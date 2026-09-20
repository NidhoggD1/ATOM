# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""GPU correctness tests for V4 TopK and head-CSA / tail-SWA packing."""

import pytest
import torch

if torch.version.hip is None or not torch.cuda.is_available():
    pytest.skip("requires ROCm GPU", allow_module_level=True)

from atom.model_ops.v4_kernels.cpp_topk import cpp_topk, cpp_topk_csa, load_cpp_topk


@pytest.fixture(scope="module", autouse=True)
def build_extension():
    load_cpp_topk()


def i32(values):
    return torch.tensor(values, dtype=torch.int32, device="cuda")


def check_selection(scores, lengths, result, k, batch_ids=None):
    scores, lengths, result = scores.cpu(), lengths.cpu(), result.cpu()
    ids = batch_ids.cpu() if batch_ids is not None else None
    for r in range(scores.shape[0]):
        n = max(0, min(int(lengths[r]), scores.shape[1]))
        if ids is not None and ids[r] < 0:
            n = 0
        count = min(n, k)
        idx = result[r, :count].long()
        assert ((idx >= 0) & (idx < n)).all()
        assert idx.unique().numel() == count
        assert (result[r, count:] == -1).all()
        # Finite-value ordering agrees with torch.topk. For NaNs, follow
        # AITER radix ordering (negative below -inf, positive above +inf).
        def ordered_keys(values):
            bits = values.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            keys = torch.where(bits & 0x80000000 != 0,
                               (~bits) & 0xFFFFFFFF, bits | 0x80000000)
            return torch.where(values == 0, 0x80000000, keys)

        actual = ordered_keys(scores[r, idx]).sort(descending=True).values
        expected = ordered_keys(scores[r, :n]).topk(count).values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", [1, 31, 1025, 8191, 8192, 8193, 16384, 16385, 32769])
@pytest.mark.parametrize("k", [1, 32, 1024, 2048])
def test_ragged_visibility_and_stride(width, k):
    # Odd row stride forces the unaligned-load path on some rows.
    torch.manual_seed(12)
    backing = torch.randn((8, width + 3), device="cuda")
    scores = backing[:, :width]
    lens = [0, min(width, 3), min(width, k - 1), min(width, k),
            max(0, width - 1), width, width + 5, -1]
    for r, n in enumerate(lens):
        scores[r, max(0, min(width, n)):] = float("inf")
    lengths = i32(lens)
    check_selection(scores, lengths, cpp_topk(scores, lengths, k), k)


@pytest.mark.parametrize("n", [2, 31, 32, 33, 64, 65, 128, 129, 1024, 2048, 2049, 32769])
@pytest.mark.parametrize("k", [1, 17])
def test_tie_paths(n, k):
    scores = torch.ones((2, n), device="cuda")
    lengths = i32([n, n])
    check_selection(scores, lengths, cpp_topk(scores, lengths, k), k)


@pytest.mark.parametrize("kind", ["overflow", "special", "negative", "zeros", "masked"])
def test_adversarial_scores(kind):
    n, k = 32769, 1024
    if kind == "overflow":
        # More than 2048 distinct fp32 values in one coarse fp16 bin.
        row = 1.0 + torch.arange(n, device="cuda") * 2.0**-26
    elif kind == "special":
        row = torch.randn(n, device="cuda")
        row[::4] = float("-inf")
        row[::17] = float("inf")
        row[::101] = float("nan")
    elif kind == "masked":
        row = torch.randn(n, device="cuda")
        row[::13] = float("-inf")
        row[::7000] = float("inf")
    elif kind == "negative":
        row = -1e30 - torch.arange(n, device="cuda") * 1e24
    else:
        row = torch.zeros(n, device="cuda")
        row[::2] = -0.0
    scores = row.unsqueeze(0)
    lengths = i32([n])
    check_selection(scores, lengths, cpp_topk(scores, lengths, k), k)


CANARY = 123456789
K, WINDOW, CAPACITY, ENVELOPE = 256, 512, 64, 448


def fused_case(position_dtype=torch.int64):
    torch.manual_seed(31)
    width = 8193
    scores = torch.randn((6, width), device="cuda")
    lengths = i32([0, 17, 384, 8192, 13, 200])
    ids = i32([0, 1, 0, 1, -1, -1])
    positions = torch.tensor([0, 67, 1535, 32767, 51, 799],
                             dtype=position_dtype, device="cuda")
    blocks = (width + CAPACITY - 1) // CAPACITY
    tables = torch.arange(2 * blocks, dtype=torch.int32, device="cuda").flip(0).reshape(2, blocks)
    # Fixed allocation, including guards and deliberately reserved pad slices.
    packed = torch.full((6 * (K + WINDOW) + 32,), CANARY, dtype=torch.int32, device="cuda")
    indptr = i32([0] * 7)
    case = [scores, lengths, tables, positions, indptr, ids, packed]
    set_indptr(case)
    return case


def set_indptr(case):
    _, lengths, _, positions, indptr, ids, _ = case
    offsets = [7]
    for length, pos, bid in zip(lengths.cpu().tolist(), positions.cpu().tolist(), ids.cpu().tolist()):
        count = min(K, max(0, length)) + min(WINDOW, max(0, pos + 1)) if bid >= 0 else 9
        offsets.append(offsets[-1] + count)
    indptr.copy_(i32(offsets))


def run_fused(case):
    return cpp_topk_csa(*case, K, ENVELOPE, CAPACITY, WINDOW)


def check_packing(case, raw):
    scores, lengths, tables, positions, indptr, ids, packed = case
    check_selection(scores, lengths, raw, K, ids)
    tables, positions, indptr, ids, raw = [x.cpu() for x in (tables, positions, indptr, ids, raw)]
    expected = torch.full_like(packed.cpu(), CANARY)
    for r, bid in enumerate(ids.tolist()):
        if bid < 0:
            continue
        base, end = int(indptr[r]), int(indptr[r + 1])
        n = end - base - min(WINDOW, max(0, int(positions[r]) + 1))
        selected = raw[r, :n].long()
        expected[base:base + n] = tables[bid, selected // CAPACITY] * ENVELOPE + selected % CAPACITY
    # Checks CSA values, every SWA tail, padding slices, and both guard regions.
    torch.testing.assert_close(packed.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_fused_mapping_and_swa_preservation(position_dtype):
    case = fused_case(position_dtype)
    check_packing(case, run_fused(case))


def test_graph_replay_reads_current_metadata():
    case = fused_case()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run_fused(case)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        raw = run_fused(case)
    # Change values but retain captured pointers and static score shape.
    for lens in ([1, 300, 80, 6000, 13, 200], [2000, 0, 17, 200, 13, 200]):
        case[1].copy_(i32(lens))
        case[3].copy_(torch.tensor([max(0, n * 4 - 1) for n in lens], dtype=case[3].dtype, device="cuda"))
        case[2].copy_(case[2].flip(1))
        set_indptr(case)
        case[6].fill_(CANARY)
        graph.replay()
        check_packing(case, raw)


def test_nondefault_stream():
    case = fused_case()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        raw = run_fused(case)
    torch.cuda.current_stream().wait_stream(stream)
    check_packing(case, raw)


def test_compile_preserves_mutation_when_raw_result_is_unused():
    case = fused_case()

    def mutate_and_read(scores, lengths, tables, positions, indptr, ids, packed):
        cpp_topk_csa(scores, lengths, tables, positions, indptr, ids, packed,
                     K, ENVELOPE, CAPACITY, WINDOW)
        return packed.clone()

    compiled = torch.compile(mutate_and_read, fullgraph=True)
    result = compiled(*case)
    # Recover raw indices from the packed physical addresses to independently
    # check selection. A deleted mutation leaves canaries and cannot pass.
    tables = case[2].cpu()
    raw = torch.full((6, K), -1, dtype=torch.int32)
    for r, bid in enumerate(case[5].cpu().tolist()):
        if bid < 0:
            continue
        n = min(K, int(case[1][r]))
        begin = int(case[4][r])
        for j, physical in enumerate(result[begin:begin + n].cpu().tolist()):
            block = (tables[bid] == physical // ENVELOPE).nonzero().flatten()
            assert block.numel() == 1
            raw[r, j] = int(block[0]) * CAPACITY + physical % ENVELOPE
    check_packing(case, raw)
    torch.testing.assert_close(result, case[6])


def test_input_validation_and_empty_batch():
    scores = torch.randn((2, 100), device="cuda")
    lengths = i32([100, 100])
    with pytest.raises(RuntimeError, match="k must"):
        cpp_topk(scores, lengths, 2049)
    with pytest.raises(RuntimeError, match="contiguous int32"):
        cpp_topk(scores, lengths.long(), 10)
    assert cpp_topk(scores[:0], lengths[:0], 10).shape == (0, 10)


@pytest.mark.parametrize("n", [33, 65, 128, 1024, 2048, 2049])
@pytest.mark.parametrize("k", [17, 1024])
def test_negative_infinity_at_cutoff(n, k):
    scores = torch.full((1, n), float("-inf"), device="cuda")
    lengths = i32([n])
    check_selection(scores, lengths, cpp_topk(scores, lengths, k), k)


@pytest.mark.parametrize("width", [8192, 8198, 16385])
def test_real_v4_negative_nan_in_column_zero(width):
    scores = torch.randn((64, width), device="cuda")
    scores.view(torch.int32)[:, 0] = -4194304  # 0xffc00000, observed FP4 logits value
    lengths = i32([width] * 64)
    result = cpp_topk(scores, lengths, 1024)
    check_selection(scores, lengths, result, 1024)
    assert (result != 0).all()


def test_nan_sign_and_payload_cutoff():
    bits = torch.tensor([0x7FC00001, 0x7FC00002, -4194303, -4194302], dtype=torch.int32, device="cuda")
    row = bits.view(torch.float32).repeat(600)
    scores = row.unsqueeze(0)
    lengths = i32([scores.shape[1]])
    for k in (1, 1024, 2048):
        check_selection(scores, lengths, cpp_topk(scores, lengths, k), k)
