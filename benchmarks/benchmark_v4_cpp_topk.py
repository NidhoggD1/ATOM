# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Same-input, CUDA-graph microbenchmark. Run on an available ROCm GPU."""
import argparse
import json

import torch
from triton.testing import do_bench_cudagraph

from aiter.ops.topk import top_k_per_row_decode
from atom.model_ops.v4_kernels.cpp_topk import cpp_topk, cpp_topk_csa, load_cpp_topk
from atom.model_ops.v4_kernels.csa_translate_pack import csa_translate_pack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rows", type=int, default=64,
        help="Query-token rows; concurrency 16 with DSpark3 uses 64 rows.",
    )
    parser.add_argument("--width", type=int, default=262144)
    parser.add_argument("--lengths", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    parser.add_argument("--k", type=int, default=1024)
    args = parser.parse_args()
    assert args.rows > 0 and args.width > 0 and 1 <= args.k <= 2048
    load_cpp_topk()
    torch.manual_seed(42)
    scores = torch.randn((args.rows, args.width), device="cuda")
    raw = torch.empty((args.rows, args.k), dtype=torch.int32, device="cuda")
    blocks = (args.width + 63) // 64
    tables = torch.arange(args.rows * blocks, dtype=torch.int32, device="cuda").reshape(args.rows, blocks)
    ids = torch.arange(args.rows, dtype=torch.int32, device="cuda")
    reports = []
    for length in args.lengths:
        assert 0 < length <= args.width
        ends = torch.full((args.rows,), length, dtype=torch.int32, device="cuda")
        positions = ends.long() * 4 - 1
        stride = min(length, args.k) + min(length * 4, 512)
        indptr = torch.arange(args.rows + 1, dtype=torch.int32, device="cuda") * stride
        packed = torch.zeros(args.rows * stride, dtype=torch.int32, device="cuda")

        def aiter_only():
            top_k_per_row_decode(scores, 1, ends, raw, args.rows,
                                 scores.stride(0), scores.stride(1), k=args.k)

        def translate():
            csa_translate_pack(raw, tables, positions, indptr, ids, None, packed,
                               envelope_rows=448, csa_block_capacity=64, window_size=512)

        def baseline():
            aiter_only()
            translate()

        def fused():
            return cpp_topk_csa(scores, ends, tables, positions, indptr, ids, packed,
                                args.k, 448, 64, 512)

        baseline()
        fused()
        torch.cuda.synchronize()
        timings = {
            "aiter_topk_us": do_bench_cudagraph(aiter_only) * 1000,
            "aiter_topk_translate_us": do_bench_cudagraph(baseline) * 1000,
            "cpp_topk_us": do_bench_cudagraph(lambda: cpp_topk(scores, ends, args.k)) * 1000,
            "cpp_topk_csa_us": do_bench_cudagraph(fused) * 1000,
        }
        timings["saved_us"] = timings["aiter_topk_translate_us"] - timings["cpp_topk_csa_us"]
        reports.append({"rows": args.rows, "width": args.width, "length": length, "k": args.k, **timings})
    print(json.dumps({"torch": torch.__version__, "hip": torch.version.hip,
                      "device": torch.cuda.get_device_name(), "results": reports}, indent=2))


if __name__ == "__main__":
    main()
