# SPDX-License-Identifier: MIT
"""GPU self-check for the M3 byte-offload path, without a vLLM server.

The unit tests next to this file pin the mapping with synthetic strides on CPU.
This script runs the real thing on a real GPU -- ATOM's `DenseKVByteCodec` and,
in the second half, an actual LMCache engine -- over an M3-shaped registration,
in both KV layouts vLLM can resolve for M3.

It exists because starting an M3 server needs the vLLM this recipe family is
built against (`Inferact/vllm-m3-amd`); everything below the connector can still
be exercised without it.

    PYTHONHASHSEED=0 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
    LMCACHE_CHUNK_SIZE=128 HIP_VISIBLE_DEVICES=0 \
    python3 tests/plugin/m3_offload_gpu_selfcheck.py

Named so pytest does not collect it: it needs a GPU and an LMCache CPU pool.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace as NS

import torch

from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._block_gpu_connector import BlockGPUConnector
from atom.kv_transfer.offload._offload_common import build_offload_engine
from atom.kv_transfer.offload.config import build_page_namespace
from atom.kv_transfer.offload.dense.kv_byte_codec import DenseKVByteCodec
from atom.plugin.vllm.kv_transfer.kv_cache_layout import (
    build_kv_cache_tensors,
    kv_layout_tag,
)
from atom.plugin.vllm.kv_transfer.offload_config import build_offload_config

BS, HD = 128, 128
DEV = "cuda:0"


class _SparseLayer:
    """M3's sparse attention: one fp32 scale per token per head, on the layer."""

    def __init__(self, nb: int) -> None:
        self.kv_scale = torch.rand((2, nb, 1, BS), dtype=torch.float32, device=DEV)

    def get_kv_transfer_scales(self, kv_cache=None):
        return self.kv_scale[0], self.kv_scale[1]


def _sparse_lhbnc(nb: int) -> torch.Tensor:
    """K/V-plane-major: the whole tensor is not contiguous, each half is."""
    return torch.empty((2, nb, BS, 1, HD), dtype=torch.uint8, device=DEV).permute(
        1, 0, 2, 3, 4
    )


def _sparse_lbhnc(nb: int) -> torch.Tensor:
    """Block-major: a block's K and V bytes are already adjacent."""
    return torch.empty((nb, 2, BS, 1, HD), dtype=torch.uint8, device=DEV)


def _registration(nb: int, dense: int, sparse: int, make_sparse, dense_c=HD):
    kv: dict[str, torch.Tensor] = {}
    layers: dict[str, _SparseLayer] = {}
    for i in range(dense):
        kv[f"model.layers.{i}.self_attn.attn"] = torch.empty(
            (nb, 1, BS, 2 * dense_c), dtype=torch.uint8, device=DEV
        )
    for i in range(dense, dense + sparse):
        name = f"model.layers.{i}.self_attn.attn"
        kv[name] = make_sparse(nb)
        kv[f"{name}.index_cache"] = torch.empty(
            (nb, BS, HD), dtype=torch.float8_e4m3fn, device=DEV
        )
        layers[name] = _SparseLayer(nb)
    return kv, layers


def _randomize(kv, layers):
    """Random bytes everywhere, so a segment silently skipped cannot pass."""
    for tensor in kv.values():
        flat = tensor.view(torch.uint8)
        flat.copy_(torch.randint(1, 255, flat.shape, dtype=torch.uint8, device=DEV))
    for layer in layers.values():
        layer.kv_scale.uniform_(0.5, 2.0)


def codec_round_trip(name: str, make_sparse, nb: int = 64) -> tuple[str, int, bool]:
    """Gather, wipe, scatter -- every byte of an M3 census must come back."""
    kv, layers = _registration(nb, 3, 57, make_sparse, dense_c=8 * HD)
    tensors = build_kv_cache_tensors(kv, layers)
    tag = kv_layout_tag(tensors)
    codec = DenseKVByteCodec({str(t.layer_num): t for t in tensors}, num_blocks=nb)
    print(
        f"[{name}] layers={len(tensors)} tag={tag} segments={len(codec._segments)} "
        f"bytes_per_block={codec.bytes_per_block}"
    )

    _randomize(kv, layers)
    want = [seg.detach().clone() for seg in codec._segments]

    block_ids = list(range(0, nb, 2))
    half = len(block_ids) // 2
    groups = [block_ids[:half], block_ids[half:]]
    staging = torch.empty(
        len(block_ids) * codec.bytes_per_block, dtype=torch.uint8, device=DEV
    )
    codec.gpu_to_chunk_major_device_buffer(staging, groups)
    torch.cuda.synchronize()

    for seg, block_bytes in zip(codec._segments, codec._seg_block_bytes):
        seg.view(torch.uint8).reshape(nb, block_bytes)[block_ids] = 0
    torch.cuda.synchronize()

    codec.chunk_major_device_buffer_to_gpu(staging, groups)
    torch.cuda.synchronize()

    bad = [
        i
        for i, (seg, ref) in enumerate(zip(codec._segments, want))
        if not torch.equal(seg.view(torch.uint8), ref.view(torch.uint8))
    ]
    if bad:
        print(f"[{name}] FAIL: {len(bad)} segments differ (first: {bad[:5]})")
    else:
        print(f"[{name}] all {len(codec._segments)} segments byte-identical")
    return tag, codec.bytes_per_block, not bad


def _vllm_config(layers: int):
    return NS(
        cache_config=NS(block_size=BS, cache_dtype="fp8"),
        model_config=NS(
            hf_config=NS(
                num_hidden_layers=layers,
                model_type="minimax_m3",
                num_attention_heads=64,
                num_key_value_heads=4,
                hidden_size=8192,
                head_dim=HD,
            ),
            dtype="bfloat16",
            model="/models/MiniMax-M3-MXFP8",
        ),
        parallel_config=NS(
            pipeline_parallel_size=1,
            tensor_parallel_size=4,
            decode_context_parallel_size=1,
        ),
        kv_transfer_config=NS(kv_role="kv_both", kv_connector_extra_config={}),
    )


def engine_round_trip(nb: int = 32) -> bool:
    """The same bytes, but through a real LMCache engine and back."""
    kv, layers = _registration(nb, 3, 5, _sparse_lbhnc)
    tensors = build_kv_cache_tensors(kv, layers)
    config = build_offload_config(_vllm_config(8))

    lm_cfg = offcfg.build_lmcache_config(config.kv_transfer_config)
    key_untagged = build_page_namespace(config, lm_cfg, 4)
    config.page_layout_tag = kv_layout_tag(tensors)
    key_tagged = build_page_namespace(config, lm_cfg, 4)
    print("[engine] layout tag        :", config.page_layout_tag)
    print("[engine] namespace untagged:", key_untagged)
    print("[engine] namespace tagged  :", key_tagged)

    codec = DenseKVByteCodec({str(t.layer_num): t for t in tensors}, num_blocks=nb)
    engine, cfg, _ = build_offload_engine(
        config,
        engine_id="atom-m3-selfcheck-0",
        block_size=BS,
        bytes_per_block=codec.bytes_per_block,
        gpu_connector_factory=lambda cfg, meta: BlockGPUConnector(
            codec, BS, chunk_size=int(cfg.chunk_size), virtual_block_size=BS
        ),
        world=4,
        rank=0,
    )
    print(
        f"[engine] built             : chunk_size={cfg.chunk_size} "
        f"bytes_per_block={codec.bytes_per_block}"
    )

    _randomize(kv, layers)
    want = [seg.detach().clone() for seg in codec._segments]

    block_ids = [0, 1, 2, 3]
    tokens = torch.arange(BS * len(block_ids), dtype=torch.int64)
    engine.store(tokens, mask=None, block_ids=block_ids, req_id="selfcheck")
    for seg, block_bytes in zip(codec._segments, codec._seg_block_bytes):
        seg.view(torch.uint8).reshape(nb, block_bytes)[block_ids] = 0
    torch.cuda.synchronize()
    engine.retrieve(tokens, mask=None, block_ids=block_ids, req_id="selfcheck")
    torch.cuda.synchronize()

    bad = [
        i
        for i, (seg, ref) in enumerate(zip(codec._segments, want))
        if not torch.equal(seg.view(torch.uint8), ref.view(torch.uint8))
    ]
    print(
        f"[engine] segments restored : {len(codec._segments) - len(bad)} / "
        f"{len(codec._segments)}"
    )

    ok = not bad
    # The namespace is a digest, so the tag shows up as a different key rather
    # than as readable text; what matters is that it moves the key at all.
    if key_tagged == key_untagged:
        print("[engine] FAIL: the layout tag did not change the namespace key")
        ok = False
    return ok


def main() -> int:
    torch.manual_seed(0)
    lhbnc = codec_round_trip("LHBNC", _sparse_lhbnc)
    lbhnc = codec_round_trip("LBHNC", _sparse_lbhnc)
    ok = lhbnc[2] and lbhnc[2]

    if (lhbnc[0], lbhnc[0]) != ("kv-split", "kv-whole"):
        print(
            f"FAIL: namespace tags did not separate the layouts "
            f"({lhbnc[0]} / {lbhnc[0]})"
        )
        ok = False
    if lhbnc[1] != lbhnc[1]:
        print(
            f"FAIL: the layouts disagree on bytes per block "
            f"({lhbnc[1]} vs {lbhnc[1]})"
        )
        ok = False
    else:
        print(f"both layouts cost {lhbnc[1]} bytes per block")

    ok = engine_round_trip() and ok
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    # LMCache leaves a non-daemon monitor thread behind, so a normal exit hangs
    # long after the answer is printed.
    os._exit(rc)
