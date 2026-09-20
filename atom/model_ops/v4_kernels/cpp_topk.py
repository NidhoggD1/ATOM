# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in HIP TopK and fused decode CSA packing for the V4 FP4 indexer.

Build once with ``load_cpp_topk()`` before CUDA graph capture. The extension
uses the current PyTorch stream and never synchronizes or copies metadata to
the host. Outputs are unsorted, with -1 after min(length, k); ties may select
any equally scoring index. NaNs follow AITER ordered-fp32 radix order:
negative NaNs rank below -inf and positive NaNs above +inf. The fused operation writes only each row's CSA
head, preserving its SWA tail. It has no SGLang runtime dependency.
"""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load_cpp_topk():
    if torch.version.hip is None:
        raise RuntimeError("V4 C++ TopK requires a ROCm PyTorch build")
    from torch.utils.cpp_extension import _get_build_directory, load
    from torch.utils.file_baton import FileBaton

    source = Path(__file__).with_name("csrc")
    inputs = {
        name: (source / name).read_bytes()
        for name in ("bindings.cpp", "v4_topk.cu", "topk_impl.cuh", "params.h")
    }
    # PyTorch's HIP ninja rule does not track device-header dependencies.
    # Include all headers in the build identity so a header-only kernel change
    # cannot silently reload an old binary. Stage under the build cache because
    # hipify also writes translated sources; installed packages can be read-only.
    digest = hashlib.sha256()
    for name, content in inputs.items():
        digest.update(name.encode())
        digest.update(content)
    name = "atom_v4_cpp_topk_" + digest.hexdigest()[:16]
    build = Path(_get_build_directory(name, verbose=False))
    staged = build / "src"
    baton = FileBaton(str(build / "source.lock"))
    if baton.try_acquire():
        try:
            staged.mkdir(exist_ok=True)
            for filename, content in inputs.items():
                path = staged / filename
                if not path.exists() or path.read_bytes() != content:
                    path.write_bytes(content)
        finally:
            baton.release()
    else:
        baton.wait()
    return load(
        name=name,
        build_directory=str(build),
        sources=[str(staged / "bindings.cpp"), str(staged / "v4_topk.cu")],
        extra_include_paths=[str(staged)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17"],
    )


@torch.library.custom_op("atom_v4::cpp_topk", mutates_args=())
def cpp_topk(scores: torch.Tensor, lengths: torch.Tensor, k: int) -> torch.Tensor:
    return load_cpp_topk().topk(scores, lengths, k)


@cpp_topk.register_fake
def _cpp_topk_fake(scores, lengths, k):
    return scores.new_empty((scores.shape[0], k), dtype=torch.int32)


@torch.library.custom_op("atom_v4::cpp_topk_csa", mutates_args=("packed_indices",))
def cpp_topk_csa(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    block_tables: torch.Tensor,
    positions: torch.Tensor,
    indptr: torch.Tensor,
    batch_ids: torch.Tensor,
    packed_indices: torch.Tensor,
    k: int,
    envelope_rows: int,
    block_capacity: int,
    window_size: int,
) -> torch.Tensor:
    return load_cpp_topk().topk_csa(
        scores, lengths, block_tables, positions, indptr, batch_ids,
        packed_indices, k, envelope_rows, block_capacity, window_size,
    )


@cpp_topk_csa.register_fake
def _cpp_topk_csa_fake(
    scores, lengths, block_tables, positions, indptr, batch_ids,
    packed_indices, k, envelope_rows, block_capacity, window_size,
):
    return scores.new_empty((scores.shape[0], k), dtype=torch.int32)
