// SPDX-License-Identifier: Apache-2.0
// Portions derived from SGLang (Apache-2.0).
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// Selection dispatch is adapted from SGLang topk_v2.cuh at 7f1f8c706ac.
// The CSA sink implements ATOM's pool geometry and head-CSA / tail-SWA layout.
#include "params.h"
#include "topk_impl.cuh"

namespace atom::v4_topk {
template <bool Packed, int Level>
__global__ __launch_bounds__(1024, 2) void topk_csa_kernel(AtomTopKParams p) {
  __shared__ TopKRegister<4>::Smem smem;
  __shared__ int32_t selected[TopKConfig::kMaxTopK];
  const int row = blockIdx.x;
  const int tx = threadIdx.x;
  const int bid = Packed ? p.batch_ids[row] : 0;
  const bool valid_row = !Packed || (bid >= 0 && bid < p.sequences);
  const uint32_t length = valid_row ? uint32_t(max(0, min(p.width, p.lengths[row]))) : 0;
  TopKProblem problem{p.scores + int64_t(row) * p.score_stride, selected, uint32_t(p.k), length};
  for (uint32_t j = tx; j < uint32_t(p.k); j += blockDim.x) selected[j] = -1;
  __syncthreads();
  if (length <= uint32_t(p.k)) {
    for (uint32_t j = tx; j < length; j += blockDim.x) selected[j] = int32_t(j);
  } else if constexpr (Level == 0) {
    TopKRegister<2>::forward<false>(problem, &smem);
  } else if constexpr (Level == 1) {
    TopKRegister<4>::forward<false>(problem, &smem);
  } else {
    if (length <= TopKRegister<2>::kMaxSeqLen) TopKRegister<2>::forward<false>(problem, &smem);
    else if (length <= TopKRegister<4>::kMaxSeqLen) TopKRegister<4>::forward<false>(problem, &smem);
    else TopKStreaming::forward<false>(problem, &smem);
  }
  __syncthreads();
  int64_t base = 0;
  int valid_k = 0;
  if constexpr (Packed) {
    if (valid_row) {
      const int64_t pos = p.positions_i64 ? static_cast<const int64_t*>(p.positions)[row]
                                         : static_cast<const int32_t*>(p.positions)[row];
      const int64_t skip = pos < 0 ? 0 : (pos >= p.window_size ? p.window_size : pos + 1);
      base = p.indptr[row];
      const int64_t end = p.indptr[row + 1];
      // Match csa_translate_pack: CSA is at the slice HEAD; SWA follows it.
      if (base >= 0 && end >= base && end <= p.packed_size)
        valid_k = int(min(max(end - base - skip, int64_t(0)), int64_t(p.k)));
    }
  }
  for (int j = tx; j < p.k; j += blockDim.x) {
    const int raw = selected[j];
    p.raw[int64_t(row) * p.k + j] = raw;
    if constexpr (Packed) {
      if (j < valid_k) {
        int32_t physical_row = -1;
        if (raw >= 0) {
          const int block = raw / p.block_capacity;
          const int slot = raw % p.block_capacity;
          if (block < p.blocks_per_seq) {
            const int physical = p.block_tables[int64_t(bid) * p.blocks_per_seq + block];
            const int64_t translated = int64_t(physical) * p.envelope_rows + slot;
            if (physical >= 0 && translated <= INT32_MAX) physical_row = int32_t(translated);
          }
        }
        p.packed[base + j] = physical_row;
      }
    }
  }
}
}  // namespace atom::v4_topk

int atom_v4_topk_launch(AtomTopKParams p, void* raw_stream) {
  if (p.rows == 0) return int(hipSuccess);
  const auto stream = static_cast<hipStream_t>(raw_stream);
  using namespace atom::v4_topk;
#define ATOM_TOPK_LAUNCH(PACKED, LEVEL) \
  hipLaunchKernelGGL((topk_csa_kernel<PACKED, LEVEL>), dim3(p.rows), dim3(1024), 0, stream, p)
  if (p.fuse_csa) {
    if (p.width <= 8192) { ATOM_TOPK_LAUNCH(true, 0); }
    else if (p.width <= 16384) { ATOM_TOPK_LAUNCH(true, 1); }
    else { ATOM_TOPK_LAUNCH(true, 2); }
  } else {
    if (p.width <= 8192) { ATOM_TOPK_LAUNCH(false, 0); }
    else if (p.width <= 16384) { ATOM_TOPK_LAUNCH(false, 1); }
    else { ATOM_TOPK_LAUNCH(false, 2); }
  }
#undef ATOM_TOPK_LAUNCH
  return int(hipGetLastError());
}
