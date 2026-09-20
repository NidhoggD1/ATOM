// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Advanced Micro Devices, Inc.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include "params.h"

static void check_index(const at::Tensor& value, const at::Tensor& scores, const char* name) {
  TORCH_CHECK(value.device() == scores.device(), name, " must be on the scores device");
  TORCH_CHECK(value.scalar_type() == at::kInt && value.is_contiguous(), name, " must be contiguous int32");
}
static AtomTopKParams params_for(const at::Tensor& scores, const at::Tensor& lengths, int64_t k) {
  TORCH_CHECK(scores.is_cuda(), "V4 TopK requires a ROCm tensor");
  TORCH_CHECK(scores.scalar_type() == at::kFloat && scores.dim() == 2 && scores.stride(1) == 1,
              "scores must be a float32 matrix with contiguous columns");
  TORCH_CHECK(scores.stride(0) >= scores.size(1), "scores rows must not overlap");
  TORCH_CHECK(scores.size(0) <= std::numeric_limits<int32_t>::max() &&
              scores.size(1) <= std::numeric_limits<int32_t>::max(), "scores shape exceeds int32");
  TORCH_CHECK(k > 0 && k <= 2048, "k must be in [1, 2048]");
  check_index(lengths, scores, "lengths");
  TORCH_CHECK(lengths.dim() == 1 && lengths.numel() >= scores.size(0), "lengths must cover every score row");
  AtomTopKParams p{};
  p.scores = scores.data_ptr<float>();
  p.lengths = lengths.data_ptr<int32_t>();
  p.score_stride = scores.stride(0);
  p.rows = scores.size(0);
  p.width = scores.size(1);
  p.k = k;
  return p;
}
static at::Tensor run(const at::Tensor& scores, AtomTopKParams p) {
  const c10::cuda::CUDAGuard guard(scores.device());
  auto raw = at::empty({p.rows, p.k}, scores.options().dtype(at::kInt));
  p.raw = raw.data_ptr<int32_t>();
  auto stream = at::cuda::getCurrentCUDAStream(scores.get_device());
  const int status = atom_v4_topk_launch(p, reinterpret_cast<void*>(stream.stream()));
  TORCH_CHECK(status == 0, "V4 TopK HIP launch failed with error code ", status);
  return raw;
}
static at::Tensor topk(const at::Tensor& scores, const at::Tensor& lengths, int64_t k) {
  return run(scores, params_for(scores, lengths, k));
}
static at::Tensor topk_csa(const at::Tensor& scores, const at::Tensor& lengths,
    const at::Tensor& tables, const at::Tensor& positions, const at::Tensor& indptr,
    const at::Tensor& batch_ids, at::Tensor packed, int64_t k, int64_t envelope_rows,
    int64_t block_capacity, int64_t window_size) {
  auto p = params_for(scores, lengths, k);
  check_index(tables, scores, "block_tables");
  check_index(indptr, scores, "indptr");
  check_index(batch_ids, scores, "batch_ids");
  check_index(packed, scores, "packed_indices");
  TORCH_CHECK(tables.dim() == 2 && tables.size(1) > 0, "block_tables must be a nonempty-width matrix");
  TORCH_CHECK(indptr.dim() == 1 && indptr.numel() >= int64_t(p.rows) + 1, "indptr is too short");
  TORCH_CHECK(batch_ids.dim() == 1 && batch_ids.numel() >= p.rows, "batch_ids is too short");
  TORCH_CHECK(packed.dim() == 1, "packed_indices must be flat");
  TORCH_CHECK(positions.device() == scores.device() && positions.dim() == 1 && positions.is_contiguous() &&
              positions.numel() >= p.rows && (positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong),
              "positions must be contiguous int32/int64 on the scores device");
  TORCH_CHECK(block_capacity > 0 && envelope_rows >= block_capacity &&
              envelope_rows <= std::numeric_limits<int32_t>::max() &&
              window_size > 0 && window_size <= std::numeric_limits<int32_t>::max(), "invalid CSA geometry");
  TORCH_CHECK(tables.size(0) <= std::numeric_limits<int32_t>::max() &&
              tables.size(1) <= std::numeric_limits<int32_t>::max(), "block_tables shape exceeds int32");
  p.block_tables = tables.data_ptr<int32_t>();
  p.positions = positions.data_ptr();
  p.indptr = indptr.data_ptr<int32_t>();
  p.batch_ids = batch_ids.data_ptr<int32_t>();
  p.fuse_csa = true;
  p.packed = packed.data_ptr<int32_t>();
  p.packed_size = packed.numel();
  p.sequences = tables.size(0);
  p.blocks_per_seq = tables.size(1);
  p.envelope_rows = envelope_rows;
  p.block_capacity = block_capacity;
  p.window_size = window_size;
  p.positions_i64 = positions.scalar_type() == at::kLong;
  return run(scores, p);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk", &topk);
  m.def("topk_csa", &topk_csa);
}
