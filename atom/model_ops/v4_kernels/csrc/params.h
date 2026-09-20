// SPDX-License-Identifier: MIT
// Copyright (C) 2026 Advanced Micro Devices, Inc.
#pragma once
#include <cstdint>
struct AtomTopKParams {
  const float* scores;
  const int32_t* lengths;
  int32_t* raw;
  const int32_t* block_tables;
  const void* positions;
  const int32_t* indptr;
  const int32_t* batch_ids;
  int32_t* packed;
  int64_t score_stride;
  int64_t packed_size;
  int32_t rows, width, k, sequences, blocks_per_seq;
  int32_t envelope_rows, block_capacity, window_size;
  bool positions_i64;
  bool fuse_csa;
};
int atom_v4_topk_launch(AtomTopKParams params, void* stream);
