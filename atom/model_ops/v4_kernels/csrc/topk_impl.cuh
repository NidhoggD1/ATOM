// SPDX-License-Identifier: Apache-2.0
// Portions derived from SGLang (Apache-2.0).
// Copyright (C) 2026 Advanced Micro Devices, Inc.
//
// Adapted from SGLang commit 7f1f8c706ac000b7a84ea0bda05135fc4177c6ca:
// python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh
// and python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh.
// Upstream credits vLLM persistent_topk.cuh and FlashInfer topk.cuh.
// Local changes: standalone HIP primitives, ATOM output contract, and exact
// whole-row fallback for overflowing coarse bins and non-finite scores.
// See THIRD_PARTY_NOTICES.md and LICENSE.apache-2.0 in this directory.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cfloat>
#include <cstdint>
#include <limits>

#define ATOM_DEVICE __device__ __forceinline__
namespace atom::v4_topk {
constexpr uint32_t kWarpSize = 32;
namespace warp {
ATOM_DEVICE uint32_t reduce_sum(uint32_t value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) value += __shfl_xor(value, offset, kWarpSize);
  return value;
}
}
template <typename T, uint32_t N>
struct alignas(sizeof(T) * N) AlignedVector {
  T values[N];
  ATOM_DEVICE T& operator[](uint32_t i) { return values[i]; }
  ATOM_DEVICE const T& operator[](uint32_t i) const { return values[i]; }
  ATOM_DEVICE void fill(T value) {
#pragma unroll
    for (uint32_t i = 0; i < N; ++i) values[i] = value;
  }
  ATOM_DEVICE void load(const T* base, uint32_t index) {
    const T* ptr = base + uint64_t(index) * N;
    if ((reinterpret_cast<uintptr_t>(ptr) % alignof(AlignedVector)) == 0) {
      *this = *reinterpret_cast<const AlignedVector*>(ptr);
    } else {
#pragma unroll
      for (uint32_t i = 0; i < N; ++i) values[i] = ptr[i];
    }
  }
};

ATOM_DEVICE uint32_t extract_exact_bin(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <uint32_t kBits>
ATOM_DEVICE uint32_t extract_coarse_bin(float x) {
  static_assert(0 < kBits && kBits < 15);
  // Match AITER's ordered-fp32 radix convention: negative NaNs precede
  // -inf; positive NaNs follow +inf. Payload ordering is resolved exactly
  // only when a NaN bucket contains the cutoff.
  if (isnan(x)) return (__float_as_uint(x) & 0x80000000u) ? 0u : ((1u << kBits) - 1u);
  const auto hx = __float2half_rn(x);
  const uint16_t bits = *reinterpret_cast<const uint16_t*>(&hx);
  const uint16_t key = (bits & 0x8000) ? ~bits : bits | 0x8000;
  return key >> (16 - kBits);
}

// Smallest fp32 value `v` for which `extract_coarse_bin<kBits>(v) >= bin`, i.e. the
// lower fp32 boundary of coarse bin `bin`. Because `extract_coarse_bin` is monotonic
// non-decreasing in its argument, the collect pass can classify an element with two
// fp32 comparisons against these boundaries instead of recomputing the fp16 bin --
// removing the F2F conversion and bit-twiddle from the (compute-bound) second pass.
// Returns -inf for bin 0 (everything qualifies) and +inf for bins past the top.
template <uint32_t kBits>
ATOM_DEVICE float coarse_bin_lower_bound(uint32_t bin) {
  constexpr uint32_t kShift = 16 - kBits;
  const uint32_t key = bin << kShift;  // ordered16 key at the low edge of `bin`
  // ordered16 -> fp16 value (inverse of the transform in extract_coarse_bin);
  // finite keys only.
  const auto to_finite_val = [](uint32_t okey) -> float {
    const uint16_t ob = static_cast<uint16_t>(okey);
    const uint16_t hb = (ob & 0x8000) ? static_cast<uint16_t>(ob ^ 0x8000) : static_cast<uint16_t>(~ob);
    return __half2float(*reinterpret_cast<const __half*>(&hb));
  };
  // Fast path, hoisted above the per-key special cases so both keys are
  // range-checked at once: `key` and `key - 1` both land in the finite band
  // [0x0401, 0xFBFF] -- every boundary a finite-score threshold produces.
  // fp16 rounds to nearest, so the fp32 boundary is the midpoint between the
  // fp16 values at `key` and `key - 1`. (Verified bit-exact against the slow
  // path for every bin of kBits 10 and 12, and measured faster than either
  // per-key dispatch or an ordered-bit decrement trick -- the two conversions
  // are independent and issue in parallel.)
  if (key - 0x0401u <= 0xFBFFu - 0x0401u && bin < (1u << kBits)) {
    return 0.5f * (to_finite_val(key) + to_finite_val(key - 1));
  }
  // Slow path: an edge of `bin` touches the +/-inf keys or NaN key space.
  // The ordered-key line is: [0, 0x03FF) negative-NaN space, 0x03FF = -inf,
  // [0x0400, 0xFC00) finite, 0xFC00 = +inf, (0xFC00, 0xFFFF] positive-NaN
  // space. Treat the +/-inf keys as +/-65536 (one ideal step past fp16 max,
  // so the midpoint lands exactly on +/-65520 -- the fp32->fp16
  // round-to-nearest overflow threshold) and saturate NaN-space keys, keeping
  // the returned boundaries finite-or-inf and monotone. Otherwise a threshold
  // bin at/next to the inf bin gets NaN boundaries, the collect pass matches
  // nothing, and rows whose scores contain >= topk (+/-)inf or >65504 values
  // come back short -- the padded slots then illegal-address downstream.
  if (bin == 0) return -FLT_MAX;
  if (bin >= (1u << kBits)) return FLT_MAX;
  const auto to_val = [&](uint32_t okey) -> float {
    constexpr float k_Inf = std::numeric_limits<float>::infinity();
    if (okey < 0x03FFu) return -k_Inf;
    if (okey == 0x03FFu) return -65536.0f;
    if (okey == 0xFC00u) return 65536.0f;
    if (okey > 0xFC00u) return FLT_MAX;
    return to_finite_val(okey);
  };
  return 0.5f * (to_val(key) + to_val(key - 1));
}

ATOM_DEVICE uint32_t warp_inclusive_sum(uint32_t lane_id, uint32_t val) {
#pragma unroll
  for (uint32_t offset = 1; offset < kWarpSize; offset *= 2) {
    const uint32_t n = __shfl_up(val, offset, kWarpSize);
    if (lane_id >= offset) val += n;
  }
  return val;
}

ATOM_DEVICE uint32_t warp_sum_bool(bool pred, uint32_t mask = 0xFFFFFFFFu) {
  // Two logical 32-lane groups share one CDNA wave64 ballot.
  const uint32_t half = (threadIdx.x % warpSize) / kWarpSize;
  return __popcll(__ballot(pred) & (uint64_t(mask) << (kWarpSize * half)));
}

struct alignas(8) TieValue {
  float value;
  uint32_t idx;
  inline static constexpr TieValue invalid() {
    return TieValue{-FLT_MAX, 0xFFFFFFFFu};
  }
};

// ---------------------------------------------------------------------------
// Per-batch problem description + page-table transform sink
// ---------------------------------------------------------------------------

// Selection writes raw seq-local indices into a CTA-local buffer. The caller
// owns the ATOM-specific packed output, not SGLang's page-table layout.
struct TopKProblem {
  const float* __restrict__ in;
  int32_t* __restrict__ out;
  uint32_t topk;
  uint32_t seq_len;
  ATOM_DEVICE void emit(uint32_t pos, uint32_t raw_idx) const {
    out[pos] = static_cast<int32_t>(raw_idx);
  }
};

// ---------------------------------------------------------------------------
// Shared configuration + tie handling (exact radix select on the threshold bin)
// ---------------------------------------------------------------------------

struct TopKConfig {
  static constexpr uint32_t kMaxTopK = 2048;
  static constexpr uint32_t kBlockSize = 1024;
  static constexpr uint32_t kOccupancy = 2;
  static constexpr uint32_t kNumWarps = kBlockSize / kWarpSize;
  // kMaxNumTie must be >= kMaxTopK: the collect pass keeps at most kMaxNumTie
  // threshold-bin candidates, and up to `topk` output slots may have to be
  // filled from them (above_count can be 0, e.g. heavily tied or all-inf
  // scores). A smaller cap leaves slots that handle_tie can only pad, and
  // padded slots inside the first min(seq_len, topk) entries are dereferenced
  // by downstream sparse attention.
  static constexpr uint32_t kMaxNumTie = 2048;
  static constexpr uint32_t kRadixSize = 1 << 8;
  static constexpr uint32_t kTopKItems = (kMaxTopK + kBlockSize - 1) / kBlockSize;
  // tie candidates owned per thread in the strided handle_tie loops
  static constexpr uint32_t kTieItems = kMaxNumTie / kBlockSize;
  static_assert(kMaxNumTie >= kMaxTopK && kMaxNumTie % kBlockSize == 0 && kBlockSize % kNumWarps == 0);

  struct TieHandleSmem {
    struct alignas(16) MatchBin {
      uint32_t bin;
      uint32_t above_count;
      uint32_t equal_count;
      uint32_t _pad;
    };
    alignas(128) uint32_t counter;
    alignas(128) uint32_t counter_final;
    MatchBin match;
    uint32_t warp_sum[kNumWarps];
    uint32_t histogram[2][kRadixSize];
  };

  /// Resolve the threshold bin's ties exactly. `base` is the number of strictly
  /// "above" elements already emitted (final output starts at slot `base`);
  /// `topk` here is the number of remaining slots to fill (== global_topk - base).
  ATOM_DEVICE static void handle_tie(  //
      const TieValue* tie_buffer,
      const TopKProblem& problem,
      const uint32_t base,
      const uint32_t num_ties,
      const uint32_t topk,
      TieHandleSmem* smem) {
    constexpr auto is_greater = [](const TieValue& a, const TieValue& b) {
      return (a.value > b.value) || (a.value == b.value && a.idx < b.idx);
    };
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = tx / kWarpSize;
    static_assert(kNumWarps == kWarpSize);

    if (num_ties <= topk) {
      for (uint32_t t = tx; t < num_ties; t += kBlockSize) {
        problem.emit(base + t, tie_buffer[t].idx);
      }
    } else if (num_ties <= kWarpSize) {
      if (lane_id >= num_ties || warp_id >= num_ties) return;  // some threads are idle
      /// NOTE: use long long to avoid mask overflow when num_tie == 32
      const uint32_t mask = (1ull << num_ties) - 1u;
      const auto tie = tie_buffer[lane_id];
      const auto target = tie_buffer[warp_id];
      const auto rank = warp_sum_bool(is_greater(tie, target), mask);
      if (lane_id == 0 && rank < topk) problem.emit(base + rank, target.idx);
    } else if (num_ties <= kWarpSize * 2) {
      // 64 x 64 topk implementation: each thread takes 2 elements
      const auto warp_id_0 = warp_id;
      const auto warp_id_1 = warp_id + kWarpSize;
      const auto lane_id_1 = lane_id + kWarpSize;
      const auto invalid = TieValue::invalid();
      const auto tie_0 = tie_buffer[lane_id];
      const auto tie_1 = lane_id_1 < num_ties ? tie_buffer[lane_id_1] : invalid;
      const auto target_0 = tie_buffer[warp_id_0];
      const auto target_1 = tie_buffer[warp_id_1];
      if (true) {  // NOTE: warp_id_0 <= kNumWarps < num_ties
        const auto rank_0 = warp_sum_bool(is_greater(tie_0, target_0));
        const auto rank_1 = warp_sum_bool(is_greater(tie_1, target_0));
        const auto rank = rank_0 + rank_1;
        if (lane_id == 0 && rank < topk) problem.emit(base + rank, target_0.idx);
      }
      if (warp_id_1 < num_ties) {
        const auto rank_0 = warp_sum_bool(is_greater(tie_0, target_1));
        const auto rank_1 = warp_sum_bool(is_greater(tie_1, target_1));
        const auto rank = rank_0 + rank_1;
        if (lane_id == 0 && rank < topk) problem.emit(base + rank, target_1.idx);
      }
    } else if (num_ties <= kWarpSize * 4) {
      // 128 x 128 topk implementation: each thread takes 4 elements and does local sort + merge
      const auto invalid = TieValue::invalid();
      const TieValue tie[] = {
          tie_buffer[lane_id + 0 * kWarpSize],
          tie_buffer[lane_id + 1 * kWarpSize],
          lane_id + 2 * kWarpSize < num_ties ? tie_buffer[lane_id + 2 * kWarpSize] : invalid,
          lane_id + 3 * kWarpSize < num_ties ? tie_buffer[lane_id + 3 * kWarpSize] : invalid,
      };
      const TieValue target[] = {
          tie_buffer[warp_id + 0 * kWarpSize],
          tie_buffer[warp_id + 1 * kWarpSize],
          tie_buffer[warp_id + 2 * kWarpSize],
          tie_buffer[warp_id + 3 * kWarpSize],
      };
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (i >= 2 && warp_id + i * kWarpSize >= num_ties) break;
        uint32_t rank = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          rank += warp_sum_bool(is_greater(tie[j], target[i]));
        }
        if (lane_id == 0 && rank < topk) problem.emit(base + rank, target[i].idx);
      }
    } else if (num_ties <= kBlockSize) {
      // Common case: one candidate per thread.
      radix_tie_select<1>(tie_buffer, problem, base, num_ties, topk, smem);
    } else {
      // Rare overflow case (kBlockSize < num_ties <= kMaxNumTie), kept out of
      // the common path so it alone pays the multi-item register cost.
      radix_tie_select<kTieItems>(tie_buffer, problem, base, num_ties, topk, smem);
    }
  }

  /// Exact radix select over the tie candidates: each thread owns kItems
  /// strided elements (inactive beyond num_ties). Requires
  /// num_ties <= kItems * kBlockSize.
  template <uint32_t kItems>
  ATOM_DEVICE static void radix_tie_select(  //
      const TieValue* tie_buffer,
      const TopKProblem& problem,
      const uint32_t base,
      const uint32_t num_ties,
      const uint32_t topk,
      TieHandleSmem* smem) {
    const auto tx = threadIdx.x;
    const auto lane_id = tx % kWarpSize;
    const auto warp_id = tx / kWarpSize;

    bool active[kItems];
    uint32_t key[kItems];
    uint32_t idx[kItems];
    uint32_t write_pos[kItems];
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      const auto t = tx + i * kBlockSize;
      active[i] = t < num_ties;
      const auto tie = active[i] ? tie_buffer[t] : TieValue::invalid();
      key[i] = extract_exact_bin(tie.value);
      idx[i] = tie.idx;
      write_pos[i] = topk;
    }
    uint32_t topk_remain = topk;
    if (tx < kRadixSize) smem->histogram[0][tx] = 0;
    if (tx == kRadixSize) smem->counter = smem->counter_final = 0;
    __syncthreads();
    uint32_t total_active = num_ties;

#pragma unroll
    for (int round = 0; round < 4; round++) {
      const uint32_t shift = 24 - round * 8;
      const auto hist_idx = round % 2;
      const auto histogram = smem->histogram[hist_idx];

#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        if (active[i]) atomicAdd(&histogram[(key[i] >> shift) & 0xFFu], 1);
      }
      if (round < 3 && tx < kRadixSize) {
        smem->histogram[hist_idx ^ 1][tx] = 0;
      }
      __syncthreads();

      uint32_t hist_val = 0;
      uint32_t warp_inc = 0;
      if (tx < kRadixSize) {
        hist_val = histogram[tx];
        warp_inc = warp_inclusive_sum(lane_id, hist_val);
        if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;
      }
      __syncthreads();
      if (tx < kRadixSize) {
        const auto inter = warp::reduce_sum(lane_id < warp_id ? smem->warp_sum[lane_id] : 0);
        const auto prefix = inter + warp_inc;      // inclusive prefix through this bin
        const auto above = total_active - prefix;  // elements in bins ABOVE this one
        // 3. Find threshold bin
        if (above < topk_remain && above + hist_val >= topk_remain) {
          smem->match = {tx, above, hist_val};
        }
      }
      __syncthreads();

      const auto [threshold_bin, above_count, equal_count, __] = smem->match;
      if (round < 3) total_active = equal_count;
      topk_remain -= above_count;

      // 4. Scatter
#pragma unroll
      for (uint32_t i = 0; i < kItems; ++i) {
        if (!active[i]) continue;
        const uint32_t bin = (key[i] >> shift) & 0xFFu;
        if (bin > threshold_bin) {
          write_pos[i] = atomicAdd(&smem->counter, 1);
          active[i] = false;
        } else if (bin < threshold_bin) {
          active[i] = false;
        } else if (round == 3) {
          write_pos[i] = topk - topk_remain + atomicAdd(&smem->counter_final, 1);
        }
        // my_bin == thr && round < 3: stay active for next round
      }

      if (round == 3 || topk_remain == 0) break;
    }

#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      if (write_pos[i] < topk) problem.emit(base + write_pos[i], idx[i]);
    }
  }
};

// ---------------------------------------------------------------------------
// Radix base: histogram storage + input iteration + threshold-bin search
// ---------------------------------------------------------------------------


// Preserve the ordered-fp32 NaN sign/payload convention used by AITER.
// Signed zeros are numerically equal and may choose either equal-score index.
ATOM_DEVICE uint32_t whole_row_key(float value) {
  if (value == 0.0f) value = 0.0f;
  return extract_exact_bin(value);
}

struct BlockCount { uint32_t rank; uint32_t total; };
ATOM_DEVICE BlockCount block_count(bool pred, uint32_t* sums) {
  const uint32_t lane = threadIdx.x % kWarpSize;
  const uint32_t group = threadIdx.x / kWarpSize;
  const uint32_t inc = warp_inclusive_sum(lane, uint32_t(pred));
  if (lane == kWarpSize - 1) sums[group] = inc;
  __syncthreads();
  const uint32_t value = sums[lane];  // 1024 threads = 32 logical groups.
  const uint32_t before = warp::reduce_sum(lane < group ? value : 0u);
  const uint32_t total = warp::reduce_sum(value);
  __syncthreads();  // All readers finish before the next scan reuses sums.
  return {before + inc - uint32_t(pred), total};
}

ATOM_DEVICE void exact_row_select(const TopKProblem& problem, TopKConfig::TieHandleSmem* smem) {
  const uint32_t tx = threadIdx.x;
  const uint32_t lane = tx % kWarpSize;
  const uint32_t group = tx / kWarpSize;
  uint32_t prefix = 0, mask = 0, remaining = problem.topk;
  uint32_t active_count = problem.seq_len;
  for (int shift = 24; shift >= 0; shift -= 8) {
    if (tx < 256) smem->histogram[0][tx] = 0;
    __syncthreads();
    for (uint32_t i = tx; i < problem.seq_len; i += TopKConfig::kBlockSize) {
      const uint32_t key = whole_row_key(problem.in[i]);
      if ((key & mask) == prefix) atomicAdd(&smem->histogram[0][(key >> shift) & 255], 1u);
    }
    __syncthreads();
    uint32_t count = 0, inc = 0;
    if (tx < 256) {
      count = smem->histogram[0][tx];
      inc = warp_inclusive_sum(lane, count);
      if (lane == kWarpSize - 1) smem->warp_sum[group] = inc;
    }
    __syncthreads();
    if (tx < 256) {
      const uint32_t before = warp::reduce_sum(lane < group ? smem->warp_sum[lane] : 0u);
      const uint32_t above = active_count - before - inc;
      if (above < remaining && above + count >= remaining) smem->match = {tx, above, count};
    }
    __syncthreads();
    const auto match = smem->match;
    prefix |= match.bin << shift;
    mask |= 255u << shift;
    remaining -= match.above_count;
    active_count = match.equal_count;
    __syncthreads();
  }
  // Deterministic smallest-index selection among values equal to the cutoff.
  // Emit in input order so large ties cannot depend on wave scheduling.
  uint32_t above_written = 0, equal_seen = 0;
  const uint32_t above_total = problem.topk - remaining;
  for (uint32_t base = 0; base < problem.seq_len; base += TopKConfig::kBlockSize) {
    const uint32_t i = base + tx;
    const uint32_t key = i < problem.seq_len ? whole_row_key(problem.in[i]) : 0;
    const bool above = i < problem.seq_len && key > prefix;
    const bool equal = i < problem.seq_len && key == prefix;
    const auto a = block_count(above, smem->warp_sum);
    if (above) problem.emit(above_written + a.rank, i);
    above_written += a.total;
    const auto e = block_count(equal, smem->warp_sum);
    if (equal && equal_seen + e.rank < remaining) problem.emit(above_total + equal_seen + e.rank, i);
    equal_seen += e.total;
  }
}

template <uint32_t kHistBits_>
struct TopKRadixBase : TopKConfig {
  static constexpr uint32_t kVecSize = 4;
  static constexpr uint32_t kHistBits = kHistBits_;
  static constexpr uint32_t kHistSize = 1 << kHistBits;
  using vec_t = AlignedVector<float, kVecSize>;

  struct Smem {
    using kHistVec = AlignedVector<uint32_t, kHistSize / kBlockSize>;
    alignas(128) uint32_t count_eq;
    alignas(128) uint32_t count_gt;
    uint32_t threshold_bin;
    uint32_t needs_exact;
    uint32_t warp_sum[kNumWarps];
    // The coarse histogram is dead once find_threshold() has published
    // threshold_bin, and the tie machinery only comes alive after that: the
    // collect pass fills tie.values, then handle_tie works over them with
    // tie.handle as scratch. Overlaying the two phases keeps the
    // kMaxNumTie-candidate buffer from growing the block's shared-memory
    // footprint. tie.handle and tie.values are live TOGETHER, so they sit
    // side by side inside the overlay, not in a union with each other.
    union {
      uint32_t histogram[kHistSize];
      kHistVec hist_vecs[kBlockSize];
      struct {
        TieHandleSmem handle;
        TieValue values[kMaxNumTie];
      } tie;
    };
  };

 protected:
  template <typename F>
  ATOM_DEVICE static void for_each_input(const float* __restrict__ in, uint32_t seq_len, F&& fn) {
    const auto tx = threadIdx.x;
    const uint32_t num_full = seq_len / kVecSize;  // fully-in-bounds vectors

    vec_t next_vec;
    uint32_t vi = tx;
    if (vi < num_full) next_vec.load(in, vi);
    while (vi < num_full) {
      const auto cur = next_vec;
      const auto base = vi * kVecSize;
      vi += kBlockSize;
      if (vi < num_full) next_vec.load(in, vi);
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j) {
        fn(cur[j], base + j);
      }
    }

    // Tail: at most one partial vector, `rem` in [0, kVecSize).
    static_assert(kVecSize <= kBlockSize);  // ensure tail correctness
    const uint32_t tail_start = num_full * kVecSize;
    if (tx < seq_len - tail_start) {
      const auto idx = tail_start + tx;
      fn(in[idx], idx);
    }
  }

  ATOM_DEVICE static void find_threshold(const uint32_t topk, const uint32_t seq_len, Smem* smem) {
    const auto tx = threadIdx.x;
    constexpr uint32_t kItems = kHistSize / kBlockSize;
    uint32_t orig[kItems];
    const auto hist_vec = smem->hist_vecs[tx];
    uint32_t tmp_local_sum = 0;

#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      orig[i] = hist_vec[i];
      tmp_local_sum += orig[i];
    }

    const auto lane_id = tx % kWarpSize;
    const auto warp_id = tx / kWarpSize;
    const auto warp_inc = warp_inclusive_sum(lane_id, tmp_local_sum);
    const auto warp_exc = warp_inc - tmp_local_sum;
    if (lane_id == kWarpSize - 1) smem->warp_sum[warp_id] = warp_inc;

    __syncthreads();

    const auto tmp = smem->warp_sum[lane_id];
    // Exactly one bin satisfies: above < K && above + count >= K
    uint32_t prefix_sum = warp::reduce_sum(lane_id < warp_id ? tmp : 0);
    prefix_sum += warp_exc;
#pragma unroll
    for (uint32_t i = 0; i < kItems; ++i) {
      prefix_sum += orig[i];
      const auto above = seq_len - prefix_sum;
      if (above < topk && above + orig[i] >= topk) {
        smem->threshold_bin = tx * kItems + i;
      }
    }
    __syncthreads();
  }
};

// ---------------------------------------------------------------------------
// Register path: scores stay resident in registers across both passes (read
// once). Templated on kLocalVecs so the caller picks the smallest covering
// kernel -- a larger kLocalVecs raises kMaxSeqLen but its fixed-unrolled loop
// wastes work on shorter sequences.
// ---------------------------------------------------------------------------

template <uint32_t kLocalVecs_>
struct TopKRegister : TopKRadixBase<12> {
  static constexpr uint32_t kLocalVecs = kLocalVecs_;
  static constexpr uint32_t kMaxSeqLen = kBlockSize * kVecSize * kLocalVecs;
  using Smem = typename TopKRadixBase<12>::Smem;

  template <bool kUsePDL>
  ATOM_DEVICE static void forward(const TopKProblem problem, void* _smem) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);

    {
      Smem::kHistVec hist_vec;
      hist_vec.fill(0);
      smem->hist_vecs[tx] = hist_vec;
    }
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
      smem->needs_exact = 0;
    }

    __syncthreads();


    // A vector `vi` is fully in bounds iff vi < num_full; only full vectors are
    // vector-loaded (16B aligned, never straddling seq_len). The <kVecSize tail is
    // a scalar remainder on the LAST lanes (which own the fewest full vectors, so
    // it overlaps the busy lanes' extra vector). The full path has no per-element
    // bounds check, keeping register pressure low enough to hold all vectors.
    const uint32_t num_full = problem.seq_len / kVecSize;
    const uint32_t tail_start = num_full * kVecSize;
    const uint32_t tail = problem.seq_len - tail_start;

    // Phase 1: load full vectors + build histogram
    vec_t local_vecs[kLocalVecs];
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      if (vi >= num_full) break;
      local_vecs[i].load(problem.in, vi);
    }
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      if (vi >= num_full) break;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j)
        atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(local_vecs[i][j])], 1);
    }
    if (tx >= kBlockSize - tail) {
      const uint32_t idx = tail_start + tx - (kBlockSize - tail);
      atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(problem.in[idx])], 1);
    }
    __syncthreads();

    // Phase 2: Find the threshold bin
    find_threshold(problem.topk, problem.seq_len, smem);

    // Phase 3: collect by two fp32 boundaries (raw indices; transform applied later)
    const auto topk = problem.topk;
    const auto threshold_bin = smem->threshold_bin;
    const auto v_hi = coarse_bin_lower_bound<kHistBits>(threshold_bin + 1);
    const auto v_lo = coarse_bin_lower_bound<kHistBits>(threshold_bin);
    const auto collect = [&](float val, uint32_t idx) {
      if (isnan(val)) {
        if (threshold_bin == 0 || threshold_bin == (1u << kHistBits) - 1u) {
          // Comparisons do not order NaNs; the exact radix fallback does.
          atomicExch(&smem->needs_exact, 1u);
        } else if ((__float_as_uint(val) & 0x80000000u) == 0) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          if (pos < topk) problem.emit(pos, idx);
        }
        // Negative NaNs lie below a numeric cutoff. This matters for V4:
        // actual FP4 logits can carry a negative NaN in column zero.
        return;
      }
      if (val >= v_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1);
        if (pos < topk) [[likely]]
          problem.emit(pos, idx);
      } else if (val >= v_lo) {
        // Keep nonfinite threshold candidates out of finite tie sorting.
        if (!isfinite(val)) atomicExch(&smem->needs_exact, 1u);
        const auto count_eq = atomicAdd(&smem->count_eq, 1);
        if (count_eq < kMaxNumTie) [[likely]]
          smem->tie.values[count_eq] = {val, idx};
      }
    };
#pragma unroll
    for (uint32_t i = 0; i < kLocalVecs; ++i) {
      const auto vi = tx + kBlockSize * i;
      const auto base = vi * kVecSize;
      if (vi >= num_full) break;
#pragma unroll
      for (uint32_t j = 0; j < kVecSize; ++j)
        collect(local_vecs[i][j], base + j);
    }
    if (tx >= kBlockSize - tail) {
      const uint32_t idx = tail_start + tx - (kBlockSize - tail);
      collect(problem.in[idx], idx);
    }

    // Phase 4: Handle ties.
    __syncthreads();
    const auto above_count = smem->count_gt;
    const auto equal_count = smem->count_eq;
    const auto remain_topk = above_count < topk ? topk - above_count : 0;
    // A coarse bin can contain more than the shared candidate capacity, even
    // when its fp32 values differ. Never truncate it and silently lose winners.
    if (smem->needs_exact || equal_count > kMaxNumTie || above_count > topk ||
        above_count + equal_count < topk) {
      exact_row_select(problem, &smem->tie.handle);
    } else if (remain_topk != 0) {
      handle_tie(smem->tie.values, problem, above_count, equal_count, remain_topk, &smem->tie.handle);
    }
  }
};

// ---------------------------------------------------------------------------
// Streaming path: seq_len > 8192 -- two vectorized passes over global memory
// ---------------------------------------------------------------------------

struct TopKStreaming : TopKRegister<2> {
 public:
  static constexpr uint32_t kMaxSeqLen = std::numeric_limits<uint32_t>::max();

  template <bool kUsePDL>
  ATOM_DEVICE static void forward(const TopKProblem problem, void* _smem) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);

    {
      Smem::kHistVec hist_vec;
      hist_vec.fill(0);
      smem->hist_vecs[tx] = hist_vec;
    }
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
      smem->needs_exact = 0;
    }
    __syncthreads();


    // Phase 1: Load and build histogram
    for_each_input(problem.in, problem.seq_len, [&](float val, uint32_t) {
      const auto bin = extract_coarse_bin<kHistBits>(val);
      atomicAdd(&smem->histogram[bin], 1);
    });
    __syncthreads();

    // Phase 2: Find the threshold bin
    find_threshold(problem.topk, problem.seq_len, smem);

    // Phase 3: Collect candidates and sort. Classify by two fp32 boundaries derived
    // from the threshold bin instead of recomputing the fp16 bin per element: an
    // element is "above" iff val >= v_hi (bin > threshold) and a "tie" iff
    // v_lo <= val < v_hi (bin == threshold). This drops the F2F + bit-twiddle from
    // the second full pass over the input.
    const auto threshold_bin = smem->threshold_bin;
    const float v_hi = coarse_bin_lower_bound<kHistBits>(threshold_bin + 1);
    const float v_lo = coarse_bin_lower_bound<kHistBits>(threshold_bin);
    const auto topk = problem.topk;
    for_each_input(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
      if (isnan(val)) {
        if (threshold_bin == 0 || threshold_bin == (1u << kHistBits) - 1u) {
          // Comparisons do not order NaNs; the exact radix fallback does.
          atomicExch(&smem->needs_exact, 1u);
        } else if ((__float_as_uint(val) & 0x80000000u) == 0) {
          const auto pos = atomicAdd(&smem->count_gt, 1);
          if (pos < topk) problem.emit(pos, idx);
        }
        // Negative NaNs lie below a numeric cutoff. This matters for V4:
        // actual FP4 logits can carry a negative NaN in column zero.
        return;
      }
      if (val >= v_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1);
        if (pos < topk) [[likely]] {
          problem.emit(pos, idx);
        }
      } else if (val >= v_lo) {
        // Keep nonfinite threshold candidates out of finite tie sorting.
        if (!isfinite(val)) atomicExch(&smem->needs_exact, 1u);
        const auto count_eq = atomicAdd(&smem->count_eq, 1);
        if (count_eq < kMaxNumTie) [[likely]] {
          smem->tie.values[count_eq] = {val, idx};
        }
      }
    });

    // Phase 4: Handle ties. Drive the output layout from the *collect* counts so it
    // is self-consistent with the fp32 classification above (rather than the fp16
    // histogram counts), even if rounding moves a boundary element between the
    // "above" and "tie" sets. above_count is < topk by the threshold-bin invariant,
    // so the count_gt guard above effectively never triggers.
    __syncthreads();
    const auto above_count = smem->count_gt;
    const auto equal_count = smem->count_eq;
    const auto remain_topk = above_count < topk ? topk - above_count : 0;
    // A coarse bin can contain more than the shared candidate capacity, even
    // when its fp32 values differ. Never truncate it and silently lose winners.
    if (smem->needs_exact || equal_count > kMaxNumTie || above_count > topk ||
        above_count + equal_count < topk) {
      exact_row_select(problem, &smem->tie.handle);
    } else if (remain_topk != 0) {
      handle_tie(smem->tie.values, problem, above_count, equal_count, remain_topk, &smem->tie.handle);
    }
  }
};


}  // namespace atom::v4_topk
