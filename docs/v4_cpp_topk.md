# Experimental V4 HIP TopK + CSA packing

This opt-in native ATOM FP4 decode path replaces AITER TopK plus the separate
CSA translation kernel with one HIP launch. It adapts SGLang's register /
streaming TopK selection and writes ATOM's head-CSA / tail-SWA layout. It
retains the raw int32 index result for diagnostics. Output indices are
unsorted; equal scores may select different indices from AITER or PyTorch.

The default remains AITER. FP8, prefill, and plugin-specific indexer paths
keep their existing implementation. Validation on MI355X (`gfx950`) covers
operator correctness and a directed 16c decode integration test. This remains
an experimental opt-in path; generation quality and full serving performance
have not been evaluated.

## Enable in a separate test server

```bash
export ATOM_V4_CPP_TOPK=1
# Keep the existing server arguments, including --index-cache-dtype fp4.
```

The extension is compiled at Indexer initialization, before graph capture,
using the installed ROCm PyTorch toolchain. Ninja and a compatible C++/HIP
compiler are required. No SGLang installation is required. The usual PyTorch
`TORCH_EXTENSIONS_DIR`, `MAX_JOBS`, and `PYTORCH_ROCM_ARCH` controls apply.
Set `ATOM_V4_CPP_TOPK=0` and restart the test server to use the baseline.
The extension build name hashes both translation units and both device
headers so a header-only change cannot reuse an old kernel. Sources are staged
in the extension cache; the installed package can remain read-only.

## Validation on an available GPU

Run from this checkout with its ATOM/AITER dependencies:

```bash
python -m pytest -q tests/test_v4_cpp_topk_schema.py tests/test_v4_cpp_topk.py
python benchmarks/benchmark_v4_cpp_topk.py --rows 64 --lengths 8192 8215 16384
```

For finite inputs, the GPU suite compares selected score multisets against
`torch.topk`. NaN cases use AITER's ordered-FP32 radix convention: negative
NaNs rank below negative infinity; positive NaNs rank above positive infinity.
The suite checks
uniqueness and per-row visibility, exercises register / streaming boundaries,
large coarse-bin overflow, ties and nonfinite values, and checks physical
mapping, SWA preservation, padding and canaries. Graph replay changes lengths,
positions and tables in place. A compile test discards the raw result to check
that the packed-output mutation survives optimization. CPU-only schema tests
can run with `HIP_VISIBLE_DEVICES=`; the GPU suite explicitly skips.

The benchmark reports AITER TopK, AITER TopK plus translation, the new raw
TopK, and the fused operation using identical inputs and CUDA graph timing.
`--rows` counts query-token rows, not concurrent requests: C16 with DSpark3
uses 64 rows. The 8192/8215 cases straddle a register-selection boundary.
Random-input microbenchmarks do not reproduce the full model's score
distribution. These are operator timings, not full decode speedup.

The latest GPU suite passed 89 tests with PyTorch
`2.10.0+rocm7.2.4.git3d3aa833` and HIP `7.2.53211`. Replaying 30 actual
FP4 indexer layers (64 rows each, visible lengths 8212-8215) passed selection,
uniqueness, visibility, packed-address mapping and SWA-preservation checks.
Each captured row contained a negative NaN at column zero; the new kernel
excludes it in agreement with native AITER. Its upstream cause is not assumed.
Same-input CUDA-graph replay averaged 10.853 us for AITER TopK plus translation
and 8.125 us for the fused operator, saving 2.728 us per layer (25.1%).

The directed 16c server test used TP8, DSpark3, FP4 index cache, FP8 KV cache,
updated ROCm, FULL capture and no TBO. Both variants used the same prompts and
server settings, with 32793 input tokens and 1024 requested output tokens per
request. Trace annotations were `decode[bs=16 tok=64 d=16 spec=3]`.

| Metric | AITER baseline | Fused HIP |
| --- | ---: | ---: |
| TopK plus CSA translation per layer | 15.009 us | 10.586 us |
| 30 layers per target graph | 450.272 us | 317.569 us |
| Kernels per target graph | 2386 | 2356 |
| Mean target graph span | 19.857 ms | 19.509 ms |

The operator group saved 4.423 us per layer (29.5%) and 132.703 us per target
graph. The separate translation kernel disappeared on all eight ranks.
The fused kernel, which includes packing, cost about the same as baseline
TopK alone (10.592 us); the observed gain primarily removes the separate
translation cost.

These figures exclude boundary and incomplete captures and include 117
baseline and 128 fused rank-local graph samples across eight ranks. They are
descriptive measurements from one short A/B trace, not independent trials.
The observed target-graph reduction was 1.75%, while the operator's direct
saved time was 0.67% of baseline graph span; the remaining difference cannot
be attributed to TopK alone. No full throughput benchmark was run. Forced
speculative acceptance was enabled in both runs, so generated synthetic text
does not validate generation quality.

Before promoting the flag, evaluate generation quality without forced
speculative acceptance and validate PIECEWISE capture separately.

## Interface and limits

- Float32 `[rows, width]` scores, contiguous columns; odd row strides work.
- Int32 per-row lengths; reads only `[0, clamp(length, 0, width))`.
- `1 <= k <= 2048`; unused raw slots contain -1.
- Int32 metadata/page tables and packed output; positions may be int32 or int64.
- Packed slices must be disjoint, metadata must match score visibility, and
  physical rows must fit int32. Invalid rows (`batch_id = -1`) do not write.
- CSA starts at `indptr[row]`. Its length is the slice length minus
  `min(position + 1, window_size)`; the following SWA tail is preserved.
- The fused custom op explicitly declares its packed buffer mutation.
- NaNs follow ordered-FP32 radix order, including payload ordering when a
  NaN bucket is the selection cutoff. Equal finite scores can select any tied
  index; output order is unspecified.
- Overflowing coarse bins, NaN cutoffs and unsafe nonfinite threshold
  candidates use exact full-row radix selection. A negative NaN below a
  finite cutoff does not force that fallback. Adversarial inputs may be slower.

Attribution and the Apache-2.0 license for adapted device code are in
`atom/model_ops/v4_kernels/csrc/THIRD_PARTY_NOTICES.md`.
