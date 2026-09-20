# Third-party attribution

`topk_impl.cuh` and the selection dispatch in `v4_topk.cu` are adapted
from SGLang, licensed under Apache-2.0, at commit
`7f1f8c706ac000b7a84ea0bda05135fc4177c6ca`:

- https://github.com/sgl-project/sglang/blob/7f1f8c706ac000b7a84ea0bda05135fc4177c6ca/python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh
- https://github.com/sgl-project/sglang/blob/7f1f8c706ac000b7a84ea0bda05135fc4177c6ca/python/sglang/kernels/jit/csrc/deepseek_v4/topk_v2.cuh

Upstream also credits these Apache-2.0 reference implementations:

- vLLM `csrc/persistent_topk.cuh`, commit `a8c6ee9b787d273916206a29b77feebadb80c368`
- FlashInfer `include/flashinfer/topk.cuh`, commit `c2b4db2b1a84448d802f0e6ac445243312bd6a4c`

The complete Apache-2.0 license is in `LICENSE.apache-2.0` beside this
file. ATOM's other files retain their existing MIT license.

Local changes: standalone HIP helpers with logical 32-lane groups on
wave64, raw-index output and ATOM CSA address packing, bounded per-row
visibility and padding, exact whole-row fallback for candidate overflow
and nonfinite scores, and PyTorch stream/extension integration. CUDA
cluster and programmatic launch dependencies were removed. These files
are modified derivatives, not unmodified upstream distributions.
