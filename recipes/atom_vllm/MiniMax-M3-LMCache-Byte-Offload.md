# MiniMax-M3 — LMCache KV offload on the vLLM plugin (byte codec)

MiniMax-M3 cannot use LMCache's own GPU connector. This recipe adds
`AtomLMCacheOffloadConnector` to the accuracy server of
[MiniMax-M3](MiniMax-M3.md) §4.1: it drives ATOM's `DenseKVByteCodec` from
vLLM's KV-connector API and leaves LMCache as a pure byte store.

Everything from that recipe still applies — the same install (§1), the same
gluon environment (§2), the same flags. This page only adds the offload tier.

For the generic plugin + `LMCacheConnectorV1` path (works on M2.5 and other
dense models), see [LMCache KV Cache Offload](LMCache-KV-Cache-Offload.md). That
path **does not work on M3** — see *Why a separate connector* below.

## Why a separate connector

M3 registers 117 KV tensors in three different physical layouts at once:

| layers | per-layer view | note |
|---|---|---|
| 3 dense | `(nb, 2, 128, C)` | gluon: K/V on the head-slot axis (`num_head_slots=2`), `C = num_kv_heads * head_size` |
| 57 sparse | `(nb, 2, 128, 128)` | same, one KV head per rank at TP4 |
| 57 index caches | `(nb, 1, 128, 128)` fp8 | DSA indexer keys, registered as `<layer>.index_cache` |

LMCache's `normalize_kv_and_discover_format()` probes for one **global** format,
so it aborts with `currently unsupported kv_caches format with list depth 1 and
tensor dimension 4`. The per-layer-format connector (V3) is off by default and
hangs on M3; the multi-process path needs cupy, which LMCache's `platform/rocm`
does not provide.

`AtomLMCacheOffloadConnector` sidesteps the whole question: ATOM gathers whole
paged blocks into a chunk-major uint8 blob and LMCache only ever stores opaque
bytes, so no format probe ever runs.

M3's fp8 KV scales live on the layer, not in vLLM's `kv_caches` dict, and are
fetched through `get_kv_transfer_scales()` — moving mantissas without them
dequantises a restored block against the previous occupant's scale, which is
silent corruption.

### Two KV layouts, and why the namespace records which one

Under `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` both M3 backends publish two
acceptable layouts (`supported_kv_cache_layouts` in
`atom/plugin/vllm/attention/backend.py`) and vLLM resolves one at startup:

| resolved | memory order | whole tensor contiguous | segmentation used |
|---|---|---|---|
| `LHBNC` | `(2, B, N, C)` | no | split into `k_cache` / `v_cache` |
| `LBHNC` | `(B, 2, N, C)` | yes | one opaque run per block |

They carry the same shape, so `split_kv_tensor` decides by contiguity, not by
shape. Both move the same bytes per block, but **arranged differently**, so the
resolved layout is folded into the offload namespace (`page_layout_tag`,
logged at registration as `layout=kv-split` / `layout=kv-whole`). Two servers
that resolved different layouts therefore miss each other's entries on a shared
disk or remote backend instead of reading them as their own.

Which one a given build resolves depends on whether its vLLM exempts a single
uniform type group from mixed-HNC narrowing; M3's key-only indexer spec
triggers that narrowing. Read the startup line rather than assuming.

## Launch

Take §4.1 of [MiniMax-M3](MiniMax-M3.md) and add the LMCache environment and the
`--kv-transfer-config` flag:

```bash
cd /root
rm -rf /root/.cache/atom/*

MODEL=/path/to/MiniMax-M3-MXFP8

# --- unchanged from MiniMax-M3.md §4.1 ---
export ATOM_M3_DENSE_ATTN_BACKEND=gluon
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
export VLLM_USE_V2_MODEL_RUNNER=0
export ATOM_M3_UNIFORM_BATCH_CAPTURE=1
export SAFETENSORS_FAST_GPU=1
export NCCL_SOCKET_IFNAME=lo
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_LOG_LEVEL=WARNING
export PYTHONNOUSERSITE=1
export VLLM_DO_NOT_TRACK=1

# --- added for the offload tier ---
export PYTHONHASHSEED=0              # mandatory, see Gotchas
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=20 # GiB **per TP rank**
export LMCACHE_CHUNK_SIZE=128        # must equal --block-size
export OFFLOAD_MIN_LOAD_TOKENS=256   # default 8192 disables the tier for chat-sized prompts

vllm serve "$MODEL" \
    --served-model-name minimax-m3 \
    --host 0.0.0.0 --port 8902 \
    --language-model-only \
    --tensor-parallel-size 4 \
    --no-trust-remote-code \
    --max-model-len 131072 \
    --block-size 128 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 88 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --enable-prompt-tokens-details \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4, "text_config": {"use_index_cache": true, "index_topk_freq": 4}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":512}' \
    --kv-transfer-config '{"kv_connector":"AtomLMCacheOffloadConnector","kv_connector_module_path":"atom.plugin.vllm.kv_transfer.connector","kv_role":"kv_both"}'
```

Select the connector through vLLM's **out-of-tree entry point** (the
`kv_connector_module_path` above). vLLM validates `kv_transfer_config` while
building VllmConfig, which happens *before* platform plugins load, so naming the
connector without the module path fails config validation.

`--enable-prompt-tokens-details` is not in §4.1; it is what makes the client-side
cached-token column non-empty.

## Verify it is actually on

Four independent checks — all of them, because each can pass for the wrong reason:

```bash
# 1. vLLM's own factory, on every worker AND the EngineCore
grep "Creating v1 connector with name: AtomLMCacheOffloadConnector" server.log

# 2. the codec saw the whole model, and which layout it resolved
grep "ATOM LMCache offload: registered 60 layers" server.log
# ... num_blocks=<N>, layout=kv-split  (LHBNC)  or  layout=kv-whole  (LBHNC)

# 3. every rank built an engine, and the scheduler sized the replica correctly
grep "LMCache offload worker rank=" server.log      # expect 4 lines at TP4
grep "LMCache offload scheduler" server.log          # expect world=4 at TP4

# 4. the tier is queried AND returns data (must be > 0)
curl -s localhost:8902/metrics | grep -E 'external_prefix_cache_(hits|queries)'
```

The registration census that follows check 2 must show **117 tensors over 60
layers** (3 dense + 57 sparse + 57 index caches), each sparse layer carrying a
`k_scale`/`v_scale`.

Two identities must hold on any interval; they catch a miscounting tier that
still looks plausible:

```
prefix_cache_queries - prefix_cache_hits == external_prefix_cache_queries
prefix_cache_hits    + external_prefix_cache_hits == prompt_tokens_cached
```

The first says the two tiers are strictly serial (HBM first, LMCache only gets
what HBM missed); the second says nothing is double-counted.

## Accuracy check

Run §4.2 of [MiniMax-M3](MiniMax-M3.md) (gsm8k 5-shot, port 8902) **twice**:
once from a cold start (`rm -rf /root/.cache/atom/*`), then again without
clearing LMCache. The two scores must match and the second run must show
non-zero `external_prefix_cache_hits`. A replay that loses points is almost
always fp8 scales not travelling with the bytes — check that
`get_kv_transfer_scales` is present on the sparse layer.

## Sizing: an oversized KV pool makes this tier look useless

```
free for caching = KV pool - concurrency x ISL
```

If HBM alone holds the working set, LMCache is queried and correctly returns
nothing — the tier is live, it simply has nothing to offer. Cap the pool
(`--num-gpu-blocks-override`) below the working set before concluding anything
about it. On the ATOM native backend the same codec measured 76.65% total cache
read with a capped pool against 56.81% on a larger one (ROCm/ATOM#2146) — same
code, same workload, sizing the only difference.

## Gotchas

- **`PYTHONHASHSEED=0` is mandatory.** Without it each TP rank derives a
  different cache key for the same prompt and the hit ratio collapses to 0.
- **`LMCACHE_CHUNK_SIZE` must equal `--block-size` (128).** ATOM refuses a load
  whose HBM frontier is not chunk-aligned; with prefix caching on that frontier
  is block-aligned, so at chunk 256 roughly every other hit is dropped.
- **`OFFLOAD_MIN_LOAD_TOKENS` defaults to 8192**, which is above every prompt in
  a chat-sized workload — the tier would never serve anything.
- **`LMCACHE_MAX_LOCAL_CPU_SIZE` is per rank.** TP4 x 20 GiB locks 80 GiB of
  pinned memory; if free memory is low the allocation reclaims page cache and
  each worker can take minutes. Startup looks hung — `EngineCore` prints
  `No available shared memory broadcast block found in 60 seconds` once a minute.
  That line alone is **not** a failure: a healthy startup prints it 5 times too.
  Check whether the workers are still emitting log lines instead.
- **Saves are fire-and-forget.** A request returning does not mean its KV has
  landed. Benchmarks that measure immediately after warm-up systematically
  under-report external hits; allow a settle period.
- **Do not share a disk or remote LMCache backend between servers built from
  different vLLM commits** without checking the `layout=` line on both. The
  namespace separates the two known layouts, but it can only separate what it is
  told about.
- **Stop the server with `podman restart`, not `pkill`.** Killing TP workers
  leaves zombies holding GPU memory (82 GiB/card observed); only restarting the
  container releases it.

## Validation status

- `layout=kv-whole` (LBHNC) is exercised end-to-end.
- `layout=kv-split` (LHBNC) is covered by unit tests
  (`tests/plugin/test_vllm_kv_cache_layout.py`, synthetic strides) but has not
  been run on a server, because the vLLM build that resolves it
  (`Inferact/vllm-m3-amd@8a9bad879`) was not reachable from here. The code path
  it takes is the one PR ROCm/ATOM#2146 validated on the ATOM native stack.

## Related

- [MiniMax-M3](MiniMax-M3.md) — base serving and accuracy recipe (this page
  layers on its §4.1)
- [LMCache KV Cache Offload](LMCache-KV-Cache-Offload.md) — generic plugin path
  (`LMCacheConnectorV1`), does not support M3's layouts
