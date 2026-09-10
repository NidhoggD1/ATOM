# MiniMax-M3 — LMCache KV offload on the vLLM plugin (byte codec)

MiniMax-M3 cannot use LMCache's own GPU connector. This recipe adds
`AtomLMCacheOffloadConnector` to the accuracy server of
[MiniMax-M3](MiniMax-M3.md) §4.1: it drives ATOM's `DenseKVByteCodec` from
vLLM's KV-connector API and leaves LMCache as a pure byte store.

Everything from that recipe still applies — the same install (§1), the same
environment (§2), the same flags — except that the dense layers run
`ATOM_M3_DENSE_ATTN_BACKEND=aiter` rather than `gluon`. This page only adds the
offload tier.

For the generic plugin + `LMCacheConnectorV1` path (works on M2.5 and other
dense models), see [LMCache KV Cache Offload](LMCache-KV-Cache-Offload.md). That
path **does not work on M3** — see *Why a separate connector* below.

## Why a separate connector

M3 registers 117 KV tensors in three different physical layouts at once:

| layers | per-layer view | note |
|---|---|---|
| 3 dense | `(nb, 1, 128, 2*hd)` | `aiter`: one KV head per rank at TP4, K and V interleaved in the content dim (`gluon` instead asks for `num_head_slots=2`, giving `(nb, 2, 128, C)`; the codec takes either) |
| 57 sparse | `(nb, 2, 128, 128)` | K/V on the head-slot axis (`num_head_slots=2`), one KV head per rank at TP4 |
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

Under `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` the sparse backend publishes two
acceptable layouts (`MiniMaxM3SparseAttentionBackend.supported_kv_cache_layouts`
in `atom/plugin/vllm/attention/backend.py`) and vLLM resolves one at startup.
This is gated on the shuffle flag and on `num_kv_heads == 1` per rank only — it
does not depend on the dense backend, so the ambiguity is there under `aiter`
just as under `gluon` (the `gluon` dense backend publishes the same pair; the
`aiter` one publishes none and its layers always travel as one opaque run):

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
triggers that narrowing. Read the startup line rather than assuming --
`Inferact/vllm-m3-amd@8a9bad879` resolves `kv-split`, stock vLLM 0.28
resolves `kv-whole`.

## Launch

Take §4.1 of [MiniMax-M3](MiniMax-M3.md) and add the LMCache environment and the
`--kv-transfer-config` flag:

```bash
cd /root
rm -rf /root/.cache/atom/*

MODEL=/path/to/MiniMax-M3-MXFP8

# --- MiniMax-M3.md §4.1, with the dense backend on aiter ---
export ATOM_M3_DENSE_ATTN_BACKEND=aiter
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
export OFFLOAD_COPY_WORKERS=4        # default 1 is a throughput cliff, see below
export OFFLOAD_GPU_STAGING_CHUNKS=8  # default 2; give the 4 workers room to overlap
export LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3  # every rank answers, see below

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

**The symptom to look for is `need=128` on the declined loads.** Start with
`OFFLOAD_PROFILE=1` and read the `[OFFLOAD-LOAD-SKIP]` records:

```
[OFFLOAD-LOAD-SKIP] hbm_cached=29952 lmc_cached=30080 need=128 reason=too_small
[OFFLOAD-LOAD-SKIP] hbm_cached=3968  lmc_cached=4096  need=128 reason=too_small
```

A `need` of exactly one chunk means LMCache holds precisely what HBM holds. The
128 tokens are the fixed offset between vLLM's `num_cached_tokens`, which drops
the last block so there is a token left to recompute, and LMCache's
chunk-aligned hit — not real content. On the radix workload at
`--num-gpu-blocks-override 16612` this was 497 of 499 declines, with
`kv_cache_usage_perc` peaking at 0.49: the pool never filled, so vLLM's prefix
cache never evicted, so the CPU tier could never hold more than a subset of
HBM. No amount of `OFFLOAD_MIN_LOAD_TOKENS` tuning recovers this — 444 of
those lookups reported a hit above 8192 tokens and every one of them still had
`need=128`.

There is a floor on how far the pool can be capped: vLLM requires it to hold
one `--max-model-len` request, so `--num-gpu-blocks-override` cannot go below
`max_model_len / block_size` (7813 at 1M context and block 128). Below that the
engine refuses to start with a KV-cache-memory `ValueError`.

## The save pipe is the throughput knob

`OFFLOAD_COPY_WORKERS` (default **1**) is the width of the per-rank save
executor, and `OFFLOAD_GPU_STAGING_CHUNKS` (default **2**) the depth of the
staging buffer it copies through. On the gfx950 box these numbers come from,
one worker sustains only ~0.85 GB/s of device-to-host writes, which is below
what a busy M3 server produces.

The failure is not a slow tier, it is block starvation. `should_defer_free`
holds a finished request's KV blocks while a save is in flight **or merely
queued**, so once the save executor saturates, blocks stop returning to the
pool: `vllm:kv_cache_usage_perc` pins near 1.0,
`vllm:num_requests_waiting_by_reason{reason="capacity"}` climbs, and TTFT goes
to tens of seconds. It drains once the load stops — it is backpressure, not a
leak — which is why it only shows up under sustained concurrency.

Measured with the talos radix generator (MiniMax-M3-MXFP8, TP=4, MTP=7, 25
concurrent sessions, `--num-gpu-blocks-override 16612`, 60 GiB CPU tier,
150 s eval). "LMCache off" is the same server with `--kv-transfer-config`
removed:

| | off | on, defaults | on, `COPY_WORKERS=4` `STAGING=8` |
|---|---|---|---|
| logical TPM | 11,281,781 | 942,378 | 10,067,546 |
| decode TPS | 1,358.3 | 181.4 | 1,144.0 |
| TTFT p50 / p90 (ms) | 248 / 955 | 75,222 / 87,478 | 306 / 1,022 |
| `kv_cache_usage_perc` | 0.54 | 0.985 | 0.49 |
| waiting (capacity) | 0 | 22.2 | 0.25 |
| save duty cycle / rank | — | 100% | 46% |

Four workers were enough here (46% duty); raise it further only if
`[OFFLOAD-SAVE-PROF]` still shows the executor saturated. To read the duty
cycle, start with `OFFLOAD_PROFILE=1` and sum `store_ms` per rank over the
wall-clock window — note `toks` in that record is the cumulative save frontier
and `skip` the floor, so bytes written are `(toks - skip) * bytes_per_block /
block_size`, not `toks`.

The staging buffer is allocated per rank at `OFFLOAD_GPU_STAGING_CHUNKS x
bytes_per_block`; at `chunk=128` on M3 that is 3,008,512 B per chunk, so 8
chunks costs 24 MiB of device memory per rank. The startup line reports the
figure it actually used.

## Lookup scope: which rank answers "is it offloaded?"

Every TP rank stores its own KV shard, but by default only rank 0 answers the
LMCache lookup. That is safe while the ranks agree, and they do agree as long as
nothing evicts — this connector saves on all ranks in lockstep. Under
CPU-tier capacity pressure they stop agreeing: each rank evicts on its own
recency, so rank 0 can report a hit whose shards are already gone elsewhere,
and the load falls back to a full recompute after paying for the attempt.

```bash
export LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3
```

makes every rank answer. The minimum across ranks is the prefix all four shards
can actually restore, and each rank refreshes its own recency and pins its own
shard. Leave it unset to keep the historical rank-0 behaviour.

`LMCACHE_CACHE_POLICY=ATOM_SLRU` is available alongside it: new chunks enter a
probationary segment and only reused ones become protected, so one long scan
cannot flush the reusable prefixes. Both are opt-in; the defaults are rank-0
lookup and plain LRU.

Only synchronous lookup is supported. `LMCACHE_ENABLE_ASYNC_LOADING=true` is
rejected at startup rather than silently issuing duplicate lookups.

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
- **This connector needs a vLLM that resolves KV layouts** (the
  `vllm.v1.kv_cache_layout` resolver). On an older vLLM the caches are bound
  through the legacy 5-D path, where a dense layer arrives as `(2, nb, ...)` and
  its block count reads as 2 while the sparse layers still read `nb`; every
  segment would then be sliced at the wrong granularity with the right dtype and
  a plausible size. Registration refuses that outright — *registered layers
  disagree on the block count* — but the same vLLM cannot import M3's attention
  backend anyway, so in practice the server never gets that far.
- **Stop the server with `podman restart`, not `pkill`.** Killing TP workers
  leaves zombies holding GPU memory (82 GiB/card observed); only restarting the
  container releases it.

## Validation status

**Both layouts are now covered, and `kv-split` (LHBNC) is the one the customer
fork actually resolves.** The M3-AMD tree imports `vllm.v1.kv_cache_layout`,
which exists only in the vLLM this recipe family is built against
(`Inferact/vllm-m3-amd@8a9bad879`); stock vLLM 0.28 carries the older
`KVCacheLayoutType` string and no layout resolver, so
`atom/plugin/vllm/attention/backend.py` does not import there. That fork was
supplied as a source archive and built locally, so the server numbers below are
from the customer's own vLLM rather than a stand-in.

### Environment the numbers come from

| piece | version |
|---|---|
| base image | `vllm/vllm-openai-rocm:nightly-27a94d1ce4e3fc100c4732439ccec10f8246a804` (torch 2.12.0) |
| vLLM | `0.28.1.dev0+g76538dc71` built from `Inferact/vllm-m3-amd@8a9bad879` |
| AITER | public `ROCm/aiter@878d60d77` — **deviation**, `zejunchen-zejun/aiter-m3` is private and unreachable; this is the same build a previously working M3 container carried, and all four M3 symbols are present |
| LMCache | 0.5.3 — **deviation**, ROCm/ATOM#2146 validated against 0.4.5 |
| hardware | 8x gfx950, TP=4 per arm, `MiniMax-M3-MXFP8` |

The fork requires torch >= 2.11 (`csrc/libtorch_stable/cuda_view.cu` uses
`torch::stable::Tensor::layout`), which rules out the `rocm/atom-dev` images
shipping torch 2.10 — use the base image above.

### What the server reports at startup

Under `ATOM_M3_DENSE_ATTN_BACKEND=aiter` and
`VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1`, all four ranks print:

```
ATOM LMCache offload: registered 60 layers, num_blocks=512, layout=kv-split
ATOM LMCache offload:   57 x index tail_shape=(1, 128, 128) dtype=torch.float8_e4m3fn
ATOM LMCache offload:   3 x kv    tail_shape=(1, 128, 256) dtype=torch.uint8
ATOM LMCache offload:   57 x kv   tail_shape=(2, 128, 128) dtype=torch.uint8
LMCache offload worker rank=N: bytes_per_block=2958336 chunk=128 ...
LMCache offload scheduler: lookup client on atom-offload-dp0 (world=4)
```

117 tensors over 60 layers, `world=4` (the `tensor_parallel_size` the shim adds
-- without it the scheduler computes `world=1` while the workers compute 4).

### A/B against the same server with the connector removed

Single variable: two servers from the same container, identical flags and an
identical 65,536-token HBM pool (`--num-gpu-blocks-override 512`), differing
only in `--kv-transfer-config` and port. Workload: 24 distinct ~10.3K-token
prompts (~246K tokens, **3.8x the pool**, so round 2 cannot hit in HBM),
`max_tokens=1` to isolate prefill, 3 rounds, 3 repeats on different seeds.

Sizing that working set above the pool is the whole experiment. If the pool
holds the working set, the offload tier has nothing to do and will tie on hits
and lose on throughput.

| steady state (rounds 1-2) | no offload | offload |
|---|---|---|
| cached-token ratio | 0.0% | **79.8%** |
| of which served by the CPU tier | n/a | **all of it** |
| prefill throughput | ~37,590 tok/s (spread 0.7%) | **~62,260 tok/s** (spread 5.9%) |
| p50 request latency | 0.273-0.275 s | **0.158-0.170 s** |

The 79.8% is the offload tier's own hit rate, not a mix. Read straight off the
counters over a 3-round run (739,191 prompt tokens): `prefix_cache_hits_total`
moved by **0** -- vLLM's GPU prefix cache contributed nothing, because the
working set is 3.8x the pool -- while `external_prefix_cache_hits_total` moved
by 393,216, which equals the API's summed `cached_tokens` exactly. Per replay
round that is 196,608 / 246,397 = **79.79%**; including the cold round,
393,216 / 739,191 = 53.20%.

1.66x on replay. The cold round pays 8-9% for the save path, so end-to-end over
cold + 2 replays it is 1.30x. Both gaps are far outside the measured noise floor
(hit rate 2.64 pp, throughput <1%).

### Correctness

| check | result |
|---|---|
| gsm8k 5-shot, 5 runs, both arms replicated | offload mean 0.9527 vs no-offload mean 0.9526 -- see the table below |
| marker recall over a pure external hit | 8192 tokens loaded from LMCache after the prompt was evicted; the answer still names the marker |
| both layouts segment and move every byte | `DenseKVByteCodec` gather/scatter over a 60-layer / 117-tensor M3 census on GPU, KV + fp8 scales + index caches, byte-compared after wiping the gathered blocks |
| an LMCache engine stores and returns those bytes | real `build_offload_engine` + `BlockGPUConnector` + LMCache LocalCPU: all 23 segments byte-identical |
| the layout reaches the namespace | `build_page_namespace` yields a different key once `page_layout_tag` is set, and the unset key is byte-identical to the native path's |
| the rest | unit tests in `tests/plugin/` |

gsm8k in full, strict-match, 1319 questions per run. Both arms are replicated
so the between-arm gap can be read against each arm's *own* run-to-run spread;
the per-run stderr is a sampling bound and does not predict re-run variance.

| arm | run | strict-match | flexible-extract |
|---|---|---|---|
| no offload | A1 | 0.9553 +/- 0.0057 | 0.9545 |
| no offload | A2 | 0.9500 +/- 0.0060 | 0.9492 |
| offload | B1, cold | 0.9484 +/- 0.0061 | 0.9477 |
| offload | B2, replay (no cache clear) | 0.9492 +/- 0.0060 | 0.9484 |
| offload | B3, replay (no cache clear) | 0.9606 +/- 0.0054 | 0.9598 |

| | no offload | offload |
|---|---|---|
| mean | **0.9526** | **0.9527** |
| own spread (max-min) | 0.0053 | 0.0122 |

The between-arm gap is **0.0001** against a within-arm spread of up to
**0.0122**: the arms are indistinguishable, and turning the offload tier on
moves accuracy less than the baseline moves itself between two runs. The
highest of all five runs is an offload run. Replay does not lose accuracy
against its own cold run -- the signal that the fp8 scales travel with the
bytes.

Run each arm at least twice. A single pair (A1 vs B1) reads as a 0.7-point
regression; it is the noise floor, not a regression.

Do **not** gate on byte-identical continuations: this build is not
bit-reproducible across a cache hit even without the offload tier. The same
prompt sent twice to the *baseline* server, hitting only vLLM's own GPU prefix
cache, already returns a different (still correct) continuation. Judge a replay
on whether it stays coherent and recalls the content, and on aggregate accuracy.

### Two-pass check: does the *load* path itself change answers

The five runs above compare cold against replay, which is a weak test of the
load path -- a first pass only ever SAVEs. To force every prefix through the
offload tier, prefix each gsm8k question with a unique salt so no two prompts
share a prefix and the working set blows past the HBM pool, then replay the
identical prompts. Accuracy is the pass1 -> pass2 delta. The noise band must
come from the same two passes with the connector **off**, not from a per-run
stderr.

```bash
python3 tests/plugin/m3_twopass_accuracy.py --port 8902 --n 1319 --shots 5 --conc 32 --salt on_r1
python3 tests/plugin/m3_twopass_accuracy.py --port 8903 --n 1319 --shots 5 --conc 32 --salt off_r1
```

5-shot is enough on M3 (chunk 128, `OFFLOAD_MIN_LOAD_TOKENS=256`). Use a
distinct salt per run so runs cannot contaminate each other.

| run | arm | pass1 | pass2 | delta | pass2 external hits |
|---|---|---|---|---|---|
| on_r1  | offload    | 0.8074 | 0.8059 | **-0.0015** | 844,672 |
| on_r2  | offload    | 0.8180 | 0.8158 | **-0.0023** | 844,672 |
| off_r1 | no offload | 0.8165 | 0.8302 | **+0.0136** | 0 |
| off_r2 | no offload | 0.8241 | 0.8218 | **-0.0023** | 0 |

Both offload deltas sit inside the no-offload arm's own delta range
[-0.0023, +0.0136] (spread **0.0159**), and the four pass1 scores -- identical
no-load work in both arms -- span **0.0167**. The offload arm reloads 844,672
tokens per pass and moves accuracy by less than the baseline moves itself while
reloading nothing: the delta does not scale with reload volume.

The salted scores (~0.81) are lower than the 5-shot scores above because the
salt header perturbs the prompt; only the within-run delta is meaningful.

The codec self-check needs a GPU but no server:

```bash
PYTHONHASHSEED=0 LMCACHE_LOCAL_CPU=True LMCACHE_MAX_LOCAL_CPU_SIZE=4 \
LMCACHE_CHUNK_SIZE=128 HIP_VISIBLE_DEVICES=0 \
python3 tests/plugin/m3_offload_gpu_selfcheck.py
```

### Known behaviour: `load failed ...; recomputing`

Under concurrent chat traffic (`num_concurrent=32`) roughly 2% of requests log

```
ATOM LMCache offload: load failed for [...]; recomputing
```

on ranks 1..N-1 and never on rank 0. That asymmetry is the signature of
`lookup_server_worker_ids: [0]`: the scheduler asks rank 0's store whether the
tokens are there and every rank then tries to load them. When another rank's
store does not have them yet, it reports `failed_loading` and vLLM recomputes
that stretch. Accuracy is unaffected (the gsm8k replay above was measured with
these warnings present) and the cost is one recompute. It does not appear under
sequential long-prompt traffic. This is a property of the ATOM offload layer
this port sits on, not of the port.

The transfer tier above the codec (`build_offload_engine`, `BlockGPUConnector`,
LMCache itself) is untouched by this port and is what ROCm/ATOM#2146 validated
on the ATOM native stack.

## Related

- [MiniMax-M3](MiniMax-M3.md) — base serving and accuracy recipe (this page
  layers on its §4.1)
- [LMCache KV Cache Offload](LMCache-KV-Cache-Offload.md) — generic plugin path
  (`LMCacheConnectorV1`), does not support M3's layouts
