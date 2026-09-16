# DSv4 native checkpoints with LMCache MP

The native ATOM `lmcache_mp` connector supports DSv4 PAGE KV and compact
`dsv4-paged-state-v3` checkpoints. It pins existing READY checkpoint PAGE units;
it does not take another snapshot of the request's Active SLOT. Every request
keeps its normal fixed SLOT while running.

## Run

Install the matching ATOM and LMCache changes. The LMCache build must include
per-group `null_block_id` and the `get_server_config` capability query. Run the
MP server on the same host, with GPU IPC access to the ATOM worker allocations:

```bash
lmcache server --host 127.0.0.1 --port 5555 \
  --chunk-size 256 --separate-object-groups \
  --supported-transfer-mode lmcache_driven --l1-size-gb 64
```

Add the following options to an otherwise working native DSv4 launch:

```bash
export LMCACHE_CHUNK_SIZE=256
export OFFLOAD_MAX_PENDING_SAVES=2

python -m atom.entrypoints.openai_server \
  --model deepseek-ai/DeepSeek-V4-Pro --kv_cache_dtype fp8 -tp 8 \
  --enable_prefix_caching --state-checkpoint-interval-tokens 8192 \
  --kv-transfer-config '{
    "kv_connector": "lmcache_mp",
    "kv_role": "offload",
    "kv_connector_extra_config": {
      "lmcache.mp.host": "tcp://127.0.0.1",
      "lmcache.mp.port": 5555,
      "lmcache.mp.tp_rank_collapse": false
    }
  }'
```

The ATOM configured chunk size must equal the MP server's chunk size. Both
must align to ATOM's PAGE/hash block size. Native checkpoints are produced by
ATOM's existing checkpoint policy, so their cadence must provide the desired
reusable boundaries. A prefix is loadable only where PAGE KV and a complete
STATE checkpoint both exist on all TP ranks.

`lmcache.mp.max_pinned_state_bytes` optionally limits native checkpoint sources
and temporary restore images together. Its default is
`OFFLOAD_MAX_PENDING_SAVES * units_per_checkpoint * page_unit_bytes`, per TP
worker's geometry. PAGE KV sources continue to use normal request ownership.
Candidates consume no image pin until admission. The shared save limit defaults
to `max(2, 2 * OFFLOAD_COPY_WORKERS)` when not configured.

The namespace includes model/PAGE geometry, TP size, speculation configuration,
native layout and image sizes, Hugging Face commit identity when available,
and `lmcache.mp.model_revision` if supplied. Set that revision string when
replacing weights in an existing local model directory. Local directory names
alone cannot identify changed weight contents.

## Lifetime and representation

Engine group 0 contains PAGE views and declares `null_block_id=None`, so PAGE 0
is ordinary data. Native image ordinal `j` uses engine group `1+j`, aliases the
same PAGE allocation, and declares a one-chunk recurrent window with null ID
`-1`. Every ordinal is present at the checkpoint endpoint; earlier chunks use
all-null STATE groups. LMCache groups all ordinals into one STATE object.
The final image region is trimmed at `image_bytes`, preserving the original
physical PAGE stride.

The scheduler dispatches one combined PAGE/STATE save generation at a time per
request, with round-robin admission and count/byte bounds. It acquires the exact
READY image only after admission. An IPC producer event orders MP reads after
native checkpoint creation. A terminal completion from every TP rank releases
the native source and settles the logical operation. Failed saves roll back the
watermark for at most three attempts at that boundary.

Restore reserves fresh PAGE units plus the request's already allocated fixed
SLOT. After MP H2D finishes, the worker invokes the native image-to-SLOT codec.
The request wakes and the temporary units release only after local restore is
complete. An aborted request keeps its allocations until the same exact
completion arrives. Transport exceptions without proof of device completion
retain the lease; elapsed time and server heartbeat failure do not free DMA
sources or destinations.

## Initial scope

- Native ATOM, one MP server, TP only. DP/PP/PCP/DCP and engine-driven transfers
  are rejected. DSv4 TP rank collapse is disabled because STATE is rank-specific.
- External restore is used when the actual post-allocation HBM hit is zero.
  Requests with a local HBM prefix follow normal local prefill. Lookups truncate
  the token list to `floor((prompt_tokens - 1) / chunk_size) * chunk_size` before
  querying, so the restored state agrees with the resumed token boundary.
- The native restore callback currently shares its descriptor buffer with
  forward. Local stream synchronization before and after restore prevents host
  descriptor reuse races; network transfer remains asynchronous.
- Finished requests currently retain their normal PAGE/SLOT allocations while
  a PAGE save is pending. Releasing their SLOT earlier needs a separate native
  PAGE lease path. Running requests never give their SLOT to the MP connector.
- The transport aliases native buffers directly, but LMCache may use its own
  GPU transfer buffers internally. This removes an additional ATOM SLOT image,
  rather than promising a completely copy-free transport.

## Validation

CPU contracts cover exact READY leases, generation replay, eviction and reset,
byte budgets, fair admission, cancellation, failures, full-prompt boundaries,
native image byte order and strided tail registration. LMCache tests cover
null markers, serialization, sparse STATE lookup and capability negotiation.

Run the real independent-process CUDA/ROCm transport test from the matching
LMCache checkout:

```bash
python -m pytest -xvs tests/v1/multiprocess/test_native_state_alias_gpu.py
```

It uses synthetic cache contents and does not establish full DSv4-Pro model
accuracy or serving performance. Those require a model run with forced HBM
misses, prefix reuse, TP quorum, cancellation, and comparison to fresh prefill.
