# MiniMax-M3 with the ATOM vLLM Plugin Backend

This recipe covers the source installation of vLLM, AITER and ATOM, the server
commands, and the accuracy-validation commands for the MiniMax-M3 gluon path.

Base image: `vllm/vllm-openai-rocm:nightly-27a94d1ce4e3fc100c4732439ccec10f8246a804`
(digest `sha256:91e381f072d6a44e1e4c97c82dce06e50e5189905cb3999a11471c5a8fc6a563`).

Path macros used below:

```bash
MODEL=/path/to/MiniMax-M3-MXFP8            # M3 MXFP8 checkpoint
DRAFT=/path/to/MiniMax-M3-EAGLE3-GQA       # EAGLE3 GQA draft (decode spec only)
```

---

## 1. Installation

### 1.1 vLLM — `Inferact/vllm-m3-amd`, commit `8a9bad879`

```bash
git clone git@github.com:Inferact/vllm-m3-amd.git vllm-m3-amd
cd vllm-m3-amd
git checkout 8a9bad879

export PYTORCH_ROCM_ARCH=gfx950 GPU_ARCHS=gfx950 MAX_JOBS=64 \
       CMAKE_BUILD_TYPE=Release VLLM_TARGET_DEVICE=rocm
python3 use_existing_torch.py          # keep the image torch
python3 -m pip install -e . --no-build-isolation -v
```

### 1.2 AITER — `zejunchen-zejun/aiter-m3`, branch `main`

```bash
git clone -b main git@github.com:zejunchen-zejun/aiter-m3.git aiter
cd aiter
PREBUILD_KERNELS=0 python3 -m pip install -e . --no-build-isolation --no-deps -v
```

### 1.3 ATOM — `zejunchen-zejun/ATOM-m3`, branch `M3-AMD`

```bash
git clone -b M3-AMD git@github.com:zejunchen-zejun/ATOM-m3.git ATOM
cd ATOM
# --no-deps keeps the transformers the image ships and vLLM was built against
python3 -m pip install -e . --no-build-isolation --no-deps -v
python3 -m pip install --no-deps pybind11 zmq msgspec xxhash setproctitle openpyxl
```

### 1.4 lm-eval

```bash
python3 -m pip install "lm_eval[api]"
```

---

## 2. Environment variables

| Variable | Value | Purpose |
|---|---|---|
| `ATOM_M3_DENSE_ATTN_BACKEND` | `gluon` | Routes M3's 3 dense layers onto AITER's shuffle kernels (`flash_attn_varlen` for prefill, `pa_decode_gluon` for decode) instead of vLLM's Triton `unified_attention`, and requests a K/V-separated KV cache. |
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` | `1` | Required by gluon — enables the page-16 SHUFFLE KV layout the shuffle kernels read. |
| `VLLM_USE_V2_MODEL_RUNNER` | `0` | ATOM's speculative-decoding integration (EAGLE3 and MTP alike) patches vLLM's V1 proposer; M3 defaults to V2 on ROCm, which bypasses it. |
| `ATOM_M3_UNIFORM_BATCH_CAPTURE` | `1` | Lets spec-verify batches (`query_len > 1`) be CUDA-graph captured under `FULL_DECODE_ONLY`; without it decode falls back to eager. |
| `USE_ATOM` | `1` | Activates the ATOM plugin inside vLLM. |
| `ATOM_FORCE_ATTN_TRITON` | `1` | Forces the Triton path for the layers gluon does not cover. |
| `VLLM_SERVER_DEV_MODE` | `1` | Mounts `POST /reset_prefix_cache`, needed to start a benchmark from a cold prefix cache (§3.2). |

---

## 3. Multi-turn throughput benchmark

End-to-end serving benchmark on a multi-turn session replay (long shared
prefixes, prefix-cache dominated).

**Result being reproduced** — 4x MI355X (gfx950), single node, TP4, concurrency
25, 300 s warmup + 600 s eval window:

| Draft | Accept len<br>configured / **engine** / client | Device KV pool | Host KV pool | GPU hit | CPU hit | Total hit | TPM/GPU | Extend TPM | Decode TPS | TTFT p50 | TPOT p50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MTP (NextN head), dummy weights | 4.0 / **4.000** / 4.06 | **5,891,590** tok<br>(engine default) | **0** | **92.39%** | 0.00% | **92.39%** | **2,754,716** | **832,937** | **1,160.2** | **345 ms** | **11.5 ms** |

Raw counters behind that row, so a repro can be checked against it: 1479 / 1479
requests succeeded, eval `duration` 621.085 s, prompt 113,340,290 tok, cache-hit
104,718,208 tok, extend 8,622,082 tok, decode 720,557 tok, mean prompt 76,501
tok, `vllm:num_preemptions_total` 0.

Three things about that table:

- **Acceptance is forced, not earned.** `rejection_sample_method: "synthetic"` +
  `synthetic_acceptance_length: 4.0` makes the verifier accept a mean of exactly
  4.0 of the 7 drafted tokens, so draft *quality* never enters the measurement —
  which is why `draft_load_config.load_format: "dummy"` (a randomly initialised
  NextN head) is fine here, and why the generated text is meaningless by design.
  The **engine** column is `1 + accepted/drafts` read off vLLM's own counters and
  is the only cross-engine comparable one (§3.3). The `4.06` is the load
  generator's client-side estimate (`completion_tokens / sse_chunk_count`); vLLM's
  API server merges several verify steps into one SSE chunk, so on some requests
  that estimate exceeds the theoretical ceiling of `num_speculative_tokens + 1 =
  8`. Do not draw conclusions from it.
- **Do not pass `--num-gpu-blocks-override`.** This arm lets the engine size the
  device pool itself: 126.43 GiB/rank of KV memory -> 46,031 blocks ->
  **5,891,590 tokens**. At concurrency 25 the in-flight working set is 1.91 M
  tokens (32 % of the pool) and `vllm:num_preemptions_total` stays 0 all run.
  Pinning the pool to 2,126,199 tokens (`--num-gpu-blocks-override 16612`, which
  is only useful when matching another engine's pool size) costs 14.2 pp of hit
  rate and 41.6 % of TPM/GPU on this workload.
- **No host-side KV tier.** No `--kv-transfer-config`, no `LMCACHE_*`. The 92.39 %
  is device prefix cache only; host KV usage is 0 by construction, so the CPU-hit
  column is structurally zero rather than merely unmeasured. Do not compare it
  against another engine's two-tier "cache hit" number without splitting that
  number into its device and host parts first.

### 3.1 Server

```bash
cd /root                             # NOT /app -- see the pitfalls below
rm -rf /root/.cache/atom/*

MODEL=/path/to/MiniMax-M3-MXFP8

export USE_ATOM=1
export ATOM_M3_DENSE_ATTN_BACKEND=gluon
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
export ATOM_FORCE_ATTN_TRITON=1
export VLLM_USE_V2_MODEL_RUNNER=0
export ATOM_M3_UNIFORM_BATCH_CAPTURE=1
export VLLM_SERVER_DEV_MODE=1        # mounts POST /reset_prefix_cache
export SAFETENSORS_FAST_GPU=1
export NCCL_SOCKET_IFNAME=lo
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_LOG_LEVEL=WARNING
export PYTHONNOUSERSITE=1
export VLLM_DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1


vllm serve "$MODEL" \
    --served-model-name minimax-m3-mxfp8 \
    --host 0.0.0.0 --port 8043 \
    --language-model-only \
    --tensor-parallel-size 4 \
    --no-trust-remote-code \
    --max-model-len 1000000 \
    --block-size 128 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 80 \
    --max-num-batched-tokens 32768 \
    --enable-prefix-caching \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4, "text_config": {"use_index_cache": true, "index_topk_freq": 4, "num_nextn_predict_layers": 1}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":640}' \
    --speculative-config '{"method":"mtp","num_speculative_tokens":7,"draft_load_config":{"load_format":"dummy"},"rejection_sample_method":"synthetic","synthetic_acceptance_length":4.0}'
```

Notes on the non-obvious flags:

- `"num_nextn_predict_layers": 1` in `--hf-overrides` is **mandatory for `method: mtp`
  on the MXFP8 checkpoint**. ATOM's generated draft config carries the key, but
  `model_wrapper.py` overwrites the draft's `hf_config` with the target's, and
  `MiniMaxM3MultiTokenPredictor.__init__` then reads `config.num_nextn_predict_layers`
  off the MXFP8 config, which only declares `text_config.num_mtp_modules = 1`.
  Without the override the server dies at load with
  `AttributeError: 'MiniMaxM3TextConfig' object has no attribute 'num_nextn_predict_layers'`.
  The two names are the same quantity (1 MTP layer, reused across the 7 steps),
  so this is an alias, not a behaviour change. The MXFP4 export declares
  `num_nextn_predict_layers` directly, which is why it starts without the override.
- `num_speculative_tokens: 7` needs an ATOM new enough to derive the sequential
  drafter's depth from the draft config
  (`max_spec = max(4, draft_cfg.num_mtp_modules)`). Older revisions hard-code
  `max_spec = 4` and reject the config outright.
- `max_cudagraph_capture_size: 640` = `80 x (7+1)`, the spec-verify batch width.
  A smaller value silently drops decode out of CUDA graphs.
- Use `synthetic_acceptance_length`, **never** hand-written
  `synthetic_acceptance_rates`: the latter are *unconditional marginals* (mean
  length = `1 + sum`, and they must be non-increasing), which is very easy to get
  wrong. The two keys are mutually exclusive; the valid range here is
  `[1, num_speculative_tokens + 1] = [1, 8]`.


### 3.2 Load generator

Run on the host. The workload is an open-loop replay of recorded agentic
sessions: each session's turns are issued with the original inter-turn gaps
(capped at 3 s), and new sessions start as needed to keep 25 concurrent.

| Item | Value |
|---|---|
| Load generator | `talos2-throughput-rust:radix-v2-20260820`, image ID `660152926bb3529ec3c7656b77db054fc17af23068e42a38238636f2b52cb57c` |
| Dataset | `talos_radix_sessions.json.gz`, SHA256 `18056a20aafab7320ee6f4a4138b8ceec35d33cb7aca7a4307ed54a97f3ef814`, 934,515 sessions |
| Window | warmup 300 s / eval 600 s / grace 30 s, seed 2026 |

```bash
ENDPOINT=http://127.0.0.1:8043
OUT=./results/c25
mkdir -p "$OUT"
MODEL=$(curl -fsS "$ENDPOINT/v1/models" \
        | python3 -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')

podman run --rm --network host \
  -v /path/to/talos_radix_sessions.json.gz:/data/radix.json.gz:ro \
  -v "$OUT":/output \
  talos2-throughput-rust:radix-v2-20260820 \
  /app/request-sim \
    --dataset-format radix-json --dataset-path /data/radix.json.gz \
    --api openai-completions --endpoint "$ENDPOINT" \
    --model-name "$MODEL" --stream \
    --workload-mode open --new-session concurrent \
    --within-session-gap original --max-within-session-gap-ms 3000 \
    --max-concurrent-session 25 \
    --warmup-time-in-seconds 300 \
    --eval-time-in-seconds 600 --grace-period-secs 30 --seed 2026 \
    --output-path /output/results.jsonl > "$OUT/talos.log" 2>&1 &

# 2. `duration` in results.summary.json covers the EVAL window only, so the
#    baseline snapshot must be taken the instant warmup ends.
until grep -qa 'warmup phase ended' "$OUT/talos.log"; do sleep 1; done
curl -fsS "$ENDPOINT/metrics" > "$OUT/metrics.before.prom"

wait
until idle; do sleep 5; done
curl -fsS "$ENDPOINT/metrics" > "$OUT/metrics.after.prom"
```

### 3.3 Deriving the numbers

Every throughput figure is `delta = after - before` over the two Prometheus
snapshots, divided by `duration` from `results.summary.json` (a Go duration
string such as `10m21.085s`, so it needs parsing). TTFT/TPOT percentiles come
straight out of `results.summary.json`.

```
prompt  = delta(vllm:prompt_tokens_total)
decode  = delta(vllm:generation_tokens_total)
gpu_hit = delta(vllm:prompt_tokens_cached_total)
cpu_hit = 0                                  # no host tier in this arm
extend  = prompt - gpu_hit

GPU hit    = gpu_hit / prompt                # 92.39 %
Total hit  = (gpu_hit + cpu_hit) / prompt    # same, 92.39 %
TPM/GPU    = (prompt + decode) / duration * 60 / 4
Extend TPM = extend / duration * 60
Decode TPS = decode / duration
```

The **engine-side** accept length, which is the only one worth quoting:

```
drafts   = delta(vllm:spec_decode_num_drafts_total)              #   180,318
drafted  = delta(vllm:spec_decode_num_draft_tokens_total)        # 1,262,226
accepted = delta(vllm:spec_decode_num_accepted_tokens_total)     #   540,954

drafted / drafts     = 7.000000    # == num_speculative_tokens
accepted / drafted   = 0.428571    # == 3/7
1 + accepted/drafts  = 4.000000    # == synthetic_acceptance_length, incl. bonus
```

### 3.4 Validity checks

Reject the run unless all of these hold:

| Check | Expected |
|---|---|
| `results.summary.json` has `duration` | yes |
| `requests_success == requests_total` | 1479 / 1479 in the reference run |
| `/v1/models` still answers after the run | yes |
| `delta(vllm:prefix_cache_hits_total) / delta(vllm:prefix_cache_queries_total)` | 0.923927 — equals the token-granularity hit rate (block vs token cross-check) |
| `delta(vllm:num_preemptions_total)` | **0** |
| `1 + accepted/drafts` | **4.000000**, i.e. exactly `synthetic_acceptance_length` |
| Other traffic on the endpoint during the window | none |

---

## 4. Accuracy validation

### 4.1 Server (no speculative decoding)

```bash
cd /root
rm -rf /root/.cache/atom/*

MODEL=/path/to/MiniMax-M3-MXFP8
 
export ATOM_M3_DENSE_ATTN_BACKEND=gluon
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1
export VLLM_USE_V2_MODEL_RUNNER=0
export SAFETENSORS_FAST_GPU=1
export NCCL_SOCKET_IFNAME=lo
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_LOG_LEVEL=WARNING
export PYTHONNOUSERSITE=1
export VLLM_DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export ATOM_FORCE_ATTN_TRITON=1
 
vllm serve "$MODEL" \
    --served-model-name minimax-m3-mxfp8 \
    --host 0.0.0.0 --port 8015 \
    --language-model-only \
    --tensor-parallel-size 4 \
    --no-trust-remote-code \
    --max-model-len 1000000 \
    --block-size 128 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 80 \
    --max-num-batched-tokens 32768 \
    --enable-prefix-caching \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4, "text_config": {"use_index_cache": true, "index_topk_freq": 4}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":80}' \
```

### 5.2 gsm8k (20-shot)

```bash
python3 -m lm_eval --model local-chat-completions --apply_chat_template --tasks gsm8k --output_path ./eval_out-tta1J8 --log_samples --num_fewshot 20 --model_args 'model=minimax-m3-mxfp8,base_url=http://0.0.0.0:8015/v1/chat/completions,api_key=EMPTY,eos_string=</s>,max_retries=5,num_concurrent=64,timeout=1800,tokenized_requests=False,max_length=1048576' --gen_kwargs max_tokens=16384,temperature=0,top_p=1
```
Results:
```bash
local-chat-completions ({'model': 'minimax-m3-mxfp8', 'base_url': 'http://0.0.0.0:8015/v1/chat/completions', 'api_key': 'EMPTY', 'eos_string': '</s>', 'max_retries': 5, 'num_concurrent': 64, 'timeout': 1800, 'tokenized_requests': False, 'max_length': 1048576}), gen_kwargs: ({'max_tokens': 16384, 'temperature': 0, 'top_p': 1}), limit: None, num_fewshot: 20, batch_size: 1
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|    20|exact_match|↑  |0.9500|±  | 0.006|
|     |       |strict-match    |    20|exact_match|↑  |0.9507|±  | 0.006|
```
