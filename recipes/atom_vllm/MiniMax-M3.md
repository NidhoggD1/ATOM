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
| `VLLM_USE_V2_MODEL_RUNNER` | `0` | ATOM's EAGLE3 integration patches vLLM's V1 proposer; M3 defaults to V2 on ROCm, which bypasses it. |
| `ATOM_M3_UNIFORM_BATCH_CAPTURE` | `1` | Lets EAGLE3 spec-verify batches (`query_len > 1`) be CUDA-graph captured under `FULL_DECODE_ONLY`; without it decode falls back to eager. |

---

## 3. Multi-turn throughput benchmark

End-to-end serving benchmark on a multi-turn session replay (long shared
prefixes, prefix-cache dominated). 

**Result being reproduced** — 4x MI355X (gfx950), single node, TP4, concurrency 25:

| Draft | Accept len<br>configured / measured | Device KV pool | Cache hit | TPM/GPU | Extend TPM | Decode TPS | TTFT p50 | TPOT p50 | Host-side cache |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| EAGLE3, real weights | 4.0 / **3.82** | `--num-gpu-blocks-override 16612`<br>-> 2,126,199 tok | 90.40% | **1,657,488** | 631,579 | 816.2 | **407 ms** | **20.5 ms** | none |

Two things about that table:

- **Acceptance is forced, not earned.** `rejection_sample_method: "synthetic"` +
  `synthetic_acceptance_length: 4.0` makes the verifier accept a mean of exactly
  4.0 of the 7 drafted tokens, so draft *quality* never enters the measurement.
  The `3.82` is the load generator's client-side estimate
  (`completion_tokens / sse_chunk_count`); short completions pay a fixed
  first/last-chunk cost that drags the mean below the configured value, while
  requests with >= 256 completion tokens land on it. Only the **configured**
  value is comparable across engines.
- **The device KV pool override is for parity, not for speed.** At concurrency 25
  the in-flight working set is 1.57 M tokens (74 % of the pinned pool) and
  `vllm:num_preemptions_total` stays 0 for the whole run. Dropping the override
  lets the engine allocate 5,037,373 tokens instead and moves TPM/GPU by -0.1 %.


### 3.1 Server

```bash
cd /root
rm -rf /root/.cache/atom/*

MODEL=/path/to/MiniMax-M3-MXFP8
DRAFT=/path/to/MiniMax-M3-EAGLE3

export USE_ATOM=1
export ATOM_M3_DENSE_ATTN_BACKEND=gluon
export VLLM_USE_V2_MODEL_RUNNER=0
export ATOM_M3_UNIFORM_BATCH_CAPTURE=1
export VLLM_SERVER_DEV_MODE=1
export SAFETENSORS_FAST_GPU=1
export NCCL_SOCKET_IFNAME=lo
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_LOG_LEVEL=WARNING
export PYTHONNOUSERSITE=1
export VLLM_DO_NOT_TRACK=1
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1

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
    --num-gpu-blocks-override 16612 \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4, "text_config": {"use_index_cache": true, "index_topk_freq": 4}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":640}' \
    --speculative-config '{"method":"eagle3","model":"'"$DRAFT"'","num_speculative_tokens":7,"draft_tensor_parallel_size":1,"attention_backend":"ROCM_AITER_FA","rejection_sample_method":"synthetic","synthetic_acceptance_length":4.0}'
```

Notes on the non-obvious flags:

- `--num-gpu-blocks-override 16612` -> `16612 x 128 = 2,126,336` nominal; the
  engine reports **2,126,199** usable KV tokens. vLLM has no `--max-total-tokens`,
  so this is the only way to pin the device pool. Without it the engine picks
  39,357 blocks (5,037,373 tokens).
- `max_cudagraph_capture_size: 640` = `80 x (7+1)`, the spec-verify batch width.
  A smaller value silently drops decode out of CUDA graphs.
- Use `synthetic_acceptance_length`, **never** hand-written
  `synthetic_acceptance_rates`: the latter are *unconditional marginals* (mean
  length = `1 + sum`, and they must be non-increasing), which is very easy to get
  wrong. The two keys are mutually exclusive; the valid range here is
  `[1, num_speculative_tokens + 1] = [1, 8]`.
- No `--kv-transfer-config` and no `LMCACHE_*`: this arm has **zero** host-side
  KV cache. The 90.40 % hit rate is device prefix cache only.
- NextN: TBD

### 3.5 Load generator

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
    --output-path /output/results.jsonl
```


## 4. Server commands

Run `rm -rf /root/.cache/atom/*` before each launch.

### 4.1 Prefill

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

vllm serve "$MODEL" \
    --served-model-name minimax-m3-mxfp8 \
    --host 0.0.0.0 --port 8000 \
    --language-model-only \
    --tensor-parallel-size 4 \
    --no-trust-remote-code \
    --block-size 128 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --max-model-len 131072 \
    --max-num-seqs 128 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --no-async-scheduling \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4, "text_config": {"use_index_cache": true, "index_topk_freq": 4}}' \
    --compilation-config '{"cudagraph_mode":"NONE"}'
```

### 4.2 Decode, with EAGLE3 speculative decoding

Performance harness only — `DecodeBenchConnector` fabricates the KV cache and
`rejection_sample_method: "synthetic"` fixes the acceptance rate. Use §5.1 for
accuracy.

```bash
cd /root
rm -rf /root/.cache/atom/*

MODEL=/path/to/MiniMax-M3-MXFP8
DRAFT=/path/to/MiniMax-M3-EAGLE3-GQA

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

vllm serve "$MODEL" \
    --served-model-name minimax-m3-mxfp8 \
    --host 0.0.0.0 --port 8000 \
    --language-model-only \
    --tensor-parallel-size 4 \
    --no-trust-remote-code \
    --max-model-len 131072 \
    --block-size 128 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 88 \
    --max-num-batched-tokens 2048 \
    --no-enable-prefix-caching \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4, "text_config": {"use_index_cache": true, "index_topk_freq": 4}}' \
    --kv-transfer-config '{"kv_connector":"DecodeBenchConnector","kv_role":"kv_both","kv_load_failure_policy":"fail","kv_buffer_device":"cuda","kv_connector_extra_config":{"fill_mean":0.015,"fill_std":0.0}}' \
    --speculative-config '{"method":"eagle3","model":"'"$DRAFT"'","num_speculative_tokens":3,"draft_tensor_parallel_size":1,"attention_backend":"ROCM_AITER_FA","rejection_sample_method":"synthetic","synthetic_acceptance_rates":[0.7,0.5,0.4]}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":512}'
```

---

## 5. Accuracy validation

### 5.1 Server (no speculative decoding)

```bash
cd /root
rm -rf /root/.cache/atom/*

MODEL=/path/to/MiniMax-M3-MXFP8

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
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4, "text_config": {"use_index_cache": true, "index_topk_freq": 4}}' \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","max_cudagraph_capture_size":512}'
```

### 5.2 gsm8k (5-shot)

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/path/to/hf_cache \
lm_eval \
  --model local-chat-completions \
  --model_args "model=minimax-m3,base_url=http://localhost:8902/v1/chat/completions,num_concurrent=32,max_gen_toks=2048,max_retries=3" \
  --tasks gsm8k --num_fewshot 5 --batch_size 65 \
  --apply_chat_template --fewshot_as_multiturn \
  --output_path ./eval_out/gsm8k
```

### 5.3 AIME25 (maj@16)

One-time setup:

```bash
HF_HOME=/path/to/hf_cache python3 -c \
  'from datasets import load_dataset; load_dataset("math-ai/aime25", split="test")'

LM_EVAL_TASKS=$(python3 -c "import lm_eval.tasks, os; print(os.path.dirname(lm_eval.tasks.__file__))")
cat > "$LM_EVAL_TASKS/aime/aime25_maj16.yaml" <<'YAML'
include: aime25.yaml
task: aime25_maj16
repeats: 16
filter_list:
  - name: maj@16
    filter:
      - function: regex
        regex_pattern: '\\boxed\{([^}]*)\}'
        group_select: -1
        fallback: "[invalid]"
      - function: majority_vote
      - function: take_first
YAML
```

Run:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/path/to/hf_cache \
lm_eval \
  --model local-chat-completions \
  --model_args "model=minimax-m3,base_url=http://localhost:8902/v1/chat/completions,num_concurrent=64,max_retries=3,timeout=7200,tokenized_requests=False" \
  --tasks aime25_maj16 \
  --apply_chat_template \
  --gen_kwargs "temperature=1.0,top_p=0.95,do_sample=True,max_gen_toks=98304" \
  --batch_size 64 --seed 42 \
  --log_samples --output_path ./eval_out/aime25
```
