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

## 3. Server commands

Run `rm -rf /root/.cache/atom/*` before each launch.

### 3.1 Prefill

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

### 3.2 Decode, with EAGLE3 speculative decoding

Performance harness only — `DecodeBenchConnector` fabricates the KV cache and
`rejection_sample_method: "synthetic"` fixes the acceptance rate. Use §4.1 for
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

## 4. Accuracy validation

### 4.1 Server (no speculative decoding)

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

### 4.2 gsm8k (5-shot)

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/path/to/hf_cache \
lm_eval \
  --model local-chat-completions \
  --model_args "model=minimax-m3,base_url=http://localhost:8902/v1/chat/completions,num_concurrent=32,max_gen_toks=2048,max_retries=3" \
  --tasks gsm8k --num_fewshot 5 --batch_size 65 \
  --apply_chat_template --fewshot_as_multiturn \
  --output_path ./eval_out/gsm8k
```

### 4.3 AIME25 (`pass_avg,all` @16 + non-stop, evalscope)

The customer scores AIME25 with **evalscope** (not lm_eval), and the reported
metric is **`pass_avg,all`** — the plain mean of the per-generation accuracy over
all `30 problems × 16 repeats = 480` generations (i.e. avg@16), **not** majority
vote. A second, independent hard requirement is the **non-stop ratio < 0.5%**:
the fraction of generations that hit `MAX_TOKENS` without a natural EOS (the model
rambling past the budget instead of converging). Baseline floor: `pass_avg,all ≥
0.927`, `non-stop < 0.5%`.

Customer-exact knobs (do **not** change the test method; only `repeats` may be
temporarily lowered for a fast smoke run):

| knob | value |
|------|-------|
| task | `aime25`, prompt **without** "step by step" |
| repeats | 16 (480 gens total) |
| `max_tokens` | 98304 |
| `temperature` / `top_p` | 1.0 / 0.95 |
| metric | `pass_avg,all` (evalscope `mean` aggregation) |
| grader | rule (`grade_answer`), invalid answers always count as a wrong vote |

One-time setup (protects the image's torch/vLLM):

```bash
pip install --no-deps evalscope colorlog filetype jsonlines editdistance overrides
# dataset: HF math-ai/aime25 (same 30 problems as evalscope/aime25)
HF_HOME=/path/to/hf_cache python3 -c \
  'from datasets import load_dataset; load_dataset("math-ai/aime25", split="test")'
```

Runner (`evalscope_aime25.py`) — the customer prompt with "step by step" removed,
and the **mandatory** client `timeout`:

```python
import os
from evalscope import run_task
from evalscope.config import TaskConfig

PORT    = os.environ.get("AIME_PORT", "8902")
REPEATS = int(os.environ.get("AIME_REPEATS", "16"))   # only debug knob; keep 16 to report
WORKDIR = os.environ.get("AIME_WORKDIR", "./eval_out/aime25")

# evalscope's default aime25 template MINUS "step by step" (customer spec)
CUSTOM_PROMPT = ("Solve the following math problem. Put your answer inside \\boxed{{}}.\n\n"
                 "{question}\n\n"
                 "Remember to put your answer inside \\boxed{{}}.")

cfg = dict(
    model="minimax-m3",
    api_url=f"http://localhost:{PORT}/v1/chat/completions",
    api_key="EMPTY", eval_type="openai_api",
    datasets=["aime25"],
    dataset_args={"aime25": {"prompt_template": CUSTOM_PROMPT, "dataset_id": "math-ai/aime25"}},
    repeats=REPEATS,
    generation_config={
        "temperature": 1.0, "top_p": 0.95, "max_tokens": 98304,
        # CRITICAL: without this, evalscope uses the openai-SDK default 600s client
        # timeout. A generation toward the 98304 cap at ~50 tok/s needs ~30 min >> 600s,
        # so every long gen times out client-side, retries 5x, then is scored FAILED —
        # silently corrupting BOTH pass_avg and non-stop. 3600s covers a full-length gen.
        "timeout": 3600,
    },
    judge={"strategy": "rule"},
    eval_batch_size=int(os.environ.get("AIME_BATCH", "64")),
    work_dir=WORKDIR, dataset_hub="huggingface",
)
run_task(task_cfg=TaskConfig(**cfg))
```

Run (full customer run; for a fast smoke set `AIME_REPEATS=1`):

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME=/path/to/hf_cache \
  AIME_WORKDIR=./eval_out/aime25 python3 evalscope_aime25.py
```

Read both metrics:

```bash
# pass_avg,all — from evalscope's report (the Accuracy / mean row)
python3 - <<'PY'
import glob, json
rep = sorted(glob.glob("./eval_out/aime25/2*/reports/minimax-m3/aime25.json"))[-1]
d = json.load(open(rep))
print("pass_avg,all =", d["metrics"][0]["score"])   # evalscope mean aggregation over 480 gens
PY

# non-stop ratio — from the raw predictions (per-choice stop_reason)
python3 - <<'PY'
import glob, json, ast
pf = sorted(glob.glob("./eval_out/aime25/2*/predictions/minimax-m3/aime25_default.jsonl"))[-1]
tot = ns = 0
for line in open(pf):
    mo = json.loads(line)["model_output"]
    mo = ast.literal_eval(mo) if isinstance(mo, str) else mo
    for ch in mo["choices"]:
        tot += 1
        if ch.get("stop_reason") not in ("stop", "eos"):   # max_tokens / length = truncated
            ns += 1
print(f"non-stop = {ns}/{tot} = {ns/tot:.4%}   (floor < 0.5%)")
PY
```
