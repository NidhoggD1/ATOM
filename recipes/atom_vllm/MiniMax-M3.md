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

### 3.3 Decode, with serial MTP speculative decoding

M3 declares `num_mtp_modules: 7`, but **no released M3 checkpoint ships MTP
weights** — the MXFP4 index has 45,475 keys and zero `*mtp*` matches, the bf16
one 23,416 keys and likewise zero. The MTP head can still be exercised with
randomly initialized draft weights: spec decode verifies every draft token
against the target, so the *output text stays correct* and only the acceptance
rate collapses. This is a plumbing / performance harness, not an accuracy or
speedup demo.

**Step 1 — build a weights-free draft config directory.**

`draft_load_config: {"load_format": "dummy"}` still needs a config to size the
draft, and it cannot be the target checkpoint: dummy init writes through
`copy_()`, which has no Half→fp4 cast, so pointing the draft at an MXFP4 path
dies with `copy_() does not support casting Float4_e2m1fn_x2 to different
types`. Derive an unquantized bf16 config instead — the directory holds only
`config.json` plus the remote-code config module, no safetensors:

```bash
cat > /tmp/make_m3_mtp_draft.py <<'PY'
import json, shutil, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.loads((src / "config.json").read_text())
text = cfg.get("text_config", cfg)

# Dummy init cannot fill MXFP4/MXFP8 params, so the draft must be plain bf16.
cfg.pop("quantization_config", None)
text.pop("quantization_config", None)
cfg["torch_dtype"] = text["torch_dtype"] = "bfloat16"

# Keep 4 backbone layers. MiniMaxM3MultiTokenPredictor keys its MTP layer at
# num_hidden_layers and only instantiates num_nextn_predict_layers of them, so
# the backbone depth is just an index base -- but the per-layer sparse-attention
# lists are indexed by it and must be truncated to match. Shrinking the expert
# count is a real saving: it sizes the MoE inside the MTP block.
KEEP = 4
text["num_hidden_layers"] = KEEP
text["num_local_experts"] = 8
text["moe_layer_freq"] = text["moe_layer_freq"][:KEEP]
sparse = text.get("sparse_attention_config", {})
for k in ("sparse_attention_freq", "sparse_disable_index_value"):
    if k in sparse:
        sparse[k] = sparse[k][:KEEP]

# num_nextn_predict_layers = how many MTP layers are instantiated.
# num_mtp_modules = the depth num_speculative_tokens may reach; the single
# instantiated layer is reused modulo the instantiated count. The MXFP8
# checkpoint declares 1, hence the max().
text["num_nextn_predict_layers"] = 1
text["num_mtp_modules"] = max(int(text.get("num_mtp_modules") or 0), 7)

dst.mkdir(parents=True, exist_ok=True)
(dst / "config.json").write_text(json.dumps(cfg, indent=2))
shutil.copy(src / "configuration_minimax_m3_vl.py", dst)
PY

python3 /tmp/make_m3_mtp_draft.py "$MODEL" /tmp/m3_mtp_draft
```

**Step 2 — serve.**

`draft_load_config` scopes dummy init to the draft alone: `get_model()` takes
`load_config or vllm_config.load_config`, and the proposer passes
`speculative_config.draft_load_config` only for the draft head, so the target
keeps loading its real quantized weights under the global `--load-format auto`.

```bash
cd /root
rm -rf /root/.cache/atom/*

MODEL=/path/to/MiniMax-M3-MXFP8
DRAFT=/tmp/m3_mtp_draft

export ATOM_FORCE_ATTN_TRITON=1
export VLLM_USE_V2_MODEL_RUNNER=0
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export AITER_LOG_LEVEL=WARNING
# Mandatory at TP>1: platforms/rocm.py imports amdsmi inside a try/except that
# only warns, but CustomAllreduce.__init__ calls amdsmi_init() unconditionally,
# so a missing amdsmi turns into `NameError: name 'amdsmi_init' is not defined`
# on every worker. The bindings ship with ROCm, just not on sys.path.
export PYTHONPATH=/opt/rocm-7.2.4/share/amd_smi${PYTHONPATH:+:$PYTHONPATH}

vllm serve "$MODEL" \
    --served-model-name m3-mtp \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 4 \
    --trust-remote-code \
    --block-size 128 \
    --max-model-len 4096 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 8 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.85 \
    --hf-overrides '{"use_index_cache": true, "index_topk_freq": 4, "text_config": {"use_index_cache": true, "index_topk_freq": 4}}' \
    --speculative-config '{"method":"mtp","model":"'"$DRAFT"'","num_speculative_tokens":7,"draft_load_config":{"load_format":"dummy"}}'
```

Do **not** drop `"model"` and let vLLM default it — for `method: "mtp"` the
default is the target checkpoint, which reintroduces the fp4 cast failure above.

`num_speculative_tokens` may go up to the draft config's `num_mtp_modules`
(7 for M3); ATOM derives the bound from that field.

Expected: `/metrics` shows `spec_decode_num_draft_tokens_total` =
7 × `spec_decode_num_drafts_total`, all seven `num_accepted_tokens_per_pos`
buckets present and at 0, and mean acceptance length 1.00. Generated text
matches the no-spec server.

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
