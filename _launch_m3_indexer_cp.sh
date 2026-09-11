#!/usr/bin/env bash
# MiniMax-M3 at TP4 for the indexer-only context-parallel A/B.
#
#   bash _launch_m3_indexer_cp.sh                       # CP arm, no spec decode
#   ATOM_M3_INDEXER_CP=0 bash _launch_m3_indexer_cp.sh  # TP arm (the baseline)
#   MTP=1 bash _launch_m3_indexer_cp.sh                 # CP + EAGLE3, 7 draft tokens, AL pinned to 4
#   MTP=1 PORT=8014 DEVICES=0,1,2,3 ATOM_M3_INDEXER_CP=0 bash ...   # paired TP arm
#   MTP=1 ACC_LEN= bash ...                             # same, but REAL acceptance (readable text)
#
# MTP=1 IS THE SWITCH FOR SPECULATIVE DECODING -- NUM_SPEC ALONE DOES NOTHING.
# `NUM_SPEC=7 bash ...` launches a no-spec server: NUM_SPEC/ACC_LEN/ACC_RATE are
# read only inside the `if MTP == 1` branch. mtp7 is `MTP=1` (NUM_SPEC already
# defaults to 7); the script now exits 2 rather than ignoring them silently.
# Confirm from the log, since a no-spec server is otherwise indistinguishable:
#   grep -c "MTP Stats" server.log     # >0 == spec decode really ran
#
# Every knob is an env var with a default: PORT DEVICES MODEL_PATH DRAFT_PATH
# MAX_LEN MAX_BATCHED MAX_SEQS MTP NUM_SPEC ACC_LEN ACC_RATE LOG CLEAR_CACHE.
# Run two arms side by side on disjoint DEVICES and disjoint PORTs; the box has
# 8 GPUs and each arm needs 4.
#
# TP4 IS REQUIRED, NOT A PREFERENCE. The gate rejects anything but
# tensor_parallel_size == num_key_value_heads (== 4 here), so the -tp 2 of
# _launch_m3_bf16.sh would silently fall back to the TP indexer path and the A/B
# would compare a config against itself.
#
# CONFIRM THE ARM FROM THE LOG, NOT FROM THE FLAG:
#   grep -c "indexer_dcp_only enabled" server.log   # 1 == CP, 0 == TP
# The "Engine kwargs" line is printed by arg_utils.py BEFORE Config.__post_init__
# applies the env override, so it reads False in BOTH arms and cannot tell them
# apart. This script prints the grep for you once the server is up.
#
# CONFIRMING THE ARM FROM A PROFILER TRACE: GREP KERNELS, NOT FUNCTIONS.
#   zcat trace/rank_0/<one-file>.pt.trace.json.gz | grep -c _context_score
# >0 means CP; the TP arm instead shows `index_topk`. Do NOT grep the Python
# names -- `indexer_context_scores`, `local_candidate_keys`, `sparse_attention_
# decode`, `minimax_m3_index_topk_decode` are all ABSENT FROM BOTH ARMS, so they
# look like proof the feature is off when it is running fine. Decode replays
# inside a CUDA graph, and `_sparse_decode`'s @mark_trace(torch_compile=False) is
# a host-side record_function: graph replay reissues only GPU kernels, so no
# annotation is ever emitted. A whole M3 trace contains zero cpu_op frames
# naming the indexer.
# Two more false negatives in the same trace:
#   - the all-to-all is NOT named all_to_all. RCCL folds it into
#     `ncclDevKernel_Generic_1(...)`, sharing that symbol with the MoE
#     all-reduce, so grepping AllToAll returns 0.
#   - prefill is TP by design (attention_mha.py:1343), so a prefill-heavy
#     capture window legitimately shows only the old path.
# Pin ONE filename: trace/rank_*/ accumulates traces from every model ever run
# here, and a `zcat *.gz` glob borrows counts from the K3/GLM DCP traces.
set -euo pipefail
cd /shared/amdgpu/home/gyu_qle/ganyi/ATOM

# ---------------------------------------------------------------- environment
export AITER_LOG_LEVEL=WARNING
# From recipes/MiniMax-M3.md. Not optional: aiter's ASM paged-attention ships no
# block_size-128 kernel (`ls hsa/*/pa/ | grep blk128` is empty), so at
# --block-size 128 the heuristic misses and the rank aborts with SIGABRT --
# "cannot get heuristic kernel! ... gqa:16 ... block_size:128". Verified to
# happen with indexer CP both ON and OFF, so it is the kernel, not this feature.
export ATOM_FORCE_ATTN_TRITON=1
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export HIP_VISIBLE_DEVICES="${DEVICES:-4,5,6,7}"
# The arm. 1 = indexer CP, 0 = today's TP path. The env override wins over the
# config field in both directions (config.py:2141), so the two arms differ in
# exactly one variable.
export ATOM_M3_INDEXER_CP="${ATOM_M3_INDEXER_CP:-1}"
export MTP="${MTP:-0}"

# Stale torch.compile cache silently serves the previous code after an edit.
# Opt-in because a cold compile costs minutes and most launches do not need it.
if [[ "${CLEAR_CACHE:-0}" == "1" ]]; then
  rm -rf /root/.cache/atom/* ~/.cache/atom/torch_compile_cache/ 2>/dev/null || true
fi

# MXFP4 by default. The bf16 checkpoint leaves only ~404 KV blocks (~51k tokens
# TOTAL, across all sequences) at TP4 because peak_torch is 205GB of the 230GB
# budget -- unusable for a 20k-491k-token agentic replay. MXFP4 is 227G vs 796G
# and has identical head geometry (4 kv / 64 q / 128 sparse block), so the
# indexer-CP gate sees the same topology.
main_model="${MODEL_PATH:-/workspace/shared/data/amd_int/models/MiniMax-M3-MXFP4}"

# ------------------------------------------------------------ optional EAGLE3
# WHY MTP IS THE CONFIGURATION THIS FEATURE IS FOR.
# indexer_context_scores sizes its MMA tile as
#     N = max(16, next_power_of_2(heads * max_query_len))
# (indexer_context_parallel.py:161). Heads and draft tokens fill ONE dimension
# together against a hardware floor of 16, where max_query_len == NUM_SPEC + 1:
#     TP,  no spec   1 x 1 =  1  ->   6% of a 16-wide tile
#     CP,  no spec   4 x 1 =  4  ->  25%
#     TP + spec3     1 x 4 =  4  ->  25%
#     CP + spec3     4 x 4 = 16  -> 100%
#     TP + spec7     1 x 8 =  8  ->  50%   (still a 16-wide tile: below the floor)
#     CP + spec7     4 x 8 = 32  -> 100% of a 32-wide tile
# Past NUM_SPEC=3 the CP arm stops gaining occupancy -- it is already full at
# spec3 -- and instead grows the tile, so spec7 is a LARGER-tile datapoint, not
# a better-packed one. The TP arm keeps gaining (25% -> 50%), which narrows the
# gap this feature exploits. Read a spec7 A/B with that in mind.
#
# An earlier revision of indexer_cp_unsupported_reason REJECTED speculative
# decoding ("qlen>1 exchange not exercised"), which silently served the TP path
# whenever --method eagle3 was passed. That reason was false -- every kernel in
# the chain takes max_query_len as a runtime argument and validates against it,
# verified out of tree over query_len {1, 4} asserting torch.equal against
# minimax_m3_index_topk_decode for every head (24 passed). That suite is NOT in
# this repo. The gate is lifted, so ATOM_M3_INDEXER_CP now means what it says
# under MTP too.
#
# ACC_LEN PINS ACCEPTANCE. DEFAULTS TO 4 OVER 7 DRAFT POSITIONS.
# --spec-decode-acceptance-length replaces real draft/target rejection sampling
# with a synthetic per-position schedule (config.py:_resolve_synthetic_acceptance
# -> rejection_sampler.acceptance_length_to_rates), so both arms execute the same
# number of forwards for the same tokens and the remaining delta is the kernel.
# That is the point for a PERFORMANCE A/B: acceptance is a per-request random
# variable, and letting it float mixes the indexer delta with a sampling
# difference that has nothing to do with the indexer.
#
# AL counts the target's own guaranteed token, so it is the same unit as vLLM's
# synthetic_acceptance_length and SGLang's SGLANG_SIMULATE_ACC_LEN -- a published
# golden AL passes through unchanged. The resolver is minimum-variance and
# deterministic, so ACC_LEN=4 over NUM_SPEC=7 is NOT "4 of 7 on average":
#     rates = [1, 1, 1, 0, 0, 0, 0]
# Exactly 3 draft tokens are accepted EVERY step; positions 4-7 are drafted and
# thrown away every step. So spec7 at AL=4 emits the same tokens per step as
# spec3 at AL=4 while paying four extra draft forwards for them. That is a
# legitimate thing to measure -- it is the tile-size sweep above -- but it is not
# a throughput win, and a spec7 arm losing to spec3 is the expected result, not
# a bug. Raise ACC_LEN toward 8 to make the extra positions pay for themselves.
#
# ACC_LEN and ACC_RATE describe the same curve (rate = (length-1)/NUM_SPEC) and
# _resolve_synthetic_acceptance RAISES if both are set, so this script emits at
# most one: ACC_LEN wins, and setting ACC_RATE requires clearing ACC_LEN.
# NUM_SPEC=7 NEEDS A max_spec BOUND THIS BRANCH NO LONGER RAISES. config.py
# derives it as max(4, draft_cfg.num_mtp_modules), and the M3 EAGLE3-GQA draft
# declares no num_mtp_modules -- so the bound is 4 and NUM_SPEC=7 is REJECTED at
# startup. The limit is a quality policy, not a structural one (the CP chain was
# verified exactly equal to the native kernel at query_len 8, out of tree); add
# num_mtp_modules to a local copy of the draft config to raise it.
#
# THE COST: THE OUTPUT TEXT IS A LITERAL PLACEHOLDER, NOT JUST WRONG.
# api_server.py:2612 replaces every generated token with the string "synthetic "
# whenever either flag is set (streaming_dispatch.py:44). Token counts, timings
# and throughput are unaffected, but the response body carries no model output at
# all -- so lm_eval AND _verify_mtp_cp_noisefloor.py are both meaningless against
# a server launched this way. The noise-floor script would report a perfect match
# between any two arms, because both emit the same placeholder. Clear ACC_LEN
# (`MTP=1 ACC_LEN= bash ...`) to get real sampling and readable text back.
#
# The GQA draft, not the MHA one also on disk. Both are published by Inferact
# under near-identical names and differ in exactly one config field --
# num_key_value_heads 4 (GQA) vs 64 (MHA) -- with every other field, including
# the file sizes, identical. 4 is the target's own num_key_value_heads.
draft_model="${DRAFT_PATH:-/workspace/shared/data/amd_int/models/MiniMax-M3-EAGLE3-GQA}"
spec_args=()
# MTP=1 IS THE ONLY SWITCH. Setting NUM_SPEC / ACC_LEN / ACC_RATE alone does
# NOTHING -- they are read only inside the branch below, so `NUM_SPEC=7 bash
# ...` silently launches a no-spec server that looks right in every log line
# except the ones nobody greps. That already cost a debugging session: the trace
# had zero draft/rejection kernels and it read as "mtp7 is broken" rather than
# "mtp was never on". Fail loudly instead.
if [[ "${MTP:-0}" != "1" ]]; then
  for v in NUM_SPEC ACC_LEN ACC_RATE DRAFT_PATH; do
    if [[ -n "${!v+set}" ]]; then
      echo "ERROR: $v is set but MTP is not 1, so speculative decoding is OFF" \
           "and $v would be silently ignored. Add MTP=1, or unset $v." >&2
      exit 2
    fi
  done
fi
if [[ "${MTP:-0}" == "1" ]]; then
  spec_args+=(--method eagle3 --draft-model "$draft_model"
              --num-speculative-tokens "${NUM_SPEC:-7}")
  # ACC_LEN defaults to 4; ACC_LEN= (empty) turns synthetic acceptance off
  # entirely. Emit at most one flag -- the resolver rejects both at once.
  acc_len="${ACC_LEN-4}"
  if [[ -n "$acc_len" && -n "${ACC_RATE:-}" ]]; then
    echo "ERROR: set ACC_LEN or ACC_RATE, not both (they describe the same" \
         "curve and _resolve_synthetic_acceptance raises). Use ACC_LEN= to" \
         "clear the default before setting ACC_RATE." >&2
    exit 2
  fi
  if [[ -n "$acc_len" ]]; then
    spec_args+=(--spec-decode-acceptance-length "$acc_len")
  elif [[ -n "${ACC_RATE:-}" ]]; then
    spec_args+=(--spec-decode-acceptance-rate "$ACC_RATE")
  fi
  # The draft adds ~5.8GB of weights per rank on top of the target, out of the
  # same --gpu-memory-utilization budget, so num_kvcache_blocks lands lower than
  # the 52,883 the no-MTP arm reports. Both MTP arms pay it identically, so
  # CP-vs-TP is unaffected; only MTP-vs-no-MTP is. Check it in the log before
  # reading any throughput number.
fi

# --------------------------------------------------------------------- launch
# MAX_LEN is a parameter because the two workloads need different values. GSM8K
# fits in 32k, but the AgentX replay traces run 20k-491k tokens: at a 32k limit
# every long trace is rejected in seconds and the "benchmark" measures nothing.
# Raising it costs almost no KV (the pool is sized by gpu-memory-utilization,
# not by max-model-len), so the long-context arm sets MAX_LEN=262144.
# Concurrency is capped by KV blocks either way, not by --max-num-seqs.
#
# No --enforce-eager / --level 0: this matches recipes/MiniMax-M3.md, which runs
# the DEFAULT level 3 (piecewise + CUDAGraph). Those two flags were added while
# debugging the ASM-PA SIGABRT (per /debug-guide) and then left in by mistake.
# Leaving them in penalizes the CP arm specifically -- CP adds one all-to-all
# per sparse layer (57 of them) that TP does not have, and CUDAGraph is exactly
# what absorbs that kind of fixed launch cost.
#
# Index cache is left at its default dtype on purpose. The recipe uses
# --index-cache-dtype fp8, but the CP scorer casts fp8 index-K with a plain
# .to() (indexer_context_parallel.py:59) where the native kernel has a dedicated
# fp8 branch -- so fp8 is a SEPARATE variable to test, after this one works.
#
# Prefix caching is left ON (the engine default). recipes/MiniMax-M3.md passes
# --no-enable_prefix_caching in all three launches without saying why, but the
# AgentX replay is 97.6% theoretically-reusable prefix, so disabling it pins the
# one axis that workload is about to a constant 0% hit. Watch "Prompt Cache
# Read" in the aiperf summary.
# MAX_LEN= (empty) OMITS --max-model-len so the engine derives the window from
# the checkpoint (max_position_embeddings = 1048576). Kept as an explicit knob
# because the flag cannot be passed as an empty string.
len_args=()
if [[ -n "${MAX_LEN-262144}" ]]; then
  len_args+=(--max-model-len "${MAX_LEN-262144}")
fi

cmd=(python -m atom.entrypoints.openai_server --model "$main_model"
  -tp 4 --server-port "${PORT:-8013}" --trust-remote-code
  "${spec_args[@]}"
  --gpu-memory-utilization 0.8
  --block-size 128
  --max-num-batched-tokens "${MAX_BATCHED:-32768}"
  "${len_args[@]}"
  --max-num-seqs "${MAX_SEQS:-128}"
  --kv_cache_dtype fp8
  --online_quant_config '{"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head", "model.embed_tokens", "vision_tower", "multi_modal_projector", "patch_merge_mlp", "*block_sparse_moe"]}'
  --torch-profiler-dir "${TRACE_DIR:-./trace}")

echo "arm=$([[ $ATOM_M3_INDEXER_CP == 1 ]] && echo CP || echo TP)" \
     "mtp=${MTP:-0} port=${PORT:-8013} gpus=$HIP_VISIBLE_DEVICES" >&2
echo "verify once up:  grep -c 'indexer_dcp_only enabled' <logfile>   # 1=CP 0=TP" >&2
echo "                 curl -sf localhost:${PORT:-8013}/v1/models && rocm-smi --showmemuse" >&2

# LOG= tees to a file AND keeps the console. Without it, redirect yourself --
# the arm-verification grep needs a log to read. Note that /health can return OK
# with no model loaded, so confirm VRAM% > 0 with rocm-smi, not just HTTP.
if [[ -n "${LOG:-}" ]]; then
  exec "${cmd[@]}" 2>&1 | tee "$LOG"
fi
exec "${cmd[@]}"
