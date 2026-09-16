# Qwen3.8-Flash-Next · SGLang + ATOM（MI308）

给接手的研发：这条分支只做 **SGLang plugin**。Native Flash（`atom/models/qwen4_exp.py`、`atom/model_ops/**`）已经在 **#2048** 合进 `main`，本 PR **不再改 Native ops**。

- 分支：`feat/sglang-atom-qwen38-flash`
- rebase 基准：当前 `origin/main`
- 权重：`/data/pretrained_model/Qwen/Qwen3.8-Flash-Next-FP8`（`qwen4_exp` / QSA，不是 Qwen3.5，也不是 2.4T MoE）

## 1. 遇到什么错、改了什么

按时间线，都是 **SGLang 侧接线**，不是 Native 算子重写。

### 1.1 Transformers / ServerArgs 不认 checkpoint

**现象：** 镜像里的 transformers < 5.16.1，`AutoConfig` 不认识 `qwen4_exp` / 嵌套 `qwen4_exp_text`，SGLang 在 Native `get_hf_config` 之前就挂。

**改动：** `atom/plugin/sglang/register.py` 里注册 `Qwen4ExpConfig` / `Qwen4ExpTextConfig`（后者 subclass SGLang `Qwen3_5TextConfig`，才能带上 `mamba2_cache_params`）。

### 1.2 被当成 Qwen3.5，或根本没开 hybrid GDN

**现象：** greedy 乱码。`is_hybrid_ssm` 为 false → 没有 `MambaPool` / `mamba_map` → GDN 输出全 0，PLE 被跳过。

**改动：**

- 独立 EntryClass `atom/plugin/sglang/models/qwen3_8_flash_next.py`（**不要**挂 `Qwen3_5*`）
- `runtime/model_arch.py` 映射 `qwen4_exp` / `qwen4_exp_text`
- `plugin/register.py` 注册 `Qwen4ExpForConditionalGeneration`
- patch `hybrid_gdn_config`、M-RoPE `get_rope_index`（`qwen4_exp` → 走 `qwen3_5` 的 index 公式）

### 1.3 QSA / PLE metadata 对不上 ForwardBatch

**现象：** Native 要 `qsa_metadata` + `ple_metadata`；SGLang 只有 `ForwardBatch`。page table 太短（按 indexer_budget=2k 建表）会在 12k 上丢掉远距离 token。

**改动：** `qwen3_8_flash_next_bridge.py`：`ForwardBatch` → Native QSA page table / slot / PLE n-gram state。page table 按 **context_length** 建，不是只按 2048 budget。

### 1.4 CUDA graph pad 行读到已结束 request 的 page / mamba slot

**现象：** decode graph 对齐到 bucket 后，`num_padding` 行仍带着刚 finish 的 `req_pool_indices` / page table，QSA/GDN 读野指针或错误 KV。

**改动（plugin 内）：**

- `backend_resolver.real_batch_size()` 同时看 `num_padding` 和 `_original_batch_size`
- GDN `SGLangGDNForwardContext` 把 pad 行的 mamba index 打成 `-1`，`query_start_loc` 在 live_bs 处截断
- bridge 里 pad 行 `seq_lens=0`、`slot_mapping=_NO_WRITE`、block table 清零

单测：`tests/plugin/test_sglang_qwen38_flash_graph_padding.py`

### 1.5 开 decode CUDA graph：短 greedy 正常，长 prefill 后第一次 decode HSA fault

**现象：** capture 时 kernel 吃到的是 `torch.empty` / `.contiguous()` 的 slab；长 eager prefill 把 caching allocator 那块收回去，replay 指针悬空。Native #2048 **没有** graph-safe PLE kernel。

**改动：** 全部放在 plugin，**不改** `atom/model_ops/*`：

| 文件 | 作用 |
|---|---|
| `flash_decode_graph_workspace.py` | GDN / MoE / RMSNorm / QKV split 的 grow-once buffer |
| `qsa_graph_workspace.py` | QSA indexer logits 等（capture 前 reserve） |
| `patches/flash_native_graph_ops_patch.py` | monkeypatch Native 的 `fused_gdn_gating`、`rearrange_mixed_qkv`、`causal_conv1d_update`、`GemmaRMSNorm`、MoE `empty`、`Qwen4ExpAttention.split`、DecoderLayer PLE |
| `patches/flash_decode_graph_ple_patch.py` | hybrid GDN `out_graph` 之后刷新 PLE（GDN 先写 `mamba_cache_indices`） |
| `patches/flash_breakable_logits_patch.py` | breakable CUDA graph 认 `LogitsProcessorOutput`（Flash `forward` 返回 logits） |
| `patches/flash_decode_graph_replay_sync_patch.py` | 仅 `ATOM_FLASH_GRAPH_REPLAY_SYNC=1` 时在长 seq replay 后 `synchronize`（默认关，避免拖 TPOT） |
| `full_attention_backend.py` | `init_forward_metadata_out_graph` 里刷新 QSA persistent buffer（replay 不能沿用 capture 时的短 page table） |

PLE：`ATOM_FLASH_PLE_EAGER` 默认开，作为 breakable graph 的 eager island；QSA / GDN / MoE / HC 仍在 graph 里。

### 1.6 流式 bench 看起来 TTFT/TPOT 全坏

**现象：** Native OpenAI completions 最后一帧 usage-only `choices: []`，`sglang.bench_serving` 读 `choices[0]` → `IndexError`。关掉 stream 后 TTFT≈E2E。**计算没坏。**

**改动：** 不在本 PR。测 Native 时 skip 空 choices，或用 completions token-id 客户端。

## 2. 本分支文件清单（相对最新 main）

只应出现 `atom/plugin/**`、`recipes/atom_sglang/**`、`tests/plugin/**`。

新增：

- `atom/plugin/sglang/models/qwen3_8_flash_next.py`
- `atom/plugin/sglang/models/qwen3_8_flash_next_processor.py`
- `atom/plugin/sglang/qwen3_8_flash_next_bridge.py`
- `atom/plugin/sglang/flash_decode_graph_workspace.py`
- `atom/plugin/sglang/qsa_graph_workspace.py`
- `atom/plugin/sglang/patches/flash_*.py`（含 `flash_native_graph_ops_patch.py`）
- `tests/plugin/test_sglang_qwen38_flash_graph_padding.py`
- 本 README

修改：

- `atom/plugin/register.py`（注册 Qwen4Exp）
- `atom/plugin/sglang/register.py`
- `atom/plugin/sglang/models/__init__.py`
- `atom/plugin/sglang/runtime/model_arch.py`
- `atom/plugin/sglang/attention_backend/{attention_gdn,backend_resolver,full_attention/...}.py`
- `tests/plugin/test_sglang_gdn_forward_context.py`（padding 行）

刻意 **没有**：`atom/config.py`、`atom/models/qwen4_exp.py`、`atom/model_ops/**`、`tests/test_attn_family.py`。  
`qwen4_exp_text` 进 Native `_PLAIN_TEXT_CONFIG_MODEL_TYPES` 如仍需要，另开 Native PR。

## 3. 启动（MI308，约 192 GB HBM）

```bash
export SGLANG_PLUGINS=atom_sglang
export SGLANG_EXTERNAL_MODEL_PACKAGE=atom.plugin.sglang.models

python -m sglang.launch_server \
  --model-path /data/pretrained_model/Qwen/Qwen3.8-Flash-Next-FP8 \
  --tp 2 --enable-expert-parallel \
  --kv-cache-dtype bf16 \
  --page-size 64 \
  --context-length 16384 \
  --max-running-requests 32 \
  --mem-fraction-static 0.95 \
  --trust-remote-code
```

TP>1 必须 EP（`moe_intermediate_size=640`）。`page-size` 要能被 `indexer_compress_ratio`（4）整除。不要 `--ple-offload-embedding`。不要照抄 MI355 的 0.98 mem。

对照 Native（#2048 已在 main）：

```bash
python -m atom.entrypoints.openai_server \
  --model /data/pretrained_model/Qwen/Qwen3.8-Flash-Next-FP8 \
  -tp 2 --enable-expert-parallel --kv-cache-dtype bf16 \
  --max-model-len 16384 --gpu-memory-utilization 0.75
```

## 4. 已在 308 上验证过的量（供对照，不是 CI）

FP8、TP2/EP2。1K/1K rebase 后 Native vs SGLang+ATOM：conc1 64.3 vs 56.6 tok/s（−12%），conc4 217 vs 200（−8%）。

12k/350、不开 APC、对 wulei SGLang：ATOM conc=32 约 **0.26 QPS/GPU**（SGLang 0.24）；conc=1 Mean TPOT ~15 ms；卡 21 ms TPOT 大约 conc=3。

精度（Native greedy, thinking off）：GSM8K blockwise 96.89% / PTPC 96.66%；MMLU-Pro 12032 题 80.83% / 80.84%。PTPC 权重加载是另一条分支 `fix/qwen38-flash-ptpc-fp8`，不要并进本 PR。

## 5. 不在本 PR

MTP、VLM、radix、speculative、PTPC-FP8 scale 修复、Native graph-safe PLE kernel、改 `sglang.bench_serving` SSE parser。
