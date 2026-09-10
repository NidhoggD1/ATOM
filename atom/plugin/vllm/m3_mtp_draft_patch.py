# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Let MiniMax-M3 serial MTP run without a hand-built draft model directory.

No released M3 checkpoint ships MTP weights -- the MXFP4 index has 45,475 keys
and zero ``*mtp*`` matches -- so the only way to exercise the head is to
randomly initialize it via ``draft_load_config: {"load_format": "dummy"}``. That
alone is not enough, because ``load_format`` governs *how* parameters are
filled, not *where the config describing them comes from*: with no ``"model"``
in ``--speculative-config``, ``SpeculativeConfig.__post_init__`` defaults the
draft to the target checkpoint **and copies the target's quantization**, and
dummy init then dies with

    RuntimeError: copy_() does not support casting Float4_e2m1fn_x2 to
    different types

(``initialize_single_dummy_weight`` takes the ``finfo(dtype).bits < 16`` branch
for ``Float4_e2m1fn_x2`` and reaches ``copy_()`` with no Half->fp4 cast).
Stripping ``quantization_config`` from a config the user supplies does not help
either: ``ModelConfig._verify_quantization`` only auto-detects when
``self.quantization`` is unset, so the explicitly-copied ``"mxfp4"`` survives.

So the draft needs its own unquantized bf16 config, and until now the user had
to build that directory by hand and pass its path. This patch generates it
instead, and fills in ``"model"`` before vLLM's defaulting ever runs, leaving

    --speculative-config '{"method":"mtp","num_speculative_tokens":7,
                           "draft_load_config":{"load_format":"dummy"}}'

as the whole invocation.
"""

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

from atom.utils.backends import VLLM_CACHE_ROOT

logger = logging.getLogger("atom")

# Backbone layers kept in the generated draft config. MiniMaxM3MultiTokenPredictor
# keys its MTP layer at num_hidden_layers and instantiates only
# num_nextn_predict_layers of them, so this depth is an index base rather than a
# layer count -- but it also sizes the per-layer sparse-attention lists, which
# the index-cache accounting sums over, so it is not free to leave at 60.
_DRAFT_BACKBONE_LAYERS = 4

# Depth num_speculative_tokens may reach. M3 declares num_mtp_modules=7; the
# MXFP8 export declares 1, hence the floor rather than a plain copy.
_DRAFT_MTP_MODULES = 7


def _is_minimax_m3_config(cfg: dict) -> bool:
    """Match the same way ``SpeculativeConfig.hf_config_override`` does.

    Architecture plus the lightning-indexer block, because M3's text_config
    carries no model_type of its own and the one spelling that does appear
    ("minimax_m2") is also a genuine MiniMax-M2's.
    """
    text = cfg.get("text_config", cfg)
    arch = (cfg.get("architectures") or text.get("architectures") or [""])[0]
    looks_like_m3 = arch.startswith("MiniMaxM3") or str(
        cfg.get("model_type") or text.get("model_type") or ""
    ).startswith("minimax_m3")
    return looks_like_m3 and bool(text.get("sparse_attention_config"))


def _resolve_config_dir(model: str, revision: str | None) -> str | None:
    """Local directory holding ``model``'s config.json, downloading if needed."""
    if os.path.isdir(model):
        return model
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            model, revision=revision, allow_patterns=["config.json", "*.py"]
        )
    except Exception as exc:  # offline, gated repo, bad id -- all non-fatal here
        logger.debug("ATOM patch: cannot resolve %s for MTP draft: %s", model, exc)
        return None


def _build_draft_config(src_dir: str) -> dict:
    """Derive an unquantized bf16 MTP-draft config from a target config."""
    cfg = json.loads((Path(src_dir) / "config.json").read_text())
    text = cfg.get("text_config", cfg)

    # Dummy init cannot fill MXFP4/MXFP8 params, so the draft must be plain bf16.
    cfg.pop("quantization_config", None)
    text.pop("quantization_config", None)
    cfg["torch_dtype"] = text["torch_dtype"] = "bfloat16"

    keep = _DRAFT_BACKBONE_LAYERS
    text["num_hidden_layers"] = keep
    for key, container in (
        ("moe_layer_freq", text),
        ("sparse_attention_freq", text.get("sparse_attention_config", {})),
        ("sparse_disable_index_value", text.get("sparse_attention_config", {})),
    ):
        value = container.get(key)
        if isinstance(value, list):
            container[key] = value[:keep]

    # num_local_experts is left at the target's value on purpose: it sizes the
    # FusedMoE, gate and correction bias inside the MTP block
    # (MiniMaxM3MTPLayer builds a MiniMaxM3MoE unconditionally), so shrinking it
    # would understate drafting cost and make any TPOT number meaningless.

    # num_nextn_predict_layers = how many MTP layers are instantiated; the single
    # one is reused modulo that count. num_mtp_modules = the depth
    # num_speculative_tokens may reach (atom.config.Config.__post_init__).
    text["num_nextn_predict_layers"] = 1
    text["num_mtp_modules"] = max(
        int(text.get("num_mtp_modules") or 0), _DRAFT_MTP_MODULES
    )
    return cfg


def _materialize_draft(src_dir: str) -> str:
    """Write the generated draft config beside the rest of ATOM's cache.

    Content-addressed by the target path and the knobs above, so switching
    checkpoints does not silently reuse a stale directory, and
    ``rm -rf ~/.cache/atom/*`` clears it like everything else ATOM caches.
    """
    cfg = _build_draft_config(src_dir)
    digest = hashlib.sha1(
        json.dumps(cfg, sort_keys=True).encode(), usedforsecurity=False
    ).hexdigest()[:12]
    dst = Path(VLLM_CACHE_ROOT) / "m3_mtp_draft" / digest
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "config.json").write_text(json.dumps(cfg, indent=2))

    # Under --trust-remote-code transformers resolves the config class through
    # auto_map, which names a module that has to sit next to config.json.
    for entry in (cfg.get("auto_map") or {}).values():
        module = str(entry).split(".", 1)[0] + ".py"
        source = Path(src_dir) / module
        if source.is_file():
            shutil.copy(source, dst / module)
    return str(dst)


def _wants_generated_draft(engine_args, spec: dict) -> bool:
    """True when the user asked for MTP + dummy draft weights and no path.

    Both spellings of every key are accepted: ``create_speculative_config``
    folds ``-`` to ``_`` and merges the ``--spec-*`` shorthands only after this
    hook has run.
    """
    method = spec.get("method") or getattr(engine_args, "spec_method", None)
    if str(method).lower() != "mtp":
        return False
    if spec.get("model") or getattr(engine_args, "spec_model", None):
        return False
    load_config = spec.get("draft_load_config") or spec.get("draft-load-config")
    if isinstance(load_config, dict):
        load_format = load_config.get("load_format") or load_config.get("load-format")
    else:
        load_format = getattr(load_config, "load_format", None)
    return str(load_format).lower() == "dummy"


def apply_vllm_m3_mtp_draft_patch() -> None:
    """Generate the M3 MTP draft config when the user omits ``model``.

    Hooks ``EngineArgs.create_speculative_config`` for the same reason
    ``dspark_dcp_patch`` does: the defaulting to fix sits inside
    ``SpeculativeConfig.__post_init__``, which pydantic binds when it builds the
    dataclass, so replacing the attribute afterwards has no effect. This is the
    last point at which the arguments are still a plain dict.
    """
    from vllm.engine.arg_utils import EngineArgs

    if getattr(EngineArgs, "_atom_m3_mtp_draft_patch", False):
        return

    original_create = EngineArgs.create_speculative_config

    def create_speculative_config(self, target_model_config, *args, **kwargs):
        spec = self.speculative_config
        if isinstance(spec, dict) and _wants_generated_draft(self, spec):
            src_dir = _resolve_config_dir(
                target_model_config.model, target_model_config.revision
            )
            if src_dir is not None and _is_minimax_m3_config(
                json.loads((Path(src_dir) / "config.json").read_text())
            ):
                draft = _materialize_draft(src_dir)
                logger.info(
                    "ATOM patch: generated a dummy-weight MiniMax-M3 MTP draft "
                    "config at %s (defaulting it to the quantized target would "
                    "fail dummy init).",
                    draft,
                )
                spec["model"] = draft
        return original_create(self, target_model_config, *args, **kwargs)

    EngineArgs.create_speculative_config = create_speculative_config
    EngineArgs._atom_m3_mtp_draft_patch = True
