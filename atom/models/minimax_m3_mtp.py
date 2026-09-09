# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Inference-only MiniMax-M3 MTP (Multi-Token Prediction) draft model for ATOM.

The M3 checkpoint ships its MTP weights under
``[language_model.]model.mtp.layers.{i}.*``, with the transformer block itself
nested one level deeper under ``transformer_layer.``. ``remap_mtp_weight_name``
flattens that layout onto this module tree:

    model.mtp.layers.{i}.{enorm,hnorm,eh_proj,final_layernorm}.*
        -> model.layers.{num_hidden_layers+i}.<same>.*
    model.mtp.layers.{i}.transformer_layer.*
        -> model.layers.{num_hidden_layers+i}.mtp_block.*
    model.embed_tokens.* / lm_head.*  -> passthrough (shared with the target)

Unlike the sglang port, ``block_sparse_moe`` is NOT renamed to ``mlp``: ATOM's
M3 decoder block already names its MoE ``block_sparse_moe``, so the checkpoint
spelling matches natively.
"""

import re
from typing import ClassVar

import torch
from torch import nn
from transformers import PretrainedConfig

from atom.config import Config, QuantizationConfig
from atom.model_ops.embed_head import ParallelLMHead, VocabParallelEmbedding
from atom.model_ops.layernorm import (
    GemmaRMSNorm,
    fused_allreduce_gemma_rms_norm,
    fused_allreduce_gemma_rms_norm_quant,
)
from atom.model_ops.linear import ReplicatedLinear
from atom.models.utils import IntermediateTensors, ckpt_has_tensor_suffix, maybe_prefix
from atom.utils.decorators import support_torch_compile

from .minimax_m3 import (
    MiniMaxM3MoE,
    MiniMaxM3SparseAttention,
    _get_text_config,
    _linear_consumes_per_token_fp8,
    make_minimax_m3_expert_params_mapping,
)


class MiniMaxM3MTPLayer(nn.Module):
    """The MTP block's transformer layer.

    Deliberately not ``MiniMaxM3DecoderLayer``: that class picks sparse-vs-dense
    attention and MoE-vs-MLP from ``sparse_attention_freq`` / ``moe_layer_freq``
    indexed by ``layer_num``, and the MTP layer id (``num_hidden_layers + i``)
    is past the end of both lists. The MTP block is unconditionally
    sparse-attention + MoE, so those choices are hard-coded here instead.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        prefix: str,
        cache_config: str = "bf16",
        index_cache_config: str = "auto",
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
        layer_num: int = 0,
    ) -> None:
        super().__init__()
        # ``layer_num`` is num_hidden_layers + i, which is absent from
        # sparse_attention_freq, so _should_skip_minimax_m3_index_topk resolves
        # to (False, -1): the draft layer always computes its own indexer top-k
        # and never reuses a target layer's cached one. That is correct -- the
        # cache is keyed per forward batch and the draft runs its own.
        self.self_attn = MiniMaxM3SparseAttention(
            config=config,
            layer_id=layer_num,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            cache_config=cache_config,
            index_cache_config=index_cache_config,
        )

        self.block_sparse_moe = MiniMaxM3MoE(
            config=config,
            layer_id=layer_num,
            quant_config=quant_config,
            params_dtype=params_dtype,
            prefix=f"{prefix}.block_sparse_moe",
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # Resolved here rather than in forward for the same reason as
        # MiniMaxM3DecoderLayer: reading QuantType off the lazy aiter proxy
        # inside the compiled region is sourceless to Dynamo and graph-breaks.
        # Only the input norm has a fused-quant variant -- the post-attention
        # norm feeds the MoE, which (like the main model's MoE layers) does not
        # take a per-token fp8 activation scale.
        self.fuse_input_ar_rmsnorm_quant = _linear_consumes_per_token_fp8(
            self.self_attn.qkv_proj
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states_scale = None
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        elif self.fuse_input_ar_rmsnorm_quant:
            hidden_states, hidden_states_scale, residual = (
                fused_allreduce_gemma_rms_norm_quant(
                    hidden_states, residual, self.input_layernorm
                )
            )
        else:
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.input_layernorm
            )

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            hidden_states_scale=hidden_states_scale,
        )
        hidden_states, residual = fused_allreduce_gemma_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class MiniMaxM3MTPPredictorLayer(nn.Module):
    """One MTP prediction step: enorm + hnorm + eh_proj + mtp_block + final norm."""

    def __init__(
        self,
        atom_config: Config,
        prefix: str,
        layer_idx: int,
    ) -> None:
        super().__init__()
        config = _get_text_config(atom_config.hf_config)
        self.config = config

        self.enorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=atom_config.quant_config,
            prefix=maybe_prefix(prefix, "eh_proj"),
        )

        self.mtp_block = MiniMaxM3MTPLayer(
            config=config,
            prefix=f"{prefix}.mtp_block",
            cache_config=atom_config.kv_cache_dtype,
            index_cache_config=atom_config.index_cache_dtype,
            quant_config=atom_config.quant_config,
            params_dtype=atom_config.torch_dtype,
            layer_num=layer_idx,
        )

        self.final_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        spec_step_index: int = 0,
    ) -> torch.Tensor:
        """Returns the POST-final-norm hidden of this MTP layer.

        Draft step 0 consumes the target's post-final-norm hidden, so every
        later step must be handed the same kind of tensor -- see
        DeepSeekMultiTokenPredictorLayer.forward for the full rationale.
        """
        assert inputs_embeds is not None
        # No fused_dual_rmsnorm_cat here: that helper folds two *RMSNorms*, and
        # these are GemmaRMSNorms (weight applied as 1 + w). Keep them explicit.
        hidden_states = self.eh_proj(
            torch.cat(
                [self.enorm(inputs_embeds), self.hnorm(previous_hidden_states)], dim=-1
            )
        )

        hidden_states, residual = self.mtp_block(
            positions=positions, hidden_states=hidden_states, residual=None
        )
        # mtp_block ends on block_sparse_moe, whose experts are built with
        # reduce_results=False, and self_attn.o_proj is likewise unreduced --
        # exactly like a main-model M3 layer. fused_allreduce_gemma_rms_norm
        # absorbs that pending all-reduce together with the residual add, which
        # is why this must NOT be written as `residual + hidden_states` followed
        # by a bare final_layernorm (that shape would leave the TP partial sums
        # unreduced).
        hidden_states, _ = fused_allreduce_gemma_rms_norm(
            hidden_states, residual, self.final_layernorm
        )
        return hidden_states


class MiniMaxM3MultiTokenPredictor(nn.Module):
    """Holds the ``num_nextn_predict_layers`` MTP steps plus the shared embedding.

    ``self.embed_tokens`` must live here (and the LM head at the top level) so
    ``Drafter.load_model`` can bind both to the target's copies -- it reaches for
    ``self.model.model.embed_tokens`` and ``self.model.lm_head``.
    """

    def __init__(
        self,
        *,
        atom_config: Config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        config = _get_text_config(atom_config.hf_config)
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = config.num_nextn_predict_layers

        # Keyed by the absolute layer index so the ModuleDict key matches the
        # rewritten checkpoint name produced by remap_mtp_weight_name.
        self.layers = torch.nn.ModuleDict(
            {
                str(idx): MiniMaxM3MTPPredictorLayer(
                    atom_config,
                    f"{prefix}.layers.{idx}",
                    layer_idx=idx,
                )
                for idx in range(
                    self.mtp_start_layer_idx,
                    self.mtp_start_layer_idx + self.num_mtp_layers,
                )
            }
        )

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        current_step_idx = spec_step_idx % self.num_mtp_layers
        return self.layers[str(self.mtp_start_layer_idx + current_step_idx)](
            input_ids,
            positions,
            previous_hidden_states,
            inputs_embeds,
            current_step_idx,
        )


@support_torch_compile
class MiniMaxM3MTP(nn.Module):
    # Copied verbatim from MiniMaxM3SparseForCausalLM: the MTP block's
    # self_attn is the same MiniMaxM3SparseAttention, so it packs the lightning
    # indexer's index_q/index_k projections into qkv_proj alongside q/k/v. The
    # leading dots are load-bearing -- they keep ".q_proj" from also matching
    # e.g. "index_q_proj".
    packed_modules_mapping: ClassVar[dict] = {
        ".index_q_proj": (".qkv_proj", "index_q"),
        ".index_k_proj": (".qkv_proj", "index_k"),
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        super().__init__()
        # atom_config here is the TARGET's config (the drafter builds the draft
        # model from it), so the text config has to be unwrapped the same way
        # MiniMaxM3SparseForCausalLM does.
        self.config = _get_text_config(atom_config.hf_config)

        # See deepseek_mtp.DeepSeekMTP for the full rationale: MTP eh_proj is
        # commonly stored as BF16 with no weight_scale even when the model's
        # global quant_config is FP8/MXFP4. Only skip quantizing eh_proj when
        # the checkpoint genuinely ships no scale tensor for it.
        if atom_config.quant_config is not None and not ckpt_has_tensor_suffix(
            atom_config.model, "eh_proj.weight_scale"
        ):
            atom_config.quant_config.apply_default_exclude_layers(["*.eh_proj"])

        self.model = MiniMaxM3MultiTokenPredictor(
            atom_config=atom_config, prefix=maybe_prefix(prefix, "model")
        )

        # Top level (not under self.model) because Drafter.load_model shares the
        # target's head by assigning onto `draft_model.lm_head`.
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            org_num_embeddings=self.config.vocab_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        return self.model(
            input_ids, positions, hidden_states, inputs_embeds, spec_step_idx
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        """``hidden_states`` is already post-final-norm (the predictor layer runs
        final_layernorm at the end of its forward), so this is a bare LM head."""
        return self.lm_head(hidden_states)

    def compute_draft_ids(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor:
        """Greedy draft token ids via distributed argmax -- each rank reduces its
        own vocab shard and only [N, 2] is all-gathered instead of the full
        [N, vocab]. Token-identical to compute_logits(...).argmax(-1)."""
        return self.lm_head.compute_argmax_token(hidden_states)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Mirrors MiniMaxM3Model.get_expert_mapping. M3 splits each expert into
        # w1/w2/w3 *and* gate_proj/down_proj/up_proj spellings, which
        # FusedMoE.make_expert_params_mapping does not cover -- hence the
        # model-specific builder. The +n_shared_experts slot is where the loader
        # parks the fused shared expert.
        num_fused_shared = getattr(self.config, "n_shared_experts", 0) or 0
        return make_minimax_m3_expert_params_mapping(
            self.config.num_local_experts + num_fused_shared
        )

    # Matches an optional ``language_model.`` prefix (present in the multimodal
    # M3 checkpoints) before the MTP path.
    _MTP_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"(?:language_model\.)?model\.mtp\.layers\.(\d+)\."
    )
    _PREDICTOR_KEYS: ClassVar[set[str]] = {
        "enorm",
        "hnorm",
        "eh_proj",
        "final_layernorm",
    }

    def remap_mtp_weight_name(self, name: str) -> str | None:
        """Remap checkpoint MTP weight names onto this module tree.

            [language_model.]model.mtp.layers.{N}.{enorm,hnorm,eh_proj,
                final_layernorm}.*      -> model.layers.{L}.<same>.*
            [language_model.]model.mtp.layers.{N}.transformer_layer.*
                                        -> model.layers.{L}.mtp_block.*
            [language_model.]model.embed_tokens.*  -> model.embed_tokens.*
            [language_model.]lm_head.*             -> lm_head.*
            everything else (model.norm.*, model.layers.*) -> None (target-only)

        where L = num_hidden_layers + N. Returning None drops the weight, which
        is what we want for the target backbone's tensors -- the draft only owns
        the mtp.* subtree plus the two shared tensors.

        Note ``transformer_layer.block_sparse_moe`` is deliberately NOT renamed
        to ``mlp`` (sglang's port does): ATOM's MiniMaxM3MTPLayer already names
        its MoE ``block_sparse_moe``, so the checkpoint spelling lines up.
        """
        # self.config (not speculative_config.draft_model_hf_config) so that L
        # is computed from exactly the same num_hidden_layers that keyed
        # MiniMaxM3MultiTokenPredictor.layers.
        cfg = self.config
        num_nextn = getattr(cfg, "num_nextn_predict_layers", 0)
        if num_nextn <= 0:
            return None

        m = self._MTP_PATTERN.match(name)
        if m is None:
            stripped = name.removeprefix("language_model.")
            # Shared with the target and rebound by Drafter.load_model; still
            # accepted here so a standalone draft checkpoint loads.
            if stripped.startswith(("model.embed_tokens.", "lm_head.")):
                return stripped
            return None

        idx = int(m.group(1))
        if idx >= num_nextn:
            return None

        layer = cfg.num_hidden_layers + idx
        suffix = name[m.end() :]

        if suffix.startswith("embed_tokens"):
            return f"model.{suffix}"
        if any(suffix.startswith(k) for k in self._PREDICTOR_KEYS):
            return f"model.layers.{layer}.{suffix}"
        suffix = suffix.removeprefix("transformer_layer.")
        return f"model.layers.{layer}.mtp_block.{suffix}"
