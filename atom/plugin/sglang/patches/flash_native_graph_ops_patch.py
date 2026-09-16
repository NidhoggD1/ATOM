# SPDX-License-Identifier: MIT
"""Pin Flash-Next decode CUDA-graph scratch without editing Native ops.

Native ``torch.empty`` / ``.contiguous()`` inside captured decode becomes a
dangling alias after a long eager prefill reclaims the caching-allocator slab
(HSA fault on first long-decode replay). The previous branch patched
``atom/model_ops`` and ``atom/models/qwen4_exp.py`` directly. This plugin
monkeypatches the same call sites so the SGLang PR stays plugin-only.

PLE cannot live in a full CUDA graph (n-gram gather / FP8 GEMM / TP all-reduce
allocate). Wrap layer-1 PLE as a SGLang breakable-graph eager island.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)


def _ws():
    from atom.plugin.sglang import flash_decode_graph_workspace as _fws

    return _fws


def _flash_ple_metadata():
    from atom.utils.forward_context import get_forward_context

    ctx = get_forward_context()
    md = getattr(ctx, "attn_metadata", None)
    if md is None:
        ple = None
    elif isinstance(md, dict):
        ple = md.get("ple_metadata")
    else:
        ple = getattr(md, "ple_metadata", None)
    if ple is not None:
        return ple
    try:
        from atom.plugin.sglang.qwen3_8_flash_next_bridge import _DECODE_GRAPH

        return getattr(_DECODE_GRAPH, "last_ple", None)
    except Exception:  # noqa: BLE001
        return None


@contextmanager
def _redirect_torch_empty(next_tensors: list[torch.Tensor | None]):
    """Return ``next_tensors[i]`` for the i-th ``torch.empty`` if not None."""

    real = torch.empty
    idx = {"n": 0}

    def empty(*args, **kwargs):
        i = idx["n"]
        idx["n"] += 1
        if i < len(next_tensors) and next_tensors[i] is not None:
            return next_tensors[i]
        return real(*args, **kwargs)

    torch.empty = empty  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.empty = real  # type: ignore[method-assign]


@contextmanager
def _redirect_empty_like(first: torch.Tensor | None):
    real = torch.empty_like
    used = {"n": False}

    def empty_like(x, *args, **kwargs):
        if first is not None and not used["n"]:
            used["n"] = True
            if tuple(first.shape) == tuple(x.shape) and first.dtype == x.dtype:
                return first
        return real(x, *args, **kwargs)

    torch.empty_like = empty_like  # type: ignore[method-assign]
    try:
        yield
    finally:
        torch.empty_like = real  # type: ignore[method-assign]


def _wrap_fused_gdn_gating() -> None:
    import atom.model_ops.attention_gdn as gdn_mod

    if getattr(gdn_mod.fused_gdn_gating, "_atom_flash_ws", False):
        return
    orig = gdn_mod.fused_gdn_gating

    def fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
        pinned = None
        try:
            pinned = _ws().gdn_gate_out(a.shape[0], a.shape[1], b.dtype)
        except Exception:  # noqa: BLE001
            pinned = None
        if pinned is None:
            return orig(A_log, a, b, dt_bias, beta=beta, threshold=threshold)
        g, beta_output = pinned
        with _redirect_torch_empty([g, beta_output]):
            return orig(A_log, a, b, dt_bias, beta=beta, threshold=threshold)

    fused_gdn_gating._atom_flash_ws = True  # type: ignore[attr-defined]
    gdn_mod.fused_gdn_gating = fused_gdn_gating


def _wrap_rearrange_mixed_qkv() -> None:
    from atom.model_ops.attention_gdn import GatedDeltaNet

    if getattr(GatedDeltaNet.rearrange_mixed_qkv, "_atom_flash_ws", False):
        return
    orig = GatedDeltaNet.rearrange_mixed_qkv

    def rearrange_mixed_qkv(self, mixed_qkv):
        query, key, value = orig(self, mixed_qkv)
        try:
            fws = _ws()
            query = fws.pin_gdn_qkv(query, kind="q")
            key = fws.pin_gdn_qkv(key, kind="k")
            value = fws.pin_gdn_qkv(value, kind="v")
        except Exception:  # noqa: BLE001
            pass
        return query, key, value

    rearrange_mixed_qkv._atom_flash_ws = True  # type: ignore[attr-defined]
    GatedDeltaNet.rearrange_mixed_qkv = rearrange_mixed_qkv


def _wrap_causal_conv1d_update() -> None:
    import atom.model_ops.mamba_ops.causal_conv1d as conv_mod

    if getattr(conv_mod.causal_conv1d_update, "_atom_flash_ws", False):
        return
    orig = conv_mod.causal_conv1d_update

    def causal_conv1d_update(*args, **kwargs):
        pinned = None
        try:
            x = args[0] if args else kwargs.get("x")
            k_dim = kwargs.get("k_dim_size", args[3] if len(args) > 3 else None)
            v_dim = kwargs.get("v_dim_size", args[4] if len(args) > 4 else None)
            if x is not None and k_dim is not None and v_dim is not None:
                pinned = _ws().gdn_conv_qkv(
                    x.shape[0], int(k_dim), int(v_dim), x.dtype
                )
        except Exception:  # noqa: BLE001
            pinned = None
        if pinned is None:
            return orig(*args, **kwargs)
        query, key, value = pinned
        with _redirect_torch_empty([query, key, value]):
            return orig(*args, **kwargs)

    causal_conv1d_update._atom_flash_ws = True  # type: ignore[attr-defined]
    conv_mod.causal_conv1d_update = causal_conv1d_update


def _wrap_gemma_rmsnorm() -> None:
    from atom.model_ops.layernorm import GemmaRMSNorm

    if getattr(GemmaRMSNorm.forward, "_atom_flash_ws", False):
        return
    orig = GemmaRMSNorm.forward

    def forward(self, x, residual=None):
        pinned = None
        try:
            x_2d = x.view(-1, x.shape[-1])
            fws = _ws()
            if residual is None and fws.fits_head_norm(
                x_2d.shape[0], x_2d.shape[1], x_2d.dtype
            ):
                pinned = fws.head_norm_out(x_2d.shape[0], x_2d.shape[1], x_2d.dtype)
        except Exception:  # noqa: BLE001
            pinned = None
        if pinned is None:
            return orig(self, x, residual)
        with _redirect_empty_like(pinned):
            return orig(self, x, residual)

    forward._atom_flash_ws = True  # type: ignore[attr-defined]
    GemmaRMSNorm.forward = forward


def _wrap_fused_moe() -> None:
    import atom.model_ops.fused_moe_triton as moe_triton

    if getattr(moe_triton.triton_kernel_fused_experts, "_atom_flash_ws", False):
        return
    orig = moe_triton.triton_kernel_fused_experts

    def triton_kernel_fused_experts(*args, **kwargs):
        if kwargs.get("intermediate_cache") is None:
            try:
                hidden_states = args[1]
                w1 = args[2]
                topk = kwargs.get("topk")
                if topk is None and len(args) > 7:
                    topk = args[7]
                M = hidden_states.shape[-2]
                half_N = w1.shape[-1] // 2
                fws = _ws()
                if fws.fits_moe(M, topk, half_N, hidden_states.dtype):
                    kwargs = dict(kwargs)
                    kwargs["intermediate_cache"] = fws.moe_inter(
                        M, topk, half_N, hidden_states.dtype
                    )
            except Exception:  # noqa: BLE001
                pass
        return orig(*args, **kwargs)

    triton_kernel_fused_experts._atom_flash_ws = True  # type: ignore[attr-defined]
    moe_triton.triton_kernel_fused_experts = triton_kernel_fused_experts


def _wrap_mxfp4_moe_apply() -> None:
    from atom.model_ops.moe import Mxfp4MoEMethod

    if getattr(Mxfp4MoEMethod.apply, "_atom_flash_ws", False):
        return
    orig = Mxfp4MoEMethod.apply

    def apply(self, layer, x, *args, **kwargs):
        pinned = None
        try:
            fws = _ws()
            if fws.fits_moe(x.shape[0], 1, 1, x.dtype):
                pinned = fws.moe_out(x.shape[0], x.shape[-1], x.dtype)
        except Exception:  # noqa: BLE001
            pinned = None
        if pinned is None:
            return orig(self, layer, x, *args, **kwargs)
        with _redirect_empty_like(pinned):
            return orig(self, layer, x, *args, **kwargs)

    apply._atom_flash_ws = True  # type: ignore[attr-defined]
    Mxfp4MoEMethod.apply = apply


def _wrap_qsa_attention_forward() -> None:
    from atom.model_ops.qwen4_exp.qsa_attention import Qwen4ExpAttention

    if getattr(Qwen4ExpAttention.forward, "_atom_flash_ws", False):
        return
    orig = Qwen4ExpAttention.forward
    orig_split = torch.Tensor.split

    def forward(self, positions, hidden_states):
        try:
            fws = _ws()
            enabled = fws.is_enabled()
        except Exception:  # noqa: BLE001
            enabled = False
        if not enabled:
            return orig(self, positions, hidden_states)

        def split(tensor, split_size, dim=0):
            parts = orig_split(tensor, split_size, dim=dim)
            if (
                dim == -1
                and len(parts) == 4
                and split_size == [self.q_size, self.q_size, self.kv_size, self.kv_size]
            ):
                gate, q, k, v = parts
                return (
                    fws.as_contiguous(gate, kind="gate"),
                    fws.as_contiguous(q, kind="q"),
                    fws.as_contiguous(k, kind="k"),
                    fws.as_contiguous(v, kind="v"),
                )
            return parts

        torch.Tensor.split = split  # type: ignore[method-assign]
        try:
            return orig(self, positions, hidden_states)
        finally:
            torch.Tensor.split = orig_split  # type: ignore[method-assign]

    forward._atom_flash_ws = True  # type: ignore[attr-defined]
    Qwen4ExpAttention.forward = forward


def _apply_flash_ple(layer, hidden_states, input_ids):
    ple_metadata = _flash_ple_metadata()
    if ple_metadata is None:
        return None
    contrib = layer.ple.forward_with_state(
        hidden_states,
        input_ids,
        ple_metadata,
    )
    return hidden_states + contrib


def _wrap_decoder_ple() -> None:
    from atom.models.qwen4_exp import Qwen4ExpDecoderLayer

    if getattr(Qwen4ExpDecoderLayer.forward, "_atom_flash_ple", False):
        return
    orig = Qwen4ExpDecoderLayer.forward
    try:
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
            eager_on_graph,
        )
    except Exception:  # pragma: no cover - Native / no SGLang

        def eager_on_graph(enable, capture_stub=None):  # type: ignore[misc]
            del enable, capture_stub

            def decorator(inner):
                return inner

            return decorator

    if os.environ.get("ATOM_FLASH_PLE_EAGER", "1") != "0":
        apply_ple: Callable = eager_on_graph(True)(_apply_flash_ple)
    else:
        apply_ple = _apply_flash_ple

    def forward(self, positions, hidden_states, input_ids):
        if self.ple is None:
            return orig(self, positions, hidden_states, input_ids)
        patched = apply_ple(self, hidden_states, input_ids)
        if patched is None:
            return orig(self, positions, hidden_states, input_ids)
        ple = self.ple
        self.ple = None
        try:
            return orig(self, positions, patched, input_ids)
        finally:
            self.ple = ple

    forward._atom_flash_ple = True  # type: ignore[attr-defined]
    Qwen4ExpDecoderLayer.forward = forward


def apply_flash_native_graph_ops_patch() -> None:
    """Install workspace + PLE wrappers on Native Flash/GDN/MoE ops."""

    try:
        _wrap_fused_gdn_gating()
        _wrap_rearrange_mixed_qkv()
        _wrap_causal_conv1d_update()
        _wrap_gemma_rmsnorm()
        _wrap_fused_moe()
        _wrap_mxfp4_moe_apply()
        _wrap_qsa_attention_forward()
        _wrap_decoder_ple()
    except Exception:
        logger.exception("Flash Native graph-ops plugin patch failed")
        return
    logger.info("Patched Native Flash/GDN/MoE ops for plugin-owned decode graph workspace")
