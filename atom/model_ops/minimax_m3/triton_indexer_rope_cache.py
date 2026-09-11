"""Opt-in M3 norm/RoPE/cache prototype; not wired into production dispatch."""

import torch
import triton
import triton.language as tl


@triton.jit
def _norm_rope_cache(
    Packed,
    Weight,
    CosSin,
    Positions,
    Slots,
    Cache,
    Scale,
    Output,
    ROW_STRIDE: tl.constexpr,
    OFFSET: tl.constexpr,
    HEADS: tl.constexpr,
    ROTARY: tl.constexpr,
    EPS: tl.constexpr,
    KIND: tl.constexpr,
    X: tl.constexpr,
    FP8: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 128)
    raw = tl.load(Packed + token * ROW_STRIDE + OFFSET + head * 128 + d)
    value = raw.to(tl.float32)
    if KIND != 2:  # V is neither normalized nor rotated.
        inv = tl.rsqrt(tl.sum(value * value, 0) / 128 + EPS)
        value = value * inv * (1.0 + tl.load(Weight + d).to(tl.float32))
        half = ROTARY // 2
        partner_d = tl.where(d < half, d + half, d - half)
        partner_d = tl.where(d < ROTARY, partner_d, d)
        partner = tl.gather(value, partner_d, 0)
        pos = tl.load(Positions + token)
        c = tl.load(CosSin + pos * ROTARY + d % half).to(tl.float32)
        s = tl.load(CosSin + pos * ROTARY + half + d % half).to(tl.float32)
        rotated = tl.where(d < half, value * c - partner * s, value * c + partner * s)
        value = tl.where(d < ROTARY, rotated, value)
    if KIND == 0:
        tl.store(Output + (token * HEADS + head) * 128 + d, value)
    else:
        slot = tl.load(Slots + token)
        if slot >= 0:
            if KIND == 3:  # Single shared index-K, unit-scale FP8.
                rounded = value.to(raw.dtype).to(tl.float32)
                if FP8:
                    rounded = tl.minimum(tl.maximum(rounded, -FP8_MAX), FP8_MAX)
                tl.store(Cache + slot * 128 + d, rounded)
            else:
                page, pos = slot // 16, slot % 16
                if FP8:
                    amax = tl.max(tl.abs(value), 0)
                    scale = tl.where(amax > 0, tl.div_rn(amax, FP8_MAX), 1.0)
                    tl.store(Scale + (page * HEADS + head) * 16 + pos, scale)
                    value = tl.div_rn(value.to(raw.dtype).to(tl.float32), scale)
                    value = tl.minimum(tl.maximum(value, -FP8_MAX), FP8_MAX)
                base = (page * HEADS + head) * 128 * 16
                if KIND == 1:
                    offset = base + (d // X) * (16 * X) + pos * X + d % X
                else:
                    offset = base + (pos // X) * (128 * X) + d * X + pos % X
                tl.store(Cache + offset, value)


def triton_indexer_rope_cache(
    qkv,
    q_norm_weight,
    k_norm_weight,
    cos_sin_cache,
    positions,
    num_heads,
    num_kv_heads,
    rotary_dim,
    eps,
    index_q_norm_weight,
    index_k_norm_weight,
    num_index_heads,
    slot_mapping,
    k_cache,
    v_cache,
    index_cache,
    index_slot_mapping,
    *,
    k_scale=None,
    v_scale=None,
):
    """Return main/index Q and insert caches with independent slot mappings.

    Caller guarantees valid positions and unique nonnegative destination slots.
    A negative slot skips only that cache's write, not query computation.
    BF16 inputs, page-16 SHUFFLE main KV and page-128 index K only.
    """
    fp8_types = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    if qkv.dtype != torch.bfloat16 or qkv.ndim != 2:
        raise ValueError("qkv must be a two-dimensional BF16 tensor")
    if min(num_heads, num_kv_heads, num_index_heads) < 1:
        raise ValueError("head counts must be positive")
    if rotary_dim not in (64, 128):
        raise ValueError("prototype supports rotary_dim 64 or 128")
    width = (num_heads + 2 * num_kv_heads + num_index_heads + 1) * 128
    if qkv.shape[1] != width:
        raise ValueError("packed projection width does not match head counts")
    weights = (q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight)
    if any(w.shape != (128,) or w.dtype != qkv.dtype for w in weights):
        raise ValueError("norm weights must be BF16 vectors of length 128")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != rotary_dim
        or cos_sin_cache.dtype != qkv.dtype
    ):
        raise ValueError("cos_sin_cache must be BF16 [positions, rotary_dim]")
    tokens = qkv.shape[0]
    for mapping in (positions, slot_mapping, index_slot_mapping):
        if mapping.shape != (tokens,) or mapping.dtype != torch.int64:
            raise ValueError("positions and slot mappings must be int64 [tokens]")
    if (
        k_cache.dtype not in (torch.bfloat16, *fp8_types)
        or v_cache.dtype != k_cache.dtype
    ):
        raise ValueError("main K/V must have matching BF16 or E4M3 FP8 dtype")
    if index_cache.dtype not in (torch.bfloat16, *fp8_types):
        raise ValueError("index cache must be BF16 or E4M3 FP8")
    x = 16 // k_cache.element_size()
    if k_cache.ndim != 5 or k_cache.shape[1:] != (num_kv_heads, 128 // x, 16, x):
        raise ValueError("K cache must use page-16 SHUFFLE layout")
    pages = k_cache.shape[0]
    if v_cache.shape != (pages, num_kv_heads, 16 // x, 128, x):
        raise ValueError("V cache must use page-16 SHUFFLE layout")
    if index_cache.ndim != 3 or index_cache.shape[1:] != (128, 128):
        raise ValueError("index cache must be [pages,128,128]")
    main_fp8 = k_cache.dtype in fp8_types
    if main_fp8:
        if any(
            s is None
            or s.shape != (pages, num_kv_heads, 16)
            or s.dtype != torch.float32
            for s in (k_scale, v_scale)
        ):
            raise ValueError("FP8 main caches require FP32 per-token K/V scales")
    elif k_scale is not None or v_scale is not None:
        raise ValueError("BF16 main caches do not use scales")
    tensors = [
        qkv,
        *weights,
        cos_sin_cache,
        positions,
        slot_mapping,
        index_slot_mapping,
        k_cache,
        v_cache,
        index_cache,
    ]
    tensors += [s for s in (k_scale, v_scale) if s is not None]
    if any(
        t.device != qkv.device or not t.is_cuda or not t.is_contiguous()
        for t in tensors
    ):
        raise ValueError("all tensors must be contiguous on the same GPU")
    q_out = torch.empty((tokens, num_heads * 128), device=qkv.device, dtype=qkv.dtype)
    iq_out = torch.empty(
        (tokens, num_index_heads * 128), device=qkv.device, dtype=qkv.dtype
    )
    if not tokens:
        return q_out, iq_out
    iq_offset = (num_heads + 2 * num_kv_heads) * 128
    jobs = (
        (num_heads, 0, q_norm_weight, slot_mapping, k_cache, k_scale, q_out, 0),
        (
            num_kv_heads,
            num_heads * 128,
            k_norm_weight,
            slot_mapping,
            k_cache,
            k_scale,
            q_out,
            1,
        ),
        (
            num_kv_heads,
            (num_heads + num_kv_heads) * 128,
            k_norm_weight,
            slot_mapping,
            v_cache,
            v_scale,
            q_out,
            2,
        ),
        (
            num_index_heads,
            iq_offset,
            index_q_norm_weight,
            index_slot_mapping,
            index_cache,
            None,
            iq_out,
            0,
        ),
        (
            1,
            iq_offset + num_index_heads * 128,
            index_k_norm_weight,
            index_slot_mapping,
            index_cache,
            None,
            iq_out,
            3,
        ),
    )
    for heads, offset, weight, slots, cache, scale, output, kind in jobs:
        fp8 = cache.dtype in fp8_types
        _norm_rope_cache[(tokens, heads)](
            qkv,
            weight,
            cos_sin_cache,
            positions,
            slots,
            cache,
            scale if scale is not None else cache,
            output,
            ROW_STRIDE=width,
            OFFSET=offset,
            HEADS=heads,
            ROTARY=rotary_dim,
            EPS=eps,
            KIND=kind,
            X=x,
            FP8=fp8,
            FP8_MAX=torch.finfo(cache.dtype).max if fp8 else 0.0,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q_out, iq_out
