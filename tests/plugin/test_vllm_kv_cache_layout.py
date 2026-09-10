"""vLLM registration -> ATOM KVCacheTensor mapping, on MiniMax-M3's real layouts.

M3 is the model that forced this mapping: it registers three different physical
layouts at once, and its K/V-separated layers arrive in one of *two* memory
orders, because the backends publish both as acceptable and vLLM picks at
startup:

    LHBNC  (2, B, N, C) in memory -- the whole tensor is not contiguous, only
                                    ``t[:, 0]`` / ``t[:, 1]`` are, so it must
                                    be split before ``DenseKVByteCodec``
    LBHNC  (B, 2, N, C) in memory -- block-compact, so the block is one opaque
                                    run and splitting it would produce two
                                    strided halves the codec must reject

Both carry the same shape, so only contiguity separates them. The LHBNC strides
here are the ones measured on M3-MXFP4 (see FINDINGS in the M3 offload notes),
scaled down in block count -- the contiguity properties depend on the axis
order, not on how many blocks there are.
"""

from __future__ import annotations

import pytest
import torch

from atom.plugin.vllm.kv_transfer.kv_cache_layout import (
    build_kv_cache_tensors,
    kv_layout_tag,
    split_kv_tensor,
)

NB, BS, HD = 8, 128, 128
DENSE_LAYERS, SPARSE_LAYERS = 3, 5


def _sparse_kv(nb: int = NB) -> torch.Tensor:
    """LHBNC: K and V in two separate regions, exactly as M3 allocates them.

    ``stride(1)`` jumps the whole K region, so the tensor is not contiguous --
    which is the entire point of the split this test covers.
    """
    k_block = BS * HD
    k_total = nb * k_block
    buf = torch.zeros(2 * k_total, dtype=torch.uint8)
    return buf.as_strided((nb, 2, BS, 1, HD), (k_block, k_total, HD, HD, 1))


def _sparse_kv_lbhnc(nb: int = NB) -> torch.Tensor:
    """LBHNC: the same logical cache, block-compact.

    Identical shape to ``_sparse_kv``; the difference is only that a block's K
    and V bytes are adjacent, which is exactly why shape cannot decide the
    segmentation.
    """
    return torch.zeros((nb, 2, BS, 1, HD), dtype=torch.uint8)


class _SparseLayer:
    """A stand-in for M3's sparse attention: fp8 scales live on the layer."""

    def __init__(self, nb: int = NB, heads: int = 1) -> None:
        self.kv_scale = torch.zeros((2, nb, heads, BS), dtype=torch.float32)

    def get_kv_transfer_scales(self, kv_cache=None):
        return self.kv_scale[0], self.kv_scale[1]


class _LayerWithUnreportableScales:
    """Per-block scales and no way to report them -- must not pass silently."""

    def __init__(self) -> None:
        self.k_scale = torch.zeros((NB, 1, BS), dtype=torch.float32)


def _m3_layers(kv: dict[str, torch.Tensor]) -> dict[str, object]:
    return {
        name: _SparseLayer()
        for name, tensor in kv.items()
        if tensor.ndim == 5  # sparse layers are the fp8-scaled ones
    }


def _m3_registration(sparse=_sparse_kv) -> dict[str, torch.Tensor]:
    kv: dict[str, torch.Tensor] = {}
    for i in range(DENSE_LAYERS):  # K/V interleaved per token
        kv[f"model.layers.{i}.self_attn.attn"] = torch.zeros(
            (NB, 1, BS, 2 * HD), dtype=torch.uint8
        )
    for i in range(DENSE_LAYERS, DENSE_LAYERS + SPARSE_LAYERS):
        name = f"model.layers.{i}.self_attn.attn"
        kv[name] = sparse()
        kv[f"{name}.index_cache"] = torch.zeros((NB, BS, HD), dtype=torch.float8_e4m3fn)
    return kv


def test_lhbnc_sparse_layer_is_only_movable_once_split():
    sparse = _sparse_kv()
    assert not sparse.is_contiguous(), "fixture no longer reproduces M3's layout"

    k, v = split_kv_tensor(sparse)
    assert k.is_contiguous() and v.is_contiguous()
    assert k.numel() == v.numel() == NB * BS * HD


def test_lbhnc_sparse_layer_travels_as_one_block_run():
    """Same shape as LHBNC, opposite answer -- so shape cannot be the test.

    Splitting a block-compact cache on dim 1 yields two strided halves the
    codec rejects, which would have taken the server down at startup.
    """
    sparse = _sparse_kv_lbhnc()
    assert sparse.is_contiguous()

    k, v = split_kv_tensor(sparse)
    assert v is None
    assert k is sparse


def test_layout_that_is_neither_is_rejected_rather_than_guessed():
    # A view whose blocks and whose K/V planes are both strided: there is no
    # segmentation that moves the right bytes, so refusing beats picking one.
    odd = torch.zeros((NB, 2, BS, 1, 2 * HD), dtype=torch.uint8)[..., :HD]
    assert not odd.is_contiguous() and not odd[:, 0].is_contiguous()

    with pytest.raises(ValueError, match="neither block-contiguous"):
        split_kv_tensor(odd)


def test_the_four_dimensional_views_split_the_same_way():
    """vLLM 0.28 folds the singleton num_kv_heads axis away.

    Under `num_head_slots=2` a layer is bound `[B, H, N, C]`, so the real gluon
    registration is 4-D, not the 5-D of the legacy `get_kv_cache_shape` path.
    Same two layouts, same decision -- and the tensor the connector measures
    `num_blocks` from is still block-first either way.
    """
    k_total = NB * BS * HD
    buf = torch.zeros(2 * k_total, dtype=torch.uint8)
    lhbnc = buf.as_strided((NB, 2, BS, HD), (BS * HD, k_total, HD, 1))
    lbhnc = torch.zeros((NB, 2, BS, HD), dtype=torch.uint8)

    k, v = split_kv_tensor(lhbnc)
    assert v is not None and k.is_contiguous() and v.is_contiguous()
    assert k.shape[0] == NB

    k, v = split_kv_tensor(lbhnc)
    assert v is None and k.shape[0] == NB


def test_kv_layout_tag_separates_the_two_packings():
    """The namespace must not let two layouts share one key space.

    Same bytes per block either way, arranged differently -- a hit across the
    two would restore transposed KV with nothing to log.
    """
    lhbnc = build_kv_cache_tensors(_m3_registration())
    lbhnc = build_kv_cache_tensors(_m3_registration(sparse=_sparse_kv_lbhnc))

    assert kv_layout_tag(lhbnc) == "kv-split"
    assert kv_layout_tag(lbhnc) == "kv-whole"


def test_both_layouts_map_and_move_the_same_bytes_per_block():
    codec_mod = pytest.importorskip(
        "atom.kv_transfer.offload.dense.kv_byte_codec",
        reason="offload codec pulls aiter",
    )

    def _bytes(sparse):
        tensors = build_kv_cache_tensors(_m3_registration(sparse=sparse))
        return codec_mod.DenseKVByteCodec(
            {str(t.layer_num): t for t in tensors}, num_blocks=NB
        ).bytes_per_block

    # One K + one V segment vs one K+V run: different segmentation, identical
    # payload. A mismatch here would mean one layout silently drops or
    # duplicates a region.
    assert _bytes(_sparse_kv) == _bytes(_sparse_kv_lbhnc)


def test_dense_layer_travels_whole():
    dense = torch.zeros((NB, 1, BS, 2 * HD), dtype=torch.uint8)
    k, v = split_kv_tensor(dense)
    assert v is None, "dense K/V interleave inside a block; splitting is meaningless"
    assert k is dense


def test_index_caches_fold_into_their_owning_layer():
    tensors = build_kv_cache_tensors(_m3_registration())

    assert (
        len(tensors) == DENSE_LAYERS + SPARSE_LAYERS
    ), "index caches must not become layers of their own"
    with_index = [t for t in tensors if t.index_cache is not None]
    assert len(with_index) == SPARSE_LAYERS


def test_layer_order_is_numeric_not_dict_order():
    kv = _m3_registration()
    shuffled = dict(reversed(list(kv.items())))

    got = build_kv_cache_tensors(shuffled)

    # Segment order must not depend on registration order: a save written in
    # one order and restored in another would scatter bytes to the wrong layers.
    assert [t.layer_num for t in got] == list(range(DENSE_LAYERS + SPARSE_LAYERS))
    assert got[0].v_cache.numel() == 0, "layer 0 is dense -> no separate V"
    assert got[-1].index_cache is not None, "last layer is sparse -> has an index cache"


def test_orphan_index_cache_is_rejected():
    with pytest.raises(ValueError, match="without their owning layer"):
        build_kv_cache_tensors(
            {"model.layers.0.self_attn.attn.index_cache": torch.zeros((NB, BS, HD))}
        )


def test_fp8_scales_travel_with_the_layer_they_belong_to():
    """Mantissas without their scales restore as fluent garbage, silently.

    M3's sparse cache scales fp8 per token AND per head, and vLLM's
    registration dict does not carry that table -- the layer does. This is the
    regression that made every warm hit produce garbage while every metric said
    the transfer had succeeded.
    """
    kv = _m3_registration()
    tensors = build_kv_cache_tensors(kv, _m3_layers(kv))

    sparse = [t for t in tensors if t.index_cache is not None]
    assert len(sparse) == SPARSE_LAYERS
    for t in sparse:
        assert t.k_scale is not None and t.v_scale is not None
        assert t.k_scale.shape[0] == NB, "scales must be block-major"
        assert t.k_scale.is_contiguous() and t.v_scale.is_contiguous()

    dense = [t for t in tensors if t.index_cache is None]
    assert all(t.k_scale is None for t in dense), "dense scales are per-tensor"


def test_scales_reach_the_codec_as_extra_segments():
    codec_mod = pytest.importorskip(
        "atom.kv_transfer.offload.dense.kv_byte_codec",
        reason="offload codec pulls aiter",
    )
    kv = _m3_registration()
    with_scales = build_kv_cache_tensors(kv, _m3_layers(kv))
    without = build_kv_cache_tensors(kv)

    def _bytes(tensors):
        return codec_mod.DenseKVByteCodec(
            {str(t.layer_num): t for t in tensors}, num_blocks=NB
        ).bytes_per_block

    # 2 scales x 1 head x BS tokens x fp32, per sparse layer.
    assert _bytes(with_scales) - _bytes(without) == SPARSE_LAYERS * 2 * BS * 4


def test_per_block_scales_without_a_reporting_hook_are_rejected():
    kv = {"model.layers.0.self_attn.attn": torch.zeros((NB, 1, BS, 2 * HD))}
    with pytest.raises(ValueError, match="per-block k_scale"):
        build_kv_cache_tensors(
            kv, {"model.layers.0.self_attn.attn": _LayerWithUnreportableScales()}
        )


def test_non_block_major_scales_are_rejected():
    class _HeadMajor:
        def get_kv_transfer_scales(self, kv_cache=None):
            scale = torch.zeros((1, NB, BS), dtype=torch.float32)
            return scale, scale

    kv = {"model.layers.0.self_attn.attn": torch.zeros((NB, 1, BS, 2 * HD))}
    with pytest.raises(ValueError, match="not block-major"):
        build_kv_cache_tensors(kv, {"model.layers.0.self_attn.attn": _HeadMajor()})


def test_codec_accepts_the_mapped_tensors():
    """The mapping's whole purpose: make M3 pass ATOM's byte codec."""
    codec_mod = pytest.importorskip(
        "atom.kv_transfer.offload.dense.kv_byte_codec",
        reason="offload codec pulls aiter",
    )
    tensors = build_kv_cache_tensors(_m3_registration())

    codec = codec_mod.DenseKVByteCodec(
        {str(t.layer_num): t for t in tensors}, num_blocks=NB
    )

    dense_bytes = BS * 2 * HD  # one opaque K/V run
    sparse_bytes = 2 * (BS * HD) + BS * HD  # K + V + index
    assert codec.bytes_per_block == (
        DENSE_LAYERS * dense_bytes + SPARSE_LAYERS * sparse_bytes
    )


def test_a_layer_whose_leading_axis_is_not_blocks_is_rejected():
    """The one symptom of a vLLM that bound the caches through the legacy path.

    There, dense arrives as ``(2, nb, ...)`` and its block count reads as 2
    while the sparse layers still read ``nb``. Nothing about the dtypes or the
    sizes looks wrong; only the disagreement does.
    """
    kv = _m3_registration()
    layers = _m3_layers(kv)  # dense layers carry a scalar scale, so no hook
    legacy = "model.layers.0.self_attn.attn"
    kv[legacy] = torch.zeros((2, NB, BS, 1, 2 * HD), dtype=torch.uint8)

    with pytest.raises(ValueError, match="disagree on the block count"):
        build_kv_cache_tensors(kv, layers)


def test_agreeing_block_counts_are_not_disturbed():
    """Both layouts already agree, so the guard must stay out of the way."""
    for sparse in (_sparse_kv, _sparse_kv_lbhnc):
        kv = _m3_registration(sparse=sparse)
        tensors = build_kv_cache_tensors(kv, _m3_layers(kv))
        assert {int(t.k_cache.shape[0]) for t in tensors} == {NB}
