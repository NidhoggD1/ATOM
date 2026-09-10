"""VllmConfig -> ATOM offload config projection.

The offload path reads these fields off ATOM's Config; in plugin mode vLLM owns
them. Getting the KV dtype wrong is not a crash but a sizing error -- the codec
derives every byte segment from it -- so the mapping fails closed on anything
it does not recognise.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest

from atom.plugin.vllm.kv_transfer.offload_config import build_offload_config


def _vllm_config(cache_dtype="fp8", model_dtype="bfloat16", block_size=128, **kw):
    return NS(
        cache_config=NS(block_size=block_size, cache_dtype=cache_dtype),
        model_config=NS(
            hf_config=NS(num_hidden_layers=60, model_type="minimax_m3"),
            dtype=model_dtype,
            model="/models/MiniMax-M3-MXFP4",
        ),
        parallel_config=NS(
            pipeline_parallel_size=kw.get("pp", 1),
            tensor_parallel_size=kw.get("tp", 1),
            decode_context_parallel_size=kw.get("dcp", 1),
        ),
        kv_transfer_config=NS(
            kv_role=kw.get("role", "kv_both"),
            kv_connector_extra_config=kw.get("extra", {}),
        ),
    )


def test_projects_the_fields_offload_reads():
    cfg = build_offload_config(_vllm_config())

    assert cfg.kv_cache_block_size == 128
    assert cfg.kv_cache_dtype == "fp8"
    assert cfg.hf_config.num_hidden_layers == 60
    assert cfg.kv_transfer_config["kv_role"] == "kv_both"


def test_tensor_parallel_size_reaches_the_scheduler_side():
    """``lmcache_replica_world_size`` reads it, in a process with no TP group.

    Left unset it defaults to 1, so a TP=4 server would size its replica as a
    single worker on the scheduler side while the workers count 4 off the aiter
    group -- two answers to one question.
    """
    from atom.kv_transfer.offload.config import lmcache_replica_world_size

    cfg = build_offload_config(_vllm_config(tp=4, pp=2))

    assert cfg.tensor_parallel_size == 4
    assert lmcache_replica_world_size(cfg) == 8


def test_page_layout_tag_starts_unset():
    """Only the worker, holding the registered tensors, can know the layout.

    Unset means "not applicable", and ``build_page_namespace`` then leaves the
    key exactly as the native path computes it.
    """
    assert build_offload_config(_vllm_config()).page_layout_tag is None


def test_page_layout_tag_changes_the_offload_namespace():
    pytest.importorskip(
        "atom.kv_transfer.offload.config", reason="offload config pulls aiter"
    )
    from atom.kv_transfer.offload.config import build_page_namespace

    lmcache_cfg = NS(chunk_size=128)
    base = build_offload_config(_vllm_config())
    split = build_offload_config(_vllm_config())
    split.page_layout_tag = "kv-split"
    whole = build_offload_config(_vllm_config())
    whole.page_layout_tag = "kv-whole"

    keys = {
        name: build_page_namespace(cfg, lmcache_cfg, 4)
        for name, cfg in (("base", base), ("split", split), ("whole", whole))
    }

    assert len(set(keys.values())) == 3, keys
    # Unset must reproduce the pre-existing key, or every native deployment
    # loses its cache on upgrade.
    assert keys["base"] == build_page_namespace(
        NS(
            hf_config=base.hf_config,
            model_tag=base.model_tag,
            kv_cache_dtype=base.kv_cache_dtype,
            kv_cache_block_size=base.kv_cache_block_size,
            decode_context_parallel_size=1,
        ),
        lmcache_cfg,
        4,
    )


def test_auto_kv_dtype_resolves_to_the_model_dtype():
    cfg = build_offload_config(_vllm_config(cache_dtype="auto", model_dtype="bfloat16"))
    assert cfg.kv_cache_dtype == "bf16"

    cfg = build_offload_config(
        _vllm_config(cache_dtype="auto", model_dtype="torch.float16")
    )
    assert cfg.kv_cache_dtype == "fp16"


@pytest.mark.parametrize("raw", ["fp8_e4m3", "fp8_e5m2", "fp8_inc"])
def test_fp8_spellings_all_map_to_fp8(raw):
    assert build_offload_config(_vllm_config(cache_dtype=raw)).kv_cache_dtype == "fp8"


def test_unknown_kv_dtype_fails_closed():
    # Guessing a width here would mis-size every byte segment silently.
    with pytest.raises(ValueError, match="unsupported KV cache dtype"):
        build_offload_config(_vllm_config(cache_dtype="int4_magic"))


def test_missing_block_size_is_rejected():
    with pytest.raises(ValueError, match="block_size"):
        build_offload_config(_vllm_config(block_size=None))


def test_missing_hf_config_is_rejected():
    bad = _vllm_config()
    bad.model_config.hf_config = None
    with pytest.raises(ValueError, match="hf_config"):
        build_offload_config(bad)


def test_extra_connector_config_is_carried_through():
    cfg = build_offload_config(_vllm_config(extra={"lmcache.foo": 7}))
    assert cfg.kv_transfer_config["lmcache.foo"] == 7


def _nested_vllm_config():
    """A multimodal config: the transformer's fields live on the text config.

    MiniMaxM3Config genuinely has no num_hidden_layers -- reading it off the
    outer config raises, which is how the first live run died.
    """
    text = NS(num_hidden_layers=60, model_type="minimax_m3_text")
    outer = NS(model_type="minimax_m3", text_config=text)  # no num_hidden_layers
    cfg = _vllm_config()
    cfg.model_config.hf_config = outer
    cfg.model_config.hf_text_config = text
    return cfg


def test_nested_text_config_supplies_the_layer_count():
    cfg = build_offload_config(_nested_vllm_config())

    assert cfg.hf_config.num_hidden_layers == 60


def test_outer_only_fields_still_resolve():
    cfg = build_offload_config(_nested_vllm_config())
    cfg._vllm_config.model_config.hf_config.architectures = ["MiniMaxM3ForCausalLM"]

    # Inner-first, outer as fallback: both readers in ATOM's offload path are
    # satisfied without either changing which attribute it asks for.
    assert cfg.hf_config.architectures == ["MiniMaxM3ForCausalLM"]


def test_flat_config_is_unaffected():
    cfg = build_offload_config(_vllm_config())
    assert cfg.hf_config.num_hidden_layers == 60
