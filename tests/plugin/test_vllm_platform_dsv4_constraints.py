# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

from vllm.config import CUDAGraphMode

from atom.plugin.vllm.platform import _enforce_deepseek_v4_constraints


def _config(architecture: str, mode: CUDAGraphMode):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            architectures=[architecture],
            max_model_len=131072,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=mode),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        scheduler_config=SimpleNamespace(
            chunked_prefill_enabled=True,
            max_num_batched_tokens=8192,
        ),
    )


def test_dsv4_forces_cudagraph_none_without_disabling_compilation(caplog):
    config = _config("DeepseekV4ForCausalLM", CUDAGraphMode.FULL_AND_PIECEWISE)
    config.compilation_config.mode = 3

    _enforce_deepseek_v4_constraints(config)

    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert config.compilation_config.mode == 3
    assert "graph replay corrupts V4 logits" in caplog.text


def test_dsv4_keeps_cudagraph_none_unchanged(caplog):
    config = _config("DeepseekV4ForCausalLM", CUDAGraphMode.NONE)

    _enforce_deepseek_v4_constraints(config)

    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.NONE
    assert "graph replay corrupts V4 logits" not in caplog.text


def test_dsv4_experimental_override_keeps_requested_graph_mode(monkeypatch):
    monkeypatch.setenv("ATOM_V4_EXPERIMENTAL_CUDAGRAPH", "1")
    config = _config("DeepseekV4ForCausalLM", CUDAGraphMode.PIECEWISE)

    _enforce_deepseek_v4_constraints(config)

    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE


def test_non_dsv4_keeps_requested_cudagraph_mode():
    config = _config("DeepseekV3ForCausalLM", CUDAGraphMode.FULL_AND_PIECEWISE)

    _enforce_deepseek_v4_constraints(config)

    assert config.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_AND_PIECEWISE
