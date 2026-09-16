# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest

from atom.kv_transfer.offload.hybrid.dsv4.policy import (
    _resolve_slot_staging_slots,
    build_dsv4_profile,
    select_pending_sidecar_boundary,
    sidecar_boundary_tokens,
)


def _config(**overrides):
    values = {
        "kv_cache_block_size": 256,
        "decode_context_parallel_size": 2,
        "state_checkpoint_interval_tokens": 9000,
        "hf_config": SimpleNamespace(kv_head_dim=576, index_head_dim=160),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_dsv4_profile_resolves_virtual_grid_and_cadence():
    profile = build_dsv4_profile(_config(), chunk_size=8192)

    assert profile.name == "deepseek-v4-page-slot"
    assert profile.block_size == 256
    assert profile.dcp_size == 2
    assert profile.hash_block_size == 512
    assert profile.resume_alignment == 8192
    assert profile.checkpoint_interval == 8704
    assert profile.sidecar_interval == 139264
    assert profile.kv_head_dim == 576
    assert profile.index_head_dim == 160


def test_dsv4_profile_rejects_chunk_that_splits_virtual_dcp_block():
    with pytest.raises(ValueError, match="virtual DCP block"):
        build_dsv4_profile(_config(), chunk_size=768)


@pytest.mark.parametrize("invalid", [True, 256.0, "256"])
@pytest.mark.parametrize(
    "field",
    ["kv_cache_block_size", "decode_context_parallel_size", "chunk_size"],
)
def test_dsv4_profile_rejects_coerced_integer_geometry(field, invalid):
    config = _config()
    chunk_size = 8192
    if field == "chunk_size":
        chunk_size = invalid
    else:
        setattr(config, field, invalid)

    with pytest.raises(ValueError, match="must be an integer"):
        build_dsv4_profile(config, chunk_size=chunk_size)


def test_state_checkpoint_can_follow_each_lmcache_chunk_with_dcp_one():
    profile = build_dsv4_profile(
        _config(
            kv_cache_block_size=4,
            decode_context_parallel_size=1,
            state_checkpoint_interval_tokens=4,
        ),
        chunk_size=4,
    )

    assert profile.hash_block_size == 4
    assert profile.resume_alignment == 4
    assert profile.checkpoint_interval == 4
    assert profile.sidecar_interval == 4
    assert sidecar_boundary_tokens(
        num_prompt_tokens=12,
        resume_alignment=profile.resume_alignment,
        sidecar_interval=profile.sidecar_interval,
    ) == (4, 8, 12)


def test_the_minus_one_interval_keeps_the_sidecar_on():
    """-1 means "no grid", not "no checkpointing" -- both halves must agree.

    `BlockManager` clamps this field to `max(-1, ...)` and reads -1 as the grid
    being off while the prompt-end anchor and the demand rung go on placing
    checkpoints. This consumer used to clamp to `max(0, ...)`, folding -1 into
    0, which here means no sidecar checkpoints at all: the engine kept
    checkpointing while offload resume silently degraded to zero reuse.

    With no grid to align to, the sidecar takes `resume_alignment` alone.
    """
    profile = build_dsv4_profile(
        _config(
            kv_cache_block_size=4,
            decode_context_parallel_size=1,
            state_checkpoint_interval_tokens=-1,
        ),
        chunk_size=4,
    )

    assert profile.checkpoint_interval == -1, "the sentinel survives, not 0"
    assert profile.sidecar_interval == profile.resume_alignment
    assert profile.sidecar_interval != 0, "0 would be checkpointing off"
    assert sidecar_boundary_tokens(
        num_prompt_tokens=12,
        resume_alignment=profile.resume_alignment,
        sidecar_interval=profile.sidecar_interval,
    ) == (4, 8, 12)


def test_interval_zero_really_does_turn_the_sidecar_off():
    """The other end of the same rule: 0 is the off switch and stays one."""
    profile = build_dsv4_profile(
        _config(
            kv_cache_block_size=4,
            decode_context_parallel_size=1,
            state_checkpoint_interval_tokens=0,
        ),
        chunk_size=4,
    )

    assert profile.checkpoint_interval == 0
    assert profile.sidecar_interval == 0


def test_sidecar_policy_skips_off_interval_terminal_boundary():
    assert sidecar_boundary_tokens(
        num_prompt_tokens=20,
        resume_alignment=4,
        sidecar_interval=8,
    ) == (8, 16)


def test_pending_policy_skips_inflight_identity_and_selects_later_boundary():
    assert (
        select_pending_sidecar_boundary(
            [(8, 101), (16, 202)],
            start=0,
            end=16,
            committed_hashes=set(),
            inflight=((8, 101),),
            failed=set(),
        )
        == (16, 202)
    )


def test_pending_policy_returns_none_when_all_identities_are_inflight():
    assert (
        select_pending_sidecar_boundary(
            [(8, 101), (16, 202)],
            start=0,
            end=16,
            committed_hashes=set(),
            inflight=((8, 101), (16, 202)),
            failed=set(),
        )
        is None
    )


def test_slot_staging_slots_default_allows_two_concurrent_sidecars(monkeypatch):
    monkeypatch.delenv("OFFLOAD_SLOT_STAGING_SLOTS", raising=False)
    assert _resolve_slot_staging_slots({}) == 2


def test_slot_staging_slots_honors_env_and_extra_config(monkeypatch):
    monkeypatch.setenv("OFFLOAD_SLOT_STAGING_SLOTS", "4")
    assert _resolve_slot_staging_slots({}) == 4
    assert (
        _resolve_slot_staging_slots(
            {"kv_connector_extra_config": {"slot_sidecar_staging_slots": 3}}
        )
        == 3
    )
