# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shared, CPU-only policy helpers for value-aware offload save admission."""

from __future__ import annotations

import array
import hashlib
import os
import time
from collections import OrderedDict
from dataclasses import dataclass

PrefixDemandKey = tuple[int, bytes]


def _nonnegative_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _nonnegative_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not 0 <= value < float("inf"):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _positive_float(name: str, default: float) -> float:
    value = _nonnegative_float(name, default)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class SaveAdmissionConfig:
    policy: str
    min_observed_count: int
    aging_weight: float
    release_weight: float
    demand_block_tokens: int
    demand_max_entries: int
    demand_ttl_seconds: float


def load_save_admission_config() -> SaveAdmissionConfig:
    """Read the shared save policy, preserving old defaults when disabled."""

    policy = os.environ.get("OFFLOAD_SAVE_POLICY", "round_robin").strip().lower()
    if policy not in {"round_robin", "priority"}:
        raise ValueError(
            f"OFFLOAD_SAVE_POLICY must be 'round_robin' or 'priority', got {policy!r}"
        )
    if policy == "round_robin":
        return SaveAdmissionConfig(policy, 2, 0.01, 1.0, 8192, 65536, 600.0)
    return SaveAdmissionConfig(
        policy=policy,
        min_observed_count=_nonnegative_int("OFFLOAD_SAVE_MIN_OBSERVED_COUNT", 2),
        aging_weight=_nonnegative_float("OFFLOAD_SAVE_AGING_WEIGHT", 0.01),
        release_weight=_nonnegative_float("OFFLOAD_SAVE_RELEASE_WEIGHT", 1.0),
        demand_block_tokens=_nonnegative_int("OFFLOAD_SAVE_DEMAND_BLOCK_TOKENS", 8192),
        demand_max_entries=_nonnegative_int("OFFLOAD_SAVE_DEMAND_MAX_ENTRIES", 65536),
        demand_ttl_seconds=_positive_float("OFFLOAD_SAVE_DEMAND_TTL_SECONDS", 600),
    )


@dataclass
class _PrefixDemandEntry:
    count: int
    last_seen: float


class PrefixDemandTracker:
    """Bounded rank-local demand counts for coarse cumulative prefixes.

    These counts deliberately do not share DP prefix-route hint state. Route
    hints are published only after prefill proves an HBM owner, whereas save
    admission needs request demand at entry, before a prefix is stored.
    """

    def __init__(
        self,
        *,
        block_tokens: int = 8192,
        max_entries: int = 65536,
        ttl_seconds: float = 600.0,
        clock=time.monotonic,
    ) -> None:
        if block_tokens <= 0 or max_entries <= 0:
            raise ValueError("prefix demand block size and capacity must be positive")
        if not 0 < ttl_seconds < float("inf"):
            raise ValueError("prefix demand TTL must be finite and positive")
        self.block_tokens = int(block_tokens)
        self.max_entries = int(max_entries)
        self.ttl_seconds = float(ttl_seconds)
        self._clock = clock
        self._entries: OrderedDict[PrefixDemandKey, _PrefixDemandEntry] = OrderedDict()

    def fingerprints(
        self, token_ids: array.array, prompt_tokens: int
    ) -> tuple[PrefixDemandKey, ...]:
        """Hash cumulative coarse prefixes without retaining token buffers."""

        prompt_tokens = int(prompt_tokens)
        if not 0 <= prompt_tokens <= len(token_ids):
            raise ValueError("prompt length exceeds the supplied token buffer")
        if not isinstance(token_ids, array.array):
            token_ids = array.array("i", token_ids)
        if token_ids.typecode != "i" or token_ids.itemsize != 4:
            raise ValueError("prefix demand expects an int32 token buffer")

        raw = memoryview(token_ids).cast("B")[: prompt_tokens * 4]
        block_bytes = self.block_tokens * 4
        digest = hashlib.blake2b(digest_size=16)
        keys = []
        for end in range(block_bytes, len(raw) + 1, block_bytes):
            digest.update(raw[end - block_bytes : end])
            keys.append((end // 4, digest.digest()))
        if not keys and raw:
            digest.update(raw)
            keys.append((prompt_tokens, digest.digest()))
        return tuple(keys)

    def observe(
        self, token_ids: array.array, prompt_tokens: int, now: float | None = None
    ) -> tuple[PrefixDemandKey, ...]:
        """Record one request and return reusable keys for later scoring."""

        observed_at = self._clock() if now is None else float(now)
        keys = self.fingerprints(token_ids, prompt_tokens)
        for key in keys:
            entry = self._entries.get(key)
            if entry is None or observed_at - entry.last_seen >= self.ttl_seconds:
                entry = _PrefixDemandEntry(count=1, last_seen=observed_at)
                self._entries[key] = entry
            else:
                entry.count += 1
                entry.last_seen = observed_at
            self._entries.move_to_end(key)
            if len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
        return keys

    def heat(
        self,
        keys: tuple[PrefixDemandKey, ...],
        *,
        max_tokens: int,
        now: float | None = None,
    ) -> tuple[int, int]:
        """Return ``(observed_count, reusable_tokens)`` for the hottest key."""

        checked_at = self._clock() if now is None else float(now)
        reusable_limit = max(0, int(max_tokens))
        if reusable_limit == 0:
            return 0, 0
        best = (0, 0)
        for key in keys:
            entry = self._entries.get(key)
            if entry is None:
                continue
            if checked_at - entry.last_seen >= self.ttl_seconds:
                del self._entries[key]
                continue
            self._entries.move_to_end(key)
            best = max(best, (entry.count, min(int(key[0]), reusable_limit)))
        return best

    def clear(self) -> None:
        self._entries.clear()


__all__ = [
    "PrefixDemandKey",
    "PrefixDemandTracker",
    "SaveAdmissionConfig",
    "load_save_admission_config",
]
