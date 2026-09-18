"""Bounded, expiring prefix hints for placing new sticky DP sessions.

These are routing estimates, not a directory of GPU cache residency. Engines
still perform their normal cache lookup and compute any missing prefix. The
CoreManager lock protects all mutable state in this helper.
"""

from __future__ import annotations

import array
import hashlib
from collections import OrderedDict

PrefixKey = tuple[int, bytes]


class PrefixRouteHints:
    def __init__(
        self,
        num_ranks: int,
        block_tokens: int = 8192,
        max_entries: int = 65536,
        ttl_seconds: float = 600.0,
        max_request_skew: int = 8,
    ):
        if num_ranks <= 0 or block_tokens <= 0 or max_entries <= 0:
            raise ValueError("rank count, block size and capacity must be positive")
        if not 0 < ttl_seconds < float("inf"):
            raise ValueError("prefix hint TTL must be finite and positive")
        if max_request_skew < 0:
            raise ValueError("prefix routing request skew must be non-negative")
        self.num_ranks = num_ranks
        self.block_tokens = block_tokens
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.max_request_skew = max_request_skew
        self._entries: OrderedDict[PrefixKey, dict[int, float]] = OrderedDict()

    def fingerprints(
        self, token_ids: array.array, prompt_tokens: int
    ) -> tuple[PrefixKey, ...]:
        """Hash cumulative prompt prefixes without boxing or retaining tokens.

        Sequence stores native int32 tokens in array('i'). All lookups happen
        in this process, so no cross-machine byte-order convention is needed.
        This read-only work can run outside the CoreManager routing lock.
        """
        if token_ids.typecode != "i" or token_ids.itemsize != 4:
            raise ValueError("prefix routing expects Sequence's int32 token buffer")
        if not 0 <= prompt_tokens <= len(token_ids):
            raise ValueError("prompt length exceeds the supplied token buffer")
        raw = memoryview(token_ids).cast("B")[: prompt_tokens * 4]
        block_bytes = self.block_tokens * 4
        digest = hashlib.blake2b(digest_size=16)
        keys = []
        for end in range(block_bytes, len(raw) + 1, block_bytes):
            digest.update(raw[end - block_bytes : end])
            keys.append((end // 4, digest.digest()))
        return tuple(keys)

    def match(self, keys: tuple[PrefixKey, ...], now: float) -> list[int]:
        """Return the longest recently observed prefix length on every rank."""
        matched = [0] * self.num_ranks
        for key in keys:
            owners = self._entries.get(key)
            if owners is None:
                continue
            stale = []
            for rank, observed in owners.items():
                if now - observed >= self.ttl_seconds:
                    stale.append(rank)
                else:
                    matched[rank] = max(matched[rank], key[0])
            for rank in stale:
                del owners[rank]
            if owners:
                self._entries.move_to_end(key)
            else:
                del self._entries[key]
        return matched

    def observe(self, keys: tuple[PrefixKey, ...], rank: int, now: float) -> None:
        """Publish a hint only after an engine output proves prefill finished."""
        if not 0 <= rank < self.num_ranks:
            raise ValueError("prefix hint rank is outside the DP group")
        for key in keys:
            owners = self._entries.setdefault(key, {})
            owners[rank] = now
            self._entries.move_to_end(key)
            if len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
