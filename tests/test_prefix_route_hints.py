import array

import pytest

from atom.model_engine.prefix_route_hints import PrefixRouteHints


def keys(cache, tokens, prompt_tokens=None):
    return cache.fingerprints(
        array.array("i", tokens),
        len(tokens) if prompt_tokens is None else prompt_tokens,
    )


def test_longest_cumulative_prefix_per_rank_excludes_generation_and_partial_block():
    cache = PrefixRouteHints(2, block_tokens=4)
    cache.observe(keys(cache, [1, 2, 3, 4, 5, 6, 7, 8, 99], 8), 0, 0)
    cache.observe(keys(cache, [1, 2, 3, 4, 10, 11, 12, 13]), 1, 0)
    assert cache.match(keys(cache, [1, 2, 3, 4, 5, 6, 7, 8, 100]), 1) == [8, 4]
    # Equal second blocks do not imply an equal cumulative prefix.
    assert cache.match(keys(cache, [9, 2, 3, 4, 5, 6, 7, 8]), 1) == [0, 0]
    assert cache.match(keys(cache, [1, 2, 3]), 1) == [0, 0]


def test_ttl_expires_individual_rank_at_exact_boundary():
    cache = PrefixRouteHints(2, block_tokens=4, ttl_seconds=10)
    prefix = keys(cache, [1, 2, 3, 4])
    cache.observe(prefix, 0, 0)
    cache.observe(prefix, 1, 5)
    assert cache.match(prefix, 10) == [0, 4]
    assert cache.match(prefix, 15) == [0, 0]


def test_capacity_evicts_least_recently_used_prefix():
    cache = PrefixRouteHints(2, block_tokens=4, max_entries=2)
    first = keys(cache, list(range(8)))
    cache.observe(first, 0, 0)
    assert cache.match(first[:1], 1) == [4, 0]
    cache.observe(keys(cache, [10, 11, 12, 13]), 1, 2)
    assert cache.match(first, 3) == [4, 0]
    cache.clear()
    assert cache.match(first, 3) == [0, 0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_tokens": 0},
        {"ttl_seconds": float("nan")},
        {"max_request_skew": -1},
    ],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        PrefixRouteHints(2, **kwargs)


def test_invalid_prompt_buffer_and_rank_rejected():
    cache = PrefixRouteHints(2, block_tokens=4)
    with pytest.raises(ValueError, match="int32"):
        cache.fingerprints(array.array("q", [1, 2, 3, 4]), 4)
    with pytest.raises(ValueError, match="prompt length"):
        cache.fingerprints(array.array("i", [1, 2]), 3)
    with pytest.raises(ValueError, match="outside"):
        cache.observe(keys(cache, [1, 2, 3, 4]), 2, 0)
