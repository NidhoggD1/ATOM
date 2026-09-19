# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""What `/metrics` may cost, which here is a correctness property.

Rendering runs inline on the loop that delivers every open SSE stream, so a
metric whose source walks the heap turns the scrape interval into a periodic
inter-token latency spike. These pin the bound, not any value: a slow source is
invisible until someone profiles a scrape.
"""

from __future__ import annotations

import gc

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.parser import text_string_to_metric_families

from atom.entrypoints.openai.metrics import AtomMetricsExporter, _gc_metrics


def _render() -> str:
    class _Collector:
        def collect(self):
            yield from _gc_metrics()

    registry = CollectorRegistry()
    registry.register(_Collector())
    return generate_latest(registry).decode()


def _series_names(exposition: str) -> set[str]:
    return {
        line.split("{")[0].split(" ")[0]
        for line in exposition.splitlines()
        if line and not line.startswith("#")
    }


def test_a_scrape_never_walks_the_heap(monkeypatch):
    """The two ways to get this wrong, named so that adding either fails here.

    `atom:gc_frozen_objects` was one of them and had to go. Caching the count
    in `gc_utils` is not the way back: `gc.collect()` moves it without going
    through that module, so any mirror drifts. See `_gc_metrics` for the cost.
    """
    walked: list[str] = []

    def watch(name, result):
        def stub(*_args, **_kwargs):
            walked.append(name)
            return result

        monkeypatch.setattr(gc, name, stub)

    watch("get_freeze_count", 0)
    watch("get_objects", [])

    _render()

    assert walked == [], f"a scrape walked the heap via {walked}"


def test_the_exported_names_are_what_the_docs_tell_operators_to_query():
    """`prometheus_client` appends `_total` to a counter and nothing to a
    gauge, so the name in the source is not the name in a PromQL rule. Every
    one of these is written out in `docs/environment_variables.md`; a rule
    copied from there returning no series is indistinguishable from a healthy
    process, which is the failure this pins.
    """
    assert _series_names(_render()) == {
        "atom:gc_collections_total",
        "atom:gc_collected_total",
        "atom:gc_uncollectable_total",
        "atom:gc_threshold",
    }


def test_every_generation_is_labelled_rather_than_summed():
    """Gen-2 is the stop-the-world one; a total that folded it in with gen-0
    would be dominated by the cheap generation and say nothing."""
    exposition = _render()

    for generation in ("0", "1", "2"):
        assert f'atom:gc_collections_total{{generation="{generation}"}}' in exposition


def test_lmcache_dp_route_probe_counters_are_exported():
    exporter = AtomMetricsExporter()
    exporter.update(
        {
            "enabled": True,
            "dp_router": {
                "lmcache_probe_hit_total": 3,
                "lmcache_probe_miss_total": 4,
                "lmcache_probe_failure_total": 5,
                "lmcache_probe_hit_tokens": 8192,
            },
        }
    )

    samples = {
        sample.name: sample.value
        for family in text_string_to_metric_families(exporter.render().decode())
        for sample in family.samples
        if sample.name.startswith("atom:dp_lmcache_probe_")
    }
    assert samples == {
        "atom:dp_lmcache_probe_hit_total": 3,
        "atom:dp_lmcache_probe_miss_total": 4,
        "atom:dp_lmcache_probe_failure_total": 5,
        "atom:dp_lmcache_probe_hit_tokens_total": 8192,
    }
