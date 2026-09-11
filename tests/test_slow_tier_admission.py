# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Contract for admitting saves to the slow tier.

The policy decides how much of the save tax to keep paying when the tier stops
returning anything. Getting it wrong is expensive in both directions: too eager
and a workload with real reuse loses its cache, too shy and the tax is paid for
nothing. The properties worth pinning are therefore the boundaries -- off, warm
-up, full payoff, no payoff -- and the floor that stops the feedback loop from
running away.
"""

from __future__ import annotations

import pytest

from atom.kv_transfer.offload._offload_common import SlowTierAdmission

PAID_OFF = "aligned_large_hit"
WASTED = "hbm_satisfies_after_alloc"


@pytest.fixture
def admission(monkeypatch):
    def build(**env):
        monkeypatch.setenv("OFFLOAD_SAVE_ADMISSION", env.pop("enabled", "1"))
        for key, value in env.items():
            monkeypatch.setenv(f"OFFLOAD_ADMISSION_{key.upper()}", str(value))
        return SlowTierAdmission()

    return build


def _feed(policy, reason, n):
    for _ in range(n):
        policy.record_load_decision(reason)


def _admitted(policy, n):
    return sum(1 for _ in range(n) if policy.admit_save())


def test_disabled_admits_everything(admission):
    policy = admission(enabled="0", warmup=0, floor=0.0)
    _feed(policy, WASTED, 500)
    assert _admitted(policy, 200) == 200


def test_warmup_admits_everything(admission):
    """A cold tier reports no payoff because nothing is in it yet.

    Throttling on that would be throttling on the absence of evidence, so the
    warm-up has to outlast it.
    """
    policy = admission(warmup=100, floor=0.0, alpha=1.0)
    _feed(policy, WASTED, 99)
    assert _admitted(policy, 50) == 50


def test_no_payoff_settles_on_the_floor(admission):
    policy = admission(warmup=10, floor=0.25, alpha=1.0)
    _feed(policy, WASTED, 200)
    assert _admitted(policy, 400) == pytest.approx(100, abs=1)


def test_full_payoff_admits_everything(admission):
    policy = admission(warmup=10, floor=0.25, target=0.10, alpha=1.0)
    _feed(policy, PAID_OFF, 200)
    assert _admitted(policy, 300) == 300


def test_payoff_at_target_admits_everything(admission):
    """The target is where saving everything is already justified.

    alpha has to be small enough to actually average: at 1.0 the estimate is
    just the last verdict, and a steady rate is never represented at all.
    """
    policy = admission(warmup=0, floor=0.1, target=0.25, alpha=0.02)
    for _ in range(400):
        policy.record_load_decision(PAID_OFF)
        _feed(policy, WASTED, 3)  # steady 25% payoff
    assert policy.stats()["payoff"] == pytest.approx(0.25, abs=0.02)
    # The estimate oscillates by about alpha around the true rate, and the
    # cycle above ends on a miss, so sitting exactly at the target admits
    # essentially everything rather than literally everything.
    assert _admitted(policy, 200) >= 190


def test_partial_payoff_admits_a_matching_share(admission):
    policy = admission(warmup=0, floor=0.0, target=0.40, alpha=0.02)
    for _ in range(400):
        policy.record_load_decision(PAID_OFF)
        _feed(policy, WASTED, 4)  # steady 20% payoff -> half of a 0.40 target
    assert policy.stats()["payoff"] == pytest.approx(0.20, abs=0.02)
    assert _admitted(policy, 400) == pytest.approx(200, rel=0.1)


def test_floor_survives_a_starved_tier(admission):
    """The loop is circular: saving less can only lower the payoff.

    Without a floor that is a ratchet to zero, and the tier can never recover
    because nothing is ever written to it again.
    """
    policy = admission(warmup=0, floor=0.2, target=1.0, alpha=1.0)
    for _ in range(20):
        _feed(policy, WASTED, 50)
        assert _admitted(policy, 100) >= 19


def test_recovers_when_the_tier_starts_paying(admission):
    policy = admission(warmup=0, floor=0.1, target=0.5, alpha=0.5)
    _feed(policy, WASTED, 200)
    assert _admitted(policy, 100) == pytest.approx(10, abs=1)
    _feed(policy, PAID_OFF, 200)
    assert _admitted(policy, 100) == 100


def test_admission_is_deterministic(admission):
    """Two runs of the same traffic admit the same requests.

    A random draw would make a short benchmark window sample a different rate
    than a long one, which is exactly the comparison this policy gets judged by.
    """

    def trace():
        policy = admission(warmup=5, floor=0.3, target=1.0, alpha=0.25)
        out = []
        for i in range(200):
            policy.record_load_decision(PAID_OFF if i % 7 == 0 else WASTED)
            out.append(policy.admit_save())
        return out

    assert trace() == trace()


def test_stats_report_the_reason_breakdown(admission):
    policy = admission(warmup=0, floor=0.5, alpha=1.0)
    _feed(policy, WASTED, 3)
    _feed(policy, "too_small", 2)
    _feed(policy, PAID_OFF, 1)
    _admitted(policy, 10)
    stats = policy.stats()
    assert stats["enabled"] is True
    assert stats["probes"] == 6
    assert stats["reasons"] == {WASTED: 3, "too_small": 2, PAID_OFF: 1}
    assert stats["saved"] + stats["skipped"] == 10


def test_a_bad_env_value_falls_back_instead_of_crashing(admission):
    policy = admission(warmup=0, floor="not-a-number", target=1.0, alpha=1.0)
    _feed(policy, WASTED, 100)
    admitted = _admitted(policy, 400)
    assert admitted == pytest.approx(100, abs=1)  # the 0.25 default floor


def test_batching_converges_to_the_reported_rate(admission):
    """A step's verdicts are folded as a uniform rate, not replayed in order.

    EWMA is order-sensitive, and the order of a handful of verdicts inside one
    engine step is arbitrary, so the closed form deliberately does not
    reproduce a particular interleaving. What has to hold is that the estimate
    still converges on the rate the scheduler is reporting, whatever the step
    boundaries happen to be.
    """
    steady = admission(warmup=0, alpha=0.05)
    for _ in range(200):
        steady.record_load_batch(3, 20)  # 15%, twenty verdicts per step
    assert steady.stats()["payoff"] == pytest.approx(0.15, abs=0.01)
    assert steady.stats()["probes"] == 4000

    # Same rate, different step sizes: the estimate must land in the same place.
    lumpy = admission(warmup=0, alpha=0.05)
    for _ in range(40):
        lumpy.record_load_batch(15, 100)
    assert lumpy.stats()["payoff"] == pytest.approx(0.15, abs=0.01)


def test_the_estimate_starts_optimistic(admission):
    """Nothing has been observed yet, so the tier gets the benefit of the doubt.

    Starting at zero would throttle a cold cache on its first step, before it
    has had any chance to be read.
    """
    assert admission(warmup=0).stats()["payoff"] == pytest.approx(1.0)


def test_batch_of_unused_verdicts_reaches_the_floor(admission):
    policy = admission(warmup=10, floor=0.25, alpha=0.5)
    policy.record_load_batch(0, 400)
    assert _admitted(policy, 400) == pytest.approx(100, abs=1)


def test_batch_counts_both_outcomes(admission):
    policy = admission(warmup=0)
    policy.record_load_batch(2, 10)
    reasons = policy.stats()["reasons"]
    assert reasons[PAID_OFF] == 2
    assert reasons["slow_tier_unused"] == 8


def test_an_empty_step_changes_nothing(admission):
    policy = admission(warmup=0, alpha=0.5)
    policy.record_load_batch(0, 400)
    before = policy.stats()["payoff"]
    policy.record_load_batch(0, 0)
    assert policy.stats()["payoff"] == before
    assert policy.stats()["probes"] == 400


def test_a_batch_cannot_claim_more_wins_than_verdicts(admission):
    policy = admission(warmup=0, alpha=1.0)
    policy.record_load_batch(99, 10)
    assert policy.stats()["reasons"][PAID_OFF] == 10
    assert policy.stats()["payoff"] == pytest.approx(1.0)
