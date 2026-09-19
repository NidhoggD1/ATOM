# SPDX-License-Identifier: MIT
# Unit tests for CoreManager DP request load-balancing (engine_core_mgr).
#
# These exercise the pure routing/bookkeeping helpers in isolation via
# ``CoreManager.__new__`` — no engine cores, sockets, or GPU are created.
# CoreManager gets ``EngineCoreRequestType`` from the lightweight
# ``engine_core_protocol`` module and only lazy-imports the heavy ``EngineCore``
# (which pulls aiter) inside ``launch_engine_core``, so importing CoreManager
# here works on the CPU-only / mocked CI runner without any sys.modules stub;
# conftest.py supplies the atom.* / zmq stubs the import chain needs.

import pickle
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock, Thread
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from atom.model_engine.engine_core_mgr import (
    DP_LB_STRATEGIES,
    CoreManager,
)

# ── Helpers ────────────────────────────────────────────────────────────────


class _FakeSeq:
    """Minimal stand-in for Sequence: routing only reads id/num_prompt_tokens."""

    def __init__(
        self,
        seq_id,
        num_prompt_tokens=1,
        data_parallel_rank=None,
        dp_session_id=None,
        dp_parent_session_id=None,
    ):
        self.id = seq_id
        self.num_prompt_tokens = num_prompt_tokens
        self.dp_session_id = dp_session_id
        self.dp_parent_session_id = dp_parent_session_id
        if data_parallel_rank is not None:
            self.data_parallel_rank = data_parallel_rank


def _make_mgr(
    n_ranks,
    strategy="least_tokens",
    req_equiv=512,
    session_affinity=False,
):
    """Build a bare CoreManager with just the routing state initialized."""
    mgr = CoreManager.__new__(CoreManager)
    mgr.label = "Engine Core Mgr"
    mgr.local_engine_count = n_ranks
    mgr.max_pool_tokens = None
    mgr._dp_lb_strategy = strategy
    mgr._dp_lb_req_equiv = req_equiv
    mgr._dp_session_affinity_enabled = session_affinity
    mgr._dp_lmcache_route_enabled = False
    mgr._dp_lmcache_route_min_gain_tokens = 8192
    mgr._dp_lmcache_route_lookup_timeout = 0.25
    mgr._dp_lmcache_route_probe = None
    mgr._dp_lmcache_route_probe_lock = Lock()
    mgr._dp_lmcache_route_descriptors = []
    mgr._dp_prefix_hints = None
    mgr._pending_dp_prefix_hints = {}
    mgr._dp_prefix_routed_total = 0
    mgr._dp_prefix_estimated_tokens = 0
    mgr._dp_session_owners = {}
    mgr._dp_session_prompt_tokens = {}
    mgr._dp_route_counters = {
        "affinity_new_total": 0,
        "affinity_owner_hit_total": 0,
        "affinity_spill_total": 0,
        "lmcache_probe_hit_total": 0,
        "lmcache_probe_miss_total": 0,
        "lmcache_probe_failure_total": 0,
        "lmcache_probe_hit_tokens": 0,
        "affinity_parent_ignored_total": 0,
        "explicit_total": 0,
        "load_balanced_total": 0,
    }
    mgr._rank_routed_total = [0] * n_ranks
    mgr._rank_rotation_cursor = 0
    mgr._rank_reqs = [0] * n_ranks
    mgr._rank_tokens = [0] * n_ranks
    mgr._seq_load = {}
    mgr._lb_lock = Lock()
    return mgr


def _route(mgr, seqs):
    """Replicate add_request's per-seq selection loop (hint > strategy)."""
    assigned = []
    with mgr._lb_lock:
        for seq in seqs:
            hint = getattr(seq, "data_parallel_rank", None)
            hint = int(hint) if hint is not None else None
            rank = mgr._select_dp_rank_for_seq_locked(seq, hint)
            mgr._charge_seq_load_locked(seq, rank)
            assigned.append(rank)
    return assigned


# ── Tests ──────────────────────────────────────────────────────────────────


def test_least_requests_spreads_uniformly():
    mgr = _make_mgr(4, strategy="least_requests")
    seqs = [_FakeSeq(i, num_prompt_tokens=10) for i in range(8)]
    _route(mgr, seqs)
    # 8 uniform requests across 4 ranks -> 2 each.
    assert mgr._rank_reqs == [2, 2, 2, 2]


def test_least_requests_tiebreak_prefers_fewer_prompt_tokens():
    # Equal in-flight request count -> the prompt-token load breaks the tie.
    mgr = _make_mgr(2, strategy="least_requests")
    mgr._rank_reqs = [1, 1]
    mgr._rank_tokens = [500, 100]
    with mgr._lb_lock:
        rank = mgr._select_dp_rank_locked()
    assert rank == 1  # equal reqs -> pick the lighter-token rank


def test_least_requests_count_dominates_token_tiebreak():
    # Request count is the primary key: a rank with fewer requests wins even
    # when it carries far more prompt tokens.
    mgr = _make_mgr(2, strategy="least_requests")
    mgr._rank_reqs = [1, 3]
    mgr._rank_tokens = [10_000, 100]
    with mgr._lb_lock:
        rank = mgr._select_dp_rank_locked()
    assert rank == 0  # fewer requests wins despite the larger token load


def test_least_requests_tiebreak_then_count_primacy_over_route():
    # Seed equal counts but skewed tokens; routing should first even the token
    # load (tie-break), then request-count primacy takes over.
    mgr = _make_mgr(2, strategy="least_requests")
    mgr._rank_reqs = [1, 1]
    mgr._rank_tokens = [1000, 100]
    ranks = _route(mgr, [_FakeSeq("a", 100), _FakeSeq("b", 100)])
    # a: reqs tie (1,1) -> tokens 1000 vs 100 -> rank 1; now reqs=[1,2].
    # b: reqs 1 vs 2 -> rank 0 has fewer requests -> rank 0 (count primary).
    assert ranks == [1, 0]


def test_least_tokens_avoids_heavy_rank():
    # Pure token balance (req_equiv=0): a long prompt should keep that rank out
    # of rotation until the others accumulate comparable token load.
    mgr = _make_mgr(2, strategy="least_tokens", req_equiv=0)
    first = _route(mgr, [_FakeSeq("big", num_prompt_tokens=1000)])[0]
    other = 1 - first
    # Next several small requests must all go to the lighter rank.
    ranks = _route(mgr, [_FakeSeq(f"s{i}", num_prompt_tokens=100) for i in range(5)])
    assert all(r == other for r in ranks)
    assert mgr._rank_tokens[other] == 500
    assert mgr._rank_tokens[first] == 1000


def test_least_tokens_combined_signal_counts_requests():
    # With req_equiv>0 a rank holding many tiny requests still looks loaded, so
    # routing balances request count even when token counts are equal-ish.
    mgr = _make_mgr(2, strategy="least_tokens", req_equiv=512)
    ranks = _route(mgr, [_FakeSeq(i, num_prompt_tokens=1) for i in range(6)])
    # 6 near-zero-token requests -> alternate evenly by the req_equiv term.
    assert mgr._rank_reqs == [3, 3]
    assert sorted(ranks) == [0, 0, 0, 1, 1, 1]


def test_affinity_new_session_uses_token_load_before_stable_tiebreak():
    mgr = _make_mgr(3, session_affinity=True)
    expected = mgr._stable_session_rank("session-root")
    mgr._rank_tokens[expected] = 500_000
    rank = _route(
        mgr,
        [_FakeSeq("root", 1000, dp_session_id="session-root")],
    )[0]
    assert rank != expected
    assert mgr._rank_tokens[rank] == 1000
    assert mgr._dp_session_owners["session-root"] == rank


def test_affinity_hash_is_deterministic_across_managers():
    first = _make_mgr(8, session_affinity=True)
    second = _make_mgr(8, session_affinity=True)
    assert first._stable_session_rank("stable-correlation-id") == (
        second._stable_session_rank("stable-correlation-id")
    )
    assert first._select_new_session_rank_locked("stable-correlation-id") == (
        second._select_new_session_rank_locked("stable-correlation-id")
    )


def test_affinity_existing_session_always_uses_cache_owner():
    mgr = _make_mgr(2, session_affinity=True)
    mgr._dp_session_owners["s"] = 0
    # Even extreme transient backlog cannot move an existing session. Replaying
    # a long agent prefix is more expensive than waiting for its cache owner.
    mgr._rank_tokens = [10_000_000, 0]
    rank = _route(mgr, [_FakeSeq("turn", 1000, dp_session_id="s")])[0]
    assert rank == 0
    assert mgr._dp_route_counters["affinity_spill_total"] == 0


def test_affinity_later_turn_charges_only_prompt_growth():
    mgr = _make_mgr(2, session_affinity=True, req_equiv=512)
    first = _FakeSeq("first", 100_000, dp_session_id="s")
    second = _FakeSeq("second", 101_500, dp_session_id="s")

    owner = _route(mgr, [first])[0]
    mgr._mark_seq_prefill_complete(first.id)
    assert mgr._rank_tokens[owner] == 0

    assert _route(mgr, [second]) == [owner]
    assert mgr._rank_tokens[owner] == 1_500
    assert mgr._seq_load[second.id][2] == 1_500


def test_affinity_new_session_accounts_for_decode_pressure():
    mgr = _make_mgr(2, session_affinity=True, req_equiv=512)
    mgr._rank_reqs = [4, 0]
    mgr._rank_tokens = [0, 1_000]

    rank = _route(mgr, [_FakeSeq("new", 100, dp_session_id="new-session")])[0]
    # score(rank0)=2048, score(rank1)=1000 before the new request is charged.
    assert rank == 1


def test_affinity_routes_across_global_ranks_on_multinode_coordinator():
    mgr = _make_mgr(2, session_affinity=True)
    # A coordinator owns two local engines but routes across all four global
    # ranks. Routing state must use the global width so a remote rank can win.
    mgr.global_engine_count = 4
    mgr._rank_reqs = [0, 0, 0, 0]
    mgr._rank_tokens = [1000, 1000, 1000, 0]
    mgr._rank_routed_total = [0, 0, 0, 0]

    rank = _route(mgr, [_FakeSeq("new", 100, dp_session_id="global-session")])[0]

    assert rank == 3
    assert mgr._dp_session_owners["global-session"] == 3
    assert len(mgr.get_dp_router_statistics()["requests_per_rank"]) == 4


def test_affinity_child_is_independent_of_parent_cache_owner():
    mgr = _make_mgr(4, session_affinity=True)
    parent_owner = 0
    mgr._dp_session_owners["parent"] = parent_owner
    mgr._rank_tokens[parent_owner] = 100_000
    rank = _route(
        mgr,
        [
            _FakeSeq(
                "child-seq",
                100,
                dp_session_id="child",
                dp_parent_session_id="parent",
            )
        ],
    )[0]
    assert rank != parent_owner
    assert mgr._dp_session_owners["child"] == rank
    assert mgr._dp_session_owners["parent"] == parent_owner
    assert mgr._dp_route_counters["affinity_parent_ignored_total"] == 1


def test_affinity_child_does_not_reserve_parent_owner():
    mgr = _make_mgr(3, session_affinity=True)
    child_rank = _route(
        mgr,
        [
            _FakeSeq(
                "child-seq",
                100,
                dp_session_id="child",
                dp_parent_session_id="parent",
            )
        ],
    )[0]
    parent_rank = _route(
        mgr,
        [_FakeSeq("parent-seq", 100, dp_session_id="parent")],
    )[0]
    # The child's first-turn charge participates in the parent's independent
    # placement instead of reserving or forcing the parent owner.
    assert parent_rank != child_rank
    assert mgr._dp_session_owners["parent"] == parent_rank


def test_explicit_rank_is_authoritative_and_becomes_session_owner():
    mgr = _make_mgr(4, session_affinity=True)
    mgr._rank_tokens = [0, 0, 0, 100_000]
    rank = _route(
        mgr,
        [
            _FakeSeq(
                "explicit",
                100,
                data_parallel_rank=3,
                dp_session_id="s",
            )
        ],
    )[0]
    assert rank == 3
    assert mgr._dp_session_owners["s"] == 3


def test_dp_router_statistics_exposes_locality_and_rank_load():
    mgr = _make_mgr(4, session_affinity=True)
    session = "observed-session"
    ranks = _route(
        mgr,
        [
            _FakeSeq("first", 100, dp_session_id=session),
            _FakeSeq("second", 200, dp_session_id=session),
        ],
    )
    owner = ranks[0]

    stats = mgr.get_dp_router_statistics()
    assert stats["affinity_new_total"] == 1
    assert stats["affinity_owner_hit_total"] == 1
    assert stats["affinity_spill_total"] == 0
    assert stats["requests_per_rank"][owner] == 2
    assert stats["inflight_requests_per_rank"][owner] == 2
    # First turn charges 100; the sticky second turn charges only its 100-token
    # growth, not the full cached 200-token prompt.
    assert stats["queued_prefill_tokens_per_rank"][owner] == 200
    assert stats["session_count_per_rank"][owner] == 1


def test_release_restores_counts():
    mgr = _make_mgr(3, strategy="least_tokens")
    seqs = [_FakeSeq(i, num_prompt_tokens=50) for i in range(6)]
    _route(mgr, seqs)
    for seq in seqs:
        mgr._release_seq_load(seq.id)
    assert mgr._rank_reqs == [0, 0, 0]
    assert mgr._rank_tokens == [0, 0, 0]
    assert mgr._seq_load == {}


def test_release_is_idempotent_no_leak_no_negative():
    mgr = _make_mgr(2, strategy="least_tokens")
    _route(mgr, [_FakeSeq("a", num_prompt_tokens=20)])
    mgr._release_seq_load("a")
    # Second release (e.g. abort after finish) must be a no-op.
    mgr._release_seq_load("a")
    # Unknown id must also be a no-op.
    mgr._release_seq_load("never-seen")
    assert mgr._rank_reqs == [0, 0]
    assert mgr._rank_tokens == [0, 0]


def test_first_output_releases_prefill_tokens_but_not_request_count():
    mgr = _make_mgr(2, strategy="least_tokens")
    seq = _FakeSeq("a", num_prompt_tokens=20)
    _route(mgr, [seq])
    rank = mgr._seq_load[seq.id][0]

    mgr._mark_seq_prefill_complete(seq.id)
    mgr._mark_seq_prefill_complete(seq.id)  # idempotent across stream chunks
    assert mgr._rank_tokens[rank] == 0
    assert mgr._rank_reqs[rank] == 1

    mgr._release_seq_load(seq.id)
    assert mgr._rank_reqs == [0, 0]


def test_burst_does_not_dogpile_one_rank():
    mgr = _make_mgr(4, strategy="least_tokens")
    # A burst of identical requests dispatched back-to-back (optimistic +1 in
    # the same locked loop) must spread, not all land on rank 0.
    _route(mgr, [_FakeSeq(i, num_prompt_tokens=128) for i in range(4)])
    assert mgr._rank_reqs == [1, 1, 1, 1]


def test_round_robin_ignores_load():
    mgr = _make_mgr(3, strategy="round_robin")
    # Pre-skew load; round_robin must still rotate purely by cursor.
    mgr._rank_tokens = [10_000, 0, 0]
    mgr._rank_reqs = [50, 0, 0]
    ranks = _route(mgr, [_FakeSeq(i) for i in range(6)])
    assert ranks == [0, 1, 2, 0, 1, 2]


def test_explicit_hint_takes_priority_and_is_charged():
    mgr = _make_mgr(4, strategy="least_tokens")
    ranks = _route(
        mgr,
        [
            _FakeSeq("h", num_prompt_tokens=30, data_parallel_rank=2),
            _FakeSeq("auto", num_prompt_tokens=30),
        ],
    )
    assert ranks[0] == 2
    # Hinted rank's load is counted so it participates in future balancing.
    assert mgr._rank_reqs[2] >= 1
    assert mgr._rank_tokens[2] >= 30


def test_dispatch_sends_explicit_hint_to_exact_engine_rank():
    class _RecordingSocket:
        def __init__(self):
            self.messages = []

        def send_multipart(self, parts, copy=False):
            self.messages.append(parts)

    mgr = _make_mgr(4, strategy="least_requests")
    mgr.engine_core_identities = [b"e0", b"e1", b"e2", b"e3"]
    mgr.input_sockets = [_RecordingSocket() for _ in range(4)]
    # Make rank 3 the least attractive load-balanced destination. An explicit
    # mesh hint must still route there exactly.
    mgr._rank_reqs = [0, 0, 0, 10]
    mgr._rank_tokens = [0, 0, 0, 10_000]

    seq = _FakeSeq("sticky", num_prompt_tokens=100, data_parallel_rank=3)
    mgr._dispatch_to_dp_ranks([seq])

    assert [len(sock.messages) for sock in mgr.input_sockets] == [0, 0, 0, 1]
    request_type, dispatched = pickle.loads(mgr.input_sockets[3].messages[0][1])
    assert request_type.name == "ADD"
    assert [item.id for item in dispatched] == [seq.id]


def test_invalid_hint_still_supported_via_add_request_validation():
    # _route mimics add_request; out-of-range hints are validated in
    # add_request itself, so here we only assert a valid hint routes exactly.
    mgr = _make_mgr(2, strategy="least_tokens")
    ranks = _route(mgr, [_FakeSeq("x", data_parallel_rank=1)])
    assert ranks == [1]


def test_reset_dp_router_clears_all_state():
    mgr = _make_mgr(3, strategy="least_tokens")
    mgr._dp_session_owners["session"] = 1
    mgr._dp_session_prompt_tokens["session"] = 1234
    _route(mgr, [_FakeSeq(i, num_prompt_tokens=40) for i in range(5)])
    mgr.reset_dp_router()
    assert mgr._rank_rotation_cursor == 0
    assert mgr._rank_reqs == [0, 0, 0]
    assert mgr._rank_tokens == [0, 0, 0]
    assert mgr._seq_load == {}
    assert mgr._dp_session_owners == {}
    assert mgr._dp_session_prompt_tokens == {}


def test_tie_break_rotates_starting_rank():
    # All ranks equal load -> successive picks rotate rather than always rank 0.
    mgr = _make_mgr(3, strategy="least_tokens")
    ranks = [mgr._select_dp_rank_locked() for _ in range(6)]
    assert ranks == [0, 1, 2, 0, 1, 2]


@pytest.mark.parametrize("strategy", ["round_robin", "least_requests", "least_tokens"])
def test_all_strategies_route_within_range(strategy):
    mgr = _make_mgr(4, strategy=strategy)
    ranks = _route(mgr, [_FakeSeq(i, num_prompt_tokens=i + 1) for i in range(20)])
    assert all(0 <= r < 4 for r in ranks)
    # Every dispatched request is accounted for.
    assert sum(mgr._rank_reqs) == 20


def test_resolve_and_validate_hints_rejects_out_of_range_without_side_effects():
    # An invalid hint anywhere in the batch must raise BEFORE any load is
    # charged, so a rejected batch cannot leak partial in-flight load.
    mgr = _make_mgr(4, strategy="least_requests")
    seqs = [_FakeSeq("ok", 100), _FakeSeq("bad", 100, data_parallel_rank=9)]
    with pytest.raises(ValueError):
        mgr._resolve_and_validate_hints(seqs)
    assert mgr._rank_reqs == [0, 0, 0, 0]
    assert mgr._rank_tokens == [0, 0, 0, 0]
    assert mgr._seq_load == {}


def test_resolve_and_validate_hints_accepts_and_returns_resolved():
    mgr = _make_mgr(4, strategy="least_requests")
    # In-range hint and no-hint (None) are both fine, and the resolved hints are
    # returned in order for the dispatch loop to reuse.
    hints = mgr._resolve_and_validate_hints(
        [_FakeSeq("a", 10, data_parallel_rank=3), _FakeSeq("b", 10)]
    )
    assert hints == [3, None]


def test_send_failure_rolls_back_undispatched_charge():
    # If a rank's send_multipart raises mid-batch, the seqs on ranks that were
    # NOT successfully handed off were charged but will never emit a finished
    # output to release them. Dispatch must roll those back so routing does not
    # skew permanently; already-sent ranks keep their (legitimate) charge.
    # dispatch pickles (EngineCoreRequestType.ADD, seqs) with the real enum.
    class _FakeSocket:
        def __init__(self, fail=False):
            self.fail = fail

        def send_multipart(self, parts, copy=False):
            if self.fail:
                raise RuntimeError("send failed")

    mgr = _make_mgr(4, strategy="least_requests")
    mgr.engine_core_identities = [b"e0", b"e1", b"e2", b"e3"]
    # Rank 2 fails; ranks 0/1 send first, rank 3 is never attempted.
    mgr.input_sockets = [
        _FakeSocket(),
        _FakeSocket(),
        _FakeSocket(fail=True),
        _FakeSocket(),
    ]
    seqs = [_FakeSeq(i, num_prompt_tokens=10) for i in range(4)]
    with pytest.raises(RuntimeError):
        mgr._dispatch_to_dp_ranks(seqs)
    # Ranks 0,1 dispatched -> stay charged; ranks 2 (failed) and 3 (never
    # attempted) -> rolled back to zero.
    assert mgr._rank_reqs == [1, 1, 0, 0]
    assert mgr._rank_tokens == [10, 10, 0, 0]
    assert set(mgr._seq_load.keys()) == {0, 1}


def test_reset_clears_even_with_charged_in_flight():
    # reset must fully clear counters even if requests are still charged (it
    # warns, but state must not be left dirty or go negative).
    mgr = _make_mgr(2, strategy="least_requests")
    _route(mgr, [_FakeSeq(i, num_prompt_tokens=50) for i in range(3)])
    assert mgr._seq_load  # charged
    mgr.reset_dp_router()
    assert mgr._rank_reqs == [0, 0]
    assert mgr._rank_tokens == [0, 0]
    assert mgr._seq_load == {}


def test_dp_lb_strategies_constant():
    assert DP_LB_STRATEGIES == ("round_robin", "least_requests", "least_tokens")
    assert "least_request" not in DP_LB_STRATEGIES  # guards against typos


def test_decode_bookkeeping_does_not_wait_for_an_unrelated_router_lock():
    mgr = _make_mgr(2, strategy="least_tokens")
    seq = _FakeSeq("a", num_prompt_tokens=20)
    _route(mgr, [seq])
    assert sum(mgr._rank_reqs) == 1
    assert sum(mgr._rank_tokens) == 20
    mgr._mark_seq_prefill_complete(seq.id)
    assert sum(mgr._rank_tokens) == 0

    completed = Event()

    def receive_decode_token():
        mgr._mark_seq_prefill_complete(seq.id)
        completed.set()

    with mgr._lb_lock:
        worker = Thread(target=receive_decode_token)
        worker.start()
        progressed_while_router_locked = completed.wait(timeout=2)
    worker.join(timeout=2)
    assert progressed_while_router_locked
    assert sum(mgr._rank_reqs) == 1
    assert sum(mgr._rank_tokens) == 0


@pytest.fixture
def prefix_router(monkeypatch):
    from atom.model_engine import engine_core_mgr
    from atom.model_engine.prefix_route_hints import PrefixRouteHints

    mgr = _make_mgr(2, req_equiv=0, session_affinity=True)
    mgr._dp_prefix_hints = PrefixRouteHints(2, block_tokens=4, ttl_seconds=10)
    sent = []
    now = [0.0]
    monkeypatch.setattr(engine_core_mgr.time, "monotonic", lambda: now[0])

    def send(rank, payload):
        _, seqs = pickle.loads(payload)
        sent.extend((seq.id, rank) for seq in seqs)

    mgr._send_request = send
    return mgr, sent, now


def _prefix_seq(seq_id, tokens, session=None, rank=None):
    import array

    seq = _FakeSeq(seq_id, len(tokens), rank, session)
    seq.token_ids = array.array("i", tokens)
    return seq


class _RouteProbe:
    """Deterministic route-probe double that verifies lock ordering."""

    def __init__(self, mgr, result=0, error=None):
        self.mgr = mgr
        self.result = result
        self.error = error
        self.calls = []

    def probe_l1_hit_tokens(self, token_ids, *, timeout):
        acquired = self.mgr._lb_lock.acquire(blocking=False)
        assert acquired, "network route probe ran while _lb_lock was held"
        self.mgr._lb_lock.release()
        self.calls.append((list(token_ids), timeout))
        if self.error is not None:
            raise self.error
        return self.result


def _enable_lmcache_route(mgr, *, result=0, error=None, min_gain=8192):
    probe = _RouteProbe(mgr, result=result, error=error)
    mgr._dp_lmcache_route_enabled = True
    mgr._dp_lmcache_route_min_gain_tokens = min_gain
    mgr._dp_lmcache_route_probe = probe
    sent = []

    def send(rank, payload):
        _, seqs = pickle.loads(payload)
        sent.extend((seq.id, rank) for seq in seqs)

    mgr._send_request = send
    return probe, sent


def _route_descriptor(**overrides):
    descriptor = {
        "version": 1,
        "server_url": "tcp://127.0.0.1:5555",
        "model_name": "native-test",
        "world_size": 2,
        "tp_size": 2,
        "num_kv_readers": 1,
        "block_size": 64,
        "lmcache_tokens_per_chunk": 256,
        "mq_timeout": 30.0,
    }
    descriptor.update(overrides)
    return descriptor


def test_lmcache_cpu_miss_keeps_existing_owner():
    mgr = _make_mgr(2, req_equiv=0, session_affinity=True)
    mgr._dp_session_owners["s"] = 0
    mgr._dp_session_prompt_tokens["s"] = 1000
    mgr._rank_tokens = [100_000, 0]
    probe, sent = _enable_lmcache_route(mgr, result=0)
    seq = _prefix_seq("turn", list(range(1200)), "s")

    mgr._dispatch_to_dp_ranks([seq])

    assert sent == [("turn", 0)]
    assert mgr._dp_session_owners["s"] == 0
    assert mgr._dp_route_counters["lmcache_probe_miss_total"] == 1
    assert mgr._dp_route_counters["affinity_spill_total"] == 0
    assert len(probe.calls) == 1


def test_lmcache_cpu_hit_below_minimum_gain_keeps_owner():
    mgr = _make_mgr(2, req_equiv=0, session_affinity=True)
    mgr._dp_session_owners["s"] = 0
    mgr._dp_session_prompt_tokens["s"] = 1000
    mgr._rank_tokens = [8000, 0]
    _probe, sent = _enable_lmcache_route(mgr, result=1024, min_gain=8192)
    seq = _prefix_seq("turn", list(range(1200)), "s")

    mgr._dispatch_to_dp_ranks([seq])

    assert sent == [("turn", 0)]
    assert mgr._dp_session_owners["s"] == 0
    assert mgr._dp_route_counters["lmcache_probe_hit_total"] == 1
    assert mgr._dp_route_counters["affinity_spill_total"] == 0


def test_lmcache_cpu_hit_spills_from_backlogged_owner_and_updates_cost():
    mgr = _make_mgr(2, req_equiv=0, session_affinity=True)
    mgr._dp_session_owners["s"] = 0
    mgr._dp_session_prompt_tokens["s"] = 1000
    mgr._rank_tokens = [20_000, 0]
    _probe, sent = _enable_lmcache_route(mgr, result=1024, min_gain=8192)
    seq = _prefix_seq("turn", list(range(1200)), "s")

    mgr._dispatch_to_dp_ranks([seq])

    assert sent == [("turn", 1)]
    assert mgr._dp_session_owners["s"] == 1
    assert mgr._seq_load[seq.id] == (1, 1, 176)
    assert mgr._dp_route_counters["affinity_spill_total"] == 1
    assert mgr._dp_route_counters["lmcache_probe_hit_total"] == 1
    assert mgr._dp_route_counters["lmcache_probe_hit_tokens"] == 1024


def test_lmcache_spill_send_failure_restores_owner_prompt_and_route_counters():
    mgr = _make_mgr(2, req_equiv=0, session_affinity=True)
    mgr._dp_session_owners["s"] = 0
    mgr._dp_session_prompt_tokens["s"] = 1000
    mgr._rank_tokens = [20_000, 0]
    _probe, _sent = _enable_lmcache_route(mgr, result=1024, min_gain=8192)

    def fail_send(*_args):
        raise RuntimeError("send failed")

    mgr._send_request = fail_send
    seq = _prefix_seq("turn", list(range(1200)), "s")

    with pytest.raises(RuntimeError, match="send failed"):
        mgr._dispatch_to_dp_ranks([seq])

    assert mgr._dp_session_owners["s"] == 0
    assert mgr._dp_session_prompt_tokens["s"] == 1000
    assert mgr._seq_load == {}
    assert mgr._dp_route_counters["affinity_spill_total"] == 0
    assert mgr._rank_routed_total == [0, 0]
    # The probe did happen, even though the subsequent engine handoff failed.
    assert mgr._dp_route_counters["lmcache_probe_hit_total"] == 1
    assert mgr._dp_route_counters["lmcache_probe_hit_tokens"] == 1024


def test_lmcache_probe_failure_fails_closed_to_owner():
    mgr = _make_mgr(2, req_equiv=0, session_affinity=True)
    mgr._dp_session_owners["s"] = 0
    mgr._dp_session_prompt_tokens["s"] = 1000
    mgr._rank_tokens = [100_000, 0]
    _probe, sent = _enable_lmcache_route(mgr, error=TimeoutError("probe timeout"))
    seq = _prefix_seq("turn", list(range(1200)), "s")

    mgr._dispatch_to_dp_ranks([seq])

    assert sent == [("turn", 0)]
    assert mgr._dp_session_owners["s"] == 0
    assert mgr._dp_route_counters["lmcache_probe_failure_total"] == 1
    assert mgr._dp_route_counters["affinity_spill_total"] == 0


def test_explicit_rank_skips_lmcache_probe_and_remains_authoritative():
    mgr = _make_mgr(2, req_equiv=0, session_affinity=True)
    mgr._dp_session_owners["s"] = 0
    probe, sent = _enable_lmcache_route(
        mgr,
        error=AssertionError("explicit routes must not probe"),
    )
    seq = _prefix_seq("turn", list(range(1200)), "s", 1)

    mgr._dispatch_to_dp_ranks([seq])

    assert sent == [("turn", 1)]
    assert probe.calls == []
    assert mgr._dp_session_owners["s"] == 1
    assert mgr._dp_route_counters["explicit_total"] == 1


def test_lmcache_route_config_requires_affinity(monkeypatch):
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_DP_PREFIX_ROUTING", False)
    monkeypatch.setattr(envs, "ATOM_DP_SESSION_AFFINITY", False)
    mgr = CoreManager.__new__(CoreManager)
    config = SimpleNamespace(
        dp_load_balance="least_tokens",
        kv_transfer_config={
            "kv_connector_extra_config": {"lmcache.mp.dp_route_enabled": True}
        },
    )

    with pytest.raises(ValueError, match="requires ATOM DP session affinity"):
        mgr._init_shared_state(
            config,
            label="lmcache-route-test",
            local_engine_count=2,
        )


def test_lmcache_route_config_initializes_conservative_defaults(monkeypatch):
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_DP_PREFIX_ROUTING", False)
    monkeypatch.setattr(envs, "ATOM_DP_SESSION_AFFINITY", True)
    mgr = CoreManager.__new__(CoreManager)
    config = SimpleNamespace(
        dp_load_balance="least_tokens",
        kv_transfer_config={
            "kv_connector_extra_config": {"lmcache.mp.dp_route_enabled": True}
        },
    )
    try:
        mgr._init_shared_state(
            config,
            label="lmcache-route-test",
            local_engine_count=2,
        )
        assert mgr._dp_lmcache_route_enabled is True
        assert mgr._dp_lmcache_route_min_gain_tokens == 8192
        assert mgr._dp_lmcache_route_lookup_timeout == 0.25
        assert mgr._dp_lmcache_route_probe is None
    finally:
        mgr.ctx.term()


def test_lmcache_route_ready_requires_descriptor_from_every_rank():
    mgr = _make_mgr(2, session_affinity=True)
    mgr._dp_lmcache_route_enabled = True
    mgr.output_sockets = [object(), object()]
    mgr._record_ready_payload({"lmcache_mp_route_lookup": _route_descriptor()})
    mgr._record_ready_payload({"max_pool_tokens": 1000})

    with pytest.raises(RuntimeError, match=r"native PAGE\+STATE"):
        mgr._initialize_dp_lmcache_route_probe()


def test_lmcache_route_ready_rejects_mismatched_descriptors():
    mgr = _make_mgr(2, session_affinity=True)
    mgr._dp_lmcache_route_enabled = True
    mgr.output_sockets = [object(), object()]
    mgr._record_ready_payload({"lmcache_mp_route_lookup": _route_descriptor()})
    mgr._record_ready_payload(
        {"lmcache_mp_route_lookup": _route_descriptor(model_name="other")}
    )

    with pytest.raises(RuntimeError, match="different route lookup descriptors"):
        mgr._initialize_dp_lmcache_route_probe()


def test_lmcache_route_ready_rejects_non_dictionary_descriptor():
    mgr = _make_mgr(2, session_affinity=True)
    mgr._dp_lmcache_route_enabled = True

    with pytest.raises(TypeError, match="descriptor in READY must be a dictionary"):
        mgr._record_ready_payload({"lmcache_mp_route_lookup": []})


def test_lmcache_route_probe_initializes_once_and_closes(monkeypatch):
    mgr = _make_mgr(2, session_affinity=True)
    mgr._dp_lmcache_route_enabled = True
    mgr.output_sockets = [object(), object()]
    mgr.ctx = object()
    descriptor = _route_descriptor()
    mgr._dp_lmcache_route_descriptors = [descriptor, dict(descriptor)]

    probe = MagicMock(name="route_probe")
    adapter = MagicMock(name="AtomMPSchedulerAdapter")
    adapter.from_route_lookup_descriptor.return_value = probe
    lmcache_module = types.ModuleType("lmcache")
    integration_module = types.ModuleType("lmcache.integration")
    atom_module = types.ModuleType("lmcache.integration.atom")
    atom_module.AtomMPSchedulerAdapter = adapter
    lmcache_module.integration = integration_module
    integration_module.atom = atom_module
    monkeypatch.setitem(sys.modules, "lmcache", lmcache_module)
    monkeypatch.setitem(sys.modules, "lmcache.integration", integration_module)
    monkeypatch.setitem(sys.modules, "lmcache.integration.atom", atom_module)

    mgr._initialize_dp_lmcache_route_probe()

    adapter.from_route_lookup_descriptor.assert_called_once_with(descriptor, mgr.ctx)
    assert mgr._dp_lmcache_route_probe is probe

    mgr._closed = False
    mgr.input_sockets = []
    mgr.control_sockets = []
    mgr.shutdown_paths = []
    mgr.output_threads = []
    mgr.engine_core_processes = []
    mgr.close()
    mgr.close()

    probe.shutdown.assert_called_once_with()
    assert mgr._dp_lmcache_route_probe is None


def _seed_prefix(mgr, tokens=None):
    tokens = list(range(8)) if tokens is None else tokens
    seq = _prefix_seq("seed", tokens, "root", 0)
    mgr._dispatch_to_dp_ranks([seq])
    mgr._mark_seq_prefill_complete(seq.id)
    mgr._release_seq_load(seq.id)
    return seq


def test_prefix_learns_on_output_then_places_new_session_and_discounts_debt(
    prefix_router,
):
    mgr, sent, now = prefix_router
    root = _prefix_seq("root", list(range(8)), "root", 0)
    mgr._dispatch_to_dp_ranks([root])
    keys = mgr._dp_prefix_hints.fingerprints(root.token_ids, 8)
    assert mgr._dp_prefix_hints.match(keys, now[0]) == [0, 0]
    mgr._mark_seq_prefill_complete(root.id)
    assert mgr._dp_prefix_hints.match(keys, now[0]) == [8, 0]

    mgr._dispatch_to_dp_ranks([_prefix_seq("busy", [90, 91, 92], "busy", 0)])
    mgr._dispatch_to_dp_ranks([_prefix_seq("child", list(range(8)) + [99], "child")])
    # rank 0 has debt 3 but saves 8 incoming tokens: 3 - 8 < rank 1's 0.
    assert sent[-1] == ("child", 0)
    assert mgr._rank_tokens == [4, 0]  # debt 3 + one uncached child token
    assert mgr._dp_prefix_routed_total == 1
    assert mgr._dp_prefix_estimated_tokens == 8


def test_prefix_credit_does_not_override_larger_pending_work_or_existing_owner(
    prefix_router,
):
    mgr, sent, _ = prefix_router
    _seed_prefix(mgr)
    mgr._dispatch_to_dp_ranks([_prefix_seq("busy0", [90] * 20, "busy0", 0)])
    child = _prefix_seq("child", list(range(8)) + [99], "child")
    mgr._dispatch_to_dp_ranks([child])
    assert sent[-1] == ("child", 1)  # 20 - 8 > 0
    assert mgr._rank_tokens == [20, 9]
    mgr._mark_seq_prefill_complete(child.id)
    mgr._dispatch_to_dp_ranks([_prefix_seq("busy1", [91] * 40, "busy1", 1)])
    mgr._dispatch_to_dp_ranks(
        [_prefix_seq("next", list(range(8)) + [99, 100], "child")]
    )
    assert sent[-1] == ("next", 1)  # existing owner remains immutable


def test_prefix_request_skew_gate_prevents_large_prefix_from_attracting_burst(
    prefix_router,
):
    mgr, sent, _ = prefix_router
    mgr._dp_prefix_hints.max_request_skew = 1
    _seed_prefix(mgr, list(range(64)))
    mgr._dispatch_to_dp_ranks(
        [_prefix_seq("busy-a", [90], "a", 0), _prefix_seq("busy-b", [91], "b", 0)]
    )
    mgr._dispatch_to_dp_ranks([_prefix_seq("child", list(range(64)) + [99], "child")])
    assert sent[-1] == ("child", 1)
    assert mgr._rank_tokens == [2, 65]


def test_prefix_explicit_hint_wins_and_is_new_owner(prefix_router):
    mgr, sent, _ = prefix_router
    _seed_prefix(mgr)
    mgr._dispatch_to_dp_ranks([_prefix_seq("forced", list(range(8)), "child", 1)])
    assert sent[-1] == ("forced", 1)
    assert mgr._dp_session_owners["child"] == 1
    assert mgr._dp_prefix_routed_total == 0


def test_prefix_send_failure_discards_unpublished_hashes_and_debt(prefix_router):
    mgr, _, now = prefix_router
    seq = _prefix_seq("failed", list(range(8)), "s", 0)

    def fail_send(rank, payload):
        raise RuntimeError("transport failed")

    mgr._send_request = fail_send
    with pytest.raises(RuntimeError, match="transport failed"):
        mgr._dispatch_to_dp_ranks([seq])
    assert mgr._pending_dp_prefix_hints == {}
    assert mgr._rank_tokens == [0, 0]
    assert mgr._rank_reqs == [0, 0]
    keys = mgr._dp_prefix_hints.fingerprints(seq.token_ids, 8)
    assert mgr._dp_prefix_hints.match(keys, now[0]) == [0, 0]


def test_prefix_cancel_before_output_does_not_publish_and_reset_clears_hints(
    prefix_router,
):
    mgr, _, now = prefix_router
    seq = _prefix_seq("cancelled", list(range(8)), "s", 0)
    mgr._dispatch_to_dp_ranks([seq])
    mgr._release_seq_load(seq.id)
    mgr._mark_seq_prefill_complete(seq.id)  # late output is ignored
    keys = mgr._dp_prefix_hints.fingerprints(seq.token_ids, 8)
    assert mgr._pending_dp_prefix_hints == {}
    assert mgr._dp_prefix_hints.match(keys, now[0]) == [0, 0]
    _seed_prefix(mgr)
    mgr.reset_dp_router()
    assert mgr._dp_prefix_hints.match(keys, now[0]) == [0, 0]


def test_prefix_zero_debt_first_output_refreshes_hint_without_decode_lock_wait(
    prefix_router,
):
    mgr, _, now = prefix_router
    root = _seed_prefix(mgr)
    now[0] = 9
    seq = _prefix_seq("cached", list(range(8)), "child")
    mgr._dispatch_to_dp_ranks([seq])
    assert mgr._seq_load[seq.id] == (0, 1, 0)
    mgr._mark_seq_prefill_complete(seq.id)
    keys = mgr._dp_prefix_hints.fingerprints(root.token_ids, 8)
    assert mgr._dp_prefix_hints.match(keys, 10) == [8, 0]

    completed = Event()

    def decode_output():
        mgr._mark_seq_prefill_complete(seq.id)
        completed.set()

    with mgr._lb_lock:
        worker = Thread(target=decode_output)
        worker.start()
        progressed = completed.wait(timeout=2)
    worker.join(timeout=2)
    assert progressed


def test_prefix_concurrent_dispatch_charges_before_next_placement(prefix_router):
    mgr, sent, _ = prefix_router
    mgr._dp_lb_req_equiv = 16
    _seed_prefix(mgr)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                mgr._dispatch_to_dp_ranks,
                [_prefix_seq(name, list(range(8)) + [99], name)],
            )
            for name in ("a", "b")
        ]
        for future in futures:
            future.result(timeout=2)
    assert sorted(rank for name, rank in sent if name in ("a", "b")) == [0, 1]
    assert mgr._rank_reqs == [1, 1]


@pytest.mark.parametrize("session_affinity", [False, True])
def test_prefix_option_requires_affinity_and_initializes_global_rank_count(
    monkeypatch, session_affinity
):
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_DP_PREFIX_ROUTING", True)
    monkeypatch.setattr(envs, "ATOM_DP_SESSION_AFFINITY", session_affinity)
    mgr = CoreManager.__new__(CoreManager)
    try:
        if not session_affinity:
            with pytest.raises(ValueError, match="requires session affinity"):
                mgr._init_shared_state(
                    SimpleNamespace(dp_load_balance="least_tokens"),
                    label="prefix-test",
                    local_engine_count=2,
                    global_engine_count=8,
                )
        else:
            mgr._init_shared_state(
                SimpleNamespace(dp_load_balance="least_tokens"),
                label="prefix-test",
                local_engine_count=2,
                global_engine_count=8,
            )
            assert mgr._dp_prefix_hints.num_ranks == 8
            assert len(mgr._rank_reqs) == 8
            assert mgr._pending_dp_prefix_hints == {}
    finally:
        mgr.ctx.term()
