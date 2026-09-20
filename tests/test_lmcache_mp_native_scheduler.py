# SPDX-License-Identifier: MIT

"""Native-state MP admission and PAGE-image lifetime contracts."""

from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.kv_transfer.disaggregation.types import (
    ConnectorCompletion,
    KVConnectorOutput,
    LoadOperationId,
    SaveOperationId,
)
from atom.kv_transfer.offload.mp import backend
from atom.kv_transfer.offload.mp.connector import LMCacheMPConnectorScheduler
from atom.kv_transfer.offload.mp.native_state_scheduler import (
    NativeStateLMCacheMPConnectorScheduler,
)
from atom.kv_transfer.offload.mp.native_state_worker import (
    NATIVE_STATE_MP_STORE_CHANNEL,
)
from atom.model_engine.block_manager import BlockManager
from atom.model_engine.block_pool import BlockPool
from atom.model_engine.page_unit_checkpoint import (
    PagedStateCheckpointCoordinator,
    PagedStateCheckpointSpec,
)
from atom.model_engine.scheduler import ScheduledBatch, ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceStatus, SequenceType
from atom.model_engine.state_runtime import StateRuntime, StateTransfer


@pytest.fixture(autouse=True)
def storage_config(monkeypatch):
    # The transport is a double; retain config/chunk validation without needing
    # the optional LMCache package or a standalone server on a CPU test runner.
    monkeypatch.setattr(
        backend.offcfg,
        "build_lmcache_config",
        lambda kvc: SimpleNamespace(
            chunk_size=kvc["kv_connector_extra_config"]["lmcache.chunk_size"]
        ),
    )


class Adapter:
    lmcache_tokens_per_chunk = 8

    def __init__(self):
        self.queries = []
        self.freed = []
        self.ended = []
        self.closed = False

    def maybe_submit_lookup_request(self, request_id, token_ids):
        self.queries.append((request_id, list(token_ids)))

    def check_lookup_result(self, request_id):
        return len(self.queries[-1][1])

    def free_lookup_locks(self, **kwargs):
        self.freed.append(kwargs)

    def cleanup_lookup_result(self, request_id):
        pass

    def end_session(self, request_id):
        self.ended.append(request_id)

    def get_route_lookup_descriptor(self):
        return {"version": 1, "model_name": "native-test"}

    def shutdown(self):
        self.closed = True


def make_scheduler(
    monkeypatch,
    *,
    capacity=2,
    budget=60,
    units=30,
    role="offload",
    policy="round_robin",
    min_observed=2,
    total_blocks=100,
    pin_ratio="0.20",
):
    monkeypatch.setenv("OFFLOAD_MAX_PENDING_SAVES", str(capacity))
    monkeypatch.setenv("OFFLOAD_SAVE_POLICY", policy)
    monkeypatch.setenv("OFFLOAD_SAVE_MIN_OBSERVED_COUNT", str(min_observed))
    monkeypatch.setenv("OFFLOAD_SAVE_MAX_PINNED_RATIO", pin_ratio)
    adapter = Adapter()
    connections = []

    def connect(config, *, checkpoint_spec=None):
        connections.append(checkpoint_spec)
        return adapter

    monkeypatch.setattr(backend, "_make_scheduler_adapter", connect)
    config = SimpleNamespace(
        kv_cache_block_size=4,
        kv_transfer_config={
            "kv_role": role,
            "kv_connector_extra_config": {
                "lmcache.mp.max_pinned_state_bytes": budget,
                "lmcache.chunk_size": 8,
            },
        },
    )
    scheduler = NativeStateLMCacheMPConnectorScheduler(config)
    assert connections == []
    checkpoints = PagedStateCheckpointCoordinator(
        BlockPool(units),
        PagedStateCheckpointSpec(10, 50, "native-test", image_bytes=25),
        enabled=True,
    )
    manager = SimpleNamespace(
        paged_state_checkpoints=checkpoints,
        hash_block_size=4,
        compute_hash=BlockManager.compute_hash,
        total_allocatable_kv_blocks=total_blocks,
    )
    scheduler.bind_block_manager(manager)
    scheduler.bind_block_manager(manager)
    assert connections == [checkpoints.store.spec]
    scheduler._min_load_tokens = 0
    return scheduler, checkpoints, adapter


def test_native_scheduler_exports_route_lookup_descriptor(monkeypatch):
    scheduler, _checkpoints, adapter = make_scheduler(monkeypatch)

    assert scheduler.get_route_lookup_descriptor() == (
        adapter.get_route_lookup_descriptor()
    )


def sequence(request_id=1, *, count=24, computed=16, token_offset=0, block_offset=0):
    seq = Sequence(
        list(range(token_offset, token_offset + count)),
        4,
        id=request_id,
        has_per_req_cache=True,
    )
    seq.num_cached_tokens = computed
    seq.state_slots = [5]
    seq.block_table = list(range(block_offset, block_offset + (count + 3) // 4))
    return seq


def checkpoint(scheduler, checkpoints, seq, boundary, *, ready=True):
    prefix_hash = scheduler._boundary_hash(seq, boundary)
    op = checkpoints.store.begin_store(prefix_hash, src_slot=seq.state_slot)
    assert op is not None
    if ready:
        checkpoints.complete_previous_batch()
    return prefix_hash, op


def terminal(scheduler, operation, *, succeeded=True):
    output = KVConnectorOutput(
        finished_saving={operation},
        connector_completions={
            ConnectorCompletion(NATIVE_STATE_MP_STORE_CHANNEL, operation, succeeded)
        },
    )
    return scheduler.process_completions(output)


def test_save_uses_ready_native_hash_and_never_copies_active_slot(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(monkeypatch)
    seq = sequence()
    assert seq.block_hashes == []
    prefix_hash, image = checkpoint(scheduler, checkpoints, seq, 16, ready=False)
    scheduler.update_state_after_alloc(seq)
    assert scheduler.build_connector_meta().requests == []
    assert scheduler.should_defer_free(seq) is False

    checkpoints.complete_previous_batch()
    seq.state_slots = [99]  # The native image is independent of this live slot.
    [request] = scheduler.build_connector_meta().requests
    assert isinstance(request.save_operation, SaveOperationId)
    assert request.native_state.prefix_hash == prefix_hash
    assert request.native_state.boundary_tokens == 16
    assert request.native_state.unit_ids == image.unit_ids
    assert request.native_state.destination_slot is None
    assert request.token_ids == list(seq.token_ids[:16])
    assert request.save_spec.skip_leading_tokens == 0
    assert checkpoints.take_checkpoint_ops() == ((), ())
    assert scheduler._pinned_state_bytes == 30
    checkpoints.clear_index()
    assert checkpoints.store.pool.num_free == 27

    terminal(scheduler, request.save_operation)
    assert checkpoints.store.pool.num_free == 30
    assert scheduler._pinned_state_bytes == 0


def test_budget_smaller_than_one_native_image_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="max_pinned_state_bytes"):
        make_scheduler(monkeypatch, budget=29)


@pytest.mark.parametrize(("capacity", "budget"), [(1, 60), (2, 30)])
def test_save_credit_and_bytes_are_reserved_before_native_pin(
    monkeypatch, capacity, budget
):
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch, capacity=capacity, budget=budget
    )
    seqs = [sequence(1), sequence(2, token_offset=100)]
    ids = []
    for seq in seqs:
        prefix_hash, _ = checkpoint(scheduler, checkpoints, seq, 16)
        ids.append(checkpoints.store.lookup(prefix_hash))
        scheduler.update_state_after_alloc(seq)
    [first] = scheduler.build_connector_meta().requests
    assert first.req_id == 1
    assert checkpoints.store.records[ids[0]].pin_count == 1
    assert checkpoints.store.records[ids[1]].pin_count == 0
    assert scheduler._save_tracker["2"][1] == 0
    assert scheduler.build_connector_meta().requests == []
    terminal(scheduler, first.save_operation)
    [second] = scheduler.build_connector_meta().requests
    assert second.req_id == 2
    assert second.native_state.unit_ids != first.native_state.unit_ids


def test_priority_save_prefers_hotter_prefix(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch, capacity=1, policy="priority", min_observed=1
    )
    cold = sequence("cold", token_offset=100)
    hot = sequence("hot")
    scheduler._prefix_demand.observe(hot.token_ids, hot.num_prompt_tokens)
    scheduler._prefix_demand.observe(hot.token_ids, hot.num_prompt_tokens)
    for seq in (cold, hot):
        checkpoint(scheduler, checkpoints, seq, 16)
        scheduler.update_state_after_alloc(seq)

    [request] = scheduler.build_connector_meta().requests
    assert request.req_id == "hot"


def test_priority_save_prefers_lower_dirty_cost(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch, capacity=1, policy="priority", min_observed=1
    )
    expensive = sequence("expensive", token_offset=100)
    cheap = sequence("cheap")
    for seq in (expensive, cheap):
        checkpoint(scheduler, checkpoints, seq, 16)
        scheduler.update_state_after_alloc(seq)
    scheduler._save_tracker["cheap"][1] = 8

    [request] = scheduler.build_connector_meta().requests
    assert request.req_id == "cheap"
    assert request.save_spec.skip_leading_tokens == 8


def test_priority_save_has_stable_request_id_tie_break(monkeypatch):
    monkeypatch.setenv("OFFLOAD_SAVE_AGING_WEIGHT", "0")
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch, capacity=1, policy="priority", min_observed=1
    )
    later_id = sequence("b", token_offset=100)
    earlier_id = sequence("a")
    for seq in (later_id, earlier_id):
        checkpoint(scheduler, checkpoints, seq, 16)
        scheduler.update_state_after_alloc(seq)

    [request] = scheduler.build_connector_meta().requests
    assert request.req_id == "a"


def test_glm_page_and_state_budgets_replace_atomically(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch,
        capacity=3,
        budget=60,
        policy="priority",
        min_observed=1,
        total_blocks=20,
    )
    lows = [
        sequence("low-a", count=8, computed=8, block_offset=0),
        sequence("low-b", count=8, computed=8, token_offset=50, block_offset=0),
    ]
    for seq in lows:
        checkpoint(scheduler, checkpoints, seq, 8)
        scheduler.update_state_after_alloc(seq)
        scheduler.request_finished(seq)
    # PAGE IDs are shared, so the new disjoint two-block candidate still fits
    # the four-block PAGE budget. The native byte budget nevertheless requires
    # one lower-score committed PAGE+STATE pair to be removed as one unit.
    high = sequence("high", count=8, computed=8, token_offset=100, block_offset=10)
    checkpoint(scheduler, checkpoints, high, 8)
    scheduler.update_state_after_alloc(high)
    scheduler._prefix_demand.observe(high.token_ids, high.num_prompt_tokens)
    scheduler.request_finished(high)

    assert len(scheduler._save_committed) == 2
    assert scheduler._save_committed["high"] is high
    assert sum(str(seq.id) in scheduler._save_committed for seq in lows) == 1
    assert scheduler.get_statistics()["save_budget_evicted"] == 1


def test_glm_failed_state_budget_simulation_keeps_all_committed_victims(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch,
        capacity=2,
        budget=60,
        policy="priority",
        min_observed=1,
        total_blocks=20,
    )
    low = sequence("low", count=8, computed=8)
    checkpoint(scheduler, checkpoints, low, 8)
    scheduler.update_state_after_alloc(low)
    scheduler.request_finished(low)
    # Model an unrelated inflight native transfer/load consuming the remaining
    # state-image bytes. Evicting the PAGE reservation still cannot admit a new
    # PAGE+STATE pair, so the simulation must make no mutation.
    scheduler._pinned_state_bytes = 60
    high = sequence("high", count=8, computed=8, token_offset=100, block_offset=10)
    checkpoint(scheduler, checkpoints, high, 8)
    scheduler.update_state_after_alloc(high)
    scheduler._prefix_demand.observe(high.token_ids, high.num_prompt_tokens)
    scheduler.request_finished(high)

    assert scheduler._save_committed == {"low": low}
    assert "low" in scheduler._save_tracker
    assert "high" not in scheduler._save_tracker
    assert scheduler.get_statistics()["save_budget_evicted"] == 0


def test_priority_aging_can_reorder_eligible_candidates_but_not_bypass_threshold(
    monkeypatch,
):
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch, capacity=1, policy="priority", min_observed=1
    )
    old = sequence("old", token_offset=100)
    hot = sequence("hot")
    scheduler._prefix_demand.observe(hot.token_ids, hot.num_prompt_tokens)
    for seq in (old, hot):
        checkpoint(scheduler, checkpoints, seq, 16)
        scheduler.update_state_after_alloc(seq)
        scheduler._build_save_candidate(str(seq.id))
    owner, since = scheduler._save_candidate_since["old"]
    scheduler._save_candidate_since["old"] = (owner, since - 10_000)

    [request] = scheduler.build_connector_meta().requests
    assert request.req_id == "old"

    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch, capacity=1, policy="priority", min_observed=2
    )
    cold = sequence("cold")
    checkpoint(scheduler, checkpoints, cold, 16)
    scheduler.update_state_after_alloc(cold)
    scheduler._save_candidate_since["cold"] = (cold, 0.0)
    assert scheduler.build_connector_meta().requests == []


def test_finished_priority_save_commits_then_dispatches(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch, capacity=1, policy="priority", min_observed=1
    )
    seq = sequence()
    checkpoint(scheduler, checkpoints, seq, 16)
    scheduler.update_state_after_alloc(seq)

    scheduler.request_finished(seq)
    assert scheduler._save_committed == {"1": seq}
    assert scheduler.should_defer_free(seq)
    [request] = scheduler.build_connector_meta().requests
    assert request.req_id == seq.id
    assert scheduler._save_committed == {}
    assert scheduler._save_inflight == {"1": request.save_operation}


def test_finished_priority_save_drops_immediately_when_capacity_is_full(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch, capacity=1, policy="priority", min_observed=1
    )
    running = sequence("running")
    finished = sequence("finished", token_offset=100)
    for seq in (running, finished):
        checkpoint(scheduler, checkpoints, seq, 16)
        scheduler.update_state_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    assert request.req_id == "finished" or request.req_id == "running"
    inflight_seq = running if request.req_id == "running" else finished
    dropped_seq = finished if inflight_seq is running else running

    scheduler.request_finished(dropped_seq)
    assert str(dropped_seq.id) not in scheduler._save_tracker
    assert not scheduler.should_defer_free(dropped_seq)
    assert scheduler.get_statistics()["save_dropped_capacity"] == 1


def test_priority_capacity_rejection_cancels_save_instead_of_retrying(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch, capacity=2, policy="priority", min_observed=1
    )
    first = sequence("first")
    second = sequence("second", token_offset=100)
    third = sequence("third", token_offset=200)
    checkpoint(scheduler, checkpoints, first, 16)
    scheduler.update_state_after_alloc(first)
    [first_request] = scheduler.build_connector_meta().requests
    for seq in (second, third):
        checkpoint(scheduler, checkpoints, seq, 16)
        scheduler.update_state_after_alloc(seq)
    scheduler.request_finished(second)
    assert scheduler._save_committed == {"second": second}

    [second_request] = scheduler.build_connector_meta().requests
    assert second_request.req_id == "second"
    assert set(scheduler._save_inflight) == {"first", "second"}
    assert scheduler.build_connector_meta().requests == []
    terminal(scheduler, first_request.save_operation)
    assert scheduler.build_connector_meta().requests == []
    assert "third" not in scheduler._save_tracker
    assert not scheduler.should_defer_free(third)


def test_finished_inflight_save_commits_one_residual_tail(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch, capacity=1, policy="priority", min_observed=1
    )
    seq = sequence(computed=8)
    checkpoint(scheduler, checkpoints, seq, 8)
    checkpoint(scheduler, checkpoints, seq, 16)
    scheduler.update_state_after_alloc(seq)
    [first] = scheduler.build_connector_meta().requests
    assert first.native_state.boundary_tokens == 8

    seq.num_cached_tokens = 16
    scheduler.request_finished(seq)
    terminal(scheduler, first.save_operation)
    assert scheduler._save_committed == {"1": seq}
    [residual] = scheduler.build_connector_meta().requests
    assert residual.native_state.boundary_tokens == 16
    assert residual.save_spec.skip_leading_tokens == 8
    terminal(scheduler, residual.save_operation)
    assert scheduler.build_connector_meta().requests == []
    assert scheduler._save_tracker == {}


def test_finished_priority_failure_drops_tail_without_retry_or_leaks(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch, capacity=1, policy="priority", min_observed=1
    )
    seq = sequence()
    checkpoint(scheduler, checkpoints, seq, 16)
    scheduler.update_state_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    scheduler.request_finished(seq)

    terminal(scheduler, request.save_operation, succeeded=False)
    assert scheduler.build_connector_meta().requests == []
    assert scheduler._save_tracker == {}
    assert scheduler._save_failures == {}
    assert scheduler._save_operation_blocks == {}
    assert scheduler._save_operation_owner == {}
    stats = scheduler.get_statistics()
    assert stats["save_dropped_terminal_failure"] == 1
    assert stats["save_dropped_tokens_terminal_failure"] == 16


def test_active_priority_failure_retains_bounded_retry(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch, capacity=1, policy="priority", min_observed=1
    )
    seq = sequence()
    checkpoint(scheduler, checkpoints, seq, 16)
    scheduler.update_state_after_alloc(seq)
    [first] = scheduler.build_connector_meta().requests
    terminal(scheduler, first.save_operation, succeeded=False)
    [second] = scheduler.build_connector_meta().requests
    assert second.save_operation != first.save_operation


def test_late_failure_from_reused_request_id_does_not_drop_new_lifecycle(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch, capacity=1, policy="priority", min_observed=1
    )
    old = sequence("same")
    checkpoint(scheduler, checkpoints, old, 16)
    scheduler.update_state_after_alloc(old)
    [old_request] = scheduler.build_connector_meta().requests
    scheduler.request_finished(old)

    new = sequence("same", token_offset=100)
    checkpoint(scheduler, checkpoints, new, 16)
    scheduler.update_state_after_alloc(new)
    terminal(scheduler, old_request.save_operation, succeeded=False)
    assert scheduler._save_tracker["same"][0] is new
    assert scheduler._retired_requests == {}

    [new_request] = scheduler.build_connector_meta().requests
    assert new_request.req_id == new.id
    assert new_request.save_operation != old_request.save_operation


def test_priority_statistics_report_candidates_reservations_drops_and_real_leases(
    monkeypatch,
):
    scheduler, checkpoints, _ = make_scheduler(
        monkeypatch, capacity=1, policy="priority", min_observed=2
    )
    cold = sequence("cold")
    checkpoint(scheduler, checkpoints, cold, 16)
    scheduler.update_state_after_alloc(cold)
    stats = scheduler.get_statistics()
    assert stats["save_candidates"] == 1
    assert stats["save_candidates_finished"] == 0
    assert stats["save_admitted"] == 0
    assert stats["save_priority_score"] > 0

    scheduler.request_finished(cold)
    stats = scheduler.get_statistics()
    assert stats["save_dropped_low_value"] == 1
    assert stats["save_dropped_tokens_low_value"] == 16
    assert stats["save_candidates"] == 0

    leased = sequence("leased", token_offset=100)
    scheduler.activate_block_leases(leased, frozenset({7, 8}))
    stats = scheduler.get_statistics()
    assert stats["save_pinned_blocks"] == 2
    assert stats["save_pinned_tokens"] == 8
    assert stats["deferred_free_requests"] == 1


def test_save_frontier_selects_existing_checkpoint_below_computed_tokens(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(monkeypatch)
    seq = sequence(computed=24)
    checkpoint(scheduler, checkpoints, seq, 8)
    scheduler.update_state_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    assert len(request.token_ids) == 8
    assert request.native_state.boundary_tokens == 8
    assert scheduler._save_tracker["1"][1] == 8


def test_store_failure_rolls_back_for_bounded_retry_and_ignores_stale_reports(
    monkeypatch,
):
    scheduler, checkpoints, _ = make_scheduler(monkeypatch)
    seq = sequence()
    checkpoint(scheduler, checkpoints, seq, 16)
    scheduler.update_state_after_alloc(seq)
    [first] = scheduler.build_connector_meta().requests
    terminal(scheduler, first.save_operation, succeeded=False)
    assert scheduler._save_tracker["1"][1] == 0
    [second] = scheduler.build_connector_meta().requests
    assert second.save_operation != first.save_operation
    terminal(scheduler, first.save_operation)
    assert scheduler._pinned_state_bytes == 30
    assert scheduler._save_inflight["1"] == second.save_operation
    terminal(scheduler, second.save_operation, succeeded=False)
    [third] = scheduler.build_connector_meta().requests
    terminal(scheduler, third.save_operation, succeeded=False)
    assert scheduler.build_connector_meta().requests == []
    assert scheduler.should_defer_free(seq) is False
    assert scheduler.get_statistics()["saved_tokens"] == 0
    scheduler.request_finished(seq)
    assert scheduler._save_tracker == {}
    assert scheduler._save_failures == {}


def test_no_timeout_or_abandon_recycles_dispatched_source(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(monkeypatch)
    seq = sequence()
    checkpoint(scheduler, checkpoints, seq, 16)
    scheduler.update_state_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    scheduler.abandon_save(request.save_operation)
    assert scheduler.save_abandon_timeout_s() == 0
    assert scheduler.reclaim_stale_leases(1e-9) == []
    assert checkpoints.reclaim_stale_offload_pins(1e-9) == 0
    assert scheduler._pinned_state_bytes == 30
    assert scheduler.should_defer_free(seq)
    terminal(scheduler, request.save_operation)
    assert scheduler._pinned_state_bytes == 0


@pytest.mark.parametrize("prompt", [8, 16, 19, 24])
def test_load_queries_safe_chunk_boundary_before_reserving_state(monkeypatch, prompt):
    scheduler, checkpoints, adapter = make_scheduler(monkeypatch)
    seq = sequence(count=prompt, computed=0)
    expected = ((prompt - 1) // 8) * 8
    assert scheduler.get_num_new_matched_tokens(seq) == (expected, expected > 0)
    assert scheduler._pinned_state_bytes == 0
    if expected == 0:
        assert adapter.queries == []
        return
    assert adapter.queries == [
        (f"atom-offload-dp0:{seq.id}", list(seq.token_ids[:expected]))
    ]
    scheduler.update_state_after_alloc(seq)
    assert scheduler.should_park_for_load_after_alloc(seq)
    operation = seq._load_operation
    assert isinstance(operation, LoadOperationId)
    assert scheduler._pinned_state_bytes == 30
    assert checkpoints.store.pool.num_free == 27
    [request] = scheduler.build_connector_meta().requests
    assert request.load_operation == operation
    assert request.native_state.boundary_tokens == expected
    assert request.native_state.destination_slot == seq.state_slot
    assert request.load_spec.lmcache_cached_tokens == expected
    assert request.load_spec.transfer_end_tokens is None
    assert request.token_ids == list(seq.token_ids[:expected])
    assert (
        scheduler.load_finished(LoadOperationId(seq.id, operation.generation + 1))
        is False
    )
    assert scheduler._pinned_state_bytes == 30
    assert scheduler.load_finished(operation) is True
    assert scheduler._pinned_state_bytes == 0
    # The completed transfer is now a reusable, unpinned READY checkpoint.
    assert checkpoints.store.pool.num_free == 27
    assert checkpoints.contains(request.native_state.prefix_hash)
    checkpoint_id = checkpoints.store.lookup(request.native_state.prefix_hash)
    assert checkpoints.store.records[checkpoint_id].pin_count == 0
    assert scheduler.load_finished(operation) is False


def test_aligned_hbm_hit_uses_incremental_native_restore(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(monkeypatch)
    seq = sequence(computed=0)
    assert scheduler.get_num_new_matched_tokens(seq) == (16, True)
    seq.num_cached_tokens = 8
    scheduler.update_state_after_alloc(seq)
    assert scheduler.should_park_for_load_after_alloc(seq) is True
    [request] = scheduler.build_connector_meta().requests
    assert request.load_spec.hbm_cached_tokens == 8
    assert request.load_spec.lmcache_cached_tokens == 16
    assert request.token_ids == list(seq.token_ids[:16])
    assert seq.offload_load_start_tokens == 8
    assert scheduler._pinned_state_bytes == 30
    assert checkpoints.store.pool.num_free == 27


def test_unaligned_hbm_hit_prefills_to_chunk_then_uses_native_restore(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(monkeypatch)
    seq = sequence(computed=0)
    assert scheduler.get_num_new_matched_tokens(seq) == (16, True)
    seq.num_cached_tokens = 4
    scheduler.update_state_after_alloc(seq)
    assert scheduler.should_park_for_load_after_alloc(seq) is False
    assert scheduler.build_connector_meta().requests == []
    assert scheduler._handoff_loads == {str(seq.id)}
    assert scheduler.adjust_prefill_chunk_after_alloc(seq, 16) == 4

    # The local prefill reaches the next LMCache chunk boundary. The same
    # lookup lease is then handed off to an incremental [8, 16) retrieve.
    seq.num_cached_tokens = 8
    assert scheduler.should_park_partial_prefill_for_load(seq) is True
    [request] = scheduler.build_connector_meta().requests
    assert request.load_spec.hbm_cached_tokens == 8
    assert request.load_spec.lmcache_cached_tokens == 16
    assert seq.offload_load_start_tokens == 8
    assert scheduler._handoff_loads == set()
    assert checkpoints.store.pool.num_free == 27


def test_incremental_restore_supersedes_queued_local_state_restore(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(monkeypatch)
    seq = sequence(computed=0)
    local_hash, _ = checkpoint(scheduler, checkpoints, seq, 8)
    assert checkpoints.begin_restore(local_hash, seq.state_slot)
    local_id = checkpoints.store.lookup(local_hash)
    assert checkpoints.store.records[local_id].pin_count == 1

    assert scheduler.get_num_new_matched_tokens(seq) == (16, True)
    seq.num_cached_tokens = 8
    scheduler.update_state_after_alloc(seq)
    assert scheduler.should_park_for_load_after_alloc(seq)
    assert not checkpoints.restore_queued_for(seq.state_slot)
    [request] = scheduler.build_connector_meta().requests

    assert scheduler.load_finished(request.load_operation)
    assert checkpoints.store.records[local_id].pin_count == 0
    assert checkpoints.contains(request.native_state.prefix_hash)


def test_incremental_restore_failure_resumes_queued_local_state_restore(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(monkeypatch)
    seq = sequence(computed=0)
    local_hash, _ = checkpoint(scheduler, checkpoints, seq, 8)
    assert checkpoints.begin_restore(local_hash, seq.state_slot)

    scheduler.get_num_new_matched_tokens(seq)
    seq.num_cached_tokens = 8
    scheduler.update_state_after_alloc(seq)
    assert scheduler.should_park_for_load_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    assert not checkpoints.restore_queued_for(seq.state_slot)

    assert scheduler.load_failed(request.load_operation)
    assert checkpoints.restore_queued_for(seq.state_slot)
    [restore] = checkpoints.take_checkpoint_ops()[1]
    assert restore.dst_slot == seq.state_slot


def test_load_capacity_failure_never_parks_or_claims_missing_state(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(monkeypatch, units=2)
    seq = sequence(computed=0)
    assert scheduler.get_num_new_matched_tokens(seq) == (16, True)
    scheduler.update_state_after_alloc(seq)
    assert scheduler.should_park_for_load_after_alloc(seq) is False
    assert seq.offload_loaded_tokens == 0
    assert scheduler._native_loads == {}
    assert scheduler._pinned_state_bytes == 0
    assert checkpoints.store.pool.num_free == 2


def test_load_failure_and_cancellation_wait_for_exact_terminal_report(monkeypatch):
    scheduler, checkpoints, adapter = make_scheduler(monkeypatch)
    seq = sequence(computed=0)
    scheduler.get_num_new_matched_tokens(seq)
    scheduler.update_state_after_alloc(seq)
    assert scheduler.should_park_for_load_after_alloc(seq)
    [request] = scheduler.build_connector_meta().requests
    scheduler.cancel_pending_load(seq)
    scheduler.request_finished(seq)
    assert scheduler.should_defer_free(seq)
    assert scheduler.has_pending_work()
    assert scheduler._pinned_state_bytes == 30
    assert adapter.ended == []
    assert scheduler.load_failed(request.load_operation)
    assert checkpoints.store.pool.num_free == 30
    assert scheduler._pinned_state_bytes == 0
    assert adapter.ended == [f"atom-offload-dp0:{seq.id}"]
    assert not scheduler.should_defer_free(seq)


def test_cancel_before_dispatch_releases_reserved_load_units(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(monkeypatch)
    seq = sequence(computed=0)
    scheduler.get_num_new_matched_tokens(seq)
    scheduler.update_state_after_alloc(seq)
    assert scheduler.should_park_for_load_after_alloc(seq)
    scheduler.cancel_pending_load(seq)
    assert scheduler._native_loads == {}
    assert scheduler._pinned_state_bytes == 0
    assert checkpoints.store.pool.num_free == 30
    assert scheduler.build_connector_meta().requests == []


def test_load_and_save_share_the_native_byte_budget(monkeypatch):
    scheduler, checkpoints, _ = make_scheduler(monkeypatch, budget=30)
    saving = sequence(1)
    checkpoint(scheduler, checkpoints, saving, 16)
    scheduler.update_state_after_alloc(saving)
    [request] = scheduler.build_connector_meta().requests
    loading = sequence(2, computed=0, token_offset=100)
    scheduler.get_num_new_matched_tokens(loading)
    scheduler.update_state_after_alloc(loading)
    assert scheduler.should_park_for_load_after_alloc(loading) is False
    assert scheduler._pinned_state_bytes == 30
    assert scheduler._native_loads == {}
    terminal(scheduler, request.save_operation)
    assert scheduler._pinned_state_bytes == 0


def engine_scheduler(monkeypatch):
    adapter = Adapter()
    monkeypatch.setenv("OFFLOAD_MIN_LOAD_TOKENS", "0")
    monkeypatch.setenv("OFFLOAD_MAX_PENDING_SAVES", "2")
    monkeypatch.setattr(
        backend, "_make_scheduler_adapter", lambda _config, **kwargs: adapter
    )
    config = MockConfig(
        num_kvcache_blocks=40,
        enable_prefix_caching=True,
        pool_entries={"state": 4},
        state_checkpoint_interval_tokens=0,
        state_checkpoint_demand=False,
        kv_transfer_config={
            "kv_connector": "lmcache_mp",
            "kv_role": "offload",
            "kv_connector_extra_config": {
                "lmcache.mp.max_pinned_state_bytes": 60,
                "lmcache.chunk_size": 8,
            },
        },
    )
    connector = LMCacheMPConnectorScheduler(config)
    monkeypatch.setattr(
        "atom.utils.forward_context.get_kvconnector", lambda _role, _config: connector
    )
    spec = PagedStateCheckpointSpec(10, 50, "native-engine-test", image_bytes=25)
    engine = Scheduler(
        config,
        state_runtime=StateRuntime(
            transfer=StateTransfer.copy(spec.layout_id), checkpoint_spec=spec
        ),
    )
    assert connector._block_manager is engine.block_manager
    return engine, connector, adapter


def start_engine_load(engine):
    seq = Sequence(list(range(24)), 4, has_per_req_cache=True)
    engine.add(seq)
    batch, scheduled = engine.schedule()
    assert scheduled == {}
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert seq.num_cached_tokens == 0
    assert seq.state_slot >= 0
    assert len(seq.block_table) == 6
    [request] = batch.connector_meta_output.requests
    assert request.load_operation == seq._load_operation
    assert request.native_state.destination_slot == seq.state_slot
    assert set(request.native_state.unit_ids).isdisjoint(seq.block_table)
    return seq, request


def test_engine_allocates_then_parks_and_wakes_at_exact_native_boundary(monkeypatch):
    engine, connector, adapter = engine_scheduler(monkeypatch)
    seq, request = start_engine_load(engine)
    block_table = list(seq.block_table)
    state_slot = seq.state_slot
    assert engine.block_manager.kv.num_free == 31
    assert connector._pinned_state_bytes == 30

    batch, scheduled = engine.schedule()
    assert scheduled == {} and batch.connector_meta_output.requests == []
    engine._update_from_kv_xfer_finished(
        KVConnectorOutput(
            finished_loading={
                LoadOperationId(seq.id, request.load_operation.generation + 1)
            }
        )
    )
    assert seq.status == SequenceStatus.WAITING_FOR_REMOTE_KVS
    assert connector._pinned_state_bytes == 30

    engine._update_from_kv_xfer_finished(
        KVConnectorOutput(finished_loading={request.load_operation})
    )
    assert connector._pinned_state_bytes == 0
    # The three transfer units remain as an unpinned READY checkpoint.
    assert engine.block_manager.kv.num_free == 31
    assert engine.block_manager.paged_state_checkpoints.contains(
        request.native_state.prefix_hash
    )
    batch, scheduled = engine.schedule()
    assert scheduled[seq.id] is seq
    assert seq.num_cached_tokens == 16
    assert list(batch.num_cached_tokens) == [16]
    assert seq.offload_promoted_tokens == 16
    assert list(seq.block_table) == block_table
    assert seq.state_slot == state_slot
    assert engine._num_parked_remote_kv == 0
    assert len(adapter.queries) == 1


@pytest.mark.parametrize("succeeded", [True, False])
def test_engine_aborted_load_keeps_page_units_and_slot_until_terminal(
    monkeypatch, succeeded
):
    engine, connector, _ = engine_scheduler(monkeypatch)
    seq, request = start_engine_load(engine)
    block_table = list(seq.block_table)
    state_slot = seq.state_slot
    seq.status = SequenceStatus.ABORTED
    engine.schedule()
    assert seq.status == SequenceStatus.FINISHED
    assert engine.deferred_free_blocks[seq.id] is seq
    assert list(seq.block_table) == block_table
    assert seq.state_slot == state_slot
    assert engine.block_manager.kv.num_free == 31
    assert connector._pinned_state_bytes == 30

    kwargs = {
        "finished_loading" if succeeded else "failed_loading": {request.load_operation}
    }
    engine._update_from_kv_xfer_finished(KVConnectorOutput(**kwargs))
    assert connector._pinned_state_bytes == 0
    assert seq.id not in engine.deferred_free_blocks
    assert list(seq.block_table) == []
    assert seq.state_slot == -1
    expected_free = 37 if succeeded else 40
    assert engine.block_manager.kv.num_free == expected_free
    assert (
        engine.block_manager.paged_state_checkpoints.contains(
            request.native_state.prefix_hash
        )
        is succeeded
    )
    assert engine._num_parked_remote_kv == 0


def test_engine_failed_native_load_recomputes_from_zero_without_reallocation(
    monkeypatch,
):
    engine, connector, adapter = engine_scheduler(monkeypatch)
    seq, request = start_engine_load(engine)
    state_slot = seq.state_slot
    block_table = list(seq.block_table)
    engine._update_from_kv_xfer_finished(
        KVConnectorOutput(failed_loading={request.load_operation})
    )
    assert connector._pinned_state_bytes == 0
    assert connector._save_tracker[str(seq.id)][1] == 0
    batch, scheduled = engine.schedule()
    assert scheduled[seq.id] is seq
    assert seq.num_cached_tokens == 0
    assert list(batch.num_cached_tokens) == [0]
    assert seq.offload_load_failed is True
    assert list(seq.block_table) == block_table
    assert seq.state_slot == state_slot
    assert engine._num_parked_remote_kv == 0
    assert len(adapter.queries) == 1


def test_engine_releases_retired_request_when_unpinned_candidate_is_evicted(
    monkeypatch,
):
    engine, connector, _ = engine_scheduler(monkeypatch)
    bm = engine.block_manager
    seq = Sequence(list(range(24)), 4, has_per_req_cache=True)
    assert bm.allocate(seq, bm.can_allocate(seq))
    bm.hash_blocks(seq, 16)
    seq.num_cached_tokens = 16
    checkpoints = bm.paged_state_checkpoints
    prefix_hash, _ = checkpoint(connector, checkpoints, seq, 16)
    connector.update_state_after_alloc(seq)
    seq.type = SequenceType.PREFILL
    seq.status = SequenceStatus.RUNNING
    engine.running.append(seq)
    batch = ScheduledBatch(
        {seq.id: seq},
        [8],
        8,
        total_tokens_num_prefill=8,
        total_seqs_num=1,
        total_seqs_num_prefill=1,
        num_cached_tokens=[16],
        is_final_chunk=[True],
    )
    engine.postprocess(
        [seq],
        ScheduledBatchOutput([seq.id], [(engine.eos_token_id,)], None, None, None),
        batch=batch,
    )
    assert seq.status == SequenceStatus.FINISHED
    # A not-yet-admitted candidate owns no PAGE lease. The request's KV blocks
    # and active SLOT are therefore released immediately.
    assert seq.id not in engine.deferred_free_blocks
    assert seq.state_slot == -1
    assert list(seq.block_table) == []
    assert connector._native_saves == {}
    checkpoint_id = checkpoints.store.lookup(prefix_hash)
    assert checkpoints.store.records[checkpoint_id].pin_count == 0

    # The next admission may evict this waiting candidate. It has never been
    # sent, so there will be no GPU completion naming this finished request.
    checkpoints.unindex(prefix_hash)
    assert connector.has_pending_work()
    engine._update_from_kv_xfer_finished(KVConnectorOutput())
    assert seq.id not in engine.deferred_free_blocks
    assert bm.kv.num_free == 40
    assert connector._retired_requests == {}
    assert connector._save_tracker == {}
    assert not connector.has_pending_work()
