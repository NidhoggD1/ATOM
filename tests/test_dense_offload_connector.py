# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
from atom.kv_transfer.disaggregation.types import (
    KVConnectorOutput,
    LoadOperationId,
    SaveOperationId,
)
from atom.kv_transfer.offload import _offload_common
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload.dense.connector import (
    DenseOffloadConnector,
    DenseOffloadScheduler,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheReqMeta,
    LoadSpec,
    SaveSpec,
)
from atom.model_engine.scheduler import Scheduler
from atom.model_engine.sequence import SequenceStatus


def _config(role="offload"):
    return SimpleNamespace(
        kv_transfer_config={"kv_role": role},
        kv_cache_block_size=4,
        decode_context_parallel_size=2,
        tensor_parallel_size=1,
    )


def _scheduler(monkeypatch, role="offload"):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(chunk_size=8),
    )
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_metadata",
        lambda *_args: object(),
    )
    return DenseOffloadScheduler(_config(role))


def _load_seq(req_id, *, num_prompt_tokens=8):
    return SimpleNamespace(
        id=req_id,
        num_cached_tokens=0,
        num_prompt_tokens=num_prompt_tokens,
        token_ids=list(range(num_prompt_tokens)),
        block_table=list(range(num_prompt_tokens // 8)),
    )


def _arm_load(scheduler, seq, *, hbm=0, lmcache=8):
    sid = str(seq.id)
    scheduler._min_load_tokens = 0
    scheduler._load_specs[sid] = LoadSpec(
        hbm_cached_tokens=hbm,
        lmcache_cached_tokens=lmcache,
        can_load=True,
    )
    scheduler._reqs_need_recv[sid] = seq


def _engine_scheduler(connector):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.kv_connector = connector
    scheduler.finished_recving_kv_req_ids = []
    scheduler.failed_recving_kv_req_ids = []
    scheduler.deferred_free_blocks = {}
    return scheduler


@pytest.mark.parametrize(
    "connector_cls", [DenseOffloadConnector, DenseOffloadScheduler]
)
def test_dense_backend_rejects_unknown_role(monkeypatch, connector_cls):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(chunk_size=8),
    )

    with pytest.raises(ValueError, match="invalid kv_role"):
        connector_cls(_config("invalid"))


def test_dense_scheduler_invalid_lmcache_config_fails_fast(monkeypatch):
    def invalid_config(_config=None):
        raise ValueError("invalid LMCache storage")

    monkeypatch.setattr(offcfg, "build_lmcache_config", invalid_config)

    with pytest.raises(ValueError, match="invalid LMCache storage"):
        DenseOffloadScheduler(_config())


def test_dense_scheduler_invalid_lmcache_metadata_fails_fast(monkeypatch):
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(chunk_size=8),
    )

    def invalid_metadata(*_args):
        raise ValueError("invalid LMCache layer geometry")

    monkeypatch.setattr(offcfg, "build_lmcache_metadata", invalid_metadata)

    with pytest.raises(ValueError, match="invalid LMCache layer geometry"):
        DenseOffloadScheduler(_config())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kv_cache_block_size", 4.5, "Dense block size must be an integer"),
        ("chunk_size", 8.5, "LMCache chunk size must be an integer"),
    ],
)
def test_dense_scheduler_rejects_coerced_geometry(
    monkeypatch,
    field,
    value,
    message,
):
    config = _config()
    if field == "kv_cache_block_size":
        config.kv_cache_block_size = value
    monkeypatch.setattr(
        offcfg,
        "build_lmcache_config",
        lambda _config=None: SimpleNamespace(
            chunk_size=value if field == "chunk_size" else 8
        ),
    )

    with pytest.raises(ValueError, match=message):
        DenseOffloadScheduler(config)


def test_dense_worker_resolves_virtual_dcp_block_size():
    worker = DenseOffloadConnector(_config())
    try:
        assert worker.block_size == 4
        assert worker.virtual_block_size == 8
    finally:
        worker._save_executor.shutdown(wait=True)
        worker._load_executor.shutdown(wait=True)


def test_dense_producer_role_tracks_only_saves(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_producer")
    seq = SimpleNamespace(
        id=1,
        num_cached_tokens=0,
        num_prompt_tokens=8,
        token_ids=list(range(8)),
        block_table=[3],
    )

    scheduler.update_state_after_alloc(seq)

    assert scheduler._do_save is True
    assert scheduler._do_load is False
    assert scheduler._save_tracker["1"] == [seq, 0]
    assert scheduler.get_num_new_matched_tokens(seq) == (0, False)


def test_dense_consumer_role_does_not_defer_for_saves(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    seq = SimpleNamespace(
        id=2,
        num_cached_tokens=8,
        num_prompt_tokens=8,
        token_ids=list(range(8)),
        block_table=[4],
    )

    scheduler.update_state_after_alloc(seq)

    assert scheduler._do_save is False
    assert scheduler._save_tracker == {}
    assert scheduler.should_defer_free(seq) is False


def test_dense_active_load_defers_only_its_concrete_lifecycle(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    seq = _load_seq(20)
    replacement = _load_seq(20)
    operation = LoadOperationId(seq.id, 3)
    scheduler._active_load_operations[str(seq.id)] = (seq, operation)

    assert scheduler.should_defer_free(seq) is True
    assert scheduler.should_defer_free(replacement) is False

    assert scheduler.load_finished(operation) is True
    assert scheduler.should_defer_free(seq) is False


def test_dense_reused_request_id_resets_save_frontier(monkeypatch):
    scheduler = _scheduler(monkeypatch)
    first = SimpleNamespace(
        id=3,
        num_cached_tokens=8,
        num_prompt_tokens=8,
        token_ids=list(range(8)),
        block_table=[5],
    )
    replacement = SimpleNamespace(
        id=3,
        num_cached_tokens=0,
        num_prompt_tokens=8,
        token_ids=list(range(8)),
        block_table=[6],
    )
    scheduler._save_tracker["3"] = [first, 8]

    scheduler.update_state_after_alloc(replacement)

    assert scheduler._save_tracker["3"] == [replacement, 0]


def test_dense_build_load_metadata_uses_increasing_exact_generations(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    seq = _load_seq(11)

    _arm_load(scheduler, seq)
    first = scheduler.build_connector_meta().requests[0].load_operation
    _arm_load(scheduler, seq)
    second = scheduler.build_connector_meta().requests[0].load_operation

    assert first == LoadOperationId(req_id=11, generation=0)
    assert second == LoadOperationId(req_id=11, generation=1)
    assert scheduler._active_load_operations["11"] == (seq, second)


def test_dense_build_save_metadata_uses_increasing_exact_generations(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_producer")
    seq = _load_seq(12, num_prompt_tokens=16)
    scheduler.update_state_after_alloc(seq)
    seq.num_cached_tokens = 8

    first_meta = scheduler.build_connector_meta()
    first = first_meta.requests[0].save_operation
    scheduler.save_finished(first)
    seq.num_cached_tokens = 16
    second_meta = scheduler.build_connector_meta()
    second = second_meta.requests[0].save_operation

    assert first == SaveOperationId(req_id=12, generation=0)
    assert second == SaveOperationId(req_id=12, generation=1)
    assert scheduler._save_inflight["12"] == second


def test_dense_stale_or_raw_save_completion_cannot_clear_exact_lifecycle(
    monkeypatch,
):
    scheduler = _scheduler(monkeypatch, "kv_producer")
    seq = _load_seq(13, num_prompt_tokens=16)
    scheduler.update_state_after_alloc(seq)
    seq.num_cached_tokens = 8
    stale = scheduler.build_connector_meta().requests[0].save_operation
    scheduler.save_finished(stale)

    seq.num_cached_tokens = 16
    current = scheduler.build_connector_meta().requests[0].save_operation
    scheduler.save_finished(stale)
    scheduler.save_finished(seq.id)

    assert scheduler._save_inflight["13"] == current

    scheduler.save_finished(current)
    assert "13" not in scheduler._save_inflight

    # A raw completion remains compatible with an explicitly legacy lifecycle.
    scheduler._save_inflight["legacy"] = "legacy"
    scheduler.save_finished("legacy")
    assert "legacy" not in scheduler._save_inflight


def test_dense_worker_exact_save_generations_do_not_form_cross_tp_quorum():
    workers = [DenseOffloadConnector(_config("kv_producer")) for _ in range(2)]
    operations = [
        SaveOperationId(req_id=14, generation=6),
        SaveOperationId(req_id=14, generation=7),
    ]

    try:
        outputs = []
        for worker_idx, (worker, operation) in enumerate(zip(workers, operations)):
            worker.chunk_size = 8
            skip = 8
            if worker_idx:
                skip = 0
                worker._engine = SimpleNamespace(
                    gpu_connector=None,
                    store=lambda _tokens, **_kwargs: None,
                )
            worker._do_save_req(
                LMCacheReqMeta(
                    req_id=14,
                    token_ids=list(range(8)),
                    block_ids=[3],
                    save_spec=SaveSpec(skip_leading_tokens=skip),
                    save_operation=operation,
                )
            )
            outputs.append(worker.get_finished())

        assert outputs[0].finished_saving == {operations[0]}
        assert outputs[1].finished_saving == {operations[1]}
        assert (
            KVOutputAggregator(world_size=2).aggregate(outputs).finished_saving == set()
        )
    finally:
        for worker in workers:
            worker._save_executor.shutdown(wait=True)
            worker._load_executor.shutdown(wait=True)


@pytest.mark.parametrize("outcome", ["exception", "miss"])
def test_dense_worker_load_failure_reports_exact_operation(outcome):
    worker = DenseOffloadConnector(_config("kv_consumer"))
    operation = LoadOperationId(req_id=21, generation=4)
    request = LMCacheReqMeta(
        req_id=21,
        token_ids=list(range(8)),
        block_ids=[3],
        load_spec=LoadSpec(
            hbm_cached_tokens=0,
            lmcache_cached_tokens=8,
            can_load=True,
        ),
        load_operation=operation,
    )
    worker.chunk_size = 8

    try:
        if outcome == "exception":

            def fail_load(_request):
                raise RuntimeError("synthetic load failure")

            worker._guard("load", fail_load, request)
        else:

            class MissEngine:
                gpu_connector = None

                @staticmethod
                def retrieve(_tokens, *, mask, **_kwargs):
                    return mask.clone().fill_(False)

                @staticmethod
                def lookup_unpin(_lookup_id):
                    pass

            worker._engine = MissEngine()
            worker._do_load_req(request)

        result = worker.get_finished()

        assert result.finished_loading == set()
        assert result.failed_loading == {operation}
    finally:
        worker._save_executor.shutdown(wait=True)
        worker._load_executor.shutdown(wait=True)


def test_dense_exact_load_failure_rolls_back_save_frontier(monkeypatch):
    connector = _scheduler(monkeypatch)
    seq = _load_seq(31, num_prompt_tokens=16)
    operation = LoadOperationId(req_id=31, generation=2)
    connector._save_tracker["31"] = [seq, 16]
    connector._load_save_floors["31"] = 8
    connector._active_load_operations["31"] = (seq, operation)
    scheduler = _engine_scheduler(connector)

    scheduler._update_from_kv_xfer_finished(
        KVConnectorOutput(failed_loading={operation})
    )

    assert connector._save_tracker["31"] == [seq, 8]
    assert "31" not in connector._active_load_operations
    assert "31" not in connector._load_save_floors
    assert scheduler.failed_recving_kv_req_ids == [31]


def test_dense_stale_load_generation_does_not_clear_active_operation(monkeypatch):
    connector = _scheduler(monkeypatch)
    seq = _load_seq(41, num_prompt_tokens=16)
    stale = LoadOperationId(req_id=41, generation=6)
    active = LoadOperationId(req_id=41, generation=7)
    connector._save_tracker["41"] = [seq, 16]
    connector._load_save_floors["41"] = 8
    connector._active_load_operations["41"] = (seq, active)
    scheduler = _engine_scheduler(connector)

    scheduler._update_from_kv_xfer_finished(KVConnectorOutput(failed_loading={stale}))

    assert connector._active_load_operations["41"] == (seq, active)
    assert connector._load_save_floors["41"] == 8
    assert connector._save_tracker["41"] == [seq, 16]
    assert scheduler.failed_recving_kv_req_ids == []


def test_dense_lookup_unpin_passes_one_string_id():
    worker = DenseOffloadConnector(_config("kv_consumer"))
    received = []
    worker._engine = SimpleNamespace(lookup_unpin=received.append)

    try:
        worker._lookup_unpin(51)

        assert received == ["51"]
    finally:
        worker._save_executor.shutdown(wait=True)
        worker._load_executor.shutdown(wait=True)


@pytest.mark.parametrize("outcome", ["hbm_hit", "small_hit", "cancel", "miss", "error"])
def test_unused_lookup_releases_worker_pin(monkeypatch, outcome):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    worker = DenseOffloadConnector(_config("kv_consumer"))
    pins = set()
    calls = []

    def lookup(_tokens, lookup_id):
        pins.add(lookup_id)
        calls.append(lookup_id)
        if outcome == "error":
            raise RuntimeError("lookup transport failed after a worker pinned")
        return 0 if outcome == "miss" else 16

    scheduler._lookup_client = SimpleNamespace(
        lookup=lookup, clear_lookup_status=lambda _sid: None
    )
    worker._engine = SimpleNamespace(lookup_unpin=pins.discard)
    seq = _load_seq(52, num_prompt_tokens=24)
    if outcome == "hbm_hit":
        seq.num_cached_tokens = 16
    try:
        scheduler.get_num_new_matched_tokens(seq)
        if outcome == "cancel":
            scheduler.cancel_pending_load(seq)
        elif outcome == "small_hit":
            scheduler.update_state_after_alloc(seq)
        else:
            # Repeated scheduler probes must not acquire another pin lease.
            scheduler.get_num_new_matched_tokens(seq)
        assert calls == ["52"]
        assert scheduler.has_pending_work()
        metadata = scheduler.build_connector_meta()
        assert metadata.lookup_requests_in_step == ["52"]
        worker.start_load_kv(metadata)
        assert pins == set()
        assert not scheduler.has_pending_work()
    finally:
        worker.close()


def test_pending_load_keeps_pin_until_cancelled(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    scheduler._lookup_client = SimpleNamespace(
        lookup=lambda *_args, **_kwargs: 16,
        clear_lookup_status=lambda _sid: None,
    )
    seq = _load_seq(53, num_prompt_tokens=24)
    assert scheduler.get_num_new_matched_tokens(seq) == (16, True)
    # A pre-allocation lookup is owned by the waiting request, not dispatchable
    # idle work. It must retain its pin without keeping the idle drain alive.
    assert not scheduler.has_pending_work()
    assert scheduler.build_connector_meta().lookup_requests_in_step == []
    scheduler.cancel_pending_load(seq)
    assert scheduler.has_pending_work()
    assert scheduler.build_connector_meta().lookup_requests_in_step == ["53"]
    assert scheduler.build_connector_meta().lookup_requests_in_step == []


def test_scheduler_abort_before_allocation_releases_lookup_pin(
    monkeypatch, scheduler, seq_factory
):
    connector = _scheduler(monkeypatch, "kv_consumer")
    worker = DenseOffloadConnector(_config("kv_consumer"))
    pins = set()
    calls = []

    def lookup(_tokens, lookup_id):
        calls.append(lookup_id)
        pins.add(lookup_id)
        return 16

    connector._lookup_client = SimpleNamespace(
        lookup=lookup, clear_lookup_status=lambda _sid: None
    )
    worker._engine = SimpleNamespace(lookup_unpin=pins.discard)
    scheduler.kv_connector = connector
    seq = seq_factory(list(range(24)))
    scheduler.add(seq)
    allocation_attempts = []

    def cannot_allocate(value):
        allocation_attempts.append(value.id)
        return -1

    monkeypatch.setattr(scheduler.block_manager, "can_allocate", cannot_allocate)
    sid = str(seq.id)
    try:
        scheduler.schedule()
        assert list(scheduler.waiting) == [seq]
        assert calls == [sid]
        assert allocation_attempts == [seq.id]
        assert pins == {sid}
        assert sid in connector._load_specs
        assert not getattr(seq, "_counted_as_inflight_load", False)
        assert connector.build_connector_meta().lookup_requests_in_step == []

        seq.status = SequenceStatus.ABORTED
        batch, scheduled = scheduler.schedule()
        assert seq.status == SequenceStatus.FINISHED
        assert scheduler._num_parked_remote_kv == 0
        assert sid not in connector._load_specs
        assert sid not in connector._load_lifecycles
        assert scheduler.deferred_free_blocks == {}
        assert scheduled == {}
        # schedule() already dispatched cleanup into this empty batch.
        meta = batch.connector_meta_output
        assert meta.requests == []
        assert meta.lookup_requests_in_step == [sid]
        worker.start_load_kv(meta)
        assert pins == set()
        assert connector._lookup_results == {}
        assert not connector.has_pending_work()
        assert connector.build_connector_meta().lookup_requests_in_step == []
    finally:
        worker.close()


def test_lookup_id_reuse_does_not_consume_an_old_pin(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    calls = []
    scheduler._lookup_client = SimpleNamespace(
        lookup=lambda _tokens, lookup_id: calls.append(lookup_id) or 16,
        clear_lookup_status=lambda _sid: None,
    )
    old = _load_seq(54, num_prompt_tokens=24)
    new = _load_seq(54, num_prompt_tokens=32)
    scheduler.get_num_new_matched_tokens(old)
    scheduler.cancel_pending_load(old)
    assert scheduler.get_num_new_matched_tokens(new) == (0, False)
    assert calls == ["54"]
    assert scheduler.build_connector_meta().lookup_requests_in_step == ["54"]
    assert scheduler.get_num_new_matched_tokens(new) == (16, True)
    assert calls == ["54", "54"]


def test_hbm_catches_up_after_a_pending_cpu_lookup(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    scheduler._lookup_client = SimpleNamespace(
        lookup=lambda *_args, **_kwargs: 16,
        clear_lookup_status=lambda _sid: None,
    )
    seq = _load_seq(55, num_prompt_tokens=24)
    assert scheduler.get_num_new_matched_tokens(seq) == (16, True)
    seq.num_cached_tokens = 16
    assert scheduler.get_num_new_matched_tokens(seq) == (0, False)
    metadata = scheduler.build_connector_meta()
    assert metadata.lookup_requests_in_step == ["55"]
    assert metadata.requests == []


def test_cpu_pin_is_retained_until_retrieve_finishes(monkeypatch):
    scheduler = _scheduler(monkeypatch, "kv_consumer")
    scheduler._min_load_tokens = 0
    scheduler._lookup_client = SimpleNamespace(
        lookup=lambda *_args, **_kwargs: 16,
        clear_lookup_status=lambda _sid: None,
    )
    seq = _load_seq(56, num_prompt_tokens=24)
    scheduler.get_num_new_matched_tokens(seq)
    scheduler.update_state_after_alloc(seq)
    metadata = scheduler.build_connector_meta()
    pins = {"56"}

    def retrieve(_tokens, *, mask, **_kwargs):
        assert pins == {"56"}
        return mask.clone()

    worker = DenseOffloadConnector(_config("kv_consumer"))
    worker.chunk_size = 8
    worker._engine = SimpleNamespace(retrieve=retrieve, lookup_unpin=pins.discard)
    try:
        worker.start_load_kv(metadata)
        worker.close()
        assert pins == set()
        assert worker.get_finished().finished_loading == {
            metadata.requests[0].load_operation
        }
    finally:
        worker.close()


def test_dense_worker_records_the_blocks_a_failed_load_left_unfilled():
    """The id alone does not truncate anything.

    vLLM caches the whole external prefix unless the failure also names blocks:
    `_update_requests_with_invalid_blocks` cuts `num_computed_tokens` at the
    first block reported here. Blocks below the HBM frontier are deliberately
    excluded -- they hold valid KV and may be shared with another request.

    The grid is the virtual block size (block_size 4 x dcp 2 = 8), matching the
    one `BlockGPUConnector` maps chunks with; the physical size would index
    different entries of the very same table.
    """
    worker = DenseOffloadConnector(_config("kv_consumer"))
    request = LMCacheReqMeta(
        req_id=61,
        token_ids=list(range(16)),
        block_ids=[10, 11, 12, 13],
        load_spec=LoadSpec(
            hbm_cached_tokens=4,
            lmcache_cached_tokens=12,
            can_load=True,
        ),
        load_operation=LoadOperationId(req_id=61, generation=1),
    )

    try:
        with worker._lock:
            worker._record_load_error_blocks(request)

        # Tokens [4, 12) on a grid of 8 are entries 0 and 1 of the table.
        assert worker.take_load_error_blocks() == {10, 11}
        # Drained, so the next step does not truncate a request all over again.
        assert worker.take_load_error_blocks() == set()
    finally:
        worker._save_executor.shutdown(wait=True)
        worker._load_executor.shutdown(wait=True)


def test_dense_worker_fences_in_flight_jobs_for_a_preempted_request():
    """Preemption hands the blocks to somebody else in the same step.

    A save still gathering from them stores the new occupant's bytes under the
    preempted request's key -- a poisoned cache entry, not a lost one. The fence
    is the only thing standing between the two.
    """
    import threading

    worker = DenseOffloadConnector(_config("kv_consumer"))
    release = threading.Event()
    finished = []

    def slow_job(_request):
        release.wait(timeout=5)
        finished.append(True)

    try:
        worker._track_job(
            71,
            worker._save_executor.submit(
                worker._guard, "save", slow_job, SimpleNamespace(req_id=71)
            ),
        )
        assert worker._inflight_jobs["71"]

        # Another request's ids are not this request's business.
        worker.wait_for_requests(["72"])
        assert finished == []

        release.set()
        worker.wait_for_requests(["71"])

        assert finished == [True]
        assert "71" not in worker._inflight_jobs
    finally:
        release.set()
        worker._save_executor.shutdown(wait=True)
        worker._load_executor.shutdown(wait=True)


def test_dense_load_failure_by_request_resolves_the_parked_generation(monkeypatch):
    """vLLM hands back plain strings; `load_failed` refuses a raw id.

    Routing the failure through `load_finished` instead would pop the floor
    recording that the [HBM, LMCache) range is NOT persisted, so the recomputed
    chunks would never be saved.
    """
    connector = _scheduler(monkeypatch)
    seq = _load_seq(81, num_prompt_tokens=16)
    operation = LoadOperationId(req_id=81, generation=3)
    connector._save_tracker["81"] = [seq, 16]
    connector._load_save_floors["81"] = 8
    connector._active_load_operations["81"] = (seq, operation)

    assert connector.load_failed_by_request("81") is True

    assert connector._save_tracker["81"] == [seq, 8]
    assert "81" not in connector._active_load_operations


def test_dense_worker_pool_widths_follow_env(monkeypatch):
    """A single load thread saturates once the CPU tier serves real traffic.

    Measured on the radix workload with the HBM pool squeezed to 7900 blocks:
    88% duty cycle inside `retrieve` on every rank, which turned a +57.8pp
    hit-rate win into a throughput loss. Both pools must be tunable, and both
    must keep their one-thread default so existing deployments are unchanged.
    """

    worker = DenseOffloadConnector(_config())
    try:
        assert (worker.save_workers, worker.load_workers) == (1, 1)
        assert worker._save_executor._max_workers == 1
        assert worker._load_executor._max_workers == 1
    finally:
        worker.close()

    monkeypatch.setenv("OFFLOAD_COPY_WORKERS", "4")
    monkeypatch.setenv("OFFLOAD_LOAD_WORKERS", "3")
    worker = DenseOffloadConnector(_config())
    try:
        assert (worker.save_workers, worker.load_workers) == (4, 3)
        assert worker._save_executor._max_workers == 4
        assert worker._load_executor._max_workers == 3
    finally:
        worker.close()


@pytest.mark.parametrize("var", ["OFFLOAD_COPY_WORKERS", "OFFLOAD_LOAD_WORKERS"])
def test_dense_worker_rejects_non_positive_pool_width(monkeypatch, var):
    monkeypatch.setenv(var, "0")
    with pytest.raises(ValueError, match="worker count must be positive"):
        DenseOffloadConnector(_config())


class _StubLookupClient:
    """Lookup client that always reports the same hit length."""

    def __init__(self, hit):
        self.hit = hit
        self.calls = 0
        self.cleared = []

    def lookup(self, token_ids, lookup_id):
        self.calls += 1
        return self.hit

    def clear_lookup_status(self, lookup_id):
        self.cleared.append(lookup_id)


def _lookup_scheduler(monkeypatch, hit, *, role="offload"):
    sched = _scheduler(monkeypatch, role)
    sched._lookup_client = _StubLookupClient(hit)
    sched._min_load_tokens = 0  # these prompts are far below the 8192 default
    return sched


def _dispatch_and_fail_load(sched, seq):
    """Run the alloc + metadata step, then fail the load it dispatched.

    The metadata build is what releases the lookup memo, so a later pass is a
    fresh lookup rather than the "older lifecycle still owns the pin" deferral.
    """

    sched.update_state_after_alloc(seq)
    meta = sched.build_connector_meta()
    (load,) = [req for req in meta.requests if req.load_spec is not None]
    assert sched.load_failed(load.load_operation) is True


def test_dense_full_prompt_hit_is_floored_to_a_loadable_chunk(monkeypatch):
    # A hit covering the whole prompt is decremented so something is left to
    # compute. With a prompt length that is an exact multiple of the chunk
    # size, that decrement lands off the chunk boundary -- and LMCache resolves
    # at chunk granularity, so the resulting spec asks for tokens the tier can
    # never return and the load fails every time. The floor is what keeps the
    # spec satisfiable. Scaled-down mirror of the production shape observed on
    # GLM-5.2 (chunk 64, prompt 32768: 32767 requested, 32704 available).
    sched = _lookup_scheduler(monkeypatch, hit=16)
    assert sched.chunk_size == 8
    seq = _load_seq(940, num_prompt_tokens=16)

    assert sched.get_num_new_matched_tokens(seq) == (8, True)
    assert sched._load_specs["940"].lmcache_cached_tokens == 8


def test_dense_partial_hit_off_a_chunk_boundary_is_floored(monkeypatch):
    # The same arithmetic with no decrement involved: any unaligned hit names
    # tokens the tier does not hold at chunk granularity.
    sched = _lookup_scheduler(monkeypatch, hit=13)
    seq = _load_seq(941, num_prompt_tokens=24)

    assert sched.get_num_new_matched_tokens(seq) == (8, True)
    assert sched._load_specs["941"].lmcache_cached_tokens == 8


def test_dense_failed_load_is_not_retried_for_the_same_request(monkeypatch):
    # `load_failed` clears the pending load and the lookup memo, so without a
    # record of the attempt the next scheduler pass looks up, hits, parks the
    # request in WAITING_FOR_REMOTE_KVS and fails again -- forever, holding its
    # KV blocks and concurrency slot. One attempt per request; after that the
    # request prefills normally.
    sched = _lookup_scheduler(monkeypatch, hit=16)
    seq = _load_seq(942, num_prompt_tokens=24)

    assert sched.get_num_new_matched_tokens(seq) == (16, True)
    _dispatch_and_fail_load(sched, seq)

    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    assert sched.total_suppressed_load_retries == 1
    assert sched._load_specs == {}
    assert sched._lookup_client.calls == 1


def test_dense_new_sequence_reusing_request_id_gets_a_fresh_attempt(monkeypatch):
    # The mark is against the sequence, not the ID: a request ID leased to a
    # new sequence has spent nothing.
    sched = _lookup_scheduler(monkeypatch, hit=16)
    seq = _load_seq(943, num_prompt_tokens=24)

    assert sched.get_num_new_matched_tokens(seq) == (16, True)
    _dispatch_and_fail_load(sched, seq)

    reused = _load_seq(943, num_prompt_tokens=24)
    assert sched.get_num_new_matched_tokens(reused) == (16, True)
    assert sched.total_suppressed_load_retries == 0


def test_dense_request_finished_releases_the_failed_load_mark(monkeypatch):
    sched = _lookup_scheduler(monkeypatch, hit=16)
    seq = _load_seq(944, num_prompt_tokens=24)

    assert sched.get_num_new_matched_tokens(seq) == (16, True)
    _dispatch_and_fail_load(sched, seq)
    assert sched._load_failed_seqs == {"944": seq}

    sched.request_finished(seq)
    assert sched._load_failed_seqs == {}


def test_dense_repeat_examination_issues_one_lookup(monkeypatch):
    """A queued request is re-examined every step; only the first may look up.

    The scheduler calls this hook for every waiting request whose
    ``num_computed_tokens`` is still zero, so a request that loses the budget
    race comes back next step. The lookup itself is memoised, and the prompt
    copy it needs belongs inside that same branch -- outside it, every losing
    examination pays to copy a prompt nobody reads.
    """
    sched = _lookup_scheduler(monkeypatch, hit=16)
    seq = _load_seq(945, num_prompt_tokens=24)

    first = sched.get_num_new_matched_tokens(seq)
    second = sched.get_num_new_matched_tokens(seq)

    assert first == second == (16, True)
    assert sched._lookup_client.calls == 1
    assert sched._lookup_calls == 1


def test_dense_lookup_cost_is_tallied(monkeypatch):
    """The synchronous lookup is timed, because admission cannot reach it.

    Save admission throttles bytes; this call is paid by every request that is
    examined, saved or not, so it needs its own number.
    """
    sched = _lookup_scheduler(monkeypatch, hit=16)
    sched.get_num_new_matched_tokens(_load_seq(946, num_prompt_tokens=24))
    sched.get_num_new_matched_tokens(_load_seq(947, num_prompt_tokens=32))

    assert sched._lookup_calls == 2
    assert sched._lookup_seconds > 0.0
    assert sched._lookup_tokens == 24 + 32


def _skip_scheduler(monkeypatch, *, enabled):
    if enabled:
        monkeypatch.setenv("OFFLOAD_LOOKUP_SKIP_UNLOADABLE", "1")
    else:
        monkeypatch.delenv("OFFLOAD_LOOKUP_SKIP_UNLOADABLE", raising=False)
    sched = _lookup_scheduler(monkeypatch, hit=20000)
    sched._min_load_tokens = 8192
    return sched


def _cached_seq(req_id, *, num_prompt_tokens, num_cached_tokens):
    seq = _load_seq(req_id, num_prompt_tokens=num_prompt_tokens)
    seq.num_cached_tokens = num_cached_tokens
    return seq


def test_dense_lookup_skipped_when_hbm_leaves_no_room_to_load(monkeypatch):
    """HBM covers all but a sub-floor tail, so no answer could be acted on."""
    sched = _skip_scheduler(monkeypatch, enabled=True)
    seq = _cached_seq(948, num_prompt_tokens=20000, num_cached_tokens=16000)

    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    assert sched._lookup_client.calls == 0
    # The floor a hit would have left. It is the connector's only record of
    # what LMCache holds; without it the next examination re-saves everything.
    assert sched._hit_save_floors["948"] == 16000

    sched.update_state_after_alloc(seq)
    assert sched._save_tracker["948"][1] == 16000


def test_dense_lookup_still_issued_when_the_tail_could_be_loaded(monkeypatch):
    sched = _skip_scheduler(monkeypatch, enabled=True)
    seq = _cached_seq(949, num_prompt_tokens=20000, num_cached_tokens=0)

    need, park = sched.get_num_new_matched_tokens(seq)

    assert sched._lookup_client.calls == 1
    # Full-prompt hit: one token back so something is left to compute, then
    # floored to a chunk the tier can actually serve.
    assert (need, park) == (19992, True)


def test_dense_lookup_skip_is_off_by_default(monkeypatch):
    sched = _skip_scheduler(monkeypatch, enabled=False)
    seq = _cached_seq(950, num_prompt_tokens=20000, num_cached_tokens=16000)

    sched.get_num_new_matched_tokens(seq)

    assert sched._lookup_client.calls == 1


def _measure_actionable(sched, actionable, lookups=64):
    """Feed the estimator as `lookups` real lookups would.

    Driving whole requests through `get_num_new_matched_tokens` would measure
    the same thing but would also have to dodge the skip under test, so the two
    accounting calls a lookup makes are invoked directly. `min_load` is 8192
    here, so a hit 20000 past the HBM frontier is actionable and one level with
    it is not.
    """
    hit, hbm = (28192, 20000) if actionable else (20000, 20000)
    for _ in range(lookups):
        sched._lookup_calls = getattr(sched, "_lookup_calls", 0) + 1
        sched._note_lookup_actionable(hit, hbm)


def test_dense_skip_stands_down_when_slow_tier_is_load_bearing(monkeypatch):
    """A tier whose answers are actionable must not be starved of writes.

    The bound still says no answer could produce a load *for this request*.
    But skipping also leaves the save floor at the HBM frontier, and where the
    tier is really serving, that suppressed write is a hit somebody loses
    later -- measured at 16612 blocks as 3.8 points of throughput.
    """
    sched = _skip_scheduler(monkeypatch, enabled=True)
    _measure_actionable(sched, True)
    seq = _cached_seq(951, num_prompt_tokens=20000, num_cached_tokens=16000)

    sched.get_num_new_matched_tokens(seq)
    assert sched._lookup_client.calls == 1


def test_dense_skip_fires_when_answers_are_never_actionable(monkeypatch):
    """The 46647-block case: LMCache answers nearly every lookup, and HBM
    already held all of it -- 2301 of 2305 hits discarded. The round trip buys
    nothing, so a hit rate would have kept it; an actionable rate drops it."""
    sched = _skip_scheduler(monkeypatch, enabled=True)
    _measure_actionable(sched, False)
    seq = _cached_seq(952, num_prompt_tokens=20000, num_cached_tokens=16000)

    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    assert sched._lookup_client.calls == 0
    assert sched._hit_save_floors["952"] == 16000


def test_dense_unmeasured_slow_tier_does_not_block_the_skip(monkeypatch):
    """Before the warmup there is no estimate, and no estimate allows the skip.

    Safe in that direction: the skip's bound needs HBM to cover all but a
    sub-floor tail, which a cold pool does not, so lookups run and feed the
    estimator before the skip can fire in a real run.
    """
    sched = _skip_scheduler(monkeypatch, enabled=True)
    _measure_actionable(sched, True, lookups=8)
    seq = _cached_seq(953, num_prompt_tokens=20000, num_cached_tokens=16000)

    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    assert sched._lookup_client.calls == 0


def test_dense_skipped_lookup_still_records_a_negative_verdict(monkeypatch):
    """A skip must not hide from save admission what it already knows.

    The skip fires only when no slow-tier answer could clear
    `_decide_load_after_alloc`, which is the same thing `paid_off=False`
    records -- so the verdict is owed whether or not the lookup ran.
    """
    sched = _skip_scheduler(monkeypatch, enabled=True)
    seq = _cached_seq(954, num_prompt_tokens=20000, num_cached_tokens=16000)
    before = sched.slow_tier_verdict_totals()
    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    assert sched._lookup_client.calls == 0
    paid_off, probes = sched.slow_tier_verdict_totals()
    assert probes == before[1] + 1
    assert paid_off == before[0]


def _feed_actionable_every(sched, every, lookups):
    """Feed `lookups` samples of which every `every`-th is actionable.

    Reproduces a measured trajectory rather than a corner: 46647 blocks
    returned 4 actionable answers in 2048 probes (every=512), and a cold 16612
    tier trickles a few percent per block long before it becomes properly
    useful (every=32). The gate has to separate those two, and they differ by
    an order of magnitude, not by the 10x margin either has to 10%.
    """
    for i in range(lookups):
        hit, hbm = (28192, 20000) if i % every == 0 else (20000, 20000)
        sched._lookup_calls = getattr(sched, "_lookup_calls", 0) + 1
        sched._note_lookup_actionable(hit, hbm)


def test_dense_cold_tier_trickle_holds_the_gate_shut(monkeypatch):
    """A filling tier must keep its writes even while it looks useless.

    At 16612 blocks the tier needs ~1536 probes before its answers are worth
    loading, and for all of that time its actionable rate reads 0-17%. A 10%
    cut opened here and starved the fill -- 1280 skips, -1.91% against the
    never-skip control. A trickle of one usable answer in 32 settles the EMA
    near 0.039, so a 1% cut keeps paying for the round trip.
    """
    sched = _skip_scheduler(monkeypatch, enabled=True)
    _feed_actionable_every(sched, every=32, lookups=256)
    assert sched._lookup_actionable_ema > 0.01
    seq = _cached_seq(955, num_prompt_tokens=20000, num_cached_tokens=16000)

    sched.get_num_new_matched_tokens(seq)
    assert sched._lookup_client.calls == 1


def test_dense_steady_redundant_tier_stays_below_the_cut(monkeypatch):
    """The 46647 rate, at the measured density: 4 usable answers in 2048.

    Same side of a 10% cut as the trickle above, opposite side of a 1% one --
    which is the whole reason the cut moved.
    """
    sched = _skip_scheduler(monkeypatch, enabled=True)
    _feed_actionable_every(sched, every=512, lookups=2048)
    assert sched._lookup_actionable_ema < 0.01
    seq = _cached_seq(956, num_prompt_tokens=20000, num_cached_tokens=16000)

    assert sched.get_num_new_matched_tokens(seq) == (0, False)
    assert sched._lookup_client.calls == 0


def test_dense_forced_probe_samples_the_tier_the_gate_is_ignoring(monkeypatch):
    """An open gate must not blind the estimator that opens it.

    Every skipped request is one the estimator never hears about, so a gate
    fed only by issued lookups can never learn that the tier has become
    useful. One skippable request in `_LOOKUP_ACT_PROBE_EVERY` pays anyway.
    """
    sched = _skip_scheduler(monkeypatch, enabled=True)
    _measure_actionable(sched, False)
    every = _offload_common._LOOKUP_ACT_PROBE_EVERY

    for i in range(1, every):
        seq = _cached_seq(f"p{i}", num_prompt_tokens=20000, num_cached_tokens=16000)
        assert sched.get_num_new_matched_tokens(seq) == (0, False)
    assert sched._lookup_client.calls == 0

    seq = _cached_seq(f"p{every}", num_prompt_tokens=20000, num_cached_tokens=16000)
    sched.get_num_new_matched_tokens(seq)
    assert sched._lookup_client.calls == 1
