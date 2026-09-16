# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.disaggregation.types import (
    KVTransferRegion,
    KVTransferTensors,
    LoadOperationId,
    SaveOperationId,
)
from atom.kv_transfer.offload.metadata import (
    LMCacheReqMeta,
    LoadSpec,
    NativeStateTransfer,
    SaveSpec,
)
from atom.kv_transfer.offload.mp.backend import _model_namespace, _tp_replication_factor
from atom.kv_transfer.offload.mp.dsv4_layout import build_dsv4_mp_layout
from atom.kv_transfer.offload.mp.dsv4_worker import (
    DSV4MPConnector,
    require_native_server,
)
from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec


class Future:
    def __init__(self, value=True, ready=False):
        self.value, self.ready = value, ready

    def query(self):
        return self.ready

    def result(self, timeout=0):
        return self.value


def config(**extra):
    return SimpleNamespace(
        kv_cache_block_size=4,
        tensor_parallel_size=2,
        hf_config=SimpleNamespace(model_type="deepseek_v4", kv_lora_rank=512),
        kv_transfer_config={
            "kv_connector": "lmcache_mp",
            "kv_role": "offload",
            "kv_connector_extra_config": extra,
        },
    )


@pytest.fixture
def worker():
    instance = DSV4MPConnector(config())
    page = torch.zeros((32, 1, 32), dtype=torch.uint8)
    spec = PagedStateCheckpointSpec(32, 128, "dsv4-paged-state-v3:test", 80)
    tensors = KVTransferTensors(
        block_regions=[
            KVTransferRegion(
                base_addr=page.data_ptr(), unit_bytes=32, total_bytes=page.numel()
            )
        ],
        slot_regions=[],
        block_tensor_views=[page],
        paged_state_checkpoint_spec=spec,
        execute_paged_state_copies=lambda *_: None,
    )
    tensors.set_block_count(32)
    instance._native_layout = build_dsv4_mp_layout(tensors, block_size=4, chunk_size=8)
    instance.chunk_size = 8
    instance.submitted = []
    instance.future = Future()

    def submit(request_id, op, event):
        instance.submitted.append(op)
        return instance.future

    instance._adapter = SimpleNamespace(
        submit_store_request=submit, submit_retrieve_request=submit
    )
    return instance


def request(*, loading=False, units=(0, 25, 31), generation=1):
    return LMCacheReqMeta(
        req_id=7,
        token_ids=list(range(16)),
        block_ids=[1, 2, 3, 4],
        load_spec=LoadSpec(0, 16) if loading else None,
        save_spec=None if loading else SaveSpec(0),
        save_operation=None if loading else SaveOperationId(7, generation),
        load_operation=LoadOperationId(7, generation) if loading else None,
        native_state=NativeStateTransfer(units, 16, 998, 2 if loading else None),
    )


def test_store_transmits_page_zero_as_real_native_unit(worker):
    req = request()
    worker._submit_save(req, object())
    assert worker.submitted[0].block_ids == [[1, 2, 3, 4], [-1, 0], [-1, 25], [-1, 31]]
    assert not worker.get_finished().connector_completions
    worker.future.ready = True
    finished = worker.get_finished()
    (completion,) = finished.connector_completions
    assert completion.operation_id == req.save_operation
    assert completion.succeeded
    assert not finished.finished_saving  # one quorum channel for the entire pair


@pytest.mark.parametrize(
    "units", [(0, -1, 31), (0, 31), (0, 0, 31), (0, 25, 32), (0, 2, 31)]
)
def test_invalid_native_source_fails_before_transport(worker, units):
    worker._submit_save(request(units=units), object())
    assert not worker.submitted
    (completion,) = worker.get_finished().connector_completions
    assert not completion.succeeded


def test_uncertain_remote_submission_retains_lease(worker):
    def uncertain(*_):
        raise ConnectionError("server may have received request")

    worker._adapter.submit_store_request = uncertain
    worker._submit_save(request(), object())
    assert not worker.get_finished().connector_completions
    assert len(worker._native_saves) == 1


def test_failed_retrieve_does_not_restore_or_report_success(worker, monkeypatch):
    monkeypatch.setattr(
        worker, "_begin_restore", lambda _: pytest.fail("unexpected restore")
    )
    worker.future.value, worker.future.ready = False, True
    req = request(loading=True)
    worker._submit_load(req, object())
    finished = worker.get_finished()
    assert finished.failed_loading == {req.load_operation}
    assert not finished.finished_loading


def test_load_completion_waits_for_native_restore(worker, monkeypatch):
    restore_event = Future(ready=False)
    restored = []

    def restore(pending):
        restored.append(pending.request.native_state.destination_slot)
        pending.restore_event = restore_event
        pending.restore_succeeded = True

    monkeypatch.setattr(worker, "_begin_restore", restore)
    req = request(loading=True)
    worker._submit_load(req, object())
    assert not worker.get_finished().finished_loading
    assert restored == []
    worker.future.ready = True
    assert not worker.get_finished().finished_loading
    assert restored == [2]
    restore_event.ready = True
    assert worker.get_finished().finished_loading == {req.load_operation}
    assert restored == [2]


def test_query_exception_never_releases_dma_source(worker):
    def broken():
        raise RuntimeError("IPC event unavailable")

    worker.future.query = broken
    worker._submit_save(request(), object())
    assert not worker.get_finished().connector_completions
    assert worker._native_saves


def test_worker_admission_bound_rejects_before_dma(worker):
    worker._max_pending_saves = 1
    worker._submit_save(request(), object())
    worker._submit_save(request(generation=2), object())
    assert len(worker.submitted) == 1
    (completion,) = worker.get_finished().connector_completions
    assert completion.operation_id == SaveOperationId(7, 2)
    assert not completion.succeeded


def test_exact_completed_generation_cannot_replay(worker):
    worker.future.ready = True
    worker._submit_save(request(), object())
    worker.get_finished()
    with pytest.raises(RuntimeError, match="duplicate"):
        worker._submit_save(request(), object())


def test_dsv4_cannot_collapse_tp_state():
    assert _tp_replication_factor(config()) == 1
    with pytest.raises(ValueError, match="every TP rank"):
        _tp_replication_factor(config(**{"lmcache.mp.tp_rank_collapse": True}))


def test_factory_selects_native_worker_and_scheduler_for_v3_schema():
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
    from atom.kv_transfer.offload.mp.dsv4_scheduler import DSV4MPConnectorScheduler

    cfg = config()
    cfg.hf_config.model_type = "deepseek_v3"
    cfg.hf_config.compress_ratios = [0, 4, 128]
    assert isinstance(
        KVConnectorFactory.create_connector(cfg, "worker"), DSV4MPConnector
    )
    assert isinstance(
        KVConnectorFactory.create_connector(cfg, "scheduler"), DSV4MPConnectorScheduler
    )


def test_native_server_chunk_mismatch_fails_before_registration(monkeypatch):
    from atom.kv_transfer.offload.mp import dsv4_worker

    monkeypatch.setattr(
        dsv4_worker.offcfg,
        "build_lmcache_config",
        lambda _: SimpleNamespace(chunk_size=256),
    )
    adapter = SimpleNamespace(lmcache_tokens_per_chunk=512)
    with pytest.raises(ValueError, match="must match"):
        require_native_server(adapter, config())


def test_native_namespace_changes_with_image_codec(monkeypatch):
    from atom.kv_transfer.offload.mp import backend

    monkeypatch.setattr(backend.offcfg, "build_lmcache_config", lambda _: object())
    monkeypatch.setattr(backend.offcfg, "lmcache_replica_world_size", lambda _: 2)
    monkeypatch.setattr(
        backend.offcfg, "build_page_namespace", lambda *_: "page-config"
    )
    first = PagedStateCheckpointSpec(32, 128, "dsv4-paged-state-v3:a", 80)
    second = replace(first, image_bytes=81)
    assert _model_namespace(config(), checkpoint_spec=first) != _model_namespace(
        config(), checkpoint_spec=second
    )
