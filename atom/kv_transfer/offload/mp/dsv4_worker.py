# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native DSV4 checkpoint transport over the standalone LMCache process."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import torch

from atom.kv_transfer.disaggregation.types import ConnectorCompletion, KVConnectorOutput
from atom.kv_transfer.offload import config as offcfg
from atom.kv_transfer.offload._offload_common import max_pending_saves
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, NativeStateTransfer
from atom.kv_transfer.offload.mp.backend import (
    LMCacheMPConnector,
    _make_worker_adapter,
    _remember_operation_tombstone,
    _storage_kv_transfer_config,
    _terminal_future_result,
    _transfer_operation_id,
    _validate_mp_config,
)
from atom.kv_transfer.offload.mp.dsv4_layout import build_dsv4_mp_layout
from atom.model_engine.page_unit_checkpoint import CheckpointRestoreOp

logger = logging.getLogger("atom")
DSV4_MP_STORE_CHANNEL = "dsv4_mp_store"


def require_native_server(adapter: Any, config: Any = None) -> None:
    """Validate native transfer geometry shared with the LMCache server."""
    if config is not None:
        configured_chunk = int(
            offcfg.build_lmcache_config(_storage_kv_transfer_config(config)).chunk_size
        )
        if configured_chunk != int(adapter.lmcache_tokens_per_chunk):
            raise ValueError(
                "DSV4 LMCache configured chunk size must match the MP server: "
                f"configured={configured_chunk}, server={adapter.lmcache_tokens_per_chunk}"
            )


class _UncertainSubmission:
    """A transport exception cannot prove that a remote DMA stopped."""

    def query(self) -> bool:
        return False


@dataclass
class _NativePending:
    request: LMCacheReqMeta
    future: Any
    restore_event: Any = None
    restore_succeeded: bool = False


class DSV4MPConnector(LMCacheMPConnector):
    """Transfer pinned native images; never read a request's active SLOT to save."""

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        self._native_saves: dict[str, _NativePending] = {}
        self._native_loads: dict[str, _NativePending] = {}
        self._native_layout = None
        self._native_copy = None
        self._compute_stream = None
        self._max_pending_saves = max_pending_saves(
            int(os.environ.get("OFFLOAD_COPY_WORKERS", "1"))
        )

    def register_kv_caches(
        self,
        _kv_caches: dict[str, Any],
        transfer_tensors: Any = None,
        num_blocks: int | None = None,
    ) -> None:
        from aiter.dist.parallel_state import get_tp_group

        spec = getattr(transfer_tensors, "paged_state_checkpoint_spec", None)
        native_copy = getattr(transfer_tensors, "execute_paged_state_copies", None)
        if spec is None or not callable(native_copy):
            raise ValueError(
                "DSV4 MP needs native checkpoint geometry and copy callback"
            )
        _validate_mp_config(self._config)
        rank = int(get_tp_group().rank_in_group)
        adapter = _make_worker_adapter(self._config, rank, checkpoint_spec=spec)
        try:
            require_native_server(adapter, self._config)
            chunk_size = int(adapter.lmcache_tokens_per_chunk)
            layout = build_dsv4_mp_layout(
                transfer_tensors,
                block_size=self.block_size,
                chunk_size=chunk_size,
                num_blocks=num_blocks,
            )
            adapter.register_kv_caches(
                {f"dsv4.{i}": tensor for i, tensor in enumerate(layout.tensors)},
                engine_group_infos=layout.engine_group_infos(),
            )
        except Exception:
            adapter.shutdown()
            raise
        self._adapter = adapter
        self.chunk_size = chunk_size
        self._native_layout = layout
        self._native_copy = native_copy
        self._compute_stream = torch.cuda.current_stream()
        logger.info(
            "DSV4 MP registered rank=%d native_image=%d units=%d groups=%d chunk=%d",
            rank,
            spec.image_bytes,
            spec.units_per_checkpoint,
            len(layout.kernel_groups),
            chunk_size,
        )

    def _native_block_ids(
        self,
        req: LMCacheReqMeta,
        start: int,
        end: int,
        *,
        loading: bool,
    ) -> list[list[int]]:
        state = req.native_state
        if state is None:
            raise ValueError("DSV4 MP transfers require an exact native checkpoint")
        if start % self.chunk_size or end % self.chunk_size or start >= end:
            raise ValueError("DSV4 MP requires a nonempty chunk-aligned range")
        if state.boundary_tokens != end or len(req.token_ids) != end:
            raise ValueError("DSV4 native STATE and PAGE endpoints must match")
        if loading and (start != 0 or state.destination_slot is None):
            raise ValueError("DSV4 MP restore requires HBM=0 and a destination SLOT")
        if not loading and state.destination_slot is not None:
            raise ValueError("DSV4 MP save must refer to immutable PAGE units")
        self._native_layout.image_plan(state.unit_ids)
        pages = self._block_slice(req, start, end)
        if set(pages) & set(state.unit_ids):
            raise ValueError("checkpoint PAGE units must not overlap KV PAGE blocks")
        count = (end - start) // self.chunk_size
        # Earlier chunks have PAGE only. A boundary image is indivisible: all
        # its ordinal groups are present at exactly the same final chunk.
        return [pages] + [[-1] * (count - 1) + [unit] for unit in state.unit_ids]

    def _submit_native(self, req: LMCacheReqMeta, event: Any, *, loading: bool) -> None:
        from lmcache.integration.atom import AtomMPTransferSpec

        completion = req.load_operation if loading else req.save_operation
        if completion is None:
            raise ValueError("DSV4 MP transfers require exact operation generations")
        operation_id = _transfer_operation_id("load" if loading else "save", completion)
        pending = self._native_loads if loading else self._native_saves
        completed = (
            self._completed_load_operations
            if loading
            else self._completed_save_operations
        )
        with self._lock:
            if operation_id in pending or operation_id in completed:
                raise RuntimeError(f"duplicate DSV4 MP operation {operation_id!r}")
            if not loading and len(pending) >= self._max_pending_saves:
                # The scheduler has the same bound. Refuse before transport;
                # a terminal False safely returns the logical admission credit.
                pending[operation_id] = _NativePending(req, None)
                return
            end = len(req.token_ids)
            start = (
                req.load_spec.hbm_cached_tokens
                if loading
                else req.save_spec.skip_leading_tokens
            )
            try:
                groups = self._native_block_ids(req, start, end, loading=loading)
                spec = AtomMPTransferSpec(
                    token_ids=list(req.token_ids),
                    block_ids=groups,
                    start=start,
                    end=end,
                )
            except Exception:
                logger.exception("Invalid DSV4 MP descriptor %s", operation_id)
                pending[operation_id] = _NativePending(req, None)
                return
            entry = _NativePending(req, _UncertainSubmission())
            pending[operation_id] = entry
        submit = (
            self._adapter.submit_retrieve_request
            if loading
            else self._adapter.submit_store_request
        )
        try:
            future = submit(str(req.req_id), spec, event)
        except Exception:
            # Retain the exact source/destination lease. The server might have
            # received the request before the connection raised an exception.
            logger.exception(
                "DSV4 MP submission uncertain; retaining lease %s", operation_id
            )
            return
        with self._lock:
            entry.future = future

    def _submit_load(self, req: LMCacheReqMeta, event: Any) -> None:
        self._submit_native(req, event, loading=True)

    def _submit_save(self, req: LMCacheReqMeta, event: Any) -> None:
        self._submit_native(req, event, loading=False)

    def _begin_restore(self, pending: _NativePending) -> None:
        state: NativeStateTransfer = pending.request.native_state
        spec = self._native_layout.checkpoint_spec
        event = torch.cuda.Event()
        pending.restore_event = event
        with torch.cuda.stream(self._compute_stream):
            # The native callback shares its pinned descriptor with forward.
            # Its CPU writes must not race the previous descriptor's async H2D.
            self._compute_stream.synchronize()
            try:
                self._native_copy(
                    (),
                    (
                        CheckpointRestoreOp(
                            dst_slot=state.destination_slot,
                            unit_ids=state.unit_ids,
                            total_bytes=spec.image_bytes,
                            layout_id=spec.layout_id,
                        ),
                    ),
                )
                pending.restore_succeeded = True
            except Exception:
                logger.exception("DSV4 MP native restore failed")
            finally:
                event.record(self._compute_stream)
                # Also protect this descriptor from the next forward's host
                # rewrite. This fences only the local native gather, never MP.
                self._compute_stream.synchronize()

    def get_finished(self) -> KVConnectorOutput:
        output = KVConnectorOutput()
        with self._lock:
            for operation_id, pending in list(self._native_saves.items()):
                terminal, result = _terminal_future_result(pending.future)
                if not terminal:
                    continue
                output.connector_completions.add(
                    ConnectorCompletion(
                        DSV4_MP_STORE_CHANNEL,
                        pending.request.save_operation,
                        succeeded=result is True,
                    )
                )
                del self._native_saves[operation_id]
                _remember_operation_tombstone(
                    operation_id,
                    self._completed_save_operations,
                    self._completed_save_operation_order,
                )
            for operation_id, pending in list(self._native_loads.items()):
                if pending.restore_event is None:
                    terminal, result = _terminal_future_result(pending.future)
                    if not terminal:
                        continue
                    if result is True:
                        try:
                            self._begin_restore(pending)
                        except Exception:
                            logger.exception(
                                "DSV4 MP restore safety unknown; retaining lease"
                            )
                            pending.future = _UncertainSubmission()
                            pending.restore_event = None
                            continue
                if pending.restore_event is not None:
                    try:
                        if not pending.restore_event.query():
                            continue
                    except Exception:
                        logger.exception(
                            "DSV4 MP restore event safety unknown; retaining lease"
                        )
                        continue
                completion = pending.request.load_operation
                target = (
                    output.finished_loading
                    if pending.restore_succeeded
                    else output.failed_loading
                )
                target.add(completion)
                del self._native_loads[operation_id]
                _remember_operation_tombstone(
                    operation_id,
                    self._completed_load_operations,
                    self._completed_load_operation_order,
                )
        return output
