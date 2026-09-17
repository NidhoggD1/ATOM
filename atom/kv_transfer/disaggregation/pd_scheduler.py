# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Source allocation ownership shared by the P/D scheduler connectors."""

from dataclasses import dataclass
from typing import Any

from atom.kv_transfer.disaggregation.base import KVConnectorSchedulerBase
from atom.kv_transfer.disaggregation.types import KVConnectorOutput


@dataclass
class _SendSource:
    published: bool = False
    completed: bool = False


class PDSchedulerBase(KVConnectorSchedulerBase):
    """Retain advertised P/D sources until the backend reports send completion.

    Before request_finished publishes remote block addresses, a prefill plan
    can be cancelled by preemption. The peer cannot read that allocation yet.
    Once advertised, it must stay alive until the send completes. Keep early
    completion in the plan too: publishing must never re-arm a completed send.

    Current P/D protocols identify sends by request ID and publish once per
    request. Preemption is therefore allowed only before publication; a new
    protocol that permits concurrent/retried published sends needs generation
    IDs as well as its own ownership implementation.
    """

    def __init__(self) -> None:
        self._send_sources: dict[str, _SendSource] = {}

    def _plan_send(self, seq: Any) -> None:
        if self.is_producer and (seq.kv_transfer_params or {}).get("do_remote_decode"):
            self._send_sources.setdefault(str(seq.id), _SendSource())

    def _publish_send(self, seq: Any) -> bool:
        """Arm ownership immediately before handing source addresses to the peer."""
        if getattr(seq, "leave_reason", None) == "aborted":
            source = self._send_sources.get(str(seq.id))
            if source is not None and not source.published:
                self._send_sources.pop(str(seq.id))
            return False
        self._plan_send(seq)
        source = self._send_sources.get(str(seq.id))
        if source is None:
            return False
        source.published = True
        if source.completed:
            self._send_sources.pop(str(seq.id))
        return True

    def should_defer_free(self, seq: Any) -> bool:
        source = self._send_sources.get(str(seq.id))
        return source is not None and source.published and not source.completed

    def process_completions(self, output: KVConnectorOutput) -> KVConnectorOutput:
        accepted = set()
        for req_id in output.finished_sending:
            source = self._send_sources.get(str(req_id))
            if source is not None:
                source.completed = True
                accepted.add(req_id)
                if source.published:
                    self._send_sources.pop(str(req_id))
        output.finished_sending = accepted
        return output

    def source_blocks_released(self, seq: Any) -> None:
        self._send_sources.pop(str(seq.id), None)
