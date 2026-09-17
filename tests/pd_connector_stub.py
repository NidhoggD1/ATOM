# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise the real P/D ownership contract without a network backend."""

from unittest.mock import Mock

from atom.kv_transfer.disaggregation.pd_scheduler import PDSchedulerBase
from atom.kv_transfer.disaggregation.types import ConnectorMetadata


class PDSchedulerStub(PDSchedulerBase):
    is_producer = True

    def __init__(self):
        super().__init__()
        self.request_finished = Mock(wraps=self.request_finished)

    def get_num_new_matched_tokens(self, seq):
        return 0, False

    def build_connector_meta(self):
        return ConnectorMetadata()

    def update_state_after_alloc(self, seq):
        self._plan_send(seq)

    def request_finished(self, seq):
        self._publish_send(seq)
