# SPDX-License-Identifier: MIT
"""Run real scheduler admission/postprocess hooks without a GPU executor."""

# Pytest discovers these imported fixtures by their original names.
# ruff: noqa: F811

import asyncio
import pickle
from types import SimpleNamespace

from conftest import MockConfig
from test_otel_tracing import clock, execute, spans  # noqa: F401

from atom.model_engine.engine_core_mgr import CoreManager
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.observability import tracing
from atom.sampling_params import SamplingParams


def output(ids, tokens):
    return ScheduledBatchOutput(
        req_ids=ids,
        token_ids=[(token,) for token in tokens],
        num_rejected=None,
        num_bonus=None,
        draft_token_ids=None,
    )


def test_real_chunked_prefill_and_decode_hooks(spans, clock):
    async def app(scope, receive, send):
        scheduler = Scheduler(MockConfig(max_num_batched_tokens=4))
        seq = Sequence(
            list(range(10, 18)),
            block_size=4,
            sampling_params=SamplingParams(max_tokens=3, ignore_eos=True),
        )
        tracing.bind_sequences([seq])
        scheduler.add(seq)
        clock[0] += 1_000_000_000
        batch, seqs = scheduler.schedule()
        scheduler.metrics.record_forward(batch, seqs)
        clock[0] += 1_000_000_000
        scheduler.postprocess(list(scheduler.running), output([], []), batch=batch)
        assert seq.is_partial_prefill
        assert seq._atom_trace_timing.prefill_end_ns is None
        batch, seqs = scheduler.schedule()
        scheduler.metrics.record_forward(batch, seqs)
        clock[0] += 1_000_000_000
        scheduler.postprocess(
            list(scheduler.running), output([seq.id], [42]), batch=batch
        )
        assert not seq.is_partial_prefill
        assert seq._atom_trace_timing.prefill_end_ns is not None
        for _ in range(4):
            if scheduler.is_finished():
                break
            batch, seqs = scheduler.schedule()
            scheduler.metrics.record_forward(batch, seqs)
            clock[0] += 1_000_000_000
            scheduler.postprocess(
                list(scheduler.running), output([seq.id], [43]), batch=batch
            )
        assert scheduler.is_finished()

    execute(app)
    result = {s.name: s for s in spans.get_finished_spans()}
    assert set(result) == {
        "atom.request",
        "engine.request",
        "prefill_waiting",
        "prefill_forward",
        "decode_forward",
    }
    assert (
        result["prefill_waiting"].end_time - result["prefill_waiting"].start_time == 1e9
    )
    assert (
        result["prefill_forward"].end_time - result["prefill_forward"].start_time == 2e9
    )
    assert result["engine.request"].attributes["gen_ai.usage.completion_tokens"] == 3


def test_real_waiting_abort_closes_queue(spans, clock):
    async def app(scope, receive, send):
        scheduler = Scheduler(MockConfig())
        seq = Sequence([10, 11, 12, 13], block_size=4)
        tracing.bind_sequences([seq])
        scheduler.add(seq)
        assert scheduler.abort_request(seq.id)
        clock[0] += 1_000_000_000
        scheduler.schedule()

    execute(app)
    assert {s.name for s in spans.get_finished_spans()} == {
        "atom.request",
        "engine.request",
        "prefill_waiting",
    }


def test_core_manager_captures_context_after_executor_preprocess(spans):
    frames = []

    async def app(scope, receive, send):
        # Match the API: Sequence construction runs in an executor, while
        # CoreManager.add_request runs back on the HTTP event-loop thread.
        seq = await asyncio.get_running_loop().run_in_executor(
            None, lambda: Sequence([10, 11, 12, 13], block_size=4)
        )
        manager = SimpleNamespace(
            pp_size=1,
            _routable_engine_count=1,
            label="test",
            _seq_id_to_callback={},
            _send_request=lambda rank, payload: frames.append(payload),
        )
        CoreManager.add_request(manager, [seq])

    execute(app)
    _, seqs = pickle.loads(frames[0])
    http = spans.get_finished_spans()[0]
    assert (
        seqs[0]._atom_trace_carrier["traceparent"].split("-")[2]
        == f"{http.context.span_id:016x}"
    )
    assert seqs[0]._atom_trace_carrier["tracestate"] == "othervendor=x,nsdye=1"
