# SPDX-License-Identifier: MIT
"""CPU tests for request context, phase boundaries, and lifecycle failures."""

import asyncio
import os
import pickle
from types import SimpleNamespace

import pytest

pytest.importorskip("opentelemetry.sdk.trace")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import StatusCode

from atom.observability import tracing

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
PARENT_ID = "00f067aa0ba902b7"


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_ON, shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_ENABLED", True)
    monkeypatch.setattr(tracing, "_provider", provider)
    monkeypatch.setattr(tracing, "_provider_pid", os.getpid())
    monkeypatch.setattr(tracing, "_init_failed_pid", None)
    monkeypatch.setattr(tracing, "_warned", False)
    yield exporter
    provider.shutdown()


@pytest.fixture
def clock(monkeypatch):
    now = [10_000_000_000]
    monkeypatch.setattr(tracing.time, "time_ns", lambda: now[0] + 100_000_000_000)
    monkeypatch.setattr(tracing.time, "perf_counter_ns", lambda: now[0])
    return now


def sequence(number=7, **kwargs):
    return SimpleNamespace(
        id=number,
        num_prompt_tokens=100,
        num_finalized_tokens=104,
        prefix_cache_hit_tokens=64,
        external_request_id=f"r-{number}",
        **kwargs,
    )


def scope(flags="01", headers=True):
    return {
        "type": "http",
        "method": "POST",
        "path": "/v1/completions",
        "headers": (
            [
                (b"traceparent", f"00-{TRACE_ID}-{PARENT_ID}-{flags}".encode()),
                (b"tracestate", b"othervendor=x,nsdye=1"),
            ]
            if headers
            else []
        ),
    }


async def no_receive():
    return {"type": "http.disconnect"}


async def no_send(message):
    pass


def execute(app, **kwargs):
    asyncio.run(
        tracing.TraceMiddleware(app, model=lambda: "test-model")(
            scope(**kwargs), no_receive, no_send
        )
    )


@pytest.mark.parametrize("flags", ["00", "01"])
def test_parent_dye_pickle_and_complete_phases(spans, clock, flags):
    async def app(scope, receive, send):
        seq = sequence()
        tracing.bind_sequences([seq])
        # This round trip is the real Sequence carrier representation, not
        # a live SDK span; engine state is created after socket receipt.
        seq = pickle.loads(pickle.dumps(seq))
        tracing.enqueue(seq)
        clock[0] += 1_000_000_000
        tracing.enqueue(seq)  # scheduler admission must not reset receipt
        tracing.forward(seq, is_decode=False)
        clock[0] += 2_000_000_000
        tracing.prefill_complete(seq)
        tracing.first_token(seq)
        clock[0] += 3_000_000_000
        tracing.finish(seq, reason="max_tokens", num_tokens=104)
        tracing.finish(seq, reason="max_tokens", num_tokens=104)
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    execute(app, flags=flags)
    result = {s.name: s for s in spans.get_finished_spans()}
    assert len(spans.get_finished_spans()) == 5
    assert set(result) == {
        "atom.request",
        "engine.request",
        "prefill_waiting",
        "prefill_forward",
        "decode_forward",
    }
    assert result["atom.request"].parent.span_id == int(PARENT_ID, 16)
    for span in result.values():
        assert span.context.trace_id == int(TRACE_ID, 16)
        assert span.context.trace_state.get("nsdye") == "1"
        assert span.status.status_code != StatusCode.ERROR
    root = result["engine.request"]
    assert root.parent.span_id == result["atom.request"].context.span_id
    for name, seconds in [
        ("prefill_waiting", 1),
        ("prefill_forward", 2),
        ("decode_forward", 3),
    ]:
        span = result[name]
        assert span.parent.span_id == root.context.span_id
        assert span.end_time - span.start_time == seconds * 1e9
    assert root.attributes["gen_ai.usage.cached_tokens"] == 64
    assert root.attributes["gen_ai.usage.completion_tokens"] == 4
    assert root.attributes["gen_ai.latency.time_to_first_token"] == 3
    assert tracing._request.get() is None


def test_queue_abort_exports_no_fabricated_execution(spans, clock):
    async def app(scope, receive, send):
        seq = sequence()
        tracing.bind_sequences([seq])
        tracing.enqueue(seq)
        clock[0] += 1_000_000_000
        tracing.finish(seq, reason="aborted", num_tokens=100)
        await receive()

    execute(app)
    result = spans.get_finished_spans()
    assert {s.name for s in result} == {
        "atom.request",
        "engine.request",
        "prefill_waiting",
    }
    assert all(s.status.status_code == StatusCode.ERROR for s in result)


def test_chunked_prefill_first_token_is_not_prompt_completion(spans, clock):
    async def app(scope, receive, send):
        seq = sequence()
        tracing.bind_sequences([seq])
        tracing.enqueue(seq)
        tracing.forward(seq, is_decode=False)
        clock[0] += 2_000_000_000
        tracing.forward(seq, is_decode=False)  # another chunk keeps first dispatch
        clock[0] += 2_000_000_000
        tracing.prefill_complete(seq)
        clock[0] += 1_000_000_000
        tracing.first_token(seq)  # deferred output delivered one step later
        clock[0] += 1_000_000_000
        tracing.finish(seq, reason="stop_sequence", num_tokens=104)

    execute(app)
    result = {s.name: s for s in spans.get_finished_spans()}
    assert (
        result["prefill_forward"].end_time - result["prefill_forward"].start_time == 4e9
    )
    assert (
        result["engine.request"].attributes["gen_ai.latency.time_to_first_token"] == 5
    )
    assert (
        result["decode_forward"].end_time - result["decode_forward"].start_time == 2e9
    )
    assert result["engine.request"].status.status_code != StatusCode.ERROR


def test_one_token_request_has_no_decode(spans, clock):
    async def app(scope, receive, send):
        seq = sequence()
        tracing.bind_sequences([seq])
        tracing.enqueue(seq)
        tracing.forward(seq, is_decode=False)
        clock[0] += 1_000_000_000
        tracing.prefill_complete(seq)
        tracing.first_token(seq)
        clock[0] += 1
        tracing.finish(seq, reason="eos", num_tokens=101)

    execute(app)
    assert "decode_forward" not in {s.name for s in spans.get_finished_spans()}


def test_http_span_lives_through_last_sse_body(spans):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        assert not spans.get_finished_spans()
        await send(
            {"type": "http.response.body", "body": b"data: x\n\n", "more_body": True}
        )
        await asyncio.sleep(0)
        assert not spans.get_finished_spans()
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    execute(app)
    assert len(spans.get_finished_spans()) == 1


def test_exception_keeps_only_type_and_reraises(spans):
    async def app(scope, receive, send):
        raise ValueError("PRIVATE_PROMPT_DO_NOT_EXPORT")

    with pytest.raises(ValueError, match="PRIVATE_PROMPT"):
        execute(app)
    span = spans.get_finished_spans()[0]
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes["error.type"] == "ValueError"
    assert not span.events
    assert "PRIVATE_PROMPT" not in str(span.attributes)


def test_concurrent_requests_and_siblings_do_not_share_parent(spans, clock):
    async def app(scope, receive, send):
        seqs = [sequence(1), sequence(2)]
        await asyncio.sleep(0)
        tracing.bind_sequences(seqs)
        for seq in seqs:
            tracing.enqueue(seq)
            tracing.forward(seq, is_decode=False)
            tracing.finish(seq, reason="max_tokens", num_tokens=101)

    async def run():
        middleware = tracing.TraceMiddleware(app)
        await asyncio.gather(
            *(middleware(scope(), no_receive, no_send) for _ in range(8))
        )

    asyncio.run(run())
    result = spans.get_finished_spans()
    http = {s.context.span_id for s in result if s.name == "atom.request"}
    engines = [s for s in result if s.name == "engine.request"]
    assert len(http) == 8
    assert len(engines) == 16
    assert all(sum(s.parent.span_id == parent for s in engines) == 2 for parent in http)


@pytest.mark.parametrize(
    "header", [b"bad", b"00-00000000000000000000000000000000-0000000000000000-01"]
)
def test_invalid_parent_creates_root(spans, header):
    async def app(scope, receive, send):
        pass

    request_scope = scope()
    request_scope["headers"] = [(b"traceparent", header)]
    asyncio.run(tracing.TraceMiddleware(app)(request_scope, no_receive, no_send))
    span = spans.get_finished_spans()[0]
    assert span.parent is None
    assert span.context.trace_id != 0


def test_disabled_path_does_not_load_sdk(monkeypatch):
    monkeypatch.setattr(tracing, "_ENABLED", False)
    monkeypatch.setattr(
        tracing, "_tracer", lambda: pytest.fail("SDK touched with tracing disabled")
    )
    seq = sequence()

    async def app(scope, receive, send):
        tracing.bind_sequences([seq])
        tracing.enqueue(seq)
        tracing.forward(seq, is_decode=False)
        tracing.prefill_complete(seq)
        tracing.first_token(seq)
        tracing.finish(seq, reason="max_tokens")

    execute(app)
    assert not hasattr(seq, "_atom_trace_carrier")


def test_pd_is_marked_unsupported_not_misattributed(spans):
    async def app(scope, receive, send):
        seq = sequence(kv_transfer_params={"do_remote_prefill": True})
        tracing.bind_sequences([seq])
        assert not hasattr(seq, "_atom_trace_carrier")

    execute(app)
    assert spans.get_finished_spans()[0].attributes["atom.trace.unsupported"] == "pd"


def test_exporter_initialization_failure_preserves_request(monkeypatch):
    monkeypatch.setattr(tracing, "_ENABLED", True)
    monkeypatch.setattr(tracing, "_provider_pid", None)
    monkeypatch.setattr(tracing, "_init_failed_pid", None)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    called = []

    async def app(scope, receive, send):
        called.append(True)

    execute(app)
    execute(app)
    assert called == [True, True]
    assert tracing._init_failed_pid == os.getpid()


def test_cancelled_stream_ends_http_span_and_preserves_cancellation(spans):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        execute(app)
    result = spans.get_finished_spans()
    assert len(result) == 1
    assert result[0].attributes["error.type"] == "CancelledError"
    assert result[0].status.status_code == StatusCode.ERROR
    assert tracing._request.get() is None


def test_disconnect_after_final_sse_body_is_normal_completion(spans):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        await send(
            {
                "type": "http.response.body",
                "body": b"data: [DONE]\n\n",
                "more_body": False,
            }
        )
        assert (await receive())["type"] == "http.disconnect"

    execute(app)
    span = spans.get_finished_spans()[0]
    assert span.status.status_code != StatusCode.ERROR
    assert "error.type" not in span.attributes


def test_public_request_id_survives_add_carrier(spans, clock):
    async def app(scope, receive, send):
        tracing.set_request_id("cmpl-public-id")
        seq = sequence()
        tracing.bind_sequences([seq])
        seq = pickle.loads(pickle.dumps(seq))
        tracing.enqueue(seq)
        clock[0] += 1
        tracing.finish(seq, reason="max_tokens", num_tokens=100)

    execute(app)
    assert all(
        s.attributes["gen_ai.request.id"] == "cmpl-public-id"
        for s in spans.get_finished_spans()
    )
