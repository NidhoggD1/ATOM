# Standalone OpenTelemetry request tracing

This experimental, opt-in implementation propagates W3C trace context from the
standalone HTTP server into the engine scheduler and exports request phases over
OTLP/HTTP. It is disabled by default. Validation results, performance measurements and
trace examples are retained in [PR #2644](https://github.com/ROCm/ATOM/pull/2644).

## Enable in an isolated service

Install the optional dependencies in the same Python environment as ATOM:

```bash
python -m pip install -e '.[tracing]'
export ATOM_ENABLE_OTEL_TRACE=1
export OTEL_SERVICE_NAME=atom
export ATOM_TRACE_VERSION='<exact-ATOM-build-version>'
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT='http://127.0.0.1:43180/v1/traces'
```

Start the standalone server with its normal model arguments. The PoC uses an
environment switch; it does not add a `--enable-trace` command-line option.
The explicit endpoint must accept OTLP/HTTP protobuf traces, including the
`/v1/traces` path. Container deployments must pass these variables into the
container. Changing them requires restarting that isolated process.

For a local receiver and a request with known upstream context:

```bash
python scripts/observability/otlp_sink.py --port 43180 --output /tmp/atom-spans.jsonl
# In another terminal, with an isolated server already listening on port 30380:
curl http://127.0.0.1:30380/v1/completions \
  -H 'Content-Type: application/json' \
  -H 'traceparent: 00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01' \
  -H 'tracestate: nsdye=1' \
  -d '{"model":"atom-otel-poc","prompt":[100,200,300],"max_tokens":8,"temperature":0,"stream":true}'
```

Use a fresh receiver output path on each run. The receiver binds only to
loopback and is a test utility, not a production collector. The benchmark
client also accepts only loopback endpoints.

## Trace contract

```text
upstream span
└── atom.request                    HTTP lifetime, including SSE teardown
    └── engine.request              one per engine sequence
        ├── prefill_waiting         engine receipt → first actual forward
        ├── prefill_forward         first forward → final prompt-chunk postprocess
        └── decode_forward          prefill completion → engine termination
```

Only primitive carrier, model and request-ID values cross the engine IPC
boundary; SDK spans and contexts remain process-local. The API's public response
ID is `gen_ai.request.id`; `atom.sequence.id` distinguishes sibling sequences.
The HTTP span is a server span. The scheduler emits completed spans using the
recorded timestamps, once per sequence, rather than keeping SDK spans in the
scheduler loop. Stages are visible after completion, not as live progress.

These phases measure wall time and include scheduling, chunk interleaving,
preemption and applicable KV waits. They are not device-event measurements or
pure GPU execution time. Engine first retained token time is recorded separately
from final prompt-chunk completion. Single-output-token requests omit decode;
requests cancelled while waiting omit phases that never executed.

Attributes include the response model, normalized finish reason, prompt/output/
cached tokens, queue/first-token/prefill/decode/end-to-end latency and errors.
Prompt text, generated text and raw exception messages are not exported.
Normal SSE final-body completion is distinguished from early disconnect.

## Export and sampling behavior

With the flag off, the module does not import the OpenTelemetry SDK. With it on,
each emitting process owns a provider with `ALWAYS_ON` sampling, including when
the incoming parent is `sampled=0`. The original trace ID and tracestate are
preserved; newly recorded children may have the sampled bit set. This deliberate
policy allows a downstream collector to calculate full phase spanmetrics before
applying its own retention sampling.

Export uses `BatchSpanProcessor` with an 8192-span queue, up to 512 spans per
batch, a 1-second schedule delay and a 1-second HTTP exporter timeout. There is no
per-token export or new GPU synchronization. Telemetry initialization failures
warn and allow inference to continue; the SDK's bounded queue can drop spans
under pressure. This does not promise lossless tracing on collector failure.
API and engine shutdown hooks make a bounded best-effort flush.

A deployment that requires tracing must separately verify receiver-observed
spans. The environment flag alone is not an acceptance check. Collector endpoint,
network reachability, header forwarding through proxies, service/version
recognition, spanmetrics and retention rules need integration validation.

## Tests and benchmark tools

The three `tests/test_otel_*.py` modules cover context propagation, real scheduler
hooks, SSE lifecycle, errors/cancellation, sibling isolation and a spawned
process sending real OTLP to a loopback receiver. Install `.[tracing]` before
running them; otherwise tracing tests may skip for missing SDK dependencies. The CPU CI job
installs the tracing extra so these tests run there.

`bench_trace_overhead.py` generates immutable, independently namespaced prompts
per input-length/concurrency point and reuses the same manifest across arms.
It checks streaming termination, response ID consistency, token counts and usage,
records output hashes without text, and measures the complete closed-loop drain.
`check_traces.py` validates the successful **cold-input** benchmark's five-span
trees, public IDs, usage, zero cached tokens, phase continuity and dye/parent
propagation. Do not apply that cold-input check to a cache-hit experiment.

PD is explicitly marked unsupported by this PoC. Multi-GPU TP/DP, PP, MTP,
other models and additional HTTP APIs are not GPU-qualified by the MI325 validation recorded in the PR.
