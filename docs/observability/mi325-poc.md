# MI325 standalone tracing PoC validation

The opt-in instrumentation emitted real request phases on MI325X and completed a
same-host trace-OFF / trace-ON / trace-OFF comparison. This is a draft PoC result,
not a production qualification or a statistical overhead bound.

## Version and execution scope

The experiment ran on 2026-10-20 in Asia/Shanghai (2026-10-19 UTC) in an isolated
non-production workspace. No production service or collector was changed.

- ATOM base: `53e4fb52403707ab07b69ee909501ea3cc25ca5c`.
- Original measured patch SHA256: `e88877b8591bfe7647b93eb60fdfa646e0c918073a239a2f771e07699136c5d1`.
- The eight instrumented runtime files match the measured source hashes in
  [runtime provenance](artifacts/mi325-runtime-provenance.json). PR packaging
  relocates the CPU receiver helper into this repository and adds documentation;
  it does not change those measured runtime files.
- Public base image: `rocm/atom-dev@sha256:566aa3dc9d5d98a737f9f5c1b775a651e44dfd8e878ab77f2536a76df21f687e`.
  A local derived image installed the PoC source; that derived image is not a
  published distribution. PyTorch `2.10.0+rocm7.2.4`, OTel `1.40.0`, transformers
  `5.16.1`, AITER `948dd267f15df3879d2099c83002b1343136979e`.
- GPU 0 only, MI325X/gfx942, aggregate TP1. Qwen/Qwen3-8B BF16 weights, revision
  `b968826d9c46dd6066d109eabc6255188de91218`; BF16 KV cache.
- Compiler level 0, FULL HIP graphs; runtime logs confirmed capture of all six
  graph sizes `[1,2,4,8,16,32]`. Prefix caching and chunked prefill enabled.
- Maximum context 10240, sequences 32, scheduled tokens 4096,
  GPU memory utilization 0.5. Export was to a same-host loopback OTLP/HTTP sink.

The fixed engine arguments were:

```bash
python -m atom.entrypoints.openai_server \
  --model /model --served-model-name atom-otel-poc \
  --host 127.0.0.1 --server-port 30380 --port 30381 -tp 1 \
  --kv-cache-dtype bf16 --level 0 --cudagraph-mode FULL \
  --cudagraph-capture-sizes '[1,2,4,8,16,32]' \
  --max-model-len 10240 --max-num-seqs 32 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.5
```

## Functional validation

The measured ON arm passed complete trace checks for **320/320** timed requests:
exactly five spans, upstream → HTTP → engine → phases, matching public response
IDs and token usage, HTTP/engine lifetime containment, contiguous phase boundaries,
`tracestate: nsdye=1`, no duplicate spans and no erroneous ERROR on normal SSE
completion. API and engine exported from different processes.

GPU probes also passed cold/hot cache (0 versus 1008 cached tokens), incoming
`sampled=0`, invalid-model HTTP error and stream cancellation. The complete ON
run received 2456 spans including warmups/probes; ERROR trace IDs were confined
to the intended error/cancellation probes. See [probe results](artifacts/mi325-functional-probe.json).

The CPU regression was **137 passed** on the measured PoC and again after PR
packaging with the repository-local receiver helper, covering the new tracing tests,
scheduler metrics, MTP token limits, partial prefill/preemption and existing
request timing. This is CPU contract coverage, not GPU qualification of those
additional execution modes.

## Performance protocol

Each arm used a new container/model load with identical runtime source, image,
command and environment except `ATOM_ENABLE_OTEL_TRACE=0/1/0`. Each completed and
released GPU processes before the next arm. The same PoC patch was present in
all arms; the cost of its disabled branches relative to unpatched stock was not
measured.

Each point generated 128 output tokens with temperature 0 and `ignore_eos`.
Input was synthetic pretokenized token IDs, so tokenizer performance is excluded.
Independent seeds per length/concurrency point prevent cross-point prefix reuse;
all arms reused each point's exact manifest. C1 used 32 timed/4 warmup requests;
C8 used 64/16; C32 used 64/64, at both 2048 and 8192 input tokens. Total: 320 timed
and 168 warmup requests per arm, **960 timed requests with zero failures**.

Prefix-cache counters before/after each point confirmed expected request counts
including warmup and zero cached-token increments; ON trace attributes were
also checked. The Completion response omits cache details, so missing usage
fields were not treated as zero cache hits.

The client is closed-loop. Timing runs from first timed dispatch through the
last complete SSE drain, including ramp/drain and excluding warmup. TTFT is first
nonempty generated-text event; TPOT is the time between first and last such
events divided by output tokens minus one. This is not per-kernel decode time.
Failed points retain errors and do not produce performance metrics.

## Observed overhead

ON changes use the arithmetic mean of A1 OFF and A2 OFF. Negative throughput
change is a slowdown; positive latency change is an increase. OFF drift is A2
relative to A1. These are descriptive observations from one A/B/A sequence, with
no confidence interval; do not interpret them as a guaranteed overhead ceiling.

| Input | Concurrency | Output token/s A1 / ON / A2 | ON throughput | ON TTFT mean | ON TPOT mean | OFF throughput drift |
| --- | --- | --- | --- | --- | --- | --- |
| 2048 | 1 | 151.1 / 151.4 / 151.5 | +0.05% | -1.81% | +0.12% | +0.30% |
| 2048 | 8 | 790.0 / 791.7 / 796.8 | -0.21% | +0.49% | +0.14% | +0.86% |
| 2048 | 32 | 1420.3 / 1416.7 / 1420.6 | -0.27% | +0.76% | +0.03% | +0.02% |
| 8192 | 1 | 106.1 / 106.3 / 106.5 | -0.02% | -0.06% | +0.04% | +0.36% |
| 8192 | 8 | 344.4 / 344.2 / 344.7 | -0.09% | +0.16% | +0.04% | +0.10% |
| 8192 | 32 | 427.9 / 428.0 / 428.2 | -0.01% | -0.03% | +0.05% | +0.06% |

Across the points, ON versus mean OFF had TTFT p95 changes of -0.04% to +1.76%
and TPOT p95 changes of -0.04% to +2.46%. Full absolute values and per-arm
CPU/memory snapshots are in [the CSV](artifacts/mi325-qwen3-8b-metrics.csv);
[comparison JSON](artifacts/mi325-comparison.json) also includes per-metric drift.

Approximate container CPU averages were 2.04–2.15 cores. ON `memory.current` was
36.6–40.2 MiB above paired OFF means. Container accounting includes the benchmark
client and, for ON, the local OTLP sink; memory includes cgroup-accounted cache.
These are not isolated SDK/engine resource costs.

## High-concurrency output differences remain unisolated

All C1 output hashes were identical across A1/B1/A2. Greedy high-concurrency
outputs were not universally bitwise stable:

| Input / concurrency | A1 vs A2 | ON vs A1 | ON vs A2 |
| --- | --- | --- | --- |
| 2048 / 8 | 6 / 64 | 5 / 64 | 5 / 64 |
| 2048 / 32 | 0 / 64 | 1 / 64 | 1 / 64 |
| All other points | 0 | 0 | 0 |

The C32 mismatch was request index 32. A supplemental fresh-container pair,
B2 ON then A3 OFF, repeated only 2K/C32 with 64 timed and 64 warmup requests per
arm. Both had zero failures/cache hits; B2's 64 complete trace trees and GPU
functional probes passed. B2 matched A1/A2 for every output, while A3 OFF matched
B1 for every output, including index 32. Thus the differing output also occurred
with tracing disabled. B2/A3 throughput was 1418.7/1420.4 output token/s.

The original A/B/A remains unchanged; these 128 supplemental requests do not
replace it or enter its means. [Pairwise hashes and metrics](artifacts/mi325-supplemental-output-check.json)
preserve the result. Batch/scheduling-dependent floating-point behavior is a
possible explanation, not an established root cause. This does not prove universal
bitwise equivalence, establish model quality, or rule out all tracing-related
behavior changes. Target-model correctness validation remains required.

## Sample and reproduction

[The five-span sample](artifacts/mi325-sample-trace.json) is a real accepted
8K/C32 request; only resource instance IDs were anonymized. Its HTTP duration is
9528.180 ms, with 7104.521 ms queue, 236.116 ms prefill and 2169.768 ms decode.
These scheduler wall-time phases are not pure GPU compute. The
[Chrome trace](artifacts/mi325-sample-chrome-trace.json) can be loaded in Perfetto.

See [setup and tools](README.md). For example, run each point with a fresh output,
a point-specific manifest shared between arms, and a distinct arm label:

```bash
python scripts/observability/bench_trace_overhead.py \
  --model atom-otel-poc --arm B1 --input-tokens 2048 --output-tokens 128 \
  --concurrency 8 --requests 64 --warmup 16 \
  --manifest /tmp/atom-otel/manifests/i2048-c8.json \
  --output /tmp/atom-otel/B1-i2048-c8.json
python scripts/observability/check_traces.py \
  --spans /tmp/atom-spans.jsonl --benchmark /tmp/atom-otel/B1-i2048-c8.json \
  --output /tmp/atom-otel/trace-check.json
```

Wait for batched export before checking. Reproduce the full six-point sequence
and OFF/ON/OFF restarts, not just the example point, to compare with this report.
The public artifacts contain synthetic aggregate results and one trace, not
model weights, prompt/output text, SSH details or internal run ledgers.

## Retained failures and unverified scope

An initial runtime failed before readiness because its older AITER lacked
`topk_select`; that is not a tracing failure. An initial SSE smoke exposed false
cancellation on normal final-body completion; the measured r02 code fixes it
and adds a regression test. An early OFF baseline reused prompts across
concurrency points and was excluded; the reported sequence uses corrected,
independently namespaced manifests and verified cold-cache counters.

Not GPU-qualified here: MI350/MI355, other models, multi-GPU TP/DP, PP,
cross-node PD/KV transfer, MTP, other HTTP APIs, tokenizer performance, production
traffic/load patterns, proxy/router/gateway propagation, collector/Tempo
integration or failure/backpressure/queue-exhaustion behavior, very long requests,
or abrupt worker crashes. PD is explicitly marked unsupported in the PoC.
No production observability requirement should be marked satisfied based only
on the tracing flag or this test.
