# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""D1 (run-to-run) determinism runner for the offline ATOM engine.

Sends the same prompts to the same engine N times and asks whether the
completions come back bit-identical. Nothing about batching is exercised here:
every prompt goes out in its own ``generate()`` call, so batch size is 1 and
batch composition is identical on every repeat. That is the point -- D1 is the
control group for any later batch-invariance (D2) work. When repeats of an
identical batch already diverge, a batch-size-1-vs-8 comparison has no baseline
to be measured against.

Signal: ``token_ids`` plus per-step ``logprobs``. The logprob is
``log_softmax(logits.float())`` gathered at the sampled token
(``model_runner.py`` sampling path), so it is a function of the whole logits
row and needs no engine instrumentation to obtain.

Stage ladder (``--stage``), each step adding exactly one variable::

    d1.0   64-token prompt, 1 output token        no batching, no chunking
    d1.1   64-token prompt, 300 output tokens     multi-step decode
    d1.2   4096-token prompt, 300 output tokens   chunked prefill boundaries

d1.0 is the watershed: one request, one output token, one process. If that
diverges, the cause is inside a kernel and has nothing to do with scheduling.

Usage::

    python -m atom.utils.determinism.d1 --model /path/to/model -tp 4 \
        --kv_cache_dtype fp8 --stage d1.0 --out /tmp/d1_p0.json

Exit code is 0 only when every prompt is bit-identical across repeats.
See ``docs/determinism_testing.md``.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import asdict, fields

from atom.model_engine.arg_utils import EngineArgs
from atom.sampling_params import SamplingParams
from atom.utils.arg_parser import FlexibleArgumentParser
from atom.utils.determinism.compare import (
    RunSample,
    build_report,
    render_text,
    save_run,
)

# (prompt_tokens, max_tokens, max_num_batched_tokens override or None)
STAGES: dict[str, dict[str, int | None]] = {
    "d1.0": {"prompt_tokens": 64, "max_tokens": 1, "max_num_batched_tokens": None},
    "d1.1": {"prompt_tokens": 64, "max_tokens": 300, "max_num_batched_tokens": None},
    # Chunking is forced by shrinking the token budget rather than by sending a
    # >16k prompt -- same code path, a fraction of the compute.
    "d1.2": {"prompt_tokens": 4096, "max_tokens": 300, "max_num_batched_tokens": 2048},
}

# Env vars whose value changes kernel selection or collective behaviour. Any
# drift between processes invalidates an inter-process comparison, so they are
# recorded alongside the engine config.
ENV_PREFIXES = ("ATOM_", "AITER_", "HIP_", "HSA_", "AMD_", "V4_", "NCCL_", "RCCL_")

# Natural-language prompts, tiled/truncated to the stage's exact token count.
# Random token ids were the first thing tried and they overstate the problem:
# garbage input leaves the model with a near-uniform next-token distribution
# (measured top-1 probability ~0.001), where every candidate is a near-tie and
# the tiniest numeric difference flips the argmax. In-distribution text keeps
# the distribution peaked, which is the regime real serving runs in.
TEXT_CORPUS = [
    "Explain how a heat pump can move more energy than it consumes, and why "
    "that does not violate conservation of energy.",
    "Write a short summary of the causes of the 1929 stock market crash and "
    "the policy responses that followed it.",
    "What is the difference between a mutex and a semaphore? Give a concrete "
    "example where using the wrong one causes a bug.",
    "Describe the water cycle in detail, from evaporation through "
    "precipitation and runoff back to the ocean.",
    "A train leaves station A at 60 km/h and another leaves station B at "
    "90 km/h toward it. Walk through how to find where they meet.",
    "Compare supervised, unsupervised, and reinforcement learning, and name a "
    "task where each is clearly the right choice.",
    "Explain why the sky is blue at noon and red at sunset, referring to "
    "Rayleigh scattering and path length through the atmosphere.",
    "Outline the steps to debug a program that crashes only under high load "
    "and never in a local test run.",
]


def add_d1_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stage",
        choices=sorted(STAGES),
        default="d1.0",
        help="D1 ladder stage; sets prompt length / output length defaults",
    )
    parser.add_argument("--repeat", type=int, default=20, help="repeats per prompt")
    parser.add_argument("--num-prompts", type=int, default=8, help="distinct prompts")
    parser.add_argument(
        "--prompt-tokens", type=int, default=None, help="override stage prompt length"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, help="override stage output length"
    )
    parser.add_argument(
        "--prompt-source",
        choices=("text", "random"),
        default="text",
        help="text: natural prompts tiled to the stage token count (default). "
        "random: synthetic token ids -- reproducible, but out of distribution, "
        "so it exaggerates argmax instability.",
    )
    parser.add_argument("--seed", type=int, default=42, help="prompt-generation seed")
    parser.add_argument("--out", default="", help="write run JSON here")
    parser.add_argument(
        "--process-id", type=int, default=0, help="label only; set by the launcher"
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip the discarded warmup generate (keeps first-call compile "
        "and autotune inside the measured repeats)",
    )
    parser.add_argument(
        "--allow-prefix-cache",
        action="store_true",
        help="do NOT force prefix caching off. Repeats then hit the cache and "
        "take a different path, which makes this a cache-consistency (D2) "
        "test rather than a D1 one.",
    )
    parser.add_argument(
        "--allow-spec",
        action="store_true",
        help="do NOT refuse --method. Speculative decoding changes batch "
        "shapes step to step, adding a variable D1 is meant to exclude.",
    )


def _engine_default(name: str):
    return next(f for f in fields(EngineArgs) if f.name == name).default


def apply_d1_overrides(args: argparse.Namespace) -> list[str]:
    """Force the config D1 requires. Returns the notes to print and record."""
    notes: list[str] = []

    if getattr(args, "method", None) and not args.allow_spec:
        raise SystemExit(
            f"refusing to run D1 with speculative decoding (--method {args.method}): "
            "acceptance/rejection changes the batch shape every step. Pass "
            "--allow-spec to override."
        )

    if args.allow_prefix_cache:
        notes.append(
            "prefix caching LEFT ON by --allow-prefix-cache: repeat 2+ will hit "
            "the cache and skip prefill, so this is no longer a D1 measurement"
        )
    else:
        args.enable_prefix_caching = False
        notes.append("prefix caching forced OFF (else repeats hit the cache)")

    # Batch size is always 1 here; pinning the capture list keeps graph padding
    # from varying with anything else on the command line.
    args.cudagraph_capture_sizes = "[1]"
    notes.append("cudagraph_capture_sizes pinned to [1]")

    stage = STAGES[args.stage]
    if args.prompt_tokens is None:
        args.prompt_tokens = stage["prompt_tokens"]
    if args.max_tokens is None:
        args.max_tokens = stage["max_tokens"]

    mnbt = stage["max_num_batched_tokens"]
    if mnbt is not None:
        if args.max_num_batched_tokens == _engine_default("max_num_batched_tokens"):
            args.max_num_batched_tokens = mnbt
            notes.append(
                f"max_num_batched_tokens set to {mnbt} by stage {args.stage} "
                "(forces chunked prefill)"
            )
        else:
            notes.append(
                f"stage {args.stage} wanted max_num_batched_tokens={mnbt} but "
                f"{args.max_num_batched_tokens} was passed explicitly; keeping it"
            )
    return notes


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def _vocab_ceiling(model_path: str) -> int:
    """Upper bound for synthetic prompt token ids.

    Falls back to a value every supported tokenizer covers rather than failing
    the run -- random ids only need to be valid and reproducible.
    """
    try:
        return max(1000, min(int(_load_tokenizer(model_path).vocab_size), 50000))
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        print(f"[d1] could not read vocab size ({exc}); using 20000")
        return 20000


def build_prompts(
    num_prompts: int,
    prompt_tokens: int,
    seed: int,
    model_path: str,
    source: str = "text",
) -> list[list[int]]:
    """Prompts as raw token ids, exactly ``prompt_tokens`` long.

    Raw ids either way: this bypasses the tokenizer and chat template at
    request time, so neither can contribute a difference between repeats.
    """
    if source == "random":
        ceiling = _vocab_ceiling(model_path)
        rng = random.Random(seed)
        return [
            [rng.randint(100, ceiling - 1) for _ in range(prompt_tokens)]
            for _ in range(num_prompts)
        ]

    tokenizer = _load_tokenizer(model_path)
    prompts = []
    for i in range(num_prompts):
        text = TEXT_CORPUS[i % len(TEXT_CORPUS)]
        ids = list(tokenizer.encode(text))
        filler = list(tokenizer.encode(" " + text, add_special_tokens=False)) or ids
        while len(ids) < prompt_tokens:
            ids.extend(filler)
        prompts.append(ids[:prompt_tokens])
    return prompts


def collect_samples(llm, prompts, sampling_params, repeat, process_id, warmup=True):
    samples: list[RunSample] = []

    if warmup:
        t0 = time.time()
        llm.generate([prompts[0]], sampling_params)
        print(f"[d1] warmup done in {time.time() - t0:.1f}s (discarded)")

    for r in range(repeat):
        t0 = time.time()
        for i, tokens in enumerate(prompts):
            # One prompt per call: batch size is 1 and the batch composition is
            # identical on every repeat, which is what makes this D1 and not D2.
            out = llm.generate([tokens], sampling_params)[0]
            samples.append(
                RunSample(
                    prompt_id=f"p{i:02d}",
                    process=process_id,
                    repeat=r,
                    token_ids=list(out["token_ids"]),
                    logprobs=(
                        None if out.get("logprobs") is None else list(out["logprobs"])
                    ),
                    finish_reason=str(out.get("finish_reason", "")),
                )
            )
        print(f"[d1] repeat {r + 1}/{repeat} done in {time.time() - t0:.1f}s")
    return samples


def main() -> int:
    parser = FlexibleArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="ATOM D1 run-to-run determinism check (offline engine).",
    )
    EngineArgs.add_cli_args(parser)
    add_d1_args(parser)
    args = parser.parse_args()

    notes = apply_d1_overrides(args)
    print("=" * 70)
    print(f"ATOM D1 determinism check -- stage {args.stage}")
    print("=" * 70)
    for n in notes:
        print(f"[d1] {n}")
    print(
        f"[d1] {args.num_prompts} {args.prompt_source} prompt(s) x "
        f"{args.prompt_tokens} tokens, max_tokens={args.max_tokens}, "
        f"repeat={args.repeat}"
    )

    engine_args = EngineArgs.from_cli_args(args)
    llm = engine_args.create_engine()

    prompts = build_prompts(
        args.num_prompts,
        args.prompt_tokens,
        args.seed,
        args.model,
        source=args.prompt_source,
    )
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens, logprobs=True
    )

    started = time.time()
    try:
        samples = collect_samples(
            llm,
            prompts,
            sampling_params,
            args.repeat,
            args.process_id,
            warmup=not args.no_warmup,
        )
    finally:
        llm.close()

    metadata = {
        "stage": args.stage,
        "repeat": args.repeat,
        "num_prompts": args.num_prompts,
        "prompt_tokens": args.prompt_tokens,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "prompt_source": args.prompt_source,
        "notes": notes,
        "elapsed_s": round(time.time() - started, 1),
        "engine_args": asdict(engine_args),
        "env": {
            k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)
        },
    }

    report = build_report(samples, f"D1-intra process {args.process_id}", metadata)
    print()
    print(render_text(report))

    if args.out:
        save_run(args.out, samples, metadata)
        print(f"\n[d1] wrote {len(samples)} samples to {args.out}")

    return 0 if report.is_deterministic else 1


if __name__ == "__main__":
    sys.exit(main())
