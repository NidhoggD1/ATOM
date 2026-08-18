# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""D1 (run-to-run) determinism metrics. Pure Python: no torch, no GPU.

The runner (``atom.utils.determinism.d1``) produces one ``RunSample`` per
(prompt, repeat) and writes them as JSON. This module turns a pile of those
samples into a verdict.

The headline metric is a **unique count**, not a pairwise diff: N repeats of
one prompt collapse to ``unique_token_seqs`` / ``unique_logprob_seqs``, and
determinism means both are 1. A single integer per prompt is what makes this
usable as a CI gate later.

Two scopes matter and they have disjoint root causes, so the CLI reports both:

- **intra-process** -- repeats inside one engine process. Dirty here means
  kernel-level nondeterminism (atomic reductions, occupancy-dependent work
  splits, per-call autotune).
- **inter-process** -- samples pooled across separately launched processes.
  Clean intra + dirty inter isolates process-level causes: Triton autotune
  selection, RCCL algorithm/channel choice at init, allocator addresses.

Logprob equality is exact (no tolerance). ``max_abs_logprob_delta`` reports the
magnitude separately, because a large text divergence and a 1-ULP numeric
divergence look identical at the token level -- one flipped near-tie argmax is
enough to fork the rest of the sequence.

CLI::

    python -m atom.utils.determinism.compare run_p0.json run_p1.json ...

See ``docs/determinism_testing.md`` for the methodology.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

SCHEMA = "atom-determinism-run/1"


@dataclass(frozen=True)
class RunSample:
    """One prompt's completion from one repeat of one process."""

    prompt_id: str
    process: int
    repeat: int
    token_ids: list[int]
    logprobs: list[float] | None = None
    finish_reason: str = ""
    # Reserved for the raw-logits digest hook (not yet wired -- see
    # docs/determinism_testing.md "Follow-ups"). None until then.
    logits_digest: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "RunSample":
        return cls(
            prompt_id=str(d["prompt_id"]),
            process=int(d["process"]),
            repeat=int(d["repeat"]),
            token_ids=list(d["token_ids"]),
            logprobs=None if d.get("logprobs") is None else list(d["logprobs"]),
            finish_reason=d.get("finish_reason", ""),
            logits_digest=d.get("logits_digest"),
        )

    @property
    def label(self) -> str:
        return f"p{self.process}r{self.repeat}"


@dataclass(frozen=True)
class PromptVerdict:
    """Determinism verdict for one prompt across all of its samples."""

    prompt_id: str
    n_samples: int
    unique_token_seqs: int
    # None when no sample carried logprobs -- absence is not a failure.
    unique_logprob_seqs: int | None
    first_token_divergence: int | None
    first_logprob_divergence: int | None
    max_abs_logprob_delta: float
    length_min_max: tuple[int, int]
    # Tooling problems (e.g. logprobs/token_ids length mismatch), NOT
    # determinism failures. Kept out of `is_deterministic` on purpose.
    alignment_warnings: list[str] = field(default_factory=list)

    @property
    def is_deterministic(self) -> bool:
        if self.unique_token_seqs != 1:
            return False
        return self.unique_logprob_seqs in (None, 1)


@dataclass(frozen=True)
class D1Report:
    scope: str
    n_samples: int
    n_prompts: int
    processes: list[int]
    verdicts: list[PromptVerdict]
    metadata: dict = field(default_factory=dict)

    @property
    def is_deterministic(self) -> bool:
        return all(v.is_deterministic for v in self.verdicts)

    @property
    def n_dirty_prompts(self) -> int:
        return sum(1 for v in self.verdicts if not v.is_deterministic)


def _first_mismatch(ref: Sequence, other: Sequence) -> int | None:
    """Index of the first differing element, or of the truncation point."""
    for i, (a, b) in enumerate(zip(ref, other)):
        if a != b:
            return i
    if len(ref) != len(other):
        return min(len(ref), len(other))
    return None


def _min_opt(values: Iterable[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    return min(present) if present else None


def compare_repeats(samples: Sequence[RunSample]) -> PromptVerdict:
    """Collapse every sample of one prompt into a single verdict.

    Divergence indices are measured against ``samples[0]``, which is the
    reference by position only -- with a unique count of 1 there is no
    reference to speak of, and with more the choice does not change the
    verdict, only which index gets reported first.
    """
    if not samples:
        raise ValueError("compare_repeats() needs at least one sample")

    prompt_ids = {s.prompt_id for s in samples}
    if len(prompt_ids) != 1:
        raise ValueError(f"samples span multiple prompts: {sorted(prompt_ids)}")

    warnings: list[str] = []
    for s in samples:
        if s.logprobs is not None and len(s.logprobs) != len(s.token_ids):
            warnings.append(
                f"{s.label}: len(logprobs)={len(s.logprobs)} != "
                f"len(token_ids)={len(s.token_ids)}"
            )

    unique_tokens = len({tuple(s.token_ids) for s in samples})

    with_lp = [s for s in samples if s.logprobs is not None]
    if not with_lp:
        unique_lp: int | None = None
        warnings.append("no sample carried logprobs (pass logprobs=True)")
    else:
        unique_lp = len({tuple(s.logprobs) for s in with_lp})
        if len(with_lp) != len(samples):
            warnings.append(
                f"logprobs present on only {len(with_lp)}/{len(samples)} samples"
            )

    ref = samples[0]
    tok_div = _min_opt(_first_mismatch(ref.token_ids, s.token_ids) for s in samples[1:])

    lp_div: int | None = None
    max_delta = 0.0
    if ref.logprobs is not None:
        lp_div = _min_opt(
            _first_mismatch(ref.logprobs, s.logprobs) for s in with_lp[1:]
        )
        for s in with_lp[1:]:
            for a, b in zip(ref.logprobs, s.logprobs):
                delta = abs(a - b)
                if delta > max_delta:
                    max_delta = delta

    lengths = [len(s.token_ids) for s in samples]

    return PromptVerdict(
        prompt_id=ref.prompt_id,
        n_samples=len(samples),
        unique_token_seqs=unique_tokens,
        unique_logprob_seqs=unique_lp,
        first_token_divergence=tok_div,
        first_logprob_divergence=lp_div,
        max_abs_logprob_delta=max_delta,
        length_min_max=(min(lengths), max(lengths)),
        alignment_warnings=warnings,
    )


def build_report(
    samples: Sequence[RunSample],
    scope: str,
    metadata: dict | None = None,
) -> D1Report:
    """Group samples by prompt and produce one verdict per prompt."""
    by_prompt: dict[str, list[RunSample]] = {}
    for s in samples:
        by_prompt.setdefault(s.prompt_id, []).append(s)

    verdicts = [
        compare_repeats(sorted(v, key=lambda s: (s.process, s.repeat)))
        for _, v in sorted(by_prompt.items())
    ]
    return D1Report(
        scope=scope,
        n_samples=len(samples),
        n_prompts=len(by_prompt),
        processes=sorted({s.process for s in samples}),
        verdicts=verdicts,
        metadata=dict(metadata or {}),
    )


def render_text(report: D1Report) -> str:
    """Human-readable report. ASCII only -- this lands in CI logs."""
    lines: list[str] = []
    head = (
        f"[{report.scope}] {report.n_prompts} prompt(s), "
        f"{report.n_samples} sample(s), processes={report.processes}"
    )
    lines.append(head)
    lines.append("-" * len(head))
    lines.append(
        f"{'prompt':<14}{'uniq_tok':>9}{'uniq_lp':>9}"
        f"{'tok_div':>9}{'lp_div':>9}{'max_dlp':>12}  {'len':<12}verdict"
    )
    for v in report.verdicts:
        lp = "-" if v.unique_logprob_seqs is None else str(v.unique_logprob_seqs)
        td = "-" if v.first_token_divergence is None else str(v.first_token_divergence)
        ld = (
            "-"
            if v.first_logprob_divergence is None
            else str(v.first_logprob_divergence)
        )
        lo, hi = v.length_min_max
        length = f"{lo}" if lo == hi else f"{lo}..{hi}"
        lines.append(
            f"{v.prompt_id:<14}{v.unique_token_seqs:>9}{lp:>9}{td:>9}{ld:>9}"
            f"{v.max_abs_logprob_delta:>12.3e}  {length:<12}"
            f"{'PASS' if v.is_deterministic else 'FAIL'}"
        )

    warned = [(v.prompt_id, w) for v in report.verdicts for w in v.alignment_warnings]
    if warned:
        lines.append("")
        lines.append("tooling warnings (not determinism failures):")
        seen: set[str] = set()
        for pid, w in warned:
            msg = f"  {pid}: {w}"
            if msg not in seen:
                seen.add(msg)
                lines.append(msg)

    lines.append("")
    lines.append(
        f"VERDICT: {'DETERMINISTIC' if report.is_deterministic else 'NON-DETERMINISTIC'}"
        f" ({report.n_dirty_prompts}/{report.n_prompts} prompt(s) diverged)"
    )
    return "\n".join(lines)


def report_to_dict(report: D1Report) -> dict:
    return asdict(report)


def diagnose(intra: Sequence[D1Report], inter: D1Report | None) -> str:
    """One-line attribution from the intra/inter split.

    This is the whole point of running more than one process: which of the two
    scopes is dirty narrows the root-cause set before any bisecting starts.
    """
    dirty_intra = [r for r in intra if not r.is_deterministic]
    if dirty_intra:
        procs = ", ".join(str(r.processes[0]) for r in dirty_intra if r.processes)
        return (
            f"KERNEL-LEVEL: repeats diverge inside a single process "
            f"(process {procs}). Suspects: atomic reductions in fused MoE, "
            f"occupancy-dependent attention work splits, per-call autotune. "
            f"Nothing to do with batching -- inter-process results are moot "
            f"until this is fixed."
        )
    if inter is None or len(inter.processes) < 2:
        return "INTRA CLEAN: only one process sampled; run more to test D1-inter."
    if inter.is_deterministic:
        return "D1 PASS: bit-identical within and across processes."
    return (
        "PROCESS-LEVEL: every process is self-consistent but they disagree "
        "with each other. Suspects: Triton autotune selection, RCCL "
        "algorithm/channel choice at init, allocator addresses. Kernel "
        "atomics are ruled out."
    )


def save_run(path: str, samples: Sequence[RunSample], metadata: dict) -> None:
    payload = {
        "schema": SCHEMA,
        "metadata": metadata,
        "samples": [asdict(s) for s in samples],
    }
    with open(path, "w", encoding="utf-8") as fp:
        # default=str: metadata carries a raw EngineArgs snapshot, which may hold
        # enums/paths that json cannot encode. Config drift only needs equality.
        json.dump(payload, fp, indent=1, default=str)


def load_run(path: str) -> tuple[list[RunSample], dict]:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if payload.get("schema") != SCHEMA:
        raise ValueError(
            f"{path}: expected schema {SCHEMA}, got {payload.get('schema')}"
        )
    samples = [RunSample.from_dict(d) for d in payload["samples"]]
    return samples, payload.get("metadata", {})


def _reindex(samples: Sequence[RunSample], process: int) -> list[RunSample]:
    """Stamp a process id, so merged files never collide on (process, repeat)."""
    return [
        RunSample(
            prompt_id=s.prompt_id,
            process=process,
            repeat=s.repeat,
            token_ids=s.token_ids,
            logprobs=s.logprobs,
            finish_reason=s.finish_reason,
            logits_digest=s.logits_digest,
        )
        for s in samples
    ]


def _metadata_diff(metas: Sequence[dict]) -> list[str]:
    """Keys whose value is not identical across runs -- a config drift check.

    A dirty inter-process verdict means nothing if the two processes were not
    launched with the same engine config, so surface that first.
    """
    ignore = {"out", "started_at", "hostname", "pid", "elapsed_s"}
    keys = {k for m in metas for k in m} - ignore
    drifted = []
    for k in sorted(keys):
        values = {json.dumps(m.get(k), sort_keys=True, default=str) for m in metas}
        if len(values) > 1:
            drifted.append(f"{k}: {' vs '.join(sorted(values))}")
    return drifted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m atom.utils.determinism.compare",
        description="Merge D1 run JSONs and report intra/inter-process verdicts.",
    )
    parser.add_argument("runs", nargs="+", help="JSON files written by d1.py --out")
    parser.add_argument(
        "--json-out", default="", help="also write the merged report as JSON"
    )
    args = parser.parse_args(argv)

    all_samples: list[RunSample] = []
    metas: list[dict] = []
    intra: list[D1Report] = []
    for i, path in enumerate(args.runs):
        samples, meta = load_run(path)
        samples = _reindex(samples, i)
        metas.append(meta)
        all_samples.extend(samples)
        label = f"D1-intra process {i}" + (" (compile-cache cold)" if i == 0 else "")
        intra.append(build_report(samples, label, meta))

    drift = _metadata_diff(metas)
    if drift:
        print("WARNING: engine config differs across runs -- inter-process")
        print("         comparison is not a controlled experiment:")
        for d in drift:
            print(f"  {d}")
        print()

    for report in intra:
        print(render_text(report))
        print()

    inter: D1Report | None = None
    if len(args.runs) > 1:
        inter = build_report(all_samples, "D1-inter (all processes pooled)", metas[0])
        print(render_text(inter))
        print()
        warm = [s for s in all_samples if s.process > 0]
        if not inter.is_deterministic and len(args.runs) > 2:
            warm_report = build_report(warm, "D1-inter excluding process 0", metas[0])
            if warm_report.is_deterministic:
                print(
                    "NOTE: excluding process 0 (compile-cache cold) makes the "
                    "inter-process verdict clean -- the divergence is a "
                    "first-run compile artifact, not steady-state."
                )
                print()

    print(diagnose(intra, inter))

    if args.json_out:
        payload = {
            "intra": [report_to_dict(r) for r in intra],
            "inter": report_to_dict(inter) if inter else None,
            "diagnosis": diagnose(intra, inter),
        }
        with open(args.json_out, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=1)

    ok = all(r.is_deterministic for r in intra) and (
        inter is None or inter.is_deterministic
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
