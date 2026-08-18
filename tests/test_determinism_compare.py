# SPDX-License-Identifier: MIT
# Unit tests for the D1 determinism metrics layer (atom.utils.determinism.compare).
#
# The metrics layer is deliberately free of torch and of any engine import, so
# these run on the CPU-only runner. What they guard is the reasoning encoded in
# the verdict: which conditions count as non-determinism, which count as a
# tooling problem, and where the first divergence is reported.

import json

import pytest

from atom.utils.determinism.compare import (
    RunSample,
    build_report,
    compare_repeats,
    diagnose,
    load_run,
    render_text,
    save_run,
)


def _sample(repeat, token_ids, logprobs=None, process=0, prompt_id="p00"):
    return RunSample(
        prompt_id=prompt_id,
        process=process,
        repeat=repeat,
        token_ids=list(token_ids),
        logprobs=None if logprobs is None else list(logprobs),
        finish_reason="length",
    )


def _identical(n, tokens=(1, 2, 3, 4), logprobs=(-0.1, -0.2, -0.3, -0.4)):
    return [_sample(i, tokens, logprobs) for i in range(n)]


class TestDeterministicCase:
    def test_identical_samples_are_deterministic(self):
        v = compare_repeats(_identical(5))
        assert v.unique_token_seqs == 1
        assert v.unique_logprob_seqs == 1
        assert v.first_token_divergence is None
        assert v.first_logprob_divergence is None
        assert v.max_abs_logprob_delta == 0.0
        assert v.is_deterministic

    def test_single_sample_does_not_false_alarm(self):
        # One sample can never disagree with itself; the verdict must not read
        # as "deterministic proven" nor as a failure.
        v = compare_repeats(_identical(1))
        assert v.n_samples == 1
        assert v.is_deterministic


class TestDivergenceLocation:
    def test_token_divergence_index_is_the_first_differing_step(self):
        samples = _identical(3)
        samples[2] = _sample(2, [1, 2, 9, 4], [-0.1, -0.2, -0.3, -0.4])
        v = compare_repeats(samples)
        assert v.unique_token_seqs == 2
        assert v.first_token_divergence == 2
        assert not v.is_deterministic

    def test_length_difference_reports_the_truncation_point(self):
        samples = _identical(2)
        samples[1] = _sample(1, [1, 2], [-0.1, -0.2])
        v = compare_repeats(samples)
        assert v.first_token_divergence == 2
        assert v.length_min_max == (2, 4)

    def test_logprob_divergence_is_caught_while_tokens_still_agree(self):
        # The interesting early signal: numerics have already moved but no
        # argmax has flipped yet. Tokens alone would call this deterministic.
        samples = _identical(2)
        samples[1] = _sample(1, [1, 2, 3, 4], [-0.1, -0.2, -0.30000001, -0.4])
        v = compare_repeats(samples)
        assert v.unique_token_seqs == 1
        assert v.unique_logprob_seqs == 2
        assert v.first_token_divergence is None
        assert v.first_logprob_divergence == 2
        assert v.max_abs_logprob_delta == pytest.approx(1e-8, rel=1e-3)
        assert not v.is_deterministic

    def test_logprob_comparison_is_exact_not_tolerant(self):
        # A one-ULP difference must fail. Tolerance here would hide exactly the
        # kind of drift that flips a near-tie argmax later in the sequence.
        base = -0.3
        samples = [
            _sample(0, [1, 2], [base, -0.4]),
            _sample(1, [1, 2], [base + 5e-17, -0.4]),
        ]
        v = compare_repeats(samples)
        assert v.unique_logprob_seqs == 2
        assert not v.is_deterministic


class TestAlignmentWarnings:
    def test_length_mismatch_warns_but_is_not_a_determinism_failure(self):
        # The scheduler's placeholder paths can append to seq.logprobs without
        # a matching token. That is a plumbing defect, not nondeterminism, and
        # must not be reported as one.
        samples = [
            _sample(0, [1, 2, 3], [-0.1, -0.2, -0.3, -0.4]),
            _sample(1, [1, 2, 3], [-0.1, -0.2, -0.3, -0.4]),
        ]
        v = compare_repeats(samples)
        assert v.alignment_warnings
        assert "len(logprobs)" in v.alignment_warnings[0]
        assert v.is_deterministic

    def test_absent_logprobs_warn_and_leave_the_token_verdict_intact(self):
        v = compare_repeats([_sample(i, [1, 2, 3]) for i in range(3)])
        assert v.unique_logprob_seqs is None
        assert v.is_deterministic
        assert any("logprobs" in w for w in v.alignment_warnings)

    def test_partial_logprob_coverage_is_flagged(self):
        samples = [_sample(0, [1, 2], [-0.1, -0.2]), _sample(1, [1, 2], None)]
        v = compare_repeats(samples)
        assert any("only 1/2" in w for w in v.alignment_warnings)


class TestReport:
    def test_report_groups_by_prompt_and_aggregates_verdicts(self):
        samples = [
            _sample(0, [1, 2], [-0.1, -0.2], prompt_id="p00"),
            _sample(1, [1, 2], [-0.1, -0.2], prompt_id="p00"),
            _sample(0, [3, 4], [-0.3, -0.4], prompt_id="p01"),
            _sample(1, [3, 5], [-0.3, -0.5], prompt_id="p01"),
        ]
        report = build_report(samples, "scope")
        assert report.n_prompts == 2
        assert report.n_dirty_prompts == 1
        assert not report.is_deterministic
        text = render_text(report)
        assert "p00" in text and "p01" in text
        assert "NON-DETERMINISTIC" in text

    def test_compare_repeats_rejects_mixed_prompts(self):
        with pytest.raises(ValueError):
            compare_repeats(
                [_sample(0, [1], prompt_id="a"), _sample(0, [1], prompt_id="b")]
            )


class TestDiagnosis:
    def test_dirty_intra_blames_the_kernel(self):
        dirty = build_report([_sample(0, [1, 2]), _sample(1, [1, 3])], "intra 0")
        assert "KERNEL-LEVEL" in diagnose([dirty], None)

    def test_clean_intra_dirty_inter_blames_the_process(self):
        p0 = [_sample(i, [1, 2], [-0.1, -0.2], process=0) for i in range(2)]
        p1 = [_sample(i, [1, 3], [-0.1, -0.3], process=1) for i in range(2)]
        intra = [build_report(p0, "intra 0"), build_report(p1, "intra 1")]
        inter = build_report(p0 + p1, "inter")
        assert all(r.is_deterministic for r in intra)
        assert "PROCESS-LEVEL" in diagnose(intra, inter)

    def test_all_clean_passes(self):
        p0 = [_sample(i, [1, 2], [-0.1, -0.2], process=0) for i in range(2)]
        p1 = [_sample(i, [1, 2], [-0.1, -0.2], process=1) for i in range(2)]
        intra = [build_report(p0, "intra 0"), build_report(p1, "intra 1")]
        assert "D1 PASS" in diagnose(intra, build_report(p0 + p1, "inter"))


class TestRoundTrip:
    def test_save_and_load_preserve_samples(self, tmp_path):
        samples = _identical(3)
        path = str(tmp_path / "run.json")
        save_run(path, samples, {"stage": "d1.0"})
        loaded, meta = load_run(path)
        assert meta["stage"] == "d1.0"
        assert loaded == samples

    def test_unknown_schema_is_rejected(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema": "nope", "samples": []}))
        with pytest.raises(ValueError):
            load_run(str(path))
