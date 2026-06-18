"""Gate test for the seed runner wiring.

Exercises the seed runner end to end on a proven-good submission (the variable
convolutional stepper, at train_steps == 0): the runner
must emit a valid SubmissionRecord and a valid EvaluationRecord through the new
schema, with the evaluation resolving back to its submission and carrying the
validated-bit score and its capability map.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import cast

from leibniz.result_schema import EvaluationDocument, SubmissionDocument
from leibniz.seed_runner import SeedSubmission, run_seed_submission

_repository_root = Path(__file__).parents[1]
_program_root = _repository_root / "tests/fixtures/programs"
_benchmark_root_base = _repository_root / "src/leibniz/benchmarks"


def test_seed_runner_emits_submission_and_evaluation(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    seed = SeedSubmission(
        name="ks-variable-conv",
        program_path=_program_root / "ks_variable_conv.py",
        benchmark_root=_benchmark_root_base / "ks",
        train_steps=0,
        learning_rate=1e-3,
    )

    result = run_seed_submission(seed, results_root=results_root, tensor_device="cpu")

    # Both records were written into the new tiers.
    assert result.submission_path.is_file()
    assert result.evaluation_path.is_file()
    assert (results_root / "submissions").is_dir()
    assert (results_root / "benchmarks").is_dir()
    assert (results_root / "evaluations").is_dir()

    submission = SubmissionDocument.from_bytes(result.submission_path.read_bytes())
    evaluation = EvaluationDocument.from_bytes(result.evaluation_path.read_bytes()).evaluation

    # The evaluation resolves to the submission that was written, and carries the
    # validated-bit score backed by a capability map.
    assert str(evaluation.submission_id) == str(submission.submission.id)
    assert str(evaluation.lineage.submission_digest) == str(submission.digest)
    assert evaluation.validated_bits >= 0.0
    capability_map = evaluation.capability_map
    assert cast(float, capability_map["value"]) == evaluation.validated_bits
    assert isinstance(capability_map["root"], dict)


def test_seed_runner_scores_diverging_submission_at_boundary(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    seed = SeedSubmission(
        name="partial-dynamics",
        program_path=_program_root / "ks_partial_dynamics.py",
        benchmark_root=_benchmark_root_base / "ks",
        train_steps=0,
        learning_rate=1e-3,
    )

    result = run_seed_submission(seed, results_root=results_root, tensor_device="cpu")

    evaluation = EvaluationDocument.from_bytes(result.evaluation_path.read_bytes()).evaluation

    assert math.isfinite(evaluation.validated_bits)
    assert evaluation.validated_bits >= 0.0
    assert cast(float, evaluation.capability_map["value"]) == evaluation.validated_bits
    assert isinstance(evaluation.capability_map["root"], dict)
