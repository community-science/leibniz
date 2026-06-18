"""Run seed submissions through train + evaluate, emitting the new result schema.

A seed submission is an ordinary program taken through the standard
``run_benchmark`` (train) and ``evaluate_benchmark_checkpoint`` (evaluate)
pipeline, so it produces a ``SubmissionRecord`` under ``submissions/`` and an
``EvaluationRecord`` under ``evaluations/`` exactly like any other submission.
Deterministic programs use ``train_steps == 0`` (there are no parameters to
fit); learned programs train for a small budget. The emitted records embed the
full program graph, so the seeded dataset is self-contained and does not depend
on the program source files.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from leibniz.benchmark_runner import (
    BenchmarkEvaluationPlan,
    BenchmarkRunPlan,
    evaluate_benchmark_checkpoint,
    run_benchmark,
)
from leibniz.documents import load_object_document
from leibniz.tensor_runtime import TensorRuntimeDevice

__all__ = [
    "SeedResult",
    "SeedRunnerError",
    "SeedSubmission",
    "run_seed_submission",
    "run_seed_submissions",
]


class SeedRunnerError(ValueError):
    """Raised when a seed submission cannot be run end to end."""


@dataclass(frozen=True, slots=True)
class SeedSubmission:
    """A program to seed, plus the benchmark and training budget to run it with."""

    name: str
    program_path: Path
    benchmark_root: Path
    train_steps: int = 0
    # A learning rate is required by the gradient optimizers even when
    # train_steps == 0 (it simply goes unused); deterministic programs have no
    # parameters to fit, so the value is immaterial for them.
    learning_rate: float | None = 1e-3
    seed: int = 101
    evaluation_half_width_threshold: float | None = None


@dataclass(frozen=True, slots=True)
class SeedResult:
    """Where a seed submission's emitted records landed."""

    name: str
    submission_path: Path
    evaluation_path: Path


def run_seed_submission(
    seed: SeedSubmission,
    *,
    results_root: Path,
    tensor_device: TensorRuntimeDevice = "auto",
) -> SeedResult:
    """Train and evaluate one seed submission into ``results_root``."""

    trains = seed.train_steps > 0
    summary = run_benchmark(
        BenchmarkRunPlan(
            program_path=seed.program_path,
            benchmark_root=seed.benchmark_root,
            results_root=results_root,
            seed=seed.seed,
            train_steps=seed.train_steps,
            gate_check_interval=1,
            model_checkpoint_gate_interval=1,
            tensor_device=tensor_device,
            # Deterministic, parameter-free programs have nothing for a gradient
            # optimizer to fit; the robust loss-search optimizer just checkpoints
            # them. Learned programs train with adam.
            optimizer="adam" if trains else "loss-search",
            learning_rate=seed.learning_rate if trains else None,
        )
    )
    checkpoint_path = _submission_selected_checkpoint_path(
        summary.training_summary_path,
        results_root=results_root,
    )
    evaluation = evaluate_benchmark_checkpoint(
        BenchmarkEvaluationPlan(
            checkpoint_artifact_path=checkpoint_path,
            benchmark_root=seed.benchmark_root,
            results_root=results_root,
            tensor_device=tensor_device,
            half_width_threshold=seed.evaluation_half_width_threshold,
        )
    )
    return SeedResult(
        name=seed.name,
        submission_path=summary.training_summary_path,
        evaluation_path=evaluation.evaluation_bundle_path,
    )


def run_seed_submissions(
    seeds: tuple[SeedSubmission, ...],
    *,
    results_root: Path,
    tensor_device: TensorRuntimeDevice = "auto",
) -> tuple[SeedResult, ...]:
    """Run every seed submission, returning where each one's records landed."""

    return tuple(
        run_seed_submission(seed, results_root=results_root, tensor_device=tensor_device)
        for seed in seeds
    )


def _submission_selected_checkpoint_path(
    submission_path: Path,
    *,
    results_root: Path,
) -> Path:
    record = load_object_document(
        submission_path.read_bytes(),
        description="submission record",
    )
    provenance = record.get("training_provenance")
    if not isinstance(provenance, Mapping):
        raise SeedRunnerError(f"{submission_path}: submission has no training_provenance")
    checkpoint = cast(Mapping[str, object], provenance).get("selected_model_checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise SeedRunnerError(
            f"{submission_path}: training_provenance has no selected_model_checkpoint"
        )
    record_path = cast(Mapping[str, object], checkpoint).get("record_path")
    if not isinstance(record_path, str) or not record_path:
        raise SeedRunnerError(
            f"{submission_path}: selected_model_checkpoint has no record_path"
        )
    path = Path(record_path)
    if path.is_absolute():
        return path
    if path.parts[:1] == (results_root.name,):
        return (results_root.parent / path).resolve()
    return results_root / path
