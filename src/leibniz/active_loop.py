"""Active local benchmark loop orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from leibniz.benchmark_runner import BenchmarkRunPlan, BenchmarkRunSummary, run_benchmark
from leibniz.documents import load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.local_results import LocalResultImportError, materialize_benchmark_result_views
from leibniz.proposal_generation import (
    ProposalGenerationPlan,
    generate_experiment_proposals,
)
from leibniz.tensor_runtime import (
    TensorRuntimeDevice,
    TensorRuntimeError,
    validate_tensor_runtime_device,
)

__all__ = [
    "ActiveTrainingLoopError",
    "ActiveTrainingLoopPlan",
    "ActiveTrainingLoopSummary",
    "run_active_training_loop",
]


class ActiveTrainingLoopError(ValueError):
    """Raised when the active benchmark loop cannot proceed."""


@dataclass(frozen=True, slots=True)
class ActiveTrainingLoopPlan:
    """Plan for a deterministic local active benchmark loop."""

    benchmark_root: Path
    results_root: Path = Path("results")
    candidate_sample_count: int = 64
    sample_count: int = 512
    evaluation_sample_count: int | None = None
    seed: int = 101
    train_steps: int | None = None
    learning_rate: float = 0.01
    optimizer: str = "sgd"
    schedule: str = "none"
    validation_interval: int = 250
    convergence_patience: int = 12
    convergence_min_delta: float = 1e-3
    convergence_min_steps: int = 500
    target_validation_loss: float | None = None
    tensor_device: TensorRuntimeDevice = "auto"
    dry_run: bool = False
    progress_callback: Callable[[BenchmarkRunSummary], None] | None = None

    def __post_init__(self) -> None:
        if type(self.candidate_sample_count) is not int or self.candidate_sample_count < 1:
            raise ActiveTrainingLoopError("candidate_sample_count must be positive")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise ActiveTrainingLoopError("sample_count must be positive")
        if (
            self.evaluation_sample_count is not None
            and (
                type(self.evaluation_sample_count) is not int
                or self.evaluation_sample_count < 1
            )
        ):
            raise ActiveTrainingLoopError("evaluation_sample_count must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ActiveTrainingLoopError("seed must be nonnegative")
        if self.train_steps is not None and (
            type(self.train_steps) is not int or self.train_steps < 0
        ):
            raise ActiveTrainingLoopError("train_steps must be nonnegative")
        if self.train_steps is None and self.schedule == "cosine":
            raise ActiveTrainingLoopError("cosine schedule requires train_steps")
        if (
            self.train_steps is None
            and self.convergence_patience == 0
            and self.target_validation_loss is None
        ):
            raise ActiveTrainingLoopError(
                "uncapped training requires convergence_patience or target_validation_loss"
            )
        if self.learning_rate <= 0:
            raise ActiveTrainingLoopError("learning_rate must be positive")
        if self.optimizer not in {"sgd", "adam", "adamw"}:
            raise ActiveTrainingLoopError(f"unsupported optimizer: {self.optimizer}")
        if self.schedule not in {"none", "cosine", "reduce-on-plateau"}:
            raise ActiveTrainingLoopError(f"unsupported schedule: {self.schedule}")
        if type(self.validation_interval) is not int or self.validation_interval < 1:
            raise ActiveTrainingLoopError("validation_interval must be positive")
        if type(self.convergence_patience) is not int or self.convergence_patience < 0:
            raise ActiveTrainingLoopError("convergence_patience must be nonnegative")
        if self.convergence_min_delta < 0:
            raise ActiveTrainingLoopError("convergence_min_delta must be nonnegative")
        if type(self.convergence_min_steps) is not int or self.convergence_min_steps < 0:
            raise ActiveTrainingLoopError("convergence_min_steps must be nonnegative")
        if self.target_validation_loss is not None and self.target_validation_loss < 0:
            raise ActiveTrainingLoopError("target_validation_loss must be nonnegative")
        try:
            validate_tensor_runtime_device(self.tensor_device)
        except TensorRuntimeError as error:
            raise ActiveTrainingLoopError(str(error)) from error


@dataclass(frozen=True, slots=True)
class ActiveTrainingLoopSummary:
    """Summary of one active benchmark loop invocation."""

    benchmark_id: ProtocolIdentifier
    completed_run_count: int
    planned_commands: tuple[tuple[str, ...], ...]
    proposal_set_paths: tuple[Path, ...]
    measurement_dataset_paths: tuple[Path, ...]
    result_view_path: Path | None
    dry_run: bool

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "benchmark_id": str(self.benchmark_id),
            "completed_run_count": self.completed_run_count,
            "planned_commands": [list(command) for command in self.planned_commands],
            "proposal_set_paths": [path.as_posix() for path in self.proposal_set_paths],
            "measurement_dataset_paths": [
                path.as_posix() for path in self.measurement_dataset_paths
            ],
            "dry_run": self.dry_run,
        }
        if self.result_view_path is not None:
            record["result_view_path"] = self.result_view_path.as_posix()
        return record


def run_active_training_loop(plan: ActiveTrainingLoopPlan) -> ActiveTrainingLoopSummary:
    """Generate one proposal, run one candidate, and refresh local result views."""

    result_view_path = _materialize_if_possible(plan.results_root)
    proposal_summary = generate_experiment_proposals(
        ProposalGenerationPlan(
            benchmark_root=plan.benchmark_root,
            results_root=plan.results_root,
            candidate_budget=1,
            candidate_sample_count=plan.candidate_sample_count,
            sample_count=plan.sample_count,
            evaluation_sample_count=plan.evaluation_sample_count,
            seed=plan.seed,
            train_steps=plan.train_steps,
            learning_rate=plan.learning_rate,
            optimizer=plan.optimizer,
            schedule=plan.schedule,
            validation_interval=plan.validation_interval,
            convergence_patience=plan.convergence_patience,
            convergence_min_delta=plan.convergence_min_delta,
            convergence_min_steps=plan.convergence_min_steps,
            target_validation_loss=plan.target_validation_loss,
            tensor_device=plan.tensor_device,
        )
    )
    proposal = _proposal_records(proposal_summary.proposal_set_path)[0]
    command = _proposal_command(proposal)
    if plan.dry_run:
        return ActiveTrainingLoopSummary(
            benchmark_id=proposal_summary.benchmark_id,
            completed_run_count=0,
            planned_commands=(command,),
            proposal_set_paths=(proposal_summary.proposal_set_path,),
            measurement_dataset_paths=(),
            result_view_path=result_view_path,
            dry_run=plan.dry_run,
        )

    architecture_path = _architecture_path_for_command(command)

    def refresh_progress(_summary: BenchmarkRunSummary) -> None:
        materialize_benchmark_result_views(
            repository_root=Path.cwd(),
            results_root=plan.results_root,
        )
        if plan.progress_callback is not None:
            plan.progress_callback(_summary)

    benchmark_summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=architecture_path,
            benchmark_root=plan.benchmark_root,
            results_root=plan.results_root,
            sample_count=plan.sample_count,
            evaluation_sample_count=plan.evaluation_sample_count,
            seed=plan.seed,
            train_steps=plan.train_steps,
            learning_rate=plan.learning_rate,
            optimizer=plan.optimizer,
            schedule=plan.schedule,
            validation_interval=plan.validation_interval,
            convergence_patience=plan.convergence_patience,
            convergence_min_delta=plan.convergence_min_delta,
            convergence_min_steps=plan.convergence_min_steps,
            target_validation_loss=plan.target_validation_loss,
            tensor_device=plan.tensor_device,
        ),
        progress_callback=refresh_progress,
    )
    result_view_path = materialize_benchmark_result_views(
        repository_root=Path.cwd(),
        results_root=plan.results_root,
    ).view_file
    return ActiveTrainingLoopSummary(
        benchmark_id=proposal_summary.benchmark_id,
        completed_run_count=1,
        planned_commands=(command,),
        proposal_set_paths=(proposal_summary.proposal_set_path,),
        measurement_dataset_paths=(benchmark_summary.measurement_dataset_path,),
        result_view_path=result_view_path,
        dry_run=plan.dry_run,
    )


def _materialize_if_possible(results_root: Path) -> Path | None:
    try:
        return materialize_benchmark_result_views(
            repository_root=Path.cwd(),
            results_root=results_root,
        ).view_file
    except LocalResultImportError as error:
        if "no benchmark result records found" in str(error):
            return None
        raise


def _proposal_records(path: Path) -> tuple[Mapping[str, object], ...]:
    record = load_object_document(path.read_bytes(), description="proposal set")
    proposals_value = record.get("proposals")
    if not isinstance(proposals_value, tuple | list) or not proposals_value:
        raise ActiveTrainingLoopError("proposal set does not contain proposals")
    proposals = tuple(cast(tuple[object, ...], proposals_value))
    if not all(isinstance(proposal, Mapping) for proposal in proposals):
        raise ActiveTrainingLoopError("proposal set contains malformed proposal")
    return cast(tuple[Mapping[str, object], ...], proposals)


def _proposal_command(proposal: Mapping[str, object]) -> tuple[str, ...]:
    command_value = proposal.get("command")
    if not isinstance(command_value, tuple | list) or not command_value:
        raise ActiveTrainingLoopError("proposal does not declare a command")
    command = tuple(cast(tuple[object, ...], command_value))
    if not all(isinstance(argument, str) and argument for argument in command):
        raise ActiveTrainingLoopError("proposal command must contain nonempty strings")
    return cast(tuple[str, ...], command)


def _architecture_path_for_command(command: tuple[str, ...]) -> Path:
    try:
        index = command.index("--architecture")
    except ValueError as error:
        raise ActiveTrainingLoopError("proposal command does not include --architecture") from error
    try:
        return Path(command[index + 1])
    except IndexError as error:
        raise ActiveTrainingLoopError("proposal command omits architecture path") from error
