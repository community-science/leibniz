"""Active local benchmark loop orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from leibniz.benchmark_runner import BenchmarkRunPlan, BenchmarkRunSummary, run_benchmark
from leibniz.content import ContentDigest
from leibniz.documents import load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.local_results import LocalResultImportError, materialize_benchmark_result_views
from leibniz.proposal_generation import (
    ProposalGenerationPlan,
    ProposalGenerationSummary,
    generate_experiment_proposals,
)
from leibniz.work_queues import (
    WorkQueueItem,
    load_work_queue_items,
    materialize_work_queue_view,
    write_work_queue_item,
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
    runs_root: Path = Path(".runs")
    iterations: int = 1
    scale: int = 1
    candidate_budget: int = 3
    candidate_sample_count: int = 64
    sample_count: int = 4
    evaluation_sample_count: int | None = None
    seed: int = 101
    train_steps: int = 1
    learning_rate: float = 0.01
    optimizer: str = "sgd"
    schedule: str = "none"
    validation_interval: int = 1
    convergence_patience: int = 0
    convergence_min_delta: float = 0.0
    target_validation_loss: float | None = None
    dry_run: bool = False
    retry_failed: bool = False

    def __post_init__(self) -> None:
        if type(self.iterations) is not int or self.iterations < 1:
            raise ActiveTrainingLoopError("iterations must be positive")
        if type(self.scale) is not int or self.scale < 1:
            raise ActiveTrainingLoopError("scale must be positive")
        if type(self.candidate_budget) is not int or self.candidate_budget < 1:
            raise ActiveTrainingLoopError("candidate_budget must be positive")
        if type(self.candidate_sample_count) is not int or self.candidate_sample_count < 1:
            raise ActiveTrainingLoopError("candidate_sample_count must be positive")
        if self.candidate_sample_count < self.candidate_budget:
            raise ActiveTrainingLoopError(
                "candidate_sample_count must be at least candidate_budget"
            )
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
        if type(self.train_steps) is not int or self.train_steps < 0:
            raise ActiveTrainingLoopError("train_steps must be nonnegative")
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
        if self.target_validation_loss is not None and self.target_validation_loss < 0:
            raise ActiveTrainingLoopError("target_validation_loss must be nonnegative")
        if type(self.retry_failed) is not bool:
            raise ActiveTrainingLoopError("retry_failed must be boolean")


@dataclass(frozen=True, slots=True)
class ActiveTrainingLoopSummary:
    """Summary of one active benchmark loop invocation."""

    benchmark_id: ProtocolIdentifier
    iteration_count: int
    completed_run_count: int
    planned_commands: tuple[tuple[str, ...], ...]
    proposal_set_paths: tuple[Path, ...]
    measurement_dataset_paths: tuple[Path, ...]
    result_view_path: Path | None
    dry_run: bool

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "benchmark_id": str(self.benchmark_id),
            "iteration_count": self.iteration_count,
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
    """Generate proposals, run selected candidates, and refresh local result views."""

    proposal_summaries: list[ProposalGenerationSummary] = []
    benchmark_summaries: list[BenchmarkRunSummary] = []
    planned_commands: list[tuple[str, ...]] = []
    benchmark_id: ProtocolIdentifier | None = None
    result_view_path = _materialize_if_possible(plan.runs_root)
    materialize_work_queue_view(plan.runs_root)

    for iteration in range(plan.iterations):
        iteration_seed = plan.seed + iteration
        proposal_summary = generate_experiment_proposals(
            ProposalGenerationPlan(
                benchmark_root=plan.benchmark_root,
                runs_root=plan.runs_root,
                scale=plan.scale,
                candidate_budget=plan.candidate_budget,
                candidate_sample_count=plan.candidate_sample_count,
                sample_count=plan.sample_count,
                evaluation_sample_count=plan.evaluation_sample_count,
                seed=iteration_seed,
                train_steps=plan.train_steps,
                learning_rate=plan.learning_rate,
                optimizer=plan.optimizer,
                schedule=plan.schedule,
                validation_interval=plan.validation_interval,
                convergence_patience=plan.convergence_patience,
                convergence_min_delta=plan.convergence_min_delta,
                target_validation_loss=plan.target_validation_loss,
            )
        )
        benchmark_id = proposal_summary.benchmark_id
        proposal_summaries.append(proposal_summary)
        proposals = _proposal_records(proposal_summary.proposal_set_path)
        for proposal_index, proposal in enumerate(proposals):
            command = _proposal_command(proposal)
            planned_commands.append(command)
            sequence = iteration * len(proposals) + proposal_index
            queue_item = _queue_item_for_proposal(
                proposal=proposal,
                proposal_summary=proposal_summary,
                command=command,
                sequence=sequence,
            )
            existing_queue_item = _matching_queue_item(
                plan.runs_root,
                benchmark_id=proposal_summary.benchmark_id,
                command=command,
            )
            if existing_queue_item is not None:
                queue_item = replace(
                    existing_queue_item,
                    candidate_id=_proposal_string(proposal, "candidate_id"),
                    proposal_id=str(proposal["id"]),
                    proposal_set_path=proposal_summary.proposal_set_path,
                    sequence=sequence,
                )
            if queue_item.status == "completed":
                materialize_work_queue_view(plan.runs_root)
                continue
            if queue_item.status == "failed" and not plan.retry_failed:
                materialize_work_queue_view(plan.runs_root)
                raise ActiveTrainingLoopError(
                    "matching failed queue item requires --retry-failed"
                )
            if queue_item.status == "failed":
                queue_item = replace(queue_item, error=None, status="pending")
            if queue_item.status == "pending":
                write_work_queue_item(plan.runs_root, queue_item)
                materialize_work_queue_view(plan.runs_root)
            if plan.dry_run:
                continue

            architecture_path = _architecture_path_for_command(command)
            write_work_queue_item(
                plan.runs_root,
                replace(queue_item, status="reserved"),
            )
            materialize_work_queue_view(plan.runs_root)
            try:
                benchmark_summary = run_benchmark(
                    BenchmarkRunPlan(
                        architecture_path=architecture_path,
                        benchmark_root=plan.benchmark_root,
                        runs_root=plan.runs_root,
                        scale=plan.scale,
                        sample_count=plan.sample_count,
                        evaluation_sample_count=plan.evaluation_sample_count,
                        seed=iteration_seed,
                        train_steps=plan.train_steps,
                        learning_rate=plan.learning_rate,
                        optimizer=plan.optimizer,
                        schedule=plan.schedule,
                        validation_interval=plan.validation_interval,
                        convergence_patience=plan.convergence_patience,
                        convergence_min_delta=plan.convergence_min_delta,
                        target_validation_loss=plan.target_validation_loss,
                    )
                )
            except Exception as error:
                write_work_queue_item(
                    plan.runs_root,
                    replace(queue_item, error=str(error), status="failed"),
                )
                materialize_work_queue_view(plan.runs_root)
                raise
            write_work_queue_item(
                plan.runs_root,
                replace(
                    queue_item,
                    measurement_dataset_path=benchmark_summary.measurement_dataset_path,
                    run_id=benchmark_summary.run_slug,
                    status="completed",
                ),
            )
            benchmark_summaries.append(benchmark_summary)
            result_view_path = materialize_benchmark_result_views(
                repository_root=Path.cwd(),
                runs_root=plan.runs_root,
            ).view_file
            materialize_work_queue_view(plan.runs_root)

    if benchmark_id is None:
        raise ActiveTrainingLoopError("active loop did not generate proposals")
    return ActiveTrainingLoopSummary(
        benchmark_id=benchmark_id,
        iteration_count=plan.iterations,
        completed_run_count=len(benchmark_summaries),
        planned_commands=tuple(planned_commands),
        proposal_set_paths=tuple(summary.proposal_set_path for summary in proposal_summaries),
        measurement_dataset_paths=tuple(
            summary.measurement_dataset_path for summary in benchmark_summaries
        ),
        result_view_path=result_view_path,
        dry_run=plan.dry_run,
    )


def _queue_item_for_proposal(
    *,
    proposal: Mapping[str, object],
    proposal_summary: ProposalGenerationSummary,
    command: tuple[str, ...],
    sequence: int,
) -> WorkQueueItem:
    rank = proposal.get("rank", 1)
    command_digest = str(ContentDigest.from_value(list(command))).split(":", maxsplit=1)[1][:12]
    return WorkQueueItem(
        id=f"sequence-{sequence + 1}-rank-{rank}-{command_digest}",
        benchmark_id=proposal_summary.benchmark_id,
        proposal_id=str(proposal["id"]),
        candidate_id=_proposal_string(proposal, "candidate_id"),
        proposal_set_path=proposal_summary.proposal_set_path,
        command=command,
        status="pending",
        sequence=sequence,
    )


def _matching_queue_item(
    runs_root: Path,
    *,
    benchmark_id: ProtocolIdentifier,
    command: tuple[str, ...],
) -> WorkQueueItem | None:
    for item in load_work_queue_items(runs_root):
        if item.benchmark_id == benchmark_id and item.command == command:
            return item
    return None


def _proposal_string(proposal: Mapping[str, object], field: str) -> str:
    value = proposal.get(field)
    if not isinstance(value, str) or not value:
        raise ActiveTrainingLoopError(f"proposal does not declare {field}")
    return value


def _materialize_if_possible(runs_root: Path) -> Path | None:
    try:
        return materialize_benchmark_result_views(
            repository_root=Path.cwd(),
            runs_root=runs_root,
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
