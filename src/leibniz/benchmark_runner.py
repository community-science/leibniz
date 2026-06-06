"""Small benchmark execution workflows for local operator runs."""

from __future__ import annotations

import hashlib
import math
import secrets
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any, Protocol, cast

from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.artifacts import ArtifactReference, reference_for_record
from leibniz.benchmark_evaluation import (
    CompetencePoint,
    ValidationCompetencePoint,
    finite_measurements_for_predictions,
    sampled_competence_curriculum_record,
    sampled_competence_frontier_score,
    sampled_competence_record,
    validation_competence_frontier_advances,
)
from leibniz.benchmark_implementations import Generator as BenchmarkGenerator
from leibniz.competition_bundles import BenchmarkCompetitionBundle
from leibniz.content import ContentDigest
from leibniz.documents import (
    canonical_document_bytes,
    document_filename_suffix,
    load_object_document,
)
from leibniz.evaluation_bundles import (
    BenchmarkEvaluationBundle,
    BenchmarkEvaluationBundleDocument,
)
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import AxisAssignment
from leibniz.measurements import MeasurementDataset
from leibniz.model_inspection import ModelInspectionRecord
from leibniz.model_interfaces import ModelInterface
from leibniz.model_manifests import (
    ModelArtifactManifest,
    ModelArtifactManifestDocument,
    ModelExecutionFamily,
)
from leibniz.model_operators import ExecutableModelOperator, summarize_architecture_operators
from leibniz.observation_generation import (
    GeneratedSample,
    GeneratedSampleSet,
    StateSpaceCandidate,
    StateSpaceMeasureRequest,
    StateSpaceMeasureValue,
    load_generator,
)
from leibniz.outcomes import OutcomeSpace
from leibniz.records import RecordExtractor
from leibniz.tensor_runtime import (
    OperationFallbackSequential,
    TensorRuntime,
    TensorRuntimeDevice,
    TensorRuntimeDeviceKind,
    TensorRuntimeError,
    apply_softmax_predictions,
    build_cosine_lr_schedule,
    build_cross_entropy_loss,
    build_optimizer,
    build_plateau_lr_schedule,
    load_tensor_runtime_state,
    make_float_tensor,
    make_long_tensor,
    no_grad_context,
    resolve_tensor_runtime,
    runtime_roofline_record,
    save_tensor_runtime_state,
    seed_runtime,
    tensor_runtime_device_kinds,
    validate_tensor_runtime_device,
)
from leibniz.timing import TimingCollector
from leibniz.training_runs import TrainingHistoryPoint, TrainingProtocol, TrainingRunRecord
from leibniz.views import MeasurementScoreView

__all__ = [
    "BenchmarkRunnerError",
    "BenchmarkEvaluationPlan",
    "BenchmarkEvaluationSummary",
    "BenchmarkCompetitionPlan",
    "BenchmarkCompetitionSummary",
    "BenchmarkRunPlan",
    "BenchmarkRunSummary",
    "CheckpointModelPredictor",
    "evaluate_benchmark_checkpoint",
    "compete_benchmark_checkpoints",
    "evaluate_model_checkpoint_artifact",
    "generate_model_checkpoint_competition_record",
    "load_model_checkpoint_artifact",
    "load_model_checkpoint_predictor",
    "ModelCheckpointArtifact",
    "has_windowed_validation_plateau",
    "run_benchmark",
    "training_stage_converged",
]

_document_suffix = document_filename_suffix()
_progress_format = "leibniz.benchmark-training-progress"
_progress_format_version = 1
_default_sample_count = 512
_default_train_steps: int | None = None
_default_gate_check_interval = 32
_default_model_checkpoint_gate_interval = 1
_default_convergence_patience = 6
_default_convergence_min_delta = 1e-3
_default_convergence_min_steps = 500
_converged_training_stage_stop_reasons = frozenset({"validation-plateau"})
_minimum_plateau_lr_reductions = 3
_state_space_target_spacing = 1.0
_full_variation_extent = 1.0


class _FieldBenchmarkGenerator(BenchmarkGenerator, Protocol):
    """Internal contract for tensor-backed benchmark training."""

    def minimum_state_space_measure(self) -> StateSpaceMeasureValue: ...

    def state_space_for_request(
        self,
        *,
        request: StateSpaceMeasureRequest,
    ) -> StateSpaceCandidate | None: ...

    def state_spaces_for_request(
        self,
        *,
        request: StateSpaceMeasureRequest,
    ) -> Sequence[StateSpaceCandidate]: ...

    def __call__(
        self,
        *,
        seed: int,
        shape: int | Sequence[int] | None = None,
        include_fields: bool = False,
        include_metadata: bool = True,
        state_space_request: StateSpaceMeasureRequest | None = None,
        component_indices: Iterable[int] | None = None,
        memory_limit_bytes: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
        variation_extent: float = 1.0,
        runtime: TensorRuntime | None = None,
        outcome_ids: tuple[str, ...] | None = None,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedSampleSet: ...


@dataclass(frozen=True, slots=True)
class _TrainingResult:
    evaluation_results: tuple[
        tuple[_CurriculumRung, tuple[tuple[float, ...], ...]],
        ...,
    ]
    training_rungs: tuple[_CurriculumRung, ...]
    training_frontier_index: int
    training_run: TrainingRunRecord
    throughput: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _EvaluationInput:
    architecture: ArchitectureManifest
    checkpoint: ModelCheckpointArtifact
    run_slug: str
    benchmark_id: ProtocolIdentifier
    evaluation_sample_count: int
    evaluation_rung_count: int
    training_compute: float | None


@dataclass(frozen=True, slots=True)
class ModelCheckpointArtifact:
    """A saved model checkpoint artifact and its manifest."""

    path: Path
    digest: ContentDigest
    manifest_path: Path
    manifest_digest: ContentDigest
    manifest: ModelArtifactManifest
    step: int
    validation_check: int
    validation_loss: float
    score_estimate: Mapping[str, object] | None = None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": "model-checkpoint",
            "path": self.path.as_posix(),
            "digest": str(self.digest),
            "manifest_path": self.manifest_path.as_posix(),
            "manifest_digest": str(self.manifest_digest),
            "step": self.step,
            "validation_check": self.validation_check,
            "validation_loss": self.validation_loss,
        }
        if self.score_estimate is not None:
            record["score_estimate"] = dict(self.score_estimate)
        return record


@dataclass(frozen=True, slots=True)
class CheckpointModelPredictor:
    """A loaded checkpoint model that can produce benchmark predictions."""

    runtime: TensorRuntime
    module: Any
    outcome_ids: tuple[str, ...]

    def predict_batch(
        self,
        batch: GeneratedSampleSet,
    ) -> tuple[tuple[float, ...], ...]:
        self.module.eval()
        fields, _labels = _batch_tensors(
            runtime=self.runtime,
            batch=batch,
            outcome_ids=self.outcome_ids,
            device=self.runtime.device,
        )
        with no_grad_context(self.runtime):
            return tuple(
                _renormalized_probabilities(row)
                for row in apply_softmax_predictions(self.runtime, self.module, fields)
            )


def _require_field_generator(generator: BenchmarkGenerator) -> _FieldBenchmarkGenerator:
    missing = tuple(
        name
        for name in (
            "materialization",
            "formation",
            "distinguishable_state_complexity",
        )
        if not hasattr(generator, name)
    )
    if missing:
        raise BenchmarkRunnerError(
            "tensor benchmark runner requires a field-generating benchmark; missing "
            + ", ".join(missing)
        )
    return cast(_FieldBenchmarkGenerator, generator)


@dataclass(frozen=True, slots=True)
class _TrainingStageResult:
    validation_history: tuple[TrainingHistoryPoint, ...]
    stop_reason: str


@dataclass(frozen=True, slots=True)
class _CurriculumRung:
    index: int
    resolution_assignment: AxisAssignment
    seed: int
    batch: GeneratedSampleSet
    sample_count: int

    @property
    def complexity(self) -> float:
        return self.batch.samples[0].complexity

    def to_record(self, *, status: str) -> dict[str, object]:
        state_space_measure = self.batch.samples[0].state_space_measure
        return {
            "index": self.index,
            "status": status,
            "resolution_assignment": self.resolution_assignment.to_record(),
            "seed": self.seed,
            "complexity_axis": _core_state_space_measure_id(),
            "complexity": self.complexity,
            "state_space_measure": (
                None if state_space_measure is None else state_space_measure.to_record()
            ),
            "state_space_request": (
                None
                if self.batch.state_space_request is None
                else self.batch.state_space_request.to_record()
            ),
            "sample_count": self.sample_count,
        }


def _rung_state_space_request(rung: _CurriculumRung) -> StateSpaceMeasureRequest:
    if rung.batch.state_space_request is not None:
        return rung.batch.state_space_request
    return StateSpaceMeasureRequest(
        minimum=rung.complexity,
        maximum=rung.complexity,
    )


def _core_state_space_measure_id() -> str:
    return StateSpaceMeasureRequest(minimum=1.0, maximum=1.0).measure_id


@dataclass(slots=True)
class _ThroughputCounter:
    seconds: float = 0.0
    samples: int = 0

    def add(self, *, seconds: float, samples: int) -> None:
        self.seconds += max(0.0, float(seconds))
        self.samples += samples

    def to_record(self, *, kind: str) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": kind,
            "sample_count": self.samples,
            "seconds": self.seconds,
        }
        record["samples_per_second"] = (
            self.samples / self.seconds if self.samples > 0 and self.seconds > 0 else 0.0
        )
        return record


@dataclass(slots=True)
class _ComputeCounter:
    compute: float = 0.0

    def add(self, *, compute_per_sample: float | None, samples: int) -> None:
        if compute_per_sample is None:
            return
        self.compute += float(compute_per_sample) * float(samples)


@dataclass(frozen=True, slots=True)
class _PhaseWorkEstimate:
    compute_per_sample: float
    bytes_per_sample: float


@dataclass(frozen=True, slots=True)
class _TrainingWorkEstimates:
    training: _PhaseWorkEstimate
    validation: _PhaseWorkEstimate
    evaluation: _PhaseWorkEstimate
    assumptions: tuple[str, ...]


@dataclass(slots=True)
class _LearningRateSchedule:
    scheduler: Any
    optimizer: Any
    update_on: str
    lr_reduction_count: int = 0
    minimum_effective_learning_rate: float | None = None
    base_learning_rates: tuple[float, ...] = ()

    def learning_rates(self) -> tuple[float, ...]:
        return tuple(float(group["lr"]) for group in self.optimizer.param_groups)

    def step_after_optimizer(self) -> None:
        if self.update_on == "optimizer-step":
            self.scheduler.step()

    def step_after_validation(self, plateau_metric: float) -> None:
        if self.update_on == "score-estimate":
            before = self.learning_rates()
            self.scheduler.step(plateau_metric)
            after = self.learning_rates()
            if any(new < old for old, new in zip(before, after, strict=True)):
                self.lr_reduction_count += 1

    def has_exhausted_plateau_response(self) -> bool:
        if self.update_on != "score-estimate":
            return True
        return (
            self.lr_reduction_count >= _minimum_plateau_lr_reductions
            or self.has_reached_minimum_effective_learning_rate()
        )

    def reset_plateau_response_count(self) -> None:
        self.lr_reduction_count = 0

    def reset_for_curriculum_expansion(self) -> None:
        self.reset_plateau_response_count()
        if self.base_learning_rates:
            for group, base_learning_rate in zip(
                self.optimizer.param_groups,
                self.base_learning_rates,
                strict=True,
            ):
                group["lr"] = base_learning_rate
        reset_scheduler = getattr(self.scheduler, "_reset", None)
        if callable(reset_scheduler):
            reset_scheduler()

    def has_reached_minimum_effective_learning_rate(self) -> bool:
        if self.minimum_effective_learning_rate is None:
            return False
        return all(
            learning_rate <= self.minimum_effective_learning_rate
            for learning_rate in self.learning_rates()
        )


class BenchmarkRunnerError(ValueError):
    """Raised when a local benchmark run cannot be planned or executed."""


_extract = RecordExtractor(error_type=BenchmarkRunnerError)


@dataclass(frozen=True, slots=True)
class BenchmarkEvaluationPlan:
    """A benchmark evaluation plan over a saved training checkpoint artifact."""

    benchmark_root: Path
    checkpoint_artifact_path: Path
    results_root: Path = Path("results")
    tensor_device: TensorRuntimeDevice = "auto"

    def __post_init__(self) -> None:
        try:
            validate_tensor_runtime_device(self.tensor_device)
        except TensorRuntimeError as error:
            raise BenchmarkRunnerError(str(error)) from error


@dataclass(frozen=True, slots=True)
class BenchmarkEvaluationSummary:
    """Summary of a benchmark evaluation generated from a checkpoint artifact."""

    run_slug: str
    benchmark_id: ProtocolIdentifier
    evaluation_bundle_path: Path
    measurement_count: int


@dataclass(frozen=True, slots=True)
class BenchmarkCompetitionPlan:
    """A pairwise benchmark competition plan over two evaluated checkpoints."""

    left_evaluation_path: Path
    right_evaluation_path: Path
    benchmark_root: Path
    results_root: Path = Path("results")
    sample_count: int = _default_sample_count
    tensor_device: TensorRuntimeDevice = "auto"

    def __post_init__(self) -> None:
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise BenchmarkRunnerError("sample_count must be a positive integer")
        try:
            validate_tensor_runtime_device(self.tensor_device)
        except TensorRuntimeError as error:
            raise BenchmarkRunnerError(str(error)) from error


@dataclass(frozen=True, slots=True)
class BenchmarkCompetitionSummary:
    """Summary of a pairwise benchmark competition record."""

    competition_id: str
    benchmark_id: ProtocolIdentifier
    competition_bundle_path: Path
    sample_count: int
    left_model_key: str
    right_model_key: str


@dataclass(frozen=True, slots=True)
class BenchmarkRunPlan:
    """A local benchmark run plan resolved from CLI or workflow inputs."""

    architecture_path: Path
    benchmark_root: Path
    results_root: Path = Path("results")
    sample_count: int = _default_sample_count
    evaluation_sample_count: int | None = None
    seed: int = 101
    train_steps: int | None = _default_train_steps
    learning_rate: float = 0.01
    optimizer: str = "adam"
    schedule: str = "reduce-on-plateau"
    gate_check_interval: int = _default_gate_check_interval
    model_checkpoint_gate_interval: int = _default_model_checkpoint_gate_interval
    gate_sample_count: int | None = None
    gate_decision_rule: str = "score-estimate-plateau"
    convergence_patience: int = _default_convergence_patience
    convergence_min_delta: float = _default_convergence_min_delta
    convergence_min_steps: int = _default_convergence_min_steps
    tensor_device: TensorRuntimeDevice = "auto"
    dry_run: bool = False

    def __post_init__(self) -> None:
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise BenchmarkRunnerError("sample_count must be a positive integer")
        if (
            self.evaluation_sample_count is not None
            and (
                type(self.evaluation_sample_count) is not int
                or self.evaluation_sample_count < 1
            )
        ):
            raise BenchmarkRunnerError("evaluation_sample_count must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise BenchmarkRunnerError("seed must be a nonnegative integer")
        if self.train_steps is not None and (
            type(self.train_steps) is not int or self.train_steps < 0
        ):
            raise BenchmarkRunnerError("train_steps must be a nonnegative integer")
        if self.train_steps is None and self.schedule == "cosine":
            raise BenchmarkRunnerError("cosine schedule requires train_steps")
        if self.train_steps is None and self.convergence_patience == 0:
            raise BenchmarkRunnerError("uncapped training requires convergence_patience")
        if self.learning_rate <= 0:
            raise BenchmarkRunnerError("learning_rate must be positive")
        if self.optimizer not in {"sgd", "adam", "adamw"}:
            raise BenchmarkRunnerError(f"unsupported optimizer: {self.optimizer}")
        if self.schedule not in {"none", "cosine", "reduce-on-plateau"}:
            raise BenchmarkRunnerError(f"unsupported schedule: {self.schedule}")
        if type(self.gate_check_interval) is not int or self.gate_check_interval < 1:
            raise BenchmarkRunnerError("gate_check_interval must be a positive integer")
        if (
            type(self.model_checkpoint_gate_interval) is not int
            or self.model_checkpoint_gate_interval < 1
        ):
            raise BenchmarkRunnerError(
                "model_checkpoint_gate_interval must be a positive integer"
            )
        if self.gate_sample_count is not None and (
            type(self.gate_sample_count) is not int or self.gate_sample_count < 1
        ):
            raise BenchmarkRunnerError("gate_sample_count must be a positive integer")
        if self.gate_decision_rule != "score-estimate-plateau":
            raise BenchmarkRunnerError(
                f"unsupported gate_decision_rule: {self.gate_decision_rule}"
            )
        if type(self.convergence_patience) is not int or self.convergence_patience < 0:
            raise BenchmarkRunnerError("convergence_patience must be nonnegative")
        if self.convergence_min_delta < 0:
            raise BenchmarkRunnerError("convergence_min_delta must be nonnegative")
        if type(self.convergence_min_steps) is not int or self.convergence_min_steps < 0:
            raise BenchmarkRunnerError("convergence_min_steps must be nonnegative")
        try:
            validate_tensor_runtime_device(self.tensor_device)
        except TensorRuntimeError as error:
            raise BenchmarkRunnerError(str(error)) from error

    @property
    def run_slug(self) -> str:
        """Return the deterministic local run suffix."""

        base = (
            f"seed{self.seed}-samples{self.sample_count}"
            f"-steps{self.train_steps if self.train_steps is not None else 'converge'}"
        )
        if self.resolved_evaluation_sample_count == self.sample_count:
            return f"{base}-{self.training_control_atom}"
        return f"{base}-eval{self.resolved_evaluation_sample_count}-{self.training_control_atom}"

    @property
    def training_control_atom(self) -> str:
        """Return a compact identity atom for training/convergence controls."""

        controls = {
            "learning_rate": float(self.learning_rate),
            "optimizer": self.optimizer,
            "schedule": self.schedule,
            "gate_check_interval": self.gate_check_interval,
            "model_checkpoint_gate_interval": self.model_checkpoint_gate_interval,
            "gate_sample_count": self.resolved_gate_sample_count,
            "gate_decision_rule": self.gate_decision_rule,
            "convergence_patience": self.convergence_patience,
            "convergence_min_delta": float(self.convergence_min_delta),
            "convergence_min_steps": self.convergence_min_steps,
            "tensor_device": self.tensor_device,
        }
        return f"train-{ContentDigest.from_value(controls).hex[:12]}"

    @property
    def resolved_evaluation_sample_count(self) -> int:
        """Return the explicit evaluation sample count for this run."""

        if self.evaluation_sample_count is None:
            return self.sample_count
        return self.evaluation_sample_count

    @property
    def resolved_gate_sample_count(self) -> int:
        """Return the generated validation sample count for competence gates."""

        if self.gate_sample_count is None:
            return self.sample_count
        return self.gate_sample_count


@dataclass(frozen=True, slots=True)
class BenchmarkRunSummary:
    """Summary of a planned or completed local benchmark run."""

    run_slug: str
    benchmark_id: ProtocolIdentifier
    architecture_path: Path
    measurement_count: int
    training_summary_path: Path
    model_artifact_root: Path
    dry_run: bool
    results_root: Path

    def to_record(self) -> dict[str, object]:
        """Return a canonical document-friendly summary record."""

        return {
            "format": "leibniz.benchmark-run",
            "format_version": 1,
            "run_slug": self.run_slug,
            "benchmark_id": str(self.benchmark_id),
            "architecture_path": _portable_record_path(
                self.architecture_path,
                results_root=self.results_root,
            ),
            "measurement_count": self.measurement_count,
            "training_summary_path": _portable_record_path(
                self.training_summary_path,
                results_root=self.results_root,
            ),
            "model_artifact_root": _portable_record_path(
                self.model_artifact_root,
                results_root=self.results_root,
            ),
            "dry_run": self.dry_run,
        }


def run_benchmark(
    plan: BenchmarkRunPlan,
    *,
    progress_callback: Callable[[BenchmarkRunSummary], None] | None = None,
) -> BenchmarkRunSummary:
    """Run or dry-run a tiny local benchmark workflow."""

    generator = _require_field_generator(load_generator(plan.benchmark_root))
    architecture = ArchitectureManifestDocument.from_bytes(
        plan.architecture_path.read_bytes()
    ).manifest
    outcome_space = generator.manifest.resolve_outcome_space()
    initial_evaluation_rung = _evaluation_curriculum_rung(
        architecture=architecture,
        generator=generator,
        sample_count=plan.resolved_evaluation_sample_count,
        seed=plan.seed,
        index=0,
    )
    evaluation_batch = initial_evaluation_rung.batch
    _validate_architecture_for_batch(
        architecture=architecture,
        batch=evaluation_batch,
        outcome_space=outcome_space,
    )

    summary = _run_summary(
        plan=plan,
        benchmark_id=generator.manifest.id,
        architecture_digest=architecture.digest,
    )
    model_inspection = ModelInspectionRecord.from_architecture(
        id=ProtocolIdentifier.parse(
            f"model-inspections.{_identifier_atom(generator.manifest.id)}."
            f"{summary.run_slug}@0.1.0"
        ),
        architecture_manifest=architecture,
    )
    if plan.dry_run:
        return summary

    progress_path = _training_progress_path(summary)
    model_interface = ModelInterface.from_outcome_space(
        id=ProtocolIdentifier.parse(
            f"model-interfaces.{_identifier_atom(generator.manifest.id)}."
            f"{summary.run_slug}@0.1.0"
        ),
        outcome_space=outcome_space,
    )
    checkpoint_artifacts: list[ModelCheckpointArtifact] = []
    progress_timings = TimingCollector()

    def checkpoint_record(
        checkpoint: ModelCheckpointArtifact,
        *,
        evaluation_rung_count: int | None,
        training_compute: float | None,
    ) -> dict[str, object]:
        return _model_checkpoint_artifact_record(
            checkpoint=checkpoint,
            architecture=architecture,
            benchmark_id=summary.benchmark_id,
            run_slug=summary.run_slug,
            evaluation_sample_count=plan.resolved_evaluation_sample_count,
            evaluation_rung_count=evaluation_rung_count,
            training_compute=training_compute,
            results_root=plan.results_root,
        )

    def publish_progress(
        training_run: TrainingRunRecord,
        throughput: Mapping[str, object],
        training_curriculum: Mapping[str, object],
        module: Any,
    ) -> None:
        with progress_timings.span("training_progress.checkpoint_decision"):
            should_write_checkpoint = _should_write_model_checkpoint(
                training_run=training_run,
                gate_interval=plan.model_checkpoint_gate_interval,
                checkpoint_artifacts=tuple(checkpoint_artifacts),
            )
        if should_write_checkpoint:
            with progress_timings.span("training_progress.checkpoint_write"):
                checkpoint_artifacts.append(
                    _write_model_checkpoint_artifact(
                        summary=summary,
                        architecture=architecture,
                        model_interface=model_interface,
                        training_run=training_run,
                        module=module,
                        runtime="pytorch",
                    )
                )
        with progress_timings.span("training_progress.checkpoint_selection"):
            selected_checkpoint = _selected_model_checkpoint(tuple(checkpoint_artifacts))
        with progress_timings.span("training_progress.inspection"):
            progress_inspection = (
                ModelInspectionRecord.from_model_manifest(
                    id=ProtocolIdentifier.parse(
                        f"model-inspections.{_identifier_atom(summary.benchmark_id)}."
                        f"{summary.run_slug}.progress@0.1.0"
                    ),
                    model_manifest=selected_checkpoint.manifest,
                    architecture_manifest=architecture,
                )
                if selected_checkpoint is not None
                else model_inspection
            )
        with progress_timings.span("training_progress.checkpoint_records"):
            progress_checkpoint_records = tuple(
                checkpoint_record(
                    checkpoint,
                    evaluation_rung_count=1,
                    training_compute=training_run.training_compute,
                )
                for checkpoint in checkpoint_artifacts
            )
            selected_checkpoint_record = (
                None
                if selected_checkpoint is None
                else checkpoint_record(
                    selected_checkpoint,
                    evaluation_rung_count=1,
                    training_compute=training_run.training_compute,
                )
            )
        with progress_timings.span("training_progress.record"):
            progress_record = _training_progress_record(
                plan=plan,
                summary=summary,
                architecture=architecture,
                inspection=progress_inspection,
                evaluation_curriculum=_curriculum_record(
                    kind="competence-gated-evaluation-curriculum",
                    rungs=(initial_evaluation_rung,),
                    frontier_index=0,
                ),
                training_curriculum=training_curriculum,
                training_run=training_run,
                throughput=_throughput_with_progress_timings(
                    throughput=throughput,
                    progress_timings=progress_timings,
                ),
                model_checkpoints=progress_checkpoint_records,
                selected_model_checkpoint=selected_checkpoint_record,
            )
        with progress_timings.span("training_progress.write"):
            _write_document_atomic(progress_path, progress_record)
        if progress_callback is not None:
            progress_callback(summary)

    training_result = _train_and_predict(
        architecture=architecture,
        initial_evaluation_rung=initial_evaluation_rung,
        generator=generator,
        outcome_space=outcome_space,
        sample_count=plan.sample_count,
        evaluation_sample_count=plan.resolved_evaluation_sample_count,
        gate_sample_count=plan.resolved_gate_sample_count,
        train_steps=plan.train_steps,
        learning_rate=float(plan.learning_rate),
        optimizer_name=plan.optimizer,
        schedule_name=plan.schedule,
        gate_check_interval=plan.gate_check_interval,
        gate_decision_rule=plan.gate_decision_rule,
        convergence_patience=plan.convergence_patience,
        convergence_min_delta=float(plan.convergence_min_delta),
        convergence_min_steps=plan.convergence_min_steps,
        tensor_device=plan.tensor_device,
        storage_bytes=model_inspection.cost_summary.storage_bytes,
        batch_size=plan.sample_count,
        seed=plan.seed,
        progress_callback=publish_progress,
    )
    selected_checkpoint = _selected_model_checkpoint(tuple(checkpoint_artifacts))
    if selected_checkpoint is None:
        raise BenchmarkRunnerError("training did not produce any model checkpoints")
    model_inspection = ModelInspectionRecord.from_model_manifest(
        id=ProtocolIdentifier.parse(
            f"model-inspections.{_identifier_atom(generator.manifest.id)}."
            f"{summary.run_slug}@0.1.0"
        ),
        model_manifest=selected_checkpoint.manifest,
        architecture_manifest=architecture,
    )
    checkpoint_records = tuple(
        checkpoint_record(
            checkpoint,
            evaluation_rung_count=len(training_result.training_rungs),
            training_compute=training_result.training_run.training_compute,
        )
        for checkpoint in checkpoint_artifacts
    )
    for record in checkpoint_records:
        _write_document(Path(_required_string(record.get("record_path"), "record_path")), record)
    selected_checkpoint_record = checkpoint_record(
        selected_checkpoint,
        evaluation_rung_count=len(training_result.training_rungs),
        training_compute=training_result.training_run.training_compute,
    )
    _write_document_atomic(
        summary.training_summary_path,
        {
            **summary.to_record(),
            "dry_run": False,
            "run_status": "completed",
            "sample_count": plan.sample_count,
            "evaluation_sample_count": plan.resolved_evaluation_sample_count,
            "training_curriculum": _curriculum_record(
                kind="competence-gated-training-curriculum",
                source="structured-training-curriculum",
                frontier_sampling_weight=0.7,
                replay_sampling_weight=0.3,
                rungs=training_result.training_rungs,
                frontier_index=training_result.training_frontier_index,
            ),
            "training_estimate": _training_estimate_record(
                summary=summary,
                training_run=training_result.training_run,
            ),
            "seed": plan.seed,
            "train_steps": plan.train_steps,
            "learning_rate": float(plan.learning_rate),
            "optimizer": plan.optimizer,
            "schedule": plan.schedule,
            "gate_check_interval": plan.gate_check_interval,
            "model_checkpoint_gate_interval": plan.model_checkpoint_gate_interval,
            "gate_sample_count": plan.resolved_gate_sample_count,
            "gate_decision_rule": plan.gate_decision_rule,
            "convergence_patience": plan.convergence_patience,
            "convergence_min_delta": float(plan.convergence_min_delta),
            "convergence_min_steps": plan.convergence_min_steps,
            "tensor_runtime": "pytorch",
            "tensor_device": training_result.training_run.protocol.tensor_device,
            "training_run": training_result.training_run.to_record(),
            "throughput": training_result.throughput,
            "architecture": model_inspection.architecture.to_record(),
            "cost_summary": _training_cost_summary(
                inspection=model_inspection,
                training_run=training_result.training_run,
            ),
            "model_checkpoints": [dict(record) for record in checkpoint_records],
            "selected_model_checkpoint": selected_checkpoint_record,
            "selected_model_checkpoint_policy": "highest-training-score-estimate",
            "evaluation_model_artifact": selected_checkpoint_record,
        },
    )
    if progress_path != summary.training_summary_path:
        progress_path.unlink(missing_ok=True)
    return summary


def _run_summary(
    *,
    plan: BenchmarkRunPlan,
    benchmark_id: ProtocolIdentifier,
    architecture_digest: ContentDigest,
) -> BenchmarkRunSummary:
    benchmark_atom = _identifier_atom(benchmark_id)
    architecture_atom = f"arch-{architecture_digest.hex[:12]}"
    run_slug = f"{benchmark_atom}-{architecture_atom}-{plan.run_slug}"
    return BenchmarkRunSummary(
        run_slug=run_slug,
        benchmark_id=benchmark_id,
        architecture_path=plan.architecture_path,
        measurement_count=plan.resolved_evaluation_sample_count,
        training_summary_path=(
            plan.results_root / "training" / benchmark_atom / f"{run_slug}{_document_suffix}"
        ),
        model_artifact_root=(plan.results_root / "models" / benchmark_atom / run_slug),
        dry_run=plan.dry_run,
        results_root=plan.results_root,
    )


def _portable_record_path(path: Path, *, results_root: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    resolved = path.resolve()
    resolved_results_root = results_root.resolve()
    if resolved.is_relative_to(resolved_results_root):
        return (Path(results_root.name) / resolved.relative_to(resolved_results_root)).as_posix()
    working_root = Path.cwd().resolve()
    if not resolved.is_relative_to(working_root):
        raise BenchmarkRunnerError(
            f"record path must be relative, inside results root, or inside working tree: {path}"
        )
    return resolved.relative_to(working_root).as_posix()


def _training_progress_path(summary: BenchmarkRunSummary) -> Path:
    return summary.training_summary_path


def evaluate_benchmark_checkpoint(plan: BenchmarkEvaluationPlan) -> BenchmarkEvaluationSummary:
    """Generate benchmark evidence from a saved training checkpoint artifact."""

    generator = _require_field_generator(load_generator(plan.benchmark_root))
    evaluation_input = _evaluation_input_from_plan(plan, generator=generator)
    outcome_space = generator.manifest.resolve_outcome_space()
    architecture = evaluation_input.architecture
    selected_checkpoint = evaluation_input.checkpoint
    run_slug = evaluation_input.run_slug
    benchmark_id = evaluation_input.benchmark_id
    benchmark_atom = _identifier_atom(benchmark_id)
    evaluation_bundle_path = (
        plan.results_root / "evaluations" / benchmark_atom / f"{run_slug}{_document_suffix}"
    )
    evaluation_seed = _unpredictable_evaluation_seed()
    evaluation_results, checkpoint_evaluation_throughput = evaluate_model_checkpoint_artifact(
        architecture=architecture,
        generator=generator,
        outcome_space=outcome_space,
        evaluation_sample_count=evaluation_input.evaluation_sample_count,
        training_rung_count=evaluation_input.evaluation_rung_count,
        seed=evaluation_seed,
        tensor_device=plan.tensor_device,
        checkpoint=selected_checkpoint,
    )
    evaluation_frontier_index = _evaluation_result_frontier_index(
        evaluation_results=evaluation_results,
        outcome_ids=tuple(outcome.id for outcome in outcome_space.outcomes),
    )
    final_evaluation_result = evaluation_results[evaluation_frontier_index]
    measurement_groups = (
        finite_measurements_for_predictions(
            batch=final_evaluation_result[0].batch,
            outcome_space=outcome_space,
            probabilities=final_evaluation_result[1],
            run_slug=f"{run_slug}.final",
        ),
    )
    sampled_competence = sampled_competence_curriculum_record(
        (
            sampled_competence_record(
                batch=final_evaluation_result[0].batch,
                measurements=measurement_groups[0],
                complexity_axis=None,
            ),
        )
    )
    measurements = tuple(
        measurement
        for group in measurement_groups
        for measurement in group
    )
    dataset = MeasurementDataset(measurements=measurements)
    dataset.validate_manifest(generator.manifest)
    model_inspection = ModelInspectionRecord.from_model_manifest(
        id=ProtocolIdentifier.parse(
            f"model-inspections.{benchmark_atom}.{run_slug}@0.1.0"
        ),
        model_manifest=selected_checkpoint.manifest,
        architecture_manifest=architecture,
    )
    evaluation_curriculum = _curriculum_record(
        kind="checkpoint-benchmark-evaluation-curriculum",
        rungs=tuple(rung for rung, _probabilities in evaluation_results),
        frontier_index=evaluation_frontier_index,
    )
    throughput = {
        "kind": "benchmark-evaluation-throughput",
        "evaluation": dict(checkpoint_evaluation_throughput),
        "checkpoint_evaluation": dict(checkpoint_evaluation_throughput),
    }
    evaluation_tensor_device = _required_string(
        checkpoint_evaluation_throughput.get("tensor_device"),
        "checkpoint_evaluation.tensor_device",
    )
    identifier_stem = f"{benchmark_atom}.{run_slug}"
    evaluation_protocol: dict[str, object] = {
        "kind": "checkpoint-benchmark-evaluation",
        "measurement_count": len(measurements),
        "evaluation_sample_count": evaluation_input.evaluation_sample_count,
        "evaluation_curriculum_rung_count": len(evaluation_results),
        "tensor_runtime": "pytorch",
        "tensor_device": evaluation_tensor_device,
        "requested_tensor_device": plan.tensor_device,
    }
    if evaluation_input.training_compute is not None:
        evaluation_protocol["training_compute"] = evaluation_input.training_compute
    checkpoint_record = _model_checkpoint_artifact_record(
        checkpoint=selected_checkpoint,
        architecture=architecture,
        benchmark_id=benchmark_id,
        run_slug=run_slug,
        evaluation_sample_count=evaluation_input.evaluation_sample_count,
        evaluation_rung_count=evaluation_input.evaluation_rung_count,
        training_compute=evaluation_input.training_compute,
        results_root=plan.results_root,
    )
    bundle = BenchmarkEvaluationBundle(
        id=ProtocolIdentifier.parse(f"benchmark-evaluations.{identifier_stem}@0.1.0"),
        run_slug=run_slug,
        benchmark_manifest=generator.manifest,
        architecture_manifest=architecture,
        model_manifest=selected_checkpoint.manifest,
        model_checkpoint=checkpoint_record,
        model_inspection=model_inspection,
        measurement_dataset=dataset,
        measurement_score_view=MeasurementScoreView.from_dataset(
            id=ProtocolIdentifier.parse(
                f"views.measurement-scores.{identifier_stem}@0.1.0"
            ),
            dataset=dataset,
        ),
        sampled_competence=sampled_competence,
        evaluation_protocol=evaluation_protocol,
        evaluation_seed=evaluation_seed,
        evaluation_curriculum=evaluation_curriculum,
        throughput=throughput,
    )
    _write_document(evaluation_bundle_path, bundle.to_record())
    return BenchmarkEvaluationSummary(
        run_slug=run_slug,
        benchmark_id=benchmark_id,
        evaluation_bundle_path=evaluation_bundle_path,
        measurement_count=len(measurements),
    )


def _evaluation_input_from_plan(
    plan: BenchmarkEvaluationPlan,
    *,
    generator: BenchmarkGenerator,
) -> _EvaluationInput:
    checkpoint_record = _load_object_record(
        plan.checkpoint_artifact_path,
        description="checkpoint artifact",
    )
    architecture = ArchitectureManifest.from_record(
        _extract_record(
            checkpoint_record.get("architecture_manifest"),
            "checkpoint_artifact.architecture_manifest",
        )
    )
    try:
        checkpoint_benchmark_id = ProtocolIdentifier.parse(
            _required_string(
                checkpoint_record.get("benchmark_id"),
                "checkpoint_artifact.benchmark_id",
            )
        )
    except ValueError as error:
        raise BenchmarkRunnerError(str(error)) from error
    if checkpoint_benchmark_id != generator.manifest.id:
        raise BenchmarkRunnerError(
            "checkpoint_artifact.benchmark_id does not match benchmark root"
        )
    run_slug = _required_string(checkpoint_record.get("run_slug"), "checkpoint_artifact.run_slug")
    evaluation_sample_count = _required_int(
        checkpoint_record.get("evaluation_sample_count"),
        "checkpoint_artifact.evaluation_sample_count",
    )
    if evaluation_sample_count < 1:
        raise BenchmarkRunnerError("checkpoint_artifact.evaluation_sample_count must be positive")
    evaluation_rung_count = _required_int(
        checkpoint_record.get("evaluation_rung_count"),
        "checkpoint_artifact.evaluation_rung_count",
    )
    if evaluation_rung_count < 1:
        raise BenchmarkRunnerError("checkpoint_artifact.evaluation_rung_count must be positive")
    training_compute = _extract.optional_float(
        checkpoint_record.get("training_compute"),
        "checkpoint_artifact.training_compute",
    )
    if training_compute is not None and training_compute < 0:
        raise BenchmarkRunnerError("checkpoint_artifact.training_compute must be nonnegative")
    return _EvaluationInput(
        architecture=architecture,
        checkpoint=load_model_checkpoint_artifact(
            checkpoint_record,
            results_root=plan.results_root,
        ),
        run_slug=run_slug,
        benchmark_id=checkpoint_benchmark_id,
        evaluation_sample_count=evaluation_sample_count,
        evaluation_rung_count=evaluation_rung_count,
        training_compute=training_compute,
    )


def compete_benchmark_checkpoints(plan: BenchmarkCompetitionPlan) -> BenchmarkCompetitionSummary:
    """Generate pairwise benchmark competition evidence from two evaluated checkpoints."""

    try:
        left_evaluation_bundle = BenchmarkEvaluationBundleDocument.from_bytes(
            plan.left_evaluation_path.read_bytes()
        ).bundle
        right_evaluation_bundle = BenchmarkEvaluationBundleDocument.from_bytes(
            plan.right_evaluation_path.read_bytes()
        ).bundle
    except ValueError as error:
        raise BenchmarkRunnerError(str(error)) from error
    left_evaluation = left_evaluation_bundle.to_record()
    right_evaluation = right_evaluation_bundle.to_record()
    generator = _require_field_generator(load_generator(plan.benchmark_root))
    benchmark_id = generator.manifest.id
    outcome_space = generator.manifest.resolve_outcome_space()
    left_architecture = left_evaluation_bundle.architecture_manifest
    right_architecture = right_evaluation_bundle.architecture_manifest
    left_checkpoint = load_model_checkpoint_artifact(
        _extract_record(
            left_evaluation.get("model_checkpoint"),
            "model_checkpoint",
        ),
        results_root=plan.results_root,
    )
    right_checkpoint = load_model_checkpoint_artifact(
        _extract_record(
            right_evaluation.get("model_checkpoint"),
            "model_checkpoint",
        ),
        results_root=plan.results_root,
    )
    left_model_key = _model_key_from_checkpoint(left_checkpoint)
    right_model_key = _model_key_from_checkpoint(right_checkpoint)
    if left_model_key == right_model_key:
        raise BenchmarkRunnerError("benchmark competition requires two distinct models")
    if right_model_key < left_model_key:
        swapped = BenchmarkCompetitionPlan(
            left_evaluation_path=plan.right_evaluation_path,
            right_evaluation_path=plan.left_evaluation_path,
            benchmark_root=plan.benchmark_root,
            results_root=plan.results_root,
            sample_count=plan.sample_count,
            tensor_device=plan.tensor_device,
        )
        return compete_benchmark_checkpoints(swapped)
    competition_seed = _unpredictable_evaluation_seed()
    competition_id = _competition_id(
        benchmark_id=benchmark_id,
        left_model_key=left_model_key,
        right_model_key=right_model_key,
        competition_seed=competition_seed,
    )
    resolution_assignment = _competition_resolution_assignment_from_evaluations(
        left_evaluation,
        right_evaluation,
    )
    competition_record, throughput = generate_model_checkpoint_competition_record(
        left_architecture=left_architecture,
        right_architecture=right_architecture,
        generator=generator,
        outcome_space=outcome_space,
        sample_count=plan.sample_count,
        seed=competition_seed,
        index=0,
        resolution_assignment=resolution_assignment,
        tensor_device=plan.tensor_device,
        left_checkpoint=left_checkpoint,
        right_checkpoint=right_checkpoint,
        left_model_key=left_model_key,
        right_model_key=right_model_key,
        benchmark_id=benchmark_id,
        competition_id=competition_id,
    )
    competition_protocol = {
        "kind": "checkpoint-benchmark-competition",
        "sample_count": plan.sample_count,
        "requested_seed": competition_seed,
        "tensor_runtime": "pytorch",
        "tensor_device": _required_string(
            throughput.get("tensor_device"),
            "checkpoint_competition.tensor_device",
        ),
        "requested_tensor_device": plan.tensor_device,
        "mechanic": "paired-prediction-accepted-mass",
    }
    competition_record["throughput"] = dict(throughput)
    competition_bundle = BenchmarkCompetitionBundle(
        id=ProtocolIdentifier.parse(f"benchmark-competitions.{competition_id}@0.1.0"),
        benchmark_manifest=generator.manifest,
        left_evaluation_bundle=left_evaluation_bundle,
        right_evaluation_bundle=right_evaluation_bundle,
        competition_result=competition_record,
        competition_protocol=competition_protocol,
        competition_seed=_required_int(competition_record.get("seed"), "competition.seed"),
        throughput=throughput,
    )
    competition_path = (
        plan.results_root
        / "evaluations"
        / _identifier_atom(benchmark_id)
        / "competitions"
        / f"{competition_id}{_document_suffix}"
    )
    _write_document(competition_path, competition_bundle.to_record())
    return BenchmarkCompetitionSummary(
        competition_id=competition_id,
        benchmark_id=benchmark_id,
        competition_bundle_path=competition_path,
        sample_count=plan.sample_count,
        left_model_key=left_model_key,
        right_model_key=right_model_key,
    )


def _evaluation_curriculum_rung(
    *,
    architecture: ArchitectureManifest,
    generator: _FieldBenchmarkGenerator,
    sample_count: int,
    seed: int,
    index: int,
) -> _CurriculumRung:
    return _curriculum_rung_from_candidates(
        architecture=architecture,
        generator=generator,
        sample_count=sample_count,
        seed=seed,
        index=index,
        candidates=_benchmark_state_space_curriculum_candidates(
            generator=generator,
            start_index=index,
        ),
    )


def _training_curriculum_rung(
    *,
    architecture: ArchitectureManifest,
    generator: _FieldBenchmarkGenerator,
    sample_count: int,
    seed: int,
    index: int,
    phase_timings: TimingCollector | None = None,
) -> _CurriculumRung:
    del architecture
    with _optional_timing_span(
        phase_timings,
        "training_frontier.rung_candidate_generation",
    ):
        candidates = _structured_training_curriculum_candidates(
            generator=generator,
            start_index=index,
            phase_timings=phase_timings,
        )
    for candidate_index, candidate in enumerate(candidates):
        if candidate_index < index:
            continue
        resolution_assignment = candidate.state_space.resolution_assignment
        if resolution_assignment is None:
            continue
        with _optional_timing_span(
            phase_timings,
            "training_frontier.rung_record_construction",
        ):
            rung_seed = seed if index == 0 else seed + 2_000_003 * index
            outcome_space = generator.manifest.resolve_outcome_space()
            if not outcome_space.outcomes:
                raise BenchmarkRunnerError("benchmark outcome space is empty")
            sample = GeneratedSample(
                index=0,
                outcome_id=outcome_space.outcomes[0].id,
                complexity=candidate.complexity,
                state_space_measure=StateSpaceMeasureValue(
                    measure_id=_core_state_space_measure_id(),
                    value=candidate.complexity,
                ),
            )
            batch = GeneratedSampleSet(
                benchmark_id=generator.manifest.id,
                generator_id=cast(ProtocolIdentifier, generator.id),
                generator_version=generator.version,
                seed=rung_seed,
                shape=(1,),
                variation_extent=_full_variation_extent,
                state_space_request=candidate.state_space_request,
                samples=(sample,),
            )
            return _CurriculumRung(
                index=index,
                resolution_assignment=resolution_assignment,
                seed=rung_seed,
                batch=batch,
                sample_count=sample_count,
            )
    raise BenchmarkRunnerError("training curriculum did not produce any rungs")


def _competition_curriculum_rung(
    *,
    generator: _FieldBenchmarkGenerator,
    sample_count: int,
    seed: int,
    index: int,
    resolution_assignment: AxisAssignment,
) -> _CurriculumRung:
    sample_set = generator(
        shape=sample_count,
        seed=seed + 9_000_009 + 2_000_003 * index,
        include_fields=True,
        resolution_assignment=resolution_assignment,
        variation_extent=_full_variation_extent,
    )
    batch = sample_set
    return _CurriculumRung(
        index=index,
        resolution_assignment=resolution_assignment,
        seed=batch.seed,
        batch=batch,
        sample_count=len(batch.samples),
    )


def _curriculum_rung_from_candidates(
    *,
    architecture: ArchitectureManifest,
    generator: _FieldBenchmarkGenerator,
    sample_count: int,
    seed: int,
    index: int,
    candidates: Sequence[_CurriculumCandidate],
) -> _CurriculumRung:
    for candidate_index, candidate in enumerate(candidates):
        if candidate_index < index:
            continue
        rung_seed = seed if index == 0 else seed + 2_000_003 * index
        sample_set = generator(
            shape=sample_count,
            seed=rung_seed,
            include_fields=True,
            state_space_request=candidate.state_space_request,
            variation_extent=_full_variation_extent,
        )
        batch = sample_set
        if not batch.samples:
            continue
        input_reason = _input_shape_boundary_reason(
            architecture=architecture,
            sample_shape=batch.samples[0].require_field().shape,
        )
        if input_reason is not None:
            if index == 0:
                raise BenchmarkRunnerError(input_reason)
            raise BenchmarkRunnerError(
                "architecture scale contract rejected the next curriculum rung: "
                f"{input_reason}"
            )
        materialization_plan = batch.samples[0].materialization_plan
        if materialization_plan is None:
            raise BenchmarkRunnerError(
                "field curriculum sample did not include a materialization plan"
            )
        return _CurriculumRung(
            index=index,
            resolution_assignment=materialization_plan.resolution_assignment,
            seed=batch.seed,
            batch=batch,
            sample_count=len(batch.samples),
        )
    raise BenchmarkRunnerError("evaluation curriculum did not produce any rungs")


@dataclass(frozen=True, slots=True)
class _CurriculumCandidate:
    state_space: StateSpaceCandidate
    source_request: StateSpaceMeasureRequest | None = None

    @property
    def state_space_request(self) -> StateSpaceMeasureRequest:
        if self.source_request is not None:
            return self.source_request
        return self.state_space.request

    @property
    def complexity(self) -> float:
        return self.state_space.complexity


def _optional_timing_span(timing: TimingCollector | None, phase: str) -> Any:
    if timing is None:
        return nullcontext()
    return timing.span(phase)


def _benchmark_state_space_curriculum_candidates(
    *,
    generator: _FieldBenchmarkGenerator,
    start_index: int,
    phase_timings: TimingCollector | None = None,
) -> Sequence[_CurriculumCandidate]:
    return _benchmark_state_space_candidates(
        generator=generator,
        start_index=start_index,
        phase_timings=phase_timings,
    )


def _structured_training_curriculum_candidates(
    *,
    generator: _FieldBenchmarkGenerator,
    start_index: int,
    phase_timings: TimingCollector | None = None,
) -> Sequence[_CurriculumCandidate]:
    return _benchmark_state_space_candidates(
        generator=generator,
        start_index=start_index,
        phase_timings=phase_timings,
    )


def _benchmark_state_space_candidates(
    *,
    generator: _FieldBenchmarkGenerator,
    start_index: int,
    phase_timings: TimingCollector | None = None,
) -> Sequence[_CurriculumCandidate]:
    stage_count = max(8, start_index + 8)
    minimum = generator.minimum_state_space_measure().value
    candidates: list[_CurriculumCandidate] = []
    seen_complexities: set[float] = set()
    for target_index in range(stage_count):
        target = minimum + target_index * _state_space_target_spacing
        request = StateSpaceMeasureRequest(
            minimum=target,
            maximum=target + _state_space_target_spacing,
        )
        with _optional_timing_span(
            phase_timings,
            "training_frontier.state_space_request",
        ):
            state_space = generator.state_space_for_request(request=request)
        if state_space is None or state_space.complexity in seen_complexities:
            continue
        seen_complexities.add(state_space.complexity)
        candidates.append(
            _CurriculumCandidate(
                state_space=state_space,
                source_request=request,
            )
        )
    return tuple(candidates)


def _curriculum_record(
    *,
    kind: str,
    rungs: Sequence[_CurriculumRung],
    frontier_index: int,
    source: str | None = None,
    frontier_sampling_weight: float | None = None,
    replay_sampling_weight: float | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "kind": kind,
        "curriculum_variable": "state-space-measure",
        "complexity_axis": _core_state_space_measure_id(),
        "sampling_levers": ["state-space-measure"],
        "state_space_measure": {
            "measure_id": _core_state_space_measure_id(),
            "scale": "log2",
        },
        "candidate_policy": {
            "kind": "benchmark-owned-target-state-space",
            "target_spacing": _state_space_target_spacing,
        },
        "gating_metric": "monotone-frontier-validation-competence",
        "rung_policy": "unbounded-competence-frontier",
        "frontier_index": frontier_index,
        "unlocked_rung_count": min(len(rungs), frontier_index + 1),
        "rungs": [
            rung.to_record(
                status=(
                    "frontier"
                    if rung.index == frontier_index
                    else "unlocked"
                    if rung.index < frontier_index
                    else "locked"
                )
            )
            for rung in rungs
        ],
    }
    if source is not None:
        record["source"] = source
    if frontier_sampling_weight is not None:
        record["frontier_sampling_weight"] = frontier_sampling_weight
    if replay_sampling_weight is not None:
        record["replay_sampling_weight"] = replay_sampling_weight
    return record


def _validate_architecture_for_batch(
    *,
    architecture: ArchitectureManifest,
    batch: GeneratedSampleSet,
    outcome_space: OutcomeSpace,
) -> None:
    sample_shape = batch.samples[0].require_field().shape
    input_reason = _input_shape_boundary_reason(
        architecture=architecture,
        sample_shape=sample_shape,
    )
    if input_reason is not None:
        raise BenchmarkRunnerError(input_reason)
    outcome_count = len(outcome_space.outcomes)
    if architecture.output_shape != (outcome_count,):
        raise BenchmarkRunnerError(
            f"architecture output_shape {architecture.output_shape} does not match "
            f"{outcome_count} resolved benchmark outcomes"
        )


def _input_shape_boundary_reason(
    *,
    architecture: ArchitectureManifest,
    sample_shape: tuple[int, ...],
) -> str | None:
    contract = architecture.model_scale_contract
    if contract is not None:
        if _shape_matches_scale_contract(contract=contract, sample_shape=sample_shape):
            return None
        return (
            f"architecture variable-shape scale contract does not accept generated "
            f"observation shape {sample_shape}"
        )
    return (
        "architecture must declare a variable-shape scale contract for generated "
        "observation shape "
        f"{sample_shape}"
    )


def _shape_matches_scale_contract(
    *,
    contract: Any,
    sample_shape: tuple[int, ...],
) -> bool:
    if contract.maximum is not None and contract.maximum == contract.minimum:
        return False
    if len(contract.axes) != len(sample_shape):
        return False
    saw_scaled_axis = False
    for axis in contract.axes:
        index = cast(int, axis["index"])
        if axis["kind"] == "fixed":
            if sample_shape[index] != cast(int, axis["size"]):
                return False
        else:
            saw_scaled_axis = True
            if not contract.accepts_scale(sample_shape[index]):
                return False
    return saw_scaled_axis


def _train_and_predict(
    *,
    architecture: ArchitectureManifest,
    initial_evaluation_rung: _CurriculumRung,
    generator: _FieldBenchmarkGenerator,
    outcome_space: OutcomeSpace,
    sample_count: int,
    evaluation_sample_count: int,
    gate_sample_count: int,
    train_steps: int | None,
    learning_rate: float,
    optimizer_name: str,
    schedule_name: str,
    gate_check_interval: int,
    gate_decision_rule: str,
    convergence_patience: int,
    convergence_min_delta: float,
    convergence_min_steps: int,
    tensor_device: TensorRuntimeDevice,
    storage_bytes: int | None,
    batch_size: int,
    seed: int,
    progress_callback: (
        Callable[[TrainingRunRecord, Mapping[str, object], Mapping[str, object], Any], None]
        | None
    ) = None,
) -> _TrainingResult:
    fallback_errors: list[tuple[str, str]] = []
    try:
        device_kinds = tensor_runtime_device_kinds(tensor_device)
    except TensorRuntimeError as error:
        raise BenchmarkRunnerError(str(error)) from error
    for index, device_kind in enumerate(device_kinds):
        try:
            return _train_and_predict_on_device(
                architecture=architecture,
                initial_evaluation_rung=initial_evaluation_rung,
                generator=generator,
                outcome_space=outcome_space,
                sample_count=sample_count,
                evaluation_sample_count=evaluation_sample_count,
                gate_sample_count=gate_sample_count,
                train_steps=train_steps,
                learning_rate=learning_rate,
                optimizer_name=optimizer_name,
                schedule_name=schedule_name,
                gate_check_interval=gate_check_interval,
                gate_decision_rule=gate_decision_rule,
                convergence_patience=convergence_patience,
                convergence_min_delta=convergence_min_delta,
                convergence_min_steps=convergence_min_steps,
                tensor_device=device_kind,
                storage_bytes=storage_bytes,
                batch_size=batch_size,
                seed=seed,
                progress_callback=progress_callback,
                fallback_errors=tuple(fallback_errors),
            )
        except RuntimeError as error:
            if tensor_device != "auto" or index == len(device_kinds) - 1:
                raise
            fallback_errors.append((device_kind, str(error)))
    raise BenchmarkRunnerError("no tensor runtime device could execute benchmark training")


def _train_and_predict_on_device(
    *,
    architecture: ArchitectureManifest,
    initial_evaluation_rung: _CurriculumRung,
    generator: _FieldBenchmarkGenerator,
    outcome_space: OutcomeSpace,
    sample_count: int,
    evaluation_sample_count: int,
    gate_sample_count: int,
    train_steps: int | None,
    learning_rate: float,
    optimizer_name: str,
    schedule_name: str,
    gate_check_interval: int,
    gate_decision_rule: str,
    convergence_patience: int,
    convergence_min_delta: float,
    convergence_min_steps: int,
    tensor_device: TensorRuntimeDeviceKind,
    storage_bytes: int | None,
    batch_size: int,
    seed: int,
    progress_callback: (
        Callable[[TrainingRunRecord, Mapping[str, object], Mapping[str, object], Any], None]
        | None
    ) = None,
    fallback_errors: tuple[tuple[str, str], ...] = (),
) -> _TrainingResult:
    try:
        runtime = resolve_tensor_runtime(tensor_device)
    except TensorRuntimeError as error:
        raise BenchmarkRunnerError(str(error)) from error
    seed_runtime(runtime, seed=seed)
    executable = ExecutableModelOperator(architecture)
    module = OperationFallbackSequential(
        runtime=runtime,
        operations=executable.operation_modules(),
    )
    outcome_ids = tuple(outcome.id for outcome in outcome_space.outcomes)
    loss_function = build_cross_entropy_loss(runtime)
    optimizer = _make_optimizer(
        runtime=runtime,
        parameters=module.parameters(),
        name=optimizer_name,
        learning_rate=learning_rate,
    )
    module.attach_optimizer(optimizer)
    scheduler = _make_scheduler(
        runtime=runtime,
        optimizer=optimizer,
        name=schedule_name,
        max_steps=train_steps,
        min_delta=convergence_min_delta,
        patience=convergence_patience,
    )
    training_counter = _ThroughputCounter()
    training_compute_counter = _ComputeCounter()
    validation_counter = _ThroughputCounter()
    evaluation_counter = _ThroughputCounter()
    phase_timings = TimingCollector()

    def training_batch_for_seed(
        batch_seed: int,
        *,
        batch_sample_count: int,
        generation_phase: str,
        tensor_phase: str,
        state_space_request: StateSpaceMeasureRequest,
    ) -> tuple[Any, Any]:
        with phase_timings.span(generation_phase, samples=batch_sample_count):
            generated = generator(
                shape=batch_sample_count,
                seed=batch_seed,
                include_metadata=False,
                state_space_request=state_space_request,
                variation_extent=_full_variation_extent,
                runtime=runtime,
                outcome_ids=outcome_ids,
                timing=phase_timings,
                timing_prefix=f"{generation_phase}.",
            )
            if generated.sample_count == 0:
                raise BenchmarkRunnerError(
                    "generator returned no samples for selected state-space measure"
                )
            with phase_timings.span(tensor_phase, samples=batch_sample_count):
                return generated.require_tensors()

    def validation_sample_batch_for_seed(
        batch_seed: int,
        *,
        batch_sample_count: int,
        generation_phase: str,
        state_space_request: StateSpaceMeasureRequest,
    ) -> GeneratedSampleSet:
        with phase_timings.span(generation_phase, samples=batch_sample_count):
            generated = generator(
                shape=batch_sample_count,
                seed=batch_seed,
                include_fields=False,
                state_space_request=state_space_request,
                variation_extent=_full_variation_extent,
                runtime=runtime,
                outcome_ids=outcome_ids,
                timing=phase_timings,
                timing_prefix=f"{generation_phase}.",
            )
            if generated.sample_count == 0:
                raise BenchmarkRunnerError(
                    "generator returned no samples for selected state-space measure"
                )
            if not generated.includes_fields:
                raise BenchmarkRunnerError("validation gate batch did not include fields")
            return generated

    training_rungs: list[_CurriculumRung] = [
        _training_curriculum_rung(
            architecture=architecture,
            generator=generator,
            sample_count=sample_count,
            seed=seed,
            index=0,
            phase_timings=phase_timings,
        )
    ]
    training_frontier_index = 0
    frontier_plateau_points: list[ValidationCompetencePoint] = []

    def current_frontier() -> _CurriculumRung:
        return training_rungs[training_frontier_index]

    def training_rung_for_step(step: int) -> _CurriculumRung:
        if training_frontier_index == 0:
            return current_frontier()
        # Keep most updates on the frontier while reserving deterministic replay
        # for previously unlocked training rungs.
        if step % 10 < 7:
            return current_frontier()
        replay_index = (step // 10) % training_frontier_index
        return training_rungs[replay_index]

    def advance_frontier(history: Sequence[TrainingHistoryPoint]) -> bool:
        nonlocal training_frontier_index
        latest = history[-1]
        chance_mass = _chance_accepted_mass(outcome_ids)
        frontier_point = _training_history_frontier_point(latest)
        with phase_timings.span("training_frontier.advance_decision"):
            should_advance = _frontier_plateau_advances(
                frontier_point=frontier_point,
                previous_frontier_points=tuple(frontier_plateau_points),
                chance_mass=chance_mass,
            )
        if not should_advance:
            return False
        next_index = training_frontier_index + 1
        with phase_timings.span("training_frontier.rung_append"):
            training_rungs.append(
                _training_curriculum_rung(
                    architecture=architecture,
                    generator=generator,
                    sample_count=sample_count,
                    seed=seed,
                    index=next_index,
                    phase_timings=phase_timings,
                )
            )
        with phase_timings.span("training_frontier.bookkeeping"):
            frontier_plateau_points.append(frontier_point)
            training_frontier_index += 1
        if scheduler is not None:
            with phase_timings.span("training_frontier.scheduler_reset"):
                scheduler.reset_for_curriculum_expansion()
        return True

    training_result = _train_until_convergence(
        runtime=runtime,
        module=module,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        train_batch=lambda step: (
            training_batch_for_seed(
                seed + step,
                batch_sample_count=sample_count,
                generation_phase="training_formation_generation",
                tensor_phase="training_tensor_batch",
                state_space_request=_rung_state_space_request(training_rung_for_step(step)),
            )
        ),
        validation_batch=lambda check: validation_sample_batch_for_seed(
            seed + 1_000_003 + check,
            batch_sample_count=gate_sample_count,
            generation_phase="validation_formation_generation",
            state_space_request=_rung_state_space_request(current_frontier()),
        ),
        outcome_space=outcome_space,
        outcome_ids=outcome_ids,
        max_steps=train_steps,
        gate_check_interval=gate_check_interval,
        patience=convergence_patience,
        min_delta=convergence_min_delta,
        min_steps=convergence_min_steps,
        batch_size=sample_count,
        gate_sample_count=gate_sample_count,
        architecture=architecture,
        training_counter=training_counter,
        training_compute_counter=training_compute_counter,
        validation_counter=validation_counter,
        phase_timings=phase_timings,
        on_plateau=advance_frontier,
        on_gate_check=lambda history, checked_module: (
            progress_callback(
                _running_training_run_record(
                    seed=seed,
                    batch_size=sample_count,
                    max_steps=train_steps,
                    learning_rate=float(learning_rate),
                    optimizer_name=optimizer_name,
                    schedule_name=schedule_name,
                    gate_check_interval=gate_check_interval,
                    gate_sample_count=gate_sample_count,
                    gate_decision_rule=gate_decision_rule,
                    convergence_patience=convergence_patience,
                    convergence_min_delta=convergence_min_delta,
                    convergence_min_steps=convergence_min_steps,
                    tensor_device=runtime.device_kind,
                    validation_history=history,
                    training_compute=training_compute_counter.compute,
                ),
                _throughput_record(
                    runtime_device=runtime.device_kind,
                    training_counter=training_counter,
                    validation_counter=validation_counter,
                    evaluation_counter=evaluation_counter,
                    competition_counter=None,
                    roofline=runtime_roofline_record(runtime),
                    work_estimates=_training_work_estimates(
                        architecture=architecture,
                        inference_compute=_training_history_max_inference_compute(history),
                        training_compute_per_sample=(
                            _training_history_latest_training_compute_per_sample(history)
                        ),
                        storage_bytes=storage_bytes,
                        batch_size=batch_size,
                    ),
                    phase_timings=phase_timings,
                    fallback_errors=fallback_errors,
                    operation_fallbacks=module.operation_fallback_records(),
                ),
                _curriculum_record(
                    kind="competence-gated-training-curriculum",
                    source="structured-training-curriculum",
                    frontier_sampling_weight=0.7,
                    replay_sampling_weight=0.3,
                    rungs=tuple(training_rungs),
                    frontier_index=training_frontier_index,
                ),
                checked_module,
            )
            if progress_callback is not None
            else None
        ),
    )
    validation_history = training_result.validation_history
    final_training_stop_reason = training_result.stop_reason
    if train_steps is None and not training_stage_converged(final_training_stop_reason):
        raise BenchmarkRunnerError("uncapped training curriculum ended before convergence")
    training_run = _training_run_record(
        seed=seed,
        batch_size=sample_count,
        max_steps=train_steps,
        learning_rate=float(learning_rate),
        optimizer_name=optimizer_name,
        schedule_name=schedule_name,
        gate_check_interval=gate_check_interval,
        gate_sample_count=gate_sample_count,
        gate_decision_rule=gate_decision_rule,
        convergence_patience=convergence_patience,
        convergence_min_delta=convergence_min_delta,
        convergence_min_steps=convergence_min_steps,
        tensor_device=runtime.device_kind,
        validation_history=tuple(validation_history),
        stop_reason=final_training_stop_reason,
        training_compute=training_compute_counter.compute,
    )
    return _TrainingResult(
        evaluation_results=(),
        training_rungs=tuple(training_rungs),
        training_frontier_index=training_frontier_index,
        training_run=training_run,
        throughput=_throughput_record(
            runtime_device=runtime.device_kind,
            training_counter=training_counter,
            validation_counter=validation_counter,
            evaluation_counter=evaluation_counter,
            competition_counter=None,
            roofline=runtime_roofline_record(runtime),
            work_estimates=_training_work_estimates(
                architecture=architecture,
                inference_compute=_training_history_max_inference_compute(validation_history),
                training_compute_per_sample=(
                    _training_history_latest_training_compute_per_sample(validation_history)
                ),
                storage_bytes=storage_bytes,
                batch_size=batch_size,
            ),
            phase_timings=phase_timings,
            fallback_errors=fallback_errors,
            operation_fallbacks=module.operation_fallback_records(),
        ),
    )


def evaluate_model_checkpoint_artifact(
    *,
    architecture: ArchitectureManifest,
    generator: _FieldBenchmarkGenerator,
    outcome_space: OutcomeSpace,
    evaluation_sample_count: int,
    training_rung_count: int,
    seed: int,
    tensor_device: TensorRuntimeDevice,
    checkpoint: ModelCheckpointArtifact,
) -> tuple[
    tuple[tuple[_CurriculumRung, tuple[tuple[float, ...], ...]], ...],
    Mapping[str, object],
]:
    """Generate benchmark evaluation evidence from a saved checkpoint artifact."""

    predictor = load_model_checkpoint_predictor(
        architecture=architecture,
        outcome_space=outcome_space,
        checkpoint=checkpoint,
        tensor_device=tensor_device,
    )
    evaluation_counter = _ThroughputCounter()
    max_inference_compute: int | None = None
    results: list[tuple[_CurriculumRung, tuple[tuple[float, ...], ...]]] = []
    for index in range(training_rung_count):
        rung = _evaluation_curriculum_rung(
            architecture=architecture,
            generator=generator,
            sample_count=evaluation_sample_count,
            seed=seed,
            index=index,
        )
        evaluation_started = time.perf_counter()
        predictions = predictor.predict_batch(rung.batch)
        evaluation_counter.add(
            seconds=time.perf_counter() - evaluation_started,
            samples=len(rung.batch.samples),
        )
        batch_max_inference_compute = _batch_max_inference_compute(
            architecture=architecture,
            batch=rung.batch,
        )
        if batch_max_inference_compute is None:
            raise BenchmarkRunnerError(
                "checkpoint evaluation could not measure max_inference_compute"
            )
        max_inference_compute = _max_optional_int(
            max_inference_compute,
            batch_max_inference_compute,
        )
        results.append((rung, predictions))
    if not results:
        raise BenchmarkRunnerError("checkpoint evaluation did not produce any results")
    throughput = evaluation_counter.to_record(kind="checkpoint-evaluation-throughput")
    throughput["tensor_runtime"] = "pytorch"
    throughput["tensor_device"] = predictor.runtime.device_kind
    if max_inference_compute is None:
        raise BenchmarkRunnerError(
            "checkpoint evaluation could not measure max_inference_compute"
        )
    throughput["max_inference_compute"] = max_inference_compute
    return tuple(results), throughput


def generate_model_checkpoint_competition_record(
    *,
    left_architecture: ArchitectureManifest,
    right_architecture: ArchitectureManifest,
    generator: _FieldBenchmarkGenerator,
    outcome_space: OutcomeSpace,
    sample_count: int,
    seed: int,
    index: int,
    resolution_assignment: AxisAssignment,
    tensor_device: TensorRuntimeDevice,
    left_checkpoint: ModelCheckpointArtifact,
    right_checkpoint: ModelCheckpointArtifact,
    left_model_key: str,
    right_model_key: str,
    benchmark_id: ProtocolIdentifier,
    competition_id: str,
) -> tuple[dict[str, object], Mapping[str, object]]:
    """Generate pairwise competition evidence from two saved checkpoint artifacts."""

    left_predictor = load_model_checkpoint_predictor(
        architecture=left_architecture,
        outcome_space=outcome_space,
        checkpoint=left_checkpoint,
        tensor_device=tensor_device,
    )
    right_predictor = load_model_checkpoint_predictor(
        architecture=right_architecture,
        outcome_space=outcome_space,
        checkpoint=right_checkpoint,
        tensor_device=tensor_device,
    )
    rung = _competition_curriculum_rung(
        generator=generator,
        sample_count=sample_count,
        seed=seed,
        index=index,
        resolution_assignment=resolution_assignment,
    )
    competition_counter = _ThroughputCounter()
    competition_started = time.perf_counter()
    left_predictions = left_predictor.predict_batch(rung.batch)
    right_predictions = right_predictor.predict_batch(rung.batch)
    competition_counter.add(
        seconds=time.perf_counter() - competition_started,
        samples=2 * len(rung.batch.samples),
    )
    throughput = competition_counter.to_record(kind="checkpoint-competition-throughput")
    throughput["tensor_runtime"] = "pytorch"
    throughput["tensor_device"] = left_predictor.runtime.device_kind
    left_max_inference_compute = _batch_max_inference_compute(
        architecture=left_architecture,
        batch=rung.batch,
    )
    right_max_inference_compute = _batch_max_inference_compute(
        architecture=right_architecture,
        batch=rung.batch,
    )
    if left_max_inference_compute is None:
        raise BenchmarkRunnerError(
            "checkpoint competition could not measure left_max_inference_compute"
        )
    if right_max_inference_compute is None:
        raise BenchmarkRunnerError(
            "checkpoint competition could not measure right_max_inference_compute"
        )
    throughput["left_max_inference_compute"] = left_max_inference_compute
    throughput["right_max_inference_compute"] = right_max_inference_compute
    return (
        _checkpoint_competition_record(
            batch=rung.batch,
            left_probabilities=left_predictions,
            right_probabilities=right_predictions,
            outcome_space=outcome_space,
            left_model_key=left_model_key,
            right_model_key=right_model_key,
            benchmark_id=benchmark_id,
            competition_id=competition_id,
        ),
        throughput,
    )


def _batch_max_inference_compute(
    *,
    architecture: ArchitectureManifest,
    batch: GeneratedSampleSet,
) -> int | None:
    tensor_input_shape = _tensor_input_shape(batch.fields)
    if tensor_input_shape is not None:
        plan = summarize_architecture_operators(
            _architecture_with_input_shape(architecture, tensor_input_shape)
        )
        return plan.inference_compute
    max_compute: int | None = None
    for input_shape in sorted({sample.require_field().shape for sample in batch.samples}):
        plan = summarize_architecture_operators(
            _architecture_with_input_shape(architecture, input_shape)
        )
        if plan.inference_compute is None:
            return None
        max_compute = _max_optional_int(max_compute, plan.inference_compute)
    return max_compute


def _batch_max_training_compute_per_sample(
    *,
    architecture: ArchitectureManifest,
    fields: Any,
) -> int | None:
    input_shape = _tensor_input_shape(fields)
    if input_shape is None:
        return None
    plan = summarize_architecture_operators(
        _architecture_with_input_shape(architecture, input_shape)
    )
    return plan.training_compute_per_sample


def _tensor_input_shape(fields: Any) -> tuple[int, ...] | None:
    shape = getattr(fields, "shape", None)
    if shape is None or len(shape) < 2:
        return None
    input_shape: list[int] = []
    for value in tuple(shape)[1:]:
        if type(value) is not int or value < 0:
            return None
        input_shape.append(value)
    return tuple(input_shape)


def _architecture_with_input_shape(
    architecture: ArchitectureManifest,
    input_shape: tuple[int, ...],
) -> ArchitectureManifest:
    record = architecture.to_record()
    record["input_shape"] = list(input_shape)
    record.pop("id", None)
    record.pop("model_scale_contract", None)
    return ArchitectureManifest.from_record(record)


def _max_optional_int(left: int | None, right: int | None) -> int | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def load_model_checkpoint_predictor(
    *,
    architecture: ArchitectureManifest,
    outcome_space: OutcomeSpace,
    checkpoint: ModelCheckpointArtifact,
    tensor_device: TensorRuntimeDevice,
) -> CheckpointModelPredictor:
    """Load a saved checkpoint artifact as an executable benchmark predictor."""

    try:
        runtime = resolve_tensor_runtime(tensor_device)
    except TensorRuntimeError as error:
        raise BenchmarkRunnerError(str(error)) from error
    executable = ExecutableModelOperator(architecture)
    module = OperationFallbackSequential(
        runtime=runtime,
        operations=executable.operation_modules(),
    )
    _load_torch_checkpoint(module=module, runtime=runtime, checkpoint=checkpoint)
    module.eval()
    outcome_ids = tuple(outcome.id for outcome in outcome_space.outcomes)
    return CheckpointModelPredictor(
        runtime=runtime,
        module=module,
        outcome_ids=outcome_ids,
    )


def load_model_checkpoint_artifact(
    record: Mapping[str, object],
    *,
    results_root: Path = Path("results"),
) -> ModelCheckpointArtifact:
    """Load and validate a checkpoint artifact record."""

    path = _resolve_artifact_record_path(
        _required_string(record.get("path"), "model checkpoint path"),
        results_root=results_root,
    )
    digest = ContentDigest.from_string(
        record.get("digest"),
        field="model checkpoint digest",
        error_type=BenchmarkRunnerError,
    )
    if not path.is_file():
        raise BenchmarkRunnerError(f"model checkpoint path does not exist: {path}")
    if _file_content_digest(path) != digest:
        raise BenchmarkRunnerError(f"model checkpoint digest mismatch: {path}")
    manifest_path = _resolve_artifact_record_path(
        _required_string(record.get("manifest_path"), "model checkpoint manifest_path"),
        results_root=results_root,
    )
    manifest_digest = ContentDigest.from_string(
        record.get("manifest_digest"),
        field="model checkpoint manifest_digest",
        error_type=BenchmarkRunnerError,
    )
    if not manifest_path.is_file():
        raise BenchmarkRunnerError(
            f"model checkpoint manifest_path does not exist: {manifest_path}"
        )
    manifest_document = ModelArtifactManifestDocument.from_bytes(manifest_path.read_bytes())
    if manifest_document.digest != manifest_digest:
        raise BenchmarkRunnerError(
            f"model checkpoint manifest digest mismatch: {manifest_path}"
        )
    return ModelCheckpointArtifact(
        path=path,
        digest=digest,
        manifest_path=manifest_path,
        manifest_digest=manifest_digest,
        manifest=manifest_document.manifest,
        step=_required_int(record.get("step"), "model checkpoint step"),
        validation_check=_required_int(
            record.get("validation_check"),
            "model checkpoint validation_check",
        ),
        validation_loss=_required_float(
            record.get("validation_loss"),
            "model checkpoint validation_loss",
        ),
    )


def _model_checkpoint_artifact_record(
    *,
    checkpoint: ModelCheckpointArtifact,
    architecture: ArchitectureManifest,
    benchmark_id: ProtocolIdentifier,
    run_slug: str,
    evaluation_sample_count: int,
    evaluation_rung_count: int | None,
    training_compute: float | None,
    results_root: Path | None = None,
) -> dict[str, object]:
    record = checkpoint.to_record()
    if results_root is not None:
        record["path"] = _artifact_record_path(checkpoint.path, results_root=results_root)
        record["manifest_path"] = _artifact_record_path(
            checkpoint.manifest_path,
            results_root=results_root,
        )
    record["record_path"] = _artifact_record_path(
        checkpoint.path.with_suffix(".checkpoint" + _document_suffix),
        results_root=results_root,
    )
    record["benchmark_id"] = str(benchmark_id)
    record["run_slug"] = run_slug
    record["architecture_manifest"] = architecture.to_record()
    record["model_manifest"] = checkpoint.manifest.to_record()
    record["evaluation_sample_count"] = evaluation_sample_count
    if evaluation_rung_count is not None:
        record["evaluation_rung_count"] = evaluation_rung_count
    if training_compute is not None:
        record["training_compute"] = training_compute
    return record


def _artifact_record_path(path: Path, *, results_root: Path | None) -> str:
    if results_root is None:
        return path.as_posix()
    resolved_results_root = results_root.resolve()
    resolved_path = path.resolve()
    if not resolved_path.is_relative_to(resolved_results_root):
        raise BenchmarkRunnerError(f"artifact path must stay inside results root: {path}")
    return (Path(results_root.name) / resolved_path.relative_to(resolved_results_root)).as_posix()


def _resolve_artifact_record_path(value: str, *, results_root: Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (results_root.parent / path).resolve()
    if not resolved.is_relative_to(results_root.resolve()):
        raise BenchmarkRunnerError(f"artifact path must stay inside results root: {value}")
    return resolved


def _load_object_record(path: Path, *, description: str) -> Mapping[str, object]:
    return load_object_document(path.read_bytes(), description=description)


def _extract_record(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkRunnerError(f"{field} must be a record")
    return cast(Mapping[str, object], value)


def _extract_sequence(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise BenchmarkRunnerError(f"{field} must be a sequence")
    return tuple(cast(Sequence[object], value))


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkRunnerError(f"{field} must be a nonempty string")
    return value


def _required_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise BenchmarkRunnerError(f"{field} must be an integer")
    return value


def _required_float(value: object, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise BenchmarkRunnerError(f"{field} must be a finite number")
    return float(value)


def _unpredictable_evaluation_seed() -> int:
    return secrets.randbelow(2**63)


def _competition_id(
    *,
    benchmark_id: ProtocolIdentifier,
    left_model_key: str,
    right_model_key: str,
    competition_seed: int,
) -> str:
    digest = ContentDigest.from_value(
        {
            "kind": "benchmark-model-competition",
            "version": 2,
            "benchmark_id": str(benchmark_id),
            "left_model_key": left_model_key,
            "right_model_key": right_model_key,
            "competition_seed": competition_seed,
        }
    )
    return f"models-{digest.hex[:16]}"


def _model_key_from_checkpoint(checkpoint: ModelCheckpointArtifact) -> str:
    return str(checkpoint.digest)


def _competition_resolution_assignment_from_evaluations(
    left_evaluation: Mapping[str, object],
    right_evaluation: Mapping[str, object],
) -> AxisAssignment:
    left_rung = _competition_frontier_rung_from_evaluation(left_evaluation, field="left")
    right_rung = _competition_frontier_rung_from_evaluation(right_evaluation, field="right")
    selected = min(
        (left_rung, right_rung),
        key=lambda rung: (
            _required_float(rung.get("complexity"), "evaluation_curriculum.rungs.complexity"),
            _required_int(rung.get("index"), "evaluation_curriculum.rungs.index"),
        ),
    )
    resolution_record = _extract_record(
        selected.get("resolution_assignment"),
        "evaluation_curriculum.rungs.resolution_assignment",
    )
    return AxisAssignment.from_record(resolution_record)


def _competition_frontier_rung_from_evaluation(
    evaluation: Mapping[str, object],
    *,
    field: str,
) -> Mapping[str, object]:
    curriculum = _extract_record(
        evaluation.get("evaluation_curriculum"),
        f"{field}.evaluation_curriculum",
    )
    rungs = _extract_sequence(
        curriculum.get("rungs"),
        f"{field}.evaluation_curriculum.rungs",
    )
    frontier_index = _required_int(
        curriculum.get("frontier_index"),
        f"{field}.evaluation_curriculum.frontier_index",
    )
    if frontier_index < 0 or frontier_index >= len(rungs):
        raise BenchmarkRunnerError(
            f"{field}.evaluation_curriculum.frontier_index is out of range"
        )
    return _extract_record(
        rungs[frontier_index],
        f"{field}.evaluation_curriculum.rungs",
    )


def training_stage_converged(stop_reason: str) -> bool:
    return stop_reason in _converged_training_stage_stop_reasons


def _train_until_convergence(
    *,
    runtime: TensorRuntime,
    module: Any,
    optimizer: Any,
    scheduler: _LearningRateSchedule | None,
    loss_function: Any,
    train_batch: Callable[[int], tuple[Any, Any]],
    validation_batch: Callable[[int], GeneratedSampleSet],
    outcome_space: OutcomeSpace,
    outcome_ids: tuple[str, ...],
    max_steps: int | None,
    gate_check_interval: int,
    patience: int,
    min_delta: float,
    min_steps: int,
    batch_size: int,
    gate_sample_count: int,
    architecture: ArchitectureManifest,
    training_counter: _ThroughputCounter,
    training_compute_counter: _ComputeCounter,
    validation_counter: _ThroughputCounter,
    phase_timings: TimingCollector,
    start_step: int = 0,
    start_check: int = 0,
    on_plateau: Callable[[tuple[TrainingHistoryPoint, ...]], bool] | None = None,
    on_gate_check: Callable[[tuple[TrainingHistoryPoint, ...], Any], None] | None = None,
) -> _TrainingStageResult:
    stage_started = time.perf_counter()
    validation_history: list[TrainingHistoryPoint] = []
    best_score = -float("inf")
    stale_checks = 0
    stop_reason = "training-stopped"
    plateau_window_start_index = 0
    plateau_window_start_step = start_step
    max_validation_inference_compute: int | None = None
    latest_training_compute_per_sample: int | None = None

    def append_validation(*, step: int, check: int) -> None:
        nonlocal best_score, max_validation_inference_compute, stale_checks
        validation_started = time.perf_counter()
        batch = validation_batch(check)
        with phase_timings.span("validation_max_inference_compute"):
            batch_max_inference_compute = _batch_max_inference_compute(
                architecture=architecture,
                batch=batch,
            )
        if batch_max_inference_compute is None:
            raise BenchmarkRunnerError(
                "training gate could not measure max_inference_compute"
            )
        max_validation_inference_compute = _max_optional_int(
            max_validation_inference_compute,
            batch_max_inference_compute,
        )
        if max_validation_inference_compute is None:
            raise BenchmarkRunnerError(
                "training gate could not measure running max_inference_compute"
            )
        with phase_timings.span("validation_tensor_batch", samples=batch.sample_count):
            fields, labels = _batch_tensors(
                runtime=runtime,
                batch=batch,
                outcome_ids=outcome_ids,
                device=runtime.device,
            )
        actual_gate_sample_count = _tensor_batch_size(fields, fallback=gate_sample_count)
        with phase_timings.span("validation_forward_loss", samples=actual_gate_sample_count):
            was_training = bool(module.training)
            module.eval()
            with no_grad_context(runtime):
                validation_loss = float(loss_function(module(fields), labels).item())
                probabilities = tuple(
                    _renormalized_probabilities(row)
                    for row in apply_softmax_predictions(runtime, module, fields)
                )
            if was_training:
                module.train()
        with phase_timings.span("validation_score_estimate", samples=batch.sample_count):
            score_estimate = _training_gate_score_estimate(
                batch=batch,
                outcome_space=outcome_space,
                probabilities=probabilities,
                validation_check=check,
                step=step,
                max_inference_compute=batch_max_inference_compute,
                running_max_inference_compute=max_validation_inference_compute,
                training_compute_per_sample=latest_training_compute_per_sample,
            )
        score = _training_score_estimate_score(score_estimate)
        if score > best_score + min_delta:
            best_score = score
            stale_checks = 0
        else:
            stale_checks += 1
        with phase_timings.span("validation_scheduler_step"):
            if scheduler is not None:
                scheduler.step_after_validation(-score)
                learning_rates = scheduler.learning_rates()
            else:
                learning_rates = tuple(float(group["lr"]) for group in optimizer.param_groups)
        validation_counter.add(
            seconds=time.perf_counter() - validation_started,
            samples=actual_gate_sample_count,
        )
        with phase_timings.span("validation_history_append"):
            validation_history.append(
                TrainingHistoryPoint(
                    step=step,
                    validation_check=check,
                    validation_loss=validation_loss,
                    stale_checks=stale_checks,
                    learning_rates=learning_rates,
                    score_estimate=score_estimate,
                )
            )
        if on_gate_check is not None:
            with phase_timings.span("validation_gate_callback"):
                on_gate_check(tuple(validation_history), module)
        phase_timings.add(
            "validation_gate_total",
            seconds=time.perf_counter() - validation_started,
            samples=actual_gate_sample_count,
        )

    append_validation(step=start_step, check=start_check)
    if max_steps == start_step:
        phase_timings.add(
            "training_stage_total",
            seconds=time.perf_counter() - stage_started,
        )
        return _TrainingStageResult(
            validation_history=tuple(validation_history),
            stop_reason="no-training-steps" if start_step == 0 else "max-steps",
        )
    validation_check = start_check + 1
    steps = count(start_step + 1) if max_steps is None else range(start_step + 1, max_steps + 1)
    for step in steps:
        training_started = time.perf_counter()
        fields, labels = train_batch(step)
        actual_batch_size = _tensor_batch_size(fields, fallback=batch_size)
        with phase_timings.span("training_max_training_compute"):
            batch_training_compute_per_sample = _batch_max_training_compute_per_sample(
                architecture=architecture,
                fields=fields,
            )
        latest_training_compute_per_sample = batch_training_compute_per_sample
        module.train()
        with phase_timings.span("training_zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        with phase_timings.span("training_forward_loss", samples=batch_size):
            loss = loss_function(module(fields), labels)
        with phase_timings.span("training_backward", samples=batch_size):
            loss.backward()
        with phase_timings.span("training_optimizer_step"):
            optimizer.step()
        if scheduler is not None:
            with phase_timings.span("training_scheduler_step"):
                scheduler.step_after_optimizer()
        training_elapsed = time.perf_counter() - training_started
        training_counter.add(
            seconds=training_elapsed,
            samples=actual_batch_size,
        )
        phase_timings.add(
            "training_step_total",
            seconds=training_elapsed,
            samples=actual_batch_size,
        )
        training_compute_counter.add(
            compute_per_sample=batch_training_compute_per_sample,
            samples=actual_batch_size,
        )
        hit_step_cap = max_steps is not None and step == max_steps
        if step % gate_check_interval != 0 and not hit_step_cap:
            continue
        append_validation(step=step, check=validation_check)
        validation_check += 1
        with phase_timings.span("validation_plateau_check"):
            should_stop_for_plateau = (
                patience > 0
                and step - plateau_window_start_step >= min_steps
                and has_windowed_validation_plateau(
                    validation_history[plateau_window_start_index:],
                    window_checks=patience,
                    min_delta=min_delta,
                )
                and (
                    scheduler is None
                    or scheduler.has_exhausted_plateau_response()
                )
            )
        if should_stop_for_plateau:
            with phase_timings.span("validation_plateau_handler"):
                advanced_frontier = (
                    on_plateau is not None and on_plateau(tuple(validation_history))
                )
            if advanced_frontier:
                plateau_window_start_index = len(validation_history) - 1
                plateau_window_start_step = step
                best_score = _training_history_score(validation_history[-1])
                stale_checks = 0
                continue
            stop_reason = "validation-plateau"
            break
        if max_steps is not None and step >= max_steps:
            stop_reason = "max-steps"
            break
    phase_timings.add(
        "training_stage_total",
        seconds=time.perf_counter() - stage_started,
    )
    return _TrainingStageResult(
        validation_history=tuple(validation_history),
        stop_reason=stop_reason,
    )


def _training_gate_score_estimate(
    *,
    batch: GeneratedSampleSet,
    outcome_space: OutcomeSpace,
    probabilities: tuple[tuple[float, ...], ...],
    validation_check: int,
    step: int,
    max_inference_compute: int,
    running_max_inference_compute: int,
    training_compute_per_sample: int | None,
) -> dict[str, object]:
    measurements = finite_measurements_for_predictions(
        batch=batch,
        outcome_space=outcome_space,
        probabilities=probabilities,
        run_slug=f"training-gate-{validation_check:04d}",
    )
    sampled_competence = sampled_competence_curriculum_record(
        (
            sampled_competence_record(
                batch=batch,
                measurements=measurements,
                complexity_axis=None,
            ),
        )
    )
    compact_sampled_competence = _compact_training_sampled_competence(sampled_competence)
    point_records = _training_score_estimate_points(compact_sampled_competence)
    chance_mass = _chance_accepted_mass(tuple(outcome.id for outcome in outcome_space.outcomes))
    score = sampled_competence_frontier_score(
        tuple(
            CompetencePoint(
                complexity=_required_float(point.get("complexity"), "score_estimate.complexity"),
                accepted_mass=_required_float(
                    point.get("mean_accepted_mass"),
                    "score_estimate.mean_accepted_mass",
                ),
            )
            for point in point_records
        ),
        chance_mass=chance_mass,
    )
    record: dict[str, object] = {
        "kind": "training-running-score-estimate",
        "status": "provisional",
        "evidence_status": "not-accepted",
        "score_frame": "none",
        "scoring_recipe": "sampled-competence-v1",
        "score": score,
        "validation_check": validation_check,
        "step": step,
        "max_inference_compute": max_inference_compute,
        "running_max_inference_compute": running_max_inference_compute,
        "chance_mass": chance_mass,
        "sampled_competence": compact_sampled_competence,
    }
    if training_compute_per_sample is not None:
        record["training_compute_per_sample"] = training_compute_per_sample
    return record


def _compact_training_sampled_competence(
    sampled_competence: Mapping[str, object],
) -> dict[str, object]:
    compact = {
        key: value
        for key, value in sampled_competence.items()
        if key not in {"measurement_ids", "observation_ids", "points"}
    }
    points = sampled_competence.get("points")
    if isinstance(points, Sequence) and not isinstance(points, str | bytes):
        compact_points: list[dict[str, object]] = []
        for raw_point in cast(Sequence[object], points):
            if not isinstance(raw_point, Mapping):
                continue
            point = cast(Mapping[str, object], raw_point)
            compact_points.append(
                {
                    key: value
                    for key, value in point.items()
                    if key not in {"measurement_ids", "observation_ids"}
                }
            )
        compact["points"] = compact_points
    return compact


def _training_score_estimate_points(
    score_estimate_or_sampled_competence: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    sampled_competence = score_estimate_or_sampled_competence.get("sampled_competence")
    record = (
        cast(Mapping[str, object], sampled_competence)
        if isinstance(sampled_competence, Mapping)
        else score_estimate_or_sampled_competence
    )
    points = record.get("points")
    if isinstance(points, list | tuple):
        raw_points = cast(Sequence[object], points)
        return tuple(
            cast(Mapping[str, object], point)
            for point in raw_points
            if isinstance(point, Mapping)
        )
    return (record,)


def _training_score_estimate_score(score_estimate: Mapping[str, object]) -> float:
    return _required_float(score_estimate.get("score"), "score_estimate.score")


def _training_history_score(point: TrainingHistoryPoint) -> float:
    if point.score_estimate is None:
        return 0.0
    return _training_score_estimate_score(point.score_estimate)


def _training_history_max_inference_compute(
    validation_history: Sequence[TrainingHistoryPoint],
) -> int | None:
    max_compute: int | None = None
    for point in validation_history:
        if point.score_estimate is None:
            continue
        value = point.score_estimate.get("running_max_inference_compute")
        if type(value) is int:
            max_compute = _max_optional_int(max_compute, value)
            continue
        current = point.score_estimate.get("max_inference_compute")
        if type(current) is int:
            max_compute = _max_optional_int(max_compute, current)
    return max_compute


def _training_history_latest_training_compute_per_sample(
    validation_history: Sequence[TrainingHistoryPoint],
) -> int | None:
    for point in reversed(validation_history):
        if point.score_estimate is None:
            continue
        value = point.score_estimate.get("training_compute_per_sample")
        if type(value) is int:
            return value
    return None


def _training_history_frontier_point(point: TrainingHistoryPoint) -> ValidationCompetencePoint:
    if point.score_estimate is None:
        raise BenchmarkRunnerError("training gate is missing score estimate")
    points = _training_score_estimate_points(point.score_estimate)
    if not points:
        raise BenchmarkRunnerError("training gate score estimate has no competence point")
    latest_point = points[-1]
    return ValidationCompetencePoint(
        complexity=_required_float(latest_point.get("complexity"), "score_estimate.complexity"),
        accepted_mass=_required_float(
            latest_point.get("mean_accepted_mass"),
            "score_estimate.mean_accepted_mass",
        ),
    )


def _tensor_batch_size(fields: Any, *, fallback: int) -> int:
    shape = getattr(fields, "shape", None)
    if shape is None or len(shape) < 1:
        return fallback
    value = shape[0]
    if type(value) is int and value >= 0:
        return value
    return fallback


def _frontier_plateau_advances(
    *,
    frontier_point: ValidationCompetencePoint,
    previous_frontier_points: tuple[ValidationCompetencePoint, ...],
    chance_mass: float,
) -> bool:
    return validation_competence_frontier_advances(
        frontier_point=frontier_point,
        previous_frontier_points=previous_frontier_points,
        chance_mass=chance_mass,
    )


def _evaluation_result_frontier_index(
    *,
    evaluation_results: Sequence[
        tuple[_CurriculumRung, tuple[tuple[float, ...], ...]]
    ],
    outcome_ids: tuple[str, ...],
) -> int:
    if not evaluation_results:
        raise BenchmarkRunnerError("evaluation did not produce any rungs")
    chance_mass = _chance_accepted_mass(outcome_ids)
    frontier_index = 0
    for index, (rung, probabilities) in enumerate(evaluation_results):
        accepted_mass = _mean_prediction_accepted_mass(
            batch=rung.batch,
            probabilities=probabilities,
            outcome_ids=outcome_ids,
        )
        if accepted_mass > chance_mass + 1e-12:
            frontier_index = index
    return frontier_index


def has_windowed_validation_plateau(
    validation_history: Sequence[TrainingHistoryPoint],
    *,
    window_checks: int,
    min_delta: float,
) -> bool:
    if window_checks <= 0 or len(validation_history) <= window_checks:
        return False
    current = validation_history[-1]
    window_start = validation_history[-1 - window_checks]
    return _training_history_score(current) - _training_history_score(window_start) < min_delta


def _training_run_record(
    *,
    seed: int,
    batch_size: int,
    max_steps: int | None,
    learning_rate: float,
    optimizer_name: str,
    schedule_name: str,
    gate_check_interval: int,
    gate_sample_count: int,
    gate_decision_rule: str,
    convergence_patience: int,
    convergence_min_delta: float,
    convergence_min_steps: int,
    tensor_device: str,
    validation_history: tuple[TrainingHistoryPoint, ...],
    stop_reason: str,
    training_compute: float | None,
) -> TrainingRunRecord:
    last_step = validation_history[-1].step
    if stop_reason == "no-training-steps":
        status = "completed"
    elif stop_reason == "validation-plateau":
        status = "converged"
    elif stop_reason == "max-steps":
        status = "budget-exhausted"
    else:
        status = "completed"
    return TrainingRunRecord(
        status=status,
        stop_reason=stop_reason,
        steps_run=last_step,
        training_compute=training_compute,
        validation_checks=len(validation_history),
        protocol=TrainingProtocol(
            kind="fixed-step-local-batch",
            objective="cross-entropy",
            optimizer=cast(Any, optimizer_name),
            learning_rate=learning_rate,
            schedule=cast(Any, schedule_name),
            seed=seed,
            batch_size=batch_size,
            max_steps=max_steps,
            gate_check_interval=gate_check_interval,
            gate_sample_count=gate_sample_count,
            gate_decision_rule=gate_decision_rule,
            min_delta=convergence_min_delta,
            patience=convergence_patience,
            validation_source="generator-resample",
            min_steps=convergence_min_steps,
            tensor_runtime="pytorch",
            tensor_device=tensor_device,
        ),
        validation_history=validation_history,
    )


def _running_training_run_record(
    *,
    seed: int,
    batch_size: int,
    max_steps: int | None,
    learning_rate: float,
    optimizer_name: str,
    schedule_name: str,
    gate_check_interval: int,
    gate_sample_count: int,
    gate_decision_rule: str,
    convergence_patience: int,
    convergence_min_delta: float,
    convergence_min_steps: int,
    tensor_device: str,
    validation_history: tuple[TrainingHistoryPoint, ...],
    training_compute: float | None,
) -> TrainingRunRecord:
    return TrainingRunRecord(
        status="running",
        stop_reason="validation-checkpoint",
        steps_run=validation_history[-1].step,
        training_compute=training_compute,
        validation_checks=len(validation_history),
        protocol=TrainingProtocol(
            kind="fixed-step-local-batch",
            objective="cross-entropy",
            optimizer=cast(Any, optimizer_name),
            learning_rate=learning_rate,
            schedule=cast(Any, schedule_name),
            seed=seed,
            batch_size=batch_size,
            max_steps=max_steps,
            gate_check_interval=gate_check_interval,
            gate_sample_count=gate_sample_count,
            gate_decision_rule=gate_decision_rule,
            min_delta=convergence_min_delta,
            patience=convergence_patience,
            validation_source="generator-resample",
            min_steps=convergence_min_steps,
            tensor_runtime="pytorch",
            tensor_device=tensor_device,
        ),
        validation_history=validation_history,
    )


def _training_progress_record(
    *,
    plan: BenchmarkRunPlan,
    summary: BenchmarkRunSummary,
    architecture: ArchitectureManifest,
    inspection: ModelInspectionRecord,
    evaluation_curriculum: Mapping[str, object],
    training_curriculum: Mapping[str, object],
    training_run: TrainingRunRecord,
    throughput: Mapping[str, object],
    model_checkpoints: tuple[Mapping[str, object], ...],
    selected_model_checkpoint: Mapping[str, object] | None,
) -> Mapping[str, object]:
    training_estimate = _training_estimate_record(
        summary=summary,
        training_run=training_run,
    )
    record: dict[str, object] = {
        "format": _progress_format,
        "format_version": _progress_format_version,
        "run_slug": summary.run_slug,
        "run_status": "running",
        "benchmark_id": str(summary.benchmark_id),
        "architecture_path": summary.architecture_path.as_posix(),
        "sample_count": plan.sample_count,
        "evaluation_sample_count": plan.resolved_evaluation_sample_count,
        "seed": plan.seed,
        "train_steps": plan.train_steps,
        "learning_rate": float(plan.learning_rate),
        "optimizer": plan.optimizer,
        "schedule": plan.schedule,
        "gate_check_interval": plan.gate_check_interval,
        "model_checkpoint_gate_interval": plan.model_checkpoint_gate_interval,
        "gate_sample_count": plan.resolved_gate_sample_count,
        "gate_decision_rule": plan.gate_decision_rule,
        "convergence_patience": plan.convergence_patience,
        "convergence_min_delta": float(plan.convergence_min_delta),
        "convergence_min_steps": plan.convergence_min_steps,
        "tensor_runtime": "pytorch",
        "tensor_device": training_run.protocol.tensor_device,
        "training_run": training_run.to_record(),
        "throughput": throughput,
        "evaluation_curriculum": dict(evaluation_curriculum),
        "training_curriculum": dict(training_curriculum),
        "architecture": inspection.architecture.to_record(),
        "cost_summary": _training_cost_summary(
            inspection=inspection,
            training_run=training_run,
        ),
        "model_inspection": inspection.to_record(),
        "architecture_digest": str(architecture.digest),
        "model_inspection_digest": str(inspection.digest),
        "provisional_score": training_estimate["score"],
        "training_estimate": training_estimate,
        "model_checkpoints": [dict(checkpoint) for checkpoint in model_checkpoints],
        "selected_model_checkpoint": (
            None if selected_model_checkpoint is None else dict(selected_model_checkpoint)
        ),
        "selected_model_checkpoint_policy": "highest-training-score-estimate",
    }
    record["sampled_competence"] = dict(
        cast(Mapping[str, object], training_estimate["sampled_competence"])
    )
    return record


def _training_estimate_record(
    *,
    summary: BenchmarkRunSummary,
    training_run: TrainingRunRecord,
) -> dict[str, object]:
    if not training_run.validation_history:
        raise BenchmarkRunnerError("training estimate requires validation history")
    latest = training_run.validation_history[-1]
    if latest.score_estimate is None:
        raise BenchmarkRunnerError("training estimate requires gate score estimate")
    score_estimate = latest.score_estimate
    sampled_competence_value = score_estimate.get("sampled_competence")
    if not isinstance(sampled_competence_value, Mapping):
        raise BenchmarkRunnerError("training gate score estimate is missing sampled competence")
    sampled_competence = cast(Mapping[str, object], sampled_competence_value)
    complexity = _required_float(
        sampled_competence.get("complexity"),
        "training_estimate.complexity",
    )
    mean_accepted_mass = _required_float(
        sampled_competence.get("mean_accepted_mass"),
        "training_estimate.mean_accepted_mass",
    )
    sample_count = _required_int(
        sampled_competence.get("sample_count"),
        "training_estimate.sample_count",
    )
    points = _training_score_estimate_points(sampled_competence)
    if not points:
        raise BenchmarkRunnerError("training estimate requires sampled competence point")
    seed = _required_int(points[0].get("seed"), "training_estimate.seed")
    return {
        "kind": "training-running-score-estimate",
        "status": "tentative",
        "evidence_status": "not-accepted",
        "score_frame": "none",
        "score_basis": "provisional-sampled-competence",
        "scoring_recipe": score_estimate.get("scoring_recipe", "sampled-competence-v1"),
        "benchmark_id": str(summary.benchmark_id),
        "complexity_axis": None,
        "complexity": complexity,
        "seed": seed,
        "sample_count": sample_count,
        "mean_accepted_mass": mean_accepted_mass,
        "score": _training_score_estimate_score(score_estimate),
        "validation_check": latest.validation_check,
        "step": latest.step,
        "max_inference_compute": _required_int(
            score_estimate.get("running_max_inference_compute"),
            "training_estimate.max_inference_compute",
        ),
        "sampled_competence": dict(sampled_competence),
    }


def _training_cost_summary(
    *,
    inspection: ModelInspectionRecord,
    training_run: TrainingRunRecord,
) -> dict[str, object]:
    cost_summary = inspection.cost_summary.to_record()
    cost_summary.pop("inference_compute", None)
    cost_summary.pop("training_compute_per_sample", None)
    max_inference_compute = _training_history_max_inference_compute(
        training_run.validation_history
    )
    if max_inference_compute is not None:
        cost_summary["inference_compute"] = max_inference_compute
    if training_run.training_compute is not None:
        cost_summary["training_compute"] = training_run.training_compute
    return cost_summary


def _should_write_model_checkpoint(
    *,
    training_run: TrainingRunRecord,
    gate_interval: int,
    checkpoint_artifacts: tuple[ModelCheckpointArtifact, ...],
) -> bool:
    if gate_interval < 1:
        raise BenchmarkRunnerError("model checkpoint gate interval must be positive")
    latest = training_run.validation_history[-1]
    if not checkpoint_artifacts:
        return True
    latest_score = _training_history_score(latest)
    saved_score = max(
        _checkpoint_estimated_score(checkpoint) for checkpoint in checkpoint_artifacts
    )
    return latest_score > saved_score


def _write_model_checkpoint_artifact(
    *,
    summary: BenchmarkRunSummary,
    architecture: ArchitectureManifest,
    model_interface: ModelInterface,
    training_run: TrainingRunRecord,
    module: Any,
    runtime: str,
) -> ModelCheckpointArtifact:
    latest = training_run.validation_history[-1]
    stem = f"gate{latest.validation_check:04d}-step{latest.step:08d}"
    checkpoint_path = summary.model_artifact_root / f"{stem}.pt"
    _write_torch_checkpoint(checkpoint_path, module=module)
    checkpoint_digest = _file_content_digest(checkpoint_path)
    checkpoint_reference = ArtifactReference(
        kind="model-checkpoint",
        content_digest=checkpoint_digest,
    )
    manifest = ModelArtifactManifest(
        id=ProtocolIdentifier.parse(
            f"model-manifests.{_identifier_atom(summary.benchmark_id)}."
            f"{summary.run_slug}.{stem}@0.1.0"
        ),
        architecture=reference_for_record(
            kind="architecture-manifest",
            record=architecture.to_record(),
        ),
        interface=reference_for_record(
            kind="model-interface",
            record=model_interface.to_record(),
        ),
        execution_family=(
            ModelExecutionFamily.reference_runner_pytorch_sequential()
            if runtime == "pytorch"
            else ModelExecutionFamily(
                kind=f"local-{runtime}",
                runtime=runtime,
                architecture_family="sequential-architecture-components",
            )
        ),
        model_artifacts=(checkpoint_reference,),
        training_provenance=(
            ArtifactReference(
                kind="training-run",
                record_digest=ContentDigest.from_value(training_run.to_record()),
            ),
        ),
    )
    manifest_path = summary.model_artifact_root / f"{stem}.model{_document_suffix}"
    _write_document(manifest_path, manifest.to_record())
    return ModelCheckpointArtifact(
        path=checkpoint_path,
        digest=checkpoint_digest,
        manifest_path=manifest_path,
        manifest_digest=manifest.digest,
        manifest=manifest,
        step=latest.step,
        validation_check=latest.validation_check,
        validation_loss=latest.validation_loss,
        score_estimate=latest.score_estimate,
    )


def _write_torch_checkpoint(path: Path, *, module: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    state = {
        "format": "leibniz.model-checkpoint.pytorch-state-dict",
        "format_version": 1,
        "state_dict": _portable_state_dict(module),
    }
    save_tensor_runtime_state(temporary, state)
    temporary.replace(path)


def _load_torch_checkpoint(
    *,
    module: Any,
    runtime: TensorRuntime,
    checkpoint: ModelCheckpointArtifact,
) -> None:
    if _file_content_digest(checkpoint.path) != checkpoint.digest:
        raise BenchmarkRunnerError(f"model checkpoint digest mismatch: {checkpoint.path}")
    payload = load_tensor_runtime_state(runtime, checkpoint.path, weights_only=False)
    if not isinstance(payload, Mapping):
        raise BenchmarkRunnerError("model checkpoint payload must be a mapping")
    payload_record = cast(Mapping[str, object], payload)
    if payload_record.get("format") != "leibniz.model-checkpoint.pytorch-state-dict":
        raise BenchmarkRunnerError("model checkpoint has unsupported format")
    if payload_record.get("format_version") != 1:
        raise BenchmarkRunnerError("model checkpoint has unsupported format_version")
    state_dict = payload_record.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise BenchmarkRunnerError("model checkpoint state_dict must be a mapping")
    module.load_state_dict(dict(cast(Mapping[str, object], state_dict)))


def _portable_state_dict(module: Any) -> Mapping[str, object]:
    state: dict[str, object] = {}
    for key, value in module.state_dict().items():
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        state[str(key)] = value
    return state


def _file_content_digest(path: Path) -> ContentDigest:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return ContentDigest(algorithm="sha256", hex=digest)


def _selected_model_checkpoint(
    checkpoints: tuple[ModelCheckpointArtifact, ...],
) -> ModelCheckpointArtifact | None:
    if not checkpoints:
        return None
    return max(
        checkpoints,
        key=lambda checkpoint: (
            _checkpoint_estimated_score(checkpoint),
            -checkpoint.validation_loss,
            checkpoint.validation_check,
            checkpoint.step,
        ),
    )


def _checkpoint_estimated_score(checkpoint: ModelCheckpointArtifact) -> float:
    if checkpoint.score_estimate is None:
        return -float("inf")
    return _training_score_estimate_score(checkpoint.score_estimate)


def _throughput_record(
    *,
    runtime_device: str,
    training_counter: _ThroughputCounter,
    validation_counter: _ThroughputCounter,
    evaluation_counter: _ThroughputCounter,
    competition_counter: _ThroughputCounter | None,
    roofline: Mapping[str, object],
    work_estimates: _TrainingWorkEstimates | None,
    phase_timings: TimingCollector,
    fallback_errors: tuple[tuple[str, str], ...] = (),
    operation_fallbacks: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    training = training_counter.to_record(kind="training-throughput")
    validation = validation_counter.to_record(kind="validation-throughput")
    evaluation = evaluation_counter.to_record(kind="evaluation-throughput")
    competition = (
        None
        if competition_counter is None
        else competition_counter.to_record(kind="competition-throughput")
    )
    record: dict[str, object] = {
        "kind": "benchmark-throughput",
        "tensor_runtime": "pytorch",
        "tensor_device": runtime_device,
        "training": training,
        "validation": validation,
        "evaluation": evaluation,
        "phase_timing": phase_timings.to_record(kind="benchmark-phase-timing"),
        "roofline": dict(roofline),
        "roofline_comparison": _roofline_comparison(
            training=training,
            validation=validation,
            evaluation=evaluation,
            roofline=roofline,
            work_estimates=work_estimates,
        ),
    }
    if competition is not None:
        record["competition"] = competition
    if fallback_errors:
        record["runtime_fallbacks"] = [
            {
                "from_device": device_kind,
                "to_device": runtime_device,
                "reason": reason,
            }
            for device_kind, reason in fallback_errors
        ]
    if operation_fallbacks:
        record["operation_runtime_fallbacks"] = [
            dict(fallback) for fallback in operation_fallbacks
        ]
    return record


def _throughput_with_progress_timings(
    *,
    throughput: Mapping[str, object],
    progress_timings: TimingCollector,
) -> dict[str, object]:
    record = dict(throughput)
    phase_timing = dict(cast(Mapping[str, object], record.get("phase_timing", {})))
    phases = dict(cast(Mapping[str, object], phase_timing.get("phases", {})))
    progress_record = progress_timings.to_record(kind="training-progress-phase-timing")
    phases.update(cast(Mapping[str, object], progress_record["phases"]))
    phase_timing["phases"] = phases
    record["phase_timing"] = phase_timing
    return record


def _roofline_comparison(
    *,
    training: Mapping[str, object],
    validation: Mapping[str, object],
    evaluation: Mapping[str, object],
    roofline: Mapping[str, object],
    work_estimates: _TrainingWorkEstimates | None,
) -> dict[str, object]:
    peak_compute = roofline.get("peak_compute_per_second")
    peak_bytes = roofline.get("peak_bytes_per_second")
    if (
        not isinstance(peak_compute, int | float)
        or not math.isfinite(float(peak_compute))
        or peak_compute <= 0
        or not isinstance(peak_bytes, int | float)
        or not math.isfinite(float(peak_bytes))
        or peak_bytes <= 0
        or work_estimates is None
    ):
        return {
            "status": "unavailable",
            "reason": roofline.get(
                "reason",
                "system roofline or per-phase work estimates are unavailable",
            ),
        }
    peak_compute_value = float(peak_compute)
    peak_bytes_value = float(peak_bytes)
    training_phase = _phase_roofline_record(
        throughput=training,
        work=work_estimates.training,
        peak_compute_per_second=peak_compute_value,
        peak_bytes_per_second=peak_bytes_value,
    )
    validation_phase = _phase_roofline_record(
        throughput=validation,
        work=work_estimates.validation,
        peak_compute_per_second=peak_compute_value,
        peak_bytes_per_second=peak_bytes_value,
    )
    evaluation_phase = _phase_roofline_record(
        throughput=evaluation,
        work=work_estimates.evaluation,
        peak_compute_per_second=peak_compute_value,
        peak_bytes_per_second=peak_bytes_value,
    )
    return {
        "status": "available",
        "model": "operational-intensity",
        "peak_compute_per_second": peak_compute_value,
        "peak_bytes_per_second": peak_bytes_value,
        "phases": {
            "training": training_phase,
            "validation": validation_phase,
            "evaluation": evaluation_phase,
        },
        "assumptions": list(work_estimates.assumptions),
        "training_fraction_of_roofline": training_phase["fraction_of_roofline"],
        "validation_fraction_of_roofline": validation_phase["fraction_of_roofline"],
        "evaluation_fraction_of_roofline": evaluation_phase["fraction_of_roofline"],
    }


def _phase_roofline_record(
    *,
    throughput: Mapping[str, object],
    work: _PhaseWorkEstimate,
    peak_compute_per_second: float,
    peak_bytes_per_second: float,
) -> dict[str, object]:
    arithmetic_intensity = work.compute_per_sample / work.bytes_per_sample
    expected_compute = min(
        peak_compute_per_second,
        peak_bytes_per_second * arithmetic_intensity,
    )
    measured = throughput.get("samples_per_second")
    observed_compute = 0.0
    if not isinstance(measured, int | float) or not math.isfinite(float(measured)):
        measured_samples = 0.0
    else:
        measured_samples = float(measured)
        observed_compute = measured_samples * work.compute_per_sample
    return {
        "compute_per_sample": work.compute_per_sample,
        "bytes_per_sample": work.bytes_per_sample,
        "arithmetic_intensity_compute_per_byte": arithmetic_intensity,
        "expected_roofline_compute_per_second": expected_compute,
        "observed_compute_per_second": observed_compute,
        "fraction_of_roofline": (
            observed_compute / expected_compute if expected_compute > 0 else 0.0
        ),
        "samples_per_second": measured_samples,
        "limiting_resource": (
            "memory-bandwidth"
            if peak_bytes_per_second * arithmetic_intensity < peak_compute_per_second
            else "compute"
        ),
    }


def _training_work_estimates(
    *,
    architecture: ArchitectureManifest,
    inference_compute: int | None,
    training_compute_per_sample: int | None,
    storage_bytes: int | None,
    batch_size: int,
) -> _TrainingWorkEstimates | None:
    if (
        inference_compute is None
        or inference_compute <= 0
        or training_compute_per_sample is None
        or training_compute_per_sample <= 0
    ):
        return None
    input_bytes = _shape_bytes(architecture.input_shape)
    output_bytes = _shape_bytes(architecture.output_shape)
    batch_size_value = max(1, batch_size)
    storage_bytes_per_sample = float(storage_bytes or 0) / batch_size_value
    inference_bytes = input_bytes + output_bytes + storage_bytes_per_sample
    formation_bytes = 8.0 * input_bytes
    training_compute = float(training_compute_per_sample)
    training_bytes = formation_bytes + 3.0 * inference_bytes + 4.0 * storage_bytes_per_sample
    validation_bytes = formation_bytes + inference_bytes
    evaluation_bytes = input_bytes + inference_bytes
    return _TrainingWorkEstimates(
        training=_PhaseWorkEstimate(
            compute_per_sample=training_compute,
            bytes_per_sample=training_bytes,
        ),
        validation=_PhaseWorkEstimate(
            compute_per_sample=float(inference_compute),
            bytes_per_sample=validation_bytes,
        ),
        evaluation=_PhaseWorkEstimate(
            compute_per_sample=float(inference_compute),
            bytes_per_sample=evaluation_bytes,
        ),
        assumptions=(
            "float32 tensor elements are four bytes",
            "training compute follows the declared per-operator training trace",
            "storage bytes are amortized across the local batch",
            "formation bytes are approximated as eight input fields per sample",
            "optimizer and gradient traffic are approximated from storage bytes",
            "cache reuse and PyTorch dispatch overhead are not modeled as protocol semantics",
        ),
    )


def _shape_bytes(shape: Sequence[int]) -> float:
    element_count = 1
    for axis in shape:
        element_count *= axis
    return float(element_count * 4)


def _make_optimizer(
    *,
    runtime: TensorRuntime,
    parameters: Any,
    name: str,
    learning_rate: float,
) -> Any:
    try:
        return build_optimizer(
            runtime, name=name, parameters=parameters, learning_rate=learning_rate
        )
    except TensorRuntimeError as error:
        raise BenchmarkRunnerError(str(error)) from error


def _make_scheduler(
    *,
    runtime: TensorRuntime,
    optimizer: Any,
    name: str,
    max_steps: int | None,
    min_delta: float,
    patience: int,
) -> _LearningRateSchedule | None:
    if name == "none":
        return None
    if name == "cosine":
        if max_steps is None:
            raise BenchmarkRunnerError("cosine schedule requires train_steps")
        return _LearningRateSchedule(
            scheduler=build_cosine_lr_schedule(runtime, optimizer, T_max=max(1, max_steps)),
            optimizer=optimizer,
            update_on="optimizer-step",
            base_learning_rates=tuple(float(group["lr"]) for group in optimizer.param_groups),
        )
    if name == "reduce-on-plateau":
        factor = 0.1
        eps = 1e-8
        return _LearningRateSchedule(
            scheduler=build_plateau_lr_schedule(
                runtime,
                optimizer,
                factor=factor,
                threshold=min_delta,
                patience=patience,
                eps=eps,
            ),
            optimizer=optimizer,
            update_on="score-estimate",
            minimum_effective_learning_rate=eps / (1.0 - factor),
            base_learning_rates=tuple(float(group["lr"]) for group in optimizer.param_groups),
        )
    raise BenchmarkRunnerError(f"unsupported schedule: {name}")


def _batch_tensors(
    *,
    runtime: TensorRuntime,
    batch: GeneratedSampleSet,
    outcome_ids: tuple[str, ...],
    device: Any,
) -> tuple[Any, Any]:
    if batch.fields is not None and batch.targets is not None:
        return batch.fields, batch.targets
    return (
        _batch_tensor(runtime=runtime, batch=batch, device=device),
        _batch_target_tensor(
            runtime=runtime,
            batch=batch,
            outcome_ids=outcome_ids,
            device=device,
        ),
    )


def _batch_tensor(*, runtime: TensorRuntime, batch: GeneratedSampleSet, device: Any) -> Any:
    values = [list(sample.require_field().values) for sample in batch.samples]
    fields = make_float_tensor(runtime, values, device=device)
    return fields.reshape((len(batch.samples), *batch.samples[0].require_field().shape))


def _batch_target_tensor(
    *,
    runtime: TensorRuntime,
    batch: GeneratedSampleSet,
    outcome_ids: tuple[str, ...],
    device: Any,
) -> Any:
    if any(sample.target_distribution is not None for sample in batch.samples):
        return make_float_tensor(
            runtime,
            [
                _target_distribution_row(
                    sample.target_distribution_or_one_hot(),
                    outcome_ids=outcome_ids,
                )
                for sample in batch.samples
            ],
            device=device,
        )
    return make_long_tensor(
        runtime,
        [outcome_ids.index(sample.outcome_id) for sample in batch.samples],
        device=device,
    )


def _target_distribution_row(
    distribution: Mapping[str, float],
    *,
    outcome_ids: tuple[str, ...],
) -> list[float]:
    unknown = tuple(outcome_id for outcome_id in distribution if outcome_id not in outcome_ids)
    if unknown:
        raise BenchmarkRunnerError(
            f"target_distribution contains unknown outcome id: {unknown[0]}"
        )
    return [float(distribution.get(outcome_id, 0.0)) for outcome_id in outcome_ids]


def _renormalized_probabilities(probabilities: Sequence[float]) -> tuple[float, ...]:
    total = sum(float(probability) for probability in probabilities)
    if total <= 0:
        raise BenchmarkRunnerError("model probabilities must contain positive mass")
    normalized = [max(0.0, float(probability) / total) for probability in probabilities]
    if len(normalized) == 1:
        return (1.0,)
    normalized[-1] = max(0.0, 1.0 - sum(normalized[:-1]))
    return tuple(normalized)


def _mean_prediction_accepted_mass(
    *,
    batch: GeneratedSampleSet,
    probabilities: tuple[tuple[float, ...], ...],
    outcome_ids: tuple[str, ...],
) -> float:
    accepted_mass = tuple(
        _prediction_target_mass(
            row,
            target_distribution=sample.target_distribution_or_one_hot(),
            outcome_ids=outcome_ids,
        )
        for sample, row in zip(batch.samples, probabilities, strict=True)
    )
    return math.fsum(accepted_mass) / len(accepted_mass)


def _prediction_target_mass(
    probabilities: Sequence[float],
    *,
    target_distribution: Mapping[str, float],
    outcome_ids: tuple[str, ...],
) -> float:
    outcome_indexes = {outcome_id: index for index, outcome_id in enumerate(outcome_ids)}
    return math.fsum(
        float(target_probability) * float(probabilities[outcome_indexes[outcome_id]])
        for outcome_id, target_probability in target_distribution.items()
    )


def _checkpoint_competition_record(
    *,
    batch: GeneratedSampleSet,
    left_probabilities: tuple[tuple[float, ...], ...],
    right_probabilities: tuple[tuple[float, ...], ...],
    outcome_space: OutcomeSpace,
    left_model_key: str,
    right_model_key: str,
    benchmark_id: ProtocolIdentifier,
    competition_id: str,
) -> dict[str, object]:
    outcome_ids = tuple(outcome.id for outcome in outcome_space.outcomes)
    entries: list[dict[str, object]] = []
    left_wins = 0
    right_wins = 0
    ties = 0
    for sample, left_row, right_row in zip(
        batch.samples,
        left_probabilities,
        right_probabilities,
        strict=True,
    ):
        target_distribution = sample.target_distribution_or_one_hot()
        left_score = _prediction_target_mass(
            left_row,
            target_distribution=target_distribution,
            outcome_ids=outcome_ids,
        )
        right_score = _prediction_target_mass(
            right_row,
            target_distribution=target_distribution,
            outcome_ids=outcome_ids,
        )
        if left_score > right_score:
            winner = "left"
            left_wins += 1
        elif right_score > left_score:
            winner = "right"
            right_wins += 1
        else:
            winner = "tie"
            ties += 1
        entries.append(
            {
                "id": (
                    f"benchmarks.{_identifier_atom(benchmark_id)}.competition."
                    f"{competition_id}.sample-{sample.index}@0.1.0"
                ),
                "observation_id": str(sample.field_record().id),
                "accepted_outcome_id": sample.outcome_id,
                "target_distribution": [
                    {"outcome_id": outcome_id, "probability": probability}
                    for outcome_id, probability in target_distribution.items()
                ],
                "left_score": left_score,
                "right_score": right_score,
                "winner": winner,
            }
        )
    sample_count = len(entries)
    left_score = 0.0 if sample_count == 0 else (left_wins + 0.5 * ties) / sample_count
    return {
        "format": "leibniz.model-competition",
        "format_version": 1,
        "benchmark_id": str(benchmark_id),
        "competition_id": competition_id,
        "mechanic": "paired-prediction-accepted-mass",
        "seed": batch.seed,
        "sample_count": sample_count,
        "outcome_space_id": str(outcome_space.id),
        "left_model_key": left_model_key,
        "right_model_key": right_model_key,
        "left_score": left_score,
        "right_score": 1.0 - left_score,
        "left_wins": left_wins,
        "right_wins": right_wins,
        "ties": ties,
        "entries": entries,
    }


def _chance_accepted_mass(outcome_ids: tuple[str, ...]) -> float:
    if not outcome_ids:
        raise BenchmarkRunnerError("outcome space must contain at least one outcome")
    return 1.0 / len(outcome_ids)


def _write_document(path: Path, record: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_document_bytes(record))


def _write_document_atomic(path: Path, record: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_document_bytes(record))
    temporary.replace(path)


def _identifier_atom(identifier: ProtocolIdentifier) -> str:
    return str(identifier.name).rsplit(".", maxsplit=1)[-1]
