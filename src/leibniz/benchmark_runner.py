"""Small benchmark execution workflows for local operator runs."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import count
from pathlib import Path
from typing import Any, cast

from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.benchmark_evaluation import (
    finite_measurements_for_predictions,
    sampled_competence_curriculum_record,
    sampled_competence_record,
    validation_competence,
)
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes, document_filename_suffix
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import AxisAssignment
from leibniz.measurements import MeasurementDataset
from leibniz.model_inspection import ModelInspectionRecord
from leibniz.model_operators import ExecutableModelOperator
from leibniz.observation_generation import (
    GeneratedObservationBatch,
    ObservationGenerator,
    load_observation_generator,
)
from leibniz.outcomes import OutcomeSpace
from leibniz.tensor_runtime import (
    FormationTensorCache,
    OperationFallbackSequential,
    TensorRuntimeDevice,
    TensorRuntimeDeviceKind,
    TensorRuntimeError,
    resolve_tensor_runtime,
    runtime_roofline_record,
    tensor_runtime_device_kinds,
    validate_tensor_runtime_device,
)
from leibniz.timing import TimingCollector
from leibniz.training_runs import TrainingHistoryPoint, TrainingProtocol, TrainingRunRecord

__all__ = [
    "BenchmarkRunnerError",
    "BenchmarkRunPlan",
    "BenchmarkRunSummary",
    "has_windowed_validation_plateau",
    "run_benchmark",
    "training_stage_converged",
]

_document_suffix = document_filename_suffix()
_progress_format = "leibniz.benchmark-training-progress"
_progress_format_version = 1
_default_sample_count = 512
_default_train_steps: int | None = None
_default_checkpoint_interval = 256
_default_gate_check_interval = 32
_default_convergence_patience = 6
_default_convergence_min_delta = 1e-3
_default_convergence_min_steps = 500
_component_count = 1
_converged_training_stage_stop_reasons = frozenset({"validation-plateau"})
_minimum_plateau_lr_reductions = 3
_nuisance_extent_curriculum = (0.0, 0.25, 0.5, 1.0)
_canvas_logarithmic_growth_factor = math.sqrt(2.0)


@dataclass(frozen=True, slots=True)
class _TrainingResult:
    evaluation_results: tuple[
        tuple[_CurriculumRung, tuple[tuple[float, ...], ...]],
        ...,
    ]
    training_run: TrainingRunRecord
    throughput: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _TrainingStageResult:
    validation_history: tuple[TrainingHistoryPoint, ...]
    stop_reason: str


@dataclass(frozen=True, slots=True)
class _CurriculumRung:
    index: int
    nuisance_extent: float
    resolution_assignment: AxisAssignment
    seed: int
    batch: GeneratedObservationBatch

    @property
    def complexity(self) -> float:
        return self.batch.samples[0].complexity

    def to_record(self, *, status: str) -> dict[str, object]:
        return {
            "index": self.index,
            "status": status,
            "resolution_assignment": self.resolution_assignment.to_record(),
            "nuisance_extent": self.nuisance_extent,
            "seed": self.seed,
            "complexity_axis": "internal-distinguishable-state-complexity",
            "complexity": self.complexity,
            "sample_count": len(self.batch.samples),
        }


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

    def step_after_validation(self, validation_loss: float) -> None:
        if self.update_on == "validation-loss":
            before = self.learning_rates()
            self.scheduler.step(validation_loss)
            after = self.learning_rates()
            if any(new < old for old, new in zip(before, after, strict=True)):
                self.lr_reduction_count += 1

    def has_exhausted_plateau_response(self) -> bool:
        if self.update_on != "validation-loss":
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
    checkpoint_interval: int = _default_checkpoint_interval
    gate_check_interval: int = _default_gate_check_interval
    gate_sample_count: int | None = None
    gate_decision_rule: str = "validation-loss-plateau"
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
        if type(self.checkpoint_interval) is not int or self.checkpoint_interval < 1:
            raise BenchmarkRunnerError("checkpoint_interval must be a positive integer")
        if type(self.gate_check_interval) is not int or self.gate_check_interval < 1:
            raise BenchmarkRunnerError("gate_check_interval must be a positive integer")
        if self.checkpoint_interval % self.gate_check_interval != 0:
            raise BenchmarkRunnerError(
                "checkpoint_interval must be an integer multiple of gate_check_interval"
            )
        if self.gate_sample_count is not None and (
            type(self.gate_sample_count) is not int or self.gate_sample_count < 1
        ):
            raise BenchmarkRunnerError("gate_sample_count must be a positive integer")
        if self.gate_decision_rule != "validation-loss-plateau":
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
            f"c{_component_count}-seed{self.seed}-samples{self.sample_count}"
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
            "checkpoint_interval": self.checkpoint_interval,
            "gate_check_interval": self.gate_check_interval,
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
    measurement_dataset_path: Path
    model_inspection_path: Path
    training_summary_path: Path
    dry_run: bool

    def to_record(self) -> dict[str, object]:
        """Return a canonical document-friendly summary record."""

        return {
            "format": "leibniz.benchmark-run",
            "format_version": 1,
            "run_slug": self.run_slug,
            "benchmark_id": str(self.benchmark_id),
            "architecture_path": self.architecture_path.as_posix(),
            "measurement_count": self.measurement_count,
            "measurement_dataset_path": self.measurement_dataset_path.as_posix(),
            "model_inspection_path": self.model_inspection_path.as_posix(),
            "training_summary_path": self.training_summary_path.as_posix(),
            "dry_run": self.dry_run,
        }


def run_benchmark(
    plan: BenchmarkRunPlan,
    *,
    progress_callback: Callable[[BenchmarkRunSummary], None] | None = None,
) -> BenchmarkRunSummary:
    """Run or dry-run a tiny local benchmark workflow."""

    generator = load_observation_generator(plan.benchmark_root)
    architecture = ArchitectureManifestDocument.from_bytes(
        plan.architecture_path.read_bytes()
    ).manifest
    outcome_space = generator.benchmark_manifest.resolve_outcome_space()
    initial_evaluation_rung = _evaluation_curriculum_rung(
        architecture=architecture,
        generator=generator,
        component_count=_component_count,
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
        benchmark_id=generator.benchmark_manifest.id,
        architecture_digest=architecture.digest,
    )
    model_inspection = ModelInspectionRecord.from_architecture(
        id=ProtocolIdentifier.parse(
            f"model-inspections.{_identifier_atom(generator.benchmark_manifest.id)}."
            f"{summary.run_slug}@0.1.0"
        ),
        architecture_manifest=architecture,
    )
    if plan.dry_run:
        return summary

    progress_path = _training_progress_path(summary)

    def publish_progress(
        training_run: TrainingRunRecord,
        throughput: Mapping[str, object],
        evaluation_curriculum: Mapping[str, object],
    ) -> None:
        _write_document_atomic(
            progress_path,
            _training_progress_record(
                plan=plan,
                summary=summary,
                architecture=architecture,
                outcome_space=outcome_space,
                evaluation_rungs=(initial_evaluation_rung,),
                evaluation_curriculum=evaluation_curriculum,
                training_run=training_run,
                throughput=throughput,
            ),
        )
        if progress_callback is not None:
            progress_callback(summary)

    training_result = _train_and_predict(
        architecture=architecture,
        initial_evaluation_rung=initial_evaluation_rung,
        generator=generator,
        outcome_space=outcome_space,
        component_count=_component_count,
        sample_count=plan.sample_count,
        evaluation_sample_count=plan.resolved_evaluation_sample_count,
        gate_sample_count=plan.resolved_gate_sample_count,
        train_steps=plan.train_steps,
        learning_rate=float(plan.learning_rate),
        optimizer_name=plan.optimizer,
        schedule_name=plan.schedule,
        gate_check_interval=plan.gate_check_interval,
        checkpoint_interval=plan.checkpoint_interval,
        gate_decision_rule=plan.gate_decision_rule,
        convergence_patience=plan.convergence_patience,
        convergence_min_delta=float(plan.convergence_min_delta),
        convergence_min_steps=plan.convergence_min_steps,
        tensor_device=plan.tensor_device,
        work_estimates=_training_work_estimates(
            architecture=architecture,
            inference_compute=model_inspection.cost_summary.inference_compute,
            training_compute_per_sample=(
                model_inspection.cost_summary.training_compute_per_sample
            ),
            storage_bytes=model_inspection.cost_summary.storage_bytes,
            batch_size=plan.sample_count,
        ),
        seed=plan.seed,
        progress_callback=publish_progress,
    )
    measurement_groups = tuple(
        finite_measurements_for_predictions(
            batch=rung.batch,
            outcome_space=outcome_space,
            probabilities=probabilities,
            run_slug=f"{summary.run_slug}.rung{index}",
        )
        for index, (rung, probabilities) in enumerate(training_result.evaluation_results)
    )
    measurements = tuple(
        measurement
        for group in measurement_groups
        for measurement in group
    )
    dataset = MeasurementDataset(measurements=measurements)
    dataset.validate_manifest(generator.benchmark_manifest)
    completed_summary = replace(summary, measurement_count=len(measurements))
    _write_document(summary.measurement_dataset_path, dataset.to_record())
    _write_document(summary.model_inspection_path, model_inspection.to_record())
    _write_document(
        summary.training_summary_path,
        {
            **completed_summary.to_record(),
            "dry_run": False,
            "component_count": _component_count,
            "sample_count": plan.sample_count,
            "evaluation_sample_count": plan.resolved_evaluation_sample_count,
            "evaluation_curriculum_rung_count": len(training_result.evaluation_results),
            "evaluation_curriculum": _curriculum_record(
                rungs=tuple(
                    rung for rung, _probabilities in training_result.evaluation_results
                ),
                frontier_index=len(training_result.evaluation_results) - 1,
            ),
            "training_curriculum": {
                "kind": "competence-gated-training-curriculum",
                "source": "evaluation_curriculum",
                "frontier_sampling_weight": 0.7,
                "replay_sampling_weight": 0.3,
                "gating_metric": "frontier-validation-loss-plateau",
            },
            "seed": plan.seed,
            "train_steps": plan.train_steps,
            "learning_rate": float(plan.learning_rate),
            "optimizer": plan.optimizer,
            "schedule": plan.schedule,
            "checkpoint_interval": plan.checkpoint_interval,
            "gate_check_interval": plan.gate_check_interval,
            "gate_sample_count": plan.resolved_gate_sample_count,
            "gate_decision_rule": plan.gate_decision_rule,
            "convergence_patience": plan.convergence_patience,
            "convergence_min_delta": float(plan.convergence_min_delta),
            "convergence_min_steps": plan.convergence_min_steps,
            "tensor_runtime": "pytorch",
            "tensor_device": training_result.training_run.protocol.tensor_device,
            "training_run": training_result.training_run.to_record(),
            "throughput": training_result.throughput,
            "sampled_competence": sampled_competence_curriculum_record(
                tuple(
                    sampled_competence_record(
                        batch=batch,
                        measurements=measurements,
                        complexity_axis=None,
                    )
                    for (rung, _probabilities), measurements in zip(
                        training_result.evaluation_results,
                        measurement_groups,
                        strict=True,
                    )
                    for batch in (rung.batch,)
                )
            ),
            "architecture": model_inspection.architecture.to_record(),
            "cost_summary": model_inspection.cost_summary.to_record(),
            "measurement_dataset_digest": str(dataset.digest),
            "model_inspection_digest": str(model_inspection.digest),
        },
    )
    progress_path.unlink(missing_ok=True)
    return completed_summary


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
        measurement_dataset_path=(
            plan.results_root / "measurements" / benchmark_atom / f"{run_slug}{_document_suffix}"
        ),
        model_inspection_path=(
            plan.results_root
            / "model-inspections"
            / benchmark_atom
            / f"{run_slug}{_document_suffix}"
        ),
        training_summary_path=(
            plan.results_root / "training" / benchmark_atom / f"{run_slug}{_document_suffix}"
        ),
        dry_run=plan.dry_run,
    )


def _training_progress_path(summary: BenchmarkRunSummary) -> Path:
    return (
        summary.training_summary_path.parent.parent.parent
        / "training-progress"
        / summary.training_summary_path.parent.name
        / summary.training_summary_path.name
    )


def _evaluation_curriculum_rung(
    *,
    architecture: ArchitectureManifest,
    generator: ObservationGenerator,
    component_count: int,
    sample_count: int,
    seed: int,
    index: int,
) -> _CurriculumRung:
    return _curriculum_rung_from_candidates(
        architecture=architecture,
        generator=generator,
        component_count=component_count,
        sample_count=sample_count,
        seed=seed,
        index=index,
        candidates=_logarithmic_curriculum_candidates(
            generator=generator,
            component_count=component_count,
            start_index=index,
        ),
    )


def _training_curriculum_rung(
    *,
    architecture: ArchitectureManifest,
    generator: ObservationGenerator,
    component_count: int,
    sample_count: int,
    seed: int,
    index: int,
) -> _CurriculumRung:
    return _curriculum_rung_from_candidates(
        architecture=architecture,
        generator=generator,
        component_count=component_count,
        sample_count=sample_count,
        seed=seed,
        index=index,
        candidates=_structured_training_curriculum_candidates(
            generator=generator,
            component_count=component_count,
            start_index=index,
        ),
    )


def _curriculum_rung_from_candidates(
    *,
    architecture: ArchitectureManifest,
    generator: ObservationGenerator,
    component_count: int,
    sample_count: int,
    seed: int,
    index: int,
    candidates: Sequence[_CurriculumCandidate],
) -> _CurriculumRung:
    for candidate_index, candidate in enumerate(candidates):
        if candidate_index < index:
            continue
        rung_seed = seed if index == 0 else seed + 2_000_003 * index
        batch = generator.sample_batch(
            component_count=component_count,
            sample_count=sample_count,
            seed=rung_seed,
            resolution_assignment=candidate.resolution_assignment,
            variation_extent=candidate.nuisance_extent,
        )
        input_reason = _input_shape_boundary_reason(
            architecture=architecture,
            sample_shape=batch.samples[0].field.shape,
        )
        if input_reason is not None:
            if index == 0:
                raise BenchmarkRunnerError(input_reason)
            raise BenchmarkRunnerError(
                "architecture scale contract rejected the next curriculum rung: "
                f"{input_reason}"
            )
        return _CurriculumRung(
            index=index,
            nuisance_extent=candidate.nuisance_extent,
            resolution_assignment=candidate.resolution_assignment,
            seed=batch.seed,
            batch=batch,
        )
    raise BenchmarkRunnerError("evaluation curriculum did not produce any rungs")


@dataclass(frozen=True, slots=True)
class _CurriculumCandidate:
    nuisance_extent: float
    resolution_assignment: AxisAssignment
    complexity: float


def _logarithmic_curriculum_candidates(
    *,
    generator: ObservationGenerator,
    component_count: int,
    start_index: int,
) -> Sequence[_CurriculumCandidate]:
    candidates = list(
        _structured_training_curriculum_candidates(
            generator=generator,
            component_count=component_count,
            start_index=start_index,
        )
    )
    return tuple(sorted(candidates, key=lambda candidate: candidate.complexity))


def _structured_training_curriculum_candidates(
    *,
    generator: ObservationGenerator,
    component_count: int,
    start_index: int,
) -> Sequence[_CurriculumCandidate]:
    minimum_assignment = generator.minimum_discriminatable_resolution_assignment(
        component_count=component_count,
        minimum_assignment=generator.materialization.minimum_resolution(),
    )
    width_axis = generator.formation.width_axis
    height_axis = generator.formation.height_axis
    minimum_width = minimum_assignment.require_axis(width_axis)
    minimum_height = minimum_assignment.require_axis(height_axis)
    lattice_steps = generator.materialization.resolution_lattice_steps()
    width_step = lattice_steps.get(width_axis, 1)
    height_step = lattice_steps.get(height_axis, 1)
    stage_count = max(8, start_index + 8)
    widths = _logarithmic_lattice_axis_values(
        minimum=minimum_width,
        step=width_step,
        count=stage_count,
    )
    heights = _logarithmic_lattice_axis_values(
        minimum=minimum_height,
        step=height_step,
        count=stage_count,
    )
    candidates: list[_CurriculumCandidate] = []
    for stage in range(stage_count):
        for width_index, width in enumerate(widths[: stage + 1]):
            for height_index, height in enumerate(heights[: stage + 1]):
                if max(width_index, height_index) != stage:
                    continue
                for nuisance_extent in _nuisance_extent_curriculum:
                    complexity = generator.distinguishable_state_complexity(
                        component_count=component_count,
                        width=width,
                        height=height,
                        variation_extent=nuisance_extent,
                    )
                    candidates.append(
                        _CurriculumCandidate(
                            nuisance_extent=nuisance_extent,
                            resolution_assignment=AxisAssignment(
                                values={
                                    **minimum_assignment.values,
                                    width_axis: width,
                                    height_axis: height,
                                }
                            ),
                            complexity=complexity,
                        )
                    )
    return tuple(candidates)


def _logarithmic_lattice_axis_values(
    *,
    minimum: int,
    step: int,
    count: int,
) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    stage = 0
    minimum_multiplier = max(1, math.ceil(minimum / step))
    while len(values) < count:
        multiplier = max(
            minimum_multiplier,
            math.ceil(minimum_multiplier * (_canvas_logarithmic_growth_factor ** stage)),
        )
        value = multiplier * step
        if value not in seen:
            values.append(value)
            seen.add(value)
        stage += 1
    return tuple(values)


def _curriculum_record(
    *,
    rungs: Sequence[_CurriculumRung],
    frontier_index: int,
) -> dict[str, object]:
    return {
        "kind": "competence-gated-evaluation-curriculum",
        "curriculum_variable": "internal-distinguishable-state-complexity",
        "complexity_axis": "internal-distinguishable-state-complexity",
        "sampling_levers": ["canvas-size", "nuisance-extent"],
        "canvas_growth": {
            "kind": "logarithmic",
            "factor": _canvas_logarithmic_growth_factor,
        },
        "nuisance_extent_curriculum": list(_nuisance_extent_curriculum),
        "gating_metric": "frontier-validation-loss-plateau",
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


def _validate_architecture_for_batch(
    *,
    architecture: ArchitectureManifest,
    batch: GeneratedObservationBatch,
    outcome_space: OutcomeSpace,
) -> None:
    sample_shape = batch.samples[0].field.shape
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
    generator: ObservationGenerator,
    outcome_space: OutcomeSpace,
    component_count: int,
    sample_count: int,
    evaluation_sample_count: int,
    gate_sample_count: int,
    train_steps: int | None,
    learning_rate: float,
    optimizer_name: str,
    schedule_name: str,
    gate_check_interval: int,
    checkpoint_interval: int,
    gate_decision_rule: str,
    convergence_patience: int,
    convergence_min_delta: float,
    convergence_min_steps: int,
    tensor_device: TensorRuntimeDevice,
    work_estimates: _TrainingWorkEstimates | None,
    seed: int,
    progress_callback: (
        Callable[[TrainingRunRecord, Mapping[str, object], Mapping[str, object]], None] | None
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
                component_count=component_count,
                sample_count=sample_count,
                evaluation_sample_count=evaluation_sample_count,
                gate_sample_count=gate_sample_count,
                train_steps=train_steps,
                learning_rate=learning_rate,
                optimizer_name=optimizer_name,
                schedule_name=schedule_name,
                gate_check_interval=gate_check_interval,
                checkpoint_interval=checkpoint_interval,
                gate_decision_rule=gate_decision_rule,
                convergence_patience=convergence_patience,
                convergence_min_delta=convergence_min_delta,
                convergence_min_steps=convergence_min_steps,
                tensor_device=device_kind,
                work_estimates=work_estimates,
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
    generator: ObservationGenerator,
    outcome_space: OutcomeSpace,
    component_count: int,
    sample_count: int,
    evaluation_sample_count: int,
    gate_sample_count: int,
    train_steps: int | None,
    learning_rate: float,
    optimizer_name: str,
    schedule_name: str,
    gate_check_interval: int,
    checkpoint_interval: int,
    gate_decision_rule: str,
    convergence_patience: int,
    convergence_min_delta: float,
    convergence_min_steps: int,
    tensor_device: TensorRuntimeDeviceKind,
    work_estimates: _TrainingWorkEstimates | None,
    seed: int,
    progress_callback: (
        Callable[[TrainingRunRecord, Mapping[str, object], Mapping[str, object]], None] | None
    ) = None,
    fallback_errors: tuple[tuple[str, str], ...] = (),
) -> _TrainingResult:
    try:
        runtime = resolve_tensor_runtime(tensor_device)
    except TensorRuntimeError as error:
        raise BenchmarkRunnerError(str(error)) from error
    torch = runtime.torch
    torch.manual_seed(seed)
    executable = ExecutableModelOperator(architecture)
    module = OperationFallbackSequential(
        runtime=runtime,
        operations=executable.torch_operation_modules(torch=torch),
    )
    outcome_ids = tuple(outcome.id for outcome in outcome_space.outcomes)
    formation_cache = FormationTensorCache(runtime=runtime, formation=generator.formation)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = _make_optimizer(
        torch=torch,
        parameters=module.parameters(),
        name=optimizer_name,
        learning_rate=learning_rate,
    )
    module.attach_optimizer(optimizer)
    scheduler = _make_scheduler(
        torch=torch,
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

    def batch_for_seed(
        batch_seed: int,
        *,
        batch_sample_count: int,
        generation_phase: str,
        tensor_phase: str,
        resolution_assignment: AxisAssignment,
        variation_extent: float,
    ) -> tuple[Any, Any]:
        with phase_timings.span(generation_phase, samples=batch_sample_count):
            generated = generator.sample_formation_batch(
                component_count=component_count,
                sample_count=batch_sample_count,
                seed=batch_seed,
                resolution_assignment=resolution_assignment,
                variation_extent=variation_extent,
                timing=phase_timings,
                timing_prefix=f"{generation_phase}.",
            )
        with phase_timings.span(tensor_phase, samples=batch_sample_count):
            tensors = formation_cache.batch_tensors(batch=generated, outcome_ids=outcome_ids)
        return tensors

    training_rungs: list[_CurriculumRung] = [
        _training_curriculum_rung(
            architecture=architecture,
            generator=generator,
            component_count=component_count,
            sample_count=sample_count,
            seed=seed,
            index=0,
        )
    ]
    training_frontier_index = 0

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
        if latest.validation_loss >= math.log(len(outcome_ids)) - convergence_min_delta:
            return False
        next_index = training_frontier_index + 1
        training_rungs.append(
            _training_curriculum_rung(
                architecture=architecture,
                generator=generator,
                component_count=component_count,
                sample_count=sample_count,
                seed=seed,
                index=next_index,
            )
        )
        training_frontier_index += 1
        if scheduler is not None:
            scheduler.reset_for_curriculum_expansion()
        return True

    training_result = _train_until_convergence(
        torch=torch,
        module=module,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        train_batch=lambda step: (
            batch_for_seed(
                seed + step,
                batch_sample_count=sample_count,
                generation_phase="training_formation_generation",
                tensor_phase="training_tensor_batch",
                resolution_assignment=training_rung_for_step(step).resolution_assignment,
                variation_extent=training_rung_for_step(step).nuisance_extent,
            )
        ),
        validation_batch=lambda check: batch_for_seed(
            seed + 1_000_003 + check,
            batch_sample_count=gate_sample_count,
            generation_phase="validation_formation_generation",
            tensor_phase="validation_tensor_batch",
            resolution_assignment=current_frontier().resolution_assignment,
            variation_extent=current_frontier().nuisance_extent,
        ),
        max_steps=train_steps,
        gate_check_interval=gate_check_interval,
        checkpoint_interval=checkpoint_interval,
        patience=convergence_patience,
        min_delta=convergence_min_delta,
        min_steps=convergence_min_steps,
        batch_size=sample_count,
        gate_sample_count=gate_sample_count,
        training_compute_per_sample=(
            None if work_estimates is None else work_estimates.training.compute_per_sample
        ),
        training_counter=training_counter,
        training_compute_counter=training_compute_counter,
        validation_counter=validation_counter,
        phase_timings=phase_timings,
        on_plateau=advance_frontier,
        on_checkpoint=lambda history: (
            progress_callback(
                _running_training_run_record(
                    seed=seed,
                    batch_size=sample_count,
                    max_steps=train_steps,
                    learning_rate=float(learning_rate),
                    optimizer_name=optimizer_name,
                    schedule_name=schedule_name,
                    gate_check_interval=gate_check_interval,
                    checkpoint_interval=checkpoint_interval,
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
                    roofline=runtime_roofline_record(runtime),
                    work_estimates=work_estimates,
                    phase_timings=phase_timings,
                    fallback_errors=fallback_errors,
                    operation_fallbacks=module.operation_fallback_records(),
                ),
                _curriculum_record(
                    rungs=tuple(training_rungs),
                    frontier_index=training_frontier_index,
                ),
            )
            if progress_callback is not None
            else None
        ),
    )
    validation_history = training_result.validation_history
    final_training_stop_reason = training_result.stop_reason
    if train_steps is None and not training_stage_converged(final_training_stop_reason):
        raise BenchmarkRunnerError("uncapped training curriculum ended before convergence")
    module.eval()
    evaluation_rungs = tuple(
        _evaluation_curriculum_rung(
            architecture=architecture,
            generator=generator,
            component_count=component_count,
            sample_count=evaluation_sample_count,
            seed=seed,
            index=index,
        )
        for index in range(len(training_rungs))
    )
    evaluation_results: list[tuple[_CurriculumRung, tuple[tuple[float, ...], ...]]] = []
    for rung in evaluation_rungs:
        candidate_batch = rung.batch
        evaluation_started = time.perf_counter()
        with phase_timings.span(
            "evaluation_tensorization",
            samples=len(candidate_batch.samples),
        ):
            eval_fields, _eval_labels = _batch_tensors(
                torch=torch,
                batch=candidate_batch,
                outcome_ids=outcome_ids,
                device=runtime.device,
            )
        with (
            phase_timings.span("evaluation_forward", samples=len(candidate_batch.samples)),
            torch.no_grad(),
        ):
            predictions = tuple(
                _renormalized_probabilities(row)
                for row in torch.softmax(module(eval_fields), dim=1).tolist()
            )
        evaluation_counter.add(
            seconds=time.perf_counter() - evaluation_started,
            samples=len(candidate_batch.samples),
        )
        evaluation_results.append((rung, predictions))
        if _mean_prediction_accepted_mass(
            batch=candidate_batch,
            probabilities=predictions,
            outcome_ids=outcome_ids,
        ) <= _chance_accepted_mass(outcome_ids) + 1e-12:
            break
    training_run = _training_run_record(
        seed=seed,
        batch_size=sample_count,
        max_steps=train_steps,
        learning_rate=float(learning_rate),
        optimizer_name=optimizer_name,
        schedule_name=schedule_name,
        gate_check_interval=gate_check_interval,
        checkpoint_interval=checkpoint_interval,
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
        evaluation_results=tuple(evaluation_results),
        training_run=training_run,
        throughput=_throughput_record(
            runtime_device=runtime.device_kind,
            training_counter=training_counter,
            validation_counter=validation_counter,
            evaluation_counter=evaluation_counter,
            roofline=runtime_roofline_record(runtime),
            work_estimates=work_estimates,
            phase_timings=phase_timings,
            fallback_errors=fallback_errors,
            operation_fallbacks=module.operation_fallback_records(),
        ),
    )


def training_stage_converged(stop_reason: str) -> bool:
    return stop_reason in _converged_training_stage_stop_reasons


def _train_until_convergence(
    *,
    torch: Any,
    module: Any,
    optimizer: Any,
    scheduler: _LearningRateSchedule | None,
    loss_function: Any,
    train_batch: Callable[[int], tuple[Any, Any]],
    validation_batch: Callable[[int], tuple[Any, Any]],
    max_steps: int | None,
    gate_check_interval: int,
    checkpoint_interval: int,
    patience: int,
    min_delta: float,
    min_steps: int,
    batch_size: int,
    gate_sample_count: int,
    training_compute_per_sample: float | None,
    training_counter: _ThroughputCounter,
    training_compute_counter: _ComputeCounter,
    validation_counter: _ThroughputCounter,
    phase_timings: TimingCollector,
    start_step: int = 0,
    start_check: int = 0,
    on_plateau: Callable[[tuple[TrainingHistoryPoint, ...]], bool] | None = None,
    on_checkpoint: Callable[[tuple[TrainingHistoryPoint, ...]], None] | None = None,
) -> _TrainingStageResult:
    validation_history: list[TrainingHistoryPoint] = []
    best_loss = float("inf")
    stale_checks = 0
    stop_reason = "training-stopped"
    plateau_window_start_index = 0
    plateau_window_start_step = start_step

    def append_validation(*, step: int, check: int) -> None:
        nonlocal best_loss, stale_checks
        validation_started = time.perf_counter()
        fields, labels = validation_batch(check)
        actual_gate_sample_count = _tensor_batch_size(fields, fallback=gate_sample_count)
        with phase_timings.span("validation_forward_loss", samples=actual_gate_sample_count):
            validation_loss = _validation_loss(
                torch=torch,
                module=module,
                fields=fields,
                labels=labels,
                loss_function=loss_function,
            )
        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            stale_checks = 0
        else:
            stale_checks += 1
        if scheduler is not None:
            scheduler.step_after_validation(validation_loss)
            learning_rates = scheduler.learning_rates()
        else:
            learning_rates = tuple(float(group["lr"]) for group in optimizer.param_groups)
        validation_counter.add(
            seconds=time.perf_counter() - validation_started,
            samples=actual_gate_sample_count,
        )
        validation_history.append(
            TrainingHistoryPoint(
                step=step,
                validation_check=check,
                validation_loss=validation_loss,
                stale_checks=stale_checks,
                learning_rates=learning_rates,
            )
        )
        if (
            on_checkpoint is not None
            and (step == start_step or step % checkpoint_interval == 0)
        ):
            on_checkpoint(tuple(validation_history))

    append_validation(step=start_step, check=start_check)
    if max_steps == start_step:
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
        training_counter.add(
            seconds=time.perf_counter() - training_started,
            samples=actual_batch_size,
        )
        training_compute_counter.add(
            compute_per_sample=training_compute_per_sample,
            samples=actual_batch_size,
        )
        hit_step_cap = max_steps is not None and step == max_steps
        if step % gate_check_interval != 0 and not hit_step_cap:
            continue
        append_validation(step=step, check=validation_check)
        validation_check += 1
        if (
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
        ):
            if on_plateau is not None and on_plateau(tuple(validation_history)):
                plateau_window_start_index = len(validation_history) - 1
                plateau_window_start_step = step
                best_loss = validation_history[-1].validation_loss
                stale_checks = 0
                continue
            stop_reason = "validation-plateau"
            break
        if max_steps is not None and step >= max_steps:
            stop_reason = "max-steps"
            break
    return _TrainingStageResult(
        validation_history=tuple(validation_history),
        stop_reason=stop_reason,
    )


def _validation_loss(
    *,
    torch: Any,
    module: Any,
    fields: Any,
    labels: Any,
    loss_function: Any,
) -> float:
    was_training = bool(module.training)
    module.eval()
    with torch.no_grad():
        loss = float(loss_function(module(fields), labels).item())
    if was_training:
        module.train()
    return loss


def _tensor_batch_size(fields: Any, *, fallback: int) -> int:
    shape = getattr(fields, "shape", None)
    if shape is None or len(shape) < 1:
        return fallback
    value = shape[0]
    if type(value) is int and value >= 0:
        return value
    return fallback


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
    return window_start.validation_loss - current.validation_loss < min_delta


def _training_run_record(
    *,
    seed: int,
    batch_size: int,
    max_steps: int | None,
    learning_rate: float,
    optimizer_name: str,
    schedule_name: str,
    gate_check_interval: int,
    checkpoint_interval: int,
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
            checkpoint_interval=checkpoint_interval,
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
    checkpoint_interval: int,
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
            checkpoint_interval=checkpoint_interval,
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
    outcome_space: OutcomeSpace,
    evaluation_rungs: tuple[_CurriculumRung, ...],
    evaluation_curriculum: Mapping[str, object],
    training_run: TrainingRunRecord,
    throughput: Mapping[str, object],
) -> Mapping[str, object]:
    inspection = ModelInspectionRecord.from_architecture(
        id=ProtocolIdentifier.parse(
            f"model-inspections.{_identifier_atom(summary.benchmark_id)}."
            f"{summary.run_slug}.progress@0.1.0"
        ),
        architecture_manifest=architecture,
    )
    provisional_score = validation_competence(
        validation_loss=training_run.validation_history[-1].validation_loss,
        outcome_count=len(outcome_space.outcomes),
    )
    chance_mass = _chance_accepted_mass(tuple(outcome.id for outcome in outcome_space.outcomes))
    accepted_mass_equivalent = chance_mass + provisional_score * (1.0 - chance_mass)
    frontier_index_value = evaluation_curriculum.get("frontier_index", 0)
    frontier_index = (
        frontier_index_value
        if isinstance(frontier_index_value, int)
        and 0 <= frontier_index_value < len(evaluation_rungs)
        else 0
    )
    evaluation_batch = evaluation_rungs[frontier_index].batch
    complexities = {sample.complexity for sample in evaluation_batch.samples}
    progress_complexity = next(iter(complexities)) if len(complexities) == 1 else None
    record: dict[str, object] = {
        "format": _progress_format,
        "format_version": _progress_format_version,
        "run_slug": summary.run_slug,
        "run_status": "running",
        "benchmark_id": str(summary.benchmark_id),
        "architecture_path": summary.architecture_path.as_posix(),
        "component_count": _component_count,
        "sample_count": plan.sample_count,
        "evaluation_sample_count": plan.resolved_evaluation_sample_count,
        "seed": plan.seed,
        "train_steps": plan.train_steps,
        "learning_rate": float(plan.learning_rate),
        "optimizer": plan.optimizer,
        "schedule": plan.schedule,
        "checkpoint_interval": plan.checkpoint_interval,
        "gate_check_interval": plan.gate_check_interval,
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
        "architecture": inspection.architecture.to_record(),
        "cost_summary": inspection.cost_summary.to_record(),
        "model_inspection": inspection.to_record(),
        "architecture_digest": str(architecture.digest),
        "model_inspection_digest": str(inspection.digest),
        "provisional_score": provisional_score,
    }
    if progress_complexity is not None:
        record["sampled_competence"] = {
            "kind": "provisional-validation-competence",
            "sampling_rule": "generator-validation-resample-v1",
            "difficulty_assumption": "validation-loss-proxy-for-sampled-competence",
            "benchmark_id": str(summary.benchmark_id),
            "component_count": evaluation_batch.component_count,
            "complexity_axis": None,
            "complexity": progress_complexity,
            "seed": evaluation_batch.seed,
            "sample_count": len(evaluation_batch.samples),
            "mean_accepted_mass": accepted_mass_equivalent,
            "validation_competence": provisional_score,
        }
    return record


def _throughput_record(
    *,
    runtime_device: str,
    training_counter: _ThroughputCounter,
    validation_counter: _ThroughputCounter,
    evaluation_counter: _ThroughputCounter,
    roofline: Mapping[str, object],
    work_estimates: _TrainingWorkEstimates | None,
    phase_timings: TimingCollector,
    fallback_errors: tuple[tuple[str, str], ...] = (),
    operation_fallbacks: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    training = training_counter.to_record(kind="training-throughput")
    validation = validation_counter.to_record(kind="validation-throughput")
    evaluation = evaluation_counter.to_record(kind="evaluation-throughput")
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
    torch: Any,
    parameters: Any,
    name: str,
    learning_rate: float,
) -> Any:
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    raise BenchmarkRunnerError(f"unsupported optimizer: {name}")


def _make_scheduler(
    *,
    torch: Any,
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
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, max_steps),
            ),
            optimizer=optimizer,
            update_on="optimizer-step",
            base_learning_rates=tuple(float(group["lr"]) for group in optimizer.param_groups),
        )
    if name == "reduce-on-plateau":
        factor = 0.1
        eps = 1e-8
        return _LearningRateSchedule(
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=factor,
                threshold=min_delta,
                threshold_mode="abs",
                patience=patience,
                eps=eps,
            ),
            optimizer=optimizer,
            update_on="validation-loss",
            minimum_effective_learning_rate=eps / (1.0 - factor),
            base_learning_rates=tuple(float(group["lr"]) for group in optimizer.param_groups),
        )
    raise BenchmarkRunnerError(f"unsupported schedule: {name}")


def _batch_tensors(
    *,
    torch: Any,
    batch: GeneratedObservationBatch,
    outcome_ids: tuple[str, ...],
    device: Any,
) -> tuple[Any, Any]:
    return (
        _batch_tensor(torch=torch, batch=batch, device=device),
        torch.tensor(
            [outcome_ids.index(sample.outcome_id) for sample in batch.samples],
            dtype=torch.long,
            device=device,
        ),
    )


def _batch_tensor(*, torch: Any, batch: GeneratedObservationBatch, device: Any) -> Any:
    values = [list(sample.field.values) for sample in batch.samples]
    fields = torch.tensor(values, dtype=torch.float32, device=device)
    return fields.reshape((len(batch.samples), *batch.samples[0].field.shape))


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
    batch: GeneratedObservationBatch,
    probabilities: tuple[tuple[float, ...], ...],
    outcome_ids: tuple[str, ...],
) -> float:
    outcome_indexes = {outcome_id: index for index, outcome_id in enumerate(outcome_ids)}
    accepted_mass = tuple(
        row[outcome_indexes[sample.outcome_id]]
        for sample, row in zip(batch.samples, probabilities, strict=True)
    )
    return math.fsum(accepted_mass) / len(accepted_mass)


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
