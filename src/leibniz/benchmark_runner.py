"""Small benchmark execution workflows for local operator runs."""

from __future__ import annotations

import hashlib
import math
import os
import secrets
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from itertools import count
from pathlib import Path
from typing import Any, Protocol, cast

from leibniz.artifacts import ArtifactReference, reference_for_record
from leibniz.benchmark_evaluation import (
    CompetencePoint,
    StateSpaceIntegral,
    StateSpaceIntegralTerm,
    ValidationCompetencePoint,
    finite_measurements_for_predictions,
    sampled_competence_curriculum_record,
    sampled_competence_frontier_integral,
    sampled_competence_metrology_cost_integral,
    sampled_competence_record,
    validation_competence_frontier_advances,
)
from leibniz.benchmark_implementations import (
    Generator as BenchmarkGenerator,
)
from leibniz.benchmark_implementations import (
    load_benchmark,
)
from leibniz.content import ContentDigest
from leibniz.cost_metrology import (
    CostMeasurement,
    CostMeter,
    CostMetrologyError,
    estimate_program_cost,
    measure_program_cost,
)
from leibniz.documents import (
    canonical_document_bytes,
    document_filename_suffix,
    load_object_document,
)
from leibniz.evaluation_bundles import (
    BenchmarkEvaluationBundle,
)
from leibniz.field_evolution import (
    FieldEvolutionError,
    field_stepper_state,
    validate_field_stepper_nondegenerate,
)
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import AxisAssignment
from leibniz.measurements import MeasurementDataset, MeasurementRecord
from leibniz.model_inspection import ModelInspectionRecord
from leibniz.model_interfaces import ModelInterface
from leibniz.model_manifests import (
    ModelArtifactManifest,
    ModelArtifactManifestDocument,
    ModelExecutionFamily,
)
from leibniz.observation_generation import (
    GeneratedSample,
    GeneratedSampleSet,
    ObservationGenerationError,
    StateSpaceVolumeRequest,
    StateSpaceVolumeValue,
)
from leibniz.program_graphs import LoadedProgramGraph, ProgramGraphError, load_program_graph
from leibniz.records import RecordExtractor
from leibniz.state_space import (
    AccessibleSubspace,
    SamplingProtocol,
    StateSpaceError,
    StateSpaceRegion,
    state_space_regions_are_disjoint,
)
from leibniz.target_contracts import TargetContract, TargetContractError
from leibniz.tensor_runtime import (
    TensorRuntime,
    TensorRuntimeDevice,
    TensorRuntimeDeviceKind,
    TensorRuntimeError,
    build_cosine_lr_schedule,
    build_loss,
    build_optimizer,
    build_plateau_lr_schedule,
    load_tensor_runtime_state,
    make_empty_float_tensor,
    make_float_tensor,
    make_long_tensor,
    no_grad_context,
    optimizer_step,
    resolve_host_tensor_runtime,
    resolve_tensor_runtime,
    runtime_capacity_error,
    runtime_roofline_record,
    save_tensor_runtime_state,
    seed_runtime,
    softmax_prediction_rows,
    softmax_target_mass_tensor,
    softmax_target_masses,
    synchronize_runtime,
    tensor_element_compile_fallback_records,
    tensor_runtime_concat,
    tensor_runtime_device_kinds,
    tensor_runtime_has_fixed_device_memory,
    tensor_runtime_total_memory_bytes,
    tensor_runtime_used_memory_bytes,
    tensor_value_to_host,
    validate_tensor_runtime_device,
)
from leibniz.timing import TimingCollector
from leibniz.training_runs import TrainingHistoryPoint, TrainingProtocol, TrainingRunRecord
from leibniz.views import MeasurementScoreView

__all__ = [
    "BenchmarkRunnerError",
    "BenchmarkEvaluationPlan",
    "BenchmarkEvaluationSummary",
    "BenchmarkRunPlan",
    "BenchmarkRunSummary",
    "CheckpointModelPredictor",
    "evaluate_benchmark_checkpoint",
    "evaluate_model_checkpoint_artifact",
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
_default_training_batch_target = 512
_default_gate_batch_target = 512
_default_evaluation_convergence_min_samples = 64
_default_evaluation_convergence_half_width = 0.05
_default_evaluation_convergence_confidence_z = 1.96
_default_evaluation_integral_relative_half_width = 0.05
_default_evaluation_integral_minimum_half_width = 0.05
_default_evaluation_terminal_failure_rungs = 3
_default_runtime_memory_budget_fraction = 0.1
_runtime_batch_memory_safety_factor = 8
_float32_bytes = 4
_default_train_steps: int | None = None
_default_gate_check_interval = 32
_default_model_checkpoint_gate_interval = 1
_default_convergence_patience = 6
_default_convergence_min_delta = 1e-3
_default_rung_competence_threshold = 0.5
_training_replay_score_window_batches = 8
_converged_training_stage_stop_reasons = frozenset({"validation-plateau"})
_legal_uncapped_training_stage_stop_reasons = frozenset(
    {"validation-plateau", "capacity-limited"}
)
_minimum_plateau_lr_reductions = 3
_full_variation_extent = 1.0
_sync_timing_environment_variable = "LEIBNIZ_SYNC_TIMING"
_field_valued_competence_kinds = frozenset(
    {
        "ambient-certified-bits",
        "convergence-resolved-bits",
        "mass-within-resolution",
    }
)


class _TensorBenchmarkGenerator(BenchmarkGenerator, Protocol):
    """Internal contract for tensor-backed benchmark training."""

    def minimum_log2_volume(self) -> StateSpaceVolumeValue: ...

    def __call__(
        self,
        *,
        seed: int,
        shape: int | Sequence[int] | None = None,
        include_fields: bool = False,
        include_metadata: bool = True,
        include_artifacts: bool = False,
        volume_request: StateSpaceVolumeRequest | None = None,
        sample_indices: Sequence[int] | None = None,
        memory_limit_bytes: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
        variation_extent: float = 1.0,
        runtime: TensorRuntime | None = None,
        outcome_ids: tuple[str, ...] | None = None,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedSampleSet: ...


@dataclass(frozen=True, slots=True)
class _FieldTrainingCompetenceRequest:
    runtime: TensorRuntime
    module: Any | None
    fields: Any | None
    predictions: Any
    targets: Any
    horizons: tuple[float, ...] | None
    batch: GeneratedSampleSet | None
    generator: _TensorBenchmarkGenerator | None
    sample_keys: tuple[Mapping[str, object], ...]


@dataclass(frozen=True, slots=True)
class _CompetenceEvaluation:
    values: tuple[float, ...]
    diagnostics: tuple[Mapping[str, object], ...] = ()


class _BenchmarkTrainingLossFactory(Protocol):
    """Benchmark-owned tensor loss factory for non-generic target losses."""

    def build_training_loss(
        self,
        runtime: TensorRuntime,
        target_contract: TargetContract,
    ) -> Any: ...


class _BenchmarkTrainingCompetenceFactory(Protocol):
    """Benchmark-owned per-sample competence factory for field targets."""

    def build_training_competence(
        self,
        runtime: TensorRuntime,
        target_contract: TargetContract,
    ) -> Callable[[_FieldTrainingCompetenceRequest], Any]: ...


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
    program_path: Path
    program_graph: Mapping[str, object]
    checkpoint: ModelCheckpointArtifact
    run_slug: str
    benchmark_id: ProtocolIdentifier


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
        return record


@dataclass(frozen=True, slots=True)
class CheckpointModelPredictor:
    """A loaded checkpoint model that can produce benchmark predictions."""

    runtime: TensorRuntime
    module: Any
    outcome_ids: tuple[str, ...]
    target_contract: TargetContract

    def predict_batch(
        self,
        batch: GeneratedSampleSet,
    ) -> Any:
        self.module.eval()
        fields, labels = _batch_tensors(
            runtime=self.runtime,
            batch=batch,
            outcome_ids=self.outcome_ids,
            device=self.runtime.device,
        )
        horizons = _field_valued_target_horizons(
            batch=batch,
            labels=labels,
            target_contract=self.target_contract,
        )
        with no_grad_context(self.runtime):
            if self.target_contract.kind == "field-valued":
                return _model_predictions(
                    runtime=self.runtime,
                    module=self.module,
                    fields=fields,
                    labels=labels,
                    horizons=horizons,
                    target_contract=self.target_contract,
                )
            return tuple(
                _renormalized_probabilities(row)
                for row in softmax_prediction_rows(self.runtime, self.module(fields))
            )


def _require_tensor_generator(generator: BenchmarkGenerator) -> _TensorBenchmarkGenerator:
    return cast(_TensorBenchmarkGenerator, generator)


@dataclass(frozen=True, slots=True)
class _TrainingStageResult:
    validation_history: tuple[TrainingHistoryPoint, ...]
    stop_reason: str
    validation_inference_cost: tuple[CostMeasurement, int] | None = None


@dataclass(frozen=True, slots=True)
class _TrainingStepBatch:
    fields: Any
    labels: Any
    horizons: tuple[float, ...] | None = None
    sample_set: GeneratedSampleSet | None = None


class _SynchronizedTimingCollector(TimingCollector):
    """Timing collector that synchronizes runtime work around measured spans."""

    def __init__(self, runtime: TensorRuntime) -> None:
        super().__init__()
        self._runtime = runtime

    @contextmanager
    def span(self, phase: str, *, samples: int = 0) -> Any:
        synchronize_runtime(self._runtime)
        with super().span(phase, samples=samples):
            yield
        synchronize_runtime(self._runtime)


@dataclass(frozen=True, slots=True)
class _PendingReplayScore:
    sample_set: GeneratedSampleSet
    accepted_mass: Any


@dataclass(frozen=True, slots=True)
class _RollingValidationCompetencePoint:
    points: tuple[ValidationCompetencePoint, ...]

    @classmethod
    def from_point(
        cls,
        point: ValidationCompetencePoint,
    ) -> _RollingValidationCompetencePoint:
        return cls(points=(point,))

    def add(
        self,
        point: ValidationCompetencePoint,
    ) -> _RollingValidationCompetencePoint:
        return _RollingValidationCompetencePoint(
            points=(*self.points, point)[-_training_replay_score_window_batches:],
        )

    @property
    def point(self) -> ValidationCompetencePoint:
        latest = self.points[-1]
        sample_count = sum(point.sample_count for point in self.points)
        accepted_mass_sum = math.fsum(
            point.accepted_mass * point.sample_count for point in self.points
        )
        return ValidationCompetencePoint(
            log2_volume=latest.log2_volume,
            accepted_mass=accepted_mass_sum / sample_count,
            sample_count=sample_count,
            seed=latest.seed,
            log2_volume_minimum=latest.log2_volume_minimum,
            log2_volume_maximum=latest.log2_volume_maximum,
            input_shape=latest.input_shape,
            region=latest.region,
        )


class _RuntimeCapacityReached(Exception):
    """Raised when the active rung cannot fit one physical execution batch."""


@dataclass(frozen=True, slots=True)
class _CurriculumRung:
    index: int
    resolution_assignment: AxisAssignment | None
    seed: int
    batch: GeneratedSampleSet
    sample_count: int = 0
    log2_volume_minimum: float | None = None
    log2_volume_maximum: float | None = None

    @property
    def log2_volume(self) -> float:
        return self.batch.log2_volume

    def to_record(self, *, status: str) -> dict[str, object]:
        record: dict[str, object] = {
            "index": self.index,
            "status": status,
            "seed": self.seed,
            "volume_axis": _core_volume_measure_id(),
            "log2_volume": self.log2_volume,
            "volume_value": {
                "measure_id": _core_volume_measure_id(),
                "value": self.log2_volume,
            },
            "volume_request": (
                None
                if self.batch.volume_request is None
                else self.batch.volume_request.to_record()
            ),
            "sample_count": self.sample_count,
        }
        if self.batch.request_outcome is not None:
            record["request_outcome"] = self.batch.request_outcome.to_record()
        interval_record = _rung_log2_volume_interval_record(self)
        if interval_record:
            record["score_interval"] = interval_record
        if self.resolution_assignment is not None:
            record["resolution_assignment"] = self.resolution_assignment.to_record()
        return record


@dataclass(frozen=True, slots=True)
class _CheckpointEvaluationRungEvidence:
    rung: _CurriculumRung
    mean_accepted_mass: float
    sample_count: int
    confidence_half_width: float
    confidence_method_id: str | None
    sampling_protocol: SamplingProtocol
    input_shape: tuple[int, ...]
    inference_cost_measurement: CostMeasurement
    inference_cost_sample_count: int


@dataclass(frozen=True, slots=True)
class _EvaluationIntegrationEvidence:
    frontier_index: int
    score_integral_value: float
    score_integral_half_width: float
    terminal_failure_count: int

    @property
    def score_integral_half_width_threshold(self) -> float:
        return max(
            _default_evaluation_integral_minimum_half_width,
            self.score_integral_value * _default_evaluation_integral_relative_half_width,
        )

    @property
    def converged(self) -> bool:
        return (
            self.terminal_failure_count >= _default_evaluation_terminal_failure_rungs
            and self.score_integral_half_width <= self.score_integral_half_width_threshold
        )


def _evaluation_sampled_competence_record(
    *,
    benchmark_id: ProtocolIdentifier,
    evaluation_results: Sequence[_CheckpointEvaluationRungEvidence],
    frontier_point: Mapping[str, object],
    frontier_index: int,
) -> dict[str, object]:
    points: list[Mapping[str, object]] = []
    for index, result in enumerate(evaluation_results):
        if index == frontier_index:
            points.append(frontier_point)
            continue
        point: dict[str, object] = {
            "kind": "sampled-state-space-volume-window",
            "sampling_rule": "generator-uniform-component-index-v1",
            "difficulty_assumption": (
                "approximately-uniform-within-volume-window"
            ),
            "benchmark_id": str(benchmark_id),
            "volume_axis": None,
            "log2_volume": result.rung.log2_volume,
            "seed": result.rung.seed,
            "sample_count": result.sample_count,
            "mean_accepted_mass": result.mean_accepted_mass,
            "confidence_half_width": result.confidence_half_width,
            "confidence_method_id": result.confidence_method_id,
            "sampling_protocol": result.sampling_protocol.to_record(),
            "sampling_seed": result.rung.seed,
            "input_shape": list(result.input_shape),
            "inference_cost_measurement": (
                result.inference_cost_measurement.without_operation_trace().to_record()
            ),
            "inference_cost_sample_count": result.inference_cost_sample_count,
            **_rung_log2_volume_interval_record(result.rung),
        }
        if result.rung.batch.region is not None:
            point["region"] = result.rung.batch.region.to_record()
        points.append(point)
    record = sampled_competence_curriculum_record(points)
    if frontier_point.get("competence_value_kind") == "validated-bits":
        record["competence_value_kind"] = "validated-bits"
    return record


def _attach_competence_diagnostics(
    record: dict[str, object],
    diagnostics: tuple[Mapping[str, object], ...],
) -> None:
    if not diagnostics:
        return
    record["competence_diagnostics"] = [dict(item) for item in diagnostics]
    boundaries = tuple(
        _optional_nonnegative_float(
            item.get("predictability_boundary"),
            "competence_diagnostics.predictability_boundary",
        )
        for item in diagnostics
    )
    finite_boundaries = tuple(value for value in boundaries if value is not None)
    if finite_boundaries:
        record["predictability_boundary"] = max(finite_boundaries)
    time_points = diagnostics[0].get("time_points")
    if isinstance(time_points, Sequence) and not isinstance(time_points, str):
        copied_points: list[dict[str, object]] = []
        for item in cast(Sequence[object], time_points):
            if isinstance(item, Mapping):
                copied_points.append(dict(cast(Mapping[str, object], item)))
        if copied_points:
            record["time_points"] = copied_points


def _rung_volume_request(rung: _CurriculumRung) -> StateSpaceVolumeRequest:
    if rung.batch.volume_request is not None:
        return rung.batch.volume_request
    return StateSpaceVolumeRequest(
        minimum=rung.log2_volume,
        maximum=rung.log2_volume,
    )


def _rung_score_volume_request(rung: _CurriculumRung) -> StateSpaceVolumeRequest:
    interval = _rung_log2_volume_interval(rung)
    if interval is None:
        return _rung_volume_request(rung)
    lower, upper = interval
    return StateSpaceVolumeRequest(minimum=lower, maximum=upper)


def _core_volume_measure_id() -> str:
    return StateSpaceVolumeRequest(minimum=1.0, maximum=1.0).measure_id


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
class _RunningMeanEstimator:
    samples: int = 0
    mean: float = 0.0
    _sum_squared_delta: float = 0.0

    def add(self, value: float) -> None:
        self.samples += 1
        delta = value - self.mean
        self.mean += delta / self.samples
        self._sum_squared_delta += delta * (value - self.mean)

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.add(float(value))

    @property
    def sample_variance(self) -> float:
        if self.samples < 2:
            return 0.25
        return min(0.25, max(0.0, self._sum_squared_delta / (self.samples - 1)))


def _evaluation_confidence_half_width(
    estimator: _RunningMeanEstimator,
    *,
    sampling_protocol: SamplingProtocol,
) -> float:
    if estimator.samples < 1:
        return math.inf
    method_id = _evaluation_confidence_method_id(sampling_protocol)
    if method_id is None:
        return 0.0
    if method_id != "wilson":
        raise BenchmarkRunnerError(f"unsupported confidence_method_id: {method_id}")
    return _wilson_confidence_half_width(
        estimator.mean,
        sample_count=estimator.samples,
        z=_default_evaluation_convergence_confidence_z,
    )


def _evaluation_confidence_method_id(protocol: SamplingProtocol) -> str | None:
    if protocol.kind == "census":
        return None
    if protocol.confidence_method_id is None:
        raise BenchmarkRunnerError("sampling protocol is missing confidence_method_id")
    return protocol.confidence_method_id


def _wilson_confidence_half_width(
    mean: float,
    *,
    sample_count: int,
    z: float,
) -> float:
    if sample_count < 1:
        return math.inf
    p = min(1.0, max(0.0, float(mean)))
    n = float(sample_count)
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    radius = (
        z
        * math.sqrt(max(0.0, p * (1.0 - p) / n + z * z / (4.0 * n * n)))
        / denominator
    )
    lower = max(0.0, center - radius)
    upper = min(1.0, center + radius)
    return max(0.0, min(1.0, max(p - lower, upper - p)))


def _evaluation_estimate_converged(
    estimator: _RunningMeanEstimator,
    *,
    sampling_protocol: SamplingProtocol,
    half_width_threshold: float = _default_evaluation_convergence_half_width,
) -> bool:
    if _evaluation_confidence_method_id(sampling_protocol) is None:
        return estimator.samples >= 1
    return (
        estimator.samples >= _default_evaluation_convergence_min_samples
        and _evaluation_confidence_half_width(
            estimator,
            sampling_protocol=sampling_protocol,
        )
        <= half_width_threshold
    )


def _evaluation_next_sample_count(
    estimator: _RunningMeanEstimator,
    *,
    half_width_threshold: float = _default_evaluation_convergence_half_width,
) -> int:
    if estimator.samples < _default_evaluation_convergence_min_samples:
        return _default_evaluation_convergence_min_samples - estimator.samples
    target = math.ceil(
        estimator.sample_variance
        * (
            _default_evaluation_convergence_confidence_z
            / half_width_threshold
        )
        ** 2
    )
    return max(1, target - estimator.samples)


def _sampling_protocol_saturates_to_census(
    *,
    protocol: SamplingProtocol,
    region: StateSpaceRegion | None,
) -> bool:
    if region is None:
        return False
    if region.ambient.distinguishability.kind != "exact":
        return False
    if region.measure_estimate is not None and region.measure_estimate.estimated:
        return False
    if protocol.census_budget is None:
        return protocol.kind == "census"
    return protocol.census_budget >= region.volume


def _census_sample_indices(region: StateSpaceRegion | None) -> tuple[int, ...] | None:
    if region is None:
        return None
    return tuple(range(region.volume))


@dataclass(frozen=True, slots=True)
class _PhaseWorkEstimate:
    compute_per_sample: float
    bytes_per_sample: float
    compute_source: str


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
    curriculum_expansion_learning_rates: tuple[float, ...] = ()

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
        if self.curriculum_expansion_learning_rates:
            for group, restart_learning_rate in zip(
                self.optimizer.param_groups,
                self.curriculum_expansion_learning_rates,
                strict=True,
            ):
                group["lr"] = max(float(group["lr"]), restart_learning_rate)
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


class _CurriculumExhausted(BenchmarkRunnerError):
    """Raised when a finite benchmark has no further curriculum windows."""


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
class BenchmarkRunPlan:
    """A local benchmark run plan resolved from CLI or workflow inputs."""

    program_path: Path
    benchmark_root: Path
    results_root: Path = Path("results")
    seed: int = 101
    train_steps: int | None = _default_train_steps
    learning_rate: float | None = None
    optimizer: str = "loss-search"
    schedule: str = "none"
    gate_check_interval: int = _default_gate_check_interval
    model_checkpoint_gate_interval: int = _default_model_checkpoint_gate_interval
    gate_decision_rule: str = "score-estimate-plateau"
    rung_competence_threshold: float = _default_rung_competence_threshold
    convergence_patience: int = _default_convergence_patience
    convergence_min_delta: float = _default_convergence_min_delta
    tensor_device: TensorRuntimeDevice = "auto"
    dry_run: bool = False

    def __post_init__(self) -> None:
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
        if self.optimizer not in {"loss-search", "sgd", "adam", "adamw"}:
            raise BenchmarkRunnerError(f"unsupported optimizer: {self.optimizer}")
        if self.optimizer == "loss-search":
            if self.learning_rate is not None:
                raise BenchmarkRunnerError("loss-search optimizer does not accept learning_rate")
            if self.schedule != "none":
                raise BenchmarkRunnerError("loss-search optimizer does not accept a schedule")
        elif self.learning_rate is None or self.learning_rate <= 0:
            raise BenchmarkRunnerError(f"{self.optimizer} optimizer requires learning_rate")
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
        if self.gate_decision_rule != "score-estimate-plateau":
            raise BenchmarkRunnerError(
                f"unsupported gate_decision_rule: {self.gate_decision_rule}"
            )
        if (
            not math.isfinite(float(self.rung_competence_threshold))
            or self.rung_competence_threshold < 0.0
            or self.rung_competence_threshold > 1.0
        ):
            raise BenchmarkRunnerError("rung_competence_threshold must be in [0, 1]")
        if type(self.convergence_patience) is not int or self.convergence_patience < 0:
            raise BenchmarkRunnerError("convergence_patience must be nonnegative")
        if self.convergence_min_delta < 0:
            raise BenchmarkRunnerError("convergence_min_delta must be nonnegative")
        try:
            validate_tensor_runtime_device(self.tensor_device)
        except TensorRuntimeError as error:
            raise BenchmarkRunnerError(str(error)) from error

    @property
    def run_slug(self) -> str:
        """Return the deterministic local run suffix."""

        return (
            f"seed{self.seed}"
            f"-steps{self.train_steps if self.train_steps is not None else 'converge'}"
            f"-{self.training_control_atom}"
        )

    @property
    def training_control_atom(self) -> str:
        """Return a compact identity atom for training/convergence controls."""

        controls = {
            "optimizer": self.optimizer,
            "schedule": self.schedule,
            "gate_check_interval": self.gate_check_interval,
            "model_checkpoint_gate_interval": self.model_checkpoint_gate_interval,
            "gate_decision_rule": self.gate_decision_rule,
            "rung_competence_threshold": float(self.rung_competence_threshold),
            "convergence_patience": self.convergence_patience,
            "convergence_min_delta": float(self.convergence_min_delta),
            "tensor_device": self.tensor_device,
        }
        if self.learning_rate is not None:
            controls["learning_rate"] = float(self.learning_rate)
        return f"train-{ContentDigest.from_value(controls).hex[:12]}"


@dataclass(frozen=True, slots=True)
class BenchmarkRunSummary:
    """Summary of a planned or completed local benchmark run."""

    run_slug: str
    benchmark_id: ProtocolIdentifier
    program_path: Path
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
            "program_path": _portable_record_path(
                self.program_path,
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

    benchmark = load_benchmark(plan.benchmark_root)
    generator = _require_tensor_generator(benchmark.generator)
    target_contract = benchmark.target_contract
    loss_factory = _optional_training_loss_factory(benchmark)
    competence_factory = _optional_training_competence_factory(benchmark)
    accessible_subspace = benchmark.accessible_subspace
    program_identity = _load_program_graph_identity(plan.program_path)
    program_graph_record = program_identity.graph.to_record()
    outcome_space = benchmark.manifest.resolve_outcome_space()
    outcome_ids = _target_contract_outcome_ids(target_contract)
    summary = _run_summary(
        plan=plan,
        benchmark_id=generator.manifest.id,
        model_source_atom=(
            f"program-{program_identity.graph.digest.hex[:12]}"
        ),
    )
    try:
        initial_evaluation_rung = _evaluation_curriculum_rung(
            generator=generator,
            sample_count=1,
            seed=plan.seed,
            index=0,
        )
    except BenchmarkRunnerError as error:
        if "must return tensors or inspectable field metadata" not in str(error):
            raise
        initial_evaluation_rung = None

    if initial_evaluation_rung is not None:
        evaluation_batch = initial_evaluation_rung.batch
        _validate_program_for_batch(
            program=program_identity,
            batch=evaluation_batch,
            target_contract=target_contract,
        )

    if plan.dry_run:
        return summary

    if initial_evaluation_rung is None:
        validation_runtime = resolve_tensor_runtime(plan.tensor_device)
        initial_evaluation_rung = _evaluation_curriculum_rung(
            generator=generator,
            sample_count=1,
            seed=plan.seed,
            index=0,
            runtime=validation_runtime,
            outcome_ids=outcome_ids,
        )
        evaluation_batch = initial_evaluation_rung.batch
        _validate_program_for_batch(
            program=program_identity,
            batch=evaluation_batch,
            target_contract=target_contract,
        )
    else:
        evaluation_batch = initial_evaluation_rung.batch
    input_shape = _batch_sample_input_shape(batch=evaluation_batch)
    output_shape = _expected_program_output_shape(
        batch=evaluation_batch,
        target_contract=target_contract,
    )
    model_inspection = ModelInspectionRecord.from_program_graph(
        id=ProtocolIdentifier.parse(
            f"model-inspections.{_identifier_atom(generator.manifest.id)}."
            f"{summary.run_slug}@0.1.0"
        ),
        program_graph=program_graph_record,
        input_shape=input_shape,
        output_shape=output_shape,
    )

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
    ) -> dict[str, object]:
        return _model_checkpoint_artifact_record(
            checkpoint=checkpoint,
            program_graph=program_graph_record,
            program_path=plan.program_path,
            benchmark_id=summary.benchmark_id,
            run_slug=summary.run_slug,
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
                        program_graph=program_graph_record,
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
                    program_graph=program_graph_record,
                    input_shape=input_shape,
                    output_shape=output_shape,
                )
                if selected_checkpoint is not None
                else model_inspection
            )
        with progress_timings.span("training_progress.checkpoint_records"):
            progress_full_checkpoint_records = tuple(
                checkpoint_record(checkpoint)
                for checkpoint in checkpoint_artifacts
            )
            progress_checkpoint_records = tuple(
                _compact_model_checkpoint_summary_record(record)
                for record in progress_full_checkpoint_records
            )
            selected_checkpoint_record = (
                None
                if selected_checkpoint is None
                else checkpoint_record(selected_checkpoint)
            )
        with progress_timings.span("training_progress.record"):
            progress_record = _training_progress_record(
                plan=plan,
                summary=summary,
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
                selected_model_checkpoint_score_estimate=(
                    None
                    if selected_checkpoint is None
                    or selected_checkpoint.score_estimate is None
                    else selected_checkpoint.score_estimate
                ),
            )
        with progress_timings.span("training_progress.write"):
            _write_document_atomic(progress_path, progress_record)
        if progress_callback is not None:
            progress_callback(summary)

    training_result = _train_and_predict(
        program_path=plan.program_path,
        initial_evaluation_rung=initial_evaluation_rung,
        generator=generator,
        target_contract=target_contract,
        accessible_subspace=accessible_subspace,
        sample_count=_default_training_batch_target,
        gate_sample_count=_default_gate_batch_target,
        train_steps=plan.train_steps,
        learning_rate=plan.learning_rate,
        optimizer_name=plan.optimizer,
        schedule_name=plan.schedule,
        gate_check_interval=plan.gate_check_interval,
        gate_decision_rule=plan.gate_decision_rule,
        rung_competence_threshold=plan.rung_competence_threshold,
        convergence_patience=plan.convergence_patience,
        convergence_min_delta=float(plan.convergence_min_delta),
        tensor_device=plan.tensor_device,
        storage_bytes=model_inspection.cost_summary.storage_bytes,
        batch_size=_default_training_batch_target,
        seed=plan.seed,
        progress_callback=publish_progress,
        loss_factory=loss_factory,
        competence_factory=competence_factory,
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
        program_graph=program_graph_record,
        input_shape=input_shape,
        output_shape=output_shape,
    )
    full_checkpoint_records = tuple(
        checkpoint_record(checkpoint)
        for checkpoint in checkpoint_artifacts
    )
    for checkpoint, record in zip(checkpoint_artifacts, full_checkpoint_records, strict=True):
        _write_document(
            checkpoint.path.with_suffix(".checkpoint" + _document_suffix),
            record,
        )
    checkpoint_records = tuple(
        _compact_model_checkpoint_summary_record(record)
        for record in full_checkpoint_records
    )
    selected_checkpoint_record = checkpoint_record(selected_checkpoint)
    completed_training_estimate = _training_estimate_record(
        summary=summary,
        training_run=training_result.training_run,
    )
    completed_record = {
        **summary.to_record(),
        "dry_run": False,
        "run_status": "completed",
        "training_curriculum": _curriculum_record(
            kind="competence-gated-training-curriculum",
            source="structured-training-curriculum",
            frontier_sampling_weight=0.5,
            replay_sampling_weight=0.5,
            rung_competence_threshold=plan.rung_competence_threshold,
            rungs=training_result.training_rungs,
            frontier_index=training_result.training_frontier_index,
        ),
        "training_estimate": completed_training_estimate,
        "seed": plan.seed,
        "train_steps": plan.train_steps,
        "optimizer": plan.optimizer,
        "schedule": plan.schedule,
        "gate_check_interval": plan.gate_check_interval,
        "model_checkpoint_gate_interval": plan.model_checkpoint_gate_interval,
        "gate_decision_rule": plan.gate_decision_rule,
        "convergence_patience": plan.convergence_patience,
        "convergence_min_delta": float(plan.convergence_min_delta),
        "tensor_runtime": "pytorch",
        "tensor_device": training_result.training_run.protocol.tensor_device,
        "training_run": _training_run_artifact_record(training_result.training_run),
        "throughput": training_result.throughput,
        "program": model_inspection.program.to_record(),
        "program_graph": program_graph_record,
        "cost_summary": _training_cost_summary(
            inspection=model_inspection,
            training_estimate=completed_training_estimate,
            training_run=training_result.training_run,
        ),
        "model_checkpoints": [dict(record) for record in checkpoint_records],
        "selected_model_checkpoint": selected_checkpoint_record,
        "selected_model_checkpoint_score_estimate": (
            None
            if selected_checkpoint.score_estimate is None
            else dict(selected_checkpoint.score_estimate)
        ),
        "selected_model_checkpoint_policy": "highest-training-score-estimate",
        "evaluation_model_artifact": selected_checkpoint_record,
    }
    if plan.learning_rate is not None:
        completed_record["learning_rate"] = float(plan.learning_rate)
    _write_document_atomic(summary.training_summary_path, completed_record)
    if progress_path != summary.training_summary_path:
        progress_path.unlink(missing_ok=True)
    return summary


def _run_summary(
    *,
    plan: BenchmarkRunPlan,
    benchmark_id: ProtocolIdentifier,
    model_source_atom: str,
) -> BenchmarkRunSummary:
    benchmark_atom = _identifier_atom(benchmark_id)
    run_slug = f"{benchmark_atom}-{model_source_atom}-{plan.run_slug}"
    return BenchmarkRunSummary(
        run_slug=run_slug,
        benchmark_id=benchmark_id,
        program_path=plan.program_path,
        measurement_count=0,
        training_summary_path=(
            plan.results_root / "training" / benchmark_atom / f"{run_slug}{_document_suffix}"
        ),
        model_artifact_root=(plan.results_root / "models" / benchmark_atom / run_slug),
        dry_run=plan.dry_run,
        results_root=plan.results_root,
    )


def _load_program_graph_identity(program_path: Path) -> LoadedProgramGraph:
    try:
        return load_program_graph(program_path, resolve_host_tensor_runtime())
    except (ProgramGraphError, TensorRuntimeError) as error:
        raise BenchmarkRunnerError(str(error)) from error


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

    workflow_timings = TimingCollector()
    with workflow_timings.span("evaluation_workflow.load_generator"):
        benchmark = load_benchmark(plan.benchmark_root)
        generator = _require_tensor_generator(benchmark.generator)
        target_contract = benchmark.target_contract
    with workflow_timings.span("evaluation_workflow.load_checkpoint_input"):
        evaluation_input = _evaluation_input_from_plan(plan, generator=generator)
        outcome_space = generator.manifest.resolve_outcome_space()
        selected_checkpoint = evaluation_input.checkpoint
        run_slug = evaluation_input.run_slug
        benchmark_id = evaluation_input.benchmark_id
        benchmark_atom = _identifier_atom(benchmark_id)
        evaluation_bundle_path = (
            plan.results_root / "evaluations" / benchmark_atom / f"{run_slug}{_document_suffix}"
        )
        evaluation_seed = _unpredictable_evaluation_seed()
    with workflow_timings.span("evaluation_workflow.checkpoint_evaluation"):
        (
            evaluation_results,
            final_evaluation_batch,
            final_evaluation_probabilities,
            final_accepted_mass,
            final_competence_diagnostics,
            checkpoint_evaluation_throughput,
        ) = evaluate_model_checkpoint_artifact(
            program_path=evaluation_input.program_path,
            generator=generator,
            target_contract=target_contract,
            competence_factory=_optional_training_competence_factory(benchmark),
            accessible_subspace=benchmark.accessible_subspace,
            sampling_protocol=benchmark.sampling_protocol,
            seed=evaluation_seed,
            tensor_device=plan.tensor_device,
            checkpoint=selected_checkpoint,
        )
    with workflow_timings.span("evaluation_workflow.integration_evidence"):
        outcome_ids = _target_contract_outcome_ids(target_contract)
        evaluation_frontier_index = _evaluation_result_frontier_index(
            evaluation_results=evaluation_results,
            outcome_ids=outcome_ids,
            target_contract=target_contract,
        )
        evaluation_integration = _evaluation_integration_evidence(
            evaluation_results=evaluation_results,
            outcome_ids=outcome_ids,
            target_contract=target_contract,
        )
    with workflow_timings.span(
        "evaluation_workflow.final_measurement_records",
        samples=final_evaluation_batch.sample_count,
    ):
        frontier_rung = evaluation_results[evaluation_frontier_index].rung
        final_scored_batch = replace(
            final_evaluation_batch,
            volume_request=_rung_score_volume_request(frontier_rung),
        )
        if target_contract.kind == "field-valued":
            measurement_groups: tuple[tuple[MeasurementRecord, ...], ...] = ()
            final_sampled_competence = _sampled_competence_record_from_accepted_mass(
                batch=final_scored_batch,
                accepted_mass=final_accepted_mass,
                volume_axis=None,
                bounded_mass=False,
            )
            _attach_competence_diagnostics(
                final_sampled_competence,
                final_competence_diagnostics,
            )
        else:
            final_measurements = finite_measurements_for_predictions(
                batch=final_evaluation_batch,
                outcome_space=outcome_space,
                probabilities=final_evaluation_probabilities,
                run_slug=f"{run_slug}.final",
            )
            measurement_groups = (final_measurements,)
            final_sampled_competence = sampled_competence_record(
                batch=final_scored_batch,
                measurements=final_measurements,
                volume_axis=None,
                input_shape=evaluation_results[evaluation_frontier_index].input_shape,
            )
        final_cost_measurement, final_cost_sample_count = (
            _checkpoint_evaluation_throughput_inference_cost_record(
                checkpoint_evaluation_throughput
            )
        )
        final_sampled_competence["inference_cost_measurement"] = final_cost_measurement
        final_sampled_competence["inference_cost_sample_count"] = final_cost_sample_count
        final_sampled_competence["confidence_half_width"] = (
            evaluation_results[evaluation_frontier_index].confidence_half_width
        )
        final_sampled_competence["confidence_method_id"] = (
            evaluation_results[evaluation_frontier_index].confidence_method_id
        )
        final_sampled_competence["sampling_protocol"] = (
            evaluation_results[evaluation_frontier_index].sampling_protocol.to_record()
        )
        final_sampled_competence["sampling_seed"] = frontier_rung.seed
        sampled_competence = _evaluation_sampled_competence_record(
            benchmark_id=benchmark_id,
            evaluation_results=evaluation_results,
            frontier_point=final_sampled_competence,
            frontier_index=evaluation_frontier_index,
        )
    with workflow_timings.span("evaluation_workflow.measurement_dataset"):
        measurements = tuple(
            measurement
            for group in measurement_groups
            for measurement in group
        )
        dataset = MeasurementDataset(measurements=measurements)
        dataset.validate_manifest(generator.manifest)
    with workflow_timings.span("evaluation_workflow.model_inspection"):
        model_inspection = ModelInspectionRecord.from_model_manifest(
            id=ProtocolIdentifier.parse(
                f"model-inspections.{benchmark_atom}.{run_slug}@0.1.0"
            ),
            model_manifest=selected_checkpoint.manifest,
            program_graph=evaluation_input.program_graph,
            input_shape=_batch_sample_input_shape(batch=final_evaluation_batch),
            output_shape=_expected_program_output_shape(
                batch=final_evaluation_batch,
                target_contract=target_contract,
            ),
        )
    with workflow_timings.span("evaluation_workflow.curriculum_record"):
        evaluation_curriculum = _curriculum_record(
            kind="checkpoint-benchmark-evaluation-curriculum",
            rungs=tuple(result.rung for result in evaluation_results),
            frontier_index=evaluation_frontier_index,
        )
        evaluation_curriculum_rungs = cast(
            list[dict[str, object]],
            evaluation_curriculum["rungs"],
        )
        for rung_record, result in zip(
            evaluation_curriculum_rungs,
            evaluation_results,
            strict=True,
        ):
            rung_record["mean_accepted_mass"] = result.mean_accepted_mass
            rung_record["confidence_half_width"] = result.confidence_half_width
            rung_record["confidence_method_id"] = result.confidence_method_id
            rung_record["sampling_protocol"] = result.sampling_protocol.to_record()
            rung_record["sampling_seed"] = result.rung.seed
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
        "score_status": (
            "provisional"
            if _checkpoint_evaluation_capacity_limited(checkpoint_evaluation_throughput)
            else "accepted"
        ),
        "evaluation_convergence": {
            "kind": "accepted-mass-confidence-half-width",
            "minimum_sample_count": _default_evaluation_convergence_min_samples,
            "confidence_z": _default_evaluation_convergence_confidence_z,
            "half_width_threshold": _default_evaluation_convergence_half_width,
        },
        "integration_convergence": {
            "kind": "adaptive-score-integral-confidence",
            "score_integral": evaluation_integration.score_integral_value,
            "score_integral_half_width": (
                evaluation_integration.score_integral_half_width
            ),
            "score_integral_half_width_threshold": (
                evaluation_integration.score_integral_half_width_threshold
            ),
            "score_integral_relative_half_width_threshold": (
                _default_evaluation_integral_relative_half_width
            ),
            "score_integral_minimum_half_width_threshold": (
                _default_evaluation_integral_minimum_half_width
            ),
            "terminal_failure_count": evaluation_integration.terminal_failure_count,
            "terminal_failure_threshold": _default_evaluation_terminal_failure_rungs,
            "converged": evaluation_integration.converged,
            "curriculum_exhausted": (
                checkpoint_evaluation_throughput.get("curriculum_exhausted") is True
            ),
        },
        "evaluation_curriculum_rung_count": len(evaluation_results),
        "tensor_runtime": "pytorch",
        "tensor_device": evaluation_tensor_device,
        "requested_tensor_device": plan.tensor_device,
    }
    checkpoint_record = _model_checkpoint_artifact_record(
        checkpoint=selected_checkpoint,
        program_graph=evaluation_input.program_graph,
        program_path=evaluation_input.program_path,
        benchmark_id=benchmark_id,
        run_slug=run_slug,
        results_root=plan.results_root,
    )
    with workflow_timings.span("evaluation_workflow.bundle_construction"):
        bundle = BenchmarkEvaluationBundle(
            id=ProtocolIdentifier.parse(f"benchmark-evaluations.{identifier_stem}@0.1.0"),
            run_slug=run_slug,
            benchmark_manifest=generator.manifest,
            program_graph=evaluation_input.program_graph,
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
    with workflow_timings.span("evaluation_workflow.bundle_record"):
        bundle_record = bundle.to_record()
    throughput["workflow_phase_timing"] = workflow_timings.to_record(
        kind="benchmark-evaluation-workflow-phase-timing"
    )
    bundle_record["throughput"] = throughput
    _write_document(evaluation_bundle_path, bundle_record)
    return BenchmarkEvaluationSummary(
        run_slug=run_slug,
        benchmark_id=benchmark_id,
        evaluation_bundle_path=evaluation_bundle_path,
        measurement_count=len(measurements),
    )


def _checkpoint_evaluation_capacity_limited(throughput: Mapping[str, object]) -> bool:
    return throughput.get("capacity_limited") is True


def _checkpoint_evaluation_throughput_inference_cost_record(
    throughput: Mapping[str, object],
) -> tuple[dict[str, object], int]:
    measurement = CostMeasurement.from_record(throughput.get("inference_cost_measurement"))
    sample_count = _required_int(
        throughput.get("inference_cost_sample_count"),
        "checkpoint_evaluation.inference_cost_sample_count",
    )
    if sample_count < 1:
        raise BenchmarkRunnerError(
            "checkpoint_evaluation.inference_cost_sample_count must be positive"
        )
    return measurement.without_operation_trace().to_record(), sample_count


def _evaluation_input_from_plan(
    plan: BenchmarkEvaluationPlan,
    *,
    generator: BenchmarkGenerator,
) -> _EvaluationInput:
    checkpoint_artifact_path = _evaluation_checkpoint_artifact_path(plan)
    checkpoint_record = _load_object_record(
        checkpoint_artifact_path,
        description="checkpoint artifact",
    )
    program_graph = _extract_record(
        checkpoint_record.get("program_graph"),
        "checkpoint_artifact.program_graph",
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
    program_path = _checkpoint_program_path(
        checkpoint_record.get("program_path"),
        results_root=plan.results_root,
    )
    return _EvaluationInput(
        program_path=program_path,
        program_graph=program_graph,
        checkpoint=load_model_checkpoint_artifact(
            checkpoint_record,
            results_root=plan.results_root,
        ),
        run_slug=run_slug,
        benchmark_id=checkpoint_benchmark_id,
    )


def _evaluation_checkpoint_artifact_path(plan: BenchmarkEvaluationPlan) -> Path:
    path = plan.checkpoint_artifact_path
    if path.is_absolute():
        return path
    if path.parts[:1] == (plan.results_root.name,):
        return _resolve_artifact_record_path(
            path.as_posix(),
            results_root=plan.results_root,
        )
    if path.exists():
        return path
    try:
        resolved = _resolve_artifact_record_path(
            path.as_posix(),
            results_root=plan.results_root,
        )
    except BenchmarkRunnerError:
        return path
    return resolved if resolved.exists() else path


def _evaluation_curriculum_rung(
    *,
    generator: _TensorBenchmarkGenerator,
    sample_count: int = 1,
    seed: int,
    index: int,
    planner: _VolumeCurriculumPlanner | None = None,
    runtime: TensorRuntime | None = None,
    outcome_ids: tuple[str, ...] | None = None,
) -> _CurriculumRung:
    planner = _VolumeCurriculumPlanner() if planner is None else planner
    return _curriculum_rung_from_window(
        generator=generator,
        sample_count=sample_count,
        seed=seed,
        index=index,
        runtime=runtime,
        outcome_ids=outcome_ids,
        window=planner.next(),
        include_fields=True,
    )


def _training_curriculum_rung(
    *,
    generator: _TensorBenchmarkGenerator,
    sample_count: int,
    seed: int,
    index: int,
    planner: _VolumeCurriculumPlanner | None = None,
    phase_timings: TimingCollector | None = None,
) -> _CurriculumRung:
    planner = _VolumeCurriculumPlanner() if planner is None else planner
    with _optional_timing_span(
        phase_timings,
        "training_frontier.rung_window_generation",
    ):
        window = planner.next()
    with _optional_timing_span(
        phase_timings,
        "training_frontier.rung_record_construction",
    ):
        return _curriculum_rung_from_window(
            generator=generator,
            sample_count=sample_count,
            seed=seed,
            index=index,
            runtime=None,
            outcome_ids=None,
            window=window,
            include_fields=False,
        )


class _EmptyCurriculumWindow(Exception):
    """Raised when a concrete curriculum window materializes no samples."""


def _curriculum_rung_from_window(
    *,
    generator: _TensorBenchmarkGenerator,
    sample_count: int,
    seed: int,
    index: int,
    runtime: TensorRuntime | None,
    outcome_ids: tuple[str, ...] | None,
    window: _CurriculumWindow,
    include_fields: bool,
) -> _CurriculumRung:
    rung_seed = seed if index == 0 else seed + 2_000_003 * index
    sample_set = generator(
        shape=sample_count,
        seed=rung_seed,
        include_fields=include_fields,
        volume_request=window.request,
        variation_extent=_full_variation_extent,
        runtime=runtime,
        outcome_ids=outcome_ids,
    )
    batch = sample_set
    if not batch.samples:
        if batch.request_outcome is not None:
            raise _EmptyCurriculumWindow(str(batch.request_outcome.kind))
        raise _EmptyCurriculumWindow()
    materialization_plan = batch.samples[0].materialization_plan
    return _CurriculumRung(
        index=index,
        resolution_assignment=(
            None if materialization_plan is None else materialization_plan.resolution_assignment
        ),
        seed=batch.seed,
        batch=batch,
        sample_count=len(batch.samples),
        log2_volume_minimum=window.minimum,
        log2_volume_maximum=window.maximum,
    )


@dataclass(frozen=True, slots=True)
class _CurriculumWindow:
    index: int

    @property
    def minimum(self) -> float:
        return float(self.index)

    @property
    def maximum(self) -> float:
        return float(self.index + 1)

    @property
    def log2_volume(self) -> float:
        return self.maximum

    @property
    def request(self) -> StateSpaceVolumeRequest:
        return StateSpaceVolumeRequest(minimum=self.minimum, maximum=self.maximum)


@dataclass(slots=True)
class _VolumeCurriculumPlanner:
    next_index: int = 0

    def next(self) -> _CurriculumWindow:
        if self.next_index < 0:
            raise BenchmarkRunnerError("curriculum window index must be nonnegative")
        window = _CurriculumWindow(index=self.next_index)
        self.next_index += 1
        return window

def _optional_timing_span(timing: TimingCollector | None, phase: str) -> Any:
    if timing is None:
        return nullcontext()
    return timing.span(phase)


def _curriculum_record(
    *,
    kind: str,
    rungs: Sequence[_CurriculumRung],
    frontier_index: int,
    source: str | None = None,
    frontier_sampling_weight: float | None = None,
    replay_sampling_weight: float | None = None,
    rung_competence_threshold: float | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "kind": kind,
        "curriculum_variable": "log2-state-space-volume",
        "volume_axis": _core_volume_measure_id(),
        "sampling_levers": ["log2-state-space-volume"],
        "volume_value": {
            "measure_id": _core_volume_measure_id(),
            "scale": "log2",
        },
        "window_policy": {
            "kind": "integer-bit-shells",
        },
        "claim_policy": {
            "kind": "benchmark-windowed-increments",
            "proposal_policy": "benchmark-canonical-integer-bit-windows",
        },
        "gating_metric": "monotone-frontier-validation-competence",
        "rung_policy": "unbounded-competence-frontier",
        "frontier_index": frontier_index,
        "unlocked_rung_count": min(len(rungs), frontier_index + 1),
        "claim_chain": [
            region.to_record()
            for region in _claim_chain_regions(
                rungs[: min(len(rungs), frontier_index + 1)]
            )
        ],
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
    if rung_competence_threshold is not None:
        record["rung_competence_threshold"] = rung_competence_threshold
    return record


def _validate_claim_chain(
    rungs: Sequence[_CurriculumRung],
    *,
    accessible_subspace: AccessibleSubspace,
) -> None:
    if not rungs:
        raise BenchmarkRunnerError("claim chain must not be empty")
    first_interval = _rung_log2_volume_interval(rungs[0])
    if first_interval is None or first_interval[0] > 0.0:
        raise BenchmarkRunnerError("claim chain must claim the base region first")
    regions = _claim_chain_regions(rungs)
    _validate_claim_chain_disjoint(regions)
    for region in regions:
        for exclusion in accessible_subspace.exclusions:
            if _state_space_regions_overlap(region, exclusion):
                raise BenchmarkRunnerError(
                    "claim chain increment intersects an accessible-subspace exclusion"
                )
    _validate_claim_chain_cumulative_brackets(rungs, regions)


def _claim_chain_regions(rungs: Sequence[_CurriculumRung]) -> tuple[StateSpaceRegion, ...]:
    regions: list[StateSpaceRegion] = []
    for rung in rungs:
        if rung.batch.region is None:
            raise BenchmarkRunnerError("claim chain increments require realized regions")
        regions.append(rung.batch.region)
    return tuple(regions)


def _validate_claim_chain_disjoint(regions: Sequence[StateSpaceRegion]) -> None:
    for earlier in range(len(regions)):
        for later in range(earlier + 1, len(regions)):
            if _state_space_regions_overlap(regions[earlier], regions[later]):
                raise BenchmarkRunnerError("claim chain increments must be pairwise disjoint")


def _state_space_regions_overlap(left: StateSpaceRegion, right: StateSpaceRegion) -> bool:
    try:
        return not state_space_regions_are_disjoint(left, right)
    except StateSpaceError as error:
        if "regions in different ambients are not comparable" in str(error):
            # Claim-chain overlap is only meaningful inside one state-space ambient.
            # Incomparable ambients are separate capacity scopes, so cumulative
            # volume brackets still constrain the chain but no state can be shared.
            return False
        raise


def _validate_claim_chain_cumulative_brackets(
    rungs: Sequence[_CurriculumRung],
    regions: Sequence[StateSpaceRegion],
) -> None:
    running_volume = 0
    for rung, region in zip(rungs, regions, strict=True):
        running_volume += region.volume
        interval = _rung_log2_volume_interval(rung)
        if interval is None:
            continue
        lower, upper = interval
        cumulative_log2_volume = math.log2(running_volume)
        if (
            cumulative_log2_volume < lower - 1e-9
            or cumulative_log2_volume > upper + 1e-9
        ):
            raise BenchmarkRunnerError(
                "claim chain cumulative volume is outside the proposed bracket"
            )


def _validate_program_for_batch(
    *,
    program: LoadedProgramGraph,
    batch: GeneratedSampleSet,
    target_contract: TargetContract,
) -> None:
    runtime = resolve_host_tensor_runtime()
    input_shapes = _program_input_shapes_for_batch(batch=batch, target_contract=target_contract)
    additional_input_shapes = _program_additional_input_shapes(
        input_shapes=input_shapes,
        target_contract=target_contract,
    )
    try:
        program.graph.validate(
            runtime,
            input_shapes=input_shapes,
            additional_input_shapes=additional_input_shapes,
            require_differentiable=False,
            batch_size=1,
        )
    except (ProgramGraphError, TensorRuntimeError) as error:
        raise BenchmarkRunnerError(str(error)) from error
    sample_shape = _batch_sample_input_shape(batch=batch)
    field_shape = sample_shape if target_contract.kind == "field-valued" else None
    try:
        expected_output_shape = target_contract.expected_output_shape(field_shape)
    except TargetContractError as error:
        raise BenchmarkRunnerError(str(error)) from error
    output_shape = _batchless_program_output_shape(
        program=program,
        input_shapes=input_shapes,
        target_contract=target_contract,
    )
    if output_shape != expected_output_shape:
        raise BenchmarkRunnerError(
            f"program output shape {output_shape} does not match "
            f"target contract output shape {expected_output_shape}"
        )


def _program_input_shapes_for_batch(
    *,
    batch: GeneratedSampleSet,
    target_contract: TargetContract,
) -> tuple[tuple[int, ...], ...]:
    sample_shape = _batch_sample_input_shape(batch=batch)
    if target_contract.kind == "field-valued":
        return (sample_shape, ())
    return (sample_shape,)


def _program_additional_input_shapes(
    *,
    input_shapes: tuple[tuple[int, ...], ...],
    target_contract: TargetContract,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if target_contract.kind != "field-valued":
        return ()
    field_shape = input_shapes[0]
    varied = (*field_shape[:-1], max(1, field_shape[-1] + 1))
    return ((varied, ()),)


def _batchless_program_output_shape(
    *,
    program: LoadedProgramGraph,
    input_shapes: tuple[tuple[int, ...], ...],
    target_contract: TargetContract,
) -> tuple[int, ...]:
    output = program.graph.validate(
        resolve_host_tensor_runtime(),
        input_shapes=input_shapes,
        additional_input_shapes=_program_additional_input_shapes(
            input_shapes=input_shapes,
            target_contract=target_contract,
        ),
        require_differentiable=False,
        batch_size=1,
    ).output_shapes[0]
    if len(output) != 1:
        raise BenchmarkRunnerError("program must produce exactly one output tensor")
    return output[0]


def _expected_program_output_shape(
    *,
    batch: GeneratedSampleSet,
    target_contract: TargetContract,
) -> tuple[int, ...]:
    sample_shape = _batch_sample_input_shape(batch=batch)
    field_shape = sample_shape if target_contract.kind == "field-valued" else None
    try:
        return target_contract.expected_output_shape(field_shape)
    except TargetContractError as error:
        raise BenchmarkRunnerError(str(error)) from error


def _train_and_predict(
    *,
    program_path: Path,
    initial_evaluation_rung: _CurriculumRung,
    generator: _TensorBenchmarkGenerator,
    target_contract: TargetContract,
    accessible_subspace: AccessibleSubspace,
    sample_count: int,
    gate_sample_count: int,
    train_steps: int | None,
    learning_rate: float | None,
    optimizer_name: str,
    schedule_name: str,
    gate_check_interval: int,
    gate_decision_rule: str,
    rung_competence_threshold: float,
    convergence_patience: int,
    convergence_min_delta: float,
    tensor_device: TensorRuntimeDevice,
    storage_bytes: int | None,
    batch_size: int,
    seed: int,
    loss_factory: _BenchmarkTrainingLossFactory | None = None,
    competence_factory: _BenchmarkTrainingCompetenceFactory | None = None,
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
                program_path=program_path,
                initial_evaluation_rung=initial_evaluation_rung,
                generator=generator,
                target_contract=target_contract,
                accessible_subspace=accessible_subspace,
                sample_count=sample_count,
                gate_sample_count=gate_sample_count,
                train_steps=train_steps,
                learning_rate=learning_rate,
                optimizer_name=optimizer_name,
                schedule_name=schedule_name,
                gate_check_interval=gate_check_interval,
                gate_decision_rule=gate_decision_rule,
                rung_competence_threshold=rung_competence_threshold,
                convergence_patience=convergence_patience,
                convergence_min_delta=convergence_min_delta,
                tensor_device=device_kind,
                storage_bytes=storage_bytes,
                batch_size=batch_size,
                seed=seed,
                loss_factory=loss_factory,
                competence_factory=competence_factory,
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
    program_path: Path,
    initial_evaluation_rung: _CurriculumRung,
    generator: _TensorBenchmarkGenerator,
    target_contract: TargetContract,
    accessible_subspace: AccessibleSubspace,
    sample_count: int,
    gate_sample_count: int,
    train_steps: int | None,
    learning_rate: float | None,
    optimizer_name: str,
    schedule_name: str,
    gate_check_interval: int,
    gate_decision_rule: str,
    rung_competence_threshold: float,
    convergence_patience: int,
    convergence_min_delta: float,
    tensor_device: TensorRuntimeDeviceKind,
    storage_bytes: int | None,
    batch_size: int,
    seed: int,
    loss_factory: _BenchmarkTrainingLossFactory | None = None,
    competence_factory: _BenchmarkTrainingCompetenceFactory | None = None,
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
    module = _build_training_module(
        runtime=runtime,
        program_path=program_path,
    )
    outcome_ids = _target_contract_outcome_ids(target_contract)
    chance_mass = _target_contract_chance_mass(target_contract)
    if target_contract.kind == "field-valued":
        _validate_training_field_stepper(
            runtime=runtime,
            module=module,
            batch=initial_evaluation_rung.batch,
            target_contract=target_contract,
            outcome_ids=outcome_ids,
        )
    loss_function = _build_training_loss(
        runtime=runtime,
        target_contract=target_contract,
        loss_factory=loss_factory,
    )
    competence_functional = _resolve_competence_functional(
        target_contract,
        runtime=runtime,
        competence_factory=competence_factory,
    )
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
    validation_counter = _ThroughputCounter()
    evaluation_counter = _ThroughputCounter()
    phase_timings = _runtime_phase_timings(runtime)
    training_curriculum_planner = _VolumeCurriculumPlanner()

    def training_batch_for_seed(
        batch_seed: int,
        *,
        batch_sample_count: int,
        generation_phase: str,
        tensor_phase: str,
        rung: _CurriculumRung,
        include_score_metadata: bool,
    ) -> _TrainingStepBatch:
        physical_sample_count = _physical_execution_sample_count(
            runtime=runtime,
            generator=generator,
            rung=rung,
            requested_sample_count=batch_sample_count,
            outcome_count=len(outcome_ids),
            phase_timings=phase_timings,
            phase=generation_phase,
        )
        with phase_timings.span(generation_phase, samples=physical_sample_count):
            generated = generator(
                shape=physical_sample_count,
                seed=batch_seed,
                include_metadata=include_score_metadata,
                volume_request=_rung_volume_request(rung),
                memory_limit_bytes=_runtime_memory_budget_bytes(runtime),
                variation_extent=_full_variation_extent,
                runtime=runtime,
                outcome_ids=outcome_ids,
                timing=phase_timings,
                timing_prefix=f"{generation_phase}.",
            )
            if generated.sample_count == 0:
                raise BenchmarkRunnerError(
                    "generator returned no samples for the selected volume window"
                )
            with phase_timings.span(tensor_phase, samples=physical_sample_count):
                fields, labels = generated.require_tensors()
                return _TrainingStepBatch(
                    fields=fields,
                    labels=labels,
                    horizons=_field_valued_target_horizons(
                        batch=generated,
                        labels=labels,
                        target_contract=target_contract,
                    ),
                    sample_set=(
                        replace(
                            generated,
                            volume_request=_rung_score_volume_request(rung),
                        )
                        if include_score_metadata
                        else None
                    ),
                )

    def validation_sample_batch_for_seed(
        batch_seed: int,
        *,
        batch_sample_count: int,
        generation_phase: str,
        rung: _CurriculumRung,
    ) -> GeneratedSampleSet:
        physical_sample_count = _physical_execution_sample_count(
            runtime=runtime,
            generator=generator,
            rung=rung,
            requested_sample_count=batch_sample_count,
            outcome_count=len(outcome_ids),
            phase_timings=phase_timings,
            phase=generation_phase,
        )
        with phase_timings.span(generation_phase, samples=physical_sample_count):
            generated = generator(
                shape=physical_sample_count,
                seed=batch_seed,
                include_fields=False,
                volume_request=_rung_volume_request(rung),
                memory_limit_bytes=_runtime_memory_budget_bytes(runtime),
                variation_extent=_full_variation_extent,
                runtime=runtime,
                outcome_ids=outcome_ids,
                timing=phase_timings,
                timing_prefix=f"{generation_phase}.",
            )
            if generated.sample_count == 0:
                raise BenchmarkRunnerError(
                    "generator returned no samples for the selected volume window"
                )
            if not generated.includes_fields:
                raise BenchmarkRunnerError("validation gate batch did not include fields")
            return replace(
                generated,
                volume_request=_rung_score_volume_request(rung),
            )

    training_rungs: list[_CurriculumRung] = [
        _training_curriculum_rung(
            generator=generator,
            sample_count=sample_count,
            seed=seed,
            index=0,
            planner=training_curriculum_planner,
            phase_timings=phase_timings,
        )
    ]
    _validate_claim_chain(
        training_rungs,
        accessible_subspace=accessible_subspace,
    )
    training_frontier_index = 0
    frontier_plateau_points: list[ValidationCompetencePoint] = []

    def current_frontier() -> _CurriculumRung:
        return training_rungs[training_frontier_index]

    def training_rung_for_step(step: int) -> _CurriculumRung:
        return training_rungs[
            _training_rung_index_for_step(
                step=step,
                frontier_index=training_frontier_index,
            )
        ]

    def training_batch_for_step(step: int) -> _TrainingStepBatch:
        rung = training_rung_for_step(step)
        return training_batch_for_seed(
            seed + step,
            batch_sample_count=sample_count,
            generation_phase="training_formation_generation",
            tensor_phase="training_tensor_batch",
            rung=rung,
            include_score_metadata=rung.index != training_frontier_index,
        )

    def advance_frontier(history: Sequence[TrainingHistoryPoint]) -> bool:
        nonlocal training_frontier_index
        latest = history[-1]
        frontier_point = _training_history_frontier_point(latest)
        with phase_timings.span("training_frontier.advance_decision"):
            should_advance = validation_competence_frontier_advances(
                frontier_point=frontier_point,
                previous_frontier_points=tuple(frontier_plateau_points),
                chance_mass=chance_mass,
            )
        if not should_advance:
            return False
        next_index = training_frontier_index + 1
        with phase_timings.span("training_frontier.rung_append"):
            try:
                next_rung = _training_curriculum_rung(
                    generator=generator,
                    sample_count=sample_count,
                    seed=seed,
                    index=next_index,
                    planner=training_curriculum_planner,
                    phase_timings=phase_timings,
                )
            except _CurriculumExhausted:
                return False
            _validate_claim_chain(
                (*training_rungs, next_rung),
                accessible_subspace=accessible_subspace,
            )
            training_rungs.append(next_rung)
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
        train_batch=training_batch_for_step,
        validation_batch=lambda check: validation_sample_batch_for_seed(
            seed + 1_000_003 + check,
            batch_sample_count=gate_sample_count,
            generation_phase="validation_formation_generation",
            rung=current_frontier(),
        ),
        generator=generator,
        target_contract=target_contract,
        competence=competence_functional,
        outcome_ids=outcome_ids,
        max_steps=train_steps,
        training_batch_target=sample_count,
        gate_check_interval=gate_check_interval,
        gate_batch_target=gate_sample_count,
        patience=convergence_patience,
        min_delta=convergence_min_delta,
        rung_competence_threshold=rung_competence_threshold,
        training_counter=training_counter,
        validation_counter=validation_counter,
        phase_timings=phase_timings,
        on_plateau=advance_frontier,
        frontier_points=lambda: tuple(frontier_plateau_points),
        on_gate_check=lambda history, checked_module: (
            progress_callback(
                _running_training_run_record(
                    seed=seed,
                    max_steps=train_steps,
                    learning_rate=learning_rate,
                    optimizer_name=optimizer_name,
                    schedule_name=schedule_name,
                    gate_check_interval=gate_check_interval,
                    gate_decision_rule=gate_decision_rule,
                    rung_competence_threshold=rung_competence_threshold,
                    convergence_patience=convergence_patience,
                    convergence_min_delta=convergence_min_delta,
                    tensor_device=runtime.device_kind,
                    runtime_memory_budget_fraction=(
                        _default_runtime_memory_budget_fraction
                        if tensor_runtime_has_fixed_device_memory(runtime)
                        else None
                    ),
                    validation_history=history,
                ),
                _throughput_record(
                    runtime_device=runtime.device_kind,
                    training_counter=training_counter,
                    validation_counter=validation_counter,
                    evaluation_counter=evaluation_counter,
                    roofline=runtime_roofline_record(runtime),
                    work_estimates=_training_work_estimates(
                        input_shape=_batch_sample_input_shape(batch=current_frontier().batch),
                        output_shape=_expected_program_output_shape(
                            batch=current_frontier().batch,
                            target_contract=target_contract,
                        ),
                        inference_cost=_module_projected_inference_cost_measurement(
                            runtime=runtime,
                            module=checked_module,
                            input_shape=_batch_sample_input_shape(batch=current_frontier().batch),
                            batch_size=batch_size,
                            target_contract=target_contract,
                        ),
                        training_cost=_training_history_latest_training_cost_measurement(
                            history
                        ),
                        storage_bytes=storage_bytes,
                        batch_size=batch_size,
                    ),
                    phase_timings=phase_timings,
                    fallback_errors=fallback_errors,
                    operation_fallbacks=module.operation_fallback_records(),
                    tensor_compile_fallbacks=tensor_element_compile_fallback_records(),
                ),
                _curriculum_record(
                    kind="competence-gated-training-curriculum",
                    source="structured-training-curriculum",
                    frontier_sampling_weight=0.5,
                    replay_sampling_weight=0.5,
                    rung_competence_threshold=rung_competence_threshold,
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
    if train_steps is None and not _training_stage_finished_legally(
        final_training_stop_reason
    ):
        raise BenchmarkRunnerError("uncapped training curriculum ended before convergence")
    training_run = _training_run_record(
        seed=seed,
        max_steps=train_steps,
        learning_rate=learning_rate,
        optimizer_name=optimizer_name,
        schedule_name=schedule_name,
        gate_check_interval=gate_check_interval,
        gate_decision_rule=gate_decision_rule,
        rung_competence_threshold=rung_competence_threshold,
        convergence_patience=convergence_patience,
        convergence_min_delta=convergence_min_delta,
        tensor_device=runtime.device_kind,
        runtime_memory_budget_fraction=(
            _default_runtime_memory_budget_fraction
            if tensor_runtime_has_fixed_device_memory(runtime)
            else None
        ),
        validation_history=tuple(validation_history),
        stop_reason=final_training_stop_reason,
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
            roofline=runtime_roofline_record(runtime),
            work_estimates=_training_work_estimates(
                input_shape=_batch_sample_input_shape(batch=current_frontier().batch),
                output_shape=_expected_program_output_shape(
                    batch=current_frontier().batch,
                    target_contract=target_contract,
                ),
                inference_cost=(
                    training_result.validation_inference_cost
                    or _module_projected_inference_cost_measurement(
                        runtime=runtime,
                        module=module,
                        input_shape=_batch_sample_input_shape(batch=current_frontier().batch),
                        batch_size=batch_size,
                        target_contract=target_contract,
                    )
                ),
                training_cost=_training_history_latest_training_cost_measurement(
                    validation_history
                ),
                storage_bytes=storage_bytes,
                batch_size=batch_size,
            ),
            phase_timings=phase_timings,
            fallback_errors=fallback_errors,
            operation_fallbacks=module.operation_fallback_records(),
            tensor_compile_fallbacks=tensor_element_compile_fallback_records(),
        ),
    )


def evaluate_model_checkpoint_artifact(
    *,
    program_path: Path,
    generator: _TensorBenchmarkGenerator,
    target_contract: TargetContract,
    competence_factory: _BenchmarkTrainingCompetenceFactory | None = None,
    accessible_subspace: AccessibleSubspace,
    sampling_protocol: SamplingProtocol,
    seed: int,
    tensor_device: TensorRuntimeDevice,
    checkpoint: ModelCheckpointArtifact,
) -> tuple[
    tuple[_CheckpointEvaluationRungEvidence, ...],
    GeneratedSampleSet,
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[Mapping[str, object], ...],
    Mapping[str, object],
]:
    """Generate benchmark evaluation evidence from a saved checkpoint artifact."""

    evaluation_counter = _ThroughputCounter()
    phase_timings = TimingCollector()
    with phase_timings.span("checkpoint_evaluation.predictor_load"):
        predictor = load_model_checkpoint_predictor(
            program_path=program_path,
            target_contract=target_contract,
            checkpoint=checkpoint,
            tensor_device=tensor_device,
        )
    runtime_phase_timings = _runtime_phase_timings(predictor.runtime)
    runtime_phase_timings.counters.update(phase_timings.counters)
    phase_timings = runtime_phase_timings
    results: list[_CheckpointEvaluationRungEvidence] = []
    outcome_ids = _target_contract_outcome_ids(target_contract)
    competence = _resolve_competence_functional(
        target_contract,
        runtime=predictor.runtime,
        competence_factory=competence_factory,
    )
    capacity_limited = False
    curriculum_exhausted = False
    planner = _VolumeCurriculumPlanner()
    while True:
        with phase_timings.span("checkpoint_evaluation.rung_preparation"):
            try:
                rung = _evaluation_curriculum_rung(
                    generator=generator,
                    sample_count=1,
                    seed=seed,
                    index=len(results),
                    planner=planner,
                    runtime=predictor.runtime,
                    outcome_ids=outcome_ids,
                )
            except _CurriculumExhausted:
                if not results:
                    raise
                curriculum_exhausted = True
                break
            except _EmptyCurriculumWindow:
                if not results:
                    raise BenchmarkRunnerError(
                        "checkpoint evaluation first curriculum rung materialized no samples"
                    ) from None
                curriculum_exhausted = True
                break
        try:
            with phase_timings.span("checkpoint_evaluation.rung_evaluation"):
                rung_evidence = _evaluate_checkpoint_rung(
                    predictor=predictor,
                    generator=generator,
                    rung=rung,
                    outcome_ids=outcome_ids,
                    target_contract=target_contract,
                    competence=competence,
                    sampling_protocol=sampling_protocol,
                    evaluation_counter=evaluation_counter,
                    phase_timings=phase_timings,
                )
        except _RuntimeCapacityReached:
            if not results:
                raise BenchmarkRunnerError(
                    "checkpoint evaluation could not fit the first evaluation rung"
                ) from None
            capacity_limited = True
            break
        results.append(rung_evidence)
        _validate_claim_chain(
            tuple(result.rung for result in results),
            accessible_subspace=accessible_subspace,
        )
        with phase_timings.span("checkpoint_evaluation.integration_check"):
            integration_evidence = _evaluation_integration_evidence(
                evaluation_results=results,
                outcome_ids=outcome_ids,
                target_contract=target_contract,
            )
        if integration_evidence.converged:
            break
    if not results:
        raise BenchmarkRunnerError("checkpoint evaluation did not produce any results")
    with phase_timings.span("checkpoint_evaluation.frontier_selection"):
        evaluation_frontier_index = _evaluation_result_frontier_index(
            evaluation_results=results,
            outcome_ids=outcome_ids,
            target_contract=target_contract,
        )
    with phase_timings.span("checkpoint_evaluation.final_measurements"):
        (
            final_batch,
            final_probabilities,
            final_accepted_mass,
            final_competence_diagnostics,
            final_inference_cost_measurement,
            final_inference_cost_sample_count,
        ) = (
            _evaluate_checkpoint_rung_measurements(
                predictor=predictor,
                generator=generator,
                rung=results[evaluation_frontier_index].rung,
                outcome_ids=outcome_ids,
                target_contract=target_contract,
                competence=competence,
                sampling_protocol=sampling_protocol,
                requested_sample_count=results[evaluation_frontier_index].sample_count,
                evaluation_counter=evaluation_counter,
                phase_timings=phase_timings,
            )
        )
    throughput = evaluation_counter.to_record(kind="checkpoint-evaluation-throughput")
    throughput["tensor_runtime"] = "pytorch"
    throughput["tensor_device"] = predictor.runtime.device_kind
    throughput["phase_timing"] = phase_timings.to_record()
    throughput["capacity_limited"] = capacity_limited
    throughput["curriculum_exhausted"] = curriculum_exhausted
    compile_fallbacks = tensor_element_compile_fallback_records()
    if compile_fallbacks:
        throughput["tensor_compile_fallbacks"] = [
            dict(fallback) for fallback in compile_fallbacks
        ]
    throughput["inference_cost_measurement"] = (
        final_inference_cost_measurement.without_operation_trace().to_record()
    )
    throughput["inference_cost_sample_count"] = final_inference_cost_sample_count
    return (
        tuple(results),
        final_batch,
        final_probabilities,
        final_accepted_mass,
        final_competence_diagnostics,
        throughput,
    )


def _checkpoint_inference_cost_measurement(
    *,
    predictor: CheckpointModelPredictor,
    batch: GeneratedSampleSet,
    outcome_ids: tuple[str, ...],
    target_contract: TargetContract,
) -> CostMeasurement:
    fields, labels = _batch_tensors(
        runtime=predictor.runtime,
        batch=batch,
        outcome_ids=outcome_ids,
        device=predictor.runtime.device,
    )
    horizons = _field_valued_target_horizons(
        batch=batch,
        labels=labels,
        target_contract=target_contract,
    )
    predictor.module.eval()

    def program(input_fields: Any) -> object:
        with no_grad_context(predictor.runtime):
            return _model_inference_for_cost(
                runtime=predictor.runtime,
                module=predictor.module,
                fields=input_fields,
                horizons=horizons,
                target_contract=target_contract,
            )

    return measure_program_cost(
        predictor.runtime,
        program,
        fields,
        strict=True,
        roofline=runtime_roofline_record(predictor.runtime),
    ).without_operation_trace()


def _model_predictions(
    *,
    runtime: TensorRuntime,
    module: Any,
    fields: Any,
    labels: Any,
    horizons: tuple[float, ...] | None = None,
    target_contract: TargetContract,
) -> Any:
    if target_contract.kind == "finite-outcome":
        return module(fields)
    return _field_valued_model_trajectory(
        runtime=runtime,
        module=module,
        fields=fields,
        labels=labels,
        horizons=horizons,
    )


def _model_inference_for_cost(
    *,
    runtime: TensorRuntime,
    module: Any,
    fields: Any,
    horizons: tuple[float, ...] | None = None,
    target_contract: TargetContract,
) -> Any:
    if target_contract.kind == "finite-outcome":
        return module(fields)
    return _field_valued_model_state_at_horizon(
        runtime=runtime,
        module=module,
        fields=fields,
        horizon=_field_valued_cost_horizon(horizons),
    )


def _field_valued_target_horizons(
    *,
    batch: GeneratedSampleSet,
    labels: Any,
    target_contract: TargetContract,
) -> tuple[float, ...] | None:
    if target_contract.kind == "finite-outcome":
        return None
    if len(tuple(labels.shape)) != 3:
        raise BenchmarkRunnerError(
            "field-valued training targets must have shape (batch, time, spatial)"
        )
    time_count = int(labels.shape[1])
    if time_count < 1:
        raise BenchmarkRunnerError("field-valued trajectory target must contain at least one time")
    if batch.region is None:
        if time_count == 1:
            return ()
        raise BenchmarkRunnerError("field-valued trajectory horizons require a sample region")
    horizon = batch.region.ambient.field_domain.get("length_y")
    if horizon is None:
        raise BenchmarkRunnerError(
            "field-valued trajectory horizons require ambient field_domain.length_y"
        )
    if isinstance(horizon, bool) or not isinstance(horizon, int | float):
        raise BenchmarkRunnerError("field-valued ambient field_domain.length_y must be numeric")
    result_horizon = float(horizon)
    if not math.isfinite(result_horizon) or result_horizon <= 0.0:
        raise BenchmarkRunnerError(
            "field-valued ambient field_domain.length_y must be positive and finite"
        )
    if time_count == 1:
        return _field_valued_ambient_horizons(batch=batch, horizon=result_horizon)
    step = result_horizon / float(time_count - 1)
    return tuple(step * index for index in range(1, time_count))


def _field_valued_ambient_horizons(
    *,
    batch: GeneratedSampleSet,
    horizon: float,
) -> tuple[float, ...]:
    if batch.region is None:
        return ()
    time_resolution = batch.region.ambient.field_domain.get("time_resolution")
    if time_resolution is None:
        return ()
    if isinstance(time_resolution, bool) or not isinstance(time_resolution, int | float):
        raise BenchmarkRunnerError(
            "field-valued ambient field_domain.time_resolution must be numeric"
        )
    step = float(time_resolution)
    if not math.isfinite(step) or step <= 0.0:
        raise BenchmarkRunnerError(
            "field-valued ambient field_domain.time_resolution must be positive and finite"
        )
    step_count = round(horizon / step)
    if step_count < 1:
        return ()
    return tuple((horizon * index) / float(step_count) for index in range(1, step_count + 1))


def _validate_training_field_stepper(
    *,
    runtime: TensorRuntime,
    module: Any,
    batch: GeneratedSampleSet,
    target_contract: TargetContract,
    outcome_ids: tuple[str, ...],
) -> None:
    fields, labels = _batch_tensors(
        runtime=runtime,
        batch=batch,
        outcome_ids=outcome_ids,
        device=runtime.device,
    )
    horizons = _field_valued_target_horizons(
        batch=batch,
        labels=labels,
        target_contract=target_contract,
    )
    dt = _field_valued_cost_horizon(horizons)
    if horizons:
        dt = horizons[0]
    try:
        validate_field_stepper_nondegenerate(
            runtime=runtime,
            module=module,
            fields=fields,
            dt=dt,
        )
    except FieldEvolutionError as error:
        raise BenchmarkRunnerError(str(error)) from error


def _field_valued_cost_horizon(horizons: tuple[float, ...] | None) -> float:
    if horizons:
        return horizons[-1]
    return 1.0


def _field_valued_model_trajectory(
    *,
    runtime: TensorRuntime,
    module: Any,
    fields: Any,
    labels: Any,
    horizons: tuple[float, ...] | None = None,
) -> Any:
    if len(tuple(fields.shape)) != 3:
        raise BenchmarkRunnerError(
            "field-valued operator input must have shape (batch, channel, spatial)"
        )
    if len(tuple(labels.shape)) != 3:
        raise BenchmarkRunnerError(
            "field-valued training targets must have shape (batch, time, spatial)"
        )
    if int(fields.shape[1]) != 1:
        raise BenchmarkRunnerError("field-valued operator input must contain one state channel")
    if int(labels.shape[0]) != int(fields.shape[0]):
        raise BenchmarkRunnerError("field-valued target batch size must match input batch size")
    if int(labels.shape[-1]) != int(fields.shape[-1]):
        raise BenchmarkRunnerError("field-valued target spatial length must match input length")
    label_time_count = int(labels.shape[1])
    if label_time_count < 1:
        raise BenchmarkRunnerError("field-valued trajectory target must contain at least one time")
    time_count = 1 + len(horizons) if horizons else label_time_count
    if time_count == 1:
        return fields
    if horizons is None:
        raise BenchmarkRunnerError("field-valued trajectory requires target horizons")
    if label_time_count > 1 and len(horizons) != label_time_count - 1:
        raise BenchmarkRunnerError(
            "field-valued target horizon count must match target time steps after "
            "the initial state"
        )
    state = fields
    states = [fields]
    previous_horizon = 0.0
    for horizon in horizons:
        dt = horizon - previous_horizon
        if dt <= 0.0 or not math.isfinite(dt):
            raise BenchmarkRunnerError("field-valued target horizons must increase")
        state = _field_valued_model_state_at_horizon(
            runtime=runtime,
            module=module,
            fields=state,
            horizon=dt,
        )
        states.append(state)
        previous_horizon = horizon
    return tensor_runtime_concat(runtime, states, dim=1)


def _field_valued_model_state_at_horizon(
    *,
    runtime: TensorRuntime,
    module: Any,
    fields: Any,
    horizon: float,
) -> Any:
    try:
        return field_stepper_state(
            runtime=runtime,
            module=module,
            fields=fields,
            dt=horizon,
        )
    except (FieldEvolutionError, TensorRuntimeError) as error:
        raise BenchmarkRunnerError(str(error)) from error


def _module_inference_cost_measurement(
    *,
    runtime: TensorRuntime,
    module: Any,
    fields: Any,
    labels: Any,
    horizons: tuple[float, ...] | None = None,
    target_contract: TargetContract,
) -> CostMeasurement:
    was_training = bool(module.training)
    module.eval()

    def program(input_fields: Any) -> object:
        with no_grad_context(runtime):
            return _model_inference_for_cost(
                runtime=runtime,
                module=module,
                fields=input_fields,
                horizons=horizons,
                target_contract=target_contract,
            )

    try:
        return measure_program_cost(
            runtime,
            program,
            fields,
            strict=True,
            roofline=runtime_roofline_record(runtime),
        ).without_operation_trace()
    finally:
        if was_training:
            module.train()


def _module_projected_inference_cost_measurement(
    *,
    runtime: TensorRuntime,
    module: Any,
    input_shape: tuple[int, ...],
    batch_size: int,
    target_contract: TargetContract,
) -> tuple[CostMeasurement, int] | None:
    sample_count = max(1, batch_size)
    was_training = bool(module.training)
    module.eval()

    def program() -> object:
        fields = make_empty_float_tensor(
            runtime,
            (sample_count, *input_shape),
            device=runtime.device,
        )
        with no_grad_context(runtime):
            return _model_inference_for_cost(
                runtime=runtime,
                module=module,
                fields=fields,
                horizons=None,
                target_contract=target_contract,
            )

    try:
        return (
            estimate_program_cost(
                runtime,
                program,
                strict=True,
                roofline=runtime_roofline_record(runtime),
            ).without_operation_trace(),
            sample_count,
        )
    except (CostMetrologyError, TensorRuntimeError):
        return None
    finally:
        if was_training:
            module.train()


def _evaluate_checkpoint_rung(
    *,
    predictor: CheckpointModelPredictor,
    generator: _TensorBenchmarkGenerator,
    rung: _CurriculumRung,
    outcome_ids: tuple[str, ...],
    target_contract: TargetContract,
    competence: _CompetenceFunctional,
    sampling_protocol: SamplingProtocol,
    evaluation_counter: _ThroughputCounter,
    phase_timings: TimingCollector,
) -> _CheckpointEvaluationRungEvidence:
    estimator = _RunningMeanEstimator()
    max_cost_measurement: tuple[CostMeasurement, int] | None = None
    chance_mass = _target_contract_chance_mass(target_contract)
    half_width_threshold = (
        _default_evaluation_convergence_half_width
        if chance_mass >= 1.0
        else _default_evaluation_convergence_half_width * (1.0 - chance_mass)
    )
    census_indices = (
        _census_sample_indices(rung.batch.region)
        if _sampling_protocol_saturates_to_census(
            protocol=sampling_protocol,
            region=rung.batch.region,
        )
        else None
    )
    effective_sampling_protocol = (
        SamplingProtocol(kind="census", census_budget=len(census_indices))
        if census_indices is not None
        else sampling_protocol
    )
    while not _evaluation_estimate_converged(
        estimator,
        sampling_protocol=effective_sampling_protocol,
        half_width_threshold=half_width_threshold,
    ):
        next_sample_count = (
            len(census_indices)
            if census_indices is not None
            else _evaluation_next_sample_count(
                estimator,
                half_width_threshold=half_width_threshold,
            )
        )
        for chunk in _checkpoint_evaluation_chunks(
            predictor=predictor,
            generator=generator,
            rung=rung,
            outcome_ids=outcome_ids,
            target_contract=target_contract,
            competence=competence,
            requested_sample_count=next_sample_count,
            sample_indices=census_indices,
            evaluation_counter=evaluation_counter,
            phase_timings=phase_timings,
            purpose="score",
        ):
            estimator.extend(chunk.accepted_mass)
            max_cost_measurement = _max_cost_measurement(
                max_cost_measurement,
                _chunk_cost_measurement_pair(chunk),
            )
    observed_sample_count = estimator.samples
    if observed_sample_count < 1:
        raise BenchmarkRunnerError("checkpoint evaluation rung produced no samples")
    if max_cost_measurement is None:
        raise BenchmarkRunnerError("checkpoint evaluation could not measure inference cost")
    return _CheckpointEvaluationRungEvidence(
        rung=replace(rung, sample_count=observed_sample_count),
        mean_accepted_mass=estimator.mean,
        sample_count=observed_sample_count,
        confidence_half_width=_evaluation_confidence_half_width(
            estimator,
            sampling_protocol=effective_sampling_protocol,
        ),
        confidence_method_id=(
            None
            if census_indices is not None
            else _evaluation_confidence_method_id(sampling_protocol)
        ),
        sampling_protocol=effective_sampling_protocol,
        input_shape=_batch_sample_input_shape(batch=rung.batch),
        inference_cost_measurement=max_cost_measurement[0],
        inference_cost_sample_count=max_cost_measurement[1],
    )


def _evaluation_integration_evidence(
    *,
    evaluation_results: Sequence[_CheckpointEvaluationRungEvidence],
    outcome_ids: tuple[str, ...],
    target_contract: TargetContract,
) -> _EvaluationIntegrationEvidence:
    """Return the explicit score-integral state that controls evaluation."""

    if not evaluation_results:
        return _EvaluationIntegrationEvidence(
            frontier_index=0,
            score_integral_value=0.0,
            score_integral_half_width=math.inf,
            terminal_failure_count=0,
        )
    chance_mass = _target_contract_chance_mass(target_contract)
    frontier_index = _evaluation_result_frontier_index(
        evaluation_results=evaluation_results,
        outcome_ids=outcome_ids,
        target_contract=target_contract,
    )
    score_integral = sampled_competence_frontier_integral(
        _evaluation_competence_points(evaluation_results),
        chance_mass=chance_mass,
    )
    return _EvaluationIntegrationEvidence(
        frontier_index=frontier_index,
        score_integral_value=score_integral.value,
        score_integral_half_width=_evaluation_score_integral_half_width(
            evaluation_results=evaluation_results,
            frontier_index=frontier_index,
            chance_mass=chance_mass,
        ),
        terminal_failure_count=_evaluation_terminal_failure_count(
            evaluation_results=evaluation_results,
            frontier_index=frontier_index,
            chance_mass=chance_mass,
        ),
    )


def _evaluation_competence_points(
    evaluation_results: Sequence[_CheckpointEvaluationRungEvidence],
) -> tuple[CompetencePoint, ...]:
    return tuple(
        CompetencePoint(
            log2_volume=result.rung.log2_volume,
            accepted_mass=result.mean_accepted_mass,
            sample_count=result.sample_count,
            seed=result.rung.seed,
            log2_volume_minimum=result.rung.log2_volume_minimum,
            log2_volume_maximum=result.rung.log2_volume_maximum,
            input_shape=result.input_shape,
            region=getattr(getattr(result.rung, "batch", None), "region", None),
            confidence_half_width=result.confidence_half_width,
            confidence_method_id=result.confidence_method_id,
        )
        for result in evaluation_results
    )


def _evaluation_score_integral_half_width(
    *,
    evaluation_results: Sequence[_CheckpointEvaluationRungEvidence],
    frontier_index: int,
    chance_mass: float,
) -> float:
    if not evaluation_results:
        return math.inf
    scale = 1.0 if chance_mass >= 1.0 else 1.0 / (1.0 - chance_mass)
    widths: list[float] = []
    frontier_results = evaluation_results[: frontier_index + 1]
    for result in frontier_results:
        lower, upper = _rung_log2_volume_interval(result.rung) or (
            max(0.0, result.rung.log2_volume - 1.0),
            result.rung.log2_volume,
        )
        widths.append(max(0.0, upper - lower))
    variance_terms = (
        (width * result.confidence_half_width * scale) ** 2
        for width, result in zip(widths, frontier_results, strict=True)
    )
    return math.sqrt(math.fsum(variance_terms))


def _evaluation_terminal_failure_count(
    *,
    evaluation_results: Sequence[_CheckpointEvaluationRungEvidence],
    frontier_index: int,
    chance_mass: float,
) -> int:
    if not evaluation_results:
        return 0
    failure_start = (
        frontier_index + 1
        if _evaluation_rung_confidently_above_chance(
            evaluation_results[frontier_index],
            chance_mass=chance_mass,
        )
        else frontier_index
    )
    return len(evaluation_results[failure_start:])


@dataclass(frozen=True, slots=True)
class _CheckpointEvaluationChunk:
    batch: GeneratedSampleSet
    probabilities: tuple[tuple[float, ...], ...]
    accepted_mass: tuple[float, ...]
    accepted_mass_sum: float
    sample_count: int
    competence_diagnostics: tuple[Mapping[str, object], ...] = ()
    inference_cost_measurement: CostMeasurement | None = None
    inference_cost_sample_count: int | None = None


def _evaluate_checkpoint_rung_measurements(
    *,
    predictor: CheckpointModelPredictor,
    generator: _TensorBenchmarkGenerator,
    rung: _CurriculumRung,
    outcome_ids: tuple[str, ...],
    target_contract: TargetContract,
    competence: _CompetenceFunctional,
    sampling_protocol: SamplingProtocol,
    requested_sample_count: int,
    evaluation_counter: _ThroughputCounter,
    phase_timings: TimingCollector,
) -> tuple[
    GeneratedSampleSet,
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[Mapping[str, object], ...],
    CostMeasurement,
    int,
]:
    samples: list[GeneratedSample] = []
    probabilities: list[tuple[float, ...]] = []
    accepted_mass: list[float] = []
    competence_diagnostics: list[Mapping[str, object]] = []
    field_batches: list[Any] = []
    target_batches: list[Any] = []
    max_cost_measurement: tuple[CostMeasurement, int] | None = None
    census_indices = (
        _census_sample_indices(rung.batch.region)
        if _sampling_protocol_saturates_to_census(
            protocol=sampling_protocol,
            region=rung.batch.region,
        )
        else None
    )
    for chunk in _checkpoint_evaluation_chunks(
        predictor=predictor,
        generator=generator,
        rung=rung,
        outcome_ids=outcome_ids,
        target_contract=target_contract,
        competence=competence,
        requested_sample_count=requested_sample_count,
        sample_indices=census_indices,
        evaluation_counter=evaluation_counter,
        phase_timings=phase_timings,
        purpose="measurements",
    ):
        sample_offset = len(samples)
        samples.extend(
            replace(sample, index=sample_offset + index)
            for index, sample in enumerate(chunk.batch.samples)
        )
        probabilities.extend(chunk.probabilities)
        accepted_mass.extend(chunk.accepted_mass)
        competence_diagnostics.extend(chunk.competence_diagnostics)
        if target_contract.kind == "field-valued":
            fields, targets = _batch_tensors(
                runtime=predictor.runtime,
                batch=chunk.batch,
                outcome_ids=outcome_ids,
                device=predictor.runtime.device,
            )
            field_batches.append(fields)
            target_batches.append(targets)
        max_cost_measurement = _max_cost_measurement(
            max_cost_measurement,
            _chunk_cost_measurement_pair(chunk),
        )
    if not samples:
        raise BenchmarkRunnerError("checkpoint evaluation final rung produced no samples")
    if max_cost_measurement is None:
        raise BenchmarkRunnerError("checkpoint evaluation could not measure inference cost")
    return (
        GeneratedSampleSet(
            benchmark_id=rung.batch.benchmark_id,
            generator_id=rung.batch.generator_id,
            generator_version=rung.batch.generator_version,
            seed=rung.seed,
            shape=(len(samples),),
            variation_extent=_full_variation_extent,
            volume_request=rung.batch.volume_request,
            region=rung.batch.region,
            request_outcome=rung.batch.request_outcome,
            samples=tuple(samples),
            fields=(
                None
                if not field_batches
                else tensor_runtime_concat(predictor.runtime, field_batches, dim=0)
            ),
            targets=(
                None
                if not target_batches
                else tensor_runtime_concat(predictor.runtime, target_batches, dim=0)
            ),
        ),
        tuple(probabilities),
        tuple(accepted_mass),
        tuple(competence_diagnostics),
        max_cost_measurement[0],
        max_cost_measurement[1],
    )


def _checkpoint_evaluation_chunks(
    *,
    predictor: CheckpointModelPredictor,
    generator: _TensorBenchmarkGenerator,
    rung: _CurriculumRung,
    outcome_ids: tuple[str, ...],
    target_contract: TargetContract,
    competence: _CompetenceFunctional,
    requested_sample_count: int,
    sample_indices: Sequence[int] | None,
    evaluation_counter: _ThroughputCounter,
    phase_timings: TimingCollector,
    purpose: str,
) -> Iterable[_CheckpointEvaluationChunk]:
    remaining = requested_sample_count
    chunk_index = 0
    sample_offset = 0
    while remaining > 0:
        physical_sample_count = _physical_execution_sample_count(
            runtime=predictor.runtime,
            generator=generator,
            rung=rung,
            requested_sample_count=remaining,
            outcome_count=len(outcome_ids),
            phase_timings=phase_timings,
            phase=f"checkpoint_evaluation_{purpose}",
        )
        chunk_seed = rung.seed + 1_000_003 * chunk_index
        chunk_sample_indices = (
            None
            if sample_indices is None
            else sample_indices[sample_offset : sample_offset + physical_sample_count]
        )
        generation_started = time.perf_counter()
        with phase_timings.span(
            f"checkpoint_evaluation_{purpose}_generation",
            samples=physical_sample_count,
        ):
            batch = generator(
                shape=physical_sample_count,
                seed=chunk_seed,
                include_fields=False,
                include_metadata=purpose == "measurements",
                volume_request=_rung_volume_request(rung),
                sample_indices=chunk_sample_indices,
                memory_limit_bytes=_runtime_memory_budget_bytes(predictor.runtime),
                variation_extent=_full_variation_extent,
                runtime=predictor.runtime,
                outcome_ids=outcome_ids,
                timing=phase_timings,
                timing_prefix=f"checkpoint_evaluation_{purpose}_generation.",
            )
        if batch.sample_count == 0:
            raise BenchmarkRunnerError(
                "generator returned no samples for the selected volume window"
            )
        prediction_started = time.perf_counter()
        with phase_timings.span(
            f"checkpoint_evaluation_{purpose}_prediction",
            samples=batch.sample_count,
        ):
            predictions = predictor.predict_batch(batch)
        evaluation_counter.add(
            seconds=time.perf_counter() - generation_started,
            samples=batch.sample_count,
        )
        phase_timings.add(
            f"checkpoint_evaluation_{purpose}_prediction_latency",
            seconds=time.perf_counter() - prediction_started,
            samples=batch.sample_count,
        )
        with phase_timings.span(
            f"checkpoint_evaluation_{purpose}_accepted_mass",
            samples=batch.sample_count,
        ):
            if target_contract.kind == "field-valued":
                fields, labels = _batch_tensors(
                    runtime=predictor.runtime,
                    batch=batch,
                    outcome_ids=outcome_ids,
                    device=predictor.runtime.device,
                )
                horizons = _field_valued_target_horizons(
                    batch=batch,
                    labels=labels,
                    target_contract=target_contract,
                )
                with no_grad_context(predictor.runtime):
                    competence_evaluation = competence.training_logit_masses_with_diagnostics(
                        predictor.runtime,
                        predictions,
                        labels,
                        module=predictor.module,
                        fields=fields,
                        horizons=horizons,
                        batch=batch,
                        generator=generator,
                    )
                    accepted_mass = competence_evaluation.values
                    competence_diagnostics = competence_evaluation.diagnostics
                probabilities: tuple[tuple[float, ...], ...] = ()
            else:
                probabilities = predictions
                competence_diagnostics = ()
                accepted_mass = competence.prediction_accepted_mass(
                    batch=batch,
                    probabilities=probabilities,
                    outcome_ids=outcome_ids,
                )
        inference_cost_measurement: CostMeasurement | None = None
        if purpose in {"score", "measurements"}:
            with phase_timings.span(
                f"checkpoint_evaluation_{purpose}_inference_cost_metrology",
                samples=batch.sample_count,
            ):
                inference_cost_measurement = _checkpoint_inference_cost_measurement(
                    predictor=predictor,
                    batch=batch,
                    outcome_ids=outcome_ids,
                    target_contract=target_contract,
                )
        if inference_cost_measurement is None:
            raise BenchmarkRunnerError("checkpoint evaluation could not measure inference cost")
        yield _CheckpointEvaluationChunk(
            batch=batch,
            probabilities=probabilities,
            accepted_mass=accepted_mass,
            accepted_mass_sum=math.fsum(accepted_mass),
            sample_count=batch.sample_count,
            competence_diagnostics=competence_diagnostics,
            inference_cost_measurement=inference_cost_measurement,
            inference_cost_sample_count=batch.sample_count,
        )
        remaining -= batch.sample_count
        sample_offset += batch.sample_count
        chunk_index += 1


def _max_cost_measurement(
    left: tuple[CostMeasurement, int] | None,
    right: tuple[CostMeasurement, int] | None,
) -> tuple[CostMeasurement, int] | None:
    if left is None:
        return right
    if right is None:
        return left
    left_cost, left_sample_count = left
    right_cost, right_sample_count = right
    if (
        right_cost.abstract_flops_per_item(right_sample_count)
        > left_cost.abstract_flops_per_item(left_sample_count)
    ):
        return right
    return left


def _chunk_cost_measurement_pair(
    chunk: _CheckpointEvaluationChunk,
) -> tuple[CostMeasurement, int] | None:
    if chunk.inference_cost_measurement is None or chunk.inference_cost_sample_count is None:
        return None
    return (chunk.inference_cost_measurement, chunk.inference_cost_sample_count)


def _measurement_ops_per_item(measurement: tuple[CostMeasurement, int]) -> float:
    cost, sample_count = measurement
    return cost.abstract_flops_per_item(sample_count)


def _cost_measurement_compute_source(measurement: CostMeasurement) -> str:
    if measurement.operations_executed:
        return "measured-forward-metrology"
    return "dry-run-metrology"


def _batch_sample_input_shape(
    *,
    batch: GeneratedSampleSet,
) -> tuple[int, ...]:
    tensor_input_shape = _tensor_input_shape(batch.fields)
    if tensor_input_shape is not None:
        return tensor_input_shape
    if batch.samples:
        try:
            return batch.samples[0].require_field().shape
        except ObservationGenerationError:
            pass
    raise BenchmarkRunnerError(
        "tensor benchmark generator must return tensors or inspectable field metadata"
    )


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


def _rung_tensor_input_shape(
    *,
    generator: _TensorBenchmarkGenerator,
    rung: _CurriculumRung,
) -> tuple[int, ...]:
    tensor_input_shape = _tensor_input_shape(rung.batch.fields)
    if tensor_input_shape is not None:
        return tensor_input_shape
    formation = getattr(generator, "formation", None)
    channel_count = getattr(formation, "channel_count", None)
    height_axis = getattr(formation, "height_axis", None)
    width_axis = getattr(formation, "width_axis", None)
    if rung.resolution_assignment is None:
        return _batch_sample_input_shape(batch=rung.batch)
    if (
        type(channel_count) is not int
        or channel_count < 1
        or type(height_axis) is not str
        or type(width_axis) is not str
    ):
        raise BenchmarkRunnerError(
            "dynamic batch sizing requires benchmark formation channel and canvas axes"
        )
    height = rung.resolution_assignment.require_axis(height_axis)
    width = rung.resolution_assignment.require_axis(width_axis)
    if height < 1 or width < 1:
        raise BenchmarkRunnerError("dynamic batch sizing requires positive canvas axes")
    return (channel_count, height, width)


def _runtime_memory_budget_bytes(runtime: TensorRuntime) -> int | None:
    total_bytes = tensor_runtime_total_memory_bytes(runtime)
    if total_bytes is None:
        return None
    return max(1, int(total_bytes * _default_runtime_memory_budget_fraction))


def _runtime_used_memory_bytes(runtime: TensorRuntime) -> int:
    return tensor_runtime_used_memory_bytes(runtime)


def _estimated_runtime_batch_sample_bytes(
    *,
    input_shape: tuple[int, ...],
    outcome_count: int,
) -> int:
    element_count = 1
    for axis in input_shape:
        element_count *= axis
    element_count += outcome_count
    return max(
        1,
        element_count * _float32_bytes * _runtime_batch_memory_safety_factor,
    )


def _physical_execution_sample_count(
    *,
    runtime: TensorRuntime,
    generator: _TensorBenchmarkGenerator,
    rung: _CurriculumRung,
    requested_sample_count: int,
    outcome_count: int,
    phase_timings: TimingCollector,
    phase: str,
) -> int:
    if requested_sample_count < 1:
        raise BenchmarkRunnerError("requested_sample_count must be positive")
    budget_bytes = _runtime_memory_budget_bytes(runtime)
    if budget_bytes is None:
        return requested_sample_count
    input_shape = _rung_tensor_input_shape(
        generator=generator,
        rung=rung,
    )
    used_bytes = _runtime_used_memory_bytes(runtime)
    available_bytes = budget_bytes
    sample_bytes = _estimated_runtime_batch_sample_bytes(
        input_shape=input_shape,
        outcome_count=outcome_count,
    )
    physical_sample_count = min(requested_sample_count, available_bytes // sample_bytes)
    phase_timings.add_counters(
        f"{phase}.dynamic_batch",
        {
            "requested_sample_count": float(requested_sample_count),
            "physical_sample_count": float(physical_sample_count),
            "runtime_memory_budget_bytes": float(budget_bytes),
            "runtime_used_memory_bytes": float(used_bytes),
            "estimated_sample_bytes": float(sample_bytes),
        },
    )
    if physical_sample_count < 1:
        raise _RuntimeCapacityReached()
    return int(physical_sample_count)


def _build_training_module(
    *,
    runtime: TensorRuntime,
    program_path: Path,
) -> Any:
    try:
        return load_program_graph(program_path, runtime).graph.build_module(runtime)
    except ProgramGraphError as error:
        raise BenchmarkRunnerError(str(error)) from error


def _is_runtime_capacity_error(error: RuntimeError) -> bool:
    return runtime_capacity_error(error)


def load_model_checkpoint_predictor(
    *,
    program_path: Path,
    target_contract: TargetContract,
    checkpoint: ModelCheckpointArtifact,
    tensor_device: TensorRuntimeDevice,
) -> CheckpointModelPredictor:
    """Load a saved checkpoint artifact as an executable benchmark predictor."""

    try:
        runtime = resolve_tensor_runtime(tensor_device)
    except TensorRuntimeError as error:
        raise BenchmarkRunnerError(str(error)) from error
    module = _build_training_module(
        runtime=runtime,
        program_path=program_path,
    )
    _load_torch_checkpoint(module=module, runtime=runtime, checkpoint=checkpoint)
    module.eval()
    return CheckpointModelPredictor(
        runtime=runtime,
        module=module,
        outcome_ids=_target_contract_outcome_ids(target_contract),
        target_contract=target_contract,
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
    program_graph: Mapping[str, object],
    program_path: Path,
    benchmark_id: ProtocolIdentifier,
    run_slug: str,
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
    record["program_graph"] = dict(program_graph)
    record["model_manifest"] = checkpoint.manifest.to_record()
    record["program_path"] = _portable_record_path(
        program_path,
        results_root=results_root or Path("results"),
    )
    if checkpoint.score_estimate is not None:
        record["score"] = _checkpoint_score_estimate_selection_score(
            checkpoint.score_estimate
        )
    return record


def _compact_model_checkpoint_summary_record(
    record: Mapping[str, object],
) -> dict[str, object]:
    summary_keys = (
        "kind",
        "path",
        "digest",
        "manifest_path",
        "manifest_digest",
        "record_path",
        "step",
        "validation_check",
        "validation_loss",
        "benchmark_id",
        "run_slug",
    )
    compact = {key: record[key] for key in summary_keys if key in record}
    score = record.get("score")
    if isinstance(score, int | float):
        compact["score"] = float(score)
    return compact


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


def _checkpoint_program_path(value: object, *, results_root: Path) -> Path:
    if value is None:
        raise BenchmarkRunnerError("checkpoint_artifact.program_path is required")
    if not isinstance(value, str) or not value:
        raise BenchmarkRunnerError("checkpoint_artifact.program_path must be a nonempty string")
    path = Path(value)
    if path.is_absolute():
        return path
    if path.parts[:1] == (results_root.name,):
        return (results_root.parent / path).resolve()
    return path


def _load_object_record(path: Path, *, description: str) -> Mapping[str, object]:
    return load_object_document(path.read_bytes(), description=description)


def _extract_record(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkRunnerError(f"{field} must be a record")
    return cast(Mapping[str, object], value)


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise BenchmarkRunnerError(f"{field} must be a nonempty string")
    return value


def _required_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise BenchmarkRunnerError(f"{field} must be an integer")
    return value


def _required_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkRunnerError(f"{field} must be an object")
    return cast(Mapping[str, object], value)


def _optional_positive_int(value: object) -> int | None:
    if type(value) is not int or value < 1:
        return None
    return value


def _required_float(value: object, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        raise BenchmarkRunnerError(f"{field} must be a finite number")
    return float(value)


def _optional_nonnegative_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    result = _required_float(value, field)
    if result < 0.0:
        raise BenchmarkRunnerError(f"{field} must be nonnegative")
    return result


def _unpredictable_evaluation_seed() -> int:
    return secrets.randbelow(2**63)


def training_stage_converged(stop_reason: str) -> bool:
    return stop_reason in _converged_training_stage_stop_reasons


def _runtime_phase_timings(runtime: TensorRuntime) -> TimingCollector:
    if os.environ.get(_sync_timing_environment_variable, "") in {"", "0"}:
        return TimingCollector()
    return _SynchronizedTimingCollector(runtime)


def _training_stage_finished_legally(stop_reason: str) -> bool:
    return stop_reason in _legal_uncapped_training_stage_stop_reasons


def _train_until_convergence(
    *,
    runtime: TensorRuntime,
    module: Any,
    optimizer: Any,
    scheduler: _LearningRateSchedule | None,
    loss_function: Any,
    train_batch: Callable[[int], _TrainingStepBatch | tuple[Any, Any]],
    validation_batch: Callable[[int], GeneratedSampleSet],
    target_contract: TargetContract,
    outcome_ids: tuple[str, ...],
    max_steps: int | None,
    gate_check_interval: int,
    patience: int,
    min_delta: float,
    rung_competence_threshold: float,
    training_counter: _ThroughputCounter,
    validation_counter: _ThroughputCounter,
    phase_timings: TimingCollector,
    generator: _TensorBenchmarkGenerator | None = None,
    training_batch_target: int = _default_training_batch_target,
    gate_batch_target: int = _default_gate_batch_target,
    start_step: int = 0,
    start_check: int = 0,
    competence: _CompetenceFunctional | None = None,
    on_plateau: Callable[[tuple[TrainingHistoryPoint, ...]], bool] | None = None,
    frontier_points: Callable[[], tuple[ValidationCompetencePoint, ...]] = tuple,
    on_gate_check: Callable[[tuple[TrainingHistoryPoint, ...], Any], None] | None = None,
) -> _TrainingStageResult:
    stage_started = time.perf_counter()
    validation_history: list[TrainingHistoryPoint] = []
    best_score = -float("inf")
    stale_checks = 0
    stop_reason = "training-stopped"
    plateau_window_start_index = 0
    max_validation_inference_cost: tuple[CostMeasurement, int] | None = None
    latest_training_cost: tuple[CostMeasurement, int] | None = None
    training_cost_by_shape: dict[object, tuple[CostMeasurement, int]] = {}
    replay_frontier_points: dict[
        tuple[float, float],
        _RollingValidationCompetencePoint,
    ] = {}
    pending_replay_scores: list[_PendingReplayScore] = []
    chance_mass = _target_contract_chance_mass(target_contract)
    if competence is None:
        competence = _resolve_competence_functional(target_contract)

    def append_validation(*, step: int, check: int) -> None:
        nonlocal best_score
        nonlocal max_validation_inference_cost, stale_checks
        with phase_timings.span("validation_replay_score_flush"):
            _flush_pending_replay_scores(
                pending_replay_scores=pending_replay_scores,
                replay_frontier_points=replay_frontier_points,
            )
        validation_started = time.perf_counter()
        batch = validation_batch(check)
        with phase_timings.span("validation_tensor_batch", samples=batch.sample_count):
            fields, labels = _batch_tensors(
                runtime=runtime,
                batch=batch,
                outcome_ids=outcome_ids,
                device=runtime.device,
            )
            horizons = _field_valued_target_horizons(
                batch=batch,
                labels=labels,
                target_contract=target_contract,
            )
        actual_gate_sample_count = _tensor_batch_size(fields, fallback=gate_batch_target)
        with phase_timings.span(
            "validation_inference_cost_metrology",
            samples=actual_gate_sample_count,
        ):
            validation_cost_measurement = _module_inference_cost_measurement(
                runtime=runtime,
                module=module,
                fields=fields,
                labels=labels,
                horizons=horizons,
                target_contract=target_contract,
            )
        max_validation_inference_cost = _max_cost_measurement(
            max_validation_inference_cost,
            (validation_cost_measurement, actual_gate_sample_count),
        )
        if max_validation_inference_cost is None:
            raise BenchmarkRunnerError("training gate could not measure inference cost")
        with phase_timings.span("validation_forward_loss", samples=actual_gate_sample_count):
            was_training = bool(module.training)
            module.eval()
            with no_grad_context(runtime):
                logits = _model_predictions(
                    runtime=runtime,
                    module=module,
                    fields=fields,
                    labels=labels,
                    horizons=horizons,
                    target_contract=target_contract,
                )
                validation_loss = float(loss_function(logits, labels).item())
                competence_evaluation = competence.training_logit_masses_with_diagnostics(
                    runtime,
                    logits,
                    labels,
                    module=module,
                    fields=fields,
                    horizons=horizons,
                    batch=batch,
                    generator=generator,
                )
                accepted_mass = competence_evaluation.values
            if was_training:
                module.train()
        with phase_timings.span("validation_score_estimate", samples=batch.sample_count):
            score_estimate = _training_gate_score_estimate(
                batch=batch,
                target_contract=target_contract,
                accepted_mass=accepted_mass,
                previous_frontier_points=_refreshed_frontier_points(
                    frontier_points(),
                    (rolling.point for rolling in replay_frontier_points.values()),
                ),
                validation_check=check,
                step=step,
                inference_cost=max_validation_inference_cost,
                training_cost=latest_training_cost,
                competence_diagnostics=competence_evaluation.diagnostics,
            )
        plateau_signal = _training_score_estimate_frontier_competence(
            score_estimate,
            chance_mass=chance_mass,
        )
        if plateau_signal > best_score + min_delta:
            best_score = plateau_signal
            stale_checks = 0
        else:
            stale_checks += 1
        with phase_timings.span("validation_scheduler_step"):
            if scheduler is not None:
                scheduler.step_after_validation(-plateau_signal)
                learning_rates = scheduler.learning_rates()
            else:
                learning_rates = _optimizer_learning_rates(optimizer)
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

    try:
        append_validation(step=start_step, check=start_check)
    except _RuntimeCapacityReached as error:
        raise BenchmarkRunnerError(
            "training memory budget cannot fit one validation sample for the first rung"
        ) from error
    except RuntimeError as error:
        if not _is_runtime_capacity_error(error):
            raise
        raise BenchmarkRunnerError(
            "training runtime could not fit one validation sample for the first rung"
        ) from error
    if max_steps == start_step:
        phase_timings.add(
            "training_stage_total",
            seconds=time.perf_counter() - stage_started,
        )
        return _TrainingStageResult(
            validation_history=tuple(validation_history),
            stop_reason="no-training-steps" if start_step == 0 else "max-steps",
            validation_inference_cost=max_validation_inference_cost,
        )
    validation_check = start_check + 1
    steps = count(start_step + 1) if max_steps is None else range(start_step + 1, max_steps + 1)
    for step in steps:
        training_started = time.perf_counter()
        try:
            raw_training_batch = train_batch(step)
        except _RuntimeCapacityReached:
            stop_reason = "capacity-limited"
            break
        except RuntimeError as error:
            if not _is_runtime_capacity_error(error):
                raise
            stop_reason = "capacity-limited"
            break
        training_batch = _coerce_training_step_batch(raw_training_batch)
        fields = training_batch.fields
        labels = training_batch.labels
        horizons = training_batch.horizons
        actual_batch_size = _tensor_batch_size(fields, fallback=training_batch_target)
        module.train()
        try:
            first_logits: Any | None = None

            def loss_closure(
                fields: Any = fields,
                labels: Any = labels,
                horizons: tuple[float, ...] | None = horizons,
                actual_batch_size: int = actual_batch_size,
            ) -> Any:
                nonlocal first_logits
                with phase_timings.span("training_zero_grad"):
                    optimizer.zero_grad(set_to_none=True)
                with phase_timings.span("training_forward_loss", samples=actual_batch_size):
                    logits = _model_predictions(
                        runtime=runtime,
                        module=module,
                        fields=fields,
                        labels=labels,
                        horizons=horizons,
                        target_contract=target_contract,
                    )
                    loss = loss_function(logits, labels)
                if first_logits is None:
                    detach_logits = getattr(logits, "detach", None)
                    first_logits = detach_logits() if callable(detach_logits) else logits
                with phase_timings.span("training_backward", samples=actual_batch_size):
                    loss.backward()
                return loss

            training_cost_shape_key = _training_cost_shape_key(fields)
            cached_training_cost = training_cost_by_shape.get(training_cost_shape_key)
            if cached_training_cost is None:
                with phase_timings.span("training_optimizer_step"):
                    with CostMeter(runtime, strict=False) as training_cost_meter:
                        optimizer_step(runtime, optimizer, loss_closure)
                    training_cost_measurement = (
                        training_cost_meter.measurement().without_operation_trace()
                    )
                cached_training_cost = (training_cost_measurement, actual_batch_size)
                training_cost_by_shape[training_cost_shape_key] = cached_training_cost
            else:
                with phase_timings.span("training_optimizer_step"):
                    optimizer_step(runtime, optimizer, loss_closure)
            latest_training_cost = cached_training_cost
            if training_batch.sample_set is not None:
                if first_logits is None:
                    raise BenchmarkRunnerError("optimizer did not evaluate training loss")
                with phase_timings.span(
                    "training_replay_score_update",
                    samples=training_batch.sample_set.sample_count,
                ):
                    pending_replay_scores.append(
                        _PendingReplayScore(
                            sample_set=training_batch.sample_set,
                            accepted_mass=competence.training_logit_mass_tensor(
                                runtime,
                                first_logits,
                                labels,
                                module=module,
                                fields=fields,
                                horizons=horizons,
                                batch=training_batch.sample_set,
                                generator=generator,
                            ),
                        )
                    )
        except RuntimeError as error:
            if not _is_runtime_capacity_error(error):
                raise
            stop_reason = "capacity-limited"
            break
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
        hit_step_cap = max_steps is not None and step == max_steps
        if step % gate_check_interval != 0 and not hit_step_cap:
            continue
        try:
            append_validation(step=step, check=validation_check)
        except _RuntimeCapacityReached:
            stop_reason = "capacity-limited"
            break
        except RuntimeError as error:
            if not _is_runtime_capacity_error(error):
                raise
            stop_reason = "capacity-limited"
            break
        validation_check += 1
        with phase_timings.span("validation_plateau_check"):
            rung_has_plateaued = (
                patience > 0
                and has_windowed_validation_plateau(
                    validation_history[plateau_window_start_index:],
                    window_checks=patience,
                    min_delta=min_delta,
                    chance_mass=chance_mass,
                )
            )
        if rung_has_plateaued:
            with phase_timings.span("validation_rung_competence_threshold"):
                best_rung_competence = _training_history_best_competence_fraction(
                    validation_history[plateau_window_start_index:],
                    chance_mass=chance_mass,
                )
            if best_rung_competence < rung_competence_threshold:
                stop_reason = "validation-plateau"
                break
            if hit_step_cap:
                stop_reason = "max-steps"
                break
            if scheduler is not None and not scheduler.has_exhausted_plateau_response():
                continue
            with phase_timings.span("validation_plateau_handler"):
                advanced_frontier = (
                    on_plateau is not None and on_plateau(tuple(validation_history))
                )
            if advanced_frontier:
                plateau_window_start_index = len(validation_history) - 1
                best_score = _training_history_frontier_competence(
                    validation_history[-1],
                    chance_mass=chance_mass,
                )
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
        validation_inference_cost=max_validation_inference_cost,
    )


def _training_gate_score_estimate(
    *,
    batch: GeneratedSampleSet,
    target_contract: TargetContract,
    accepted_mass: tuple[float, ...],
    previous_frontier_points: tuple[ValidationCompetencePoint, ...] = (),
    validation_check: int,
    step: int,
    inference_cost: tuple[CostMeasurement, int],
    training_cost: tuple[CostMeasurement, int] | None,
    competence_diagnostics: tuple[Mapping[str, object], ...] = (),
) -> dict[str, object]:
    current_point = _sampled_competence_record_from_accepted_mass(
        batch=batch,
        accepted_mass=accepted_mass,
        volume_axis=None,
        bounded_mass=target_contract.kind != "field-valued",
    )
    sampled_competence = _training_sampled_competence_record(
        benchmark_id=batch.benchmark_id,
        previous_frontier_points=previous_frontier_points,
        current_point=current_point,
        bounded_mass=target_contract.kind != "field-valued",
    )
    compact_sampled_competence = _compact_training_sampled_competence(sampled_competence)
    _assign_sampled_competence_inference_cost(
        compact_sampled_competence,
        measurement=inference_cost[0],
        sample_count=inference_cost[1],
    )
    point_records = _training_score_estimate_points(compact_sampled_competence)
    chance_mass = _target_contract_chance_mass(target_contract)
    competence_points = tuple(
        CompetencePoint.from_sampled_record(
            point,
            field_prefix="score_estimate",
            error_type=BenchmarkRunnerError,
        )
        for point in point_records
    )
    score_integral = _training_score_integral(
        target_contract=target_contract,
        points=competence_points,
        chance_mass=chance_mass,
    )
    score = score_integral.value
    record: dict[str, object] = {
        "kind": "training-running-score-estimate",
        "status": "provisional",
        "evidence_status": "not-accepted",
        "score_frame": "none",
        "scoring_recipe": "sampled-competence-v1",
        "score": score,
        "validation_check": validation_check,
        "step": step,
        "inference_cost_measurement": inference_cost[0].without_operation_trace().to_record(),
        "inference_cost_sample_count": inference_cost[1],
        "chance_mass": chance_mass,
        "score_integral": score_integral.to_record(kind="sampled-competence-integral"),
        "sampled_competence": compact_sampled_competence,
    }
    if training_cost is not None:
        record["training_cost_measurement"] = (
            training_cost[0].without_operation_trace().to_record()
        )
        record["training_cost_sample_count"] = training_cost[1]
    if competence_diagnostics:
        record["competence_diagnostics"] = [dict(item) for item in competence_diagnostics]
    return record


def _training_score_integral(
    *,
    target_contract: TargetContract,
    points: tuple[CompetencePoint, ...],
    chance_mass: float,
) -> StateSpaceIntegral:
    if target_contract.kind != "field-valued":
        return sampled_competence_frontier_integral(points, chance_mass=chance_mass)
    terms: list[StateSpaceIntegralTerm] = []
    cursor = 0.0
    for point in sorted(points, key=_competence_point_interval_sort_key):
        lower, upper = _competence_point_interval(point)
        measured_lower = max(lower, cursor)
        if upper > measured_lower:
            terms.append(
                StateSpaceIntegralTerm(
                    lower=measured_lower,
                    upper=upper,
                    competence_density=point.accepted_mass,
                    kind="measured-state-space-validated-bits",
                    representative_log2_volume=point.log2_volume,
                    sample_count=point.sample_count,
                    confidence_half_width=point.confidence_half_width,
                    confidence_method_id=point.confidence_method_id,
                    region=point.region,
                )
            )
        cursor = max(cursor, lower, upper)
    return StateSpaceIntegral(terms=tuple(terms))


def _competence_point_interval_sort_key(point: CompetencePoint) -> tuple[float, float]:
    lower, upper = _competence_point_interval(point)
    return (lower, upper)


def _competence_point_interval(point: CompetencePoint) -> tuple[float, float]:
    lower = (
        point.log2_volume_minimum
        if point.log2_volume_minimum is not None
        else point.log2_volume
    )
    upper = (
        point.log2_volume_maximum
        if point.log2_volume_maximum is not None
        else point.log2_volume
    )
    if upper <= lower:
        upper = point.log2_volume
        lower = 0.0
    return (lower, upper)


def _sampled_competence_record_from_accepted_mass(
    *,
    batch: GeneratedSampleSet,
    accepted_mass: tuple[float, ...],
    volume_axis: str | None,
    bounded_mass: bool = True,
) -> dict[str, object]:
    """Return sampled competence from target probability mass."""

    if len(batch.samples) != len(accepted_mass):
        raise BenchmarkRunnerError("sampled competence requires one mass per sample")
    record: dict[str, object] = {
        "kind": "sampled-state-space-volume-window",
        "sampling_rule": "generator-uniform-component-index-v1",
        "difficulty_assumption": "approximately-uniform-within-volume-window",
        "benchmark_id": str(batch.benchmark_id),
        "volume_axis": volume_axis,
        "log2_volume": batch.log2_volume,
        "seed": batch.seed,
        "sample_count": len(batch.samples),
        "mean_accepted_mass": math.fsum(accepted_mass) / len(accepted_mass),
    }
    if bounded_mass:
        finite_losses = tuple(-math.log(mass) for mass in accepted_mass if mass > 0.0)
        mean_negative_log_score: float | str
        if len(finite_losses) != len(accepted_mass):
            mean_negative_log_score = "infinity"
        else:
            mean_negative_log_score = math.fsum(finite_losses) / len(finite_losses)
        record["mean_negative_log_score"] = mean_negative_log_score
    else:
        record["competence_value_kind"] = "validated-bits"
    input_shape = _optional_batch_sample_input_shape(batch=batch)
    if input_shape is not None:
        record["input_shape"] = list(input_shape)
    if batch.volume_request is not None:
        record["log2_volume_minimum"] = batch.volume_request.minimum
        record["log2_volume_maximum"] = batch.volume_request.maximum
    if batch.region is not None:
        record["region"] = batch.region.to_record()
    return record


def _optional_batch_sample_input_shape(
    *,
    batch: GeneratedSampleSet,
) -> tuple[int, ...] | None:
    try:
        return _batch_sample_input_shape(batch=batch)
    except BenchmarkRunnerError:
        return None


def _coerce_training_step_batch(
    value: _TrainingStepBatch | tuple[Any, Any],
) -> _TrainingStepBatch:
    if isinstance(value, _TrainingStepBatch):
        return value
    fields, labels = value
    return _TrainingStepBatch(fields=fields, labels=labels)


def _field_training_sample_keys(
    batch: GeneratedSampleSet | None,
) -> tuple[Mapping[str, object], ...]:
    if batch is None:
        return ()
    return tuple(sample.to_record(include_field=False) for sample in batch.samples)


def _refreshed_frontier_points(
    historical_points: Iterable[ValidationCompetencePoint],
    replay_points: Iterable[ValidationCompetencePoint],
) -> tuple[ValidationCompetencePoint, ...]:
    by_interval = {
        _validation_competence_point_interval_key(point): point
        for point in historical_points
    }
    for point in replay_points:
        by_interval[_validation_competence_point_interval_key(point)] = point
    return tuple(
        by_interval[key]
        for key in sorted(by_interval, key=lambda interval: (interval[0], interval[1]))
    )


def _accumulate_replay_frontier_point(
    replay_frontier_points: dict[
        tuple[float, float],
        _RollingValidationCompetencePoint,
    ],
    point: ValidationCompetencePoint,
) -> None:
    key = _validation_competence_point_interval_key(point)
    existing = replay_frontier_points.get(key)
    if existing is None:
        replay_frontier_points[key] = _RollingValidationCompetencePoint.from_point(
            point
        )
        return
    replay_frontier_points[key] = existing.add(point)


def _training_cost_shape_key(fields: Any) -> object:
    """Return the cache key for one-measurement-per-shape training cost metering.

    Operation streams are deterministic per module and input shape, so one
    metered step per distinct field shape carries the same evidence as metering
    every step while keeping per-op dispatch interception off the hot path.
    """

    shape = getattr(fields, "shape", None)
    if shape is None:
        return "shapeless"
    try:
        return tuple(int(extent) for extent in shape)
    except (TypeError, ValueError):
        return "shapeless"


def _flush_pending_replay_scores(
    *,
    pending_replay_scores: list[_PendingReplayScore],
    replay_frontier_points: dict[
        tuple[float, float],
        _RollingValidationCompetencePoint,
    ],
) -> None:
    if not pending_replay_scores:
        return
    for pending in pending_replay_scores:
        accepted_mass = tuple(float(value) for value in pending.accepted_mass.tolist())
        replay_point = ValidationCompetencePoint.from_sampled_record(
            _sampled_competence_record_from_accepted_mass(
                batch=pending.sample_set,
                accepted_mass=accepted_mass,
                volume_axis=None,
            ),
            field_prefix="score_estimate",
            error_type=BenchmarkRunnerError,
        )
        _accumulate_replay_frontier_point(
            replay_frontier_points,
            replay_point,
        )
    pending_replay_scores.clear()


def _validation_competence_point_interval_key(
    point: ValidationCompetencePoint,
) -> tuple[float, float]:
    return (
        point.log2_volume if point.log2_volume_minimum is None else point.log2_volume_minimum,
        point.log2_volume if point.log2_volume_maximum is None else point.log2_volume_maximum,
    )


def _training_sampled_competence_record(
    *,
    benchmark_id: ProtocolIdentifier,
    previous_frontier_points: tuple[ValidationCompetencePoint, ...],
    current_point: Mapping[str, object],
    bounded_mass: bool = True,
) -> dict[str, object]:
    points: list[Mapping[str, object]] = [
        {
            "kind": "sampled-state-space-volume-window",
            "sampling_rule": "generator-uniform-component-index-v1",
            "difficulty_assumption": "approximately-uniform-within-volume-window",
            "benchmark_id": str(benchmark_id),
            "volume_axis": None,
            "log2_volume": point.log2_volume,
            "seed": point.seed,
            "sample_count": point.sample_count,
            "mean_accepted_mass": point.accepted_mass,
            **_competence_point_input_shape_record(point),
            **_competence_point_interval_record(point),
        }
        for point in previous_frontier_points
    ]
    points.append(current_point)
    if bounded_mass:
        return sampled_competence_curriculum_record(points)
    return _sampled_unbounded_competence_curriculum_record(points)


def _sampled_unbounded_competence_curriculum_record(
    points: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if not points:
        raise BenchmarkRunnerError("sampled competence curriculum requires at least one point")
    sorted_points = tuple(
        sorted(
            points,
            key=lambda point: _required_float(
                point.get("log2_volume"),
                "sampled_competence.log2_volume",
            ),
        )
    )
    first = sorted_points[0]
    sample_counts = tuple(
        _required_int(point.get("sample_count"), "sampled_competence.sample_count")
        for point in sorted_points
    )
    values = tuple(
        _required_float(
            point.get("mean_accepted_mass"),
            "sampled_competence.mean_accepted_mass",
        )
        for point in sorted_points
    )
    total_samples = sum(sample_counts)
    if total_samples < 1:
        raise BenchmarkRunnerError("sampled competence sample count must be positive")
    weighted_value = (
        math.fsum(
            value * sample_count
            for value, sample_count in zip(values, sample_counts, strict=True)
        )
        / total_samples
    )
    return {
        "kind": "sampled-competence-curriculum",
        "sampling_rule": first.get("sampling_rule"),
        "difficulty_assumption": first.get("difficulty_assumption"),
        "benchmark_id": first.get("benchmark_id"),
        "volume_axis": first.get("volume_axis"),
        "log2_volume": first.get("log2_volume"),
        "sample_count": total_samples,
        "mean_accepted_mass": weighted_value,
        "competence_value_kind": "validated-bits",
        "points": [dict(point) for point in sorted_points],
    }


def _rung_log2_volume_interval(rung: _CurriculumRung) -> tuple[float, float] | None:
    if rung.log2_volume_minimum is not None and rung.log2_volume_maximum is not None:
        return (rung.log2_volume_minimum, rung.log2_volume_maximum)
    if rung.batch.volume_request is None:
        return None
    return (
        rung.batch.volume_request.minimum,
        rung.batch.volume_request.maximum,
    )


def _rung_log2_volume_interval_record(rung: _CurriculumRung) -> dict[str, object]:
    interval = _rung_log2_volume_interval(rung)
    if interval is None:
        return {}
    lower, upper = interval
    return {
        "log2_volume_minimum": lower,
        "log2_volume_maximum": upper,
    }


def _competence_point_interval_record(point: ValidationCompetencePoint) -> dict[str, object]:
    if point.log2_volume_minimum is None or point.log2_volume_maximum is None:
        return {}
    return {
        "log2_volume_minimum": point.log2_volume_minimum,
        "log2_volume_maximum": point.log2_volume_maximum,
    }


def _competence_point_input_shape_record(
    point: ValidationCompetencePoint,
) -> dict[str, object]:
    if point.input_shape is None:
        return {}
    return {"input_shape": list(point.input_shape)}


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


def _assign_sampled_competence_inference_cost(
    sampled_competence: dict[str, object],
    *,
    measurement: CostMeasurement,
    sample_count: int,
) -> None:
    sampled_competence["inference_cost_measurement"] = (
        measurement.without_operation_trace().to_record()
    )
    sampled_competence["inference_cost_sample_count"] = sample_count
    points = sampled_competence.get("points")
    if not isinstance(points, list):
        return
    for point in cast(list[object], points):
        if isinstance(point, dict):
            point["inference_cost_measurement"] = (
                measurement.without_operation_trace().to_record()
            )
            point["inference_cost_sample_count"] = sample_count


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


def _training_history_checkpoint_selection_score(point: TrainingHistoryPoint) -> float:
    if point.score_estimate is None:
        return -float("inf")
    return _checkpoint_score_estimate_selection_score(point.score_estimate)


def _training_score_estimate_frontier_competence(
    score_estimate: Mapping[str, object],
    *,
    chance_mass: float,
) -> float:
    points = _training_score_estimate_points(score_estimate)
    if not points:
        return 0.0
    latest_point = points[-1]
    return _competence_fraction(
        accepted_mass=_required_float(
            latest_point.get("mean_accepted_mass"),
            "score_estimate.mean_accepted_mass",
        ),
        chance_mass=chance_mass,
    )


def _training_history_frontier_competence(
    point: TrainingHistoryPoint,
    *,
    chance_mass: float,
) -> float:
    if point.score_estimate is None:
        return 0.0
    return _training_score_estimate_frontier_competence(
        point.score_estimate,
        chance_mass=chance_mass,
    )


def _training_history_latest_cost_measurement(
    validation_history: Sequence[TrainingHistoryPoint],
) -> tuple[CostMeasurement, int] | None:
    for point in validation_history:
        if point.score_estimate is None:
            continue
        measurement_record = point.score_estimate.get("inference_cost_measurement")
        sample_count = point.score_estimate.get("inference_cost_sample_count")
        if measurement_record is None:
            continue
        sample_count_value = _optional_positive_int(sample_count)
        if sample_count_value is None:
            continue
        try:
            measurement = CostMeasurement.from_record(measurement_record)
        except ValueError:
            continue
        return (measurement, sample_count_value)
    return None


def _training_history_latest_training_cost_measurement(
    validation_history: Sequence[TrainingHistoryPoint],
) -> tuple[CostMeasurement, int] | None:
    for point in reversed(validation_history):
        if point.score_estimate is None:
            continue
        measurement_record = point.score_estimate.get("training_cost_measurement")
        sample_count = point.score_estimate.get("training_cost_sample_count")
        if measurement_record is None:
            continue
        sample_count_value = _optional_positive_int(sample_count)
        if sample_count_value is None:
            continue
        try:
            measurement = CostMeasurement.from_record(measurement_record)
        except ValueError:
            continue
        return (measurement, sample_count_value)
    return None


def _training_history_frontier_point(point: TrainingHistoryPoint) -> ValidationCompetencePoint:
    if point.score_estimate is None:
        raise BenchmarkRunnerError("training gate is missing score estimate")
    points = _training_score_estimate_points(point.score_estimate)
    if not points:
        raise BenchmarkRunnerError("training gate score estimate has no competence point")
    latest_point = points[-1]
    return ValidationCompetencePoint.from_sampled_record(
        latest_point,
        field_prefix="score_estimate",
        error_type=BenchmarkRunnerError,
    )


def _training_rung_index_for_step(*, step: int, frontier_index: int) -> int:
    if frontier_index <= 0:
        return 0
    if step % 2 == 0:
        return frontier_index
    return (step // 2) % frontier_index


def _training_history_best_competence_fraction(
    validation_history: Sequence[TrainingHistoryPoint],
    *,
    chance_mass: float,
) -> float:
    best = 0.0
    for point in validation_history:
        if point.score_estimate is None:
            continue
        best = max(
            best,
            _training_score_estimate_frontier_competence(
                point.score_estimate,
                chance_mass=chance_mass,
            ),
        )
    return best


def _competence_fraction(*, accepted_mass: float, chance_mass: float) -> float:
    if chance_mass >= 1.0:
        return 1.0 if accepted_mass >= 1.0 else 0.0
    competence = (accepted_mass - chance_mass) / (1.0 - chance_mass)
    return min(1.0, max(0.0, competence))


def _tensor_batch_size(fields: Any, *, fallback: int) -> int:
    shape = getattr(fields, "shape", None)
    if shape is None or len(shape) < 1:
        return fallback
    value = shape[0]
    if type(value) is int and value >= 0:
        return value
    return fallback


def _evaluation_result_frontier_index(
    *,
    evaluation_results: Sequence[_CheckpointEvaluationRungEvidence],
    outcome_ids: tuple[str, ...],
    target_contract: TargetContract,
) -> int:
    if not evaluation_results:
        raise BenchmarkRunnerError("evaluation did not produce any rungs")
    _ = outcome_ids
    chance_mass = _target_contract_chance_mass(target_contract)
    frontier_index = 0
    for index, result in enumerate(evaluation_results):
        if _evaluation_rung_confidently_above_chance(
            result,
            chance_mass=chance_mass,
        ):
            frontier_index = index
            continue
        break
    return frontier_index


def _evaluation_rung_confidently_above_chance(
    result: _CheckpointEvaluationRungEvidence,
    *,
    chance_mass: float,
) -> bool:
    return result.mean_accepted_mass - result.confidence_half_width > chance_mass


def has_windowed_validation_plateau(
    validation_history: Sequence[TrainingHistoryPoint],
    *,
    window_checks: int,
    min_delta: float,
    chance_mass: float,
) -> bool:
    if window_checks <= 0 or len(validation_history) <= window_checks:
        return False
    current = validation_history[-1]
    window_start = validation_history[-1 - window_checks]
    current_competence = _training_history_frontier_competence(
        current,
        chance_mass=chance_mass,
    )
    window_start_competence = _training_history_frontier_competence(
        window_start,
        chance_mass=chance_mass,
    )
    return current_competence - window_start_competence < min_delta


def _training_run_record(
    *,
    seed: int,
    max_steps: int | None,
    learning_rate: float | None,
    optimizer_name: str,
    schedule_name: str,
    gate_check_interval: int,
    gate_decision_rule: str,
    rung_competence_threshold: float,
    convergence_patience: int,
    convergence_min_delta: float,
    tensor_device: str,
    runtime_memory_budget_fraction: float | None = None,
    validation_history: tuple[TrainingHistoryPoint, ...],
    stop_reason: str,
) -> TrainingRunRecord:
    last_step = validation_history[-1].step
    if stop_reason == "no-training-steps":
        status = "completed"
    elif stop_reason == "validation-plateau":
        status = "converged"
    elif stop_reason in {"max-steps", "capacity-limited"}:
        status = "budget-exhausted"
    else:
        status = "completed"
    return TrainingRunRecord(
        status=status,
        stop_reason=stop_reason,
        steps_run=last_step,
        validation_checks=len(validation_history),
        protocol=TrainingProtocol(
            kind="fixed-step-local-batch",
            objective="cross-entropy",
            optimizer=cast(Any, optimizer_name),
            learning_rate=learning_rate,
            schedule=cast(Any, schedule_name),
            seed=seed,
            max_steps=max_steps,
            gate_check_interval=gate_check_interval,
            gate_decision_rule=gate_decision_rule,
            rung_competence_threshold=rung_competence_threshold,
            min_delta=convergence_min_delta,
            patience=convergence_patience,
            validation_source="generator-resample",
            tensor_runtime="pytorch",
            tensor_device=tensor_device,
            runtime_memory_budget_fraction=runtime_memory_budget_fraction,
        ),
        validation_history=validation_history,
    )


def _running_training_run_record(
    *,
    seed: int,
    max_steps: int | None,
    learning_rate: float | None,
    optimizer_name: str,
    schedule_name: str,
    gate_check_interval: int,
    gate_decision_rule: str,
    rung_competence_threshold: float,
    convergence_patience: int,
    convergence_min_delta: float,
    tensor_device: str,
    runtime_memory_budget_fraction: float | None = None,
    validation_history: tuple[TrainingHistoryPoint, ...],
) -> TrainingRunRecord:
    return TrainingRunRecord(
        status="running",
        stop_reason="validation-checkpoint",
        steps_run=validation_history[-1].step,
        validation_checks=len(validation_history),
        protocol=TrainingProtocol(
            kind="fixed-step-local-batch",
            objective="cross-entropy",
            optimizer=cast(Any, optimizer_name),
            learning_rate=learning_rate,
            schedule=cast(Any, schedule_name),
            seed=seed,
            max_steps=max_steps,
            gate_check_interval=gate_check_interval,
            gate_decision_rule=gate_decision_rule,
            rung_competence_threshold=rung_competence_threshold,
            min_delta=convergence_min_delta,
            patience=convergence_patience,
            validation_source="generator-resample",
            tensor_runtime="pytorch",
            tensor_device=tensor_device,
            runtime_memory_budget_fraction=runtime_memory_budget_fraction,
        ),
        validation_history=validation_history,
    )


def _training_progress_record(
    *,
    plan: BenchmarkRunPlan,
    summary: BenchmarkRunSummary,
    inspection: ModelInspectionRecord,
    evaluation_curriculum: Mapping[str, object],
    training_curriculum: Mapping[str, object],
    training_run: TrainingRunRecord,
    throughput: Mapping[str, object],
    model_checkpoints: tuple[Mapping[str, object], ...],
    selected_model_checkpoint: Mapping[str, object] | None,
    selected_model_checkpoint_score_estimate: Mapping[str, object] | None,
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
        "program_path": summary.program_path.as_posix(),
        "seed": plan.seed,
        "train_steps": plan.train_steps,
        "optimizer": plan.optimizer,
        "schedule": plan.schedule,
        "gate_check_interval": plan.gate_check_interval,
        "model_checkpoint_gate_interval": plan.model_checkpoint_gate_interval,
        "gate_decision_rule": plan.gate_decision_rule,
        "convergence_patience": plan.convergence_patience,
        "convergence_min_delta": float(plan.convergence_min_delta),
        "tensor_runtime": "pytorch",
        "tensor_device": training_run.protocol.tensor_device,
        "training_run": _training_run_artifact_record(training_run),
        "throughput": throughput,
        "evaluation_curriculum": dict(evaluation_curriculum),
        "training_curriculum": dict(training_curriculum),
        "program": inspection.program.to_record(),
        "program_graph": dict(inspection.program_graph),
        "cost_summary": _training_cost_summary(
            inspection=inspection,
            training_estimate=training_estimate,
            training_run=training_run,
        ),
        "model_inspection": inspection.to_record(),
        "program_digest": str(inspection.program.record_digest),
        "model_inspection_digest": str(inspection.digest),
        "training_estimate": training_estimate,
        "model_checkpoints": [dict(checkpoint) for checkpoint in model_checkpoints],
        "selected_model_checkpoint": (
            None if selected_model_checkpoint is None else dict(selected_model_checkpoint)
        ),
        "selected_model_checkpoint_score_estimate": (
            None
            if selected_model_checkpoint_score_estimate is None
            else dict(selected_model_checkpoint_score_estimate)
        ),
        "selected_model_checkpoint_policy": "highest-training-score-estimate",
    }
    if plan.learning_rate is not None:
        record["learning_rate"] = float(plan.learning_rate)
    record["sampled_competence"] = dict(
        cast(Mapping[str, object], training_estimate["sampled_competence"])
    )
    return record


def _training_run_artifact_record(training_run: TrainingRunRecord) -> dict[str, object]:
    record = training_run.to_record()
    record["validation_history"] = [
        _training_history_artifact_point_record(point)
        for point in training_run.validation_history
    ]
    return record


def _training_history_artifact_point_record(
    point: TrainingHistoryPoint,
) -> dict[str, object]:
    record: dict[str, object] = {
        "step": point.step,
        "validation_check": point.validation_check,
        "validation_loss": point.validation_loss,
        "stale_checks": point.stale_checks,
    }
    if point.learning_rates:
        record["learning_rates"] = list(point.learning_rates)
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
    log2_volume = _required_float(
        sampled_competence.get("log2_volume"),
        "training_estimate.log2_volume",
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
    score_integral_value = score_estimate.get("score_integral")
    if not isinstance(score_integral_value, Mapping):
        raise BenchmarkRunnerError("training gate score estimate is missing score integral")
    score_integral = cast(Mapping[str, object], score_integral_value)
    cost_integral = sampled_competence_metrology_cost_integral(
        points=points,
        error_type=BenchmarkRunnerError,
        field_prefix="training_estimate.cost_point",
    )
    return {
        "kind": "training-running-score-estimate",
        "status": "provisional",
        "evidence_status": "not-accepted",
        "score_frame": "none",
        "scoring_recipe": score_estimate.get("scoring_recipe", "sampled-competence-v1"),
        "benchmark_id": str(summary.benchmark_id),
        "volume_axis": None,
        "log2_volume": log2_volume,
        "seed": seed,
        "sample_count": sample_count,
        "mean_accepted_mass": mean_accepted_mass,
        "chance_mass": score_estimate.get("chance_mass"),
        "score": _training_score_estimate_score(score_estimate),
        "score_integral": dict(score_integral),
        "cost": cost_integral.value,
        "cost_integral": cost_integral.to_record(kind="compute-cost-integral"),
        "validation_check": latest.validation_check,
        "step": latest.step,
        "inference_cost_measurement": dict(
            _required_mapping(
                score_estimate.get("inference_cost_measurement"),
                "training_estimate.inference_cost_measurement",
            )
        ),
        "inference_cost_sample_count": _required_int(
            score_estimate.get("inference_cost_sample_count"),
            "training_estimate.inference_cost_sample_count",
        ),
        "sampled_competence": dict(sampled_competence),
    }


def _training_cost_summary(
    *,
    inspection: ModelInspectionRecord,
    training_estimate: Mapping[str, object],
    training_run: TrainingRunRecord,
) -> dict[str, object]:
    cost_summary = inspection.cost_summary.to_record()
    cost = _optional_nonnegative_float(
        training_estimate.get("cost"),
        "training_estimate.cost",
    )
    if cost is not None:
        cost_summary["cost"] = cost
    inference_cost = _training_history_latest_cost_measurement(
        training_run.validation_history
    )
    if inference_cost is not None:
        cost_summary["inference_cost_measurement"] = (
            inference_cost[0].without_operation_trace().to_record()
        )
        cost_summary["inference_cost_sample_count"] = inference_cost[1]
    training_cost = _training_history_latest_training_cost_measurement(
        training_run.validation_history
    )
    if training_cost is not None:
        cost_summary["training_cost_measurement"] = (
            training_cost[0].without_operation_trace().to_record()
        )
        cost_summary["training_cost_sample_count"] = training_cost[1]
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
    latest_score = _training_history_checkpoint_selection_score(latest)
    saved_score = max(
        _checkpoint_selection_score(checkpoint) for checkpoint in checkpoint_artifacts
    )
    return latest_score > saved_score


def _write_model_checkpoint_artifact(
    *,
    summary: BenchmarkRunSummary,
    program_graph: Mapping[str, object],
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
    program_reference = reference_for_record(kind="program-graph", record=program_graph)
    manifest = ModelArtifactManifest(
        id=ProtocolIdentifier.parse(
            f"model-manifests.{_identifier_atom(summary.benchmark_id)}."
            f"{summary.run_slug}.{stem}@0.1.0"
        ),
        program=program_reference,
        interface=reference_for_record(
            kind="model-interface",
            record=model_interface.to_record(),
        ),
        execution_family=(
            ModelExecutionFamily.submitted_program_graph()
            if runtime == "pytorch"
            else ModelExecutionFamily(
                kind=f"local-{runtime}",
                runtime=runtime,
                program_family="open-node-program-graph",
            )
        ),
        model_artifacts=(checkpoint_reference,),
        training_provenance=(
            ArtifactReference(
                kind="training-run",
                record_digest=ContentDigest.from_value(
                    _training_run_artifact_record(training_run)
                ),
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
        state[str(key)] = tensor_value_to_host(value)
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
            _checkpoint_selection_score(checkpoint),
            -checkpoint.validation_loss,
            checkpoint.validation_check,
            checkpoint.step,
        ),
    )


def _checkpoint_selection_score(checkpoint: ModelCheckpointArtifact) -> float:
    if checkpoint.score_estimate is None:
        return -float("inf")
    return _checkpoint_score_estimate_selection_score(checkpoint.score_estimate)


def _checkpoint_score_estimate_selection_score(
    score_estimate: Mapping[str, object],
) -> float:
    return _training_score_estimate_score(score_estimate)


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
    tensor_compile_fallbacks: Sequence[Mapping[str, object]] = (),
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
    if tensor_compile_fallbacks:
        record["tensor_compile_fallbacks"] = [
            dict(fallback) for fallback in tensor_compile_fallbacks
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
    arithmetic_intensity = CostMeasurement.abstract_flops_per_byte_value(
        work.compute_per_sample,
        work.bytes_per_sample,
    )
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
        observed_compute = CostMeasurement.abstract_flops_rate_value(
            work.compute_per_sample,
            measured_samples,
        )
    return {
        "compute_per_sample": work.compute_per_sample,
        "compute_source": work.compute_source,
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
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    inference_cost: tuple[CostMeasurement, int] | None,
    training_cost: tuple[CostMeasurement, int] | None,
    storage_bytes: int | None,
    batch_size: int,
) -> _TrainingWorkEstimates | None:
    if inference_cost is None:
        return None
    inference_measurement = inference_cost[0]
    inference_ops_per_sample = _measurement_ops_per_item(inference_cost)
    training_ops_per_sample = (
        None if training_cost is None else _measurement_ops_per_item(training_cost)
    )
    if (
        inference_ops_per_sample <= 0
        or training_ops_per_sample is None
        or training_ops_per_sample <= 0
    ):
        return None
    input_bytes = _shape_bytes(input_shape)
    output_bytes = _shape_bytes(output_shape)
    batch_size_value = max(1, batch_size)
    storage_bytes_per_sample = float(storage_bytes or 0) / batch_size_value
    inference_bytes = input_bytes + output_bytes + storage_bytes_per_sample
    inference_cost_source = _cost_measurement_compute_source(inference_measurement)
    formation_bytes = 8.0 * input_bytes
    training_cost_per_sample = float(training_ops_per_sample)
    training_bytes = formation_bytes + 3.0 * inference_bytes + 4.0 * storage_bytes_per_sample
    validation_bytes = formation_bytes + inference_bytes
    evaluation_bytes = input_bytes + inference_bytes
    return _TrainingWorkEstimates(
        training=_PhaseWorkEstimate(
            compute_per_sample=training_cost_per_sample,
            bytes_per_sample=training_bytes,
            compute_source="measured-training-metrology",
        ),
        validation=_PhaseWorkEstimate(
            compute_per_sample=inference_ops_per_sample,
            bytes_per_sample=validation_bytes,
            compute_source=inference_cost_source,
        ),
        evaluation=_PhaseWorkEstimate(
            compute_per_sample=inference_ops_per_sample,
            bytes_per_sample=evaluation_bytes,
            compute_source=inference_cost_source,
        ),
        assumptions=(
            "float32 tensor elements are four bytes",
            "training compute uses measured runtime metrology from optimizer steps",
            (
                "validation and evaluation compute use measured forward-pass metrology"
                if inference_measurement.operations_executed
                else (
                    "validation and evaluation compute use dry-run cost metrology "
                    f"from {inference_measurement.operation_stream_source}"
                )
            ),
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
    learning_rate: float | None,
) -> Any:
    try:
        return build_optimizer(
            runtime, name=name, parameters=parameters, learning_rate=learning_rate
        )
    except TensorRuntimeError as error:
        raise BenchmarkRunnerError(str(error)) from error


def _optimizer_learning_rates(optimizer: Any) -> tuple[float, ...]:
    rates: list[float] = []
    for group in getattr(optimizer, "param_groups", ()):
        if not isinstance(group, Mapping) or "lr" not in group:
            continue
        group_record = cast(Mapping[str, object], group)
        value = group_record["lr"]
        if isinstance(value, int | float) and not isinstance(value, bool):
            rates.append(float(value))
    return tuple(rates)


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
            curriculum_expansion_learning_rates=tuple(
                float(group["lr"]) * factor for group in optimizer.param_groups
            ),
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


def _batch_prediction_accepted_mass(
    *,
    batch: GeneratedSampleSet,
    probabilities: tuple[tuple[float, ...], ...],
    outcome_ids: tuple[str, ...],
) -> tuple[float, ...]:
    if batch.samples:
        return tuple(
            _prediction_target_mass(
                row,
                target_distribution=sample.target_distribution_or_one_hot(),
                outcome_ids=outcome_ids,
            )
            for sample, row in zip(batch.samples, probabilities, strict=True)
        )
    if batch.targets is None:
        raise BenchmarkRunnerError("prediction scoring requires targets or sample metadata")
    targets = tensor_value_to_host(batch.targets)
    to_list = getattr(targets, "tolist", None)
    if callable(to_list):
        targets = to_list()
    if batch.sample_count == 1 and type(targets) in {int, float}:
        target_rows: Sequence[object] = (targets,)
    else:
        target_rows = cast(Sequence[object], targets)
    if len(target_rows) != len(probabilities):
        raise BenchmarkRunnerError("prediction scoring requires one target per prediction")
    return tuple(
        _target_tensor_prediction_mass(row, target=target)
        for row, target in zip(probabilities, target_rows, strict=True)
    )


def _target_tensor_prediction_mass(
    probabilities: Sequence[float],
    *,
    target: object,
) -> float:
    if type(target) is int:
        if target < 0 or target >= len(probabilities):
            raise BenchmarkRunnerError("target label is outside prediction range")
        return float(probabilities[target])
    if isinstance(target, Sequence) and not isinstance(target, str):
        target_values = cast(Sequence[object], target)
        if len(target_values) != len(probabilities):
            raise BenchmarkRunnerError(
                "target distribution width does not match prediction width"
            )
        return math.fsum(
            _target_probability_value(target_probability) * float(probability)
            for target_probability, probability in zip(
                target_values,
                probabilities,
                strict=True,
            )
        )
    raise BenchmarkRunnerError("unsupported target tensor shape for prediction scoring")


def _target_probability_value(value: object) -> float:
    if not isinstance(value, int | float):
        raise BenchmarkRunnerError("target distribution contains nonnumeric probability")
    probability = float(value)
    if not math.isfinite(probability):
        raise BenchmarkRunnerError("target distribution contains nonfinite probability")
    return probability


def _finite_outcome_ids(contract: TargetContract) -> tuple[str, ...]:
    if contract.kind != "finite-outcome" or contract.outcome_ids is None:
        raise BenchmarkRunnerError("runner path requires a finite-outcome target contract")
    return contract.outcome_ids


def _target_contract_outcome_ids(contract: TargetContract) -> tuple[str, ...]:
    if contract.kind == "field-valued":
        return ()
    return _finite_outcome_ids(contract)


def _target_contract_chance_mass(contract: TargetContract) -> float:
    chance_mass = contract.chance_mass()
    if chance_mass is None and contract.kind == "field-valued":
        return 0.0
    if chance_mass is None:
        raise BenchmarkRunnerError("runner path requires finite-outcome chance mass")
    return chance_mass


def _optional_training_loss_factory(
    benchmark: object,
) -> _BenchmarkTrainingLossFactory | None:
    factory = getattr(benchmark, "build_training_loss", None)
    if factory is None:
        return None
    if not callable(factory):
        raise BenchmarkRunnerError("benchmark build_training_loss must be callable")
    return cast(_BenchmarkTrainingLossFactory, benchmark)


def _optional_training_competence_factory(
    benchmark: object,
) -> _BenchmarkTrainingCompetenceFactory | None:
    factory = getattr(benchmark, "build_training_competence", None)
    if factory is None:
        return None
    if not callable(factory):
        raise BenchmarkRunnerError("benchmark build_training_competence must be callable")
    return cast(_BenchmarkTrainingCompetenceFactory, benchmark)


def _build_training_loss(
    *,
    runtime: TensorRuntime,
    target_contract: TargetContract,
    loss_factory: _BenchmarkTrainingLossFactory | None,
) -> Any:
    if target_contract.loss_id == "equation-residual":
        if loss_factory is None:
            raise BenchmarkRunnerError(
                "equation-residual target contracts require benchmark build_training_loss"
            )
        return loss_factory.build_training_loss(runtime, target_contract)
    return build_loss(runtime, target_contract)


@dataclass(frozen=True, slots=True)
class _CompetenceFunctional:
    """Per-sample competence selected by a benchmark target contract.

    The runner turns model outputs and targets into per-sample accepted mass
    through this functional, chosen by ``contract.competence.kind``. Only the
    finite-outcome ``above-chance-accepted-mass`` functional is implemented in
    the runner today; field-valued competence kinds resolve here once a
    benchmark declares them, so the per-sample competence step is a contract
    dispatch rather than a hardcoded softmax call.
    """

    kind: str
    field_mass_tensor: Callable[[_FieldTrainingCompetenceRequest], Any] | None = None

    def training_logit_masses(
        self,
        runtime: TensorRuntime,
        logits: Any,
        labels: Any,
        *,
        module: Any | None = None,
        fields: Any | None = None,
        horizons: tuple[float, ...] | None = None,
        batch: GeneratedSampleSet | None = None,
        generator: _TensorBenchmarkGenerator | None = None,
    ) -> tuple[float, ...]:
        return self.training_logit_masses_with_diagnostics(
            runtime,
            logits,
            labels,
            module=module,
            fields=fields,
            horizons=horizons,
            batch=batch,
            generator=generator,
        ).values

    def training_logit_masses_with_diagnostics(
        self,
        runtime: TensorRuntime,
        logits: Any,
        labels: Any,
        *,
        module: Any | None = None,
        fields: Any | None = None,
        horizons: tuple[float, ...] | None = None,
        batch: GeneratedSampleSet | None = None,
        generator: _TensorBenchmarkGenerator | None = None,
    ) -> _CompetenceEvaluation:
        if self.kind in _field_valued_competence_kinds:
            if self.field_mass_tensor is None:
                raise BenchmarkRunnerError(
                    f"{self.kind} requires benchmark training competence"
                )
            mass_tensor = self.training_logit_mass_tensor(
                runtime,
                logits,
                labels,
                module=module,
                fields=fields,
                horizons=horizons,
                batch=batch,
                generator=generator,
            )
            return _CompetenceEvaluation(
                values=tuple(float(value) for value in mass_tensor.detach().tolist()),
                diagnostics=_competence_tensor_diagnostics(mass_tensor),
            )
        return _CompetenceEvaluation(
            values=tuple(softmax_target_masses(runtime, logits, labels))
        )

    def training_logit_mass_tensor(
        self,
        runtime: TensorRuntime,
        logits: Any,
        labels: Any,
        *,
        module: Any | None = None,
        fields: Any | None = None,
        horizons: tuple[float, ...] | None = None,
        batch: GeneratedSampleSet | None = None,
        generator: _TensorBenchmarkGenerator | None = None,
    ) -> Any:
        if self.kind in _field_valued_competence_kinds:
            if self.field_mass_tensor is None:
                raise BenchmarkRunnerError(
                    f"{self.kind} requires benchmark training competence"
                )
            return self.field_mass_tensor(
                _FieldTrainingCompetenceRequest(
                    runtime=runtime,
                    module=module,
                    fields=fields,
                    predictions=logits,
                    targets=labels,
                    horizons=horizons,
                    batch=batch,
                    generator=generator,
                    sample_keys=_field_training_sample_keys(batch),
                )
            )
        return softmax_target_mass_tensor(runtime, logits, labels)

    def prediction_accepted_mass(
        self,
        *,
        batch: GeneratedSampleSet,
        probabilities: tuple[tuple[float, ...], ...],
        outcome_ids: tuple[str, ...],
    ) -> tuple[float, ...]:
        return _batch_prediction_accepted_mass(
            batch=batch,
            probabilities=probabilities,
            outcome_ids=outcome_ids,
        )


def _resolve_competence_functional(
    contract: TargetContract,
    *,
    runtime: TensorRuntime | None = None,
    competence_factory: _BenchmarkTrainingCompetenceFactory | None = None,
) -> _CompetenceFunctional:
    if contract.competence.kind not in {
        "above-chance-accepted-mass",
        *_field_valued_competence_kinds,
    }:
        raise BenchmarkRunnerError(
            "runner path does not support competence kind "
            f"{contract.competence.kind!r}"
        )
    if contract.competence.kind in _field_valued_competence_kinds:
        if runtime is None or competence_factory is None:
            raise BenchmarkRunnerError(
                f"{contract.competence.kind} requires benchmark training competence"
            )
        return _CompetenceFunctional(
            kind=contract.competence.kind,
            field_mass_tensor=competence_factory.build_training_competence(
                runtime,
                contract,
            ),
        )
    return _CompetenceFunctional(kind=contract.competence.kind)


def _competence_tensor_diagnostics(tensor: Any) -> tuple[Mapping[str, object], ...]:
    diagnostics = getattr(tensor, "leibniz_competence_diagnostics", ())
    if not isinstance(diagnostics, tuple):
        return ()
    diagnostic_items = cast(tuple[object, ...], diagnostics)
    return tuple(
        cast(Mapping[str, object], item)
        for item in diagnostic_items
        if isinstance(item, Mapping)
    )


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
