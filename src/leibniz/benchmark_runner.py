"""Small benchmark execution workflows for local operator runs."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes, document_filename_suffix
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset, MeasurementRecord
from leibniz.model_inspection import ModelInspectionRecord
from leibniz.model_operators import ExecutableModelOperator
from leibniz.observation_generation import (
    GeneratedObservationBatch,
    GeneratedObservationSample,
    ObservationGenerator,
    load_observation_generator,
)
from leibniz.outcomes import (
    AcceptedEvent,
    FiniteProbabilityMeasure,
    OutcomeSpace,
    ProbabilityMass,
    RawScoringEvidence,
)
from leibniz.tensor_runtime import (
    FormationTensorCache,
    TensorRuntimeDevice,
    TensorRuntimeError,
    resolve_tensor_runtime,
    runtime_roofline_record,
    validate_tensor_runtime_device,
)
from leibniz.training_runs import TrainingHistoryPoint, TrainingProtocol, TrainingRunRecord

__all__ = [
    "BenchmarkRunnerError",
    "BenchmarkRunPlan",
    "BenchmarkRunSummary",
    "run_benchmark",
]

_document_suffix = document_filename_suffix()
_progress_format = "leibniz.benchmark-training-progress"
_progress_format_version = 1
_default_sample_count = 512
_default_train_steps = 50_000
_default_validation_interval = 250
_default_convergence_patience = 12
_default_convergence_min_delta = 1e-3
_default_convergence_min_steps = 500


@dataclass(frozen=True, slots=True)
class _TrainingResult:
    probabilities: tuple[tuple[float, ...], ...]
    training_run: TrainingRunRecord
    throughput: Mapping[str, object]


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


@dataclass(frozen=True, slots=True)
class _PhaseWorkEstimate:
    flops_per_sample: float
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

    def learning_rates(self) -> tuple[float, ...]:
        return tuple(float(group["lr"]) for group in self.optimizer.param_groups)

    def step_after_optimizer(self) -> None:
        if self.update_on == "optimizer-step":
            self.scheduler.step()

    def step_after_validation(self, validation_loss: float) -> None:
        if self.update_on == "validation-loss":
            self.scheduler.step(validation_loss)


class BenchmarkRunnerError(ValueError):
    """Raised when a local benchmark run cannot be planned or executed."""


@dataclass(frozen=True, slots=True)
class BenchmarkRunPlan:
    """A local benchmark run plan resolved from CLI or workflow inputs."""

    architecture_path: Path
    benchmark_root: Path
    runs_root: Path = Path(".runs")
    scale: int = 1
    sample_count: int = _default_sample_count
    evaluation_sample_count: int | None = None
    seed: int = 101
    train_steps: int = _default_train_steps
    learning_rate: float = 0.01
    optimizer: str = "sgd"
    schedule: str = "none"
    validation_interval: int = _default_validation_interval
    convergence_patience: int = _default_convergence_patience
    convergence_min_delta: float = _default_convergence_min_delta
    convergence_min_steps: int = _default_convergence_min_steps
    target_validation_loss: float | None = None
    tensor_device: TensorRuntimeDevice = "auto"
    dry_run: bool = False

    def __post_init__(self) -> None:
        if type(self.scale) is not int or self.scale < 1:
            raise BenchmarkRunnerError("scale must be a positive integer")
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
        if type(self.train_steps) is not int or self.train_steps < 0:
            raise BenchmarkRunnerError("train_steps must be a nonnegative integer")
        if self.learning_rate <= 0:
            raise BenchmarkRunnerError("learning_rate must be positive")
        if self.optimizer not in {"sgd", "adam", "adamw"}:
            raise BenchmarkRunnerError(f"unsupported optimizer: {self.optimizer}")
        if self.schedule not in {"none", "cosine", "reduce-on-plateau"}:
            raise BenchmarkRunnerError(f"unsupported schedule: {self.schedule}")
        if type(self.validation_interval) is not int or self.validation_interval < 1:
            raise BenchmarkRunnerError("validation_interval must be a positive integer")
        if type(self.convergence_patience) is not int or self.convergence_patience < 0:
            raise BenchmarkRunnerError("convergence_patience must be nonnegative")
        if self.convergence_min_delta < 0:
            raise BenchmarkRunnerError("convergence_min_delta must be nonnegative")
        if type(self.convergence_min_steps) is not int or self.convergence_min_steps < 0:
            raise BenchmarkRunnerError("convergence_min_steps must be nonnegative")
        if self.target_validation_loss is not None and self.target_validation_loss < 0:
            raise BenchmarkRunnerError("target_validation_loss must be nonnegative")
        try:
            validate_tensor_runtime_device(self.tensor_device)
        except TensorRuntimeError as error:
            raise BenchmarkRunnerError(str(error)) from error

    @property
    def run_slug(self) -> str:
        """Return the deterministic local run suffix."""

        base = (
            f"l{self.scale}-seed{self.seed}-samples{self.sample_count}"
            f"-steps{self.train_steps}"
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
            "validation_interval": self.validation_interval,
            "convergence_patience": self.convergence_patience,
            "convergence_min_delta": float(self.convergence_min_delta),
            "convergence_min_steps": self.convergence_min_steps,
            "target_validation_loss": (
                None if self.target_validation_loss is None else float(self.target_validation_loss)
            ),
            "tensor_device": self.tensor_device,
        }
        return f"train-{ContentDigest.from_value(controls).hex[:12]}"

    @property
    def resolved_evaluation_sample_count(self) -> int:
        """Return the explicit evaluation sample count for this run."""

        if self.evaluation_sample_count is None:
            return self.sample_count
        return self.evaluation_sample_count


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
    evaluation_batch = generator.sample_batch(
        scale=plan.scale,
        sample_count=plan.resolved_evaluation_sample_count,
        seed=plan.seed,
    )
    outcome_space = generator.benchmark_manifest.resolve_outcome_space(scale=plan.scale)
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
    ) -> None:
        _write_document_atomic(
            progress_path,
            _training_progress_record(
                plan=plan,
                summary=summary,
                architecture=architecture,
                outcome_space=outcome_space,
                training_run=training_run,
                throughput=throughput,
            ),
        )
        if progress_callback is not None:
            progress_callback(summary)

    training_result = _train_and_predict(
        architecture=architecture,
        evaluation_batch=evaluation_batch,
        generator=generator,
        outcome_space=outcome_space,
        scale=plan.scale,
        sample_count=plan.sample_count,
        train_steps=plan.train_steps,
        learning_rate=float(plan.learning_rate),
        optimizer_name=plan.optimizer,
        schedule_name=plan.schedule,
        validation_interval=plan.validation_interval,
        convergence_patience=plan.convergence_patience,
        convergence_min_delta=float(plan.convergence_min_delta),
        convergence_min_steps=plan.convergence_min_steps,
        target_validation_loss=plan.target_validation_loss,
        tensor_device=plan.tensor_device,
        work_estimates=_training_work_estimates(
            architecture=architecture,
            inference_flops=model_inspection.cost_summary.inference_flops,
            parameter_bytes=model_inspection.cost_summary.parameter_bytes,
            batch_size=plan.sample_count,
        ),
        seed=plan.seed,
        progress_callback=publish_progress,
    )
    measurements = _measurements_for_predictions(
        generator=generator,
        batch=evaluation_batch,
        outcome_space=outcome_space,
        probabilities=training_result.probabilities,
        run_slug=summary.run_slug,
    )
    dataset = MeasurementDataset(measurements=measurements)
    dataset.validate_manifest(generator.benchmark_manifest, scale=plan.scale)
    _write_document(summary.measurement_dataset_path, dataset.to_record())
    _write_document(summary.model_inspection_path, model_inspection.to_record())
    _write_document(
        summary.training_summary_path,
        {
            **summary.to_record(),
            "dry_run": False,
            "scale": plan.scale,
            "sample_count": plan.sample_count,
            "evaluation_sample_count": plan.resolved_evaluation_sample_count,
            "seed": plan.seed,
            "train_steps": plan.train_steps,
            "learning_rate": float(plan.learning_rate),
            "optimizer": plan.optimizer,
            "schedule": plan.schedule,
            "validation_interval": plan.validation_interval,
            "convergence_patience": plan.convergence_patience,
            "convergence_min_delta": float(plan.convergence_min_delta),
            "convergence_min_steps": plan.convergence_min_steps,
            "tensor_runtime": "pytorch",
            "tensor_device": training_result.training_run.protocol.tensor_device,
            "training_run": training_result.training_run.to_record(),
            "throughput": training_result.throughput,
            "sampled_competence": _sampled_competence_record(
                generator=generator,
                batch=evaluation_batch,
                measurements=measurements,
            ),
            "architecture": model_inspection.architecture.to_record(),
            "cost_summary": model_inspection.cost_summary.to_record(),
            "measurement_dataset_digest": str(dataset.digest),
            "model_inspection_digest": str(model_inspection.digest),
        },
    )
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
        measurement_dataset_path=(
            plan.runs_root / "measurements" / benchmark_atom / f"{run_slug}{_document_suffix}"
        ),
        model_inspection_path=(
            plan.runs_root
            / "model-inspections"
            / benchmark_atom
            / f"{run_slug}{_document_suffix}"
        ),
        training_summary_path=(
            plan.runs_root / "training" / benchmark_atom / f"{run_slug}{_document_suffix}"
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


def _validate_architecture_for_batch(
    *,
    architecture: ArchitectureManifest,
    batch: GeneratedObservationBatch,
    outcome_space: OutcomeSpace,
) -> None:
    sample_shape = batch.samples[0].field.shape
    if architecture.input_shape != sample_shape:
        raise BenchmarkRunnerError(
            f"architecture input_shape {architecture.input_shape} does not match "
            f"generated observation shape {sample_shape}"
        )
    outcome_count = len(outcome_space.outcomes)
    if architecture.output_shape != (outcome_count,):
        raise BenchmarkRunnerError(
            f"architecture output_shape {architecture.output_shape} does not match "
            f"{outcome_count} resolved benchmark outcomes"
        )


def _train_and_predict(
    *,
    architecture: ArchitectureManifest,
    evaluation_batch: GeneratedObservationBatch,
    generator: ObservationGenerator,
    outcome_space: OutcomeSpace,
    scale: int,
    sample_count: int,
    train_steps: int,
    learning_rate: float,
    optimizer_name: str,
    schedule_name: str,
    validation_interval: int,
    convergence_patience: int,
    convergence_min_delta: float,
    convergence_min_steps: int,
    target_validation_loss: float | None,
    tensor_device: TensorRuntimeDevice,
    work_estimates: _TrainingWorkEstimates | None,
    seed: int,
    progress_callback: Callable[[TrainingRunRecord, Mapping[str, object]], None] | None = None,
) -> _TrainingResult:
    try:
        runtime = resolve_tensor_runtime(tensor_device)
    except TensorRuntimeError as error:
        raise BenchmarkRunnerError(str(error)) from error
    torch = runtime.torch
    torch.manual_seed(seed)
    module = ExecutableModelOperator(architecture).torch_module().to(runtime.device)
    outcome_ids = tuple(outcome.id for outcome in outcome_space.outcomes)
    formation_cache = FormationTensorCache(runtime=runtime, formation=generator.formation)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = _make_optimizer(
        torch=torch,
        parameters=module.parameters(),
        name=optimizer_name,
        learning_rate=learning_rate,
    )
    scheduler = _make_scheduler(
        torch=torch,
        optimizer=optimizer,
        name=schedule_name,
        max_steps=train_steps,
        min_delta=convergence_min_delta,
    )
    training_counter = _ThroughputCounter()
    validation_counter = _ThroughputCounter()
    evaluation_counter = _ThroughputCounter()

    def batch_for_seed(batch_seed: int) -> tuple[Any, Any]:
        generated = generator.sample_formation_batch(
            scale=scale,
            sample_count=sample_count,
            seed=batch_seed,
        )
        return formation_cache.batch_tensors(batch=generated, outcome_ids=outcome_ids)

    validation_history = _train_until_convergence(
        torch=torch,
        module=module,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        train_batch=lambda step: batch_for_seed(seed + step),
        validation_batch=lambda check: batch_for_seed(seed + 1_000_003 + check),
        max_steps=train_steps,
        validation_interval=validation_interval,
        patience=convergence_patience,
        min_delta=convergence_min_delta,
        min_steps=convergence_min_steps,
        target_validation_loss=target_validation_loss,
        batch_size=sample_count,
        training_counter=training_counter,
        validation_counter=validation_counter,
        on_validation=lambda history: (
            progress_callback(
                _running_training_run_record(
                    seed=seed,
                    batch_size=sample_count,
                    max_steps=train_steps,
                    learning_rate=float(learning_rate),
                    optimizer_name=optimizer_name,
                    schedule_name=schedule_name,
                    validation_interval=validation_interval,
                    convergence_patience=convergence_patience,
                    convergence_min_delta=convergence_min_delta,
                    convergence_min_steps=convergence_min_steps,
                    tensor_device=runtime.device_kind,
                    validation_history=tuple(history),
                ),
                _throughput_record(
                    runtime_device=runtime.device_kind,
                    training_counter=training_counter,
                    validation_counter=validation_counter,
                    evaluation_counter=evaluation_counter,
                    roofline=runtime_roofline_record(runtime),
                    work_estimates=work_estimates,
                ),
            )
            if progress_callback is not None
            else None
        ),
    )
    evaluation_started = time.perf_counter()
    eval_fields, _eval_labels = _batch_tensors(
        torch=torch,
        batch=evaluation_batch,
        outcome_ids=outcome_ids,
        device=runtime.device,
    )
    module.eval()
    with torch.no_grad():
        predictions = torch.softmax(module(eval_fields), dim=1).tolist()
    evaluation_counter.add(
        seconds=time.perf_counter() - evaluation_started,
        samples=len(evaluation_batch.samples),
    )
    training_run = _training_run_record(
        seed=seed,
        batch_size=sample_count,
        max_steps=train_steps,
        learning_rate=float(learning_rate),
        optimizer_name=optimizer_name,
        schedule_name=schedule_name,
        validation_interval=validation_interval,
        convergence_patience=convergence_patience,
        convergence_min_delta=convergence_min_delta,
        convergence_min_steps=convergence_min_steps,
        tensor_device=runtime.device_kind,
        validation_history=tuple(validation_history),
    )
    return _TrainingResult(
        probabilities=tuple(_renormalized_probabilities(row) for row in predictions),
        training_run=training_run,
        throughput=_throughput_record(
            runtime_device=runtime.device_kind,
            training_counter=training_counter,
            validation_counter=validation_counter,
            evaluation_counter=evaluation_counter,
            roofline=runtime_roofline_record(runtime),
            work_estimates=work_estimates,
        ),
    )


def _train_until_convergence(
    *,
    torch: Any,
    module: Any,
    optimizer: Any,
    scheduler: _LearningRateSchedule | None,
    loss_function: Any,
    train_batch: Callable[[int], tuple[Any, Any]],
    validation_batch: Callable[[int], tuple[Any, Any]],
    max_steps: int,
    validation_interval: int,
    patience: int,
    min_delta: float,
    min_steps: int,
    target_validation_loss: float | None,
    batch_size: int,
    training_counter: _ThroughputCounter,
    validation_counter: _ThroughputCounter,
    on_validation: Callable[[tuple[TrainingHistoryPoint, ...]], None] | None = None,
) -> list[TrainingHistoryPoint]:
    validation_history: list[TrainingHistoryPoint] = []
    best_loss = float("inf")
    best_step = 0
    best_check = 0
    stale_checks = 0

    def append_validation(*, step: int, check: int) -> None:
        nonlocal best_loss, best_step, best_check, stale_checks
        validation_started = time.perf_counter()
        fields, labels = validation_batch(check)
        validation_loss = _validation_loss(
            torch=torch,
            module=module,
            fields=fields,
            labels=labels,
            loss_function=loss_function,
        )
        if validation_loss < best_loss - min_delta:
            best_loss = validation_loss
            best_step = step
            best_check = check
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
            samples=batch_size,
        )
        validation_history.append(
            TrainingHistoryPoint(
                step=step,
                validation_check=check,
                validation_loss=validation_loss,
                best_validation_loss=best_loss,
                best_validation_step=best_step,
                best_validation_check=best_check,
                stale_checks=stale_checks,
                learning_rates=learning_rates,
            )
        )
        if on_validation is not None:
            on_validation(tuple(validation_history))

    append_validation(step=0, check=0)
    validation_check = 1
    for step in range(1, max_steps + 1):
        training_started = time.perf_counter()
        fields, labels = train_batch(step)
        module.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(module(fields), labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step_after_optimizer()
        training_counter.add(
            seconds=time.perf_counter() - training_started,
            samples=batch_size,
        )
        if step % validation_interval != 0 and step != max_steps:
            continue
        append_validation(step=step, check=validation_check)
        validation_check += 1
        if (
            target_validation_loss is not None
            and step >= min_steps
            and best_loss <= target_validation_loss
        ):
            break
        if patience > 0 and step >= min_steps and stale_checks >= patience:
            break
    return validation_history


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


def _training_run_record(
    *,
    seed: int,
    batch_size: int,
    max_steps: int,
    learning_rate: float,
    optimizer_name: str,
    schedule_name: str,
    validation_interval: int,
    convergence_patience: int,
    convergence_min_delta: float,
    convergence_min_steps: int,
    tensor_device: str,
    validation_history: tuple[TrainingHistoryPoint, ...],
) -> TrainingRunRecord:
    best = validation_history[-1]
    last_step = validation_history[-1].step
    last_stale_checks = validation_history[-1].stale_checks
    if max_steps == 0:
        stop_reason = "no-training-steps"
        status = "completed"
    elif convergence_patience > 0 and last_stale_checks >= convergence_patience:
        stop_reason = "validation-plateau"
        status = "converged"
    elif last_step < max_steps:
        stop_reason = "target-validation-loss"
        status = "converged"
    else:
        stop_reason = "max-steps"
        status = "budget-exhausted"
    return TrainingRunRecord(
        status=status,
        stop_reason=stop_reason,
        steps_run=last_step,
        validation_checks=len(validation_history),
        best_validation_loss=best.best_validation_loss,
        best_validation_step=best.best_validation_step,
        best_validation_check=best.best_validation_check,
        protocol=TrainingProtocol(
            kind="fixed-step-local-batch",
            objective="cross-entropy",
            optimizer=cast(Any, optimizer_name),
            learning_rate=learning_rate,
            schedule=cast(Any, schedule_name),
            seed=seed,
            batch_size=batch_size,
            max_steps=max_steps,
            validation_interval=validation_interval,
            validation_sample_count=batch_size,
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
    max_steps: int,
    learning_rate: float,
    optimizer_name: str,
    schedule_name: str,
    validation_interval: int,
    convergence_patience: int,
    convergence_min_delta: float,
    convergence_min_steps: int,
    tensor_device: str,
    validation_history: tuple[TrainingHistoryPoint, ...],
) -> TrainingRunRecord:
    best = validation_history[-1]
    return TrainingRunRecord(
        status="running",
        stop_reason="validation-checkpoint",
        steps_run=validation_history[-1].step,
        validation_checks=len(validation_history),
        best_validation_loss=best.best_validation_loss,
        best_validation_step=best.best_validation_step,
        best_validation_check=best.best_validation_check,
        protocol=TrainingProtocol(
            kind="fixed-step-local-batch",
            objective="cross-entropy",
            optimizer=cast(Any, optimizer_name),
            learning_rate=learning_rate,
            schedule=cast(Any, schedule_name),
            seed=seed,
            batch_size=batch_size,
            max_steps=max_steps,
            validation_interval=validation_interval,
            validation_sample_count=batch_size,
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
    return {
        "format": _progress_format,
        "format_version": _progress_format_version,
        "run_slug": summary.run_slug,
        "run_status": "running",
        "benchmark_id": str(summary.benchmark_id),
        "architecture_path": summary.architecture_path.as_posix(),
        "scale": plan.scale,
        "sample_count": plan.sample_count,
        "evaluation_sample_count": plan.resolved_evaluation_sample_count,
        "seed": plan.seed,
        "train_steps": plan.train_steps,
        "learning_rate": float(plan.learning_rate),
        "optimizer": plan.optimizer,
        "schedule": plan.schedule,
        "validation_interval": plan.validation_interval,
        "convergence_patience": plan.convergence_patience,
        "convergence_min_delta": float(plan.convergence_min_delta),
        "convergence_min_steps": plan.convergence_min_steps,
        "tensor_runtime": "pytorch",
        "tensor_device": training_run.protocol.tensor_device,
        "target_validation_loss": (
            None
            if plan.target_validation_loss is None
            else float(plan.target_validation_loss)
        ),
        "training_run": training_run.to_record(),
        "throughput": throughput,
        "architecture": inspection.architecture.to_record(),
        "cost_summary": inspection.cost_summary.to_record(),
        "model_inspection": inspection.to_record(),
        "architecture_digest": str(architecture.digest),
        "model_inspection_digest": str(inspection.digest),
        "provisional_score": _validation_competence(
            best_validation_loss=training_run.best_validation_loss,
            outcome_count=len(outcome_space.outcomes),
        ),
    }


def _validation_competence(*, best_validation_loss: float, outcome_count: int) -> float:
    reference_cross_entropy = math.log(outcome_count)
    if reference_cross_entropy <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - best_validation_loss / reference_cross_entropy))


def _throughput_record(
    *,
    runtime_device: str,
    training_counter: _ThroughputCounter,
    validation_counter: _ThroughputCounter,
    evaluation_counter: _ThroughputCounter,
    roofline: Mapping[str, object],
    work_estimates: _TrainingWorkEstimates | None,
) -> dict[str, object]:
    training = training_counter.to_record(kind="training-throughput")
    validation = validation_counter.to_record(kind="validation-throughput")
    evaluation = evaluation_counter.to_record(kind="evaluation-throughput")
    return {
        "kind": "benchmark-throughput",
        "tensor_runtime": "pytorch",
        "tensor_device": runtime_device,
        "training": training,
        "validation": validation,
        "evaluation": evaluation,
        "roofline": dict(roofline),
        "roofline_comparison": _roofline_comparison(
            training=training,
            validation=validation,
            evaluation=evaluation,
            roofline=roofline,
            work_estimates=work_estimates,
        ),
    }


def _roofline_comparison(
    *,
    training: Mapping[str, object],
    validation: Mapping[str, object],
    evaluation: Mapping[str, object],
    roofline: Mapping[str, object],
    work_estimates: _TrainingWorkEstimates | None,
) -> dict[str, object]:
    peak_flops = roofline.get("peak_flops_per_second")
    peak_bytes = roofline.get("peak_bytes_per_second")
    if (
        not isinstance(peak_flops, int | float)
        or not math.isfinite(float(peak_flops))
        or peak_flops <= 0
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
    peak_flops_value = float(peak_flops)
    peak_bytes_value = float(peak_bytes)
    training_phase = _phase_roofline_record(
        throughput=training,
        work=work_estimates.training,
        peak_flops_per_second=peak_flops_value,
        peak_bytes_per_second=peak_bytes_value,
    )
    validation_phase = _phase_roofline_record(
        throughput=validation,
        work=work_estimates.validation,
        peak_flops_per_second=peak_flops_value,
        peak_bytes_per_second=peak_bytes_value,
    )
    evaluation_phase = _phase_roofline_record(
        throughput=evaluation,
        work=work_estimates.evaluation,
        peak_flops_per_second=peak_flops_value,
        peak_bytes_per_second=peak_bytes_value,
    )
    return {
        "status": "available",
        "model": "operational-intensity",
        "peak_flops_per_second": peak_flops_value,
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
    peak_flops_per_second: float,
    peak_bytes_per_second: float,
) -> dict[str, object]:
    arithmetic_intensity = work.flops_per_sample / work.bytes_per_sample
    expected_flops = min(
        peak_flops_per_second,
        peak_bytes_per_second * arithmetic_intensity,
    )
    measured = throughput.get("samples_per_second")
    observed_flops = 0.0
    if not isinstance(measured, int | float) or not math.isfinite(float(measured)):
        measured_samples = 0.0
    else:
        measured_samples = float(measured)
        observed_flops = measured_samples * work.flops_per_sample
    return {
        "flops_per_sample": work.flops_per_sample,
        "bytes_per_sample": work.bytes_per_sample,
        "arithmetic_intensity_flops_per_byte": arithmetic_intensity,
        "expected_roofline_flops_per_second": expected_flops,
        "observed_flops_per_second": observed_flops,
        "fraction_of_roofline": (
            observed_flops / expected_flops if expected_flops > 0 else 0.0
        ),
        "samples_per_second": measured_samples,
        "limiting_resource": (
            "memory-bandwidth"
            if peak_bytes_per_second * arithmetic_intensity < peak_flops_per_second
            else "compute"
        ),
    }


def _training_work_estimates(
    *,
    architecture: ArchitectureManifest,
    inference_flops: int | None,
    parameter_bytes: int | None,
    batch_size: int,
) -> _TrainingWorkEstimates | None:
    if inference_flops is None or inference_flops <= 0:
        return None
    input_bytes = _shape_bytes(architecture.input_shape)
    output_bytes = _shape_bytes(architecture.output_shape)
    batch_size_value = max(1, batch_size)
    parameter_bytes_per_sample = float(parameter_bytes or 0) / batch_size_value
    inference_bytes = input_bytes + output_bytes + parameter_bytes_per_sample
    formation_bytes = 8.0 * input_bytes
    training_flops = 3.0 * float(inference_flops)
    training_bytes = formation_bytes + 3.0 * inference_bytes + 4.0 * parameter_bytes_per_sample
    validation_bytes = formation_bytes + inference_bytes
    evaluation_bytes = input_bytes + inference_bytes
    return _TrainingWorkEstimates(
        training=_PhaseWorkEstimate(
            flops_per_sample=training_flops,
            bytes_per_sample=training_bytes,
        ),
        validation=_PhaseWorkEstimate(
            flops_per_sample=float(inference_flops),
            bytes_per_sample=validation_bytes,
        ),
        evaluation=_PhaseWorkEstimate(
            flops_per_sample=float(inference_flops),
            bytes_per_sample=evaluation_bytes,
        ),
        assumptions=(
            "float32 tensor elements are four bytes",
            "training FLOPs are approximated as three times declared inference FLOPs",
            "parameter bytes are amortized across the local batch",
            "formation bytes are approximated as eight input fields per sample",
            "optimizer and gradient traffic are approximated from parameter bytes",
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
    max_steps: int,
    min_delta: float,
) -> _LearningRateSchedule | None:
    if name == "none":
        return None
    if name == "cosine":
        return _LearningRateSchedule(
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, max_steps),
            ),
            optimizer=optimizer,
            update_on="optimizer-step",
        )
    if name == "reduce-on-plateau":
        return _LearningRateSchedule(
            scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                threshold=min_delta,
                threshold_mode="abs",
            ),
            optimizer=optimizer,
            update_on="validation-loss",
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


def _measurements_for_predictions(
    *,
    generator: ObservationGenerator,
    batch: GeneratedObservationBatch,
    outcome_space: OutcomeSpace,
    probabilities: tuple[tuple[float, ...], ...],
    run_slug: str,
) -> tuple[MeasurementRecord, ...]:
    outcome_ids = tuple(outcome.id for outcome in outcome_space.outcomes)
    measurements: list[MeasurementRecord] = []
    for sample, sample_probabilities in zip(batch.samples, probabilities, strict=True):
        accepted_event = AcceptedEvent.from_record(
            {
                "id": str(_sample_identifier("events", run_slug, sample)),
                "outcome_space_id": str(outcome_space.id),
                "outcomes": [sample.outcome_id],
            },
            outcome_space=outcome_space,
        )
        probability_measure = FiniteProbabilityMeasure(
            id=_sample_identifier("measures", run_slug, sample),
            outcome_space_id=outcome_space.id,
            probabilities=tuple(
                ProbabilityMass(outcome_id, probability)
                for outcome_id, probability in zip(
                    outcome_ids,
                    sample_probabilities,
                    strict=True,
                )
                if probability > 0
            ),
        )
        measurements.append(
            MeasurementRecord(
                benchmark_id=generator.benchmark_manifest.id,
                outcome_space=outcome_space,
                accepted_event=accepted_event,
                probability_measure=probability_measure,
                raw_scoring_evidence=RawScoringEvidence.from_event_and_measure(
                    id=_sample_identifier("evidence", run_slug, sample),
                    observation_id=str(sample.observation.id),
                    event=accepted_event,
                    measure=probability_measure,
                ),
                evidence_artifacts=(
                    sample.observation.formation_declaration,
                    sample.observation.materialization_plan,
                    ArtifactReference(
                        kind="formed-observation",
                        protocol_id=sample.observation.id,
                        record_digest=sample.observation.digest,
                    ),
                ),
            )
        )
    return tuple(measurements)


def _sampled_competence_record(
    *,
    generator: ObservationGenerator,
    batch: GeneratedObservationBatch,
    measurements: tuple[MeasurementRecord, ...],
) -> dict[str, object]:
    if len(batch.samples) != len(measurements):
        raise BenchmarkRunnerError("sampled competence requires one measurement per sample")
    complexities = {sample.complexity for sample in batch.samples}
    if len(complexities) != 1:
        raise BenchmarkRunnerError("sampled competence requires one complexity class")
    accepted_mass = tuple(
        measurement.raw_scoring_evidence.accepted_mass for measurement in measurements
    )
    finite_losses = tuple(
        measurement.raw_scoring_evidence.negative_log_score
        for measurement in measurements
        if math.isfinite(measurement.raw_scoring_evidence.negative_log_score)
    )
    mean_negative_log_score: float | str
    if len(finite_losses) != len(measurements):
        mean_negative_log_score = "infinity"
    else:
        mean_negative_log_score = math.fsum(finite_losses) / len(finite_losses)
    return {
        "kind": "sampled-complexity-class",
        "sampling_rule": "generator-uniform-component-sequence-v1",
        "difficulty_assumption": "approximately-uniform-within-complexity-class",
        "benchmark_id": str(generator.benchmark_manifest.id),
        "scale": batch.scale,
        "complexity_axis": generator.benchmark_manifest.complexity_coordinate,
        "complexity": next(iter(complexities)),
        "seed": batch.seed,
        "sample_count": len(batch.samples),
        "mean_accepted_mass": math.fsum(accepted_mass) / len(accepted_mass),
        "mean_negative_log_score": mean_negative_log_score,
        "observation_ids": [str(sample.observation.id) for sample in batch.samples],
        "measurement_ids": [
            str(measurement.raw_scoring_evidence.id) for measurement in measurements
        ],
    }


def _sample_identifier(
    family: str,
    run_slug: str,
    sample: GeneratedObservationSample,
) -> ProtocolIdentifier:
    return _child_identifier(
        sample.observation.benchmark_id,
        f"{family}.{run_slug}.sample-{sample.index}",
    )


def _renormalized_probabilities(probabilities: Sequence[float]) -> tuple[float, ...]:
    total = sum(float(probability) for probability in probabilities)
    if total <= 0:
        raise BenchmarkRunnerError("model probabilities must contain positive mass")
    normalized = [max(0.0, float(probability) / total) for probability in probabilities]
    if len(normalized) == 1:
        return (1.0,)
    normalized[-1] = max(0.0, 1.0 - sum(normalized[:-1]))
    return tuple(normalized)


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


def _child_identifier(parent: ProtocolIdentifier, suffix: str) -> ProtocolIdentifier:
    return ProtocolIdentifier.parse(f"{parent.name}.{suffix}@{parent.version}")
