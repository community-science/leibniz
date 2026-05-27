"""Small benchmark execution workflows for local operator runs."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.artifacts import ArtifactReference
from leibniz.documents import canonical_document_bytes, document_filename_suffix
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset, MeasurementRecord
from leibniz.model_inspection import ModelInspectionRecord
from leibniz.model_operators import ExecutableModelOperator, ModelOperatorExecutionError
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
from leibniz.training_runs import TrainingHistoryPoint, TrainingProtocol, TrainingRunRecord

__all__ = [
    "BenchmarkRunnerError",
    "BenchmarkRunPlan",
    "BenchmarkRunSummary",
    "run_benchmark",
]

_document_suffix = document_filename_suffix()


@dataclass(frozen=True, slots=True)
class _TrainingResult:
    probabilities: tuple[tuple[float, ...], ...]
    training_run: TrainingRunRecord


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
    sample_count: int = 4
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

    def __post_init__(self) -> None:
        if type(self.scale) is not int or self.scale < 1:
            raise BenchmarkRunnerError("scale must be a positive integer")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise BenchmarkRunnerError("sample_count must be a positive integer")
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
        if self.target_validation_loss is not None and self.target_validation_loss < 0:
            raise BenchmarkRunnerError("target_validation_loss must be nonnegative")

    @property
    def run_slug(self) -> str:
        """Return the deterministic local run suffix."""

        return f"l{self.scale}-seed{self.seed}-samples{self.sample_count}-steps{self.train_steps}"


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


def run_benchmark(plan: BenchmarkRunPlan) -> BenchmarkRunSummary:
    """Run or dry-run a tiny local benchmark workflow."""

    generator = load_observation_generator(plan.benchmark_root)
    architecture = ArchitectureManifestDocument.from_bytes(
        plan.architecture_path.read_bytes()
    ).manifest
    batch = generator.sample_batch(
        scale=plan.scale,
        sample_count=plan.sample_count,
        seed=plan.seed,
    )
    outcome_space = generator.benchmark_manifest.resolve_outcome_space(scale=plan.scale)
    _validate_architecture_for_batch(
        architecture=architecture,
        batch=batch,
        outcome_space=outcome_space,
    )

    summary = _run_summary(plan=plan, benchmark_id=generator.benchmark_manifest.id)
    if plan.dry_run:
        return summary

    training_result = _train_and_predict(
        architecture=architecture,
        evaluation_batch=batch,
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
        target_validation_loss=plan.target_validation_loss,
        seed=plan.seed,
    )
    measurements = _measurements_for_predictions(
        generator=generator,
        batch=batch,
        outcome_space=outcome_space,
        probabilities=training_result.probabilities,
        run_slug=summary.run_slug,
    )
    dataset = MeasurementDataset(measurements=measurements)
    dataset.validate_manifest(generator.benchmark_manifest, scale=plan.scale)
    model_inspection = ModelInspectionRecord.from_architecture(
        id=ProtocolIdentifier.parse(
            f"model-inspections.{_identifier_atom(generator.benchmark_manifest.id)}."
            f"{plan.run_slug}@0.1.0"
        ),
        architecture_manifest=architecture,
    )

    _write_document(summary.measurement_dataset_path, dataset.to_record())
    _write_document(summary.model_inspection_path, model_inspection.to_record())
    _write_document(
        summary.training_summary_path,
        {
            **summary.to_record(),
            "dry_run": False,
            "scale": plan.scale,
            "sample_count": plan.sample_count,
            "seed": plan.seed,
            "train_steps": plan.train_steps,
            "learning_rate": float(plan.learning_rate),
            "optimizer": plan.optimizer,
            "schedule": plan.schedule,
            "validation_interval": plan.validation_interval,
            "convergence_patience": plan.convergence_patience,
            "convergence_min_delta": float(plan.convergence_min_delta),
            "training_run": training_result.training_run.to_record(),
            "architecture": model_inspection.architecture.to_record(),
            "cost_summary": model_inspection.cost_summary.to_record(),
            "measurement_dataset_digest": str(dataset.digest),
            "model_inspection_digest": str(model_inspection.digest),
        },
    )
    return summary


def _run_summary(
    *,
    plan: BenchmarkRunPlan,
    benchmark_id: ProtocolIdentifier,
) -> BenchmarkRunSummary:
    benchmark_atom = _identifier_atom(benchmark_id)
    run_slug = f"{benchmark_atom}-{plan.run_slug}"
    return BenchmarkRunSummary(
        run_slug=run_slug,
        benchmark_id=benchmark_id,
        architecture_path=plan.architecture_path,
        measurement_count=plan.sample_count,
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
    target_validation_loss: float | None,
    seed: int,
) -> _TrainingResult:
    torch = _torch()
    torch.manual_seed(seed)
    module = ExecutableModelOperator(architecture).torch_module()
    outcome_ids = tuple(outcome.id for outcome in outcome_space.outcomes)
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

    def batch_for_seed(batch_seed: int) -> tuple[Any, Any]:
        generated = generator.sample_batch(
            scale=scale,
            sample_count=sample_count,
            seed=batch_seed,
        )
        return _batch_tensors(torch=torch, batch=generated, outcome_ids=outcome_ids)

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
        target_validation_loss=target_validation_loss,
    )
    eval_fields, _eval_labels = _batch_tensors(
        torch=torch,
        batch=evaluation_batch,
        outcome_ids=outcome_ids,
    )
    module.eval()
    with torch.no_grad():
        predictions = torch.softmax(module(eval_fields), dim=1).tolist()
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
        validation_history=tuple(validation_history),
    )
    return _TrainingResult(
        probabilities=tuple(_renormalized_probabilities(row) for row in predictions),
        training_run=training_run,
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
    target_validation_loss: float | None,
) -> list[TrainingHistoryPoint]:
    validation_history: list[TrainingHistoryPoint] = []
    best_loss = float("inf")
    best_step = 0
    best_check = 0
    stale_checks = 0

    def append_validation(*, step: int, check: int) -> None:
        nonlocal best_loss, best_step, best_check, stale_checks
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

    append_validation(step=0, check=0)
    validation_check = 1
    for step in range(1, max_steps + 1):
        fields, labels = train_batch(step)
        module.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(module(fields), labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step_after_optimizer()
        if step % validation_interval != 0 and step != max_steps:
            continue
        append_validation(step=step, check=validation_check)
        validation_check += 1
        if target_validation_loss is not None and best_loss <= target_validation_loss:
            break
        if patience > 0 and stale_checks >= patience:
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
        ),
        validation_history=validation_history,
    )


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
) -> tuple[Any, Any]:
    return (
        _batch_tensor(torch=torch, batch=batch),
        torch.tensor(
            [outcome_ids.index(sample.outcome_id) for sample in batch.samples],
            dtype=torch.long,
        ),
    )


def _batch_tensor(*, torch: Any, batch: GeneratedObservationBatch) -> Any:
    values = [
        list(sample.field.values)
        for sample in batch.samples
    ]
    fields = torch.tensor(values, dtype=torch.float32)
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


def _torch() -> Any:
    try:
        return cast(Any, importlib.import_module("torch"))
    except ImportError as error:
        raise ModelOperatorExecutionError(
            "PyTorch is required to run benchmark training"
        ) from error


def _identifier_atom(identifier: ProtocolIdentifier) -> str:
    return str(identifier.name).rsplit(".", maxsplit=1)[-1]


def _child_identifier(parent: ProtocolIdentifier, suffix: str) -> ProtocolIdentifier:
    return ProtocolIdentifier.parse(f"{parent.name}.{suffix}@{parent.version}")
