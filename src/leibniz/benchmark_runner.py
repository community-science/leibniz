"""Small benchmark execution workflows for local operator runs."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
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

__all__ = [
    "BenchmarkRunnerError",
    "BenchmarkRunPlan",
    "BenchmarkRunSummary",
    "run_benchmark",
]

_document_suffix = document_filename_suffix()


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

    probabilities = _train_and_predict(
        architecture=architecture,
        batch=batch,
        outcome_space=outcome_space,
        train_steps=plan.train_steps,
        learning_rate=float(plan.learning_rate),
        seed=plan.seed,
    )
    measurements = _measurements_for_predictions(
        generator=generator,
        batch=batch,
        outcome_space=outcome_space,
        probabilities=probabilities,
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
    batch: GeneratedObservationBatch,
    outcome_space: OutcomeSpace,
    train_steps: int,
    learning_rate: float,
    seed: int,
) -> tuple[tuple[float, ...], ...]:
    torch = _torch()
    torch.manual_seed(seed)
    module = ExecutableModelOperator(architecture).torch_module()
    fields = _batch_tensor(torch=torch, batch=batch)
    outcome_ids = tuple(outcome.id for outcome in outcome_space.outcomes)
    labels = torch.tensor(
        [outcome_ids.index(sample.outcome_id) for sample in batch.samples],
        dtype=torch.long,
    )
    if train_steps:
        optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)
        loss_function = torch.nn.CrossEntropyLoss()
        module.train()
        for _step in range(train_steps):
            optimizer.zero_grad()
            loss = loss_function(module(fields), labels)
            loss.backward()
            optimizer.step()
    module.eval()
    with torch.no_grad():
        predictions = torch.softmax(module(fields), dim=1).tolist()
    return tuple(_renormalized_probabilities(row) for row in predictions)


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
