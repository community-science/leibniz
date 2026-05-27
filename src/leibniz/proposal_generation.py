"""Deterministic local proposal generation for benchmark experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from leibniz.architecture_candidates import (
    ArchitectureCandidate,
    default_architecture_candidate_space,
    generate_architecture_candidates,
)
from leibniz.architectures import ArchitectureManifest
from leibniz.candidate_observations import (
    ArchitectureCandidateObservation,
    ArchitectureMeasurementEvidence,
    project_architecture_candidate_observations,
)
from leibniz.content import ContentDigest
from leibniz.documents import (
    canonical_document_bytes,
    document_filename_suffix,
    load_object_document,
)
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import (
    MeasurementDataset,
    MeasurementDatasetDocument,
    MeasurementRecord,
)
from leibniz.model_inspection import ModelInspectionDocument
from leibniz.model_operators import summarize_architecture_operators
from leibniz.observation_generation import load_observation_generator
from leibniz.proposal_selection import CandidateProposalSelection, select_candidate_proposals
from leibniz.proposals import ExperimentProposal, ExperimentProposalSet
from leibniz.publications import SubmissionPublicationDocument

__all__ = [
    "ProposalGenerationError",
    "ProposalGenerationPlan",
    "ProposalGenerationSummary",
    "generate_experiment_proposals",
]

_document_suffix = document_filename_suffix()


class ProposalGenerationError(ValueError):
    """Raised when local proposal generation cannot produce candidates."""


@dataclass(frozen=True, slots=True)
class ProposalGenerationPlan:
    """Inputs for deterministic local proposal generation."""

    benchmark_root: Path
    runs_root: Path = Path(".runs")
    scale: int = 1
    candidate_budget: int = 3
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

    def __post_init__(self) -> None:
        if type(self.scale) is not int or self.scale < 1:
            raise ProposalGenerationError("scale must be a positive integer")
        if type(self.candidate_budget) is not int or self.candidate_budget < 1:
            raise ProposalGenerationError("candidate_budget must be positive")
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise ProposalGenerationError("sample_count must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ProposalGenerationError("seed must be nonnegative")
        if type(self.train_steps) is not int or self.train_steps < 0:
            raise ProposalGenerationError("train_steps must be nonnegative")
        if self.learning_rate <= 0:
            raise ProposalGenerationError("learning_rate must be positive")
        if self.optimizer not in {"sgd", "adam", "adamw"}:
            raise ProposalGenerationError(f"unsupported optimizer: {self.optimizer}")
        if self.schedule not in {"none", "cosine", "reduce-on-plateau"}:
            raise ProposalGenerationError(f"unsupported schedule: {self.schedule}")
        if type(self.validation_interval) is not int or self.validation_interval < 1:
            raise ProposalGenerationError("validation_interval must be positive")
        if type(self.convergence_patience) is not int or self.convergence_patience < 0:
            raise ProposalGenerationError("convergence_patience must be nonnegative")
        if self.convergence_min_delta < 0:
            raise ProposalGenerationError("convergence_min_delta must be nonnegative")
        if self.target_validation_loss is not None and self.target_validation_loss < 0:
            raise ProposalGenerationError("target_validation_loss must be nonnegative")


@dataclass(frozen=True, slots=True)
class ProposalGenerationSummary:
    """Summary of generated local proposal artifacts."""

    proposal_set_path: Path
    architecture_paths: tuple[Path, ...]
    benchmark_id: ProtocolIdentifier
    proposal_count: int

    def to_record(self) -> dict[str, object]:
        return {
            "proposal_set_path": self.proposal_set_path.as_posix(),
            "architecture_paths": [path.as_posix() for path in self.architecture_paths],
            "benchmark_id": str(self.benchmark_id),
            "proposal_count": self.proposal_count,
        }


@dataclass(frozen=True, slots=True)
class _MeasuredArchitecture:
    digest: ContentDigest
    score: float
    parameter_count: int


@dataclass(frozen=True, slots=True)
class _CandidateArchitecture:
    architecture: ArchitectureManifest
    search_rank: int
    parameter_count: int
    rationale: str
    selector_name: str
    source_candidate_rank: int
    comparable_cost_best_score: float
    resource_stratum_index: int | None
    resource_stratum_count: int | None
    capability_family_kind: str
    capability_operator_kinds: tuple[str, ...]
    predicted_score: float
    uncertainty: float
    acquisition_value: float
    novelty: float
    expected_frontier_improvement: float


def generate_experiment_proposals(plan: ProposalGenerationPlan) -> ProposalGenerationSummary:
    """Generate deterministic candidate architecture proposals for one benchmark."""

    generator = load_observation_generator(plan.benchmark_root)
    manifest = generator.benchmark_manifest
    outcome_space = manifest.resolve_outcome_space(scale=plan.scale)
    sample = generator.sample_batch(scale=plan.scale, sample_count=1, seed=plan.seed).samples[0]
    dataset = _measurement_dataset(plan.runs_root, benchmark_id=manifest.id)
    measured = _measured_architectures(plan.runs_root, benchmark_id=manifest.id)
    candidate_space = default_architecture_candidate_space()
    candidate_observations = project_architecture_candidate_observations(
        tuple(
            generate_architecture_candidates(
                candidate_space,
                input_shape=sample.field.shape,
                output_count=len(outcome_space.outcomes),
            )
        ),
        measured=tuple(
            ArchitectureMeasurementEvidence(
                architecture_digest=item.digest,
                score=item.score,
                parameter_count=item.parameter_count,
            )
            for item in measured
        ),
    )
    candidates = _candidate_architectures(
        candidate_observations=candidate_observations,
        input_shape=sample.field.shape,
        output_count=len(outcome_space.outcomes),
        candidate_budget=plan.candidate_budget,
    )
    if not candidates:
        raise ProposalGenerationError("no unmeasured candidate architectures are available")

    benchmark_atom = _identifier_atom(manifest.id)
    architecture_root = plan.runs_root / "proposals" / "architectures" / benchmark_atom
    proposal_root = plan.runs_root / "proposals" / benchmark_atom
    architecture_root.mkdir(parents=True, exist_ok=True)
    proposal_root.mkdir(parents=True, exist_ok=True)

    architecture_paths: list[Path] = []
    proposals: list[ExperimentProposal] = []
    for rank, candidate in enumerate(candidates, start=1):
        architecture_path = architecture_root / (
            f"{_identifier_atom(candidate.architecture.id)}{_document_suffix}"
        )
        architecture_path.write_bytes(
            canonical_document_bytes(candidate.architecture.to_record()) + b"\n"
        )
        architecture_paths.append(architecture_path)
        proposals.append(
            ExperimentProposal(
                id=_proposal_identifier(manifest.id, rank),
                rank=rank,
                candidate_kind="architecture",
                candidate_id=candidate.architecture.id,
                rationale=candidate.rationale,
                predicted_score=candidate.predicted_score,
                uncertainty=candidate.uncertainty,
                acquisition_value=candidate.acquisition_value,
                novelty=candidate.novelty,
                expected_frontier_improvement=candidate.expected_frontier_improvement,
                selector_name=candidate.selector_name,
                source_candidate_rank=candidate.source_candidate_rank,
                comparable_cost_best_score=candidate.comparable_cost_best_score,
                resource_stratum_index=candidate.resource_stratum_index,
                resource_stratum_count=candidate.resource_stratum_count,
                capability_family_kind=candidate.capability_family_kind,
                capability_operator_kinds=candidate.capability_operator_kinds,
                command=(
                    "leibniz",
                    "benchmark",
                    "run",
                    "--benchmark-root",
                    plan.benchmark_root.as_posix(),
                    "--architecture",
                    architecture_path.as_posix(),
                    "--runs-root",
                    plan.runs_root.as_posix(),
                    "--scale",
                    str(plan.scale),
                    "--sample-count",
                    str(plan.sample_count),
                    "--seed",
                    str(plan.seed),
                    "--train-steps",
                    str(plan.train_steps),
                    "--learning-rate",
                    str(float(plan.learning_rate)),
                    "--optimizer",
                    plan.optimizer,
                    "--schedule",
                    plan.schedule,
                    "--validation-interval",
                    str(plan.validation_interval),
                    "--convergence-patience",
                    str(plan.convergence_patience),
                    "--convergence-min-delta",
                    str(float(plan.convergence_min_delta)),
                    *(
                        ()
                        if plan.target_validation_loss is None
                        else (
                            "--target-validation-loss",
                            str(float(plan.target_validation_loss)),
                        )
                    ),
                ),
            )
        )

    proposal_set = ExperimentProposalSet(
        id=_proposal_set_identifier(manifest.id),
        source_dataset_digest=dataset.digest,
        proposals=tuple(proposals),
    )
    proposal_set.validate_sources(
        dataset=dataset,
        architectures=tuple(candidate.architecture for candidate in candidates),
    )
    proposal_set_path = proposal_root / ("proposal_set" + _document_suffix)
    proposal_set_path.write_bytes(canonical_document_bytes(proposal_set.to_record()) + b"\n")
    return ProposalGenerationSummary(
        proposal_set_path=proposal_set_path,
        architecture_paths=tuple(architecture_paths),
        benchmark_id=manifest.id,
        proposal_count=len(proposals),
    )


def _measurement_dataset(
    runs_root: Path,
    *,
    benchmark_id: ProtocolIdentifier,
) -> MeasurementDataset:
    records: dict[ProtocolIdentifier, Mapping[str, object]] = {}
    for dataset in _measurement_datasets(runs_root, benchmark_id=benchmark_id):
        for measurement in dataset.measurements:
            measurement_id = measurement.raw_scoring_evidence.id
            record = measurement.to_record()
            previous = records.get(measurement_id)
            if previous is not None and previous != record:
                raise ProposalGenerationError(f"conflicting measurement record: {measurement_id}")
            records[measurement_id] = record
    return MeasurementDataset(
        measurements=tuple(
            MeasurementRecord.from_record(record)
            for _measurement_id, record in sorted(records.items(), key=lambda item: str(item[0]))
        )
    )


def _measurement_datasets(
    runs_root: Path,
    *,
    benchmark_id: ProtocolIdentifier,
) -> tuple[MeasurementDataset, ...]:
    datasets: list[MeasurementDataset] = []
    measurement_root = runs_root / "measurements" / _identifier_atom(benchmark_id)
    if measurement_root.is_dir():
        for path in sorted(measurement_root.rglob("*" + _document_suffix)):
            datasets.append(MeasurementDatasetDocument.from_bytes(path.read_bytes()).dataset)
    import_root = runs_root / "imports" / "publication_bundles"
    if import_root.is_dir():
        for path in sorted(import_root.rglob("*" + _document_suffix)):
            bundle = SubmissionPublicationDocument.from_bytes(path.read_bytes()).bundle
            if bundle.submission_package.benchmark_manifest.id == benchmark_id:
                datasets.append(bundle.measurement_dataset)
    return tuple(datasets)


def _measured_architectures(
    runs_root: Path,
    *,
    benchmark_id: ProtocolIdentifier,
) -> tuple[_MeasuredArchitecture, ...]:
    measured: list[_MeasuredArchitecture] = []
    for summary in _training_summaries(runs_root, benchmark_id=benchmark_id):
        inspection_path = _summary_path(
            runs_root=runs_root,
            value=summary.get("model_inspection_path"),
            field="model_inspection_path",
        )
        dataset_path = _summary_path(
            runs_root=runs_root,
            value=summary.get("measurement_dataset_path"),
            field="measurement_dataset_path",
        )
        inspection = ModelInspectionDocument.from_bytes(inspection_path.read_bytes()).inspection
        dataset = MeasurementDatasetDocument.from_bytes(dataset_path.read_bytes()).dataset
        measured.append(
            _MeasuredArchitecture(
                digest=_architecture_digest(inspection.architecture.to_record()),
                score=_mean_score(dataset),
                parameter_count=_parameter_count(inspection.cost_summary.to_record()),
            )
        )

    import_root = runs_root / "imports" / "publication_bundles"
    if import_root.is_dir():
        for path in sorted(import_root.rglob("*" + _document_suffix)):
            bundle = SubmissionPublicationDocument.from_bytes(path.read_bytes()).bundle
            package = bundle.submission_package
            if package.benchmark_manifest.id != benchmark_id:
                continue
            plan = summarize_architecture_operators(package.architecture_manifest)
            if plan.parameter_count is None:
                continue
            measured.append(
                _MeasuredArchitecture(
                    digest=package.architecture_manifest.digest,
                    score=_mean_score(bundle.measurement_dataset),
                    parameter_count=plan.parameter_count,
                )
            )
    return tuple(measured)


def _training_summaries(
    runs_root: Path,
    *,
    benchmark_id: ProtocolIdentifier,
) -> tuple[Mapping[str, object], ...]:
    training_root = runs_root / "training" / _identifier_atom(benchmark_id)
    if not training_root.is_dir():
        return ()
    summaries: list[Mapping[str, object]] = []
    for path in sorted(training_root.rglob("*" + _document_suffix)):
        record = load_object_document(path.read_bytes(), description="training summary")
        if record.get("format") == "leibniz.benchmark-run":
            summaries.append(record)
    return tuple(summaries)


def _candidate_architectures(
    *,
    candidate_observations: tuple[ArchitectureCandidateObservation, ...],
    input_shape: tuple[int, ...],
    output_count: int,
    candidate_budget: int,
) -> tuple[_CandidateArchitecture, ...]:
    _require_candidate_shape(
        tuple(observation.candidate for observation in candidate_observations),
        input_shape=input_shape,
        output_count=output_count,
    )
    observed_parameters = tuple(
        observation.parameter_count
        for observation in candidate_observations
        if observation.is_measured
    )
    best_score = max(
        (observation.best_measured_score for observation in candidate_observations),
        default=0.0,
    )
    candidates: list[_CandidateArchitecture] = []
    selections = select_candidate_proposals(
        candidate_observations,
        budget=candidate_budget,
    )
    for selection in selections:
        observation = selection.observation
        candidate = observation.candidate
        architecture = candidate.architecture
        parameter_count = observation.parameter_count
        novelty = _novelty(parameter_count, observed_parameters)
        uncertainty = min(1.0, 0.2 + novelty / 2.0)
        predicted_score = min(1.0, max(best_score, 1.0 / output_count) + novelty * 0.05)
        frontier_improvement = max(
            0.0,
            predicted_score - observation.best_measured_score_at_or_below_cost,
        )
        acquisition_value = predicted_score + uncertainty * 0.25 + novelty * 0.1
        candidates.append(
            _CandidateArchitecture(
                architecture=architecture,
                search_rank=observation.source_candidate_rank,
                parameter_count=parameter_count,
                rationale=_selection_rationale(selection),
                selector_name=selection.selector_name,
                source_candidate_rank=selection.source_candidate_rank,
                comparable_cost_best_score=selection.comparable_cost_best_score,
                resource_stratum_index=selection.resource_stratum_index,
                resource_stratum_count=selection.resource_stratum_count,
                capability_family_kind=selection.capability_key.family_kind,
                capability_operator_kinds=selection.capability_key.operator_kinds,
                predicted_score=predicted_score,
                uncertainty=uncertainty,
                acquisition_value=acquisition_value,
                novelty=novelty,
                expected_frontier_improvement=frontier_improvement,
            )
        )
    return tuple(candidates)


def _selection_rationale(selection: CandidateProposalSelection) -> str:
    if (
        selection.resource_stratum_index is not None
        and selection.resource_stratum_count is not None
    ):
        return (
            f"{selection.selector_name} selected resource stratum "
            f"{selection.resource_stratum_index + 1}/{selection.resource_stratum_count}"
        )
    return (
        f"{selection.selector_name} selected capability "
        f"{selection.capability_key.family_kind}"
    )


def _require_candidate_shape(
    candidates: tuple[ArchitectureCandidate, ...],
    *,
    input_shape: tuple[int, ...],
    output_count: int,
) -> None:
    if not candidates:
        raise ProposalGenerationError("candidate space did not generate architectures")
    expected_output_shape = (output_count,)
    for candidate in candidates:
        if candidate.architecture.input_shape != input_shape:
            raise ProposalGenerationError(
                "candidate architecture input_shape does not match generated observations"
            )
        if candidate.architecture.output_shape != expected_output_shape:
            raise ProposalGenerationError(
                "candidate architecture output_shape does not match resolved outcomes"
            )


def _novelty(parameter_count: int, observed_parameters: tuple[int, ...]) -> float:
    if not observed_parameters:
        return 1.0
    distances = tuple(
        abs(math.log1p(parameter_count) - math.log1p(observed))
        for observed in observed_parameters
    )
    return min(1.0, min(distances))


def _mean_score(dataset: MeasurementDataset) -> float:
    if not dataset.measurements:
        return 0.0
    return sum(
        measurement.raw_scoring_evidence.accepted_mass
        for measurement in dataset.measurements
    ) / len(dataset.measurements)


def _architecture_digest(record: Mapping[str, object]) -> ContentDigest:
    value = record.get("record_digest")
    if not isinstance(value, str):
        raise ProposalGenerationError("architecture reference must include record_digest")
    algorithm, separator, digest_hex = value.partition(":")
    if separator == "":
        raise ProposalGenerationError("architecture record_digest must be algorithm:digest")
    return ContentDigest(algorithm=algorithm, hex=digest_hex)


def _parameter_count(record: Mapping[str, object]) -> int:
    value = record.get("parameter_count")
    if type(value) is not int or value < 0:
        raise ProposalGenerationError("cost summary must include parameter_count")
    return value


def _summary_path(*, runs_root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ProposalGenerationError(f"{field}: expected path string")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (runs_root.parent / path).resolve()
    if not resolved.is_relative_to(runs_root.resolve()):
        raise ProposalGenerationError(f"{field} must stay inside runs root")
    if not resolved.is_file():
        raise ProposalGenerationError(f"{field} does not exist: {path}")
    return resolved


def _proposal_set_identifier(benchmark_id: ProtocolIdentifier) -> ProtocolIdentifier:
    return ProtocolIdentifier.parse(
        f"experiment-proposal-sets.{benchmark_id.name}.active@{benchmark_id.version}"
    )


def _proposal_identifier(benchmark_id: ProtocolIdentifier, rank: int) -> ProtocolIdentifier:
    return ProtocolIdentifier.parse(
        f"experiment-proposals.{benchmark_id.name}.rank-{rank}@{benchmark_id.version}"
    )


def _identifier_atom(identifier: ProtocolIdentifier) -> str:
    return str(identifier.name).rsplit(".", maxsplit=1)[-1]
