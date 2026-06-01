"""Deterministic local proposal generation for benchmark experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from leibniz.acquisition import score_candidate_acquisition
from leibniz.architecture_candidates import (
    ArchitectureCandidate,
    ArchitectureSearchDistribution,
    default_architecture_search_distribution,
    sample_architecture_candidates,
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
from leibniz.tensor_runtime import (
    TensorRuntimeDevice,
    TensorRuntimeError,
    architecture_supported_by_tensor_runtime,
    preferred_tensor_runtime_device_kind,
    validate_tensor_runtime_device,
)

__all__ = [
    "ProposalGenerationError",
    "ProposalGenerationPlan",
    "ProposalGenerationSummary",
    "generate_experiment_proposals",
]

_document_suffix = document_filename_suffix()
_component_count = 1


class ProposalGenerationError(ValueError):
    """Raised when local proposal generation cannot produce candidates."""


@dataclass(frozen=True, slots=True)
class ProposalGenerationPlan:
    """Inputs for deterministic local proposal generation."""

    benchmark_root: Path
    results_root: Path = Path("results")
    candidate_budget: int = 3
    candidate_sample_count: int = 64
    sample_count: int = 512
    evaluation_sample_count: int | None = None
    seed: int = 101
    train_steps: int | None = None
    learning_rate: float = 0.01
    optimizer: str = "adam"
    schedule: str = "reduce-on-plateau"
    validation_interval: int = 250
    convergence_patience: int = 12
    convergence_min_delta: float = 1e-3
    convergence_min_steps: int = 500
    tensor_device: TensorRuntimeDevice = "auto"

    def __post_init__(self) -> None:
        if type(self.candidate_budget) is not int or self.candidate_budget < 1:
            raise ProposalGenerationError("candidate_budget must be positive")
        if type(self.candidate_sample_count) is not int or self.candidate_sample_count < 1:
            raise ProposalGenerationError("candidate_sample_count must be positive")
        if self.candidate_sample_count < self.candidate_budget:
            raise ProposalGenerationError(
                "candidate_sample_count must be at least candidate_budget"
            )
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise ProposalGenerationError("sample_count must be positive")
        if (
            self.evaluation_sample_count is not None
            and (
                type(self.evaluation_sample_count) is not int
                or self.evaluation_sample_count < 1
            )
        ):
            raise ProposalGenerationError("evaluation_sample_count must be positive")
        if type(self.seed) is not int or self.seed < 0:
            raise ProposalGenerationError("seed must be nonnegative")
        if self.train_steps is not None and (
            type(self.train_steps) is not int or self.train_steps < 0
        ):
            raise ProposalGenerationError("train_steps must be nonnegative")
        if self.train_steps is None and self.schedule == "cosine":
            raise ProposalGenerationError("cosine schedule requires train_steps")
        if self.train_steps is None and self.convergence_patience == 0:
            raise ProposalGenerationError("uncapped training requires convergence_patience")
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
        if type(self.convergence_min_steps) is not int or self.convergence_min_steps < 0:
            raise ProposalGenerationError("convergence_min_steps must be nonnegative")
        try:
            validate_tensor_runtime_device(self.tensor_device)
        except TensorRuntimeError as error:
            raise ProposalGenerationError(str(error)) from error

    @property
    def resolved_evaluation_sample_count(self) -> int:
        """Return the evaluation sample count encoded into proposal commands."""

        if self.evaluation_sample_count is None:
            return self.sample_count
        return self.evaluation_sample_count


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
    predicted_score: float
    uncertainty: float
    acquisition_value: float
    novelty: float
    expected_frontier_improvement: float
    acquisition_model: str
    acquisition_components: Mapping[str, object]
    search_diagnostics: Mapping[str, object]


def generate_experiment_proposals(plan: ProposalGenerationPlan) -> ProposalGenerationSummary:
    """Generate deterministic candidate architecture proposals for one benchmark."""

    generator = load_observation_generator(plan.benchmark_root)
    manifest = generator.benchmark_manifest
    outcome_space = manifest.resolve_outcome_space()
    sample = generator.sample_batch(
        component_count=_component_count,
        sample_count=1,
        seed=plan.seed,
    ).samples[0]
    dataset = _measurement_dataset(plan.results_root, benchmark_id=manifest.id)
    measured = _measured_architectures(plan.results_root, benchmark_id=manifest.id)
    search_distribution = default_architecture_search_distribution()
    preferred_device = preferred_tensor_runtime_device_kind(plan.tensor_device)
    sampled_candidates = tuple(
        candidate
        for candidate in sample_architecture_candidates(
            search_distribution,
            input_shape=sample.field.shape,
            output_count=len(outcome_space.outcomes),
            sample_count=plan.candidate_sample_count,
            seed=plan.seed,
        )
        if architecture_supported_by_tensor_runtime(
            candidate.architecture,
            device_kind=preferred_device,
        )
    )
    candidate_observations = project_architecture_candidate_observations(
        sampled_candidates,
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
        search_distribution=search_distribution,
        measured=tuple(measured),
    )
    if not candidates:
        raise ProposalGenerationError("no unmeasured candidate architectures are available")

    benchmark_atom = _identifier_atom(manifest.id)
    architecture_root = plan.results_root / "proposals" / "architectures" / benchmark_atom
    proposal_root = plan.results_root / "proposals" / benchmark_atom
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
                acquisition_model=candidate.acquisition_model,
                acquisition_components=candidate.acquisition_components,
                search_diagnostics=candidate.search_diagnostics,
                selector_name=candidate.selector_name,
                source_candidate_rank=candidate.source_candidate_rank,
                comparable_cost_best_score=candidate.comparable_cost_best_score,
                resource_stratum_index=candidate.resource_stratum_index,
                resource_stratum_count=candidate.resource_stratum_count,
                command=(
                    "leibniz",
                    "benchmark",
                    "run",
                    "--benchmark-root",
                    plan.benchmark_root.as_posix(),
                    "--architecture",
                    architecture_path.as_posix(),
                    "--results-root",
                    plan.results_root.as_posix(),
                    "--sample-count",
                    str(plan.sample_count),
                    "--evaluation-sample-count",
                    str(plan.resolved_evaluation_sample_count),
                    "--seed",
                    str(plan.seed),
                    *(
                        ()
                        if plan.train_steps is None
                        else ("--train-steps", str(plan.train_steps))
                    ),
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
                    "--convergence-min-steps",
                    str(plan.convergence_min_steps),
                    "--device",
                    plan.tensor_device,
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
    results_root: Path,
    *,
    benchmark_id: ProtocolIdentifier,
) -> MeasurementDataset:
    records: dict[ProtocolIdentifier, Mapping[str, object]] = {}
    for dataset in _measurement_datasets(results_root, benchmark_id=benchmark_id):
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
    results_root: Path,
    *,
    benchmark_id: ProtocolIdentifier,
) -> tuple[MeasurementDataset, ...]:
    datasets: list[MeasurementDataset] = []
    measurement_root = results_root / "measurements" / _identifier_atom(benchmark_id)
    if measurement_root.is_dir():
        for path in sorted(measurement_root.rglob("*" + _document_suffix)):
            datasets.append(MeasurementDatasetDocument.from_bytes(path.read_bytes()).dataset)
    import_root = results_root / "imports" / "publication_bundles"
    if import_root.is_dir():
        for path in sorted(import_root.rglob("*" + _document_suffix)):
            bundle = SubmissionPublicationDocument.from_bytes(path.read_bytes()).bundle
            if bundle.submission_package.benchmark_manifest.id == benchmark_id:
                datasets.append(bundle.measurement_dataset)
    return tuple(datasets)


def _measured_architectures(
    results_root: Path,
    *,
    benchmark_id: ProtocolIdentifier,
) -> tuple[_MeasuredArchitecture, ...]:
    measured: list[_MeasuredArchitecture] = []
    for summary in _training_summaries(results_root, benchmark_id=benchmark_id):
        inspection_path = _summary_path(
            results_root=results_root,
            value=summary.get("model_inspection_path"),
            field="model_inspection_path",
        )
        dataset_path = _summary_path(
            results_root=results_root,
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

    import_root = results_root / "imports" / "publication_bundles"
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
    results_root: Path,
    *,
    benchmark_id: ProtocolIdentifier,
) -> tuple[Mapping[str, object], ...]:
    training_root = results_root / "training" / _identifier_atom(benchmark_id)
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
    search_distribution: ArchitectureSearchDistribution,
    measured: tuple[_MeasuredArchitecture, ...],
) -> tuple[_CandidateArchitecture, ...]:
    _require_candidate_shape(
        tuple(observation.candidate for observation in candidate_observations),
        input_shape=input_shape,
        output_count=output_count,
    )
    acquisition_scores = score_candidate_acquisition(
        candidate_observations,
        output_count=output_count,
    )
    candidates: list[_CandidateArchitecture] = []
    selections = select_candidate_proposals(
        candidate_observations,
        budget=candidate_budget,
        acquisition_scores=acquisition_scores,
    )
    for selection in selections:
        observation = selection.observation
        candidate = observation.candidate
        architecture = candidate.architecture
        parameter_count = observation.parameter_count
        score = selection.acquisition_score
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
                predicted_score=score.estimated_score,
                uncertainty=score.uncertainty,
                acquisition_value=score.acquisition_value,
                novelty=score.resource_novelty,
                expected_frontier_improvement=score.expected_frontier_improvement,
                acquisition_model=score.model_name,
                acquisition_components=score.to_component_record(),
                search_diagnostics=_search_diagnostics(
                    observation,
                    selection=selection,
                    search_distribution=search_distribution,
                    measured=measured,
                ),
            )
        )
    return tuple(candidates)


def _search_diagnostics(
    observation: ArchitectureCandidateObservation,
    *,
    selection: CandidateProposalSelection,
    search_distribution: ArchitectureSearchDistribution,
    measured: tuple[_MeasuredArchitecture, ...],
) -> dict[str, object]:
    record: dict[str, object] = {
        "search_distribution_id": str(search_distribution.id),
        "semantic_coordinates": [
            coordinate.to_record() for coordinate in observation.semantic_coordinates
        ],
    }
    if (
        selection.resource_stratum_index is not None
        and selection.resource_stratum_count is not None
    ):
        record["sampled_resource_stratum"] = {
            "index": selection.resource_stratum_index,
            "count": selection.resource_stratum_count,
        }
    nearest = _nearest_measured_support(observation, measured=measured)
    if nearest is not None:
        record["nearest_measured_support"] = nearest
    return record


def _nearest_measured_support(
    observation: ArchitectureCandidateObservation,
    *,
    measured: tuple[_MeasuredArchitecture, ...],
) -> dict[str, object] | None:
    if not measured:
        return None
    coordinate = math.log1p(observation.parameter_count)
    nearest = min(
        measured,
        key=lambda item: (
            abs(math.log1p(item.parameter_count) - coordinate),
            item.parameter_count,
            item.digest.hex,
        ),
    )
    return {
        "architecture_digest": f"{nearest.digest.algorithm}:{nearest.digest.hex}",
        "parameter_count": nearest.parameter_count,
        "score": nearest.score,
        "log_parameter_distance": abs(math.log1p(nearest.parameter_count) - coordinate),
    }


def _selection_rationale(selection: CandidateProposalSelection) -> str:
    if (
        selection.resource_stratum_index is not None
        and selection.resource_stratum_count is not None
    ):
        return (
            f"{selection.selector_name} selected resource stratum "
            f"{selection.resource_stratum_index + 1}/{selection.resource_stratum_count}"
        )
    return f"{selection.selector_name} selected resource candidate"


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


def _summary_path(*, results_root: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ProposalGenerationError(f"{field}: expected path string")
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (results_root.parent / path).resolve()
    if not resolved.is_relative_to(results_root.resolve()):
        raise ProposalGenerationError(f"{field} must stay inside results root")
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
