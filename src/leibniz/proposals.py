"""Experiment proposal records for declared artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from leibniz.architectures import ArchitectureManifest
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec
from leibniz.relationships import RelationshipFitRecord
from leibniz.submissions import SubmissionPackageManifest

__all__ = [
    "ExperimentProposal",
    "ExperimentProposalDocument",
    "ExperimentProposalSet",
    "ExperimentProposalValidationError",
]

_CandidateKind = Literal["architecture", "submission-package"]

_proposal_set_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "source_dataset_digest": FieldSpec(kind="string"),
        "relationship_fit_digest": FieldSpec(kind="string", required=False),
        "proposals": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
    }
)
_proposal_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "rank": FieldSpec(kind="integer"),
        "candidate_kind": FieldSpec(kind="string"),
        "candidate_id": FieldSpec(kind="identifier"),
        "rationale": FieldSpec(kind="string"),
        "source_relationship_fit_id": FieldSpec(kind="identifier", required=False),
        "predicted_score": FieldSpec(kind="number", required=False),
        "uncertainty": FieldSpec(kind="number", required=False),
        "acquisition_value": FieldSpec(kind="number", required=False),
        "acquisition_model": FieldSpec(kind="string", required=False),
        "acquisition_components": FieldSpec(kind="record", required=False),
        "search_diagnostics": FieldSpec(kind="record", required=False),
        "novelty": FieldSpec(kind="number", required=False),
        "expected_frontier_improvement": FieldSpec(kind="number", required=False),
        "selector_name": FieldSpec(kind="string", required=False),
        "source_candidate_rank": FieldSpec(kind="integer", required=False),
        "comparable_cost_best_score": FieldSpec(kind="number", required=False),
        "resource_stratum_index": FieldSpec(kind="integer", required=False),
        "resource_stratum_count": FieldSpec(kind="integer", required=False),
        "command": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="string"),
            required=False,
        ),
    }
)


class ExperimentProposalValidationError(ValueError):
    """Raised when an experiment proposal record is invalid."""


_extract = RecordExtractor(error_type=ExperimentProposalValidationError)


@dataclass(frozen=True, slots=True)
class ExperimentProposal:
    """One candidate selected for future measurement."""

    id: ProtocolIdentifier
    rank: int
    candidate_kind: _CandidateKind
    candidate_id: ProtocolIdentifier
    rationale: str
    source_relationship_fit_id: ProtocolIdentifier | None = None
    predicted_score: float | None = None
    uncertainty: float | None = None
    acquisition_value: float | None = None
    acquisition_model: str | None = None
    acquisition_components: Mapping[str, object] | None = None
    search_diagnostics: Mapping[str, object] | None = None
    novelty: float | None = None
    expected_frontier_improvement: float | None = None
    selector_name: str | None = None
    source_candidate_rank: int | None = None
    comparable_cost_best_score: float | None = None
    resource_stratum_index: int | None = None
    resource_stratum_count: int | None = None
    command: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
            self.candidate_id.require_unreleased()
            if self.source_relationship_fit_id is not None:
                self.source_relationship_fit_id.require_unreleased()
        except ValueError as error:
            raise ExperimentProposalValidationError(str(error)) from error
        if not str(self.id.name).startswith("experiment-proposals."):
            raise ExperimentProposalValidationError("id must be a valid experiment proposal id")
        if self.rank <= 0:
            raise ExperimentProposalValidationError("rank must be positive")
        if self.candidate_kind not in {"architecture", "submission-package"}:
            raise ExperimentProposalValidationError(
                f"unsupported candidate_kind: {self.candidate_kind}"
            )
        if not self.rationale:
            raise ExperimentProposalValidationError("rationale must be nonempty")
        _require_optional_probability(self.predicted_score, field="predicted_score")
        _require_optional_nonnegative(self.uncertainty, field="uncertainty")
        _require_optional_nonnegative(self.acquisition_value, field="acquisition_value")
        if self.acquisition_model is not None and not self.acquisition_model:
            raise ExperimentProposalValidationError("acquisition_model must be nonempty")
        _require_optional_nonnegative(self.novelty, field="novelty")
        _require_optional_nonnegative(
            self.expected_frontier_improvement,
            field="expected_frontier_improvement",
        )
        if self.selector_name is not None and not self.selector_name:
            raise ExperimentProposalValidationError("selector_name must be nonempty")
        if self.source_candidate_rank is not None and self.source_candidate_rank <= 0:
            raise ExperimentProposalValidationError("source_candidate_rank must be positive")
        _require_optional_probability(
            self.comparable_cost_best_score,
            field="comparable_cost_best_score",
        )
        _require_optional_nonnegative_int(
            self.resource_stratum_index,
            field="resource_stratum_index",
        )
        _require_optional_positive_int(
            self.resource_stratum_count,
            field="resource_stratum_count",
        )
        if (self.resource_stratum_index is None) != (self.resource_stratum_count is None):
            raise ExperimentProposalValidationError(
                "resource stratum fields must be provided together"
            )
        if (
            self.resource_stratum_index is not None
            and self.resource_stratum_count is not None
            and self.resource_stratum_index >= self.resource_stratum_count
        ):
            raise ExperimentProposalValidationError(
                "resource_stratum_index must be less than resource_stratum_count"
            )
        if any(not argument for argument in self.command):
            raise ExperimentProposalValidationError("command arguments must be nonempty")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ExperimentProposal:
        try:
            validated = _proposal_record.validate(record)
        except ValueError as error:
            raise ExperimentProposalValidationError(str(error)) from error
        return cls(
            id=_extract.identifier(validated["id"], "id"),
            rank=_extract.integer(validated["rank"], "rank"),
            candidate_kind=cast(_CandidateKind, str(validated["candidate_kind"])),
            candidate_id=_extract.identifier(validated["candidate_id"], "candidate_id"),
            rationale=str(validated["rationale"]),
            source_relationship_fit_id=(
                _extract.identifier(
                    validated["source_relationship_fit_id"],
                    "source_relationship_fit_id",
                )
                if "source_relationship_fit_id" in validated
                else None
            ),
            predicted_score=_extract.optional_float(
                validated.get("predicted_score"), "predicted_score"
            ),
            uncertainty=_extract.optional_float(validated.get("uncertainty"), "uncertainty"),
            acquisition_value=_extract.optional_float(
                validated.get("acquisition_value"),
                "acquisition_value",
            ),
            acquisition_model=(
                _extract.non_empty_string(validated["acquisition_model"], "acquisition_model")
                if "acquisition_model" in validated
                else None
            ),
            acquisition_components=(
                _extract.mapping(
                    validated["acquisition_components"],
                    "acquisition_components",
                )
                if "acquisition_components" in validated
                else None
            ),
            search_diagnostics=(
                _extract.mapping(
                    validated["search_diagnostics"],
                    "search_diagnostics",
                )
                if "search_diagnostics" in validated
                else None
            ),
            novelty=_extract.optional_float(validated.get("novelty"), "novelty"),
            expected_frontier_improvement=_extract.optional_float(
                validated.get("expected_frontier_improvement"),
                "expected_frontier_improvement",
            ),
            selector_name=(
                _extract.non_empty_string(validated["selector_name"], "selector_name")
                if "selector_name" in validated
                else None
            ),
            source_candidate_rank=(
                _extract.integer(validated["source_candidate_rank"], "source_candidate_rank")
                if "source_candidate_rank" in validated
                else None
            ),
            comparable_cost_best_score=_extract.optional_float(
                validated.get("comparable_cost_best_score"),
                "comparable_cost_best_score",
            ),
            resource_stratum_index=(
                _extract.integer(validated["resource_stratum_index"], "resource_stratum_index")
                if "resource_stratum_index" in validated
                else None
            ),
            resource_stratum_count=(
                _extract.integer(validated["resource_stratum_count"], "resource_stratum_count")
                if "resource_stratum_count" in validated
                else None
            ),
            command=tuple(
                _extract.non_empty_string(argument, "command")
                for argument in _extract.sequence(validated.get("command", ()), "command")
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "rank": self.rank,
            "candidate_kind": self.candidate_kind,
            "candidate_id": str(self.candidate_id),
            "rationale": self.rationale,
        }
        if self.source_relationship_fit_id is not None:
            record["source_relationship_fit_id"] = str(self.source_relationship_fit_id)
        if self.predicted_score is not None:
            record["predicted_score"] = self.predicted_score
        if self.uncertainty is not None:
            record["uncertainty"] = self.uncertainty
        if self.acquisition_value is not None:
            record["acquisition_value"] = self.acquisition_value
        if self.acquisition_model is not None:
            record["acquisition_model"] = self.acquisition_model
        if self.acquisition_components is not None:
            record["acquisition_components"] = dict(self.acquisition_components)
        if self.search_diagnostics is not None:
            record["search_diagnostics"] = dict(self.search_diagnostics)
        if self.novelty is not None:
            record["novelty"] = self.novelty
        if self.expected_frontier_improvement is not None:
            record["expected_frontier_improvement"] = self.expected_frontier_improvement
        if self.selector_name is not None:
            record["selector_name"] = self.selector_name
        if self.source_candidate_rank is not None:
            record["source_candidate_rank"] = self.source_candidate_rank
        if self.comparable_cost_best_score is not None:
            record["comparable_cost_best_score"] = self.comparable_cost_best_score
        if self.resource_stratum_index is not None:
            record["resource_stratum_index"] = self.resource_stratum_index
        if self.resource_stratum_count is not None:
            record["resource_stratum_count"] = self.resource_stratum_count
        if self.command:
            record["command"] = list(self.command)
        return record


@dataclass(frozen=True, slots=True)
class ExperimentProposalSet:
    """An auditable proposal set derived from public artifacts."""

    id: ProtocolIdentifier
    source_dataset_digest: ContentDigest
    proposals: tuple[ExperimentProposal, ...]
    relationship_fit_digest: ContentDigest | None = None

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise ExperimentProposalValidationError(str(error)) from error
        if not str(self.id.name).startswith("experiment-proposal-sets."):
            raise ExperimentProposalValidationError("id must be a valid experiment proposal set id")
        if not self.proposals:
            raise ExperimentProposalValidationError("proposals must contain at least one proposal")
        _reject_duplicate_ranks(self.proposals)
        _reject_duplicate_proposal_ids(self.proposals)
        ordered = tuple(sorted(self.proposals, key=lambda proposal: proposal.rank))
        if self.proposals != ordered:
            raise ExperimentProposalValidationError("proposals must be sorted by rank")

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        dataset: MeasurementDataset,
        relationship_fit: RelationshipFitRecord | None = None,
        architectures: tuple[ArchitectureManifest, ...] = (),
        submission_packages: tuple[SubmissionPackageManifest, ...] = (),
    ) -> ExperimentProposalSet:
        try:
            validated = _proposal_set_record.validate(record)
            proposals = tuple(
                ExperimentProposal.from_record(_extract.mapping(item, "proposals"))
                for item in _extract.sequence(validated["proposals"], "proposals")
            )
        except ValueError as error:
            raise ExperimentProposalValidationError(str(error)) from error
        proposal_set = cls(
            id=_extract.identifier(validated["id"], "id"),
            source_dataset_digest=_as_digest(
                validated["source_dataset_digest"],
                field="source_dataset_digest",
            ),
            relationship_fit_digest=(
                _as_digest(validated["relationship_fit_digest"], field="relationship_fit_digest")
                if "relationship_fit_digest" in validated
                else None
            ),
            proposals=proposals,
        )
        proposal_set.validate_sources(
            dataset=dataset,
            relationship_fit=relationship_fit,
            architectures=architectures,
            submission_packages=submission_packages,
        )
        return proposal_set

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_sources(
        self,
        *,
        dataset: MeasurementDataset,
        relationship_fit: RelationshipFitRecord | None = None,
        architectures: tuple[ArchitectureManifest, ...] = (),
        submission_packages: tuple[SubmissionPackageManifest, ...] = (),
    ) -> None:
        if self.source_dataset_digest != dataset.digest:
            raise ExperimentProposalValidationError("source_dataset_digest does not match dataset")
        if self.relationship_fit_digest is not None:
            if relationship_fit is None:
                raise ExperimentProposalValidationError("relationship_fit source is required")
            if self.relationship_fit_digest != relationship_fit.digest:
                raise ExperimentProposalValidationError(
                    "relationship_fit_digest does not match relationship fit"
                )
        architecture_ids = {architecture.id for architecture in architectures}
        submission_ids = {package.id for package in submission_packages}
        for proposal in self.proposals:
            if proposal.source_relationship_fit_id is not None:
                if relationship_fit is None:
                    raise ExperimentProposalValidationError(
                        "relationship_fit source is required"
                    )
                if proposal.source_relationship_fit_id != relationship_fit.id:
                    raise ExperimentProposalValidationError(
                        "source_relationship_fit_id does not match relationship fit"
                    )
            if proposal.candidate_kind == "architecture":
                if proposal.candidate_id not in architecture_ids:
                    raise ExperimentProposalValidationError(
                        f"unknown architecture candidate: {proposal.candidate_id}"
                    )
            elif proposal.candidate_id not in submission_ids:
                raise ExperimentProposalValidationError(
                    f"unknown submission package candidate: {proposal.candidate_id}"
                )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "source_dataset_digest": str(self.source_dataset_digest),
            "proposals": [proposal.to_record() for proposal in self.proposals],
        }
        if self.relationship_fit_digest is not None:
            record["relationship_fit_digest"] = str(self.relationship_fit_digest)
        return record


@dataclass(frozen=True, slots=True)
class ExperimentProposalDocument:
    """A loaded experiment proposal set and its canonical digest."""

    proposal_set: ExperimentProposalSet
    digest: ContentDigest

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        dataset: MeasurementDataset,
        relationship_fit: RelationshipFitRecord | None = None,
        architectures: tuple[ArchitectureManifest, ...] = (),
        submission_packages: tuple[SubmissionPackageManifest, ...] = (),
    ) -> ExperimentProposalDocument:
        try:
            record = load_object_document(data, description="experiment proposal document")
        except ContentEncodingError as error:
            raise ExperimentProposalValidationError(str(error)) from error
        proposal_set = ExperimentProposalSet.from_record(
            record,
            dataset=dataset,
            relationship_fit=relationship_fit,
            architectures=architectures,
            submission_packages=submission_packages,
        )
        return cls(proposal_set=proposal_set, digest=proposal_set.digest)
def _as_digest(value: object, *, field: str) -> ContentDigest:
    return ContentDigest.from_string(
        value,
        field=field,
        error_type=ExperimentProposalValidationError,
    )


def _require_optional_probability(value: float | None, *, field: str) -> None:
    if value is None:
        return
    _require_optional_nonnegative(value, field=field)
    if value > 1:
        raise ExperimentProposalValidationError(f"{field} must not exceed 1")


def _require_optional_nonnegative_int(value: int | None, *, field: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ExperimentProposalValidationError(f"{field} must be nonnegative")


def _require_optional_positive_int(value: int | None, *, field: str) -> None:
    if value is not None and (type(value) is not int or value < 1):
        raise ExperimentProposalValidationError(f"{field} must be positive")


def _require_optional_nonnegative(value: float | None, *, field: str) -> None:
    if value is None:
        return
    if value < 0 or not math.isfinite(value):
        raise ExperimentProposalValidationError(f"{field} must be finite and nonnegative")


def _reject_duplicate_ranks(proposals: tuple[ExperimentProposal, ...]) -> None:
    seen: set[int] = set()
    for proposal in proposals:
        if proposal.rank in seen:
            raise ExperimentProposalValidationError(f"duplicate proposal rank: {proposal.rank}")
        seen.add(proposal.rank)


def _reject_duplicate_proposal_ids(proposals: tuple[ExperimentProposal, ...]) -> None:
    seen: set[ProtocolIdentifier] = set()
    for proposal in proposals:
        if proposal.id in seen:
            raise ExperimentProposalValidationError(f"duplicate proposal id: {proposal.id}")
        seen.add(proposal.id)
