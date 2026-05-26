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
from leibniz.records import FieldSpec, RecordSpec
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
        "novelty": FieldSpec(kind="number", required=False),
        "expected_frontier_improvement": FieldSpec(kind="number", required=False),
        "command": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="string"),
            required=False,
        ),
    }
)


class ExperimentProposalValidationError(ValueError):
    """Raised when an experiment proposal record is invalid."""


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
    novelty: float | None = None
    expected_frontier_improvement: float | None = None
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
        _require_optional_nonnegative(self.novelty, field="novelty")
        _require_optional_nonnegative(
            self.expected_frontier_improvement,
            field="expected_frontier_improvement",
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
            id=_as_identifier(validated["id"], field="id"),
            rank=_as_int(validated["rank"], field="rank"),
            candidate_kind=cast(_CandidateKind, str(validated["candidate_kind"])),
            candidate_id=_as_identifier(validated["candidate_id"], field="candidate_id"),
            rationale=str(validated["rationale"]),
            source_relationship_fit_id=(
                _as_identifier(
                    validated["source_relationship_fit_id"],
                    field="source_relationship_fit_id",
                )
                if "source_relationship_fit_id" in validated
                else None
            ),
            predicted_score=_optional_float(validated.get("predicted_score"), "predicted_score"),
            uncertainty=_optional_float(validated.get("uncertainty"), "uncertainty"),
            acquisition_value=_optional_float(
                validated.get("acquisition_value"),
                "acquisition_value",
            ),
            novelty=_optional_float(validated.get("novelty"), "novelty"),
            expected_frontier_improvement=_optional_float(
                validated.get("expected_frontier_improvement"),
                "expected_frontier_improvement",
            ),
            command=tuple(
                _as_string(argument, field="command")
                for argument in _as_sequence(validated.get("command", ()), field="command")
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
        if self.novelty is not None:
            record["novelty"] = self.novelty
        if self.expected_frontier_improvement is not None:
            record["expected_frontier_improvement"] = self.expected_frontier_improvement
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
                ExperimentProposal.from_record(_as_mapping(item, field="proposals"))
                for item in _as_sequence(validated["proposals"], field="proposals")
            )
        except ValueError as error:
            raise ExperimentProposalValidationError(str(error)) from error
        proposal_set = cls(
            id=_as_identifier(validated["id"], field="id"),
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


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise ExperimentProposalValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ExperimentProposalValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ExperimentProposalValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_digest(value: object, *, field: str) -> ContentDigest:
    if not isinstance(value, str):
        raise ExperimentProposalValidationError(f"{field}: expected digest string")
    algorithm, separator, digest_hex = value.partition(":")
    if separator == "":
        raise ExperimentProposalValidationError(f"{field}: expected algorithm:digest")
    try:
        return ContentDigest(algorithm=algorithm, hex=digest_hex)
    except ContentEncodingError as error:
        raise ExperimentProposalValidationError(str(error)) from error


def _as_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ExperimentProposalValidationError(f"{field}: expected parsed integer")
    return value


def _as_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExperimentProposalValidationError(f"{field}: expected nonempty string")
    return value


def _optional_float(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExperimentProposalValidationError(f"{field}: expected number")
    return float(value)


def _require_optional_probability(value: float | None, *, field: str) -> None:
    if value is None:
        return
    _require_optional_nonnegative(value, field=field)
    if value > 1:
        raise ExperimentProposalValidationError(f"{field} must not exceed 1")


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
