"""Submission publication bundle records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz._documents import ContentEncodingError, load_object_document
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset
from leibniz.proposals import ExperimentProposalSet
from leibniz.records import FieldSpec, RecordSpec
from leibniz.submissions import SubmissionPackageManifest
from leibniz.surrogates import ArchitectureSurrogateRecord
from leibniz.views import MeasurementScoreView

__all__ = [
    "SubmissionPublicationBundle",
    "SubmissionPublicationDocument",
    "SubmissionPublicationValidationError",
]

_publication_bundle_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "submission_package": FieldSpec(kind="record"),
        "measurement_dataset": FieldSpec(kind="record"),
        "measurement_score_view": FieldSpec(kind="record"),
        "proposal_set": FieldSpec(kind="record", required=False),
        "architecture_surrogate": FieldSpec(kind="record", required=False),
    }
)


class SubmissionPublicationValidationError(ValueError):
    """Raised when a submission publication bundle is invalid."""


@dataclass(frozen=True, slots=True)
class SubmissionPublicationBundle:
    """A local, checkable bundle of artifacts prepared for publication."""

    id: ProtocolIdentifier
    submission_package: SubmissionPackageManifest
    measurement_dataset: MeasurementDataset
    measurement_score_view: MeasurementScoreView
    proposal_set: ExperimentProposalSet | None = None
    architecture_surrogate: ArchitectureSurrogateRecord | None = None

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise SubmissionPublicationValidationError(str(error)) from error
        if not str(self.id.name).startswith("publication-bundles."):
            raise SubmissionPublicationValidationError(
                "id must be a valid publication bundle id"
            )
        self.validate_sources()

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SubmissionPublicationBundle:
        try:
            validated = _publication_bundle_record.validate(record)
            measurement_dataset = MeasurementDataset.from_record(
                _as_mapping(validated["measurement_dataset"], field="measurement_dataset")
            )
            submission_package = SubmissionPackageManifest.from_record(
                _as_mapping(validated["submission_package"], field="submission_package")
            )
            measurement_score_view = MeasurementScoreView.from_record(
                _as_mapping(validated["measurement_score_view"], field="measurement_score_view"),
                dataset=measurement_dataset,
            )
            proposal_set = (
                ExperimentProposalSet.from_record(
                    _as_mapping(validated["proposal_set"], field="proposal_set"),
                    dataset=measurement_dataset,
                    architectures=(submission_package.architecture_manifest,),
                    submission_packages=(submission_package,),
                )
                if "proposal_set" in validated
                else None
            )
            architecture_surrogate = (
                ArchitectureSurrogateRecord.from_record(
                    _as_mapping(
                        validated["architecture_surrogate"],
                        field="architecture_surrogate",
                    ),
                    dataset=measurement_dataset,
                )
                if "architecture_surrogate" in validated
                else None
            )
        except ValueError as error:
            raise SubmissionPublicationValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            submission_package=submission_package,
            measurement_dataset=measurement_dataset,
            measurement_score_view=measurement_score_view,
            proposal_set=proposal_set,
            architecture_surrogate=architecture_surrogate,
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_sources(self) -> None:
        if self.submission_package.measurement_dataset != self.measurement_dataset:
            raise SubmissionPublicationValidationError(
                "submission_package measurement_dataset does not match bundle dataset"
            )
        if self.measurement_score_view.source_dataset_digest != self.measurement_dataset.digest:
            raise SubmissionPublicationValidationError(
                "measurement_score_view source_dataset_digest does not match dataset"
            )
        if self.proposal_set is not None:
            if self.proposal_set.source_dataset_digest != self.measurement_dataset.digest:
                raise SubmissionPublicationValidationError(
                    "proposal_set source_dataset_digest does not match dataset"
                )
            try:
                self.proposal_set.validate_sources(
                    dataset=self.measurement_dataset,
                    architectures=(self.submission_package.architecture_manifest,),
                    submission_packages=(self.submission_package,),
                )
            except ValueError as error:
                raise SubmissionPublicationValidationError(str(error)) from error
        if self.architecture_surrogate is not None:
            try:
                self.architecture_surrogate.validate_sources(dataset=self.measurement_dataset)
            except ValueError as error:
                raise SubmissionPublicationValidationError(str(error)) from error

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "submission_package": self.submission_package.to_record(),
            "measurement_dataset": self.measurement_dataset.to_record(),
            "measurement_score_view": self.measurement_score_view.to_record(),
        }
        if self.proposal_set is not None:
            record["proposal_set"] = self.proposal_set.to_record()
        if self.architecture_surrogate is not None:
            record["architecture_surrogate"] = self.architecture_surrogate.to_record()
        return record


@dataclass(frozen=True, slots=True)
class SubmissionPublicationDocument:
    """A loaded submission publication bundle and its canonical digest."""

    bundle: SubmissionPublicationBundle
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> SubmissionPublicationDocument:
        try:
            record = load_object_document(data, description="submission publication document")
        except ContentEncodingError as error:
            raise SubmissionPublicationValidationError(str(error)) from error
        bundle = SubmissionPublicationBundle.from_record(record)
        return cls(bundle=bundle, digest=bundle.digest)


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise SubmissionPublicationValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SubmissionPublicationValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)
