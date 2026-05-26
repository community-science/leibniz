"""Submission package manifests for local artifact bundles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz.architectures import ArchitectureManifest
from leibniz.benchmarks import BenchmarkManifest
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "SubmissionArtifact",
    "SubmissionPackageDocument",
    "SubmissionPackageManifest",
    "SubmissionPackageValidationError",
]

_submission_package_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "benchmark_manifest": FieldSpec(kind="record"),
        "architecture_manifest": FieldSpec(kind="record"),
        "measurement_dataset": FieldSpec(kind="record"),
        "artifacts": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
            required=False,
        ),
    }
)
_submission_artifact_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "digest": FieldSpec(kind="string"),
        "description": FieldSpec(kind="string", required=False),
    }
)


class SubmissionPackageValidationError(ValueError):
    """Raised when a submission package manifest is invalid."""


@dataclass(frozen=True, slots=True)
class SubmissionArtifact:
    """Durable metadata for an artifact included in a submission package."""

    id: ProtocolIdentifier
    digest: ContentDigest
    description: str | None = None

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise SubmissionPackageValidationError(str(error)) from error
        if self.description == "":
            raise SubmissionPackageValidationError("artifact description must be nonempty")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SubmissionArtifact:
        try:
            validated = _submission_artifact_record.validate(record)
        except ValueError as error:
            raise SubmissionPackageValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="artifacts.id"),
            digest=_as_digest(validated["digest"], field="artifacts.digest"),
            description=_as_optional_string(validated.get("description")),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "digest": str(self.digest),
        }
        if self.description is not None:
            record["description"] = self.description
        return record


@dataclass(frozen=True, slots=True)
class SubmissionPackageManifest:
    """A local package manifest tying submission artifacts to evidence."""

    id: ProtocolIdentifier
    benchmark_manifest: BenchmarkManifest
    architecture_manifest: ArchitectureManifest
    measurement_dataset: MeasurementDataset
    artifacts: tuple[SubmissionArtifact, ...] = ()

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise SubmissionPackageValidationError(str(error)) from error
        if not str(self.id.name).startswith("submissions."):
            raise SubmissionPackageValidationError("id must be a valid submission package id")
        if not self.measurement_dataset.measurements:
            raise SubmissionPackageValidationError(
                "measurement_dataset must contain at least one measurement"
            )
        _reject_duplicate_artifact_ids(self.artifacts)
        try:
            _validate_measurement_dataset_manifest(
                dataset=self.measurement_dataset,
                manifest=self.benchmark_manifest,
            )
        except ValueError as error:
            raise SubmissionPackageValidationError(str(error)) from error

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SubmissionPackageManifest:
        try:
            validated = _submission_package_record.validate(record)
            benchmark_manifest = BenchmarkManifest.from_record(
                _as_mapping(validated["benchmark_manifest"], field="benchmark_manifest")
            )
            architecture_manifest = ArchitectureManifest.from_record(
                _as_mapping(validated["architecture_manifest"], field="architecture_manifest")
            )
            measurement_dataset = MeasurementDataset.from_record(
                _as_mapping(validated["measurement_dataset"], field="measurement_dataset")
            )
            artifacts = tuple(
                SubmissionArtifact.from_record(_as_mapping(item, field="artifacts"))
                for item in _as_sequence(validated.get("artifacts", ()), field="artifacts")
            )
        except ValueError as error:
            raise SubmissionPackageValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            benchmark_manifest=benchmark_manifest,
            architecture_manifest=architecture_manifest,
            measurement_dataset=measurement_dataset,
            artifacts=artifacts,
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "benchmark_manifest": self.benchmark_manifest.to_record(),
            "architecture_manifest": self.architecture_manifest.to_record(),
            "measurement_dataset": self.measurement_dataset.to_record(),
        }
        if self.artifacts:
            record["artifacts"] = [
                artifact.to_record()
                for artifact in sorted(self.artifacts, key=lambda artifact: str(artifact.id))
            ]
        return record


@dataclass(frozen=True, slots=True)
class SubmissionPackageDocument:
    """A loaded submission package manifest and its canonical digest."""

    manifest: SubmissionPackageManifest
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> SubmissionPackageDocument:
        try:
            record = load_object_document(data, description="submission package document")
        except ContentEncodingError as error:
            raise SubmissionPackageValidationError(str(error)) from error
        manifest = SubmissionPackageManifest.from_record(record)
        return cls(manifest=manifest, digest=manifest.digest)


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise SubmissionPackageValidationError(f"{field}: expected parsed identifier")
    return value


def _validate_measurement_dataset_manifest(
    *,
    dataset: MeasurementDataset,
    manifest: BenchmarkManifest,
) -> None:
    if manifest.outcome_space is not None:
        dataset.validate_manifest(manifest)
        return
    for measurement in dataset.measurements:
        measurement.validate_manifest(
            manifest,
            scale=_scale_from_measurement_outcome_space(manifest, measurement.outcome_space.id),
        )


def _scale_from_measurement_outcome_space(
    manifest: BenchmarkManifest,
    outcome_space_id: ProtocolIdentifier,
) -> int:
    prefix = f"{manifest.id.name}.outcomes.l"
    outcome_space_name = str(outcome_space_id.name)
    if not outcome_space_name.startswith(prefix):
        raise SubmissionPackageValidationError(
            "measurement outcome_space does not match scale-indexed manifest"
        )
    scale_text = outcome_space_name.removeprefix(prefix)
    if not scale_text.isdecimal():
        raise SubmissionPackageValidationError(
            "measurement outcome_space does not declare an integer scale"
        )
    return int(scale_text)


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SubmissionPackageValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise SubmissionPackageValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SubmissionPackageValidationError("artifact description must be a string")
    return value


def _as_digest(value: object, *, field: str) -> ContentDigest:
    if not isinstance(value, str):
        raise SubmissionPackageValidationError(f"{field}: expected digest string")
    algorithm, separator, digest_hex = value.partition(":")
    if separator == "":
        raise SubmissionPackageValidationError(f"{field}: expected algorithm:digest")
    try:
        return ContentDigest(algorithm=algorithm, hex=digest_hex)
    except ContentEncodingError as error:
        raise SubmissionPackageValidationError(str(error)) from error


def _reject_duplicate_artifact_ids(artifacts: tuple[SubmissionArtifact, ...]) -> None:
    seen: set[ProtocolIdentifier] = set()
    for artifact in artifacts:
        if artifact.id in seen:
            raise SubmissionPackageValidationError(f"duplicate artifact id: {artifact.id}")
        seen.add(artifact.id)
