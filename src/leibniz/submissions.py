"""Submission package manifests for local artifact bundles."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz.architectures import ArchitectureManifest
from leibniz.benchmarks import BenchmarkManifest
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec

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
        "sampled_competence": FieldSpec(kind="record", required=False),
        "model_metadata": FieldSpec(kind="record", required=False),
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


_extract = RecordExtractor(error_type=SubmissionPackageValidationError)


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
            id=_extract.identifier(validated["id"], "artifacts.id"),
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
    sampled_competence: Mapping[str, object] | None = None
    model_metadata: Mapping[str, object] | None = None
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
        if self.sampled_competence is not None:
            _validate_sampled_competence(
                self.sampled_competence,
                benchmark_manifest=self.benchmark_manifest,
            )
        if self.model_metadata is not None:
            _validate_model_metadata(self.model_metadata)
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
                _extract.mapping(validated["benchmark_manifest"], "benchmark_manifest")
            )
            architecture_manifest = ArchitectureManifest.from_record(
                _extract.mapping(validated["architecture_manifest"], "architecture_manifest")
            )
            measurement_dataset = MeasurementDataset.from_record(
                _extract.mapping(validated["measurement_dataset"], "measurement_dataset")
            )
            artifacts = tuple(
                SubmissionArtifact.from_record(_extract.mapping(item, "artifacts"))
                for item in _extract.sequence(validated.get("artifacts", ()), "artifacts")
            )
            sampled_competence = _extract.optional_mapping(
                validated.get("sampled_competence"),
                "sampled_competence",
            )
            model_metadata = _extract.optional_mapping(
                validated.get("model_metadata"),
                "model_metadata",
            )
        except ValueError as error:
            raise SubmissionPackageValidationError(str(error)) from error
        return cls(
            id=_extract.identifier(validated["id"], "id"),
            benchmark_manifest=benchmark_manifest,
            architecture_manifest=architecture_manifest,
            measurement_dataset=measurement_dataset,
            sampled_competence=sampled_competence,
            model_metadata=model_metadata,
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
        if self.sampled_competence is not None:
            record["sampled_competence"] = dict(self.sampled_competence)
        if self.model_metadata is not None:
            record["model_metadata"] = dict(self.model_metadata)
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


def _validate_measurement_dataset_manifest(
    *,
    dataset: MeasurementDataset,
    manifest: BenchmarkManifest,
) -> None:
    dataset.validate_manifest(manifest)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if isinstance(value, list):
        return tuple(cast(list[object], value))
    if isinstance(value, tuple):
        return cast(tuple[object, ...], value)
    raise SubmissionPackageValidationError(f"{field}: expected sequence")


def _as_optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise SubmissionPackageValidationError("artifact description must be a string")
    return value


def _as_digest(value: object, *, field: str) -> ContentDigest:
    return ContentDigest.from_string(
        value,
        field=field,
        error_type=SubmissionPackageValidationError,
    )


def _reject_duplicate_artifact_ids(artifacts: tuple[SubmissionArtifact, ...]) -> None:
    seen: set[ProtocolIdentifier] = set()
    for artifact in artifacts:
        if artifact.id in seen:
            raise SubmissionPackageValidationError(f"duplicate artifact id: {artifact.id}")
        seen.add(artifact.id)


def _validate_sampled_competence(
    record: Mapping[str, object],
    *,
    benchmark_manifest: BenchmarkManifest,
) -> None:
    benchmark_id = record.get("benchmark_id")
    if benchmark_id != str(benchmark_manifest.id):
        raise SubmissionPackageValidationError(
            "sampled_competence benchmark_id does not match benchmark_manifest"
        )
    kind = record.get("kind")
    if kind not in {"sampled-complexity-class", "sampled-competence-curriculum"}:
        raise SubmissionPackageValidationError("sampled_competence has unsupported kind")
    _positive_int(record.get("sample_count"), field="sampled_competence.sample_count")
    _score(record.get("mean_accepted_mass"), field="sampled_competence.mean_accepted_mass")
    _nonnegative_number(record.get("complexity"), field="sampled_competence.complexity")
    points = record.get("points")
    if points is not None:
        for point in _as_sequence(points, field="sampled_competence.points"):
            point_record = _extract.mapping(point, "sampled_competence.points")
            _positive_int(
                point_record.get("sample_count"),
                field="sampled_competence.points.sample_count",
            )
            _score(
                point_record.get("mean_accepted_mass"),
                field="sampled_competence.points.mean_accepted_mass",
            )
            _nonnegative_number(
                point_record.get("complexity"),
                field="sampled_competence.points.complexity",
            )


def _validate_model_metadata(record: Mapping[str, object]) -> None:
    cost_summary = record.get("cost_summary")
    if cost_summary is None:
        return
    summary = _extract.mapping(cost_summary, "model_metadata.cost_summary")
    for key, value in summary.items():
        if key.endswith("_components"):
            _as_sequence(value, field=f"model_metadata.cost_summary.{key}")
        else:
            _nonnegative_number(value, field=f"model_metadata.cost_summary.{key}")


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise SubmissionPackageValidationError(f"{field}: expected positive integer")
    return value


def _nonnegative_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SubmissionPackageValidationError(f"{field}: expected nonnegative number")
    number = float(value)
    if number < 0.0 or not math.isfinite(number):
        raise SubmissionPackageValidationError(f"{field}: expected nonnegative number")
    return number


def _score(value: object, *, field: str) -> float:
    score = _nonnegative_number(value, field=field)
    if score > 1.0:
        raise SubmissionPackageValidationError(f"{field}: expected score at most 1")
    return score
