"""Measurement records for finite-outcome scoring evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz._documents import ContentEncodingError, load_object_document
from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestValidationError
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.outcomes import FiniteOutcomeScoringBundle
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "MeasurementDocument",
    "MeasurementRecord",
    "MeasurementRecordValidationError",
]

_measurement_record = RecordSpec(
    fields={
        "benchmark_id": FieldSpec(kind="identifier"),
        "scoring_bundle": FieldSpec(kind="record", required=False),
        "id": FieldSpec(kind="identifier", required=False),
        "observation_id": FieldSpec(kind="string", required=False),
        "outcome_space": FieldSpec(kind="record", required=False),
        "accepted_event": FieldSpec(kind="record", required=False),
        "probability_measure": FieldSpec(kind="record", required=False),
        "raw_scoring_evidence": FieldSpec(kind="record", required=False),
    }
)
_top_level_scoring_fields = frozenset(
    {
        "id",
        "observation_id",
        "outcome_space",
        "accepted_event",
        "probability_measure",
        "raw_scoring_evidence",
    }
)


class MeasurementRecordValidationError(ValueError):
    """Raised when a measurement record is invalid."""


@dataclass(frozen=True, slots=True)
class MeasurementRecord:
    """A durable finite-outcome measurement for one benchmark observation."""

    benchmark_id: ProtocolIdentifier
    scoring_bundle: FiniteOutcomeScoringBundle

    def __post_init__(self) -> None:
        try:
            self.benchmark_id.require_unreleased()
        except ValueError as error:
            raise MeasurementRecordValidationError(str(error)) from error

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> MeasurementRecord:
        try:
            validated = _measurement_record.validate(record)
            scoring_bundle = _measurement_scoring_bundle(record, validated=validated)
        except ValueError as error:
            raise MeasurementRecordValidationError(str(error)) from error
        return cls(
            benchmark_id=_as_identifier(validated["benchmark_id"], field="benchmark_id"),
            scoring_bundle=scoring_bundle,
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_manifest(self, manifest: BenchmarkManifest) -> None:
        if self.benchmark_id != manifest.id:
            raise MeasurementRecordValidationError(
                f"benchmark_id {self.benchmark_id} does not match manifest {manifest.id}"
            )
        try:
            manifest.validate_bundle(self.scoring_bundle)
        except BenchmarkManifestValidationError as error:
            raise MeasurementRecordValidationError(str(error)) from error

    def to_record(self) -> dict[str, object]:
        return {
            "benchmark_id": str(self.benchmark_id),
            "scoring_bundle": self.scoring_bundle.to_record(),
        }


@dataclass(frozen=True, slots=True)
class MeasurementDocument:
    """A loaded measurement record and the digest of its canonical record."""

    measurement: MeasurementRecord
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> MeasurementDocument:
        try:
            record = load_object_document(data, description="measurement document")
        except ContentEncodingError as error:
            raise MeasurementRecordValidationError(str(error)) from error
        measurement = MeasurementRecord.from_record(record)
        return cls(measurement=measurement, digest=measurement.digest)


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise MeasurementRecordValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MeasurementRecordValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _measurement_scoring_bundle(
    record: Mapping[str, object],
    *,
    validated: Mapping[str, object],
) -> FiniteOutcomeScoringBundle:
    scoring_bundle = validated.get("scoring_bundle")
    top_level = {
        field: record[field]
        for field in _top_level_scoring_fields
        if field in record
    }
    if scoring_bundle is not None:
        if top_level:
            raise MeasurementRecordValidationError(
                "scoring_bundle cannot be combined with top-level scoring fields"
            )
        return FiniteOutcomeScoringBundle.from_record(
            _as_mapping(scoring_bundle, field="scoring_bundle")
        )
    if not top_level:
        raise MeasurementRecordValidationError("measurement scoring fields are missing")
    return FiniteOutcomeScoringBundle.from_record(top_level)
