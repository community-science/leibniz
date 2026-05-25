"""Measurement records for finite-answer scoring evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz.answers import FiniteAnswerScoringBundle
from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestValidationError
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier, require_unreleased_identifier
from leibniz.records import RecordSpec, required, validate_record

__all__ = [
    "MeasurementRecord",
    "MeasurementRecordValidationError",
]

_measurement_record = RecordSpec(
    fields={
        "benchmark_id": required("identifier"),
        "scoring_bundle": required("record"),
    }
)


class MeasurementRecordValidationError(ValueError):
    """Raised when a measurement record is invalid."""


@dataclass(frozen=True, slots=True)
class MeasurementRecord:
    """A durable finite-answer measurement for one benchmark observation."""

    benchmark_id: ProtocolIdentifier
    scoring_bundle: FiniteAnswerScoringBundle

    def __post_init__(self) -> None:
        try:
            require_unreleased_identifier(self.benchmark_id)
        except ValueError as error:
            raise MeasurementRecordValidationError(str(error)) from error

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> MeasurementRecord:
        try:
            validated = validate_record(record, _measurement_record)
            scoring_bundle = FiniteAnswerScoringBundle.from_record(
                _as_mapping(validated["scoring_bundle"], field="scoring_bundle")
            )
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


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise MeasurementRecordValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MeasurementRecordValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)
