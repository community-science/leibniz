"""Measurement records for finite-outcome scoring evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from leibniz._documents import ContentEncodingError, load_object_document
from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestValidationError
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.outcomes import (
    AcceptedEvent,
    FiniteOutcomeScoringGraph,
    FiniteProbabilityMeasure,
    OutcomeSpace,
    RawScoringEvidence,
)
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "MeasurementDocument",
    "MeasurementRecord",
    "MeasurementRecordValidationError",
]

_measurement_record = RecordSpec(
    fields={
        "benchmark_id": FieldSpec(kind="identifier"),
        "id": FieldSpec(kind="identifier", required=False),
        "observation_id": FieldSpec(kind="string", required=False),
        "outcome_space": FieldSpec(kind="record", required=False),
        "accepted_event": FieldSpec(kind="record", required=False),
        "probability_measure": FieldSpec(kind="record", required=False),
        "raw_scoring_evidence": FieldSpec(kind="record", required=False),
    }
)
_measurement_scoring_fields = frozenset(
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
    outcome_space: OutcomeSpace
    accepted_event: AcceptedEvent
    probability_measure: FiniteProbabilityMeasure
    raw_scoring_evidence: RawScoringEvidence

    def __post_init__(self) -> None:
        try:
            self.benchmark_id.require_unreleased()
        except ValueError as error:
            raise MeasurementRecordValidationError(str(error)) from error

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> MeasurementRecord:
        try:
            validated = _measurement_record.validate(record)
            scoring_graph = _measurement_scoring_graph(record)
        except ValueError as error:
            raise MeasurementRecordValidationError(str(error)) from error
        return cls(
            benchmark_id=_as_identifier(validated["benchmark_id"], field="benchmark_id"),
            outcome_space=scoring_graph.outcome_space,
            accepted_event=scoring_graph.accepted_event,
            probability_measure=scoring_graph.probability_measure,
            raw_scoring_evidence=scoring_graph.raw_scoring_evidence,
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
            manifest.validate_measurement(
                outcome_space_id=self.outcome_space.id,
                observation_id=self.raw_scoring_evidence.observation_id,
            )
        except BenchmarkManifestValidationError as error:
            raise MeasurementRecordValidationError(str(error)) from error

    def to_record(self) -> dict[str, object]:
        return {
            "benchmark_id": str(self.benchmark_id),
            **self._scoring_graph().to_record(),
        }

    def _scoring_graph(self) -> FiniteOutcomeScoringGraph:
        return FiniteOutcomeScoringGraph(
            outcome_space=self.outcome_space,
            accepted_event=self.accepted_event,
            probability_measure=self.probability_measure,
            raw_scoring_evidence=self.raw_scoring_evidence,
        )


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


def _measurement_scoring_graph(record: Mapping[str, object]) -> FiniteOutcomeScoringGraph:
    scoring_fields = {
        field: record[field]
        for field in _measurement_scoring_fields
        if field in record
    }
    if not scoring_fields:
        raise MeasurementRecordValidationError("measurement scoring fields are missing")
    return FiniteOutcomeScoringGraph.from_record(scoring_fields)
