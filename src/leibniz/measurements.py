"""Measurement records for finite-outcome scoring evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz.artifacts import ArtifactReference
from leibniz.benchmarks import BenchmarkManifest
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.outcomes import (
    AcceptedEvent,
    AcceptedMassScore,
    FiniteProbabilityMeasure,
    OutcomeSpace,
    RawScoringEvidence,
)
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "MeasurementDataset",
    "MeasurementDatasetDocument",
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
        "evidence_artifacts": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
            required=False,
        ),
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
_finite_outcome_scoring_graph_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier", required=False),
        "observation_id": FieldSpec(kind="string", required=False),
        "outcome_space": FieldSpec(kind="record"),
        "accepted_event": FieldSpec(kind="record"),
        "probability_measure": FieldSpec(kind="record"),
        "raw_scoring_evidence": FieldSpec(kind="record", required=False),
    },
    allow_unknown=True,
)
_finite_outcome_scoring_graph_expected_fields = frozenset(_measurement_scoring_fields)
_measurement_dataset_record = RecordSpec(
    fields={
        "measurements": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
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
    evidence_artifacts: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        try:
            self.benchmark_id.require_unreleased()
        except ValueError as error:
            raise MeasurementRecordValidationError(str(error)) from error

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> MeasurementRecord:
        try:
            validated = _measurement_record.validate(record)
            (
                outcome_space,
                accepted_event,
                probability_measure,
                raw_scoring_evidence,
            ) = _measurement_scoring_parts(record)
        except ValueError as error:
            raise MeasurementRecordValidationError(str(error)) from error
        return cls(
            benchmark_id=_as_identifier(validated["benchmark_id"], field="benchmark_id"),
            outcome_space=outcome_space,
            accepted_event=accepted_event,
            probability_measure=probability_measure,
            raw_scoring_evidence=raw_scoring_evidence,
            evidence_artifacts=_evidence_artifacts(validated),
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_manifest(self, manifest: BenchmarkManifest, *, scale: int | None = None) -> None:
        if manifest.outcome_space is None and scale is None:
            raise MeasurementRecordValidationError(
                "scale-indexed benchmark manifests require resolved outcome spaces"
            )
        if self.benchmark_id != manifest.id:
            raise MeasurementRecordValidationError(
                f"benchmark_id {self.benchmark_id} does not match manifest {manifest.id}"
            )
        expected_outcome_space = (
            manifest.outcome_space if scale is None else manifest.resolve_outcome_space(scale=scale)
        )
        if expected_outcome_space is None:
            raise MeasurementRecordValidationError(
                "scale-indexed benchmark manifests require resolved outcome spaces"
            )
        if self.outcome_space != expected_outcome_space:
            raise MeasurementRecordValidationError(
                "measurement outcome_space does not match manifest outcome_space "
                f"{expected_outcome_space.id}"
            )
        if (
            manifest.observation_ids is not None
            and self.raw_scoring_evidence.observation_id not in manifest.observation_ids
        ):
            raise MeasurementRecordValidationError(
                "observation_id "
                f"{self.raw_scoring_evidence.observation_id!r} is not declared by "
                f"{manifest.id}"
            )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "benchmark_id": str(self.benchmark_id),
            "outcome_space": self.outcome_space.to_record(),
            "accepted_event": self.accepted_event.to_record(),
            "probability_measure": self.probability_measure.to_record(),
            "raw_scoring_evidence": self.raw_scoring_evidence.to_record(),
        }
        if self.evidence_artifacts:
            record["evidence_artifacts"] = [
                artifact.to_record() for artifact in self.evidence_artifacts
            ]
        return record


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


@dataclass(frozen=True, slots=True)
class MeasurementDataset:
    """A canonical collection of finite-outcome measurements."""

    measurements: tuple[MeasurementRecord, ...]

    def __post_init__(self) -> None:
        measurement_ids = tuple(_measurement_id(measurement) for measurement in self.measurements)
        duplicate_id = _first_duplicate(measurement_ids)
        if duplicate_id is not None:
            raise MeasurementRecordValidationError(f"duplicate measurement id: {duplicate_id}")

        ordered = tuple(
            sorted(self.measurements, key=lambda measurement: str(_measurement_id(measurement)))
        )
        object.__setattr__(self, "measurements", ordered)

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> MeasurementDataset:
        try:
            validated = _measurement_dataset_record.validate(record)
            measurement_records = _as_sequence(
                validated["measurements"],
                field="measurements",
            )
            measurements = tuple(
                MeasurementRecord.from_record(_scoring_mapping(item, field="measurements"))
                for item in measurement_records
            )
        except ValueError as error:
            raise MeasurementRecordValidationError(str(error)) from error
        return cls(measurements=measurements)

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_manifest(self, manifest: BenchmarkManifest, *, scale: int | None = None) -> None:
        for measurement in self.measurements:
            measurement.validate_manifest(manifest, scale=scale)

    def to_record(self) -> dict[str, object]:
        return {
            "measurements": [
                measurement.to_record()
                for measurement in self.measurements
            ]
        }


@dataclass(frozen=True, slots=True)
class MeasurementDatasetDocument:
    """A loaded measurement dataset and the digest of its canonical record."""

    dataset: MeasurementDataset
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> MeasurementDatasetDocument:
        try:
            record = load_object_document(data, description="measurement dataset document")
        except ContentEncodingError as error:
            raise MeasurementRecordValidationError(str(error)) from error
        dataset = MeasurementDataset.from_record(record)
        return cls(dataset=dataset, digest=dataset.digest)


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise MeasurementRecordValidationError(f"{field}: expected parsed identifier")
    return value


def _measurement_scoring_parts(
    record: Mapping[str, object],
) -> tuple[OutcomeSpace, AcceptedEvent, FiniteProbabilityMeasure, RawScoringEvidence]:
    scoring_fields = {
        field: record[field]
        for field in _measurement_scoring_fields
        if field in record
    }
    if not scoring_fields:
        raise MeasurementRecordValidationError("measurement scoring fields are missing")
    return _scoring_parts_from_record(scoring_fields)


def _scoring_parts_from_record(
    record: Mapping[str, object],
) -> tuple[OutcomeSpace, AcceptedEvent, FiniteProbabilityMeasure, RawScoringEvidence]:
    validated = _finite_outcome_scoring_graph_record.validate(record)
    outcome_space = OutcomeSpace.from_record(
        _scoring_mapping(validated["outcome_space"], field="outcome_space")
    )
    accepted_event = AcceptedEvent.from_record(
        _scoring_mapping(validated["accepted_event"], field="accepted_event"),
        outcome_space=outcome_space,
    )
    probability_measure = FiniteProbabilityMeasure.from_record(
        _scoring_mapping(
            validated["probability_measure"],
            field="probability_measure",
        ),
        outcome_space=outcome_space,
    )
    raw_scoring_evidence = _raw_scoring_evidence(
        record=record,
        validated=validated,
        accepted_event=accepted_event,
        probability_measure=probability_measure,
    )
    _validate_scoring_parts(
        outcome_space=outcome_space,
        accepted_event=accepted_event,
        probability_measure=probability_measure,
        raw_scoring_evidence=raw_scoring_evidence,
    )
    return outcome_space, accepted_event, probability_measure, raw_scoring_evidence


def _validate_scoring_parts(
    *,
    outcome_space: OutcomeSpace,
    accepted_event: AcceptedEvent,
    probability_measure: FiniteProbabilityMeasure,
    raw_scoring_evidence: RawScoringEvidence,
) -> None:
    _require_matching_identifier(
        field="accepted_event.outcome_space_id",
        actual=accepted_event.outcome_space_id,
        expected=outcome_space.id,
    )
    _require_matching_identifier(
        field="probability_measure.outcome_space_id",
        actual=probability_measure.outcome_space_id,
        expected=outcome_space.id,
    )
    _require_matching_identifier(
        field="raw_scoring_evidence.outcome_space_id",
        actual=raw_scoring_evidence.outcome_space_id,
        expected=outcome_space.id,
    )
    _require_matching_identifier(
        field="raw_scoring_evidence.accepted_event_id",
        actual=raw_scoring_evidence.accepted_event_id,
        expected=accepted_event.id,
    )
    _require_matching_identifier(
        field="raw_scoring_evidence.probability_measure_id",
        actual=raw_scoring_evidence.probability_measure_id,
        expected=probability_measure.id,
    )

    score = AcceptedMassScore.from_event_and_measure(
        event=accepted_event,
        measure=probability_measure,
    )
    if not math.isclose(
        raw_scoring_evidence.accepted_mass,
        score.accepted_mass,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise MeasurementRecordValidationError(
            "raw_scoring_evidence.accepted_mass must equal recomputed accepted mass"
        )
    if not math.isclose(
        raw_scoring_evidence.negative_log_score,
        score.negative_log_score,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise MeasurementRecordValidationError(
            "raw_scoring_evidence.negative_log_score must equal recomputed score"
        )


def _raw_scoring_evidence(
    *,
    record: Mapping[str, object],
    validated: Mapping[str, object],
    accepted_event: AcceptedEvent,
    probability_measure: FiniteProbabilityMeasure,
) -> RawScoringEvidence:
    unknown_fields = tuple(
        sorted(
            field
            for field in record
            if field not in _finite_outcome_scoring_graph_expected_fields
        )
    )
    if unknown_fields:
        raise MeasurementRecordValidationError(f"{unknown_fields[0]}: unknown field")

    raw_value = validated.get("raw_scoring_evidence")
    explicit: RawScoringEvidence | None = None
    if raw_value is not None:
        explicit = RawScoringEvidence.from_record(
            _scoring_mapping(
                raw_value,
                field="raw_scoring_evidence",
            )
        )

    evidence_id = validated.get("id")
    observation_id = validated.get("observation_id")
    if evidence_id is None:
        if explicit is None:
            raise MeasurementRecordValidationError("id: missing required field")
        evidence_id = explicit.id
    if observation_id is None:
        if explicit is None:
            raise MeasurementRecordValidationError("observation_id: missing required field")
        observation_id = explicit.observation_id

    derived = RawScoringEvidence.from_event_and_measure(
        id=_as_identifier(evidence_id, field="id"),
        observation_id=str(observation_id),
        event=accepted_event,
        measure=probability_measure,
    )
    if explicit is None:
        return derived
    if explicit != derived:
        raise MeasurementRecordValidationError(
            "raw_scoring_evidence must equal derived scoring evidence"
        )
    return explicit


def _scoring_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MeasurementRecordValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise MeasurementRecordValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _evidence_artifacts(record: Mapping[str, object]) -> tuple[ArtifactReference, ...]:
    raw_artifacts = record.get("evidence_artifacts")
    if raw_artifacts is None:
        return ()
    artifacts = tuple(
        ArtifactReference.from_record(_scoring_mapping(item, field="evidence_artifacts"))
        for item in _as_sequence(raw_artifacts, field="evidence_artifacts")
    )
    duplicate = _first_duplicate_references(artifacts)
    if duplicate is not None:
        raise MeasurementRecordValidationError("duplicate evidence artifact")
    return artifacts


def _measurement_id(measurement: MeasurementRecord) -> ProtocolIdentifier:
    return measurement.raw_scoring_evidence.id


def _first_duplicate(values: tuple[ProtocolIdentifier, ...]) -> ProtocolIdentifier | None:
    seen: set[ProtocolIdentifier] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _first_duplicate_references(
    values: tuple[ArtifactReference, ...],
) -> ArtifactReference | None:
    seen: set[ArtifactReference] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _require_matching_identifier(
    *,
    field: str,
    actual: ProtocolIdentifier,
    expected: ProtocolIdentifier,
) -> None:
    if actual != expected:
        raise MeasurementRecordValidationError(f"{field} {actual} does not match {expected}")
