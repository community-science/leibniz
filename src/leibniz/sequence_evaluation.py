"""Exact finite-token sequence scoring records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.prediction_results import (
    PredictionResultValidationError,
    TokenSequencePrediction,
)
from leibniz.prediction_spaces import PredictionSpaceValidationError
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "ExactSequenceScore",
    "SequenceMeasurementDocument",
    "SequenceMeasurementRecord",
    "SequenceMeasurementValidationError",
]

_sequence_measurement_record = RecordSpec(
    fields={
        "format": FieldSpec(kind="literal", literal="leibniz.sequence-measurement"),
        "format_version": FieldSpec(kind="literal", literal=1),
        "benchmark_id": FieldSpec(kind="identifier"),
        "observation_id": FieldSpec(kind="string"),
        "prediction": FieldSpec(kind="record"),
        "accepted_sequence": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "accepted_mass": FieldSpec(kind="number"),
        "scoring_rule": FieldSpec(kind="literal", literal="exact-sequence-probability"),
    },
    allow_unknown=True,
)


class SequenceMeasurementValidationError(ValueError):
    """Raised when an exact sequence measurement is invalid."""


@dataclass(frozen=True, slots=True)
class ExactSequenceScore:
    """Exact full-sequence probability and negative-log score."""

    accepted_mass: float
    negative_log_score: float

    @classmethod
    def from_prediction(
        cls,
        *,
        prediction: TokenSequencePrediction,
        accepted_sequence: Sequence[int],
    ) -> ExactSequenceScore:
        accepted_mass = prediction.probability_of(tuple(accepted_sequence))
        if accepted_mass == 0.0:
            return cls(accepted_mass=0.0, negative_log_score=math.inf)
        return cls(accepted_mass=accepted_mass, negative_log_score=-math.log(accepted_mass))


@dataclass(frozen=True, slots=True)
class SequenceMeasurementRecord:
    """A durable exact-sequence measurement for one benchmark observation."""

    benchmark_id: ProtocolIdentifier
    observation_id: str
    prediction: TokenSequencePrediction
    accepted_sequence: tuple[int, ...]
    accepted_mass: float
    negative_log_score: float
    scoring_rule: str = "exact-sequence-probability"

    def __post_init__(self) -> None:
        try:
            self.benchmark_id.require_unreleased()
            self.prediction.prediction_space.require_sequence(self.accepted_sequence)
        except (ValueError, PredictionSpaceValidationError) as error:
            raise SequenceMeasurementValidationError(str(error)) from error
        if not self.observation_id:
            raise SequenceMeasurementValidationError("observation_id must be nonempty")
        if self.scoring_rule != "exact-sequence-probability":
            raise SequenceMeasurementValidationError(
                f"unsupported scoring_rule: {self.scoring_rule}"
            )
        score = ExactSequenceScore.from_prediction(
            prediction=self.prediction,
            accepted_sequence=self.accepted_sequence,
        )
        if not math.isclose(score.accepted_mass, self.accepted_mass, abs_tol=1e-12):
            raise SequenceMeasurementValidationError(
                "accepted_mass must equal prediction probability for accepted_sequence"
            )
        if score.negative_log_score == math.inf:
            if self.negative_log_score != math.inf:
                raise SequenceMeasurementValidationError(
                    "zero accepted_mass requires infinite negative_log_score"
                )
        elif not math.isclose(
            score.negative_log_score,
            self.negative_log_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise SequenceMeasurementValidationError(
                "negative_log_score must equal -log(accepted_mass)"
            )

    @classmethod
    def from_prediction(
        cls,
        *,
        benchmark_id: ProtocolIdentifier,
        observation_id: str,
        prediction: TokenSequencePrediction,
        accepted_sequence: Sequence[int],
    ) -> SequenceMeasurementRecord:
        sequence = tuple(accepted_sequence)
        score = ExactSequenceScore.from_prediction(
            prediction=prediction,
            accepted_sequence=sequence,
        )
        return cls(
            benchmark_id=benchmark_id,
            observation_id=observation_id,
            prediction=prediction,
            accepted_sequence=sequence,
            accepted_mass=score.accepted_mass,
            negative_log_score=score.negative_log_score,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SequenceMeasurementRecord:
        try:
            validated = _sequence_measurement_record.validate(record)
            prediction = TokenSequencePrediction.from_record(
                _as_mapping(validated["prediction"], "prediction")
            )
        except (ValueError, PredictionResultValidationError) as error:
            raise SequenceMeasurementValidationError(str(error)) from error
        if "negative_log_score" not in record:
            raise SequenceMeasurementValidationError(
                "negative_log_score: missing required field"
            )
        negative_log_score_value = record["negative_log_score"]
        negative_log_score = (
            math.inf
            if negative_log_score_value == "infinity"
            else _as_float(negative_log_score_value, "negative_log_score")
        )
        return cls(
            benchmark_id=_as_identifier(validated["benchmark_id"], "benchmark_id"),
            observation_id=str(validated["observation_id"]),
            prediction=prediction,
            accepted_sequence=tuple(
                _as_int(token, "accepted_sequence")
                for token in _as_tuple(validated["accepted_sequence"], "accepted_sequence")
            ),
            accepted_mass=_as_float(validated["accepted_mass"], "accepted_mass"),
            negative_log_score=negative_log_score,
            scoring_rule=str(validated["scoring_rule"]),
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "format": "leibniz.sequence-measurement",
            "format_version": 1,
            "benchmark_id": str(self.benchmark_id),
            "observation_id": self.observation_id,
            "prediction": self.prediction.to_record(),
            "accepted_sequence": list(self.accepted_sequence),
            "accepted_mass": self.accepted_mass,
            "negative_log_score": (
                "infinity"
                if self.negative_log_score == math.inf
                else self.negative_log_score
            ),
            "scoring_rule": self.scoring_rule,
        }


@dataclass(frozen=True, slots=True)
class SequenceMeasurementDocument:
    """A loaded exact sequence measurement and canonical digest."""

    measurement: SequenceMeasurementRecord
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> SequenceMeasurementDocument:
        try:
            record = load_object_document(data, description="sequence measurement document")
        except ContentEncodingError as error:
            raise SequenceMeasurementValidationError(str(error)) from error
        measurement = SequenceMeasurementRecord.from_record(record)
        return cls(measurement=measurement, digest=measurement.digest)


def _as_identifier(value: object, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise SequenceMeasurementValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SequenceMeasurementValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_tuple(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise SequenceMeasurementValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise SequenceMeasurementValidationError(f"{field}: expected integer")
    return value


def _as_float(value: object, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise SequenceMeasurementValidationError(f"{field}: expected number")
    result = float(value)
    if not math.isfinite(result):
        raise SequenceMeasurementValidationError(f"{field}: expected finite number")
    return result
