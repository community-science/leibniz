"""Prediction result records for model outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import cast

from leibniz.identifiers import ProtocolIdentifier
from leibniz.outcomes import FiniteProbabilityMeasure, OutcomeSpace, ProbabilityMass
from leibniz.prediction_spaces import (
    FiniteOutcomeSpace,
    FiniteTokenSequenceSpace,
    PredictionSpaceValidationError,
)
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "DirectFiniteProbabilityPrediction",
    "PredictionMass",
    "PredictionResultContract",
    "PredictionResultValidationError",
    "TokenSequenceProbability",
    "TokenSequencePrediction",
]

_prediction_mass_record = RecordSpec(
    fields={
        "outcome_index": FieldSpec(kind="integer"),
        "probability": FieldSpec(kind="number"),
    }
)
_direct_finite_probability_prediction_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "prediction_space": FieldSpec(kind="record"),
        "prediction_kind": FieldSpec(
            kind="literal",
            literal="direct-finite-probability-measure",
        ),
        "output_encoding": FieldSpec(
            kind="literal",
            literal="probability-mass-sequence",
        ),
        "probabilities": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record", record=_prediction_mass_record),
        ),
    }
)
_token_sequence_probability_record = RecordSpec(
    fields={
        "tokens": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "probability": FieldSpec(kind="number"),
    }
)
_token_sequence_prediction_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "prediction_space": FieldSpec(kind="record"),
        "prediction_kind": FieldSpec(
            kind="literal",
            literal="finite-token-sequence-probability",
        ),
        "output_encoding": FieldSpec(
            kind="literal",
            literal="sequence-probability",
        ),
        "sequence_probabilities": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record", record=_token_sequence_probability_record),
        ),
    }
)


class PredictionResultValidationError(ValueError):
    """Raised when a prediction result record is invalid."""


@dataclass(frozen=True, slots=True)
class PredictionResultContract:
    """Runtime-neutral prediction result interface metadata."""

    prediction_space: object
    prediction_kind: str
    output_encoding: str

    def __post_init__(self) -> None:
        if type(self.prediction_kind) is not str or not self.prediction_kind:
            raise PredictionResultValidationError("prediction_kind must be nonempty")
        if type(self.output_encoding) is not str or not self.output_encoding:
            raise PredictionResultValidationError("output_encoding must be nonempty")

    @classmethod
    def from_prediction(cls, prediction: object) -> PredictionResultContract:
        prediction_space = getattr(prediction, "prediction_space", None)
        prediction_kind = getattr(prediction, "prediction_kind", None)
        output_encoding = getattr(prediction, "output_encoding", None)
        if prediction_space is None:
            raise PredictionResultValidationError(
                "prediction result does not expose prediction_space"
            )
        if not isinstance(prediction_kind, str) or not prediction_kind:
            raise PredictionResultValidationError(
                "prediction result does not expose prediction_kind"
            )
        if not isinstance(output_encoding, str) or not output_encoding:
            raise PredictionResultValidationError(
                "prediction result does not expose output_encoding"
            )
        return cls(
            prediction_space=prediction_space,
            prediction_kind=prediction_kind,
            output_encoding=output_encoding,
        )

    def require_matches(
        self,
        *,
        prediction_space: object,
        prediction_kind: str,
        output_encoding: str,
    ) -> None:
        if self.prediction_space != prediction_space:
            raise PredictionResultValidationError(
                "prediction_space does not match model interface"
            )
        if self.prediction_kind != prediction_kind:
            raise PredictionResultValidationError(
                "prediction_kind does not match model interface"
            )
        if self.output_encoding != output_encoding:
            raise PredictionResultValidationError(
                "output_encoding does not match model interface"
            )


@dataclass(frozen=True, slots=True)
class PredictionMass:
    """Probability assigned to one finite prediction-space index."""

    outcome_index: int
    probability: float

    def __post_init__(self) -> None:
        if type(self.outcome_index) is not int or self.outcome_index < 0:
            raise PredictionResultValidationError(
                "outcome_index must be a nonnegative integer"
            )
        if not math.isfinite(self.probability):
            raise PredictionResultValidationError("probability must be finite")
        if self.probability < 0:
            raise PredictionResultValidationError("probability must be nonnegative")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> PredictionMass:
        try:
            validated = _prediction_mass_record.validate(record)
        except ValueError as error:
            raise PredictionResultValidationError(str(error)) from error
        return cls(
            outcome_index=_as_int(validated["outcome_index"], field="outcome_index"),
            probability=float(cast(float | int, validated["probability"])),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "outcome_index": self.outcome_index,
            "probability": self.probability,
        }


@dataclass(frozen=True, slots=True)
class TokenSequenceProbability:
    """Probability assigned to one concrete token sequence."""

    tokens: tuple[int, ...]
    probability: float

    def __post_init__(self) -> None:
        if not self.tokens:
            raise PredictionResultValidationError("tokens must be nonempty")
        if not all(type(token) is int for token in self.tokens):
            raise PredictionResultValidationError("tokens must be integers")
        if not math.isfinite(self.probability):
            raise PredictionResultValidationError("probability must be finite")
        if self.probability < 0 or self.probability > 1:
            raise PredictionResultValidationError("probability must be in [0, 1]")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> TokenSequenceProbability:
        try:
            validated = _token_sequence_probability_record.validate(record)
        except ValueError as error:
            raise PredictionResultValidationError(str(error)) from error
        return cls(
            tokens=tuple(
                _as_int(token, field="tokens")
                for token in _as_tuple(validated["tokens"], field="tokens")
            ),
            probability=float(cast(float | int, validated["probability"])),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "tokens": list(self.tokens),
            "probability": self.probability,
        }


@dataclass(frozen=True, slots=True)
class TokenSequencePrediction:
    """A sparse exact-sequence probability prediction.

    The prediction space may be unbounded when sequences are variable length. A
    concrete prediction record only needs to expose the sequence probabilities
    being scored for evaluated observations.
    """

    id: ProtocolIdentifier
    prediction_space: FiniteTokenSequenceSpace
    sequence_probabilities: tuple[TokenSequenceProbability, ...]
    prediction_kind: str = "finite-token-sequence-probability"
    output_encoding: str = "sequence-probability"

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise PredictionResultValidationError(str(error)) from error
        if self.prediction_kind != "finite-token-sequence-probability":
            raise PredictionResultValidationError(
                f"unsupported prediction_kind: {self.prediction_kind}"
            )
        if self.output_encoding != "sequence-probability":
            raise PredictionResultValidationError(
                f"unsupported output_encoding: {self.output_encoding}"
            )
        sequences = tuple(item.tokens for item in self.sequence_probabilities)
        if len(set(sequences)) != len(sequences):
            raise PredictionResultValidationError("token sequences must be unique")
        object.__setattr__(
            self,
            "sequence_probabilities",
            tuple(sorted(self.sequence_probabilities, key=lambda item: item.tokens)),
        )
        for item in self.sequence_probabilities:
            try:
                self.prediction_space.require_sequence(item.tokens)
            except PredictionSpaceValidationError as error:
                raise PredictionResultValidationError(str(error)) from error

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> TokenSequencePrediction:
        try:
            validated = _token_sequence_prediction_record.validate(record)
            prediction_space = FiniteTokenSequenceSpace.from_record(
                _as_mapping(validated["prediction_space"], field="prediction_space")
            )
        except (ValueError, PredictionSpaceValidationError) as error:
            raise PredictionResultValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            prediction_space=prediction_space,
            prediction_kind=str(validated["prediction_kind"]),
            output_encoding=str(validated["output_encoding"]),
            sequence_probabilities=tuple(
                TokenSequenceProbability.from_record(
                    _as_mapping(item, field="sequence_probabilities")
                )
                for item in _as_tuple(
                    validated["sequence_probabilities"],
                    field="sequence_probabilities",
                )
            ),
        )

    @property
    def contract(self) -> PredictionResultContract:
        return PredictionResultContract(
            prediction_space=self.prediction_space,
            prediction_kind=self.prediction_kind,
            output_encoding=self.output_encoding,
        )

    def probability_of(self, tokens: Sequence[int]) -> float:
        token_values = tuple(_as_int(token, field="tokens") for token in tokens)
        try:
            self.prediction_space.require_sequence(token_values)
        except PredictionSpaceValidationError as error:
            raise PredictionResultValidationError(str(error)) from error
        for item in self.sequence_probabilities:
            if item.tokens == token_values:
                return item.probability
        return 0.0

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "prediction_space": self.prediction_space.to_record(),
            "prediction_kind": self.prediction_kind,
            "output_encoding": self.output_encoding,
            "sequence_probabilities": [
                item.to_record()
                for item in sorted(
                    self.sequence_probabilities,
                    key=lambda probability: probability.tokens,
                )
            ],
        }


@dataclass(frozen=True, slots=True)
class DirectFiniteProbabilityPrediction:
    """A direct finite probability prediction over an indexed finite space."""

    id: ProtocolIdentifier
    prediction_space: FiniteOutcomeSpace
    probabilities: tuple[PredictionMass, ...]
    prediction_kind: str = "direct-finite-probability-measure"
    output_encoding: str = "probability-mass-sequence"
    normalization_tolerance: float = field(
        default=1e-12,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise PredictionResultValidationError(str(error)) from error
        if self.prediction_kind != "direct-finite-probability-measure":
            raise PredictionResultValidationError(
                f"unsupported prediction_kind: {self.prediction_kind}"
            )
        if self.output_encoding != "probability-mass-sequence":
            raise PredictionResultValidationError(
                f"unsupported output_encoding: {self.output_encoding}"
            )
        if not math.isfinite(self.normalization_tolerance) or self.normalization_tolerance < 0:
            raise PredictionResultValidationError(
                "normalization tolerance must be finite and nonnegative"
            )
        outcome_indices = tuple(mass.outcome_index for mass in self.probabilities)
        if len(set(outcome_indices)) != len(outcome_indices):
            raise PredictionResultValidationError("outcome indices must be unique")
        outside = tuple(
            index
            for index in sorted(outcome_indices)
            if index >= self.prediction_space.outcome_count
        )
        if outside:
            raise PredictionResultValidationError(
                f"outcome_index {outside[0]} is outside prediction space"
            )
        total = self.total_probability
        if abs(total - 1.0) > self.normalization_tolerance:
            raise PredictionResultValidationError(
                "probabilities must sum to 1 within tolerance "
                f"{self.normalization_tolerance:g}; got {total:g}"
            )

    @classmethod
    def from_probabilities(
        cls,
        *,
        id: ProtocolIdentifier,
        prediction_space: FiniteOutcomeSpace,
        probabilities: Sequence[float],
        normalization_tolerance: float = 1e-12,
    ) -> DirectFiniteProbabilityPrediction:
        if len(probabilities) != prediction_space.outcome_count:
            raise PredictionResultValidationError(
                "probability sequence length "
                f"{len(probabilities)} does not match prediction space outcome_count "
                f"{prediction_space.outcome_count}"
            )
        return cls(
            id=id,
            prediction_space=prediction_space,
            probabilities=tuple(
                PredictionMass(outcome_index=index, probability=float(probability))
                for index, probability in enumerate(probabilities)
                if probability > 0
            ),
            normalization_tolerance=normalization_tolerance,
        )

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        outcome_space: OutcomeSpace,
        normalization_tolerance: float = 1e-12,
    ) -> DirectFiniteProbabilityPrediction:
        try:
            validated = _direct_finite_probability_prediction_record.validate(record)
            prediction_space = FiniteOutcomeSpace.from_record(
                _as_mapping(validated["prediction_space"], field="prediction_space")
            )
            prediction_space.validate_outcome_space(outcome_space)
        except (ValueError, PredictionSpaceValidationError) as error:
            raise PredictionResultValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            prediction_space=prediction_space,
            prediction_kind=str(validated["prediction_kind"]),
            output_encoding=str(validated["output_encoding"]),
            probabilities=tuple(
                PredictionMass.from_record(_as_mapping(mass, field="probabilities"))
                for mass in _as_tuple(validated["probabilities"], field="probabilities")
            ),
            normalization_tolerance=normalization_tolerance,
        )

    @classmethod
    def from_probability_measure(
        cls,
        *,
        prediction_space: FiniteOutcomeSpace,
        outcome_space: OutcomeSpace,
        measure: FiniteProbabilityMeasure,
        normalization_tolerance: float = 1e-12,
    ) -> DirectFiniteProbabilityPrediction:
        prediction_space.validate_outcome_space(outcome_space)
        if measure.outcome_space_id != outcome_space.id:
            raise PredictionResultValidationError(
                f"outcome_space_id {measure.outcome_space_id} does not match {outcome_space.id}"
            )
        probabilities_by_outcome = {
            mass.outcome_id: mass.probability for mass in measure.probabilities
        }
        return cls.from_probabilities(
            id=measure.id,
            prediction_space=prediction_space,
            probabilities=tuple(
                probabilities_by_outcome.get(outcome.id, 0.0)
                for outcome in outcome_space.outcomes
            ),
            normalization_tolerance=normalization_tolerance,
        )

    @property
    def total_probability(self) -> float:
        return math.fsum(mass.probability for mass in self.probabilities)

    @property
    def contract(self) -> PredictionResultContract:
        return PredictionResultContract(
            prediction_space=self.prediction_space,
            prediction_kind=self.prediction_kind,
            output_encoding=self.output_encoding,
        )

    def probability_at(self, outcome_index: int) -> float:
        if type(outcome_index) is not int or outcome_index < 0:
            raise PredictionResultValidationError(
                "outcome_index must be a nonnegative integer"
            )
        if outcome_index >= self.prediction_space.outcome_count:
            raise PredictionResultValidationError(
                f"outcome_index {outcome_index} is outside prediction space"
            )
        for mass in self.probabilities:
            if mass.outcome_index == outcome_index:
                return mass.probability
        return 0.0

    def to_probability_measure(self, *, outcome_space: OutcomeSpace) -> FiniteProbabilityMeasure:
        try:
            self.prediction_space.validate_outcome_space(outcome_space)
        except PredictionSpaceValidationError as error:
            raise PredictionResultValidationError(str(error)) from error
        return FiniteProbabilityMeasure(
            id=self.id,
            outcome_space_id=outcome_space.id,
            probabilities=tuple(
                ProbabilityMass(outcome.id, self.probability_at(index))
                for index, outcome in enumerate(outcome_space.outcomes)
                if self.probability_at(index) > 0
            ),
            normalization_tolerance=self.normalization_tolerance,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "prediction_space": self.prediction_space.to_record(),
            "prediction_kind": self.prediction_kind,
            "output_encoding": self.output_encoding,
            "probabilities": [
                mass.to_record()
                for mass in sorted(self.probabilities, key=lambda item: item.outcome_index)
            ],
        }


def _as_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise PredictionResultValidationError(f"{field}: expected integer")
    return value


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise PredictionResultValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PredictionResultValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_tuple(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise PredictionResultValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)
