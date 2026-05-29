"""Prediction target-space declarations for benchmark contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from leibniz.identifiers import ProtocolIdentifier
from leibniz.outcomes import Outcome, OutcomeSpace
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "FiniteOutcomeSpace",
    "FiniteTokenSequenceSpace",
    "FiniteTokenVocabulary",
    "PredictionSpaceValidationError",
    "RealVectorSpace",
]

_SequenceBoundary: TypeAlias = Literal["fixed-length"]
_RealCoordinateMeasure: TypeAlias = Literal["lebesgue"]

_token_vocabulary_record = RecordSpec(
    fields={
        "token_count": FieldSpec(kind="integer"),
        "token_name": FieldSpec(kind="string"),
    }
)
_finite_token_sequence_space_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="literal", literal="finite-token-sequence"),
        "vocabulary": FieldSpec(kind="record", record=_token_vocabulary_record),
        "length": FieldSpec(kind="integer"),
        "sequence_boundary": FieldSpec(kind="literal", literal="fixed-length"),
    }
)
_real_vector_space_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="literal", literal="real-vector"),
        "dimension": FieldSpec(kind="integer"),
        "coordinate_name": FieldSpec(kind="string"),
        "measure": FieldSpec(kind="literal", literal="lebesgue"),
    }
)
_finite_outcome_space_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="literal", literal="finite-outcome-space"),
        "outcome_space_id": FieldSpec(kind="identifier"),
        "outcome_count": FieldSpec(kind="integer"),
        "source_space": FieldSpec(kind="record", required=False),
    }
)


class PredictionSpaceValidationError(ValueError):
    """Raised when a prediction target-space declaration is invalid."""


@dataclass(frozen=True, slots=True)
class FiniteTokenVocabulary:
    """A finite vocabulary of integer-indexed tokens."""

    token_count: int
    token_name: str

    def __post_init__(self) -> None:
        if type(self.token_count) is not int:
            raise PredictionSpaceValidationError("token_count must be an integer")
        if self.token_count < 2:
            raise PredictionSpaceValidationError("token_count must be at least 2")
        if not self.token_name:
            raise PredictionSpaceValidationError("token_name must be nonempty")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> FiniteTokenVocabulary:
        try:
            validated = _token_vocabulary_record.validate(record)
        except ValueError as error:
            raise PredictionSpaceValidationError(str(error)) from error
        return cls(
            token_count=_as_int(validated["token_count"], field="token_count"),
            token_name=str(validated["token_name"]),
        )

    def contains(self, token: int) -> bool:
        return type(token) is int and 0 <= token < self.token_count

    def require_token(self, token: int) -> int:
        if not self.contains(token):
            raise PredictionSpaceValidationError(
                f"token {token} is outside 0..{self.token_count - 1}"
            )
        return token

    def to_record(self) -> dict[str, object]:
        return {
            "token_count": self.token_count,
            "token_name": self.token_name,
        }


@dataclass(frozen=True, slots=True)
class FiniteTokenSequenceSpace:
    """Fixed-length sequences over a finite token vocabulary."""

    vocabulary: FiniteTokenVocabulary
    length: int
    sequence_boundary: _SequenceBoundary = "fixed-length"

    def __post_init__(self) -> None:
        if type(self.length) is not int or self.length < 1:
            raise PredictionSpaceValidationError("length must be a positive integer")
        if self.sequence_boundary != "fixed-length":
            raise PredictionSpaceValidationError("sequence_boundary must be fixed-length")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> FiniteTokenSequenceSpace:
        try:
            validated = _finite_token_sequence_space_record.validate(record)
        except ValueError as error:
            raise PredictionSpaceValidationError(str(error)) from error
        return cls(
            vocabulary=FiniteTokenVocabulary.from_record(
                _as_mapping(validated["vocabulary"], field="vocabulary")
            ),
            length=_as_int(validated["length"], field="length"),
            sequence_boundary=cast(_SequenceBoundary, validated["sequence_boundary"]),
        )

    @property
    def cardinality(self) -> int:
        return self.vocabulary.token_count**self.length

    def sequence_index(self, tokens: Sequence[int]) -> int:
        """Return the lexicographic outcome index for a token sequence."""

        token_values = tuple(_as_int(token, field="tokens") for token in tokens)
        if len(token_values) != self.length:
            raise PredictionSpaceValidationError(
                f"token sequence length must be {self.length}"
            )
        index = 0
        for token in token_values:
            index = index * self.vocabulary.token_count + self.vocabulary.require_token(token)
        return index

    def sequence_for_index(self, index: int) -> tuple[int, ...]:
        """Return the token sequence at one lexicographic outcome index."""

        if type(index) is not int or index < 0 or index >= self.cardinality:
            raise PredictionSpaceValidationError("sequence index is outside token space")
        tokens = [0] * self.length
        cursor = index
        for position in range(self.length - 1, -1, -1):
            tokens[position] = cursor % self.vocabulary.token_count
            cursor //= self.vocabulary.token_count
        return tuple(tokens)

    def outcome_id(self, tokens: Sequence[int]) -> str:
        token_values = tuple(_as_int(token, field="tokens") for token in tokens)
        self.sequence_index(token_values)
        return "-".join((self.vocabulary.token_name, *(str(token) for token in token_values)))

    def outcome_space(self, *, id: ProtocolIdentifier) -> OutcomeSpace:
        return OutcomeSpace(
            id=id,
            outcomes=tuple(
                Outcome(self.outcome_id(self.sequence_for_index(index)))
                for index in range(self.cardinality)
            ),
        )

    def finite_outcome_space(self, *, id: ProtocolIdentifier) -> FiniteOutcomeSpace:
        """Return the finite outcome prediction space induced by this sequence space."""

        return FiniteOutcomeSpace(
            outcome_space_id=id,
            outcome_count=self.cardinality,
            source_space=self.to_record(),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "finite-token-sequence",
            "vocabulary": self.vocabulary.to_record(),
            "length": self.length,
            "sequence_boundary": self.sequence_boundary,
        }


@dataclass(frozen=True, slots=True)
class FiniteOutcomeSpace:
    """A finite prediction target space with explicitly enumerated outcomes."""

    outcome_space_id: ProtocolIdentifier
    outcome_count: int
    source_space: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        try:
            self.outcome_space_id.require_unreleased()
        except ValueError as error:
            raise PredictionSpaceValidationError(str(error)) from error
        if type(self.outcome_count) is not int or self.outcome_count < 1:
            raise PredictionSpaceValidationError("outcome_count must be a positive integer")
        if self.source_space is not None:
            _validate_source_space(self.source_space)

    @classmethod
    def from_outcome_space(
        cls,
        outcome_space: OutcomeSpace,
        *,
        source_space: Mapping[str, object] | None = None,
    ) -> FiniteOutcomeSpace:
        return cls(
            outcome_space_id=outcome_space.id,
            outcome_count=len(outcome_space.outcomes),
            source_space=source_space,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> FiniteOutcomeSpace:
        try:
            validated = _finite_outcome_space_record.validate(record)
        except ValueError as error:
            raise PredictionSpaceValidationError(str(error)) from error
        return cls(
            outcome_space_id=_as_identifier(
                validated["outcome_space_id"],
                field="outcome_space_id",
            ),
            outcome_count=_as_int(validated["outcome_count"], field="outcome_count"),
            source_space=(
                None
                if "source_space" not in validated
                else _as_mapping(validated["source_space"], field="source_space")
            ),
        )

    def validate_outcome_space(self, outcome_space: OutcomeSpace) -> None:
        if self.outcome_space_id != outcome_space.id:
            raise PredictionSpaceValidationError(
                f"outcome_space_id {self.outcome_space_id} does not match {outcome_space.id}"
            )
        if self.outcome_count != len(outcome_space.outcomes):
            raise PredictionSpaceValidationError(
                f"outcome_count {self.outcome_count} does not match "
                f"{len(outcome_space.outcomes)} outcomes"
            )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": "finite-outcome-space",
            "outcome_space_id": str(self.outcome_space_id),
            "outcome_count": self.outcome_count,
        }
        if self.source_space is not None:
            record["source_space"] = dict(self.source_space)
        return record


@dataclass(frozen=True, slots=True)
class RealVectorSpace:
    """A continuous real-valued prediction target space."""

    dimension: int
    coordinate_name: str = "value"
    measure: _RealCoordinateMeasure = "lebesgue"

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension < 1:
            raise PredictionSpaceValidationError("dimension must be a positive integer")
        if not self.coordinate_name:
            raise PredictionSpaceValidationError("coordinate_name must be nonempty")
        if self.measure != "lebesgue":
            raise PredictionSpaceValidationError("measure must be lebesgue")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> RealVectorSpace:
        try:
            validated = _real_vector_space_record.validate(record)
        except ValueError as error:
            raise PredictionSpaceValidationError(str(error)) from error
        return cls(
            dimension=_as_int(validated["dimension"], field="dimension"),
            coordinate_name=str(validated["coordinate_name"]),
            measure=cast(_RealCoordinateMeasure, validated["measure"]),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "real-vector",
            "dimension": self.dimension,
            "coordinate_name": self.coordinate_name,
            "measure": self.measure,
        }


def _as_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise PredictionSpaceValidationError(f"{field}: expected integer")
    return value


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise PredictionSpaceValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PredictionSpaceValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _validate_source_space(record: Mapping[str, object]) -> None:
    kind = record.get("kind")
    if kind == "finite-token-sequence":
        FiniteTokenSequenceSpace.from_record(record)
        return
    raise PredictionSpaceValidationError("source_space kind is unsupported")
