"""Finite answer-space declarations."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from leibniz.identifiers import ProtocolIdentifier, require_unreleased_identifier
from leibniz.records import RecordSpec, RecordValidationError, required, validate_record

_element_id_pattern = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")

_answer_element_record = RecordSpec(
    fields={
        "id": required("string"),
    }
)
_answer_space_record = RecordSpec(
    fields={
        "id": required("identifier"),
        "elements": required("sequence", item=required("record", record=_answer_element_record)),
    }
)
_accepted_event_record = RecordSpec(
    fields={
        "id": required("identifier"),
        "answer_space_id": required("identifier"),
        "elements": required("sequence", item=required("string")),
    }
)
_probability_mass_record = RecordSpec(
    fields={
        "element_id": required("string"),
        "probability": required("number"),
    }
)
_finite_probability_measure_record = RecordSpec(
    fields={
        "id": required("identifier"),
        "answer_space_id": required("identifier"),
        "probabilities": required(
            "sequence",
            item=required("record", record=_probability_mass_record),
        ),
    }
)
_raw_scoring_evidence_base_record = RecordSpec(
    fields={
        "id": required("identifier"),
        "observation_id": required("string"),
        "answer_space_id": required("identifier"),
        "accepted_event_id": required("identifier"),
        "probability_measure_id": required("identifier"),
        "accepted_mass": required("number"),
    },
    allow_unknown=True,
)


class AnswerSpaceValidationError(ValueError):
    """Raised when an answer element or answer space is invalid."""


class AcceptedEventValidationError(ValueError):
    """Raised when an accepted event is invalid."""


class ProbabilityMeasureValidationError(ValueError):
    """Raised when a finite probability measure is invalid."""


class AcceptedMassScoreError(ValueError):
    """Raised when accepted-mass scoring inputs are invalid."""


class RawScoringEvidenceValidationError(ValueError):
    """Raised when raw finite-answer scoring evidence is invalid."""


@dataclass(frozen=True, slots=True)
class AnswerElement:
    """One possible answer inside a finite answer space."""

    id: str

    def __post_init__(self) -> None:
        if _element_id_pattern.fullmatch(self.id) is None:
            raise AnswerSpaceValidationError(f"invalid answer element id: {self.id!r}")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> AnswerElement:
        try:
            validated = validate_record(record, _answer_element_record)
        except RecordValidationError as error:
            raise AnswerSpaceValidationError(str(error)) from error
        return cls(id=str(validated["id"]))

    def to_record(self) -> dict[str, object]:
        return {"id": self.id}


@dataclass(frozen=True, slots=True)
class AnswerSpace:
    """A finite set of possible answers."""

    id: ProtocolIdentifier
    elements: tuple[AnswerElement, ...]

    def __post_init__(self) -> None:
        try:
            require_unreleased_identifier(self.id)
        except ValueError as error:
            raise AnswerSpaceValidationError(str(error)) from error
        if not self.elements:
            raise AnswerSpaceValidationError("answer space must contain at least one element")
        element_ids = tuple(element.id for element in self.elements)
        if len(set(element_ids)) != len(element_ids):
            raise AnswerSpaceValidationError("answer element ids must be unique")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> AnswerSpace:
        try:
            validated = validate_record(record, _answer_space_record)
        except RecordValidationError as error:
            raise AnswerSpaceValidationError(str(error)) from error
        elements = tuple(
            AnswerElement.from_record(_as_mapping(element, field="elements"))
            for element in _as_tuple(validated["elements"], field="elements")
        )
        identifier = _as_identifier(validated["id"], field="id")
        return cls(id=identifier, elements=elements)

    @property
    def element_ids(self) -> frozenset[str]:
        return frozenset(element.id for element in self.elements)

    def contains(self, element_id: str) -> bool:
        return element_id in self.element_ids

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "elements": [element.to_record() for element in self.elements],
        }


@dataclass(frozen=True, slots=True)
class AcceptedEvent:
    """A nonempty subset of a finite answer space."""

    id: ProtocolIdentifier
    answer_space_id: ProtocolIdentifier
    elements: frozenset[str]

    def __post_init__(self) -> None:
        try:
            require_unreleased_identifier(self.id)
        except ValueError as error:
            raise AcceptedEventValidationError(str(error)) from error
        if not self.elements:
            raise AcceptedEventValidationError("accepted event must contain at least one element")
        for element_id in self.elements:
            if _element_id_pattern.fullmatch(element_id) is None:
                raise AcceptedEventValidationError(
                    f"invalid accepted element id: {element_id!r}"
                )

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], *, answer_space: AnswerSpace
    ) -> AcceptedEvent:
        try:
            validated = validate_record(record, _accepted_event_record)
        except RecordValidationError as error:
            raise AcceptedEventValidationError(str(error)) from error

        identifier = _as_identifier(validated["id"], field="id")
        answer_space_id = _as_identifier(validated["answer_space_id"], field="answer_space_id")
        if answer_space_id != answer_space.id:
            raise AcceptedEventValidationError(
                f"answer_space_id {answer_space_id} does not match {answer_space.id}"
            )

        element_ids = tuple(
            str(element) for element in _as_tuple(validated["elements"], field="elements")
        )
        if len(set(element_ids)) != len(element_ids):
            raise AcceptedEventValidationError("accepted element ids must be unique")

        event = cls(
            id=identifier,
            answer_space_id=answer_space_id,
            elements=frozenset(element_ids),
        )
        unknown = tuple(
            element_id
            for element_id in sorted(event.elements)
            if not answer_space.contains(element_id)
        )
        if unknown:
            raise AcceptedEventValidationError(
                f"accepted elements are not in answer space: {', '.join(unknown)}"
            )
        return event

    def accepts(self, element_id: str) -> bool:
        return element_id in self.elements

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "answer_space_id": str(self.answer_space_id),
            "elements": sorted(self.elements),
        }


@dataclass(frozen=True, slots=True)
class ProbabilityMass:
    """Probability assigned to one answer element."""

    element_id: str
    probability: float

    def __post_init__(self) -> None:
        if _element_id_pattern.fullmatch(self.element_id) is None:
            raise ProbabilityMeasureValidationError(
                f"invalid probability element id: {self.element_id!r}"
            )
        if not math.isfinite(self.probability):
            raise ProbabilityMeasureValidationError("probability must be finite")
        if self.probability < 0:
            raise ProbabilityMeasureValidationError("probability must be nonnegative")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ProbabilityMass:
        try:
            validated = validate_record(record, _probability_mass_record)
        except RecordValidationError as error:
            raise ProbabilityMeasureValidationError(str(error)) from error
        return cls(
            element_id=str(validated["element_id"]),
            probability=float(cast(float | int, validated["probability"])),
        )

    def to_record(self) -> dict[str, object]:
        return {"element_id": self.element_id, "probability": self.probability}


@dataclass(frozen=True, slots=True)
class FiniteProbabilityMeasure:
    """A normalized finite probability measure over an answer space.

    Elements omitted from ``probabilities`` have zero probability mass.
    """

    id: ProtocolIdentifier
    answer_space_id: ProtocolIdentifier
    probabilities: tuple[ProbabilityMass, ...]
    normalization_tolerance: float = field(
        default=1e-12,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        try:
            require_unreleased_identifier(self.id)
        except ValueError as error:
            raise ProbabilityMeasureValidationError(str(error)) from error
        if not math.isfinite(self.normalization_tolerance) or self.normalization_tolerance < 0:
            raise ProbabilityMeasureValidationError(
                "normalization tolerance must be finite and nonnegative"
            )

        element_ids = tuple(mass.element_id for mass in self.probabilities)
        if len(set(element_ids)) != len(element_ids):
            raise ProbabilityMeasureValidationError("probability element ids must be unique")

        total = self.total_probability
        if abs(total - 1.0) > self.normalization_tolerance:
            raise ProbabilityMeasureValidationError(
                "probabilities must sum to 1 within tolerance "
                f"{self.normalization_tolerance:g}; got {total:g}"
            )

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        answer_space: AnswerSpace,
        normalization_tolerance: float = 1e-12,
    ) -> FiniteProbabilityMeasure:
        try:
            validated = validate_record(record, _finite_probability_measure_record)
        except RecordValidationError as error:
            raise ProbabilityMeasureValidationError(str(error)) from error

        identifier = _as_identifier(validated["id"], field="id")
        answer_space_id = _as_identifier(validated["answer_space_id"], field="answer_space_id")
        if answer_space_id != answer_space.id:
            raise ProbabilityMeasureValidationError(
                f"answer_space_id {answer_space_id} does not match {answer_space.id}"
            )

        probabilities = tuple(
            ProbabilityMass.from_record(_as_mapping(probability, field="probabilities"))
            for probability in _as_tuple(validated["probabilities"], field="probabilities")
        )
        unknown = tuple(
            element_id
            for element_id in sorted({mass.element_id for mass in probabilities})
            if not answer_space.contains(element_id)
        )
        if unknown:
            raise ProbabilityMeasureValidationError(
                f"probability elements are not in answer space: {', '.join(unknown)}"
            )
        return cls(
            id=identifier,
            answer_space_id=answer_space_id,
            probabilities=probabilities,
            normalization_tolerance=normalization_tolerance,
        )

    @property
    def total_probability(self) -> float:
        return math.fsum(mass.probability for mass in self.probabilities)

    def probability_of(self, element_id: str) -> float:
        for mass in self.probabilities:
            if mass.element_id == element_id:
                return mass.probability
        return 0.0

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "answer_space_id": str(self.answer_space_id),
            "probabilities": [
                mass.to_record()
                for mass in sorted(self.probabilities, key=lambda item: item.element_id)
            ],
        }


@dataclass(frozen=True, slots=True)
class AcceptedMassScore:
    """Accepted probability mass and negative-log score for one event."""

    accepted_mass: float
    negative_log_score: float

    @classmethod
    def from_event_and_measure(
        cls,
        *,
        event: AcceptedEvent,
        measure: FiniteProbabilityMeasure,
    ) -> AcceptedMassScore:
        if event.answer_space_id != measure.answer_space_id:
            raise AcceptedMassScoreError(
                "accepted event answer_space_id "
                f"{event.answer_space_id} does not match probability measure "
                f"{measure.answer_space_id}"
            )

        accepted_mass = math.fsum(
            measure.probability_of(element_id) for element_id in event.elements
        )
        if accepted_mass < 0:
            raise AcceptedMassScoreError("accepted mass must be nonnegative")
        if accepted_mass > 1.0 and math.isclose(
            accepted_mass,
            1.0,
            rel_tol=measure.normalization_tolerance,
            abs_tol=measure.normalization_tolerance,
        ):
            accepted_mass = 1.0
        elif accepted_mass > 1.0:
            raise AcceptedMassScoreError(
                "accepted mass must not exceed 1 within measure normalization tolerance"
            )
        if accepted_mass == 0.0:
            return cls(accepted_mass=0.0, negative_log_score=math.inf)
        return cls(accepted_mass=accepted_mass, negative_log_score=-math.log(accepted_mass))


@dataclass(frozen=True, slots=True)
class RawScoringEvidence:
    """Per-observation evidence for one finite-answer score."""

    id: ProtocolIdentifier
    observation_id: str
    answer_space_id: ProtocolIdentifier
    accepted_event_id: ProtocolIdentifier
    probability_measure_id: ProtocolIdentifier
    accepted_mass: float
    negative_log_score: float

    def __post_init__(self) -> None:
        try:
            require_unreleased_identifier(self.id)
        except ValueError as error:
            raise RawScoringEvidenceValidationError(str(error)) from error
        if not self.observation_id:
            raise RawScoringEvidenceValidationError("observation_id must be nonempty")
        if not math.isfinite(self.accepted_mass):
            raise RawScoringEvidenceValidationError("accepted_mass must be finite")
        if self.accepted_mass < 0:
            raise RawScoringEvidenceValidationError("accepted_mass must be nonnegative")
        if self.accepted_mass > 1:
            raise RawScoringEvidenceValidationError("accepted_mass must not exceed 1")
        if self.accepted_mass == 0:
            if self.negative_log_score != math.inf:
                raise RawScoringEvidenceValidationError(
                    "zero accepted_mass requires infinite negative_log_score"
                )
            return
        if not math.isfinite(self.negative_log_score):
            raise RawScoringEvidenceValidationError("negative_log_score must be finite")
        expected_score = -math.log(self.accepted_mass)
        if not math.isclose(
            self.negative_log_score,
            expected_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise RawScoringEvidenceValidationError(
                "negative_log_score must equal -log(accepted_mass)"
            )

    @classmethod
    def from_event_and_measure(
        cls,
        *,
        id: ProtocolIdentifier,
        observation_id: str,
        event: AcceptedEvent,
        measure: FiniteProbabilityMeasure,
    ) -> RawScoringEvidence:
        score = AcceptedMassScore.from_event_and_measure(event=event, measure=measure)
        return cls(
            id=id,
            observation_id=observation_id,
            answer_space_id=event.answer_space_id,
            accepted_event_id=event.id,
            probability_measure_id=measure.id,
            accepted_mass=score.accepted_mass,
            negative_log_score=score.negative_log_score,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> RawScoringEvidence:
        try:
            validated = validate_record(record, _raw_scoring_evidence_base_record)
        except RecordValidationError as error:
            raise RawScoringEvidenceValidationError(str(error)) from error
        expected_fields = {
            "id",
            "observation_id",
            "answer_space_id",
            "accepted_event_id",
            "probability_measure_id",
            "accepted_mass",
            "negative_log_score",
        }
        unknown_fields = tuple(sorted(field for field in record if field not in expected_fields))
        if unknown_fields:
            raise RawScoringEvidenceValidationError(
                f"{unknown_fields[0]}: unknown field"
            )
        if "negative_log_score" not in record:
            raise RawScoringEvidenceValidationError(
                "negative_log_score: missing required field"
            )

        score_value = record["negative_log_score"]
        if score_value == "infinity":
            negative_log_score = math.inf
        elif isinstance(score_value, int) and not isinstance(score_value, bool):
            negative_log_score = float(score_value)
        elif isinstance(score_value, float) and math.isfinite(score_value):
            negative_log_score = score_value
        else:
            raise RawScoringEvidenceValidationError(
                "negative_log_score: expected finite number or 'infinity'"
            )

        return cls(
            id=_as_identifier(validated["id"], field="id"),
            observation_id=str(validated["observation_id"]),
            answer_space_id=_as_identifier(
                validated["answer_space_id"], field="answer_space_id"
            ),
            accepted_event_id=_as_identifier(
                validated["accepted_event_id"], field="accepted_event_id"
            ),
            probability_measure_id=_as_identifier(
                validated["probability_measure_id"], field="probability_measure_id"
            ),
            accepted_mass=float(cast(float | int, validated["accepted_mass"])),
            negative_log_score=negative_log_score,
        )

    def to_record(self) -> dict[str, object]:
        negative_log_score: float | str
        if self.negative_log_score == math.inf:
            negative_log_score = "infinity"
        else:
            negative_log_score = self.negative_log_score
        return {
            "id": str(self.id),
            "observation_id": self.observation_id,
            "answer_space_id": str(self.answer_space_id),
            "accepted_event_id": str(self.accepted_event_id),
            "probability_measure_id": str(self.probability_measure_id),
            "accepted_mass": self.accepted_mass,
            "negative_log_score": negative_log_score,
        }


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AnswerSpaceValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_tuple(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise AnswerSpaceValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise AnswerSpaceValidationError(f"{field}: expected parsed identifier")
    return value
