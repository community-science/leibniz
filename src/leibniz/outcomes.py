"""Finite outcome-space declarations."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from leibniz.identifiers import ProtocolIdentifier
from leibniz.records import FieldSpec, RecordSpec, RecordValidationError

__all__ = [
    "AcceptedEvent",
    "AcceptedEventValidationError",
    "AcceptedMassScore",
    "AcceptedMassScoreError",
    "Outcome",
    "OutcomeSpace",
    "OutcomeSpaceValidationError",
    "FiniteOutcomeScoringGraph",
    "FiniteOutcomeScoringGraphValidationError",
    "FiniteProbabilityMeasure",
    "ProbabilityMass",
    "ProbabilityMeasureValidationError",
    "RawScoringEvidence",
    "RawScoringEvidenceValidationError",
]

_outcome_id_pattern = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")

_outcome_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="string"),
    }
)
_outcome_space_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "outcomes": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record", record=_outcome_record),
        ),
    }
)
_accepted_event_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "outcome_space_id": FieldSpec(kind="identifier"),
        "outcomes": FieldSpec(kind="sequence", item=FieldSpec(kind="string")),
    }
)
_probability_mass_record = RecordSpec(
    fields={
        "outcome_id": FieldSpec(kind="string"),
        "probability": FieldSpec(kind="number"),
    }
)
_finite_probability_measure_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "outcome_space_id": FieldSpec(kind="identifier"),
        "probabilities": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record", record=_probability_mass_record),
        ),
    }
)
_raw_scoring_evidence_base_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "observation_id": FieldSpec(kind="string"),
        "outcome_space_id": FieldSpec(kind="identifier"),
        "accepted_event_id": FieldSpec(kind="identifier"),
        "probability_measure_id": FieldSpec(kind="identifier"),
        "accepted_mass": FieldSpec(kind="number"),
    },
    allow_unknown=True,
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
_finite_outcome_scoring_graph_expected_fields = frozenset(
    {
        "id",
        "observation_id",
        "outcome_space",
        "accepted_event",
        "probability_measure",
        "raw_scoring_evidence",
    }
)


class OutcomeSpaceValidationError(ValueError):
    """Raised when an outcome or outcome space is invalid."""


class AcceptedEventValidationError(ValueError):
    """Raised when an accepted event is invalid."""


class ProbabilityMeasureValidationError(ValueError):
    """Raised when a finite probability measure is invalid."""


class AcceptedMassScoreError(ValueError):
    """Raised when accepted-mass scoring inputs are invalid."""


class RawScoringEvidenceValidationError(ValueError):
    """Raised when raw finite-outcome scoring evidence is invalid."""


class FiniteOutcomeScoringGraphValidationError(ValueError):
    """Raised when a finite-outcome scoring graph is invalid."""


@dataclass(frozen=True, slots=True)
class Outcome:
    """One possible outcome inside a finite outcome space."""

    id: str

    def __post_init__(self) -> None:
        if _outcome_id_pattern.fullmatch(self.id) is None:
            raise OutcomeSpaceValidationError(f"invalid outcome id: {self.id!r}")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> Outcome:
        try:
            validated = _outcome_record.validate(record)
        except RecordValidationError as error:
            raise OutcomeSpaceValidationError(str(error)) from error
        return cls(id=str(validated["id"]))

    def to_record(self) -> dict[str, object]:
        return {"id": self.id}


@dataclass(frozen=True, slots=True)
class OutcomeSpace:
    """A finite set of possible outcomes."""

    id: ProtocolIdentifier
    outcomes: tuple[Outcome, ...]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise OutcomeSpaceValidationError(str(error)) from error
        if not self.outcomes:
            raise OutcomeSpaceValidationError("outcome space must contain at least one outcome")
        outcome_ids = tuple(outcome.id for outcome in self.outcomes)
        if len(set(outcome_ids)) != len(outcome_ids):
            raise OutcomeSpaceValidationError("outcome ids must be unique")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> OutcomeSpace:
        try:
            validated = _outcome_space_record.validate(record)
        except RecordValidationError as error:
            raise OutcomeSpaceValidationError(str(error)) from error
        outcomes = tuple(
            Outcome.from_record(_as_mapping(outcome, field="outcomes"))
            for outcome in _as_tuple(validated["outcomes"], field="outcomes")
        )
        identifier = _as_identifier(validated["id"], field="id")
        return cls(id=identifier, outcomes=outcomes)

    @property
    def outcome_ids(self) -> frozenset[str]:
        return frozenset(outcome.id for outcome in self.outcomes)

    def contains(self, outcome_id: str) -> bool:
        return outcome_id in self.outcome_ids

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "outcomes": [outcome.to_record() for outcome in self.outcomes],
        }


@dataclass(frozen=True, slots=True)
class AcceptedEvent:
    """A nonempty subset of a finite outcome space."""

    id: ProtocolIdentifier
    outcome_space_id: ProtocolIdentifier
    outcomes: frozenset[str]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise AcceptedEventValidationError(str(error)) from error
        if not self.outcomes:
            raise AcceptedEventValidationError("accepted event must contain at least one outcome")
        for outcome_id in self.outcomes:
            if _outcome_id_pattern.fullmatch(outcome_id) is None:
                raise AcceptedEventValidationError(
                    f"invalid accepted outcome id: {outcome_id!r}"
                )

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], *, outcome_space: OutcomeSpace
    ) -> AcceptedEvent:
        try:
            validated = _accepted_event_record.validate(record)
        except RecordValidationError as error:
            raise AcceptedEventValidationError(str(error)) from error

        identifier = _as_identifier(validated["id"], field="id")
        outcome_space_id = _as_identifier(validated["outcome_space_id"], field="outcome_space_id")
        if outcome_space_id != outcome_space.id:
            raise AcceptedEventValidationError(
                f"outcome_space_id {outcome_space_id} does not match {outcome_space.id}"
            )

        outcome_ids = tuple(
            str(outcome) for outcome in _as_tuple(validated["outcomes"], field="outcomes")
        )
        if len(set(outcome_ids)) != len(outcome_ids):
            raise AcceptedEventValidationError("accepted outcome ids must be unique")

        event = cls(
            id=identifier,
            outcome_space_id=outcome_space_id,
            outcomes=frozenset(outcome_ids),
        )
        unknown = tuple(
            outcome_id
            for outcome_id in sorted(event.outcomes)
            if not outcome_space.contains(outcome_id)
        )
        if unknown:
            raise AcceptedEventValidationError(
                f"accepted outcomes are not in outcome space: {', '.join(unknown)}"
            )
        return event

    def accepts(self, outcome_id: str) -> bool:
        return outcome_id in self.outcomes

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "outcome_space_id": str(self.outcome_space_id),
            "outcomes": sorted(self.outcomes),
        }


@dataclass(frozen=True, slots=True)
class ProbabilityMass:
    """Probability assigned to one outcome."""

    outcome_id: str
    probability: float

    def __post_init__(self) -> None:
        if _outcome_id_pattern.fullmatch(self.outcome_id) is None:
            raise ProbabilityMeasureValidationError(
                f"invalid probability outcome id: {self.outcome_id!r}"
            )
        if not math.isfinite(self.probability):
            raise ProbabilityMeasureValidationError("probability must be finite")
        if self.probability < 0:
            raise ProbabilityMeasureValidationError("probability must be nonnegative")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ProbabilityMass:
        try:
            validated = _probability_mass_record.validate(record)
        except RecordValidationError as error:
            raise ProbabilityMeasureValidationError(str(error)) from error
        return cls(
            outcome_id=str(validated["outcome_id"]),
            probability=float(cast(float | int, validated["probability"])),
        )

    def to_record(self) -> dict[str, object]:
        return {"outcome_id": self.outcome_id, "probability": self.probability}


@dataclass(frozen=True, slots=True)
class FiniteProbabilityMeasure:
    """A normalized finite probability measure over an outcome space.

    Outcomes omitted from ``probabilities`` have zero probability mass.
    """

    id: ProtocolIdentifier
    outcome_space_id: ProtocolIdentifier
    probabilities: tuple[ProbabilityMass, ...]
    normalization_tolerance: float = field(
        default=1e-12,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise ProbabilityMeasureValidationError(str(error)) from error
        if not math.isfinite(self.normalization_tolerance) or self.normalization_tolerance < 0:
            raise ProbabilityMeasureValidationError(
                "normalization tolerance must be finite and nonnegative"
            )

        outcome_ids = tuple(mass.outcome_id for mass in self.probabilities)
        if len(set(outcome_ids)) != len(outcome_ids):
            raise ProbabilityMeasureValidationError("probability outcome ids must be unique")

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
        outcome_space: OutcomeSpace,
        normalization_tolerance: float = 1e-12,
    ) -> FiniteProbabilityMeasure:
        try:
            validated = _finite_probability_measure_record.validate(record)
        except RecordValidationError as error:
            raise ProbabilityMeasureValidationError(str(error)) from error

        identifier = _as_identifier(validated["id"], field="id")
        outcome_space_id = _as_identifier(validated["outcome_space_id"], field="outcome_space_id")
        if outcome_space_id != outcome_space.id:
            raise ProbabilityMeasureValidationError(
                f"outcome_space_id {outcome_space_id} does not match {outcome_space.id}"
            )

        probabilities = tuple(
            ProbabilityMass.from_record(_as_mapping(probability, field="probabilities"))
            for probability in _as_tuple(validated["probabilities"], field="probabilities")
        )
        unknown = tuple(
            outcome_id
            for outcome_id in sorted({mass.outcome_id for mass in probabilities})
            if not outcome_space.contains(outcome_id)
        )
        if unknown:
            raise ProbabilityMeasureValidationError(
                f"probability outcomes are not in outcome space: {', '.join(unknown)}"
            )
        return cls(
            id=identifier,
            outcome_space_id=outcome_space_id,
            probabilities=probabilities,
            normalization_tolerance=normalization_tolerance,
        )

    @property
    def total_probability(self) -> float:
        return math.fsum(mass.probability for mass in self.probabilities)

    def probability_of(self, outcome_id: str) -> float:
        for mass in self.probabilities:
            if mass.outcome_id == outcome_id:
                return mass.probability
        return 0.0

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "outcome_space_id": str(self.outcome_space_id),
            "probabilities": [
                mass.to_record()
                for mass in sorted(self.probabilities, key=lambda item: item.outcome_id)
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
        if event.outcome_space_id != measure.outcome_space_id:
            raise AcceptedMassScoreError(
                "accepted event outcome_space_id "
                f"{event.outcome_space_id} does not match probability measure "
                f"{measure.outcome_space_id}"
            )

        accepted_mass = math.fsum(
            measure.probability_of(outcome_id) for outcome_id in event.outcomes
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
    """Per-observation evidence for one finite-outcome score."""

    id: ProtocolIdentifier
    observation_id: str
    outcome_space_id: ProtocolIdentifier
    accepted_event_id: ProtocolIdentifier
    probability_measure_id: ProtocolIdentifier
    accepted_mass: float
    negative_log_score: float

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
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
            outcome_space_id=event.outcome_space_id,
            accepted_event_id=event.id,
            probability_measure_id=measure.id,
            accepted_mass=score.accepted_mass,
            negative_log_score=score.negative_log_score,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> RawScoringEvidence:
        try:
            validated = _raw_scoring_evidence_base_record.validate(record)
        except RecordValidationError as error:
            raise RawScoringEvidenceValidationError(str(error)) from error
        expected_fields = {
            "id",
            "observation_id",
            "outcome_space_id",
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
            outcome_space_id=_as_identifier(
                validated["outcome_space_id"], field="outcome_space_id"
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
            "outcome_space_id": str(self.outcome_space_id),
            "accepted_event_id": str(self.accepted_event_id),
            "probability_measure_id": str(self.probability_measure_id),
            "accepted_mass": self.accepted_mass,
            "negative_log_score": negative_log_score,
        }


@dataclass(frozen=True, slots=True)
class FiniteOutcomeScoringGraph:
    """A finite-outcome scoring graph for one observation."""

    outcome_space: OutcomeSpace
    accepted_event: AcceptedEvent
    probability_measure: FiniteProbabilityMeasure
    raw_scoring_evidence: RawScoringEvidence

    def __post_init__(self) -> None:
        _require_matching_identifier(
            field="accepted_event.outcome_space_id",
            actual=self.accepted_event.outcome_space_id,
            expected=self.outcome_space.id,
        )
        _require_matching_identifier(
            field="probability_measure.outcome_space_id",
            actual=self.probability_measure.outcome_space_id,
            expected=self.outcome_space.id,
        )
        _require_matching_identifier(
            field="raw_scoring_evidence.outcome_space_id",
            actual=self.raw_scoring_evidence.outcome_space_id,
            expected=self.outcome_space.id,
        )
        _require_matching_identifier(
            field="raw_scoring_evidence.accepted_event_id",
            actual=self.raw_scoring_evidence.accepted_event_id,
            expected=self.accepted_event.id,
        )
        _require_matching_identifier(
            field="raw_scoring_evidence.probability_measure_id",
            actual=self.raw_scoring_evidence.probability_measure_id,
            expected=self.probability_measure.id,
        )

        score = AcceptedMassScore.from_event_and_measure(
            event=self.accepted_event,
            measure=self.probability_measure,
        )
        if not math.isclose(
            self.raw_scoring_evidence.accepted_mass,
            score.accepted_mass,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise FiniteOutcomeScoringGraphValidationError(
                "raw_scoring_evidence.accepted_mass must equal recomputed accepted mass"
            )
        if not math.isclose(
            self.raw_scoring_evidence.negative_log_score,
            score.negative_log_score,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise FiniteOutcomeScoringGraphValidationError(
                "raw_scoring_evidence.negative_log_score must equal recomputed score"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> FiniteOutcomeScoringGraph:
        try:
            validated = _finite_outcome_scoring_graph_record.validate(record)
            outcome_space = OutcomeSpace.from_record(
                _graph_mapping(validated["outcome_space"], field="outcome_space")
            )
            accepted_event = AcceptedEvent.from_record(
                _graph_mapping(validated["accepted_event"], field="accepted_event"),
                outcome_space=outcome_space,
            )
            probability_measure = FiniteProbabilityMeasure.from_record(
                _graph_mapping(
                    validated["probability_measure"],
                    field="probability_measure",
                ),
                outcome_space=outcome_space,
            )
            raw_scoring_evidence = _graph_raw_scoring_evidence(
                record=record,
                validated=validated,
                accepted_event=accepted_event,
                probability_measure=probability_measure,
            )
        except ValueError as error:
            raise FiniteOutcomeScoringGraphValidationError(str(error)) from error
        return cls(
            outcome_space=outcome_space,
            accepted_event=accepted_event,
            probability_measure=probability_measure,
            raw_scoring_evidence=raw_scoring_evidence,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "outcome_space": self.outcome_space.to_record(),
            "accepted_event": self.accepted_event.to_record(),
            "probability_measure": self.probability_measure.to_record(),
            "raw_scoring_evidence": self.raw_scoring_evidence.to_record(),
        }


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise OutcomeSpaceValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_tuple(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise OutcomeSpaceValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise OutcomeSpaceValidationError(f"{field}: expected parsed identifier")
    return value


def _graph_raw_scoring_evidence(
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
        raise FiniteOutcomeScoringGraphValidationError(f"{unknown_fields[0]}: unknown field")

    raw_value = validated.get("raw_scoring_evidence")
    explicit: RawScoringEvidence | None = None
    if raw_value is not None:
        explicit = RawScoringEvidence.from_record(
            _graph_mapping(
                raw_value,
                field="raw_scoring_evidence",
            )
        )

    evidence_id = validated.get("id")
    observation_id = validated.get("observation_id")
    if evidence_id is None:
        if explicit is None:
            raise FiniteOutcomeScoringGraphValidationError("id: missing required field")
        evidence_id = explicit.id
    if observation_id is None:
        if explicit is None:
            raise FiniteOutcomeScoringGraphValidationError(
                "observation_id: missing required field"
            )
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
        raise FiniteOutcomeScoringGraphValidationError(
            "raw_scoring_evidence must equal derived scoring evidence"
        )
    return explicit


def _graph_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise FiniteOutcomeScoringGraphValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _require_matching_identifier(
    *,
    field: str,
    actual: ProtocolIdentifier,
    expected: ProtocolIdentifier,
) -> None:
    if actual != expected:
        raise FiniteOutcomeScoringGraphValidationError(
            f"{field} {actual} does not match {expected}"
        )
