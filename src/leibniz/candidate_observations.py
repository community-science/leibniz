"""Projection records for architecture candidates and measured evidence."""

from __future__ import annotations

from dataclasses import dataclass

from leibniz.architecture_candidates import ArchitectureCandidate
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier

__all__ = [
    "ArchitectureCandidateObservation",
    "ArchitectureMeasurementEvidence",
    "CandidateObservationProjectionError",
    "project_architecture_candidate_observations",
]


class CandidateObservationProjectionError(ValueError):
    """Raised when candidate observations cannot be projected consistently."""


@dataclass(frozen=True, slots=True)
class ArchitectureMeasurementEvidence:
    """Measured score and cost evidence for one architecture digest."""

    architecture_digest: ContentDigest
    score: float
    parameter_count: int

    def __post_init__(self) -> None:
        if self.score < 0 or self.score > 1:
            raise CandidateObservationProjectionError("score must be a probability")
        if type(self.parameter_count) is not int or self.parameter_count < 0:
            raise CandidateObservationProjectionError("parameter_count must be nonnegative")


@dataclass(frozen=True, slots=True)
class ArchitectureCandidateObservation:
    """One selector-ready projection of a generated architecture candidate."""

    candidate: ArchitectureCandidate
    source_candidate_rank: int
    candidate_id: ProtocolIdentifier
    architecture_digest: ContentDigest
    operator_kinds: tuple[str, ...]
    parameter_count: int
    inference_flops: int | None
    is_measured: bool
    measured_score: float | None
    best_measured_score_at_or_below_cost: float
    best_measured_score: float

    def __post_init__(self) -> None:
        if type(self.source_candidate_rank) is not int or self.source_candidate_rank < 1:
            raise CandidateObservationProjectionError("source_candidate_rank must be positive")
        if self.candidate_id != self.candidate.architecture.id:
            raise CandidateObservationProjectionError("candidate_id must match candidate")
        if self.architecture_digest != self.candidate.architecture.digest:
            raise CandidateObservationProjectionError("architecture_digest must match candidate")
        if not self.operator_kinds or any(not operator for operator in self.operator_kinds):
            raise CandidateObservationProjectionError("operator_kinds must be nonempty")
        if type(self.parameter_count) is not int or self.parameter_count < 0:
            raise CandidateObservationProjectionError("parameter_count must be nonnegative")
        if self.inference_flops is not None and (
            type(self.inference_flops) is not int or self.inference_flops < 0
        ):
            raise CandidateObservationProjectionError("inference_flops must be nonnegative")
        if self.is_measured != (self.measured_score is not None):
            raise CandidateObservationProjectionError(
                "is_measured must agree with measured_score"
            )
        if self.measured_score is not None and (
            self.measured_score < 0 or self.measured_score > 1
        ):
            raise CandidateObservationProjectionError("measured_score must be a probability")
        _require_probability(
            self.best_measured_score_at_or_below_cost,
            field="best_measured_score_at_or_below_cost",
        )
        _require_probability(self.best_measured_score, field="best_measured_score")


def project_architecture_candidate_observations(
    candidates: tuple[ArchitectureCandidate, ...],
    *,
    measured: tuple[ArchitectureMeasurementEvidence, ...] = (),
) -> tuple[ArchitectureCandidateObservation, ...]:
    """Project generated candidates and measured evidence into selector observations."""

    if not candidates:
        raise CandidateObservationProjectionError("candidates must contain at least one item")
    measured_by_digest = _measured_by_digest(measured)
    best_measured_score = max(
        (evidence.score for evidence in measured_by_digest.values()),
        default=0.0,
    )
    observations: list[ArchitectureCandidateObservation] = []
    for rank, candidate in enumerate(candidates, start=1):
        parameter_count = candidate.parameter_count
        measured_evidence = measured_by_digest.get(candidate.architecture.digest)
        measured_score = None if measured_evidence is None else measured_evidence.score
        observations.append(
            ArchitectureCandidateObservation(
                candidate=candidate,
                source_candidate_rank=rank,
                candidate_id=candidate.architecture.id,
                architecture_digest=candidate.architecture.digest,
                operator_kinds=tuple(
                    operator.descriptor.kind for operator in candidate.operator_plan.operators
                ),
                parameter_count=parameter_count,
                inference_flops=candidate.operator_plan.inference_flops,
                is_measured=measured_evidence is not None,
                measured_score=measured_score,
                best_measured_score_at_or_below_cost=_best_score_at_or_below_cost(
                    measured_by_digest,
                    parameter_count,
                ),
                best_measured_score=best_measured_score,
            )
        )
    return tuple(observations)


def _measured_by_digest(
    measured: tuple[ArchitectureMeasurementEvidence, ...],
) -> dict[ContentDigest, ArchitectureMeasurementEvidence]:
    by_digest: dict[ContentDigest, ArchitectureMeasurementEvidence] = {}
    for evidence in measured:
        previous = by_digest.get(evidence.architecture_digest)
        if previous is not None and evidence.parameter_count != previous.parameter_count:
            raise CandidateObservationProjectionError(
                "conflicting parameter_count for measured architecture"
            )
        if previous is None or evidence.score > previous.score:
            by_digest[evidence.architecture_digest] = evidence
    return by_digest


def _best_score_at_or_below_cost(
    measured_by_digest: dict[ContentDigest, ArchitectureMeasurementEvidence],
    parameter_count: int,
) -> float:
    return max(
        (
            evidence.score
            for evidence in measured_by_digest.values()
            if evidence.parameter_count <= parameter_count
        ),
        default=0.0,
    )


def _require_probability(value: float, *, field: str) -> None:
    if value < 0 or value > 1:
        raise CandidateObservationProjectionError(f"{field} must be a probability")
