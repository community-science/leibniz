"""Deterministic acquisition scoring over candidate observations."""

from __future__ import annotations

import math
from dataclasses import dataclass

from leibniz.candidate_observations import ArchitectureCandidateObservation

__all__ = [
    "AcquisitionScoringError",
    "CandidateAcquisitionScore",
    "score_candidate_acquisition",
]

_model_name = "frontier-resource-gap"


class AcquisitionScoringError(ValueError):
    """Raised when candidate acquisition scores cannot be computed."""


@dataclass(frozen=True, slots=True)
class CandidateAcquisitionScore:
    """Evidence-derived score components for one architecture candidate."""

    observation: ArchitectureCandidateObservation
    model_name: str
    chance_score: float
    best_observed_score: float
    estimated_score: float
    uncertainty: float
    exploration_value: float
    resource_novelty: float
    expected_frontier_improvement: float
    comparable_cost_best_score: float
    acquisition_value: float

    def __post_init__(self) -> None:
        if not self.model_name:
            raise AcquisitionScoringError("model_name must be nonempty")
        for field in (
            "chance_score",
            "best_observed_score",
            "estimated_score",
            "uncertainty",
            "exploration_value",
            "resource_novelty",
            "expected_frontier_improvement",
            "comparable_cost_best_score",
            "acquisition_value",
        ):
            _require_nonnegative(getattr(self, field), field=field)
        for field in (
            "chance_score",
            "best_observed_score",
            "estimated_score",
            "uncertainty",
            "resource_novelty",
            "comparable_cost_best_score",
        ):
            _require_probability(getattr(self, field), field=field)

    def to_component_record(self) -> dict[str, object]:
        """Return auditable scalar components used to produce acquisition_value."""

        return {
            "chance_score": self.chance_score,
            "best_observed_score": self.best_observed_score,
            "estimated_score": self.estimated_score,
            "uncertainty": self.uncertainty,
            "exploration_value": self.exploration_value,
            "resource_novelty": self.resource_novelty,
            "expected_frontier_improvement": self.expected_frontier_improvement,
            "comparable_cost_best_score": self.comparable_cost_best_score,
            "acquisition_value": self.acquisition_value,
        }


def score_candidate_acquisition(
    observations: tuple[ArchitectureCandidateObservation, ...],
    *,
    output_count: int,
) -> tuple[CandidateAcquisitionScore, ...]:
    """Score candidate observations with a deterministic frontier-gap rule."""

    if type(output_count) is not int or output_count < 1:
        raise AcquisitionScoringError("output_count must be positive")
    if not observations:
        raise AcquisitionScoringError("observations must contain at least one item")

    observed_parameters = tuple(
        observation.parameter_count for observation in observations if observation.is_measured
    )
    best_score = max(
        (observation.best_measured_score for observation in observations),
        default=0.0,
    )
    chance_score = 1.0 / output_count
    return tuple(
        _score_observation(
            observation,
            observed_parameters=observed_parameters,
            best_score=best_score,
            chance_score=chance_score,
        )
        for observation in observations
    )


def _score_observation(
    observation: ArchitectureCandidateObservation,
    *,
    observed_parameters: tuple[int, ...],
    best_score: float,
    chance_score: float,
) -> CandidateAcquisitionScore:
    resource_novelty = _resource_novelty(observation.parameter_count, observed_parameters)
    uncertainty = min(1.0, 0.2 + resource_novelty / 2.0)
    estimated_score = min(1.0, max(best_score, chance_score) + resource_novelty * 0.05)
    frontier_improvement = max(
        0.0,
        estimated_score - observation.best_measured_score_at_or_below_cost,
    )
    exploration_value = uncertainty * 0.25
    acquisition_value = estimated_score + exploration_value + resource_novelty * 0.1
    return CandidateAcquisitionScore(
        observation=observation,
        model_name=_model_name,
        chance_score=chance_score,
        best_observed_score=best_score,
        estimated_score=estimated_score,
        uncertainty=uncertainty,
        exploration_value=exploration_value,
        resource_novelty=resource_novelty,
        expected_frontier_improvement=frontier_improvement,
        comparable_cost_best_score=observation.best_measured_score_at_or_below_cost,
        acquisition_value=acquisition_value,
    )


def _resource_novelty(parameter_count: int, observed_parameters: tuple[int, ...]) -> float:
    if not observed_parameters:
        return 1.0
    distances = tuple(
        abs(math.log1p(parameter_count) - math.log1p(observed))
        for observed in observed_parameters
    )
    return min(1.0, min(distances))


def _require_probability(value: float, *, field: str) -> None:
    _require_nonnegative(value, field=field)
    if value > 1:
        raise AcquisitionScoringError(f"{field} must not exceed 1")


def _require_nonnegative(value: float, *, field: str) -> None:
    if value < 0 or not math.isfinite(value):
        raise AcquisitionScoringError(f"{field} must be finite and nonnegative")
