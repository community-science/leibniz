"""Benchmark evaluation helpers independent of local training workflows."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementRecord
from leibniz.observation_generation import GeneratedSample, GeneratedSampleSet
from leibniz.outcomes import AcceptedEvent, OutcomeSpace, RawScoringEvidence
from leibniz.prediction_results import DirectFiniteProbabilityPrediction
from leibniz.prediction_spaces import FiniteOutcomeSpace

__all__ = [
    "CompetencePoint",
    "ComputeCostPoint",
    "finite_measurements_for_predictions",
    "integrated_compute_cost",
    "sampled_competence_curriculum_record",
    "sampled_competence_record",
    "sampled_competence_frontier_score",
    "ValidationCompetencePoint",
    "validation_competence",
    "validation_competence_frontier_advances",
    "validation_competence_frontier_score",
]


@dataclass(frozen=True, slots=True)
class CompetencePoint:
    """A measured point on a complexity frontier."""

    complexity: float
    accepted_mass: float
    sample_count: int = 1
    seed: int = 0
    complexity_minimum: float | None = None
    complexity_maximum: float | None = None


@dataclass(frozen=True, slots=True)
class ComputeCostPoint:
    """A measured per-sample compute density over a complexity interval."""

    complexity: float
    compute_per_sample: float
    bit_length_per_op: float
    complexity_minimum: float | None = None
    complexity_maximum: float | None = None


@dataclass(frozen=True, slots=True)
class ValidationCompetencePoint(CompetencePoint):
    """A validation-loss-derived point on a complexity frontier."""


def finite_measurements_for_predictions(
    *,
    batch: GeneratedSampleSet,
    outcome_space: OutcomeSpace,
    probabilities: tuple[tuple[float, ...], ...],
    run_slug: str,
) -> tuple[MeasurementRecord, ...]:
    """Return finite-outcome measurement records for evaluated predictions."""

    prediction_space = FiniteOutcomeSpace.from_outcome_space(outcome_space)
    measurements: list[MeasurementRecord] = []
    for sample, sample_probabilities in zip(batch.samples, probabilities, strict=True):
        observation_id = _sample_observation_id(batch=batch, sample=sample)
        accepted_event = AcceptedEvent.from_record(
            {
                "id": str(
                    _sample_identifier(
                        "events",
                        run_slug,
                        sample,
                        benchmark_id=batch.benchmark_id,
                    )
                ),
                "outcome_space_id": str(outcome_space.id),
                "outcomes": [sample.outcome_id],
            },
            outcome_space=outcome_space,
        )
        prediction = DirectFiniteProbabilityPrediction.from_probabilities(
            id=_sample_identifier(
                "measures",
                run_slug,
                sample,
                benchmark_id=batch.benchmark_id,
            ),
            prediction_space=prediction_space,
            probabilities=sample_probabilities,
        )
        probability_measure = prediction.to_probability_measure(
            outcome_space=outcome_space,
        )
        measurements.append(
            MeasurementRecord(
                benchmark_id=batch.benchmark_id,
                outcome_space=outcome_space,
                accepted_event=accepted_event,
                probability_measure=probability_measure,
                raw_scoring_evidence=RawScoringEvidence.from_event_and_measure(
                    id=_sample_identifier(
                        "evidence",
                        run_slug,
                        sample,
                        benchmark_id=batch.benchmark_id,
                    ),
                    observation_id=observation_id,
                    event=accepted_event,
                    measure=probability_measure,
                ),
            )
        )
    return tuple(measurements)


def sampled_competence_record(
    *,
    batch: GeneratedSampleSet,
    measurements: tuple[MeasurementRecord, ...],
    complexity_axis: str | None,
) -> dict[str, object]:
    """Return aggregate competence evidence for one sampled complexity class."""

    if len(batch.samples) != len(measurements):
        raise ValueError("sampled competence requires one measurement per sample")
    complexities = {sample.complexity for sample in batch.samples}
    if len(complexities) != 1:
        raise ValueError("sampled competence requires one complexity class")
    accepted_mass = tuple(
        measurement.raw_scoring_evidence.accepted_mass for measurement in measurements
    )
    finite_losses = tuple(
        measurement.raw_scoring_evidence.negative_log_score
        for measurement in measurements
        if math.isfinite(measurement.raw_scoring_evidence.negative_log_score)
    )
    mean_negative_log_score: float | str
    if len(finite_losses) != len(measurements):
        mean_negative_log_score = "infinity"
    else:
        mean_negative_log_score = math.fsum(finite_losses) / len(finite_losses)
    record: dict[str, object] = {
        "kind": "sampled-complexity-class",
        "sampling_rule": "generator-uniform-component-index-v1",
        "difficulty_assumption": "approximately-uniform-within-complexity-class",
        "benchmark_id": str(batch.benchmark_id),
        "complexity_axis": complexity_axis,
        "complexity": next(iter(complexities)),
        "seed": batch.seed,
        "sample_count": len(batch.samples),
        "mean_accepted_mass": math.fsum(accepted_mass) / len(accepted_mass),
        "mean_negative_log_score": mean_negative_log_score,
        "observation_ids": [
            measurement.raw_scoring_evidence.observation_id
            for measurement in measurements
        ],
        "measurement_ids": [
            str(measurement.raw_scoring_evidence.id) for measurement in measurements
        ],
    }
    if batch.complexity_request is not None:
        record["complexity_minimum"] = batch.complexity_request.minimum
        record["complexity_maximum"] = batch.complexity_request.maximum
    return record


def sampled_competence_curriculum_record(
    points: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return aggregate competence evidence for a progressive complexity sweep."""

    if not points:
        raise ValueError("sampled competence curriculum requires at least one point")
    sorted_points = tuple(
        sorted(
            points,
            key=lambda point: _finite_nonnegative_number(
                point.get("complexity"),
                field="sampled_competence.complexity",
            ),
        )
    )
    first = sorted_points[0]
    sample_counts = tuple(
        _positive_int(point.get("sample_count"), field="sampled_competence.sample_count")
        for point in sorted_points
    )
    accepted_masses = tuple(
        _finite_score(
            point.get("mean_accepted_mass"),
            field="sampled_competence.mean_accepted_mass",
        )
        for point in sorted_points
    )
    total_samples = sum(sample_counts)
    weighted_score = (
        math.fsum(
            mass * sample_count
            for mass, sample_count in zip(accepted_masses, sample_counts, strict=True)
        )
        / total_samples
    )
    return {
        "kind": "sampled-competence-curriculum",
        "sampling_rule": first.get("sampling_rule"),
        "difficulty_assumption": first.get("difficulty_assumption"),
        "benchmark_id": first.get("benchmark_id"),
        "complexity_axis": first.get("complexity_axis"),
        "complexity": first.get("complexity"),
        "sample_count": total_samples,
        "mean_accepted_mass": weighted_score,
        "points": [dict(point) for point in sorted_points],
    }


def validation_competence(*, validation_loss: float, outcome_count: int) -> float:
    """Convert finite-outcome cross-entropy to bounded local competence."""

    reference_cross_entropy = math.log(outcome_count)
    if reference_cross_entropy <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - validation_loss / reference_cross_entropy))


def sampled_competence_frontier_score(
    points: Sequence[CompetencePoint],
    *,
    chance_mass: float,
) -> float:
    """Return competence bits over the full frontier from zero complexity."""

    if not points:
        return 0.0
    ordered = tuple(sorted(points, key=lambda point: point.complexity))
    area = 0.0
    cursor = 0.0
    for point in sorted(ordered, key=_competence_point_interval_sort_key):
        lower, upper = _competence_point_interval(point)
        if lower > cursor:
            area += lower - cursor
        measured_lower = max(lower, cursor)
        if upper <= measured_lower:
            continue
        area += (upper - measured_lower) * _above_chance_competence(
            point.accepted_mass,
            chance_mass=chance_mass,
        )
        cursor = upper
    return area


def integrated_compute_cost(points: Sequence[ComputeCostPoint]) -> float:
    """Return bit-op cost integrated over observed complexity intervals."""

    total = 0.0
    for point in points:
        lower, upper = _complexity_point_interval(point)
        if upper <= lower:
            continue
        compute_per_sample = _finite_nonnegative_number(
            point.compute_per_sample,
            field="compute_cost.compute_per_sample",
        )
        bit_length_per_op = _finite_nonnegative_number(
            point.bit_length_per_op,
            field="compute_cost.bit_length_per_op",
        )
        total += (upper - lower) * compute_per_sample * bit_length_per_op
    return total


def _competence_point_interval_sort_key(point: CompetencePoint) -> tuple[float, float]:
    lower, upper = _complexity_point_interval(point)
    return (lower, upper)


def _competence_point_interval(point: CompetencePoint) -> tuple[float, float]:
    return _complexity_point_interval(point)


def _complexity_point_interval(
    point: CompetencePoint | ComputeCostPoint,
) -> tuple[float, float]:
    complexity = _finite_nonnegative_number(
        point.complexity,
        field="complexity_interval.complexity",
    )
    lower = (
        max(0.0, complexity - 1.0)
        if point.complexity_minimum is None
        else _finite_nonnegative_number(
            point.complexity_minimum,
            field="complexity_interval.complexity_minimum",
        )
    )
    upper = (
        complexity
        if point.complexity_maximum is None
        else _finite_nonnegative_number(
            point.complexity_maximum,
            field="complexity_interval.complexity_maximum",
        )
    )
    if upper < lower:
        raise ValueError("complexity interval maximum is below minimum")
    return (lower, upper)


def validation_competence_frontier_score(
    points: Sequence[ValidationCompetencePoint],
    *,
    chance_mass: float,
) -> float:
    """Return the benchmark frontier score for validation-derived points."""

    return sampled_competence_frontier_score(points, chance_mass=chance_mass)


def validation_competence_frontier_advances(
    *,
    frontier_point: ValidationCompetencePoint,
    previous_frontier_points: Sequence[ValidationCompetencePoint],
    chance_mass: float,
) -> bool:
    """Return whether a measured frontier point advances the benchmark ladder."""

    if frontier_point.accepted_mass <= chance_mass + 1e-12:
        return False
    previous_score = validation_competence_frontier_score(
        previous_frontier_points,
        chance_mass=chance_mass,
    )
    next_score = validation_competence_frontier_score(
        (*previous_frontier_points, frontier_point),
        chance_mass=chance_mass,
    )
    if next_score <= 0.0:
        return False
    return next_score > previous_score


def _above_chance_competence(score: float, *, chance_mass: float) -> float:
    chance = _finite_score(chance_mass, field="competence_frontier.chance_mass")
    if chance >= 1.0:
        return 0.0
    accepted_mass = _finite_score(score, field="competence_frontier.accepted_mass")
    return max(0.0, min(1.0, (accepted_mass - chance) / (1.0 - chance)))


def _positive_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _finite_score(value: object, *, field: str) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    score = float(value)
    if score < 0.0 or score > 1.0:
        raise ValueError(f"{field} must be between 0 and 1")
    return score


def _finite_nonnegative_number(value: object, *, field: str) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{field} must be finite")
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{field} must be nonnegative")
    return number


def _sample_identifier(
    family: str,
    run_slug: str,
    sample: GeneratedSample,
    *,
    benchmark_id: ProtocolIdentifier,
) -> ProtocolIdentifier:
    return _child_identifier(
        benchmark_id,
        f"{family}.{run_slug}.sample-{sample.index}",
    )


def _sample_observation_id(
    *,
    batch: GeneratedSampleSet,
    sample: GeneratedSample,
) -> str:
    if sample.observable_state_id is not None:
        return sample.observable_state_id
    return f"{batch.generator_id}.seed-{batch.seed}.sample-{sample.index}"


def _child_identifier(parent: ProtocolIdentifier, suffix: str) -> ProtocolIdentifier:
    return ProtocolIdentifier.parse(f"{parent.name}.{suffix}@{parent.version}")
