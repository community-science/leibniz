"""Benchmark evaluation helpers independent of local training workflows."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from leibniz.artifacts import ArtifactReference
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementRecord
from leibniz.observation_generation import GeneratedObservationBatch, GeneratedObservationSample
from leibniz.outcomes import AcceptedEvent, OutcomeSpace, RawScoringEvidence
from leibniz.prediction_results import DirectFiniteProbabilityPrediction
from leibniz.prediction_spaces import FiniteOutcomeSpace

__all__ = [
    "finite_measurements_for_predictions",
    "sampled_competence_curriculum_record",
    "sampled_competence_record",
    "validation_competence",
]


def finite_measurements_for_predictions(
    *,
    batch: GeneratedObservationBatch,
    outcome_space: OutcomeSpace,
    probabilities: tuple[tuple[float, ...], ...],
    run_slug: str,
) -> tuple[MeasurementRecord, ...]:
    """Return finite-outcome measurement records for evaluated predictions."""

    prediction_space = FiniteOutcomeSpace.from_outcome_space(outcome_space)
    measurements: list[MeasurementRecord] = []
    for sample, sample_probabilities in zip(batch.samples, probabilities, strict=True):
        accepted_event = AcceptedEvent.from_record(
            {
                "id": str(_sample_identifier("events", run_slug, sample)),
                "outcome_space_id": str(outcome_space.id),
                "outcomes": [sample.outcome_id],
            },
            outcome_space=outcome_space,
        )
        prediction = DirectFiniteProbabilityPrediction.from_probabilities(
            id=_sample_identifier("measures", run_slug, sample),
            prediction_space=prediction_space,
            probabilities=sample_probabilities,
        )
        probability_measure = prediction.to_probability_measure(
            outcome_space=outcome_space,
        )
        measurements.append(
            MeasurementRecord(
                benchmark_id=sample.observation.benchmark_id,
                outcome_space=outcome_space,
                accepted_event=accepted_event,
                probability_measure=probability_measure,
                raw_scoring_evidence=RawScoringEvidence.from_event_and_measure(
                    id=_sample_identifier("evidence", run_slug, sample),
                    observation_id=str(sample.observation.id),
                    event=accepted_event,
                    measure=probability_measure,
                ),
                evidence_artifacts=(
                    sample.observation.formation_declaration,
                    sample.observation.materialization_plan,
                    ArtifactReference(
                        kind="formed-observation",
                        protocol_id=sample.observation.id,
                        record_digest=sample.observation.digest,
                    ),
                ),
            )
        )
    return tuple(measurements)


def sampled_competence_record(
    *,
    batch: GeneratedObservationBatch,
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
    return {
        "kind": "sampled-complexity-class",
        "sampling_rule": "generator-uniform-component-sequence-v1",
        "difficulty_assumption": "approximately-uniform-within-complexity-class",
        "benchmark_id": str(batch.benchmark_id),
        "component_count": batch.component_count,
        "complexity_axis": complexity_axis,
        "complexity": next(iter(complexities)),
        "seed": batch.seed,
        "sample_count": len(batch.samples),
        "mean_accepted_mass": math.fsum(accepted_mass) / len(accepted_mass),
        "mean_negative_log_score": mean_negative_log_score,
        "observation_ids": [str(sample.observation.id) for sample in batch.samples],
        "measurement_ids": [
            str(measurement.raw_scoring_evidence.id) for measurement in measurements
        ],
    }


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
        "component_count": first.get("component_count"),
        "complexity_axis": first.get("complexity_axis"),
        "complexity": first.get("complexity"),
        "sample_count": total_samples,
        "mean_accepted_mass": weighted_score,
        "points": [dict(point) for point in sorted_points],
    }


def validation_competence(*, best_validation_loss: float, outcome_count: int) -> float:
    """Convert finite-outcome cross-entropy to bounded local competence."""

    reference_cross_entropy = math.log(outcome_count)
    if reference_cross_entropy <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - best_validation_loss / reference_cross_entropy))


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
    sample: GeneratedObservationSample,
) -> ProtocolIdentifier:
    return _child_identifier(
        sample.observation.benchmark_id,
        f"{family}.{run_slug}.sample-{sample.index}",
    )


def _child_identifier(parent: ProtocolIdentifier, suffix: str) -> ProtocolIdentifier:
    return ProtocolIdentifier.parse(f"{parent.name}.{suffix}@{parent.version}")
