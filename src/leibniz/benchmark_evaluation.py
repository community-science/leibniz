"""Benchmark evaluation helpers independent of local training workflows."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self, TypeVar, cast

from leibniz.architectures import ArchitectureManifest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementRecord
from leibniz.model_operators import (
    architecture_with_input_shape,
    summarize_architecture_operators,
)
from leibniz.observation_generation import GeneratedSample, GeneratedSampleSet
from leibniz.outcomes import AcceptedEvent, OutcomeSpace, RawScoringEvidence
from leibniz.prediction_results import DirectFiniteProbabilityPrediction
from leibniz.prediction_spaces import FiniteOutcomeSpace
from leibniz.state_space import StateSpaceError, StateSpaceRegion, state_space_region_from_record
from leibniz.tensor_runtime import tensor_runtime_ops_bit_density

__all__ = [
    "CompetencePoint",
    "StateSpaceIntegral",
    "finite_measurements_for_predictions",
    "sampled_competence_curriculum_record",
    "sampled_competence_planning_cost_integral",
    "sampled_competence_record",
    "sampled_competence_frontier_integral",
    "StateSpaceIntegralTerm",
    "ValidationCompetencePoint",
    "validation_competence",
    "validation_competence_frontier_advances",
]

_ErrorT = TypeVar("_ErrorT", bound=ValueError)

@dataclass(frozen=True, slots=True)
class CompetencePoint:
    """A measured point on a state-space volume frontier."""

    log2_volume: float
    accepted_mass: float
    sample_count: int = 1
    seed: int = 0
    log2_volume_minimum: float | None = None
    log2_volume_maximum: float | None = None
    input_shape: tuple[int, ...] | None = None
    region: StateSpaceRegion | None = None

    @classmethod
    def from_sampled_record(
        cls,
        record: Mapping[str, object],
        *,
        field_prefix: str,
        error_type: type[_ErrorT] = ValueError,
    ) -> Self:
        seed = record.get("seed")
        return cls(
            log2_volume=_record_nonnegative_number(
                record.get("log2_volume"),
                field=f"{field_prefix}.log2_volume",
                error_type=error_type,
            ),
            accepted_mass=_record_nonnegative_number(
                record.get("mean_accepted_mass"),
                field=f"{field_prefix}.mean_accepted_mass",
                error_type=error_type,
            ),
            sample_count=_record_positive_int(
                record.get("sample_count"),
                field=f"{field_prefix}.sample_count",
                error_type=error_type,
            ),
            seed=seed if type(seed) is int else 0,
            log2_volume_minimum=_record_optional_nonnegative_number(
                record.get("log2_volume_minimum"),
                field=f"{field_prefix}.log2_volume_minimum",
                error_type=error_type,
            ),
            log2_volume_maximum=_record_optional_nonnegative_number(
                record.get("log2_volume_maximum"),
                field=f"{field_prefix}.log2_volume_maximum",
                error_type=error_type,
            ),
            input_shape=_record_optional_input_shape(
                record,
                field=f"{field_prefix}.input_shape",
                error_type=error_type,
            ),
            region=_record_optional_state_space_region(
                record.get("region"),
                field=f"{field_prefix}.region",
                error_type=error_type,
            ),
        )


@dataclass(frozen=True, slots=True)
class StateSpaceIntegralTerm:
    """One explicit interval contribution to an integral over state-space volume."""

    lower: float
    upper: float
    competence_density: float
    kind: str
    representative_log2_volume: float | None = None
    sample_count: int | None = None
    confidence_half_width: float | None = None
    region: StateSpaceRegion | None = None

    @property
    def width(self) -> float:
        return self.upper - self.lower

    @property
    def contribution(self) -> float:
        return self.width * self.competence_density

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": self.kind,
            "log2_volume_minimum": self.lower,
            "log2_volume_maximum": self.upper,
            "width_in_bits": self.width,
            "competence_density": self.competence_density,
            "contribution": self.contribution,
        }
        if self.representative_log2_volume is not None:
            record["representative_log2_volume"] = self.representative_log2_volume
        if self.sample_count is not None:
            record["sample_count"] = self.sample_count
        if self.confidence_half_width is not None:
            record["confidence_half_width"] = self.confidence_half_width
        if self.region is not None:
            record["region"] = self.region.to_record()
        return record


@dataclass(frozen=True, slots=True)
class StateSpaceIntegral:
    """A human-readable numerical integral over the log2 state-space volume axis."""

    terms: tuple[StateSpaceIntegralTerm, ...]

    @property
    def value(self) -> float:
        return math.fsum(term.contribution for term in self.terms)

    def to_record(self, *, kind: str) -> dict[str, object]:
        return {
            "kind": kind,
            "value": self.value,
            "terms": [term.to_record() for term in self.terms],
        }


@dataclass(frozen=True, slots=True)
class ValidationCompetencePoint(CompetencePoint):
    """A validation-loss-derived point on a state-space volume frontier."""


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
                "outcomes": list(sample.target_distribution_or_one_hot()),
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
    volume_axis: str | None,
    input_shape: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Return aggregate competence evidence for one sampled state-space volume window."""

    if len(batch.samples) != len(measurements):
        raise ValueError("sampled competence requires one measurement per sample")
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
        "kind": "sampled-state-space-volume-window",
        "sampling_rule": "generator-uniform-component-index-v1",
        "difficulty_assumption": "approximately-uniform-within-volume-window",
        "benchmark_id": str(batch.benchmark_id),
        "volume_axis": volume_axis,
        "log2_volume": batch.log2_volume,
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
    resolved_input_shape = input_shape
    if resolved_input_shape is None:
        resolved_input_shape = _sampled_input_shape(batch)
    record["input_shape"] = list(resolved_input_shape)
    if batch.volume_request is not None:
        record["log2_volume_minimum"] = batch.volume_request.minimum
        record["log2_volume_maximum"] = batch.volume_request.maximum
    if batch.region is not None:
        record["region"] = batch.region.to_record()
    return record


def _sampled_input_shape(batch: GeneratedSampleSet) -> tuple[int, ...]:
    tensor_shape = getattr(batch.fields, "shape", None)
    if tensor_shape is not None and len(tensor_shape) >= 2:
        input_shape: list[int] = []
        for axis in tuple(tensor_shape)[1:]:
            if type(axis) is not int or axis < 1:
                input_shape = []
                break
            input_shape.append(axis)
        if input_shape:
            return tuple(input_shape)
    sample_shapes = {
        sample.field.shape
        for sample in batch.samples
        if sample.field is not None
    }
    if len(sample_shapes) == 1:
        return next(iter(sample_shapes))
    raise ValueError("sampled competence requires an inspectable input shape")


def sampled_competence_curriculum_record(
    points: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return aggregate competence evidence for a progressive state-space volume sweep."""

    if not points:
        raise ValueError("sampled competence curriculum requires at least one point")
    sorted_points = tuple(
        sorted(
            points,
            key=lambda point: _finite_nonnegative_number(
                point.get("log2_volume"),
                field="sampled_competence.log2_volume",
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
        "volume_axis": first.get("volume_axis"),
        "log2_volume": first.get("log2_volume"),
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


def sampled_competence_frontier_integral(
    points: Sequence[CompetencePoint],
    *,
    chance_mass: float,
) -> StateSpaceIntegral:
    """Return the explicit competence integral over measured log2-volume intervals."""

    terms: list[StateSpaceIntegralTerm] = []
    cursor = 0.0
    for point in sorted(points, key=_competence_point_interval_sort_key):
        lower, upper = _log2_volume_point_interval(point)
        measured_lower = max(lower, cursor)
        if upper > measured_lower:
            terms.append(
                StateSpaceIntegralTerm(
                    lower=measured_lower,
                    upper=upper,
                    competence_density=_above_chance_competence(
                        point.accepted_mass,
                        chance_mass=chance_mass,
                    ),
                    kind="measured-state-space-competence",
                    representative_log2_volume=point.log2_volume,
                    sample_count=point.sample_count,
                    region=point.region,
                )
            )
        cursor = max(cursor, lower, upper)
    return StateSpaceIntegral(terms=tuple(terms))


def sampled_competence_planning_cost_integral(
    *,
    points: Sequence[Mapping[str, object]],
    architecture: ArchitectureManifest,
    error_type: type[_ErrorT] = ValueError,
    field_prefix: str = "compute_cost_point",
) -> StateSpaceIntegral:
    """Return a planning compute-cost integral for sampled competence intervals."""

    ordered = sorted(
        points,
        key=lambda point: _record_nonnegative_number(
            point.get("log2_volume"),
            field=f"{field_prefix}.log2_volume",
            error_type=error_type,
        ),
    )
    cursor = 0.0
    terms: list[StateSpaceIntegralTerm] = []
    for point in ordered:
        log2_volume = _record_nonnegative_number(
            point.get("log2_volume"),
            field=f"{field_prefix}.log2_volume",
            error_type=error_type,
        )
        minimum = _record_optional_nonnegative_number(
            point.get("log2_volume_minimum"),
            field=f"{field_prefix}.log2_volume_minimum",
            error_type=error_type,
        )
        maximum = _record_optional_nonnegative_number(
            point.get("log2_volume_maximum"),
            field=f"{field_prefix}.log2_volume_maximum",
            error_type=error_type,
        )
        if minimum is None:
            minimum = cursor
        if maximum is None:
            maximum = log2_volume
        if maximum <= minimum:
            minimum = cursor
            maximum = log2_volume
        if maximum > minimum:
            terms.append(
                StateSpaceIntegralTerm(
                    lower=minimum,
                    upper=maximum,
                    competence_density=tensor_runtime_ops_bit_density(
                        _sampled_point_inference_compute(
                            point,
                            architecture=architecture,
                            error_type=error_type,
                            field_prefix=field_prefix,
                        )
                    ),
                    kind="planning-compute-cost",
                    representative_log2_volume=log2_volume,
                )
            )
        cursor = max(cursor, maximum)
    return StateSpaceIntegral(terms=tuple(terms))


def _competence_point_interval_sort_key(point: CompetencePoint) -> tuple[float, float]:
    lower, upper = _log2_volume_point_interval(point)
    return (lower, upper)


def _sampled_point_inference_compute(
    point: Mapping[str, object],
    *,
    architecture: ArchitectureManifest,
    error_type: type[_ErrorT],
    field_prefix: str,
) -> float:
    plan = summarize_architecture_operators(
        architecture_with_input_shape(
            architecture,
            _sampled_point_input_shape(
                point,
                error_type=error_type,
                field=f"{field_prefix}.input_shape",
            ),
        )
    )
    if plan.inference_compute is None:
        raise error_type(f"{field_prefix} input_shape has unknown inference compute")
    return float(plan.inference_compute)


def _sampled_point_input_shape(
    point: Mapping[str, object],
    *,
    error_type: type[_ErrorT],
    field: str,
) -> tuple[int, ...]:
    value = point.get("input_shape")
    if not isinstance(value, list | tuple):
        raise error_type(f"{field}: expected shape")
    shape: list[int] = []
    for axis in cast(Sequence[object], value):
        if type(axis) is not int or axis < 1:
            raise error_type(f"{field}: expected positive integer shape")
        shape.append(axis)
    if not shape:
        raise error_type(f"{field}: expected nonempty shape")
    return tuple(shape)


def _record_optional_nonnegative_number(
    value: object,
    *,
    field: str,
    error_type: type[_ErrorT],
) -> float | None:
    if value is None:
        return None
    return _record_nonnegative_number(value, field=field, error_type=error_type)


def _record_positive_int(
    value: object,
    *,
    field: str,
    error_type: type[_ErrorT],
) -> int:
    if type(value) is not int or value < 1:
        raise error_type(f"{field}: expected positive integer")
    return value


def _record_nonnegative_number(
    value: object,
    *,
    field: str,
    error_type: type[_ErrorT],
) -> float:
    if not isinstance(value, int | float):
        raise error_type(f"{field}: expected number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise error_type(f"{field}: expected finite nonnegative number")
    return result


def _record_optional_input_shape(
    point: Mapping[str, object],
    *,
    field: str,
    error_type: type[_ErrorT],
) -> tuple[int, ...] | None:
    if "input_shape" not in point:
        return None
    return _sampled_point_input_shape(point, field=field, error_type=error_type)


def _record_optional_state_space_region(
    value: object,
    *,
    field: str,
    error_type: type[_ErrorT],
) -> StateSpaceRegion | None:
    if value is None:
        return None
    try:
        return state_space_region_from_record(value)
    except StateSpaceError as error:
        raise error_type(f"{field}: {error}") from error


def _log2_volume_point_interval(
    point: CompetencePoint,
) -> tuple[float, float]:
    log2_volume = _finite_nonnegative_number(
        point.log2_volume,
        field="log2_volume_interval.log2_volume",
    )
    lower = (
        max(0.0, log2_volume - 1.0)
        if point.log2_volume_minimum is None
        else _finite_nonnegative_number(
            point.log2_volume_minimum,
            field="log2_volume_interval.log2_volume_minimum",
        )
    )
    upper = (
        log2_volume
        if point.log2_volume_maximum is None
        else _finite_nonnegative_number(
            point.log2_volume_maximum,
            field="log2_volume_interval.log2_volume_maximum",
        )
    )
    if upper < lower:
        raise ValueError("volume interval maximum is below minimum")
    return (lower, upper)


def validation_competence_frontier_advances(
    *,
    frontier_point: ValidationCompetencePoint,
    previous_frontier_points: Sequence[ValidationCompetencePoint],
    chance_mass: float,
) -> bool:
    """Return whether a measured frontier point advances the benchmark ladder."""

    return _above_chance_competence(
        frontier_point.accepted_mass,
        chance_mass=chance_mass,
    ) > 0.0


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
