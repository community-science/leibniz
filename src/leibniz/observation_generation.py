"""Runtime generation of benchmark samples."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

from leibniz.benchmark_implementations import Generator as BenchmarkGenerator
from leibniz.benchmark_implementations import load_benchmark
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import (
    AxisAssignment,
    MaterializationPlan,
)
from leibniz.observation_formation import (
    FieldObservation,
    FormedObservation,
)

__all__ = [
    "GeneratedSample",
    "GeneratedSampleSet",
    "ObservationGenerationError",
    "ComplexityCandidate",
    "ComplexityRequest",
    "ComplexityValue",
    "load_generator",
]

_core_complexity_measure_id = "log2_complexity_class_size"
_core_complexity_measure_ids = frozenset({_core_complexity_measure_id})
_minimum_complexity_value = 1.0


def _empty_metadata() -> Mapping[str, object]:
    return {}


class ObservationGenerationError(ValueError):
    """Raised when observation generation cannot satisfy benchmark declarations."""


@dataclass(frozen=True, slots=True)
class ComplexityRequest:
    """A core complexity interval requested by a runner."""

    minimum: float
    maximum: float
    measure_id: str = _core_complexity_measure_id

    def __post_init__(self) -> None:
        if not self.measure_id:
            raise ObservationGenerationError("complexity id must be nonempty")
        if self.measure_id not in _core_complexity_measure_ids:
            raise ObservationGenerationError("complexity id is not a core measure")
        if not math.isfinite(float(self.minimum)):
            raise ObservationGenerationError("complexity minimum must be finite")
        if not math.isfinite(float(self.maximum)):
            raise ObservationGenerationError("complexity maximum must be finite")
        if self.minimum < _minimum_complexity_value:
            raise ObservationGenerationError(
                "complexity minimum must be at least 1"
            )
        if self.maximum < self.minimum:
            raise ObservationGenerationError(
                "complexity maximum must be at least the minimum"
            )

    def contains(self, value: ComplexityValue) -> bool:
        """Return whether a measured sample satisfies this interval."""

        if value.measure_id != self.measure_id:
            return False
        return self.minimum <= value.value <= self.maximum

    def to_record(self) -> dict[str, object]:
        """Return a record for this request."""

        return {
            "measure_id": self.measure_id,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }


@dataclass(frozen=True, slots=True)
class ComplexityValue:
    """A generated sample's value for a core complexity."""

    value: float
    measure_id: str = _core_complexity_measure_id

    def __post_init__(self) -> None:
        if not self.measure_id:
            raise ObservationGenerationError("complexity id must be nonempty")
        if self.measure_id not in _core_complexity_measure_ids:
            raise ObservationGenerationError("complexity id is not a core measure")
        if not math.isfinite(float(self.value)):
            raise ObservationGenerationError("complexity value must be finite")
        if self.value < _minimum_complexity_value:
            raise ObservationGenerationError(
                "complexity value must be at least 1"
            )

    def to_record(self) -> dict[str, object]:
        """Return a record for this measured value."""

        return {
            "measure_id": self.measure_id,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class ComplexityCandidate:
    """A benchmark-owned concrete complexity class for curriculum sampling."""

    request: ComplexityRequest
    cardinality: int | None = None
    resolution_assignment: AxisAssignment | None = None
    metadata: Mapping[str, object] = dataclass_field(default_factory=_empty_metadata)

    def __post_init__(self) -> None:
        if self.request.minimum != self.request.maximum:
            raise ObservationGenerationError(
                "complexity candidate request must name one realized cardinality"
            )
        if self.cardinality is not None:
            if type(self.cardinality) is not int or self.cardinality < 1:
                raise ObservationGenerationError(
                    "complexity candidate cardinality must be a positive integer"
                )
            if not math.isclose(
                math.log2(self.cardinality),
                self.complexity,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ObservationGenerationError(
                    "complexity candidate cardinality must match log2 cardinality"
                )

    @property
    def complexity(self) -> float:
        """Return the candidate's realized log2 cardinality."""

        return self.request.minimum

    def to_record(self) -> dict[str, object]:
        """Return a record for this benchmark-owned complexity candidate."""

        record: dict[str, object] = {
            "request": self.request.to_record(),
            "complexity": self.complexity,
        }
        if self.cardinality is not None:
            record["cardinality"] = self.cardinality
        if self.resolution_assignment is not None:
            record["resolution_assignment"] = self.resolution_assignment.to_record()
        if self.metadata:
            record["metadata"] = dict(self.metadata)
        return record


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    """One generated sample from a benchmark data source."""

    index: int
    outcome_id: str
    complexity: float
    complexity_value: ComplexityValue | None = None
    observable_state_id: str | None = None
    target_distribution: Mapping[str, float] | None = None
    latent_coordinates: tuple[Mapping[str, object], ...] = ()
    materialization_plan: MaterializationPlan | None = None
    width: int | None = None
    height: int | None = None
    component_index: int | None = None
    variation_coordinates: tuple[Mapping[str, object], ...] = ()
    variation_values: Mapping[str, object] | None = None
    field: FieldObservation | None = None
    _field_record: FormedObservation | None = dataclass_field(
        default=None,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.outcome_id:
            raise ObservationGenerationError("sample outcome_id must be nonempty")
        if self.observable_state_id is not None and not self.observable_state_id:
            raise ObservationGenerationError("sample observable_state_id must be nonempty")
        if self.target_distribution is not None:
            _validate_target_distribution(self.target_distribution)

    def require_field(self) -> FieldObservation:
        """Return the generated field or fail with a domain error."""

        if self.field is None:
            raise ObservationGenerationError("sample does not include generated field data")
        return self.field

    def target_distribution_or_one_hot(self) -> Mapping[str, float]:
        """Return the sample target distribution, defaulting to its accepted outcome."""

        if self.target_distribution is not None:
            return self.target_distribution
        return {self.outcome_id: 1.0}

    def field_record(self) -> FormedObservation:
        """Return the backing generated-field record for evidence plumbing."""

        if self._field_record is None:
            raise ObservationGenerationError("sample does not include generated field data")
        return self._field_record

    def to_record(self, *, include_field: bool = False) -> dict[str, object]:
        """Return a record for this generated sample."""

        record: dict[str, object] = {
            "index": self.index,
            "outcome_id": self.outcome_id,
            "complexity": self.complexity,
            "latent_coordinates": [dict(coordinate) for coordinate in self.latent_coordinates],
        }
        if self.materialization_plan is not None:
            record["materialization_plan"] = self.materialization_plan.to_record()
        if self.width is not None:
            record["width"] = self.width
        if self.height is not None:
            record["height"] = self.height
        if self.component_index is not None:
            record["component_index"] = self.component_index
        if self.observable_state_id is not None:
            record["observable_state_id"] = self.observable_state_id
        if self.target_distribution is not None:
            record["target_distribution"] = [
                {"outcome_id": outcome_id, "probability": probability}
                for outcome_id, probability in self.target_distribution.items()
            ]
        if self.variation_coordinates:
            record["variation_coordinates"] = [
                dict(item) for item in self.variation_coordinates
            ]
        if self.variation_values is not None:
            record["variation_values"] = dict(self.variation_values)
        if self.complexity_value is not None:
            record["complexity_value"] = self.complexity_value.to_record()
        if self.field is not None and include_field:
            record["field"] = self.field.to_record()
        return record


@dataclass(frozen=True, slots=True)
class GeneratedSampleSet:
    """Tensor-first shaped samples returned by a reusable data generator."""

    benchmark_id: ProtocolIdentifier
    generator_id: ProtocolIdentifier
    generator_version: str
    seed: int
    shape: tuple[int, ...]
    samples: tuple[GeneratedSample, ...] = ()
    fields: Any | None = None
    targets: Any | None = None
    variation_extent: float | None = None
    complexity_request: ComplexityRequest | None = None

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        if self.variation_extent is not None:
            if not math.isfinite(float(self.variation_extent)):
                raise ObservationGenerationError("variation_extent must be finite")
            if self.variation_extent < 0.0 or self.variation_extent > 1.0:
                raise ObservationGenerationError("variation_extent must be between 0 and 1")
        has_tensors = self.fields is not None or self.targets is not None
        if self.fields is None and self.targets is not None:
            raise ObservationGenerationError("generated targets require generated fields")
        if self.samples or has_tensors:
            if any(type(axis) is not int or axis < 1 for axis in self.shape):
                raise ObservationGenerationError("sample shape axes must be positive integers")
        elif self.shape != (0,):
            raise ObservationGenerationError("empty sample sets must have shape [0]")
        if self.samples and len(self.samples) != self.sample_count:
            raise ObservationGenerationError("sample count does not match sample shape")
        if not self.samples and not has_tensors and self.complexity_request is None:
            raise ObservationGenerationError(
                "empty sample sets require a complexity request"
            )
        for sample in self.samples:
            if (
                sample.materialization_plan is not None
                and sample.materialization_plan.benchmark_id != self.benchmark_id
            ):
                raise ObservationGenerationError("sample benchmark_id does not match sample set")
            if self.complexity_request is not None:
                if sample.complexity_value is None:
                    raise ObservationGenerationError(
                        "complexity request requires sample complexity values"
                    )
                if not self.complexity_request.contains(sample.complexity_value):
                    raise ObservationGenerationError(
                        "sample complexity is outside requested interval"
                    )

    @property
    def includes_fields(self) -> bool:
        """Return whether all samples include generated field data."""

        if self.fields is not None:
            return True
        return bool(self.samples) and all(sample.field is not None for sample in self.samples)

    @property
    def sample_count(self) -> int:
        """Return the number of scalar samples represented by this set."""

        if not self.shape:
            return 1
        count = 1
        for axis in self.shape:
            count *= axis
        return count

    @property
    def outcomes(self) -> tuple[str, ...]:
        """Return generated outcome identifiers."""

        return tuple(sample.outcome_id for sample in self.samples)

    @property
    def complexities(self) -> tuple[float, ...]:
        """Return generated sample complexities."""

        return tuple(sample.complexity for sample in self.samples)

    def require_tensors(self) -> tuple[Any, Any]:
        """Return generated field and target tensors or fail with a domain error."""

        if self.fields is None or self.targets is None:
            raise ObservationGenerationError("generated sample does not include tensors")
        return self.fields, self.targets

    def to_record(self, *, include_fields: bool = False) -> dict[str, object]:
        """Return a record for this generated sample set."""

        record: dict[str, object] = {
            "benchmark_id": str(self.benchmark_id),
            "generator_id": str(self.generator_id),
            "generator_version": self.generator_version,
            "seed": self.seed,
            "shape": list(self.shape),
            "sample_count": self.sample_count,
            "complexity_request": (
                None
                if self.complexity_request is None
                else self.complexity_request.to_record()
            ),
            "includes_fields": self.includes_fields,
            "includes_tensors": self.fields is not None,
            "samples": [
                sample.to_record(include_field=include_fields) for sample in self.samples
            ],
        }
        if self.variation_extent is not None:
            record["variation_extent"] = self.variation_extent
        return record


def load_generator(benchmark_root: Path) -> BenchmarkGenerator:
    """Load a first-class data generator from a benchmark package root."""

    generator = load_benchmark(benchmark_root).generator
    return generator


def _validate_target_distribution(distribution: Mapping[str, float]) -> None:
    if not distribution:
        raise ObservationGenerationError("target_distribution must not be empty")
    total = 0.0
    for outcome_id, probability in distribution.items():
        if not outcome_id:
            raise ObservationGenerationError(
                "target_distribution outcome ids must be nonempty"
            )
        if not math.isfinite(float(probability)):
            raise ObservationGenerationError(
                "target_distribution probabilities must be finite"
            )
        if probability < 0.0:
            raise ObservationGenerationError(
                "target_distribution probabilities must be nonnegative"
            )
        total += float(probability)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ObservationGenerationError("target_distribution probabilities must sum to 1")
