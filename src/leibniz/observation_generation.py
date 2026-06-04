"""Runtime generation of benchmark samples."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

from leibniz.benchmark_implementations import Generator as BenchmarkGenerator
from leibniz.benchmark_implementations import load_benchmark
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import (
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
    "StateSpaceMeasureRequest",
    "StateSpaceMeasureValue",
    "load_generator",
]

_core_state_space_measure_id = "log2_state_space_size"
_core_state_space_measure_ids = frozenset({_core_state_space_measure_id})

class ObservationGenerationError(ValueError):
    """Raised when observation generation cannot satisfy benchmark declarations."""


@dataclass(frozen=True, slots=True)
class StateSpaceMeasureRequest:
    """A core state-space measure interval requested by a runner."""

    minimum: float
    maximum: float
    measure_id: str = _core_state_space_measure_id

    def __post_init__(self) -> None:
        if not self.measure_id:
            raise ObservationGenerationError("state-space measure id must be nonempty")
        if self.measure_id not in _core_state_space_measure_ids:
            raise ObservationGenerationError("state-space measure id is not a core measure")
        if not math.isfinite(float(self.minimum)):
            raise ObservationGenerationError("state-space measure minimum must be finite")
        if not math.isfinite(float(self.maximum)):
            raise ObservationGenerationError("state-space measure maximum must be finite")
        if self.minimum < 0.0:
            raise ObservationGenerationError("state-space measure minimum must be nonnegative")
        if self.maximum < self.minimum:
            raise ObservationGenerationError(
                "state-space measure maximum must be at least the minimum"
            )

    def contains(self, value: StateSpaceMeasureValue) -> bool:
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
class StateSpaceMeasureValue:
    """A generated sample's value for a core state-space measure."""

    value: float
    measure_id: str = _core_state_space_measure_id

    def __post_init__(self) -> None:
        if not self.measure_id:
            raise ObservationGenerationError("state-space measure id must be nonempty")
        if self.measure_id not in _core_state_space_measure_ids:
            raise ObservationGenerationError("state-space measure id is not a core measure")
        if not math.isfinite(float(self.value)):
            raise ObservationGenerationError("state-space measure value must be finite")
        if self.value < 0.0:
            raise ObservationGenerationError("state-space measure value must be nonnegative")

    def to_record(self) -> dict[str, object]:
        """Return a record for this measured value."""

        return {
            "measure_id": self.measure_id,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    """One generated sample from a benchmark data source."""

    index: int
    outcome_id: str
    complexity: float
    state_space_measure: StateSpaceMeasureValue | None = None
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

    def require_field(self) -> FieldObservation:
        """Return the generated field or fail with a domain error."""

        if self.field is None:
            raise ObservationGenerationError("sample does not include generated field data")
        return self.field

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
        if self.variation_coordinates:
            record["variation_coordinates"] = [
                dict(item) for item in self.variation_coordinates
            ]
        if self.variation_values is not None:
            record["variation_values"] = dict(self.variation_values)
        if self.state_space_measure is not None:
            record["state_space_measure"] = self.state_space_measure.to_record()
        if self.field is not None and include_field:
            record["field"] = self.field.to_record()
        return record


@dataclass(frozen=True, slots=True)
class GeneratedSampleSet:
    """Shape-aware samples returned by a reusable data generator."""

    benchmark_id: ProtocolIdentifier
    generator_id: ProtocolIdentifier
    generator_version: str
    seed: int
    shape: tuple[int, ...]
    samples: tuple[GeneratedSample, ...]
    variation_extent: float | None = None
    state_space_request: StateSpaceMeasureRequest | None = None

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        if self.variation_extent is not None:
            if not math.isfinite(float(self.variation_extent)):
                raise ObservationGenerationError("variation_extent must be finite")
            if self.variation_extent < 0.0 or self.variation_extent > 1.0:
                raise ObservationGenerationError("variation_extent must be between 0 and 1")
        if self.samples:
            if any(type(axis) is not int or axis < 1 for axis in self.shape):
                raise ObservationGenerationError("sample shape axes must be positive integers")
        elif self.shape != (0,):
            raise ObservationGenerationError("empty sample sets must have shape [0]")
        if len(self.samples) != self.sample_count:
            raise ObservationGenerationError("sample count does not match sample shape")
        if not self.samples and self.state_space_request is None:
            raise ObservationGenerationError(
                "empty sample sets require a state-space request"
            )
        for sample in self.samples:
            if (
                sample.materialization_plan is not None
                and sample.materialization_plan.benchmark_id != self.benchmark_id
            ):
                raise ObservationGenerationError("sample benchmark_id does not match sample set")
            if self.state_space_request is not None:
                if sample.state_space_measure is None:
                    raise ObservationGenerationError(
                        "state-space request requires sample measure values"
                    )
                if not self.state_space_request.contains(sample.state_space_measure):
                    raise ObservationGenerationError(
                        "sample state-space measure is outside requested interval"
                    )

    @property
    def includes_fields(self) -> bool:
        """Return whether all samples include generated field data."""

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

    def to_record(self, *, include_fields: bool = False) -> dict[str, object]:
        """Return a record for this generated sample set."""

        record: dict[str, object] = {
            "benchmark_id": str(self.benchmark_id),
            "generator_id": str(self.generator_id),
            "generator_version": self.generator_version,
            "seed": self.seed,
            "shape": list(self.shape),
            "sample_count": len(self.samples),
            "state_space_request": (
                None
                if self.state_space_request is None
                else self.state_space_request.to_record()
            ),
            "includes_fields": self.includes_fields,
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
