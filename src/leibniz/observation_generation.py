"""Runtime generation of benchmark samples."""

from __future__ import annotations

import base64
import math
import random
import struct
import zlib
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import product
from pathlib import Path
from typing import cast

from leibniz.artifacts import ArtifactReference
from leibniz.benchmark_implementations import Generator as BenchmarkGenerator
from leibniz.benchmark_implementations import load_benchmark
from leibniz.benchmarks import BenchmarkManifest
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.latent_factors import (
    LatentFactorDeclaration,
    SampleLatentFactor,
)
from leibniz.materialization import (
    AxisAssignment,
    MaterializationDeclaration,
    MaterializationPlan,
    MaterializationValidationError,
)
from leibniz.observation_formation import (
    FieldObservation,
    FormedObservation,
    ObservationFormationDeclaration,
    SpatialAffineVariation,
    VariationTransformDeclaration,
)
from leibniz.timing import TimingCollector

__all__ = [
    "GeneratedSample",
    "GeneratedSampleSet",
    "ObservationGenerationError",
    "StateSpaceMeasureRequest",
    "StateSpaceMeasureValue",
    "field_to_png_bytes",
    "field_to_png_data_url",
    "load_generator",
    "sample_variation_transform_coordinates",
]

_discriminatable_resolution_cache: dict[
    tuple[str, str, str, int, int, float],
    tuple[int, int],
] = {}
_canvas_fit_component_bounds_cache: dict[
    tuple[str, int, int, int],
    tuple[tuple[int, int, int, int] | None, ...],
] = {}
_rejection_cache_bins_per_axis = 8
_rejection_cache_cell_limit = 4096
_variation_state_bins_per_axis = 8
_field_scalar_construction_bytes = 64
_default_memory_budget_fraction = 0.10
_default_generation_memory_limit_bytes = 32_768_000

class ObservationGenerationError(ValueError):
    """Raised when observation generation cannot satisfy benchmark declarations."""


@dataclass(frozen=True, slots=True)
class StateSpaceMeasureRequest:
    """A benchmark-declared state-space measure interval requested by a runner."""

    measure_id: str
    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not self.measure_id:
            raise ObservationGenerationError("state-space measure id must be nonempty")
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
    """A generated sample's value for a benchmark-declared state-space measure."""

    measure_id: str
    value: float

    def __post_init__(self) -> None:
        if not self.measure_id:
            raise ObservationGenerationError("state-space measure id must be nonempty")
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
class _ResolutionSampling:
    width_axis: str
    height_axis: str


@dataclass(frozen=True, slots=True)
class GeneratedSample:
    """One generated sample from a benchmark data source."""

    index: int
    materialization_plan: MaterializationPlan
    width: int
    height: int
    component_index: int
    variation_coordinates: tuple[Mapping[str, object], ...]
    variation_values: Mapping[str, object]
    outcome_id: str
    complexity: float
    state_space_measure: StateSpaceMeasureValue | None = None
    latent_coordinates: tuple[Mapping[str, object], ...] = ()
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
            "materialization_plan": self.materialization_plan.to_record(),
            "width": self.width,
            "height": self.height,
            "component_index": self.component_index,
            "variation_coordinates": [dict(item) for item in self.variation_coordinates],
            "variation_values": dict(self.variation_values),
            "outcome_id": self.outcome_id,
            "complexity": self.complexity,
            "latent_coordinates": [dict(coordinate) for coordinate in self.latent_coordinates],
        }
        if self.state_space_measure is not None:
            record["state_space_measure"] = self.state_space_measure.to_record()
        if self.field is not None and include_field:
            record["field"] = self.field.to_record()
        return record


@dataclass(frozen=True, slots=True)
class _FormationSamples:
    """A deterministic batch of formation specifications."""

    benchmark_id: ProtocolIdentifier
    seed: int
    samples: tuple[GeneratedSample, ...]

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        if not self.samples:
            raise ObservationGenerationError("samples must not be empty")


@dataclass(frozen=True, slots=True)
class GeneratedSampleSet:
    """Shape-aware samples returned by a reusable data generator."""

    benchmark_id: ProtocolIdentifier
    generator_id: ProtocolIdentifier
    generator_version: str
    seed: int
    shape: tuple[int, ...]
    variation_extent: float
    samples: tuple[GeneratedSample, ...]
    state_space_request: StateSpaceMeasureRequest | None = None

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
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
            if sample.materialization_plan.benchmark_id != self.benchmark_id:
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

        return {
            "benchmark_id": str(self.benchmark_id),
            "generator_id": str(self.generator_id),
            "generator_version": self.generator_version,
            "seed": self.seed,
            "shape": list(self.shape),
            "variation_extent": self.variation_extent,
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


@dataclass(slots=True)
class _BoundedRejectionCache:
    cells: set[tuple[object, ...]] = dataclass_field(default_factory=lambda: set())
    max_cells: int = _rejection_cache_cell_limit

    def contains(self, key: tuple[object, ...]) -> bool:
        return key in self.cells

    def add(self, key: tuple[object, ...]) -> bool:
        if key in self.cells:
            return True
        if len(self.cells) >= self.max_cells:
            return False
        self.cells.add(key)
        return True


@dataclass(frozen=True, slots=True)
class _ObservationGenerationEngine:  # pyright: ignore[reportUnusedClass]
    """Generate samples from manifest, latent, materialization, and formation records."""

    benchmark_manifest: BenchmarkManifest
    latent_factors: LatentFactorDeclaration
    materialization: MaterializationDeclaration
    formation: ObservationFormationDeclaration
    rejection_cache: _BoundedRejectionCache = dataclass_field(
        default_factory=_BoundedRejectionCache,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.materialization.benchmark_id != self.benchmark_manifest.id:
            raise ObservationGenerationError(
                "materialization benchmark_id does not match benchmark manifest"
            )
        if self.formation.benchmark_id != self.benchmark_manifest.id:
            raise ObservationGenerationError(
                "formation benchmark_id does not match benchmark manifest"
            )
        if self.benchmark_manifest.latent_factor_declaration is None:
            raise ObservationGenerationError("benchmark manifest must declare latent factors")
        try:
            self.benchmark_manifest.validate_latent_factor_declaration(self.latent_factors)
        except ValueError as error:
            raise ObservationGenerationError(str(error)) from error

    def _sample_formation_batch(
        self,
        *,
        sample_count: int,
        seed: int,
        component_indices: Iterable[int] | None = None,
        memory_limit_bytes: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
        variation_extent: float = 1.0,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> _FormationSamples:
        """Generate deterministic formation specs without materializing fields."""

        if type(sample_count) is not int or sample_count < 1:
            raise ObservationGenerationError("sample_count must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        try:
            variation_extent_value = float(variation_extent)
        except (TypeError, ValueError) as error:
            raise ObservationGenerationError("variation_extent must be finite") from error
        if not math.isfinite(variation_extent_value):
            raise ObservationGenerationError("variation_extent must be finite")
        if variation_extent_value < 0.0 or variation_extent_value > 1.0:
            raise ObservationGenerationError("variation_extent must be between 0 and 1")
        with _timing_span(timing, f"{timing_prefix}component_indices"):
            indices = tuple(component_indices) if component_indices is not None else ()
        if indices and len(indices) != sample_count:
            raise ObservationGenerationError("component_indices length must match sample_count")
        requested_resolution_assignment = resolution_assignment
        resolved_resolution_assignment = self.materialization.minimum_resolution()
        resolved_resolution_assignment = self._minimum_discriminatable_resolution_assignment(
            minimum_assignment=resolved_resolution_assignment,
        )
        if requested_resolution_assignment is None:
            resolved_resolution_assignment = self._sample_resolution_assignment(
                sample_count=sample_count,
                seed=seed,
                minimum_assignment=resolved_resolution_assignment,
                memory_limit_bytes=memory_limit_bytes,
            )
        else:
            resolved_resolution_assignment = self._requested_resolution_assignment(
                minimum_assignment=resolved_resolution_assignment,
                requested_assignment=requested_resolution_assignment,
            )
        try:
            self.materialization.require_resolution(
                resolution_assignment=resolved_resolution_assignment,
            )
        except MaterializationValidationError as error:
            raise ObservationGenerationError(str(error)) from error
        resolution_assignment = resolved_resolution_assignment
        width = resolution_assignment.require_axis(self.formation.width_axis)
        height = resolution_assignment.require_axis(self.formation.height_axis)
        transform = _variation_transform_at_extent(
            self.formation.variation_transform,
            extent=variation_extent_value,
        )
        transform_record = transform.to_record()
        with _timing_span(timing, f"{timing_prefix}complexity"):
            complexity = self._distinguishable_state_complexity(
                width=width,
                height=height,
                variation_extent=variation_extent_value,
            )
        materialization_declaration = ArtifactReference(
            kind="materialization-declaration",
            protocol_id=self.materialization.id,
            record_digest=self.materialization.digest,
        )
        variation_transform_digest = str(ContentDigest.from_value(transform_record))
        component_generator = random.Random(f"{seed}:component-sequence")
        variation_generator = random.Random(f"{seed}:variation:{variation_transform_digest}")

        with _timing_span(
            timing,
            f"{timing_prefix}materialization_plan",
            samples=sample_count,
        ):
            plans = tuple(
                self._materialization_plan(
                    seed=seed,
                    index=index,
                    resolution_assignment=resolution_assignment,
                    materialization_declaration=materialization_declaration,
                )
                for index in range(sample_count)
            )
        with _timing_span(timing, f"{timing_prefix}component_index", samples=sample_count):
            component_index_samples = tuple(
                (
                    indices[index]
                    if indices
                    else _sample_component_index(
                        generator=component_generator,
                        component_vocabulary_size=len(self.formation.components),
                    )
                )
                for index in range(sample_count)
            )
        if any(
            index < 0 or index >= len(self.formation.components)
            for index in component_index_samples
        ):
            raise ObservationGenerationError("component index is outside component vocabulary")
        variation_samples: list[
            tuple[
                Mapping[str, object],
                tuple[Mapping[str, object], ...],
            ]
        ] = []
        variation_timing_phase = f"{timing_prefix}variation_coordinates"
        with _timing_span(timing, variation_timing_phase, samples=sample_count):
            for plan in plans:
                variation_samples.append(
                    _variation_transform_values_and_coordinates(
                        formation=self.formation,
                        transform=transform,
                        transform_record=transform_record,
                        generator=variation_generator,
                        width=plan.resolution_assignment.require_axis(self.formation.width_axis),
                        height=plan.resolution_assignment.require_axis(self.formation.height_axis),
                        affine_acceptance_thresholds=(
                            self.benchmark_manifest.affine_acceptance_thresholds()
                        ),
                        rejection_cache=self.rejection_cache,
                        timing=timing,
                        timing_phase=variation_timing_phase,
                    )
                )
        samples: list[GeneratedSample] = []
        with _timing_span(timing, f"{timing_prefix}sample_assembly", samples=sample_count):
            for index, plan, component_index, variation_sample in zip(
                range(sample_count),
                plans,
                component_index_samples,
                variation_samples,
                strict=True,
            ):
                variation_values, variation_coordinates = variation_sample
                samples.append(
                    GeneratedSample(
                        index=index,
                        materialization_plan=plan,
                        width=plan.resolution_assignment.require_axis(self.formation.width_axis),
                        height=plan.resolution_assignment.require_axis(self.formation.height_axis),
                        component_index=component_index,
                        variation_coordinates=variation_coordinates,
                        variation_values=variation_values,
                        outcome_id=self._outcome_id(component_index),
                        complexity=complexity,
                    )
                )

        return _FormationSamples(
            benchmark_id=self.benchmark_manifest.id,
            seed=seed,
            samples=tuple(samples),
        )

    def _distinguishable_state_complexity(
        self,
        *,
        width: int,
        height: int,
        variation_extent: float = 1.0,
    ) -> float:
        return self.distinguishable_state_complexity(
            width=width,
            height=height,
            variation_extent=variation_extent,
        )

    def distinguishable_state_complexity(
        self,
        *,
        width: int,
        height: int,
        variation_extent: float = 1.0,
    ) -> float:
        component_complexity = math.log2(len(self.formation.components))
        variation_complexity = _variation_transform_complexity(
            _variation_transform_at_extent(
                self.formation.variation_transform,
                extent=variation_extent,
            ),
            width=width,
            height=height,
        )
        return component_complexity + variation_complexity

    def _materialization_plan(
        self,
        *,
        seed: int,
        index: int,
        resolution_assignment: AxisAssignment,
        materialization_declaration: ArtifactReference,
    ) -> MaterializationPlan:
        return MaterializationPlan(
            id=self._plan_id(seed=seed, index=index),
            benchmark_id=self.materialization.benchmark_id,
            materialization_declaration=materialization_declaration,
            resolution_assignment=resolution_assignment,
            seed=seed,
            latent_factor_declaration=self.materialization.latent_factor_declaration,
        )

    def _sample_resolution_assignment(
        self,
        *,
        sample_count: int,
        seed: int,
        minimum_assignment: AxisAssignment,
        memory_limit_bytes: int | None,
    ) -> AxisAssignment:
        sampling = _resolution_sampling(self.materialization.layout)
        if sampling is None:
            return minimum_assignment
        minimum_width = minimum_assignment.require_axis(sampling.width_axis)
        minimum_height = minimum_assignment.require_axis(sampling.height_axis)
        lattice_steps = self.materialization.resolution_lattice_steps()
        width_step = lattice_steps.get(sampling.width_axis, 1)
        height_step = lattice_steps.get(sampling.height_axis, 1)
        maximum_pixel_count = _batch_sample_pixel_limit(
            memory_limit_bytes=(
                memory_limit_bytes
                if memory_limit_bytes is not None
                else _default_generation_memory_limit_bytes
            ),
            memory_budget_fraction=_default_memory_budget_fraction,
            sample_count=sample_count,
            channel_count=self.formation.channel_count,
        )
        maximum_width, maximum_height = _sampled_resolution_maximum(
            minimum_width=minimum_width,
            minimum_height=minimum_height,
            maximum_pixel_count=maximum_pixel_count,
        )
        if maximum_width < minimum_width:
            raise ObservationGenerationError(
                f"{sampling.width_axis} maximum {maximum_width} is below "
                f"required minimum {minimum_width}"
            )
        if maximum_height < minimum_height:
            raise ObservationGenerationError(
                f"{sampling.height_axis} maximum {maximum_height} is below "
                f"required minimum {minimum_height}"
            )
        minimum_width_multiplier = _minimum_axis_multiplier(
            minimum_width,
            step=width_step,
        )
        minimum_height_multiplier = _minimum_axis_multiplier(
            minimum_height,
            step=height_step,
        )
        maximum_width_multiplier = maximum_width // width_step
        maximum_height_multiplier = maximum_height // height_step
        if maximum_width_multiplier < minimum_width_multiplier:
            maximum_width_multiplier = minimum_width_multiplier
        if maximum_height_multiplier < minimum_height_multiplier:
            maximum_height_multiplier = minimum_height_multiplier
        generator = random.Random(
            f"{seed}:resolution:{sampling.width_axis}:{minimum_width}:"
            f"{maximum_width}:{width_step}:{sampling.height_axis}:{minimum_height}:"
            f"{maximum_height}:{height_step}:"
            f"{maximum_pixel_count}"
        )
        values = dict(minimum_assignment.values)
        values[sampling.width_axis] = (
            generator.randint(minimum_width_multiplier, maximum_width_multiplier)
            * width_step
        )
        values[sampling.height_axis] = (
            generator.randint(minimum_height_multiplier, maximum_height_multiplier)
            * height_step
        )
        return AxisAssignment(values=values)

    def _requested_resolution_assignment(
        self,
        *,
        minimum_assignment: AxisAssignment,
        requested_assignment: AxisAssignment,
    ) -> AxisAssignment:
        values = dict(minimum_assignment.values)
        for axis, requested_value in requested_assignment.values.items():
            minimum_value = minimum_assignment.require_axis(axis)
            if requested_value < minimum_value:
                raise ObservationGenerationError(
                    f"{axis} requested resolution {requested_value} is below "
                    f"required minimum {minimum_value}"
                )
            values[axis] = requested_value
        return AxisAssignment(values=values)

    def _minimum_discriminatable_resolution_assignment(
        self,
        *,
        minimum_assignment: AxisAssignment,
    ) -> AxisAssignment:
        return self.minimum_discriminatable_resolution_assignment(
            minimum_assignment=minimum_assignment,
        )

    def minimum_discriminatable_resolution_assignment(
        self,
        *,
        minimum_assignment: AxisAssignment,
    ) -> AxisAssignment:
        sampling = _resolution_sampling(self.materialization.layout)
        if sampling is None:
            return minimum_assignment
        minimum_width = minimum_assignment.require_axis(sampling.width_axis)
        minimum_height = minimum_assignment.require_axis(sampling.height_axis)
        margin = self.benchmark_manifest.resolution_discriminability_margin()
        cache_key = (
            str(self.formation.digest),
            sampling.width_axis,
            sampling.height_axis,
            minimum_width,
            minimum_height,
            margin,
        )
        cached = _discriminatable_resolution_cache.get(cache_key)
        if cached is None:
            cached = self.formation.minimum_discriminatable_resolution(
                minimum_width=minimum_width,
                minimum_height=minimum_height,
                maximum_width=max(minimum_width * 64, 64),
                maximum_height=max(minimum_height * 64, 64),
                minimum_pairwise_l1=margin,
            )
            _discriminatable_resolution_cache[cache_key] = cached
        width, height = cached
        values = dict(minimum_assignment.values)
        values[sampling.width_axis] = max(values.get(sampling.width_axis, 0), width)
        values[sampling.height_axis] = max(values.get(sampling.height_axis, 0), height)
        return AxisAssignment(values=values)

    def _outcome_id(self, component_index: int) -> str:
        if component_index >= len(self.benchmark_manifest.outcome_space.outcomes):
            raise ObservationGenerationError("component index is outside outcome space")
        return self.benchmark_manifest.outcome_space.outcomes[component_index].id

    def _latent_coordinates(
        self,
        *,
        component_index: int,
        scaled_factors: tuple[SampleLatentFactor, ...],
        plan: MaterializationPlan,
        variation_values: Mapping[str, object],
    ) -> tuple[Mapping[str, object], ...]:
        records: list[Mapping[str, object]] = []
        for factor in scaled_factors:
            if factor.role == "content":
                values: object = component_index
            elif factor.role == "materialization":
                values = dict(plan.resolution_assignment.values)
            else:
                values = variation_values
            records.append(
                {
                    "name": str(factor.name),
                    "role": factor.role,
                    "degree_measure": factor.degree_measure.to_record(),
                    "multiplicity": factor.multiplicity,
                    "values": values,
                }
            )
        return tuple(records)

    def _plan_id(
        self,
        *,
        seed: int,
        index: int,
    ) -> ProtocolIdentifier:
        return _child_identifier(
            self.benchmark_manifest.id,
            f"materialization-plans.seed{seed}.sample-{index}",
        )

    def _observation_id(
        self,
        *,
        seed: int,
        index: int,
    ) -> ProtocolIdentifier:
        return _child_identifier(
            self.benchmark_manifest.id,
            f"observations.seed{seed}.sample-{index}",
        )


def load_generator(benchmark_root: Path) -> BenchmarkGenerator:
    """Load a first-class data generator from a benchmark package root."""

    generator = load_benchmark(benchmark_root).generator
    return generator


def sample_variation_transform_coordinates(
    *,
    transform: VariationTransformDeclaration,
    seed: int,
    sample_index: int,
    component_index: int = 0,
) -> Mapping[str, object]:
    """Sample one deterministic component variation coordinate."""

    if type(seed) is not int or seed < 0:
        raise ObservationGenerationError("seed must be a nonnegative integer")
    if type(sample_index) is not int or sample_index < 0:
        raise ObservationGenerationError("sample_index must be a nonnegative integer")
    if type(component_index) is not int or component_index < 0:
        raise ObservationGenerationError("component_index must be a nonnegative integer")
    generator = _variation_random(
        seed=seed,
        sample_index=sample_index,
        component_index=component_index,
        transform_digest=str(ContentDigest.from_value(transform.to_record())),
    )
    return _variation_coordinate_record(
        transform=transform,
        generator=generator,
        component_index=component_index,
    )


def _variation_random(
    *,
    seed: int,
    sample_index: int,
    component_index: int,
    transform_digest: str,
) -> random.Random:
    return random.Random(
        ":".join((str(seed), str(sample_index), str(component_index), transform_digest))
    )


def _variation_transform_at_extent(
    transform: VariationTransformDeclaration,
    *,
    extent: float,
) -> VariationTransformDeclaration:
    if extent == 1.0:
        return transform
    spatial = transform.spatial_affine
    matrix: list[tuple[tuple[float, float], ...]] = []
    for row_index, row in enumerate(spatial.matrix):
        scaled_row: list[tuple[float, float]] = []
        for column_index, (lower, upper) in enumerate(row):
            center = 1.0 if row_index == column_index else 0.0
            scaled_row.append(
                (
                    center + (lower - center) * extent,
                    center + (upper - center) * extent,
                )
            )
        matrix.append(tuple(scaled_row))
    return VariationTransformDeclaration(
        kind=transform.kind,
        spatial_affine=SpatialAffineVariation(
            kind=spatial.kind,
            coordinate_system=spatial.coordinate_system,
            spatial_rank=spatial.spatial_rank,
            matrix=tuple(matrix),
        ),
    )


def _variation_coordinate_record(
    *,
    transform: VariationTransformDeclaration,
    generator: random.Random,
    component_index: int,
    thresholds: Mapping[str, float] | None = None,
) -> Mapping[str, object]:
    spatial = transform.spatial_affine
    return {
        "kind": "field-variation-transform-coordinate",
        "component_index": component_index,
        "spatial_affine": {
            "kind": "spatial-affine-coordinate",
            "coordinate_system": spatial.coordinate_system,
            "matrix": [
                [_sample_interval(generator, bounds) for bounds in row]
                for row in spatial.matrix
            ],
        },
    }


def _timing_span(
    timing: TimingCollector | None,
    phase: str,
    *,
    samples: int = 0,
) -> AbstractContextManager[None]:
    if timing is None:
        return nullcontext()
    return timing.span(phase, samples=samples)


def field_to_png_bytes(field: FieldObservation) -> bytes:
    """Encode a one-channel field as a grayscale PNG image."""

    channels, height, width = field.shape
    if channels != 1:
        raise ObservationGenerationError("PNG encoding currently requires one channel")
    rows: list[bytes] = []
    for y_index in range(height):
        offset = y_index * width
        row = bytes(_uint8(field.values[offset + x_index]) for x_index in range(width))
        rows.append(b"\x00" + row)
    payload = b"".join(rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(payload))
        + _png_chunk(b"IEND", b"")
    )


def field_to_png_data_url(field: FieldObservation) -> str:
    """Encode a one-channel field as a browser data URL."""

    encoded = base64.b64encode(field_to_png_bytes(field)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _variation_transform_values_and_coordinates(
    *,
    formation: ObservationFormationDeclaration,
    transform: VariationTransformDeclaration,
    transform_record: Mapping[str, object],
    generator: random.Random,
    width: int,
    height: int,
    affine_acceptance_thresholds: Mapping[str, float],
    rejection_cache: _BoundedRejectionCache | None = None,
    timing: TimingCollector | None = None,
    timing_phase: str = "",
) -> tuple[
    Mapping[str, object],
    tuple[Mapping[str, object], ...],
]:
    counters: dict[str, float] = {}
    coordinate = _accepted_variation_coordinate(
        formation=formation,
        transform=transform,
        generator=generator,
        component_index=0,
        width=width,
        height=height,
        affine_acceptance_thresholds=affine_acceptance_thresholds,
        rejection_cache=rejection_cache,
        counters=counters,
    )
    coordinates = (coordinate,)
    if timing is not None and counters:
        timing.add_counters(timing_phase, counters)
    return (
        {
            "kind": "field-variation-transform-samples",
            "bounds": transform_record,
            "coordinates": [dict(item) for item in coordinates],
        },
        coordinates,
    )


def _accepted_variation_coordinate(
    *,
    formation: ObservationFormationDeclaration,
    transform: VariationTransformDeclaration,
    generator: random.Random,
    component_index: int,
    width: int,
    height: int,
    affine_acceptance_thresholds: Mapping[str, float],
    rejection_cache: _BoundedRejectionCache | None,
    counters: dict[str, float],
) -> Mapping[str, object]:
    cache_scope = _rejection_cache_scope(
        transform=transform,
        thresholds=affine_acceptance_thresholds,
    )
    for _attempt in range(640):
        _increment_counter(counters, "candidate_count")
        _increment_counter(counters, "declared_distribution_candidate_count")
        coordinate = dict(
            _variation_coordinate_record(
                transform=transform,
                generator=generator,
                component_index=component_index,
            )
        )
        cache_cell = _rejection_cache_cell_key(
            coordinate=coordinate,
            transform=transform,
            cache_scope=cache_scope,
        )
        if (
            rejection_cache is not None
            and cache_cell is not None
            and rejection_cache.contains(cache_cell[0])
        ):
            _increment_counter(counters, "cached_reject_count")
            continue
        if not _affine_coordinate_passes_fast_thresholds(
            coordinate,
            affine_acceptance_thresholds,
        ):
            _increment_counter(counters, "fast_reject_count")
            if (
                rejection_cache is not None
                and cache_cell is not None
                and _affine_cell_is_certified_fast_reject(
                    cell_bounds=cache_cell[1],
                    thresholds=affine_acceptance_thresholds,
                )
            ):
                _increment_counter(counters, "rejection_certificate_count")
                if rejection_cache.add(cache_cell[0]):
                    _increment_counter(counters, "rejection_cache_insert_count")
                else:
                    _increment_counter(counters, "rejection_cache_saturated_count")
            continue
        if not _affine_coordinate_fits_canvas(
            formation=formation,
            width=width,
            height=height,
            coordinate=coordinate,
        ):
            _increment_counter(counters, "canvas_fit_reject_count")
            continue
        _increment_counter(counters, "accepted_count")
        return coordinate
    identity_coordinate = _identity_variation_coordinate_record(
        transform=transform,
        component_index=component_index,
    )
    _increment_counter(counters, "identity_fallback_count")
    return identity_coordinate
    raise ObservationGenerationError(
        "could not sample an identity-preserving affine coordinate"
    )


def _increment_counter(counters: dict[str, float], name: str, value: float = 1.0) -> None:
    counters[name] = counters.get(name, 0.0) + value


def _rejection_cache_scope(
    *,
    transform: VariationTransformDeclaration,
    thresholds: Mapping[str, float],
) -> tuple[object, ...]:
    return (
        str(ContentDigest.from_value(transform.to_record())),
        tuple(sorted((key, float(value)) for key, value in thresholds.items())),
    )


def _rejection_cache_cell_key(
    *,
    coordinate: Mapping[str, object],
    transform: VariationTransformDeclaration,
    cache_scope: tuple[object, ...],
) -> tuple[tuple[object, ...], tuple[tuple[float, float], ...]] | None:
    spatial = transform.spatial_affine
    if spatial.spatial_rank != 2:
        return None
    matrix = _affine_coordinate_matrix(coordinate)
    values = (matrix[0][0], matrix[0][1], matrix[0][2], matrix[1][0], matrix[1][1], matrix[1][2])
    intervals = (
        spatial.matrix[0][0],
        spatial.matrix[0][1],
        spatial.matrix[0][2],
        spatial.matrix[1][0],
        spatial.matrix[1][1],
        spatial.matrix[1][2],
    )
    indices: list[int] = []
    cell_bounds: list[tuple[float, float]] = []
    for value, bounds in zip(values, intervals, strict=True):
        lower, upper = bounds
        if upper < lower:
            return None
        if upper == lower:
            index = 0
            cell_lower = lower
            cell_upper = upper
        else:
            position = (value - lower) / (upper - lower)
            index = max(
                0,
                min(
                    _rejection_cache_bins_per_axis - 1,
                    math.floor(position * _rejection_cache_bins_per_axis),
                ),
            )
            cell_width = (upper - lower) / _rejection_cache_bins_per_axis
            cell_lower = lower + index * cell_width
            cell_upper = cell_lower + cell_width
        indices.append(index)
        cell_bounds.append((cell_lower, cell_upper))
    return ((*cache_scope, tuple(indices)), tuple(cell_bounds))


def _affine_cell_is_certified_fast_reject(
    *,
    cell_bounds: tuple[tuple[float, float], ...],
    thresholds: Mapping[str, float],
) -> bool:
    if len(cell_bounds) != 6:
        return False
    a_bounds, b_bounds, tx_bounds, c_bounds, d_bounds, ty_bounds = cell_bounds
    determinant_bounds = _interval_subtract(
        _interval_multiply(a_bounds, d_bounds),
        _interval_multiply(b_bounds, c_bounds),
    )
    if determinant_bounds[1] <= 0.0:
        return True
    minimum_determinant = thresholds.get("affine_minimum_absolute_determinant")
    if minimum_determinant is not None and determinant_bounds[1] < minimum_determinant:
        return True
    minimum_axis_alignment = thresholds.get("affine_minimum_axis_alignment")
    if minimum_axis_alignment is not None:
        if _maximum_axis_alignment(a_bounds, c_bounds) < minimum_axis_alignment:
            return True
        if _maximum_axis_alignment(d_bounds, b_bounds) < minimum_axis_alignment:
            return True
    minimum_extent = thresholds.get("affine_minimum_projected_extent")
    maximum_extent = thresholds.get("affine_maximum_projected_extent")
    if minimum_extent is not None or maximum_extent is not None:
        projected_x = _projected_unit_interval((a_bounds, b_bounds, tx_bounds))
        projected_y = _projected_unit_interval((c_bounds, d_bounds, ty_bounds))
        if minimum_extent is not None and (
            projected_x[1] < minimum_extent or projected_y[1] < minimum_extent
        ):
            return True
        if maximum_extent is not None and (
            projected_x[0] > maximum_extent or projected_y[0] > maximum_extent
        ):
            return True
    return False


def _interval_multiply(
    left: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float]:
    products = tuple(a * b for a, b in product(left, right))
    return (min(products), max(products))


def _interval_subtract(
    left: tuple[float, float],
    right: tuple[float, float],
) -> tuple[float, float]:
    return (left[0] - right[1], left[1] - right[0])


def _maximum_axis_alignment(
    aligned_bounds: tuple[float, float],
    cross_bounds: tuple[float, float],
) -> float:
    maximum_aligned = max(0.0, aligned_bounds[1])
    if maximum_aligned <= 0.0:
        return -1.0
    minimum_cross_abs = _minimum_interval_abs(cross_bounds)
    return maximum_aligned / math.hypot(maximum_aligned, minimum_cross_abs)


def _minimum_interval_abs(bounds: tuple[float, float]) -> float:
    if bounds[0] <= 0.0 <= bounds[1]:
        return 0.0
    return min(abs(bounds[0]), abs(bounds[1]))


def _projected_unit_interval(
    bounds: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
) -> tuple[float, float]:
    first, second, translation = bounds
    possible_ranges: list[float] = []
    for first_value, second_value, translation_value in product(
        first,
        second,
        translation,
    ):
        points = (
            translation_value,
            first_value + translation_value,
            second_value + translation_value,
            first_value + second_value + translation_value,
        )
        possible_ranges.append(max(points) - min(points))
    return (min(possible_ranges), max(possible_ranges))


def _identity_variation_coordinate_record(
    *,
    transform: VariationTransformDeclaration,
    component_index: int,
) -> Mapping[str, object]:
    spatial = transform.spatial_affine
    spatial_rank = spatial.spatial_rank
    matrix = [
        [1.0 if row_index == column_index else 0.0 for column_index in range(spatial_rank + 1)]
        for row_index in range(spatial_rank + 1)
    ]
    return {
        "kind": "field-variation-transform-coordinate",
        "component_index": component_index,
        "spatial_affine": {
            "kind": "spatial-affine-coordinate",
            "coordinate_system": spatial.coordinate_system,
            "matrix": matrix,
        },
    }


def _affine_coordinate_passes_fast_thresholds(
    coordinate: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> bool:
    if not thresholds:
        return True
    matrix = _affine_coordinate_matrix(coordinate)
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    signed_determinant = a * d - b * c
    if signed_determinant <= 0.0:
        return False
    determinant = abs(signed_determinant)
    minimum_determinant = thresholds.get("affine_minimum_absolute_determinant")
    if minimum_determinant is not None and determinant < minimum_determinant:
        return False
    minimum_axis_alignment = thresholds.get("affine_minimum_axis_alignment")
    if minimum_axis_alignment is not None:
        x_axis_extent = math.hypot(a, c)
        y_axis_extent = math.hypot(b, d)
        if x_axis_extent <= 0.0 or y_axis_extent <= 0.0:
            return False
        if a / x_axis_extent < minimum_axis_alignment:
            return False
        if d / y_axis_extent < minimum_axis_alignment:
            return False
    singular_minimum, singular_maximum = _affine_singular_values(a, b, c, d)
    minimum_singular_value = thresholds.get("affine_minimum_singular_value")
    if minimum_singular_value is not None and singular_minimum < minimum_singular_value:
        return False
    maximum_singular_value = thresholds.get("affine_maximum_singular_value")
    if maximum_singular_value is not None and singular_maximum > maximum_singular_value:
        return False
    maximum_condition_number = thresholds.get("affine_maximum_condition_number")
    if (
        maximum_condition_number is not None
        and singular_maximum / singular_minimum > maximum_condition_number
    ):
        return False

    corners = (
        (tx, ty),
        (a + tx, c + ty),
        (b + tx, d + ty),
        (a + b + tx, c + d + ty),
    )
    xs = tuple(point[0] for point in corners)
    ys = tuple(point[1] for point in corners)
    minimum_x = min(xs)
    maximum_x = max(xs)
    minimum_y = min(ys)
    maximum_y = max(ys)
    extent_x = maximum_x - minimum_x
    extent_y = maximum_y - minimum_y
    minimum_extent = thresholds.get("affine_minimum_projected_extent")
    if minimum_extent is not None and (extent_x < minimum_extent or extent_y < minimum_extent):
        return False
    maximum_extent = thresholds.get("affine_maximum_projected_extent")
    if maximum_extent is not None and (extent_x > maximum_extent or extent_y > maximum_extent):
        return False

    minimum_overlap = thresholds.get("affine_minimum_cell_overlap_ratio")
    if minimum_overlap is None:
        return True
    envelope_area = extent_x * extent_y
    if envelope_area <= 0.0:
        return False
    overlap_width = max(0.0, min(maximum_x, 1.0) - max(minimum_x, 0.0))
    overlap_height = max(0.0, min(maximum_y, 1.0) - max(minimum_y, 0.0))
    return (overlap_width * overlap_height) / envelope_area >= minimum_overlap


def _affine_coordinate_fits_canvas(
    *,
    formation: ObservationFormationDeclaration,
    width: int,
    height: int,
    coordinate: Mapping[str, object],
) -> bool:
    matrix = _affine_coordinate_matrix(coordinate)
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    center = (0.5, 0.5)
    for bounds in _canvas_fit_component_bounds(
        formation=formation,
        width=width,
        height=height,
    ):
        if bounds is None:
            continue
        min_x, min_y, max_x, max_y = bounds
        source_corners = (
            ((min_x - 1.5) / width, (min_y - 1.5) / height),
            ((max_x + 2.5) / width, (min_y - 1.5) / height),
            ((min_x - 1.5) / width, (max_y + 2.5) / height),
            ((max_x + 2.5) / width, (max_y + 2.5) / height),
        )
        for x, y in source_corners:
            transformed_x = center[0] + a * (x - center[0]) + b * (y - center[1]) + tx
            transformed_y = center[1] + c * (x - center[0]) + d * (y - center[1]) + ty
            if transformed_x < 0.0 or transformed_x > 1.0:
                return False
            if transformed_y < 0.0 or transformed_y > 1.0:
                return False
    return True


def _canvas_fit_component_bounds(
    *,
    formation: ObservationFormationDeclaration,
    width: int,
    height: int,
) -> tuple[tuple[int, int, int, int] | None, ...]:
    key = (str(formation.digest), width, height, 0)
    cached = _canvas_fit_component_bounds_cache.get(key)
    if cached is not None:
        return cached
    bounds = tuple(
        _positive_field_bounds(
            formation.component_field(
                width=width,
                height=height,
                component_index=component_index,
            )
        )
        for component_index in range(len(formation.components))
    )
    _canvas_fit_component_bounds_cache[key] = bounds
    return bounds


def _positive_field_bounds(
    field: FieldObservation,
) -> tuple[int, int, int, int] | None:
    channels, height, width = field.shape
    min_x = width
    min_y = height
    max_x = -1
    max_y = -1
    for channel in range(channels):
        channel_offset = channel * width * height
        for y_index in range(height):
            row_offset = channel_offset + y_index * width
            for x_index in range(width):
                if field.values[row_offset + x_index] <= 0.0:
                    continue
                min_x = min(min_x, x_index)
                min_y = min(min_y, y_index)
                max_x = max(max_x, x_index)
                max_y = max(max_y, y_index)
    if max_x < min_x or max_y < min_y:
        return None
    return (min_x, min_y, max_x, max_y)


def _affine_singular_values(
    a: float,
    b: float,
    c: float,
    d: float,
) -> tuple[float, float]:
    trace = a * a + b * b + c * c + d * d
    determinant = a * d - b * c
    discriminant = max(0.0, trace * trace - 4.0 * determinant * determinant)
    root = math.sqrt(discriminant)
    maximum_eigenvalue = max(0.0, (trace + root) / 2.0)
    minimum_eigenvalue = max(0.0, (trace - root) / 2.0)
    return (math.sqrt(minimum_eigenvalue), math.sqrt(maximum_eigenvalue))


def _affine_coordinate_matrix(coordinate: Mapping[str, object]) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    spatial_affine = coordinate.get("spatial_affine")
    if not isinstance(spatial_affine, Mapping):
        raise ObservationGenerationError("variation coordinate missing spatial_affine")
    spatial_affine_mapping = cast(Mapping[str, object], spatial_affine)
    matrix_object = spatial_affine_mapping.get("matrix")
    if not isinstance(matrix_object, Sequence):
        raise ObservationGenerationError("spatial_affine matrix must have three rows")
    matrix = cast(Sequence[object], matrix_object)
    if len(matrix) != 3:
        raise ObservationGenerationError("spatial_affine matrix must have three rows")
    rows: list[tuple[float, float, float]] = []
    for row_object in matrix:
        if not isinstance(row_object, Sequence):
            raise ObservationGenerationError("spatial_affine matrix rows must have three values")
        row = cast(Sequence[object], row_object)
        if len(row) != 3:
            raise ObservationGenerationError("spatial_affine matrix rows must have three values")
        values: list[float] = []
        for value in row:
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ObservationGenerationError("spatial_affine matrix values must be numeric")
            values.append(float(value))
        rows.append((values[0], values[1], values[2]))
    return (rows[0], rows[1], rows[2])


def _resolution_sampling(
    layout: Mapping[str, object] | None,
) -> _ResolutionSampling | None:
    if layout is None or "resolution_sampling" not in layout:
        return None
    value = layout["resolution_sampling"]
    if not isinstance(value, Mapping):
        raise ObservationGenerationError("resolution_sampling must be a record")
    value = cast(Mapping[str, object], value)
    kind = value.get("kind")
    if kind != "uniform-integer-rectangle":
        raise ObservationGenerationError(f"unsupported resolution_sampling kind: {kind}")
    width_axis = value.get("width_axis")
    if not isinstance(width_axis, str) or not width_axis:
        raise ObservationGenerationError("resolution_sampling width_axis must be nonempty")
    height_axis = value.get("height_axis")
    if not isinstance(height_axis, str) or not height_axis:
        raise ObservationGenerationError("resolution_sampling height_axis must be nonempty")
    return _ResolutionSampling(
        width_axis=width_axis,
        height_axis=height_axis,
    )


def _variation_transform_complexity(
    transform: VariationTransformDeclaration,
    *,
    width: int,
    height: int,
) -> float:
    row_extents = (width, height, 1)
    complexity = 0.0
    for row_index, row in enumerate(transform.spatial_affine.matrix):
        extent = row_extents[row_index] if row_index < len(row_extents) else 1
        for lower, upper in row:
            if upper > lower:
                complexity += _distinguishable_interval_complexity(
                    lower=lower,
                    upper=upper,
                    extent=extent,
                )
    return complexity


def _distinguishable_interval_complexity(
    *,
    lower: float,
    upper: float,
    extent: int,
) -> float:
    if extent < 1:
        return 0.0
    return math.log2(max(1, math.floor((upper - lower) * extent) + 1))


def _sampled_resolution_maximum(
    *,
    minimum_width: int,
    minimum_height: int,
    maximum_pixel_count: int,
) -> tuple[int, int]:
    minimum_pixel_count = minimum_width * minimum_height
    if maximum_pixel_count < minimum_pixel_count:
        return (minimum_width, minimum_height)
    side_multiplier = math.sqrt(maximum_pixel_count / minimum_pixel_count)
    return (
        math.floor(minimum_width * side_multiplier),
        math.floor(minimum_height * side_multiplier),
    )


def _minimum_axis_multiplier(value: int, *, step: int) -> int:
    return max(1, math.ceil(value / step))


def _batch_sample_pixel_limit(
    *,
    memory_limit_bytes: int,
    memory_budget_fraction: float,
    sample_count: int,
    channel_count: int,
) -> int:
    if type(memory_limit_bytes) is not int or memory_limit_bytes < 1:
        raise ObservationGenerationError("memory_limit_bytes must be a positive integer")
    if not math.isfinite(memory_budget_fraction) or memory_budget_fraction <= 0.0:
        raise ObservationGenerationError("memory_budget_fraction must be positive")
    if memory_budget_fraction > 1.0:
        raise ObservationGenerationError("memory_budget_fraction must not exceed 1")
    if sample_count < 1:
        raise ObservationGenerationError("sample_count must be positive")
    if channel_count < 1:
        raise ObservationGenerationError("channel_count must be positive")
    budget = math.floor(memory_limit_bytes * memory_budget_fraction)
    per_sample_denominator = sample_count * channel_count * _field_scalar_construction_bytes
    return max(1, budget // per_sample_denominator)


def _sample_component_index(
    *,
    generator: random.Random,
    component_vocabulary_size: int,
) -> int:
    return generator.randrange(component_vocabulary_size)


def _sample_interval(
    generator: random.Random,
    bounds: tuple[float, float],
) -> float:
    low, high = bounds
    return generator.uniform(low, high)


def _child_identifier(parent: ProtocolIdentifier, suffix: str) -> ProtocolIdentifier:
    return ProtocolIdentifier(
        name=ProtocolName.parse(f"{parent.name}.{suffix}"),
        version=parent.version,
    )


def _uint8(value: float) -> int:
    return max(0, min(255, round(value * 255)))


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    digest = zlib.crc32(kind)
    digest = zlib.crc32(data, digest)
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", digest)
