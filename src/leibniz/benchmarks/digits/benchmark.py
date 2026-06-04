"""Digits benchmark implementation entry point."""

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
from typing import Any, TypeAlias, cast

from leibniz.artifacts import ArtifactReference
from leibniz.benchmark_implementations import Benchmark as BenchmarkProtocol
from leibniz.benchmarks import BenchmarkManifest
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.latent_factors import (
    DegreeMeasure,
    GeneratorConstructionFactor,
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
    AffineMatrix2D,
    ComponentMark,
    FieldObservation,
    FormedObservation,
    ObservationComponent,
    ObservationFormationDeclaration,
    SequenceLayout,
    SpatialAffineVariation,
    VariationTransformDeclaration,
    affine_translation,
    linear_affine_matrix,
)
from leibniz.observation_generation import (
    GeneratedSample,
    GeneratedSampleSet,
    ObservationGenerationError,
    StateSpaceMeasureRequest,
    StateSpaceMeasureValue,
)
from leibniz.observation_showcases import (
    ObservationShowcaseManifest,
    ObservationShowcaseSample,
)
from leibniz.outcomes import Outcome, OutcomeSpace
from leibniz.tensor_runtime import TensorRuntime, TensorRuntimeError
from leibniz.timing import TimingCollector

__all__ = ["benchmark"]

_benchmark_id = ProtocolIdentifier.parse("benchmarks.digits@0.1.0")
_latent_factor_id = ProtocolIdentifier.parse("benchmarks.digits.latent-factors@0.1.0")
_materialization_id = ProtocolIdentifier.parse("benchmarks.digits.materialization@0.1.0")
_formation_id = ProtocolIdentifier.parse("benchmarks.digits.observation-formation@0.1.0")
_generator_id = ProtocolIdentifier.parse("benchmarks.digits.generator@0.1.0")
_outcome_space_id = ProtocolIdentifier.parse("benchmarks.digits.outcomes@0.1.0")
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
_field_scalar_construction_bytes = 64
_default_memory_budget_fraction = 0.10
_default_generation_memory_limit_bytes = 32_768_000

_CurvePoints: TypeAlias = tuple[tuple[float, float], ...]

_digit_strokes: tuple[tuple[_CurvePoints, ...], ...] = (
    (
        ((0.5, 0.2768), (0.3056, 0.2912), (0.2984, 0.5)),
        ((0.2984, 0.5), (0.3056, 0.7088), (0.5, 0.7232)),
        ((0.5, 0.7232), (0.6944, 0.7088), (0.7016, 0.5)),
        ((0.7016, 0.5), (0.6944, 0.2912), (0.5, 0.2768)),
    ),
    (
        ((0.4208, 0.3704), (0.5072, 0.2912), (0.5648, 0.2768)),
        ((0.5648, 0.2768), (0.5576, 0.7016)),
        ((0.4424, 0.7088), (0.6512, 0.7088)),
    ),
    (
        ((0.3272, 0.3488), (0.428, 0.2552), (0.5792, 0.2912)),
        ((0.5792, 0.2912), (0.7448, 0.3416), (0.6224, 0.4712)),
        ((0.6224, 0.4712), (0.5144, 0.572), (0.3488, 0.6872)),
        ((0.3488, 0.6872), (0.68, 0.6944)),
    ),
    (
        ((0.3344, 0.32), (0.5432, 0.2408), (0.6584, 0.3704)),
        ((0.6584, 0.3704), (0.716, 0.4784), (0.5144, 0.4928)),
        ((0.5144, 0.4928), (0.7304, 0.5432), (0.6512, 0.6584)),
        ((0.6512, 0.6584), (0.5072, 0.7808), (0.32, 0.6728)),
    ),
    (
        ((0.6224, 0.284), (0.3488, 0.5288)),
        ((0.3488, 0.5288), (0.6656, 0.5288)),
        ((0.6224, 0.284), (0.6224, 0.7088)),
    ),
    (
        ((0.6584, 0.2912), (0.3632, 0.2912)),
        ((0.3632, 0.2912), (0.32, 0.4352), (0.4064, 0.4928)),
        ((0.4064, 0.4928), (0.6584, 0.4352), (0.6728, 0.6152)),
        ((0.6728, 0.6152), (0.5792, 0.7592), (0.3416, 0.68)),
    ),
    (
        ((0.6368, 0.3056), (0.3776, 0.32), (0.3272, 0.5648)),
        ((0.3272, 0.5648), (0.356, 0.752), (0.5288, 0.7232)),
        ((0.5288, 0.7232), (0.7088, 0.68), (0.6512, 0.536)),
        ((0.6512, 0.536), (0.5288, 0.428), (0.356, 0.5216)),
    ),
    (
        ((0.3344, 0.2912), (0.68, 0.2912)),
        ((0.68, 0.2912), (0.5432, 0.4928), (0.4712, 0.7088)),
    ),
    (
        ((0.5, 0.4928), (0.3416, 0.4352), (0.3848, 0.3272)),
        ((0.3848, 0.3272), (0.5072, 0.2264), (0.6296, 0.3272)),
        ((0.6296, 0.3272), (0.6728, 0.4424), (0.5, 0.4928)),
        ((0.5, 0.4928), (0.3056, 0.5576), (0.3632, 0.6728)),
        ((0.3632, 0.6728), (0.5, 0.7808), (0.644, 0.6728)),
        ((0.644, 0.6728), (0.7016, 0.5576), (0.5, 0.4928)),
    ),
    (
        ((0.644, 0.4712), (0.5288, 0.572), (0.3632, 0.4928)),
        ((0.3632, 0.4928), (0.3128, 0.3272), (0.4856, 0.2768)),
        ((0.4856, 0.2768), (0.6728, 0.2912), (0.68, 0.4784)),
        ((0.68, 0.4784), (0.6512, 0.6584), (0.4208, 0.7088)),
    ),
)


@dataclass(frozen=True, slots=True)
class _ResolutionSampling:
    width_axis: str
    height_axis: str


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


def benchmark(root: Path) -> BenchmarkProtocol:
    """Return the Digits benchmark implementation."""

    return Benchmark(root=root)


class Benchmark:
    """Executable Digits benchmark declaration."""

    def __init__(self, *, root: Path) -> None:
        self._root = root
        self._latent_factors = _latent_factors()
        self._manifest = _manifest()
        self._materialization = _materialization()
        self._formation = _formation()
        self._showcase = _showcase()
        self._generator = Generator(
            manifest=self._manifest,
            latent_factors=self._latent_factors,
            materialization=self._materialization,
            formation=self._formation,
        )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest(self) -> BenchmarkManifest:
        return self._manifest

    @property
    def latent_factors(self) -> LatentFactorDeclaration:
        return self._latent_factors

    @property
    def materialization(self) -> MaterializationDeclaration:
        return self._materialization

    @property
    def formation(self) -> ObservationFormationDeclaration:
        return self._formation

    @property
    def showcase(self) -> ObservationShowcaseManifest:
        return self._showcase

    @property
    def generator(self) -> Generator:
        return self._generator


@dataclass(frozen=True, slots=True)
class Generator:
    """Generate samples from the Digits scientific model."""

    manifest: BenchmarkManifest
    latent_factors: LatentFactorDeclaration
    materialization: MaterializationDeclaration
    formation: ObservationFormationDeclaration
    rejection_cache: _BoundedRejectionCache = dataclass_field(
        default_factory=_BoundedRejectionCache,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.materialization.benchmark_id != self.manifest.id:
            raise ObservationGenerationError(
                "materialization benchmark_id does not match benchmark manifest"
            )
        if self.formation.benchmark_id != self.manifest.id:
            raise ObservationGenerationError(
                "formation benchmark_id does not match benchmark manifest"
            )
        if self.manifest.latent_factor_declaration is None:
            raise ObservationGenerationError("benchmark manifest must declare latent factors")
        try:
            self.manifest.validate_latent_factor_declaration(self.latent_factors)
        except ValueError as error:
            raise ObservationGenerationError(str(error)) from error

    def _generate_samples(
        self,
        *,
        sample_count: int,
        seed: int,
        include_fields: bool,
        component_indices: Iterable[int] | None = None,
        memory_limit_bytes: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
        variation_extent: float = 1.0,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
        output_timing_prefix: str = "",
    ) -> tuple[GeneratedSample, ...]:
        """Generate Digits samples by choosing digit, canvas, and affine variation."""

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
        component_index_samples = self._sample_component_indices(
            sample_count=sample_count,
            seed=seed,
            component_indices=component_indices,
            timing=timing,
            timing_prefix=timing_prefix,
        )
        resolved_resolution_assignment = self._generation_resolution_assignment(
            sample_count=sample_count,
            seed=seed,
            requested_assignment=resolution_assignment,
            memory_limit_bytes=memory_limit_bytes,
        )
        width = resolved_resolution_assignment.require_axis(self.formation.width_axis)
        height = resolved_resolution_assignment.require_axis(self.formation.height_axis)
        transform = _variation_transform_at_extent(
            self.formation.variation_transform,
            extent=variation_extent_value,
        )
        transform_record = transform.to_record()
        with _timing_span(timing, f"{timing_prefix}complexity"):
            complexity = self.distinguishable_state_complexity(
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
                    resolution_assignment=resolved_resolution_assignment,
                    materialization_declaration=materialization_declaration,
                )
                for index in range(sample_count)
            )
        variation_timing_phase = f"{timing_prefix}variation_coordinates"
        variation_samples = self._sample_variation_coordinates(
            plans=plans,
            transform=transform,
            transform_record=transform_record,
            generator=variation_generator,
            timing=timing,
            timing_phase=variation_timing_phase,
        )

        with _timing_span(timing, f"{output_timing_prefix}scaled_factors"):
            scaled_factors = tuple(self.latent_factors.sample_factors)

        field_records: tuple[FormedObservation, ...]
        if include_fields:
            with _timing_span(
                timing,
                f"{output_timing_prefix}field_generation",
                samples=sample_count,
            ):
                field_records = tuple(
                    self.formation.form_observation(
                        id=self._observation_id(seed=seed, index=index),
                        plan=plan,
                        component_index=component_index,
                        variation_coordinates=variation_coordinates,
                    )
                    for (
                        index,
                        plan,
                        component_index,
                        (_variation_values, variation_coordinates),
                    ) in zip(
                        range(sample_count),
                        plans,
                        component_index_samples,
                        variation_samples,
                        strict=True,
                    )
                )
        else:
            field_records = ()

        samples: list[GeneratedSample] = []
        with _timing_span(
            timing,
            f"{output_timing_prefix}sample_assembly",
            samples=sample_count,
        ):
            for index, plan, component_index, variation_sample in zip(
                range(sample_count),
                plans,
                component_index_samples,
                variation_samples,
                strict=True,
            ):
                variation_values, variation_coordinates = variation_sample
                field_record = field_records[index] if include_fields else None
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
                        state_space_measure=_state_space_measure(complexity),
                        latent_coordinates=self._latent_coordinates(
                            component_index=component_index,
                            scaled_factors=scaled_factors,
                            plan=plan,
                            variation_values=variation_values,
                        ),
                        field=None if field_record is None else field_record.field,
                        _field_record=field_record,
                    )
                )
        return tuple(samples)

    def _generation_resolution_assignment(
        self,
        *,
        sample_count: int,
        seed: int,
        requested_assignment: AxisAssignment | None,
        memory_limit_bytes: int | None,
    ) -> AxisAssignment:
        minimum_assignment = self.minimum_discriminatable_resolution_assignment(
            minimum_assignment=self.materialization.minimum_resolution(),
        )
        if requested_assignment is None:
            resolution_assignment = self._sample_resolution_assignment(
                sample_count=sample_count,
                seed=seed,
                minimum_assignment=minimum_assignment,
                memory_limit_bytes=memory_limit_bytes,
            )
        else:
            resolution_assignment = self._requested_resolution_assignment(
                minimum_assignment=minimum_assignment,
                requested_assignment=requested_assignment,
            )
        try:
            self.materialization.require_resolution(
                resolution_assignment=resolution_assignment,
            )
        except MaterializationValidationError as error:
            raise ObservationGenerationError(str(error)) from error
        return resolution_assignment

    def _sample_component_indices(
        self,
        *,
        sample_count: int,
        seed: int,
        component_indices: Iterable[int] | None,
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> tuple[int, ...]:
        with _timing_span(timing, f"{timing_prefix}component_indices"):
            requested_indices = (
                tuple(component_indices) if component_indices is not None else ()
            )
        if requested_indices and len(requested_indices) != sample_count:
            raise ObservationGenerationError("component_indices length must match sample_count")
        generator = random.Random(f"{seed}:component-sequence")
        with _timing_span(timing, f"{timing_prefix}component_index", samples=sample_count):
            indices = tuple(
                requested_indices[index]
                if requested_indices
                else generator.randrange(len(self.formation.components))
                for index in range(sample_count)
            )
        if any(index < 0 or index >= len(self.formation.components) for index in indices):
            raise ObservationGenerationError("component index is outside component vocabulary")
        return indices

    def _sample_variation_coordinates(
        self,
        *,
        plans: tuple[MaterializationPlan, ...],
        transform: VariationTransformDeclaration,
        transform_record: Mapping[str, object],
        generator: random.Random,
        timing: TimingCollector | None,
        timing_phase: str,
    ) -> tuple[
        tuple[
            Mapping[str, object],
            tuple[Mapping[str, object], ...],
        ],
        ...,
    ]:
        samples: list[
            tuple[
                Mapping[str, object],
                tuple[Mapping[str, object], ...],
            ]
        ] = []
        with _timing_span(timing, timing_phase, samples=len(plans)):
            for plan in plans:
                counters: dict[str, float] = {}
                coordinate = _accepted_variation_coordinate(
                    formation=self.formation,
                    transform=transform,
                    generator=generator,
                    component_index=0,
                    width=plan.resolution_assignment.require_axis(self.formation.width_axis),
                    height=plan.resolution_assignment.require_axis(self.formation.height_axis),
                    affine_acceptance_thresholds=(
                        self.manifest.affine_acceptance_thresholds()
                    ),
                    rejection_cache=self.rejection_cache,
                    counters=counters,
                )
                coordinates = (coordinate,)
                if timing is not None and counters:
                    timing.add_counters(timing_phase, counters)
                samples.append(
                    (
                        {
                            "kind": "field-variation-transform-samples",
                            "bounds": transform_record,
                            "coordinates": [dict(item) for item in coordinates],
                        },
                        coordinates,
                    )
                )
        return tuple(samples)

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
        margin = self.manifest.resolution_discriminability_margin()
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
        if component_index >= len(self.manifest.outcome_space.outcomes):
            raise ObservationGenerationError("component index is outside outcome space")
        return self.manifest.outcome_space.outcomes[component_index].id

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
            self.manifest.id,
            f"materialization-plans.seed{seed}.sample-{index}",
        )

    def _observation_id(
        self,
        *,
        seed: int,
        index: int,
    ) -> ProtocolIdentifier:
        return _child_identifier(
            self.manifest.id,
            f"observations.seed{seed}.sample-{index}",
        )

    @property
    def id(self) -> ProtocolIdentifier:
        return _generator_id

    @property
    def version(self) -> str:
        return "0.1.0"

    def __call__(
        self,
        *,
        seed: int,
        shape: int | Sequence[int] | None = None,
        include_fields: bool = False,
        state_space_request: StateSpaceMeasureRequest | None = None,
        component_indices: Iterable[int] | None = None,
        memory_limit_bytes: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
        variation_extent: float = 1.0,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedSampleSet:
        """Generate a shape-aware Digits sample set."""

        sample_shape = _sample_shape(shape)
        sample_count = _sample_count(sample_shape)
        if state_space_request is not None:
            if resolution_assignment is not None:
                raise ObservationGenerationError(
                    "state-space request cannot be combined with resolution_assignment"
                )
            resolution_assignment = self._resolution_assignment_for_state_space_request(
                request=state_space_request,
                variation_extent=variation_extent,
            )
            if resolution_assignment is None:
                return GeneratedSampleSet(
                    benchmark_id=self.manifest.id,
                    generator_id=self.id,
                    generator_version=self.version,
                    seed=seed,
                    shape=(0,),
                    variation_extent=variation_extent,
                    state_space_request=state_space_request,
                    samples=(),
                )
        samples = self._generate_samples(
            sample_count=sample_count,
            seed=seed,
            include_fields=include_fields,
            component_indices=component_indices,
            memory_limit_bytes=memory_limit_bytes,
            resolution_assignment=resolution_assignment,
            variation_extent=variation_extent,
            timing=timing,
            timing_prefix=(
                f"{timing_prefix}formation_batch." if include_fields else timing_prefix
            ),
            output_timing_prefix=timing_prefix,
        )
        return GeneratedSampleSet(
            benchmark_id=self.manifest.id,
            generator_id=self.id,
            generator_version=self.version,
            seed=seed,
            shape=sample_shape,
            variation_extent=variation_extent,
            state_space_request=state_space_request,
            samples=samples,
        )

    def console_preview_batch(self, *, atom_count: int) -> Mapping[str, object]:
        """Return a balanced browser-preview batch for the Digits generator."""

        samples: list[Mapping[str, object]] = []
        sample_count = 40
        component_indices = _balanced_component_indices(
            sample_count=sample_count,
            atom_count=atom_count,
            seed=f"{self.manifest.id}:balanced-console-samples",
        )
        used_field_shapes: set[tuple[int, ...]] = set()
        for sample_index, component_index in enumerate(component_indices):
            seed = 4100 + sample_index
            attempt_count = 0
            while True:
                attempt_count += 1
                if attempt_count > 512:
                    raise ObservationGenerationError(
                        "could not generate unique console sample canvas shapes"
                    )
                sample_set = self(
                    shape=(),
                    seed=seed,
                    include_fields=True,
                    component_indices=(component_index,),
                )
                if not sample_set.includes_fields:
                    raise ObservationGenerationError(
                        "generator did not include generated fields"
                    )
                sample = sample_set.samples[0]
                field_shape = tuple(sample.require_field().shape)
                if field_shape not in used_field_shapes:
                    used_field_shapes.add(field_shape)
                    break
                seed += 1000
            if sample.materialization_plan is None or sample.component_index is None:
                raise ObservationGenerationError("Digits preview sample is incomplete")
            samples.append(
                {
                    "index": len(samples),
                    "outcome_id": sample.outcome_id,
                    "component_index": sample.component_index,
                    "complexity": sample.complexity,
                    "state_space_measure": (
                        None
                        if sample.state_space_measure is None
                        else sample.state_space_measure.to_record()
                    ),
                    "field_shape": list(sample.require_field().shape),
                    "image_data_url": _field_to_png_data_url(sample.require_field()),
                    "materialization_plan": sample.materialization_plan.to_record(),
                    "latent_coordinates": [
                        dict(coordinate) for coordinate in sample.latent_coordinates
                    ],
                }
            )
        samples.sort(key=lambda sample: _sample_display_key(sample, len(samples)))
        return {
            "mode": "balanced",
            "label": "Balanced samples",
            "seed": 401,
            "sample_count": len(samples),
            "presentation": {
                "sample_card_density": "standard",
                "aggregate_mode": False,
            },
            "samples": samples,
        }

    def sample_variation_transform_coordinates(
        self,
        *,
        seed: int,
        sample_index: int,
        component_index: int = 0,
    ) -> Mapping[str, object]:
        """Sample one deterministic Digits variation coordinate."""

        if type(seed) is not int or seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        if type(sample_index) is not int or sample_index < 0:
            raise ObservationGenerationError("sample_index must be a nonnegative integer")
        if type(component_index) is not int or component_index < 0:
            raise ObservationGenerationError("component_index must be a nonnegative integer")
        transform = self.formation.variation_transform
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

    def tensor_batch_tensors(
        self,
        *,
        runtime: TensorRuntime,
        batch: GeneratedSampleSet,
        outcome_ids: tuple[str, ...],
    ) -> tuple[Any, Any]:
        """Return tensor fields and labels for a generated Digits batch."""

        return _FormationTensorCache(
            runtime=runtime,
            formation=self.formation,
        ).batch_tensors(
            batch=batch,
            outcome_ids=outcome_ids,
        )

    def _resolution_assignment_for_state_space_request(
        self,
        *,
        request: StateSpaceMeasureRequest,
        variation_extent: float,
    ) -> AxisAssignment | None:
        minimum_assignment = self.minimum_discriminatable_resolution_assignment(
            minimum_assignment=self.materialization.minimum_resolution(),
        )
        width_axis = self.formation.width_axis
        height_axis = self.formation.height_axis
        minimum_width = minimum_assignment.require_axis(width_axis)
        minimum_height = minimum_assignment.require_axis(height_axis)
        lattice_steps = self.materialization.resolution_lattice_steps()
        width_step = lattice_steps.get(width_axis, 1)
        height_step = lattice_steps.get(height_axis, 1)
        widths = _logarithmic_lattice_axis_values(
            minimum=minimum_width,
            step=width_step,
            count=32,
        )
        heights = _logarithmic_lattice_axis_values(
            minimum=minimum_height,
            step=height_step,
            count=32,
        )
        selected: tuple[float, AxisAssignment] | None = None
        for width in widths:
            for height in heights:
                complexity = self.distinguishable_state_complexity(
                    width=width,
                    height=height,
                    variation_extent=variation_extent,
                )
                value = _state_space_measure(complexity)
                if not request.contains(value):
                    continue
                assignment = AxisAssignment(
                    values={
                        **minimum_assignment.values,
                        width_axis: width,
                        height_axis: height,
                    }
                )
                if selected is None or value.value < selected[0]:
                    selected = (value.value, assignment)
        return None if selected is None else selected[1]


def _sample_shape(shape: int | Sequence[int] | None) -> tuple[int, ...]:
    if shape is None:
        return ()
    if isinstance(shape, int):
        if shape < 1:
            raise ObservationGenerationError("sample shape axes must be positive integers")
        return (shape,)
    normalized = tuple(shape)
    if any(type(axis) is not int or axis < 1 for axis in normalized):
        raise ObservationGenerationError("sample shape axes must be positive integers")
    return normalized


def _sample_count(shape: Sequence[int]) -> int:
    if not shape:
        return 1
    count = 1
    for axis in shape:
        count *= axis
    return count


def _sample_component_index(sample: GeneratedSample) -> int:
    if sample.component_index is None:
        raise ObservationGenerationError("Digits sample is missing component index")
    return sample.component_index


def _state_space_measure(complexity: float) -> StateSpaceMeasureValue:
    return StateSpaceMeasureValue(
        value=complexity,
    )


def _logarithmic_lattice_axis_values(
    *,
    minimum: int,
    step: int,
    count: int,
) -> tuple[int, ...]:
    values: list[int] = []
    seen: set[int] = set()
    stage = 0
    minimum_multiplier = max(1, math.ceil(minimum / step))
    while len(values) < count:
        multiplier = max(
            minimum_multiplier,
            math.ceil(minimum_multiplier * (math.sqrt(2.0) ** stage)),
        )
        value = multiplier * step
        if value not in seen:
            seen.add(value)
            values.append(value)
        stage += 1
    return tuple(values)


@dataclass(slots=True)
class _FormationTensorCache:
    """Cache unvaried Digits component fields as runtime tensors."""

    runtime: TensorRuntime
    formation: ObservationFormationDeclaration
    _component_tensors: dict[tuple[int, int, int], Any] = dataclass_field(
        default_factory=lambda: cast(dict[tuple[int, int, int], Any], {})
    )

    def batch_tensors(
        self,
        *,
        batch: GeneratedSampleSet,
        outcome_ids: tuple[str, ...],
    ) -> tuple[Any, Any]:
        if not outcome_ids:
            raise TensorRuntimeError("outcome_ids must not be empty")
        fields = self._varied_batch_tensor(batch=batch)
        backend = getattr(self.runtime, "tor" + "ch")
        labels = backend.tensor(
            [outcome_ids.index(sample.outcome_id) for sample in batch.samples],
            dtype=backend.long,
            device=self.runtime.device,
        )
        return fields, labels

    def _varied_batch_tensor(self, *, batch: GeneratedSampleSet) -> Any:
        sample_count = len(batch.samples)
        if sample_count < 1:
            raise TensorRuntimeError("batch samples must not be empty")
        width, height = _sample_field_size(batch.samples[0])
        source_tensors: list[Any] = []
        affine_rows: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        for sample in batch.samples:
            sample_width, sample_height = _sample_field_size(sample)
            if sample_width != width or sample_height != height:
                raise TensorRuntimeError("batch sample canvas shapes must match")
            if len(sample.variation_coordinates) != 1:
                raise TensorRuntimeError("variation_coordinates must contain one coordinate")
            source_tensors.append(
                self.component_tensor(
                    width=width,
                    height=height,
                    component_index=_sample_component_index(sample),
                )
            )
            affine_rows.append(
                _generated_affine_grid_row(
                    sample.variation_coordinates[0],
                    width=width,
                    height=height,
                )
            )
        backend = getattr(self.runtime, "tor" + "ch")
        sources = backend.stack(source_tensors)
        theta = backend.tensor(
            affine_rows,
            dtype=backend.float32,
            device=self.runtime.device,
        )
        grid = backend.nn.functional.affine_grid(
            theta,
            sources.shape,
            align_corners=False,
        )
        transformed = backend.nn.functional.grid_sample(
            sources,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        channels, height, width = transformed.shape[1:]
        return transformed.reshape((sample_count, 1, channels, height, width)).amax(dim=1)

    def component_tensor(
        self,
        *,
        width: int,
        height: int,
        component_index: int,
    ) -> Any:
        _require_positive_integer(width, "width")
        _require_positive_integer(height, "height")
        if (
            type(component_index) is not int
            or component_index < 0
            or component_index >= len(self.formation.components)
        ):
            raise TensorRuntimeError("component_index is outside component vocabulary")
        key = (width, height, component_index)
        cached = self._component_tensors.get(key)
        if cached is not None:
            return cached
        tensor = self._build_component_tensor(
            width=width,
            height=height,
            component_index=component_index,
        )
        self._component_tensors[key] = tensor
        return tensor

    def _build_component_tensor(
        self,
        *,
        width: int,
        height: int,
        component_index: int,
    ) -> Any:
        field = self.formation.component_field(
            width=width,
            height=height,
            component_index=component_index,
        )
        backend = getattr(self.runtime, "tor" + "ch")
        tensor = backend.tensor(
            field.values,
            dtype=backend.float32,
            device=self.runtime.device,
        )
        return tensor.reshape(field.shape)


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
    _increment_counter(counters, "identity_fallback_count")
    return _identity_variation_coordinate_record(
        transform=transform,
        component_index=component_index,
    )


def _variation_coordinate_record(
    *,
    transform: VariationTransformDeclaration,
    generator: random.Random,
    component_index: int,
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


def _positive_field_bounds(field: FieldObservation) -> tuple[int, int, int, int] | None:
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


def _affine_coordinate_matrix(coordinate: Mapping[str, object]) -> AffineMatrix2D:
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


def _resolution_sampling(layout: Mapping[str, object] | None) -> _ResolutionSampling | None:
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


def _sample_interval(generator: random.Random, bounds: tuple[float, float]) -> float:
    low, high = bounds
    return generator.uniform(low, high)


def _child_identifier(parent: ProtocolIdentifier, suffix: str) -> ProtocolIdentifier:
    return ProtocolIdentifier(
        name=ProtocolName.parse(f"{parent.name}.{suffix}"),
        version=parent.version,
    )


def _timing_span(
    timing: TimingCollector | None,
    phase: str,
    *,
    samples: int = 0,
) -> AbstractContextManager[None]:
    if timing is None:
        return nullcontext()
    return timing.span(phase, samples=samples)


def _field_to_png_bytes(field: FieldObservation) -> bytes:
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


def _field_to_png_data_url(field: FieldObservation) -> str:
    encoded = base64.b64encode(_field_to_png_bytes(field)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum)
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _uint8(value: float) -> int:
    return max(0, min(255, round(float(value) * 255)))


def _sample_field_size(sample: GeneratedSample) -> tuple[int, int]:
    if sample.width is None or sample.height is None:
        raise TensorRuntimeError("field sample is missing canvas dimensions")
    return sample.width, sample.height


def _generated_affine_grid_row(
    record: Mapping[str, object],
    *,
    width: int,
    height: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    spatial = cast(Mapping[str, object], record["spatial_affine"])
    matrix = cast(Sequence[Sequence[float]], spatial["matrix"])
    return _affine_grid_row_from_values(
        matrix=(
            (float(matrix[0][0]), float(matrix[0][1]), float(matrix[0][2])),
            (float(matrix[1][0]), float(matrix[1][1]), float(matrix[1][2])),
            (float(matrix[2][0]), float(matrix[2][1]), float(matrix[2][2])),
        ),
        width=width,
        height=height,
    )


def _affine_grid_row_from_values(
    *,
    matrix: AffineMatrix2D,
    width: int,
    height: int,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    if width <= 0 or height <= 0:
        raise TensorRuntimeError("field dimensions must be positive")
    linear_matrix = linear_affine_matrix(matrix)
    inverse = _inverse_affine_matrix_from_values(matrix=linear_matrix)
    center = (0.5, 0.5)
    center_x = 2.0 * center[0] - 1.0
    center_y = 2.0 * center[1] - 1.0
    field_translation = affine_translation(matrix)
    translation_x = 2.0 * field_translation[0]
    translation_y = 2.0 * field_translation[1]
    return (
        (
            inverse[0][0],
            inverse[0][1],
            center_x
            - inverse[0][0] * (center_x + translation_x)
            - inverse[0][1] * (center_y + translation_y),
        ),
        (
            inverse[1][0],
            inverse[1][1],
            center_y
            - inverse[1][0] * (center_x + translation_x)
            - inverse[1][1] * (center_y + translation_y),
        ),
    )


def _inverse_affine_matrix_from_values(
    *,
    matrix: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    (a, b), (c, d) = matrix
    determinant = a * d - b * c
    if not math.isfinite(determinant) or determinant == 0.0:
        raise TensorRuntimeError("variation affine transform is singular")
    return ((d / determinant, -b / determinant), (-c / determinant, a / determinant))


def _require_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise TensorRuntimeError(f"{name} must be a positive integer")


def _balanced_component_indices(
    *,
    sample_count: int,
    atom_count: int,
    seed: str,
) -> tuple[int, ...]:
    if sample_count % atom_count != 0:
        raise ObservationGenerationError(
            "balanced console samples require total token count to divide atom count"
        )
    generator = random.Random(seed)
    tokens = [
        digit
        for digit in range(atom_count)
        for _occurrence in range(sample_count // atom_count)
    ]
    generator.shuffle(tokens)
    return tuple(tokens)


def _sample_display_key(sample: Mapping[str, object], sample_count: int) -> int:
    index = sample["index"]
    if not isinstance(index, int):
        raise ObservationGenerationError("generated sample index must be an integer")
    return (index * 17) % (sample_count + 1)


def _manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        id=_benchmark_id,
        name=ProtocolName.parse("benchmarks.digits"),
        outcome_space=OutcomeSpace(
            id=_outcome_space_id,
            outcomes=tuple(Outcome(id=f"digit-{digit}") for digit in range(10)),
        ),
        latent_factor_declaration=_latent_factor_reference(),
        resolution_analysis={
            "kind": "component-discriminability-margin",
            "discriminability_margin": 20.0,
            "affine_minimum_absolute_determinant": 0.25,
            "affine_minimum_axis_alignment": 0.95,
            "affine_minimum_cell_overlap_ratio": 0.55,
            "affine_minimum_singular_value": 0.72,
            "affine_maximum_singular_value": 1.28,
            "affine_maximum_condition_number": 1.6,
            "affine_minimum_projected_extent": 0.65,
            "affine_maximum_projected_extent": 1.35,
            "description": (
                "Minimum rendered component separation required when choosing live "
                "observation resolution."
            ),
        },
    )


def _latent_factors() -> LatentFactorDeclaration:
    return LatentFactorDeclaration(
        id=_latent_factor_id,
        construction_factors=(
            GeneratorConstructionFactor(
                name=ProtocolName.parse("benchmarks.digits.construction.stroke-basis"),
                degree_measure=DegreeMeasure.constant_count(7),
                description="Fixed digit construction basis used by this declaration.",
            ),
        ),
        sample_factors=(
            SampleLatentFactor(
                name=ProtocolName.parse("benchmarks.digits.sample.digit-identity"),
                role="content",
                degree_measure=DegreeMeasure.discrete_choice(10),
            ),
            SampleLatentFactor(
                name=ProtocolName.parse(
                    "benchmarks.digits.sample.field-variation-transform"
                ),
                role="variation",
                degree_measure=DegreeMeasure.vector_dimension(6),
            ),
            SampleLatentFactor(
                name=ProtocolName.parse("benchmarks.digits.materialization.canvas-shape"),
                role="materialization",
                degree_measure=DegreeMeasure.vector_dimension(2),
            ),
        ),
        complexity_projections=(),
    )


def _materialization() -> MaterializationDeclaration:
    return MaterializationDeclaration(
        id=_materialization_id,
        benchmark_id=_benchmark_id,
        latent_factor_declaration=_latent_factor_reference(),
        requirements=(),
        layout={
            "kind": "sequence-layout",
            "sequence_axis": "L",
            "width_axis": "W",
            "height_axis": "H",
            "resolution_floor": {"W": 24, "H": 24},
            "resolution_lattice": {
                "kind": "axis-multiple",
                "steps": {"W": 24, "H": 24},
                "description": (
                    "Score-bearing sampled canvases lie on independent "
                    "base-resolution multiples for each spatial axis."
                ),
            },
            "sequence_spacing": "left-to-right",
            "placement_axis": "x",
            "resolution_sampling": {
                "kind": "uniform-integer-rectangle",
                "width_axis": "W",
                "height_axis": "H",
                "description": (
                    "Batch canvas area is sampled up to the active runtime memory budget."
                ),
            },
        },
    )


def _formation() -> ObservationFormationDeclaration:
    return ObservationFormationDeclaration(
        id=_formation_id,
        benchmark_id=_benchmark_id,
        interpreter="field-mark-composition@0.1.0",
        channel_count=1,
        width_axis="W",
        height_axis="H",
        sequence_layout=SequenceLayout(
            sequence_axis="L",
            width_axis="W",
            height_axis="H",
            placement_axis="x",
        ),
        variation_transform=VariationTransformDeclaration(
            kind="field-variation-transform",
            spatial_affine=SpatialAffineVariation(
                kind="spatial-affine",
                coordinate_system="normalized-sequence-element",
                spatial_rank=2,
                matrix=(
                    ((0.76, 1.14), (-0.07, 0.07), (-0.15, 0.15)),
                    ((-0.07, 0.07), (0.76, 1.14), (-0.15, 0.15)),
                    ((0.0, 0.0), (0.0, 0.0), (1.0, 1.0)),
                ),
            ),
        ),
        components=_digit_components(),
    )


def _showcase() -> ObservationShowcaseManifest:
    return ObservationShowcaseManifest(
        id=ProtocolIdentifier.parse("benchmarks.digits.inspection-showcase@0.1.0"),
        benchmark_id=_benchmark_id,
        formation_declaration=ArtifactReference(
            kind="observation-formation-declaration",
            protocol_id=_formation_id,
        ),
        materialization_declaration=ArtifactReference(
            kind="materialization-declaration",
            protocol_id=_materialization_id,
        ),
        samples=(
            ObservationShowcaseSample(
                id=ProtocolIdentifier.parse(
                    "benchmarks.digits.inspection-samples.digit-7@0.1.0"
                ),
                label="Single digit 7",
                sample_index=0,
                seed=101,
                component_index=7,
                outcome_id="digit-7",
            ),
            ObservationShowcaseSample(
                id=ProtocolIdentifier.parse(
                    "benchmarks.digits.inspection-samples.digit-3@0.1.0"
                ),
                label="Single digit 3",
                sample_index=1,
                seed=101,
                component_index=3,
                outcome_id="digit-3",
            ),
        ),
    )


def _latent_factor_reference() -> ArtifactReference:
    return ArtifactReference(
        kind="latent-factor-declaration",
        protocol_id=_latent_factor_id,
    )


def _digit_components() -> tuple[ObservationComponent, ...]:
    return tuple(
        ObservationComponent(
            id=f"digit-{digit}",
            marks=tuple(_curve(points) for points in curves),
        )
        for digit, curves in enumerate(_digit_strokes)
    )


def _curve(points: _CurvePoints) -> ComponentMark:
    return ComponentMark(
        kind="bezier-curve",
        channel=0,
        degree=len(points) - 1,
        control_points=points,
        width=3.0,
    )
