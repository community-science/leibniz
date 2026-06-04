"""Digits benchmark implementation entry point."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path

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
    ComponentMark,
    ObservationComponent,
    ObservationFormationDeclaration,
    SequenceLayout,
    SpatialAffineVariation,
    VariationTransformDeclaration,
)
from leibniz.observation_generation import (
    GeneratedSample,
    GeneratedSampleSet,
    ObservationGenerationError,
    StateSpaceMeasureRequest,
    StateSpaceMeasureValue,
    _batch_sample_pixel_limit,  # pyright: ignore[reportPrivateUsage]
    _BoundedRejectionCache,  # pyright: ignore[reportPrivateUsage]
    _child_identifier,  # pyright: ignore[reportPrivateUsage]
    _default_generation_memory_limit_bytes,  # pyright: ignore[reportPrivateUsage]
    _default_memory_budget_fraction,  # pyright: ignore[reportPrivateUsage]
    _discriminatable_resolution_cache,  # pyright: ignore[reportPrivateUsage]
    _FormationSamples,  # pyright: ignore[reportPrivateUsage]
    _minimum_axis_multiplier,  # pyright: ignore[reportPrivateUsage]
    _resolution_sampling,  # pyright: ignore[reportPrivateUsage]
    _sampled_resolution_maximum,  # pyright: ignore[reportPrivateUsage]
    _timing_span,  # pyright: ignore[reportPrivateUsage]
    _variation_transform_at_extent,  # pyright: ignore[reportPrivateUsage]
    _variation_transform_complexity,  # pyright: ignore[reportPrivateUsage]
    _variation_transform_values_and_coordinates,  # pyright: ignore[reportPrivateUsage]
)
from leibniz.observation_generation import (
    _sample_component_index as _sample_component_index_from_vocabulary,  # pyright: ignore[reportPrivateUsage]
)
from leibniz.observation_showcases import (
    ObservationShowcaseManifest,
    ObservationShowcaseSample,
)
from leibniz.outcomes import Outcome, OutcomeSpace
from leibniz.timing import TimingCollector

__all__ = ["benchmark"]

_benchmark_id = ProtocolIdentifier.parse("benchmarks.digits@0.1.0")
_latent_factor_id = ProtocolIdentifier.parse("benchmarks.digits.latent-factors@0.1.0")
_materialization_id = ProtocolIdentifier.parse("benchmarks.digits.materialization@0.1.0")
_formation_id = ProtocolIdentifier.parse("benchmarks.digits.observation-formation@0.1.0")
_generator_id = ProtocolIdentifier.parse("benchmarks.digits.generator@0.1.0")
_outcome_space_id = ProtocolIdentifier.parse("benchmarks.digits.outcomes@0.1.0")


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
                    else _sample_component_index_from_vocabulary(
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
                            self.manifest.affine_acceptance_thresholds()
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
            benchmark_id=self.manifest.id,
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
        formation_timing_prefix = (
            f"{timing_prefix}formation_batch." if include_fields else timing_prefix
        )
        formation_batch = self._sample_formation_batch(
            sample_count=sample_count,
            seed=seed,
            component_indices=component_indices,
            memory_limit_bytes=memory_limit_bytes,
            resolution_assignment=resolution_assignment,
            variation_extent=variation_extent,
            timing=timing,
            timing_prefix=formation_timing_prefix,
        )
        samples = formation_batch.samples
        if include_fields:
            with _timing_span(timing, f"{timing_prefix}scaled_factors"):
                scaled_factors = tuple(self.latent_factors.sample_factors)
            with _timing_span(
                timing,
                f"{timing_prefix}field_generation",
                samples=sample_count,
            ):
                field_records = tuple(
                    self.formation.form_observation(
                        id=self._observation_id(
                            seed=seed,
                            index=sample.index,
                        ),
                        plan=_sample_materialization_plan(sample),
                        component_index=_sample_component_index(sample),
                        variation_coordinates=sample.variation_coordinates,
                    )
                    for sample in samples
                )
            with _timing_span(timing, f"{timing_prefix}latent_coordinates", samples=sample_count):
                latent_coordinate_samples = tuple(
                    self._latent_coordinates(
                        component_index=_sample_component_index(sample),
                        scaled_factors=scaled_factors,
                        plan=_sample_materialization_plan(sample),
                        variation_values=_sample_variation_values(sample),
                    )
                    for sample in samples
                )
            samples = tuple(
                GeneratedSample(
                    index=sample.index,
                    materialization_plan=sample.materialization_plan,
                    width=sample.width,
                    height=sample.height,
                    component_index=sample.component_index,
                    variation_coordinates=sample.variation_coordinates,
                    variation_values=sample.variation_values,
                    outcome_id=sample.outcome_id,
                    complexity=sample.complexity,
                    state_space_measure=_state_space_measure(sample.complexity),
                    latent_coordinates=latent_coordinates,
                    field=field_record.field,
                    _field_record=field_record,
                )
                for sample, field_record, latent_coordinates in zip(
                    samples,
                    field_records,
                    latent_coordinate_samples,
                    strict=True,
                )
            )
        else:
            with _timing_span(timing, f"{timing_prefix}latent_coordinates", samples=sample_count):
                scaled_factors = tuple(self.latent_factors.sample_factors)
                samples = tuple(
                    GeneratedSample(
                        index=sample.index,
                        materialization_plan=sample.materialization_plan,
                        width=sample.width,
                        height=sample.height,
                        component_index=sample.component_index,
                        variation_coordinates=sample.variation_coordinates,
                        variation_values=sample.variation_values,
                        outcome_id=sample.outcome_id,
                        complexity=sample.complexity,
                        state_space_measure=_state_space_measure(sample.complexity),
                        latent_coordinates=self._latent_coordinates(
                            component_index=_sample_component_index(sample),
                            scaled_factors=scaled_factors,
                            plan=_sample_materialization_plan(sample),
                            variation_values=_sample_variation_values(sample),
                        ),
                    )
                    for sample in samples
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


def _sample_materialization_plan(sample: GeneratedSample) -> MaterializationPlan:
    if sample.materialization_plan is None:
        raise ObservationGenerationError("Digits sample is missing materialization plan")
    return sample.materialization_plan


def _sample_component_index(sample: GeneratedSample) -> int:
    if sample.component_index is None:
        raise ObservationGenerationError("Digits sample is missing component index")
    return sample.component_index


def _sample_variation_values(sample: GeneratedSample) -> Mapping[str, object]:
    if sample.variation_values is None:
        raise ObservationGenerationError("Digits sample is missing variation values")
    return sample.variation_values


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
        components=(
            _component(
                "digit-0",
                (
                    _curve(((0.5, 0.2768), (0.3056, 0.2912), (0.2984, 0.5))),
                    _curve(((0.2984, 0.5), (0.3056, 0.7088), (0.5, 0.7232))),
                    _curve(((0.5, 0.7232), (0.6944, 0.7088), (0.7016, 0.5))),
                    _curve(((0.7016, 0.5), (0.6944, 0.2912), (0.5, 0.2768))),
                ),
            ),
            _component(
                "digit-1",
                (
                    _curve(((0.4208, 0.3704), (0.5072, 0.2912), (0.5648, 0.2768))),
                    _curve(((0.5648, 0.2768), (0.5576, 0.7016))),
                    _curve(((0.4424, 0.7088), (0.6512, 0.7088))),
                ),
            ),
            _component(
                "digit-2",
                (
                    _curve(((0.3272, 0.3488), (0.428, 0.2552), (0.5792, 0.2912))),
                    _curve(((0.5792, 0.2912), (0.7448, 0.3416), (0.6224, 0.4712))),
                    _curve(((0.6224, 0.4712), (0.5144, 0.572), (0.3488, 0.6872))),
                    _curve(((0.3488, 0.6872), (0.68, 0.6944))),
                ),
            ),
            _component(
                "digit-3",
                (
                    _curve(((0.3344, 0.32), (0.5432, 0.2408), (0.6584, 0.3704))),
                    _curve(((0.6584, 0.3704), (0.716, 0.4784), (0.5144, 0.4928))),
                    _curve(((0.5144, 0.4928), (0.7304, 0.5432), (0.6512, 0.6584))),
                    _curve(((0.6512, 0.6584), (0.5072, 0.7808), (0.32, 0.6728))),
                ),
            ),
            _component(
                "digit-4",
                (
                    _curve(((0.6224, 0.284), (0.3488, 0.5288))),
                    _curve(((0.3488, 0.5288), (0.6656, 0.5288))),
                    _curve(((0.6224, 0.284), (0.6224, 0.7088))),
                ),
            ),
            _component(
                "digit-5",
                (
                    _curve(((0.6584, 0.2912), (0.3632, 0.2912))),
                    _curve(((0.3632, 0.2912), (0.32, 0.4352), (0.4064, 0.4928))),
                    _curve(((0.4064, 0.4928), (0.6584, 0.4352), (0.6728, 0.6152))),
                    _curve(((0.6728, 0.6152), (0.5792, 0.7592), (0.3416, 0.68))),
                ),
            ),
            _component(
                "digit-6",
                (
                    _curve(((0.6368, 0.3056), (0.3776, 0.32), (0.3272, 0.5648))),
                    _curve(((0.3272, 0.5648), (0.356, 0.752), (0.5288, 0.7232))),
                    _curve(((0.5288, 0.7232), (0.7088, 0.68), (0.6512, 0.536))),
                    _curve(((0.6512, 0.536), (0.5288, 0.428), (0.356, 0.5216))),
                ),
            ),
            _component(
                "digit-7",
                (
                    _curve(((0.3344, 0.2912), (0.68, 0.2912))),
                    _curve(((0.68, 0.2912), (0.5432, 0.4928), (0.4712, 0.7088))),
                ),
            ),
            _component(
                "digit-8",
                (
                    _curve(((0.5, 0.4928), (0.3416, 0.4352), (0.3848, 0.3272))),
                    _curve(((0.3848, 0.3272), (0.5072, 0.2264), (0.6296, 0.3272))),
                    _curve(((0.6296, 0.3272), (0.6728, 0.4424), (0.5, 0.4928))),
                    _curve(((0.5, 0.4928), (0.3056, 0.5576), (0.3632, 0.6728))),
                    _curve(((0.3632, 0.6728), (0.5, 0.7808), (0.644, 0.6728))),
                    _curve(((0.644, 0.6728), (0.7016, 0.5576), (0.5, 0.4928))),
                ),
            ),
            _component(
                "digit-9",
                (
                    _curve(((0.644, 0.4712), (0.5288, 0.572), (0.3632, 0.4928))),
                    _curve(((0.3632, 0.4928), (0.3128, 0.3272), (0.4856, 0.2768))),
                    _curve(((0.4856, 0.2768), (0.6728, 0.2912), (0.68, 0.4784))),
                    _curve(((0.68, 0.4784), (0.6512, 0.6584), (0.4208, 0.7088))),
                ),
            ),
        ),
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


def _component(id: str, marks: tuple[ComponentMark, ...]) -> ObservationComponent:
    return ObservationComponent(id=id, marks=marks)


def _curve(points: tuple[tuple[float, float], ...]) -> ComponentMark:
    return ComponentMark(
        kind="bezier-curve",
        channel=0,
        degree=len(points) - 1,
        control_points=points,
        width=3.0,
    )
