"""Digits benchmark implementation entry point."""

from __future__ import annotations

import base64
import hashlib
import importlib
import itertools
import linecache
import math
import random
import struct
import zlib
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
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
    ComponentMark,
    FieldObservation,
    FormedObservation,
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
    StateSpaceCandidate,
    StateSpaceMeasureRequest,
    StateSpaceMeasureValue,
)
from leibniz.observation_showcases import (
    ObservationShowcaseManifest,
    ObservationShowcaseSample,
)
from leibniz.outcomes import Outcome, OutcomeSpace
from leibniz.tensor_runtime import (
    TensorRuntime,
    TensorRuntimeError,
    resolve_host_tensor_runtime,
    tensor_runtime_backend,
    tensor_runtime_prefers_compiled_renderer,
    tensor_value_to_host,
)
from leibniz.timing import TimingCollector

__all__ = ["benchmark"]

_benchmark_id = ProtocolIdentifier.parse("benchmarks.digits@0.1.0")
_latent_factor_id = ProtocolIdentifier.parse("benchmarks.digits.latent-factors@0.1.0")
_materialization_id = ProtocolIdentifier.parse("benchmarks.digits.materialization@0.1.0")
_formation_id = ProtocolIdentifier.parse("benchmarks.digits.observation-formation@0.1.0")
_generator_id = ProtocolIdentifier.parse("benchmarks.digits.generator@0.1.0")
_outcome_space_id = ProtocolIdentifier.parse("benchmarks.digits.outcomes@0.1.0")
_field_scalar_construction_bytes = 64
_default_memory_budget_fraction = 0.10
_default_generation_memory_limit_bytes = 32_768_000
_state_space_digit_count = 10
_state_space_canvas_minimum_side = 16
_state_space_canvas_side_step = 4
_state_space_canvas_density = 1.0
_maximum_state_space_candidates_per_request = 64
_state_space_cardinality_relative_tolerance = 1e-12
_default_constructed_affine_transform_count = 2
_canonical_digits_state_count = _state_space_digit_count
_constructed_affine_preset_max_count = 8
_constructed_affine_translation_bounds = (-0.15, 0.15)
_constructed_affine_effective_radius_fraction = 0.25
_constructed_affine_axis_density = 0.25
_constructed_affine_scale_bounds = (0.92, 1.08)
_constructed_affine_rotation_bounds = (-0.03, 0.03)
_constructed_affine_shear_bounds = (-0.03, 0.03)
_batch_render_curve_sample_count = 25
_constructed_affine_axis_names = (
    "x_translation",
    "y_translation",
    "scale",
    "rotation",
    "x_shear",
)
_console_preview_state_space_windows = (
    (3.0, 4.0),
    (5.0, 6.0),
    (8.0, 9.0),
)
_console_preview_sample_limit = 50

_CurvePoints: TypeAlias = tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class _ConstructedAffineGrid:
    x_translation: int
    y_translation: int
    scale: int
    rotation: int
    x_shear: int
    x_translation_bounds: tuple[float, float]
    y_translation_bounds: tuple[float, float]
    scale_bounds: tuple[float, float]
    rotation_bounds: tuple[float, float]
    x_shear_bounds: tuple[float, float]
    preset_count: int | None = None

    @property
    def counts(self) -> tuple[int, int, int, int, int]:
        return (
            self.x_translation,
            self.y_translation,
            self.scale,
            self.rotation,
            self.x_shear,
        )

    @property
    def transform_count(self) -> int:
        if self.preset_count is not None:
            return self.preset_count
        count = 1
        for axis_count in self.counts:
            count *= axis_count
        return count

    def to_record(self) -> dict[str, int]:
        record: dict[str, int] = dict(
            zip(_constructed_affine_axis_names, self.counts, strict=True)
        )
        if self.preset_count is not None:
            record["preset_count"] = self.preset_count
        return record

    def bounds_record(self) -> dict[str, list[float]]:
        return {
            "x_translation": list(self.x_translation_bounds),
            "y_translation": list(self.y_translation_bounds),
            "scale": list(self.scale_bounds),
            "rotation": list(self.rotation_bounds),
            "x_shear": list(self.x_shear_bounds),
        }


@dataclass(frozen=True, slots=True)
class _DigitsStateSpace:
    affine_grid: _ConstructedAffineGrid
    requested_state_count: int
    resolution_assignment: AxisAssignment | None = None

    @property
    def digit_count(self) -> int:
        return _state_space_digit_count

    @property
    def affine_transform_count(self) -> int:
        return self.affine_grid.transform_count

    @property
    def cardinality(self) -> int:
        return self.digit_count * self.affine_transform_count

    @property
    def complexity(self) -> float:
        return math.log2(self.cardinality)

    def measure(self) -> StateSpaceMeasureValue:
        return _state_space_measure(self.complexity)

    def metadata(self) -> dict[str, object]:
        return {
            "kind": "digits-requested-finite-state-space",
            "digit_count": self.digit_count,
            "affine_transform_count": self.affine_transform_count,
            "latent_state_count": self.cardinality,
            "requested_state_count": self.requested_state_count,
            "realized_state_count": self.cardinality,
            "construction": "symmetric-digits-over-finite-affine-product-grid",
            "affine_grid": self.affine_grid.to_record(),
            "affine_bounds": self.affine_grid.bounds_record(),
            "affine_parameters": list(_constructed_affine_axis_names),
        }

    def candidate(self) -> StateSpaceCandidate:
        return StateSpaceCandidate(
            request=StateSpaceMeasureRequest(
                minimum=self.complexity,
                maximum=self.complexity,
            ),
            cardinality=self.cardinality,
            resolution_assignment=self.resolution_assignment,
            metadata=self.metadata(),
        )

_SegmentWindow: TypeAlias = tuple[int, int, int, int]

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
        component_indices: tuple[int, ...] | None = None,
        transform_indices: tuple[int, ...] | None = None,
        memory_limit_bytes: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
        state_space: _DigitsStateSpace,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
        output_timing_prefix: str = "",
    ) -> tuple[GeneratedSample, ...]:
        """Generate Digits samples by choosing digit, canvas, and affine variation."""

        if type(sample_count) is not int or sample_count < 1:
            raise ObservationGenerationError("sample_count must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        component_index_samples = self._sample_component_indices(
            sample_count=sample_count,
            seed=seed,
            component_indices=component_indices,
            component_count=state_space.digit_count,
            timing=timing,
            timing_prefix=timing_prefix,
        )
        resolved_resolution_assignment = self._generation_resolution_assignment(
            sample_count=sample_count,
            seed=seed,
            requested_assignment=resolution_assignment,
            memory_limit_bytes=memory_limit_bytes,
        )
        transform = self.formation.variation_transform
        transform_record = transform.to_record()
        with _timing_span(timing, f"{timing_prefix}complexity"):
            complexity = state_space.complexity
        materialization_declaration = ArtifactReference(
            kind="materialization-declaration",
            protocol_id=self.materialization.id,
            record_digest=self.materialization.digest,
        )
        transform_index_samples = self._sample_transform_indices(
            sample_count=sample_count,
            seed=seed,
            transform_indices=transform_indices,
            state_space=state_space,
            timing=timing,
            timing_prefix=timing_prefix,
        )

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
            component_indices=component_index_samples,
            transform_indices=transform_index_samples,
            state_space=state_space,
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
                fields = self._generate_tensor_fields(
                    sample_shape=(sample_count,),
                    seed=seed,
                    component_indices=component_index_samples,
                    transform_indices=transform_index_samples,
                    state_space=state_space,
                    resolution_assignment=resolved_resolution_assignment,
                    memory_limit_bytes=memory_limit_bytes,
                    runtime=resolve_host_tensor_runtime(),
                    timing=timing,
                    timing_prefix=output_timing_prefix,
                )
                field_records = self._formed_observations_from_tensor_fields(
                    seed=seed,
                    plans=plans,
                    component_indices=component_index_samples,
                    fields=fields,
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
                digit_variant_index = 0
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
                            digit_variant_index=digit_variant_index,
                            scaled_factors=scaled_factors,
                            plan=plan,
                            variation_values=variation_values,
                        ),
                        field=None if field_record is None else field_record.field,
                        _field_record=field_record,
                    )
                )
        return tuple(samples)

    def _generate_tensor_metadata_samples(
        self,
        *,
        sample_count: int,
        component_indices: tuple[int, ...],
        state_space: _DigitsStateSpace,
    ) -> tuple[GeneratedSample, ...]:
        if len(component_indices) != sample_count:
            raise ObservationGenerationError("component_indices length must match sample_count")
        complexity = state_space.complexity
        return tuple(
            GeneratedSample(
                index=index,
                outcome_id=self._outcome_id(component_index),
                complexity=complexity,
                state_space_measure=_state_space_measure(complexity),
                component_index=component_index,
            )
            for index, component_index in enumerate(component_indices)
        )

    def _generation_resolution_assignment(
        self,
        *,
        sample_count: int,
        seed: int,
        requested_assignment: AxisAssignment | None,
        memory_limit_bytes: int | None,
    ) -> AxisAssignment:
        minimum_assignment = self.materialization.minimum_resolution()
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
        component_indices: tuple[int, ...] | None,
        component_count: int,
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> tuple[int, ...]:
        _require_generation_positive_integer(component_count, "component_count")
        if component_count > len(self.formation.components):
            raise ObservationGenerationError("component_count exceeds component vocabulary")
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
                else generator.randrange(component_count)
                for index in range(sample_count)
            )
        if any(index < 0 or index >= component_count for index in indices):
            raise ObservationGenerationError("component index is outside active component set")
        return indices

    def _sample_variation_coordinates(
        self,
        *,
        plans: tuple[MaterializationPlan, ...],
        transform: VariationTransformDeclaration,
        transform_record: Mapping[str, object],
        component_indices: tuple[int, ...],
        transform_indices: tuple[int, ...],
        state_space: _DigitsStateSpace,
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
        if len(component_indices) != len(plans):
            raise ObservationGenerationError("component_indices length must match sample_count")
        if len(transform_indices) != len(plans):
            raise ObservationGenerationError("transform_indices length must match sample_count")
        with _timing_span(timing, timing_phase, samples=len(plans)):
            for index, _plan in enumerate(plans):
                component_index = component_indices[index]
                transform_index = transform_indices[index]
                if (
                    transform_index < 0
                    or transform_index >= state_space.affine_transform_count
                ):
                    raise ObservationGenerationError(
                        "transform index is outside active transform set"
                    )
                coordinate = _constructed_variation_coordinate_record(
                    transform=transform,
                    component_index=component_index,
                    transform_index=transform_index,
                    grid=state_space.affine_grid,
                )
                coordinates = (coordinate,)
                samples.append(
                    (
                        {
                            "kind": "constructed-field-variation-transform-samples",
                            "bounds": transform_record,
                            "state_space": state_space.metadata(),
                            "transform_index": transform_index,
                            "transform_count": state_space.affine_transform_count,
                            "coordinates": [dict(item) for item in coordinates],
                        },
                        coordinates,
                    )
                )
        return tuple(samples)

    def _sample_transform_indices(
        self,
        *,
        sample_count: int,
        seed: int,
        transform_indices: tuple[int, ...] | None = None,
        state_space: _DigitsStateSpace,
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> tuple[int, ...]:
        state_space_digest = str(ContentDigest.from_value(state_space.metadata()))
        generator = random.Random(f"{seed}:variation:{state_space_digest}")
        with _timing_span(timing, f"{timing_prefix}transform_index", samples=sample_count):
            if transform_indices is not None:
                if len(transform_indices) != sample_count:
                    raise ObservationGenerationError(
                        "transform_indices length must match sample_count"
                    )
                if any(
                    index < 0 or index >= state_space.affine_transform_count
                    for index in transform_indices
                ):
                    raise ObservationGenerationError(
                        "transform index is outside active transform set"
                    )
                return transform_indices
            return tuple(
                generator.randrange(state_space.affine_transform_count)
                for _index in range(sample_count)
            )

    def _component_index_tensor(
        self,
        *,
        sample_count: int,
        seed: int,
        component_indices: tuple[int, ...] | None,
        component_count: int,
        runtime: TensorRuntime,
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> Any:
        _require_generation_positive_integer(component_count, "component_count")
        if component_count > len(self.formation.components):
            raise ObservationGenerationError("component_count exceeds component vocabulary")
        backend = tensor_runtime_backend(runtime)
        with _timing_span(timing, f"{timing_prefix}component_index", samples=sample_count):
            if component_indices is not None:
                if len(component_indices) != sample_count:
                    raise ObservationGenerationError(
                        "component_indices length must match sample_count"
                    )
                if any(index < 0 or index >= component_count for index in component_indices):
                    raise ObservationGenerationError(
                        "component index is outside active component set"
                    )
                return backend.tensor(
                    component_indices,
                    dtype=backend.long,
                    device=runtime.device,
                )
            generator = _runtime_generator(
                runtime=runtime,
                seed=f"{seed}:component-sequence",
            )
            return backend.randint(
                low=0,
                high=component_count,
                size=(sample_count,),
                dtype=backend.long,
                device=runtime.device,
                generator=generator,
            )

    def _transform_index_tensor(
        self,
        *,
        sample_count: int,
        seed: int,
        transform_indices: tuple[int, ...] | None,
        state_space: _DigitsStateSpace,
        runtime: TensorRuntime,
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> Any:
        backend = tensor_runtime_backend(runtime)
        state_space_digest = str(ContentDigest.from_value(state_space.metadata()))
        with _timing_span(timing, f"{timing_prefix}transform_index", samples=sample_count):
            if transform_indices is not None:
                if len(transform_indices) != sample_count:
                    raise ObservationGenerationError(
                        "transform_indices length must match sample_count"
                    )
                if any(
                    index < 0 or index >= state_space.affine_transform_count
                    for index in transform_indices
                ):
                    raise ObservationGenerationError(
                        "transform index is outside active transform set"
                    )
                return backend.tensor(
                    transform_indices,
                    dtype=backend.long,
                    device=runtime.device,
                )
            generator = _runtime_generator(
                runtime=runtime,
                seed=f"{seed}:variation:{state_space_digest}",
            )
            return backend.randint(
                low=0,
                high=state_space.affine_transform_count,
                size=(sample_count,),
                dtype=backend.long,
                device=runtime.device,
                generator=generator,
            )

    def _generate_tensors(
        self,
        *,
        sample_shape: tuple[int, ...],
        seed: int,
        component_indices: tuple[int, ...] | None,
        transform_indices: tuple[int, ...] | None,
        state_space: _DigitsStateSpace,
        resolution_assignment: AxisAssignment | None,
        memory_limit_bytes: int | None,
        runtime: TensorRuntime,
        outcome_ids: tuple[str, ...],
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> tuple[Any, Any]:
        """Generate tensor fields and targets directly from the Digits state space."""

        if not outcome_ids:
            raise ObservationGenerationError("tensor generation requires outcome_ids")
        sample_count = _sample_count(sample_shape)
        component_index_samples = self._component_index_tensor(
            sample_count=sample_count,
            seed=seed,
            component_indices=component_indices,
            component_count=state_space.digit_count,
            runtime=runtime,
            timing=timing,
            timing_prefix=timing_prefix,
        )
        transform_index_samples = self._transform_index_tensor(
            sample_count=sample_count,
            seed=seed,
            transform_indices=transform_indices,
            state_space=state_space,
            runtime=runtime,
            timing=timing,
            timing_prefix=timing_prefix,
        )
        backend = tensor_runtime_backend(runtime)
        fields = self._generate_tensor_fields(
            sample_shape=sample_shape,
            seed=seed,
            component_indices=component_index_samples,
            transform_indices=transform_index_samples,
            state_space=state_space,
            resolution_assignment=resolution_assignment,
            memory_limit_bytes=memory_limit_bytes,
            runtime=runtime,
            timing=timing,
            timing_prefix=timing_prefix,
        )
        with _timing_span(timing, f"{timing_prefix}target_tensor", samples=sample_count):
            component_outcome_ids = tuple(
                outcome.id for outcome in self.manifest.outcome_space.outcomes
            )
            unknown = tuple(
                outcome_id
                for outcome_id in component_outcome_ids[: state_space.digit_count]
                if outcome_id not in outcome_ids
            )
            if unknown:
                raise TensorRuntimeError(f"unknown target outcome id: {unknown[0]}")
            component_to_outcome = backend.tensor(
                [
                    outcome_ids.index(outcome_id)
                    for outcome_id in component_outcome_ids[: state_space.digit_count]
                ],
                dtype=backend.long,
                device=runtime.device,
            )
            target_indices = component_to_outcome.index_select(0, component_index_samples)
            labels = backend.nn.functional.one_hot(
                target_indices,
                num_classes=len(outcome_ids),
            ).to(dtype=backend.float32)
            if sample_shape:
                labels = labels.reshape((*sample_shape, len(outcome_ids)))
            else:
                labels = labels.reshape((len(outcome_ids),))
        return fields, labels

    def _generate_tensor_fields(
        self,
        *,
        sample_shape: tuple[int, ...],
        seed: int,
        component_indices: Any,
        transform_indices: Any,
        state_space: _DigitsStateSpace,
        resolution_assignment: AxisAssignment | None,
        memory_limit_bytes: int | None,
        runtime: TensorRuntime,
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> Any:
        sample_count = _sample_count(sample_shape)
        resolved_resolution_assignment = self._generation_resolution_assignment(
            sample_count=sample_count,
            seed=seed,
            requested_assignment=resolution_assignment,
            memory_limit_bytes=memory_limit_bytes,
        )
        width = resolved_resolution_assignment.require_axis(self.formation.width_axis)
        height = resolved_resolution_assignment.require_axis(self.formation.height_axis)
        with _timing_span(timing, f"{timing_prefix}batch_tensor_render", samples=sample_count):
            fields = self._build_batch_tensor(
                width=width,
                height=height,
                digit_count=state_space.digit_count,
                component_indices=component_indices,
                transform_indices=transform_indices,
                transform=self.formation.variation_transform,
                grid=state_space.affine_grid,
                runtime=runtime,
                timing=timing,
                timing_prefix=timing_prefix,
            )
        if sample_shape:
            return fields.reshape((*sample_shape, *tuple(fields.shape[1:])))
        return fields.reshape(tuple(fields.shape[1:]))

    def _formed_observations_from_tensor_fields(
        self,
        *,
        seed: int,
        plans: tuple[MaterializationPlan, ...],
        component_indices: tuple[int, ...],
        fields: Any,
        sample_indices: tuple[int, ...] | None = None,
    ) -> tuple[FormedObservation, ...]:
        if len(plans) != len(component_indices):
            raise ObservationGenerationError("field plan count must match component indices")
        if sample_indices is not None and len(sample_indices) != len(plans):
            raise ObservationGenerationError("field sample index count must match plans")
        flat_fields = fields.reshape(
            (
                len(plans),
                self.formation.channel_count,
                plans[0].resolution_assignment.require_axis(self.formation.height_axis),
                plans[0].resolution_assignment.require_axis(self.formation.width_axis),
            )
        )
        host_fields = tensor_value_to_host(flat_fields)
        formation_reference = ArtifactReference(
            kind="observation-formation-declaration",
            protocol_id=self.formation.id,
            record_digest=self.formation.digest,
        )
        records: list[FormedObservation] = []
        for index, (plan, component_index) in enumerate(
            zip(plans, component_indices, strict=True)
        ):
            field_shape = tuple(int(size) for size in host_fields[index].shape)
            if len(field_shape) != 3:
                raise ObservationGenerationError("rendered field shape must have rank 3")
            sample_index = sample_indices[index] if sample_indices is not None else index
            field = FieldObservation(
                shape=(field_shape[0], field_shape[1], field_shape[2]),
                values=tuple(float(value) for value in host_fields[index].reshape(-1).tolist()),
            )
            records.append(
                FormedObservation(
                    id=self._observation_id(seed=seed, index=sample_index),
                    benchmark_id=self.manifest.id,
                    formation_declaration=formation_reference,
                    materialization_plan=ArtifactReference(
                        kind="materialization-plan",
                        protocol_id=plan.id,
                        record_digest=plan.digest,
                    ),
                    component_index=component_index,
                    field=field,
                )
            )
        return tuple(records)

    def _build_batch_tensor(
        self,
        *,
        width: int,
        height: int,
        digit_count: int,
        component_indices: Any,
        transform_indices: Any,
        transform: VariationTransformDeclaration,
        grid: _ConstructedAffineGrid,
        runtime: TensorRuntime,
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> Any:
        _require_positive_integer(width, "width")
        _require_positive_integer(height, "height")
        _require_positive_integer(digit_count, "digit_count")
        if digit_count > len(self.formation.components):
            raise TensorRuntimeError("digit_count exceeds component vocabulary")
        backend = tensor_runtime_backend(runtime)
        component_index_tensor = _runtime_long_tensor(
            runtime=runtime,
            values=component_indices,
        ).reshape(-1)
        transform_index_tensor = _runtime_long_tensor(
            runtime=runtime,
            values=transform_indices,
        ).reshape(-1)
        if int(component_index_tensor.numel()) != int(transform_index_tensor.numel()):
            raise TensorRuntimeError("component and transform index counts must match")
        sample_count = int(component_index_tensor.numel())
        fields = backend.zeros(
            (sample_count, self.formation.channel_count, height, width),
            dtype=backend.float32,
            device=runtime.device,
        )
        if sample_count == 0:
            return fields
        if tensor_runtime_prefers_compiled_renderer(runtime) and self.formation.channel_count == 1:
            return self._build_batch_tensor_triton(
                width=width,
                height=height,
                digit_count=digit_count,
                component_index_tensor=component_index_tensor,
                transform_index_tensor=transform_index_tensor,
                transform=transform,
                grid=grid,
                runtime=runtime,
            )
        matrices = _constructed_affine_matrix_tensors(
            transform=transform,
            grid=grid,
            transform_indices=transform_index_tensor,
            runtime=runtime,
        )
        t = backend.linspace(
            0.0,
            1.0,
            _batch_render_curve_sample_count,
            dtype=backend.float32,
            device=runtime.device,
        ).reshape(1, _batch_render_curve_sample_count)
        one_minus_t = 1.0 - t
        with backend.profiler.record_function("digits.render.component_grouping"):
            component_counts = tuple(
                int(count)
                for count in tensor_value_to_host(
                    backend.bincount(
                        component_index_tensor,
                        minlength=digit_count,
                    )
                ).tolist()
            )
            sorted_sample_indices = component_index_tensor.argsort()
        component_mark_offsets: list[int] = [0]
        mark_channels: list[int] = []
        mark_values: list[float] = []
        mark_widths: list[float] = []
        mark_controls: list[
            tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
        ] = []
        mark_segment_windows: list[tuple[_SegmentWindow, ...]] = []
        for component_index in range(digit_count):
            for mark in self.formation.components[component_index].marks:
                mark_channels.append(mark.channel)
                mark_values.append(float(mark.value))
                mark_widths.append(float(mark.width))
                mark_controls.append(_quadratic_control_points(mark))
                mark_segment_windows.append(
                    _mark_segment_windows(
                        mark=mark,
                        transform=transform,
                        grid=grid,
                        width=width,
                        height=height,
                    )
                )
            component_mark_offsets.append(len(mark_channels))
        if not mark_channels:
            return fields
        with backend.profiler.record_function("digits.render.mark_tables"):
            value_tensor = backend.tensor(
                mark_values,
                dtype=backend.float32,
                device=runtime.device,
            )
            width_tensor = backend.tensor(
                mark_widths,
                dtype=backend.float32,
                device=runtime.device,
            )
            control_tensor = backend.tensor(
                mark_controls,
                dtype=backend.float32,
                device=runtime.device,
            )
        sample_offset = 0
        for component_index in range(digit_count):
            component_sample_count = component_counts[component_index]
            if component_sample_count == 0:
                continue
            sample_indices = sorted_sample_indices[
                sample_offset : sample_offset + component_sample_count
            ]
            sample_offset += component_sample_count
            component_mark_start = component_mark_offsets[component_index]
            component_mark_stop = component_mark_offsets[component_index + 1]
            component_mark_count = component_mark_stop - component_mark_start
            if component_mark_count == 0:
                continue
            component_matrices = tuple(
                matrix.index_select(0, sample_indices) for matrix in matrices
            )
            m00, m01, m02, m10, m11, m12, width_scale = component_matrices
            for mark_index in range(component_mark_start, component_mark_stop):
                with _timing_span(
                    timing,
                    f"{timing_prefix}batch_tensor_render.mark",
                    samples=component_sample_count,
                ), backend.profiler.record_function("digits.render.mark"):
                    controls = control_tensor[mark_index]
                    control_x = controls[:, 0].reshape(1, 3) - 0.5
                    control_y = controls[:, 1].reshape(1, 3) - 0.5
                    transformed_x = (
                        0.5
                        + m00.reshape(component_sample_count, 1) * control_x
                        + m01.reshape(component_sample_count, 1) * control_y
                        + m02.reshape(component_sample_count, 1)
                    )
                    transformed_y = (
                        0.5
                        + m10.reshape(component_sample_count, 1) * control_x
                        + m11.reshape(component_sample_count, 1) * control_y
                        + m12.reshape(component_sample_count, 1)
                    )
                    curve_x = (
                        one_minus_t * one_minus_t * transformed_x[:, 0:1]
                        + 2.0 * one_minus_t * t * transformed_x[:, 1:2]
                        + t * t * transformed_x[:, 2:3]
                    ) * width
                    curve_y = (
                        one_minus_t * one_minus_t * transformed_y[:, 0:1]
                        + 2.0 * one_minus_t * t * transformed_y[:, 1:2]
                        + t * t * transformed_y[:, 2:3]
                    ) * height
                    threshold = (
                        width_scale.reshape(component_sample_count, 1, 1)
                        * width_tensor[mark_index]
                        / 2.0
                    ) ** 2
                    mark_value = value_tensor[mark_index]
                    channel = mark_channels[mark_index]
                    x_start, x_stop, y_start, y_stop = _combined_segment_window(
                        mark_segment_windows[mark_index]
                    )
                    window_width = x_stop - x_start
                    window_height = y_stop - y_start
                    if window_width <= 0 or window_height <= 0:
                        continue
                    xs = backend.arange(
                        x_start,
                        x_stop,
                        dtype=backend.float32,
                        device=runtime.device,
                    ).reshape(1, 1, 1, window_width) + 0.5
                    ys = backend.arange(
                        y_start,
                        y_stop,
                        dtype=backend.float32,
                        device=runtime.device,
                    ).reshape(1, 1, window_height, 1) + 0.5
                    sx = curve_x[:, :-1].reshape(
                        component_sample_count,
                        _batch_render_curve_sample_count - 1,
                        1,
                        1,
                    )
                    sy = curve_y[:, :-1].reshape(
                        component_sample_count,
                        _batch_render_curve_sample_count - 1,
                        1,
                        1,
                    )
                    ex = curve_x[:, 1:].reshape(
                        component_sample_count,
                        _batch_render_curve_sample_count - 1,
                        1,
                        1,
                    )
                    ey = curve_y[:, 1:].reshape(
                        component_sample_count,
                        _batch_render_curve_sample_count - 1,
                        1,
                        1,
                    )
                    dx = ex - sx
                    dy = ey - sy
                    length_squared = dx * dx + dy * dy
                    safe_length_squared = backend.where(
                        length_squared == 0.0,
                        backend.ones_like(length_squared),
                        length_squared,
                    )
                    segment_t = ((xs - sx) * dx + (ys - sy) * dy) / safe_length_squared
                    segment_t = segment_t.clamp(0.0, 1.0)
                    closest_x = sx + segment_t * dx
                    closest_y = sy + segment_t * dy
                    segment_distance_squared = (
                        (xs - closest_x) ** 2 + (ys - closest_y) ** 2
                    )
                    point_distance_squared = (xs - sx) ** 2 + (ys - sy) ** 2
                    segment_distance_squared = backend.where(
                        length_squared == 0.0,
                        point_distance_squared,
                        segment_distance_squared,
                    )
                    distance_squared = segment_distance_squared.min(dim=1).values
                    mark_values_tensor = (
                        (distance_squared <= threshold).to(dtype=backend.float32)
                        * mark_value
                    )
                    with backend.profiler.record_function("digits.render.field_update"):
                        current_channel = fields.index_select(0, sample_indices)[
                            :,
                            channel,
                            y_start:y_stop,
                            x_start:x_stop,
                        ]
                        updated_channel = backend.maximum(
                            current_channel,
                            mark_values_tensor,
                        )
                        fields[:, channel, y_start:y_stop, x_start:x_stop].index_copy_(
                            0,
                            sample_indices,
                            updated_channel,
                        )
        return fields

    def _build_batch_tensor_triton(
        self,
        *,
        width: int,
        height: int,
        digit_count: int,
        component_index_tensor: Any,
        transform_index_tensor: Any,
        transform: VariationTransformDeclaration,
        grid: _ConstructedAffineGrid,
        runtime: TensorRuntime,
    ) -> Any:
        backend = tensor_runtime_backend(runtime)
        sample_count = int(component_index_tensor.numel())
        mark_offsets: list[int] = [0]
        mark_values: list[float] = []
        mark_widths: list[float] = []
        curve_points: list[tuple[tuple[float, float], ...]] = []
        for component_index in range(digit_count):
            for mark in self.formation.components[component_index].marks:
                if mark.channel != 0:
                    raise TensorRuntimeError("Triton Digits renderer requires single-channel marks")
                mark_values.append(float(mark.value))
                mark_widths.append(float(mark.width))
                curve_points.append(
                    _sampled_quadratic_points(_quadratic_control_points(mark))
                )
            mark_offsets.append(len(mark_values))
        fields = backend.empty(
            (sample_count, 1, height, width),
            dtype=backend.float32,
            device=runtime.device,
        )
        if not mark_values:
            fields.zero_()
            return fields
        max_component_mark_count = max(
            stop - start for start, stop in zip(mark_offsets, mark_offsets[1:], strict=False)
        )
        matrices = _constructed_affine_matrix_tensors(
            transform=transform,
            grid=grid,
            transform_indices=transform_index_tensor,
            runtime=runtime,
        )
        mark_offsets_tensor = backend.tensor(
            mark_offsets,
            dtype=backend.int32,
            device=runtime.device,
        )
        mark_values_tensor = backend.tensor(
            mark_values,
            dtype=backend.float32,
            device=runtime.device,
        )
        mark_widths_tensor = backend.tensor(
            mark_widths,
            dtype=backend.float32,
            device=runtime.device,
        )
        curve_points_tensor = backend.tensor(
            curve_points,
            dtype=backend.float32,
            device=runtime.device,
        )
        kernel, triton = _digits_triton_render_kernel()
        block_size = 256
        total_elements = sample_count * height * width
        grid_shape = (triton.cdiv(total_elements, block_size),)
        kernel[grid_shape](
            fields,
            component_index_tensor,
            matrices[0],
            matrices[1],
            matrices[2],
            matrices[3],
            matrices[4],
            matrices[5],
            matrices[6],
            mark_offsets_tensor,
            mark_values_tensor,
            mark_widths_tensor,
            curve_points_tensor,
            max_component_mark_count,
            total_elements,
            height,
            width,
            block_size,
        )
        return fields

    def distinguishable_state_complexity(
        self,
        *,
        width: int,
        height: int,
        variation_extent: float = 1.0,
    ) -> float:
        _require_generation_positive_integer(width, "width")
        _require_generation_positive_integer(height, "height")
        try:
            variation_extent_value = float(variation_extent)
        except (TypeError, ValueError) as error:
            raise ObservationGenerationError("variation_extent must be finite") from error
        if not math.isfinite(variation_extent_value):
            raise ObservationGenerationError("variation_extent must be finite")
        if variation_extent_value < 0.0 or variation_extent_value > 1.0:
            raise ObservationGenerationError("variation_extent must be between 0 and 1")
        return self._default_state_space(
            canonical_variation=variation_extent_value == 0.0,
        ).complexity

    def constructed_state_space_complexity(
        self,
        *,
        affine_transform_count: int,
    ) -> float:
        """Return the exact log2 count of constructed single-digit states."""

        return self._state_space_for_requested_state_count(
            requested_state_count=_state_space_digit_count * affine_transform_count,
            affine_transform_count=affine_transform_count,
        ).complexity

    def minimum_state_space_measure(self) -> StateSpaceMeasureValue:
        """Return the smallest score-bearing Digits state-space measure."""

        return _state_space_measure(math.log2(_canonical_digits_state_count))

    def state_space_for_request(
        self,
        *,
        request: StateSpaceMeasureRequest,
    ) -> StateSpaceCandidate | None:
        """Return the smallest symmetric finite state space inside a request band."""

        state_space = self._first_symmetric_state_space_in_request(request=request)
        if state_space is None:
            return None
        return state_space.candidate()

    def state_spaces_for_request(
        self,
        *,
        request: StateSpaceMeasureRequest,
    ) -> tuple[StateSpaceCandidate, ...]:
        """Return symmetric Digits state-space candidates inside a request band."""

        minimum_complexity = self.minimum_state_space_measure().value
        if request.maximum < minimum_complexity:
            return ()
        minimum_cardinality = _ceil_state_space_cardinality(
            max(request.minimum, minimum_complexity)
        )
        maximum_cardinality = _floor_state_space_cardinality(request.maximum)
        if maximum_cardinality < minimum_cardinality:
            return ()
        resolution_assignment = self._resolution_assignment_for_state_space_request(
            request
        )
        candidates: list[StateSpaceCandidate] = []
        for state_space in self._symmetric_state_spaces_in_cardinality_range(
            minimum_cardinality=minimum_cardinality,
            maximum_cardinality=maximum_cardinality,
            resolution_assignment=resolution_assignment,
        ):
            measure = state_space.measure()
            if request.contains(measure):
                candidates.append(state_space.candidate())
        return tuple(candidates)

    def _default_state_space(self, *, canonical_variation: bool = False) -> _DigitsStateSpace:
        return self._state_space_for_affine_transform_count(
            affine_transform_count=(
                1 if canonical_variation else _default_constructed_affine_transform_count
            ),
        )

    def _state_space_for_affine_transform_count(
        self,
        *,
        affine_transform_count: int,
        resolution_assignment: AxisAssignment | None = None,
    ) -> _DigitsStateSpace:
        return self._state_space_for_requested_state_count(
            requested_state_count=_state_space_digit_count * affine_transform_count,
            affine_transform_count=affine_transform_count,
            resolution_assignment=resolution_assignment,
        )

    def _state_space_for_requested_state_count(
        self,
        *,
        requested_state_count: int,
        affine_transform_count: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
    ) -> _DigitsStateSpace:
        _require_generation_positive_integer(
            requested_state_count,
            "requested_state_count",
        )
        if affine_transform_count is None:
            affine_transform_count = max(
                1,
                math.ceil(requested_state_count / _state_space_digit_count),
            )
        requested_state_count = _state_space_digit_count * affine_transform_count
        return _DigitsStateSpace(
            affine_grid=_constructed_affine_grid(
                affine_transform_count,
                resolution_assignment=resolution_assignment,
            ),
            requested_state_count=requested_state_count,
            resolution_assignment=resolution_assignment,
        )

    def _symmetric_state_spaces_in_cardinality_range(
        self,
        *,
        minimum_cardinality: int,
        maximum_cardinality: int,
        resolution_assignment: AxisAssignment,
    ) -> tuple[_DigitsStateSpace, ...]:
        _require_generation_positive_integer(minimum_cardinality, "minimum_cardinality")
        _require_generation_positive_integer(maximum_cardinality, "maximum_cardinality")
        if maximum_cardinality < minimum_cardinality:
            return ()
        state_spaces: list[_DigitsStateSpace] = []
        minimum_transform_count = max(
            1,
            math.ceil(minimum_cardinality / _state_space_digit_count),
        )
        maximum_transform_count = max(
            1,
            maximum_cardinality // _state_space_digit_count,
        )
        for affine_grid in _constructed_affine_grids_in_transform_count_range(
            minimum_transform_count=minimum_transform_count,
            maximum_transform_count=maximum_transform_count,
            resolution_assignment=resolution_assignment,
        ):
            if len(state_spaces) >= _maximum_state_space_candidates_per_request:
                break
            state_spaces.append(
                _DigitsStateSpace(
                    affine_grid=affine_grid,
                    requested_state_count=(
                        _state_space_digit_count * affine_grid.transform_count
                    ),
                    resolution_assignment=resolution_assignment,
                )
            )
        return tuple(
            sorted(
                state_spaces,
                key=lambda state_space: (
                    state_space.cardinality,
                    state_space.digit_count,
                    state_space.affine_transform_count,
                ),
            )
        )

    def _first_symmetric_state_space_in_request(
        self,
        *,
        request: StateSpaceMeasureRequest,
    ) -> _DigitsStateSpace | None:
        minimum_complexity = self.minimum_state_space_measure().value
        if request.maximum < minimum_complexity:
            return None
        minimum_cardinality = _ceil_state_space_cardinality(
            max(request.minimum, minimum_complexity)
        )
        maximum_cardinality = _floor_state_space_cardinality(request.maximum)
        minimum_transform_count = max(
            1,
            math.ceil(minimum_cardinality / _state_space_digit_count),
        )
        maximum_transform_count = max(
            1,
            maximum_cardinality // _state_space_digit_count,
        )
        if maximum_transform_count < minimum_transform_count:
            return None
        resolution_assignment = self._resolution_assignment_for_state_space_request(
            request
        )
        for transform_count in range(
            minimum_transform_count,
            min(maximum_transform_count, minimum_transform_count + 4096) + 1,
        ):
            try:
                state_space = self._state_space_for_affine_transform_count(
                    affine_transform_count=transform_count,
                    resolution_assignment=resolution_assignment,
                )
            except ObservationGenerationError:
                continue
            if request.contains(state_space.measure()):
                return state_space
        candidates = self.state_spaces_for_request(request=request)
        if not candidates:
            return None
        return self._state_space_for_candidate(candidates[0])

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

    def _outcome_id(self, component_index: int) -> str:
        if component_index >= len(self.manifest.outcome_space.outcomes):
            raise ObservationGenerationError("component index is outside outcome space")
        return self.manifest.outcome_space.outcomes[component_index].id

    def _latent_coordinates(
        self,
        *,
        component_index: int,
        digit_variant_index: int,
        scaled_factors: tuple[SampleLatentFactor, ...],
        plan: MaterializationPlan,
        variation_values: Mapping[str, object],
    ) -> tuple[Mapping[str, object], ...]:
        records: list[Mapping[str, object]] = []
        for factor in scaled_factors:
            if factor.role == "content":
                values: object = {
                    "digit_index": component_index,
                    "digit_variant_index": digit_variant_index,
                    "outcome_id": self._outcome_id(component_index),
                }
            elif factor.role == "materialization":
                values = dict(plan.resolution_assignment.values)
            else:
                values = dict(variation_values)
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
        include_metadata: bool = True,
        state_space_request: StateSpaceMeasureRequest | None = None,
        component_indices: Iterable[int] | None = None,
        memory_limit_bytes: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
        variation_extent: float = 1.0,
        runtime: TensorRuntime | None = None,
        outcome_ids: tuple[str, ...] | None = None,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedSampleSet:
        """Generate a shape-aware Digits sample set."""

        sample_shape = _sample_shape(shape)
        sample_count = _sample_count(sample_shape)
        requested_component_indices = (
            tuple(component_indices) if component_indices is not None else None
        )
        variation_extent_value = _variation_extent_value(variation_extent)
        state_space = self._default_state_space(
            canonical_variation=variation_extent_value == 0.0,
        )
        if state_space_request is not None:
            if resolution_assignment is not None:
                raise ObservationGenerationError(
                    "state-space request cannot be combined with resolution_assignment"
            )
            candidate = self.state_space_for_request(request=state_space_request)
            if candidate is None:
                if runtime is not None:
                    raise ObservationGenerationError(
                        "tensor generation state-space request matched no candidate"
                    )
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
            resolution_assignment = candidate.resolution_assignment
            if resolution_assignment is None:
                raise ObservationGenerationError(
                    "Digits state-space candidate is missing a resolution assignment"
                )
            if candidate.cardinality is None:
                raise ObservationGenerationError(
                    "Digits state-space candidate is missing cardinality"
                )
            state_space = self._state_space_for_requested_state_count(
                requested_state_count=candidate.cardinality,
                affine_transform_count=_state_space_affine_transform_count(candidate),
                resolution_assignment=resolution_assignment,
            )
        if runtime is not None and outcome_ids is None:
            raise ObservationGenerationError("tensor generation requires outcome_ids")
        shared_component_indices = requested_component_indices
        shared_transform_indices: tuple[int, ...] | None = None
        if runtime is not None and include_metadata:
            if shared_component_indices is None:
                shared_component_indices = self._sample_component_indices(
                    sample_count=sample_count,
                    seed=seed,
                    component_indices=None,
                    component_count=state_space.digit_count,
                    timing=timing,
                    timing_prefix=timing_prefix,
                )
            shared_transform_indices = self._sample_transform_indices(
                sample_count=sample_count,
                seed=seed,
                state_space=state_space,
                timing=timing,
                timing_prefix=timing_prefix,
            )
        fields = None
        targets = None
        if runtime is not None and outcome_ids is not None:
            fields, targets = self._generate_tensors(
                sample_shape=sample_shape,
                seed=seed,
                component_indices=shared_component_indices,
                transform_indices=shared_transform_indices,
                memory_limit_bytes=memory_limit_bytes,
                resolution_assignment=resolution_assignment,
                state_space=state_space,
                runtime=runtime,
                outcome_ids=outcome_ids,
                timing=timing,
                timing_prefix=timing_prefix,
            )
        samples: tuple[GeneratedSample, ...] = ()
        if include_metadata:
            if runtime is not None and not include_fields:
                if shared_component_indices is None:
                    raise ObservationGenerationError(
                        "tensor metadata generation requires component indices"
                    )
                samples = self._generate_tensor_metadata_samples(
                    sample_count=sample_count,
                    component_indices=shared_component_indices,
                    state_space=state_space,
                )
            else:
                samples = self._generate_samples(
                    sample_count=sample_count,
                    seed=seed,
                    include_fields=include_fields,
                    component_indices=shared_component_indices,
                    transform_indices=shared_transform_indices,
                    memory_limit_bytes=memory_limit_bytes,
                    resolution_assignment=resolution_assignment,
                    state_space=state_space,
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
            fields=fields,
            targets=targets,
        )

    def console_preview_batches(self, *, atom_count: int) -> tuple[Mapping[str, object], ...]:
        """Return browser-preview batches for declared state-space windows."""

        if atom_count != len(self.manifest.outcome_space.outcomes):
            raise ObservationGenerationError("atom_count does not match outcome space")
        return tuple(
            self._console_preview_state_space_window(
                minimum=minimum,
                maximum=maximum,
                seed=401 + index,
            )
            for index, (minimum, maximum) in enumerate(
                _console_preview_state_space_windows
            )
        )

    def _console_preview_state_space_window(
        self,
        *,
        minimum: float,
        maximum: float,
        seed: int,
    ) -> Mapping[str, object]:
        request = StateSpaceMeasureRequest(minimum=minimum, maximum=maximum)
        candidates = self.state_spaces_for_request(request=request)
        samples: list[Mapping[str, object]] = []
        materialization_declaration = ArtifactReference(
            kind="materialization-declaration",
            protocol_id=self.materialization.id,
            record_digest=self.materialization.digest,
        )
        transform = self.formation.variation_transform
        transform_record = transform.to_record()
        scaled_factors = tuple(self.latent_factors.sample_factors)
        state_spaces = tuple(self._state_space_for_candidate(candidate) for candidate in candidates)

        for candidate_index, component_index, transform_index in _preview_sample_coordinates(
            state_spaces,
            limit=_console_preview_sample_limit,
        ):
            state_space = state_spaces[candidate_index]
            if state_space.resolution_assignment is None:
                raise ObservationGenerationError(
                    "Digits state-space candidate is missing a resolution assignment"
                )
            sample_index = len(samples)
            plan = self._materialization_plan(
                seed=seed,
                index=sample_index,
                resolution_assignment=state_space.resolution_assignment,
                materialization_declaration=materialization_declaration,
            )
            variation_coordinate = _constructed_variation_coordinate_record(
                transform=transform,
                component_index=component_index,
                transform_index=transform_index,
                grid=state_space.affine_grid,
            )
            variation_values: Mapping[str, object] = {
                "kind": "constructed-field-variation-transform-samples",
                "bounds": transform_record,
                "state_space": state_space.metadata(),
                "candidate_index": candidate_index,
                "transform_index": transform_index,
                "transform_count": state_space.affine_transform_count,
                "coordinates": [dict(variation_coordinate)],
            }
            field_tensor = self._generate_tensor_fields(
                sample_shape=(1,),
                seed=seed,
                component_indices=(component_index,),
                transform_indices=(transform_index,),
                state_space=state_space,
                resolution_assignment=state_space.resolution_assignment,
                memory_limit_bytes=None,
                runtime=resolve_host_tensor_runtime(),
                timing=None,
                timing_prefix="",
            )
            field_record = self._formed_observations_from_tensor_fields(
                seed=seed,
                plans=(plan,),
                component_indices=(component_index,),
                fields=field_tensor,
                sample_indices=(sample_index,),
            )[0]
            samples.append(
                {
                    "index": sample_index,
                    "outcome_id": self._outcome_id(component_index),
                    "component_index": component_index,
                    "complexity": state_space.complexity,
                    "state_space_measure": state_space.measure().to_record(),
                    "field_shape": list(field_record.field.shape),
                    "image_data_url": _field_to_png_data_url(field_record.field),
                    "materialization_plan": plan.to_record(),
                    "latent_coordinates": [
                        dict(coordinate)
                        for coordinate in self._latent_coordinates(
                            component_index=component_index,
                            digit_variant_index=0,
                            scaled_factors=scaled_factors,
                            plan=plan,
                            variation_values=variation_values,
                        )
                    ],
                }
            )
        samples.sort(key=lambda sample: _sample_display_key(sample, len(samples)))
        return {
            "mode": "state-space-window",
            "label": f"[{minimum:g}, {maximum:g}]",
            "seed": seed,
            "sample_count": len(samples),
            "state_space_window": {
                "measure_id": "log2_state_space_size",
                "minimum": minimum,
                "maximum": maximum,
            },
            "state_space_sizes": [
                candidate.cardinality
                for candidate in candidates
                if candidate.cardinality is not None
            ],
            "presentation": {
                "sample_card_density": "compact" if len(samples) > 80 else "standard",
                "aggregate_mode": False,
            },
            "samples": samples,
        }

    def _state_space_for_candidate(
        self,
        candidate: StateSpaceCandidate,
    ) -> _DigitsStateSpace:
        if candidate.cardinality is None:
            raise ObservationGenerationError(
                "Digits state-space candidate is missing cardinality"
            )
        if candidate.resolution_assignment is None:
            raise ObservationGenerationError(
                "Digits state-space candidate is missing a resolution assignment"
            )
        return self._state_space_for_requested_state_count(
            requested_state_count=candidate.cardinality,
            affine_transform_count=_state_space_affine_transform_count(candidate),
            resolution_assignment=candidate.resolution_assignment,
        )

    def _resolution_assignment_for_state_space_request(
        self,
        request: StateSpaceMeasureRequest,
    ) -> AxisAssignment:
        side = _state_space_canvas_side_for_complexity(request.maximum)
        minimum_assignment = self.materialization.minimum_resolution()
        values = dict(minimum_assignment.values)
        values[self.formation.width_axis] = max(
            values.get(self.formation.width_axis, 1),
            side,
        )
        values[self.formation.height_axis] = max(
            values.get(self.formation.height_axis, 1),
            side,
        )
        return AxisAssignment(values=values)

    def _resolution_assignment_for_requested_state_count(
        self,
        requested_state_count: int,
    ) -> AxisAssignment:
        _require_generation_positive_integer(
            requested_state_count,
            "requested_state_count",
        )
        minimum_assignment = self.materialization.minimum_resolution()
        width_axis = self.formation.width_axis
        height_axis = self.formation.height_axis
        side = _state_space_canvas_side_for_complexity(math.log2(requested_state_count))
        values = dict(minimum_assignment.values)
        values[width_axis] = max(values.get(width_axis, 1), side)
        values[height_axis] = max(values.get(height_axis, 1), side)
        return AxisAssignment(values=values)


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


def _state_space_measure(complexity: float) -> StateSpaceMeasureValue:
    return StateSpaceMeasureValue(
        value=complexity,
    )


def _state_space_canvas_side_for_complexity(complexity: float) -> int:
    if not math.isfinite(float(complexity)):
        raise ObservationGenerationError("state-space complexity must be finite")
    if complexity < 1.0:
        raise ObservationGenerationError("state-space complexity must be at least 1")
    frontier_area = (2.0**complexity) / _state_space_canvas_density
    side = max(_state_space_canvas_minimum_side, math.ceil(math.sqrt(frontier_area)))
    return _state_space_canvas_side_step * math.ceil(
        side / _state_space_canvas_side_step
    )


def _ceil_state_space_cardinality(complexity: float) -> int:
    value = _state_space_cardinality_float(complexity)
    tolerance = max(1.0, abs(value)) * _state_space_cardinality_relative_tolerance
    return max(2, math.ceil(value - tolerance))


def _floor_state_space_cardinality(complexity: float) -> int:
    value = _state_space_cardinality_float(complexity)
    tolerance = max(1.0, abs(value)) * _state_space_cardinality_relative_tolerance
    return max(2, math.floor(value + tolerance))


def _state_space_cardinality_float(complexity: float) -> float:
    if not math.isfinite(float(complexity)):
        raise ObservationGenerationError("state-space complexity must be finite")
    return 2.0**complexity


def _state_space_canvas_side_from_assignment(
    resolution_assignment: AxisAssignment | None,
) -> int:
    if resolution_assignment is None:
        return _state_space_canvas_minimum_side
    values = tuple(resolution_assignment.values.values())
    if not values:
        return _state_space_canvas_minimum_side
    return max(_state_space_canvas_minimum_side, max(values))


def _constructed_affine_bounds_for_canvas_side(
    side: int,
) -> dict[str, tuple[float, float]]:
    _require_generation_positive_integer(side, "side")
    return {
        "x_translation": _constructed_affine_translation_bounds,
        "y_translation": _constructed_affine_translation_bounds,
        "scale": _constructed_affine_scale_bounds,
        "rotation": _constructed_affine_rotation_bounds,
        "x_shear": _constructed_affine_shear_bounds,
    }


def _constructed_affine_axis_capacities(
    *,
    bounds: Mapping[str, tuple[float, float]],
    side: int,
) -> tuple[int, int, int, int, int]:
    _require_generation_positive_integer(side, "side")
    radius_pixels = max(1.0, side * _constructed_affine_effective_radius_fraction)
    translation_step = 1.0 / side
    radius_step = 1.0 / radius_pixels
    return (
        _grid_capacity(bounds["x_translation"], minimum_step=translation_step),
        _grid_capacity(bounds["y_translation"], minimum_step=translation_step),
        _grid_capacity(bounds["scale"], minimum_step=radius_step),
        _grid_capacity(bounds["rotation"], minimum_step=radius_step),
        _grid_capacity(bounds["x_shear"], minimum_step=radius_step),
    )


def _grid_capacity(bounds: tuple[float, float], *, minimum_step: float) -> int:
    lower, upper = bounds
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ObservationGenerationError("grid bounds must be finite")
    if upper < lower:
        raise ObservationGenerationError("grid upper bound must not be below lower bound")
    if not math.isfinite(minimum_step) or minimum_step <= 0.0:
        raise ObservationGenerationError("grid minimum step must be positive")
    width = upper - lower
    if width <= 0.0:
        return 1
    return max(1, math.floor(width / minimum_step) + 1)


def _variation_extent_value(variation_extent: float) -> float:
    try:
        value = float(variation_extent)
    except (TypeError, ValueError) as error:
        raise ObservationGenerationError("variation_extent must be finite") from error
    if not math.isfinite(value):
        raise ObservationGenerationError("variation_extent must be finite")
    if value < 0.0 or value > 1.0:
        raise ObservationGenerationError("variation_extent must be between 0 and 1")
    return value


def _state_space_affine_transform_count(candidate: StateSpaceCandidate) -> int:
    value = candidate.metadata.get("affine_transform_count")
    if type(value) is not int or value <= 0:
        raise ObservationGenerationError(
            "Digits state-space candidate metadata is missing affine_transform_count"
        )
    return value


def _preview_index_selection(count: int, *, limit: int) -> tuple[int, ...]:
    if count <= 0 or limit <= 0:
        return ()
    if count <= limit:
        return tuple(range(count))
    if limit == 1:
        return (0,)
    indices = {
        round(index * (count - 1) / (limit - 1))
        for index in range(limit)
    }
    return tuple(sorted(indices))


def _preview_sample_coordinates(
    state_spaces: tuple[_DigitsStateSpace, ...],
    *,
    limit: int,
) -> tuple[tuple[int, int, int], ...]:
    full_count = sum(
        state_space.digit_count * state_space.affine_transform_count
        for state_space in state_spaces
    )
    if full_count <= limit:
        return tuple(
            (state_space_index, component_index, transform_index)
            for state_space_index, state_space in enumerate(state_spaces)
            for component_index in range(state_space.digit_count)
            for transform_index in range(state_space.affine_transform_count)
        )
    coordinates: list[tuple[int, int, int]] = []
    offset = 0
    selected_offsets = set(_preview_index_selection(full_count, limit=limit))
    for state_space_index, state_space in enumerate(state_spaces):
        state_space_count = state_space.digit_count * state_space.affine_transform_count
        for selected_offset in sorted(
            value
            for value in selected_offsets
            if offset <= value < offset + state_space_count
        ):
            local_offset = selected_offset - offset
            component_index = local_offset // state_space.affine_transform_count
            transform_index = local_offset % state_space.affine_transform_count
            coordinates.append((state_space_index, component_index, transform_index))
        offset += state_space_count
    return tuple(coordinates)


def _constructed_affine_grid(
    transform_count: int,
    *,
    resolution_assignment: AxisAssignment | None = None,
) -> _ConstructedAffineGrid:
    _require_generation_positive_integer(transform_count, "transform_count")
    side = _state_space_canvas_side_from_assignment(resolution_assignment)
    bounds = _constructed_affine_bounds_for_canvas_side(side)
    if transform_count <= _constructed_affine_preset_max_count:
        return _ConstructedAffineGrid(
            x_translation=1,
            y_translation=1,
            scale=1,
            rotation=1,
            x_shear=1,
            x_translation_bounds=bounds["x_translation"],
            y_translation_bounds=bounds["y_translation"],
            scale_bounds=bounds["scale"],
            rotation_bounds=bounds["rotation"],
            x_shear_bounds=bounds["x_shear"],
            preset_count=transform_count,
        )
    capacities = _constructed_affine_axis_capacities(bounds=bounds, side=side)
    counts = _constructed_affine_counts_for_transform_count(
        transform_count=transform_count,
        capacities=capacities,
    )
    return _ConstructedAffineGrid(
        x_translation=counts[0],
        y_translation=counts[1],
        scale=counts[2],
        rotation=counts[3],
        x_shear=counts[4],
        x_translation_bounds=bounds["x_translation"],
        y_translation_bounds=bounds["y_translation"],
        scale_bounds=bounds["scale"],
        rotation_bounds=bounds["rotation"],
        x_shear_bounds=bounds["x_shear"],
    )


def _constructed_affine_grids_in_transform_count_range(
    *,
    minimum_transform_count: int,
    maximum_transform_count: int,
    resolution_assignment: AxisAssignment,
) -> tuple[_ConstructedAffineGrid, ...]:
    _require_generation_positive_integer(
        minimum_transform_count,
        "minimum_transform_count",
    )
    _require_generation_positive_integer(
        maximum_transform_count,
        "maximum_transform_count",
    )
    if maximum_transform_count < minimum_transform_count:
        return ()
    side = _state_space_canvas_side_from_assignment(resolution_assignment)
    bounds = _constructed_affine_bounds_for_canvas_side(side)
    capacities = _constructed_affine_axis_capacities(bounds=bounds, side=side)
    grids: list[_ConstructedAffineGrid] = []
    seen_products: set[int] = set()
    for preset_count in range(
        minimum_transform_count,
        min(maximum_transform_count, _constructed_affine_preset_max_count) + 1,
    ):
        grids.append(
            _ConstructedAffineGrid(
                x_translation=1,
                y_translation=1,
                scale=1,
                rotation=1,
                x_shear=1,
                x_translation_bounds=bounds["x_translation"],
                y_translation_bounds=bounds["y_translation"],
                scale_bounds=bounds["scale"],
                rotation_bounds=bounds["rotation"],
                x_shear_bounds=bounds["x_shear"],
                preset_count=preset_count,
            )
        )
        seen_products.add(preset_count)
    for transform_count in range(
        max(
            minimum_transform_count,
            _constructed_affine_preset_max_count + 1,
        ),
        maximum_transform_count + 1,
    ):
        if transform_count in seen_products:
            continue
        try:
            counts = _constructed_affine_counts_for_transform_count(
                transform_count=transform_count,
                capacities=capacities,
            )
        except ObservationGenerationError:
            continue
        seen_products.add(transform_count)
        grids.append(
            _ConstructedAffineGrid(
                x_translation=counts[0],
                y_translation=counts[1],
                scale=counts[2],
                rotation=counts[3],
                x_shear=counts[4],
                x_translation_bounds=bounds["x_translation"],
                y_translation_bounds=bounds["y_translation"],
                scale_bounds=bounds["scale"],
                rotation_bounds=bounds["rotation"],
                x_shear_bounds=bounds["x_shear"],
            )
        )
        if len(grids) >= _maximum_state_space_candidates_per_request:
            break
    return tuple(sorted(grids, key=lambda grid: (grid.transform_count, grid.counts)))


def _constructed_affine_counts_for_transform_count(
    *,
    transform_count: int,
    capacities: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    counts = [1, 1, 1, 1, 1]
    for factor in sorted(_prime_factors(transform_count), reverse=True):
        candidates = tuple(
            index
            for index, (count, capacity) in enumerate(zip(counts, capacities, strict=True))
            if count * factor <= capacity
        )
        if not candidates:
            break
        selected = min(
            candidates,
            key=lambda index: _constructed_affine_count_sort_key(
                _with_affine_count_factor(counts=counts, index=index, factor=factor)
            ),
        )
        counts[selected] *= factor
    if math.prod(counts) == transform_count:
        return (counts[0], counts[1], counts[2], counts[3], counts[4])
    raise ObservationGenerationError(
        "requested affine transform count exceeds resolution-aware grid capacity"
    )


def _with_affine_count_factor(
    *,
    counts: Sequence[int],
    index: int,
    factor: int,
) -> tuple[int, int, int, int, int]:
    return (
        counts[0] * factor if index == 0 else counts[0],
        counts[1] * factor if index == 1 else counts[1],
        counts[2] * factor if index == 2 else counts[2],
        counts[3] * factor if index == 3 else counts[3],
        counts[4] * factor if index == 4 else counts[4],
    )


def _constructed_affine_count_sort_key(
    counts: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int, int]:
    return (
        max(counts),
        counts[4],
        counts[3],
        counts[2],
        counts[1],
        counts[0],
    )


def _prime_factors(value: int) -> tuple[int, ...]:
    _require_generation_positive_integer(value, "value")
    factors: list[int] = []
    divisor = 2
    remaining = value
    while divisor * divisor <= remaining:
        while remaining % divisor == 0:
            factors.append(divisor)
            remaining //= divisor
        divisor += 1 if divisor == 2 else 2
    if remaining > 1:
        factors.append(remaining)
    return tuple(factors)


def _constructed_affine_indices(
    *,
    transform_index: int,
    grid: _ConstructedAffineGrid,
) -> dict[str, int]:
    if type(transform_index) is not int or transform_index < 0:
        raise ObservationGenerationError("transform_index must be a nonnegative integer")
    if transform_index >= grid.transform_count:
        raise ObservationGenerationError("transform_index must be below transform_count")
    if grid.preset_count is not None:
        return {"preset": transform_index}
    remainder = transform_index
    indices: list[int] = []
    for axis_count in grid.counts:
        indices.append(remainder % axis_count)
        remainder //= axis_count
    return dict(zip(_constructed_affine_axis_names, indices, strict=True))


def _constructed_variation_coordinate_record(
    *,
    transform: VariationTransformDeclaration,
    component_index: int,
    transform_index: int,
    grid: _ConstructedAffineGrid,
) -> Mapping[str, object]:
    if type(transform_index) is not int or transform_index < 0:
        raise ObservationGenerationError("transform_index must be a nonnegative integer")
    if transform_index >= grid.transform_count:
        raise ObservationGenerationError("transform_index must be below transform_count")
    if grid.transform_count == 1:
        return _identity_variation_coordinate_record(
            transform=transform,
            component_index=component_index,
        )
    spatial = transform.spatial_affine
    indices = _constructed_affine_indices(
        transform_index=transform_index,
        grid=grid,
    )
    parameters = _constructed_affine_parameters(
        spatial=spatial,
        grid=grid,
        indices=indices,
    )
    scale = parameters["scale"]
    rotation = parameters["rotation"]
    shear = parameters["x_shear"]
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    matrix = [
        [scale * cosine, shear - scale * sine, parameters["x_translation"]],
        [scale * sine, scale * cosine, parameters["y_translation"]],
        [0.0, 0.0, 1.0],
    ]
    return {
        "kind": "field-variation-transform-coordinate",
        "component_index": component_index,
        "spatial_affine": {
            "kind": "spatial-affine-coordinate",
            "coordinate_system": spatial.coordinate_system,
            "matrix": matrix,
        },
        "constructed_affine_indices": indices,
        "constructed_affine_parameters": parameters,
    }


def _constructed_affine_parameters(
    *,
    spatial: SpatialAffineVariation,
    grid: _ConstructedAffineGrid,
    indices: Mapping[str, int],
) -> dict[str, float]:
    if grid.preset_count is not None:
        preset_index = indices.get("preset")
        if type(preset_index) is not int:
            raise ObservationGenerationError("affine preset index is missing")
        return _constructed_affine_preset_parameters(
            spatial=spatial,
            grid=grid,
            preset_index=preset_index,
        )
    return {
        "x_translation": _grid_value(
            _bounded_interval(
                grid.x_translation_bounds,
                lower_bound=spatial.matrix[0][2][0],
                upper_bound=spatial.matrix[0][2][1],
            ),
            index=indices["x_translation"],
            count=grid.x_translation,
        ),
        "y_translation": _grid_value(
            _bounded_interval(
                grid.y_translation_bounds,
                lower_bound=spatial.matrix[1][2][0],
                upper_bound=spatial.matrix[1][2][1],
            ),
            index=indices["y_translation"],
            count=grid.y_translation,
        ),
        "scale": _grid_value(
            _bounded_interval(
                grid.scale_bounds,
                lower_bound=max(spatial.matrix[0][0][0], spatial.matrix[1][1][0]),
                upper_bound=min(spatial.matrix[0][0][1], spatial.matrix[1][1][1]),
            ),
            index=indices["scale"],
            count=grid.scale,
        ),
        "rotation": _grid_value(
            _bounded_interval(
                grid.rotation_bounds,
                lower_bound=spatial.matrix[1][0][0],
                upper_bound=spatial.matrix[1][0][1],
            ),
            index=indices["rotation"],
            count=grid.rotation,
        ),
        "x_shear": _grid_value(
            _bounded_interval(
                grid.x_shear_bounds,
                lower_bound=spatial.matrix[0][1][0],
                upper_bound=spatial.matrix[0][1][1],
            ),
            index=indices["x_shear"],
            count=grid.x_shear,
        ),
    }


def _constructed_affine_preset_parameters(
    *,
    spatial: SpatialAffineVariation,
    grid: _ConstructedAffineGrid,
    preset_index: int,
) -> dict[str, float]:
    if grid.preset_count is None:
        raise ObservationGenerationError("affine preset count is missing")
    if preset_index < 0 or preset_index >= grid.preset_count:
        raise ObservationGenerationError("affine preset index is out of range")
    coordinates = _constructed_affine_preset_unit_coordinates(
        preset_index=preset_index,
        preset_count=grid.preset_count,
    )
    return {
        "x_translation": _preset_grid_value(
            _bounded_interval(
                grid.x_translation_bounds,
                lower_bound=spatial.matrix[0][2][0],
                upper_bound=spatial.matrix[0][2][1],
            ),
            fraction=coordinates[0],
        ),
        "y_translation": _preset_grid_value(
            _bounded_interval(
                grid.y_translation_bounds,
                lower_bound=spatial.matrix[1][2][0],
                upper_bound=spatial.matrix[1][2][1],
            ),
            fraction=coordinates[1],
        ),
        "scale": _preset_grid_value(
            _bounded_interval(
                grid.scale_bounds,
                lower_bound=max(spatial.matrix[0][0][0], spatial.matrix[1][1][0]),
                upper_bound=min(spatial.matrix[0][0][1], spatial.matrix[1][1][1]),
            ),
            fraction=coordinates[2],
        ),
        "rotation": _preset_grid_value(
            _bounded_interval(
                grid.rotation_bounds,
                lower_bound=spatial.matrix[1][0][0],
                upper_bound=spatial.matrix[1][0][1],
            ),
            fraction=coordinates[3],
        ),
        "x_shear": _preset_grid_value(
            _bounded_interval(
                grid.x_shear_bounds,
                lower_bound=spatial.matrix[0][1][0],
                upper_bound=spatial.matrix[0][1][1],
            ),
            fraction=coordinates[4],
        ),
    }


def _constructed_affine_preset_unit_coordinates(
    *,
    preset_index: int,
    preset_count: int,
) -> tuple[float, float, float, float, float]:
    if preset_count == 1:
        return (0.5, 0.5, 0.5, 0.5, 0.5)
    presets = (
        (0.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0, 1.0),
        (0.5, 0.5, 0.5, 0.5, 0.5),
        (0.0, 1.0, 1.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 1.0, 0.0),
        (0.25, 0.75, 0.25, 0.75, 0.25),
        (0.75, 0.25, 0.75, 0.25, 0.75),
        (0.5, 0.0, 0.5, 1.0, 0.0),
    )
    if preset_count > len(presets):
        raise ObservationGenerationError("affine preset count exceeds preset table")
    return presets[preset_index]


def _bounded_interval(
    requested: tuple[float, float],
    *,
    lower_bound: float,
    upper_bound: float,
) -> tuple[float, float]:
    lower = max(requested[0], lower_bound)
    upper = min(requested[1], upper_bound)
    if upper < lower:
        center = (lower_bound + upper_bound) / 2.0
        return (center, center)
    return (lower, upper)


def _preset_grid_value(bounds: tuple[float, float], *, fraction: float) -> float:
    lower, upper = bounds
    return lower + (upper - lower) * fraction


def _grid_value(bounds: tuple[float, float], *, index: int, count: int) -> float:
    lower, upper = bounds
    if count <= 1:
        return (lower + upper) / 2.0
    return lower + (upper - lower) * (index / (count - 1))


def _runtime_long_tensor(*, runtime: TensorRuntime, values: Any) -> Any:
    backend = tensor_runtime_backend(runtime)
    if hasattr(values, "to") and hasattr(values, "dtype"):
        return values.to(device=runtime.device, dtype=backend.long)
    return backend.tensor(values, dtype=backend.long, device=runtime.device)


def _constructed_affine_matrix_tensors(
    *,
    transform: VariationTransformDeclaration,
    grid: _ConstructedAffineGrid,
    transform_indices: Any,
    runtime: TensorRuntime,
) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    backend = tensor_runtime_backend(runtime)
    transform_indices = transform_indices.reshape(-1)
    sample_count = int(transform_indices.numel())
    if grid.transform_count == 1:
        ones = backend.ones(sample_count, dtype=backend.float32, device=runtime.device)
        zeros = backend.zeros(sample_count, dtype=backend.float32, device=runtime.device)
        return ones, zeros, zeros, zeros, ones, zeros, ones
    spatial = transform.spatial_affine
    if grid.preset_count is not None:
        preset_coordinates = backend.tensor(
            [
                _constructed_affine_preset_unit_coordinates(
                    preset_index=index,
                    preset_count=grid.preset_count,
                )
                for index in range(grid.preset_count)
            ],
            dtype=backend.float32,
            device=runtime.device,
        ).index_select(0, transform_indices)
        x_translation = _tensor_preset_grid_value(
            _bounded_interval(
                grid.x_translation_bounds,
                lower_bound=spatial.matrix[0][2][0],
                upper_bound=spatial.matrix[0][2][1],
            ),
            fraction=preset_coordinates[:, 0],
        )
        y_translation = _tensor_preset_grid_value(
            _bounded_interval(
                grid.y_translation_bounds,
                lower_bound=spatial.matrix[1][2][0],
                upper_bound=spatial.matrix[1][2][1],
            ),
            fraction=preset_coordinates[:, 1],
        )
        scale = _tensor_preset_grid_value(
            _bounded_interval(
                grid.scale_bounds,
                lower_bound=max(spatial.matrix[0][0][0], spatial.matrix[1][1][0]),
                upper_bound=min(spatial.matrix[0][0][1], spatial.matrix[1][1][1]),
            ),
            fraction=preset_coordinates[:, 2],
        )
        rotation = _tensor_preset_grid_value(
            _bounded_interval(
                grid.rotation_bounds,
                lower_bound=spatial.matrix[1][0][0],
                upper_bound=spatial.matrix[1][0][1],
            ),
            fraction=preset_coordinates[:, 3],
        )
        x_shear = _tensor_preset_grid_value(
            _bounded_interval(
                grid.x_shear_bounds,
                lower_bound=spatial.matrix[0][1][0],
                upper_bound=spatial.matrix[0][1][1],
            ),
            fraction=preset_coordinates[:, 4],
        )
    else:
        affine_indices = _constructed_affine_index_tensors(
            transform_indices=transform_indices,
            grid=grid,
            runtime=runtime,
        )
        x_translation = _tensor_grid_value(
            _bounded_interval(
                grid.x_translation_bounds,
                lower_bound=spatial.matrix[0][2][0],
                upper_bound=spatial.matrix[0][2][1],
            ),
            indices=affine_indices[0],
            count=grid.x_translation,
        )
        y_translation = _tensor_grid_value(
            _bounded_interval(
                grid.y_translation_bounds,
                lower_bound=spatial.matrix[1][2][0],
                upper_bound=spatial.matrix[1][2][1],
            ),
            indices=affine_indices[1],
            count=grid.y_translation,
        )
        scale = _tensor_grid_value(
            _bounded_interval(
                grid.scale_bounds,
                lower_bound=max(spatial.matrix[0][0][0], spatial.matrix[1][1][0]),
                upper_bound=min(spatial.matrix[0][0][1], spatial.matrix[1][1][1]),
            ),
            indices=affine_indices[2],
            count=grid.scale,
        )
        rotation = _tensor_grid_value(
            _bounded_interval(
                grid.rotation_bounds,
                lower_bound=spatial.matrix[1][0][0],
                upper_bound=spatial.matrix[1][0][1],
            ),
            indices=affine_indices[3],
            count=grid.rotation,
        )
        x_shear = _tensor_grid_value(
            _bounded_interval(
                grid.x_shear_bounds,
                lower_bound=spatial.matrix[0][1][0],
                upper_bound=spatial.matrix[0][1][1],
            ),
            indices=affine_indices[4],
            count=grid.x_shear,
        )
    cosine = backend.cos(rotation)
    sine = backend.sin(rotation)
    m00 = scale * cosine
    m01 = x_shear - scale * sine
    m10 = scale * sine
    m11 = scale * cosine
    width_scale = backend.maximum(
        backend.sqrt(m00 * m00 + m10 * m10),
        backend.sqrt(m01 * m01 + m11 * m11),
    )
    return m00, m01, x_translation, m10, m11, y_translation, width_scale


def _constructed_affine_index_tensors(
    *,
    transform_indices: Any,
    grid: _ConstructedAffineGrid,
    runtime: TensorRuntime,
) -> tuple[Any, Any, Any, Any, Any]:
    backend = tensor_runtime_backend(runtime)
    remainder = transform_indices
    indices: list[Any] = []
    for axis_count in grid.counts:
        indices.append(remainder.remainder(axis_count))
        remainder = backend.div(remainder, axis_count, rounding_mode="floor")
    return (indices[0], indices[1], indices[2], indices[3], indices[4])


def _tensor_grid_value(bounds: tuple[float, float], *, indices: Any, count: int) -> Any:
    lower, upper = bounds
    float_indices = indices.float()
    if count <= 1:
        return float_indices * 0.0 + ((lower + upper) / 2.0)
    return lower + (upper - lower) * (float_indices / (count - 1))


def _tensor_preset_grid_value(bounds: tuple[float, float], *, fraction: Any) -> Any:
    lower, upper = bounds
    return lower + (upper - lower) * fraction


def _quadratic_control_points(
    mark: ComponentMark,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    points = mark.control_points
    if len(points) == 3:
        return (points[0], points[1], points[2])
    if len(points) == 2:
        midpoint = (
            (points[0][0] + points[1][0]) / 2.0,
            (points[0][1] + points[1][1]) / 2.0,
        )
        return (points[0], midpoint, points[1])
    raise TensorRuntimeError("Digits marks must be linear or quadratic curves")


def _mark_segment_windows(
    *,
    mark: ComponentMark,
    transform: VariationTransformDeclaration,
    grid: _ConstructedAffineGrid,
    width: int,
    height: int,
) -> tuple[_SegmentWindow, ...]:
    curve_points = _sampled_quadratic_points(_quadratic_control_points(mark))
    matrices = _constructed_affine_window_matrices(transform=transform, grid=grid)
    windows: list[_SegmentWindow] = []
    for start, stop in zip(curve_points, curve_points[1:], strict=False):
        x_values: list[float] = []
        y_values: list[float] = []
        radii: list[float] = []
        for m00, m01, m02, m10, m11, m12, width_scale in matrices:
            radii.append(width_scale * float(mark.width) / 2.0)
            for x_value, y_value in (start, stop):
                centered_x = x_value - 0.5
                centered_y = y_value - 0.5
                x_values.append(
                    (0.5 + m00 * centered_x + m01 * centered_y + m02) * width
                )
                y_values.append(
                    (0.5 + m10 * centered_x + m11 * centered_y + m12) * height
                )
        radius = max(radii)
        x_start = max(0, math.floor(min(x_values) - radius - 0.5))
        x_stop = min(width, math.ceil(max(x_values) + radius + 0.5))
        y_start = max(0, math.floor(min(y_values) - radius - 0.5))
        y_stop = min(height, math.ceil(max(y_values) + radius + 0.5))
        windows.append((x_start, x_stop, y_start, y_stop))
    return tuple(windows)


def _combined_segment_window(windows: tuple[_SegmentWindow, ...]) -> _SegmentWindow:
    if not windows:
        return (0, 0, 0, 0)
    return (
        min(window[0] for window in windows),
        max(window[1] for window in windows),
        min(window[2] for window in windows),
        max(window[3] for window in windows),
    )


def _sampled_quadratic_points(
    controls: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    points: list[tuple[float, float]] = []
    for index in range(_batch_render_curve_sample_count):
        t = index / (_batch_render_curve_sample_count - 1)
        one_minus_t = 1.0 - t
        x_value = (
            one_minus_t * one_minus_t * controls[0][0]
            + 2.0 * one_minus_t * t * controls[1][0]
            + t * t * controls[2][0]
        )
        y_value = (
            one_minus_t * one_minus_t * controls[0][1]
            + 2.0 * one_minus_t * t * controls[1][1]
            + t * t * controls[2][1]
        )
        points.append((x_value, y_value))
    return tuple(points)


def _constructed_affine_window_matrices(
    *,
    transform: VariationTransformDeclaration,
    grid: _ConstructedAffineGrid,
) -> tuple[tuple[float, float, float, float, float, float, float], ...]:
    if grid.transform_count == 1:
        return ((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),)
    spatial = transform.spatial_affine
    if grid.preset_count is not None:
        parameters = tuple(
            _constructed_affine_preset_parameters(
                spatial=spatial,
                grid=grid,
                preset_index=index,
            )
            for index in range(grid.preset_count)
        )
    else:
        ranges = _constructed_affine_parameter_ranges(spatial=spatial, grid=grid)
        parameters = tuple(
            {
                "x_translation": x_translation,
                "y_translation": y_translation,
                "scale": scale,
                "rotation": rotation,
                "x_shear": x_shear,
            }
            for x_translation, y_translation, scale, rotation, x_shear in itertools.product(
                _range_endpoints(ranges["x_translation"]),
                _range_endpoints(ranges["y_translation"]),
                _range_endpoints(ranges["scale"]),
                _range_endpoints(ranges["rotation"]),
                _range_endpoints(ranges["x_shear"]),
            )
        )
    return tuple(_constructed_affine_matrix_from_parameters(parameter) for parameter in parameters)


def _constructed_affine_parameter_ranges(
    *,
    spatial: SpatialAffineVariation,
    grid: _ConstructedAffineGrid,
) -> dict[str, tuple[float, float]]:
    return {
        "x_translation": _bounded_interval(
            grid.x_translation_bounds,
            lower_bound=spatial.matrix[0][2][0],
            upper_bound=spatial.matrix[0][2][1],
        ),
        "y_translation": _bounded_interval(
            grid.y_translation_bounds,
            lower_bound=spatial.matrix[1][2][0],
            upper_bound=spatial.matrix[1][2][1],
        ),
        "scale": _bounded_interval(
            grid.scale_bounds,
            lower_bound=max(spatial.matrix[0][0][0], spatial.matrix[1][1][0]),
            upper_bound=min(spatial.matrix[0][0][1], spatial.matrix[1][1][1]),
        ),
        "rotation": _bounded_interval(
            grid.rotation_bounds,
            lower_bound=spatial.matrix[1][0][0],
            upper_bound=spatial.matrix[1][0][1],
        ),
        "x_shear": _bounded_interval(
            grid.x_shear_bounds,
            lower_bound=spatial.matrix[0][1][0],
            upper_bound=spatial.matrix[0][1][1],
        ),
    }


def _range_endpoints(bounds: tuple[float, float]) -> tuple[float, ...]:
    if bounds[0] == bounds[1]:
        return (bounds[0],)
    return bounds


def _constructed_affine_matrix_from_parameters(
    parameters: Mapping[str, float],
) -> tuple[float, float, float, float, float, float, float]:
    scale = parameters["scale"]
    rotation = parameters["rotation"]
    shear = parameters["x_shear"]
    cosine = math.cos(rotation)
    sine = math.sin(rotation)
    m00 = scale * cosine
    m01 = shear - scale * sine
    m10 = scale * sine
    m11 = scale * cosine
    width_scale = max(
        math.sqrt(m00 * m00 + m10 * m10),
        math.sqrt(m01 * m01 + m11 * m11),
    )
    return (
        m00,
        m01,
        parameters["x_translation"],
        m10,
        m11,
        parameters["y_translation"],
        width_scale,
    )


_digits_triton_render_kernel_cache: tuple[Any, Any] | None = None


def _digits_triton_render_kernel() -> tuple[Any, Any]:
    global _digits_triton_render_kernel_cache
    if _digits_triton_render_kernel_cache is not None:
        return _digits_triton_render_kernel_cache
    triton = importlib.import_module("triton")
    tl = importlib.import_module("triton.language")
    namespace = {"triton": triton, "tl": tl}
    source = """
@triton.jit
def kernel(
    fields,
    component_indices,
    m00_values,
    m01_values,
    m02_values,
    m10_values,
    m11_values,
    m12_values,
    width_scale_values,
    mark_offsets,
    mark_values,
    mark_widths,
    curve_points,
    max_component_mark_count,
    total_elements,
    height,
    width,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    active = offsets < total_elements
    pixel_index = offsets % (height * width)
    sample_index = offsets // (height * width)
    y_index = pixel_index // width
    x_index = pixel_index % width
    x_center = x_index.to(tl.float32) + 0.5
    y_center = y_index.to(tl.float32) + 0.5

    component_index = tl.load(component_indices + sample_index, mask=active, other=0)
    mark_index = tl.load(mark_offsets + component_index, mask=active, other=0)
    mark_stop = tl.load(mark_offsets + component_index + 1, mask=active, other=0)

    m00 = tl.load(m00_values + sample_index, mask=active, other=1.0)
    m01 = tl.load(m01_values + sample_index, mask=active, other=0.0)
    m02 = tl.load(m02_values + sample_index, mask=active, other=0.0)
    m10 = tl.load(m10_values + sample_index, mask=active, other=0.0)
    m11 = tl.load(m11_values + sample_index, mask=active, other=1.0)
    m12 = tl.load(m12_values + sample_index, mask=active, other=0.0)
    width_scale = tl.load(width_scale_values + sample_index, mask=active, other=1.0)
    value = tl.full((block_size,), 0.0, tl.float32)

    mark_slot = 0
    while mark_slot < max_component_mark_count:
        current_mark = mark_index + mark_slot
        live_mark = active & (current_mark < mark_stop)
        mark_value = tl.load(mark_values + current_mark, mask=live_mark, other=0.0)
        mark_width = tl.load(mark_widths + current_mark, mask=live_mark, other=0.0)
        threshold = (width_scale * mark_width / 2.0) * (width_scale * mark_width / 2.0)
        distance_squared = tl.full((block_size,), float("inf"), tl.float32)

        segment_index = 0
        while segment_index < 24:
            start_offset = ((current_mark * 25 + segment_index) * 2)
            stop_offset = start_offset + 2
            raw_sx = tl.load(curve_points + start_offset, mask=live_mark, other=0.5)
            raw_sy = tl.load(curve_points + start_offset + 1, mask=live_mark, other=0.5)
            raw_ex = tl.load(curve_points + stop_offset, mask=live_mark, other=0.5)
            raw_ey = tl.load(curve_points + stop_offset + 1, mask=live_mark, other=0.5)

            centered_sx = raw_sx - 0.5
            centered_sy = raw_sy - 0.5
            centered_ex = raw_ex - 0.5
            centered_ey = raw_ey - 0.5
            sx = (0.5 + m00 * centered_sx + m01 * centered_sy + m02) * width
            sy = (0.5 + m10 * centered_sx + m11 * centered_sy + m12) * height
            ex = (0.5 + m00 * centered_ex + m01 * centered_ey + m02) * width
            ey = (0.5 + m10 * centered_ex + m11 * centered_ey + m12) * height
            dx = ex - sx
            dy = ey - sy
            length_squared = dx * dx + dy * dy
            safe_length_squared = tl.where(length_squared == 0.0, 1.0, length_squared)
            segment_t = ((x_center - sx) * dx + (y_center - sy) * dy) / safe_length_squared
            segment_t = tl.minimum(tl.maximum(segment_t, 0.0), 1.0)
            closest_x = sx + segment_t * dx
            closest_y = sy + segment_t * dy
            segment_distance_squared = (
                (x_center - closest_x) * (x_center - closest_x)
                + (y_center - closest_y) * (y_center - closest_y)
            )
            point_distance_squared = (
                (x_center - sx) * (x_center - sx)
                + (y_center - sy) * (y_center - sy)
            )
            segment_distance_squared = tl.where(
                length_squared == 0.0,
                point_distance_squared,
                segment_distance_squared,
            )
            distance_squared = tl.minimum(distance_squared, segment_distance_squared)
            segment_index += 1

        value = tl.maximum(
            value,
            tl.where(live_mark & (distance_squared <= threshold), mark_value, 0.0),
        )
        mark_slot += 1
    tl.store(fields + offsets, value, mask=active)
"""
    filename = f"{__file__}::_digits_triton_render_kernel"
    linecache.cache[filename] = (
        len(source),
        None,
        [line + "\n" for line in source.splitlines()],
        filename,
    )
    exec(compile(source, filename, "exec"), namespace)
    kernel = namespace["kernel"]
    _digits_triton_render_kernel_cache = (kernel, triton)
    return _digits_triton_render_kernel_cache


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


def _require_generation_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ObservationGenerationError(f"{name} must be a positive integer")


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


def _require_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise TensorRuntimeError(f"{name} must be a positive integer")


def _runtime_generator(*, runtime: TensorRuntime, seed: str) -> Any:
    backend = tensor_runtime_backend(runtime)
    generator_seed = int.from_bytes(
        hashlib.sha256(seed.encode("utf-8")).digest()[:8],
        "big",
    ) & ((1 << 63) - 1)
    try:
        generator = backend.Generator(device=runtime.device)
    except (TypeError, RuntimeError):
        generator = backend.Generator()
    generator.manual_seed(generator_seed)
    return generator


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
            "state_space_measure": {
                "kind": "constructed-finite-state-space",
                "measure_id": "log2_state_space_size",
                "formula": "log2(realized_state_count)",
                "digit_count": _state_space_digit_count,
                "affine_transform_family": "constructed-finite-affine-product-grid",
                "target_policy": "symmetric-realized-cardinalities-inside-request-band",
                "description": (
                    "Score-bearing Digits state spaces are requested finite "
                    "single-digit slices. The minimum non-null request is the "
                    "canonical 10-way digit classification problem. Larger "
                    "requests add symmetric finite affine choices for every "
                    "digit, and the benchmark reports the realized cardinality "
                    "instead of forcing exact powers of two."
                ),
            },
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
