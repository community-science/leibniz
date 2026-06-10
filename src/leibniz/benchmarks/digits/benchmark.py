"""Digits benchmark implementation entry point."""

from __future__ import annotations

import base64
import math
import random
import struct
import zlib
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

from leibniz.artifacts import ArtifactReference
from leibniz.benchmark_implementations import Benchmark as BenchmarkProtocol
from leibniz.benchmarks import BenchmarkManifest
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
    ObservationComponent,
    ObservationFormationDeclaration,
    SequenceLayout,
    SpatialAffineVariation,
    VariationTransformDeclaration,
)
from leibniz.observation_generation import (
    ComplexityRequest,
    ComplexityValue,
    GeneratedSample,
    GeneratedSampleSet,
    GenerationRequestOutcome,
    ObservationGenerationError,
)
from leibniz.observation_showcases import (
    ObservationShowcaseManifest,
    ObservationShowcaseSample,
)
from leibniz.outcomes import Outcome, OutcomeSpace
from leibniz.state_space import (
    AxisRegion,
    Distinguishability,
    EnumeratedCellsDomain,
    IntegerRangeDomain,
    ProductRegion,
    RealGridDomain,
    StateSpaceAmbient,
    StateSpaceAxis,
    StateSpaceRegion,
)
from leibniz.tensor_runtime import (
    TensorElementParameter,
    TensorElementProgram,
    TensorElementRecipe,
    TensorRuntime,
    TensorRuntimeError,
    resolve_host_tensor_runtime,
    tensor_runtime_construct_tensor,
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
_complexity_class_digit_count = 10
_complexity_class_canvas_minimum_side = 16
_complexity_class_canvas_side_step = 4
_complexity_class_cardinality_relative_tolerance = 1e-12
_default_constructed_affine_transform_count = 2
_canonical_digits_cardinality = _complexity_class_digit_count
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
class _DigitsComplexityClass:
    affine_grid: _ConstructedAffineGrid
    requested_cardinality: int
    minimum_address: int = 0
    resolution_assignment: AxisAssignment | None = None

    @property
    def digit_count(self) -> int:
        return _complexity_class_digit_count

    @property
    def affine_transform_count(self) -> int:
        return self.affine_grid.transform_count

    @property
    def cardinality(self) -> int:
        return self.requested_cardinality

    @property
    def maximum_address(self) -> int:
        return self.minimum_address + self.requested_cardinality - 1

    @property
    def complexity(self) -> float:
        return math.log2(self.cardinality)

    def measure(self) -> ComplexityValue:
        return _complexity_value(self.complexity)

    def metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "kind": "digits-requested-finite-complexity-class",
            "digit_count": self.digit_count,
            "output_digit_count": _complexity_class_digit_count,
            "affine_transform_count": self.affine_transform_count,
            "latent_cardinality": self.cardinality,
            "minimum_address": self.minimum_address,
            "maximum_address": self.maximum_address,
            "requested_cardinality": self.requested_cardinality,
            "realized_cardinality": self.cardinality,
            "affine_product_cardinality": self.digit_count * self.affine_transform_count,
            "construction": "symmetric-digits-over-finite-affine-product-grid",
            "affine_grid": self.affine_grid.to_record(),
            "affine_bounds": self.affine_grid.bounds_record(),
            "affine_parameters": list(_constructed_affine_axis_names),
        }
        if self.resolution_assignment is not None:
            width = self.resolution_assignment.require_axis("W")
            height = self.resolution_assignment.require_axis("H")
            metadata["oracle_inference_compute"] = {
                "kind": "oracle-inference-compute-reference-v1",
                "unit": "abstract-ops",
                "value": width * height,
                "components": {
                    "height": height,
                    "width": width,
                    "pixel_count": width * height,
                },
            }
        return metadata

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
        sample_indices: tuple[int, ...],
        include_fields: bool,
        include_artifacts: bool,
        memory_limit_bytes: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
        complexity_class: _DigitsComplexityClass,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
        output_timing_prefix: str = "",
    ) -> tuple[GeneratedSample, ...]:
        """Generate Digits samples by choosing digit, canvas, and affine variation."""

        if type(sample_count) is not int or sample_count < 1:
            raise ObservationGenerationError("sample_count must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        component_index_samples, transform_index_samples = (
            self._sample_state_coordinates(
                sample_count=sample_count,
                seed=seed,
                sample_indices=sample_indices,
                complexity_class=complexity_class,
                timing=timing,
                timing_prefix=timing_prefix,
            )
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
            complexity = complexity_class.complexity
        materialization_declaration = ArtifactReference(
            kind="materialization-declaration",
            protocol_id=self.materialization.id,
            record_digest=self.materialization.digest,
        )
        with _timing_span(
            timing,
            f"{timing_prefix}materialization_plan",
            samples=sample_count,
        ):
            plans = tuple(
                self._materialization_plan(
                    seed=seed,
                    index=sample_indices[index],
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
            complexity_class=complexity_class,
            timing=timing,
            timing_phase=variation_timing_phase,
        )

        with _timing_span(timing, f"{output_timing_prefix}scaled_factors"):
            scaled_factors = tuple(self.latent_factors.sample_factors)

        field_records: tuple[FieldObservation, ...]
        if include_fields:
            with _timing_span(
                timing,
                f"{output_timing_prefix}field_generation",
                samples=sample_count,
            ):
                fields = self._generate_tensor_fields(
                    sample_shape=(sample_count,),
                    seed=seed,
                    sample_indices=sample_indices,
                    complexity_class=complexity_class,
                    resolution_assignment=resolved_resolution_assignment,
                    memory_limit_bytes=memory_limit_bytes,
                    runtime=resolve_host_tensor_runtime(),
                    timing=timing,
                    timing_prefix=output_timing_prefix,
                )
                field_records = self._field_observations_from_tensor_fields(
                    plans=plans,
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
            for row_index, (
                index,
                plan,
                component_index,
                transform_index,
                variation_sample,
            ) in enumerate(
                zip(
                    sample_indices,
                    plans,
                    component_index_samples,
                    transform_index_samples,
                    variation_samples,
                    strict=True,
                )
            ):
                variation_values, variation_coordinates = variation_sample
                field_record = field_records[row_index] if include_fields else None
                artifacts = (
                    {
                        "field_shape": list(field_record.shape),
                        "image_data_url": _field_to_png_data_url(field_record),
                    }
                    if include_artifacts and field_record is not None
                    else None
                )
                digit_variant_index = 0
                samples.append(
                    GeneratedSample(
                        index=index,
                        materialization_plan=plan,
                        width=plan.resolution_assignment.require_axis(self.formation.width_axis),
                        height=plan.resolution_assignment.require_axis(self.formation.height_axis),
                        component_index=component_index,
                        region_component_index=_digits_region_component_index(
                            component_index=component_index,
                            complexity_class=complexity_class,
                        ),
                        axis_coordinates=_digits_region_axis_coordinates(
                            transform_index=transform_index,
                            grid=complexity_class.affine_grid,
                            resolution_assignment=resolved_resolution_assignment,
                        ),
                        variation_coordinates=variation_coordinates,
                        variation_values=variation_values,
                        outcome_id=self._outcome_id(component_index),
                        complexity=complexity,
                        complexity_value=_complexity_value(complexity),
                        latent_coordinates=self._latent_coordinates(
                            component_index=component_index,
                            digit_variant_index=digit_variant_index,
                            scaled_factors=scaled_factors,
                            plan=plan,
                            variation_values=variation_values,
                        ),
                        field=field_record,
                        artifacts=artifacts,
                    )
                )
        return tuple(samples)

    def _generate_tensor_metadata_samples(
        self,
        *,
        sample_count: int,
        seed: int,
        sample_indices: tuple[int, ...],
        complexity_class: _DigitsComplexityClass,
        resolution_assignment: AxisAssignment,
    ) -> tuple[GeneratedSample, ...]:
        component_indices, transform_indices = self._sample_state_coordinates(
            sample_count=sample_count,
            seed=seed,
            sample_indices=sample_indices,
            complexity_class=complexity_class,
            timing=None,
            timing_prefix="",
        )
        complexity = complexity_class.complexity
        return tuple(
            GeneratedSample(
                index=sample_indices[index],
                outcome_id=self._outcome_id(component_index),
                complexity=complexity,
                complexity_value=_complexity_value(complexity),
                component_index=component_index,
                region_component_index=_digits_region_component_index(
                    component_index=component_index,
                    complexity_class=complexity_class,
                ),
                axis_coordinates=_digits_region_axis_coordinates(
                    transform_index=transform_indices[index],
                    grid=complexity_class.affine_grid,
                    resolution_assignment=resolution_assignment,
                ),
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

    def _sample_variation_coordinates(
        self,
        *,
        plans: tuple[MaterializationPlan, ...],
        transform: VariationTransformDeclaration,
        transform_record: Mapping[str, object],
        component_indices: tuple[int, ...],
        transform_indices: tuple[int, ...],
        complexity_class: _DigitsComplexityClass,
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
                    or transform_index >= complexity_class.affine_transform_count
                ):
                    raise ObservationGenerationError(
                        "transform index is outside active transform set"
                    )
                coordinate = _constructed_variation_coordinate_record(
                    transform=transform,
                    component_index=component_index,
                    transform_index=transform_index,
                    grid=complexity_class.affine_grid,
                )
                coordinates = (coordinate,)
                samples.append(
                    (
                        {
                            "kind": "constructed-field-variation-transform-samples",
                            "bounds": transform_record,
                            "complexity_class": complexity_class.metadata(),
                            "transform_index": transform_index,
                            "transform_count": complexity_class.affine_transform_count,
                            "coordinates": [dict(item) for item in coordinates],
                        },
                        coordinates,
                    )
                )
        return tuple(samples)

    def _sample_state_coordinates(
        self,
        *,
        sample_count: int,
        seed: int,
        sample_indices: tuple[int, ...],
        complexity_class: _DigitsComplexityClass,
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        component_indices: list[int] = []
        transform_indices: list[int] = []
        with _timing_span(timing, f"{timing_prefix}sample_state", samples=sample_count):
            for index in range(sample_count):
                state_index = _digits_local_state_index(
                    seed=seed,
                    sample_index=sample_indices[index],
                    cardinality=complexity_class.cardinality,
                )
                component_index, transform_index = _digits_state_coordinate(
                    state_index=state_index,
                    complexity_class=complexity_class,
                )
                component_indices.append(component_index)
                transform_indices.append(transform_index)
        return (tuple(component_indices), tuple(transform_indices))

    def _generate_tensors(
        self,
        *,
        sample_shape: tuple[int, ...],
        seed: int,
        sample_indices: tuple[int, ...],
        complexity_class: _DigitsComplexityClass,
        resolution_assignment: AxisAssignment | None,
        memory_limit_bytes: int | None,
        runtime: TensorRuntime,
        outcome_ids: tuple[str, ...],
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> tuple[Any, Any]:
        """Generate tensor fields and targets directly from the Digits complexity shell."""

        if not outcome_ids:
            raise ObservationGenerationError("tensor generation requires outcome_ids")
        sample_count = _sample_count(sample_shape)
        fields = self._generate_tensor_fields(
            sample_shape=sample_shape,
            seed=seed,
            sample_indices=sample_indices,
            complexity_class=complexity_class,
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
                for outcome_id in component_outcome_ids[: complexity_class.digit_count]
                if outcome_id not in outcome_ids
            )
            if unknown:
                raise TensorRuntimeError(f"unknown target outcome id: {unknown[0]}")
            component_to_outcome = tuple(
                outcome_ids.index(outcome_id)
                for outcome_id in component_outcome_ids[: complexity_class.digit_count]
            )
            labels = tensor_runtime_construct_tensor(
                runtime,
                recipe=TensorElementRecipe(
                    shape=(*sample_shape, len(outcome_ids))
                    if sample_shape
                    else (len(outcome_ids),),
                    dtype="float32",
                    program=_target_tensor_program(
                        seed=seed,
                        sample_indices=sample_indices,
                        cardinality=complexity_class.cardinality,
                        minimum_address=complexity_class.minimum_address,
                        digit_count=complexity_class.digit_count,
                        component_to_outcome=component_to_outcome,
                        outcome_count=len(outcome_ids),
                    ),
                ),
            )
        return fields, labels

    def _generate_tensor_fields(
        self,
        *,
        sample_shape: tuple[int, ...],
        seed: int,
        sample_indices: tuple[int, ...],
        complexity_class: _DigitsComplexityClass,
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
                sample_count=sample_count,
                width=width,
                height=height,
                digit_count=complexity_class.digit_count,
                transform=self.formation.variation_transform,
                grid=complexity_class.affine_grid,
                seed=seed,
                sample_indices=sample_indices,
                cardinality=complexity_class.cardinality,
                minimum_address=complexity_class.minimum_address,
                runtime=runtime,
                timing=timing,
                timing_prefix=timing_prefix,
            )
        if sample_shape:
            return fields.reshape((*sample_shape, *tuple(fields.shape[1:])))
        return fields.reshape(tuple(fields.shape[1:]))

    def _field_observations_from_tensor_fields(
        self,
        *,
        plans: tuple[MaterializationPlan, ...],
        fields: Any,
    ) -> tuple[FieldObservation, ...]:
        flat_fields = fields.reshape(
            (
                len(plans),
                self.formation.channel_count,
                plans[0].resolution_assignment.require_axis(self.formation.height_axis),
                plans[0].resolution_assignment.require_axis(self.formation.width_axis),
            )
        )
        host_fields = tensor_value_to_host(flat_fields)
        records: list[FieldObservation] = []
        for index, _plan in enumerate(plans):
            field_shape = tuple(int(size) for size in host_fields[index].shape)
            if len(field_shape) != 3:
                raise ObservationGenerationError("rendered field shape must have rank 3")
            records.append(
                FieldObservation(
                    shape=(field_shape[0], field_shape[1], field_shape[2]),
                    values=tuple(
                        float(value) for value in host_fields[index].reshape(-1).tolist()
                    ),
                )
            )
        return tuple(records)

    def _build_batch_tensor(
        self,
        *,
        sample_count: int,
        width: int,
        height: int,
        digit_count: int,
        transform: VariationTransformDeclaration,
        grid: _ConstructedAffineGrid,
        seed: int,
        sample_indices: tuple[int, ...],
        cardinality: int,
        minimum_address: int,
        runtime: TensorRuntime,
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> Any:
        _require_positive_integer(width, "width")
        _require_positive_integer(height, "height")
        _require_positive_integer(digit_count, "digit_count")
        if digit_count > len(self.formation.components):
            raise TensorRuntimeError("digit_count exceeds component vocabulary")
        if self.formation.channel_count != 1:
            raise TensorRuntimeError("Digits tensor renderer requires one channel")
        max_mark_count = max(
            len(self.formation.components[component_index].marks)
            for component_index in range(digit_count)
        )
        component_mark_counts: list[int] = []
        component_mark_values: list[list[float]] = []
        component_mark_widths: list[list[float]] = []
        component_control_x_points: list[list[list[float]]] = []
        component_control_y_points: list[list[list[float]]] = []
        for component_index in range(digit_count):
            component_values: list[float] = []
            component_widths: list[float] = []
            component_control_x: list[list[float]] = []
            component_control_y: list[list[float]] = []
            for mark in self.formation.components[component_index].marks:
                if mark.channel != 0:
                    raise TensorRuntimeError("Digits tensor renderer requires single-channel marks")
                controls = _quadratic_control_points(mark)
                component_values.append(float(mark.value))
                component_widths.append(float(mark.width))
                component_control_x.append([point[0] for point in controls])
                component_control_y.append([point[1] for point in controls])
            component_mark_counts.append(len(component_values))
            while len(component_values) < max_mark_count:
                component_values.append(0.0)
                component_widths.append(0.0)
                component_control_x.append([0.0, 0.0, 0.0])
                component_control_y.append([0.0, 0.0, 0.0])
            component_mark_values.append(component_values)
            component_mark_widths.append(component_widths)
            component_control_x_points.append(component_control_x)
            component_control_y_points.append(component_control_y)
        if not max_mark_count:
            return tensor_runtime_construct_tensor(
                runtime,
                recipe=TensorElementRecipe(
                    shape=(sample_count, 1, height, width),
                    dtype="float32",
                    program=_constant_tensor_program(0.0),
                ),
            )
        return tensor_runtime_construct_tensor(
            runtime,
            recipe=TensorElementRecipe(
                shape=(sample_count, 1, height, width),
                dtype="float32",
                program=_digits_tensor_program(
                    seed=seed,
                    sample_indices=sample_indices,
                    cardinality=cardinality,
                    minimum_address=minimum_address,
                    digit_count=digit_count,
                    transform=transform,
                    grid=grid,
                    component_mark_counts=tuple(component_mark_counts),
                    component_mark_values=tuple(
                        tuple(row) for row in component_mark_values
                    ),
                    component_mark_widths=tuple(
                        tuple(row) for row in component_mark_widths
                    ),
                    component_control_x_points=tuple(
                        tuple(tuple(points) for points in component)
                        for component in component_control_x_points
                    ),
                    component_control_y_points=tuple(
                        tuple(tuple(points) for points in component)
                        for component in component_control_y_points
                    ),
                    height=height,
                    width=width,
                ),
            ),
        )

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
        return self._default_complexity_class(
            canonical_variation=variation_extent_value == 0.0,
        ).complexity

    def constructed_complexity_class_complexity(
        self,
        *,
        affine_transform_count: int,
    ) -> float:
        """Return the exact log2 count of constructed single-digit choices."""

        return self._complexity_class_for_requested_cardinality(
            requested_cardinality=_complexity_class_digit_count * affine_transform_count,
            affine_transform_count=affine_transform_count,
        ).complexity

    def minimum_complexity(self) -> ComplexityValue:
        """Return the smallest score-bearing Digits complexity."""

        return _complexity_value(0.0)

    def _complexity_class_for_request(
        self,
        *,
        request: ComplexityRequest,
    ) -> _DigitsComplexityClass | None:
        """Return the integer address shell represented by a complexity request."""

        if request.maximum < self.minimum_complexity().value:
            return None
        requested_cardinality = _ceil_complexity_class_cardinality(
            max(request.minimum, self.minimum_complexity().value)
        )
        maximum_cardinality = _floor_complexity_class_cardinality(request.maximum)
        if maximum_cardinality < requested_cardinality:
            return None
        complexity_class = self._complexity_class_for_requested_cardinality(
            requested_cardinality=requested_cardinality,
            resolution_assignment=self._resolution_assignment_for_complexity_request(
                request
            ),
        )
        if not request.contains(complexity_class.measure()):
            return None
        return complexity_class

    def oracle_inference_reference_points(
        self,
        *,
        maximum_cost: float,
    ) -> tuple[dict[str, object], ...]:
        """Return deterministic Digits oracle-reference points across inference cost."""

        if not math.isfinite(maximum_cost) or maximum_cost <= 0.0:
            raise ObservationGenerationError("maximum_cost must be positive and finite")
        sides = _oracle_reference_canvas_sides(maximum_cost=maximum_cost)
        points: list[dict[str, object]] = [
            _oracle_reference_point_for_cardinality(
                cardinality=1,
                side=_complexity_class_canvas_minimum_side,
            )
        ]
        for side in sides:
            bounds = _constructed_affine_bounds_for_canvas_side(side)
            capacities = _constructed_affine_axis_capacities(
                bounds=bounds,
                side=side,
            )
            sample_cardinality = _complexity_class_digit_count * math.prod(capacities)
            points.append(
                _oracle_reference_point_for_cardinality(
                    cardinality=sample_cardinality,
                    side=side,
                    affine_axis_capacities=capacities,
                )
            )
        return tuple(points)

    def _default_complexity_class(
        self,
        *,
        canonical_variation: bool = False,
    ) -> _DigitsComplexityClass:
        return self._complexity_class_for_affine_transform_count(
            affine_transform_count=(
                1 if canonical_variation else _default_constructed_affine_transform_count
            ),
        )

    def _complexity_class_for_affine_transform_count(
        self,
        *,
        affine_transform_count: int,
        resolution_assignment: AxisAssignment | None = None,
    ) -> _DigitsComplexityClass:
        return self._complexity_class_for_requested_cardinality(
            requested_cardinality=_complexity_class_digit_count * affine_transform_count,
            affine_transform_count=affine_transform_count,
            minimum_address=0,
            resolution_assignment=resolution_assignment,
        )

    def _complexity_class_for_requested_cardinality(
        self,
        *,
        requested_cardinality: int,
        affine_transform_count: int | None = None,
        minimum_address: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
    ) -> _DigitsComplexityClass:
        _require_generation_positive_integer(
            requested_cardinality,
            "requested_cardinality",
        )
        if resolution_assignment is None:
            resolution_assignment = self._resolution_assignment_for_requested_cardinality(
                requested_cardinality
            )
        shell_minimum_address = (
            requested_cardinality - 1 if minimum_address is None else minimum_address
        )
        if type(shell_minimum_address) is not int or shell_minimum_address < 0:
            raise ObservationGenerationError("minimum_address must be a nonnegative integer")
        if affine_transform_count is None:
            affine_transform_count = _affine_transform_count_for_address_range(
                minimum_address=shell_minimum_address,
                cardinality=requested_cardinality,
            )
            affine_grid = _constructed_affine_grid_for_minimum_transform_count(
                minimum_transform_count=affine_transform_count,
                resolution_assignment=resolution_assignment,
            )
        else:
            affine_grid = _constructed_affine_grid(
                affine_transform_count,
                resolution_assignment=resolution_assignment,
            )
        return _DigitsComplexityClass(
            affine_grid=affine_grid,
            requested_cardinality=requested_cardinality,
            minimum_address=shell_minimum_address,
            resolution_assignment=resolution_assignment,
        )

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
        include_artifacts: bool = False,
        complexity_request: ComplexityRequest | None = None,
        sample_indices: Sequence[int] | None = None,
        memory_limit_bytes: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
        variation_extent: float = 1.0,
        runtime: TensorRuntime | None = None,
        outcome_ids: tuple[str, ...] | None = None,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedSampleSet:
        """Generate a shape-aware Digits sample set."""

        if include_artifacts:
            include_fields = True
        sample_shape = _sample_shape(shape)
        sample_count = _sample_count(sample_shape)
        resolved_sample_indices = _sample_indices(
            sample_count=sample_count,
            sample_indices=sample_indices,
        )
        variation_extent_value = _variation_extent_value(variation_extent)
        complexity_class = self._default_complexity_class(
            canonical_variation=variation_extent_value == 0.0,
        )
        if complexity_request is not None:
            if resolution_assignment is not None:
                raise ObservationGenerationError(
                    "complexity request cannot be combined with resolution_assignment"
            )
            requested_complexity_class = self._complexity_class_for_request(
                request=complexity_request
            )
            if requested_complexity_class is None:
                return GeneratedSampleSet(
                    benchmark_id=self.manifest.id,
                    generator_id=self.id,
                    generator_version=self.version,
                    seed=seed,
                    shape=(0,),
                    variation_extent=variation_extent,
                    complexity_request=complexity_request,
                    samples=(),
                    request_outcome=_digits_unrealized_request_outcome(
                        request=complexity_request,
                        minimum_complexity=self.minimum_complexity().value,
                    ),
                )
            resolution_assignment = requested_complexity_class.resolution_assignment
            if resolution_assignment is None:
                raise ObservationGenerationError(
                    "Digits complexity shell is missing a resolution assignment"
                )
            complexity_class = requested_complexity_class
        if runtime is not None and outcome_ids is None:
            raise ObservationGenerationError("tensor generation requires outcome_ids")
        resolved_resolution_assignment = self._generation_resolution_assignment(
            sample_count=sample_count,
            seed=seed,
            requested_assignment=resolution_assignment,
            memory_limit_bytes=memory_limit_bytes,
        )
        region = _digits_state_space_region(
            complexity_class=complexity_class,
            resolution_assignment=resolved_resolution_assignment,
            margin=self.manifest.resolution_discriminability_margin(),
        )
        fields = None
        targets = None
        if runtime is not None and outcome_ids is not None:
            fields, targets = self._generate_tensors(
                sample_shape=sample_shape,
                seed=seed,
                sample_indices=resolved_sample_indices,
                memory_limit_bytes=memory_limit_bytes,
                resolution_assignment=resolved_resolution_assignment,
                complexity_class=complexity_class,
                runtime=runtime,
                outcome_ids=outcome_ids,
                timing=timing,
                timing_prefix=timing_prefix,
            )
        samples: tuple[GeneratedSample, ...] = ()
        if include_metadata:
            if runtime is not None:
                samples = self._generate_tensor_metadata_samples(
                    sample_count=sample_count,
                    seed=seed,
                    sample_indices=resolved_sample_indices,
                    complexity_class=complexity_class,
                    resolution_assignment=resolved_resolution_assignment,
                )
            else:
                samples = self._generate_samples(
                    sample_count=sample_count,
                    seed=seed,
                    sample_indices=resolved_sample_indices,
                    include_fields=include_fields,
                    include_artifacts=include_artifacts,
                    memory_limit_bytes=memory_limit_bytes,
                    resolution_assignment=resolved_resolution_assignment,
                    complexity_class=complexity_class,
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
            complexity_request=complexity_request,
            samples=samples,
            fields=fields,
            targets=targets,
            region=region,
        )

    def _resolution_assignment_for_complexity_request(
        self,
        request: ComplexityRequest,
    ) -> AxisAssignment:
        minimum_complexity = self.minimum_complexity().value
        if request.maximum < minimum_complexity:
            side = _complexity_class_canvas_minimum_side
        else:
            minimum_cardinality = _ceil_complexity_class_cardinality(
                max(request.minimum, minimum_complexity)
            )
            maximum_cardinality = _floor_complexity_class_cardinality(request.maximum)
            side = _complexity_class_canvas_side_for_cardinality_range(
                minimum_cardinality=minimum_cardinality,
                maximum_cardinality=maximum_cardinality,
            )
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

    def _resolution_assignment_for_requested_cardinality(
        self,
        requested_cardinality: int,
    ) -> AxisAssignment:
        _require_generation_positive_integer(
            requested_cardinality,
            "requested_cardinality",
        )
        minimum_assignment = self.materialization.minimum_resolution()
        width_axis = self.formation.width_axis
        height_axis = self.formation.height_axis
        side = _complexity_class_canvas_side_for_cardinality_range(
            minimum_cardinality=requested_cardinality,
            maximum_cardinality=requested_cardinality,
        )
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


def _sample_indices(
    *,
    sample_count: int,
    sample_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if sample_indices is None:
        return tuple(range(sample_count))
    normalized = tuple(sample_indices)
    if len(normalized) != sample_count:
        raise ObservationGenerationError("sample_indices length must match sample shape")
    if any(type(index) is not int or index < 0 for index in normalized):
        raise ObservationGenerationError("sample_indices must be nonnegative integers")
    return normalized


def _complexity_value(complexity: float) -> ComplexityValue:
    return ComplexityValue(
        value=complexity,
    )


def _complexity_class_canvas_side_for_cardinality_range(
    *,
    minimum_cardinality: int,
    maximum_cardinality: int,
) -> int:
    _require_generation_positive_integer(minimum_cardinality, "minimum_cardinality")
    _require_generation_positive_integer(maximum_cardinality, "maximum_cardinality")
    if maximum_cardinality < minimum_cardinality:
        raise ObservationGenerationError(
            "complexity cardinality range maximum is below minimum"
        )
    maximum_transform_count = _affine_transform_count_for_sample_cardinality(
        maximum_cardinality
    )
    required_transform_count = maximum_transform_count
    side = _complexity_class_canvas_minimum_side
    while True:
        bounds = _constructed_affine_bounds_for_canvas_side(side)
        capacities = _constructed_affine_axis_capacities(bounds=bounds, side=side)
        if math.prod(capacities) >= required_transform_count:
            assignment = AxisAssignment(values={"H": side, "W": side})
            if _constructed_affine_grid_in_transform_count_range(
                minimum_transform_count=required_transform_count,
                maximum_transform_count=max(
                    required_transform_count,
                    required_transform_count * 2,
                ),
                resolution_assignment=assignment,
            ) is not None:
                return side
        side += _complexity_class_canvas_side_step


def _oracle_reference_canvas_sides(*, maximum_cost: float) -> tuple[int, ...]:
    if not math.isfinite(maximum_cost) or maximum_cost <= 0.0:
        raise ObservationGenerationError("maximum_cost must be positive and finite")
    sides = {_complexity_class_canvas_minimum_side}
    exponent = 0
    while 10**exponent <= maximum_cost:
        for multiplier in (1, 2, 5):
            target_cost = multiplier * 10**exponent
            side = _complexity_class_canvas_side_for_cost(target_cost)
            sides.add(side)
            if side * side >= maximum_cost:
                return tuple(sorted(sides))
        exponent += 1
    sides.add(_complexity_class_canvas_side_for_cost(maximum_cost))
    return tuple(sorted(sides))


def _oracle_reference_point_for_cardinality(
    *,
    cardinality: int,
    side: int,
    affine_axis_capacities: tuple[int, int, int, int, int] | None = None,
) -> dict[str, object]:
    _require_generation_positive_integer(cardinality, "cardinality")
    _require_generation_positive_integer(side, "side")
    complexity = math.log2(cardinality)
    cost = side * side
    components: dict[str, object] = {
        "height": side,
        "width": side,
        "pixel_count": cost,
        "sample_cardinality": cardinality,
    }
    if affine_axis_capacities is not None:
        components["affine_axis_capacities"] = {
            "x_translation": affine_axis_capacities[0],
            "y_translation": affine_axis_capacities[1],
            "scale": affine_axis_capacities[2],
            "rotation": affine_axis_capacities[3],
            "x_shear": affine_axis_capacities[4],
        }
    return {
        "complexity": complexity,
        "score": complexity,
        "cost": cost,
        "metadata": {
            "kind": "oracle-inference-compute-reference-v1",
            "unit": "abstract-ops",
            "value": cost,
            "components": components,
        },
    }


def _complexity_class_canvas_side_for_cost(cost: float) -> int:
    if not math.isfinite(cost) or cost <= 0.0:
        raise ObservationGenerationError("cost must be positive and finite")
    side = max(_complexity_class_canvas_minimum_side, math.ceil(math.sqrt(cost)))
    offset = max(0, side - _complexity_class_canvas_minimum_side)
    steps = math.ceil(offset / _complexity_class_canvas_side_step)
    return _complexity_class_canvas_minimum_side + steps * _complexity_class_canvas_side_step


def _ceil_complexity_class_cardinality(complexity: float) -> int:
    value = _complexity_class_cardinality_float(complexity)
    tolerance = max(1.0, abs(value)) * _complexity_class_cardinality_relative_tolerance
    return max(1, math.ceil(value - tolerance))


def _floor_complexity_class_cardinality(complexity: float) -> int:
    value = _complexity_class_cardinality_float(complexity)
    tolerance = max(1.0, abs(value)) * _complexity_class_cardinality_relative_tolerance
    return max(1, math.floor(value + tolerance))


def _complexity_class_cardinality_float(complexity: float) -> float:
    if not math.isfinite(float(complexity)):
        raise ObservationGenerationError("complexity must be finite")
    return 2.0**complexity


def _affine_transform_count_for_sample_cardinality(cardinality: int) -> int:
    _require_generation_positive_integer(cardinality, "cardinality")
    digit_count = min(_complexity_class_digit_count, cardinality)
    return max(1, math.ceil(cardinality / digit_count))


def _affine_transform_count_for_address_range(
    *,
    minimum_address: int,
    cardinality: int,
) -> int:
    if type(minimum_address) is not int or minimum_address < 0:
        raise ObservationGenerationError("minimum_address must be a nonnegative integer")
    _require_generation_positive_integer(cardinality, "cardinality")
    maximum_address = minimum_address + cardinality - 1
    return maximum_address // _complexity_class_digit_count + 1


def _constructed_affine_grid_for_minimum_transform_count(
    *,
    minimum_transform_count: int,
    resolution_assignment: AxisAssignment | None,
) -> _ConstructedAffineGrid:
    _require_generation_positive_integer(
        minimum_transform_count,
        "minimum_transform_count",
    )
    assignment = (
        resolution_assignment
        if resolution_assignment is not None
        else AxisAssignment(
            values={
                "H": _complexity_class_canvas_minimum_side,
                "W": _complexity_class_canvas_minimum_side,
            }
        )
    )
    maximum_transform_count = max(minimum_transform_count, minimum_transform_count * 2)
    grid = _constructed_affine_grid_in_transform_count_range(
        minimum_transform_count=minimum_transform_count,
        maximum_transform_count=maximum_transform_count,
        resolution_assignment=assignment,
    )
    if grid is None:
        raise ObservationGenerationError(
            "requested sample cardinality exceeds resolution-aware affine grid capacity"
        )
    return grid


def _digits_state_coordinate(
    *,
    state_index: int,
    complexity_class: _DigitsComplexityClass,
) -> tuple[int, int]:
    if type(state_index) is not int or state_index < 0:
        raise ObservationGenerationError("state_index must be a nonnegative integer")
    if state_index >= complexity_class.cardinality:
        raise ObservationGenerationError("state_index must be below cardinality")
    sample_address = complexity_class.minimum_address + state_index
    component_index = sample_address % _complexity_class_digit_count
    transform_index = sample_address // _complexity_class_digit_count
    if transform_index >= complexity_class.affine_transform_count:
        raise ObservationGenerationError("sample address exceeds active transform set")
    return (component_index, transform_index)


def _digits_state_space_region(
    *,
    complexity_class: _DigitsComplexityClass,
    resolution_assignment: AxisAssignment,
    margin: float,
) -> StateSpaceRegion:
    width = resolution_assignment.require_axis("W")
    height = resolution_assignment.require_axis("H")
    ambient = StateSpaceAmbient(
        field_domain_kind="lattice-2d",
        field_domain={"height": height, "width": width},
        field_codomain_id="unit-intensity",
        distinguishability=Distinguishability(
            kind="metric-resolution",
            metric_id="l1-field-distance",
            resolution=margin,
            certificate_id="component-discriminability-margin",
        ),
    )
    transform_indices_by_digit = _digits_transform_indices_by_digit(complexity_class)
    components = tuple(
        _digits_product_region(
            digit_index=digit_index,
            transform_indices=transform_indices,
            grid=complexity_class.affine_grid,
            width=width,
            height=height,
        )
        for digit_index, transform_indices in transform_indices_by_digit.items()
    )
    return StateSpaceRegion(
        id=(
            "benchmarks.digits.realized-region."
            f"addresses-{complexity_class.minimum_address}-{complexity_class.maximum_address}."
            f"canvas-{width}x{height}"
        ),
        ambient=ambient,
        components=components,
        union_rule="disjoint-union",
        volume=complexity_class.cardinality,
        log2_volume=complexity_class.complexity,
    )


def _digits_product_region(
    *,
    digit_index: int,
    transform_indices: tuple[int, ...],
    grid: _ConstructedAffineGrid,
    width: int,
    height: int,
) -> ProductRegion:
    if not transform_indices:
        raise ObservationGenerationError("digits region stratum must not be empty")
    axis_regions = (
        *_digits_transform_axis_regions(
            transform_indices=transform_indices,
            grid=grid,
        ),
        _singleton_integer_axis_region("canvas_width", width),
        _singleton_integer_axis_region("canvas_height", height),
    )
    measure_rule = (
        "product-of-counts"
        if math.prod(axis_region.count for axis_region in axis_regions)
        == len(transform_indices)
        else "benchmark-computed-finite-count"
    )
    return ProductRegion(
        axis_regions=axis_regions,
        measure_rule=measure_rule,
        volume=len(transform_indices),
        log2_volume=math.log2(len(transform_indices)),
        stratum_id=f"digit-{digit_index}",
        stratum_target={
            "digit_index": digit_index,
            "outcome_id": f"digit-{digit_index}",
        },
    )


def _digits_transform_axis_regions(
    *,
    transform_indices: tuple[int, ...],
    grid: _ConstructedAffineGrid,
) -> tuple[AxisRegion, ...]:
    if grid.preset_count is not None:
        cells = tuple(f"preset-{index}" for index in range(grid.preset_count))
        selected = tuple(f"preset-{index}" for index in transform_indices)
        return (
            AxisRegion(
                axis=StateSpaceAxis(
                    id="affine_preset",
                    domain=EnumeratedCellsDomain(cells=cells),
                ),
                coordinate_region=selected,
                count=len(selected),
                log2_count=math.log2(len(selected)),
            ),
        )
    indices_by_axis = {
        axis_name: tuple(
            _constructed_affine_indices(transform_index=transform_index, grid=grid)[axis_name]
            for transform_index in transform_indices
        )
        for axis_name in _constructed_affine_axis_names
    }
    bounds_by_axis = {
        "x_translation": grid.x_translation_bounds,
        "y_translation": grid.y_translation_bounds,
        "scale": grid.scale_bounds,
        "rotation": grid.rotation_bounds,
        "x_shear": grid.x_shear_bounds,
    }
    counts_by_axis = dict(zip(_constructed_affine_axis_names, grid.counts, strict=True))
    return tuple(
        _real_grid_axis_region(
            axis_id=axis_name,
            bounds=bounds_by_axis[axis_name],
            domain_count=counts_by_axis[axis_name],
            selected_indices=indices_by_axis[axis_name],
        )
        for axis_name in _constructed_affine_axis_names
    )


def _real_grid_axis_region(
    *,
    axis_id: str,
    bounds: tuple[float, float],
    domain_count: int,
    selected_indices: tuple[int, ...],
) -> AxisRegion:
    lower = min(selected_indices)
    upper = max(selected_indices)
    count = upper - lower + 1
    return AxisRegion(
        axis=StateSpaceAxis(
            id=axis_id,
            domain=RealGridDomain(
                lower=bounds[0],
                upper=bounds[1],
                count=domain_count,
            ),
        ),
        coordinate_region=(lower, upper),
        count=count,
        log2_count=math.log2(count),
    )


def _singleton_integer_axis_region(axis_id: str, value: int) -> AxisRegion:
    return AxisRegion(
        axis=StateSpaceAxis(
            id=axis_id,
            domain=IntegerRangeDomain(lower=value, upper=value),
        ),
        coordinate_region=(value, value),
        count=1,
        log2_count=0.0,
    )


def _digits_region_component_index(
    *,
    component_index: int,
    complexity_class: _DigitsComplexityClass,
) -> int:
    digits = tuple(_digits_transform_indices_by_digit(complexity_class))
    try:
        return digits.index(component_index)
    except ValueError as error:
        raise ObservationGenerationError(
            "digit component is outside the realized region"
        ) from error


def _digits_region_axis_coordinates(
    *,
    transform_index: int,
    grid: _ConstructedAffineGrid,
    resolution_assignment: AxisAssignment,
) -> Mapping[str, object]:
    coordinates: dict[str, object]
    if grid.preset_count is not None:
        coordinates = {"affine_preset": f"preset-{transform_index}"}
    else:
        coordinates = dict(_constructed_affine_indices(transform_index=transform_index, grid=grid))
    coordinates["canvas_width"] = resolution_assignment.require_axis("W")
    coordinates["canvas_height"] = resolution_assignment.require_axis("H")
    return coordinates


def _digits_transform_indices_by_digit(
    complexity_class: _DigitsComplexityClass,
) -> dict[int, tuple[int, ...]]:
    transform_indices_by_digit: dict[int, list[int]] = {}
    for state_index in range(complexity_class.cardinality):
        component_index, transform_index = _digits_state_coordinate(
            state_index=state_index,
            complexity_class=complexity_class,
        )
        transform_indices_by_digit.setdefault(component_index, []).append(transform_index)
    return {
        digit_index: tuple(transform_indices)
        for digit_index, transform_indices in sorted(transform_indices_by_digit.items())
    }


def _digits_unrealized_request_outcome(
    *,
    request: ComplexityRequest,
    minimum_complexity: float,
) -> GenerationRequestOutcome:
    _ = minimum_complexity
    return GenerationRequestOutcome(kind="unrepresentable-below-minimum")


def _digits_local_state_index(
    *,
    seed: int,
    sample_index: int,
    cardinality: int,
) -> int:
    if type(sample_index) is not int or sample_index < 0:
        raise ObservationGenerationError("sample_index must be a nonnegative integer")
    if type(cardinality) is not int or cardinality < 1:
        raise ObservationGenerationError("cardinality must be positive")
    return (seed + sample_index) % cardinality


def _complexity_class_canvas_side_from_assignment(
    resolution_assignment: AxisAssignment | None,
) -> int:
    if resolution_assignment is None:
        return _complexity_class_canvas_minimum_side
    values = tuple(resolution_assignment.values.values())
    if not values:
        return _complexity_class_canvas_minimum_side
    return max(_complexity_class_canvas_minimum_side, max(values))


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


def _constructed_affine_grid(
    transform_count: int,
    *,
    resolution_assignment: AxisAssignment | None = None,
) -> _ConstructedAffineGrid:
    _require_generation_positive_integer(transform_count, "transform_count")
    side = _complexity_class_canvas_side_from_assignment(resolution_assignment)
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


def _constructed_affine_grid_in_transform_count_range(
    *,
    minimum_transform_count: int,
    maximum_transform_count: int,
    resolution_assignment: AxisAssignment,
) -> _ConstructedAffineGrid | None:
    _require_generation_positive_integer(
        minimum_transform_count,
        "minimum_transform_count",
    )
    _require_generation_positive_integer(
        maximum_transform_count,
        "maximum_transform_count",
    )
    if maximum_transform_count < minimum_transform_count:
        return None
    side = _complexity_class_canvas_side_from_assignment(resolution_assignment)
    bounds = _constructed_affine_bounds_for_canvas_side(side)
    capacities = _constructed_affine_axis_capacities(bounds=bounds, side=side)
    preset_count = _preset_count_in_range(
        minimum_transform_count=minimum_transform_count,
        maximum_transform_count=maximum_transform_count,
    )
    if preset_count is not None:
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
            preset_count=preset_count,
        )
    counts = _constructed_affine_counts_in_transform_count_range(
        minimum_transform_count=max(
            minimum_transform_count,
            _constructed_affine_preset_max_count + 1,
        ),
        maximum_transform_count=maximum_transform_count,
        capacities=capacities,
    )
    if counts is None:
        return None
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


def _preset_count_in_range(
    *,
    minimum_transform_count: int,
    maximum_transform_count: int,
) -> int | None:
    if minimum_transform_count > _constructed_affine_preset_max_count:
        return None
    if maximum_transform_count < minimum_transform_count:
        return None
    return minimum_transform_count


def _constructed_affine_counts_in_transform_count_range(
    *,
    minimum_transform_count: int,
    maximum_transform_count: int,
    capacities: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int] | None:
    if maximum_transform_count < minimum_transform_count:
        return None
    best: tuple[int, int, int, int, int] | None = None
    best_key: tuple[int, tuple[float, ...], tuple[int, ...]] | None = None
    x_capacity, y_capacity, scale_capacity, rotation_capacity, shear_capacity = capacities
    for scale_count in range(1, scale_capacity + 1):
        for rotation_count in range(1, rotation_capacity + 1):
            for shear_count in range(1, shear_capacity + 1):
                base_count = scale_count * rotation_count * shear_count
                minimum_xy_count = math.ceil(minimum_transform_count / base_count)
                maximum_xy_count = maximum_transform_count // base_count
                xy_counts = _constructed_translation_counts_in_product_range(
                    minimum_product=minimum_xy_count,
                    maximum_product=maximum_xy_count,
                    x_capacity=x_capacity,
                    y_capacity=y_capacity,
                )
                if xy_counts is None:
                    continue
                counts = (
                    xy_counts[0],
                    xy_counts[1],
                    scale_count,
                    rotation_count,
                    shear_count,
                )
                product = math.prod(counts)
                key = (
                    product,
                    _constructed_affine_count_sort_key(counts),
                    counts,
                )
                if best_key is None or key < best_key:
                    best = counts
                    best_key = key
    return best


def _constructed_translation_counts_in_product_range(
    *,
    minimum_product: int,
    maximum_product: int,
    x_capacity: int,
    y_capacity: int,
) -> tuple[int, int] | None:
    minimum_product = max(1, minimum_product)
    if maximum_product < minimum_product:
        return None
    if x_capacity * y_capacity < minimum_product:
        return None
    best: tuple[int, int] | None = None
    best_key: tuple[int, float, tuple[int, int]] | None = None
    for x_count in range(1, min(x_capacity, maximum_product) + 1):
        y_count = math.ceil(minimum_product / x_count)
        if y_count < 1 or y_count > y_capacity:
            continue
        product = x_count * y_count
        if product > maximum_product:
            continue
        balance_penalty = abs(math.log2(x_count / y_count))
        key = (product, balance_penalty, (x_count, y_count))
        if best_key is None or key < best_key:
            best = (x_count, y_count)
            best_key = key
    return best


def _constructed_affine_counts_for_transform_count(
    *,
    transform_count: int,
    capacities: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    counts = [1, 1, 1, 1, 1]
    for factor in sorted(_prime_factors(transform_count), reverse=True):
        factor_options = tuple(
            index
            for index, (count, capacity) in enumerate(zip(counts, capacities, strict=True))
            if count * factor <= capacity
        )
        if not factor_options:
            break
        selected = min(
            factor_options,
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


def _quadratic_tensor_point(
    controls: Any,
    component_index: Any,
    mark_slot: int,
    t: Any,
) -> Any:
    t_by_segment = t.reshape((-1, 1))
    one_minus_t = 1.0 - t_by_segment
    start = controls[component_index, mark_slot, 0].reshape((1, -1))
    control = controls[component_index, mark_slot, 1].reshape((1, -1))
    end = controls[component_index, mark_slot, 2].reshape((1, -1))
    return (
        one_minus_t * one_minus_t * start
        + 2.0 * one_minus_t * t_by_segment * control
        + t_by_segment * t_by_segment * end
    )


def _constant_tensor_program(value: float) -> TensorElementProgram:
    def element_function(coordinates: tuple[Any, ...], flat_indices: Any) -> Any:
        _ = flat_indices
        return coordinates[0] * 0.0 + value

    return TensorElementProgram(
        kernel=element_function,
        parameters={},
        cache_key=("constant", value),
    )


def _target_tensor_program(
    *,
    seed: int,
    sample_indices: tuple[int, ...],
    cardinality: int,
    minimum_address: int,
    digit_count: int,
    component_to_outcome: tuple[int, ...],
    outcome_count: int,
) -> TensorElementProgram:
    def element_function(
        coordinates: tuple[Any, ...],
        flat_indices: Any,
        *,
        seed_value: Any,
        sample_indices_value: Any,
        cardinality_value: Any,
        minimum_address_value: Any,
        component_to_outcome: Any,
    ) -> Any:
        _ = coordinates
        sample_axis_index = flat_indices.div(outcome_count, rounding_mode="floor")
        sample_index = sample_indices_value[sample_axis_index]
        outcome_index = flat_indices.remainder(outcome_count)
        state_index = (sample_index + seed_value).remainder(cardinality_value)
        sample_address = minimum_address_value + state_index
        component_index = sample_address.remainder(digit_count)
        target_index = component_to_outcome[component_index]
        return (target_index == outcome_index).to(dtype=target_index.dtype)

    return TensorElementProgram(
        kernel=element_function,
        parameters={
            "seed_value": TensorElementParameter(
                dtype="int64",
                shape=(),
                values=(seed,),
            ),
            "sample_indices_value": TensorElementParameter(
                dtype="int64",
                shape=(len(sample_indices),),
                values=sample_indices,
                dynamic_axes=(0,),
            ),
            "cardinality_value": TensorElementParameter(
                dtype="int64",
                shape=(),
                values=(cardinality,),
            ),
            "minimum_address_value": TensorElementParameter(
                dtype="int64",
                shape=(),
                values=(minimum_address,),
            ),
            "component_to_outcome": TensorElementParameter(
                dtype="int64",
                shape=(len(component_to_outcome),),
                values=component_to_outcome,
            ),
        },
        cache_key=("target", outcome_count),
    )


def _digits_tensor_program(
    *,
    seed: int,
    sample_indices: tuple[int, ...],
    cardinality: int,
    minimum_address: int,
    digit_count: int,
    transform: VariationTransformDeclaration,
    grid: _ConstructedAffineGrid,
    component_mark_counts: tuple[int, ...],
    component_mark_values: tuple[tuple[float, ...], ...],
    component_mark_widths: tuple[tuple[float, ...], ...],
    component_control_x_points: tuple[tuple[tuple[float, ...], ...], ...],
    component_control_y_points: tuple[tuple[tuple[float, ...], ...], ...],
    height: int,
    width: int,
) -> TensorElementProgram:
    component_count = len(component_mark_counts)
    max_mark_count = len(component_mark_values[0]) if component_count else 0
    control_point_count = 3
    spatial = transform.spatial_affine
    x_translation_bounds = _bounded_interval(
        grid.x_translation_bounds,
        lower_bound=spatial.matrix[0][2][0],
        upper_bound=spatial.matrix[0][2][1],
    )
    y_translation_bounds = _bounded_interval(
        grid.y_translation_bounds,
        lower_bound=spatial.matrix[1][2][0],
        upper_bound=spatial.matrix[1][2][1],
    )
    scale_bounds = _bounded_interval(
        grid.scale_bounds,
        lower_bound=max(spatial.matrix[0][0][0], spatial.matrix[1][1][0]),
        upper_bound=min(spatial.matrix[0][0][1], spatial.matrix[1][1][1]),
    )
    rotation_bounds = _bounded_interval(
        grid.rotation_bounds,
        lower_bound=spatial.matrix[1][0][0],
        upper_bound=spatial.matrix[1][0][1],
    )
    x_shear_bounds = _bounded_interval(
        grid.x_shear_bounds,
        lower_bound=spatial.matrix[0][1][0],
        upper_bound=spatial.matrix[0][1][1],
    )
    preset_units = (
        tuple(
            value
            for preset_index in range(grid.preset_count)
            for value in _constructed_affine_preset_unit_coordinates(
                preset_index=preset_index,
                preset_count=grid.preset_count,
            )
        )
        if grid.preset_count is not None
        else ()
    )

    def tensor_grid_value(lower: Any, upper: Any, index: Any, count: int) -> Any:
        if count <= 1:
            return (lower + upper) / 2.0
        return lower + (upper - lower) * (index / float(count - 1))

    def element_function(
        coordinates: tuple[Any, ...],
        flat_indices: Any,
        *,
        seed_value: Any,
        sample_indices_value: Any,
        cardinality_value: Any,
        minimum_address_value: Any,
        preset_units_tensor: Any,
        component_mark_counts: Any,
        component_mark_values: Any,
        component_mark_widths: Any,
        component_control_x_points: Any,
        component_control_y_points: Any,
        segment_start_t: Any,
        segment_end_t: Any,
        image_height: Any,
        image_width: Any,
        x_translation_minimum: Any,
        x_translation_maximum: Any,
        y_translation_minimum: Any,
        y_translation_maximum: Any,
        scale_minimum: Any,
        scale_maximum: Any,
        rotation_minimum: Any,
        rotation_maximum: Any,
        x_shear_minimum: Any,
        x_shear_maximum: Any,
    ) -> Any:
        _ = flat_indices
        sample_axis_index, channel_index, y_index, x_index = coordinates
        sample_index = sample_indices_value[sample_axis_index]
        x_center = x_index + 0.5
        y_center = y_index + 0.5
        state_index = (sample_index + seed_value).remainder(cardinality_value)
        sample_address = minimum_address_value + state_index
        component_index = sample_address.remainder(digit_count)
        transform_index = sample_address.div(digit_count, rounding_mode="floor")
        if grid.transform_count == 1:
            m00 = x_center * 0.0 + 1.0
            m01 = x_center * 0.0
            m02 = x_center * 0.0
            m10 = x_center * 0.0
            m11 = x_center * 0.0 + 1.0
            m12 = x_center * 0.0
            width_scale = x_center * 0.0 + 1.0
        else:
            def affine_component(
                axis_slot: int,
                count: int,
                lower: Any,
                upper: Any,
            ) -> Any:
                if grid.preset_count is not None:
                    return lower + (upper - lower) * preset_units_tensor[
                        transform_index,
                        axis_slot,
                    ]
                remainder = transform_index
                for prior_count in grid.counts[:axis_slot]:
                    remainder = remainder.div(prior_count, rounding_mode="floor")
                return tensor_grid_value(lower, upper, remainder.remainder(count), count)

            x_translation = affine_component(
                0,
                grid.x_translation,
                x_translation_minimum,
                x_translation_maximum,
            )
            y_translation = affine_component(
                1,
                grid.y_translation,
                y_translation_minimum,
                y_translation_maximum,
            )
            scale = affine_component(
                2,
                grid.scale,
                scale_minimum,
                scale_maximum,
            )
            rotation = affine_component(
                3,
                grid.rotation,
                rotation_minimum,
                rotation_maximum,
            )
            x_shear = affine_component(
                4,
                grid.x_shear,
                x_shear_minimum,
                x_shear_maximum,
            )
            cosine = rotation.cos()
            sine = rotation.sin()
            m00 = scale * cosine
            m01 = x_shear - scale * sine
            m02 = x_translation
            m10 = scale * sine
            m11 = scale * cosine
            m12 = y_translation
            width_scale = ((m00 * m00 + m10 * m10).sqrt()).maximum(
                (m01 * m01 + m11 * m11).sqrt()
            )
        value = x_center * 0.0
        active_channel = channel_index == 0
        mark_count = component_mark_counts[component_index]
        for mark_slot in range(max_mark_count):
            active_mark = active_channel & (mark_slot < mark_count)
            raw_sx = _quadratic_tensor_point(
                component_control_x_points,
                component_index,
                mark_slot,
                segment_start_t,
            )
            raw_sy = _quadratic_tensor_point(
                component_control_y_points,
                component_index,
                mark_slot,
                segment_start_t,
            )
            raw_ex = _quadratic_tensor_point(
                component_control_x_points,
                component_index,
                mark_slot,
                segment_end_t,
            )
            raw_ey = _quadratic_tensor_point(
                component_control_y_points,
                component_index,
                mark_slot,
                segment_end_t,
            )
            sx = (0.5 + m00 * (raw_sx - 0.5) + m01 * (raw_sy - 0.5) + m02) * image_width
            sy = (0.5 + m10 * (raw_sx - 0.5) + m11 * (raw_sy - 0.5) + m12) * image_height
            ex = (0.5 + m00 * (raw_ex - 0.5) + m01 * (raw_ey - 0.5) + m02) * image_width
            ey = (0.5 + m10 * (raw_ex - 0.5) + m11 * (raw_ey - 0.5) + m12) * image_height
            dx = ex - sx
            dy = ey - sy
            length_squared = dx * dx + dy * dy
            safe_length_squared = length_squared.where(
                length_squared != 0.0,
                length_squared * 0.0 + 1.0,
            )
            segment_t = ((x_center - sx) * dx + (y_center - sy) * dy) / safe_length_squared
            segment_t = segment_t.clamp(0.0, 1.0)
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
            segment_distance_squared = segment_distance_squared.where(
                length_squared != 0.0,
                point_distance_squared,
            )
            distance_squared = segment_distance_squared.min(dim=0).values
            mark_width = component_mark_widths[component_index, mark_slot]
            threshold = (width_scale * mark_width / 2.0) * (width_scale * mark_width / 2.0)
            contribution = (
                active_mark & (distance_squared <= threshold)
            ) * component_mark_values[component_index, mark_slot]
            value = value.maximum(contribution)
        return value

    return TensorElementProgram(
        kernel=element_function,
        parameters={
            "seed_value": TensorElementParameter(
                dtype="int64",
                shape=(),
                values=(seed,),
            ),
            "sample_indices_value": TensorElementParameter(
                dtype="int64",
                shape=(len(sample_indices),),
                values=sample_indices,
                dynamic_axes=(0,),
            ),
            "cardinality_value": TensorElementParameter(
                dtype="int64",
                shape=(),
                values=(cardinality,),
            ),
            "minimum_address_value": TensorElementParameter(
                dtype="int64",
                shape=(),
                values=(minimum_address,),
            ),
            "preset_units_tensor": TensorElementParameter(
                dtype="float32",
                shape=(grid.preset_count or 1, 5),
                values=preset_units if preset_units else (0.0, 0.0, 0.0, 0.0, 0.0),
            ),
            "component_mark_counts": TensorElementParameter(
                dtype="int64",
                shape=(component_count,),
                values=component_mark_counts,
            ),
            "component_mark_values": TensorElementParameter(
                dtype="float32",
                shape=(component_count, max_mark_count),
                values=tuple(value for row in component_mark_values for value in row),
            ),
            "component_mark_widths": TensorElementParameter(
                dtype="float32",
                shape=(component_count, max_mark_count),
                values=tuple(value for row in component_mark_widths for value in row),
            ),
            "component_control_x_points": TensorElementParameter(
                dtype="float32",
                shape=(component_count, max_mark_count, control_point_count),
                values=tuple(
                    value
                    for component in component_control_x_points
                    for mark in component
                    for value in mark
                ),
            ),
            "component_control_y_points": TensorElementParameter(
                dtype="float32",
                shape=(component_count, max_mark_count, control_point_count),
                values=tuple(
                    value
                    for component in component_control_y_points
                    for mark in component
                    for value in mark
                ),
            ),
            "segment_start_t": TensorElementParameter(
                dtype="float32",
                shape=(_batch_render_curve_sample_count - 1,),
                values=tuple(
                    index / float(_batch_render_curve_sample_count - 1)
                    for index in range(_batch_render_curve_sample_count - 1)
                ),
            ),
            "segment_end_t": TensorElementParameter(
                dtype="float32",
                shape=(_batch_render_curve_sample_count - 1,),
                values=tuple(
                    (index + 1) / float(_batch_render_curve_sample_count - 1)
                    for index in range(_batch_render_curve_sample_count - 1)
                ),
            ),
            "image_height": TensorElementParameter(
                dtype="float32",
                shape=(),
                values=(float(height),),
            ),
            "image_width": TensorElementParameter(
                dtype="float32",
                shape=(),
                values=(float(width),),
            ),
            "x_translation_minimum": TensorElementParameter(
                dtype="float32",
                shape=(),
                values=(x_translation_bounds[0],),
            ),
            "x_translation_maximum": TensorElementParameter(
                dtype="float32",
                shape=(),
                values=(x_translation_bounds[1],),
            ),
            "y_translation_minimum": TensorElementParameter(
                dtype="float32",
                shape=(),
                values=(y_translation_bounds[0],),
            ),
            "y_translation_maximum": TensorElementParameter(
                dtype="float32",
                shape=(),
                values=(y_translation_bounds[1],),
            ),
            "scale_minimum": TensorElementParameter(
                dtype="float32",
                shape=(),
                values=(scale_bounds[0],),
            ),
            "scale_maximum": TensorElementParameter(
                dtype="float32",
                shape=(),
                values=(scale_bounds[1],),
            ),
            "rotation_minimum": TensorElementParameter(
                dtype="float32",
                shape=(),
                values=(rotation_bounds[0],),
            ),
            "rotation_maximum": TensorElementParameter(
                dtype="float32",
                shape=(),
                values=(rotation_bounds[1],),
            ),
            "x_shear_minimum": TensorElementParameter(
                dtype="float32",
                shape=(),
                values=(x_shear_bounds[0],),
            ),
            "x_shear_maximum": TensorElementParameter(
                dtype="float32",
                shape=(),
                values=(x_shear_bounds[1],),
            ),
        },
        cache_key=(
            "digits-field",
            component_count,
            max_mark_count,
            _batch_render_curve_sample_count,
            grid.counts,
            grid.preset_count,
        ),
    )


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
            "complexity_value": {
                "kind": "constructed-finite-complexity-shell",
                "measure_id": "log2-state-space-volume",
                "formula": "log2(realized_cardinality)",
                "digit_count": _complexity_class_digit_count,
                "affine_transform_family": "constructed-finite-affine-product-grid",
                "target_policy": "symmetric-realized-cardinalities-inside-request-band",
                "description": (
                    "Score-bearing Digits complexity shells are requested finite "
                    "single-digit windows. The minimum non-null request is the "
                    "canonical 10-way digit classification problem. Larger "
                    "requests add symmetric finite affine choices for every "
                    "digit, and the benchmark reports the realized cardinality "
                    "instead of forcing exact powers of two. Canvas resolution "
                    "is the smallest square lattice, rounded to the benchmark "
                    "resolution step, whose finite affine grid can express a "
                    "cardinality inside the requested complexity band."
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
