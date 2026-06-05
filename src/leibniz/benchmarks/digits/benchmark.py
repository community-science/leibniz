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
from leibniz.tensor_runtime import TensorRuntime, TensorRuntimeError
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
_default_constructed_affine_transform_count = 2
_constructed_affine_translation_bounds = (-0.15, 0.15)
_constructed_affine_effective_radius_fraction = 0.25
_constructed_affine_axis_density = 0.25
_constructed_affine_scale_bounds = (0.92, 1.08)
_constructed_affine_rotation_bounds = (-0.03, 0.03)
_constructed_affine_shear_bounds = (-0.03, 0.03)
_constructed_affine_axis_names = (
    "x_translation",
    "y_translation",
    "scale",
    "rotation",
    "x_shear",
)
_console_preview_state_space_windows = (
    (1.0, 2.0),
    (2.0, 4.0),
    (4.0, 8.0),
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
        count = 1
        for axis_count in self.counts:
            count *= axis_count
        return count

    def to_record(self) -> dict[str, int]:
        return dict(zip(_constructed_affine_axis_names, self.counts, strict=True))

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
    digit_variant_counts: tuple[int, ...]
    affine_grid: _ConstructedAffineGrid
    requested_state_count: int
    resolution_assignment: AxisAssignment | None = None

    @property
    def digit_count(self) -> int:
        return len(self.digit_variant_counts)

    @property
    def digit_state_count(self) -> int:
        return sum(self.digit_variant_counts)

    @property
    def affine_transform_count(self) -> int:
        return self.affine_grid.transform_count

    @property
    def cardinality(self) -> int:
        return self.digit_state_count * self.affine_transform_count

    @property
    def complexity(self) -> float:
        return math.log2(self.cardinality)

    def measure(self) -> StateSpaceMeasureValue:
        return _state_space_measure(self.complexity)

    def metadata(self) -> dict[str, object]:
        return {
            "kind": "digits-requested-finite-state-space",
            "digit_count": self.digit_count,
            "digit_variant_counts": list(self.digit_variant_counts),
            "digit_state_count": self.digit_state_count,
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
        component_indices: Iterable[int] | None = None,
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
        state_space_digest = str(ContentDigest.from_value(state_space.metadata()))
        variation_generator = random.Random(f"{seed}:variation:{state_space_digest}")

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
        component_indices: Iterable[int] | None,
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
        generator: random.Random,
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
        with _timing_span(timing, timing_phase, samples=len(plans)):
            for _plan in plans:
                transform_index = generator.randrange(state_space.affine_transform_count)
                coordinate = _constructed_variation_coordinate_record(
                    transform=transform,
                    component_index=0,
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

        return _state_space_measure(1.0)

    def state_space_for_request(
        self,
        *,
        request: StateSpaceMeasureRequest,
    ) -> StateSpaceCandidate | None:
        """Return the smallest symmetric finite state space inside a request band."""

        candidates = self.state_spaces_for_request(request=request)
        if not candidates:
            return None
        return candidates[0]

    def state_spaces_for_request(
        self,
        *,
        request: StateSpaceMeasureRequest,
    ) -> tuple[StateSpaceCandidate, ...]:
        """Return symmetric Digits state-space candidates inside a request band."""

        minimum_complexity = self.minimum_state_space_measure().value
        if request.maximum < minimum_complexity:
            return ()
        minimum_cardinality = max(
            2,
            math.ceil(2.0 ** max(request.minimum, minimum_complexity)),
        )
        maximum_cardinality = max(2, math.floor(2.0**request.maximum))
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
        digit_variant_counts = tuple(1 for _component in self.formation.components)
        return self._state_space_for_requested_state_count(
            requested_state_count=sum(digit_variant_counts) * affine_transform_count,
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
        active_digit_count = min(_state_space_digit_count, requested_state_count)
        digit_variant_counts = tuple(1 for _index in range(active_digit_count))
        digit_state_count = sum(digit_variant_counts)
        if affine_transform_count is None:
            affine_transform_count = max(
                1,
                math.ceil(requested_state_count / digit_state_count),
            )
        requested_state_count = digit_state_count * affine_transform_count
        return _DigitsStateSpace(
            digit_variant_counts=digit_variant_counts,
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
        for digit_count in range(2, _state_space_digit_count):
            if minimum_cardinality <= digit_count <= maximum_cardinality:
                state_spaces.append(
                    _DigitsStateSpace(
                        digit_variant_counts=tuple(1 for _index in range(digit_count)),
                        affine_grid=_constructed_affine_grid(
                            1,
                            resolution_assignment=resolution_assignment,
                        ),
                        requested_state_count=digit_count,
                        resolution_assignment=resolution_assignment,
                    )
                )
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
                    digit_variant_counts=tuple(
                        1 for _index in range(_state_space_digit_count)
                    ),
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
        samples = self._generate_samples(
            sample_count=sample_count,
            seed=seed,
            include_fields=include_fields,
            component_indices=component_indices,
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

        for candidate_index, candidate in enumerate(candidates):
            state_space = self._state_space_for_candidate(candidate)
            if state_space.resolution_assignment is None:
                raise ObservationGenerationError(
                    "Digits state-space candidate is missing a resolution assignment"
                )
            for component_index in range(state_space.digit_count):
                for transform_index in range(state_space.affine_transform_count):
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
                    field_record = self.formation.form_observation(
                        id=self._observation_id(seed=seed, index=sample_index),
                        plan=plan,
                        component_index=component_index,
                        variation_coordinates=(variation_coordinate,),
                    )
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

    def sample_variation_transform_coordinates(
        self,
        *,
        seed: int,
        sample_index: int,
        component_index: int = 0,
        affine_transform_count: int = _default_constructed_affine_transform_count,
    ) -> Mapping[str, object]:
        """Sample one deterministic Digits variation coordinate."""

        if type(seed) is not int or seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        if type(sample_index) is not int or sample_index < 0:
            raise ObservationGenerationError("sample_index must be a nonnegative integer")
        if type(component_index) is not int or component_index < 0:
            raise ObservationGenerationError("component_index must be a nonnegative integer")
        _require_generation_positive_integer(
            affine_transform_count,
            "affine_transform_count",
        )
        transform = self.formation.variation_transform
        transform_index = (
            seed + sample_index * 9973 + component_index * 101
        ) % affine_transform_count
        return _constructed_variation_coordinate_record(
            transform=transform,
            component_index=component_index,
            transform_index=transform_index,
            grid=_constructed_affine_grid(
                affine_transform_count,
                resolution_assignment=self._resolution_assignment_for_requested_state_count(
                    _state_space_digit_count * affine_transform_count,
                ),
            ),
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


def _sample_component_index(sample: GeneratedSample) -> int:
    if sample.component_index is None:
        raise ObservationGenerationError("Digits sample is missing component index")
    return sample.component_index


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


def _target_distribution_row(
    distribution: Mapping[str, float],
    *,
    outcome_ids: tuple[str, ...],
) -> list[float]:
    unknown = tuple(outcome_id for outcome_id in distribution if outcome_id not in outcome_ids)
    if unknown:
        raise TensorRuntimeError(f"unknown target outcome id: {unknown[0]}")
    return [float(distribution.get(outcome_id, 0.0)) for outcome_id in outcome_ids]


def _constructed_affine_grid(
    transform_count: int,
    *,
    resolution_assignment: AxisAssignment | None = None,
) -> _ConstructedAffineGrid:
    _require_generation_positive_integer(transform_count, "transform_count")
    side = _state_space_canvas_side_from_assignment(resolution_assignment)
    bounds = _constructed_affine_bounds_for_canvas_side(side)
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
    for counts in _constructed_affine_count_products(
        capacities=capacities,
        minimum_product=minimum_transform_count,
        maximum_product=maximum_transform_count,
    ):
        transform_count = _sample_count(counts)
        if transform_count in seen_products:
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
    products = _constructed_affine_count_products(
        capacities=capacities,
        minimum_product=transform_count,
        maximum_product=transform_count,
    )
    if products:
        return products[0]
    raise ObservationGenerationError(
        "requested affine transform count exceeds resolution-aware grid capacity"
    )


def _constructed_affine_count_products(
    *,
    capacities: tuple[int, int, int, int, int],
    minimum_product: int,
    maximum_product: int,
) -> tuple[tuple[int, int, int, int, int], ...]:
    products: list[tuple[int, int, int, int, int]] = []
    for x_translation in range(1, capacities[0] + 1):
        partial_x = x_translation
        if partial_x > maximum_product:
            break
        for y_translation in range(1, capacities[1] + 1):
            partial_y = partial_x * y_translation
            if partial_y > maximum_product:
                break
            for scale in range(1, capacities[2] + 1):
                partial_scale = partial_y * scale
                if partial_scale > maximum_product:
                    break
                for rotation in range(1, capacities[3] + 1):
                    partial_rotation = partial_scale * rotation
                    if partial_rotation > maximum_product:
                        break
                    for x_shear in range(1, capacities[4] + 1):
                        product = partial_rotation * x_shear
                        if product > maximum_product:
                            break
                        if product >= minimum_product:
                            products.append(
                                (
                                    x_translation,
                                    y_translation,
                                    scale,
                                    rotation,
                                    x_shear,
                                )
                            )
    return tuple(
        sorted(
            products,
            key=lambda counts: (
                _sample_count(counts),
                max(counts),
                counts[4],
                counts[3],
                counts[2],
                counts[1],
                counts[0],
            ),
        )
    )


def _constructed_affine_indices(
    *,
    transform_index: int,
    grid: _ConstructedAffineGrid,
) -> dict[str, int]:
    if type(transform_index) is not int or transform_index < 0:
        raise ObservationGenerationError("transform_index must be a nonnegative integer")
    if transform_index >= grid.transform_count:
        raise ObservationGenerationError("transform_index must be below transform_count")
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


def _grid_value(bounds: tuple[float, float], *, index: int, count: int) -> float:
    lower, upper = bounds
    if count <= 1:
        return (lower + upper) / 2.0
    return lower + (upper - lower) * (index / (count - 1))


@dataclass(slots=True)
class _FormationTensorCache:
    """Cache unvaried Digits component fields as runtime tensors."""

    runtime: TensorRuntime
    formation: ObservationFormationDeclaration
    _component_tensors: dict[tuple[int, int, int, str], Any] = dataclass_field(
        default_factory=lambda: cast(dict[tuple[int, int, int, str], Any], {})
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
            [
                _target_distribution_row(
                    sample.target_distribution_or_one_hot(),
                    outcome_ids=outcome_ids,
                )
                for sample in batch.samples
            ],
            dtype=backend.float32,
            device=self.runtime.device,
        )
        return fields, labels

    def _varied_batch_tensor(self, *, batch: GeneratedSampleSet) -> Any:
        sample_count = len(batch.samples)
        if sample_count < 1:
            raise TensorRuntimeError("batch samples must not be empty")
        width, height = _sample_field_size(batch.samples[0])
        source_tensors: list[Any] = []
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
                    variation_coordinate=sample.variation_coordinates[0],
                )
            )
        backend = getattr(self.runtime, "tor" + "ch")
        return backend.stack(source_tensors)

    def component_tensor(
        self,
        *,
        width: int,
        height: int,
        component_index: int,
        variation_coordinate: Mapping[str, object] | None = None,
    ) -> Any:
        _require_positive_integer(width, "width")
        _require_positive_integer(height, "height")
        if (
            type(component_index) is not int
            or component_index < 0
            or component_index >= len(self.formation.components)
        ):
            raise TensorRuntimeError("component_index is outside component vocabulary")
        coordinate_digest = (
            ""
            if variation_coordinate is None
            else str(ContentDigest.from_value(variation_coordinate))
        )
        key = (width, height, component_index, coordinate_digest)
        cached = self._component_tensors.get(key)
        if cached is not None:
            return cached
        tensor = self._build_component_tensor(
            width=width,
            height=height,
            component_index=component_index,
            variation_coordinate=variation_coordinate,
        )
        self._component_tensors[key] = tensor
        return tensor

    def _build_component_tensor(
        self,
        *,
        width: int,
        height: int,
        component_index: int,
        variation_coordinate: Mapping[str, object] | None,
    ) -> Any:
        field = self.formation.component_field(
            width=width,
            height=height,
            component_index=component_index,
            variation_coordinate=variation_coordinate,
        )
        backend = getattr(self.runtime, "tor" + "ch")
        tensor = backend.tensor(
            field.values,
            dtype=backend.float32,
            device=self.runtime.device,
        )
        return tensor.reshape(field.shape)


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


def _sample_field_size(sample: GeneratedSample) -> tuple[int, int]:
    if sample.width is None or sample.height is None:
        raise TensorRuntimeError("field sample is missing canvas dimensions")
    return sample.width, sample.height


def _require_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise TensorRuntimeError(f"{name} must be a positive integer")


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
                    "single-digit slices. Requests smaller than the full digit "
                    "vocabulary activate only a prefix of digit classes; after "
                    "all digit classes are active, each digit receives the same "
                    "finite affine choices. The benchmark reports the realized "
                    "cardinality instead of forcing exact powers of two."
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
