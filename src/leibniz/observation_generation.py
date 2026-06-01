"""Runtime generation of declaration-backed benchmark observations."""

from __future__ import annotations

import base64
import math
import random
import struct
import zlib
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import cast

from leibniz.artifacts import ArtifactReference
from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestDocument
from leibniz.content import ContentDigest
from leibniz.documents import document_filename_suffix
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.latent_factors import (
    LatentFactorDeclaration,
    LatentFactorDeclarationDocument,
    SampleLatentFactor,
)
from leibniz.materialization import (
    AxisAssignment,
    MaterializationDeclaration,
    MaterializationDeclarationDocument,
    MaterializationPlan,
)
from leibniz.observation_formation import (
    FieldObservation,
    FormedObservation,
    ObservationFormationDeclaration,
    ObservationFormationDeclarationDocument,
    ObservationFormationValidationError,
    VariationTransformDeclaration,
)
from leibniz.timing import TimingCollector

__all__ = [
    "GeneratedFormationBatch",
    "GeneratedFormationSample",
    "GeneratedObservationBatch",
    "GeneratedObservationSample",
    "ObservationGenerator",
    "ObservationGenerationError",
    "field_to_png_bytes",
    "field_to_png_data_url",
    "load_observation_generator",
    "sample_variation_transform_coordinates",
]

_document_suffix = document_filename_suffix()
_discriminatable_resolution_cache: dict[
    tuple[str, int, str, str, int, int, float],
    tuple[int, int],
] = {}
_rejection_cache_bins_per_axis = 8
_rejection_cache_cell_limit = 4096
_variation_state_bins_per_axis = 8
_field_scalar_construction_bytes = 64
_default_memory_budget_fraction = 0.10
_default_generation_memory_limit_bytes = 1_024_000

class ObservationGenerationError(ValueError):
    """Raised when observation generation cannot satisfy benchmark declarations."""


@dataclass(frozen=True, slots=True)
class _ResolutionSampling:
    width_axis: str
    height_axis: str


@dataclass(frozen=True, slots=True)
class GeneratedObservationSample:
    """One generated observation with its scientific coordinates."""

    index: int
    materialization_plan: MaterializationPlan
    observation: FormedObservation
    outcome_id: str
    complexity: float
    latent_coordinates: tuple[Mapping[str, object], ...]

    @property
    def field(self) -> FieldObservation:
        """Return the generated channel-first field."""

        return self.observation.field

    def to_record(self, *, include_field: bool = False) -> dict[str, object]:
        record: dict[str, object] = {
            "index": self.index,
            "materialization_plan": self.materialization_plan.to_record(),
            "observation": self.observation.to_record(),
            "component_sequence": list(self.observation.component_sequence),
            "outcome_id": self.outcome_id,
            "complexity": self.complexity,
            "latent_coordinates": [dict(coordinate) for coordinate in self.latent_coordinates],
        }
        if include_field:
            record["field"] = self.field.to_record()
        return record


@dataclass(frozen=True, slots=True)
class GeneratedObservationBatch:
    """A deterministic batch of generated observations."""

    benchmark_id: ProtocolIdentifier
    component_count: int
    seed: int
    samples: tuple[GeneratedObservationSample, ...]

    def __post_init__(self) -> None:
        if type(self.component_count) is not int or self.component_count < 1:
            raise ObservationGenerationError("component_count must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        if not self.samples:
            raise ObservationGenerationError("samples must not be empty")

    def to_record(self, *, include_fields: bool = False) -> dict[str, object]:
        return {
            "benchmark_id": str(self.benchmark_id),
            "component_count": self.component_count,
            "seed": self.seed,
            "samples": [sample.to_record(include_field=include_fields) for sample in self.samples],
        }


@dataclass(frozen=True, slots=True)
class GeneratedFormationSample:
    """One generated formation specification without materializing the field."""

    index: int
    materialization_plan: MaterializationPlan
    width: int
    height: int
    component_sequence: tuple[int, ...]
    variation_coordinates: tuple[Mapping[str, object], ...]
    variation_values: Mapping[str, object]
    outcome_id: str
    complexity: float


@dataclass(frozen=True, slots=True)
class GeneratedFormationBatch:
    """A deterministic batch of formation specifications."""

    benchmark_id: ProtocolIdentifier
    component_count: int
    seed: int
    samples: tuple[GeneratedFormationSample, ...]

    def __post_init__(self) -> None:
        if type(self.component_count) is not int or self.component_count < 1:
            raise ObservationGenerationError("component_count must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        if not self.samples:
            raise ObservationGenerationError("samples must not be empty")


@dataclass(slots=True)
class _BoundedRejectionCache:
    cells: set[tuple[object, ...]] = field(default_factory=lambda: set())
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
class ObservationGenerator:
    """Generate observations from manifest, latent, materialization, and formation records."""

    benchmark_manifest: BenchmarkManifest
    latent_factors: LatentFactorDeclaration
    materialization: MaterializationDeclaration
    formation: ObservationFormationDeclaration
    rejection_cache: _BoundedRejectionCache = field(
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

    def sample_batch(
        self,
        *,
        component_count: int,
        sample_count: int,
        seed: int,
        component_sequences: Iterable[Sequence[int]] | None = None,
        memory_limit_bytes: int | None = None,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedObservationBatch:
        """Generate a deterministic batch with one internal component count."""

        with _timing_span(timing, f"{timing_prefix}formation_batch", samples=sample_count):
            formation_batch = self.sample_formation_batch(
                component_count=component_count,
                sample_count=sample_count,
                seed=seed,
                component_sequences=component_sequences,
                memory_limit_bytes=memory_limit_bytes,
                timing=timing,
                timing_prefix=f"{timing_prefix}formation_batch.",
            )
        with _timing_span(timing, f"{timing_prefix}scaled_factors"):
            scaled_factors = tuple(self.latent_factors.sample_factors)
        samples: list[GeneratedObservationSample] = []
        with _timing_span(
            timing,
            f"{timing_prefix}materialized_observation",
            samples=sample_count,
        ):
            observations = tuple(
                self.formation.form_observation(
                    id=self._observation_id(
                        component_count=component_count,
                        seed=seed,
                        index=spec.index,
                    ),
                    plan=spec.materialization_plan,
                    component_sequence=spec.component_sequence,
                    variation_coordinates=spec.variation_coordinates,
                )
                for spec in formation_batch.samples
            )
        with _timing_span(timing, f"{timing_prefix}latent_coordinates", samples=sample_count):
            latent_coordinate_samples = tuple(
                self._latent_coordinates(
                    sequence=spec.component_sequence,
                    scaled_factors=scaled_factors,
                    plan=spec.materialization_plan,
                    variation_values=spec.variation_values,
                )
                for spec in formation_batch.samples
            )
        with _timing_span(timing, f"{timing_prefix}sample_assembly", samples=sample_count):
            for spec, observation, latent_coordinates in zip(
                formation_batch.samples,
                observations,
                latent_coordinate_samples,
                strict=True,
            ):
                samples.append(
                    GeneratedObservationSample(
                        index=spec.index,
                        materialization_plan=spec.materialization_plan,
                        observation=observation,
                        outcome_id=spec.outcome_id,
                        complexity=spec.complexity,
                        latent_coordinates=latent_coordinates,
                    )
                )

        return GeneratedObservationBatch(
            benchmark_id=self.benchmark_manifest.id,
            component_count=component_count,
            seed=seed,
            samples=tuple(samples),
        )

    def sample_formation_batch(
        self,
        *,
        component_count: int,
        sample_count: int,
        seed: int,
        component_sequences: Iterable[Sequence[int]] | None = None,
        memory_limit_bytes: int | None = None,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedFormationBatch:
        """Generate deterministic formation specs without materializing fields."""

        if type(sample_count) is not int or sample_count < 1:
            raise ObservationGenerationError("sample_count must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        if type(component_count) is not int or component_count < 1:
            raise ObservationGenerationError("component_count must be a positive integer")
        if component_count != 1:
            raise ObservationGenerationError(
                "fixed-outcome component observations require one component"
            )
        with _timing_span(timing, f"{timing_prefix}component_sequences"):
            sequences = tuple(component_sequences) if component_sequences is not None else ()
        if sequences and len(sequences) != sample_count:
            raise ObservationGenerationError("component_sequences length must match sample_count")
        resolution_assignment = self.materialization.minimum_resolution()
        resolution_assignment = self._minimum_discriminatable_resolution_assignment(
            component_count=component_count,
            minimum_assignment=resolution_assignment,
        )
        resolution_assignment = self._sample_resolution_assignment(
            component_count=component_count,
            sample_count=sample_count,
            seed=seed,
            minimum_assignment=resolution_assignment,
            memory_limit_bytes=memory_limit_bytes,
        )
        self.materialization.require_resolution(
            resolution_assignment=resolution_assignment,
        )
        width = resolution_assignment.require_axis(self.formation.width_axis)
        height = resolution_assignment.require_axis(self.formation.height_axis)
        with _timing_span(timing, f"{timing_prefix}complexity"):
            distinguishable_state_count = self._distinguishable_state_count(
                component_count=component_count,
                width=width,
                height=height,
            )
            complexity = math.log2(distinguishable_state_count)
        materialization_declaration = ArtifactReference(
            kind="materialization-declaration",
            protocol_id=self.materialization.id,
            record_digest=self.materialization.digest,
        )
        sequence_length = component_count
        variation_transform_record = self.formation.variation_transform.to_record()
        variation_transform_digest = str(ContentDigest.from_value(variation_transform_record))
        component_generator = random.Random(f"{seed}:component-sequence")
        variation_generator = random.Random(f"{seed}:variation:{variation_transform_digest}")

        with _timing_span(
            timing,
            f"{timing_prefix}materialization_plan",
            samples=sample_count,
        ):
            plans = tuple(
                self._materialization_plan(
                    component_count=component_count,
                    seed=seed,
                    index=index,
                    resolution_assignment=resolution_assignment,
                    materialization_declaration=materialization_declaration,
                )
                for index in range(sample_count)
            )
        with _timing_span(timing, f"{timing_prefix}component_sequence", samples=sample_count):
            sequence_samples = tuple(
                (
                    tuple(sequences[index])
                    if sequences
                    else _sample_component_sequence(
                        generator=component_generator,
                        sequence_length=sequence_length,
                        component_count=len(self.formation.components),
                    )
                )
                for index in range(sample_count)
            )
        variation_samples: list[
            tuple[
                Mapping[str, object],
                tuple[Mapping[str, object], ...],
            ]
        ] = []
        variation_timing_phase = f"{timing_prefix}variation_coordinates"
        with _timing_span(timing, variation_timing_phase, samples=sample_count):
            for sequence, plan in zip(sequence_samples, plans, strict=True):
                variation_samples.append(
                    _variation_transform_values_and_coordinates(
                        formation=self.formation,
                        transform=self.formation.variation_transform,
                        transform_record=variation_transform_record,
                        generator=variation_generator,
                        sequence_length=len(sequence),
                        width=plan.resolution_assignment.require_axis(self.formation.width_axis),
                        height=plan.resolution_assignment.require_axis(self.formation.height_axis),
                        minimum_pairwise_l1=(
                            self.benchmark_manifest.resolution_discriminability_margin()
                        ),
                        affine_acceptance_thresholds=(
                            self.benchmark_manifest.affine_acceptance_thresholds()
                        ),
                        rejection_cache=self.rejection_cache,
                        timing=timing,
                        timing_phase=variation_timing_phase,
                    )
                )
        samples: list[GeneratedFormationSample] = []
        with _timing_span(timing, f"{timing_prefix}sample_assembly", samples=sample_count):
            for index, plan, sequence, variation_sample in zip(
                range(sample_count),
                plans,
                sequence_samples,
                variation_samples,
                strict=True,
            ):
                variation_values, variation_coordinates = variation_sample
                samples.append(
                    GeneratedFormationSample(
                        index=index,
                        materialization_plan=plan,
                        width=plan.resolution_assignment.require_axis(self.formation.width_axis),
                        height=plan.resolution_assignment.require_axis(self.formation.height_axis),
                        component_sequence=sequence,
                        variation_coordinates=variation_coordinates,
                        variation_values=variation_values,
                        outcome_id=self._outcome_id(sequence),
                        complexity=complexity,
                    )
                )

        return GeneratedFormationBatch(
            benchmark_id=self.benchmark_manifest.id,
            component_count=component_count,
            seed=seed,
            samples=tuple(samples),
        )

    def _distinguishable_state_count(
        self,
        *,
        component_count: int,
        width: int,
        height: int,
    ) -> int:
        component_states = len(self.formation.components) ** component_count
        variation_states = _variation_transform_state_count(
            self.formation.variation_transform,
            sequence_length=component_count,
            width=width,
            height=height,
        ) ** component_count
        return component_states * variation_states

    def _materialization_plan(
        self,
        *,
        component_count: int,
        seed: int,
        index: int,
        resolution_assignment: AxisAssignment,
        materialization_declaration: ArtifactReference,
    ) -> MaterializationPlan:
        return MaterializationPlan(
            id=self._plan_id(component_count=component_count, seed=seed, index=index),
            benchmark_id=self.materialization.benchmark_id,
            materialization_declaration=materialization_declaration,
            resolution_assignment=resolution_assignment,
            seed=seed,
            latent_factor_declaration=self.materialization.latent_factor_declaration,
        )

    def _sample_resolution_assignment(
        self,
        *,
        component_count: int,
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
        generator = random.Random(
            f"{seed}:resolution:{component_count}:{sampling.width_axis}:{minimum_width}:"
            f"{maximum_width}:{sampling.height_axis}:{minimum_height}:{maximum_height}:"
            f"{maximum_pixel_count}"
        )
        values = dict(minimum_assignment.values)
        values[sampling.width_axis] = generator.randint(minimum_width, maximum_width)
        values[sampling.height_axis] = generator.randint(minimum_height, maximum_height)
        return AxisAssignment(values=values)

    def _minimum_discriminatable_resolution_assignment(
        self,
        *,
        component_count: int,
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
            component_count,
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
                sequence_length=component_count,
                maximum_width=max(minimum_width * 64, component_count * 64),
                maximum_height=max(minimum_height * 64, 64),
                minimum_pairwise_l1=margin,
            )
            _discriminatable_resolution_cache[cache_key] = cached
        width, height = cached
        values = dict(minimum_assignment.values)
        values[sampling.width_axis] = max(values.get(sampling.width_axis, 0), width)
        values[sampling.height_axis] = max(values.get(sampling.height_axis, 0), height)
        return AxisAssignment(values=values)

    def _outcome_id(self, sequence: tuple[int, ...]) -> str:
        if len(sequence) != 1:
            raise ObservationGenerationError(
                "fixed-outcome component observations require one component"
            )
        index = sequence[0]
        if index >= len(self.benchmark_manifest.outcome_space.outcomes):
            raise ObservationGenerationError("component index is outside outcome space")
        return self.benchmark_manifest.outcome_space.outcomes[index].id

    def _latent_coordinates(
        self,
        *,
        sequence: tuple[int, ...],
        scaled_factors: tuple[SampleLatentFactor, ...],
        plan: MaterializationPlan,
        variation_values: Mapping[str, object],
    ) -> tuple[Mapping[str, object], ...]:
        records: list[Mapping[str, object]] = []
        for factor in scaled_factors:
            if factor.role == "content":
                values: object = list(sequence)
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
        component_count: int,
        seed: int,
        index: int,
    ) -> ProtocolIdentifier:
        return _child_identifier(
            self.benchmark_manifest.id,
            f"materialization-plans.c{component_count}.seed{seed}.sample-{index}",
        )

    def _observation_id(
        self,
        *,
        component_count: int,
        seed: int,
        index: int,
    ) -> ProtocolIdentifier:
        return _child_identifier(
            self.benchmark_manifest.id,
            f"observations.c{component_count}.seed{seed}.sample-{index}",
        )


def load_observation_generator(benchmark_root: Path) -> ObservationGenerator:
    """Load an observation generator from a benchmark declaration directory."""

    return ObservationGenerator(
        benchmark_manifest=BenchmarkManifestDocument.from_bytes(
            (benchmark_root / ("manifest" + _document_suffix)).read_bytes()
        ).manifest,
        latent_factors=LatentFactorDeclarationDocument.from_bytes(
            (benchmark_root / ("latent_factors" + _document_suffix)).read_bytes()
        ).declaration,
        materialization=MaterializationDeclarationDocument.from_bytes(
            (benchmark_root / ("materialization" + _document_suffix)).read_bytes()
        ).declaration,
        formation=ObservationFormationDeclarationDocument.from_bytes(
            (benchmark_root / ("observation_formation" + _document_suffix)).read_bytes()
        ).declaration,
    )


def sample_variation_transform_coordinates(
    *,
    transform: VariationTransformDeclaration,
    seed: int,
    sample_index: int,
    sequence_index: int,
) -> Mapping[str, object]:
    """Sample one deterministic per-sequence-position variation coordinate."""

    if type(seed) is not int or seed < 0:
        raise ObservationGenerationError("seed must be a nonnegative integer")
    if type(sample_index) is not int or sample_index < 0:
        raise ObservationGenerationError("sample_index must be a nonnegative integer")
    if type(sequence_index) is not int or sequence_index < 0:
        raise ObservationGenerationError("sequence_index must be a nonnegative integer")
    generator = _variation_random(
        seed=seed,
        sample_index=sample_index,
        sequence_index=sequence_index,
        transform_digest=str(ContentDigest.from_value(transform.to_record())),
    )
    return _variation_coordinate_record(
        transform=transform,
        generator=generator,
        sequence_index=sequence_index,
    )


def _variation_random(
    *,
    seed: int,
    sample_index: int,
    sequence_index: int,
    transform_digest: str,
) -> random.Random:
    return random.Random(
        ":".join((str(seed), str(sample_index), str(sequence_index), transform_digest))
    )


def _variation_coordinate_record(
    *,
    transform: VariationTransformDeclaration,
    generator: random.Random,
    sequence_index: int,
    thresholds: Mapping[str, float] | None = None,
) -> Mapping[str, object]:
    spatial = transform.spatial_affine
    return {
        "kind": "field-variation-transform-coordinate",
        "sequence_index": sequence_index,
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
    sequence_length: int,
    width: int,
    height: int,
    minimum_pairwise_l1: float,
    affine_acceptance_thresholds: Mapping[str, float],
    rejection_cache: _BoundedRejectionCache | None = None,
    timing: TimingCollector | None = None,
    timing_phase: str = "",
) -> tuple[
    Mapping[str, object],
    tuple[Mapping[str, object], ...],
]:
    if sequence_length < 1:
        raise ObservationGenerationError("sequence_length must be positive")
    counters: dict[str, float] = {}
    coordinates = tuple(
        _accepted_variation_coordinate(
            formation=formation,
            transform=transform,
            generator=generator,
            sequence_length=sequence_length,
            sequence_index=sequence_index,
            width=width,
            height=height,
            minimum_pairwise_l1=minimum_pairwise_l1,
            affine_acceptance_thresholds=affine_acceptance_thresholds,
            rejection_cache=rejection_cache,
            counters=counters,
        )
        for sequence_index in range(sequence_length)
    )
    if timing is not None and counters:
        timing.add_counters(timing_phase, counters)
    return (
        {
            "kind": "field-variation-transform-samples",
            "bounds": transform_record,
            "coordinates": [dict(coordinate) for coordinate in coordinates],
        },
        coordinates,
    )


def _accepted_variation_coordinate(
    *,
    formation: ObservationFormationDeclaration,
    transform: VariationTransformDeclaration,
    generator: random.Random,
    sequence_length: int,
    sequence_index: int,
    width: int,
    height: int,
    minimum_pairwise_l1: float,
    affine_acceptance_thresholds: Mapping[str, float],
    rejection_cache: _BoundedRejectionCache | None,
    counters: dict[str, float],
) -> Mapping[str, object]:
    cache_scope = _rejection_cache_scope(
        transform=transform,
        thresholds=affine_acceptance_thresholds,
    )
    for sampler, attempt_count in (
        (_variation_coordinate_record, 512),
        (_readable_variation_coordinate_record, 128),
    ):
        sampler_name = "broad" if sampler is _variation_coordinate_record else "readable"
        for _attempt in range(attempt_count):
            _increment_counter(counters, "candidate_count")
            _increment_counter(counters, f"{sampler_name}_candidate_count")
            coordinate = dict(
                sampler(
                    transform=transform,
                    generator=generator,
                    sequence_index=sequence_index,
                    thresholds=affine_acceptance_thresholds,
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
            try:
                analysis_width, analysis_height = _variation_analysis_extent(
                    formation=formation,
                    sequence_length=sequence_length,
                    width=width,
                    height=height,
                )
                analysis_coordinate = dict(coordinate)
                analysis_coordinate["sequence_index"] = 0
                _increment_counter(counters, "analysis_discriminability_check_count")
                if not formation.component_discriminability_passes(
                    width=analysis_width,
                    height=analysis_height,
                    sequence_length=1,
                    sequence_index=0,
                    variation_coordinates=(analysis_coordinate,),
                    minimum_pairwise_l1=minimum_pairwise_l1,
                ):
                    _increment_counter(counters, "analysis_reject_count")
                    continue
                _increment_counter(counters, "canvas_discriminability_check_count")
                accepted = formation.component_discriminability_passes(
                    width=width,
                    height=height,
                    sequence_length=sequence_length,
                    sequence_index=sequence_index,
                    variation_coordinates=(coordinate,),
                    minimum_pairwise_l1=minimum_pairwise_l1,
                )
            except ObservationFormationValidationError:
                _increment_counter(counters, "validation_error_count")
                continue
            if accepted:
                _increment_counter(counters, "accepted_count")
                return coordinate
            _increment_counter(counters, "canvas_reject_count")
    identity_coordinate = _identity_variation_coordinate_record(
        transform=transform,
        sequence_index=sequence_index,
    )
    try:
        _increment_counter(counters, "identity_fallback_check_count")
        if formation.component_discriminability_passes(
            width=width,
            height=height,
            sequence_length=sequence_length,
            sequence_index=sequence_index,
            variation_coordinates=(identity_coordinate,),
            minimum_pairwise_l1=minimum_pairwise_l1,
        ):
            _increment_counter(counters, "identity_fallback_count")
            return identity_coordinate
    except ObservationFormationValidationError:
        _increment_counter(counters, "validation_error_count")
        pass
    raise ObservationGenerationError(
        "could not sample an identity-preserving affine coordinate"
    )


def _readable_variation_coordinate_record(
    *,
    transform: VariationTransformDeclaration,
    generator: random.Random,
    sequence_index: int,
    thresholds: Mapping[str, float],
) -> Mapping[str, object]:
    spatial = transform.spatial_affine
    if spatial.spatial_rank != 2:
        return _variation_coordinate_record(
            transform=transform,
            generator=generator,
            sequence_index=sequence_index,
        )
    minimum_scale = thresholds.get("affine_minimum_singular_value", 0.75)
    maximum_scale = thresholds.get("affine_maximum_singular_value", 1.25)
    maximum_condition = thresholds.get("affine_maximum_condition_number", 1.35)
    minimum_alignment = thresholds.get("affine_minimum_axis_alignment", 0.9)
    maximum_angle = math.acos(max(-1.0, min(1.0, minimum_alignment)))
    rotation = generator.uniform(-maximum_angle, maximum_angle)
    scale_x = generator.uniform(minimum_scale, maximum_scale)
    scale_y = generator.uniform(
        max(minimum_scale, scale_x / maximum_condition),
        min(maximum_scale, scale_x * maximum_condition),
    )
    shear = generator.uniform(-math.tan(maximum_angle) * 0.25, math.tan(maximum_angle) * 0.25)
    cos_angle = math.cos(rotation)
    sin_angle = math.sin(rotation)
    a = scale_x * cos_angle
    c = scale_x * sin_angle
    b = scale_y * (shear * cos_angle - sin_angle)
    d = scale_y * (shear * sin_angle + cos_angle)
    tx = generator.uniform(-0.15, 0.15)
    ty = generator.uniform(-0.15, 0.15)
    return {
        "kind": "field-variation-transform-coordinate",
        "sequence_index": sequence_index,
        "spatial_affine": {
            "kind": "spatial-affine-coordinate",
            "coordinate_system": spatial.coordinate_system,
            "matrix": [[a, b, tx], [c, d, ty], [0.0, 0.0, 1.0]],
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


def _variation_analysis_extent(
    *,
    formation: ObservationFormationDeclaration,
    sequence_length: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    if formation.variation_transform.spatial_affine.coordinate_system != (
        "normalized-sequence-element"
    ):
        return (width, height)
    if formation.sequence_layout.placement_axis == "x":
        return (max(1, width // sequence_length), height)
    return (width, max(1, height // sequence_length))


def _identity_variation_coordinate_record(
    *,
    transform: VariationTransformDeclaration,
    sequence_index: int,
) -> Mapping[str, object]:
    spatial = transform.spatial_affine
    spatial_rank = spatial.spatial_rank
    matrix = [
        [1.0 if row_index == column_index else 0.0 for column_index in range(spatial_rank + 1)]
        for row_index in range(spatial_rank + 1)
    ]
    return {
        "kind": "field-variation-transform-coordinate",
        "sequence_index": sequence_index,
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


def _variation_transform_state_count(
    transform: VariationTransformDeclaration,
    *,
    sequence_length: int,
    width: int,
    height: int,
) -> int:
    if sequence_length < 1:
        raise ObservationGenerationError("sequence_length must be positive")
    element_width = max(1, width // sequence_length)
    row_extents = (element_width, height, 1)
    count = 1
    for row_index, row in enumerate(transform.spatial_affine.matrix):
        extent = row_extents[row_index] if row_index < len(row_extents) else 1
        for lower, upper in row:
            if upper > lower:
                count *= _distinguishable_interval_count(
                    lower=lower,
                    upper=upper,
                    extent=extent,
                )
    return count


def _distinguishable_interval_count(
    *,
    lower: float,
    upper: float,
    extent: int,
) -> int:
    if extent < 1:
        return 1
    return max(1, math.floor((upper - lower) * extent) + 1)


def _sampled_resolution_maximum(
    *,
    minimum_width: int,
    minimum_height: int,
    maximum_pixel_count: int,
) -> tuple[int, int]:
    minimum_pixel_count = minimum_width * minimum_height
    if maximum_pixel_count < minimum_pixel_count:
        raise ObservationGenerationError(
            "resource budget cannot fit minimum observation canvas"
        )
    side_multiplier = math.sqrt(maximum_pixel_count / minimum_pixel_count)
    return (
        math.floor(minimum_width * side_multiplier),
        math.floor(minimum_height * side_multiplier),
    )


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


def _sample_component_sequence(
    *,
    generator: random.Random,
    sequence_length: int,
    component_count: int,
) -> tuple[int, ...]:
    if sequence_length < 1:
        raise ObservationGenerationError("sequence length must be positive")
    return tuple(
        generator.randrange(component_count) for _sequence_element in range(sequence_length)
    )


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
