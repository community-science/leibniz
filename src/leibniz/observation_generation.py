"""Runtime generation of declaration-backed benchmark observations."""

from __future__ import annotations

import base64
import math
import random
import struct
import zlib
from collections.abc import Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
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


class ObservationGenerationError(ValueError):
    """Raised when observation generation cannot satisfy benchmark declarations."""


@dataclass(frozen=True, slots=True)
class _ResolutionSampling:
    width_axis: str
    height_axis: str
    maximum_pixel_multiplier: float


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
    scale: int
    seed: int
    samples: tuple[GeneratedObservationSample, ...]

    def __post_init__(self) -> None:
        if type(self.scale) is not int or self.scale < 1:
            raise ObservationGenerationError("scale must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        if not self.samples:
            raise ObservationGenerationError("samples must not be empty")

    def to_record(self, *, include_fields: bool = False) -> dict[str, object]:
        return {
            "benchmark_id": str(self.benchmark_id),
            "scale": self.scale,
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
    scale: int
    seed: int
    samples: tuple[GeneratedFormationSample, ...]

    def __post_init__(self) -> None:
        if type(self.scale) is not int or self.scale < 1:
            raise ObservationGenerationError("scale must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        if not self.samples:
            raise ObservationGenerationError("samples must not be empty")


@dataclass(frozen=True, slots=True)
class ObservationGenerator:
    """Generate observations from manifest, latent, materialization, and formation records."""

    benchmark_manifest: BenchmarkManifest
    latent_factors: LatentFactorDeclaration
    materialization: MaterializationDeclaration
    formation: ObservationFormationDeclaration

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
        scale: int,
        sample_count: int,
        seed: int,
        component_sequences: Iterable[Sequence[int]] | None = None,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedObservationBatch:
        """Generate a deterministic batch at one scale."""

        with _timing_span(timing, f"{timing_prefix}formation_batch", samples=sample_count):
            formation_batch = self.sample_formation_batch(
                scale=scale,
                sample_count=sample_count,
                seed=seed,
                component_sequences=component_sequences,
                timing=timing,
                timing_prefix=f"{timing_prefix}formation_batch.",
            )
        with _timing_span(timing, f"{timing_prefix}scaled_factors"):
            scaled_factors = tuple(self._scaled_sample_factors(scale))
        samples: list[GeneratedObservationSample] = []
        with _timing_span(
            timing,
            f"{timing_prefix}materialized_observation",
            samples=sample_count,
        ):
            observations = tuple(
                self.formation.form_observation(
                    id=self._observation_id(scale=scale, seed=seed, index=spec.index),
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
            scale=scale,
            seed=seed,
            samples=tuple(samples),
        )

    def sample_formation_batch(
        self,
        *,
        scale: int,
        sample_count: int,
        seed: int,
        component_sequences: Iterable[Sequence[int]] | None = None,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedFormationBatch:
        """Generate deterministic formation specs without materializing fields."""

        if type(sample_count) is not int or sample_count < 1:
            raise ObservationGenerationError("sample_count must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        with _timing_span(timing, f"{timing_prefix}scaled_factors"):
            scaled_factors = tuple(self._scaled_sample_factors(scale))
        with _timing_span(timing, f"{timing_prefix}complexity"):
            complexity = self._complexity(scaled_factors)
            complexity_axis = self._complexity_axis()
        with _timing_span(timing, f"{timing_prefix}component_sequences"):
            sequences = tuple(component_sequences) if component_sequences is not None else ()
        if sequences and len(sequences) != sample_count:
            raise ObservationGenerationError("component_sequences length must match sample_count")
        if self.benchmark_manifest.scale_parameter is None:
            raise ObservationGenerationError("benchmark manifest must declare scale")
        scale_assignment = AxisAssignment(
            values={self.benchmark_manifest.scale_parameter.symbol: scale}
        )
        complexity_assignment = AxisAssignment(values={complexity_axis: int(complexity)})
        resolution_assignment = self.materialization.minimum_resolution(scale_assignment)
        resolution_assignment = self._minimum_discriminatable_resolution_assignment(
            scale=scale,
            minimum_assignment=resolution_assignment,
        )
        resolution_assignment = self._sample_resolution_assignment(
            scale=scale,
            seed=seed,
            minimum_assignment=resolution_assignment,
        )
        self.materialization.require_resolution(
            scale_assignment=scale_assignment,
            resolution_assignment=resolution_assignment,
        )
        materialization_declaration = ArtifactReference(
            kind="materialization-declaration",
            protocol_id=self.materialization.id,
            record_digest=self.materialization.digest,
        )
        sequence_length = scale_assignment.require_axis(
            self.formation.sequence_layout.sequence_axis
        )
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
                    scale=scale,
                    seed=seed,
                    index=index,
                    scale_assignment=scale_assignment,
                    complexity_assignment=complexity_assignment,
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
        with _timing_span(timing, f"{timing_prefix}variation_coordinates", samples=sample_count):
            for sequence in sequence_samples:
                variation_samples.append(
                    _variation_transform_values_and_coordinates(
                        transform=self.formation.variation_transform,
                        transform_record=variation_transform_record,
                        generator=variation_generator,
                        sequence_length=len(sequence),
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
            scale=scale,
            seed=seed,
            samples=tuple(samples),
        )

    def _scaled_sample_factors(self, scale: int) -> tuple[SampleLatentFactor, ...]:
        if type(scale) is not int or scale < 1:
            raise ObservationGenerationError("scale must be a positive integer")
        if self.benchmark_manifest.outcome_sequence is None:
            return self.latent_factors.sample_factors
        scale_axis = self.benchmark_manifest.outcome_sequence.length_parameter
        if self.benchmark_manifest.scale_parameter is None:
            raise ObservationGenerationError("sequence benchmarks require scale_parameter")
        if scale_axis != self.benchmark_manifest.scale_parameter.symbol:
            raise ObservationGenerationError("outcome sequence scale axis mismatch")

        factors: list[SampleLatentFactor] = []
        for factor in self.latent_factors.sample_factors:
            if factor.role in {"content", "variation"} and factor.multiplicity == 1:
                factors.append(
                    SampleLatentFactor(
                        name=factor.name,
                        role=factor.role,
                        degree_measure=factor.degree_measure,
                        multiplicity=scale,
                        description=factor.description,
                    )
                )
            else:
                factors.append(factor)
        return tuple(factors)

    def _complexity(self, scaled_factors: tuple[SampleLatentFactor, ...]) -> float:
        value = self.latent_factors.projection(self._complexity_axis()).evaluate(scaled_factors)
        if not value.is_integer():
            raise ObservationGenerationError("complexity coordinate must be integral")
        return value

    def _complexity_axis(self) -> str:
        axis = self.benchmark_manifest.complexity_coordinate
        if axis is None:
            raise ObservationGenerationError("benchmark manifest must declare complexity")
        return axis

    def _materialization_plan(
        self,
        *,
        scale: int,
        seed: int,
        index: int,
        scale_assignment: AxisAssignment,
        complexity_assignment: AxisAssignment,
        resolution_assignment: AxisAssignment,
        materialization_declaration: ArtifactReference,
    ) -> MaterializationPlan:
        return MaterializationPlan(
            id=self._plan_id(scale=scale, seed=seed, index=index),
            benchmark_id=self.materialization.benchmark_id,
            materialization_declaration=materialization_declaration,
            scale_assignment=scale_assignment,
            complexity_assignment=complexity_assignment,
            resolution_assignment=resolution_assignment,
            seed=seed,
            latent_factor_declaration=self.materialization.latent_factor_declaration,
        )

    def _sample_resolution_assignment(
        self,
        *,
        scale: int,
        seed: int,
        minimum_assignment: AxisAssignment,
    ) -> AxisAssignment:
        sampling = _resolution_sampling(self.materialization.layout)
        if sampling is None:
            return minimum_assignment
        minimum_width = minimum_assignment.require_axis(sampling.width_axis)
        minimum_height = minimum_assignment.require_axis(sampling.height_axis)
        side_multiplier = math.sqrt(sampling.maximum_pixel_multiplier)
        maximum_width = math.floor(minimum_width * side_multiplier)
        maximum_height = math.floor(minimum_height * side_multiplier)
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
            f"{seed}:resolution:{scale}:{sampling.width_axis}:{minimum_width}:"
            f"{maximum_width}:{sampling.height_axis}:{minimum_height}:{maximum_height}:"
            f"{sampling.maximum_pixel_multiplier}"
        )
        values = dict(minimum_assignment.values)
        values[sampling.width_axis] = generator.randint(minimum_width, maximum_width)
        values[sampling.height_axis] = generator.randint(minimum_height, maximum_height)
        return AxisAssignment(values=values)

    def _minimum_discriminatable_resolution_assignment(
        self,
        *,
        scale: int,
        minimum_assignment: AxisAssignment,
    ) -> AxisAssignment:
        sampling = _resolution_sampling(self.materialization.layout)
        if sampling is None:
            return minimum_assignment
        minimum_width = minimum_assignment.require_axis(sampling.width_axis)
        minimum_height = minimum_assignment.require_axis(sampling.height_axis)
        cache_key = (
            str(self.formation.digest),
            scale,
            sampling.width_axis,
            sampling.height_axis,
            minimum_width,
            minimum_height,
            self.benchmark_manifest.resolution_discriminability_margin(),
        )
        cached = _discriminatable_resolution_cache.get(cache_key)
        if cached is None:
            cached = self.formation.minimum_discriminatable_resolution(
                minimum_width=minimum_width,
                minimum_height=minimum_height,
                sequence_length=scale,
                maximum_width=max(minimum_width * 32, scale * 64),
                maximum_height=max(minimum_height * 32, 64),
                variation_coordinates=self.formation.boundary_variation_coordinates(
                    sequence_index=0
                ),
                minimum_pairwise_l1=self.benchmark_manifest.resolution_discriminability_margin(),
            )
            _discriminatable_resolution_cache[cache_key] = cached
        width, height = cached
        values = dict(minimum_assignment.values)
        values[sampling.width_axis] = max(values.get(sampling.width_axis, 0), width)
        values[sampling.height_axis] = max(values.get(sampling.height_axis, 0), height)
        return AxisAssignment(values=values)

    def _outcome_id(self, sequence: tuple[int, ...]) -> str:
        if self.benchmark_manifest.outcome_sequence is None:
            raise ObservationGenerationError("benchmark manifest must declare outcome sequence")
        return self.benchmark_manifest.outcome_sequence.outcome_id(sequence)

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

    def _plan_id(self, *, scale: int, seed: int, index: int) -> ProtocolIdentifier:
        return _child_identifier(
            self.benchmark_manifest.id,
            f"materialization-plans.l{scale}.seed{seed}.sample-{index}",
        )

    def _observation_id(self, *, scale: int, seed: int, index: int) -> ProtocolIdentifier:
        return _child_identifier(
            self.benchmark_manifest.id,
            f"observations.l{scale}.seed{seed}.sample-{index}",
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
) -> Mapping[str, object]:
    spatial = transform.spatial_affine
    return {
        "kind": "field-variation-transform-coordinate",
        "sequence_index": sequence_index,
        "spatial_affine": {
            "kind": "spatial-affine-coordinate",
            "coordinate_system": spatial.coordinate_system,
            "translation": [_sample_interval(generator, bounds) for bounds in spatial.translation],
            "scale": [_sample_interval(generator, bounds) for bounds in spatial.scale],
            "rotation_degrees": [
                _sample_symmetric(generator, bound) for bound in spatial.rotation_degrees
            ],
            "shear_degrees": [
                _sample_symmetric(generator, bound) for bound in spatial.shear_degrees
            ],
        },
        "value_scale": {
            "kind": "value-scale-coordinate",
            "scale": _sample_interval(generator, transform.value_scale.scale),
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
    transform: VariationTransformDeclaration,
    transform_record: Mapping[str, object],
    generator: random.Random,
    sequence_length: int,
) -> tuple[
    Mapping[str, object],
    tuple[Mapping[str, object], ...],
]:
    if sequence_length < 1:
        raise ObservationGenerationError("sequence_length must be positive")
    coordinates = tuple(
        dict(
            _variation_coordinate_record(
                transform=transform,
                generator=generator,
                sequence_index=sequence_index,
            )
        )
        for sequence_index in range(sequence_length)
    )
    return (
        {
            "kind": "field-variation-transform-samples",
            "bounds": transform_record,
            "coordinates": [dict(coordinate) for coordinate in coordinates],
        },
        coordinates,
    )


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
    multiplier = value.get("maximum_pixel_multiplier")
    if (
        not isinstance(multiplier, int | float)
        or isinstance(multiplier, bool)
        or not math.isfinite(float(multiplier))
        or float(multiplier) < 1.0
    ):
        raise ObservationGenerationError(
            "resolution_sampling maximum_pixel_multiplier must be at least 1"
        )
    return _ResolutionSampling(
        width_axis=width_axis,
        height_axis=height_axis,
        maximum_pixel_multiplier=float(multiplier),
    )


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


def _sample_symmetric(generator: random.Random, bound: float) -> float:
    return generator.uniform(-bound, bound)


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
