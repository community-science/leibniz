"""Runtime generation of declaration-backed benchmark observations."""

from __future__ import annotations

import base64
import random
import struct
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestDocument
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
)

__all__ = [
    "GeneratedObservationBatch",
    "GeneratedObservationSample",
    "ObservationGenerator",
    "ObservationGenerationError",
    "field_to_png_bytes",
    "field_to_png_data_url",
    "load_observation_generator",
]

_document_suffix = "." + "".join(("j", "s", "o", "n"))


class ObservationGenerationError(ValueError):
    """Raised when observation generation cannot satisfy benchmark declarations."""


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
            "samples": [
                sample.to_record(include_field=include_fields)
                for sample in self.samples
            ],
        }


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
    ) -> GeneratedObservationBatch:
        """Generate a deterministic batch at one scale."""

        if type(sample_count) is not int or sample_count < 1:
            raise ObservationGenerationError("sample_count must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        scaled_factors = tuple(self._scaled_sample_factors(scale))
        complexity = self._complexity(scaled_factors)
        complexity_axis = self._complexity_axis()
        sequences = tuple(component_sequences) if component_sequences is not None else ()
        if sequences and len(sequences) != sample_count:
            raise ObservationGenerationError(
                "component_sequences length must match sample_count"
            )

        samples: list[GeneratedObservationSample] = []
        for index in range(sample_count):
            plan = self._materialization_plan(
                scale=scale,
                seed=seed,
                index=index,
                complexity_axis=complexity_axis,
                complexity=complexity,
            )
            sequence = (
                tuple(sequences[index])
                if sequences
                else self.formation.sample_component_sequence(plan=plan, sample_index=index)
            )
            observation = self.formation.form_observation(
                id=self._observation_id(scale=scale, seed=seed, index=index),
                plan=plan,
                component_sequence=sequence,
            )
            samples.append(
                GeneratedObservationSample(
                    index=index,
                    materialization_plan=plan,
                    observation=observation,
                    outcome_id=self._outcome_id(sequence),
                    complexity=complexity,
                    latent_coordinates=self._latent_coordinates(
                        sequence=sequence,
                        scaled_factors=scaled_factors,
                        plan=plan,
                        sample_index=index,
                    ),
                )
            )

        return GeneratedObservationBatch(
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
            if factor.role in {"content", "nuisance"} and factor.multiplicity == 1:
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
        complexity_axis: str,
        complexity: float,
    ) -> MaterializationPlan:
        if self.benchmark_manifest.scale_parameter is None:
            raise ObservationGenerationError("benchmark manifest must declare scale")
        plan = MaterializationPlan.resolve(
            id=self._plan_id(scale=scale, seed=seed, index=index),
            declaration=self.materialization,
            scale_assignment=AxisAssignment(
                values={self.benchmark_manifest.scale_parameter.symbol: scale}
            ),
            complexity_assignment=AxisAssignment(values={complexity_axis: int(complexity)}),
            seed=seed,
        )
        plan.validate_declaration(self.materialization)
        return plan

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
        sample_index: int,
    ) -> tuple[Mapping[str, object], ...]:
        records: list[Mapping[str, object]] = []
        for factor in scaled_factors:
            if factor.role == "content":
                values: object = list(sequence)
            elif factor.role == "materialization":
                values = dict(plan.resolution_assignment.values)
            else:
                values = _nuisance_values(
                    seed=plan.seed,
                    sample_index=sample_index,
                    factor_name=str(factor.name),
                    count=int(factor.contribution),
                )
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


def field_to_png_bytes(field: FieldObservation) -> bytes:
    """Encode a one-channel field as a grayscale PNG image."""

    channels, height, width = field.shape
    if channels != 1:
        raise ObservationGenerationError("PNG encoding currently requires one channel")
    rows: list[bytes] = []
    for y_index in range(height):
        offset = y_index * width
        row = bytes(
            _uint8(field.values[offset + x_index])
            for x_index in range(width)
        )
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


def _nuisance_values(
    *,
    seed: int,
    sample_index: int,
    factor_name: str,
    count: int,
) -> list[float]:
    generator = random.Random(f"{seed}:{sample_index}:{factor_name}")
    return [generator.uniform(-0.5, 0.5) for _item in range(count)]


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
