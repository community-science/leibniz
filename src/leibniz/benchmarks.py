"""Benchmark manifests for finite-outcome scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.outcomes import OutcomeSpace
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "BenchmarkManifestDocument",
    "BenchmarkManifest",
    "BenchmarkManifestValidationError",
]

_benchmark_manifest_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "name": FieldSpec(kind="name", required=False),
        "outcome_space": FieldSpec(kind="record", required=False),
        "observation_ids": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="string"),
            required=False,
        ),
        "latent_factor_declaration": FieldSpec(kind="record", required=False),
        "resolution_analysis": FieldSpec(kind="record", required=False),
    }
)


class BenchmarkManifestValidationError(ValueError):
    """Raised when a benchmark manifest is invalid."""


class _RecordSerializable(Protocol):
    def to_record(self) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """A benchmark manifest for fixed finite outcomes."""

    id: ProtocolIdentifier
    name: ProtocolName
    outcome_space: OutcomeSpace
    observation_ids: frozenset[str] | None = None
    latent_factor_declaration: ArtifactReference | None = None
    resolution_analysis: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise BenchmarkManifestValidationError(str(error)) from error
        if self.id.name != self.name:
            raise BenchmarkManifestValidationError(
                f"name {self.name} does not match id name {self.id.name}"
            )
        if self.observation_ids is not None:
            if not self.observation_ids:
                raise BenchmarkManifestValidationError(
                    "observation_ids must contain at least one observation id"
                )
            if any(not observation_id for observation_id in self.observation_ids):
                raise BenchmarkManifestValidationError("observation_ids must be nonempty")
        if (
            self.latent_factor_declaration is not None
            and self.latent_factor_declaration.kind != "latent-factor-declaration"
        ):
            raise BenchmarkManifestValidationError(
                "latent_factor_declaration reference must have kind "
                "latent-factor-declaration"
            )
        if self.resolution_analysis is not None:
            if self.resolution_analysis.get("kind") != "component-discriminability-margin":
                raise BenchmarkManifestValidationError(
                    "resolution_analysis kind must be component-discriminability-margin"
                )
            margin = self.resolution_analysis.get("discriminability_margin")
            if (
                not isinstance(margin, int | float)
                or isinstance(margin, bool)
                or float(margin) <= 0.0
            ):
                raise BenchmarkManifestValidationError(
                    "resolution_analysis discriminability_margin must be positive"
                )
            for field in (
                "affine_minimum_absolute_determinant",
                "affine_minimum_axis_alignment",
                "affine_minimum_cell_overlap_ratio",
                "affine_minimum_singular_value",
                "affine_maximum_singular_value",
                "affine_maximum_condition_number",
                "affine_minimum_projected_extent",
                "affine_maximum_projected_extent",
            ):
                value = self.resolution_analysis.get(field)
                if value is None:
                    continue
                if (
                    not isinstance(value, int | float)
                    or isinstance(value, bool)
                    or float(value) <= 0.0
                ):
                    raise BenchmarkManifestValidationError(
                        f"resolution_analysis {field} must be positive"
                    )
            minimum_extent = self.resolution_analysis.get("affine_minimum_projected_extent")
            maximum_extent = self.resolution_analysis.get("affine_maximum_projected_extent")
            if (
                isinstance(minimum_extent, int | float)
                and not isinstance(minimum_extent, bool)
                and isinstance(maximum_extent, int | float)
                and not isinstance(maximum_extent, bool)
                and float(minimum_extent) > float(maximum_extent)
            ):
                raise BenchmarkManifestValidationError(
                    "resolution_analysis affine_minimum_projected_extent must not exceed "
                    "affine_maximum_projected_extent"
                )
            minimum_singular_value = self.resolution_analysis.get(
                "affine_minimum_singular_value"
            )
            maximum_singular_value = self.resolution_analysis.get(
                "affine_maximum_singular_value"
            )
            if (
                isinstance(minimum_singular_value, int | float)
                and not isinstance(minimum_singular_value, bool)
                and isinstance(maximum_singular_value, int | float)
                and not isinstance(maximum_singular_value, bool)
                and float(minimum_singular_value) > float(maximum_singular_value)
            ):
                raise BenchmarkManifestValidationError(
                    "resolution_analysis affine_minimum_singular_value must not exceed "
                    "affine_maximum_singular_value"
                )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BenchmarkManifest:
        try:
            validated = _benchmark_manifest_record.validate(record)
        except ValueError as error:
            raise BenchmarkManifestValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            name=_manifest_name(validated),
            outcome_space=_manifest_outcome_space(validated),
            observation_ids=_manifest_observation_ids(validated),
            latent_factor_declaration=_manifest_latent_factor_declaration(validated),
            resolution_analysis=_manifest_resolution_analysis(validated),
        )

    def validate_latent_factor_declaration(self, declaration: _RecordSerializable) -> None:
        """Validate this manifest's latent factor declaration reference."""

        if self.latent_factor_declaration is None:
            raise BenchmarkManifestValidationError(
                "manifest does not declare a latent factor reference"
            )
        if not self.latent_factor_declaration.matches_record(declaration.to_record()):
            raise BenchmarkManifestValidationError(
                "latent_factor_declaration reference does not match declaration"
            )

    def resolve_outcome_space(self) -> OutcomeSpace:
        """Return this benchmark's fixed finite outcome space."""

        return self.outcome_space

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "name": str(self.name),
        }
        record["outcome_space"] = self.outcome_space.to_record()
        if self.observation_ids is not None:
            record["observation_ids"] = sorted(self.observation_ids)
        if self.latent_factor_declaration is not None:
            record["latent_factor_declaration"] = self.latent_factor_declaration.to_record()
        if self.resolution_analysis is not None:
            record["resolution_analysis"] = dict(self.resolution_analysis)
        return record

    def resolution_discriminability_margin(self) -> float:
        """Return the requested component-separation margin for resolution analysis."""

        if self.resolution_analysis is None:
            return 0.0
        value = self.resolution_analysis["discriminability_margin"]
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise BenchmarkManifestValidationError(
                "resolution_analysis discriminability_margin must be numeric"
            )
        return float(value)

    def affine_acceptance_thresholds(self) -> dict[str, float]:
        """Return optional fast affine proposal acceptance thresholds."""

        if self.resolution_analysis is None:
            return {}
        thresholds: dict[str, float] = {}
        for field in (
            "affine_minimum_absolute_determinant",
            "affine_minimum_axis_alignment",
            "affine_minimum_cell_overlap_ratio",
            "affine_minimum_singular_value",
            "affine_maximum_singular_value",
            "affine_maximum_condition_number",
            "affine_minimum_projected_extent",
            "affine_maximum_projected_extent",
        ):
            value = self.resolution_analysis.get(field)
            if value is None:
                continue
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise BenchmarkManifestValidationError(
                    f"resolution_analysis {field} must be numeric"
                )
            thresholds[field] = float(value)
        return thresholds


@dataclass(frozen=True, slots=True)
class BenchmarkManifestDocument:
    """A loaded benchmark manifest and the digest of its canonical record."""

    manifest: BenchmarkManifest
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> BenchmarkManifestDocument:
        try:
            record = load_object_document(data, description="manifest document")
        except ContentEncodingError as error:
            raise BenchmarkManifestValidationError(str(error)) from error
        manifest = BenchmarkManifest.from_record(record)
        return cls(manifest=manifest, digest=ContentDigest.from_value(manifest.to_record()))


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise BenchmarkManifestValidationError(f"{field}: expected parsed identifier")
    return value


def _as_name(value: object, *, field: str) -> ProtocolName:
    if not isinstance(value, ProtocolName):
        raise BenchmarkManifestValidationError(f"{field}: expected parsed name")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkManifestValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _manifest_name(validated: Mapping[str, object]) -> ProtocolName:
    identifier = _as_identifier(validated["id"], field="id")
    value = validated.get("name")
    if value is None:
        return identifier.name
    return _as_name(value, field="name")


def _manifest_outcome_space(validated: Mapping[str, object]) -> OutcomeSpace:
    value = validated.get("outcome_space")
    if value is None:
        raise BenchmarkManifestValidationError("manifest must declare outcome_space")
    try:
        return OutcomeSpace.from_record(_as_mapping(value, field="outcome_space"))
    except ValueError as error:
        raise BenchmarkManifestValidationError(str(error)) from error


def _manifest_observation_ids(validated: Mapping[str, object]) -> frozenset[str] | None:
    value = validated.get("observation_ids")
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise BenchmarkManifestValidationError("observation_ids: expected parsed sequence")
    observation_ids = tuple(
        str(observation_id) for observation_id in cast(tuple[object, ...], value)
    )
    if len(set(observation_ids)) != len(observation_ids):
        raise BenchmarkManifestValidationError("observation_ids must be unique")
    return frozenset(observation_ids)


def _manifest_latent_factor_declaration(
    validated: Mapping[str, object],
) -> ArtifactReference | None:
    value = validated.get("latent_factor_declaration")
    if value is None:
        return None
    try:
        return ArtifactReference.from_record(
            _as_mapping(value, field="latent_factor_declaration")
        )
    except ValueError as error:
        raise BenchmarkManifestValidationError(str(error)) from error


def _manifest_resolution_analysis(
    validated: Mapping[str, object],
) -> Mapping[str, object] | None:
    value = validated.get("resolution_analysis")
    if value is None:
        return None
    return dict(_as_mapping(value, field="resolution_analysis"))
