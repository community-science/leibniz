"""Benchmark manifests for finite-outcome scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.latent_factors import LatentFactorDeclaration
from leibniz.outcomes import OutcomeSpace
from leibniz.prediction_spaces import FiniteTokenSequenceSpace, FiniteTokenVocabulary
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "BenchmarkManifestDocument",
    "BenchmarkManifest",
    "BenchmarkOutcomeSequence",
    "BenchmarkScaleParameter",
    "BenchmarkManifestValidationError",
]

_benchmark_manifest_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "name": FieldSpec(kind="name", required=False),
        "outcome_space": FieldSpec(kind="record", required=False),
        "outcome_sequence": FieldSpec(kind="record", required=False),
        "scale_parameter": FieldSpec(kind="record", required=False),
        "observation_ids": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="string"),
            required=False,
        ),
        "latent_factor_declaration": FieldSpec(kind="record", required=False),
        "complexity_coordinate": FieldSpec(kind="string", required=False),
        "resolution_analysis": FieldSpec(kind="record", required=False),
    }
)
_scale_parameter_record = RecordSpec(
    fields={
        "symbol": FieldSpec(kind="string"),
        "minimum": FieldSpec(kind="integer"),
        "description": FieldSpec(kind="string", required=False),
    }
)
_outcome_sequence_record = RecordSpec(
    fields={
        "atom_count": FieldSpec(kind="integer"),
        "atom_name": FieldSpec(kind="string"),
        "length_parameter": FieldSpec(kind="string"),
    }
)


class BenchmarkManifestValidationError(ValueError):
    """Raised when a benchmark manifest is invalid."""


@dataclass(frozen=True, slots=True)
class BenchmarkScaleParameter:
    """The single unbounded scale parameter for a benchmark family."""

    symbol: str
    minimum: int
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise BenchmarkManifestValidationError("scale parameter symbol must be nonempty")
        if isinstance(self.minimum, bool):
            raise BenchmarkManifestValidationError("scale parameter minimum must be an integer")
        if self.minimum < 1:
            raise BenchmarkManifestValidationError("scale parameter minimum must be positive")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BenchmarkScaleParameter:
        try:
            validated = _scale_parameter_record.validate(record)
        except ValueError as error:
            raise BenchmarkManifestValidationError(str(error)) from error
        return cls(
            symbol=str(validated["symbol"]),
            minimum=_as_int(validated["minimum"], field="minimum"),
            description=_optional_string(validated.get("description"), field="description"),
        )

    def contains(self, value: int) -> bool:
        return value >= self.minimum

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "symbol": self.symbol,
            "minimum": self.minimum,
        }
        if self.description is not None:
            record["description"] = self.description
        return record


@dataclass(frozen=True, slots=True)
class BenchmarkOutcomeSequence:
    """A finite atom vocabulary lifted to fixed-length sequences by scale."""

    atom_count: int
    atom_name: str
    length_parameter: str

    def __post_init__(self) -> None:
        if isinstance(self.atom_count, bool):
            raise BenchmarkManifestValidationError("atom_count must be an integer")
        if self.atom_count < 2:
            raise BenchmarkManifestValidationError("atom_count must be at least 2")
        if not self.atom_name:
            raise BenchmarkManifestValidationError("atom_name must be nonempty")
        if not self.length_parameter:
            raise BenchmarkManifestValidationError("length_parameter must be nonempty")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BenchmarkOutcomeSequence:
        try:
            validated = _outcome_sequence_record.validate(record)
        except ValueError as error:
            raise BenchmarkManifestValidationError(str(error)) from error
        return cls(
            atom_count=_as_int(validated["atom_count"], field="atom_count"),
            atom_name=str(validated["atom_name"]),
            length_parameter=str(validated["length_parameter"]),
        )

    def outcome_count(self, scale: int) -> int:
        return self.token_sequence_space(length=scale).cardinality

    def outcome_index(self, atoms: Sequence[int]) -> int:
        """Return the lexicographic outcome index for a token sequence."""

        atom_values = tuple(_as_int(atom, field="atoms") for atom in atoms)
        try:
            return self.token_sequence_space(length=len(atom_values)).sequence_index(atom_values)
        except ValueError as error:
            raise BenchmarkManifestValidationError(str(error)) from error

    def atoms_for_outcome_index(self, *, index: int, length: int) -> tuple[int, ...]:
        """Return the token sequence at one lexicographic outcome index."""

        try:
            return self.token_sequence_space(length=length).sequence_for_index(index)
        except ValueError as error:
            raise BenchmarkManifestValidationError(str(error)) from error

    def outcome_id(self, atoms: Sequence[int]) -> str:
        atom_values = tuple(_as_int(atom, field="atoms") for atom in atoms)
        try:
            return self.token_sequence_space(length=len(atom_values)).outcome_id(atom_values)
        except ValueError as error:
            raise BenchmarkManifestValidationError(str(error)) from error

    def resolve_outcome_space(
        self,
        *,
        id: ProtocolIdentifier,
        length: int,
    ) -> OutcomeSpace:
        if isinstance(length, bool):
            raise BenchmarkManifestValidationError("length must be an integer")
        if length < 1:
            raise BenchmarkManifestValidationError("length must be positive")
        return self.token_sequence_space(length=length).outcome_space(id=id)

    @property
    def token_vocabulary(self) -> FiniteTokenVocabulary:
        """Return the generic token vocabulary represented by this manifest field."""

        return FiniteTokenVocabulary(
            token_count=self.atom_count,
            token_name=self.atom_name,
        )

    def token_sequence_space(self, *, length: int) -> FiniteTokenSequenceSpace:
        """Return the generic fixed-length token sequence prediction space."""

        return FiniteTokenSequenceSpace(
            vocabulary=self.token_vocabulary,
            length=length,
        )

    def to_record(self) -> dict[str, object]:
        return {
            "atom_count": self.atom_count,
            "atom_name": self.atom_name,
            "length_parameter": self.length_parameter,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """A benchmark manifest for fixed or scale-indexed finite outcomes."""

    id: ProtocolIdentifier
    name: ProtocolName
    outcome_space: OutcomeSpace | None = None
    outcome_sequence: BenchmarkOutcomeSequence | None = None
    scale_parameter: BenchmarkScaleParameter | None = None
    observation_ids: frozenset[str] | None = None
    latent_factor_declaration: ArtifactReference | None = None
    complexity_coordinate: str | None = None
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
        if (self.outcome_space is None) == (self.outcome_sequence is None):
            raise BenchmarkManifestValidationError(
                "manifest must declare exactly one of outcome_space or outcome_sequence"
            )
        if self.outcome_sequence is not None:
            if self.scale_parameter is None:
                raise BenchmarkManifestValidationError(
                    "outcome_sequence requires scale_parameter"
                )
            if self.outcome_sequence.length_parameter != self.scale_parameter.symbol:
                raise BenchmarkManifestValidationError(
                    "outcome_sequence length_parameter must match scale_parameter symbol"
                )
        elif self.scale_parameter is not None:
            raise BenchmarkManifestValidationError(
                "scale_parameter requires outcome_sequence"
            )
        if self.observation_ids is not None:
            if not self.observation_ids:
                raise BenchmarkManifestValidationError(
                    "observation_ids must contain at least one observation id"
                )
            if any(not observation_id for observation_id in self.observation_ids):
                raise BenchmarkManifestValidationError("observation_ids must be nonempty")
        if self.complexity_coordinate is not None:
            if not self.complexity_coordinate:
                raise BenchmarkManifestValidationError("complexity_coordinate must be nonempty")
            if self.latent_factor_declaration is None:
                raise BenchmarkManifestValidationError(
                    "complexity_coordinate requires latent_factor_declaration"
                )
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
            outcome_sequence=_manifest_outcome_sequence(validated),
            scale_parameter=_manifest_scale_parameter(validated),
            observation_ids=_manifest_observation_ids(validated),
            latent_factor_declaration=_manifest_latent_factor_declaration(validated),
            complexity_coordinate=_optional_string(
                validated.get("complexity_coordinate"),
                field="complexity_coordinate",
            ),
            resolution_analysis=_manifest_resolution_analysis(validated),
        )

    def validate_latent_factor_declaration(
        self,
        declaration: LatentFactorDeclaration,
    ) -> None:
        """Validate this manifest's complexity reference against a declaration."""

        if self.latent_factor_declaration is None:
            raise BenchmarkManifestValidationError(
                "manifest does not declare a latent factor reference"
            )
        if not self.latent_factor_declaration.matches_record(declaration.to_record()):
            raise BenchmarkManifestValidationError(
                "latent_factor_declaration reference does not match declaration"
            )
        if self.complexity_coordinate is None:
            raise BenchmarkManifestValidationError(
                "manifest does not declare a complexity coordinate"
            )
        try:
            declaration.projection(self.complexity_coordinate)
        except ValueError as error:
            raise BenchmarkManifestValidationError(str(error)) from error

    def resolve_outcome_space(self, *, scale: int) -> OutcomeSpace:
        """Resolve this benchmark's finite outcome space at one scale."""

        if self.outcome_space is not None:
            return self.outcome_space
        if self.outcome_sequence is None or self.scale_parameter is None:
            raise BenchmarkManifestValidationError("manifest does not declare outcomes")
        if not self.scale_parameter.contains(scale):
            raise BenchmarkManifestValidationError(
                f"scale {scale} is below minimum {self.scale_parameter.minimum}"
            )
        return self.outcome_sequence.resolve_outcome_space(
            id=ProtocolIdentifier.parse(f"{self.id.name}.outcomes.l{scale}@0.1.0"),
            length=scale,
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "name": str(self.name),
        }
        if self.outcome_space is not None:
            record["outcome_space"] = self.outcome_space.to_record()
        if self.outcome_sequence is not None:
            record["outcome_sequence"] = self.outcome_sequence.to_record()
        if self.scale_parameter is not None:
            record["scale_parameter"] = self.scale_parameter.to_record()
        if self.observation_ids is not None:
            record["observation_ids"] = sorted(self.observation_ids)
        if self.latent_factor_declaration is not None:
            record["latent_factor_declaration"] = self.latent_factor_declaration.to_record()
        if self.complexity_coordinate is not None:
            record["complexity_coordinate"] = self.complexity_coordinate
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


def _as_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise BenchmarkManifestValidationError(f"{field}: expected integer")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise BenchmarkManifestValidationError(f"{field}: expected string")
    return value


def _manifest_name(validated: Mapping[str, object]) -> ProtocolName:
    identifier = _as_identifier(validated["id"], field="id")
    value = validated.get("name")
    if value is None:
        return identifier.name
    return _as_name(value, field="name")


def _manifest_outcome_space(validated: Mapping[str, object]) -> OutcomeSpace | None:
    value = validated.get("outcome_space")
    if value is None:
        return None
    try:
        return OutcomeSpace.from_record(_as_mapping(value, field="outcome_space"))
    except ValueError as error:
        raise BenchmarkManifestValidationError(str(error)) from error


def _manifest_outcome_sequence(
    validated: Mapping[str, object],
) -> BenchmarkOutcomeSequence | None:
    value = validated.get("outcome_sequence")
    if value is None:
        return None
    return BenchmarkOutcomeSequence.from_record(_as_mapping(value, field="outcome_sequence"))


def _manifest_scale_parameter(
    validated: Mapping[str, object],
) -> BenchmarkScaleParameter | None:
    value = validated.get("scale_parameter")
    if value is None:
        return None
    return BenchmarkScaleParameter.from_record(_as_mapping(value, field="scale_parameter"))


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
