"""Latent factors and complexity projections for benchmark generators."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "ComplexityProjection",
    "DegreeMeasure",
    "GeneratorConstructionFactor",
    "LatentFactorDeclaration",
    "LatentFactorDeclarationDocument",
    "LatentFactorRole",
    "LatentFactorValidationError",
    "ResolutionRequirement",
    "SampleLatentFactor",
]

LatentFactorRole: TypeAlias = Literal["content", "nuisance", "materialization"]
_degree_measure_kinds = (
    "constant-count",
    "discrete-choice",
    "scalar",
    "vector-dimension",
)
_latent_factor_roles = ("content", "nuisance", "materialization")
_resolution_bases = ("declared-minimum", "analytic-bound", "certified-search")

_degree_measure_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="string"),
        "count": FieldSpec(kind="number"),
        "domain_size": FieldSpec(kind="integer", required=False),
    }
)
_construction_factor_record = RecordSpec(
    fields={
        "name": FieldSpec(kind="name"),
        "degree_measure": FieldSpec(kind="record"),
        "description": FieldSpec(kind="string", required=False),
    }
)
_sample_factor_record = RecordSpec(
    fields={
        "name": FieldSpec(kind="name"),
        "role": FieldSpec(kind="string"),
        "degree_measure": FieldSpec(kind="record"),
        "multiplicity": FieldSpec(kind="integer", required=False),
        "description": FieldSpec(kind="string", required=False),
    }
)
_complexity_projection_record = RecordSpec(
    fields={
        "name": FieldSpec(kind="name"),
        "coordinate": FieldSpec(kind="string"),
        "included_roles": FieldSpec(kind="sequence", item=FieldSpec(kind="string")),
        "description": FieldSpec(kind="string", required=False),
    }
)
_resolution_requirement_record = RecordSpec(
    fields={
        "name": FieldSpec(kind="name"),
        "resolution_axis": FieldSpec(kind="string"),
        "content_coordinate": FieldSpec(kind="string"),
        "content_complexity": FieldSpec(kind="number"),
        "minimum_resolution": FieldSpec(kind="integer"),
        "basis": FieldSpec(kind="string"),
        "description": FieldSpec(kind="string", required=False),
    }
)
_latent_factor_declaration_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "construction_factors": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "sample_factors": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "complexity_projections": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "resolution_requirements": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
            required=False,
        ),
    }
)


class LatentFactorValidationError(ValueError):
    """Raised when a latent-factor declaration is invalid."""


@dataclass(frozen=True, slots=True)
class DegreeMeasure:
    """The dimensional or counted contribution made by one factor."""

    kind: str
    count: float
    domain_size: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in _degree_measure_kinds:
            raise LatentFactorValidationError(f"unsupported degree measure kind: {self.kind}")
        if not math.isfinite(self.count) or self.count < 0:
            raise LatentFactorValidationError("degree count must be finite and nonnegative")
        if self.kind == "discrete-choice":
            if self.domain_size is None:
                raise LatentFactorValidationError(
                    "discrete-choice degree measure requires domain_size"
                )
            if self.count <= 0:
                raise LatentFactorValidationError("discrete-choice count must be positive")
            if self.domain_size < 2:
                raise LatentFactorValidationError("domain_size must be at least 2")
        elif self.kind == "scalar":
            if self.count != 1.0:
                raise LatentFactorValidationError("scalar degree measure count must be 1")
            if self.domain_size is not None:
                raise LatentFactorValidationError(
                    "scalar degree measure must not declare domain_size"
                )
        elif self.kind == "vector-dimension":
            if self.count < 1 or not self.count.is_integer():
                raise LatentFactorValidationError(
                    "vector-dimension count must be a positive integer"
                )
            if self.domain_size is not None:
                raise LatentFactorValidationError(
                    "vector-dimension degree measure must not declare domain_size"
                )
        elif self.domain_size is not None:
            raise LatentFactorValidationError(
                f"{self.kind} degree measure must not declare domain_size"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> DegreeMeasure:
        try:
            validated = _degree_measure_record.validate(record)
        except ValueError as error:
            raise LatentFactorValidationError(str(error)) from error
        return cls(
            kind=str(validated["kind"]),
            count=_as_float(validated["count"], field="count"),
            domain_size=_optional_int(validated.get("domain_size"), field="domain_size"),
        )

    @classmethod
    def constant_count(cls, count: float) -> DegreeMeasure:
        return cls(kind="constant-count", count=float(count))

    @classmethod
    def discrete_choice(cls, domain_size: int) -> DegreeMeasure:
        return cls(kind="discrete-choice", count=1.0, domain_size=domain_size)

    @classmethod
    def scalar(cls) -> DegreeMeasure:
        return cls(kind="scalar", count=1.0)

    @classmethod
    def vector_dimension(cls, dimension: int) -> DegreeMeasure:
        return cls(kind="vector-dimension", count=float(dimension))

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": self.kind,
            "count": self.count,
        }
        if self.domain_size is not None:
            record["domain_size"] = self.domain_size
        return record


@dataclass(frozen=True, slots=True)
class GeneratorConstructionFactor:
    """One fixed component of a generator family."""

    name: ProtocolName
    degree_measure: DegreeMeasure
    description: str | None = None

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
    ) -> GeneratorConstructionFactor:
        try:
            validated = _construction_factor_record.validate(record)
        except ValueError as error:
            raise LatentFactorValidationError(str(error)) from error
        return cls(
            name=_as_name(validated["name"], field="name"),
            degree_measure=DegreeMeasure.from_record(
                _as_mapping(validated["degree_measure"], field="degree_measure")
            ),
            description=_optional_string(validated.get("description"), field="description"),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "name": str(self.name),
            "degree_measure": self.degree_measure.to_record(),
        }
        if self.description is not None:
            record["description"] = self.description
        return record


@dataclass(frozen=True, slots=True)
class SampleLatentFactor:
    """One latent variable or component drawn for a concrete sample."""

    name: ProtocolName
    role: LatentFactorRole
    degree_measure: DegreeMeasure
    multiplicity: int = 1
    description: str | None = None

    def __post_init__(self) -> None:
        if self.role not in _latent_factor_roles:
            raise LatentFactorValidationError(f"unsupported latent factor role: {self.role}")
        if isinstance(self.multiplicity, bool):
            raise LatentFactorValidationError("multiplicity must be an integer")
        if self.multiplicity < 1:
            raise LatentFactorValidationError("multiplicity must be positive")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SampleLatentFactor:
        try:
            validated = _sample_factor_record.validate(record)
        except ValueError as error:
            raise LatentFactorValidationError(str(error)) from error
        return cls(
            name=_as_name(validated["name"], field="name"),
            role=_as_role(validated["role"]),
            degree_measure=DegreeMeasure.from_record(
                _as_mapping(validated["degree_measure"], field="degree_measure")
            ),
            multiplicity=_optional_int(validated.get("multiplicity"), field="multiplicity")
            or 1,
            description=_optional_string(validated.get("description"), field="description"),
        )

    @property
    def contribution(self) -> float:
        return self.degree_measure.count * self.multiplicity

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "name": str(self.name),
            "role": self.role,
            "degree_measure": self.degree_measure.to_record(),
        }
        if self.multiplicity != 1:
            record["multiplicity"] = self.multiplicity
        if self.description is not None:
            record["description"] = self.description
        return record


@dataclass(frozen=True, slots=True)
class ComplexityProjection:
    """A scalar coordinate derived from sample latent factors."""

    name: ProtocolName
    coordinate: str
    included_roles: frozenset[LatentFactorRole]
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.coordinate:
            raise LatentFactorValidationError("coordinate must be nonempty")
        if self.coordinate == "N":
            raise LatentFactorValidationError("N is a resolution axis, not a complexity axis")
        if not self.included_roles:
            raise LatentFactorValidationError("included_roles must be nonempty")
        for role in self.included_roles:
            if role not in _latent_factor_roles:
                raise LatentFactorValidationError(
                    f"unsupported included latent factor role: {role}"
                )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ComplexityProjection:
        try:
            validated = _complexity_projection_record.validate(record)
        except ValueError as error:
            raise LatentFactorValidationError(str(error)) from error
        roles: frozenset[LatentFactorRole] = frozenset(
            _as_role(role) for role in _as_sequence(validated["included_roles"], field="roles")
        )
        return cls(
            name=_as_name(validated["name"], field="name"),
            coordinate=str(validated["coordinate"]),
            included_roles=roles,
            description=_optional_string(validated.get("description"), field="description"),
        )

    def evaluate(self, factors: Iterable[SampleLatentFactor]) -> float:
        return sum(
            factor.contribution for factor in factors if factor.role in self.included_roles
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "name": str(self.name),
            "coordinate": self.coordinate,
            "included_roles": sorted(self.included_roles),
        }
        if self.description is not None:
            record["description"] = self.description
        return record


@dataclass(frozen=True, slots=True)
class ResolutionRequirement:
    """A minimum resolution needed for a content-complexity level."""

    name: ProtocolName
    resolution_axis: str
    content_coordinate: str
    content_complexity: float
    minimum_resolution: int
    basis: str
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.resolution_axis:
            raise LatentFactorValidationError("resolution_axis must be nonempty")
        if not self.content_coordinate:
            raise LatentFactorValidationError("content_coordinate must be nonempty")
        if not math.isfinite(self.content_complexity) or self.content_complexity < 0:
            raise LatentFactorValidationError(
                "content_complexity must be finite and nonnegative"
            )
        if isinstance(self.minimum_resolution, bool):
            raise LatentFactorValidationError("minimum_resolution must be an integer")
        if self.minimum_resolution < 1:
            raise LatentFactorValidationError("minimum_resolution must be positive")
        if self.basis not in _resolution_bases:
            raise LatentFactorValidationError(f"unsupported resolution basis: {self.basis}")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ResolutionRequirement:
        try:
            validated = _resolution_requirement_record.validate(record)
        except ValueError as error:
            raise LatentFactorValidationError(str(error)) from error
        return cls(
            name=_as_name(validated["name"], field="name"),
            resolution_axis=str(validated["resolution_axis"]),
            content_coordinate=str(validated["content_coordinate"]),
            content_complexity=_as_float(
                validated["content_complexity"],
                field="content_complexity",
            ),
            minimum_resolution=_as_int(
                validated["minimum_resolution"],
                field="minimum_resolution",
            ),
            basis=str(validated["basis"]),
            description=_optional_string(validated.get("description"), field="description"),
        )

    def require_resolution(self, resolution: int) -> None:
        if resolution < self.minimum_resolution:
            raise LatentFactorValidationError(
                f"{self.resolution_axis}={resolution} is below minimum "
                f"{self.minimum_resolution} for {self.content_coordinate}="
                f"{self.content_complexity:g}"
            )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "name": str(self.name),
            "resolution_axis": self.resolution_axis,
            "content_coordinate": self.content_coordinate,
            "content_complexity": self.content_complexity,
            "minimum_resolution": self.minimum_resolution,
            "basis": self.basis,
        }
        if self.description is not None:
            record["description"] = self.description
        return record


@dataclass(frozen=True, slots=True)
class LatentFactorDeclaration:
    """A generator-owned declaration of latent factors and complexity views."""

    id: ProtocolIdentifier
    construction_factors: tuple[GeneratorConstructionFactor, ...]
    sample_factors: tuple[SampleLatentFactor, ...]
    complexity_projections: tuple[ComplexityProjection, ...]
    resolution_requirements: tuple[ResolutionRequirement, ...] = ()

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise LatentFactorValidationError(str(error)) from error
        _require_nonempty(self.construction_factors, "construction_factors")
        _require_nonempty(self.sample_factors, "sample_factors")
        _require_nonempty(self.complexity_projections, "complexity_projections")
        _reject_duplicate_names(
            (factor.name for factor in self.construction_factors),
            description="construction factor",
        )
        _reject_duplicate_names(
            (factor.name for factor in self.sample_factors),
            description="sample factor",
        )
        _reject_duplicate_names(
            (projection.name for projection in self.complexity_projections),
            description="complexity projection",
        )
        _reject_duplicate_names(
            (requirement.name for requirement in self.resolution_requirements),
            description="resolution requirement",
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> LatentFactorDeclaration:
        try:
            validated = _latent_factor_declaration_record.validate(record)
        except ValueError as error:
            raise LatentFactorValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            construction_factors=tuple(
                GeneratorConstructionFactor.from_record(
                    _as_mapping(factor, field="construction_factors")
                )
                for factor in _as_sequence(
                    validated["construction_factors"],
                    field="construction_factors",
                )
            ),
            sample_factors=tuple(
                SampleLatentFactor.from_record(_as_mapping(factor, field="sample_factors"))
                for factor in _as_sequence(validated["sample_factors"], field="sample_factors")
            ),
            complexity_projections=tuple(
                ComplexityProjection.from_record(
                    _as_mapping(projection, field="complexity_projections")
                )
                for projection in _as_sequence(
                    validated["complexity_projections"],
                    field="complexity_projections",
                )
            ),
            resolution_requirements=tuple(
                ResolutionRequirement.from_record(
                    _as_mapping(requirement, field="resolution_requirements")
                )
                for requirement in _as_sequence(
                    validated.get("resolution_requirements", ()),
                    field="resolution_requirements",
                )
            ),
        )

    def projection(self, coordinate: str) -> ComplexityProjection:
        matches = tuple(
            projection
            for projection in self.complexity_projections
            if projection.coordinate == coordinate
        )
        if len(matches) != 1:
            raise LatentFactorValidationError(
                f"expected exactly one complexity projection for coordinate {coordinate!r}"
            )
        return matches[0]

    def evaluate_complexity(self, coordinate: str) -> float:
        return self.projection(coordinate).evaluate(self.sample_factors)

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "construction_factors": [
                factor.to_record() for factor in self.construction_factors
            ],
            "sample_factors": [factor.to_record() for factor in self.sample_factors],
            "complexity_projections": [
                projection.to_record() for projection in self.complexity_projections
            ],
        }
        if self.resolution_requirements:
            record["resolution_requirements"] = [
                requirement.to_record() for requirement in self.resolution_requirements
            ]
        return record


@dataclass(frozen=True, slots=True)
class LatentFactorDeclarationDocument:
    """A loaded latent-factor declaration and its canonical digest."""

    declaration: LatentFactorDeclaration
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> LatentFactorDeclarationDocument:
        try:
            record = load_object_document(data, description="latent factor declaration")
        except ContentEncodingError as error:
            raise LatentFactorValidationError(str(error)) from error
        declaration = LatentFactorDeclaration.from_record(record)
        return cls(declaration=declaration, digest=declaration.digest)


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise LatentFactorValidationError(f"{field}: expected parsed identifier")
    return value


def _as_name(value: object, *, field: str) -> ProtocolName:
    if not isinstance(value, ProtocolName):
        raise LatentFactorValidationError(f"{field}: expected parsed name")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LatentFactorValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise LatentFactorValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_role(value: object) -> LatentFactorRole:
    if not isinstance(value, str):
        raise LatentFactorValidationError("role: expected string")
    if value not in _latent_factor_roles:
        raise LatentFactorValidationError(f"unsupported latent factor role: {value}")
    return value


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, float):
        return value
    raise LatentFactorValidationError(f"{field}: expected number")


def _as_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise LatentFactorValidationError(f"{field}: expected integer")
    return value


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, field=field)


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LatentFactorValidationError(f"{field}: expected string")
    return value


def _require_nonempty(values: tuple[object, ...], description: str) -> None:
    if not values:
        raise LatentFactorValidationError(f"{description} must contain at least one item")


def _reject_duplicate_names(names: Iterable[ProtocolName], *, description: str) -> None:
    seen: set[ProtocolName] = set()
    for name in names:
        if name in seen:
            raise LatentFactorValidationError(f"duplicate {description}: {name}")
        seen.add(name)
