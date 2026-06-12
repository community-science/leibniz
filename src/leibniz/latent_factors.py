"""Latent factors for benchmark generators."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec

__all__ = [
    "DegreeMeasure",
    "GeneratorConstructionFactor",
    "LatentFactorDeclaration",
    "LatentFactorDeclarationDocument",
    "LatentFactorRole",
    "LatentFactorValidationError",
    "SampleLatentFactor",
]

LatentFactorRole: TypeAlias = Literal["content", "variation", "materialization"]
_degree_measure_kinds = (
    "constant-count",
    "discrete-choice",
    "scalar",
    "vector-dimension",
)
_latent_factor_roles = ("content", "variation", "materialization")
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
    }
)


class LatentFactorValidationError(ValueError):
    """Raised when a latent-factor declaration is invalid."""


_extract = RecordExtractor(error_type=LatentFactorValidationError)


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
            count=_extract.float(validated["count"], "count"),
            domain_size=_extract.optional_integer(validated.get("domain_size"), "domain_size"),
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
                _extract.mapping(validated["degree_measure"], "degree_measure")
            ),
            description=_extract.optional_string(validated.get("description"), "description"),
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
                _extract.mapping(validated["degree_measure"], "degree_measure")
            ),
            multiplicity=_extract.optional_integer(validated.get("multiplicity"), "multiplicity")
            or 1,
            description=_extract.optional_string(validated.get("description"), "description"),
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
class LatentFactorDeclaration:
    """A generator-owned declaration of latent factors."""

    id: ProtocolIdentifier
    construction_factors: tuple[GeneratorConstructionFactor, ...]
    sample_factors: tuple[SampleLatentFactor, ...]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise LatentFactorValidationError(str(error)) from error
        _require_nonempty(self.construction_factors, "construction_factors")
        _require_nonempty(self.sample_factors, "sample_factors")
        _reject_duplicate_names(
            (factor.name for factor in self.construction_factors),
            description="construction factor",
        )
        _reject_duplicate_names(
            (factor.name for factor in self.sample_factors),
            description="sample factor",
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> LatentFactorDeclaration:
        try:
            validated = _latent_factor_declaration_record.validate(record)
        except ValueError as error:
            raise LatentFactorValidationError(str(error)) from error
        return cls(
            id=_extract.identifier(validated["id"], "id"),
            construction_factors=tuple(
                GeneratorConstructionFactor.from_record(
                    _extract.mapping(factor, "construction_factors")
                )
                for factor in _extract.sequence(
                    validated["construction_factors"],
                    "construction_factors",
                )
            ),
            sample_factors=tuple(
                SampleLatentFactor.from_record(_extract.mapping(factor, "sample_factors"))
                for factor in _extract.sequence(validated["sample_factors"], "sample_factors")
            ),
        )

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
        }
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


def _as_name(value: object, *, field: str) -> ProtocolName:
    if not isinstance(value, ProtocolName):
        raise LatentFactorValidationError(f"{field}: expected parsed name")
    return value


def _as_role(value: object) -> LatentFactorRole:
    if not isinstance(value, str):
        raise LatentFactorValidationError("role: expected string")
    if value not in _latent_factor_roles:
        raise LatentFactorValidationError(f"unsupported latent factor role: {value}")
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
