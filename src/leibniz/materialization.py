"""Resolution and materialization records for benchmark observations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "AxisAssignment",
    "LinearResolutionRequirement",
    "MaterializationDeclaration",
    "MaterializationDeclarationDocument",
    "MaterializationPlan",
    "MaterializationPlanDocument",
    "MaterializationValidationError",
]

_axis_assignment_record = RecordSpec(
    fields={
        "values": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
    }
)
_axis_value_record = RecordSpec(
    fields={
        "axis": FieldSpec(kind="string"),
        "value": FieldSpec(kind="integer"),
    }
)
_linear_requirement_record = RecordSpec(
    fields={
        "name": FieldSpec(kind="name"),
        "source_axis": FieldSpec(kind="string"),
        "resolution_axis": FieldSpec(kind="string"),
        "coefficient": FieldSpec(kind="number"),
        "intercept": FieldSpec(kind="number", required=False),
        "minimum": FieldSpec(kind="integer", required=False),
        "basis": FieldSpec(kind="string"),
        "description": FieldSpec(kind="string", required=False),
    }
)
_materialization_declaration_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "benchmark_id": FieldSpec(kind="identifier"),
        "requirements": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "latent_factor_declaration": FieldSpec(kind="record", required=False),
        "layout": FieldSpec(kind="record", required=False),
    }
)
_materialization_plan_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "benchmark_id": FieldSpec(kind="identifier"),
        "materialization_declaration": FieldSpec(kind="record"),
        "scale_assignment": FieldSpec(kind="record"),
        "complexity_assignment": FieldSpec(kind="record"),
        "resolution_assignment": FieldSpec(kind="record"),
        "seed": FieldSpec(kind="integer"),
        "latent_factor_declaration": FieldSpec(kind="record", required=False),
    }
)


class MaterializationValidationError(ValueError):
    """Raised when materialization records are invalid."""


@dataclass(frozen=True, slots=True)
class AxisAssignment:
    """Integer values assigned to named scale, complexity, or resolution axes."""

    values: Mapping[str, int]

    def __post_init__(self) -> None:
        if not self.values:
            raise MaterializationValidationError("axis assignment must not be empty")
        normalized: dict[str, int] = {}
        for axis, value in self.values.items():
            if not axis:
                raise MaterializationValidationError("axis names must be nonempty")
            if type(value) is not int:
                raise MaterializationValidationError(f"{axis}: axis value must be an integer")
            if value < 0:
                raise MaterializationValidationError(
                    f"{axis}: axis value must be nonnegative"
                )
            normalized[str(axis)] = value
        if len(normalized) != len(self.values):
            raise MaterializationValidationError("axis names must be unique")
        object.__setattr__(self, "values", normalized)

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> AxisAssignment:
        try:
            validated = _axis_assignment_record.validate(record)
            items = tuple(
                _axis_value_record.validate(_as_mapping(item, field="values"))
                for item in _as_sequence(validated["values"], field="values")
            )
        except ValueError as error:
            raise MaterializationValidationError(str(error)) from error
        values: dict[str, int] = {}
        for item in items:
            axis = str(item["axis"])
            if axis in values:
                raise MaterializationValidationError(f"duplicate axis assignment: {axis}")
            values[axis] = _as_int(item["value"], field=axis)
        return cls(values=values)

    def require_axis(self, axis: str) -> int:
        try:
            return self.values[axis]
        except KeyError as error:
            raise MaterializationValidationError(
                f"missing axis assignment: {axis}"
            ) from error

    def to_record(self) -> dict[str, object]:
        return {
            "values": [
                {"axis": axis, "value": value}
                for axis, value in sorted(self.values.items())
            ]
        }


@dataclass(frozen=True, slots=True)
class LinearResolutionRequirement:
    """A linear lower bound for a resolution axis."""

    name: ProtocolName
    source_axis: str
    resolution_axis: str
    coefficient: float
    intercept: float = 0.0
    minimum: int | None = None
    basis: str = "analytic-bound"
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.source_axis:
            raise MaterializationValidationError("source_axis must be nonempty")
        if not self.resolution_axis:
            raise MaterializationValidationError("resolution_axis must be nonempty")
        if not math.isfinite(self.coefficient) or self.coefficient < 0:
            raise MaterializationValidationError(
                "coefficient must be finite and nonnegative"
            )
        if not math.isfinite(self.intercept) or self.intercept < 0:
            raise MaterializationValidationError("intercept must be finite and nonnegative")
        if self.minimum is not None:
            if isinstance(self.minimum, bool):
                raise MaterializationValidationError("minimum must be an integer")
            if self.minimum < 1:
                raise MaterializationValidationError("minimum must be positive")
        if self.basis not in {"declared-minimum", "analytic-bound", "certified-search"}:
            raise MaterializationValidationError(f"unsupported requirement basis: {self.basis}")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> LinearResolutionRequirement:
        try:
            validated = _linear_requirement_record.validate(record)
        except ValueError as error:
            raise MaterializationValidationError(str(error)) from error
        return cls(
            name=_as_name(validated["name"], field="name"),
            source_axis=str(validated["source_axis"]),
            resolution_axis=str(validated["resolution_axis"]),
            coefficient=_as_float(validated["coefficient"], field="coefficient"),
            intercept=_optional_float(validated.get("intercept"), field="intercept") or 0.0,
            minimum=_optional_int(validated.get("minimum"), field="minimum"),
            basis=str(validated["basis"]),
            description=_optional_string(validated.get("description"), field="description"),
        )

    def minimum_resolution(self, assignment: AxisAssignment) -> int:
        source_value = assignment.require_axis(self.source_axis)
        minimum = math.ceil(self.intercept + self.coefficient * source_value)
        if self.minimum is not None:
            minimum = max(minimum, self.minimum)
        return max(minimum, 1)

    def require_resolution(
        self,
        *,
        scale_assignment: AxisAssignment,
        resolution_assignment: AxisAssignment,
    ) -> None:
        actual = resolution_assignment.require_axis(self.resolution_axis)
        minimum = self.minimum_resolution(scale_assignment)
        if actual < minimum:
            raise MaterializationValidationError(
                f"{self.resolution_axis}={actual} is below minimum {minimum} "
                f"for {self.source_axis}={scale_assignment.require_axis(self.source_axis)}"
            )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "name": str(self.name),
            "source_axis": self.source_axis,
            "resolution_axis": self.resolution_axis,
            "coefficient": self.coefficient,
            "basis": self.basis,
        }
        if self.intercept != 0:
            record["intercept"] = self.intercept
        if self.minimum is not None:
            record["minimum"] = self.minimum
        if self.description is not None:
            record["description"] = self.description
        return record


@dataclass(frozen=True, slots=True)
class MaterializationDeclaration:
    """Reusable requirements for resolving benchmark materialization axes."""

    id: ProtocolIdentifier
    benchmark_id: ProtocolIdentifier
    requirements: tuple[LinearResolutionRequirement, ...]
    latent_factor_declaration: ArtifactReference | None = None
    layout: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
            self.benchmark_id.require_unreleased()
        except ValueError as error:
            raise MaterializationValidationError(str(error)) from error
        if not self.requirements:
            raise MaterializationValidationError("requirements must not be empty")
        duplicate = _first_duplicate(tuple(requirement.name for requirement in self.requirements))
        if duplicate is not None:
            raise MaterializationValidationError(f"duplicate requirement: {duplicate}")
        if (
            self.latent_factor_declaration is not None
            and self.latent_factor_declaration.kind != "latent-factor-declaration"
        ):
            raise MaterializationValidationError(
                "latent_factor_declaration reference must have kind "
                "latent-factor-declaration"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> MaterializationDeclaration:
        try:
            validated = _materialization_declaration_record.validate(record)
        except ValueError as error:
            raise MaterializationValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            benchmark_id=_as_identifier(validated["benchmark_id"], field="benchmark_id"),
            requirements=tuple(
                LinearResolutionRequirement.from_record(
                    _as_mapping(requirement, field="requirements")
                )
                for requirement in _as_sequence(
                    validated["requirements"],
                    field="requirements",
                )
            ),
            latent_factor_declaration=_optional_reference(
                validated.get("latent_factor_declaration"),
                field="latent_factor_declaration",
            ),
            layout=_optional_mapping(validated.get("layout"), field="layout"),
        )

    def minimum_resolution(self, scale_assignment: AxisAssignment) -> AxisAssignment:
        values: dict[str, int] = {}
        for requirement in self.requirements:
            minimum = requirement.minimum_resolution(scale_assignment)
            current = values.get(requirement.resolution_axis)
            if current is None or minimum > current:
                values[requirement.resolution_axis] = minimum
        return AxisAssignment(values=values)

    def require_resolution(
        self,
        *,
        scale_assignment: AxisAssignment,
        resolution_assignment: AxisAssignment,
    ) -> None:
        for requirement in self.requirements:
            requirement.require_resolution(
                scale_assignment=scale_assignment,
                resolution_assignment=resolution_assignment,
            )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "benchmark_id": str(self.benchmark_id),
            "requirements": [
                requirement.to_record() for requirement in self.requirements
            ],
        }
        if self.latent_factor_declaration is not None:
            record["latent_factor_declaration"] = self.latent_factor_declaration.to_record()
        if self.layout is not None:
            record["layout"] = _canonical_mapping(self.layout)
        return record


@dataclass(frozen=True, slots=True)
class MaterializationDeclarationDocument:
    """A loaded materialization declaration and its canonical digest."""

    declaration: MaterializationDeclaration
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> MaterializationDeclarationDocument:
        try:
            record = load_object_document(data, description="materialization declaration")
        except ContentEncodingError as error:
            raise MaterializationValidationError(str(error)) from error
        declaration = MaterializationDeclaration.from_record(record)
        return cls(declaration=declaration, digest=declaration.digest)


@dataclass(frozen=True, slots=True)
class MaterializationPlan:
    """A deterministic resolved materialization request."""

    id: ProtocolIdentifier
    benchmark_id: ProtocolIdentifier
    materialization_declaration: ArtifactReference
    scale_assignment: AxisAssignment
    complexity_assignment: AxisAssignment
    resolution_assignment: AxisAssignment
    seed: int
    latent_factor_declaration: ArtifactReference | None = None

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
            self.benchmark_id.require_unreleased()
        except ValueError as error:
            raise MaterializationValidationError(str(error)) from error
        if self.materialization_declaration.kind != "materialization-declaration":
            raise MaterializationValidationError(
                "materialization_declaration reference must have kind "
                "materialization-declaration"
            )
        if (
            self.latent_factor_declaration is not None
            and self.latent_factor_declaration.kind != "latent-factor-declaration"
        ):
            raise MaterializationValidationError(
                "latent_factor_declaration reference must have kind "
                "latent-factor-declaration"
            )
        if type(self.seed) is not int:
            raise MaterializationValidationError("seed must be an integer")
        if self.seed < 0:
            raise MaterializationValidationError("seed must be nonnegative")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> MaterializationPlan:
        try:
            validated = _materialization_plan_record.validate(record)
        except ValueError as error:
            raise MaterializationValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            benchmark_id=_as_identifier(validated["benchmark_id"], field="benchmark_id"),
            materialization_declaration=_reference(
                validated["materialization_declaration"],
                field="materialization_declaration",
            ),
            scale_assignment=AxisAssignment.from_record(
                _as_mapping(validated["scale_assignment"], field="scale_assignment")
            ),
            complexity_assignment=AxisAssignment.from_record(
                _as_mapping(validated["complexity_assignment"], field="complexity_assignment")
            ),
            resolution_assignment=AxisAssignment.from_record(
                _as_mapping(validated["resolution_assignment"], field="resolution_assignment")
            ),
            seed=_as_int(validated["seed"], field="seed"),
            latent_factor_declaration=_optional_reference(
                validated.get("latent_factor_declaration"),
                field="latent_factor_declaration",
            ),
        )

    @classmethod
    def resolve(
        cls,
        *,
        id: ProtocolIdentifier,
        declaration: MaterializationDeclaration,
        scale_assignment: AxisAssignment,
        complexity_assignment: AxisAssignment,
        seed: int,
    ) -> MaterializationPlan:
        declaration_reference = ArtifactReference(
            kind="materialization-declaration",
            protocol_id=declaration.id,
            record_digest=declaration.digest,
        )
        return cls(
            id=id,
            benchmark_id=declaration.benchmark_id,
            materialization_declaration=declaration_reference,
            scale_assignment=scale_assignment,
            complexity_assignment=complexity_assignment,
            resolution_assignment=declaration.minimum_resolution(scale_assignment),
            seed=seed,
            latent_factor_declaration=declaration.latent_factor_declaration,
        )

    def validate_declaration(self, declaration: MaterializationDeclaration) -> None:
        if self.benchmark_id != declaration.benchmark_id:
            raise MaterializationValidationError(
                f"benchmark_id {self.benchmark_id} does not match declaration "
                f"{declaration.benchmark_id}"
            )
        if not self.materialization_declaration.matches_record(declaration.to_record()):
            raise MaterializationValidationError(
                "materialization_declaration reference does not match declaration"
            )
        declaration.require_resolution(
            scale_assignment=self.scale_assignment,
            resolution_assignment=self.resolution_assignment,
        )
        if self.latent_factor_declaration != declaration.latent_factor_declaration:
            raise MaterializationValidationError(
                "latent_factor_declaration reference does not match declaration"
            )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "benchmark_id": str(self.benchmark_id),
            "materialization_declaration": self.materialization_declaration.to_record(),
            "scale_assignment": self.scale_assignment.to_record(),
            "complexity_assignment": self.complexity_assignment.to_record(),
            "resolution_assignment": self.resolution_assignment.to_record(),
            "seed": self.seed,
        }
        if self.latent_factor_declaration is not None:
            record["latent_factor_declaration"] = self.latent_factor_declaration.to_record()
        return record


@dataclass(frozen=True, slots=True)
class MaterializationPlanDocument:
    """A loaded materialization plan and its canonical digest."""

    plan: MaterializationPlan
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> MaterializationPlanDocument:
        try:
            record = load_object_document(data, description="materialization plan")
        except ContentEncodingError as error:
            raise MaterializationValidationError(str(error)) from error
        plan = MaterializationPlan.from_record(record)
        return cls(plan=plan, digest=plan.digest)


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise MaterializationValidationError(f"{field}: expected parsed identifier")
    return value


def _as_name(value: object, *, field: str) -> ProtocolName:
    if not isinstance(value, ProtocolName):
        raise MaterializationValidationError(f"{field}: expected parsed name")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MaterializationValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _optional_mapping(value: object, *, field: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return _as_mapping(value, field=field)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise MaterializationValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise MaterializationValidationError(f"{field}: expected integer")
    return value


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, field=field)


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, float):
        return value
    raise MaterializationValidationError(f"{field}: expected number")


def _optional_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _as_float(value, field=field)


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise MaterializationValidationError(f"{field}: expected string")
    return value


def _reference(value: object, *, field: str) -> ArtifactReference:
    try:
        return ArtifactReference.from_record(_as_mapping(value, field=field))
    except ValueError as error:
        raise MaterializationValidationError(str(error)) from error


def _optional_reference(value: object, *, field: str) -> ArtifactReference | None:
    if value is None:
        return None
    return _reference(value, field=field)


def _canonical_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): value[key] for key in sorted(value)}


def _first_duplicate(values: tuple[object, ...]) -> object | None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
