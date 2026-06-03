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
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec

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
        "source_assignment": FieldSpec(kind="record", required=False),
        "resolution_assignment": FieldSpec(kind="record"),
        "seed": FieldSpec(kind="integer"),
        "latent_factor_declaration": FieldSpec(kind="record", required=False),
    }
)


class MaterializationValidationError(ValueError):
    """Raised when materialization records are invalid."""


_extract = RecordExtractor(error_type=MaterializationValidationError)


@dataclass(frozen=True, slots=True)
class AxisAssignment:
    """Integer values assigned to named complexity or resolution axes."""

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
                _axis_value_record.validate(_extract.mapping(item, "values"))
                for item in _extract.sequence(validated["values"], "values")
            )
        except ValueError as error:
            raise MaterializationValidationError(str(error)) from error
        values: dict[str, int] = {}
        for item in items:
            axis = str(item["axis"])
            if axis in values:
                raise MaterializationValidationError(f"duplicate axis assignment: {axis}")
            values[axis] = _extract.integer(item["value"], axis)
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
            coefficient=_extract.float(validated["coefficient"], "coefficient"),
            intercept=_extract.optional_float(validated.get("intercept"), "intercept") or 0.0,
            minimum=_extract.optional_integer(validated.get("minimum"), "minimum"),
            basis=str(validated["basis"]),
            description=_extract.optional_string(validated.get("description"), "description"),
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
        source_assignment: AxisAssignment,
        resolution_assignment: AxisAssignment,
    ) -> None:
        actual = resolution_assignment.require_axis(self.resolution_axis)
        minimum = self.minimum_resolution(source_assignment)
        if actual < minimum:
            raise MaterializationValidationError(
                f"{self.resolution_axis}={actual} is below minimum {minimum} "
                f"for {self.source_axis}={source_assignment.require_axis(self.source_axis)}"
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
            id=_extract.identifier(validated["id"], "id"),
            benchmark_id=_extract.identifier(validated["benchmark_id"], "benchmark_id"),
            requirements=tuple(
                LinearResolutionRequirement.from_record(
                    _extract.mapping(requirement, "requirements")
                )
                for requirement in _extract.sequence(
                    validated["requirements"],
                    "requirements",
                )
            ),
            latent_factor_declaration=_optional_reference(
                validated.get("latent_factor_declaration"),
                field="latent_factor_declaration",
            ),
            layout=_extract.optional_mapping(validated.get("layout"), "layout"),
        )

    def minimum_resolution(
        self,
        source_assignment: AxisAssignment | None = None,
    ) -> AxisAssignment:
        values: dict[str, int] = (
            _layout_resolution_floor(self.layout) if self.layout is not None else {}
        )
        for requirement in self.requirements:
            if source_assignment is None:
                raise MaterializationValidationError(
                    "source_assignment is required by materialization requirements"
                )
            minimum = requirement.minimum_resolution(source_assignment)
            current = values.get(requirement.resolution_axis)
            if current is None or minimum > current:
                values[requirement.resolution_axis] = minimum
        return AxisAssignment(values=values)

    def resolution_lattice_steps(self) -> dict[str, int]:
        """Return declared per-axis resolution lattice steps."""

        return _layout_resolution_lattice(self.layout) if self.layout is not None else {}

    def require_resolution(
        self,
        *,
        source_assignment: AxisAssignment | None = None,
        resolution_assignment: AxisAssignment,
    ) -> None:
        if self.layout is not None:
            for axis, minimum in _layout_resolution_floor(self.layout).items():
                actual = resolution_assignment.require_axis(axis)
                if actual < minimum:
                    raise MaterializationValidationError(
                        f"{axis}={actual} is below layout minimum {minimum}"
                    )
            for axis, step in _layout_resolution_lattice(self.layout).items():
                actual = resolution_assignment.require_axis(axis)
                if actual % step != 0:
                    raise MaterializationValidationError(
                        f"{axis}={actual} is not an integer multiple of "
                        f"layout lattice step {step}"
                    )
        for requirement in self.requirements:
            if source_assignment is None:
                raise MaterializationValidationError(
                    "source_assignment is required by materialization requirements"
                )
            requirement.require_resolution(
                source_assignment=source_assignment,
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
    resolution_assignment: AxisAssignment
    seed: int
    source_assignment: AxisAssignment | None = None
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
            id=_extract.identifier(validated["id"], "id"),
            benchmark_id=_extract.identifier(validated["benchmark_id"], "benchmark_id"),
            materialization_declaration=_reference(
                validated["materialization_declaration"],
                field="materialization_declaration",
            ),
            resolution_assignment=AxisAssignment.from_record(
                _extract.mapping(validated["resolution_assignment"], "resolution_assignment")
            ),
            seed=_extract.integer(validated["seed"], "seed"),
            source_assignment=_optional_axis_assignment(
                validated.get("source_assignment"),
                field="source_assignment",
            ),
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
        seed: int,
        source_assignment: AxisAssignment | None = None,
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
            resolution_assignment=declaration.minimum_resolution(source_assignment),
            seed=seed,
            source_assignment=source_assignment,
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
            source_assignment=self.source_assignment,
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
            "resolution_assignment": self.resolution_assignment.to_record(),
            "seed": self.seed,
        }
        if self.source_assignment is not None:
            record["source_assignment"] = self.source_assignment.to_record()
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
def _as_name(value: object, *, field: str) -> ProtocolName:
    if not isinstance(value, ProtocolName):
        raise MaterializationValidationError(f"{field}: expected parsed name")
    return value
def _reference(value: object, *, field: str) -> ArtifactReference:
    try:
        return ArtifactReference.from_record(_extract.mapping(value, field))
    except ValueError as error:
        raise MaterializationValidationError(str(error)) from error


def _optional_reference(value: object, *, field: str) -> ArtifactReference | None:
    if value is None:
        return None
    return _reference(value, field=field)


def _optional_axis_assignment(value: object, *, field: str) -> AxisAssignment | None:
    if value is None:
        return None
    return AxisAssignment.from_record(_extract.mapping(value, field))


def _canonical_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {str(key): value[key] for key in sorted(value)}


def _layout_resolution_floor(layout: Mapping[str, object]) -> dict[str, int]:
    if layout.get("kind") != "sequence-layout":
        return {}
    width_axis = layout.get("width_axis")
    height_axis = layout.get("height_axis")
    if not isinstance(width_axis, str) or not width_axis:
        return {}
    if not isinstance(height_axis, str) or not height_axis:
        return {}
    floor = {width_axis: 1, height_axis: 1}
    floor_record = layout.get("resolution_floor")
    if floor_record is None:
        return floor
    if not isinstance(floor_record, Mapping):
        raise MaterializationValidationError("layout resolution_floor must be a record")
    floor_mapping = cast(Mapping[str, object], floor_record)
    for axis in (width_axis, height_axis):
        value = floor_mapping.get(axis)
        if value is None:
            continue
        if not isinstance(value, int) or isinstance(value, bool):
            raise MaterializationValidationError(
                f"layout resolution_floor {axis} must be an integer"
            )
        if value < 1:
            raise MaterializationValidationError(
                f"layout resolution_floor {axis} must be positive"
            )
        floor[axis] = value
    return floor


def _layout_resolution_lattice(layout: Mapping[str, object]) -> dict[str, int]:
    if layout.get("kind") != "sequence-layout":
        return {}
    lattice = layout.get("resolution_lattice")
    if lattice is None:
        return {}
    if not isinstance(lattice, Mapping):
        raise MaterializationValidationError("layout resolution_lattice must be a record")
    lattice_record = cast(Mapping[str, object], lattice)
    kind = lattice_record.get("kind")
    if kind != "axis-multiple":
        raise MaterializationValidationError(
            f"unsupported layout resolution_lattice kind: {kind}"
        )
    steps = lattice_record.get("steps")
    if not isinstance(steps, Mapping):
        raise MaterializationValidationError(
            "layout resolution_lattice steps must be a record"
        )
    step_mapping = cast(Mapping[str, object], steps)
    result: dict[str, int] = {}
    for axis, value in step_mapping.items():
        if not axis:
            raise MaterializationValidationError(
                "layout resolution_lattice axes must be nonempty strings"
            )
        if not isinstance(value, int) or isinstance(value, bool):
            raise MaterializationValidationError(
                f"layout resolution_lattice {axis} step must be an integer"
            )
        if value < 1:
            raise MaterializationValidationError(
                f"layout resolution_lattice {axis} step must be positive"
            )
        result[axis] = value
    if not result:
        raise MaterializationValidationError(
            "layout resolution_lattice steps must not be empty"
        )
    return result


def _first_duplicate(values: tuple[object, ...]) -> object | None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
