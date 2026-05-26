"""Manifest-driven formation of benchmark observation fields."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import MaterializationPlan
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "ComponentMark",
    "FieldObservation",
    "FormedObservation",
    "ObservationComponent",
    "ObservationFormationDeclaration",
    "ObservationFormationDeclarationDocument",
    "ObservationFormationValidationError",
    "SlotComposition",
]

_interpreter = "field-mark-composition@0.1.0"
_curve_samples_per_degree = 12

_mark_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="string"),
        "channel": FieldSpec(kind="integer"),
        "degree": FieldSpec(kind="integer"),
        "control_points": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="sequence", item=FieldSpec(kind="number")),
        ),
        "width": FieldSpec(kind="number"),
        "value": FieldSpec(kind="number", required=False),
    }
)
_component_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="string"),
        "marks": FieldSpec(kind="sequence", item=FieldSpec(kind="record")),
    }
)
_slot_composition_record = RecordSpec(
    fields={
        "count_axis": FieldSpec(kind="string"),
        "resolution_axis": FieldSpec(kind="string"),
        "slot_axis": FieldSpec(kind="string"),
    }
)
_output_field_record = RecordSpec(
    fields={
        "channel_count": FieldSpec(kind="integer"),
        "resolution_axis": FieldSpec(kind="string"),
    }
)
_observation_formation_declaration_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "benchmark_id": FieldSpec(kind="identifier"),
        "interpreter": FieldSpec(kind="string"),
        "output_field": FieldSpec(kind="record"),
        "slot_composition": FieldSpec(kind="record"),
        "components": FieldSpec(kind="sequence", item=FieldSpec(kind="record")),
    }
)


class ObservationFormationValidationError(ValueError):
    """Raised when observation formation records are invalid."""


@dataclass(frozen=True, slots=True)
class ComponentMark:
    """A primitive mark in local component coordinates."""

    kind: str
    channel: int
    degree: int
    control_points: tuple[tuple[float, float], ...]
    width: float
    value: float = 1.0

    def __post_init__(self) -> None:
        if self.kind != "bezier-curve":
            raise ObservationFormationValidationError(f"unsupported mark kind: {self.kind}")
        if isinstance(self.channel, bool) or self.channel < 0:
            raise ObservationFormationValidationError("channel must be a nonnegative integer")
        if isinstance(self.degree, bool) or self.degree < 1:
            raise ObservationFormationValidationError("degree must be a positive integer")
        if len(self.control_points) != self.degree + 1:
            raise ObservationFormationValidationError(
                "control_points length must equal degree plus one"
            )
        if not math.isfinite(self.width) or self.width <= 0:
            raise ObservationFormationValidationError("width must be finite and positive")
        if not math.isfinite(self.value):
            raise ObservationFormationValidationError("value must be finite")
        for index, point in enumerate(self.control_points):
            _validate_point(point, f"control_points.{index}")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ComponentMark:
        try:
            validated = _mark_record.validate(record)
        except ValueError as error:
            raise ObservationFormationValidationError(str(error)) from error
        value = _optional_float(validated.get("value"), field="value")
        return cls(
            kind=str(validated["kind"]),
            channel=_as_int(validated["channel"], field="channel"),
            degree=_as_int(validated["degree"], field="degree"),
            control_points=tuple(
                _point(point, field=f"control_points.{index}")
                for index, point in enumerate(
                    _as_sequence(validated["control_points"], field="control_points")
                )
            ),
            width=_as_float(validated["width"], field="width"),
            value=1.0 if value is None else value,
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": self.kind,
            "channel": self.channel,
            "degree": self.degree,
            "control_points": [list(point) for point in self.control_points],
            "width": self.width,
        }
        if self.value != 1.0:
            record["value"] = self.value
        return record


@dataclass(frozen=True, slots=True)
class ObservationComponent:
    """A finite-vocabulary component formed from primitive marks."""

    id: str
    marks: tuple[ComponentMark, ...]

    def __post_init__(self) -> None:
        if not self.id:
            raise ObservationFormationValidationError("component id must be nonempty")
        if not self.marks:
            raise ObservationFormationValidationError("component marks must not be empty")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ObservationComponent:
        try:
            validated = _component_record.validate(record)
        except ValueError as error:
            raise ObservationFormationValidationError(str(error)) from error
        return cls(
            id=str(validated["id"]),
            marks=tuple(
                ComponentMark.from_record(_as_mapping(mark, field="marks"))
                for mark in _as_sequence(validated["marks"], field="marks")
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "marks": [mark.to_record() for mark in self.marks],
        }


@dataclass(frozen=True, slots=True)
class SlotComposition:
    """Repeat components across slots controlled by a scale axis."""

    count_axis: str
    resolution_axis: str
    slot_axis: str

    def __post_init__(self) -> None:
        if not self.count_axis:
            raise ObservationFormationValidationError("count_axis must be nonempty")
        if not self.resolution_axis:
            raise ObservationFormationValidationError("resolution_axis must be nonempty")
        if self.slot_axis not in {"x", "y"}:
            raise ObservationFormationValidationError("slot_axis must be x or y")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SlotComposition:
        try:
            validated = _slot_composition_record.validate(record)
        except ValueError as error:
            raise ObservationFormationValidationError(str(error)) from error
        return cls(
            count_axis=str(validated["count_axis"]),
            resolution_axis=str(validated["resolution_axis"]),
            slot_axis=str(validated["slot_axis"]),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "count_axis": self.count_axis,
            "resolution_axis": self.resolution_axis,
            "slot_axis": self.slot_axis,
        }


@dataclass(frozen=True, slots=True)
class FieldObservation:
    """A formed dense scalar field in channel-first square layout."""

    shape: tuple[int, int, int]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.shape) != 3:
            raise ObservationFormationValidationError("field shape must have rank 3")
        if any(size < 1 for size in self.shape):
            raise ObservationFormationValidationError("field shape sizes must be positive")
        expected = self.shape[0] * self.shape[1] * self.shape[2]
        if len(self.values) != expected:
            raise ObservationFormationValidationError(
                f"field value count {len(self.values)} does not match shape size {expected}"
            )
        for value in self.values:
            if not math.isfinite(value):
                raise ObservationFormationValidationError("field values must be finite")

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class FormedObservation:
    """A formed observation field and the provenance needed to reproduce it."""

    id: ProtocolIdentifier
    benchmark_id: ProtocolIdentifier
    formation_declaration: ArtifactReference
    materialization_plan: ArtifactReference
    component_sequence: tuple[int, ...]
    field: FieldObservation

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
            self.benchmark_id.require_unreleased()
        except ValueError as error:
            raise ObservationFormationValidationError(str(error)) from error
        if self.formation_declaration.kind != "observation-formation-declaration":
            raise ObservationFormationValidationError(
                "formation_declaration reference must have kind "
                "observation-formation-declaration"
            )
        if self.materialization_plan.kind != "materialization-plan":
            raise ObservationFormationValidationError(
                "materialization_plan reference must have kind materialization-plan"
            )
        if not self.component_sequence:
            raise ObservationFormationValidationError("component_sequence must not be empty")
        if any(index < 0 for index in self.component_sequence):
            raise ObservationFormationValidationError(
                "component_sequence indexes must be nonnegative"
            )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "benchmark_id": str(self.benchmark_id),
            "formation_declaration": self.formation_declaration.to_record(),
            "materialization_plan": self.materialization_plan.to_record(),
            "component_sequence": list(self.component_sequence),
            "field_shape": list(self.field.shape),
            "field_digest": str(self.field.digest),
        }


@dataclass(frozen=True, slots=True)
class ObservationFormationDeclaration:
    """A manifest-driven interpreter declaration for forming observations."""

    id: ProtocolIdentifier
    benchmark_id: ProtocolIdentifier
    interpreter: str
    channel_count: int
    resolution_axis: str
    slot_composition: SlotComposition
    components: tuple[ObservationComponent, ...]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
            self.benchmark_id.require_unreleased()
        except ValueError as error:
            raise ObservationFormationValidationError(str(error)) from error
        if self.interpreter != _interpreter:
            raise ObservationFormationValidationError(
                f"unsupported observation formation interpreter: {self.interpreter}"
            )
        if isinstance(self.channel_count, bool) or self.channel_count < 1:
            raise ObservationFormationValidationError("channel_count must be positive")
        if not self.resolution_axis:
            raise ObservationFormationValidationError("resolution_axis must be nonempty")
        if self.slot_composition.resolution_axis != self.resolution_axis:
            raise ObservationFormationValidationError(
                "slot composition resolution_axis must match output field resolution_axis"
            )
        if not self.components:
            raise ObservationFormationValidationError("components must not be empty")
        duplicate = _first_duplicate(tuple(component.id for component in self.components))
        if duplicate is not None:
            raise ObservationFormationValidationError(f"duplicate component id: {duplicate}")
        for component in self.components:
            for mark in component.marks:
                if mark.channel >= self.channel_count:
                    raise ObservationFormationValidationError(
                        f"mark channel {mark.channel} is outside channel_count "
                        f"{self.channel_count}"
                    )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ObservationFormationDeclaration:
        try:
            validated = _observation_formation_declaration_record.validate(record)
            output_field = _as_mapping(validated["output_field"], field="output_field")
            output = _output_field_record.validate(output_field)
        except ValueError as error:
            raise ObservationFormationValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            benchmark_id=_as_identifier(validated["benchmark_id"], field="benchmark_id"),
            interpreter=str(validated["interpreter"]),
            channel_count=_as_int(output["channel_count"], field="channel_count"),
            resolution_axis=str(output["resolution_axis"]),
            slot_composition=SlotComposition.from_record(
                _as_mapping(validated["slot_composition"], field="slot_composition")
            ),
            components=tuple(
                ObservationComponent.from_record(_as_mapping(component, field="components"))
                for component in _as_sequence(validated["components"], field="components")
            ),
        )

    def sample_component_sequence(
        self,
        *,
        plan: MaterializationPlan,
        sample_index: int,
    ) -> tuple[int, ...]:
        if sample_index < 0:
            raise ObservationFormationValidationError("sample_index must be nonnegative")
        slot_count = plan.scale_assignment.require_axis(self.slot_composition.count_axis)
        if slot_count < 1:
            raise ObservationFormationValidationError("slot count must be positive")
        generator = random.Random(plan.seed + sample_index)
        return tuple(generator.randrange(len(self.components)) for _slot in range(slot_count))

    def form_observation(
        self,
        *,
        id: ProtocolIdentifier,
        plan: MaterializationPlan,
        component_sequence: Sequence[int],
    ) -> FormedObservation:
        if plan.benchmark_id != self.benchmark_id:
            raise ObservationFormationValidationError(
                f"plan benchmark_id {plan.benchmark_id} does not match {self.benchmark_id}"
            )
        slots = plan.scale_assignment.require_axis(self.slot_composition.count_axis)
        resolution = plan.resolution_assignment.require_axis(self.resolution_axis)
        sequence = tuple(_as_int(index, field="component_sequence") for index in component_sequence)
        if len(sequence) != slots:
            raise ObservationFormationValidationError(
                f"component_sequence length {len(sequence)} does not match slot count {slots}"
            )
        if any(index >= len(self.components) for index in sequence):
            raise ObservationFormationValidationError(
                "component_sequence index is outside component vocabulary"
            )
        field = self._form_field(sequence=sequence, resolution=resolution)
        return FormedObservation(
            id=id,
            benchmark_id=self.benchmark_id,
            formation_declaration=ArtifactReference(
                kind="observation-formation-declaration",
                protocol_id=self.id,
                record_digest=self.digest,
            ),
            materialization_plan=ArtifactReference(
                kind="materialization-plan",
                protocol_id=plan.id,
                record_digest=plan.digest,
            ),
            component_sequence=sequence,
            field=field,
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "benchmark_id": str(self.benchmark_id),
            "interpreter": self.interpreter,
            "output_field": {
                "channel_count": self.channel_count,
                "resolution_axis": self.resolution_axis,
            },
            "slot_composition": self.slot_composition.to_record(),
            "components": [component.to_record() for component in self.components],
        }

    def _form_field(self, *, sequence: tuple[int, ...], resolution: int) -> FieldObservation:
        if resolution < 1:
            raise ObservationFormationValidationError("resolution must be positive")
        values = [0.0] * (self.channel_count * resolution * resolution)
        for slot_index, component_index in enumerate(sequence):
            component = self.components[component_index]
            for mark in component.marks:
                _draw_mark(
                    values=values,
                    channel_count=self.channel_count,
                    resolution=resolution,
                    slot_count=len(sequence),
                    slot_index=slot_index,
                    slot_axis=self.slot_composition.slot_axis,
                    mark=mark,
                )
        return FieldObservation(
            shape=(self.channel_count, resolution, resolution),
            values=tuple(values),
        )


@dataclass(frozen=True, slots=True)
class ObservationFormationDeclarationDocument:
    """A loaded observation formation declaration and its canonical digest."""

    declaration: ObservationFormationDeclaration
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> ObservationFormationDeclarationDocument:
        try:
            record = load_object_document(data, description="observation formation declaration")
        except ContentEncodingError as error:
            raise ObservationFormationValidationError(str(error)) from error
        declaration = ObservationFormationDeclaration.from_record(record)
        return cls(declaration=declaration, digest=declaration.digest)


def _draw_mark(
    *,
    values: list[float],
    channel_count: int,
    resolution: int,
    slot_count: int,
    slot_index: int,
    slot_axis: str,
    mark: ComponentMark,
) -> None:
    curve_points = tuple(
        _slot_point(point, slot_count=slot_count, slot_index=slot_index, axis=slot_axis)
        for point in _sample_bezier_curve(mark.control_points)
    )
    threshold = mark.width / (2.0 * resolution)
    x_range = _pixel_range(
        min(point[0] for point in curve_points) - threshold,
        max(point[0] for point in curve_points) + threshold,
        resolution=resolution,
    )
    y_range = _pixel_range(
        min(point[1] for point in curve_points) - threshold,
        max(point[1] for point in curve_points) + threshold,
        resolution=resolution,
    )
    for y_index in y_range:
        y = (y_index + 0.5) / resolution
        for x_index in x_range:
            x = (x_index + 0.5) / resolution
            if _polyline_distance((x, y), curve_points) <= threshold:
                value_index = (
                    mark.channel * resolution * resolution
                    + y_index * resolution
                    + x_index
                )
                values[value_index] = max(values[value_index], mark.value)


def _pixel_range(lower: float, upper: float, *, resolution: int) -> range:
    start = max(0, math.floor(lower * resolution))
    stop = min(resolution, math.ceil(upper * resolution))
    return range(start, stop)


def _sample_bezier_curve(
    control_points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    sample_count = max(1, (len(control_points) - 1) * _curve_samples_per_degree)
    return tuple(
        _bezier_point(control_points, index / sample_count)
        for index in range(sample_count + 1)
    )


def _bezier_point(
    control_points: tuple[tuple[float, float], ...],
    t: float,
) -> tuple[float, float]:
    working: list[tuple[float, float]] = list(control_points)
    while len(working) > 1:
        working = [
            (
                (1.0 - t) * left[0] + t * right[0],
                (1.0 - t) * left[1] + t * right[1],
            )
            for left, right in zip(working, working[1:], strict=False)
        ]
    return working[0]


def _slot_point(
    point: tuple[float, float],
    *,
    slot_count: int,
    slot_index: int,
    axis: str,
) -> tuple[float, float]:
    x, y = point
    if axis == "x":
        return ((slot_index + x) / slot_count, y)
    return (x, (slot_index + y) / slot_count)


def _polyline_distance(
    point: tuple[float, float],
    points: tuple[tuple[float, float], ...],
) -> float:
    return min(
        _segment_distance(point, start, end)
        for start, end in zip(points, points[1:], strict=False)
    )


def _segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length_squared = dx * dx + dy * dy
    if length_squared == 0:
        return math.hypot(px - sx, py - sy)
    t = ((px - sx) * dx + (py - sy) * dy) / length_squared
    t = min(1.0, max(0.0, t))
    closest_x = sx + t * dx
    closest_y = sy + t * dy
    return math.hypot(px - closest_x, py - closest_y)


def _validate_point(point: tuple[float, float], description: str) -> None:
    for value in point:
        if not math.isfinite(value):
            raise ObservationFormationValidationError(f"{description} values must be finite")
        if value < 0 or value > 1:
            raise ObservationFormationValidationError(
                f"{description} values must lie between 0 and 1"
            )


def _point(value: object, *, field: str) -> tuple[float, float]:
    sequence = _as_sequence(value, field=field)
    if len(sequence) != 2:
        raise ObservationFormationValidationError(f"{field}: expected two coordinates")
    return (
        _as_float(sequence[0], field=f"{field}.0"),
        _as_float(sequence[1], field=f"{field}.1"),
    )


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise ObservationFormationValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ObservationFormationValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ObservationFormationValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise ObservationFormationValidationError(f"{field}: expected integer")
    return value


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, float):
        return value
    raise ObservationFormationValidationError(f"{field}: expected number")


def _optional_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    return _as_float(value, field=field)


def _first_duplicate(values: tuple[object, ...]) -> object | None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
