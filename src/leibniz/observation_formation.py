"""Manifest-driven formation of benchmark observation fields."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
    "VariationTransformDeclaration",
    "ObservationComponent",
    "ObservationFormationDeclaration",
    "ObservationFormationDeclarationDocument",
    "ObservationFormationValidationError",
    "SequenceLayout",
    "SpatialAffineVariation",
    "ValueScaleVariation",
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
_sequence_layout_record = RecordSpec(
    fields={
        "sequence_axis": FieldSpec(kind="string"),
        "width_axis": FieldSpec(kind="string"),
        "height_axis": FieldSpec(kind="string"),
        "placement_axis": FieldSpec(kind="string"),
    }
)
_output_field_record = RecordSpec(
    fields={
        "channel_count": FieldSpec(kind="integer"),
        "width_axis": FieldSpec(kind="string"),
        "height_axis": FieldSpec(kind="string"),
    }
)
_spatial_affine_variation_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="string"),
        "coordinate_system": FieldSpec(kind="string"),
        "spatial_rank": FieldSpec(kind="integer"),
        "translation": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="sequence", item=FieldSpec(kind="number")),
        ),
        "scale": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="sequence", item=FieldSpec(kind="number")),
        ),
        "rotation_degrees": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="number"),
            required=False,
        ),
        "shear_degrees": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="number"),
            required=False,
        ),
    }
)
_value_scale_variation_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="string"),
        "scale": FieldSpec(kind="sequence", item=FieldSpec(kind="number")),
    }
)
_variation_transform_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="string"),
        "spatial_affine": FieldSpec(kind="record", required=False),
        "value_scale": FieldSpec(kind="record", required=False),
    }
)
_observation_formation_declaration_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "benchmark_id": FieldSpec(kind="identifier"),
        "interpreter": FieldSpec(kind="string"),
        "output_field": FieldSpec(kind="record"),
        "sequence_layout": FieldSpec(kind="record"),
        "variation_transform": FieldSpec(kind="record", required=False),
        "components": FieldSpec(kind="sequence", item=FieldSpec(kind="record")),
    }
)


class ObservationFormationValidationError(ValueError):
    """Raised when observation formation records are invalid."""


@dataclass(frozen=True, slots=True)
class _SpatialAffineCoordinate:
    coordinate_system: str
    translation: tuple[float, float]
    scale: tuple[float, float]
    rotation_degrees: float
    shear_degrees: float


@dataclass(frozen=True, slots=True)
class _VariationCoordinate:
    sequence_index: int
    spatial_affine: _SpatialAffineCoordinate
    value_scale: float


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
class SequenceLayout:
    """Place ordered sequence components along a declared field axis."""

    sequence_axis: str
    placement_axis: str
    width_axis: str
    height_axis: str

    def __post_init__(self) -> None:
        if not self.sequence_axis:
            raise ObservationFormationValidationError("sequence_axis must be nonempty")
        if not self.width_axis:
            raise ObservationFormationValidationError("width_axis must be nonempty")
        if not self.height_axis:
            raise ObservationFormationValidationError("height_axis must be nonempty")
        if self.placement_axis not in {"x", "y"}:
            raise ObservationFormationValidationError("placement_axis must be x or y")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SequenceLayout:
        try:
            validated = _sequence_layout_record.validate(record)
        except ValueError as error:
            raise ObservationFormationValidationError(str(error)) from error
        return cls(
            sequence_axis=str(validated["sequence_axis"]),
            placement_axis=str(validated["placement_axis"]),
            width_axis=str(validated["width_axis"]),
            height_axis=str(validated["height_axis"]),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "sequence_axis": self.sequence_axis,
            "placement_axis": self.placement_axis,
            "width_axis": self.width_axis,
            "height_axis": self.height_axis,
        }


@dataclass(frozen=True, slots=True)
class SpatialAffineVariation:
    """Declared spatial affine variation bounds in normalized field coordinates."""

    kind: str
    coordinate_system: str
    spatial_rank: int
    translation: tuple[tuple[float, float], ...]
    scale: tuple[tuple[float, float], ...]
    rotation_degrees: tuple[float, ...] = ()
    shear_degrees: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.kind != "spatial-affine":
            raise ObservationFormationValidationError(
                f"unsupported spatial affine variation kind: {self.kind}"
            )
        if self.coordinate_system != "normalized-sequence-element":
            raise ObservationFormationValidationError(
                "spatial affine coordinate_system must be normalized-sequence-element"
            )
        if type(self.spatial_rank) is not int or self.spatial_rank < 1:
            raise ObservationFormationValidationError("spatial_rank must be positive")
        if len(self.translation) != self.spatial_rank:
            raise ObservationFormationValidationError(
                "translation bounds length must equal spatial_rank"
            )
        if len(self.scale) != self.spatial_rank:
            raise ObservationFormationValidationError("scale bounds length must equal spatial_rank")
        for index, bounds in enumerate(self.translation):
            _validate_interval(bounds, field=f"translation.{index}")
        for index, bounds in enumerate(self.scale):
            _validate_interval(bounds, field=f"scale.{index}", positive=True)
        plane_count = self.spatial_rank * (self.spatial_rank - 1) // 2
        if len(self.rotation_degrees) != plane_count:
            raise ObservationFormationValidationError(
                "rotation_degrees length must match spatial coordinate plane count"
            )
        if len(self.shear_degrees) != plane_count:
            raise ObservationFormationValidationError(
                "shear_degrees length must match spatial coordinate plane count"
            )
        for index, value in enumerate((*self.rotation_degrees, *self.shear_degrees)):
            if not math.isfinite(value) or value < 0:
                field = (
                    "rotation_degrees" if index < len(self.rotation_degrees) else "shear_degrees"
                )
                raise ObservationFormationValidationError(
                    f"{field} values must be finite nonnegative numbers"
                )

    @classmethod
    def identity(cls, *, spatial_rank: int) -> SpatialAffineVariation:
        return cls(
            kind="spatial-affine",
            coordinate_system="normalized-sequence-element",
            spatial_rank=spatial_rank,
            translation=tuple((0.0, 0.0) for _axis in range(spatial_rank)),
            scale=tuple((1.0, 1.0) for _axis in range(spatial_rank)),
            rotation_degrees=tuple(0.0 for _plane in range(spatial_rank * (spatial_rank - 1) // 2)),
            shear_degrees=tuple(0.0 for _plane in range(spatial_rank * (spatial_rank - 1) // 2)),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SpatialAffineVariation:
        try:
            validated = _spatial_affine_variation_record.validate(record)
        except ValueError as error:
            raise ObservationFormationValidationError(str(error)) from error
        spatial_rank = _as_int(validated["spatial_rank"], field="spatial_rank")
        plane_count = spatial_rank * (spatial_rank - 1) // 2
        return cls(
            kind=str(validated["kind"]),
            coordinate_system=str(validated["coordinate_system"]),
            spatial_rank=spatial_rank,
            translation=_interval_sequence(validated["translation"], field="translation"),
            scale=_interval_sequence(validated["scale"], field="scale"),
            rotation_degrees=_number_sequence(
                validated.get("rotation_degrees", tuple(0.0 for _plane in range(plane_count))),
                field="rotation_degrees",
            ),
            shear_degrees=_number_sequence(
                validated.get("shear_degrees", tuple(0.0 for _plane in range(plane_count))),
                field="shear_degrees",
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "coordinate_system": self.coordinate_system,
            "spatial_rank": self.spatial_rank,
            "translation": [list(bounds) for bounds in self.translation],
            "scale": [list(bounds) for bounds in self.scale],
            "rotation_degrees": list(self.rotation_degrees),
            "shear_degrees": list(self.shear_degrees),
        }


@dataclass(frozen=True, slots=True)
class ValueScaleVariation:
    """Declared scalar intensity variation bounds."""

    kind: str
    scale: tuple[float, float]

    def __post_init__(self) -> None:
        if self.kind != "value-scale":
            raise ObservationFormationValidationError(
                f"unsupported value scale variation kind: {self.kind}"
            )
        _validate_interval(self.scale, field="value_scale.scale", positive=True)

    @classmethod
    def identity(cls) -> ValueScaleVariation:
        return cls(kind="value-scale", scale=(1.0, 1.0))

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ValueScaleVariation:
        try:
            validated = _value_scale_variation_record.validate(record)
        except ValueError as error:
            raise ObservationFormationValidationError(str(error)) from error
        return cls(
            kind=str(validated["kind"]),
            scale=_interval(validated["scale"], field="value_scale.scale"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "scale": list(self.scale),
        }


@dataclass(frozen=True, slots=True)
class VariationTransformDeclaration:
    """Declared variation transforms available to observation formation."""

    kind: str
    spatial_affine: SpatialAffineVariation
    value_scale: ValueScaleVariation

    def __post_init__(self) -> None:
        if self.kind != "field-variation-transform":
            raise ObservationFormationValidationError(
                f"unsupported variation transform kind: {self.kind}"
            )

    @classmethod
    def identity(cls, *, spatial_rank: int = 2) -> VariationTransformDeclaration:
        return cls(
            kind="field-variation-transform",
            spatial_affine=SpatialAffineVariation.identity(spatial_rank=spatial_rank),
            value_scale=ValueScaleVariation.identity(),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> VariationTransformDeclaration:
        try:
            validated = _variation_transform_record.validate(record)
        except ValueError as error:
            raise ObservationFormationValidationError(str(error)) from error
        return cls(
            kind=str(validated["kind"]),
            spatial_affine=(
                SpatialAffineVariation.identity(spatial_rank=2)
                if "spatial_affine" not in validated
                else SpatialAffineVariation.from_record(
                    _as_mapping(validated["spatial_affine"], field="spatial_affine")
                )
            ),
            value_scale=(
                ValueScaleVariation.identity()
                if "value_scale" not in validated
                else ValueScaleVariation.from_record(
                    _as_mapping(validated["value_scale"], field="value_scale")
                )
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "spatial_affine": self.spatial_affine.to_record(),
            "value_scale": self.value_scale.to_record(),
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
                "formation_declaration reference must have kind observation-formation-declaration"
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
    width_axis: str
    height_axis: str
    sequence_layout: SequenceLayout
    components: tuple[ObservationComponent, ...]
    variation_transform: VariationTransformDeclaration = field(
        default_factory=VariationTransformDeclaration.identity
    )

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
        if not self.width_axis:
            raise ObservationFormationValidationError("width_axis must be nonempty")
        if not self.height_axis:
            raise ObservationFormationValidationError("height_axis must be nonempty")
        if self.sequence_layout.width_axis != self.width_axis:
            raise ObservationFormationValidationError(
                "sequence layout width_axis must match output field width_axis"
            )
        if self.sequence_layout.height_axis != self.height_axis:
            raise ObservationFormationValidationError(
                "sequence layout height_axis must match output field height_axis"
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
                        f"mark channel {mark.channel} is outside channel_count {self.channel_count}"
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
            width_axis=str(output["width_axis"]),
            height_axis=str(output["height_axis"]),
            sequence_layout=SequenceLayout.from_record(
                _as_mapping(validated["sequence_layout"], field="sequence_layout")
            ),
            components=tuple(
                ObservationComponent.from_record(_as_mapping(component, field="components"))
                for component in _as_sequence(validated["components"], field="components")
            ),
            variation_transform=(
                VariationTransformDeclaration.identity()
                if "variation_transform" not in validated
                else VariationTransformDeclaration.from_record(
                    _as_mapping(validated["variation_transform"], field="variation_transform")
                )
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
        sequence_length = plan.scale_assignment.require_axis(self.sequence_layout.sequence_axis)
        if sequence_length < 1:
            raise ObservationFormationValidationError("sequence length must be positive")
        generator = random.Random(plan.seed + sample_index)
        return tuple(
            generator.randrange(len(self.components))
            for _sequence_element in range(sequence_length)
        )

    def form_observation(
        self,
        *,
        id: ProtocolIdentifier,
        plan: MaterializationPlan,
        component_sequence: Sequence[int],
        variation_coordinates: Sequence[Mapping[str, object]] | None = None,
    ) -> FormedObservation:
        if plan.benchmark_id != self.benchmark_id:
            raise ObservationFormationValidationError(
                f"plan benchmark_id {plan.benchmark_id} does not match {self.benchmark_id}"
            )
        sequence_elements = plan.scale_assignment.require_axis(self.sequence_layout.sequence_axis)
        width = plan.resolution_assignment.require_axis(self.width_axis)
        height = plan.resolution_assignment.require_axis(self.height_axis)
        sequence = tuple(_as_int(index, field="component_sequence") for index in component_sequence)
        if len(sequence) != sequence_elements:
            raise ObservationFormationValidationError(
                f"component_sequence length {len(sequence)} does not match "
                f"sequence length {sequence_elements}"
            )
        if any(index >= len(self.components) for index in sequence):
            raise ObservationFormationValidationError(
                "component_sequence index is outside component vocabulary"
            )
        coordinates = self._variation_coordinates(
            sequence_length=sequence_elements,
            variation_coordinates=variation_coordinates,
        )
        field = self._form_field(
            sequence=sequence,
            width=width,
            height=height,
            variation_coordinates=coordinates,
        )
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

    def component_field(
        self,
        *,
        width: int,
        height: int,
        sequence_length: int,
        sequence_index: int,
        component_index: int,
    ) -> FieldObservation:
        """Form one unvaried component at one sequence position in the output field."""

        if width < 1:
            raise ObservationFormationValidationError("width must be positive")
        if height < 1:
            raise ObservationFormationValidationError("height must be positive")
        if sequence_length < 1:
            raise ObservationFormationValidationError("sequence_length must be positive")
        if sequence_index < 0 or sequence_index >= sequence_length:
            raise ObservationFormationValidationError(
                "sequence_index must be within sequence_length"
            )
        if component_index < 0 or component_index >= len(self.components):
            raise ObservationFormationValidationError(
                "component_index is outside component vocabulary"
            )
        values = [0.0] * (self.channel_count * width * height)
        component = self.components[component_index]
        for mark in component.marks:
            _draw_mark(
                values=values,
                channel_count=self.channel_count,
                width=width,
                height=height,
                sequence_length=sequence_length,
                sequence_index=sequence_index,
                placement_axis=self.sequence_layout.placement_axis,
                mark=mark,
            )
        return FieldObservation(
            shape=(self.channel_count, height, width),
            values=tuple(values),
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
                "width_axis": self.width_axis,
                "height_axis": self.height_axis,
            },
            "sequence_layout": self.sequence_layout.to_record(),
            "variation_transform": self.variation_transform.to_record(),
            "components": [component.to_record() for component in self.components],
        }

    def _variation_coordinates(
        self,
        *,
        sequence_length: int,
        variation_coordinates: Sequence[Mapping[str, object]] | None,
    ) -> tuple[_VariationCoordinate, ...] | None:
        if variation_coordinates is None:
            return None
        coordinates = tuple(variation_coordinates)
        if len(coordinates) != sequence_length:
            raise ObservationFormationValidationError(
                "variation_coordinates length must match sequence length"
            )
        parsed_coordinates: list[_VariationCoordinate] = []
        for sequence_index, coordinate in enumerate(coordinates):
            parsed = _parse_variation_coordinate(coordinate, field="variation_coordinates")
            if parsed.sequence_index != sequence_index:
                raise ObservationFormationValidationError(
                    "variation coordinate sequence_index must match coordinate position"
                )
            parsed_coordinates.append(parsed)
        return tuple(parsed_coordinates)

    def _form_field(
        self,
        *,
        sequence: tuple[int, ...],
        width: int,
        height: int,
        variation_coordinates: tuple[_VariationCoordinate, ...] | None = None,
    ) -> FieldObservation:
        if width < 1:
            raise ObservationFormationValidationError("width must be positive")
        if height < 1:
            raise ObservationFormationValidationError("height must be positive")
        values = [0.0] * (self.channel_count * width * height)
        for sequence_index, component_index in enumerate(sequence):
            target_values = values
            if variation_coordinates is not None:
                target_values = [0.0] * len(values)
            component = self.components[component_index]
            for mark in component.marks:
                _draw_mark(
                    values=target_values,
                    channel_count=self.channel_count,
                    width=width,
                    height=height,
                    sequence_length=len(sequence),
                    sequence_index=sequence_index,
                    placement_axis=self.sequence_layout.placement_axis,
                    mark=mark,
                )
            if variation_coordinates is not None:
                _merge_transformed_sequence_element(
                    values=values,
                    source_values=target_values,
                    channel_count=self.channel_count,
                    width=width,
                    height=height,
                    sequence_length=len(sequence),
                    sequence_index=sequence_index,
                    placement_axis=self.sequence_layout.placement_axis,
                    coordinate=variation_coordinates[sequence_index],
                )
        return FieldObservation(
            shape=(self.channel_count, height, width),
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
    width: int,
    height: int,
    sequence_length: int,
    sequence_index: int,
    placement_axis: str,
    mark: ComponentMark,
) -> None:
    curve_points = tuple(
        _field_pixel_point(
            point,
            width=width,
            height=height,
            sequence_length=sequence_length,
            sequence_index=sequence_index,
            placement_axis=placement_axis,
        )
        for point in _sample_bezier_curve(mark.control_points)
    )
    threshold = mark.width / 2.0
    x_range = _pixel_index_range(
        min(point[0] for point in curve_points) - threshold,
        max(point[0] for point in curve_points) + threshold,
        size=width,
    )
    y_range = _pixel_index_range(
        min(point[1] for point in curve_points) - threshold,
        max(point[1] for point in curve_points) + threshold,
        size=height,
    )
    for y_index in y_range:
        y = y_index + 0.5
        for x_index in x_range:
            x = x_index + 0.5
            if _polyline_distance((x, y), curve_points) <= threshold:
                value_index = mark.channel * width * height + y_index * width + x_index
                values[value_index] = max(values[value_index], mark.value)


def _merge_transformed_sequence_element(
    *,
    values: list[float],
    source_values: list[float],
    channel_count: int,
    width: int,
    height: int,
    sequence_length: int,
    sequence_index: int,
    placement_axis: str,
    coordinate: _VariationCoordinate,
) -> None:
    if coordinate.spatial_affine.coordinate_system != "normalized-sequence-element":
        raise ObservationFormationValidationError(
            "variation coordinate coordinate_system must be normalized-sequence-element"
        )
    if _is_identity_variation_coordinate(coordinate):
        for index, value in enumerate(source_values):
            if value > values[index]:
                values[index] = value
        return
    center = _sequence_center(
        width=width,
        height=height,
        sequence_length=sequence_length,
        sequence_index=sequence_index,
        placement_axis=placement_axis,
    )
    inverse = _inverse_affine_matrix(coordinate.spatial_affine)
    translation = _sequence_relative_translation(
        coordinate.spatial_affine.translation,
        sequence_length=sequence_length,
        placement_axis=placement_axis,
    )
    for channel in range(channel_count):
        channel_offset = channel * width * height
        for y_index in range(height):
            y = (y_index + 0.5) / height
            for x_index in range(width):
                x = (x_index + 0.5) / width
                source_x, source_y = _inverse_transform_point(
                    (x, y),
                    center=center,
                    translation=translation,
                    inverse_matrix=inverse,
                )
                value = _bilinear_sample(
                    source_values,
                    channel_offset=channel_offset,
                    width=width,
                    height=height,
                    x=source_x,
                    y=source_y,
                )
                if value <= 0.0:
                    continue
                target_index = channel_offset + y_index * width + x_index
                values[target_index] = max(
                    values[target_index],
                    min(1.0, value * coordinate.value_scale),
                )


def _is_identity_variation_coordinate(coordinate: _VariationCoordinate) -> bool:
    return (
        coordinate.spatial_affine.translation == (0.0, 0.0)
        and coordinate.spatial_affine.scale == (1.0, 1.0)
        and coordinate.spatial_affine.rotation_degrees == 0.0
        and coordinate.spatial_affine.shear_degrees == 0.0
        and coordinate.value_scale == 1.0
    )


def _sequence_center(
    *,
    width: int,
    height: int,
    sequence_length: int,
    sequence_index: int,
    placement_axis: str,
) -> tuple[float, float]:
    return _sequence_point(
        (0.5, 0.5),
        width=width,
        height=height,
        sequence_length=sequence_length,
        sequence_index=sequence_index,
        placement_axis=placement_axis,
    )


def _sequence_relative_translation(
    translation: tuple[float, float],
    *,
    sequence_length: int,
    placement_axis: str,
) -> tuple[float, float]:
    if placement_axis == "x":
        return (translation[0] / sequence_length, translation[1])
    return (translation[0], translation[1] / sequence_length)


def _inverse_affine_matrix(
    coordinate: _SpatialAffineCoordinate,
) -> tuple[tuple[float, float], tuple[float, float]]:
    angle = math.radians(coordinate.rotation_degrees)
    shear = math.tan(math.radians(coordinate.shear_degrees))
    scale_x, scale_y = coordinate.scale
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    # Forward matrix is rotation * x-shear * axis scale.
    a = cos_angle * scale_x
    b = (cos_angle * shear - sin_angle) * scale_y
    c = sin_angle * scale_x
    d = (sin_angle * shear + cos_angle) * scale_y
    determinant = a * d - b * c
    if not math.isfinite(determinant) or determinant == 0.0:
        raise ObservationFormationValidationError("variation affine transform is singular")
    return ((d / determinant, -b / determinant), (-c / determinant, a / determinant))


def _inverse_transform_point(
    point: tuple[float, float],
    *,
    center: tuple[float, float],
    translation: tuple[float, float],
    inverse_matrix: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    dx = point[0] - center[0] - translation[0]
    dy = point[1] - center[1] - translation[1]
    return (
        center[0] + inverse_matrix[0][0] * dx + inverse_matrix[0][1] * dy,
        center[1] + inverse_matrix[1][0] * dx + inverse_matrix[1][1] * dy,
    )


def _bilinear_sample(
    values: Sequence[float],
    *,
    channel_offset: int,
    width: int,
    height: int,
    x: float,
    y: float,
) -> float:
    pixel_x = x * width - 0.5
    pixel_y = y * height - 0.5
    if pixel_x < 0 or pixel_y < 0 or pixel_x > width - 1 or pixel_y > height - 1:
        return 0.0
    left = math.floor(pixel_x)
    top = math.floor(pixel_y)
    right = min(width - 1, left + 1)
    bottom = min(height - 1, top + 1)
    x_weight = pixel_x - left
    y_weight = pixel_y - top
    top_left = values[channel_offset + top * width + left]
    top_right = values[channel_offset + top * width + right]
    bottom_left = values[channel_offset + bottom * width + left]
    bottom_right = values[channel_offset + bottom * width + right]
    top_value = (1.0 - x_weight) * top_left + x_weight * top_right
    bottom_value = (1.0 - x_weight) * bottom_left + x_weight * bottom_right
    return (1.0 - y_weight) * top_value + y_weight * bottom_value


def _pixel_index_range(lower: float, upper: float, *, size: int) -> range:
    start = max(0, math.floor(lower))
    stop = min(size, math.ceil(upper))
    return range(start, stop)


def _sample_bezier_curve(
    control_points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    sample_count = max(1, (len(control_points) - 1) * _curve_samples_per_degree)
    return tuple(
        _bezier_point(control_points, index / sample_count) for index in range(sample_count + 1)
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


def _sequence_point(
    point: tuple[float, float],
    *,
    width: int,
    height: int,
    sequence_length: int,
    sequence_index: int,
    placement_axis: str,
) -> tuple[float, float]:
    x, y = point
    if placement_axis == "x":
        return ((sequence_index + x) / sequence_length, y)
    return (x, (sequence_index + y) / sequence_length)


def _field_pixel_point(
    point: tuple[float, float],
    *,
    width: int,
    height: int,
    sequence_length: int,
    sequence_index: int,
    placement_axis: str,
) -> tuple[float, float]:
    x, y = _sequence_point(
        point,
        width=width,
        height=height,
        sequence_length=sequence_length,
        sequence_index=sequence_index,
        placement_axis=placement_axis,
    )
    return (x * width, y * height)


def _polyline_distance(
    point: tuple[float, float],
    points: tuple[tuple[float, float], ...],
) -> float:
    return min(
        _segment_distance(point, start, end) for start, end in zip(points, points[1:], strict=False)
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


def _parse_variation_coordinate(
    value: Mapping[str, object],
    *,
    field: str,
) -> _VariationCoordinate:
    if str(value.get("kind")) != "field-variation-transform-coordinate":
        raise ObservationFormationValidationError(
            f"{field}: expected field-variation-transform-coordinate"
        )
    spatial = _as_mapping(value.get("spatial_affine"), field=f"{field}.spatial_affine")
    if str(spatial.get("kind")) != "spatial-affine-coordinate":
        raise ObservationFormationValidationError(
            f"{field}.spatial_affine: expected spatial-affine-coordinate"
        )
    value_scale = _as_mapping(value.get("value_scale"), field=f"{field}.value_scale")
    if str(value_scale.get("kind")) != "value-scale-coordinate":
        raise ObservationFormationValidationError(
            f"{field}.value_scale: expected value-scale-coordinate"
        )
    translation = _coordinate_pair(
        spatial.get("translation"),
        field=f"{field}.spatial_affine.translation",
    )
    scale = _coordinate_pair(spatial.get("scale"), field=f"{field}.spatial_affine.scale")
    if scale[0] <= 0.0 or scale[1] <= 0.0:
        raise ObservationFormationValidationError(
            f"{field}.spatial_affine.scale values must be positive"
        )
    value_scale_number = _as_float(value_scale.get("scale"), field=f"{field}.value_scale.scale")
    if not math.isfinite(value_scale_number) or value_scale_number <= 0.0:
        raise ObservationFormationValidationError(
            f"{field}.value_scale.scale must be finite and positive"
        )
    return _VariationCoordinate(
        sequence_index=_as_int(value.get("sequence_index"), field=f"{field}.sequence_index"),
        spatial_affine=_SpatialAffineCoordinate(
            coordinate_system=str(spatial.get("coordinate_system")),
            translation=translation,
            scale=scale,
            rotation_degrees=_single_coordinate_number(
                spatial.get("rotation_degrees"),
                field=f"{field}.spatial_affine.rotation_degrees",
            ),
            shear_degrees=_single_coordinate_number(
                spatial.get("shear_degrees"),
                field=f"{field}.spatial_affine.shear_degrees",
            ),
        ),
        value_scale=value_scale_number,
    )


def _coordinate_pair(value: object, *, field: str) -> tuple[float, float]:
    sequence = _coordinate_sequence(value, field=field)
    if len(sequence) != 2:
        raise ObservationFormationValidationError(f"{field}: expected two values")
    first = _as_float(sequence[0], field=f"{field}.0")
    second = _as_float(sequence[1], field=f"{field}.1")
    if not math.isfinite(first) or not math.isfinite(second):
        raise ObservationFormationValidationError(f"{field} values must be finite")
    return (first, second)


def _single_coordinate_number(value: object, *, field: str) -> float:
    sequence = _coordinate_sequence(value, field=field)
    if len(sequence) != 1:
        raise ObservationFormationValidationError(f"{field}: expected one value")
    number = _as_float(sequence[0], field=f"{field}.0")
    if not math.isfinite(number):
        raise ObservationFormationValidationError(f"{field} value must be finite")
    return number


def _coordinate_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return cast(tuple[object, ...], value)
    if isinstance(value, list):
        return tuple(cast(list[object], value))
    raise ObservationFormationValidationError(f"{field}: expected sequence")


def _interval(value: object, *, field: str) -> tuple[float, float]:
    sequence = _as_sequence(value, field=field)
    if len(sequence) != 2:
        raise ObservationFormationValidationError(f"{field}: expected two bounds")
    return (
        _as_float(sequence[0], field=f"{field}.0"),
        _as_float(sequence[1], field=f"{field}.1"),
    )


def _interval_sequence(value: object, *, field: str) -> tuple[tuple[float, float], ...]:
    return tuple(
        _interval(item, field=f"{field}.{index}")
        for index, item in enumerate(_as_sequence(value, field=field))
    )


def _number_sequence(value: object, *, field: str) -> tuple[float, ...]:
    return tuple(
        _as_float(item, field=f"{field}.{index}")
        for index, item in enumerate(_as_sequence(value, field=field))
    )


def _validate_interval(
    interval: tuple[float, float],
    *,
    field: str,
    positive: bool = False,
) -> None:
    low, high = interval
    if not math.isfinite(low) or not math.isfinite(high):
        raise ObservationFormationValidationError(f"{field} bounds must be finite")
    if low > high:
        raise ObservationFormationValidationError(f"{field} lower bound must not exceed upper")
    if positive and low <= 0:
        raise ObservationFormationValidationError(f"{field} bounds must be positive")


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
