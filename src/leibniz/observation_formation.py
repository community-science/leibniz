"""Manifest-driven formation of benchmark observation fields."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import cast

from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec

__all__ = [
    "AffineMatrix2D",
    "ComponentMark",
    "FieldObservation",
    "ObservationComponentDiscriminabilityReport",
    "VariationTransformDeclaration",
    "ObservationComponent",
    "ObservationFormationDeclaration",
    "ObservationFormationDeclarationDocument",
    "ObservationFormationValidationError",
    "SequenceLayout",
    "SpatialAffineVariation",
    "VariationCoordinate",
    "affine_translation",
    "linear_affine_matrix",
]

_interpreter = "field-mark-composition@0.1.0"
_curve_samples_per_degree = 12
_component_analysis_field_cache: dict[
    tuple[str, int, int, int],
    FieldObservation,
] = {}

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
        "matrix": FieldSpec(
            kind="sequence",
            item=FieldSpec(
                kind="sequence",
                item=FieldSpec(kind="sequence", item=FieldSpec(kind="number")),
            ),
        ),
    }
)
_variation_transform_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="string"),
        "spatial_affine": FieldSpec(kind="record", required=False),
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


_extract = RecordExtractor(error_type=ObservationFormationValidationError)


AffineMatrix2D = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


@dataclass(frozen=True, slots=True)
class VariationCoordinate:
    component_index: int
    matrix: AffineMatrix2D
    native_footprint_side: float | None = None


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
        value = _extract.optional_float(validated.get("value"), "value")
        return cls(
            kind=str(validated["kind"]),
            channel=_extract.integer(validated["channel"], "channel"),
            degree=_extract.integer(validated["degree"], "degree"),
            control_points=tuple(
                _point(point, field=f"control_points.{index}")
                for index, point in enumerate(
                    _extract.sequence(validated["control_points"], "control_points")
                )
            ),
            width=_extract.float(validated["width"], "width"),
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
                ComponentMark.from_record(_extract.mapping(mark, "marks"))
                for mark in _extract.sequence(validated["marks"], "marks")
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
    """Declared spatial affine proposal bounds in normalized field coordinates."""

    kind: str
    coordinate_system: str
    spatial_rank: int
    matrix: tuple[tuple[tuple[float, float], ...], ...]

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
        affine_rank = self.spatial_rank + 1
        if len(self.matrix) != affine_rank:
            raise ObservationFormationValidationError(
                "matrix row count must equal spatial_rank plus one"
            )
        for row_index, row in enumerate(self.matrix):
            if len(row) != affine_rank:
                raise ObservationFormationValidationError(
                    "matrix column count must equal spatial_rank plus one"
                )
            for column_index, bounds in enumerate(row):
                _validate_interval(bounds, field=f"matrix.{row_index}.{column_index}")
                final_row_value = 1.0 if column_index == self.spatial_rank else 0.0
                if row_index == self.spatial_rank and bounds != (
                    final_row_value,
                    final_row_value,
                ):
                    raise ObservationFormationValidationError(
                        "matrix final row must be fixed affine coordinates"
                    )

    @classmethod
    def identity(cls, *, spatial_rank: int) -> SpatialAffineVariation:
        return cls(
            kind="spatial-affine",
            coordinate_system="normalized-sequence-element",
            spatial_rank=spatial_rank,
            matrix=tuple(
                tuple(
                    (1.0, 1.0) if row == column else (0.0, 0.0)
                    for column in range(spatial_rank + 1)
                )
                for row in range(spatial_rank + 1)
            ),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SpatialAffineVariation:
        try:
            validated = _spatial_affine_variation_record.validate(record)
        except ValueError as error:
            raise ObservationFormationValidationError(str(error)) from error
        spatial_rank = _extract.integer(validated["spatial_rank"], "spatial_rank")
        return cls(
            kind=str(validated["kind"]),
            coordinate_system=str(validated["coordinate_system"]),
            spatial_rank=spatial_rank,
            matrix=_matrix_interval_sequence(validated["matrix"], field="matrix"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "coordinate_system": self.coordinate_system,
            "spatial_rank": self.spatial_rank,
            "matrix": [[list(bounds) for bounds in row] for row in self.matrix],
        }


@dataclass(frozen=True, slots=True)
class VariationTransformDeclaration:
    """Declared variation transforms available to observation formation."""

    kind: str
    spatial_affine: SpatialAffineVariation

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
                    _extract.mapping(validated["spatial_affine"], "spatial_affine")
                )
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "spatial_affine": self.spatial_affine.to_record(),
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
class ObservationComponentDiscriminabilityReport:
    """Pairwise discriminability report for one observation formation envelope."""

    width: int
    height: int
    component_vocabulary_size: int
    variation_case_count: int
    required_pairwise_l1: float
    minimum_pairwise_l1: float
    nearest_pair: tuple[int, int] | None

    @property
    def passed(self) -> bool:
        """Return whether every component pair has positive distance."""

        return (
            self.nearest_pair is not None
            and self.minimum_pairwise_l1 > 0.0
            and self.minimum_pairwise_l1 >= self.required_pairwise_l1
        )


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
    _digest: ContentDigest = field(init=False, repr=False, compare=False)

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
        object.__setattr__(self, "_digest", ContentDigest.from_value(self.to_record()))

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ObservationFormationDeclaration:
        try:
            validated = _observation_formation_declaration_record.validate(record)
            output_field = _extract.mapping(validated["output_field"], "output_field")
            output = _output_field_record.validate(output_field)
        except ValueError as error:
            raise ObservationFormationValidationError(str(error)) from error
        return cls(
            id=_extract.identifier(validated["id"], "id"),
            benchmark_id=_extract.identifier(validated["benchmark_id"], "benchmark_id"),
            interpreter=str(validated["interpreter"]),
            channel_count=_extract.integer(output["channel_count"], "channel_count"),
            width_axis=str(output["width_axis"]),
            height_axis=str(output["height_axis"]),
            sequence_layout=SequenceLayout.from_record(
                _extract.mapping(validated["sequence_layout"], "sequence_layout")
            ),
            components=tuple(
                ObservationComponent.from_record(_extract.mapping(component, "components"))
                for component in _extract.sequence(validated["components"], "components")
            ),
            variation_transform=(
                VariationTransformDeclaration.identity()
                if "variation_transform" not in validated
                else VariationTransformDeclaration.from_record(
                    _extract.mapping(validated["variation_transform"], "variation_transform")
                )
            ),
        )

    def sample_component_index(
        self,
        *,
        seed: int,
        sample_index: int,
    ) -> int:
        if sample_index < 0:
            raise ObservationFormationValidationError("sample_index must be nonnegative")
        generator = random.Random(seed + sample_index)
        return generator.randrange(len(self.components))

    def component_field(
        self,
        *,
        width: int,
        height: int,
        component_index: int,
        variation_coordinate: Mapping[str, object] | VariationCoordinate | None = None,
    ) -> FieldObservation:
        """Form one component in the output field."""

        if width < 1:
            raise ObservationFormationValidationError("width must be positive")
        if height < 1:
            raise ObservationFormationValidationError("height must be positive")
        if component_index < 0 or component_index >= len(self.components):
            raise ObservationFormationValidationError(
                "component_index is outside component vocabulary"
            )
        parsed_coordinate = (
            None
            if variation_coordinate is None
            else (
                variation_coordinate
                if isinstance(variation_coordinate, VariationCoordinate)
                else _parse_variation_coordinate(
                    variation_coordinate,
                    field="variation_coordinates",
                )
            )
        )
        values = [0.0] * (self.channel_count * width * height)
        component = self.components[component_index]
        for mark in component.marks:
            _draw_mark(
                values=values,
                channel_count=self.channel_count,
                width=width,
                height=height,
                mark=(
                    mark
                    if parsed_coordinate is None
                    else _transform_mark(mark, parsed_coordinate, field_width=width)
                ),
            )
        return FieldObservation(
            shape=(self.channel_count, height, width),
            values=tuple(values),
        )

    def component_discriminability_report(
        self,
        *,
        width: int,
        height: int,
        variation_coordinates: Sequence[Mapping[str, object] | None] = (None,),
        minimum_pairwise_l1: float = 0.0,
    ) -> ObservationComponentDiscriminabilityReport:
        """Measure whether component renderings remain pairwise distinct."""

        if width < 1:
            raise ObservationFormationValidationError("width must be positive")
        if height < 1:
            raise ObservationFormationValidationError("height must be positive")
        if not math.isfinite(minimum_pairwise_l1) or minimum_pairwise_l1 < 0.0:
            raise ObservationFormationValidationError(
                "minimum_pairwise_l1 must be finite and nonnegative"
            )
        coordinate_cases = tuple(variation_coordinates)
        if not coordinate_cases:
            raise ObservationFormationValidationError(
                "variation_coordinates must not be empty"
            )
        minimum_distance = float("inf")
        nearest_pair: tuple[int, int] | None = None
        for coordinate in coordinate_cases:
            fields = tuple(
                self._component_analysis_field(
                    width=width,
                    height=height,
                    component_index=component_index,
                    variation_coordinate=coordinate,
                )
                for component_index in range(len(self.components))
            )
            for left_index, right_index in combinations(range(len(fields)), 2):
                distance = _l1_distance(fields[left_index].values, fields[right_index].values)
                if distance < minimum_distance:
                    minimum_distance = distance
                    nearest_pair = (left_index, right_index)
                    if minimum_distance < minimum_pairwise_l1:
                        return ObservationComponentDiscriminabilityReport(
                            width=width,
                            height=height,
                            component_vocabulary_size=len(self.components),
                            variation_case_count=len(coordinate_cases),
                            required_pairwise_l1=minimum_pairwise_l1,
                            minimum_pairwise_l1=minimum_distance,
                            nearest_pair=nearest_pair,
                        )
        return ObservationComponentDiscriminabilityReport(
            width=width,
            height=height,
            component_vocabulary_size=len(self.components),
            variation_case_count=len(coordinate_cases),
            required_pairwise_l1=minimum_pairwise_l1,
            minimum_pairwise_l1=minimum_distance if nearest_pair is not None else 0.0,
            nearest_pair=nearest_pair,
        )

    def component_discriminability_passes(
        self,
        *,
        width: int,
        height: int,
        variation_coordinates: Sequence[Mapping[str, object] | None] = (None,),
        minimum_pairwise_l1: float = 0.0,
    ) -> bool:
        """Return whether component renderings clear a pairwise distance margin."""

        if minimum_pairwise_l1 <= 0.0:
            return self.component_discriminability_report(
                width=width,
                height=height,
                variation_coordinates=variation_coordinates,
                minimum_pairwise_l1=minimum_pairwise_l1,
            ).passed
        if width < 1:
            raise ObservationFormationValidationError("width must be positive")
        if height < 1:
            raise ObservationFormationValidationError("height must be positive")
        if not math.isfinite(minimum_pairwise_l1):
            raise ObservationFormationValidationError(
                "minimum_pairwise_l1 must be finite and nonnegative"
            )
        coordinate_cases = tuple(variation_coordinates)
        if not coordinate_cases:
            raise ObservationFormationValidationError(
                "variation_coordinates must not be empty"
            )
        return self._component_fields_clear_margin(
            width=width,
            height=height,
            variation_coordinates=coordinate_cases,
            minimum_pairwise_l1=minimum_pairwise_l1,
        )

    def minimum_discriminatable_resolution(
        self,
        *,
        minimum_width: int,
        minimum_height: int,
        maximum_width: int | None = None,
        maximum_height: int | None = None,
        variation_coordinates: Sequence[Mapping[str, object] | None] | None = None,
        minimum_pairwise_l1: float = 0.0,
    ) -> tuple[int, int]:
        """Return the smallest searched resolution with distinct components."""

        if minimum_width < 1:
            raise ObservationFormationValidationError("minimum_width must be positive")
        if minimum_height < 1:
            raise ObservationFormationValidationError("minimum_height must be positive")
        width_default = max(minimum_width * 4, 64)
        width_limit = maximum_width if maximum_width is not None else width_default
        height_limit = maximum_height if maximum_height is not None else max(minimum_height * 4, 64)
        if width_limit < minimum_width:
            raise ObservationFormationValidationError("maximum_width is below minimum_width")
        if height_limit < minimum_height:
            raise ObservationFormationValidationError("maximum_height is below minimum_height")
        coordinate_cases = tuple(variation_coordinates or (None,))
        for total in range(minimum_width + minimum_height, width_limit + height_limit + 1):
            width_start = max(minimum_width, total - height_limit)
            width_stop = min(width_limit, total - minimum_height)
            for width in range(width_start, width_stop + 1):
                height = total - width
                if self._resolution_is_discriminatable(
                    width=width,
                    height=height,
                    variation_coordinates=coordinate_cases,
                    minimum_pairwise_l1=minimum_pairwise_l1,
                ):
                    return (width, height)
        raise ObservationFormationValidationError(
            "no discriminatable resolution found within search bounds"
        )

    def _component_cell_is_discriminatable(
        self,
        *,
        width: int,
        height: int,
        variation_coordinates: tuple[Mapping[str, object] | None, ...],
        minimum_pairwise_l1: float,
    ) -> bool:
        if minimum_pairwise_l1 <= 0.0:
            return self._component_cell_report_is_discriminatable(
                width=width,
                height=height,
                variation_coordinates=variation_coordinates,
                minimum_pairwise_l1=minimum_pairwise_l1,
            )
        if not self._component_fields_clear_margin(
            width=width,
            height=height,
            variation_coordinates=(None,),
            minimum_pairwise_l1=minimum_pairwise_l1,
        ):
            return False
        return self._component_fields_clear_margin(
            width=width,
            height=height,
            variation_coordinates=variation_coordinates,
            minimum_pairwise_l1=minimum_pairwise_l1,
        )

    def _component_cell_report_is_discriminatable(
        self,
        *,
        width: int,
        height: int,
        variation_coordinates: tuple[Mapping[str, object] | None, ...],
        minimum_pairwise_l1: float,
    ) -> bool:
        unvaried = self.component_discriminability_report(
            width=width,
            height=height,
            minimum_pairwise_l1=minimum_pairwise_l1,
        )
        if not unvaried.passed:
            return False
        varied = self.component_discriminability_report(
            width=width,
            height=height,
            variation_coordinates=variation_coordinates,
            minimum_pairwise_l1=minimum_pairwise_l1,
        )
        return varied.passed

    def _component_fields_clear_margin(
        self,
        *,
        width: int,
        height: int,
        variation_coordinates: tuple[Mapping[str, object] | None, ...],
        minimum_pairwise_l1: float,
    ) -> bool:
        for coordinate in variation_coordinates:
            fields = tuple(
                self._component_analysis_field(
                    width=width,
                    height=height,
                    component_index=component_index,
                    variation_coordinate=coordinate,
                )
                for component_index in range(len(self.components))
            )
            for left_index, right_index in combinations(range(len(fields)), 2):
                if not _l1_distance_reaches(
                    fields[left_index].values,
                    fields[right_index].values,
                    minimum_pairwise_l1,
                ):
                    return False
        return True

    def boundary_variation_coordinates(
        self,
    ) -> tuple[Mapping[str, object], ...]:
        """Return transform-bound coordinates for live resolution analysis."""

        spatial = self.variation_transform.spatial_affine
        cases: list[Mapping[str, object]] = []
        for values in product(*(bounds for row in spatial.matrix for bounds in row)):
            matrix = [
                list(values[index : index + spatial.spatial_rank + 1])
                for index in range(0, len(values), spatial.spatial_rank + 1)
            ]
            cases.append(
                {
                    "kind": "field-variation-transform-coordinate",
                    "component_index": 0,
                    "spatial_affine": {
                        "kind": "spatial-affine-coordinate",
                        "coordinate_system": spatial.coordinate_system,
                        "matrix": matrix,
                    },
                }
            )
        return tuple(cases)

    def _resolution_is_discriminatable(
        self,
        *,
        width: int,
        height: int,
        variation_coordinates: tuple[Mapping[str, object] | None, ...],
        minimum_pairwise_l1: float,
    ) -> bool:
        report = self.component_discriminability_report(
            width=width,
            height=height,
            variation_coordinates=variation_coordinates,
            minimum_pairwise_l1=minimum_pairwise_l1,
        )
        return report.passed

    def _component_analysis_field(
        self,
        *,
        width: int,
        height: int,
        component_index: int,
        variation_coordinate: Mapping[str, object] | None,
    ) -> FieldObservation:
        if variation_coordinate is None:
            return self._cached_component_analysis_field(
                width=width,
                height=height,
                component_index=component_index,
            )
        return self.component_field(
            width=width,
            height=height,
            component_index=component_index,
            variation_coordinate=variation_coordinate,
        )

    def _cached_component_analysis_field(
        self,
        *,
        width: int,
        height: int,
        component_index: int,
    ) -> FieldObservation:
        key = (str(self.digest), width, height, component_index)
        cached = _component_analysis_field_cache.get(key)
        if cached is None:
            cached = self.component_field(
                width=width,
                height=height,
                component_index=component_index,
            )
            _component_analysis_field_cache[key] = cached
        return cached

    @property
    def digest(self) -> ContentDigest:
        return self._digest

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
        variation_coordinates: Sequence[Mapping[str, object]] | None,
    ) -> tuple[VariationCoordinate, ...] | None:
        if variation_coordinates is None:
            return None
        coordinates = tuple(variation_coordinates)
        if len(coordinates) != 1:
            raise ObservationFormationValidationError(
                "variation_coordinates must contain one coordinate"
            )
        parsed_coordinates: list[VariationCoordinate] = []
        for coordinate in coordinates:
            parsed = _parse_variation_coordinate(coordinate, field="variation_coordinates")
            parsed_coordinates.append(parsed)
        return tuple(parsed_coordinates)

    def _form_field(
        self,
        *,
        component_index: int,
        width: int,
        height: int,
        variation_coordinates: tuple[VariationCoordinate, ...] | None = None,
    ) -> FieldObservation:
        if width < 1:
            raise ObservationFormationValidationError("width must be positive")
        if height < 1:
            raise ObservationFormationValidationError("height must be positive")
        return self.component_field(
            width=width,
            height=height,
            component_index=component_index,
            variation_coordinate=(
                None if variation_coordinates is None else variation_coordinates[0]
            ),
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
    mark: ComponentMark,
) -> None:
    curve_points = tuple(
        _field_pixel_point(
            point,
            width=width,
            height=height,
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


def _transform_mark(
    mark: ComponentMark,
    coordinate: VariationCoordinate,
    *,
    field_width: int,
) -> ComponentMark:
    linear_matrix = linear_affine_matrix(coordinate.matrix)
    translation = affine_translation(coordinate.matrix)
    center = (0.5, 0.5)
    width_scale = _affine_width_scale(linear_matrix)
    if coordinate.native_footprint_side is not None:
        width_scale *= field_width / coordinate.native_footprint_side
    return ComponentMark(
        kind=mark.kind,
        channel=mark.channel,
        degree=mark.degree,
        control_points=tuple(
            _transform_point(
                point,
                center=center,
                translation=translation,
                matrix=linear_matrix,
            )
            for point in mark.control_points
        ),
        width=mark.width * width_scale,
        value=mark.value,
    )


def _affine_width_scale(
    matrix: tuple[tuple[float, float], tuple[float, float]],
) -> float:
    first_column = math.hypot(matrix[0][0], matrix[1][0])
    second_column = math.hypot(matrix[0][1], matrix[1][1])
    return max(first_column, second_column)


def linear_affine_matrix(
    matrix: AffineMatrix2D,
) -> tuple[tuple[float, float], tuple[float, float]]:
    return ((matrix[0][0], matrix[0][1]), (matrix[1][0], matrix[1][1]))


def affine_translation(
    matrix: AffineMatrix2D,
) -> tuple[float, float]:
    return (matrix[0][2], matrix[1][2])


def _transform_point(
    point: tuple[float, float],
    *,
    center: tuple[float, float],
    translation: tuple[float, float],
    matrix: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float]:
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    return (
        center[0] + matrix[0][0] * dx + matrix[0][1] * dy + translation[0],
        center[1] + matrix[1][0] * dx + matrix[1][1] * dy + translation[1],
    )


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


def _field_pixel_point(
    point: tuple[float, float],
    *,
    width: int,
    height: int,
) -> tuple[float, float]:
    x, y = point
    return (x * width, y * height)


def _polyline_distance(
    point: tuple[float, float],
    points: tuple[tuple[float, float], ...],
) -> float:
    return min(
        _segment_distance(point, start, end) for start, end in zip(points, points[1:], strict=False)
    )


def _l1_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(
        abs(left_value - right_value)
        for left_value, right_value in zip(left, right, strict=True)
    )


def _l1_distance_reaches(left: Sequence[float], right: Sequence[float], threshold: float) -> bool:
    distance = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        distance += abs(left_value - right_value)
        if distance >= threshold:
            return True
    return False


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
    sequence = _extract.sequence(value, field)
    if len(sequence) != 2:
        raise ObservationFormationValidationError(f"{field}: expected two coordinates")
    return (
        _extract.float(sequence[0], f"{field}.0"),
        _extract.float(sequence[1], f"{field}.1"),
    )


def _parse_variation_coordinate(
    value: Mapping[str, object],
    *,
    field: str,
) -> VariationCoordinate:
    if str(value.get("kind")) != "field-variation-transform-coordinate":
        raise ObservationFormationValidationError(
            f"{field}: expected field-variation-transform-coordinate"
        )
    spatial = _extract.mapping(value.get("spatial_affine"), f"{field}.spatial_affine")
    if str(spatial.get("kind")) != "spatial-affine-coordinate":
        raise ObservationFormationValidationError(
            f"{field}.spatial_affine: expected spatial-affine-coordinate"
        )
    if str(spatial.get("coordinate_system")) != "normalized-sequence-element":
        raise ObservationFormationValidationError(
            f"{field}.spatial_affine: coordinate_system must be normalized-sequence-element"
        )
    return VariationCoordinate(
        component_index=_extract.integer(
            value.get("component_index", 0),
            f"{field}.component_index",
        ),
        matrix=_coordinate_matrix(spatial.get("matrix"), field=f"{field}.spatial_affine.matrix"),
        native_footprint_side=_optional_positive_float(
            value.get("native_footprint_side"),
            field=f"{field}.native_footprint_side",
        ),
    )


def _optional_positive_float(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    parsed = _extract.float(value, field)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ObservationFormationValidationError(f"{field} must be positive and finite")
    return parsed


def _coordinate_matrix(
    value: object,
    *,
    field: str,
) -> AffineMatrix2D:
    rows = _coordinate_sequence(value, field=field)
    if len(rows) != 3:
        raise ObservationFormationValidationError(f"{field}: expected three rows")
    matrix = (
        _coordinate_triplet(rows[0], field=f"{field}.0"),
        _coordinate_triplet(rows[1], field=f"{field}.1"),
        _coordinate_triplet(rows[2], field=f"{field}.2"),
    )
    if matrix[2] != (0.0, 0.0, 1.0):
        raise ObservationFormationValidationError(
            f"{field} final row must be fixed affine coordinates"
        )
    return matrix


def _coordinate_triplet(value: object, *, field: str) -> tuple[float, float, float]:
    sequence = _coordinate_sequence(value, field=field)
    if len(sequence) != 3:
        raise ObservationFormationValidationError(f"{field}: expected three values")
    values = (
        _extract.float(sequence[0], f"{field}.0"),
        _extract.float(sequence[1], f"{field}.1"),
        _extract.float(sequence[2], f"{field}.2"),
    )
    if not all(math.isfinite(value) for value in values):
        raise ObservationFormationValidationError(f"{field} values must be finite")
    return values


def _coordinate_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return cast(tuple[object, ...], value)
    if isinstance(value, list):
        return tuple(cast(list[object], value))
    raise ObservationFormationValidationError(f"{field}: expected sequence")


def _interval(value: object, *, field: str) -> tuple[float, float]:
    sequence = _extract.sequence(value, field)
    if len(sequence) != 2:
        raise ObservationFormationValidationError(f"{field}: expected two bounds")
    return (
        _extract.float(sequence[0], f"{field}.0"),
        _extract.float(sequence[1], f"{field}.1"),
    )


def _interval_sequence(value: object, *, field: str) -> tuple[tuple[float, float], ...]:
    return tuple(
        _interval(item, field=f"{field}.{index}")
        for index, item in enumerate(_extract.sequence(value, field))
    )


def _matrix_interval_sequence(
    value: object,
    *,
    field: str,
) -> tuple[tuple[tuple[float, float], ...], ...]:
    return tuple(
        _interval_sequence(row, field=f"{field}.{row_index}")
        for row_index, row in enumerate(_extract.sequence(value, field))
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
def _first_duplicate(values: tuple[object, ...]) -> object | None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
