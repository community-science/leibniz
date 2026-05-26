"""Read-only inspection records for formed benchmark observations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest, ContentEncodingError
from leibniz.documents import canonical_document_bytes, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import AxisAssignment, MaterializationPlan
from leibniz.observation_formation import FieldObservation, FormedObservation
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "FieldPreview",
    "FieldPreviewRun",
    "ObservationInspectionDocument",
    "ObservationInspectionRecord",
    "ObservationInspectionValidationError",
]

_preview_encoding = "uint8-rle"
_preview_run_record = RecordSpec(
    fields={
        "value": FieldSpec(kind="integer"),
        "count": FieldSpec(kind="integer"),
    }
)
_field_preview_record = RecordSpec(
    fields={
        "encoding": FieldSpec(kind="literal", literal=_preview_encoding),
        "shape": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "runs": FieldSpec(kind="sequence", item=FieldSpec(kind="record")),
    }
)
_inspection_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "benchmark_id": FieldSpec(kind="identifier"),
        "formed_observation": FieldSpec(kind="record"),
        "formation_declaration": FieldSpec(kind="record"),
        "materialization_plan": FieldSpec(kind="record"),
        "sample_index": FieldSpec(kind="integer"),
        "component_sequence": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "scale_assignment": FieldSpec(kind="record"),
        "complexity_assignment": FieldSpec(kind="record"),
        "resolution_assignment": FieldSpec(kind="record"),
        "field_shape": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "field_digest": FieldSpec(kind="string"),
        "field_preview": FieldSpec(kind="record", required=False),
        "outcome_id": FieldSpec(kind="string", required=False),
    }
)


class ObservationInspectionValidationError(ValueError):
    """Raised when an observation inspection record is invalid."""


@dataclass(frozen=True, slots=True)
class FieldPreviewRun:
    """One run in a channel-first quantized field preview."""

    value: int
    count: int

    def __post_init__(self) -> None:
        if type(self.value) is not int:
            raise ObservationInspectionValidationError("preview run value must be an integer")
        if self.value < 0 or self.value > 255:
            raise ObservationInspectionValidationError(
                "preview run value must lie between 0 and 255"
            )
        if type(self.count) is not int:
            raise ObservationInspectionValidationError("preview run count must be an integer")
        if self.count < 1:
            raise ObservationInspectionValidationError("preview run count must be positive")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> FieldPreviewRun:
        try:
            validated = _preview_run_record.validate(record)
        except ValueError as error:
            raise ObservationInspectionValidationError(str(error)) from error
        return cls(
            value=_as_int(validated["value"], field="value"),
            count=_as_int(validated["count"], field="count"),
        )

    def to_record(self) -> dict[str, object]:
        return {"value": self.value, "count": self.count}


@dataclass(frozen=True, slots=True)
class FieldPreview:
    """A compact display preview for a formed scalar field."""

    shape: tuple[int, int, int]
    runs: tuple[FieldPreviewRun, ...]

    def __post_init__(self) -> None:
        _validate_shape(self.shape, field="shape")
        if not self.runs:
            raise ObservationInspectionValidationError("preview runs must not be empty")
        expected = _shape_size(self.shape)
        actual = sum(run.count for run in self.runs)
        if actual != expected:
            raise ObservationInspectionValidationError(
                f"preview run length {actual} does not match shape size {expected}"
            )

    @classmethod
    def from_field(cls, field: FieldObservation) -> FieldPreview:
        """Build a deterministic byte-valued preview from a formed field."""

        return cls(shape=field.shape, runs=_rle_runs(_quantized_values(field.values)))

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> FieldPreview:
        try:
            validated = _field_preview_record.validate(record)
        except ValueError as error:
            raise ObservationInspectionValidationError(str(error)) from error
        return cls(
            shape=_shape(validated["shape"], field="shape"),
            runs=tuple(
                FieldPreviewRun.from_record(_as_mapping(run, field="runs"))
                for run in _as_sequence(validated["runs"], field="runs")
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "encoding": _preview_encoding,
            "shape": list(self.shape),
            "runs": [run.to_record() for run in self.runs],
        }


@dataclass(frozen=True, slots=True)
class ObservationInspectionRecord:
    """A read-only inspection summary for one formed observation."""

    id: ProtocolIdentifier
    benchmark_id: ProtocolIdentifier
    formed_observation: ArtifactReference
    formation_declaration: ArtifactReference
    materialization_plan: ArtifactReference
    sample_index: int
    component_sequence: tuple[int, ...]
    scale_assignment: AxisAssignment
    complexity_assignment: AxisAssignment
    resolution_assignment: AxisAssignment
    field_shape: tuple[int, int, int]
    field_digest: ContentDigest
    field_preview: FieldPreview | None = None
    outcome_id: str | None = None

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
            self.benchmark_id.require_unreleased()
        except ValueError as error:
            raise ObservationInspectionValidationError(str(error)) from error
        _require_kind(self.formed_observation, "formed-observation", "formed_observation")
        _require_kind(
            self.formation_declaration,
            "observation-formation-declaration",
            "formation_declaration",
        )
        _require_kind(self.materialization_plan, "materialization-plan", "materialization_plan")
        if type(self.sample_index) is not int:
            raise ObservationInspectionValidationError("sample_index must be an integer")
        if self.sample_index < 0:
            raise ObservationInspectionValidationError("sample_index must be nonnegative")
        if not self.component_sequence:
            raise ObservationInspectionValidationError("component_sequence must not be empty")
        if any(type(index) is not int or index < 0 for index in self.component_sequence):
            raise ObservationInspectionValidationError(
                "component_sequence values must be nonnegative integers"
            )
        _validate_shape(self.field_shape, field="field_shape")
        if self.field_preview is not None and self.field_preview.shape != self.field_shape:
            raise ObservationInspectionValidationError(
                "field_preview shape must match field_shape"
            )
        if self.outcome_id is not None and self.outcome_id == "":
            raise ObservationInspectionValidationError("outcome_id must be nonempty")

    @classmethod
    def from_formed_observation(
        cls,
        *,
        id: ProtocolIdentifier,
        observation: FormedObservation,
        materialization_plan: MaterializationPlan,
        sample_index: int,
        outcome_id: str | None = None,
        include_preview: bool = True,
    ) -> ObservationInspectionRecord:
        if observation.benchmark_id != materialization_plan.benchmark_id:
            raise ObservationInspectionValidationError(
                "observation benchmark_id does not match materialization plan"
            )
        if not observation.materialization_plan.matches_record(materialization_plan.to_record()):
            raise ObservationInspectionValidationError(
                "observation materialization_plan reference does not match plan"
            )
        return cls(
            id=id,
            benchmark_id=observation.benchmark_id,
            formed_observation=ArtifactReference(
                kind="formed-observation",
                protocol_id=observation.id,
                record_digest=observation.digest,
            ),
            formation_declaration=observation.formation_declaration,
            materialization_plan=observation.materialization_plan,
            sample_index=sample_index,
            component_sequence=observation.component_sequence,
            scale_assignment=materialization_plan.scale_assignment,
            complexity_assignment=materialization_plan.complexity_assignment,
            resolution_assignment=materialization_plan.resolution_assignment,
            field_shape=observation.field.shape,
            field_digest=observation.field.digest,
            field_preview=FieldPreview.from_field(observation.field) if include_preview else None,
            outcome_id=outcome_id,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ObservationInspectionRecord:
        try:
            validated = _inspection_record.validate(record)
        except ValueError as error:
            raise ObservationInspectionValidationError(str(error)) from error
        preview = validated.get("field_preview")
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            benchmark_id=_as_identifier(validated["benchmark_id"], field="benchmark_id"),
            formed_observation=ArtifactReference.from_record(
                _as_mapping(validated["formed_observation"], field="formed_observation")
            ),
            formation_declaration=ArtifactReference.from_record(
                _as_mapping(validated["formation_declaration"], field="formation_declaration")
            ),
            materialization_plan=ArtifactReference.from_record(
                _as_mapping(validated["materialization_plan"], field="materialization_plan")
            ),
            sample_index=_as_int(validated["sample_index"], field="sample_index"),
            component_sequence=tuple(
                _as_int(index, field="component_sequence")
                for index in _as_sequence(
                    validated["component_sequence"],
                    field="component_sequence",
                )
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
            field_shape=_shape(validated["field_shape"], field="field_shape"),
            field_digest=_digest(validated["field_digest"], field="field_digest"),
            field_preview=(
                None
                if preview is None
                else FieldPreview.from_record(_as_mapping(preview, field="field_preview"))
            ),
            outcome_id=_optional_string(validated.get("outcome_id"), field="outcome_id"),
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "benchmark_id": str(self.benchmark_id),
            "formed_observation": self.formed_observation.to_record(),
            "formation_declaration": self.formation_declaration.to_record(),
            "materialization_plan": self.materialization_plan.to_record(),
            "sample_index": self.sample_index,
            "component_sequence": list(self.component_sequence),
            "scale_assignment": self.scale_assignment.to_record(),
            "complexity_assignment": self.complexity_assignment.to_record(),
            "resolution_assignment": self.resolution_assignment.to_record(),
            "field_shape": list(self.field_shape),
            "field_digest": str(self.field_digest),
        }
        if self.field_preview is not None:
            record["field_preview"] = self.field_preview.to_record()
        if self.outcome_id is not None:
            record["outcome_id"] = self.outcome_id
        return record

    def to_bytes(self) -> bytes:
        return canonical_document_bytes(self.to_record()) + b"\n"


@dataclass(frozen=True, slots=True)
class ObservationInspectionDocument:
    """A loaded observation inspection record and its canonical digest."""

    inspection: ObservationInspectionRecord
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> ObservationInspectionDocument:
        try:
            record = load_object_document(data, description="observation inspection document")
        except ContentEncodingError as error:
            raise ObservationInspectionValidationError(str(error)) from error
        inspection = ObservationInspectionRecord.from_record(record)
        return cls(inspection=inspection, digest=inspection.digest)


def _quantized_values(values: tuple[float, ...]) -> tuple[int, ...]:
    return tuple(max(0, min(255, round(value * 255))) for value in values)


def _rle_runs(values: tuple[int, ...]) -> tuple[FieldPreviewRun, ...]:
    if not values:
        return ()
    runs: list[FieldPreviewRun] = []
    current = values[0]
    count = 1
    for value in values[1:]:
        if value == current:
            count += 1
            continue
        runs.append(FieldPreviewRun(value=current, count=count))
        current = value
        count = 1
    runs.append(FieldPreviewRun(value=current, count=count))
    return tuple(runs)


def _require_kind(reference: ArtifactReference, expected: str, field: str) -> None:
    if reference.kind != expected:
        raise ObservationInspectionValidationError(
            f"{field} reference must have kind {expected}"
        )


def _shape(value: object, *, field: str) -> tuple[int, int, int]:
    sequence = tuple(_as_int(item, field=field) for item in _as_sequence(value, field=field))
    _validate_shape(sequence, field=field)
    return cast(tuple[int, int, int], sequence)


def _validate_shape(value: tuple[int, ...], *, field: str) -> None:
    if len(value) != 3:
        raise ObservationInspectionValidationError(f"{field} must have rank 3")
    if any(type(size) is not int or size < 1 for size in value):
        raise ObservationInspectionValidationError(f"{field} sizes must be positive integers")


def _shape_size(shape: tuple[int, int, int]) -> int:
    channels, height, width = shape
    return channels * height * width


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise ObservationInspectionValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ObservationInspectionValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ObservationInspectionValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise ObservationInspectionValidationError(f"{field}: expected integer")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ObservationInspectionValidationError(f"{field}: expected string")
    return value


def _digest(value: object, *, field: str) -> ContentDigest:
    if not isinstance(value, str):
        raise ObservationInspectionValidationError(f"{field}: expected digest string")
    algorithm, separator, digest_hex = value.partition(":")
    if separator == "":
        raise ObservationInspectionValidationError(f"{field}: expected algorithm:digest")
    try:
        return ContentDigest(algorithm=algorithm, hex=digest_hex)
    except ContentEncodingError as error:
        raise ObservationInspectionValidationError(str(error)) from error
