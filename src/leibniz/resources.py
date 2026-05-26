"""Declared resource accounting records for protocol artifacts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from leibniz._documents import ContentEncodingError, load_object_document
from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "ResourceAxis",
    "ResourcePayload",
    "ResourceReport",
    "ResourceReportDocument",
    "ResourceReportSet",
    "ResourceReportSetDocument",
    "ResourceValidationError",
]

_PayloadKind: TypeAlias = Literal["tensor", "table"]

_name = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_resource_axis_record = RecordSpec(
    fields={
        "name": FieldSpec(kind="string"),
        "value": FieldSpec(kind="number"),
        "unit": FieldSpec(kind="string"),
    }
)
_resource_payload_record = RecordSpec(
    fields={
        "name": FieldSpec(kind="string"),
        "kind": FieldSpec(kind="string"),
        "shape": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="integer"),
            required=False,
        ),
        "entries": FieldSpec(kind="integer", required=False),
        "entry_bits": FieldSpec(kind="integer", required=False),
        "element_bits": FieldSpec(kind="integer", required=False),
        "total_bits": FieldSpec(kind="integer"),
        "total_bytes": FieldSpec(kind="integer"),
    }
)
_resource_report_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "artifact": FieldSpec(kind="record"),
        "parameter_count": FieldSpec(kind="integer", required=False),
        "parameter_bits": FieldSpec(kind="integer", required=False),
        "parameter_bytes": FieldSpec(kind="integer", required=False),
        "payloads": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
            required=False,
        ),
        "inference_axes": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
            required=False,
        ),
    }
)
_resource_report_set_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "reports": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
    }
)


class ResourceValidationError(ValueError):
    """Raised when a resource accounting record is invalid."""


@dataclass(frozen=True, slots=True)
class ResourceAxis:
    """One declared nonnegative resource axis."""

    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        _validate_name(self.name, field="axis name")
        if not self.unit:
            raise ResourceValidationError("axis unit must be nonempty")
        _require_finite_nonnegative(self.value, field="axis value")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ResourceAxis:
        try:
            validated = _resource_axis_record.validate(record)
        except ValueError as error:
            raise ResourceValidationError(str(error)) from error
        return cls(
            name=_as_string(validated["name"], field="name"),
            value=_as_float(validated["value"], field="value"),
            unit=_as_string(validated["unit"], field="unit"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": _canonical_number(self.value),
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class ResourcePayload:
    """A declared tensor or table resource payload."""

    name: str
    kind: _PayloadKind
    total_bits: int
    total_bytes: int
    shape: tuple[int, ...] | None = None
    entries: int | None = None
    entry_bits: int | None = None
    element_bits: int | None = None

    def __post_init__(self) -> None:
        _validate_name(self.name, field="payload name")
        if self.kind == "tensor":
            self._validate_tensor_payload()
        elif self.kind == "table":
            self._validate_table_payload()
        else:
            raise ResourceValidationError(f"unsupported payload kind: {self.kind}")
        _require_nonnegative_integer(self.total_bits, field="total_bits")
        expected_bytes = _bytes_for_bits(self.total_bits)
        if self.total_bytes != expected_bytes:
            raise ResourceValidationError("total_bytes must derive from total_bits")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ResourcePayload:
        try:
            validated = _resource_payload_record.validate(record)
        except ValueError as error:
            raise ResourceValidationError(str(error)) from error
        return cls(
            name=_as_string(validated["name"], field="name"),
            kind=cast(_PayloadKind, _as_string(validated["kind"], field="kind")),
            shape=(
                _as_shape(validated["shape"], field="shape")
                if "shape" in validated
                else None
            ),
            entries=_as_optional_integer(validated.get("entries"), field="entries"),
            entry_bits=_as_optional_integer(validated.get("entry_bits"), field="entry_bits"),
            element_bits=_as_optional_integer(
                validated.get("element_bits"),
                field="element_bits",
            ),
            total_bits=_as_integer(validated["total_bits"], field="total_bits"),
            total_bytes=_as_integer(validated["total_bytes"], field="total_bytes"),
        )

    @classmethod
    def tensor(
        cls,
        *,
        name: str,
        shape: tuple[int, ...],
        element_bits: int,
    ) -> ResourcePayload:
        total_bits = math.prod(shape) * element_bits
        return cls(
            name=name,
            kind="tensor",
            shape=shape,
            element_bits=element_bits,
            total_bits=total_bits,
            total_bytes=_bytes_for_bits(total_bits),
        )

    @classmethod
    def table(
        cls,
        *,
        name: str,
        entries: int,
        entry_bits: int,
    ) -> ResourcePayload:
        total_bits = entries * entry_bits
        return cls(
            name=name,
            kind="table",
            entries=entries,
            entry_bits=entry_bits,
            total_bits=total_bits,
            total_bytes=_bytes_for_bits(total_bits),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "name": self.name,
            "kind": self.kind,
            "total_bits": self.total_bits,
            "total_bytes": self.total_bytes,
        }
        if self.shape is not None:
            record["shape"] = list(self.shape)
        if self.entries is not None:
            record["entries"] = self.entries
        if self.entry_bits is not None:
            record["entry_bits"] = self.entry_bits
        if self.element_bits is not None:
            record["element_bits"] = self.element_bits
        return record

    def _validate_tensor_payload(self) -> None:
        if self.shape is None:
            raise ResourceValidationError("tensor payloads require shape")
        _require_positive_shape(self.shape, field="shape")
        if self.element_bits is None:
            raise ResourceValidationError("tensor payloads require element_bits")
        _require_positive_integer(self.element_bits, field="element_bits")
        if self.entries is not None or self.entry_bits is not None:
            raise ResourceValidationError("tensor payloads must not declare table fields")
        if self.total_bits != math.prod(self.shape) * self.element_bits:
            raise ResourceValidationError(
                "tensor total_bits must equal shape product times element_bits"
            )

    def _validate_table_payload(self) -> None:
        if self.entries is None:
            raise ResourceValidationError("table payloads require entries")
        if self.entry_bits is None:
            raise ResourceValidationError("table payloads require entry_bits")
        _require_nonnegative_integer(self.entries, field="entries")
        _require_positive_integer(self.entry_bits, field="entry_bits")
        if self.shape is not None or self.element_bits is not None:
            raise ResourceValidationError("table payloads must not declare tensor fields")
        if self.total_bits != self.entries * self.entry_bits:
            raise ResourceValidationError("table total_bits must equal entries times entry_bits")


@dataclass(frozen=True, slots=True)
class ResourceReport:
    """Declared resource metadata for one referenced artifact."""

    id: ProtocolIdentifier
    artifact: ArtifactReference
    parameter_count: int | None = None
    parameter_bits: int | None = None
    parameter_bytes: int | None = None
    payloads: tuple[ResourcePayload, ...] = ()
    inference_axes: tuple[ResourceAxis, ...] = ()

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise ResourceValidationError(str(error)) from error
        if not str(self.id.name).startswith("resource-reports."):
            raise ResourceValidationError("id must be a valid resource report id")
        if self.parameter_count is not None:
            _require_nonnegative_integer(self.parameter_count, field="parameter_count")
        if self.parameter_bits is not None:
            _require_nonnegative_integer(self.parameter_bits, field="parameter_bits")
        if self.parameter_bytes is not None:
            if self.parameter_bits is None:
                raise ResourceValidationError("parameter_bits is required with parameter_bytes")
            if self.parameter_bytes != _bytes_for_bits(self.parameter_bits):
                raise ResourceValidationError("parameter_bytes must derive from parameter_bits")
        if self.parameter_count is None and self.parameter_bits is None and not self.payloads:
            raise ResourceValidationError("resource report must declare at least one resource")
        duplicate_payload = _first_duplicate(tuple(payload.name for payload in self.payloads))
        if duplicate_payload is not None:
            raise ResourceValidationError(f"duplicate payload name: {duplicate_payload}")
        duplicate_axis = _first_duplicate(tuple(axis.name for axis in self.inference_axes))
        if duplicate_axis is not None:
            raise ResourceValidationError(f"duplicate inference axis name: {duplicate_axis}")
        object.__setattr__(
            self,
            "payloads",
            tuple(sorted(self.payloads, key=lambda item: item.name)),
        )
        object.__setattr__(
            self,
            "inference_axes",
            tuple(sorted(self.inference_axes, key=lambda item: item.name)),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ResourceReport:
        try:
            validated = _resource_report_record.validate(record)
            payloads = tuple(
                ResourcePayload.from_record(_as_mapping(item, field="payloads"))
                for item in _as_sequence(validated.get("payloads", ()), field="payloads")
            )
            inference_axes = tuple(
                ResourceAxis.from_record(_as_mapping(item, field="inference_axes"))
                for item in _as_sequence(
                    validated.get("inference_axes", ()),
                    field="inference_axes",
                )
            )
        except ValueError as error:
            raise ResourceValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            artifact=ArtifactReference.from_record(
                _as_mapping(validated["artifact"], field="artifact")
            ),
            parameter_count=_as_optional_integer(
                validated.get("parameter_count"),
                field="parameter_count",
            ),
            parameter_bits=_as_optional_integer(
                validated.get("parameter_bits"),
                field="parameter_bits",
            ),
            parameter_bytes=_as_optional_integer(
                validated.get("parameter_bytes"),
                field="parameter_bytes",
            ),
            payloads=payloads,
            inference_axes=inference_axes,
        )

    @property
    def total_bits(self) -> int:
        return (self.parameter_bits or 0) + sum(payload.total_bits for payload in self.payloads)

    @property
    def total_bytes(self) -> int:
        return _bytes_for_bits(self.total_bits)

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "artifact": self.artifact.to_record(),
        }
        if self.parameter_count is not None:
            record["parameter_count"] = self.parameter_count
        if self.parameter_bits is not None:
            record["parameter_bits"] = self.parameter_bits
        if self.parameter_bytes is not None:
            record["parameter_bytes"] = self.parameter_bytes
        if self.payloads:
            record["payloads"] = [payload.to_record() for payload in self.payloads]
        if self.inference_axes:
            record["inference_axes"] = [axis.to_record() for axis in self.inference_axes]
        return record


@dataclass(frozen=True, slots=True)
class ResourceReportDocument:
    """A loaded resource report and its canonical digest."""

    report: ResourceReport
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> ResourceReportDocument:
        try:
            record = load_object_document(data, description="resource report document")
        except ContentEncodingError as error:
            raise ResourceValidationError(str(error)) from error
        report = ResourceReport.from_record(record)
        return cls(report=report, digest=report.digest)


@dataclass(frozen=True, slots=True)
class ResourceReportSet:
    """A deterministic collection of declared resource reports."""

    id: ProtocolIdentifier
    reports: tuple[ResourceReport, ...]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise ResourceValidationError(str(error)) from error
        if not str(self.id.name).startswith("resource-report-sets."):
            raise ResourceValidationError("id must be a valid resource report set id")
        if not self.reports:
            raise ResourceValidationError("reports must contain at least one resource report")
        duplicate_report = _first_duplicate(tuple(report.id for report in self.reports))
        if duplicate_report is not None:
            raise ResourceValidationError(f"duplicate resource report id: {duplicate_report}")
        object.__setattr__(
            self,
            "reports",
            tuple(sorted(self.reports, key=lambda item: str(item.id))),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ResourceReportSet:
        try:
            validated = _resource_report_set_record.validate(record)
            reports = tuple(
                ResourceReport.from_record(_as_mapping(item, field="reports"))
                for item in _as_sequence(validated["reports"], field="reports")
            )
        except ValueError as error:
            raise ResourceValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            reports=reports,
        )

    @property
    def total_bits(self) -> int:
        return sum(report.total_bits for report in self.reports)

    @property
    def total_bytes(self) -> int:
        return _bytes_for_bits(self.total_bits)

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "reports": [report.to_record() for report in self.reports],
        }


@dataclass(frozen=True, slots=True)
class ResourceReportSetDocument:
    """A loaded resource report set and its canonical digest."""

    report_set: ResourceReportSet
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> ResourceReportSetDocument:
        try:
            record = load_object_document(data, description="resource report set document")
        except ContentEncodingError as error:
            raise ResourceValidationError(str(error)) from error
        report_set = ResourceReportSet.from_record(record)
        return cls(report_set=report_set, digest=report_set.digest)


def _validate_name(value: str, *, field: str) -> None:
    if _name.fullmatch(value) is None:
        raise ResourceValidationError(f"{field} must be a stable lowercase name")


def _require_finite_nonnegative(value: float, *, field: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ResourceValidationError(f"{field} must be finite and nonnegative")


def _require_nonnegative_integer(value: int, *, field: str) -> None:
    if value < 0:
        raise ResourceValidationError(f"{field} must be nonnegative")


def _require_positive_integer(value: int, *, field: str) -> None:
    if value <= 0:
        raise ResourceValidationError(f"{field} must be positive")


def _require_positive_shape(shape: tuple[int, ...], *, field: str) -> None:
    if not shape:
        raise ResourceValidationError(f"{field} must contain at least one axis")
    if any(axis <= 0 for axis in shape):
        raise ResourceValidationError(f"{field} axes must be positive integers")


def _bytes_for_bits(bits: int) -> int:
    return (bits + 7) // 8


def _canonical_number(value: float) -> int | float:
    if value.is_integer():
        return int(value)
    return value


def _as_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ResourceValidationError(f"{field}: expected string")
    return value


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise ResourceValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ResourceValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ResourceValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_integer(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ResourceValidationError(f"{field}: expected parsed integer")
    return value


def _as_optional_integer(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _as_integer(value, field=field)


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, float):
        return value
    raise ResourceValidationError(f"{field}: expected parsed number")


def _as_shape(value: object, *, field: str) -> tuple[int, ...]:
    shape = tuple(_as_integer(axis, field=field) for axis in _as_sequence(value, field=field))
    _require_positive_shape(shape, field=field)
    return shape


def _first_duplicate(values: tuple[object, ...]) -> object | None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
