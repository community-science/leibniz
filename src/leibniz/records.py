"""Validation for mapping-shaped data objects.

A record is a concrete object that claims to satisfy a declared structural
contract. The contract supplies the protocol-facing rules; the record is the
data being checked against those rules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from leibniz.identifiers import (
    IdentifierSyntaxError,
    ProtocolIdentifier,
    ProtocolName,
    SemanticVersion,
)

ScalarKind: TypeAlias = Literal["boolean", "integer", "number", "string"]
FieldKind: TypeAlias = ScalarKind | Literal[
    "identifier",
    "literal",
    "name",
    "record",
    "sequence",
    "version",
]
RecordValue: TypeAlias = object
ValidatedRecord: TypeAlias = Mapping[str, RecordValue]
_UNSET = object()


class RecordValidationError(ValueError):
    """Raised when a data object does not satisfy a record specification."""

    def __init__(self, violations: Sequence[RecordViolation]) -> None:
        self.violations = tuple(violations)
        message = "; ".join(violation.format() for violation in self.violations)
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RecordViolation:
    """A single failure found while checking a data object."""

    path: tuple[str, ...]
    message: str

    def format(self) -> str:
        location = ".".join(self.path) if self.path else "<record>"
        return f"{location}: {self.message}"


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """Validation rule for one field in a mapping-shaped data object."""

    kind: FieldKind
    required: bool = True
    literal: object = _UNSET
    item: FieldSpec | None = None
    record: RecordSpec | None = None


@dataclass(frozen=True, slots=True)
class RecordSpec:
    """A structural contract for a mapping-shaped data object."""

    fields: Mapping[str, FieldSpec]
    allow_unknown: bool = False


def validate_record(record: Mapping[str, object], spec: RecordSpec) -> ValidatedRecord:
    """Validate a data object against a record specification."""

    validated, violations = _validate_record(record=record, spec=spec, path=())
    if violations:
        raise RecordValidationError(violations)
    return validated


def collect_record_violations(
    record: Mapping[str, object],
    spec: RecordSpec,
) -> tuple[RecordViolation, ...]:
    """Return specification violations without raising."""

    _, violations = _validate_record(record=record, spec=spec, path=())
    return violations


def required(
    kind: FieldKind,
    *,
    literal: object = _UNSET,
    item: FieldSpec | None = None,
    record: RecordSpec | None = None,
) -> FieldSpec:
    return FieldSpec(kind=kind, required=True, literal=literal, item=item, record=record)


def optional(
    kind: FieldKind,
    *,
    literal: object = _UNSET,
    item: FieldSpec | None = None,
    record: RecordSpec | None = None,
) -> FieldSpec:
    return FieldSpec(kind=kind, required=False, literal=literal, item=item, record=record)


def _validate_record(
    *,
    record: Mapping[str, object],
    spec: RecordSpec,
    path: tuple[str, ...],
) -> tuple[dict[str, RecordValue], tuple[RecordViolation, ...]]:
    validated: dict[str, RecordValue] = {}
    violations: list[RecordViolation] = []

    for field_name, field_spec in spec.fields.items():
        field_path = (*path, field_name)
        if field_name not in record:
            if field_spec.required:
                violations.append(
                    RecordViolation(path=field_path, message="missing required field")
                )
            continue
        value, field_violations = _validate_value(
            value=record[field_name],
            spec=field_spec,
            path=field_path,
        )
        violations.extend(field_violations)
        if not field_violations:
            validated[field_name] = value

    if not spec.allow_unknown:
        for field_name in record:
            if field_name not in spec.fields:
                violations.append(
                    RecordViolation(path=(*path, field_name), message="unknown field")
                )

    return validated, tuple(violations)


def _validate_value(
    *,
    value: object,
    spec: FieldSpec,
    path: tuple[str, ...],
) -> tuple[RecordValue, tuple[RecordViolation, ...]]:
    if spec.kind == "literal":
        if spec.literal is _UNSET:
            return value, (RecordViolation(path=path, message="missing literal value"),)
        if value == spec.literal:
            return value, ()
        return value, (RecordViolation(path=path, message=f"expected literal {spec.literal!r}"),)

    if spec.kind == "boolean":
        if isinstance(value, bool):
            return value, ()
        return value, (RecordViolation(path=path, message="expected boolean"),)

    if spec.kind == "integer":
        if isinstance(value, int) and not isinstance(value, bool):
            return value, ()
        return value, (RecordViolation(path=path, message="expected integer"),)

    if spec.kind == "number":
        if isinstance(value, int) and not isinstance(value, bool):
            return value, ()
        if isinstance(value, float) and math.isfinite(value):
            return value, ()
        return value, (RecordViolation(path=path, message="expected finite number"),)

    if spec.kind == "string":
        if isinstance(value, str):
            return value, ()
        return value, (RecordViolation(path=path, message="expected string"),)

    if spec.kind == "identifier":
        return _parse_identifier(value=value, path=path)

    if spec.kind == "name":
        return _parse_name(value=value, path=path)

    if spec.kind == "version":
        return _parse_version(value=value, path=path)

    if spec.kind == "record":
        return _parse_record(value=value, spec=spec, path=path)

    if spec.kind == "sequence":
        return _parse_sequence(value=value, spec=spec, path=path)

    return value, (RecordViolation(path=path, message=f"unsupported field kind {spec.kind!r}"),)


def _parse_identifier(
    *,
    value: object,
    path: tuple[str, ...],
) -> tuple[RecordValue, tuple[RecordViolation, ...]]:
    if not isinstance(value, str):
        return value, (RecordViolation(path=path, message="expected identifier string"),)
    try:
        return ProtocolIdentifier.parse(value), ()
    except IdentifierSyntaxError as error:
        return value, (RecordViolation(path=path, message=str(error)),)


def _parse_name(
    *,
    value: object,
    path: tuple[str, ...],
) -> tuple[RecordValue, tuple[RecordViolation, ...]]:
    if not isinstance(value, str):
        return value, (RecordViolation(path=path, message="expected name string"),)
    try:
        return ProtocolName.parse(value), ()
    except IdentifierSyntaxError as error:
        return value, (RecordViolation(path=path, message=str(error)),)


def _parse_version(
    *,
    value: object,
    path: tuple[str, ...],
) -> tuple[RecordValue, tuple[RecordViolation, ...]]:
    if not isinstance(value, str):
        return value, (RecordViolation(path=path, message="expected version string"),)
    try:
        return SemanticVersion.parse(value), ()
    except IdentifierSyntaxError as error:
        return value, (RecordViolation(path=path, message=str(error)),)


def _parse_record(
    *,
    value: object,
    spec: FieldSpec,
    path: tuple[str, ...],
) -> tuple[RecordValue, tuple[RecordViolation, ...]]:
    if spec.record is None:
        return value, (RecordViolation(path=path, message="missing nested record specification"),)
    if not isinstance(value, Mapping):
        return value, (RecordViolation(path=path, message="expected record"),)
    return _validate_record(record=cast(Mapping[str, object], value), spec=spec.record, path=path)


def _parse_sequence(
    *,
    value: object,
    spec: FieldSpec,
    path: tuple[str, ...],
) -> tuple[RecordValue, tuple[RecordViolation, ...]]:
    if spec.item is None:
        return value, (RecordViolation(path=path, message="missing sequence item specification"),)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        return value, (RecordViolation(path=path, message="expected sequence"),)

    parsed: list[RecordValue] = []
    violations: list[RecordViolation] = []
    sequence = cast(Sequence[object], value)
    for index, item in enumerate(sequence):
        item_value, item_violations = _validate_value(
            value=item,
            spec=spec.item,
            path=(*path, str(index)),
        )
        violations.extend(item_violations)
        if not item_violations:
            parsed.append(item_value)
    return tuple(parsed), tuple(violations)
