"""Validation for mapping-shaped data objects.

A record is a concrete object that claims to satisfy a declared structural
contract. The contract supplies the protocol-facing rules; the record is the
data being checked against those rules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

from leibniz.contracts import ContractRuntimeSupport
from leibniz.identifiers import (
    IdentifierSyntaxError,
    ProtocolIdentifier,
    ProtocolName,
    SemanticVersion,
)
from leibniz.record_contracts import (
    FieldContract,
    FieldKind,
    RecordContract,
    RecordContractValidationError,
    ScalarKind,
    record_contract_set_from_contract,
)

__all__ = [
    "ContractRuntimeSupport",
    "FieldKind",
    "FieldSpec",
    "RecordExtractor",
    "RecordSpec",
    "RecordValidationError",
    "RecordValue",
    "RecordViolation",
    "ScalarKind",
    "ValidatedRecord",
    "record_specs_from_contract",
]

RecordValue: TypeAlias = object
ValidatedRecord: TypeAlias = Mapping[str, RecordValue]
_unset = object()


class RecordValidationError(ValueError):
    """Raised when a data object does not satisfy a record specification."""

    def __init__(self, violations: Sequence[RecordViolation]) -> None:
        self.violations = tuple(violations)
        message = "; ".join(violation.format() for violation in self.violations)
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class RecordViolation(ContractRuntimeSupport):
    """A single failure found while checking a data object."""

    path: tuple[str, ...]
    message: str

    @property
    def contract_runtime_role(self) -> str:
        return "record-violation"

    def format(self) -> str:
        location = ".".join(self.path) if self.path else "<record>"
        return f"{location}: {self.message}"


@dataclass(frozen=True, slots=True)
class FieldSpec(ContractRuntimeSupport):
    """Validation rule for one field in a mapping-shaped data object."""

    kind: FieldKind
    required: bool = True
    literal: object = _unset
    item: FieldSpec | None = None
    record: RecordSpec | None = None
    values: tuple[object, ...] | None = None

    @property
    def contract_runtime_role(self) -> str:
        return "field-spec"


@dataclass(frozen=True, slots=True)
class RecordSpec(ContractRuntimeSupport):
    """A structural contract for a mapping-shaped data object."""

    fields: Mapping[str, FieldSpec]
    allow_unknown: bool = False

    @property
    def contract_runtime_role(self) -> str:
        return "record-spec"

    def validate(self, record: Mapping[str, object]) -> ValidatedRecord:
        """Validate a data object against this record specification."""

        validated, violations = _validate_record(record=record, spec=self, path=())
        if violations:
            raise RecordValidationError(violations)
        return validated

    def collect_violations(self, record: Mapping[str, object]) -> tuple[RecordViolation, ...]:
        """Return specification violations without raising."""

        _, violations = _validate_record(record=record, spec=self, path=())
        return violations


@dataclass(frozen=True, slots=True)
class RecordExtractor(ContractRuntimeSupport):
    """Typed accessors for values already checked by record validation."""

    error_type: type[ValueError] = ValueError

    @property
    def contract_runtime_role(self) -> str:
        return "record-extractor"

    def string(self, value: object, field: str) -> str:
        return _require_string(value, field=field, error_type=self.error_type)

    def boolean(self, value: object, field: str) -> bool:
        return _require_boolean(value, field=field, error_type=self.error_type)

    def integer(self, value: object, field: str) -> int:
        return _require_integer(value, field=field, error_type=self.error_type)

    def identifier(self, value: object, field: str) -> ProtocolIdentifier:
        return _require_identifier(value, field=field, error_type=self.error_type)

    def mapping(self, value: object, field: str) -> Mapping[str, object]:
        return _require_mapping(value, field=field, error_type=self.error_type)

    def sequence(self, value: object, field: str) -> tuple[object, ...]:
        return _require_sequence(value, field=field, error_type=self.error_type)


def record_specs_from_contract(contract: Mapping[str, object]) -> dict[str, RecordSpec]:
    """Generate record specifications from an authored record-contract document."""

    try:
        contract_set = record_contract_set_from_contract(contract)
    except RecordContractValidationError as error:
        raise RecordValidationError(
            tuple(
                RecordViolation(path=violation.path, message=violation.message)
                for violation in error.violations
            )
        ) from error
    return _record_specs_from_contract_set(tuple(contract_set.records.values()))


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
        if (
            not field_violations
            and field_spec.values is not None
            and value not in field_spec.values
        ):
            field_violations = (
                RecordViolation(path=field_path, message="expected allowed value"),
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
        if spec.literal is _unset:
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
    if not isinstance(value, Mapping):
        return value, (RecordViolation(path=path, message="expected record"),)
    if spec.record is None:
        return cast(Mapping[str, object], value), ()
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


def _record_specs_from_contract_set(
    contracts: Sequence[RecordContract],
) -> dict[str, RecordSpec]:
    return {contract.name: _record_spec_from_contract(contract) for contract in contracts}


def _record_spec_from_contract(contract: RecordContract) -> RecordSpec:
    return RecordSpec(
        fields={
            field.name: _field_spec_from_contract(field)
            for field in contract.fields
            if field.name is not None
        },
        allow_unknown=contract.allow_unknown,
    )


def _field_spec_from_contract(field: FieldContract) -> FieldSpec:
    return FieldSpec(
        kind=field.kind,
        required=field.required,
        literal=field.literal_or(_unset),
        item=(
            _field_spec_from_contract(field.item)
            if field.item is not None
            else None
        ),
        values=field.values,
    )


def _require_string(
    value: object,
    *,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> str:
    if not isinstance(value, str):
        raise error_type(f"{field}: expected string")
    return value


def _require_boolean(
    value: object,
    *,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> bool:
    if not isinstance(value, bool):
        raise error_type(f"{field}: expected boolean")
    return value


def _require_integer(
    value: object,
    *,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> int:
    if type(value) is not int:
        raise error_type(f"{field}: expected integer")
    return value


def _require_identifier(
    value: object,
    *,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise error_type(f"{field}: expected parsed identifier")
    return value


def _require_mapping(
    value: object,
    *,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise error_type(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _require_sequence(
    value: object,
    *,
    field: str,
    error_type: type[ValueError] = ValueError,
) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise error_type(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)
