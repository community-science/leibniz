"""Typed model for authored record-contract documents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Literal, TypeAlias, cast

from leibniz.documents import document_filename_suffix, load_object_document

__all__ = [
    "ContractRuntimeSupport",
    "FieldContract",
    "FieldKind",
    "RecordContract",
    "RecordContractSet",
    "RecordContractValidationError",
    "RecordContractViolation",
    "ScalarKind",
    "TypeScriptRecordModule",
    "record_contract_set_from_contract",
    "record_contract_set_from_package",
    "typescript_literal",
]

ScalarKind: TypeAlias = Literal["boolean", "integer", "number", "string"]
FieldKind: TypeAlias = ScalarKind | Literal[
    "identifier",
    "literal",
    "name",
    "record",
    "sequence",
    "version",
]
_field_kinds = frozenset(
    {
        "boolean",
        "identifier",
        "integer",
        "literal",
        "name",
        "number",
        "record",
        "sequence",
        "string",
        "version",
    }
)
_missing_literal = object()


class ContractRuntimeSupport(ABC):
    """Marker base for generic handwritten record contract runtime support."""

    @property
    @abstractmethod
    def contract_runtime_role(self) -> str:
        """Return the generic record-runtime role this object serves."""


@dataclass(frozen=True, slots=True)
class RecordContractViolation:
    """A single failure found while checking a contract document."""

    path: tuple[str, ...]
    message: str


class RecordContractValidationError(ValueError):
    """Raised when an authored record-contract document is invalid."""

    def __init__(self, violations: Sequence[RecordContractViolation]) -> None:
        self.violations = tuple(violations)
        message = "; ".join(_format_violation(violation) for violation in self.violations)
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class FieldContract(ContractRuntimeSupport):
    """A parsed field declaration from an authored record contract."""

    kind: FieldKind
    name: str | None = None
    required: bool = True
    literal: object = _missing_literal
    item: FieldContract | None = None
    values: tuple[object, ...] | None = None

    @property
    def contract_runtime_role(self) -> str:
        return "field-contract"

    def literal_or(self, default: object) -> object:
        """Return the declared literal or a caller-owned sentinel."""

        return default if self.literal is _missing_literal else self.literal


@dataclass(frozen=True, slots=True)
class RecordContract(ContractRuntimeSupport):
    """A parsed record declaration from an authored record contract."""

    name: str
    fields: tuple[FieldContract, ...]
    allow_unknown: bool = False

    @property
    def contract_runtime_role(self) -> str:
        return "record-contract"

    def to_typescript_module(
        self,
        *,
        exported_type: str,
        parser_name: str,
        error_name: str,
        literal_expressions: Mapping[str, str] | None = None,
        imports: str = "",
    ) -> str:
        """Generate a TypeScript record type and parser for this contract."""

        return _typescript_record_contract_module(
            TypeScriptRecordModule(
                record=self,
                exported_type=exported_type,
                parser_name=parser_name,
                error_name=error_name,
                literal_expressions=literal_expressions or {},
                imports=imports,
            )
        )


@dataclass(frozen=True, slots=True)
class TypeScriptRecordModule:
    """Configuration for TypeScript generated directly from a record contract."""

    record: RecordContract
    exported_type: str
    parser_name: str
    error_name: str
    literal_expressions: Mapping[str, str]
    imports: str = ""


@dataclass(frozen=True, slots=True)
class RecordContractSet(ContractRuntimeSupport):
    """A parsed authored record-contract document."""

    records: Mapping[str, RecordContract]

    @property
    def contract_runtime_role(self) -> str:
        return "record-contract-set"

    def require_record(self, name: str) -> RecordContract:
        try:
            return self.records[name]
        except KeyError as error:
            raise ValueError(f"missing contract record: {name}") from error


def record_contract_set_from_contract(contract: Mapping[str, object]) -> RecordContractSet:
    """Parse an authored record-contract document into a typed contract model."""

    if contract.get("format") != "leibniz.record-contract-set":
        raise RecordContractValidationError(
            (
                RecordContractViolation(
                    path=("format",),
                    message="expected record-contract set",
                ),
            )
        )
    if contract.get("format_version") != 1:
        raise RecordContractValidationError(
            (RecordContractViolation(path=("format_version",), message="expected version 1"),)
        )
    records = _require_contract_sequence(contract.get("records"), path=("records",))
    parsed: dict[str, RecordContract] = {}
    for index, record in enumerate(records):
        record_path = ("records", str(index))
        if not isinstance(record, Mapping):
            raise RecordContractValidationError(
                (RecordContractViolation(path=record_path, message="expected record"),)
            )
        record_map = cast(Mapping[str, object], record)
        name = _require_contract_string(record_map.get("name"), path=(*record_path, "name"))
        if name in parsed:
            raise RecordContractValidationError(
                (RecordContractViolation(path=(*record_path, "name"), message="duplicate record"),)
            )
        fields = _require_contract_sequence(
            record_map.get("fields"),
            path=(*record_path, "fields"),
        )
        parsed[name] = RecordContract(
            name=name,
            fields=_field_contracts_from_contract(fields=fields, path=(*record_path, "fields")),
            allow_unknown=_optional_contract_boolean(
                record_map.get("allow_unknown", False),
                path=(*record_path, "allow_unknown"),
            ),
        )
    return RecordContractSet(records=parsed)


def record_contract_set_from_package(
    package: str,
    name: str,
    *,
    description: str,
) -> RecordContractSet:
    """Parse a bundled authored record-contract document."""

    contract = load_object_document(
        files(package).joinpath(f"{name}{document_filename_suffix()}").read_bytes(),
        description=description,
    )
    return record_contract_set_from_contract(contract)


def _field_contract_from_contract(
    field: object,
    *,
    path: tuple[str, ...],
) -> FieldContract:
    if not isinstance(field, Mapping):
        raise RecordContractValidationError(
            (RecordContractViolation(path=path, message="expected record"),)
        )
    field_map = cast(Mapping[str, object], field)
    return FieldContract(
        name=_require_contract_string(field_map.get("name"), path=(*path, "name")),
        kind=_field_kind_from_contract(field_map.get("kind"), path=(*path, "kind")),
        required=_optional_contract_boolean(
            field_map.get("required", True),
            path=(*path, "required"),
        ),
        item=(
            _anonymous_field_contract_from_contract(field_map["item"], path=(*path, "item"))
            if "item" in field_map
            else None
        ),
        literal=field_map.get("literal", _missing_literal),
        values=_optional_contract_values(field_map.get("values"), path=(*path, "values")),
    )


def _anonymous_field_contract_from_contract(
    field: object,
    *,
    path: tuple[str, ...],
) -> FieldContract:
    if not isinstance(field, Mapping):
        raise RecordContractValidationError(
            (RecordContractViolation(path=path, message="expected record"),)
        )
    field_map = cast(Mapping[str, object], field)
    return FieldContract(
        kind=_field_kind_from_contract(field_map.get("kind"), path=(*path, "kind")),
        item=(
            _anonymous_field_contract_from_contract(field_map["item"], path=(*path, "item"))
            if "item" in field_map
            else None
        ),
        literal=field_map.get("literal", _missing_literal),
        values=_optional_contract_values(field_map.get("values"), path=(*path, "values")),
    )


def _field_contracts_from_contract(
    *,
    fields: Sequence[object],
    path: tuple[str, ...],
) -> tuple[FieldContract, ...]:
    parsed: list[FieldContract] = []
    field_names: set[str] = set()
    for offset, field in enumerate(fields):
        field_contract = _field_contract_from_contract(field, path=(*path, str(offset)))
        assert field_contract.name is not None
        if field_contract.name in field_names:
            raise RecordContractValidationError(
                (
                    RecordContractViolation(
                        path=(*path, str(offset), "name"),
                        message="duplicate field",
                    ),
                )
            )
        field_names.add(field_contract.name)
        parsed.append(field_contract)
    return tuple(parsed)


def _field_kind_from_contract(value: object, *, path: tuple[str, ...]) -> FieldKind:
    kind = _require_contract_string(value, path=path)
    if kind not in _field_kinds:
        raise RecordContractValidationError(
            (RecordContractViolation(path=path, message="unsupported field kind"),)
        )
    return cast(FieldKind, kind)


def _require_contract_sequence(value: object, *, path: tuple[str, ...]) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise RecordContractValidationError(
            (RecordContractViolation(path=path, message="expected sequence"),)
        )
    return cast(Sequence[object], value)


def _require_contract_string(value: object, *, path: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise RecordContractValidationError(
            (RecordContractViolation(path=path, message="expected string"),)
        )
    return value


def _optional_contract_boolean(value: object, *, path: tuple[str, ...]) -> bool:
    if not isinstance(value, bool):
        raise RecordContractValidationError(
            (RecordContractViolation(path=path, message="expected boolean"),)
        )
    return value


def _optional_contract_values(
    value: object,
    *,
    path: tuple[str, ...],
) -> tuple[object, ...] | None:
    if value is None:
        return None
    return tuple(_require_contract_sequence(value, path=path))


def _format_violation(violation: RecordContractViolation) -> str:
    location = ".".join(violation.path) if violation.path else "<record>"
    return f"{location}: {violation.message}"


def typescript_literal(value: object) -> str:
    """Render a small Python value as a TypeScript literal expression."""

    return _typescript_literal_lines(value, indent=0)


def _typescript_record_contract_module(config: TypeScriptRecordModule) -> str:
    named_unions = _named_union_fields(fields=config.record.fields)
    imports = config.imports.rstrip()
    type_aliases = "\n".join(
        _typescript_union_alias(
            name=_union_name(field_name=field_name, exported_type=config.exported_type),
            values=values,
        )
        for field_name, values in named_unions.items()
    )
    record_type = _typescript_record_type(
        exported_type=config.exported_type,
        fields=config.record.fields,
        named_unions=named_unions,
        literal_expressions=config.literal_expressions,
    )
    parser = _typescript_record_parser(
        record=config.record,
        exported_type=config.exported_type,
        parser_name=config.parser_name,
        error_name=config.error_name,
        named_unions=named_unions,
        literal_expressions=config.literal_expressions,
    )
    helpers = _typescript_record_parser_helpers(
        error_name=config.error_name,
        required_helpers=_required_typescript_helpers(record=config.record),
    )
    sections = [
        section
        for section in (imports, type_aliases, record_type, parser, helpers)
        if section
    ]
    return "\n\n".join(sections) + "\n"


def _typescript_literal_lines(value: object, *, indent: int) -> str:
    prefix = " " * indent
    child_indent = indent + 2
    child_prefix = " " * child_indent
    if isinstance(value, str):
        return _typescript_string(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        if not mapping:
            return "{}"
        lines = ["{"]
        items = sorted(mapping.items())
        for index, (key, item) in enumerate(items):
            suffix = "," if index < len(items) - 1 else ""
            rendered = _typescript_literal_lines(item, indent=child_indent)
            lines.append(f"{child_prefix}{_typescript_string(str(key))}: {rendered}{suffix}")
        lines.append(f"{prefix}}}")
        return "\n".join(lines)
    raise TypeError(f"unsupported generated TypeScript value: {type(value).__name__}")


def _typescript_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _typescript_record_type(
    *,
    exported_type: str,
    fields: tuple[FieldContract, ...],
    named_unions: Mapping[str, tuple[str, ...]],
    literal_expressions: Mapping[str, str],
) -> str:
    lines = [f"export type {exported_type} = {{"]
    for field in fields:
        field_name = _required_field_name(field)
        marker = "?" if not field.required else ""
        field_type = _typescript_field_type(
            field,
            exported_type=exported_type,
            named_unions=named_unions,
            literal_expressions=literal_expressions,
        )
        lines.append(f"  {field_name}{marker}: {field_type};")
    lines.append("};")
    return "\n".join(lines)


def _typescript_record_parser(
    *,
    record: RecordContract,
    exported_type: str,
    parser_name: str,
    error_name: str,
    named_unions: Mapping[str, tuple[str, ...]],
    literal_expressions: Mapping[str, str],
) -> str:
    known_fields = ", ".join(
        _typescript_string(_required_field_name(field))
        for field in record.fields
    )
    lines = [
        f"export function {parser_name}(value: unknown, path: string): {exported_type} {{",
        "  const record = requireRecord(value, path);",
    ]
    if not record.allow_unknown:
        lines.append(f"  rejectUnknownFields(record, path, [{known_fields}]);")
    lines.append("  return {")
    for field in record.fields:
        field_name = _required_field_name(field)
        rendered = _typescript_field_parser(
            field,
            exported_type=exported_type,
            named_unions=named_unions,
            literal_expressions=literal_expressions,
        )
        if field.required:
            lines.append(f"    {field_name}: {rendered},")
        else:
            lines.extend(
                [
                    f"    {field_name}:",
                    f"      record.{field_name} === undefined",
                    "        ? undefined",
                    f"        : {rendered},",
                ]
            )
    lines.extend(["  };", "}"])
    parser_sections = ["\n".join(lines)]
    parser_sections.extend(
        _typescript_union_parser(
            name=_union_name(field_name=field_name, exported_type=exported_type),
            values=values,
            error_name=error_name,
        )
        for field_name, values in named_unions.items()
    )
    return "\n\n".join(parser_sections)


def _typescript_union_alias(*, name: str, values: tuple[str, ...]) -> str:
    return f"export type {name} = {' | '.join(_typescript_string(value) for value in values)};"


def _typescript_union_parser(
    *,
    name: str,
    values: tuple[str, ...],
    error_name: str,
) -> str:
    parser_name = f"parse{name}"
    allowed = ", ".join(_typescript_string(value) for value in values)
    return f"""function {parser_name}(value: unknown, path: string): {name} {{
  const parsed = requireString(value, path);
  if (![{allowed}].includes(parsed)) {{
    throw new {error_name}(`${{path}}: expected {_humanize_type_name(name)}`);
  }}
  return parsed as {name};
}}"""


def _typescript_record_parser_helpers(
    *,
    error_name: str,
    required_helpers: set[str],
) -> str:
    sections = [
        f"""class {error_name} extends Error {{
  constructor(message: string) {{
    super(message);
    this.name = '{error_name}';
  }}
}}""",
        f"""function requireRecord(value: unknown, path: string): Record<string, unknown> {{
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {{
    throw new {error_name}(`${{path}}: expected record`);
  }}
  return value as Record<string, unknown>;
}}""",
    ]
    if "rejectUnknownFields" in required_helpers:
        sections.append(
            f"""function rejectUnknownFields(
  record: Record<string, unknown>,
  path: string,
  fields: string[],
): void {{
  const known = new Set(fields);
  for (const field of Object.keys(record)) {{
    if (!known.has(field)) {{
      throw new {error_name}(`${{path}}.${{field}}: unknown field`);
    }}
  }}
}}"""
        )
    if "array" in required_helpers:
        sections.append(
            f"""function requireArray(value: unknown, path: string): unknown[] {{
  if (!Array.isArray(value)) {{
    throw new {error_name}(`${{path}}: expected array`);
  }}
  return value;
}}"""
        )
    if "string" in required_helpers:
        sections.append(
            f"""function requireString(value: unknown, path: string): string {{
  if (typeof value !== 'string') {{
    throw new {error_name}(`${{path}}: expected string`);
  }}
  return value;
}}"""
        )
    if "boolean" in required_helpers:
        sections.append(
            f"""function requireBoolean(value: unknown, path: string): boolean {{
  if (typeof value !== 'boolean') {{
    throw new {error_name}(`${{path}}: expected boolean`);
  }}
  return value;
}}"""
        )
    if "integer" in required_helpers:
        sections.append(
            f"""function requireInteger(value: unknown, path: string): number {{
  if (typeof value !== 'number' || !Number.isInteger(value)) {{
    throw new {error_name}(`${{path}}: expected integer`);
  }}
  return value;
}}"""
        )
    if "number" in required_helpers:
        sections.append(
            f"""function requireNumber(value: unknown, path: string): number {{
  if (typeof value !== 'number' || !Number.isFinite(value)) {{
    throw new {error_name}(`${{path}}: expected number`);
  }}
  return value;
}}"""
        )
    if "literal" in required_helpers:
        sections.append(
            f"""function requireLiteral<T extends string | number>(
  value: unknown,
  path: string,
  expected: T,
): T {{
  if (value !== expected) {{
    throw new {error_name}(`${{path}}: expected ${{String(expected)}}`);
  }}
  return expected;
}}"""
        )
    return "\n\n".join(sections)


def _required_typescript_helpers(*, record: RecordContract) -> set[str]:
    helpers: set[str] = set()
    if not record.allow_unknown:
        helpers.add("rejectUnknownFields")
    for field in record.fields:
        _collect_required_typescript_helpers(field=field, helpers=helpers)
    return helpers


def _collect_required_typescript_helpers(
    *,
    field: FieldContract,
    helpers: set[str],
) -> None:
    if field.values is not None:
        helpers.add("string")
        return
    if field.kind in {"identifier", "name", "string", "version"}:
        helpers.add("string")
    elif field.kind == "integer":
        helpers.add("integer")
    elif field.kind == "number":
        helpers.add("number")
    elif field.kind == "boolean":
        helpers.add("boolean")
    elif field.kind == "literal":
        helpers.add("literal")
    elif field.kind == "sequence":
        helpers.add("array")
        if field.item is not None:
            _collect_required_typescript_helpers(field=field.item, helpers=helpers)


def _typescript_field_type(
    field: FieldContract,
    *,
    exported_type: str,
    named_unions: Mapping[str, tuple[str, ...]],
    literal_expressions: Mapping[str, str],
) -> str:
    field_name = field.name or ""
    if field_name in named_unions:
        return _union_name(field_name=field_name, exported_type=exported_type)
    if field_name in literal_expressions:
        return f"typeof {literal_expressions[field_name]}"
    if field.kind == "literal":
        return _typescript_literal_type(field.literal)
    if field.kind in {"identifier", "name", "string", "version"}:
        return "string"
    if field.kind in {"integer", "number"}:
        return "number"
    if field.kind == "boolean":
        return "boolean"
    if field.kind == "record":
        return "Record<string, unknown>"
    if field.kind == "sequence":
        if field.item is None:
            raise ValueError(f"{field_name}.item: missing sequence item contract")
        item_type = _typescript_field_type(
            field.item,
            exported_type=exported_type,
            named_unions={},
            literal_expressions={},
        )
        return f"{item_type}[]"
    raise ValueError(f"{field_name}: unsupported contract field kind: {field.kind}")


def _typescript_field_parser(
    field: FieldContract,
    *,
    exported_type: str,
    named_unions: Mapping[str, tuple[str, ...]],
    literal_expressions: Mapping[str, str],
) -> str:
    field_name = _required_field_name(field)
    value = f"record.{field_name}"
    path = f"`${{path}}.{field_name}`"
    return _typescript_value_parser(
        field,
        value=value,
        path=path,
        exported_type=exported_type,
        named_unions=named_unions,
        literal_expressions=literal_expressions,
    )


def _typescript_value_parser(
    field: FieldContract,
    *,
    value: str,
    path: str,
    exported_type: str,
    named_unions: Mapping[str, tuple[str, ...]],
    literal_expressions: Mapping[str, str],
) -> str:
    field_name = field.name or ""
    if field_name in named_unions:
        return (
            f"parse{_union_name(field_name=field_name, exported_type=exported_type)}"
            f"({value}, {path})"
        )
    if field_name in literal_expressions:
        return f"requireLiteral({value}, {path}, {literal_expressions[field_name]})"
    if field.kind == "literal":
        return f"requireLiteral({value}, {path}, {typescript_literal(field.literal)})"
    if field.kind in {"identifier", "name", "string", "version"}:
        return f"requireString({value}, {path})"
    if field.kind == "integer":
        return f"requireInteger({value}, {path})"
    if field.kind == "number":
        return f"requireNumber({value}, {path})"
    if field.kind == "boolean":
        return f"requireBoolean({value}, {path})"
    if field.kind == "record":
        return f"requireRecord({value}, {path})"
    if field.kind == "sequence":
        if field.item is None:
            raise ValueError(f"{field_name}.item: missing sequence item contract")
        item_parser = _typescript_value_parser(
            field.item,
            value="item",
            path=_typescript_index_path(path),
            exported_type=exported_type,
            named_unions={},
            literal_expressions={},
        )
        return f"requireArray({value}, {path}).map((item, index) =>\n    {item_parser},\n  )"
    raise ValueError(f"{field_name}: unsupported contract field kind: {field.kind}")


def _named_union_fields(*, fields: tuple[FieldContract, ...]) -> dict[str, tuple[str, ...]]:
    unions: dict[str, tuple[str, ...]] = {}
    for field in fields:
        if field.values is None:
            continue
        unions[_required_field_name(field)] = tuple(str(value) for value in field.values)
    return unions


def _required_field_name(field: FieldContract) -> str:
    if field.name is None:
        raise ValueError("named record field is missing name")
    return field.name


def _union_name(*, field_name: str, exported_type: str) -> str:
    record_stem = exported_type.removesuffix("Record")
    return record_stem + _pascal_case(field_name)


def _pascal_case(value: str) -> str:
    return "".join(part.capitalize() for part in value.split("_") if part)


def _typescript_literal_type(value: object) -> str:
    if isinstance(value, str):
        return _typescript_string(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if value is None or value is _missing_literal:
        return "null"
    raise ValueError(f"unsupported TypeScript literal type: {value!r}")


def _humanize_type_name(name: str) -> str:
    return " ".join(part.lower() for part in _split_pascal_case(name))


def _typescript_index_path(path: str) -> str:
    if path.startswith("`") and path.endswith("`"):
        return path[:-1] + ".${index}`"
    raise ValueError(f"unsupported generated TypeScript path expression: {path}")


def _split_pascal_case(value: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    for index, character in enumerate(value):
        if index > 0 and character.isupper():
            parts.append(value[start:index])
            start = index
    parts.append(value[start:])
    return tuple(part for part in parts if part)
