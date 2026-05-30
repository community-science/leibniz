"""Generate small console web-source modules from Python-owned protocol metadata."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import cast

from leibniz.console.protocol import (
    console_protocol_format_versions,
    console_protocol_formats,
)
from leibniz.documents import document_filename_suffix, load_object_document

__all__ = [
    "generated_console_protocol_module",
    "generated_console_result_view_records_module",
    "generated_console_work_queue_records_module",
    "write_generated_console_protocol_module",
    "write_generated_console_result_view_records_module",
    "write_generated_console_work_queue_records_module",
    "write_generated_console_web_modules",
]

_generated_protocol_module_path = (
    Path(__file__).parent / "_web_src" / "src" / "generated" / "protocolVocabulary.ts"
)
_generated_result_view_records_module_path = (
    Path(__file__).parent / "_web_src" / "src" / "generated" / "resultViewRecords.ts"
)
_generated_work_queue_records_module_path = (
    Path(__file__).parent / "_web_src" / "src" / "generated" / "workQueueRecords.ts"
)


@dataclass(frozen=True, slots=True)
class _TypeScriptContractModule:
    contract_name: str
    record_name: str
    exported_type: str
    parser_name: str
    error_name: str
    literal_expressions: Mapping[str, str]
    imports: str = ""


def generated_console_protocol_module() -> str:
    """Return the generated TypeScript console protocol vocabulary module."""

    formats = _console_protocol_formats()
    versions = _console_protocol_format_versions()
    return (
        "export const consoleProtocolFormats = "
        f"{_typescript_literal(formats)} as const;\n\n"
        "export const consoleProtocolFormatVersions = "
        f"{_typescript_literal(versions)} as const;\n"
    )


def write_generated_console_protocol_module(
    path: Path = _generated_protocol_module_path,
) -> Path:
    """Write the generated TypeScript console protocol vocabulary module."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generated_console_protocol_module(), encoding="utf-8")
    return path


def generated_console_result_view_records_module() -> str:
    """Return the generated TypeScript result-view record parser module."""

    return _result_view_records_module


def generated_console_work_queue_records_module() -> str:
    """Return the generated TypeScript work-queue record parser module."""

    return _typescript_record_contract_module(
        _TypeScriptContractModule(
            contract_name="work_queue_items",
            record_name="work_queue_item",
            exported_type="WorkQueueItemRecord",
            parser_name="parseWorkQueueItem",
            error_name="WorkQueueTransportError",
            literal_expressions={
                "format": "workQueueItemFormat",
                "format_version": "workQueueItemFormatVersion",
            },
            imports="""import {
  consoleProtocolFormats,
  consoleProtocolFormatVersions,
} from './protocolVocabulary.ts';

const workQueueItemFormat = consoleProtocolFormats.workQueueItem;
const workQueueItemFormatVersion = consoleProtocolFormatVersions.workQueueItem;
""",
        )
    )


def write_generated_console_result_view_records_module(
    path: Path = _generated_result_view_records_module_path,
) -> Path:
    """Write the generated TypeScript result-view record parser module."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generated_console_result_view_records_module(), encoding="utf-8")
    return path


def write_generated_console_work_queue_records_module(
    path: Path = _generated_work_queue_records_module_path,
) -> Path:
    """Write the generated TypeScript work-queue record parser module."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generated_console_work_queue_records_module(), encoding="utf-8")
    return path


def write_generated_console_web_modules() -> tuple[Path, ...]:
    """Write every generated console web-source module."""

    return (
        write_generated_console_protocol_module(),
        write_generated_console_result_view_records_module(),
        write_generated_console_work_queue_records_module(),
    )


def _console_protocol_formats() -> Mapping[str, object]:
    formats = console_protocol_formats()
    return {
        "consoleData": formats.console_data,
        "artifactIndex": formats.artifact_index,
        "resultViews": {
            "importedResults": formats.imported_result_view,
            "benchmarkResults": formats.benchmark_result_view,
            "workQueue": formats.work_queue_view,
        },
        "workQueueItem": formats.work_queue_item,
    }


def _console_protocol_format_versions() -> Mapping[str, object]:
    versions = console_protocol_format_versions()
    return {
        "consoleData": versions.console_data,
        "artifactIndex": versions.artifact_index,
        "resultView": versions.result_view,
        "workQueueItem": versions.work_queue_item,
    }


def _typescript_literal(value: object) -> str:
    return _typescript_literal_lines(value, indent=0)


def _typescript_literal_lines(value: object, *, indent: int) -> str:
    prefix = " " * indent
    child_indent = indent + 2
    child_prefix = " " * child_indent
    if isinstance(value, str):
        return _typescript_string(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        if not value:
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


def _typescript_record_contract_module(config: _TypeScriptContractModule) -> str:
    contract = load_object_document(
        files("leibniz.contracts")
        .joinpath(f"{config.contract_name}{document_filename_suffix()}")
        .read_bytes(),
        description=f"{config.record_name} record contracts",
    )
    fields = _contract_fields(contract=contract, record_name=config.record_name)
    named_unions = _named_union_fields(fields=fields)
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
        fields=fields,
        named_unions=named_unions,
        literal_expressions=config.literal_expressions,
    )
    parser = _typescript_record_parser(
        exported_type=config.exported_type,
        parser_name=config.parser_name,
        error_name=config.error_name,
        fields=fields,
        named_unions=named_unions,
        literal_expressions=config.literal_expressions,
    )
    helpers = _typescript_record_parser_helpers(
        error_name=config.error_name,
        required_helpers=_required_typescript_helpers(fields=fields),
    )
    sections = [
        section
        for section in (imports, type_aliases, record_type, parser, helpers)
        if section
    ]
    return "\n\n".join(sections) + "\n"


def _typescript_record_type(
    *,
    exported_type: str,
    fields: tuple[Mapping[str, object], ...],
    named_unions: Mapping[str, tuple[str, ...]],
    literal_expressions: Mapping[str, str],
) -> str:
    lines = [f"export type {exported_type} = {{"]
    for field in fields:
        field_name = _field_name(field)
        marker = "?" if not _field_required(field) else ""
        field_type = _typescript_field_type(
            field,
            exported_type=exported_type,
            named_unions=named_unions,
            literal_expressions=literal_expressions,
        )
        lines.append(
            f"  {field_name}{marker}: {field_type};"
        )
    lines.append("};")
    return "\n".join(lines)


def _typescript_record_parser(
    *,
    exported_type: str,
    parser_name: str,
    error_name: str,
    fields: tuple[Mapping[str, object], ...],
    named_unions: Mapping[str, tuple[str, ...]],
    literal_expressions: Mapping[str, str],
) -> str:
    known_fields = ", ".join(_typescript_string(_field_name(field)) for field in fields)
    lines = [
        f"export function {parser_name}(value: unknown, path: string): {exported_type} {{",
        "  const record = requireRecord(value, path);",
        f"  rejectUnknownFields(record, path, [{known_fields}]);",
        "  return {",
    ]
    for field in fields:
        field_name = _field_name(field)
        rendered = _typescript_field_parser(
            field,
            exported_type=exported_type,
            named_unions=named_unions,
            literal_expressions=literal_expressions,
        )
        if _field_required(field):
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
}}""",
    ]
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


def _required_typescript_helpers(*, fields: tuple[Mapping[str, object], ...]) -> set[str]:
    helpers: set[str] = set()
    for field in fields:
        _collect_required_typescript_helpers(field=field, helpers=helpers)
    return helpers


def _collect_required_typescript_helpers(
    *,
    field: Mapping[str, object],
    helpers: set[str],
) -> None:
    if "values" in field:
        helpers.add("string")
        return
    kind = _field_kind(field)
    if kind in {"identifier", "name", "string", "version"}:
        helpers.add("string")
    elif kind == "integer":
        helpers.add("integer")
    elif kind == "number":
        helpers.add("number")
    elif kind == "boolean":
        helpers.add("boolean")
    elif kind == "literal":
        helpers.add("literal")
    elif kind == "sequence":
        helpers.add("array")
        _collect_required_typescript_helpers(
            field=_mapping_value(field["item"], field=f"{_field_name(field)}.item"),
            helpers=helpers,
        )


def _typescript_field_type(
    field: Mapping[str, object],
    *,
    exported_type: str,
    named_unions: Mapping[str, tuple[str, ...]],
    literal_expressions: Mapping[str, str],
) -> str:
    field_name = _field_name(field, default="")
    if field_name in named_unions:
        return _union_name(field_name=field_name, exported_type=exported_type)
    kind = _field_kind(field)
    if field_name in literal_expressions:
        return f"typeof {literal_expressions[field_name]}"
    if kind == "literal":
        return _typescript_literal_type(field["literal"])
    if kind in {"identifier", "name", "string", "version"}:
        return "string"
    if kind in {"integer", "number"}:
        return "number"
    if kind == "boolean":
        return "boolean"
    if kind == "record":
        return "Record<string, unknown>"
    if kind == "sequence":
        item = _mapping_value(field["item"], field=f"{field_name}.item")
        item_type = _typescript_field_type(
            item,
            exported_type=exported_type,
            named_unions={},
            literal_expressions={},
        )
        return f"{item_type}[]"
    raise ValueError(f"{field_name}: unsupported contract field kind: {kind}")


def _typescript_field_parser(
    field: Mapping[str, object],
    *,
    exported_type: str,
    named_unions: Mapping[str, tuple[str, ...]],
    literal_expressions: Mapping[str, str],
) -> str:
    field_name = _field_name(field)
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
    field: Mapping[str, object],
    *,
    value: str,
    path: str,
    exported_type: str,
    named_unions: Mapping[str, tuple[str, ...]],
    literal_expressions: Mapping[str, str],
) -> str:
    field_name = _field_name(field, default="")
    if field_name in named_unions:
        return (
            f"parse{_union_name(field_name=field_name, exported_type=exported_type)}"
            f"({value}, {path})"
        )
    kind = _field_kind(field)
    if field_name in literal_expressions:
        return f"requireLiteral({value}, {path}, {literal_expressions[field_name]})"
    if kind == "literal":
        return f"requireLiteral({value}, {path}, {_typescript_literal(field['literal'])})"
    if kind in {"identifier", "name", "string", "version"}:
        return f"requireString({value}, {path})"
    if kind == "integer":
        return f"requireInteger({value}, {path})"
    if kind == "number":
        return f"requireNumber({value}, {path})"
    if kind == "boolean":
        return f"requireBoolean({value}, {path})"
    if kind == "record":
        return f"requireRecord({value}, {path})"
    if kind == "sequence":
        item = _mapping_value(field["item"], field=f"{field_name}.item")
        item_parser = _typescript_value_parser(
            item,
            value="item",
            path=_typescript_index_path(path),
            exported_type=exported_type,
            named_unions={},
            literal_expressions={},
        )
        return f"requireArray({value}, {path}).map((item, index) =>\n    {item_parser},\n  )"
    raise ValueError(f"{field_name}: unsupported contract field kind: {kind}")


def _named_union_fields(*, fields: tuple[Mapping[str, object], ...]) -> dict[str, tuple[str, ...]]:
    unions: dict[str, tuple[str, ...]] = {}
    for field in fields:
        if "values" not in field:
            continue
        field_name = _field_name(field)
        unions[field_name] = tuple(
            str(value) for value in _sequence_value(field["values"], field=_field_name(field))
        )
    return unions


def _field_name(field: Mapping[str, object], *, default: str | None = None) -> str:
    if "name" not in field:
        if default is not None:
            return default
        raise ValueError("contract field is missing name")
    value = field["name"]
    if not isinstance(value, str) or not value:
        raise ValueError("contract field name must be nonempty string")
    return value


def _field_kind(field: Mapping[str, object]) -> str:
    value = field.get("kind")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{_field_name(field, default='<field>')}: missing field kind")
    return value


def _field_required(field: Mapping[str, object]) -> bool:
    value = field.get("required", True)
    if not isinstance(value, bool):
        raise ValueError(f"{_field_name(field)}.required: expected boolean")
    return value


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
    if value is None:
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


def _contract_fields(
    *,
    contract: Mapping[str, object],
    record_name: str,
) -> tuple[Mapping[str, object], ...]:
    for record_value in _sequence_value(contract["records"], field="records"):
        record = _mapping_value(record_value, field="records")
        if record["name"] == record_name:
            return tuple(
                _mapping_value(field, field=f"{record_name}.fields")
                for field in _sequence_value(record["fields"], field=f"{record_name}.fields")
            )
    raise ValueError(f"missing contract record: {record_name}")


def _mapping_value(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _sequence_value(value: object, *, field: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{field}: expected sequence")
    return cast(Sequence[object], value)


_result_view_records_module = """import {
  parseModelInspectionRecord,
  type ModelInspectionRecord,
} from '../modelInspections.ts';
import {
  consoleProtocolFormats,
  consoleProtocolFormatVersions,
} from './protocolVocabulary.ts';
import {
  parseWorkQueueItem,
  type WorkQueueItemRecord,
} from './workQueueRecords.ts';

export type { WorkQueueItemRecord, WorkQueueItemStatus } from './workQueueRecords.ts';

const resultViewFormats = consoleProtocolFormats.resultViews;
const resultViewFormatVersion = consoleProtocolFormatVersions.resultView;

export type ResultViewRecord =
  | ImportedResultViewRecord
  | BenchmarkResultViewRecord
  | WorkQueueViewRecord;

export type ImportedResultViewRecord = {
  format: typeof resultViewFormats.importedResults;
  format_version: typeof resultViewFormatVersion;
  source_path: string;
  source_mtime_ms?: number;
  source_size_bytes?: number;
  publication_bundles: ImportedPublicationBundleRecord[];
};

class ResultViewTransportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ResultViewTransportError';
  }
}

export type ImportedPublicationBundleRecord = {
  id: string;
  digest: string;
  source_path: string;
  submission_package_id: string;
  benchmark_ids: string[];
  measurement_count: number;
  measurement_dataset: Record<string, unknown>;
  measurement_score_view: Record<string, unknown>;
};

export type BenchmarkResultViewRecord = {
  format: typeof resultViewFormats.benchmarkResults;
  format_version: typeof resultViewFormatVersion;
  source_path: string;
  source_mtime_ms?: number;
  source_size_bytes?: number;
  benchmark_results: BenchmarkResultRecord[];
};

export type WorkQueueViewRecord = {
  format: typeof resultViewFormats.workQueue;
  format_version: typeof resultViewFormatVersion;
  source_path: string;
  source_mtime_ms?: number;
  source_size_bytes?: number;
  queue_items: WorkQueueItemRecord[];
};

export type BenchmarkResultRecord = {
  benchmark_id: string;
  complexity_axis?: string;
  scale_axis?: string;
  cost_axes: CostAxisRecord[];
  leaderboard: ModelResultRecord[];
  frontiers: Record<string, ModelResultRecord[]>;
  training_history: RunResultRecord[];
  model_inspections: ModelInspectionRecord[];
  proposals: ProposalRecord[];
};

export type CostAxisRecord = {
  key: string;
  label: string;
};

export type ModelResultRecord = {
  model_key: string;
  architecture_digest: string;
  benchmark_id: string;
  score: number;
  observed_complexities: number[];
  points: CompetencePointRecord[];
  cost_summary: CostSummaryRecord;
  run_ids: string[];
  measurement_count: number;
  source_kinds: string[];
  console_view_model?: RunDetailViewModelRecord;
};

export type CompetencePointRecord = {
  complexity: number;
  score: number;
  sample_count?: number;
  run_ids: string[];
};

export type RunResultRecord = {
  source_kind: string;
  source_path: string;
  run_id: string;
  run_slug: string;
  benchmark_id: string;
  architecture_digest: string;
  model_key: string;
  scale?: number;
  complexity?: number;
  measurement_count: number;
  score: number;
  cost_summary: CostSummaryRecord;
  architecture: Record<string, unknown>;
  model_inspection_digest?: string;
  model_inspection_path?: string;
  measurement_dataset_digest: string;
  sampled_competence?: Record<string, unknown>;
  training_diagnostics?: TrainingDiagnosticsRecord;
  console_view_model?: RunDetailViewModelRecord;
};

export type RunDetailViewModelRecord = {
  detail_sections: RunDetailSectionRecord[];
};

export type RunDetailSectionRecord = {
  title: string;
  entries?: RunDetailEntryRecord[];
  table?: RunDetailTableRecord;
};

export type RunDetailEntryRecord = {
  label: string;
  value: string;
};

export type RunDetailTableRecord = {
  aria_label: string;
  columns: string[];
  rows: string[][];
};

export type TrainingDiagnosticsRecord = {
  status: string;
  stop_reason: string;
  steps_run: number;
  validation_checks: number;
  best_validation_loss: number;
  best_validation_step: number;
  best_validation_check: number;
  final_validation_loss: number;
  final_validation_step: number;
  final_validation_check: number;
  protocol: TrainingProtocolRecord;
  validation_history: TrainingHistoryPointRecord[];
  artifacts: TrainingArtifactReferenceRecord[];
};

export type TrainingProtocolRecord = {
  kind: string;
  objective: string;
  optimizer: string;
  learning_rate: number;
  schedule: string;
  seed: number;
  batch_size: number;
  max_steps?: number;
  validation_interval: number;
  validation_sample_count: number;
  min_delta: number;
  patience: number;
  min_steps?: number;
  validation_source: string;
};

export type TrainingHistoryPointRecord = {
  step: number;
  validation_check: number;
  validation_loss: number;
  best_validation_loss: number;
  best_validation_step: number;
  best_validation_check: number;
  stale_checks: number;
  learning_rates?: number[];
};

export type TrainingArtifactReferenceRecord = {
  kind: string;
  digest: string;
  path?: string;
};

export type ProposalRecord = {
  id: string;
  rank: number;
  candidate_kind: string;
  candidate_id: string;
  rationale: string;
  predicted_score?: number;
  uncertainty?: number;
  acquisition_value?: number;
  acquisition_model?: string;
  acquisition_components?: Record<string, unknown>;
  search_diagnostics?: Record<string, unknown>;
  novelty?: number;
  expected_frontier_improvement?: number;
  selector_name?: string;
  source_candidate_rank?: number;
  comparable_cost_best_score?: number;
  resource_stratum_index?: number;
  resource_stratum_count?: number;
  command: string[];
};

export type CostSummaryRecord = {
  component_count: number;
  parameter_count: number;
  parameter_bytes: number;
  inference_flops: number;
  unknown_parameter_components?: number[];
};

export function isBenchmarkResultView(
  view: ResultViewRecord,
): view is BenchmarkResultViewRecord {
  return view.format === resultViewFormats.benchmarkResults;
}

export function isImportedResultView(
  view: ResultViewRecord,
): view is ImportedResultViewRecord {
  return view.format === resultViewFormats.importedResults;
}

export function isWorkQueueView(
  view: ResultViewRecord,
): view is WorkQueueViewRecord {
  return view.format === resultViewFormats.workQueue;
}

export function parseResultViewRecords(value: unknown): ResultViewRecord[] {
  return requireArray(value, 'result_views').map((view, index) =>
    parseResultViewRecord(view, `result_views.${index}`),
  );
}

function parseResultViewRecord(value: unknown, path: string): ResultViewRecord {
  const record = requireRecord(value, path);
  if (record.format === resultViewFormats.benchmarkResults) {
    return parseBenchmarkResultViewRecord(record, path);
  }
  if (record.format === resultViewFormats.workQueue) {
    return parseWorkQueueViewRecord(record, path);
  }
  return parseImportedResultViewRecord(record, path);
}

function parseImportedResultViewRecord(
  record: Record<string, unknown>,
  path: string,
): ImportedResultViewRecord {
  const format = requireLiteral(record.format, `${path}.format`, resultViewFormats.importedResults);
  const formatVersion = requireLiteral(
    record.format_version,
    `${path}.format_version`,
    resultViewFormatVersion,
  );
  const sourcePath = requireString(record.source_path, `${path}.source_path`);
  const sourceMtimeMs = optionalNumber(record.source_mtime_ms, `${path}.source_mtime_ms`);
  const sourceSizeBytes = optionalNumber(record.source_size_bytes, `${path}.source_size_bytes`);
  const publicationBundles = requireArray(
    record.publication_bundles,
    `${path}.publication_bundles`,
  ).map((bundle, index) =>
    parseImportedPublicationBundleRecord(bundle, `${path}.publication_bundles.${index}`),
  );
  return {
    format,
    format_version: formatVersion,
    source_path: sourcePath,
    source_mtime_ms: sourceMtimeMs,
    source_size_bytes: sourceSizeBytes,
    publication_bundles: publicationBundles,
  };
}

function parseBenchmarkResultViewRecord(
  record: Record<string, unknown>,
  path: string,
): BenchmarkResultViewRecord {
  const format = requireLiteral(
    record.format,
    `${path}.format`,
    resultViewFormats.benchmarkResults,
  );
  const formatVersion = requireLiteral(
    record.format_version,
    `${path}.format_version`,
    resultViewFormatVersion,
  );
  const sourcePath = requireString(record.source_path, `${path}.source_path`);
  const sourceMtimeMs = optionalNumber(record.source_mtime_ms, `${path}.source_mtime_ms`);
  const sourceSizeBytes = optionalNumber(record.source_size_bytes, `${path}.source_size_bytes`);
  const benchmarkResults = requireArray(
    record.benchmark_results,
    `${path}.benchmark_results`,
  ).map((result, index) => parseBenchmarkResult(result, `${path}.benchmark_results.${index}`));
  return {
    format,
    format_version: formatVersion,
    source_path: sourcePath,
    source_mtime_ms: sourceMtimeMs,
    source_size_bytes: sourceSizeBytes,
    benchmark_results: benchmarkResults,
  };
}

function parseWorkQueueViewRecord(
  record: Record<string, unknown>,
  path: string,
): WorkQueueViewRecord {
  const format = requireLiteral(record.format, `${path}.format`, resultViewFormats.workQueue);
  const formatVersion = requireLiteral(
    record.format_version,
    `${path}.format_version`,
    resultViewFormatVersion,
  );
  const sourcePath = requireString(record.source_path, `${path}.source_path`);
  const sourceMtimeMs = optionalNumber(record.source_mtime_ms, `${path}.source_mtime_ms`);
  const sourceSizeBytes = optionalNumber(record.source_size_bytes, `${path}.source_size_bytes`);
  const queueItems = requireArray(record.queue_items, `${path}.queue_items`).map((item, index) =>
    parseWorkQueueItem(item, `${path}.queue_items.${index}`),
  );
  return {
    format,
    format_version: formatVersion,
    source_path: sourcePath,
    source_mtime_ms: sourceMtimeMs,
    source_size_bytes: sourceSizeBytes,
    queue_items: queueItems,
  };
}

function parseBenchmarkResult(value: unknown, path: string): BenchmarkResultRecord {
  const record = requireRecord(value, path);
  const costAxes = requireArray(record.cost_axes, `${path}.cost_axes`).map((axis, index) => {
    const axisRecord = requireRecord(axis, `${path}.cost_axes.${index}`);
    return {
      key: requireString(axisRecord.key, `${path}.cost_axes.${index}.key`),
      label: requireString(axisRecord.label, `${path}.cost_axes.${index}.label`),
    };
  });
  const frontiersRecord = requireRecord(record.frontiers, `${path}.frontiers`);
  const frontiers = Object.fromEntries(
    Object.entries(frontiersRecord).map(([key, value]) => [
      key,
      requireArray(value, `${path}.frontiers.${key}`).map((model, index) =>
        parseModelResult(model, `${path}.frontiers.${key}.${index}`),
      ),
    ]),
  );
  return {
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
    complexity_axis:
      record.complexity_axis === undefined
        ? undefined
        : requireString(record.complexity_axis, `${path}.complexity_axis`),
    scale_axis:
      record.scale_axis === undefined
        ? undefined
        : requireString(record.scale_axis, `${path}.scale_axis`),
    cost_axes: costAxes,
    leaderboard: requireArray(record.leaderboard, `${path}.leaderboard`).map((model, index) =>
      parseModelResult(model, `${path}.leaderboard.${index}`),
    ),
    frontiers,
    training_history: requireArray(record.training_history, `${path}.training_history`).map(
      (run, index) => parseRunResult(run, `${path}.training_history.${index}`),
    ),
    model_inspections:
      record.model_inspections === undefined
        ? []
        : requireArray(record.model_inspections, `${path}.model_inspections`).map(
            (inspection, index) =>
              parseModelInspectionRecord(inspection, `${path}.model_inspections.${index}`),
          ),
    proposals:
      record.proposals === undefined
        ? []
        : requireArray(record.proposals, `${path}.proposals`).map((proposal, index) =>
            parseProposal(proposal, `${path}.proposals.${index}`),
          ),
  };
}

function parseModelResult(value: unknown, path: string): ModelResultRecord {
  const record = requireRecord(value, path);
  return {
    model_key: requireString(record.model_key, `${path}.model_key`),
    architecture_digest: requireString(record.architecture_digest, `${path}.architecture_digest`),
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
    score: requireNumber(record.score, `${path}.score`),
    observed_complexities: parseNumberArray(
      record.observed_complexities,
      `${path}.observed_complexities`,
    ),
    points: requireArray(record.points, `${path}.points`).map((point, index) =>
      parseCompetencePoint(point, `${path}.points.${index}`),
    ),
    cost_summary: parseCostSummary(record.cost_summary, `${path}.cost_summary`),
    run_ids: parseStringArray(record.run_ids, `${path}.run_ids`),
    measurement_count: requireNumber(record.measurement_count, `${path}.measurement_count`),
    source_kinds: parseStringArray(record.source_kinds, `${path}.source_kinds`),
    console_view_model:
      record.console_view_model === undefined
        ? undefined
        : parseRunDetailViewModel(record.console_view_model, `${path}.console_view_model`),
  };
}

function parseCompetencePoint(value: unknown, path: string): CompetencePointRecord {
  const record = requireRecord(value, path);
  return {
    complexity: requireNumber(record.complexity, `${path}.complexity`),
    score: requireNumber(record.score, `${path}.score`),
    sample_count:
      record.sample_count === undefined
        ? undefined
        : requireNumber(record.sample_count, `${path}.sample_count`),
    run_ids: parseStringArray(record.run_ids, `${path}.run_ids`),
  };
}

function parseRunResult(value: unknown, path: string): RunResultRecord {
  const record = requireRecord(value, path);
  return {
    source_kind: requireString(record.source_kind, `${path}.source_kind`),
    source_path: requireString(record.source_path, `${path}.source_path`),
    run_id: requireString(record.run_id, `${path}.run_id`),
    run_slug: requireString(record.run_slug, `${path}.run_slug`),
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
    architecture_digest: requireString(record.architecture_digest, `${path}.architecture_digest`),
    model_key: requireString(record.model_key, `${path}.model_key`),
    scale:
      record.scale === undefined ? undefined : requireNumber(record.scale, `${path}.scale`),
    complexity:
      record.complexity === undefined
        ? undefined
        : requireNumber(record.complexity, `${path}.complexity`),
    measurement_count: requireNumber(record.measurement_count, `${path}.measurement_count`),
    score: requireNumber(record.score, `${path}.score`),
    cost_summary: parseCostSummary(record.cost_summary, `${path}.cost_summary`),
    architecture: requireRecord(record.architecture, `${path}.architecture`),
    model_inspection_digest:
      record.model_inspection_digest === undefined
        ? undefined
        : requireString(record.model_inspection_digest, `${path}.model_inspection_digest`),
    model_inspection_path:
      record.model_inspection_path === undefined
        ? undefined
        : requireString(record.model_inspection_path, `${path}.model_inspection_path`),
    measurement_dataset_digest: requireString(
      record.measurement_dataset_digest,
      `${path}.measurement_dataset_digest`,
    ),
    sampled_competence:
      record.sampled_competence === undefined
        ? undefined
        : requireRecord(record.sampled_competence, `${path}.sampled_competence`),
    training_diagnostics:
      record.training_diagnostics === undefined
        ? undefined
        : parseTrainingDiagnostics(
            record.training_diagnostics,
            `${path}.training_diagnostics`,
          ),
    console_view_model:
      record.console_view_model === undefined
        ? undefined
        : parseRunDetailViewModel(record.console_view_model, `${path}.console_view_model`),
  };
}

function parseRunDetailViewModel(value: unknown, path: string): RunDetailViewModelRecord {
  const record = requireRecord(value, path);
  return {
    detail_sections: requireArray(record.detail_sections, `${path}.detail_sections`).map(
      (section, index) => parseRunDetailSection(section, `${path}.detail_sections.${index}`),
    ),
  };
}

function parseRunDetailSection(value: unknown, path: string): RunDetailSectionRecord {
  const record = requireRecord(value, path);
  return {
    title: requireString(record.title, `${path}.title`),
    entries:
      record.entries === undefined
        ? undefined
        : requireArray(record.entries, `${path}.entries`).map((entry, index) =>
            parseRunDetailEntry(entry, `${path}.entries.${index}`),
          ),
    table:
      record.table === undefined
        ? undefined
        : parseRunDetailTable(record.table, `${path}.table`),
  };
}

function parseRunDetailEntry(value: unknown, path: string): RunDetailEntryRecord {
  const record = requireRecord(value, path);
  return {
    label: requireString(record.label, `${path}.label`),
    value: requireString(record.value, `${path}.value`),
  };
}

function parseRunDetailTable(value: unknown, path: string): RunDetailTableRecord {
  const record = requireRecord(value, path);
  return {
    aria_label: requireString(record.aria_label, `${path}.aria_label`),
    columns: parseStringArray(record.columns, `${path}.columns`),
    rows: requireArray(record.rows, `${path}.rows`).map((row, index) =>
      parseStringArray(row, `${path}.rows.${index}`),
    ),
  };
}

function parseTrainingDiagnostics(value: unknown, path: string): TrainingDiagnosticsRecord {
  const record = requireRecord(value, path);
  return {
    status: requireString(record.status, `${path}.status`),
    stop_reason: requireString(record.stop_reason, `${path}.stop_reason`),
    steps_run: requireNumber(record.steps_run, `${path}.steps_run`),
    validation_checks: requireNumber(record.validation_checks, `${path}.validation_checks`),
    best_validation_loss: requireNumber(
      record.best_validation_loss,
      `${path}.best_validation_loss`,
    ),
    best_validation_step: requireNumber(
      record.best_validation_step,
      `${path}.best_validation_step`,
    ),
    best_validation_check: requireNumber(
      record.best_validation_check,
      `${path}.best_validation_check`,
    ),
    final_validation_loss: requireNumber(
      record.final_validation_loss,
      `${path}.final_validation_loss`,
    ),
    final_validation_step: requireNumber(
      record.final_validation_step,
      `${path}.final_validation_step`,
    ),
    final_validation_check: requireNumber(
      record.final_validation_check,
      `${path}.final_validation_check`,
    ),
    protocol: parseTrainingProtocol(record.protocol, `${path}.protocol`),
    validation_history: requireArray(record.validation_history, `${path}.validation_history`).map(
      (point, index) => parseTrainingHistoryPoint(point, `${path}.validation_history.${index}`),
    ),
    artifacts: requireArray(record.artifacts, `${path}.artifacts`).map((artifact, index) =>
      parseTrainingArtifactReference(artifact, `${path}.artifacts.${index}`),
    ),
  };
}

function parseTrainingProtocol(value: unknown, path: string): TrainingProtocolRecord {
  const record = requireRecord(value, path);
  return {
    kind: requireString(record.kind, `${path}.kind`),
    objective: requireString(record.objective, `${path}.objective`),
    optimizer: requireString(record.optimizer, `${path}.optimizer`),
    learning_rate: requireNumber(record.learning_rate, `${path}.learning_rate`),
    schedule: requireString(record.schedule, `${path}.schedule`),
    seed: requireNumber(record.seed, `${path}.seed`),
    batch_size: requireNumber(record.batch_size, `${path}.batch_size`),
    max_steps:
      record.max_steps === undefined
        ? undefined
        : requireNumber(record.max_steps, `${path}.max_steps`),
    validation_interval: requireNumber(
      record.validation_interval,
      `${path}.validation_interval`,
    ),
    validation_sample_count: requireNumber(
      record.validation_sample_count,
      `${path}.validation_sample_count`,
    ),
    min_delta: requireNumber(record.min_delta, `${path}.min_delta`),
    patience: requireNumber(record.patience, `${path}.patience`),
    min_steps:
      record.min_steps === undefined
        ? undefined
        : requireNumber(record.min_steps, `${path}.min_steps`),
    validation_source: requireString(record.validation_source, `${path}.validation_source`),
  };
}

function parseTrainingHistoryPoint(value: unknown, path: string): TrainingHistoryPointRecord {
  const record = requireRecord(value, path);
  return {
    step: requireNumber(record.step, `${path}.step`),
    validation_check: requireNumber(record.validation_check, `${path}.validation_check`),
    validation_loss: requireNumber(record.validation_loss, `${path}.validation_loss`),
    best_validation_loss: requireNumber(
      record.best_validation_loss,
      `${path}.best_validation_loss`,
    ),
    best_validation_step: requireNumber(
      record.best_validation_step,
      `${path}.best_validation_step`,
    ),
    best_validation_check: requireNumber(
      record.best_validation_check,
      `${path}.best_validation_check`,
    ),
    stale_checks: requireNumber(record.stale_checks, `${path}.stale_checks`),
    learning_rates:
      record.learning_rates === undefined
        ? undefined
        : parseNumberArray(record.learning_rates, `${path}.learning_rates`),
  };
}

function parseTrainingArtifactReference(
  value: unknown,
  path: string,
): TrainingArtifactReferenceRecord {
  const record = requireRecord(value, path);
  return {
    kind: requireString(record.kind, `${path}.kind`),
    digest: requireString(record.digest, `${path}.digest`),
    path: record.path === undefined ? undefined : requireString(record.path, `${path}.path`),
  };
}

function parseProposal(value: unknown, path: string): ProposalRecord {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    rank: requireNumber(record.rank, `${path}.rank`),
    candidate_kind: requireString(record.candidate_kind, `${path}.candidate_kind`),
    candidate_id: requireString(record.candidate_id, `${path}.candidate_id`),
    rationale: requireString(record.rationale, `${path}.rationale`),
    predicted_score: optionalNumber(record.predicted_score, `${path}.predicted_score`),
    uncertainty: optionalNumber(record.uncertainty, `${path}.uncertainty`),
    acquisition_value: optionalNumber(record.acquisition_value, `${path}.acquisition_value`),
    acquisition_model:
      record.acquisition_model === undefined
        ? undefined
        : requireString(record.acquisition_model, `${path}.acquisition_model`),
    acquisition_components:
      record.acquisition_components === undefined
        ? undefined
        : requireRecord(record.acquisition_components, `${path}.acquisition_components`),
    search_diagnostics:
      record.search_diagnostics === undefined
        ? undefined
        : requireRecord(record.search_diagnostics, `${path}.search_diagnostics`),
    novelty: optionalNumber(record.novelty, `${path}.novelty`),
    expected_frontier_improvement: optionalNumber(
      record.expected_frontier_improvement,
      `${path}.expected_frontier_improvement`,
    ),
    selector_name:
      record.selector_name === undefined
        ? undefined
        : requireString(record.selector_name, `${path}.selector_name`),
    source_candidate_rank: optionalNumber(
      record.source_candidate_rank,
      `${path}.source_candidate_rank`,
    ),
    comparable_cost_best_score: optionalNumber(
      record.comparable_cost_best_score,
      `${path}.comparable_cost_best_score`,
    ),
    resource_stratum_index: optionalNumber(
      record.resource_stratum_index,
      `${path}.resource_stratum_index`,
    ),
    resource_stratum_count: optionalNumber(
      record.resource_stratum_count,
      `${path}.resource_stratum_count`,
    ),
    command:
      record.command === undefined ? [] : parseStringArray(record.command, `${path}.command`),
  };
}

function parseCostSummary(value: unknown, path: string): CostSummaryRecord {
  const record = requireRecord(value, path);
  const costSummary: CostSummaryRecord = {
    component_count: requireNumber(record.component_count, `${path}.component_count`),
    parameter_count: requireNumber(record.parameter_count, `${path}.parameter_count`),
    parameter_bytes: requireNumber(record.parameter_bytes, `${path}.parameter_bytes`),
    inference_flops: requireNumber(record.inference_flops, `${path}.inference_flops`),
  };
  if (record.unknown_parameter_components !== undefined) {
    costSummary.unknown_parameter_components = parseNumberArray(
      record.unknown_parameter_components,
      `${path}.unknown_parameter_components`,
    );
  }
  return costSummary;
}

function parseImportedPublicationBundleRecord(
  value: unknown,
  path: string,
): ImportedPublicationBundleRecord {
  const record = requireRecord(value, path);
  const benchmarkIds = requireArray(record.benchmark_ids, `${path}.benchmark_ids`).map(
    (item, index) => requireString(item, `${path}.benchmark_ids.${index}`),
  );
  return {
    id: requireString(record.id, `${path}.id`),
    digest: requireString(record.digest, `${path}.digest`),
    source_path: requireString(record.source_path, `${path}.source_path`),
    submission_package_id: requireString(
      record.submission_package_id,
      `${path}.submission_package_id`,
    ),
    benchmark_ids: benchmarkIds,
    measurement_count: requireNumber(record.measurement_count, `${path}.measurement_count`),
    measurement_dataset: requireRecord(record.measurement_dataset, `${path}.measurement_dataset`),
    measurement_score_view: requireRecord(
      record.measurement_score_view,
      `${path}.measurement_score_view`,
    ),
  };
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ResultViewTransportError(`${path}: expected record`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new ResultViewTransportError(`${path}: expected array`);
  }
  return value;
}

function parseNumberArray(value: unknown, path: string): number[] {
  return requireArray(value, path).map((item, index) => requireNumber(item, `${path}.${index}`));
}

function parseStringArray(value: unknown, path: string): string[] {
  return requireArray(value, path).map((item, index) => requireString(item, `${path}.${index}`));
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new ResultViewTransportError(`${path}: expected string`);
  }
  return value;
}

function requireNumber(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ResultViewTransportError(`${path}: expected number`);
  }
  return value;
}

function optionalNumber(value: unknown, path: string): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  return requireNumber(value, path);
}

function requireLiteral<const Literal extends string | number>(
  value: unknown,
  path: string,
  expected: Literal,
): Literal {
  if (value !== expected) {
    throw new ResultViewTransportError(`${path}: expected ${String(expected)}`);
  }
  return expected;
}
"""


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate console web-source modules.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the generated module is not up to date",
    )
    parser.add_argument(
        "--path",
        default=None,
        type=Path,
        help="generated protocol vocabulary module path",
    )
    parser.add_argument(
        "--result-view-records-path",
        default=None,
        type=Path,
        help="generated result-view records module path",
    )
    parser.add_argument(
        "--work-queue-records-path",
        default=None,
        type=Path,
        help="generated work-queue records module path",
    )
    args = parser.parse_args(argv)

    protocol_path = args.path or _generated_protocol_module_path
    result_view_records_path = (
        args.result_view_records_path or _generated_result_view_records_module_path
    )
    work_queue_records_path = (
        args.work_queue_records_path or _generated_work_queue_records_module_path
    )
    expected_modules = (
        (protocol_path, generated_console_protocol_module()),
        (result_view_records_path, generated_console_result_view_records_module()),
        (work_queue_records_path, generated_console_work_queue_records_module()),
    )
    if args.check:
        for path, expected in expected_modules:
            actual = path.read_text(encoding="utf-8")
            if actual != expected:
                raise SystemExit(f"{path}: generated console web module is out of date")
        return 0
    write_generated_console_protocol_module(protocol_path)
    write_generated_console_result_view_records_module(result_view_records_path)
    write_generated_console_work_queue_records_module(work_queue_records_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
