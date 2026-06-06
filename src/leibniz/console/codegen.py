"""Generate small console web-source modules from Python-owned protocol metadata."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from leibniz.console.protocol import (
    console_protocol_format_versions,
    console_protocol_formats,
)
from leibniz.record_contracts import typescript_literal

__all__ = [
    "generated_console_protocol_module",
    "generated_console_result_view_records_module",
    "write_generated_console_protocol_module",
    "write_generated_console_result_view_records_module",
    "write_generated_console_web_modules",
]

_generated_protocol_module_path = (
    Path(__file__).parent / "_web_src" / "src" / "generated" / "protocolVocabulary.ts"
)
_generated_result_view_records_module_path = (
    Path(__file__).parent / "_web_src" / "src" / "generated" / "resultViewRecords.ts"
)


def generated_console_protocol_module() -> str:
    """Return the generated TypeScript console protocol vocabulary module."""

    formats = _console_protocol_formats()
    versions = _console_protocol_format_versions()
    return (
        "export const consoleProtocolFormats = "
        f"{typescript_literal(formats)} as const;\n\n"
        "export const consoleProtocolFormatVersions = "
        f"{typescript_literal(versions)} as const;\n"
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

    return _result_view_records_module.rstrip("\n") + "\n"


def write_generated_console_result_view_records_module(
    path: Path = _generated_result_view_records_module_path,
) -> Path:
    """Write the generated TypeScript result-view record parser module."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generated_console_result_view_records_module(), encoding="utf-8")
    return path


def write_generated_console_web_modules() -> tuple[Path, ...]:
    """Write every generated console web-source module."""

    return (
        write_generated_console_protocol_module(),
        write_generated_console_result_view_records_module(),
    )


def _console_protocol_formats() -> Mapping[str, object]:
    formats = console_protocol_formats()
    return {
        "consoleData": formats.console_data,
        "artifactIndex": formats.artifact_index,
        "resultViews": {
            "benchmarkResults": formats.benchmark_result_view,
        },
    }


def _console_protocol_format_versions() -> Mapping[str, object]:
    versions = console_protocol_format_versions()
    return {
        "consoleData": versions.console_data,
        "artifactIndex": versions.artifact_index,
        "resultView": versions.result_view,
    }


_result_view_records_module = """import {
  parseModelInspectionRecord,
  type ModelInspectionRecord,
} from '../modelInspections.ts';
import {
  requireArray,
  requireLiteral,
  requireNumber,
  requireRecord,
  requireString,
  optionalNumber,
} from '../transport.ts';
import {
  consoleProtocolFormats,
  consoleProtocolFormatVersions,
} from './protocolVocabulary.ts';

const resultViewFormats = consoleProtocolFormats.resultViews;
const resultViewFormatVersion = consoleProtocolFormatVersions.resultView;

type ResultViewBaseRecord = {
  format_version: typeof resultViewFormatVersion;
  source_path: string;
  source_mtime_ms?: number;
  source_size_bytes?: number;
};

export type ResultViewRecord = BenchmarkResultViewRecord;

export type BenchmarkResultViewRecord = ResultViewBaseRecord & {
  format: typeof resultViewFormats.benchmarkResults;
  benchmark_results: BenchmarkResultRecord[];
};

export type BenchmarkResultRecord = {
  benchmark_id: string;
  complexity_axis?: string;
  cost_axes: CostAxisRecord[];
  score_axes?: ScoreAxisRecord[];
  leaderboard: ModelResultRecord[];
  model_candidates: ModelResultRecord[];
  frontiers: Record<string, ModelResultRecord[]>;
  training_history: RunResultRecord[];
  plot_runs: RunResultRecord[];
  model_inspections: ModelInspectionRecord[];
};

export type CostAxisRecord = {
  key: string;
  label: string;
};

export type ScoreAxisRecord = {
  key: string;
  label: string;
};

export type ScoreViewRecord = {
  key: string;
  label: string;
  score: number;
  basis?: Record<string, unknown>;
};

export type CompetencePointRecord = {
  complexity: number;
  complexity_minimum?: number;
  complexity_maximum?: number;
  score: number;
  sample_count?: number;
  run_ids: string[];
};

export type TrainingEstimateComparisonPointRecord = {
  complexity: number;
  complexity_minimum?: number;
  complexity_maximum?: number;
  status: 'matched' | 'accepted-only' | 'training-only';
  accepted_score?: number;
  training_score?: number;
  score_delta?: number;
  accepted_sample_count?: number;
  training_sample_count?: number;
};

export type TrainingEstimateComparisonRecord = {
  kind: 'training-vs-accepted-sampled-competence-v1';
  accepted_score: number;
  training_score: number;
  score_delta: number;
  accepted_sample_count: number;
  training_sample_count: number;
  point_count: number;
  matched_point_count: number;
  points: TrainingEstimateComparisonPointRecord[];
};

export type CostSummaryRecord = {
  component_count: number;
  parameter_count?: number;
  storage_bytes?: number;
  inference_compute?: number;
  training_compute_per_sample?: number;
  training_compute?: number;
  unknown_parameter_components?: number[];
};

export type ModelResultRecord = {
  model_key: string;
  result_status: 'accepted' | 'provisional';
  architecture_digest: string;
  benchmark_id: string;
  score: number;
  score_basis?: Record<string, unknown>;
  score_views?: Record<string, ScoreViewRecord>;
  observed_complexities: number[];
  points: CompetencePointRecord[];
  cost_summary: CostSummaryRecord;
  run_ids: string[];
  measurement_count: number;
  source_kinds: string[];
  training_estimate_comparison?: TrainingEstimateComparisonRecord;
  console_view_model?: RunDetailViewModelRecord;
};

export type RunResultRecord = {
  source_kind: string;
  result_status: 'accepted' | 'provisional';
  source_path: string;
  run_id: string;
  run_slug: string;
  benchmark_id: string;
  architecture_digest: string;
  model_key: string;
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
  final_validation_loss: number;
  final_validation_step: number;
  final_validation_check: number;
  training_compute?: number;
  protocol: TrainingProtocolRecord;
  validation_history: TrainingHistoryPointRecord[];
  artifacts: TrainingArtifactReferenceRecord[];
  throughput?: Record<string, unknown>;
  evaluation_curriculum?: Record<string, unknown>;
};

export type TrainingProtocolRecord = {
  kind: string;
  objective: string;
  optimizer: string;
  learning_rate?: number;
  schedule: string;
  seed: number;
  training_batch_target: number;
  max_steps?: number;
  gate_check_interval: number;
  gate_batch_target: number;
  gate_decision_rule: string;
  min_delta: number;
  patience: number;
  validation_source: string;
};

export type TrainingHistoryPointRecord = {
  step: number;
  validation_check: number;
  validation_loss: number;
  stale_checks: number;
  learning_rates?: number[];
};

export type TrainingArtifactReferenceRecord = {
  kind: string;
  digest: string;
  path?: string;
};

const transportError = (message: string) => new Error(message);

export function isBenchmarkResultView(
  view: ResultViewRecord,
): view is BenchmarkResultViewRecord {
  return view.format === resultViewFormats.benchmarkResults;
}

export function parseResultViewRecords(value: unknown): ResultViewRecord[] {
  return requireArray(value, 'result_views', transportError).map((view, index) =>
    parseResultViewRecord(view, `result_views.${index}`),
  );
}

function parseResultViewRecord(value: unknown, path: string): ResultViewRecord {
  const record = requireRecord(value, path, transportError);
  requireLiteral(record.format_version, `${path}.format_version`, resultViewFormatVersion, transportError);
  requireString(record.source_path, `${path}.source_path`, transportError);
  optionalNumber(record.source_mtime_ms, `${path}.source_mtime_ms`, transportError);
  optionalNumber(record.source_size_bytes, `${path}.source_size_bytes`, transportError);
  requireLiteral(record.format, `${path}.format`, resultViewFormats.benchmarkResults, transportError);
  return parseBenchmarkResultViewRecord(record, path);
}

function parseBenchmarkResultViewRecord(
  record: Record<string, unknown>,
  path: string,
): BenchmarkResultViewRecord {
  return withFields(record, {
    benchmark_results: arrayOf(record.benchmark_results, `${path}.benchmark_results`, parseBenchmarkResult),
  }) as BenchmarkResultViewRecord;
}

function parseBenchmarkResult(value: unknown, path: string): BenchmarkResultRecord {
  const record = requireRecord(value, path, transportError);
  return {
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`, transportError),
    complexity_axis: optional(record.complexity_axis, `${path}.complexity_axis`, (item, itemPath) => requireString(item, itemPath, transportError)),
    cost_axes: arrayOf(record.cost_axes, `${path}.cost_axes`, parseCostAxis),
    score_axes: optional(record.score_axes, `${path}.score_axes`, (value, valuePath) => arrayOf(value, valuePath, parseScoreAxis)),
    leaderboard: arrayOf(record.leaderboard, `${path}.leaderboard`, parseModelResult),
    model_candidates: arrayOf(record.model_candidates, `${path}.model_candidates`, parseModelResult),
    frontiers: parseFrontiers(record.frontiers, `${path}.frontiers`),
    training_history: arrayOf(record.training_history, `${path}.training_history`, parseRunResult),
    plot_runs: arrayOf(record.plot_runs, `${path}.plot_runs`, parseRunResult),
    model_inspections: arrayOf(record.model_inspections ?? [], `${path}.model_inspections`, parseModelInspectionRecord),
  };
}

function parseModelResult(value: unknown, path: string): ModelResultRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, ['model_key', 'result_status', 'architecture_digest', 'benchmark_id']);
  if (record.result_status !== 'accepted' && record.result_status !== 'provisional') {
    throw transportError(`${path}.result_status must be accepted or provisional`);
  }
  return withFields(record, {
    model_key: requireString(record.model_key, `${path}.model_key`, transportError),
    result_status: requireString(record.result_status, `${path}.result_status`, transportError) as 'accepted' | 'provisional',
    architecture_digest: requireString(record.architecture_digest, `${path}.architecture_digest`, transportError),
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`, transportError),
    score: requireNumber(record.score, `${path}.score`, transportError),
    score_basis: optional(record.score_basis, `${path}.score_basis`, parseScoreBasis),
    score_views: optional(record.score_views, `${path}.score_views`, parseScoreViews),
    observed_complexities: numberArray(record.observed_complexities, `${path}.observed_complexities`),
    points: arrayOf(record.points, `${path}.points`, parseCompetencePoint),
    cost_summary: parseCostSummary(record.cost_summary, `${path}.cost_summary`),
    run_ids: stringArray(record.run_ids, `${path}.run_ids`),
    measurement_count: requireNumber(record.measurement_count, `${path}.measurement_count`, transportError),
    source_kinds: stringArray(record.source_kinds, `${path}.source_kinds`),
    training_estimate_comparison: optional(record.training_estimate_comparison, `${path}.training_estimate_comparison`, parseTrainingEstimateComparison),
    console_view_model: optional(record.console_view_model, `${path}.console_view_model`, parseRunDetailViewModel),
  }) as ModelResultRecord;
}

function parseTrainingEstimateComparison(value: unknown, path: string): TrainingEstimateComparisonRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, ['kind']);
  if (record.kind !== 'training-vs-accepted-sampled-competence-v1') {
    throw transportError(`${path}.kind is invalid`);
  }
  return withFields(record, {
    kind: requireString(record.kind, `${path}.kind`, transportError) as 'training-vs-accepted-sampled-competence-v1',
    accepted_score: requireNumber(record.accepted_score, `${path}.accepted_score`, transportError),
    training_score: requireNumber(record.training_score, `${path}.training_score`, transportError),
    score_delta: requireNumber(record.score_delta, `${path}.score_delta`, transportError),
    accepted_sample_count: requireNumber(record.accepted_sample_count, `${path}.accepted_sample_count`, transportError),
    training_sample_count: requireNumber(record.training_sample_count, `${path}.training_sample_count`, transportError),
    point_count: requireNumber(record.point_count, `${path}.point_count`, transportError),
    matched_point_count: requireNumber(record.matched_point_count, `${path}.matched_point_count`, transportError),
    points: arrayOf(record.points, `${path}.points`, parseTrainingEstimateComparisonPoint),
  }) as TrainingEstimateComparisonRecord;
}

function parseTrainingEstimateComparisonPoint(value: unknown, path: string): TrainingEstimateComparisonPointRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, ['status']);
  if (record.status !== 'matched' && record.status !== 'accepted-only' && record.status !== 'training-only') {
    throw transportError(`${path}.status is invalid`);
  }
  return withFields(record, {
    complexity: requireNumber(record.complexity, `${path}.complexity`, transportError),
    complexity_minimum: optional(record.complexity_minimum, `${path}.complexity_minimum`, parseNumber),
    complexity_maximum: optional(record.complexity_maximum, `${path}.complexity_maximum`, parseNumber),
    status: requireString(record.status, `${path}.status`, transportError) as 'matched' | 'accepted-only' | 'training-only',
    accepted_score: optional(record.accepted_score, `${path}.accepted_score`, parseNumber),
    training_score: optional(record.training_score, `${path}.training_score`, parseNumber),
    score_delta: optional(record.score_delta, `${path}.score_delta`, parseNumber),
    accepted_sample_count: optional(record.accepted_sample_count, `${path}.accepted_sample_count`, parseNumber),
    training_sample_count: optional(record.training_sample_count, `${path}.training_sample_count`, parseNumber),
  }) as TrainingEstimateComparisonPointRecord;
}

function parseRunResult(value: unknown, path: string): RunResultRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, [
    'source_kind',
    'result_status',
    'source_path',
    'run_id',
    'run_slug',
    'benchmark_id',
    'architecture_digest',
    'model_key',
    'measurement_dataset_digest',
  ]);
  if (record.result_status !== 'accepted' && record.result_status !== 'provisional') {
    throw transportError(`${path}.result_status must be accepted or provisional`);
  }
  return withFields(record, {
    measurement_count: requireNumber(record.measurement_count, `${path}.measurement_count`, transportError),
    score: requireNumber(record.score, `${path}.score`, transportError),
    cost_summary: parseCostSummary(record.cost_summary, `${path}.cost_summary`),
    architecture: requireRecord(record.architecture, `${path}.architecture`, transportError),
    training_diagnostics: optional(record.training_diagnostics, `${path}.training_diagnostics`, parseTrainingDiagnostics),
    console_view_model: optional(record.console_view_model, `${path}.console_view_model`, parseRunDetailViewModel),
  }) as RunResultRecord;
}

function parseCostAxis(value: unknown, path: string): CostAxisRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, ['key', 'label']);
  return record as CostAxisRecord;
}

function parseScoreAxis(value: unknown, path: string): ScoreAxisRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, ['key', 'label']);
  return record as ScoreAxisRecord;
}

function parseScoreViews(value: unknown, path: string): Record<string, ScoreViewRecord> {
  const record = requireRecord(value, path, transportError);
  return Object.fromEntries(
    Object.entries(record).map(([key, item]) => {
      const itemPath = `${path}.${key}`;
      const view = requireRecord(item, itemPath, transportError);
      requireStrings(view, itemPath, ['key', 'label']);
      return [
        key,
        {
          key: requireString(view.key, `${itemPath}.key`, transportError),
          label: requireString(view.label, `${itemPath}.label`, transportError),
          score: requireNumber(view.score, `${itemPath}.score`, transportError),
          basis: optional(view.basis, `${itemPath}.basis`, parseScoreBasis),
        },
      ];
    }),
  );
}

function parseCompetencePoint(value: unknown, path: string): CompetencePointRecord {
  const record = requireRecord(value, path, transportError);
  return {
    complexity: requireNumber(record.complexity, `${path}.complexity`, transportError),
    score: requireNumber(record.score, `${path}.score`, transportError),
    sample_count: optionalNumber(record.sample_count, `${path}.sample_count`, transportError),
    run_ids: stringArray(record.run_ids, `${path}.run_ids`),
  };
}

function parseRunDetailViewModel(value: unknown, path: string): RunDetailViewModelRecord {
  const record = requireRecord(value, path, transportError);
  return {
    detail_sections: arrayOf(record.detail_sections, `${path}.detail_sections`, (section, sectionPath) => {
      const sectionRecord = requireRecord(section, sectionPath, transportError);
      return {
        title: requireString(sectionRecord.title, `${sectionPath}.title`, transportError),
        entries: optional(sectionRecord.entries, `${sectionPath}.entries`, parseDetailEntries),
        table: optional(sectionRecord.table, `${sectionPath}.table`, parseRunDetailTable),
      };
    }),
  };
}

function parseDetailEntries(value: unknown, path: string): RunDetailEntryRecord[] {
  return arrayOf(value, path, (entry, entryPath) => {
    const record = requireRecord(entry, entryPath, transportError);
    requireStrings(record, entryPath, ['label', 'value']);
    return record as RunDetailEntryRecord;
  });
}

function parseRunDetailTable(value: unknown, path: string): RunDetailTableRecord {
  const record = requireRecord(value, path, transportError);
  const columns = stringArray(record.columns, `${path}.columns`);
  const rows = arrayOf(record.rows, `${path}.rows`, stringArray);
  rows.forEach((row, index) => {
    if (row.length !== columns.length) {
      throw transportError(`${path}.rows.${index}: expected ${columns.length} cells`);
    }
  });
  return {
    aria_label: requireString(record.aria_label, `${path}.aria_label`, transportError),
    columns,
    rows,
  };
}

function parseTrainingDiagnostics(value: unknown, path: string): TrainingDiagnosticsRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, ['status', 'stop_reason']);
  requireNumbers(record, path, [
    'steps_run',
    'validation_checks',
    'final_validation_loss',
    'final_validation_step',
    'final_validation_check',
  ]);
  return withFields(record, {
    training_compute: optionalNumber(record.training_compute, `${path}.training_compute`, transportError),
    protocol: requireRecord(record.protocol, `${path}.protocol`, transportError) as TrainingProtocolRecord,
    validation_history: requireArray(record.validation_history, `${path}.validation_history`, transportError) as TrainingHistoryPointRecord[],
    artifacts: requireArray(record.artifacts, `${path}.artifacts`, transportError) as TrainingArtifactReferenceRecord[],
    throughput:
      record.throughput === undefined
        ? undefined
        : requireRecord(record.throughput, `${path}.throughput`, transportError),
    evaluation_curriculum:
      record.evaluation_curriculum === undefined
        ? undefined
        : requireRecord(record.evaluation_curriculum, `${path}.evaluation_curriculum`, transportError),
  }) as TrainingDiagnosticsRecord;
}

function parseCostSummary(value: unknown, path: string): CostSummaryRecord {
  const record = requireRecord(value, path, transportError);
  return {
    ...record,
    component_count: requireNumber(record.component_count, `${path}.component_count`, transportError),
    parameter_count: optionalNumber(record.parameter_count, `${path}.parameter_count`, transportError),
    storage_bytes: optionalNumber(record.storage_bytes, `${path}.storage_bytes`, transportError),
    inference_compute: optionalNumber(record.inference_compute, `${path}.inference_compute`, transportError),
    training_compute_per_sample: optionalNumber(record.training_compute_per_sample, `${path}.training_compute_per_sample`, transportError),
    training_compute: optionalNumber(record.training_compute, `${path}.training_compute`, transportError),
    unknown_parameter_components:
      record.unknown_parameter_components === undefined
        ? undefined
        : numberArray(record.unknown_parameter_components, `${path}.unknown_parameter_components`),
  };
}

function parseScoreBasis(value: unknown, path: string): Record<string, unknown> {
  return requireRecord(value, path, transportError);
}

function parseFrontiers(value: unknown, path: string): Record<string, ModelResultRecord[]> {
  const record = requireRecord(value, path, transportError);
  return Object.fromEntries(
    Object.entries(record).map(([key, models]) => [
      key,
      arrayOf(models, `${path}.${key}`, parseModelResult),
    ]),
  );
}

function arrayOf<T>(value: unknown, path: string, parse: (item: unknown, path: string) => T): T[] {
  return requireArray(value, path, transportError).map((item, index) => parse(item, `${path}.${index}`));
}

function optional<T>(value: unknown, path: string, parse: (item: unknown, path: string) => T): T | undefined {
  return value === undefined ? undefined : parse(value, path);
}

function parseNumber(value: unknown, path: string): number {
  return requireNumber(value, path, transportError);
}

function requireStrings(record: Record<string, unknown>, path: string, fields: string[]): void {
  fields.forEach((field) => requireString(record[field], `${path}.${field}`, transportError));
}

function requireNumbers(record: Record<string, unknown>, path: string, fields: string[]): void {
  fields.forEach((field) => requireNumber(record[field], `${path}.${field}`, transportError));
}

function withFields<T extends Record<string, unknown>>(
  record: Record<string, unknown>,
  fields: T,
): Record<string, unknown> & T {
  return { ...record, ...fields };
}

function stringArray(value: unknown, path: string): string[] {
  return arrayOf(value, path, (item, itemPath) => requireString(item, itemPath, transportError));
}

function numberArray(value: unknown, path: string): number[] {
  return arrayOf(value, path, (item, itemPath) => requireNumber(item, itemPath, transportError));
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
    args = parser.parse_args(argv)

    protocol_path = args.path or _generated_protocol_module_path
    result_view_records_path = (
        args.result_view_records_path or _generated_result_view_records_module_path
    )
    expected_modules = (
        (protocol_path, generated_console_protocol_module()),
        (result_view_records_path, generated_console_result_view_records_module()),
    )
    if args.check:
        for path, expected in expected_modules:
            actual = path.read_text(encoding="utf-8")
            if actual != expected:
                raise SystemExit(f"{path}: generated console web module is out of date")
        return 0
    write_generated_console_protocol_module(protocol_path)
    write_generated_console_result_view_records_module(result_view_records_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
