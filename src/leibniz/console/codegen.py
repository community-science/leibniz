"""Generate small console web-source modules from Python-owned protocol metadata."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from leibniz.console.protocol import (
    console_protocol_format_versions,
    console_protocol_formats,
)
from leibniz.record_contracts import typescript_literal
from leibniz.work_queues import WorkQueueItem

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

    return _result_view_records_module


def generated_console_work_queue_records_module() -> str:
    """Return the generated TypeScript work-queue record parser module."""

    return WorkQueueItem.typescript_record_module()


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
import {
  parseWorkQueueItem,
  type WorkQueueItemRecord,
} from './workQueueRecords.ts';

export type { WorkQueueItemRecord, WorkQueueItemStatus } from './workQueueRecords.ts';

const resultViewFormats = consoleProtocolFormats.resultViews;
const resultViewFormatVersion = consoleProtocolFormatVersions.resultView;

type ResultViewBaseRecord = {
  format_version: typeof resultViewFormatVersion;
  source_path: string;
  source_mtime_ms?: number;
  source_size_bytes?: number;
};

export type ResultViewRecord =
  | ImportedResultViewRecord
  | BenchmarkResultViewRecord
  | WorkQueueViewRecord;

export type ImportedResultViewRecord = ResultViewBaseRecord & {
  format: typeof resultViewFormats.importedResults;
  publication_bundles: ImportedPublicationBundleRecord[];
};

export type BenchmarkResultViewRecord = ResultViewBaseRecord & {
  format: typeof resultViewFormats.benchmarkResults;
  benchmark_results: BenchmarkResultRecord[];
};

export type WorkQueueViewRecord = ResultViewBaseRecord & {
  format: typeof resultViewFormats.workQueue;
  queue_items: WorkQueueItemRecord[];
};

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

export type CompetencePointRecord = {
  complexity: number;
  score: number;
  sample_count?: number;
  run_ids: string[];
};

export type CostSummaryRecord = {
  component_count: number;
  parameter_count?: number;
  parameter_bytes?: number;
  inference_flops?: number;
  unknown_parameter_components?: number[];
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

const transportError = (message: string) => new Error(message);

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
  return requireArray(value, 'result_views', transportError).map((view, index) =>
    parseResultViewRecord(view, `result_views.${index}`),
  );
}

function parseResultViewRecord(value: unknown, path: string): ResultViewRecord {
  const record = requireRecord(value, path, transportError);
  requireLiteral(
    record.format_version,
    `${path}.format_version`,
    resultViewFormatVersion,
    transportError,
  );
  requireString(record.source_path, `${path}.source_path`, transportError);
  optionalNumber(record.source_mtime_ms, `${path}.source_mtime_ms`, transportError);
  optionalNumber(record.source_size_bytes, `${path}.source_size_bytes`, transportError);
  if (record.format === resultViewFormats.benchmarkResults) {
    return parseBenchmarkResultViewRecord(record, path);
  }
  if (record.format === resultViewFormats.workQueue) {
    return parseWorkQueueViewRecord(record, path);
  }
  requireLiteral(
    record.format,
    `${path}.format`,
    resultViewFormats.importedResults,
    transportError,
  );
  requireArray(record.publication_bundles, `${path}.publication_bundles`, transportError);
  return record as unknown as ImportedResultViewRecord;
}

function parseBenchmarkResultViewRecord(
  record: Record<string, unknown>,
  path: string,
): BenchmarkResultViewRecord {
  requireArray(record.benchmark_results, `${path}.benchmark_results`, transportError).forEach(
    (result, index) => parseBenchmarkResult(result, `${path}.benchmark_results.${index}`),
  );
  return record as unknown as BenchmarkResultViewRecord;
}

function parseWorkQueueViewRecord(
  record: Record<string, unknown>,
  path: string,
): WorkQueueViewRecord {
  requireArray(record.queue_items, `${path}.queue_items`, transportError).forEach((item, index) =>
    parseWorkQueueItem(item, `${path}.queue_items.${index}`),
  );
  return record as unknown as WorkQueueViewRecord;
}

function parseBenchmarkResult(value: unknown, path: string): void {
  const record = requireRecord(value, path, transportError);
  requireString(record.benchmark_id, `${path}.benchmark_id`, transportError);
  requireArray(record.cost_axes, `${path}.cost_axes`, transportError);
  requireArray(record.leaderboard, `${path}.leaderboard`, transportError).forEach((model, index) =>
    parseModelResult(model, `${path}.leaderboard.${index}`),
  );
  requireRecord(record.frontiers, `${path}.frontiers`, transportError);
  requireArray(record.training_history, `${path}.training_history`, transportError).forEach(
    (run, index) => parseRunResult(run, `${path}.training_history.${index}`),
  );
  if (record.model_inspections !== undefined) {
    requireArray(record.model_inspections, `${path}.model_inspections`, transportError).forEach(
      (inspection, index) =>
        parseModelInspectionRecord(inspection, `${path}.model_inspections.${index}`),
    );
  }
}

function parseModelResult(value: unknown, path: string): void {
  const record = requireRecord(value, path, transportError);
  requireString(record.model_key, `${path}.model_key`, transportError);
  requireNumber(record.score, `${path}.score`, transportError);
  requireRecord(record.cost_summary, `${path}.cost_summary`, transportError);
  requireNumber(record.measurement_count, `${path}.measurement_count`, transportError);
}

function parseRunResult(value: unknown, path: string): void {
  const record = requireRecord(value, path, transportError);
  requireString(record.source_kind, `${path}.source_kind`, transportError);
  requireString(record.run_id, `${path}.run_id`, transportError);
  requireString(record.model_key, `${path}.model_key`, transportError);
  requireNumber(record.measurement_count, `${path}.measurement_count`, transportError);
  requireNumber(record.score, `${path}.score`, transportError);
  requireRecord(record.cost_summary, `${path}.cost_summary`, transportError);
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
