import type { ModelInspectionRecord } from './modelInspections.ts';
import {
  isBenchmarkResultView,
  type BenchmarkResultRecord,
  type ModelResultRecord,
  type ResultViewRecord,
} from './resultViews.ts';
import type { PerformanceViewRecord } from './performanceViews.ts';

export type BenchmarkResultEntry = {
  sourcePath: string;
  result: BenchmarkResultRecord;
};

export type BenchmarkModelComparisonRow = {
  model: ModelResultRecord;
  inspection?: ModelInspectionRecord;
};

export function benchmarkResultsForTask(
  resultViews: ResultViewRecord[],
  benchmarkId: string,
): BenchmarkResultEntry[] {
  return resultViews
    .filter(isBenchmarkResultView)
    .flatMap((view) =>
      view.benchmark_results.map((result) => ({ sourcePath: view.source_path, result })),
    )
    .filter((entry) => entry.result.benchmark_id === benchmarkId);
}

export function performanceViewsForTask(
  views: PerformanceViewRecord[],
  benchmarkId: string,
): PerformanceViewRecord[] {
  return views.filter((view) =>
    view.manifest.benchmark_manifest.protocol_id === benchmarkId ||
    view.competence_integral_view.entries.some((entry) => entry.benchmark_id === benchmarkId),
  );
}

export function modelComparisonRows(
  result: BenchmarkResultRecord | undefined,
  inspections: ModelInspectionRecord[],
): BenchmarkModelComparisonRow[] {
  if (result === undefined) {
    return [];
  }

  const inspectionsByDigest = new Map<string, ModelInspectionRecord>();
  for (const inspection of inspections) {
    for (const digest of [
      inspection.architecture.record_digest,
      inspection.architecture.content_digest,
    ]) {
      if (digest !== undefined) {
        inspectionsByDigest.set(normalizedDigest(digest), inspection);
      }
    }
  }

  return result.leaderboard.map((model) => ({
    model,
    inspection: inspectionsByDigest.get(normalizedDigest(model.architecture_digest)),
  }));
}

export function costValue(costSummary: Record<string, unknown>, costAxis: string): number {
  const value = costSummary[costAxis];
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

export function formatCost(value: number): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

export function scoreLabel(value: number | undefined): string {
  return value === undefined ? 'n/a' : value.toFixed(4);
}

export function shortDigest(value: string): string {
  const digest = normalizedDigest(value);
  return digest.slice(0, 12);
}

function normalizedDigest(value: string): string {
  return value.includes(':') ? value.split(':').at(-1) ?? value : value;
}
