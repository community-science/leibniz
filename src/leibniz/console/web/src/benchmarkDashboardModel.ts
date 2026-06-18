import type { ModelInspectionRecord } from './modelInspections.ts';
import {
  isBenchmarkResultView,
  type BenchmarkResultRecord,
  type ModelResultRecord,
  type ResultViewRecord,
  type RunResultRecord,
} from './resultViews.ts';

export type BenchmarkResultEntry = {
  sourcePath: string;
  sourceMtimeMs?: number;
  sourceSizeBytes?: number;
  result: BenchmarkResultRecord;
};

export type BenchmarkModelComparisonRow = {
  model: ModelResultRecord;
  inspection?: ModelInspectionRecord;
};

export type BenchmarkPlotModelPoint = {
  id: string;
  label: string;
  cost: number;
  logCost: number;
  score: number;
  frontier: boolean;
  resultStatus: 'accepted' | 'provisional';
  model?: ModelResultRecord;
  run?: RunResultRecord;
};

export type BenchmarkReferenceCurvePoint = {
  cost: number;
  logCost: number;
  score: number;
};

export type BenchmarkReferenceCurve = {
  key: string;
  label: string;
  points: BenchmarkReferenceCurvePoint[];
};

export type BenchmarkPlotModel = {
  points: BenchmarkPlotModelPoint[];
  frontierPoints: BenchmarkPlotModelPoint[];
  referenceCurves: BenchmarkReferenceCurve[];
  xDomain: [number, number];
  xLogBase: number;
  yDomain: [number, number];
  xTicks: number[];
  xMajorTicks: number[];
  xMinorTicks: number[];
  yTicks: number[];
  staircase: [number, number][];
};
const fallbackLogCostDomain: [number, number] = [0, 10];
const defaultScoreMaximum = 1;
const denseLogTickThreshold = 14;
const targetScoreTickCount = 8;
export const benchmarkCostAxisKey = 'cost';
export const benchmarkCostAxisLabel = 'Cost';

export type ModelResultSortKey = 'score' | 'cost' | 'model' | 'runs' | 'measurements';
export type SortDirection = 'ascending' | 'descending';

export type ModelResultSort = {
  key: ModelResultSortKey;
  direction: SortDirection;
};

export type BenchmarkSelection = {
  selectedModel?: ModelResultRecord;
  selectedRun?: RunResultRecord;
};

export function emptyFrontiersForCostAxis(): Record<string, ModelResultRecord[]> {
  return { [benchmarkCostAxisKey]: [] };
}

export function benchmarkResultsForTask(
  resultViews: ResultViewRecord[],
  benchmarkId: string,
): BenchmarkResultEntry[] {
  return resultViews
    .filter(isBenchmarkResultView)
    .flatMap((view) =>
      view.benchmark_results.map((result) => ({
        sourceMtimeMs: view.source_mtime_ms,
        sourcePath: view.source_path,
        sourceSizeBytes: view.source_size_bytes,
        result,
      })),
    )
    .filter((entry) => entry.result.benchmark_id === benchmarkId);
}

export function modelComparisonRows(
  result: BenchmarkResultRecord | undefined,
  inspections: ModelInspectionRecord[],
): BenchmarkModelComparisonRow[] {
  if (result === undefined) {
    return [];
  }

  const inspectionsByDigest = new Map<string, ModelInspectionRecord>();
  for (const inspection of [...inspections, ...result.model_inspections]) {
    for (const digest of [
      inspection.program.record_digest,
      inspection.program.content_digest,
    ]) {
      if (digest !== undefined) {
        inspectionsByDigest.set(normalizedDigest(digest), inspection);
      }
    }
  }

  return result.model_candidates.map((model) => ({
    model,
    inspection: inspectionsByDigest.get(normalizedDigest(model.program_digest)),
  }));
}

export function benchmarkPlotModel(
  result: BenchmarkResultRecord,
): BenchmarkPlotModel {
  const frontierModels = frontierModelResults(result.leaderboard);
  const frontierKeys = new Set(frontierModels.map((model) => model.model_key));
  const leaderboardModels = modelLookup(result.leaderboard);
  const candidateModels = modelLookup(result.model_candidates);
  const points = result.plot_runs
    .map((run) => plotRunPoint(run, leaderboardModels, candidateModels, frontierKeys))
    .filter((point): point is BenchmarkPlotModelPoint => point !== null)
    .sort((left, right) => left.cost - right.cost || right.score - left.score);
  const frontierPoints = frontierModels
    .map((model) => plotPoint(model, true))
    .filter((point): point is BenchmarkPlotModelPoint => point !== null)
    .sort((left, right) => left.cost - right.cost || right.score - left.score);
  const referenceCurves = referenceCurvePlotModels(result);
  const xLogBase = costAxisLogBase();
  const costLogs = points.map((point) => point.logCost);
  const scores = [...points, ...frontierPoints].map((point) => point.score);
  const xDomain = logCostDomain(costLogs);
  const yDomain = scoreDomain(scores);
  const xTicks = logCostTicks(xDomain, xLogBase);
  return {
    points,
    frontierPoints,
    referenceCurves,
    xDomain,
    xLogBase,
    yDomain,
    xTicks: [...xTicks.major, ...xTicks.minor].sort((left, right) => left - right),
    xMajorTicks: xTicks.major,
    xMinorTicks: xTicks.minor,
    yTicks: scoreTicks(yDomain),
    staircase: staircasePoints(frontierPoints),
  };
}

function referenceCurvePlotModels(
  result: BenchmarkResultRecord,
): BenchmarkReferenceCurve[] {
  return (result.reference_curves ?? [])
    .filter((curve) => curve.x_axis === benchmarkCostAxisKey && curve.y_axis === 'score')
    .map((curve) => ({
      key: curve.key,
      label: curve.label,
      points: curve.points
        .filter((point) =>
          Number.isFinite(point.cost) &&
          point.cost > 0 &&
          Number.isFinite(point.score),
        )
        .map((point) => ({
          cost: point.cost,
          logCost: logCost(point.cost),
          score: point.score,
        }))
        .sort((left, right) => left.cost - right.cost || left.score - right.score),
    }))
    .filter((curve) => curve.points.length >= 2);
}

export function sortedModelResults(
  models: ModelResultRecord[],
  sort: ModelResultSort,
): ModelResultRecord[] {
  return [...models].sort((left, right) => compareModelResults(left, right, sort));
}

export function nextModelResultSort(
  current: ModelResultSort,
  key: ModelResultSortKey,
): ModelResultSort {
  if (current.key !== key) {
    return { key, direction: defaultSortDirection(key) };
  }
  return {
    key,
    direction: current.direction === 'ascending' ? 'descending' : 'ascending',
  };
}

export function selectionForId(
  result: BenchmarkResultRecord,
  selectedId: string | null,
): BenchmarkSelection {
  if (selectedId === null) {
    return {};
  }
  return {
    selectedModel: result.leaderboard.find((model) => model.model_key === selectedId),
    selectedRun: [...result.plot_runs, ...result.training_history].find(
      (run) => runSelectionId(run) === selectedId,
    ),
  };
}

export function costValue(costSummary: Record<string, unknown>): number {
  const value = costSummary[benchmarkCostAxisKey];
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

export function scoreValue(model: ModelResultRecord): number {
  return model.score;
}

export function formatCost(value: number): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

export function scoreLabel(value: number | undefined): string {
  return value === undefined || !Number.isFinite(value) ? 'n/a' : value.toFixed(4);
}

export function scoreTickLabel(value: number): string {
  const abs = Math.abs(value);
  const maximumFractionDigits = abs >= 100 ? 0 : abs >= 10 ? 1 : abs >= 1 ? 2 : 3;
  return value.toLocaleString(undefined, { maximumFractionDigits });
}

export function shortDigest(value: string): string {
  const digest = normalizedDigest(value);
  return digest.slice(0, 12);
}

export function runSelectionId(run: RunResultRecord): string {
  return `run:${run.source_kind}:${run.run_id}`;
}

function normalizedDigest(value: string): string {
  return value.includes(':') ? value.split(':').at(-1) ?? value : value;
}

function modelLookup(models: ModelResultRecord[]): {
  byModelKey: Map<string, ModelResultRecord>;
} {
  return {
    byModelKey: new Map(models.map((model) => [model.model_key, model])),
  };
}

function compareModelResults(
  left: ModelResultRecord,
  right: ModelResultRecord,
  sort: ModelResultSort,
): number {
  const direction = sort.direction === 'ascending' ? 1 : -1;
  const result = compareBySortKey(left, right, sort.key);
  if (result !== 0) {
    return result * direction;
  }
  return (
    sortableScoreValue(right) - sortableScoreValue(left) ||
    costValue(left.cost_summary) - costValue(right.cost_summary)
  );
}

function compareBySortKey(
  left: ModelResultRecord,
  right: ModelResultRecord,
  key: ModelResultSortKey,
): number {
  if (key === 'score') {
    return sortableScoreValue(left) - sortableScoreValue(right);
  }
  if (key === 'cost') {
    return costValue(left.cost_summary) - costValue(right.cost_summary);
  }
  if (key === 'runs') {
    return left.run_ids.length - right.run_ids.length;
  }
  if (key === 'measurements') {
    return left.measurement_count - right.measurement_count;
  }
  return left.model_key.localeCompare(right.model_key);
}

function sortableScoreValue(model: ModelResultRecord): number {
  const value = scoreValue(model);
  return Number.isFinite(value) ? value : -Infinity;
}

function defaultSortDirection(key: ModelResultSortKey): SortDirection {
  return key === 'cost' || key === 'model' ? 'ascending' : 'descending';
}

function plotPoint(
  model: ModelResultRecord,
  frontier: boolean,
): BenchmarkPlotModelPoint | null {
  const cost = costValue(model.cost_summary);
  const score = scoreValue(model);
  if (!Number.isFinite(cost) || cost <= 0 || !Number.isFinite(score)) {
    return null;
  }
  return {
    id: model.model_key,
    label: shortDigest(model.program_digest),
    cost,
    logCost: logCost(cost),
    score,
    frontier,
    resultStatus: 'accepted',
    model,
  };
}

function plotRunPoint(
  run: RunResultRecord,
  leaderboardModels: ReturnType<typeof modelLookup>,
  candidateModels: ReturnType<typeof modelLookup>,
  frontierKeys: Set<string>,
): BenchmarkPlotModelPoint | null {
  const leaderboardModel = leaderboardModels.byModelKey.get(run.model_key);
  const candidateModel = candidateModels.byModelKey.get(run.model_key);
  const model = run.result_status === 'provisional'
    ? candidateModel ?? leaderboardModel
    : leaderboardModel;
  const cost = costValue(run.cost_summary);
  const score = model === undefined
    ? run.result_status === 'provisional' ? run.score : Number.NaN
    : scoreValue(model);
  if (!Number.isFinite(cost) || cost <= 0 || !Number.isFinite(score)) {
    return null;
  }
  return {
    id: runSelectionId(run),
    label: shortDigest(run.program_digest),
    cost,
    logCost: logCost(cost),
    score,
    frontier: run.result_status === 'accepted' && model !== undefined && frontierKeys.has(model.model_key),
    resultStatus: run.result_status,
    model,
    run,
  };
}

function frontierModelResults(
  models: ModelResultRecord[],
): ModelResultRecord[] {
  const ordered = [...models]
    .filter((model) => {
      const cost = costValue(model.cost_summary);
      return cost > 0 && Number.isFinite(cost) && Number.isFinite(scoreValue(model));
    })
    .sort((left, right) =>
      costValue(left.cost_summary) - costValue(right.cost_summary) ||
      scoreValue(right) - scoreValue(left),
    );
  const frontier: ModelResultRecord[] = [];
  let bestScore = -Infinity;
  for (const model of ordered) {
    const score = scoreValue(model);
    if (score > bestScore) {
      frontier.push(model);
      bestScore = score;
    }
  }
  return frontier;
}

function logCostDomain(values: number[]): [number, number] {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) {
    return fallbackLogCostDomain;
  }
  const min = Math.min(fallbackLogCostDomain[0], Math.floor(Math.min(...finite)));
  const max = Math.max(fallbackLogCostDomain[1], Math.ceil(Math.max(...finite)));
  return [min, max];
}

function costAxisLogBase(): number {
  return 10;
}

function logCost(cost: number): number {
  return Math.log(cost) / Math.log(costAxisLogBase());
}

function scoreDomain(values: number[]): [number, number] {
  const defaultMaximum = defaultScoreMaximum;
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) {
    return [0, defaultMaximum];
  }
  const min = Math.min(0, Math.floor(Math.min(...finite)));
  const max = Math.max(...finite);
  if (max <= defaultMaximum) {
    return [min, defaultMaximum];
  }
  const expandedMaximum = 2 * max;
  const step = niceScoreTickStep(max / Math.max(1, targetScoreTickCount - 1));
  return [min, normalizedTick(Math.ceil(expandedMaximum / step) * step)];
}

function scoreTicks([min, max]: [number, number]): number[] {
  const span = max - min;
  if (!Number.isFinite(span) || span <= 0) {
    return [min];
  }
  const step = niceScoreTickStep(span / Math.max(1, targetScoreTickCount - 1));
  const first = Math.ceil(min / step) * step;
  const last = Math.floor(max / step) * step;
  const ticks: number[] = [];
  for (let tick = first; tick <= last + step / 2; tick += step) {
    ticks.push(normalizedTick(tick));
  }
  if (ticks.length === 0) {
    ticks.push(normalizedTick((min + max) / 2));
  }
  return ticks;
}

function niceScoreTickStep(rawStep: number): number {
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  const multiplier = normalized <= 1
    ? 1
    : normalized <= 2
      ? 2
      : normalized <= 2.5
        ? 2.5
        : normalized <= 5
          ? 5
          : 10;
  return multiplier * magnitude;
}

function normalizedTick(value: number): number {
  return Number(value.toPrecision(12));
}

function logCostTicks(
  [min, max]: [number, number],
  logBase: number,
): { major: number[]; minor: number[] } {
  const first = Math.ceil(min);
  const last = Math.floor(max);
  const labelStep = logBase === 10 || last - first <= denseLogTickThreshold ? 1 : 2;
  const major: number[] = [];
  const minor: number[] = [];
  for (let exponent = first; exponent <= last; exponent += 1) {
    const tick = logBase ** exponent;
    if ((exponent - first) % labelStep === 0) {
      major.push(tick);
    } else {
      minor.push(tick);
    }
  }
  return { major, minor };
}

function staircasePoints(points: BenchmarkPlotModelPoint[]): [number, number][] {
  if (points.length === 0) {
    return [];
  }
  const sorted = [...points].sort((left, right) => left.cost - right.cost || right.score - left.score);
  const coordinates: [number, number][] = [];
  let previousScore = sorted[0].score;
  coordinates.push([sorted[0].logCost, previousScore]);
  for (const point of sorted.slice(1)) {
    coordinates.push([point.logCost, previousScore]);
    coordinates.push([point.logCost, point.score]);
    previousScore = point.score;
  }
  return coordinates;
}
