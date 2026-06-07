import type { ModelInspectionRecord } from './modelInspections.ts';
import {
  isBenchmarkResultView,
  type BenchmarkResultRecord,
  type CostAxisRecord,
  type ModelResultRecord,
  type ResultViewRecord,
  type RunResultRecord,
  type ScoreAxisRecord,
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
export type BenchmarkCostAxisGroup = {
  axes: CostAxisRecord[];
  key: string;
};

const fallbackLogCostDomain: [number, number] = [0, 10];
const defaultAbsoluteScoreMaximum = 1;
const defaultRelativeScoreMaximum = 2000;
const denseLogTickThreshold = 14;
const targetScoreTickCount = 8;
const expandedRelativeScoreTickCount = 12;
const standardScoreAxes: ScoreAxisRecord[] = [
  { key: 'absolute', label: 'Absolute Score' },
  { key: 'relative', label: 'Relative Score' },
];
const standardCostAxisGroups: BenchmarkCostAxisGroup[] = [
  {
    axes: [{ key: 'inference_compute', label: 'Inference Compute' }],
    key: 'inference',
  },
  {
    axes: [{ key: 'storage_bytes', label: 'Model Size' }],
    key: 'model',
  },
  {
    axes: [{ key: 'training_compute', label: 'Training Compute' }],
    key: 'training',
  },
];
const standardCostAxes = standardCostAxisGroups.flatMap((group) => group.axes);

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

export function benchmarkCostAxes(
  result: BenchmarkResultRecord | undefined,
): CostAxisRecord[] {
  const axes = result?.cost_axes ?? [];
  const seen = new Set(axes.map((axis) => axis.key));
  return [
    ...axes,
    ...standardCostAxes.filter((axis) => !seen.has(axis.key)),
  ];
}

export function benchmarkCostAxisGroups(costAxes: CostAxisRecord[]): BenchmarkCostAxisGroup[] {
  const axesByKey = new Map(costAxes.map((axis) => [axis.key, axis]));
  const standardKeys = new Set(standardCostAxes.map((axis) => axis.key));
  const groups = standardCostAxisGroups.map((group) => ({
    ...group,
    axes: group.axes
      .map((axis) => axesByKey.get(axis.key))
      .filter((axis): axis is CostAxisRecord => axis !== undefined),
  }));
  const customAxes = costAxes.filter((axis) => !standardKeys.has(axis.key));
  if (customAxes.length > 0) {
    groups[0] = {
      ...groups[0]!,
      axes: [...groups[0]!.axes, ...customAxes],
    };
  }
  return groups.filter((group) => group.axes.length > 0);
}

export function benchmarkCostAxis(
  selectedAxis: string,
  axes: CostAxisRecord[],
): string {
  if (axes.some((axis) => axis.key === selectedAxis)) {
    return selectedAxis;
  }
  return axes[0]?.key ?? standardCostAxes[0]!.key;
}

export function benchmarkScoreAxes(
  result: BenchmarkResultRecord | undefined,
): ScoreAxisRecord[] {
  const axes = result?.score_axes ?? [];
  const seen = new Set(axes.map((axis) => axis.key));
  return [
    ...axes,
    ...standardScoreAxes.filter((axis) => !seen.has(axis.key)),
  ];
}

export function benchmarkScoreAxis(
  selectedAxis: string,
  axes: ScoreAxisRecord[],
): string {
  if (axes.some((axis) => axis.key === selectedAxis)) {
    return selectedAxis;
  }
  return axes[0]?.key ?? standardScoreAxes[0]!.key;
}

export function emptyFrontiersForCostAxes(
  axes: CostAxisRecord[],
): Record<string, ModelResultRecord[]> {
  return Object.fromEntries(axes.map((axis) => [axis.key, []]));
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
      inspection.architecture.record_digest,
      inspection.architecture.content_digest,
    ]) {
      if (digest !== undefined) {
        inspectionsByDigest.set(normalizedDigest(digest), inspection);
      }
    }
  }

  return result.model_candidates.map((model) => ({
    model,
    inspection: inspectionsByDigest.get(normalizedDigest(model.architecture_digest)),
  }));
}

export function benchmarkPlotModel(
  result: BenchmarkResultRecord,
  costAxis: string,
  scoreAxis = 'absolute',
): BenchmarkPlotModel {
  const frontierModels = frontierModelResults(result.leaderboard, costAxis, scoreAxis);
  const frontierKeys = new Set(frontierModels.map((model) => model.model_key));
  const leaderboardModels = modelLookup(result.leaderboard);
  const candidateModels = modelLookup(result.model_candidates);
  const points = result.plot_runs
    .map((run) => plotRunPoint(run, leaderboardModels, candidateModels, costAxis, scoreAxis, frontierKeys))
    .filter((point): point is BenchmarkPlotModelPoint => point !== null)
    .sort((left, right) => left.cost - right.cost || right.score - left.score);
  const frontierPoints = frontierModels
    .map((model) => plotPoint(model, costAxis, scoreAxis, true))
    .filter((point): point is BenchmarkPlotModelPoint => point !== null)
    .sort((left, right) => left.cost - right.cost || right.score - left.score);
  const referenceCurves = referenceCurvePlotModels(result, costAxis, scoreAxis);
  const xLogBase = costAxisLogBase();
  const costLogs = points.map((point) => point.logCost);
  const scores = [...points, ...frontierPoints].map((point) => point.score);
  const xDomain = logCostDomain(costLogs);
  const yDomain = scoreDomain(scores, scoreAxis);
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
  costAxis: string,
  scoreAxis: string,
): BenchmarkReferenceCurve[] {
  return (result.reference_curves ?? [])
    .filter((curve) => curve.x_axis === costAxis && curve.y_axis === scoreAxis)
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
  costAxis: string,
  scoreAxis: string,
  sort: ModelResultSort,
): ModelResultRecord[] {
  return [...models].sort((left, right) => compareModelResults(left, right, costAxis, scoreAxis, sort));
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

export function costValue(costSummary: Record<string, unknown>, costAxis: string): number {
  const value = costSummary[costAxis];
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

export function scoreValue(model: ModelResultRecord, scoreAxis: string): number {
  const viewScore = model.score_views?.[scoreAxis]?.score;
  if (typeof viewScore === 'number' && Number.isFinite(viewScore)) {
    return viewScore;
  }
  return scoreAxis === 'absolute' ? model.score : Number.NaN;
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
  costAxis: string,
  scoreAxis: string,
  sort: ModelResultSort,
): number {
  const direction = sort.direction === 'ascending' ? 1 : -1;
  const result = compareBySortKey(left, right, costAxis, scoreAxis, sort.key);
  if (result !== 0) {
    return result * direction;
  }
  return (
    sortableScoreValue(right, scoreAxis) - sortableScoreValue(left, scoreAxis) ||
    costValue(left.cost_summary, costAxis) - costValue(right.cost_summary, costAxis)
  );
}

function compareBySortKey(
  left: ModelResultRecord,
  right: ModelResultRecord,
  costAxis: string,
  scoreAxis: string,
  key: ModelResultSortKey,
): number {
  if (key === 'score') {
    return sortableScoreValue(left, scoreAxis) - sortableScoreValue(right, scoreAxis);
  }
  if (key === 'cost') {
    return costValue(left.cost_summary, costAxis) - costValue(right.cost_summary, costAxis);
  }
  if (key === 'runs') {
    return left.run_ids.length - right.run_ids.length;
  }
  if (key === 'measurements') {
    return left.measurement_count - right.measurement_count;
  }
  return left.model_key.localeCompare(right.model_key);
}

function sortableScoreValue(model: ModelResultRecord, scoreAxis: string): number {
  const value = scoreValue(model, scoreAxis);
  return Number.isFinite(value) ? value : -Infinity;
}

function defaultSortDirection(key: ModelResultSortKey): SortDirection {
  return key === 'cost' || key === 'model' ? 'ascending' : 'descending';
}

function plotPoint(
  model: ModelResultRecord,
  costAxis: string,
  scoreAxis: string,
  frontier: boolean,
): BenchmarkPlotModelPoint | null {
  const cost = costValue(model.cost_summary, costAxis);
  const score = scoreValue(model, scoreAxis);
  if (!Number.isFinite(cost) || cost <= 0 || !Number.isFinite(score)) {
    return null;
  }
  return {
    id: model.model_key,
    label: shortDigest(model.architecture_digest),
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
  costAxis: string,
  scoreAxis: string,
  frontierKeys: Set<string>,
): BenchmarkPlotModelPoint | null {
  const leaderboardModel = leaderboardModels.byModelKey.get(run.model_key);
  const candidateModel = candidateModels.byModelKey.get(run.model_key);
  const model = run.result_status === 'provisional'
    ? candidateModel ?? leaderboardModel
    : leaderboardModel;
  const cost = costValue(run.cost_summary, costAxis);
  const score = model === undefined
    ? run.result_status === 'provisional' && scoreAxis === 'absolute'
      ? run.score
      : Number.NaN
    : scoreValue(model, scoreAxis);
  if (!Number.isFinite(cost) || cost <= 0 || !Number.isFinite(score)) {
    return null;
  }
  return {
    id: runSelectionId(run),
    label: shortDigest(run.architecture_digest),
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
  costAxis: string,
  scoreAxis: string,
): ModelResultRecord[] {
  const ordered = [...models]
    .filter((model) => {
      const cost = costValue(model.cost_summary, costAxis);
      return cost > 0 && Number.isFinite(cost) && Number.isFinite(scoreValue(model, scoreAxis));
    })
    .sort((left, right) =>
      costValue(left.cost_summary, costAxis) - costValue(right.cost_summary, costAxis) ||
      scoreValue(right, scoreAxis) - scoreValue(left, scoreAxis),
    );
  const frontier: ModelResultRecord[] = [];
  let bestScore = -Infinity;
  for (const model of ordered) {
    const score = scoreValue(model, scoreAxis);
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

function scoreDomain(values: number[], scoreAxis: string): [number, number] {
  const defaultMaximum = defaultScoreMaximum(scoreAxis);
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

function defaultScoreMaximum(scoreAxis: string): number {
  return scoreAxis === 'relative' ? defaultRelativeScoreMaximum : defaultAbsoluteScoreMaximum;
}

function scoreTicks([min, max]: [number, number]): number[] {
  const span = max - min;
  if (!Number.isFinite(span) || span <= 0) {
    return [min];
  }
  const targetTickCount = max > defaultRelativeScoreMaximum
    ? expandedRelativeScoreTickCount
    : targetScoreTickCount;
  const step = niceScoreTickStep(span / Math.max(1, targetTickCount - 1));
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
