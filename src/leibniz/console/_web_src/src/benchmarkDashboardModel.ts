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
  resultStatus: 'accepted' | 'tentative';
  model?: ModelResultRecord;
  run?: RunResultRecord;
};

export type BenchmarkPlotModel = {
  points: BenchmarkPlotModelPoint[];
  frontierPoints: BenchmarkPlotModelPoint[];
  xDomain: [number, number];
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
  label: string;
};

const fallbackLogCostDomain: [number, number] = [0, 20];
const fallbackScoreDomain: [number, number] = [0, 1.05];
const denseLogTickThreshold = 14;
const standardScoreAxes: ScoreAxisRecord[] = [
  { key: 'absolute', label: 'Absolute' },
  { key: 'relative', label: 'Relative' },
];
const standardCostAxisGroups: BenchmarkCostAxisGroup[] = [
  {
    axes: [
      { key: 'parameter_count', label: 'Parameters' },
      { key: 'storage_bytes', label: 'Storage' },
    ],
    key: 'model',
    label: 'Model',
  },
  {
    axes: [{ key: 'inference_compute', label: 'Compute' }],
    key: 'inference',
    label: 'Inference',
  },
  {
    axes: [{ key: 'training_compute', label: 'Compute' }],
    key: 'training',
    label: 'Training',
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
  const models = modelLookup(result.leaderboard);
  const points = result.plot_runs
    .map((run) => plotRunPoint(run, models, costAxis, scoreAxis, frontierKeys))
    .filter((point): point is BenchmarkPlotModelPoint => point !== null)
    .sort((left, right) => left.cost - right.cost || right.score - left.score);
  const frontierPoints = frontierModels
    .map((model) => plotPoint(model, costAxis, scoreAxis, true))
    .filter((point): point is BenchmarkPlotModelPoint => point !== null)
    .sort((left, right) => left.cost - right.cost || right.score - left.score);
  const costLogs = points.map((point) => point.logCost);
  const scores = points.map((point) => point.score);
  const xDomain = logCostDomain(costLogs);
  const yDomain = scoreDomain(scores);
  const xTicks = logCostTicks(xDomain);
  return {
    points,
    frontierPoints,
    xDomain,
    yDomain,
    xTicks: [...xTicks.major, ...xTicks.minor].sort((left, right) => left - right),
    xMajorTicks: xTicks.major,
    xMinorTicks: xTicks.minor,
    yTicks: scoreTicks(yDomain),
    staircase: staircasePoints(frontierPoints),
  };
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
  byArchitecture: Map<string, ModelResultRecord>;
  byModelKey: Map<string, ModelResultRecord>;
} {
  return {
    byArchitecture: new Map(
      models.map((model) => [normalizedDigest(model.architecture_digest), model]),
    ),
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
    logCost: Math.log2(cost),
    score,
    frontier,
    resultStatus: 'accepted',
    model,
  };
}

function plotRunPoint(
  run: RunResultRecord,
  models: ReturnType<typeof modelLookup>,
  costAxis: string,
  scoreAxis: string,
  frontierKeys: Set<string>,
): BenchmarkPlotModelPoint | null {
  const model =
    models.byModelKey.get(run.model_key) ??
    models.byArchitecture.get(normalizedDigest(run.architecture_digest));
  const cost = costValue(run.cost_summary, costAxis);
  const score = run.result_status === 'tentative' && scoreAxis === 'absolute'
    ? run.score
    : model === undefined ? Number.NaN : scoreValue(model, scoreAxis);
  if (!Number.isFinite(cost) || cost <= 0 || !Number.isFinite(score)) {
    return null;
  }
  return {
    id: runSelectionId(run),
    label: shortDigest(run.architecture_digest),
    cost,
    logCost: Math.log2(cost),
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

function scoreDomain(values: number[]): [number, number] {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) {
    return fallbackScoreDomain;
  }
  const max = Math.max(1, Math.ceil(Math.max(...finite)));
  return [Math.min(0, Math.floor(Math.min(...finite))), max * 1.05];
}

function scoreTicks([min, max]: [number, number]): number[] {
  const first = Math.ceil(min);
  const last = Math.floor(max);
  const labelStep = last - first > denseLogTickThreshold ? 2 : 1;
  const ticks: number[] = [];
  for (let exponent = first; exponent <= last; exponent += labelStep) {
    ticks.push(exponent);
  }
  if (ticks.length === 0) {
    ticks.push(Math.round((min + max) / 2));
  }
  return ticks;
}

function logCostTicks([min, max]: [number, number]): { major: number[]; minor: number[] } {
  const first = Math.ceil(min);
  const last = Math.floor(max);
  const labelStep = last - first > denseLogTickThreshold ? 2 : 1;
  const major: number[] = [];
  const minor: number[] = [];
  for (let exponent = first; exponent <= last; exponent += 1) {
    const tick = 2 ** exponent;
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
