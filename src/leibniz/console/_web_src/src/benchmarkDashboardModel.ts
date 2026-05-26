import type { ModelInspectionRecord } from './modelInspections.ts';
import {
  isBenchmarkResultView,
  type BenchmarkResultRecord,
  type ModelResultRecord,
  type ProposalRecord,
  type ResultViewRecord,
  type RunResultRecord,
} from './resultViews.ts';

export type BenchmarkResultEntry = {
  sourcePath: string;
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
  model: ModelResultRecord;
};

export type BenchmarkPlotProposal = {
  id: string;
  label: string;
  cost: number;
  logCost: number;
  predictedScore: number;
  uncertainty?: number;
  proposal: ProposalRecord;
};

export type BenchmarkPlotModel = {
  points: BenchmarkPlotModelPoint[];
  frontierPoints: BenchmarkPlotModelPoint[];
  proposals: BenchmarkPlotProposal[];
  xDomain: [number, number];
  yDomain: [number, number];
  xTicks: number[];
  yTicks: number[];
  staircase: [number, number][];
};

export type ModelResultSortKey = 'score' | 'cost' | 'model' | 'runs' | 'measurements';
export type SortDirection = 'ascending' | 'descending';

export type ModelResultSort = {
  key: ModelResultSortKey;
  direction: SortDirection;
};

export type BenchmarkProposalAssociation = {
  proposal: ProposalRecord;
  model?: ModelResultRecord;
};

export type BenchmarkRunDetail = {
  run: RunResultRecord;
  model?: ModelResultRecord;
};

export type BenchmarkSelection = {
  selectedModel?: ModelResultRecord;
  selectedProposal?: ProposalRecord;
  selectedRun?: RunResultRecord;
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

export function benchmarkPlotModel(
  result: BenchmarkResultRecord,
  costAxis: string,
): BenchmarkPlotModel {
  const frontierKeys = new Set((result.frontiers[costAxis] ?? []).map((model) => model.model_key));
  const points = result.leaderboard
    .map((model) => plotPoint(model, costAxis, frontierKeys.has(model.model_key)))
    .filter((point): point is BenchmarkPlotModelPoint => point !== null)
    .sort((left, right) => left.cost - right.cost || right.score - left.score);
  const frontierPoints = (result.frontiers[costAxis] ?? [])
    .map((model) => plotPoint(model, costAxis, true))
    .filter((point): point is BenchmarkPlotModelPoint => point !== null)
    .sort((left, right) => left.cost - right.cost || right.score - left.score);
  const pointByModelKey = new Map(points.map((point) => [point.model.model_key, point]));
  const pointByArchitecture = new Map(
    points.map((point) => [normalizedDigest(point.model.architecture_digest), point]),
  );
  const proposals = result.proposals
    .map((proposal) => plotProposal(proposal, pointByModelKey, pointByArchitecture))
    .filter((proposal): proposal is BenchmarkPlotProposal => proposal !== null);
  const costLogs = [
    ...points.map((point) => point.logCost),
    ...proposals.map((proposal) => proposal.logCost),
  ];
  const scores = [
    ...points.map((point) => point.score),
    ...proposals.map((proposal) => proposal.predictedScore),
    ...proposals.flatMap((proposal) =>
      proposal.uncertainty === undefined
        ? []
        : [
            proposal.predictedScore - proposal.uncertainty,
            proposal.predictedScore + proposal.uncertainty,
          ],
    ),
  ];
  const xDomain = paddedDomain(costLogs, 0.5, [0, 1]);
  const yDomain = scoreDomain(scores);
  return {
    points,
    frontierPoints,
    proposals,
    xDomain,
    yDomain,
    xTicks: logCostTicks(xDomain),
    yTicks: linearTicks(yDomain, 5),
    staircase: staircasePoints(frontierPoints),
  };
}

export function sortedModelResults(
  models: ModelResultRecord[],
  costAxis: string,
  sort: ModelResultSort,
): ModelResultRecord[] {
  return [...models].sort((left, right) => compareModelResults(left, right, costAxis, sort));
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

export function proposalAssociations(
  result: BenchmarkResultRecord,
): BenchmarkProposalAssociation[] {
  const models = modelLookup(result.leaderboard);
  return result.proposals.map((proposal) => ({
    proposal,
    model:
      models.byModelKey.get(proposal.candidate_id) ??
      models.byArchitecture.get(normalizedDigest(proposal.candidate_id)),
  }));
}

export function runDetails(result: BenchmarkResultRecord): BenchmarkRunDetail[] {
  const models = modelLookup(result.leaderboard);
  return result.training_history.map((run) => ({
    run,
    model:
      models.byModelKey.get(run.model_key) ??
      models.byArchitecture.get(normalizedDigest(run.architecture_digest)),
  }));
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
    selectedProposal: result.proposals.find((proposal) => proposal.id === selectedId),
    selectedRun: result.training_history.find((run) => runSelectionId(run) === selectedId),
  };
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
  sort: ModelResultSort,
): number {
  const direction = sort.direction === 'ascending' ? 1 : -1;
  const result = compareBySortKey(left, right, costAxis, sort.key);
  if (result !== 0) {
    return result * direction;
  }
  return (
    right.score - left.score ||
    costValue(left.cost_summary, costAxis) - costValue(right.cost_summary, costAxis)
  );
}

function compareBySortKey(
  left: ModelResultRecord,
  right: ModelResultRecord,
  costAxis: string,
  key: ModelResultSortKey,
): number {
  if (key === 'score') {
    return left.score - right.score;
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

function defaultSortDirection(key: ModelResultSortKey): SortDirection {
  return key === 'cost' || key === 'model' ? 'ascending' : 'descending';
}

function plotPoint(
  model: ModelResultRecord,
  costAxis: string,
  frontier: boolean,
): BenchmarkPlotModelPoint | null {
  const cost = costValue(model.cost_summary, costAxis);
  if (!Number.isFinite(cost) || cost <= 0 || !Number.isFinite(model.score)) {
    return null;
  }
  return {
    id: model.model_key,
    label: shortDigest(model.architecture_digest),
    cost,
    logCost: Math.log2(cost),
    score: model.score,
    frontier,
    model,
  };
}

function plotProposal(
  proposal: ProposalRecord,
  pointByModelKey: Map<string, BenchmarkPlotModelPoint>,
  pointByArchitecture: Map<string, BenchmarkPlotModelPoint>,
): BenchmarkPlotProposal | null {
  if (proposal.predicted_score === undefined || !Number.isFinite(proposal.predicted_score)) {
    return null;
  }
  const point =
    pointByModelKey.get(proposal.candidate_id) ??
    pointByArchitecture.get(normalizedDigest(proposal.candidate_id));
  if (point === undefined) {
    return null;
  }
  return {
    id: proposal.id,
    label: proposal.candidate_id,
    cost: point.cost,
    logCost: point.logCost,
    predictedScore: proposal.predicted_score,
    uncertainty:
      proposal.uncertainty === undefined || !Number.isFinite(proposal.uncertainty)
        ? undefined
        : proposal.uncertainty,
    proposal,
  };
}

function paddedDomain(values: number[], padding: number, fallback: [number, number]): [number, number] {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) {
    return fallback;
  }
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  if (min === max) {
    return [min - padding, max + padding];
  }
  return [min - padding, max + padding];
}

function scoreDomain(values: number[]): [number, number] {
  const finite = values.filter(Number.isFinite);
  if (finite.length === 0) {
    return [0, 1];
  }
  const min = Math.min(0, ...finite);
  const max = Math.max(1, ...finite);
  const padding = Math.max(0.02, (max - min) * 0.05);
  return [min, max + padding];
}

function logCostTicks([min, max]: [number, number]): number[] {
  const first = Math.ceil(min);
  const last = Math.floor(max);
  return Array.from({ length: Math.max(0, last - first + 1) }, (_value, index) => 2 ** (first + index));
}

function linearTicks([min, max]: [number, number], count: number): number[] {
  if (count <= 1 || min === max) {
    return [min];
  }
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_value, index) => min + step * index);
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
