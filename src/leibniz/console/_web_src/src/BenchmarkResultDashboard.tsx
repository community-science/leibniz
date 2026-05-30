import {
  ArrowLeft,
  ArrowRight,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { useState } from 'react';

import {
  benchmarkPlotModel,
  benchmarkCostAxes,
  benchmarkCostAxis,
  costValue,
  formatCost,
  nextModelResultSort,
  proposalAssociations,
  runDetails,
  runSelectionId,
  scoreLabel,
  selectionForId,
  shortDigest,
  sortedModelResults,
  type BenchmarkProposalAssociation,
  type BenchmarkRunDetail,
  type ModelResultSort,
  type ModelResultSortKey,
} from './benchmarkDashboardModel.ts';
import type {
  BenchmarkResultRecord,
  CostAxisRecord,
  ModelResultRecord,
  ProposalRecord,
  RunDetailSectionRecord,
} from './resultViews.ts';
import {
  coordinateDisplayName,
  type OperatorVocabularyRecord,
} from './operatorVocabulary.ts';

type PlotView = {
  xDomain: [number, number];
  yDomain: [number, number];
};

const plotWidth = 960;
const plotHeight = 440;
const plotMargin = {
  bottom: 58,
  left: 72,
  right: 26,
  top: 26,
};
const plotBodyWidth = plotWidth - plotMargin.left - plotMargin.right;
const plotBodyHeight = plotHeight - plotMargin.top - plotMargin.bottom;
const plotZoomInFactor = 0.72;
const plotZoomOutFactor = 1.28;
const plotPanFraction = 0.18;
const plotTickOffset = 18;
const plotYTickLabelOffset = 10;
const plotTickLabelBaselineOffset = 4;
const plotAxisSelectorTopOffset = 31;
const plotAxisSelectorHeight = 25;
const plotAxisSelectorMinWidth = 264;
const plotAxisSelectorButtonWidth = 84;
const proposalIntervalCapHalfWidth = 7;

export function BenchmarkResultDashboard({
  operatorVocabulary,
  result,
}: {
  operatorVocabulary: OperatorVocabularyRecord;
  result: BenchmarkResultRecord;
}) {
  const costAxes = benchmarkCostAxes(result);
  const [selectedCostAxis, setSelectedCostAxis] = useState(costAxes[0]?.key ?? 'parameter_count');
  const costAxis = benchmarkCostAxis(selectedCostAxis, costAxes);
  const [plotView, setPlotView] = useState<PlotView | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [leaderboardSort, setLeaderboardSort] = useState<ModelResultSort>({
    key: 'score',
    direction: 'descending',
  });
  const plot = benchmarkPlotModel(result, costAxis);
  const selection = selectionForId(result, selectedId);
  const proposalRows = proposalAssociations(result);
  const selectedProposalAssociation = proposalRows.find(
    ({ proposal }) => proposal.id === selection.selectedProposal?.id,
  );
  const runRows = runDetails(result);
  const selectedRunDetail = runRows.find(
    ({ run }) => runSelectionId(run) === selectedId,
  );
  const selectedModelKey =
    selection.selectedModel?.model_key ??
    selectedProposalAssociation?.model?.model_key ??
    selectedRunDetail?.model?.model_key;
  const activeView = plotView ?? {
    xDomain: plot.xDomain,
    yDomain: plot.yDomain,
  };

  return (
    <section className="performance-section benchmark-result-dashboard">
      <BenchmarkFrontierPlot
        costAxes={costAxes}
        costAxis={costAxis}
        model={plot}
        onCostAxisChange={(axis) => {
          setSelectedCostAxis(axis);
          setPlotView(null);
          setHoveredId(null);
          setSelectedId(null);
        }}
        onHover={setHoveredId}
        onPan={(direction) => setPlotView(pannedView(activeView, direction, plot))}
        onReset={() => setPlotView(null)}
        onSelect={setSelectedId}
        onZoom={(factor) => setPlotView(zoomedView(activeView, factor, plot))}
        selectedId={selectedId}
        view={activeView}
        hoveredId={hoveredId}
      />
      <ModelResultTable
        complexityAxis={result.complexity_axis}
        costAxis={costAxis}
        models={result.leaderboard}
        onSelect={setSelectedId}
        onSort={(key) => setLeaderboardSort((current) => nextModelResultSort(current, key))}
        selectedModelKey={selectedModelKey}
        sort={leaderboardSort}
        title="Leaderboard"
      />
      <ProposalCards
        associations={proposalRows}
        onSelect={setSelectedId}
        operatorVocabulary={operatorVocabulary}
        selectedId={selectedId}
      />
      <RunHistoryTable
        complexityAxis={result.complexity_axis}
        costAxis={costAxis}
        onSelect={setSelectedId}
        rows={runRows}
        selectedId={selectedId}
        selectedRunDetail={selectedRunDetail}
      />
    </section>
  );
}

function BenchmarkFrontierPlot({
  costAxes,
  costAxis,
  hoveredId,
  model,
  onCostAxisChange,
  onHover,
  onPan,
  onReset,
  onSelect,
  onZoom,
  selectedId,
  view,
}: {
  costAxes: CostAxisRecord[];
  costAxis: string;
  hoveredId: string | null;
  model: ReturnType<typeof benchmarkPlotModel>;
  onCostAxisChange: (axis: string) => void;
  onHover: (id: string | null) => void;
  onPan: (direction: 'left' | 'right') => void;
  onReset: () => void;
  onSelect: (id: string | null) => void;
  onZoom: (factor: number) => void;
  selectedId: string | null;
  view: PlotView;
}) {
  const x = (logCost: number) =>
    plotMargin.left +
    ((logCost - view.xDomain[0]) / (view.xDomain[1] - view.xDomain[0])) * plotBodyWidth;
  const y = (score: number) =>
    plotMargin.top +
    (1 - (score - view.yDomain[0]) / (view.yDomain[1] - view.yDomain[0])) * plotBodyHeight;
  const visiblePoints = model.points.filter(
    (point) =>
      point.logCost >= view.xDomain[0] &&
      point.logCost <= view.xDomain[1] &&
      point.score >= view.yDomain[0] &&
      point.score <= view.yDomain[1],
  );
  const visibleProposals = model.proposals.filter(
    (proposal) =>
      proposal.logCost >= view.xDomain[0] &&
      proposal.logCost <= view.xDomain[1] &&
      proposal.predictedScore >= view.yDomain[0] &&
      proposal.predictedScore <= view.yDomain[1],
  );
  const selectedPoint =
    model.points.find((point) => point.id === selectedId) ??
    model.proposals.find((proposal) => proposal.id === selectedId);
  const hoveredPoint =
    model.points.find((point) => point.id === hoveredId) ??
    model.proposals.find((proposal) => proposal.id === hoveredId);
  const activePoint = hoveredPoint ?? selectedPoint;
  const axisSelectorWidth = Math.min(
    plotBodyWidth,
    Math.max(plotAxisSelectorMinWidth, costAxes.length * plotAxisSelectorButtonWidth),
  );
  const axisSelectorX = plotMargin.left + plotBodyWidth / 2 - axisSelectorWidth / 2;
  const axisSelectorY = plotMargin.top + plotBodyHeight + plotAxisSelectorTopOffset;

  return (
    <section className="benchmark-result-table-section">
      <div className="benchmark-plot-heading">
        <div>
          <h3>Measurements Plot</h3>
        </div>
        <div className="benchmark-plot-actions">
          <button
            aria-label="Pan left"
            onClick={() => onPan('left')}
            title="Pan left"
            type="button"
          >
            <ArrowLeft aria-hidden="true" size={14} />
          </button>
          <button
            aria-label="Pan right"
            onClick={() => onPan('right')}
            title="Pan right"
            type="button"
          >
            <ArrowRight aria-hidden="true" size={14} />
          </button>
          <button
            aria-label="Zoom in"
            onClick={() => onZoom(plotZoomInFactor)}
            title="Zoom in"
            type="button"
          >
            <ZoomIn aria-hidden="true" size={14} />
          </button>
          <button
            aria-label="Zoom out"
            onClick={() => onZoom(plotZoomOutFactor)}
            title="Zoom out"
            type="button"
          >
            <ZoomOut aria-hidden="true" size={14} />
          </button>
          <button
            aria-label="Reset plot"
            onClick={onReset}
            title="Reset plot"
            type="button"
          >
            <RotateCcw aria-hidden="true" size={14} />
          </button>
        </div>
      </div>
      <div className="frontier-chart">
        <div className="frontier-chart-legend" aria-label="Measurements plot legend">
          <span><i className="frontier" />Frontier</span>
          <span><i className="measured" />Measured</span>
          <span><i className="proposal" />Proposal</span>
        </div>
        <svg
          aria-label={`Measurements by ${costAxis}`}
          className="frontier-chart-svg"
          role="img"
          viewBox={`0 0 ${plotWidth} ${plotHeight}`}
          onClick={() => onSelect(null)}
        >
            <rect
              className="frontier-chart-frame"
              height={plotBodyHeight}
              width={plotBodyWidth}
              x={plotMargin.left}
              y={plotMargin.top}
            />
            {model.xMinorTicks.map((tick) => {
              const logTick = Math.log2(tick);
              if (logTick < view.xDomain[0] || logTick > view.xDomain[1]) {
                return null;
              }
              const tickX = x(logTick);
              return (
                <line
                  className="frontier-chart-grid frontier-chart-grid-minor"
                  key={`x-minor-${tick}`}
                  x1={tickX}
                  x2={tickX}
                  y1={plotMargin.top}
                  y2={plotMargin.top + plotBodyHeight}
                />
              );
            })}
            {model.xMajorTicks.map((tick) => {
              const logTick = Math.log2(tick);
              if (logTick < view.xDomain[0] || logTick > view.xDomain[1]) {
                return null;
              }
              const tickX = x(logTick);
              return (
                <g key={`x-major-${tick}`}>
                  <line
                    className="frontier-chart-grid"
                    x1={tickX}
                    x2={tickX}
                    y1={plotMargin.top}
                    y2={plotMargin.top + plotBodyHeight}
                  />
                  <line
                    className="frontier-chart-axis-tick"
                    x1={tickX}
                    x2={tickX}
                    y1={plotMargin.top + plotBodyHeight}
                    y2={plotMargin.top + plotBodyHeight + 5}
                  />
                  <text
                    className="frontier-chart-tick"
                    textAnchor="middle"
                    x={tickX}
                    y={plotMargin.top + plotBodyHeight + plotTickOffset}
                  >
                    2<tspan dy="-5" fontSize="0.72em">{Math.round(logTick)}</tspan><tspan dy="5"> </tspan>
                  </text>
                </g>
              );
            })}
            {model.yTicks.map((tick) => {
              if (tick < view.yDomain[0] || tick > view.yDomain[1]) {
                return null;
              }
              const tickY = y(tick);
              return (
                <g key={`y-${tick}`}>
                  <line
                    className="frontier-chart-grid"
                    x1={plotMargin.left}
                    x2={plotMargin.left + plotBodyWidth}
                    y1={tickY}
                    y2={tickY}
                  />
                  <line
                    className="frontier-chart-axis-tick"
                    x1={plotMargin.left - 4}
                    x2={plotMargin.left}
                    y1={tickY}
                    y2={tickY}
                  />
                  <text
                    className="frontier-chart-tick"
                    textAnchor="end"
                    x={plotMargin.left - plotYTickLabelOffset}
                    y={tickY + plotTickLabelBaselineOffset}
                  >
                    {tick.toFixed(2)}
                  </text>
                </g>
              );
            })}
            {model.staircase.length > 0 ? (
              <polyline
                className="frontier-chart-staircase"
                fill="none"
                points={model.staircase.map(([logCost, score]) => `${x(logCost)},${y(score)}`).join(' ')}
              />
            ) : null}
            {visibleProposals.map((proposal) => {
              const proposalX = x(proposal.logCost);
              const proposalY = y(proposal.predictedScore);
              const uncertainty = proposal.uncertainty;
              return (
                <g key={proposal.id}>
                  {uncertainty === undefined ? null : (
                    <g className="frontier-chart-proposal-interval">
                      <line
                        className="frontier-chart-proposal-band"
                        x1={proposalX}
                        x2={proposalX}
                        y1={y(proposal.predictedScore - uncertainty)}
                        y2={y(proposal.predictedScore + uncertainty)}
                      />
                      <line
                        className="frontier-chart-proposal-cap"
                        x1={proposalX - proposalIntervalCapHalfWidth}
                        x2={proposalX + proposalIntervalCapHalfWidth}
                        y1={y(proposal.predictedScore - uncertainty)}
                        y2={y(proposal.predictedScore - uncertainty)}
                      />
                      <line
                        className="frontier-chart-proposal-cap"
                        x1={proposalX - proposalIntervalCapHalfWidth}
                        x2={proposalX + proposalIntervalCapHalfWidth}
                        y1={y(proposal.predictedScore + uncertainty)}
                        y2={y(proposal.predictedScore + uncertainty)}
                      />
                    </g>
                  )}
                  <line
                    className="frontier-chart-proposal-guide"
                    x1={proposalX}
                    x2={proposalX}
                    y1={plotMargin.top}
                    y2={plotMargin.top + plotBodyHeight}
                  />
                  <circle
                    className={[
                      'frontier-chart-proposal',
                      selectedId === proposal.id ? 'selected' : '',
                      hoveredId === proposal.id ? 'hovered' : '',
                    ].filter(Boolean).join(' ')}
                    cx={proposalX}
                    cy={proposalY}
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelect(proposal.id);
                    }}
                    onMouseEnter={() => onHover(proposal.id)}
                    onMouseLeave={() => onHover(null)}
                    r={6}
                  />
                </g>
              );
            })}
            {visiblePoints.map((point) => (
              <circle
                className={[
                  'frontier-chart-point',
                  point.frontier ? 'frontier' : '',
                  selectedId === point.id ? 'selected' : '',
                  hoveredId === point.id ? 'hovered' : '',
                ].filter(Boolean).join(' ')}
                cx={x(point.logCost)}
                cy={y(point.score)}
                key={point.id}
                onClick={(event) => {
                  event.stopPropagation();
                  onSelect(point.id);
                }}
                onMouseEnter={() => onHover(point.id)}
                onMouseLeave={() => onHover(null)}
                r={point.frontier ? 5 : 3}
              />
            ))}
            <foreignObject
              height={plotAxisSelectorHeight}
              width={axisSelectorWidth}
              x={axisSelectorX}
              y={axisSelectorY}
            >
              <div className="frontier-chart-axis-selector">
                {costAxes.map((axis) => (
                  <button
                    aria-pressed={axis.key === costAxis}
                    className={axis.key === costAxis ? 'active' : ''}
                    key={axis.key}
                    onClick={(event) => {
                      event.stopPropagation();
                      onCostAxisChange(axis.key);
                    }}
                    onPointerDown={(event) => event.stopPropagation()}
                    type="button"
                  >
                    {axis.label}
                  </button>
                ))}
              </div>
            </foreignObject>
            <text
              className="frontier-chart-axis-label"
              textAnchor="middle"
              transform={`rotate(-90 ${18} ${plotMargin.top + plotBodyHeight / 2})`}
              x={18}
              y={plotMargin.top + plotBodyHeight / 2}
            >
              Score
            </text>
            {model.points.length === 0 ? (
              <text
                className="frontier-chart-empty-label"
                textAnchor="middle"
                x={plotMargin.left + plotBodyWidth / 2}
                y={plotMargin.top + plotBodyHeight / 2}
              >
                No model results yet
              </text>
            ) : null}
        </svg>
        {activePoint === undefined ? null : (
          <div className="frontier-chart-tooltip">
            <span className="frontier-chart-tooltip-kicker">
              {'predictedScore' in activePoint ? 'Proposal' : activePoint.frontier ? 'Frontier highlight' : 'Measured model'}
            </span>
            <strong>{activePoint.label}</strong>
            <span>{formatCost(activePoint.cost)} cost</span>
            <span>
              {'predictedScore' in activePoint
                ? `prediction ${scoreLabel(activePoint.predictedScore)}`
                : `score ${scoreLabel(activePoint.score)}`}
            </span>
            {'uncertainty' in activePoint && activePoint.uncertainty !== undefined ? (
              <span>uncertainty +/- {scoreLabel(activePoint.uncertainty)}</span>
            ) : null}
          </div>
        )}
      </div>
    </section>
  );
}

function ProposalCards({
  associations,
  onSelect,
  operatorVocabulary,
  selectedId,
}: {
  associations: BenchmarkProposalAssociation[];
  onSelect: (id: string) => void;
  operatorVocabulary: OperatorVocabularyRecord;
  selectedId: string | null;
}) {
  if (associations.length === 0) {
    return null;
  }

  return (
    <section className="benchmark-result-table-section">
      <h3>Proposals</h3>
      <div className="proposal-card-grid">
        {associations.map(({ model, proposal }) => {
          return (
            <button
              className={`proposal-card ${selectedId === proposal.id ? 'selected' : ''}`}
              key={proposal.id}
              onClick={() => onSelect(proposal.id)}
              type="button"
            >
              <div className="proposal-card-heading">
                <span>Rank {proposal.rank}</span>
                <strong>{scoreLabel(proposal.acquisition_value)}</strong>
              </div>
              <dl>
                <dt>Candidate</dt>
                <dd>{proposal.candidate_id}</dd>
                <dt>Acquisition</dt>
                <dd>{proposal.acquisition_model ?? 'not recorded'}</dd>
                <dt>Prediction</dt>
                <dd>{scoreLabel(proposal.predicted_score)}</dd>
                <dt>Uncertainty</dt>
                <dd>{scoreLabel(proposal.uncertainty)}</dd>
                <dt>Improvement</dt>
                <dd>{scoreLabel(proposal.expected_frontier_improvement)}</dd>
                <dt>Selector</dt>
                <dd>{proposal.selector_name ?? 'none'}</dd>
                <dt>Cost Stratum</dt>
                <dd>{resourceStratumLabel(proposal)}</dd>
                <dt>Search</dt>
                <dd>{searchDistributionLabel(proposal)}</dd>
                <dt>Coordinates</dt>
                <dd>{semanticCoordinateSummary(proposal, operatorVocabulary)}</dd>
                <dt>Nearest Evidence</dt>
                <dd>{nearestMeasuredSupportLabel(proposal)}</dd>
                <dt>Comparable Score</dt>
                <dd>{scoreLabel(proposal.comparable_cost_best_score)}</dd>
                <dt>Matched Model</dt>
                <dd>{model === undefined ? 'none' : shortDigest(model.architecture_digest)}</dd>
              </dl>
              <ProposalAcquisitionComponents proposal={proposal} />
              <p>{proposal.rationale}</p>
              {proposal.command.length === 0 ? null : (
                <code className="proposal-card-command">{proposal.command.join(' ')}</code>
              )}
            </button>
          );
        })}
      </div>
    </section>
  );
}

function ProposalAcquisitionComponents({ proposal }: { proposal: ProposalRecord }) {
  const rows = acquisitionComponentRows(proposal);
  if (rows.length === 0) {
    return null;
  }
  return (
    <dl className="proposal-acquisition-components">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{scoreLabel(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function acquisitionComponentRows(proposal: ProposalRecord): Array<[string, number]> {
  const components = proposal.acquisition_components;
  if (components === undefined) {
    return [];
  }
  return [
    ['Estimated', componentNumber(components, 'estimated_score')],
    ['Explore', componentNumber(components, 'exploration_value')],
    ['Novelty', componentNumber(components, 'resource_novelty')],
    ['Frontier', componentNumber(components, 'expected_frontier_improvement')],
    ['Baseline', componentNumber(components, 'comparable_cost_best_score')],
  ].filter((row): row is [string, number] => row[1] !== undefined);
}

function componentNumber(components: Record<string, unknown>, key: string): number | undefined {
  const value = components[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function resourceStratumLabel(proposal: ProposalRecord): string {
  if (
    proposal.resource_stratum_index === undefined ||
    proposal.resource_stratum_count === undefined
  ) {
    return 'none';
  }
  return `${proposal.resource_stratum_index + 1}/${proposal.resource_stratum_count}`;
}

function searchDistributionLabel(proposal: ProposalRecord): string {
  const value = proposal.search_diagnostics?.search_distribution_id;
  return typeof value === 'string' ? shortDigest(value) : 'not recorded';
}

function semanticCoordinateSummary(
  proposal: ProposalRecord,
  operatorVocabulary: OperatorVocabularyRecord,
): string {
  const coordinates = proposal.search_diagnostics?.semantic_coordinates;
  if (!Array.isArray(coordinates)) {
    return 'not recorded';
  }
  const primaryCoordinate = coordinates.find(isNamedCoordinate);
  if (primaryCoordinate !== undefined) {
    return `${coordinates.length} coordinates, ${coordinateDisplayName(operatorVocabulary, primaryCoordinate.name)} ${primaryCoordinate.value}`;
  }
  return `${coordinates.length} coordinates`;
}

function nearestMeasuredSupportLabel(proposal: ProposalRecord): string {
  const support = proposal.search_diagnostics?.nearest_measured_support;
  if (!isRecord(support)) {
    return 'none';
  }
  const parameterCount = support.parameter_count;
  const score = support.score;
  if (typeof parameterCount !== 'number' || typeof score !== 'number') {
    return 'none';
  }
  return `${formatCost(parameterCount)} params at ${scoreLabel(score)}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isNamedCoordinate(
  value: unknown,
): value is { name: string; value: string | number } {
  if (!isRecord(value) || typeof value.name !== 'string') {
    return false;
  }
  return typeof value.value === 'string' || typeof value.value === 'number';
}

function ModelResultTable({
  complexityAxis,
  costAxis,
  models,
  onSelect,
  onSort,
  selectedModelKey,
  sort,
  title,
}: {
  complexityAxis: string | undefined;
  costAxis: string;
  models: ModelResultRecord[];
  onSelect: (id: string) => void;
  onSort: (key: ModelResultSortKey) => void;
  selectedModelKey: string | undefined;
  sort: ModelResultSort;
  title: string;
}) {
  if (models.length === 0) {
    return null;
  }

  return (
    <section className="benchmark-result-table-section">
      <h3>{title}</h3>
      <div className="benchmark-result-table" role="table" aria-label={title}>
        <div className="benchmark-result-row benchmark-model-result-row header" role="row">
          <SortHeader
            active={sort.key === 'model'}
            direction={sort.direction}
            label="Model"
            onClick={() => onSort('model')}
          />
          <SortHeader
            active={sort.key === 'score'}
            direction={sort.direction}
            label="Score"
            onClick={() => onSort('score')}
          />
          <SortHeader
            active={sort.key === 'cost'}
            direction={sort.direction}
            label="Cost"
            onClick={() => onSort('cost')}
          />
          <span role="columnheader">{complexityAxis ?? 'Complexity'}</span>
          <SortHeader
            active={sort.key === 'runs'}
            direction={sort.direction}
            label="Runs"
            onClick={() => onSort('runs')}
          />
          <SortHeader
            active={sort.key === 'measurements'}
            direction={sort.direction}
            label="Measurements"
            onClick={() => onSort('measurements')}
          />
        </div>
        {sortedModelResults(models, costAxis, sort).map((model) => (
          <button
            className={[
              'benchmark-result-row',
              'benchmark-model-result-row',
              selectedModelKey === model.model_key ? 'selected' : '',
            ].filter(Boolean).join(' ')}
            key={model.model_key}
            onClick={() => onSelect(model.model_key)}
            role="row"
            type="button"
          >
            <span role="cell">{shortDigest(model.architecture_digest)}</span>
            <span role="cell">{model.score.toFixed(4)}</span>
            <span role="cell">{formatCost(costValue(model.cost_summary, costAxis))}</span>
            <span role="cell">{model.observed_complexities.join(', ') || 'none'}</span>
            <span role="cell">{model.run_ids.length}</span>
            <span role="cell">{model.measurement_count}</span>
          </button>
        ))}
      </div>
    </section>
  );
}

function SortHeader({
  active,
  direction,
  label,
  onClick,
}: {
  active: boolean;
  direction: 'ascending' | 'descending';
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      aria-sort={active ? direction : 'none'}
      className={`benchmark-sort-header ${active ? 'active' : ''}`}
      onClick={onClick}
      role="columnheader"
      type="button"
    >
      <span>{label}</span>
      {active ? <span aria-hidden="true">{direction === 'ascending' ? 'asc' : 'desc'}</span> : null}
    </button>
  );
}

function RunHistoryTable({
  complexityAxis,
  costAxis,
  onSelect,
  rows,
  selectedId,
  selectedRunDetail,
}: {
  complexityAxis: string | undefined;
  costAxis: string;
  onSelect: (id: string) => void;
  rows: BenchmarkRunDetail[];
  selectedId: string | null;
  selectedRunDetail: BenchmarkRunDetail | undefined;
}) {
  if (rows.length === 0) {
    return null;
  }

  return (
    <section className="benchmark-result-table-section">
      <h3>Training History</h3>
      <div className="benchmark-result-table" role="table" aria-label="Training history">
        <div className="benchmark-result-row header" role="row">
          <span role="columnheader">Run</span>
          <span role="columnheader">Score</span>
          <span role="columnheader">Cost</span>
          <span role="columnheader">{complexityAxis ?? 'Complexity'}</span>
          <span role="columnheader">Measurements</span>
        </div>
        {rows.map(({ run }) => {
          const id = runSelectionId(run);
          return (
            <button
              className={`benchmark-result-row ${selectedId === id ? 'selected' : ''}`}
              key={id}
              onClick={() => onSelect(id)}
              role="row"
              type="button"
            >
              <span role="cell">{run.run_slug}</span>
              <span role="cell">{run.score.toFixed(4)}</span>
              <span role="cell">{formatCost(costValue(run.cost_summary, costAxis))}</span>
              <span role="cell">{run.complexity ?? run.scale ?? 'n/a'}</span>
              <span role="cell">{run.measurement_count}</span>
            </button>
          );
        })}
      </div>
      {selectedRunDetail === undefined ? null : (
        <RunDetailCard
          costAxis={costAxis}
          detail={selectedRunDetail}
        />
      )}
    </section>
  );
}

function RunDetailCard({
  costAxis,
  detail,
}: {
  costAxis: string;
  detail: BenchmarkRunDetail;
}) {
  const { model, run } = detail;
  const detailSections = run.console_view_model?.detail_sections ?? [];
  return (
    <article className="run-detail-card">
      <div>
        <h4>{run.run_slug}</h4>
        <p>{run.source_path}</p>
      </div>
      <dl>
        <dt>Score</dt>
        <dd>{run.score.toFixed(4)}</dd>
        <dt>Cost</dt>
        <dd>{formatCost(costValue(run.cost_summary, costAxis))}</dd>
        <dt>Architecture</dt>
        <dd>{shortDigest(run.architecture_digest)}</dd>
        <dt>Matched Model</dt>
        <dd>{model === undefined ? 'none' : shortDigest(model.architecture_digest)}</dd>
        <dt>Measurements</dt>
        <dd>{run.measurement_count}</dd>
        <dt>Dataset</dt>
        <dd>{shortDigest(run.measurement_dataset_digest)}</dd>
        <dt>Source</dt>
        <dd>{run.source_kind}</dd>
        <dt>Model Inspection</dt>
        <dd>
          {run.model_inspection_digest === undefined
            ? 'none'
            : shortDigest(run.model_inspection_digest)}
        </dd>
      </dl>
      {detailSections.map((section) => (
        <RunDetailSection key={section.title} section={section} />
      ))}
    </article>
  );
}

function RunDetailSection({ section }: { section: RunDetailSectionRecord }) {
  const entries = section.entries ?? [];
  const table = section.table;
  return (
    <section className="run-evidence-panel">
      <h5>{section.title}</h5>
      {entries.length === 0 ? null : (
        <dl>
          {entries.map((entry) => (
            <div key={entry.label}>
              <dt>{entry.label}</dt>
              <dd>{entry.value}</dd>
            </div>
          ))}
        </dl>
      )}
      {table === undefined ? null : (
        <div className="run-validation-table" role="table" aria-label={table.aria_label}>
          <div className="run-validation-row header" role="row">
            {table.columns.map((column) => (
              <span key={column} role="columnheader">{column}</span>
            ))}
          </div>
          {table.rows.map((row, index) => (
            <div className="run-validation-row" key={index} role="row">
              {row.map((value, valueIndex) => (
                <span key={`${index}-${valueIndex}`} role="cell">{value}</span>
              ))}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function zoomedView(
  view: PlotView,
  factor: number,
  model: ReturnType<typeof benchmarkPlotModel>,
): PlotView {
  const nextX = zoomedDomain(view.xDomain, factor, model.xDomain);
  const nextY = zoomedDomain(view.yDomain, factor, model.yDomain);
  return { xDomain: nextX, yDomain: nextY };
}

function pannedView(
  view: PlotView,
  direction: 'left' | 'right',
  model: ReturnType<typeof benchmarkPlotModel>,
): PlotView {
  return {
    xDomain: pannedDomain(view.xDomain, direction, model.xDomain),
    yDomain: view.yDomain,
  };
}

function zoomedDomain(
  [min, max]: [number, number],
  factor: number,
  bounds: [number, number],
): [number, number] {
  const midpoint = (min + max) / 2;
  const span = Math.min(bounds[1] - bounds[0], (max - min) * factor);
  const rawMin = midpoint - span / 2;
  const rawMax = midpoint + span / 2;
  if (rawMin < bounds[0]) {
    return [bounds[0], bounds[0] + span];
  }
  if (rawMax > bounds[1]) {
    return [bounds[1] - span, bounds[1]];
  }
  return [rawMin, rawMax];
}

function pannedDomain(
  [min, max]: [number, number],
  direction: 'left' | 'right',
  bounds: [number, number],
): [number, number] {
  const span = max - min;
  const offset = span * plotPanFraction * (direction === 'left' ? -1 : 1);
  const rawMin = min + offset;
  const rawMax = max + offset;
  if (rawMin < bounds[0]) {
    return [bounds[0], bounds[0] + span];
  }
  if (rawMax > bounds[1]) {
    return [bounds[1] - span, bounds[1]];
  }
  return [rawMin, rawMax];
}
