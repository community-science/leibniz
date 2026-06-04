import {
  ArrowLeft,
  ArrowRight,
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  RotateCcw,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { useEffect, useState } from 'react';

import {
  benchmarkPlotModel,
  benchmarkCostAxisGroups,
  benchmarkCostAxes,
  benchmarkCostAxis,
  benchmarkScoreAxes,
  benchmarkScoreAxis,
  costValue,
  formatCost,
  nextModelResultSort,
  scoreLabel,
  scoreTickLabel,
  scoreValue,
  selectionForId,
  shortDigest,
  sortedModelResults,
  type ModelResultSort,
  type ModelResultSortKey,
} from './benchmarkDashboardModel.ts';
import type {
  BenchmarkResultRecord,
  CostAxisRecord,
  ModelResultRecord,
  ScoreAxisRecord,
} from './resultViews.ts';
import { usePersistentState } from './persistentState.ts';

type PlotView = {
  xDomain: [number, number];
  yDomain: [number, number];
};

type PlotAxisSelectorAxis = {
  key: string;
  label: string;
};

type PlotAxisSelectorGroup = {
  axes: PlotAxisSelectorAxis[];
  key: string;
};

const plotWidth = 960;
const plotHeight = 440;
const plotMargin = {
  bottom: 78,
  left: 92,
  right: 26,
  top: 36,
};
const plotBodyWidth = plotWidth - plotMargin.left - plotMargin.right;
const plotBodyHeight = plotHeight - plotMargin.top - plotMargin.bottom;
const plotZoomInFactor = 0.72;
const plotZoomOutFactor = 1.28;
const plotPanFraction = 0.18;
const plotTickOffset = 18;
const plotYTickLabelOffset = 10;
const plotTickLabelBaselineOffset = 4;
const plotAxisSelectorTopOffset = 20;
const plotAxisSelectorHeight = 28;
const plotAxisSelectorMinWidth = 720;
const plotAxisSelectorButtonWidth = 118;
const plotScoreSelectorWidth = 300;
const defaultLeaderboardSort: ModelResultSort = {
  key: 'score',
  direction: 'descending',
};

export function BenchmarkResultDashboard({
  onModelSelect,
  result,
  selectedModelKey,
}: {
  onModelSelect: (modelKey: string) => void;
  result: BenchmarkResultRecord;
  selectedModelKey: string;
}) {
  const costAxes = benchmarkCostAxes(result);
  const scoreAxes = benchmarkScoreAxes(result);
  const stateKeyPrefix = `leibniz.console.benchmarks.${result.benchmark_id}.performance`;
  const [selectedCostAxis, setSelectedCostAxis] = usePersistentState(
    `${stateKeyPrefix}.costAxis`,
    costAxes[0]?.key ?? 'storage_bytes',
  );
  const costAxis = benchmarkCostAxis(selectedCostAxis, costAxes);
  const costAxisLabel = costAxes.find((axis) => axis.key === costAxis)?.label ?? 'Cost';
  const [selectedScoreAxis, setSelectedScoreAxis] = usePersistentState(
    `${stateKeyPrefix}.scoreAxis`,
    scoreAxes[0]?.key ?? 'absolute',
  );
  const scoreAxis = benchmarkScoreAxis(selectedScoreAxis, scoreAxes);
  const scoreAxisLabel = scoreAxes.find((axis) => axis.key === scoreAxis)?.label ?? 'Score';
  const [plotView, setPlotView] = usePersistentState<PlotView | null>(
    `${stateKeyPrefix}.plotView`,
    null,
  );
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = usePersistentState<string | null>(
    `${stateKeyPrefix}.selectedId`,
    null,
  );
  const [leaderboardSort, setLeaderboardSort] = usePersistentState<ModelResultSort>(
    `${stateKeyPrefix}.leaderboardSort`,
    defaultLeaderboardSort,
  );
  const plot = benchmarkPlotModel(result, costAxis, scoreAxis);
  const selection = selectionForId(result, selectedId);
  const selectedSelectionModelKey =
    selection.selectedModel?.model_key ??
    selection.selectedRun?.model_key;
  const activeSelectedModelKey = selectedSelectionModelKey ?? selectedModelKey;
  const activeView = plotView ?? {
    xDomain: plot.xDomain,
    yDomain: plot.yDomain,
  };

  useEffect(() => {
    if (
      selectedSelectionModelKey !== undefined &&
      selectedSelectionModelKey !== selectedModelKey
    ) {
      onModelSelect(selectedSelectionModelKey);
    }
  }, [onModelSelect, selectedModelKey, selectedSelectionModelKey]);

  return (
    <section className="performance-section benchmark-result-dashboard">
      <BenchmarkFrontierPlot
        costAxes={costAxes}
        costAxis={costAxis}
        scoreAxes={scoreAxes}
        scoreAxis={scoreAxis}
        model={plot}
        onCostAxisChange={(axis) => {
          setSelectedCostAxis(axis);
          setPlotView(null);
          setHoveredId(null);
        }}
        onScoreAxisChange={(axis) => {
          setSelectedScoreAxis(axis);
          setPlotView(null);
          setHoveredId(null);
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
        costAxis={costAxis}
        costAxisLabel={costAxisLabel}
        models={result.leaderboard}
        onSelect={setSelectedId}
        onSort={(key) => setLeaderboardSort((current) => nextModelResultSort(current, key))}
        scoreAxis={scoreAxis}
        scoreAxisLabel={scoreAxisLabel}
        selectedModelKey={activeSelectedModelKey}
        sort={leaderboardSort}
        title="Leaderboard"
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
  onScoreAxisChange,
  onSelect,
  onZoom,
  scoreAxes,
  scoreAxis,
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
  onScoreAxisChange: (axis: string) => void;
  onSelect: (id: string | null) => void;
  onZoom: (factor: number) => void;
  scoreAxes: ScoreAxisRecord[];
  scoreAxis: string;
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
  const selectedPoint = model.points.find((point) => point.id === selectedId);
  const hoveredPoint = model.points.find((point) => point.id === hoveredId);
  const activePoint = hoveredPoint ?? selectedPoint;
  const costAxisGroups = benchmarkCostAxisGroups(costAxes);
  const axisButtonCount = costAxisGroups.reduce(
    (count, group) => count + group.axes.length,
    0,
  );
  const axisSelectorWidth = Math.min(
    plotBodyWidth,
    Math.max(plotAxisSelectorMinWidth, axisButtonCount * plotAxisSelectorButtonWidth),
  );
  const axisSelectorX = plotMargin.left + plotBodyWidth / 2 - axisSelectorWidth / 2;
  const axisSelectorY = plotMargin.top + plotBodyHeight + plotAxisSelectorTopOffset;
  const yAxisSelectorCenterY = plotMargin.top + plotBodyHeight / 2;
  const scoreAxisGroups: PlotAxisSelectorGroup[] = [
    {
      axes: scoreAxes,
      key: 'score',
    },
  ];

  return (
    <section className="benchmark-result-table-section">
      <div className="benchmark-plot-heading">
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
          <span><i className="tentative" />Tentative</span>
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
              const logTick = Math.log(tick) / Math.log(model.xLogBase);
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
              const logTick = Math.log(tick) / Math.log(model.xLogBase);
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
                    {model.xLogBase}<tspan dy="-5" fontSize="0.72em">{Math.round(logTick)}</tspan><tspan dy="5"> </tspan>
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
                    {scoreTickLabel(tick)}
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
            {visiblePoints.map((point) => (
              <circle
                className={[
                  'frontier-chart-point',
                  point.frontier ? 'frontier' : '',
                  point.resultStatus === 'tentative' ? 'tentative' : '',
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
                <PlotAxisSelector
                  activeAxis={costAxis}
                  groups={costAxisGroups}
                  onAxisChange={onCostAxisChange}
                />
              </div>
            </foreignObject>
            <foreignObject
              height={plotAxisSelectorHeight}
              width={plotScoreSelectorWidth}
              transform={
                `translate(18 ${yAxisSelectorCenterY}) rotate(-90) ` +
                `translate(${-plotScoreSelectorWidth / 2} ${-plotAxisSelectorHeight / 2})`
              }
              x={0}
              y={0}
            >
              <div className="frontier-chart-axis-selector">
                <PlotAxisSelector
                  activeAxis={scoreAxis}
                  groups={scoreAxisGroups}
                  onAxisChange={onScoreAxisChange}
                />
              </div>
            </foreignObject>
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
              {activePoint.resultStatus === 'tentative'
                ? 'Training estimate'
                : activePoint.frontier ? 'Frontier highlight' : 'Accepted result'}
            </span>
            <strong>{activePoint.label}</strong>
            <span>{formatCost(activePoint.cost)} cost</span>
            <span>{`score ${scoreLabel(activePoint.score)}`}</span>
          </div>
        )}
      </div>
    </section>
  );
}

function PlotAxisSelector({
  activeAxis,
  groups,
  onAxisChange,
}: {
  activeAxis: string;
  groups: PlotAxisSelectorGroup[];
  onAxisChange: (axis: string) => void;
}) {
  return (
    <>
      {groups.map((group) => (
        <div
          className="frontier-chart-axis-group"
          key={group.key}
          style={{ flexGrow: group.axes.length }}
        >
          <div>
            {group.axes.map((axis) => (
              <button
                aria-pressed={axis.key === activeAxis}
                className={axis.key === activeAxis ? 'active' : ''}
                key={axis.key}
                onClick={(event) => {
                  event.stopPropagation();
                  onAxisChange(axis.key);
                }}
                onPointerDown={(event) => event.stopPropagation()}
                type="button"
              >
                {axis.label}
              </button>
            ))}
          </div>
        </div>
      ))}
    </>
  );
}

function ModelResultTable({
  costAxis,
  costAxisLabel,
  models,
  onSelect,
  onSort,
  scoreAxis,
  scoreAxisLabel,
  selectedModelKey,
  sort,
  title,
}: {
  costAxis: string;
  costAxisLabel: string;
  models: ModelResultRecord[];
  onSelect: (id: string) => void;
  onSort: (key: ModelResultSortKey) => void;
  scoreAxis: string;
  scoreAxisLabel: string;
  selectedModelKey: string | undefined;
  sort: ModelResultSort;
  title: string;
}) {
  if (models.length === 0) {
    return null;
  }

  return (
    <section className="benchmark-result-table-section">
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
            label={scoreAxisLabel}
            onClick={() => onSort('score')}
          />
          <SortHeader
            active={sort.key === 'cost'}
            direction={sort.direction}
            label={costAxisLabel}
            onClick={() => onSort('cost')}
          />
        </div>
        {sortedModelResults(models, costAxis, scoreAxis, sort).map((model) => (
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
            <span role="cell">{scoreLabel(scoreValue(model, scoreAxis))}</span>
            <span role="cell">{formatCost(costValue(model.cost_summary, costAxis))}</span>
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
      <span className="benchmark-sort-header-icon" aria-hidden="true">
        {active ? (
          direction === 'ascending' ? (
            <ArrowUp size={13} />
          ) : (
            <ArrowDown size={13} />
          )
        ) : (
          <ArrowUpDown size={13} />
        )}
      </span>
    </button>
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
