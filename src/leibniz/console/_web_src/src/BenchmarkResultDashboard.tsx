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
  costValue,
  formatCost,
  nextModelResultSort,
  runDetails,
  runSelectionId,
  scoreLabel,
  selectionForId,
  shortDigest,
  sortedModelResults,
  type BenchmarkRunDetail,
  type ModelResultSort,
  type ModelResultSortKey,
} from './benchmarkDashboardModel.ts';
import type {
  BenchmarkResultRecord,
  CostAxisRecord,
  ModelResultRecord,
  RunDetailSectionRecord,
} from './resultViews.ts';
import { usePersistentState } from './persistentState.ts';

type PlotView = {
  xDomain: [number, number];
  yDomain: [number, number];
};

const plotWidth = 960;
const plotHeight = 440;
const plotMargin = {
  bottom: 78,
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
const plotAxisSelectorTopOffset = 20;
const plotAxisSelectorHeight = 28;
const plotAxisSelectorMinWidth = 720;
const plotAxisSelectorButtonWidth = 118;
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
  const stateKeyPrefix = `leibniz.console.benchmarks.${result.benchmark_id}.performance`;
  const [selectedCostAxis, setSelectedCostAxis] = usePersistentState(
    `${stateKeyPrefix}.costAxis`,
    costAxes[0]?.key ?? 'parameter_count',
  );
  const costAxis = benchmarkCostAxis(selectedCostAxis, costAxes);
  const costAxisLabel = costAxes.find((axis) => axis.key === costAxis)?.label ?? 'Cost';
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
  const plot = benchmarkPlotModel(result, costAxis);
  const selection = selectionForId(result, selectedId);
  const runRows = runDetails(result);
  const selectedRunDetail = runRows.find(
    ({ run }) => runSelectionId(run) === selectedId,
  );
  const selectedSelectionModelKey =
    selection.selectedModel?.model_key ??
    selectedRunDetail?.model?.model_key;
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
        model={plot}
        onCostAxisChange={(axis) => {
          setSelectedCostAxis(axis);
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
        selectedModelKey={activeSelectedModelKey}
        sort={leaderboardSort}
        title="Leaderboard"
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
                    {tick.toFixed(0)}
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
                {costAxisGroups.map((group) => (
                  <div
                    className="frontier-chart-axis-group"
                    key={group.key}
                    style={{ flexGrow: group.axes.length }}
                  >
                    <span>{group.label}</span>
                    <div>
                      {group.axes.map((axis) => (
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
                  </div>
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
              {activePoint.frontier ? 'Frontier highlight' : 'Measured model'}
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

function ModelResultTable({
  costAxis,
  costAxisLabel,
  models,
  onSelect,
  onSort,
  selectedModelKey,
  sort,
  title,
}: {
  costAxis: string;
  costAxisLabel: string;
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
            label={costAxisLabel}
            onClick={() => onSort('cost')}
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
              <span role="cell">{run.complexity ?? 'n/a'}</span>
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
