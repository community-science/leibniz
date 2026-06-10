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
import { useEffect, useId, useState } from 'react';

import {
  benchmarkPlotModel,
  benchmarkCostAxisLabel,
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
  type BenchmarkPlotModelPoint,
} from './benchmarkDashboardModel.ts';
import type {
  BenchmarkResultRecord,
  ModelResultRecord,
  StateSpaceIntegralTermRecord,
} from './resultViews.ts';
import { usePersistentState } from './persistentState.ts';
import type { AxisRegionRecord, StateSpaceRegionRecord } from './stateSpaceRecords.ts';

type PlotView = {
  xDomain: [number, number];
  yDomain: [number, number];
};

const plotWidth = 960;
const plotHeight = 440;
const plotMargin = {
  bottom: 58,
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
const plotXAxisLabelOffset = 44;
const plotYTickLabelOffset = 10;
const plotYAxisLabelOffset = 52;
const plotTickLabelBaselineOffset = 4;
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
  const stateKeyPrefix = `leibniz.console.benchmarks.${result.benchmark_id}.performance`;
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
  const plot = benchmarkPlotModel(result);
  const frontierModels = plot.frontierPoints
    .map((point) => point.model)
    .filter((model): model is ModelResultRecord => model !== undefined);
  const selection = selectionForId(result, selectedId);
  const selectedSelectionModelKey =
    selection.selectedModel?.model_key ??
    selection.selectedRun?.model_key;
  const activeSelectedModelKey = selectedSelectionModelKey ?? selectedModelKey;
  const activeModel = result.model_candidates.find(
    (model) => model.model_key === activeSelectedModelKey,
  );
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
        model={plot}
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
        costAxisLabel={benchmarkCostAxisLabel}
        models={frontierModels}
        onSelect={setSelectedId}
        onSort={(key) => setLeaderboardSort((current) => nextModelResultSort(current, key))}
        scoreAxisLabel="Score"
        selectedModelKey={activeSelectedModelKey}
        sort={leaderboardSort}
        title="Leaderboard"
      />
      <ModelIntegralDetails model={activeModel} />
    </section>
  );
}

function BenchmarkFrontierPlot({
  hoveredId,
  model,
  onHover,
  onPan,
  onReset,
  onSelect,
  onZoom,
  selectedId,
  view,
}: {
  hoveredId: string | null;
  model: ReturnType<typeof benchmarkPlotModel>;
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
  const renderedPoints = [...visiblePoints].sort(comparePlotPointRenderOrder);
  const selectedPoint = model.points.find((point) => point.id === selectedId);
  const hoveredPoint = model.points.find((point) => point.id === hoveredId);
  const activePoint = hoveredPoint ?? selectedPoint;
  const plotClipId = `frontier-plot-clip-${useId().replaceAll(':', '')}`;

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
          <span><i className="provisional" />Provisional</span>
          {model.referenceCurves.length > 0 ? (
            <span><i className="reference" />Oracle Reference</span>
          ) : null}
        </div>
        <svg
          aria-label={`Measurements by ${benchmarkCostAxisLabel}`}
          className="frontier-chart-svg"
          role="img"
          viewBox={`0 0 ${plotWidth} ${plotHeight}`}
          onClick={() => onSelect(null)}
        >
          <defs>
            <clipPath id={plotClipId}>
              <rect
                height={plotBodyHeight}
                width={plotBodyWidth}
                x={plotMargin.left}
                y={plotMargin.top}
              />
            </clipPath>
          </defs>
          <rect
            className="frontier-chart-frame"
            height={plotBodyHeight}
            width={plotBodyWidth}
            x={plotMargin.left}
            y={plotMargin.top}
          />
          <text
            className="frontier-chart-axis-label"
            textAnchor="middle"
            transform={`translate(${plotMargin.left - plotYAxisLabelOffset} ${plotMargin.top + plotBodyHeight / 2}) rotate(-90)`}
          >
            Score
          </text>
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
          <text
            className="frontier-chart-axis-label"
            textAnchor="middle"
            x={plotMargin.left + plotBodyWidth / 2}
            y={plotMargin.top + plotBodyHeight + plotXAxisLabelOffset}
          >
            {benchmarkCostAxisLabel}
          </text>
          <g clipPath={`url(#${plotClipId})`}>
            {model.staircase.length > 0 ? (
              <polyline
                className="frontier-chart-staircase"
                fill="none"
                points={model.staircase.map(([logCost, score]) => `${x(logCost)},${y(score)}`).join(' ')}
              />
            ) : null}
            {model.referenceCurves.map((curve) => (
              <polyline
                className="frontier-chart-reference-curve"
                fill="none"
                key={curve.key}
                points={curve.points.map((point) => `${x(point.logCost)},${y(point.score)}`).join(' ')}
              />
            ))}
            {renderedPoints.map((point) => (
              <circle
                className={[
                  'frontier-chart-point',
                  point.frontier ? 'frontier' : '',
                  point.resultStatus === 'provisional' ? 'provisional' : '',
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
            </g>
        </svg>
        {activePoint === undefined ? null : (
          <div className="frontier-chart-tooltip">
            <span className="frontier-chart-tooltip-kicker">
              {activePoint.resultStatus === 'provisional'
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

function comparePlotPointRenderOrder(
  left: BenchmarkPlotModelPoint,
  right: BenchmarkPlotModelPoint,
): number {
  if (left.frontier !== right.frontier) {
    return left.frontier ? 1 : -1;
  }
  return left.cost - right.cost || left.score - right.score;
}

function ModelResultTable({
  costAxisLabel,
  models,
  onSelect,
  onSort,
  scoreAxisLabel,
  selectedModelKey,
  sort,
  title,
}: {
  costAxisLabel: string;
  models: ModelResultRecord[];
  onSelect: (id: string) => void;
  onSort: (key: ModelResultSortKey) => void;
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
        {sortedModelResults(models, sort).map((model) => (
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
            <span role="cell">{scoreLabel(scoreValue(model))}</span>
            <span role="cell">{formatCost(costValue(model.cost_summary))}</span>
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

function ModelIntegralDetails({ model }: { model: ModelResultRecord | undefined }) {
  if (model === undefined) {
    return null;
  }
  return (
    <section className="benchmark-result-table-section benchmark-integral-details">
      <div className="benchmark-integral-header">
        <h3>{shortDigest(model.architecture_digest)}</h3>
        <span>{scoreLabel(model.score)}</span>
      </div>
      <IntegralTermTable title="Score Integral" terms={model.score_integral.terms} />
      {model.cost_integral === undefined ? null : (
        <IntegralTermTable title="Cost Integral" terms={model.cost_integral.terms} />
      )}
    </section>
  );
}

function IntegralTermTable({
  terms,
  title,
}: {
  terms: StateSpaceIntegralTermRecord[];
  title: string;
}) {
  if (terms.length === 0) {
    return null;
  }
  return (
    <section className="benchmark-integral-term-section" aria-label={title}>
      <h4>{title}</h4>
      <div className="benchmark-integral-term-grid" role="table">
        <div className="benchmark-integral-term-row header" role="row">
          <span role="columnheader">Region</span>
          <span role="columnheader">Width</span>
          <span role="columnheader">Density</span>
          <span role="columnheader">Contribution</span>
          <span role="columnheader">Samples</span>
        </div>
        {terms.map((term, index) => (
          <div
            className="benchmark-integral-term-row"
            key={`${term.kind}:${term.log2_volume_minimum}:${term.log2_volume_maximum}:${index}`}
            role="row"
          >
            <span role="cell">{integralTermRegionLabel(term)}</span>
            <span role="cell">{formatBits(term.width_in_bits)}</span>
            <span role="cell">{formatRatio(term.competence_density)}</span>
            <span role="cell">{formatBits(term.contribution)}</span>
            <span role="cell">{term.sample_count?.toLocaleString() ?? 'unknown'}</span>
          </div>
        ))}
      </div>
      {terms
        .filter((term) => term.region !== undefined)
        .map((term, index) => (
          <StateSpaceRegionSummary
            key={`${term.kind}:region:${index}`}
            region={term.region}
            sampleCount={term.sample_count}
          />
        ))}
    </section>
  );
}

function StateSpaceRegionSummary({
  region,
  sampleCount,
}: {
  region: StateSpaceRegionRecord | undefined;
  sampleCount?: number;
}) {
  if (region === undefined) {
    return null;
  }
  const componentPreview = region.components.slice(0, 4);
  return (
    <section className="state-space-region-summary" aria-label="State-space region">
      <dl>
        <div>
          <dt>Ambient</dt>
          <dd>{regionAmbientLabel(region)}</dd>
        </div>
        <div>
          <dt>Volume</dt>
          <dd>{`${region.volume.toLocaleString()} states / ${formatBits(region.log2_volume)}`}</dd>
        </div>
        <div>
          <dt>Components</dt>
          <dd>{region.components.length.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Samples</dt>
          <dd>{sampleCount?.toLocaleString() ?? 'unknown'}</dd>
        </div>
      </dl>
      <div className="state-space-component-list">
        {componentPreview.map((component, index) => (
          <div className="state-space-component" key={`${component.stratum_id ?? 'component'}:${index}`}>
            <div className="state-space-component-heading">
              <span>{component.stratum_id ?? `component ${index + 1}`}</span>
              <span>{`${component.volume.toLocaleString()} / ${formatBits(component.log2_volume)}`}</span>
            </div>
            <div className="state-space-component-meta">
              <span>{component.measure_rule}</span>
              <span>{component.axis_regions.length.toLocaleString()} axes</span>
            </div>
            <div className="state-space-axis-list">
              {component.axis_regions.slice(0, 4).map((axisRegion) => (
                <span key={axisRegion.axis.id} title={axisRegionLabel(axisRegion)}>
                  {axisRegion.axis.domain.kind}: {axisRegion.count.toLocaleString()}
                </span>
              ))}
            </div>
          </div>
        ))}
        {region.components.length > componentPreview.length ? (
          <p>{`${region.components.length - componentPreview.length} more components`}</p>
        ) : null}
      </div>
    </section>
  );
}

function integralTermRegionLabel(term: StateSpaceIntegralTermRecord): string {
  return `${formatBits(term.log2_volume_minimum)} to ${formatBits(term.log2_volume_maximum)}`;
}

function regionAmbientLabel(region: StateSpaceRegionRecord): string {
  const domain = Object.entries(region.ambient.field_domain)
    .map(([key, value]) => `${key}=${value}`)
    .join(', ');
  const distinguishability = region.ambient.distinguishability.kind;
  return `${region.ambient.field_domain_kind} (${domain}) -> ${region.ambient.field_codomain_id}, ${distinguishability}`;
}

function formatBits(value: number): string {
  return `${formatCompactNumber(value)} bits`;
}

function formatRatio(value: number): string {
  return formatCompactNumber(value);
}

function axisRegionLabel(axisRegion: AxisRegionRecord): string {
  const coordinates = axisRegion.coordinate_region.length === 0
    ? 'singleton'
    : axisRegion.coordinate_region.join(', ');
  return `${axisRegion.axis.id}: ${axisRegion.axis.domain.kind}, ${coordinates}`;
}

function formatCompactNumber(value: number): string {
  return Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, {
    maximumFractionDigits: 4,
  });
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
