import { Activity, BarChart3, Boxes, Gauge, Images } from 'lucide-react';
import type { ReactNode } from 'react';
import { useMemo, useState } from 'react';

import { BenchmarkPerformanceBundle } from './BenchmarkPerformanceBundle.tsx';
import { BenchmarkResultDashboard } from './BenchmarkResultDashboard.tsx';
import {
  benchmarkResultsForTask,
  costValue,
  formatCost,
  modelComparisonRows,
  performanceViewsForTask,
  scoreLabel,
  shortDigest,
} from './benchmarkDashboardModel.ts';
import type {
  BenchmarkTaskRecord,
  GeneratedObservationBatchRecord,
  GeneratedObservationSampleRecord,
} from './benchmarkTasks.ts';
import type { ModelInspectionRecord } from './modelInspections.ts';
import type { PerformanceViewRecord } from './performanceViews.ts';
import type { BenchmarkResultRecord, ResultViewRecord } from './resultViews.ts';

type SampleCardDensity = 'standard' | 'compact';
type BenchmarkPane = 'samples' | 'performance' | 'models';

const benchmarkPanes: { id: BenchmarkPane; label: string; icon: ReactNode }[] = [
  { id: 'samples', label: 'Samples', icon: <Images size={16} /> },
  { id: 'performance', label: 'Performance', icon: <BarChart3 size={16} /> },
  { id: 'models', label: 'Models', icon: <Boxes size={16} /> },
];

export function BenchmarksPanel({
  modelInspections,
  performanceViews,
  resultViews,
  tasks,
}: {
  modelInspections: ModelInspectionRecord[];
  performanceViews: PerformanceViewRecord[];
  resultViews: ResultViewRecord[];
  tasks: BenchmarkTaskRecord[];
}) {
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState(tasks[0]?.benchmark_id ?? '');
  const [currentPane, setCurrentPane] = useState<BenchmarkPane>('samples');
  const selected = tasks.find((task) => task.benchmark_id === selectedBenchmarkId) ?? tasks[0];
  const benchmarkResults = useMemo(
    () =>
      selected === undefined
        ? []
        : benchmarkResultsForTask(resultViews, selected.benchmark_id),
    [resultViews, selected],
  );
  const selectedResult = benchmarkResults[0];
  const benchmarkPerformanceViews = useMemo(
    () =>
      selected === undefined
        ? []
        : performanceViewsForTask(performanceViews, selected.benchmark_id),
    [performanceViews, selected],
  );

  if (selected === undefined) {
    return (
      <section className="benchmark-layout">
        <p className="artifact-detail-note">No benchmark tasks are available.</p>
      </section>
    );
  }

  return (
    <section className="benchmark-layout" aria-label="Benchmarks">
      <aside className="benchmark-sidebar" aria-label="Benchmarks">
        {tasks.map((task) => (
          <button
            className={`benchmark-list-item ${task.benchmark_id === selected.benchmark_id ? 'active' : ''}`}
            key={task.benchmark_id}
            onClick={() => setSelectedBenchmarkId(task.benchmark_id)}
            type="button"
          >
            <span>{task.label}</span>
            <small>{task.benchmark_id}</small>
          </button>
        ))}
      </aside>
      <div className="benchmark-main">
        <div className="benchmark-header">
          <div>
            <h2>{selected.label}</h2>
            <p>{selected.source_path}</p>
          </div>
          <nav className="benchmark-pane-tabs" aria-label="Benchmark panes">
            {benchmarkPanes.map((pane) => (
              <button
                className={`benchmark-pane-tab ${currentPane === pane.id ? 'active' : ''}`}
                key={pane.id}
                onClick={() => setCurrentPane(pane.id)}
                type="button"
              >
                {pane.icon}
                {pane.label}
              </button>
            ))}
          </nav>
        </div>
        {currentPane === 'samples' ? (
          <BenchmarkTaskPane task={selected} />
        ) : currentPane === 'performance' ? (
          <BenchmarkPerformancePane
            performanceViews={benchmarkPerformanceViews}
            resultEntry={selectedResult}
          />
        ) : (
          <BenchmarkModelsPane
            inspections={modelInspections}
            result={selectedResult?.result}
          />
        )}
      </div>
    </section>
  );
}

function BenchmarkPerformancePane({
  performanceViews,
  resultEntry,
}: {
  performanceViews: PerformanceViewRecord[];
  resultEntry:
    | {
        sourcePath: string;
        result: BenchmarkResultRecord;
      }
    | undefined;
}) {
  if (resultEntry === undefined && performanceViews.length === 0) {
    return (
      <div className="benchmark-task">
        <p className="artifact-detail-note">No benchmark performance records are available.</p>
      </div>
    );
  }

  return (
    <div className="benchmark-task">
      {resultEntry === undefined ? null : (
        <BenchmarkResultDashboard
          result={resultEntry.result}
          sourcePath={resultEntry.sourcePath}
        />
      )}
      {performanceViews.map((view) => (
        <BenchmarkPerformanceBundle key={view.id} view={view} />
      ))}
    </div>
  );
}

function BenchmarkModelsPane({
  inspections,
  result,
}: {
  inspections: ModelInspectionRecord[];
  result: BenchmarkResultRecord | undefined;
}) {
  const rows = modelComparisonRows(result, inspections);
  const costAxis = result?.cost_axes[0]?.key ?? 'parameter_count';

  if (rows.length === 0) {
    return (
      <div className="benchmark-task">
        <p className="artifact-detail-note">No benchmark model comparisons are available.</p>
      </div>
    );
  }

  return (
    <div className="benchmark-task">
      <section className="benchmark-result-table-section">
        <h3>Model Comparison</h3>
        <div className="benchmark-model-grid">
          {rows.map(({ inspection, model }) => (
            <article className="benchmark-model-card" key={model.model_key}>
              <div className="benchmark-model-heading">
                <strong>{shortDigest(model.architecture_digest)}</strong>
                <span>{scoreLabel(model.score)}</span>
              </div>
              <dl>
                <dt>Cost</dt>
                <dd>{formatCost(costValue(model.cost_summary, costAxis))}</dd>
                <dt>Complexity</dt>
                <dd>{model.observed_complexities.join(', ') || 'none'}</dd>
                <dt>Measurements</dt>
                <dd>{model.measurement_count}</dd>
                <dt>Runs</dt>
                <dd>{model.run_ids.length}</dd>
                <dt>Layers</dt>
                <dd>{inspection?.layers.length ?? model.cost_summary.layer_count}</dd>
                <dt>Source</dt>
                <dd>{model.source_kinds.join(', ') || 'unknown'}</dd>
              </dl>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function BenchmarkTaskPane({ task }: { task: BenchmarkTaskRecord }) {
  const modes = useMemo(() => unique(task.batches.map((batch) => batch.mode)), [task.batches]);
  const [mode, setMode] = useState(modes[0] ?? '');
  const matchingMode = matchingBatches(task.batches, { mode });
  const scales = unique(matchingMode.map((batch) => batch.scale));
  const [scale, setScale] = useState(scales[0] ?? task.batches[0]?.scale ?? 1);
  const matchingScale = matchingBatches(task.batches, { mode, scale });
  const seeds = unique(matchingScale.map((batch) => batch.seed));
  const [seed, setSeed] = useState(seeds[0] ?? task.batches[0]?.seed ?? 0);
  const matchingSeed = matchingBatches(task.batches, { mode, scale, seed });
  const sampleCounts = unique(matchingSeed.map((batch) => batch.sample_count));
  const [sampleCount, setSampleCount] = useState(
    sampleCounts[0] ?? task.batches[0]?.sample_count ?? 1,
  );
  const selected =
    task.batches.find(
      (batch) =>
        batch.mode === mode &&
        batch.scale === scale &&
        batch.seed === seed &&
        batch.sample_count === sampleCount,
    ) ??
    matchingSeed[0] ??
    matchingScale[0] ??
    matchingMode[0] ??
    task.batches[0];
  const aggregateMode = selected?.presentation.aggregate_mode === true;
  const visibleBatches = aggregateMode
    ? matchingBatches(task.batches, { mode, seed: selected?.seed }).sort(
        (left, right) => left.scale - right.scale,
      )
    : selected === undefined
      ? []
      : [selected];
  const visibleSamples = visibleBatches.flatMap((batch) =>
    batch.samples.map((sample) => ({ batch, sample })),
  );

  if (selected === undefined) {
    return <p className="artifact-detail-note">No generated samples are available.</p>;
  }

  return (
    <div className="benchmark-task">
      <section className="benchmark-task-controls" aria-label="Benchmark sample controls">
        <SelectControl
          label="Mode"
          value={mode}
          values={modes}
          labelFor={modeLabel}
          onChange={setMode}
        />
        {aggregateMode ? null : (
          <SelectControl
            label={task.scale_axis}
            value={String(selected.scale)}
            values={unique(
              matchingBatches(task.batches, { mode }).map((batch) => String(batch.scale)),
            )}
            onChange={(value) => setScale(Number(value))}
          />
        )}
        <SelectControl
          label="Seed"
          value={String(selected.seed)}
          values={unique(
            matchingBatches(
              task.batches,
              aggregateMode ? { mode } : { mode, scale: selected.scale },
            ).map((batch) => String(batch.seed)),
          )}
          onChange={(value) => setSeed(Number(value))}
        />
        {aggregateMode ? null : (
          <SelectControl
            label="Samples"
            value={String(selected.sample_count)}
            values={unique(
              matchingBatches(task.batches, {
                mode,
                scale: selected.scale,
                seed: selected.seed,
              }).map((batch) => String(batch.sample_count)),
            )}
            onChange={(value) => setSampleCount(Number(value))}
          />
        )}
      </section>
      <section className="benchmark-task-summary" aria-label="Benchmark batch summary">
        <Metric
          icon={<Gauge size={16} />}
          label={task.complexity_axis}
          value={
            aggregateMode
              ? complexityRange(visibleBatches)
              : String(selected.samples[0]?.complexity ?? selected.scale)
          }
        />
        <Metric icon={<Activity size={16} />} label="Mode" value={modeLabel(selected.mode)} />
        <Metric label="Outcomes" value={`${task.outcome_atom_count} ${task.outcome_atom_name}s`} />
        <Metric
          label="Batch"
          value={aggregateMode ? `${visibleSamples.length} samples` : selected.label}
        />
      </section>
      <section
        className={`benchmark-sample-grid ${selected.presentation.sample_card_density}`}
        aria-label="Generated benchmark samples"
      >
        {visibleSamples.map(({ batch, sample }) => (
          <BenchmarkSampleCard
            density={batch.presentation.sample_card_density}
            key={`${batch.mode}-${batch.scale}-${sample.index}-${sample.outcome_id}`}
            sample={sample}
          />
        ))}
      </section>
    </div>
  );
}

function BenchmarkSampleCard({
  density,
  sample,
}: {
  density: SampleCardDensity;
  sample: GeneratedObservationSampleRecord;
}) {
  const content = sample.latent_coordinates.find((coordinate) => coordinate.role === 'content');
  const nuisance = sample.latent_coordinates.find((coordinate) => coordinate.role === 'nuisance');
  return (
    <article className={`benchmark-sample-card ${density}`}>
      <div className="benchmark-image-shell">
        <img alt={sample.outcome_id} src={sample.image_data_url} />
      </div>
      {density === 'compact' ? (
        <div className="benchmark-compact-label">
          <strong>{sample.component_sequence.join('')}</strong>
          <span>{sample.outcome_id}</span>
        </div>
      ) : (
        <dl>
          <dt>Outcome</dt>
          <dd>{sample.outcome_id}</dd>
          <dt>Sequence</dt>
          <dd>{sample.component_sequence.join(', ')}</dd>
          <dt>Complexity</dt>
          <dd>{sample.complexity}</dd>
          <dt>Shape</dt>
          <dd>{sample.field_shape.join(' x ')}</dd>
          <dt>Content DOF</dt>
          <dd>{content?.multiplicity ?? 'n/a'}</dd>
          <dt>Nuisance DOF</dt>
          <dd>{nuisance?.multiplicity ?? 'n/a'}</dd>
        </dl>
      )}
    </article>
  );
}

function SelectControl({
  label,
  labelFor = (value: string) => value,
  onChange,
  value,
  values,
}: {
  label: string;
  labelFor?: (value: string) => string;
  onChange: (value: string) => void;
  value: string;
  values: string[];
}) {
  return (
    <label className="benchmark-task-control">
      <span>{label}</span>
      <select onChange={(event) => onChange(event.target.value)} value={value}>
        {values.map((item) => (
          <option key={item} value={item}>
            {labelFor(item)}
          </option>
        ))}
      </select>
    </label>
  );
}

function Metric({
  icon,
  label,
  value,
}: {
  icon?: ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="benchmark-metric">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function matchingBatches(
  batches: GeneratedObservationBatchRecord[],
  criteria: { mode?: string; scale?: number; seed?: number },
): GeneratedObservationBatchRecord[] {
  return batches.filter(
    (batch) =>
      (criteria.mode === undefined || batch.mode === criteria.mode) &&
      (criteria.scale === undefined || batch.scale === criteria.scale) &&
      (criteria.seed === undefined || batch.seed === criteria.seed),
  );
}

function modeLabel(mode: string): string {
  return mode
    .split('-')
    .map((part) => `${part.slice(0, 1).toUpperCase()}${part.slice(1)}`)
    .join(' ');
}

function complexityRange(batches: GeneratedObservationBatchRecord[]): string {
  const complexities = batches
    .map((batch) => batch.samples[0]?.complexity)
    .filter((complexity): complexity is number => complexity !== undefined);
  if (complexities.length === 0) {
    return 'n/a';
  }
  return `${Math.min(...complexities)}-${Math.max(...complexities)}`;
}

function unique<T extends string | number>(values: T[]): T[] {
  return Array.from(new Set(values));
}
