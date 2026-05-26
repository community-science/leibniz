import { Activity, ChevronDown, Gauge } from 'lucide-react';
import type { ReactNode } from 'react';
import { useMemo, useState } from 'react';

import { BenchmarkResultDashboard } from './BenchmarkResultDashboard.tsx';
import {
  type BenchmarkResultEntry,
  benchmarkResultsForTask,
  costValue,
  formatCost,
  modelComparisonRows,
  scoreLabel,
  shortDigest,
} from './benchmarkDashboardModel.ts';
import type { ArtifactReferenceRecord } from './artifactIndex.ts';
import type {
  BenchmarkTaskRecord,
  GeneratedObservationBatchRecord,
  GeneratedObservationSampleRecord,
} from './benchmarkTasks.ts';
import type { ModelInspectionLayerRecord, ModelInspectionRecord } from './modelInspections.ts';
import type { BenchmarkResultRecord, ResultViewRecord } from './resultViews.ts';

type SampleCardDensity = 'standard' | 'compact';

export function BenchmarksPanel({
  modelInspections,
  resultViews,
  tasks,
}: {
  modelInspections: ModelInspectionRecord[];
  resultViews: ResultViewRecord[];
  tasks: BenchmarkTaskRecord[];
}) {
  const [selectedBenchmarkId, setSelectedBenchmarkId] = useState(tasks[0]?.benchmark_id ?? '');
  const selected = tasks.find((task) => task.benchmark_id === selectedBenchmarkId) ?? tasks[0];
  const benchmarkResults = useMemo(
    () =>
      selected === undefined
        ? []
        : benchmarkResultsForTask(resultViews, selected.benchmark_id),
    [resultViews, selected],
  );
  const selectedResult = benchmarkResults[0];

  if (selected === undefined) {
    return (
      <section className="benchmark-workbench">
        <p className="artifact-detail-note">No benchmark tasks are available.</p>
      </section>
    );
  }

  const result = selectedResult?.result;
  const sampleCount = selected.batches.reduce(
    (total, batch) => total + batch.samples.length,
    0,
  );

  return (
    <section className="benchmark-workbench" aria-label="Benchmarks">
      <div className="benchmark-workbench-content">
        <div className="benchmark-header">
          <h2>{selected.label}</h2>
          <p>{selected.source_path}</p>
        </div>

        <div className="benchmark-selector-row" aria-label="Benchmarks">
          <span>Benchmarks:</span>
          {tasks.map((task) => (
            <button
              className={task.benchmark_id === selected.benchmark_id ? 'active' : ''}
              key={task.benchmark_id}
              onClick={() => setSelectedBenchmarkId(task.benchmark_id)}
              type="button"
            >
              {task.label}
            </button>
          ))}
        </div>

        <BenchmarkStatusRow
          resultEntry={selectedResult}
          sampleCount={sampleCount}
          task={selected}
        />

        <div className="benchmark-section-stack">
          <CollapsibleBenchmarkSection
            label="Samples"
            summary={`${selected.batches.length} batches / ${sampleCount} samples`}
          >
            <BenchmarkTaskPane task={selected} />
          </CollapsibleBenchmarkSection>
          <CollapsibleBenchmarkSection
            actions={<BenchmarkPerformanceActions />}
            label="Performance"
            summary={`${result?.leaderboard.length ?? 0} models / ${result?.training_history.length ?? 0} runs`}
          >
            <BenchmarkPerformancePane benchmark={selected} resultEntry={selectedResult} />
          </CollapsibleBenchmarkSection>
          <CollapsibleBenchmarkSection
            label="Models"
            summary={`${modelComparisonRows(result, modelInspections).length} inspected candidates`}
          >
            <BenchmarkModelsPane
              inspections={modelInspections}
              result={result}
            />
          </CollapsibleBenchmarkSection>
        </div>
      </div>
    </section>
  );
}

function CollapsibleBenchmarkSection({
  actions,
  children,
  label,
  summary,
}: {
  actions?: ReactNode;
  children: ReactNode;
  label: string;
  summary?: string;
}) {
  const [expanded, setExpanded] = useState(true);
  return (
    <section className="benchmark-collapsible-section">
      <div className="benchmark-section-heading">
        <button
          aria-expanded={expanded}
          className="benchmark-section-toggle"
          onClick={() => setExpanded((value) => !value)}
          type="button"
        >
          <ChevronDown aria-hidden="true" className={expanded ? 'expanded' : ''} size={16} />
          <span>{label}</span>
        </button>
        {summary === undefined ? null : <span className="benchmark-section-summary">{summary}</span>}
        {actions === undefined ? null : <div className="benchmark-section-actions">{actions}</div>}
      </div>
      <div hidden={!expanded}>{children}</div>
    </section>
  );
}

function BenchmarkPerformanceActions() {
  return <span>Read-only frontier</span>;
}

function BenchmarkStatusRow({
  resultEntry,
  sampleCount,
  task,
}: {
  resultEntry: BenchmarkResultEntry | undefined;
  sampleCount: number;
  task: BenchmarkTaskRecord;
}) {
  const status = resultEntry === undefined ? 'o AWAITING RESULTS' : 'o RESULT VIEW LOADED';
  const source = resultEntry?.sourcePath ?? task.source_path;
  return (
    <div className="benchmark-status-row">
      <span>{status}</span>
      <span>/</span>
      <span>{sampleCount} generated samples</span>
      <span>/</span>
      <span>benchmark={task.benchmark_id}</span>
      <span>/</span>
      <span>{source}</span>
    </div>
  );
}

function BenchmarkPerformancePane({
  benchmark,
  resultEntry,
}: {
  benchmark: BenchmarkTaskRecord;
  resultEntry:
    | BenchmarkResultEntry
    | undefined;
}) {
  const result = resultEntry?.result ?? emptyBenchmarkResult(benchmark);
  const sourcePath = resultEntry?.sourcePath ?? 'No result view loaded';

  return (
    <div className="benchmark-task">
      <ResultSourceStatus result={result} resultEntry={resultEntry} />
      <BenchmarkResultDashboard
        result={result}
        sourcePath={sourcePath}
      />
    </div>
  );
}

function ResultSourceStatus({
  result,
  resultEntry,
}: {
  result: BenchmarkResultRecord;
  resultEntry: BenchmarkResultEntry | undefined;
}) {
  const status = resultEntry === undefined ? 'Awaiting result view' : 'Loaded result view';
  const updatedAt =
    resultEntry?.sourceMtimeMs === undefined
      ? 'Not reported'
      : new Intl.DateTimeFormat(undefined, {
          dateStyle: 'medium',
          timeStyle: 'short',
        }).format(new Date(resultEntry.sourceMtimeMs));
  const size =
    resultEntry?.sourceSizeBytes === undefined
      ? 'Not reported'
      : new Intl.NumberFormat(undefined, {
          maximumFractionDigits: 1,
          minimumFractionDigits: 0,
          style: 'unit',
          unit: 'byte',
          unitDisplay: 'narrow',
        }).format(resultEntry.sourceSizeBytes);

  return (
    <section className="benchmark-result-source" aria-label="Result source">
      <div>
        <span>{status}</span>
        <p>{resultEntry?.sourcePath ?? 'Materialize benchmark results into .runs/views to populate the frontier.'}</p>
      </div>
      <dl>
        <div>
          <dt>Updated</dt>
          <dd>{updatedAt}</dd>
        </div>
        <div>
          <dt>Size</dt>
          <dd>{size}</dd>
        </div>
        <div>
          <dt>Models</dt>
          <dd>{result.leaderboard.length}</dd>
        </div>
        <div>
          <dt>Runs</dt>
          <dd>{result.training_history.length}</dd>
        </div>
        <div>
          <dt>Proposals</dt>
          <dd>{result.proposals.length}</dd>
        </div>
      </dl>
    </section>
  );
}

function emptyBenchmarkResult(benchmark: BenchmarkTaskRecord): BenchmarkResultRecord {
  return {
    benchmark_id: benchmark.benchmark_id,
    complexity_axis: benchmark.complexity_axis,
    cost_axes: [{ key: 'parameter_count', label: 'Parameters' }],
    frontiers: {},
    leaderboard: [],
    proposals: [],
    scale_axis: benchmark.scale_axis,
    training_history: [],
  };
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
  const [selectedModelKey, setSelectedModelKey] = useState(rows[0]?.model.model_key ?? '');
  const selectedRow =
    rows.find(({ model }) => model.model_key === selectedModelKey) ?? rows[0];

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
        <div className="benchmark-model-inspector-layout">
          <div className="benchmark-model-grid">
          {rows.map(({ inspection, model }) => (
            <button
              className={`benchmark-model-card ${model.model_key === selectedRow?.model.model_key ? 'selected' : ''}`}
              key={model.model_key}
              onClick={() => setSelectedModelKey(model.model_key)}
              type="button"
            >
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
            </button>
          ))}
          </div>
          {selectedRow === undefined ? null : (
            <BenchmarkModelInspector
              costAxis={costAxis}
              inspection={selectedRow.inspection}
              model={selectedRow.model}
            />
          )}
        </div>
      </section>
    </div>
  );
}

function BenchmarkModelInspector({
  costAxis,
  inspection,
  model,
}: {
  costAxis: string;
  inspection: ModelInspectionRecord | undefined;
  model: BenchmarkResultRecord['leaderboard'][number];
}) {
  return (
    <article className="benchmark-model-detail">
      <header className="benchmark-model-detail-header">
        <div>
          <h3>{shortDigest(model.architecture_digest)}</h3>
          <p>{model.model_key}</p>
        </div>
        <dl className="benchmark-model-detail-metrics">
          <div>
            <dt>Score</dt>
            <dd>{model.score.toFixed(4)}</dd>
          </div>
          <div>
            <dt>Cost</dt>
            <dd>{formatCost(costValue(model.cost_summary, costAxis))}</dd>
          </div>
          <div>
            <dt>Measurements</dt>
            <dd>{model.measurement_count}</dd>
          </div>
          <div>
            <dt>Runs</dt>
            <dd>{model.run_ids.length}</dd>
          </div>
        </dl>
      </header>
      <section className="benchmark-model-detail-section">
        <h4>Architecture</h4>
        <dl className="benchmark-model-detail-grid">
          <dt>Digest</dt>
          <dd>{model.architecture_digest}</dd>
          <dt>Input</dt>
          <dd>{inspection === undefined ? 'unknown' : shapeLabel(inspection.input_shape)}</dd>
          <dt>Output</dt>
          <dd>{inspection === undefined ? 'unknown' : shapeLabel(inspection.output_shape)}</dd>
          <dt>Observed C</dt>
          <dd>{model.observed_complexities.join(', ') || 'none'}</dd>
          <dt>Sources</dt>
          <dd>{model.source_kinds.join(', ') || 'unknown'}</dd>
        </dl>
      </section>
      <ModelCostDetail inspection={inspection} model={model} />
      <ModelLayerTrace inspection={inspection} />
      <ModelProvenanceDetail inspection={inspection} />
    </article>
  );
}

function ModelCostDetail({
  inspection,
  model,
}: {
  inspection: ModelInspectionRecord | undefined;
  model: BenchmarkResultRecord['leaderboard'][number];
}) {
  const summary = inspection?.cost_summary ?? model.cost_summary;
  return (
    <section className="benchmark-model-detail-section">
      <h4>Cost Summary</h4>
      <dl className="benchmark-model-detail-grid">
        <dt>Layers</dt>
        <dd>{summary.layer_count}</dd>
        <dt>Parameters</dt>
        <dd>{optionalNumberLabel(summary.parameter_count)}</dd>
        <dt>Bytes</dt>
        <dd>{optionalNumberLabel(summary.parameter_bytes)}</dd>
        <dt>FLOPs</dt>
        <dd>{optionalNumberLabel(summary.inference_flops)}</dd>
        <dt>Unknown Parameters</dt>
        <dd>{unknownLayerLabel(inspection?.cost_summary.unknown_parameter_layers)}</dd>
        <dt>Unknown FLOPs</dt>
        <dd>{unknownLayerLabel(inspection?.cost_summary.unknown_flop_layers)}</dd>
      </dl>
    </section>
  );
}

function ModelLayerTrace({ inspection }: { inspection: ModelInspectionRecord | undefined }) {
  if (inspection === undefined) {
    return (
      <section className="benchmark-model-detail-section">
        <h4>Layer Trace</h4>
        <p className="artifact-detail-note">No model inspection record matches this model.</p>
      </section>
    );
  }
  return (
    <section className="benchmark-model-detail-section">
      <h4>Layer Trace</h4>
      <div className="benchmark-model-layer-list">
        {inspection.layers.map((layer) => (
          <article className="benchmark-model-layer" key={layer.index}>
            <div className="benchmark-model-layer-heading">
              <span>{layer.index}</span>
              <strong>{layer.kind}</strong>
            </div>
            <dl className="benchmark-model-detail-grid">
              <dt>Input</dt>
              <dd>{optionalShapeLabel(layer.input_shape)}</dd>
              <dt>Output</dt>
              <dd>{optionalShapeLabel(layer.output_shape)}</dd>
              <dt>Parameters</dt>
              <dd>{optionalNumberLabel(layer.parameter_count)}</dd>
              <dt>Bytes</dt>
              <dd>{optionalNumberLabel(layer.parameter_bytes)}</dd>
              <dt>FLOPs</dt>
              <dd>{optionalNumberLabel(layer.inference_flops)}</dd>
              <dt>Operator</dt>
              <dd>{operatorSummary(layer)}</dd>
              <dt>Config</dt>
              <dd>{recordLabel(layer.parameters)}</dd>
            </dl>
            {operatorEntries(layer).length === 0 ? null : (
              <dl className="benchmark-model-operator-grid">
                {operatorEntries(layer).map(([key, value]) => (
                  <div key={key}>
                    <dt>{key}</dt>
                    <dd>{value}</dd>
                  </div>
                ))}
              </dl>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}

function ModelProvenanceDetail({
  inspection,
}: {
  inspection: ModelInspectionRecord | undefined;
}) {
  if (inspection === undefined) {
    return null;
  }
  const references = [
    referenceEntry('Architecture', inspection.architecture),
    referenceEntry('Model manifest', inspection.model_manifest),
    referenceEntry('Submission package', inspection.submission_package),
    referenceEntry('Benchmark', inspection.benchmark_manifest),
    referenceEntry('Measurements', inspection.measurement_dataset),
    ...inspection.model_artifacts.map((reference, index) =>
      referenceEntry(`Model artifact ${index + 1}`, reference),
    ),
    ...inspection.training_provenance.map((reference, index) =>
      referenceEntry(`Training provenance ${index + 1}`, reference),
    ),
  ].filter((entry): entry is { label: string; reference: ArtifactReferenceRecord } => entry !== null);
  if (references.length === 0) {
    return null;
  }
  return (
    <section className="benchmark-model-detail-section">
      <h4>Provenance</h4>
      <div className="benchmark-model-source-list">
        {references.map(({ label, reference }) => (
          <dl key={`${label}:${referenceLabel(reference)}`}>
            <dt>{label}</dt>
            <dd>{referenceLabel(reference)}</dd>
          </dl>
        ))}
      </div>
    </section>
  );
}

function shapeLabel(shape: number[]): string {
  return shape.join(' x ');
}

function optionalShapeLabel(shape: number[] | undefined): string {
  return shape === undefined ? 'unknown' : shapeLabel(shape);
}

function optionalNumberLabel(value: number | undefined): string {
  return value === undefined ? 'unknown' : value.toLocaleString();
}

function unknownLayerLabel(layers: number[] | undefined): string {
  return layers === undefined || layers.length === 0 ? 'none' : layers.join(', ');
}

function operatorSummary(layer: ModelInspectionLayerRecord): string {
  const operator = layer.operator;
  if (operator === undefined) {
    return 'unknown';
  }
  return typeof operator.kind === 'string' ? operator.kind : 'unknown';
}

function operatorEntries(layer: ModelInspectionLayerRecord): [string, string][] {
  const operator = layer.operator;
  if (operator === undefined) {
    return [];
  }
  return Object.entries(operator)
    .filter(([key]) => key !== 'kind')
    .map(([key, value]) => [key, parameterValueLabel(value)]);
}

function recordLabel(record: Record<string, unknown>): string {
  const entries = Object.entries(record);
  if (entries.length === 0) {
    return 'none';
  }
  return entries.map(([key, value]) => `${key}: ${parameterValueLabel(value)}`).join(', ');
}

function parameterValueLabel(value: unknown): string {
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
    return String(value);
  }
  return JSON.stringify(value);
}

function referenceEntry(label: string, reference: ArtifactReferenceRecord | undefined) {
  if (reference === undefined) {
    return null;
  }
  return { label, reference };
}

function referenceLabel(reference: ArtifactReferenceRecord): string {
  return (
    reference.protocol_id ??
    reference.record_digest ??
    reference.content_digest ??
    reference.external_uri ??
    reference.kind
  );
}

function BenchmarkTaskPane({ task }: { task: BenchmarkTaskRecord }) {
  const modes = useMemo(() => unique(task.batches.map((batch) => batch.mode)), [task.batches]);
  const [mode, setMode] = useState(modes[0] ?? '');
  const [selectedSampleKey, setSelectedSampleKey] = useState<string | null>(null);
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
  const selectedSample =
    visibleSamples.find(({ batch, sample }) => sampleKey(batch, sample) === selectedSampleKey) ??
    visibleSamples[0];
  const selectedKey =
    selectedSample === undefined
      ? null
      : sampleKey(selectedSample.batch, selectedSample.sample);

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
      {selectedSample === undefined ? null : (
        <BenchmarkSampleCoordinateInspector
          batch={selectedSample.batch}
          sample={selectedSample.sample}
          task={task}
        />
      )}
      <section
        className={`benchmark-sample-grid ${selected.presentation.sample_card_density}`}
        aria-label="Generated benchmark samples"
      >
        {visibleSamples.map(({ batch, sample }) => (
          <BenchmarkSampleCard
            density={batch.presentation.sample_card_density}
            key={`${batch.mode}-${batch.scale}-${sample.index}-${sample.outcome_id}`}
            onSelect={() => setSelectedSampleKey(sampleKey(batch, sample))}
            sample={sample}
            selected={sampleKey(batch, sample) === selectedKey}
          />
        ))}
      </section>
      {selectedSample === undefined ? null : (
        <BenchmarkSampleDetail
          sample={selectedSample.sample}
        />
      )}
    </div>
  );
}

function BenchmarkSampleCard({
  density,
  onSelect,
  sample,
  selected,
}: {
  density: SampleCardDensity;
  onSelect: () => void;
  sample: GeneratedObservationSampleRecord;
  selected: boolean;
}) {
  return (
    <button
      className={`benchmark-sample-card ${density} ${selected ? 'selected' : ''}`}
      onClick={onSelect}
      type="button"
    >
      <div className="benchmark-image-shell">
        <img alt={sample.outcome_id} src={sample.image_data_url} />
      </div>
      <div className="benchmark-sample-caption">
        <span>{sample.component_sequence.join('')}</span>
        <span>{sample.outcome_id}</span>
      </div>
    </button>
  );
}

function BenchmarkSampleCoordinateInspector({
  batch,
  sample,
  task,
}: {
  batch: GeneratedObservationBatchRecord;
  sample: GeneratedObservationSampleRecord;
  task: BenchmarkTaskRecord;
}) {
  const content = sample.latent_coordinates.find((coordinate) => coordinate.role === 'content');
  const nuisance = sample.latent_coordinates.find((coordinate) => coordinate.role === 'nuisance');
  const entries: [string, string][] = [
    ['Outcome', sample.outcome_id],
    ['Components', sample.component_sequence.join('')],
    [task.scale_axis, String(batch.scale)],
    [task.complexity_axis, String(sample.complexity)],
    ['Seed', String(batch.seed)],
    ['Sample', `${sample.index + 1} / ${batch.sample_count}`],
    ['Field', sample.field_shape.join(' x ')],
    ['Content DOF', String(content?.multiplicity ?? 'n/a')],
    ['Nuisance DOF', String(nuisance?.multiplicity ?? 'n/a')],
  ];
  return (
    <section
      className="benchmark-sample-coordinate-inspector"
      aria-label="Selected sample coordinates"
    >
      <div className="benchmark-sample-coordinate-title">Selected Coordinates</div>
      <dl>
        {entries.map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd title={value}>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function BenchmarkSampleDetail({
  sample,
}: {
  sample: GeneratedObservationSampleRecord;
}) {
  const latentRoles = unique(sample.latent_coordinates.map((coordinate) => coordinate.role));
  return (
    <section className="benchmark-sample-detail" aria-label="Selected sample detail">
      <section className="benchmark-sample-detail-section">
        <h4>Materialization</h4>
        <dl className="benchmark-sample-detail-grid">
          {materializationEntries(sample).map(([key, value]) => (
            <div key={key}>
              <dt>{key}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      </section>
      <section className="benchmark-sample-detail-section">
        <h4>Latent Coordinates</h4>
        <div className="benchmark-latent-role-list">
          {latentRoles.map((role) => (
            <article className="benchmark-latent-role" key={role}>
              <h5>{modeLabel(role)}</h5>
              {sample.latent_coordinates
                .filter((coordinate) => coordinate.role === role)
                .map((coordinate) => (
                  <dl className="benchmark-sample-detail-grid" key={coordinate.name}>
                    <dt>Name</dt>
                    <dd>{coordinate.name}</dd>
                    <dt>Role</dt>
                    <dd>{coordinate.role}</dd>
                    <dt>Multiplicity</dt>
                    <dd>{coordinate.multiplicity}</dd>
                    <dt>Measure</dt>
                    <dd>{recordLabel(coordinate.degree_measure)}</dd>
                    <dt>Values</dt>
                    <dd>{parameterValueLabel(coordinate.values)}</dd>
                  </dl>
                ))}
            </article>
          ))}
        </div>
      </section>
    </section>
  );
}

function sampleKey(
  batch: GeneratedObservationBatchRecord,
  sample: GeneratedObservationSampleRecord,
): string {
  return `${batch.mode}:${batch.scale}:${batch.seed}:${batch.sample_count}:${sample.index}:${sample.outcome_id}`;
}

function materializationEntries(sample: GeneratedObservationSampleRecord): [string, string][] {
  return [
    ['Plan', recordString(sample.materialization_plan, 'id')],
    ['Benchmark', recordString(sample.materialization_plan, 'benchmark_id')],
    ['Scale', assignmentLabel(sample.materialization_plan.scale_assignment)],
    ['Complexity', assignmentLabel(sample.materialization_plan.complexity_assignment)],
    ['Resolution', assignmentLabel(sample.materialization_plan.resolution_assignment)],
    ['Seed', recordString(sample.materialization_plan, 'seed')],
  ];
}

function assignmentLabel(value: unknown): string {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return 'unknown';
  }
  const values = (value as Record<string, unknown>).values;
  if (!Array.isArray(values)) {
    return 'unknown';
  }
  return values
    .map((entry) => {
      if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
        return null;
      }
      const record = entry as Record<string, unknown>;
      return `${String(record.axis)}=${String(record.value)}`;
    })
    .filter((entry): entry is string => entry !== null)
    .join(', ') || 'unknown';
}

function recordString(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  return typeof value === 'string' || typeof value === 'number' ? String(value) : 'unknown';
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
