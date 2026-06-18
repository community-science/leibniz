import {
  Boxes,
  ChevronDown,
  Fingerprint,
  GitBranch,
  PackageCheck,
  type LucideIcon,
} from 'lucide-react';
import type { ReactNode } from 'react';
import { useMemo } from 'react';

import { BenchmarkResultDashboard } from './BenchmarkResultDashboard.tsx';
import {
  type BenchmarkResultEntry,
  benchmarkResultsForTask,
  costValue,
  emptyFrontiersForCostAxis,
  formatCost,
  modelComparisonRows,
  shortDigest,
} from './benchmarkDashboardModel.ts';
import type { ArtifactReferenceRecord } from './artifactReferences.ts';
import type { BenchmarkTaskRecord } from './benchmarkTasks.ts';
import type {
  ModelInspectionRecord,
} from './modelInspections.ts';
import { usePersistentState } from './persistentState.ts';
import type {
  BenchmarkResultRecord,
  CapabilityMapNodeRecord,
  CapabilityMapRecord,
  ResultViewRecord,
  RunResultRecord,
  TrainingHistoryPointRecord,
  TrainingProtocolRecord,
} from './resultViews.ts';

type BenchmarkModelCandidate = BenchmarkResultRecord['model_candidates'][number];
type ModelArtifactView = 'model' | 'program' | 'measurements' | 'training' | 'provenance';
type ModelArtifactFlowItem = {
  icon: LucideIcon;
  label: string;
  value: string;
  view: ModelArtifactView;
};
type ModelLineageNode = {
  detail?: string;
  kind: string;
  role: 'input' | 'operation' | 'output';
  value: string;
};
type ValidationHistoryPoint = {
  stale_checks?: number;
  step: number;
  validation_loss: number;
  validation_loss_reference?: number;
};

const modelValidationPlotWidth = 560;
const modelValidationPlotHeight = 180;
const modelValidationPlotMargin = {
  bottom: 30,
  left: 46,
  right: 16,
  top: 16,
};
const modelValidationPlotBodyWidth =
  modelValidationPlotWidth - modelValidationPlotMargin.left - modelValidationPlotMargin.right;
const modelValidationPlotBodyHeight =
  modelValidationPlotHeight - modelValidationPlotMargin.top - modelValidationPlotMargin.bottom;

export function BenchmarksPanel({
  modelInspections,
  resultViews,
  tasks,
}: {
  modelInspections: ModelInspectionRecord[];
  resultViews: ResultViewRecord[];
  tasks: BenchmarkTaskRecord[];
}) {
  const [selectedBenchmarkId, setSelectedBenchmarkId] = usePersistentState(
    'leibniz.console.benchmarks.selectedBenchmark',
    tasks[0]?.benchmark_id ?? '',
  );
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
  const modelRows = modelComparisonRows(result, modelInspections);
  const [selectedModelKey, setSelectedModelKey] = usePersistentState(
    `leibniz.console.benchmarks.${selected.benchmark_id}.selectedModel`,
    modelRows[0]?.model.model_key ?? '',
  );
  const effectiveSelectedModelKey =
    modelRows.some(({ model }) => model.model_key === selectedModelKey)
      ? selectedModelKey
      : modelRows[0]?.model.model_key ?? '';

  return (
    <section className="benchmark-workbench" aria-label="Benchmarks">
      <div className="benchmark-workbench-content">
        <div className="benchmark-header">
          <div>
            <h2>
              <select
                aria-label="Benchmark"
                className="benchmark-title-select"
                onChange={(event) => setSelectedBenchmarkId(event.target.value)}
                value={selected.benchmark_id}
              >
                {tasks.map((task) => (
                  <option key={task.benchmark_id} value={task.benchmark_id}>
                    {task.label}
                  </option>
                ))}
              </select>
            </h2>
          </div>
        </div>

        <div className="benchmark-section-stack">
          <CollapsibleBenchmarkSection
            label="Performance"
            summary={`${result?.leaderboard.length ?? 0} models`}
            storageKey="leibniz.console.benchmarks.section.performance.expanded"
          >
            <BenchmarkPerformancePane
              benchmark={selected}
              onModelSelect={setSelectedModelKey}
              resultEntry={selectedResult}
              selectedModelKey={effectiveSelectedModelKey}
            />
          </CollapsibleBenchmarkSection>
          <CollapsibleBenchmarkSection
            label="Model Inspector"
            summary={`${modelRows.length} inspected candidates`}
            storageKey="leibniz.console.benchmarks.section.models.expanded"
          >
            <BenchmarkModelsPane
              rows={modelRows}
              result={result}
              selectedModelKey={effectiveSelectedModelKey}
            />
          </CollapsibleBenchmarkSection>
        </div>
      </div>
    </section>
  );
}

function CollapsibleBenchmarkSection({
  children,
  label,
  storageKey,
  summary,
}: {
  children: ReactNode;
  label: string;
  storageKey: string;
  summary?: string;
}) {
  const [expanded, setExpanded] = usePersistentState(storageKey, true);
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
      </div>
      <div hidden={!expanded}>{children}</div>
    </section>
  );
}

function BenchmarkPerformancePane({
  benchmark,
  onModelSelect,
  resultEntry,
  selectedModelKey,
}: {
  benchmark: BenchmarkTaskRecord;
  onModelSelect: (modelKey: string) => void;
  resultEntry:
    | BenchmarkResultEntry
    | undefined;
  selectedModelKey: string;
}) {
  const result = resultEntry?.result ?? emptyBenchmarkResult(benchmark);

  return (
    <div className="benchmark-task">
      <BenchmarkResultDashboard
        onModelSelect={onModelSelect}
        result={result}
        selectedModelKey={selectedModelKey}
      />
    </div>
  );
}

function emptyBenchmarkResult(benchmark: BenchmarkTaskRecord): BenchmarkResultRecord {
  return {
    benchmark_id: benchmark.benchmark_id,
    volume_axis: benchmark.volume_axis,
    frontiers: emptyFrontiersForCostAxis(),
    leaderboard: [],
    model_candidates: [],
    model_inspections: [],
    plot_runs: [],
    training_history: [],
  };
}

function BenchmarkModelsPane({
  rows,
  result,
  selectedModelKey,
}: {
  rows: ReturnType<typeof modelComparisonRows>;
  result: BenchmarkResultRecord | undefined;
  selectedModelKey: string;
}) {
  const selectedRow =
    rows.find(({ model }) => model.model_key === selectedModelKey) ?? rows[0];

  return (
    <div className="benchmark-task">
      <section className="benchmark-model-workbench">
        <div className="benchmark-model-inspector-layout single">
          {selectedRow === undefined ? (
            <p className="artifact-detail-note">No model runs are available.</p>
          ) : (
            <BenchmarkModelInspector
              volumeAxis={result?.volume_axis}
              inspection={selectedRow.inspection}
              model={selectedRow.model}
              runs={runsForModel(result, selectedRow.model)}
            />
          )}
        </div>
      </section>
    </div>
  );
}

function BenchmarkModelInspector({
  volumeAxis,
  inspection,
  model,
  runs,
}: {
  volumeAxis: string | undefined;
  inspection: ModelInspectionRecord | undefined;
  model: BenchmarkModelCandidate;
  runs: RunResultRecord[];
}) {
  const [artifactView, setArtifactView] = usePersistentState<ModelArtifactView>(
    `leibniz.console.benchmarks.${model.model_key}.artifactView`,
    'model',
  );
  return (
    <article className="benchmark-model-detail">
      <header className="benchmark-model-artifact-hero">
        <div className="benchmark-model-artifact-mark" aria-hidden="true">
          <PackageCheck size={20} />
        </div>
        <div>
          <span>Model artifact</span>
          <h3>{shortDigest(model.program_digest)}</h3>
          <code>{model.model_key}</code>
        </div>
      </header>
      <ModelArtifactFlow
        active={artifactView}
        inspection={inspection}
        model={model}
        onSelect={setArtifactView}
      />
      <ModelLineageGraph inspection={inspection} model={model} runs={runs} />
      <dl className="benchmark-model-detail-metrics">
        <div>
          <dt>Score</dt>
          <dd>{model.score.toFixed(4)}</dd>
        </div>
        <div>
          <dt>Cost</dt>
          <dd>{formatCost(costValue(model.cost_summary))}</dd>
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
      {artifactView === 'model' ? (
        <ModelManifestDetail
          volumeAxis={volumeAxis}
          inspection={inspection}
          model={model}
        />
      ) : null}
      {artifactView === 'measurements' ? (
        <ModelMeasurementDetail model={model} />
      ) : null}
      {artifactView === 'program' ? (
        <>
          <ModelProgramDetail
            volumeAxis={volumeAxis}
            inspection={inspection}
            model={model}
          />
          <ModelCostDetail inspection={inspection} model={model} />
          <ModelGraphOperations inspection={inspection} />
        </>
      ) : null}
      {artifactView === 'training' ? (
        <ModelTrainingDetail runs={runs} />
      ) : null}
      {artifactView === 'provenance' ? (
        <ModelProvenanceDetail inspection={inspection} />
      ) : null}
    </article>
  );
}

function ModelArtifactFlow({
  active,
  inspection,
  model,
  onSelect,
}: {
  active: ModelArtifactView;
  inspection: ModelInspectionRecord | undefined;
  model: BenchmarkModelCandidate;
  onSelect: (view: ModelArtifactView) => void;
}) {
  const items: ModelArtifactFlowItem[] = [
    {
      icon: PackageCheck,
      label: 'Program',
      value: inspection === undefined
        ? shortDigest(model.program_digest)
        : referenceLabel(inspection.program),
      view: 'program',
    },
    {
      icon: Boxes,
      label: 'Measurements',
      value: inspection?.measurement_dataset === undefined
        ? `${model.measurement_count} records`
        : referenceLabel(inspection.measurement_dataset),
      view: 'measurements',
    },
    {
      icon: GitBranch,
      label: 'Training',
      value: inspection?.training_provenance.length
        ? `${inspection.training_provenance.length} records`
        : `${model.run_ids.length} runs`,
      view: 'training',
    },
    {
      icon: Fingerprint,
      label: 'Provenance',
      value: inspection === undefined
        ? 'inspection pending'
        : `${modelProvenanceReferences(inspection).length} references`,
      view: 'provenance',
    },
  ];
  return (
    <section className="benchmark-model-artifact-flow" aria-label="Selected model artifact flow">
      {items.map((item) => {
        const Icon = item.icon;
        return (
          <button
            className={active === item.view ? 'active' : ''}
            key={item.label}
            onClick={() => onSelect(item.view)}
            type="button"
          >
            <Icon aria-hidden="true" size={15} />
            <span>{item.label}</span>
            <strong>{item.value}</strong>
          </button>
        );
      })}
    </section>
  );
}

function ModelLineageGraph({
  inspection,
  model,
  runs,
}: {
  inspection: ModelInspectionRecord | undefined;
  model: BenchmarkModelCandidate;
  runs: RunResultRecord[];
}) {
  const inputNodes = [
    lineageNode('input', 'program', model.program_digest),
    inspection?.measurement_dataset === undefined
      ? lineageNode('input', 'measurements', `${model.measurement_count} records`)
      : lineageNode('input', inspection.measurement_dataset.kind, referenceLabel(inspection.measurement_dataset)),
  ];
  const operationNode = lineageNode(
    'operation',
    runs.length === 0 ? 'training' : 'training run',
    runs[0]?.run_slug ?? `${model.run_ids.length} runs`,
    runs[0]?.source_kind,
  );
  const outputNode = lineageNode(
    'output',
    inspection?.model_manifest?.kind ?? 'model',
    model.model_key,
    inspection?.model_manifest === undefined ? undefined : referenceLabel(inspection.model_manifest),
  );

  return (
    <section className="benchmark-model-lineage-graph" aria-label="Model lineage">
      <div className="benchmark-model-lineage-column">
        {inputNodes.map((node) => (
          <ModelLineageCard key={`${node.kind}:${node.value}`} node={node} />
        ))}
      </div>
      <div className="benchmark-model-lineage-arrow" aria-hidden="true">-&gt;</div>
      <ModelLineageCard node={operationNode} />
      <div className="benchmark-model-lineage-arrow" aria-hidden="true">-&gt;</div>
      <ModelLineageCard node={outputNode} />
    </section>
  );
}

function ModelLineageCard({ node }: { node: ModelLineageNode }) {
  return (
    <div className={`benchmark-model-lineage-card ${node.role}`}>
      <span>{node.role}</span>
      <strong>{node.kind}</strong>
      <code>{node.value}</code>
      {node.detail === undefined ? null : <em>{node.detail}</em>}
    </div>
  );
}

function lineageNode(
  role: ModelLineageNode['role'],
  kind: string,
  value: string,
  detail?: string,
): ModelLineageNode {
  return { detail, kind, role, value };
}

function ModelManifestDetail({
  volumeAxis,
  inspection,
  model,
}: {
  volumeAxis: string | undefined;
  inspection: ModelInspectionRecord | undefined;
  model: BenchmarkModelCandidate;
}) {
  return (
    <section className="benchmark-model-detail-section">
      <h4>Model Manifest</h4>
      <dl className="benchmark-model-detail-grid">
        <dt>Model Key</dt>
        <dd>{model.model_key}</dd>
        <dt>Benchmark</dt>
        <dd>{model.benchmark_id}</dd>
        <dt>Program</dt>
        <dd>{model.program_digest}</dd>
        <dt>Observed {volumeAxis ?? 'Volume (bits)'}</dt>
        <dd>{observedVolumeLabel(model)}</dd>
        <dt>Manifest</dt>
        <dd>{inspection?.model_manifest === undefined ? 'not recorded' : referenceLabel(inspection.model_manifest)}</dd>
      </dl>
    </section>
  );
}

function ModelMeasurementDetail({ model }: { model: BenchmarkModelCandidate }) {
  return (
    <section className="benchmark-model-detail-section">
      <h4>Measurements</h4>
      <dl className="benchmark-model-cost-grid">
        <div>
          <dt>Score</dt>
          <dd>{model.score.toFixed(4)}</dd>
        </div>
        <div>
          <dt>Measurements</dt>
          <dd>{model.measurement_count.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Score Terms</dt>
          <dd>{model.score_integral.terms.length.toLocaleString()}</dd>
        </div>
        <div>
          <dt>Cost Terms</dt>
          <dd>{model.cost_integral?.terms.length.toLocaleString() ?? 'none'}</dd>
        </div>
      </dl>
      <CapabilityMapPanel map={model.capability_map} />
    </section>
  );
}

function CapabilityMapPanel({ map }: { map: CapabilityMapRecord | undefined }) {
  if (map === undefined) {
    return null;
  }
  const ladder = map.refinement_ladder;
  return (
    <section className="benchmark-capability-map" aria-label="Capability map">
      <div className="benchmark-capability-map-heading">
        <h5>Capability Map</h5>
        <dl>
          <div>
            <dt>Score</dt>
            <dd>{formatMetricNumber(map.value)}</dd>
          </div>
          <div>
            <dt>Uncertainty</dt>
            <dd>{formatMetricNumber(map.confidence_half_width)}</dd>
          </div>
          <div>
            <dt>Leaves</dt>
            <dd>{map.leaf_count.toLocaleString()}</dd>
          </div>
        </dl>
      </div>
      <div className="benchmark-capability-map-body">
        <CapabilityMapNode node={map.root} depth={0} />
      </div>
      {ladder.length === 0 ? null : (
        <div className="benchmark-capability-ladder" role="table" aria-label="Capability refinement ladder">
          <div className="benchmark-capability-ladder-row header" role="row">
            <span role="columnheader">Depth</span>
            <span role="columnheader">Leaves</span>
            <span role="columnheader">Score</span>
            <span role="columnheader">Movement</span>
          </div>
          {ladder.map((step) => (
            <div className="benchmark-capability-ladder-row" role="row" key={step.depth}>
              <span role="cell">{step.depth.toLocaleString()}</span>
              <span role="cell">{step.leaf_count.toLocaleString()}</span>
              <span role="cell">{formatMetricNumber(step.value)}</span>
              <span role="cell">
                {step.movement === undefined ? 'baseline' : formatMetricNumber(step.movement)}
              </span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function CapabilityMapNode({
  depth,
  node,
}: {
  depth: number;
  node: CapabilityMapNodeRecord;
}) {
  const competence = Math.max(0, Math.min(1, node.competence));
  const background = `linear-gradient(90deg, rgba(174, 55, 62, ${0.2 + (1 - competence) * 0.35}), rgba(58, 132, 94, ${0.18 + competence * 0.42}))`;
  return (
    <div className="benchmark-capability-node-group">
      <div
        className="benchmark-capability-node"
        style={{ background, marginLeft: `${depth * 1.25}rem` }}
      >
        <span>{node.label}</span>
        <span>{formatMetricNumber(node.competence)}</span>
        <span>{node.sample_count.toLocaleString()}</span>
      </div>
      {node.children.map((child) => (
        <CapabilityMapNode depth={depth + 1} key={`${node.label}:${child.label}`} node={child} />
      ))}
    </div>
  );
}

function ModelProgramDetail({
  volumeAxis,
  inspection,
  model,
}: {
  volumeAxis: string | undefined;
  inspection: ModelInspectionRecord | undefined;
  model: BenchmarkModelCandidate;
}) {
  return (
    <section className="benchmark-model-detail-section">
      <h4>Program</h4>
      <dl className="benchmark-model-detail-grid">
        <dt>Digest</dt>
        <dd>{model.program_digest}</dd>
        <dt>Input</dt>
        <dd>{inspection === undefined ? 'unknown' : shapeLabel(inspection.input_shape)}</dd>
        <dt>Output</dt>
        <dd>{inspection === undefined ? 'unknown' : shapeLabel(inspection.output_shape)}</dd>
        <dt>Observed {volumeAxis ?? 'Volume (bits)'}</dt>
        <dd>{observedVolumeLabel(model)}</dd>
        <dt>Sources</dt>
        <dd>{model.source_kinds.join(', ') || 'unknown'}</dd>
      </dl>
    </section>
  );
}

function ModelTrainingDetail({ runs }: { runs: RunResultRecord[] }) {
  if (runs.length === 0) {
    return (
      <section className="benchmark-model-detail-section">
        <p className="artifact-detail-note">No training runs match this model.</p>
      </section>
    );
  }
  const history = runs.flatMap((run) =>
    trainingValidationHistory(run).map((point) => ({
      ...point,
      run,
    })),
  );
  const latestRun = runs[0];
  const diagnostics = latestRun.training_diagnostics;
  const protocol = diagnostics?.protocol;
  return (
    <section className="benchmark-model-detail-section">
      <dl className="benchmark-model-training-grid">
        <div>
          <dt>Runs</dt>
          <dd>{runs.length}</dd>
        </div>
        <div>
          <dt>Latest</dt>
          <dd>{latestRun.run_slug}</dd>
        </div>
        <div>
          <dt>Score</dt>
          <dd>{latestRun.score.toFixed(4)}</dd>
        </div>
        <div>
          <dt>Measurements</dt>
          <dd>{latestRun.measurement_count}</dd>
        </div>
        {diagnostics === undefined ? null : (
          <>
            <div>
              <dt>Status</dt>
              <dd>{parameterValueLabel(diagnostics.status)}</dd>
            </div>
            <div>
              <dt>Stop</dt>
              <dd>{parameterValueLabel(diagnostics.stop_reason)}</dd>
            </div>
            <div>
              <dt>Final Loss</dt>
              <dd>{diagnostics.final_validation_loss.toFixed(4)}</dd>
            </div>
          </>
        )}
      </dl>
      {protocol === undefined ? null : (
        <dl className="benchmark-model-training-grid">
          {trainingProtocolEntries(protocol).map(([label, value]) => (
            <div key={label}>
              <dt>{label}</dt>
              <dd>{value}</dd>
            </div>
          ))}
        </dl>
      )}
      {history.length === 0 ? null : <ModelValidationChart points={history} />}
    </section>
  );
}

function ModelValidationChart({
  points,
}: {
  points: Array<ValidationHistoryPoint & { run: RunResultRecord }>;
}) {
  const steps = points.map((point) => point.step);
  const losses = points.map((point) => point.validation_loss);
  const references = points
    .map((point) => point.validation_loss_reference)
    .filter((value): value is number => value !== undefined && Number.isFinite(value) && value > 0);
  const xMin = Math.min(...steps);
  const xMax = Math.max(...steps, xMin + 1);
  const yMin = 0;
  const yMax = references.length === 0
    ? Math.max(...losses, Number.EPSILON)
    : Math.max(...references);
  const x = (step: number) =>
    modelValidationPlotMargin.left +
    ((step - xMin) / (xMax - xMin)) * modelValidationPlotBodyWidth;
  const y = (loss: number) => {
    const chartLoss = Math.max(yMin, Math.min(loss, yMax));
    return modelValidationPlotMargin.top +
      (1 - (chartLoss - yMin) / (yMax - yMin)) * modelValidationPlotBodyHeight;
  };
  const line = points.map((point) => `${x(point.step)},${y(point.validation_loss)}`).join(' ');
  return (
    <div className="benchmark-model-validation-chart">
      <svg
        aria-label="Validation loss history"
        role="img"
        viewBox={`0 0 ${modelValidationPlotWidth} ${modelValidationPlotHeight}`}
      >
        <rect
          className="benchmark-model-validation-frame"
          height={modelValidationPlotBodyHeight}
          width={modelValidationPlotBodyWidth}
          x={modelValidationPlotMargin.left}
          y={modelValidationPlotMargin.top}
        />
        <text
          className="benchmark-model-validation-tick"
          textAnchor="end"
          x={modelValidationPlotMargin.left - 8}
          y={y(yMax) + 4}
        >
          {yMax.toFixed(2)}
        </text>
        <text
          className="benchmark-model-validation-tick"
          textAnchor="middle"
          x={modelValidationPlotMargin.left}
          y={modelValidationPlotHeight - 8}
        >
          {xMin}
        </text>
        <text
          className="benchmark-model-validation-tick"
          textAnchor="middle"
          x={modelValidationPlotMargin.left + modelValidationPlotBodyWidth}
          y={modelValidationPlotHeight - 8}
        >
          {xMax}
        </text>
        <polyline
          className="benchmark-model-validation-loss"
          fill="none"
          points={line}
        />
      </svg>
    </div>
  );
}

function ModelCostDetail({
  inspection,
  model,
}: {
  inspection: ModelInspectionRecord | undefined;
  model: BenchmarkModelCandidate;
}) {
  const summary = model.cost_summary;
  return (
    <section className="benchmark-model-detail-section">
      <h4>Cost Summary</h4>
      <dl className="benchmark-model-cost-grid">
        <div>
          <dt>Components</dt>
          <dd>{modelComponentCount(inspection, model)}</dd>
        </div>
        <div>
          <dt>Graph Edges</dt>
          <dd>{optionalNumberLabel(inspection?.program_graph.edges.length)}</dd>
        </div>
        <div>
          <dt>Graph Inputs</dt>
          <dd>{optionalNumberLabel(inspection?.program_graph.inputs.length)}</dd>
        </div>
        <div>
          <dt>Graph Outputs</dt>
          <dd>{optionalNumberLabel(inspection?.program_graph.outputs.length)}</dd>
        </div>
        <div>
          <dt>Model Size</dt>
          <dd>{optionalNumberLabel(summary.storage_bytes)}</dd>
        </div>
        <div>
          <dt>Cost</dt>
          <dd>{optionalNumberLabel(summary.cost)}</dd>
        </div>
        <div>
          <dt>Inference Cost</dt>
          <dd>
            {optionalNumberLabel(costMeasurementPerSample(summary.inference_cost_measurement, summary.inference_cost_sample_count))}
            {summary.inference_cost_measurement?.operation_stream_source === undefined
              ? ''
              : ` (${summary.inference_cost_measurement.operation_stream_source})`}
          </dd>
        </div>
        <div>
          <dt>Unknown Parameter Components</dt>
          <dd>
            {unknownComponentLabel(
              inspection?.cost_summary.unknown_parameter_components,
            )}
          </dd>
        </div>
        <div>
          <dt>Unknown compute Components</dt>
          <dd>
            {unknownComponentLabel(inspection?.cost_summary.unknown_cost_components)}
          </dd>
        </div>
      </dl>
    </section>
  );
}

function ModelGraphOperations({
  inspection,
}: {
  inspection: ModelInspectionRecord | undefined;
}) {
  if (inspection === undefined) {
    return (
      <section className="benchmark-model-detail-section">
        <h4>Graph Operations</h4>
        <p className="artifact-detail-note">No model inspection record matches this model.</p>
      </section>
    );
  }
  return (
    <section className="benchmark-model-detail-section">
      <h4>Graph Operations</h4>
      <div className="benchmark-model-operation-list">
        {inspection.program_graph.nodes.map((graphNode, index) => {
          const component = inspection.components[index];
          return (
            <article className="benchmark-model-operation" key={graphNode.id}>
              <div className="benchmark-model-operation-heading">
                <span>{graphNode.id}</span>
                <div>
                  <strong>{graphNode.kind}</strong>
                  <small>{incomingEdgeLabel(inspection, graphNode.id)}</small>
                </div>
              </div>
              <dl className="benchmark-model-operation-shape-grid">
                <div>
                  <dt>Outgoing Edges</dt>
                  <dd>{outgoingEdgeCount(inspection, graphNode.id)}</dd>
                </div>
                <div>
                  <dt>Parameters</dt>
                  <dd>{Object.keys(graphNode.parameters ?? {}).length}</dd>
                </div>
              </dl>
              <p className="benchmark-model-operation-config">
                {component === undefined
                  ? 'none'
                  : recordLabel(component.parameters)}
              </p>
              {Object.entries(graphNode.parameters ?? {}).length === 0 ? null : (
                <dl className="benchmark-model-operator-grid">
                  {Object.entries(graphNode.parameters ?? {}).map(([key, value]) => (
                    <div key={key}>
                      <dt>{key}</dt>
                      <dd>{parameterValueLabel(value)}</dd>
                    </div>
                  ))}
                </dl>
              )}
            </article>
          );
        })}
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
  const references = modelProvenanceReferences(inspection);
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
            <dt>Kind</dt>
            <dd>{reference.kind}</dd>
          </dl>
        ))}
      </div>
    </section>
  );
}

function modelProvenanceReferences(
  inspection: ModelInspectionRecord,
): { label: string; reference: ArtifactReferenceRecord }[] {
  return [
    referenceEntry('Program', inspection.program),
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
}

function runsForModel(
  result: BenchmarkResultRecord | undefined,
  model: BenchmarkModelCandidate,
): RunResultRecord[] {
  if (result === undefined) {
    return [];
  }
  const runIds = new Set(model.run_ids);
  return result.training_history.filter(
    (run) =>
      run.model_key === model.model_key ||
      runIds.has(run.run_id),
  );
}

function trainingProtocolEntries(protocol: TrainingProtocolRecord): [string, string][] {
  const entries: [string, unknown][] = [
    ['Objective', protocol.objective],
    ['Optimizer', protocol.optimizer],
    ['Schedule', protocol.schedule],
    ['Steps', protocol.max_steps],
    ['Gate Check', protocol.gate_check_interval],
    ['Gate Rule', protocol.gate_decision_rule],
    ['Patience', protocol.patience],
    ['Validation', protocol.validation_source],
  ];
  if (protocol.learning_rate !== undefined) {
    entries.splice(3, 0, ['Learning Rate', protocol.learning_rate]);
  }
  return entries
    .filter(([, value]) => value !== undefined && value !== null && value !== '')
    .map(([label, value]) => [label, parameterValueLabel(value)]);
}

function trainingValidationHistory(run: RunResultRecord): ValidationHistoryPoint[] {
  return (run.training_diagnostics?.validation_history ?? []).map(
    (point: TrainingHistoryPointRecord) => ({
      stale_checks: point.stale_checks,
      step: point.step,
      validation_loss: point.validation_loss,
      validation_loss_reference: run.training_diagnostics?.validation_loss_reference,
    }),
  );
}

function shapeLabel(shape: number[]): string {
  return shape.join(' x ');
}

function observedVolumeLabel(model: BenchmarkModelCandidate): string {
  const volumes = model.points.map((point) => point.log2_volume);
  return volumes.length === 0 ? 'none' : volumes.join(', ');
}

function optionalNumberLabel(value: number | undefined): string {
  return value === undefined ? 'unknown' : value.toLocaleString();
}

function costMeasurementPerSample(
  measurement: { abstract_flops?: number } | undefined,
  sampleCount: number | undefined,
): number | undefined {
  if (measurement?.abstract_flops === undefined || sampleCount === undefined || sampleCount < 1) {
    return undefined;
  }
  return measurement.abstract_flops / sampleCount;
}

function formatMetricNumber(value: number): string {
  return Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, {
    maximumFractionDigits: 4,
  });
}

function modelComponentCount(
  inspection: ModelInspectionRecord | undefined,
  model: BenchmarkModelCandidate,
): number {
  return inspection?.components.length ?? model.cost_summary.component_count;
}

function unknownComponentLabel(components: number[] | undefined): string {
  return components === undefined || components.length === 0 ? 'none' : components.join(', ');
}

function recordLabel(
  record: Record<string, unknown>,
): string {
  const entries = Object.entries(record);
  if (entries.length === 0) {
    return 'none';
  }
  return entries
    .map(([key, value]) => `${key}: ${parameterValueLabel(value)}`)
    .join(', ');
}

function incomingEdgeLabel(inspection: ModelInspectionRecord, nodeId: string): string {
  const incoming = inspection.program_graph.edges
    .filter((edge) => edge.target_id === nodeId)
    .sort((left, right) => left.target_input_index - right.target_input_index);
  return incoming.length === 0
    ? 'no graph inputs'
    : incoming.map((edge) => `${edge.source_id} -> ${edge.target_input_index}`).join(', ');
}

function outgoingEdgeCount(inspection: ModelInspectionRecord, nodeId: string): number {
  return inspection.program_graph.edges.filter((edge) => edge.source_id === nodeId).length;
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
