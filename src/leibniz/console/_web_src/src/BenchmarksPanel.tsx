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
  benchmarkCostAxes,
  benchmarkResultsForTask,
  costValue,
  emptyFrontiersForCostAxes,
  formatCost,
  modelComparisonRows,
  scoreLabel,
  shortDigest,
} from './benchmarkDashboardModel.ts';
import type { ArtifactReferenceRecord } from './artifactReferences.ts';
import type {
  BenchmarkTaskRecord,
  GeneratedObservationBatchRecord,
  GeneratedObservationSampleRecord,
} from './benchmarkTasks.ts';
import type {
  ModelInspectionRecord,
  ModelInspectionTraceStageRecord,
} from './modelInspections.ts';
import {
  descriptorAxisDisplayName,
  descriptorValueDisplayName,
  operatorDisplayName,
  parameterDisplayName,
  syntaxAliasDisplayName,
  type OperatorVocabularyRecord,
} from './operatorVocabulary.ts';
import { usePersistentState } from './persistentState.ts';
import type {
  BenchmarkResultRecord,
  ResultViewRecord,
  RunDetailSectionRecord,
  RunResultRecord,
  TrainingHistoryPointRecord,
  TrainingProtocolRecord,
} from './resultViews.ts';

type SampleCardDensity = 'standard' | 'compact';
type ModelArtifactView = 'model' | 'architecture' | 'training' | 'artifacts';
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
  best_validation_loss: number;
  best_validation_step: number;
  stale_checks?: number;
  step: number;
  validation_loss: number;
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
  operatorVocabulary,
  resultViews,
  tasks,
}: {
  modelInspections: ModelInspectionRecord[];
  operatorVocabulary: OperatorVocabularyRecord;
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
            label="Samples"
            storageKey="leibniz.console.benchmarks.section.samples.expanded"
          >
            <BenchmarkTaskPane task={selected} />
          </CollapsibleBenchmarkSection>
          <CollapsibleBenchmarkSection
            label="Performance"
            summary={`${result?.leaderboard.length ?? 0} models`}
            storageKey="leibniz.console.benchmarks.section.performance.expanded"
          >
            <BenchmarkPerformancePane
              benchmark={selected}
              operatorVocabulary={operatorVocabulary}
              resultEntry={selectedResult}
            />
          </CollapsibleBenchmarkSection>
          {modelRows.length === 0 ? null : (
            <CollapsibleBenchmarkSection
              label="Models"
              summary={`${modelRows.length} inspected candidates`}
              storageKey="leibniz.console.benchmarks.section.models.expanded"
            >
              <BenchmarkModelsPane
                operatorVocabulary={operatorVocabulary}
                rows={modelRows}
                result={result}
              />
            </CollapsibleBenchmarkSection>
          )}
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
  operatorVocabulary,
  resultEntry,
}: {
  benchmark: BenchmarkTaskRecord;
  operatorVocabulary: OperatorVocabularyRecord;
  resultEntry:
    | BenchmarkResultEntry
    | undefined;
}) {
  const result = resultEntry?.result ?? emptyBenchmarkResult(benchmark);

  return (
    <div className="benchmark-task">
      <BenchmarkResultDashboard
        operatorVocabulary={operatorVocabulary}
        result={result}
      />
    </div>
  );
}

function emptyBenchmarkResult(benchmark: BenchmarkTaskRecord): BenchmarkResultRecord {
  const costAxes = benchmarkCostAxes(undefined);
  return {
    benchmark_id: benchmark.benchmark_id,
    complexity_axis: benchmark.complexity_axis,
    cost_axes: costAxes,
    frontiers: emptyFrontiersForCostAxes(costAxes),
    leaderboard: [],
    model_inspections: [],
    proposals: [],
    training_history: [],
  };
}

function BenchmarkModelsPane({
  operatorVocabulary,
  rows,
  result,
}: {
  operatorVocabulary: OperatorVocabularyRecord;
  rows: ReturnType<typeof modelComparisonRows>;
  result: BenchmarkResultRecord | undefined;
}) {
  const costAxis = benchmarkCostAxes(result)[0]?.key ?? 'parameter_count';
  const [selectedModelKey, setSelectedModelKey] = usePersistentState(
    `leibniz.console.benchmarks.${result?.benchmark_id ?? 'empty'}.selectedModel`,
    rows[0]?.model.model_key ?? '',
  );
  const selectedRow =
    rows.find(({ model }) => model.model_key === selectedModelKey) ?? rows[0];

  return (
    <div className="benchmark-task">
      <section className="benchmark-model-workbench">
        <div className="benchmark-model-workbench-heading">
          <h3>Model Workbench</h3>
          <span>{rows.length} candidates</span>
        </div>
        <div className="benchmark-model-inspector-layout">
          <aside className="benchmark-model-rail" aria-label="Benchmark model candidates">
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
                  <div>
                    <dt>Cost</dt>
                    <dd>{formatCost(costValue(model.cost_summary, costAxis))}</dd>
                  </div>
                  <div>
                    <dt>Components</dt>
                    <dd>{modelComponentCount(inspection, model)}</dd>
                  </div>
                  <div>
                    <dt>Runs</dt>
                    <dd>{model.run_ids.length}</dd>
                  </div>
                  <div>
                    <dt>Sources</dt>
                    <dd>{model.source_kinds.length}</dd>
                  </div>
                </dl>
              </button>
            ))}
          </aside>
          {selectedRow === undefined ? null : (
            <BenchmarkModelInspector
              complexityAxis={result?.complexity_axis}
              costAxis={costAxis}
              inspection={selectedRow.inspection}
              model={selectedRow.model}
              operatorVocabulary={operatorVocabulary}
              runs={runsForModel(result, selectedRow.model)}
            />
          )}
        </div>
      </section>
    </div>
  );
}

function BenchmarkModelInspector({
  complexityAxis,
  costAxis,
  inspection,
  model,
  operatorVocabulary,
  runs,
}: {
  complexityAxis: string | undefined;
  costAxis: string;
  inspection: ModelInspectionRecord | undefined;
  model: BenchmarkResultRecord['leaderboard'][number];
  operatorVocabulary: OperatorVocabularyRecord;
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
          <h3>{shortDigest(model.architecture_digest)}</h3>
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
      {artifactView === 'model' ? (
        <ModelManifestDetail
          complexityAxis={complexityAxis}
          inspection={inspection}
          model={model}
        />
      ) : null}
      {artifactView === 'architecture' ? (
        <>
          <ModelArchitectureDetail
            complexityAxis={complexityAxis}
            inspection={inspection}
            model={model}
          />
          <ModelCostDetail inspection={inspection} model={model} />
          <ModelGraphOperations inspection={inspection} operatorVocabulary={operatorVocabulary} />
        </>
      ) : null}
      {artifactView === 'training' ? (
        <ModelTrainingDetail runs={runs} />
      ) : null}
      {artifactView === 'artifacts' ? (
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
  model: BenchmarkResultRecord['leaderboard'][number];
  onSelect: (view: ModelArtifactView) => void;
}) {
  const items: ModelArtifactFlowItem[] = [
    {
      icon: PackageCheck,
      label: 'Architecture',
      value: inspection === undefined
        ? shortDigest(model.architecture_digest)
        : referenceLabel(inspection.architecture),
      view: 'architecture',
    },
    {
      icon: Boxes,
      label: 'Measurements',
      value: inspection?.measurement_dataset === undefined
        ? `${model.measurement_count} records`
        : referenceLabel(inspection.measurement_dataset),
      view: 'model',
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
      label: 'Artifacts',
      value: inspection?.model_artifacts.length
        ? `${inspection.model_artifacts.length} records`
        : 'not recorded',
      view: 'artifacts',
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
  model: BenchmarkResultRecord['leaderboard'][number];
  runs: RunResultRecord[];
}) {
  const inputNodes = [
    lineageNode('input', 'architecture', model.architecture_digest),
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
  complexityAxis,
  inspection,
  model,
}: {
  complexityAxis: string | undefined;
  inspection: ModelInspectionRecord | undefined;
  model: BenchmarkResultRecord['leaderboard'][number];
}) {
  const sections = model.console_view_model?.detail_sections ?? [];
  return (
    <section className="benchmark-model-detail-section">
      <h4>Model Manifest</h4>
      {sections.length === 0 ? null : (
        <div className="benchmark-model-generated-summary">
          {sections.map((section) => (
            <ModelGeneratedSummarySection key={section.title} section={section} />
          ))}
        </div>
      )}
      <dl className="benchmark-model-detail-grid">
        <dt>Model Key</dt>
        <dd>{model.model_key}</dd>
        <dt>Benchmark</dt>
        <dd>{model.benchmark_id}</dd>
        <dt>Architecture</dt>
        <dd>{model.architecture_digest}</dd>
        <dt>Observed {complexityAxis ?? 'Complexity'}</dt>
        <dd>{model.observed_complexities.join(', ') || 'none'}</dd>
        <dt>Manifest</dt>
        <dd>{inspection?.model_manifest === undefined ? 'not recorded' : referenceLabel(inspection.model_manifest)}</dd>
      </dl>
    </section>
  );
}

function ModelGeneratedSummarySection({ section }: { section: RunDetailSectionRecord }) {
  const entries = section.entries ?? [];
  if (entries.length === 0) {
    return null;
  }
  return (
    <section className="run-evidence-panel">
      <h5>{section.title}</h5>
      <dl>
        {entries.map((entry) => (
          <div key={entry.label}>
            <dt>{entry.label}</dt>
            <dd>{entry.value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function ModelArchitectureDetail({
  complexityAxis,
  inspection,
  model,
}: {
  complexityAxis: string | undefined;
  inspection: ModelInspectionRecord | undefined;
  model: BenchmarkResultRecord['leaderboard'][number];
}) {
  return (
    <section className="benchmark-model-detail-section">
      <h4>Architecture</h4>
      <dl className="benchmark-model-detail-grid">
        <dt>Digest</dt>
        <dd>{model.architecture_digest}</dd>
        <dt>Input</dt>
        <dd>{inspection === undefined ? 'unknown' : shapeLabel(inspection.input_shape)}</dd>
        <dt>Output</dt>
        <dd>{inspection === undefined ? 'unknown' : shapeLabel(inspection.output_shape)}</dd>
        <dt>Observed {complexityAxis ?? 'Complexity'}</dt>
        <dd>{model.observed_complexities.join(', ') || 'none'}</dd>
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
        <h4>Training History</h4>
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
      <h4>Training History</h4>
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
              <dt>Best Loss</dt>
              <dd>{diagnostics.best_validation_loss.toFixed(4)}</dd>
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
      {diagnostics === undefined ? null : (
        <TrainingArtifactReferences artifacts={diagnostics.artifacts} />
      )}
      {history.length === 0 ? null : <ModelValidationChart points={history} />}
    </section>
  );
}

function TrainingArtifactReferences({
  artifacts,
}: {
  artifacts: NonNullable<RunResultRecord['training_diagnostics']>['artifacts'];
}) {
  if (artifacts.length === 0) {
    return null;
  }
  return (
    <dl className="benchmark-model-training-artifacts">
      {artifacts.map((artifact) => (
        <div key={`${artifact.kind}:${artifact.digest}`}>
          <dt>{parameterValueLabel(artifact.kind)}</dt>
          <dd>{artifact.path ?? shortDigest(artifact.digest)}</dd>
        </div>
      ))}
    </dl>
  );
}

function ModelValidationChart({
  points,
}: {
  points: Array<ValidationHistoryPoint & { run: RunResultRecord }>;
}) {
  const steps = points.map((point) => point.step);
  const losses = points.flatMap((point) => [
    point.validation_loss,
    point.best_validation_loss,
  ]);
  const xMin = Math.min(...steps);
  const xMax = Math.max(...steps, xMin + 1);
  const yMin = 0;
  const yMax = Math.max(...losses, Number.EPSILON);
  const x = (step: number) =>
    modelValidationPlotMargin.left +
    ((step - xMin) / (xMax - xMin)) * modelValidationPlotBodyWidth;
  const y = (loss: number) =>
    modelValidationPlotMargin.top +
    (1 - (loss - yMin) / (yMax - yMin)) * modelValidationPlotBodyHeight;
  const line = (key: 'validation_loss' | 'best_validation_loss') =>
    points.map((point) => `${x(point.step)},${y(point[key])}`).join(' ');
  return (
    <div className="benchmark-model-validation-chart">
      <div className="benchmark-model-validation-legend">
        <span><i className="loss" />Loss</span>
        <span><i className="best" />Best</span>
      </div>
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
          points={line('validation_loss')}
        />
        <polyline
          className="benchmark-model-validation-best"
          fill="none"
          points={line('best_validation_loss')}
        />
        {points.map((point, index) => (
          <circle
            className="benchmark-model-validation-point"
            cx={x(point.step)}
            cy={y(point.validation_loss)}
            key={`${point.run.run_id}:${point.step}:${index}`}
            r={3}
          />
        ))}
      </svg>
    </div>
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
      <dl className="benchmark-model-cost-grid">
        <div>
          <dt>Components</dt>
          <dd>{modelComponentCount(inspection, model)}</dd>
        </div>
        <div>
          <dt>Graph Edges</dt>
          <dd>{optionalNumberLabel(inspection?.architecture_summary.edge_count)}</dd>
        </div>
        <div>
          <dt>Graph Inputs</dt>
          <dd>{optionalNumberLabel(inspection?.architecture_summary.input_count)}</dd>
        </div>
        <div>
          <dt>Graph Outputs</dt>
          <dd>{optionalNumberLabel(inspection?.architecture_summary.output_count)}</dd>
        </div>
        <div>
          <dt>Parameters</dt>
          <dd>{optionalNumberLabel(summary.parameter_count)}</dd>
        </div>
        <div>
          <dt>Bytes</dt>
          <dd>{optionalNumberLabel(summary.parameter_bytes)}</dd>
        </div>
        <div>
          <dt>FLOPs</dt>
          <dd>{optionalNumberLabel(summary.inference_flops)}</dd>
        </div>
        <div>
          <dt>Unknown Parameter Components</dt>
          <dd>
            {unknownComponentLabel(
              inspection?.architecture_summary.unsupported_parameter_components,
            )}
          </dd>
        </div>
        <div>
          <dt>Unknown FLOP Components</dt>
          <dd>
            {unknownComponentLabel(inspection?.architecture_summary.unsupported_flop_components)}
          </dd>
        </div>
      </dl>
    </section>
  );
}

function ModelGraphOperations({
  inspection,
  operatorVocabulary,
}: {
  inspection: ModelInspectionRecord | undefined;
  operatorVocabulary: OperatorVocabularyRecord;
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
        {inspection.architecture_trace.stages.map((stage) => {
          const component = inspection.components[stage.index];
          const graphNode = inspection.architecture_graph.nodes[stage.index];
          return (
            <article className="benchmark-model-operation" key={stage.index}>
              <div className="benchmark-model-operation-heading">
                <span>{graphNode?.id ?? stage.index}</span>
                <div>
                  <strong>{operatorDisplayName(operatorVocabulary, stage.operator_kind)}</strong>
                  <small>{syntaxAliasDisplayName(operatorVocabulary, stage.syntax_alias)}</small>
                </div>
              </div>
              <dl className="benchmark-model-operation-shape-grid">
                <div>
                  <dt>Input</dt>
                  <dd>{shapeLabel(stage.input_shape)}</dd>
                </div>
                <div>
                  <dt>Output</dt>
                  <dd>{shapeLabel(stage.output_shape)}</dd>
                </div>
                <div>
                  <dt>Parameters</dt>
                  <dd>{optionalNumberLabel(stage.parameter_count)}</dd>
                </div>
                <div>
                  <dt>FLOPs</dt>
                  <dd>{optionalNumberLabel(stage.inference_flops)}</dd>
                </div>
              </dl>
              <p className="benchmark-model-operation-config">
                {component === undefined
                  ? 'none'
                  : recordLabel(component.parameters, component.operator, operatorVocabulary)}
              </p>
              {traceStageEntries(stage, operatorVocabulary).length === 0 ? null : (
                <dl className="benchmark-model-operator-grid">
                  {traceStageEntries(stage, operatorVocabulary).map(([key, value]) => (
                    <div key={key}>
                      <dt>{key}</dt>
                      <dd>{value}</dd>
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
            <dt>Kind</dt>
            <dd>{reference.kind}</dd>
          </dl>
        ))}
      </div>
    </section>
  );
}

function runsForModel(
  result: BenchmarkResultRecord | undefined,
  model: BenchmarkResultRecord['leaderboard'][number],
): RunResultRecord[] {
  if (result === undefined) {
    return [];
  }
  const runIds = new Set(model.run_ids);
  return result.training_history.filter(
    (run) =>
      run.model_key === model.model_key ||
      run.architecture_digest === model.architecture_digest ||
      runIds.has(run.run_id),
  );
}

function trainingProtocolEntries(protocol: TrainingProtocolRecord): [string, string][] {
  const entries: [string, unknown][] = [
    ['Objective', protocol.objective],
    ['Optimizer', protocol.optimizer],
    ['Schedule', protocol.schedule],
    ['Learning Rate', protocol.learning_rate],
    ['Steps', protocol.max_steps],
    ['Min Steps', protocol.min_steps],
    ['Batch', protocol.batch_size],
    ['Interval', protocol.validation_interval],
    ['Patience', protocol.patience],
    ['Validation', protocol.validation_source],
  ];
  return entries
    .filter(([, value]) => value !== undefined && value !== null && value !== '')
    .map(([label, value]) => [label, parameterValueLabel(value)]);
}

function trainingValidationHistory(run: RunResultRecord): ValidationHistoryPoint[] {
  return (run.training_diagnostics?.validation_history ?? []).map(
    (point: TrainingHistoryPointRecord) => ({
      best_validation_loss: point.best_validation_loss,
      best_validation_step: point.best_validation_step,
      stale_checks: point.stale_checks,
      step: point.step,
      validation_loss: point.validation_loss,
    }),
  );
}

function shapeLabel(shape: number[]): string {
  return shape.join(' x ');
}

function optionalNumberLabel(value: number | undefined): string {
  return value === undefined ? 'unknown' : value.toLocaleString();
}

function modelComponentCount(
  inspection: ModelInspectionRecord | undefined,
  model: BenchmarkResultRecord['leaderboard'][number],
): number {
  return inspection?.architecture_summary.component_count ?? model.cost_summary.component_count;
}

function unknownComponentLabel(components: number[] | undefined): string {
  return components === undefined || components.length === 0 ? 'none' : components.join(', ');
}

function traceStageEntries(
  stage: ModelInspectionTraceStageRecord,
  vocabulary: OperatorVocabularyRecord,
): [string, string][] {
  const descriptorEntry = (axis: string, value: string): [string, string] => [
    descriptorAxisDisplayName(vocabulary, axis),
    descriptorValueDisplayName(vocabulary, axis, value),
  ];
  return [
    descriptorEntry('tensor_relation', stage.descriptor_axes.tensor_relation),
    descriptorEntry('support', stage.descriptor_axes.support),
    descriptorEntry('state', stage.descriptor_axes.state),
    descriptorEntry('shape_law', stage.shape_law),
    descriptorEntry('cost_law', stage.cost_law),
  ].filter((entry): entry is [string, string] => entry[1] !== undefined && entry[1] !== '');
}

function recordLabel(
  record: Record<string, unknown>,
  operator?: Record<string, unknown>,
  vocabulary?: OperatorVocabularyRecord,
): string {
  const entries = Object.entries(record);
  if (entries.length === 0) {
    return 'none';
  }
  const operatorKind = typeof operator?.kind === 'string' ? operator.kind : undefined;
  return entries
    .map(
      ([key, value]) =>
        `${vocabulary === undefined ? key : parameterDisplayName(vocabulary, operatorKind, key)}: ${parameterValueLabel(value)}`,
    )
    .join(', ');
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
  const [selectedSampleKey, setSelectedSampleKey] = usePersistentState<string | null>(
    `leibniz.console.benchmarks.${task.benchmark_id}.selectedSample`,
    null,
  );
  const selected = task.batches[0];
  const visibleSamples = selected?.samples.map((sample) => ({ batch: selected, sample })) ?? [];
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
      {selectedSample === undefined ? null : (
        <BenchmarkSampleCoordinateInspector
          sample={selectedSample.sample}
        />
      )}
      <section
        className={`benchmark-sample-grid ${selected.presentation.sample_card_density}`}
        aria-label="Generated benchmark samples"
      >
        {visibleSamples.map(({ batch, sample }) => (
          <BenchmarkSampleCard
            density={batch.presentation.sample_card_density}
            key={`${batch.mode}-${batch.component_count}-${sample.index}-${sample.outcome_id}`}
            onSelect={() => setSelectedSampleKey(sampleKey(batch, sample))}
            sample={sample}
            selected={sampleKey(batch, sample) === selectedKey}
          />
        ))}
      </section>
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
        <div className="benchmark-image-fit">
          <img alt={sample.outcome_id} src={sample.image_data_url} />
        </div>
      </div>
    </button>
  );
}

function BenchmarkSampleCoordinateInspector({
  sample,
}: {
  sample: GeneratedObservationSampleRecord;
}) {
  const entries: [string, string][] = [
    ['Components', sample.component_sequence.join(' ')],
    ['Complexity', String(sample.complexity)],
    ['Field', sample.field_shape.join(' x ')],
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

function sampleKey(
  batch: GeneratedObservationBatchRecord,
  sample: GeneratedObservationSampleRecord,
): string {
  return `${batch.mode}:${batch.component_count}:${batch.seed}:${batch.sample_count}:${sample.index}:${sample.outcome_id}`;
}
