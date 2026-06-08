import {
  Boxes,
  ChevronDown,
  Code2,
  Fingerprint,
  GitBranch,
  PackageCheck,
  type LucideIcon,
} from 'lucide-react';
import type { ReactNode } from 'react';
import { useMemo } from 'react';
import { useState } from 'react';

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
import type {
  BenchmarkCodeSurfaceRecord,
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
type BenchmarkModelCandidate = BenchmarkResultRecord['model_candidates'][number];
type ModelArtifactView = 'model' | 'architecture' | 'training' | 'provenance';
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
              operatorVocabulary={operatorVocabulary}
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
    complexity_axis: benchmark.complexity_axis,
    frontiers: emptyFrontiersForCostAxis(),
    leaderboard: [],
    model_candidates: [],
    model_inspections: [],
    plot_runs: [],
    training_history: [],
  };
}

function BenchmarkModelsPane({
  operatorVocabulary,
  rows,
  result,
  selectedModelKey,
}: {
  operatorVocabulary: OperatorVocabularyRecord;
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
              complexityAxis={result?.complexity_axis}
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
  inspection,
  model,
  operatorVocabulary,
  runs,
}: {
  complexityAxis: string | undefined;
  inspection: ModelInspectionRecord | undefined;
  model: BenchmarkModelCandidate;
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
  model: BenchmarkModelCandidate;
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
        <dd>{observedComplexityLabel(model)}</dd>
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
  model: BenchmarkModelCandidate;
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
        <dd>{observedComplexityLabel(model)}</dd>
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
          <dt>Model Size</dt>
          <dd>{optionalNumberLabel(summary.storage_bytes)}</dd>
        </div>
        <div>
          <dt>Cost</dt>
          <dd>{optionalNumberLabel(summary.cost)}</dd>
        </div>
        <div>
          <dt>Inference Compute</dt>
          <dd>{optionalNumberLabel(summary.inference_compute)}</dd>
        </div>
        <div>
          <dt>Training Compute</dt>
          <dd>{optionalNumberLabel(summary.training_compute)}</dd>
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
          <dt>Unknown compute Components</dt>
          <dd>
            {unknownComponentLabel(inspection?.architecture_summary.unsupported_compute_components)}
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
                  <dt>Inference Compute</dt>
                  <dd>{optionalNumberLabel(stage.inference_compute)}</dd>
                </div>
                <div>
                  <dt>Training Compute / Sample</dt>
                  <dd>{optionalNumberLabel(stage.training_compute_per_sample)}</dd>
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
    ['Training Batch Target', protocol.training_batch_target],
    ['Gate Check', protocol.gate_check_interval],
    ['Gate Batch Target', protocol.gate_batch_target],
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

function observedComplexityLabel(model: BenchmarkModelCandidate): string {
  const complexities = model.points.map((point) => point.complexity);
  return complexities.length === 0 ? 'none' : complexities.join(', ');
}

function optionalNumberLabel(value: number | undefined): string {
  return value === undefined ? 'unknown' : value.toLocaleString();
}

function modelComponentCount(
  inspection: ModelInspectionRecord | undefined,
  model: BenchmarkModelCandidate,
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
  const defaultBatch = task.batches.find((batch) => batch.sample_count > 0) ?? task.batches[0];
  const defaultBatchKey = defaultBatch === undefined ? null : batchKey(defaultBatch);
  const [selectedBatchKey, setSelectedBatchKey] = usePersistentState<string | null>(
    `leibniz.console.benchmarks.${task.benchmark_id}.selectedBatch`,
    defaultBatchKey,
  );
  const [selectedSampleKey, setSelectedSampleKey] = usePersistentState<string | null>(
    `leibniz.console.benchmarks.${task.benchmark_id}.selectedSample`,
    null,
  );

  if (task.batches.length === 0) {
    return <p className="artifact-detail-note">No generated samples are available.</p>;
  }

  const selectedBatch =
    task.batches.find((batch) => batchKey(batch) === selectedBatchKey) ??
    task.batches.find((batch) => batch.sample_count > 0) ??
    task.batches[0];
  const visibleSamples = selectedBatch.samples.map((sample) => ({
    batch: selectedBatch,
    sample,
  }));
  const selectedSample =
    visibleSamples.find(
      ({ batch, sample }) => sampleKey(batch, sample) === selectedSampleKey,
    ) ?? visibleSamples[0];
  const selectedKey =
    selectedSample === undefined
      ? null
      : sampleKey(selectedSample.batch, selectedSample.sample);
  const hasInteractiveImageOverlay = selectedBatch.samples.some(
    (sample) => sample.image_overlay?.kind === 'grid-move-highlights',
  );

  return (
    <div className="benchmark-task">
      <BenchmarkCodeSurfaceInspector surfaces={task.code_surfaces} />
      <BenchmarkSampleCoordinateInspector
        batches={task.batches}
        onBatchChange={(key) => {
          setSelectedBatchKey(key);
          setSelectedSampleKey(null);
        }}
        sample={selectedSample?.sample}
        selectedBatch={selectedBatch}
      />
      <section
        aria-label={`Generated benchmark samples ${selectedBatch.label}`}
        className="benchmark-sample-window"
      >
        {selectedBatch.samples.length === 0 ? (
          <p className="artifact-detail-note">No generated samples in this range.</p>
        ) : (
          <div
            className={`benchmark-sample-grid ${
              selectedBatch.presentation.sample_card_density
            } ${hasInteractiveImageOverlay ? 'interactive-image-overlay' : ''}`}
          >
            {selectedBatch.samples.map((sample) => (
              <BenchmarkSampleCard
                density={selectedBatch.presentation.sample_card_density}
                key={sampleKey(selectedBatch, sample)}
                onSelect={() => setSelectedSampleKey(sampleKey(selectedBatch, sample))}
                sample={sample}
                selected={sampleKey(selectedBatch, sample) === selectedKey}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function BenchmarkCodeSurfaceInspector({
  surfaces,
}: {
  surfaces: BenchmarkCodeSurfaceRecord[];
}) {
  const [expanded, setExpanded] = usePersistentState(
    'leibniz.console.benchmarks.codeInspector.expanded',
    false,
  );
  const selected = surfaces[0];
  if (selected === undefined) {
    return null;
  }
  return (
    <section className="benchmark-code-inspector" aria-label="Benchmark implementation code">
      <div className="benchmark-code-inspector-header">
        <button
          aria-expanded={expanded}
          className="benchmark-code-toggle"
          onClick={() => setExpanded((value) => !value)}
          type="button"
        >
          <ChevronDown aria-hidden="true" className={expanded ? 'expanded' : ''} size={16} />
          <Code2 aria-hidden="true" size={16} />
          <span>Implementation</span>
        </button>
        <div className="benchmark-code-source">
          {selected.source_path}:{selected.start_line}-{selected.end_line}
        </div>
      </div>
      <div className="benchmark-code-inspector-body" hidden={!expanded}>
        <ol className="benchmark-code-call-path" aria-label="Implementation call path">
          {selected.call_path.map((step) => (
            <li key={step}>{step}</li>
          ))}
        </ol>
        <pre className="benchmark-code-excerpt">
          <code>{selected.code}</code>
        </pre>
      </div>
    </section>
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
  const imageUrl = sample.image_data_url;
  const overlay = sample.image_overlay;
  const [hoveredSource, setHoveredSource] = useState<string | null>(null);
  return (
    <button
      className={`benchmark-sample-card ${density} ${
        overlay?.kind === 'grid-move-highlights' ? 'interactive-image-overlay' : ''
      } ${selected ? 'selected' : ''}`}
      onClick={onSelect}
      type="button"
    >
      {imageUrl === undefined ? (
        <div className="benchmark-text-sample-shell">
          <div className="benchmark-text-sample-outcome">{sample.outcome_id}</div>
          {sample.observable_state_id === undefined ? null : (
            <div className="benchmark-text-sample-state">{sample.observable_state_id}</div>
          )}
        </div>
      ) : (
        <div className="benchmark-image-shell">
          <div className="benchmark-image-fit">
            <img alt={sample.outcome_id} src={imageUrl} />
            {overlay?.kind === 'grid-move-highlights' ? (
              <BenchmarkGridMoveOverlay
                hoveredSource={hoveredSource}
                onHoverSource={setHoveredSource}
                overlay={overlay}
              />
            ) : null}
          </div>
        </div>
      )}
    </button>
  );
}

function BenchmarkGridMoveOverlay({
  hoveredSource,
  onHoverSource,
  overlay,
}: {
  hoveredSource: string | null;
  onHoverSource: (source: string | null) => void;
  overlay: NonNullable<GeneratedObservationSampleRecord['image_overlay']>;
}) {
  const sourceKeys = new Set(overlay.moves.map((move) => gridCoordinateKey(move.from)));
  const destinationKeys = new Set(
    overlay.moves
      .filter((move) => gridCoordinateKey(move.from) === hoveredSource)
      .map((move) => gridCoordinateKey(move.to)),
  );
  const targetDestinationKeys = new Set(
    overlay.moves
      .filter(
        (move) =>
          gridCoordinateKey(move.from) === hoveredSource &&
          (move.target_probability ?? 0) > 0,
      )
      .map((move) => gridCoordinateKey(move.to)),
  );
  const cells = [];
  for (let row = 0; row < overlay.rows; row += 1) {
    for (let column = 0; column < overlay.columns; column += 1) {
      const key = `${column},${row}`;
      const isSource = sourceKeys.has(key);
      const isDestination = destinationKeys.has(key);
      const isTargetDestination = targetDestinationKeys.has(key);
      const className = [
        'benchmark-grid-move-cell',
        isSource ? 'source' : '',
        isDestination ? 'destination' : '',
        isTargetDestination ? 'target-destination' : '',
      ].filter(Boolean).join(' ');
      cells.push(
        <div
          aria-hidden="true"
          className={className}
          key={key}
          onMouseEnter={() => {
            if (isSource) {
              onHoverSource(key);
            }
          }}
          onMouseLeave={() => {
            if (isSource) {
              onHoverSource(null);
            }
          }}
        />,
      );
    }
  }
  return (
    <div
      className="benchmark-grid-move-overlay"
      style={{
        gridTemplateColumns: `repeat(${overlay.columns}, 1fr)`,
        gridTemplateRows: `repeat(${overlay.rows}, 1fr)`,
      }}
    >
      {cells}
    </div>
  );
}

function gridCoordinateKey(coordinate: [number, number]): string {
  return `${coordinate[0]},${coordinate[1]}`;
}

function BenchmarkSampleCoordinateInspector({
  batches,
  onBatchChange,
  sample,
  selectedBatch,
}: {
  batches: GeneratedObservationBatchRecord[];
  onBatchChange: (key: string) => void;
  sample: GeneratedObservationSampleRecord | undefined;
  selectedBatch: GeneratedObservationBatchRecord;
}) {
  const complexityCardinalities = realizedComplexityCardinalities(selectedBatch);
  const entries: [string, string][] = [
    ['Complexity Classes', complexityCardinalities.length === 0 ? 'null set' : complexityCardinalities.join(', ')],
  ];
  if (sample !== undefined) {
    entries.push(['Outcome', sample.outcome_id]);
    if (sample.component_index !== undefined) {
      entries.push(['Component', String(sample.component_index)]);
    }
    if (sample.field_shape !== undefined) {
      entries.push(['Field', sample.field_shape.join(' x ')]);
    }
    if (sample.available_outcome_ids !== undefined) {
      entries.push(['Available Outcomes', String(sample.available_outcome_ids.length)]);
    }
    if (sample.observable_state_id !== undefined) {
      entries.push(['State', sample.observable_state_id]);
    }
  }
  return (
    <section
      className="benchmark-sample-coordinate-inspector"
      aria-label="Selected sample coordinates"
    >
      <div className="benchmark-sample-coordinate-range">
        <label htmlFor="benchmark-complexity-range">Complexity Range</label>
        <select
          className="benchmark-sample-window-select"
          id="benchmark-complexity-range"
          onChange={(event) => onBatchChange(event.target.value)}
          value={batchKey(selectedBatch)}
        >
          {batches.map((batch) => (
            <option key={batchKey(batch)} value={batchKey(batch)}>
              {batch.label} ({batch.sample_count})
            </option>
          ))}
        </select>
      </div>
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

function realizedComplexityCardinalities(batch: GeneratedObservationBatchRecord): string[] {
  if (batch.complexity_cardinalities !== undefined) {
    return batch.complexity_cardinalities.map((size) => String(size));
  }
  const sizes = new Set<number>();
  batch.samples.forEach((sample) => {
    if (sample.complexity_value === undefined || sample.complexity_value === null) {
      return;
    }
    sizes.add(Math.round(2 ** sample.complexity_value.value));
  });
  return [...sizes].sort((left, right) => left - right).map((size) => String(size));
}

function sampleKey(
  batch: GeneratedObservationBatchRecord,
  sample: GeneratedObservationSampleRecord,
): string {
  return `${batchKey(batch)}:${sample.index}:${sample.outcome_id}`;
}

function batchKey(batch: GeneratedObservationBatchRecord): string {
  return `${batch.mode}:${batch.label}:${batch.seed}:${batch.sample_count}`;
}
