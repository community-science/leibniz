import { useState } from 'react';

import type {
  CompetenceIntegralEntryRecord,
  CompetenceIntegralPointRecord,
  MeasurementRecord,
  PerformanceViewRecord,
} from './performanceViews.ts';
import type {
  BenchmarkResultRecord,
  BenchmarkResultViewRecord,
  ImportedResultViewRecord,
  ModelResultRecord,
  ProposalRecord,
  ResultViewRecord,
  RunResultRecord,
} from './resultViews.ts';

export function PerformanceViewPanel({
  resultViews,
  views,
}: {
  resultViews: ResultViewRecord[];
  views: PerformanceViewRecord[];
}) {
  const [selectedId, setSelectedId] = useState(views[0]?.id ?? '');
  const selected = views.find((view) => view.id === selectedId) ?? views[0];
  const benchmarkResults = resultViews.filter(isBenchmarkResultView).flatMap((view) =>
    view.benchmark_results.map((result) => ({ sourcePath: view.source_path, result })),
  );

  if (selected === undefined) {
    return (
      <section className="performance-layout">
        <ImportedResultViews views={resultViews} />
        {benchmarkResults[0] === undefined ? (
          <p className="artifact-detail-note">No performance views are available.</p>
        ) : (
          <BenchmarkResultDashboard
            result={benchmarkResults[0].result}
            sourcePath={benchmarkResults[0].sourcePath}
          />
        )}
      </section>
    );
  }

  const entry = selected.competence_integral_view.entries[0];

  return (
    <section className="performance-layout" aria-label="Performance views">
      <aside className="performance-list" aria-label="Performance view bundles">
        <ImportedResultViews views={resultViews} />
        {views.map((view) => (
          <button
            className={`performance-list-item ${view.id === selected.id ? 'active' : ''}`}
            key={view.id}
            onClick={() => setSelectedId(view.id)}
            type="button"
          >
            <span>{view.manifest.view_id}</span>
            <small>{view.source_path}</small>
          </button>
        ))}
      </aside>

      <article className="performance-detail">
        <header className="performance-header">
          <div>
            <h2>{selected.manifest.view_id}</h2>
            <p>{selected.source_path}</p>
          </div>
          {entry === undefined ? null : <PerformanceMetrics entry={entry} />}
        </header>

        {entry === undefined ? (
          benchmarkResults[0] === undefined ? (
            <p className="artifact-detail-note">No competence integral entries are available.</p>
          ) : (
            <BenchmarkResultDashboard
              result={benchmarkResults[0].result}
              sourcePath={benchmarkResults[0].sourcePath}
            />
          )
        ) : (
          <>
            {benchmarkResults[0] === undefined ? null : (
              <BenchmarkResultDashboard
                result={benchmarkResults[0].result}
                sourcePath={benchmarkResults[0].sourcePath}
              />
            )}
            <CompetencePoints
              measurements={selected.measurement_dataset.measurements}
              points={entry.points}
            />
            <PerformanceEvidence view={selected} />
          </>
        )}
      </article>
    </section>
  );
}

function BenchmarkResultDashboard({
  result,
  sourcePath,
}: {
  result: BenchmarkResultRecord;
  sourcePath: string;
}) {
  const [costAxis, setCostAxis] = useState(result.cost_axes[0]?.key ?? 'parameter_count');
  const frontier = result.frontiers[costAxis] ?? [];
  const topModel = result.leaderboard[0];

  return (
    <section className="performance-section benchmark-result-dashboard">
      <div className="benchmark-result-heading">
        <div>
          <h3>Benchmark Results</h3>
          <p>{sourcePath}</p>
        </div>
        <label className="benchmark-result-axis">
          <span>Cost Axis</span>
          <select value={costAxis} onChange={(event) => setCostAxis(event.target.value)}>
            {result.cost_axes.map((axis) => (
              <option key={axis.key} value={axis.key}>
                {axis.label}
              </option>
            ))}
          </select>
        </label>
      </div>
      {topModel === undefined ? (
        <p className="artifact-detail-note">No model results are available.</p>
      ) : (
        <dl className="performance-metrics">
          <div>
            <dt>Benchmark</dt>
            <dd>{result.benchmark_id}</dd>
          </div>
          <div>
            <dt>Best Score</dt>
            <dd>{topModel.score.toFixed(4)}</dd>
          </div>
          <div>
            <dt>Models</dt>
            <dd>{result.leaderboard.length}</dd>
          </div>
          <div>
            <dt>Runs</dt>
            <dd>{result.training_history.length}</dd>
          </div>
        </dl>
      )}
      <ModelResultTable
        costAxis={costAxis}
        models={frontier}
        title="Frontier"
      />
      <ModelResultTable
        costAxis={costAxis}
        models={result.leaderboard}
        title="Leaderboard"
      />
      <ProposalCards proposals={result.proposals} />
      <RunHistoryTable costAxis={costAxis} runs={result.training_history} />
    </section>
  );
}

function ProposalCards({ proposals }: { proposals: ProposalRecord[] }) {
  if (proposals.length === 0) {
    return <p className="artifact-detail-note">No active proposals are available.</p>;
  }

  return (
    <section className="benchmark-result-table-section">
      <h3>Proposals</h3>
      <div className="proposal-card-grid">
        {proposals.map((proposal) => (
          <article className="proposal-card" key={proposal.id}>
            <div className="proposal-card-heading">
              <span>Rank {proposal.rank}</span>
              <strong>{scoreLabel(proposal.acquisition_value)}</strong>
            </div>
            <dl>
              <dt>Candidate</dt>
              <dd>{proposal.candidate_id}</dd>
              <dt>Prediction</dt>
              <dd>{scoreLabel(proposal.predicted_score)}</dd>
              <dt>Uncertainty</dt>
              <dd>{scoreLabel(proposal.uncertainty)}</dd>
              <dt>Novelty</dt>
              <dd>{scoreLabel(proposal.novelty)}</dd>
              <dt>Improvement</dt>
              <dd>{scoreLabel(proposal.expected_frontier_improvement)}</dd>
            </dl>
            <p>{proposal.rationale}</p>
            {proposal.command.length === 0 ? null : <code>{proposal.command.join(' ')}</code>}
          </article>
        ))}
      </div>
    </section>
  );
}

function ModelResultTable({
  costAxis,
  models,
  title,
}: {
  costAxis: string;
  models: ModelResultRecord[];
  title: string;
}) {
  if (models.length === 0) {
    return <p className="artifact-detail-note">No {title.toLowerCase()} records are available.</p>;
  }

  return (
    <section className="benchmark-result-table-section">
      <h3>{title}</h3>
      <div className="benchmark-result-table" role="table" aria-label={title}>
        <div className="benchmark-result-row header" role="row">
          <span role="columnheader">Model</span>
          <span role="columnheader">Score</span>
          <span role="columnheader">Cost</span>
          <span role="columnheader">C</span>
          <span role="columnheader">Runs</span>
        </div>
        {models.map((model) => (
          <div className="benchmark-result-row" key={model.model_key} role="row">
            <span role="cell">{shortDigest(model.architecture_digest)}</span>
            <span role="cell">{model.score.toFixed(4)}</span>
            <span role="cell">{formatCost(costValue(model.cost_summary, costAxis))}</span>
            <span role="cell">{model.observed_complexities.join(', ') || 'none'}</span>
            <span role="cell">{model.run_ids.length}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function RunHistoryTable({ costAxis, runs }: { costAxis: string; runs: RunResultRecord[] }) {
  if (runs.length === 0) {
    return <p className="artifact-detail-note">No training history is available.</p>;
  }

  return (
    <section className="benchmark-result-table-section">
      <h3>Training History</h3>
      <div className="benchmark-result-table" role="table" aria-label="Training history">
        <div className="benchmark-result-row header" role="row">
          <span role="columnheader">Run</span>
          <span role="columnheader">Score</span>
          <span role="columnheader">Cost</span>
          <span role="columnheader">Scale</span>
          <span role="columnheader">Measurements</span>
        </div>
        {runs.map((run) => (
          <div className="benchmark-result-row" key={`${run.source_kind}:${run.run_id}`} role="row">
            <span role="cell">{run.run_slug}</span>
            <span role="cell">{run.score.toFixed(4)}</span>
            <span role="cell">{formatCost(costValue(run.cost_summary, costAxis))}</span>
            <span role="cell">{run.scale ?? 'n/a'}</span>
            <span role="cell">{run.measurement_count}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

function ImportedResultViews({ views }: { views: ResultViewRecord[] }) {
  if (views.length === 0) {
    return null;
  }

  const benchmarkResultViews = views.filter(isBenchmarkResultView);
  const importedResultViews = views.filter(isImportedResultView);

  return (
    <section className="imported-results" aria-label="Imported result views">
      <h3>Result Views</h3>
      {benchmarkResultViews.map((view) => (
        <article className="imported-result-view" key={view.source_path}>
          <span>{view.source_path}</span>
          <dl>
            <dt>Benchmarks</dt>
            <dd>{view.benchmark_results.length}</dd>
            <dt>Models</dt>
            <dd>
              {view.benchmark_results.reduce(
                (sum, result) => sum + result.leaderboard.length,
                0,
              )}
            </dd>
            <dt>Runs</dt>
            <dd>
              {view.benchmark_results.reduce(
                (sum, result) => sum + result.training_history.length,
                0,
              )}
            </dd>
          </dl>
        </article>
      ))}
      {importedResultViews.map((view) => (
        <article className="imported-result-view" key={view.source_path}>
          <span>{view.source_path}</span>
          <dl>
            <dt>Bundles</dt>
            <dd>{view.publication_bundles.length}</dd>
            <dt>Measurements</dt>
            <dd>
              {view.publication_bundles.reduce(
                (sum, bundle) => sum + bundle.measurement_count,
                0,
              )}
            </dd>
          </dl>
        </article>
      ))}
    </section>
  );
}

function isBenchmarkResultView(view: ResultViewRecord): view is BenchmarkResultViewRecord {
  return view.format === 'leibniz.console.benchmark-results';
}

function isImportedResultView(view: ResultViewRecord): view is ImportedResultViewRecord {
  return view.format === 'leibniz.console.imported-results';
}

function costValue(costSummary: Record<string, unknown>, costAxis: string): number {
  const value = costSummary[costAxis];
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function formatCost(value: number): string {
  return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
}

function scoreLabel(value: number | undefined): string {
  return value === undefined ? 'n/a' : value.toFixed(4);
}

function shortDigest(value: string): string {
  const digest = value.includes(':') ? value.split(':')[1] : value;
  return digest.slice(0, 12);
}

function PerformanceMetrics({ entry }: { entry: CompetenceIntegralEntryRecord }) {
  return (
    <dl className="performance-metrics">
      <div>
        <dt>Integral</dt>
        <dd>{entry.integral.toFixed(4)}</dd>
      </div>
      <div>
        <dt>Coverage</dt>
        <dd>{entry.coverage.toFixed(4)}</dd>
      </div>
      <div>
        <dt>Observed C</dt>
        <dd>{entry.observed_complexities.join(', ')}</dd>
      </div>
      <div>
        <dt>Missing C</dt>
        <dd>{entry.missing_complexities.join(', ') || 'none'}</dd>
      </div>
    </dl>
  );
}

function CompetencePoints({
  measurements,
  points,
}: {
  measurements: MeasurementRecord[];
  points: CompetenceIntegralPointRecord[];
}) {
  const measurementById = new Map(
    measurements.map((measurement) => [measurement.raw_scoring_evidence.id, measurement]),
  );

  return (
    <section className="performance-section">
      <h3>Competence By Complexity</h3>
      <div className="performance-points">
        {points.map((point) => {
          const measurement = measurementById.get(point.measurement_id);
          return (
            <article className="performance-point" key={point.measurement_id}>
              <div className="performance-point-heading">
                <span>C={point.complexity}</span>
                <strong>{point.competence.toFixed(4)}</strong>
              </div>
              <div className="performance-bar" aria-hidden="true">
                <div style={{ width: `${point.competence * 100}%` }} />
              </div>
              <dl>
                <dt>Measurement</dt>
                <dd>{point.measurement_id}</dd>
                <dt>Accepted</dt>
                <dd>{measurement?.accepted_event.outcomes.join(', ') ?? 'unknown'}</dd>
                <dt>Probability</dt>
                <dd>{probabilityLabel(measurement)}</dd>
              </dl>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function PerformanceEvidence({ view }: { view: PerformanceViewRecord }) {
  return (
    <section className="performance-section">
      <h3>Generated Sources</h3>
      <div className="performance-source-grid">
        {view.manifest.measurement_cases.map((measurementCase) => (
          <article className="performance-source" key={measurementCase.id}>
            <span>{axisAssignmentLabel(measurementCase.complexity_assignment)}</span>
            <dl>
              <dt>Case</dt>
              <dd>{measurementCase.id}</dd>
              <dt>Components</dt>
              <dd>{measurementCase.component_sequence.join(', ')}</dd>
              <dt>Accepted</dt>
              <dd>{measurementCase.accepted_outcome_sequence.join(', ')}</dd>
              <dt>Probabilities</dt>
              <dd>{measurementCase.probabilities.map(probabilityMassLabel).join('; ')}</dd>
            </dl>
          </article>
        ))}
      </div>
    </section>
  );
}

function probabilityLabel(measurement: MeasurementRecord | undefined): string {
  if (measurement === undefined) {
    return 'unknown';
  }
  return measurement.probability_measure.probabilities
    .map((probability) => `${probability.outcome_id}: ${probability.probability}`)
    .join(', ');
}

function probabilityMassLabel(probability: {
  outcome_id?: string;
  outcome_sequence?: number[];
  probability: number;
}): string {
  const outcome =
    probability.outcome_id ?? `sequence ${probability.outcome_sequence?.join(', ') ?? ''}`;
  return `${outcome}: ${probability.probability}`;
}

function axisAssignmentLabel(assignment: { values: { axis: string; value: number }[] }): string {
  return assignment.values.map((item) => `${item.axis}: ${item.value}`).join(', ');
}
