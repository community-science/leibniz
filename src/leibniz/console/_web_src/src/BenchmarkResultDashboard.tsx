import { useState } from 'react';
import type { CSSProperties } from 'react';

import {
  costValue,
  formatCost,
  scoreLabel,
  shortDigest,
} from './benchmarkDashboardModel.ts';
import type {
  BenchmarkResultRecord,
  ModelResultRecord,
  ProposalRecord,
  RunResultRecord,
} from './resultViews.ts';

export function BenchmarkResultDashboard({
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
      <FrontierChart costAxis={costAxis} models={frontier} />
      <ModelResultTable costAxis={costAxis} models={frontier} title="Frontier" />
      <ModelResultTable costAxis={costAxis} models={result.leaderboard} title="Leaderboard" />
      <ProposalCards proposals={result.proposals} />
      <RunHistoryTable costAxis={costAxis} runs={result.training_history} />
    </section>
  );
}

function FrontierChart({
  costAxis,
  models,
}: {
  costAxis: string;
  models: ModelResultRecord[];
}) {
  if (models.length === 0) {
    return null;
  }

  const costs = models.map((model) => costValue(model.cost_summary, costAxis));
  const minCost = Math.min(...costs);
  const maxCost = Math.max(...costs);
  const minScore = Math.min(...models.map((model) => model.score));
  const maxScore = Math.max(...models.map((model) => model.score));

  return (
    <section className="benchmark-result-table-section">
      <h3>Frontier Chart</h3>
      <div className="frontier-chart" role="img" aria-label="Frontier score by selected cost">
        {models.map((model) => {
          const x = normalizedPosition(costValue(model.cost_summary, costAxis), minCost, maxCost);
          const y = normalizedPosition(model.score, minScore, maxScore);
          return (
            <span
              className="frontier-chart-point"
              key={model.model_key}
              style={{ '--point-x': `${x}%`, '--point-y': `${100 - y}%` } as CSSProperties}
              title={`${shortDigest(model.architecture_digest)} score ${model.score.toFixed(4)}`}
            />
          );
        })}
      </div>
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

function normalizedPosition(value: number, min: number, max: number): number {
  if (max === min) {
    return 50;
  }
  return ((value - min) / (max - min)) * 100;
}
