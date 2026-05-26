import { useState } from 'react';

import type {
  CompetenceIntegralEntryRecord,
  CompetenceIntegralPointRecord,
  MeasurementRecord,
  PerformanceViewRecord,
} from './performanceViews.ts';
import type { ImportedResultViewRecord } from './resultViews.ts';

export function PerformanceViewPanel({
  resultViews,
  views,
}: {
  resultViews: ImportedResultViewRecord[];
  views: PerformanceViewRecord[];
}) {
  const [selectedId, setSelectedId] = useState(views[0]?.id ?? '');
  const selected = views.find((view) => view.id === selectedId) ?? views[0];

  if (selected === undefined) {
    return (
      <section className="performance-layout">
        <ImportedResultViews views={resultViews} />
        <p className="artifact-detail-note">No performance views are available.</p>
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
          <p className="artifact-detail-note">No competence integral entries are available.</p>
        ) : (
          <>
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

function ImportedResultViews({ views }: { views: ImportedResultViewRecord[] }) {
  if (views.length === 0) {
    return null;
  }

  return (
    <section className="imported-results" aria-label="Imported result views">
      <h3>Imported Results</h3>
      {views.map((view) => (
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
