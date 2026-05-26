import type {
  CompetenceIntegralEntryRecord,
  CompetenceIntegralPointRecord,
  MeasurementRecord,
  PerformanceViewRecord,
} from './performanceViews.ts';

export function BenchmarkPerformanceBundle({ view }: { view: PerformanceViewRecord }) {
  const entry = view.competence_integral_view.entries[0];

  return (
    <section className="performance-section benchmark-performance-bundle">
      <header className="performance-header">
        <div>
          <h3>{view.manifest.view_id}</h3>
          <p>{view.source_path}</p>
        </div>
        {entry === undefined ? null : <PerformanceMetrics entry={entry} />}
      </header>
      {entry === undefined ? (
        <p className="artifact-detail-note">No competence integral entries are available.</p>
      ) : (
        <>
          <CompetencePoints
            measurements={view.measurement_dataset.measurements}
            points={entry.points}
          />
          <PerformanceEvidence view={view} />
        </>
      )}
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
