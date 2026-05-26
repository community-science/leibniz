import type { ConsoleArtifactIndexEntryRecord } from './artifactIndex';
import type { ConsoleArtifactDetailRecord } from './artifactDetails';
import { DetailItem, DetailSection } from './ArtifactDetailPrimitives';

const probabilityFractionDigits = 2;

export function ArtifactTypedDetail({
  artifact,
  detail,
}: {
  artifact: ConsoleArtifactIndexEntryRecord;
  detail: ConsoleArtifactDetailRecord | undefined;
}) {
  if (detail === undefined) {
    return (
      <DetailSection title="Document">
        <p className="artifact-detail-note">No typed detail fixture is available.</p>
      </DetailSection>
    );
  }

  if (detail.kind !== artifact.kind) {
    return (
      <DetailSection title="Document">
        <p className="artifact-detail-note">Typed detail fixture does not match this artifact.</p>
      </DetailSection>
    );
  }

  if (detail.kind === 'architecture-manifest') {
    return (
      <DetailSection title="Architecture">
        <DetailItem label="Input Shape" value={shapeLabel(detail.input_shape)} />
        <DetailItem label="Output Shape" value={shapeLabel(detail.output_shape)} />
        <DetailItem label="Layers" value={String(detail.layers.length)} />
        <ul className="artifact-detail-list">
          {detail.layers.map((layer, index) => (
            <li key={`${index}:${layer.kind}`}>
              <span>{layer.kind}</span>
              {layer.parameters === undefined ? null : (
                <small>{parameterLabel(layer.parameters)}</small>
              )}
            </li>
          ))}
        </ul>
      </DetailSection>
    );
  }

  if (detail.kind === 'benchmark-manifest') {
    return (
      <DetailSection title="Benchmark">
        <DetailItem label="Benchmark ID" value={detail.id} />
        <DetailItem label="Outcome Space" value={detail.outcome_space.id} />
        <DetailItem label="Outcomes" value={String(detail.outcome_space.outcomes.length)} />
        {detail.observation_ids === undefined ? null : (
          <DetailItem label="Observations" value={String(detail.observation_ids.length)} />
        )}
        <OutcomeList outcomes={detail.outcome_space.outcomes.map((outcome) => outcome.id)} />
      </DetailSection>
    );
  }

  return (
    <DetailSection title="Measurement">
      <DetailItem label="Evidence ID" value={detail.id} />
      <DetailItem label="Benchmark ID" value={detail.benchmark_id} />
      <DetailItem label="Observation" value={detail.observation_id} />
      <DetailItem label="Accepted Outcomes" value={detail.accepted_event.outcomes.join(', ')} />
      <DetailItem
        label="Probabilities"
        value={String(detail.probability_measure.probabilities.length)}
      />
      <ul className="artifact-detail-list">
        {detail.probability_measure.probabilities.map((probability) => (
          <li key={probability.outcome_id}>
            <span>{probability.outcome_id}</span>
            <small>{probability.probability.toFixed(probabilityFractionDigits)}</small>
          </li>
        ))}
      </ul>
    </DetailSection>
  );
}

function OutcomeList({ outcomes }: { outcomes: string[] }) {
  return (
    <ul className="artifact-detail-list">
      {outcomes.map((outcome) => (
        <li key={outcome}>
          <span>{outcome}</span>
        </li>
      ))}
    </ul>
  );
}

function parameterLabel(parameters: Record<string, string | number | boolean>): string {
  return Object.entries(parameters)
    .map(([key, value]) => `${key}: ${String(value)}`)
    .join(', ');
}

function shapeLabel(shape: number[]): string {
  return shape.join(' x ');
}
