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
        <p className="artifact-detail-note">No typed detail is available.</p>
      </DetailSection>
    );
  }

  if (detail.kind !== artifact.kind) {
    return (
      <DetailSection title="Document">
        <p className="artifact-detail-note">Typed detail does not match this artifact.</p>
      </DetailSection>
    );
  }

  if (detail.kind === 'architecture-manifest') {
    return (
      <DetailSection title="Architecture">
        <DetailItem label="Input Shape" value={shapeLabel(detail.input_shape)} />
        <DetailItem label="Output Shape" value={shapeLabel(detail.output_shape)} />
        <DetailItem label="Components" value={String(detail.architecture_graph.nodes.length)} />
        <DetailItem label="Graph Edges" value={String(detail.architecture_graph.edges.length)} />
        <DetailItem
          label="Graph Inputs"
          value={detail.architecture_graph.input_node_ids.join(', ')}
        />
        <DetailItem
          label="Graph Outputs"
          value={detail.architecture_graph.output_node_ids.join(', ')}
        />
        <ul className="artifact-detail-list">
          {detail.architecture_graph.nodes.map((node) => (
            <li key={node.id}>
              <span>{node.component.kind}</span>
              {node.component.parameters === undefined ? null : (
                <small>{`${node.id}, ${parameterLabel(node.component.parameters)}`}</small>
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
        {detail.outcome_space === undefined ? null : (
          <>
            <DetailItem label="Outcome Space" value={detail.outcome_space.id} />
            <DetailItem label="Outcomes" value={String(detail.outcome_space.outcomes.length)} />
          </>
        )}
        {detail.outcome_sequence === undefined ? null : (
          <>
            <DetailItem
              label="Outcome Atoms"
              value={`${detail.outcome_sequence.atom_count} ${detail.outcome_sequence.atom_name}s`}
            />
            <DetailItem
              label="Sequence Length"
              value={detail.outcome_sequence.length_parameter}
            />
          </>
        )}
        {detail.scale_parameter === undefined ? null : (
          <DetailItem
            label="Scale Parameter"
            value={`${detail.scale_parameter.symbol} >= ${detail.scale_parameter.minimum}`}
          />
        )}
        {detail.complexity_coordinate === undefined ? null : (
          <DetailItem label="Complexity" value={detail.complexity_coordinate} />
        )}
        {detail.latent_factor_declaration?.protocol_id === undefined ? null : (
          <DetailItem
            label="Latent Factors"
            value={detail.latent_factor_declaration.protocol_id}
          />
        )}
        {detail.observation_ids === undefined ? null : (
          <DetailItem label="Observations" value={String(detail.observation_ids.length)} />
        )}
        {detail.outcome_space === undefined ? null : (
          <OutcomeList outcomes={detail.outcome_space.outcomes.map((outcome) => outcome.id)} />
        )}
      </DetailSection>
    );
  }

  if (detail.kind === 'latent-factor-declaration') {
    return (
      <DetailSection title="Latent Factors">
        <DetailItem label="Declaration ID" value={detail.id} />
        <DetailItem label="Construction Factors" value={String(detail.construction_factors.length)} />
        <DetailItem label="Sample Factors" value={String(detail.sample_factors.length)} />
        <DetailItem label="Complexity Views" value={String(detail.complexity_projections.length)} />
        <ul className="artifact-detail-list">
          {detail.sample_factors.map((factor) => (
            <li key={factor.name}>
              <span>{factor.name}</span>
              <small>
                {factor.role}, {degreeMeasureLabel(factor.degree_measure)}
              </small>
            </li>
          ))}
        </ul>
        {detail.resolution_requirements.length === 0 ? null : (
          <ul className="artifact-detail-list">
            {detail.resolution_requirements.map((requirement) => (
              <li key={requirement.name}>
                <span>{requirement.name}</span>
                <small>
                  {requirement.resolution_axis} {'>='} {requirement.minimum_resolution}
                </small>
              </li>
            ))}
          </ul>
        )}
      </DetailSection>
    );
  }

  if (detail.kind === 'materialization-declaration') {
    return (
      <DetailSection title="Materialization">
        <DetailItem label="Declaration ID" value={detail.id} />
        <DetailItem label="Benchmark ID" value={detail.benchmark_id} />
        <DetailItem label="Requirements" value={String(detail.requirements.length)} />
        {detail.latent_factor_declaration?.protocol_id === undefined ? null : (
          <DetailItem
            label="Latent Factors"
            value={detail.latent_factor_declaration.protocol_id}
          />
        )}
        <ul className="artifact-detail-list">
          {detail.requirements.map((requirement) => (
            <li key={requirement.name}>
              <span>{requirement.name}</span>
              <small>
                {requirement.resolution_axis} {'>='} {linearRequirementLabel(requirement)}
              </small>
            </li>
          ))}
        </ul>
      </DetailSection>
    );
  }

  if (detail.kind === 'materialization-plan') {
    return (
      <DetailSection title="Materialization Plan">
        <DetailItem label="Plan ID" value={detail.id} />
        <DetailItem label="Benchmark ID" value={detail.benchmark_id} />
        <DetailItem label="Scale" value={axisAssignmentLabel(detail.scale_assignment)} />
        <DetailItem label="Complexity" value={axisAssignmentLabel(detail.complexity_assignment)} />
        <DetailItem label="Resolution" value={axisAssignmentLabel(detail.resolution_assignment)} />
        <DetailItem label="Seed" value={String(detail.seed)} />
        {detail.materialization_declaration.protocol_id === undefined ? null : (
          <DetailItem
            label="Declaration"
            value={detail.materialization_declaration.protocol_id}
          />
        )}
      </DetailSection>
    );
  }

  if (detail.kind === 'observation-formation-declaration') {
    return (
      <DetailSection title="Observation Formation">
        <DetailItem label="Declaration ID" value={detail.id} />
        <DetailItem label="Benchmark ID" value={detail.benchmark_id} />
        <DetailItem label="Interpreter" value={detail.interpreter} />
        <DetailItem
          label="Output Field"
          value={`${detail.output_field.channel_count} channel(s), ${detail.output_field.resolution_axis}`}
        />
        <DetailItem
          label="Slots"
          value={`${detail.slot_composition.count_axis} along ${detail.slot_composition.slot_axis}`}
        />
        <DetailItem label="Components" value={String(detail.component_count)} />
        <DetailItem label="Marks" value={String(detail.mark_count)} />
        <ul className="artifact-detail-list">
          {detail.components.map((component) => (
            <li key={component.id}>
              <span>{component.id}</span>
              <small>{component.marks.map((mark) => `${mark.kind} d${mark.degree}`).join(', ')}</small>
            </li>
          ))}
        </ul>
      </DetailSection>
    );
  }

  if (detail.kind === 'observation-showcase') {
    return (
      <DetailSection title="Observation Showcase">
        <DetailItem label="Showcase ID" value={detail.id} />
        <DetailItem label="Benchmark ID" value={detail.benchmark_id} />
        <DetailItem label="Samples" value={String(detail.samples.length)} />
        {detail.formation_declaration.protocol_id === undefined ? null : (
          <DetailItem label="Formation" value={detail.formation_declaration.protocol_id} />
        )}
        {detail.materialization_declaration.protocol_id === undefined ? null : (
          <DetailItem
            label="Materialization"
            value={detail.materialization_declaration.protocol_id}
          />
        )}
        <ul className="artifact-detail-list">
          {detail.samples.map((sample) => (
            <li key={sample.id}>
              <span>{sample.label}</span>
              <small>
                {axisAssignmentLabel(sample.scale_assignment)}, {sample.component_sequence.join('-')}
              </small>
            </li>
          ))}
        </ul>
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

function degreeMeasureLabel(measure: { kind: string; count: number; domain_size?: number }): string {
  if (measure.domain_size === undefined) {
    return `${measure.kind}: ${measure.count}`;
  }
  return `${measure.kind}: ${measure.domain_size}`;
}

function axisAssignmentLabel(assignment: { values: { axis: string; value: number }[] }): string {
  return assignment.values.map((item) => `${item.axis}: ${item.value}`).join(', ');
}

function linearRequirementLabel(requirement: {
  source_axis: string;
  coefficient: number;
  intercept?: number;
  minimum?: number;
}): string {
  const base = `${requirement.coefficient} * ${requirement.source_axis}`;
  const shifted = requirement.intercept === undefined ? base : `${base} + ${requirement.intercept}`;
  return requirement.minimum === undefined ? shifted : `max(${shifted}, ${requirement.minimum})`;
}
