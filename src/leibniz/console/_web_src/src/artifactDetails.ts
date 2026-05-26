import type { ArtifactReferenceRecord, ConsoleArtifactIndexEntryRecord } from './artifactIndex';
import { artifactKey } from './artifactBrowserModel.ts';

export type ArchitectureManifestDetailRecord = {
  kind: 'architecture-manifest';
  source_path: string;
  input_shape: number[];
  output_shape: number[];
  layers: LayerSummaryRecord[];
};

export type BenchmarkManifestDetailRecord = {
  kind: 'benchmark-manifest';
  source_path: string;
  id: string;
  outcome_space?: OutcomeSpaceSummaryRecord;
  outcome_sequence?: OutcomeSequenceSummaryRecord;
  scale_parameter?: ScaleParameterSummaryRecord;
  observation_ids?: string[];
  latent_factor_declaration?: ArtifactReferenceRecord;
  complexity_coordinate?: string;
};

export type MeasurementDetailRecord = {
  kind: 'measurement';
  source_path: string;
  id: string;
  benchmark_id: string;
  observation_id: string;
  outcome_space: OutcomeSpaceSummaryRecord;
  accepted_event: {
    id: string;
    outcome_space_id: string;
    outcomes: string[];
  };
  probability_measure: {
    id: string;
    outcome_space_id: string;
    probabilities: ProbabilitySummaryRecord[];
  };
};

export type LatentFactorDeclarationDetailRecord = {
  kind: 'latent-factor-declaration';
  source_path: string;
  id: string;
  construction_factors: LatentFactorSummaryRecord[];
  sample_factors: SampleLatentFactorSummaryRecord[];
  complexity_projections: ComplexityProjectionSummaryRecord[];
  resolution_requirements: ResolutionRequirementSummaryRecord[];
};

export type MaterializationDeclarationDetailRecord = {
  kind: 'materialization-declaration';
  source_path: string;
  id: string;
  benchmark_id: string;
  requirements: LinearResolutionRequirementSummaryRecord[];
  latent_factor_declaration?: ArtifactReferenceRecord;
  layout?: Record<string, unknown>;
};

export type MaterializationPlanDetailRecord = {
  kind: 'materialization-plan';
  source_path: string;
  id: string;
  benchmark_id: string;
  materialization_declaration: ArtifactReferenceRecord;
  scale_assignment: AxisAssignmentSummaryRecord;
  complexity_assignment: AxisAssignmentSummaryRecord;
  resolution_assignment: AxisAssignmentSummaryRecord;
  seed: number;
  latent_factor_declaration?: ArtifactReferenceRecord;
};

export type ObservationFormationDeclarationDetailRecord = {
  kind: 'observation-formation-declaration';
  source_path: string;
  id: string;
  benchmark_id: string;
  interpreter: string;
  output_field: OutputFieldSummaryRecord;
  slot_composition: SlotCompositionSummaryRecord;
  component_count: number;
  mark_count: number;
  components: ObservationComponentSummaryRecord[];
};

export type ObservationShowcaseDetailRecord = {
  kind: 'observation-showcase';
  source_path: string;
  id: string;
  benchmark_id: string;
  formation_declaration: ArtifactReferenceRecord;
  materialization_declaration: ArtifactReferenceRecord;
  samples: ObservationShowcaseSampleSummaryRecord[];
};

export type ConsoleArtifactDetailRecord =
  | ArchitectureManifestDetailRecord
  | BenchmarkManifestDetailRecord
  | LatentFactorDeclarationDetailRecord
  | MaterializationDeclarationDetailRecord
  | MaterializationPlanDetailRecord
  | MeasurementDetailRecord
  | ObservationFormationDeclarationDetailRecord
  | ObservationShowcaseDetailRecord;

export type ConsoleArtifactDetailMap = ReadonlyMap<string, ConsoleArtifactDetailRecord>;

type LayerSummaryRecord = {
  kind: string;
  parameters?: Record<string, string | number | boolean>;
};

type OutcomeSpaceSummaryRecord = {
  id: string;
  outcomes: { id: string }[];
};

type OutcomeSequenceSummaryRecord = {
  atom_count: number;
  atom_name: string;
  length_parameter: string;
};

type ScaleParameterSummaryRecord = {
  symbol: string;
  minimum: number;
  description?: string;
};

type ProbabilitySummaryRecord = {
  outcome_id: string;
  probability: number;
};

type DegreeMeasureSummaryRecord = {
  kind: string;
  count: number;
  domain_size?: number;
};

type LatentFactorSummaryRecord = {
  name: string;
  degree_measure: DegreeMeasureSummaryRecord;
  description?: string;
};

type SampleLatentFactorSummaryRecord = LatentFactorSummaryRecord & {
  role: string;
  multiplicity?: number;
};

type ComplexityProjectionSummaryRecord = {
  name: string;
  coordinate: string;
  included_roles: string[];
  description?: string;
};

type ResolutionRequirementSummaryRecord = {
  name: string;
  resolution_axis: string;
  content_coordinate: string;
  content_complexity: number;
  minimum_resolution: number;
  basis: string;
  description?: string;
};

type LinearResolutionRequirementSummaryRecord = {
  name: string;
  source_axis: string;
  resolution_axis: string;
  coefficient: number;
  intercept?: number;
  minimum?: number;
  basis: string;
  description?: string;
};

type AxisAssignmentSummaryRecord = {
  values: { axis: string; value: number }[];
};

type OutputFieldSummaryRecord = {
  channel_count: number;
  resolution_axis: string;
};

type SlotCompositionSummaryRecord = {
  count_axis: string;
  resolution_axis: string;
  slot_axis: string;
};

type ObservationComponentSummaryRecord = {
  id: string;
  marks: ObservationMarkSummaryRecord[];
};

type ObservationMarkSummaryRecord = {
  kind: string;
  channel: number;
  degree: number;
  control_points: number[][];
  width: number;
  value?: number;
};

type ObservationShowcaseSampleSummaryRecord = {
  id: string;
  label: string;
  sample_index: number;
  scale_assignment: AxisAssignmentSummaryRecord;
  complexity_assignment: AxisAssignmentSummaryRecord;
  seed: number;
  component_sequence: number[];
  outcome_id?: string;
};

export class ConsoleArtifactDetailError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConsoleArtifactDetailError';
  }
}

export function artifactDetailKey(detail: ConsoleArtifactDetailRecord): string {
  return `${detail.kind}:${detail.source_path}`;
}

export function artifactDetailKeyForEntry(artifact: ConsoleArtifactIndexEntryRecord): string {
  return `${artifact.kind}:${artifactKey(artifact)}`;
}

export function detailForArtifact(
  details: ConsoleArtifactDetailMap | undefined,
  artifact: ConsoleArtifactIndexEntryRecord,
): ConsoleArtifactDetailRecord | undefined {
  if (details === undefined) {
    return undefined;
  }
  return details.get(artifactDetailKeyForEntry(artifact));
}

export function parseConsoleArtifactDetailRecords(value: unknown): ConsoleArtifactDetailMap {
  const records = requireArray(value, 'artifact details').map((detail, index) =>
    parseDetailRecord(detail, `artifact details.${index}`),
  );
  return new Map(records.map((detail) => [artifactDetailKey(detail), detail]));
}

function parseDetailRecord(value: unknown, path: string): ConsoleArtifactDetailRecord {
  const record = requireRecord(value, path);
  const kind = requireString(record.kind, `${path}.kind`);

  if (kind === 'architecture-manifest') {
    return {
      kind,
      source_path: requireString(record.source_path, `${path}.source_path`),
      input_shape: parseNumberArray(record.input_shape, `${path}.input_shape`),
      output_shape: parseNumberArray(record.output_shape, `${path}.output_shape`),
      layers: requireArray(record.layers, `${path}.layers`).map((layer, index) =>
        parseLayerSummary(layer, `${path}.layers.${index}`),
      ),
    };
  }

  if (kind === 'benchmark-manifest') {
    const detail: BenchmarkManifestDetailRecord = {
      kind,
      source_path: requireString(record.source_path, `${path}.source_path`),
      id: requireString(record.id, `${path}.id`),
    };
    if (record.outcome_space !== undefined) {
      detail.outcome_space = parseOutcomeSpace(record.outcome_space, `${path}.outcome_space`);
    }
    if (record.outcome_sequence !== undefined) {
      detail.outcome_sequence = parseOutcomeSequence(
        record.outcome_sequence,
        `${path}.outcome_sequence`,
      );
    }
    if (record.scale_parameter !== undefined) {
      detail.scale_parameter = parseScaleParameter(
        record.scale_parameter,
        `${path}.scale_parameter`,
      );
    }
    if (record.observation_ids !== undefined) {
      detail.observation_ids = parseStringArray(record.observation_ids, `${path}.observation_ids`);
    }
    if (record.latent_factor_declaration !== undefined) {
      detail.latent_factor_declaration = parseReferenceRecord(
        record.latent_factor_declaration,
        `${path}.latent_factor_declaration`,
      );
    }
    if (record.complexity_coordinate !== undefined) {
      detail.complexity_coordinate = requireString(
        record.complexity_coordinate,
        `${path}.complexity_coordinate`,
      );
    }
    return detail;
  }

  if (kind === 'measurement') {
    return {
      kind,
      source_path: requireString(record.source_path, `${path}.source_path`),
      id: requireString(record.id, `${path}.id`),
      benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
      observation_id: requireString(record.observation_id, `${path}.observation_id`),
      outcome_space: parseOutcomeSpace(record.outcome_space, `${path}.outcome_space`),
      accepted_event: parseAcceptedEvent(record.accepted_event, `${path}.accepted_event`),
      probability_measure: parseProbabilityMeasure(
        record.probability_measure,
        `${path}.probability_measure`,
      ),
    };
  }

  if (kind === 'latent-factor-declaration') {
    return {
      kind,
      source_path: requireString(record.source_path, `${path}.source_path`),
      id: requireString(record.id, `${path}.id`),
      construction_factors: requireArray(
        record.construction_factors,
        `${path}.construction_factors`,
      ).map((factor, index) => parseLatentFactor(factor, `${path}.construction_factors.${index}`)),
      sample_factors: requireArray(record.sample_factors, `${path}.sample_factors`).map(
        (factor, index) => parseSampleLatentFactor(factor, `${path}.sample_factors.${index}`),
      ),
      complexity_projections: requireArray(
        record.complexity_projections,
        `${path}.complexity_projections`,
      ).map((projection, index) =>
        parseComplexityProjection(projection, `${path}.complexity_projections.${index}`),
      ),
      resolution_requirements: requireArray(
        record.resolution_requirements,
        `${path}.resolution_requirements`,
      ).map((requirement, index) =>
        parseResolutionRequirement(requirement, `${path}.resolution_requirements.${index}`),
      ),
    };
  }

  if (kind === 'materialization-declaration') {
    const detail: MaterializationDeclarationDetailRecord = {
      kind,
      source_path: requireString(record.source_path, `${path}.source_path`),
      id: requireString(record.id, `${path}.id`),
      benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
      requirements: requireArray(record.requirements, `${path}.requirements`).map(
        (requirement, index) =>
          parseLinearResolutionRequirement(requirement, `${path}.requirements.${index}`),
      ),
    };
    if (record.latent_factor_declaration !== undefined) {
      detail.latent_factor_declaration = parseReferenceRecord(
        record.latent_factor_declaration,
        `${path}.latent_factor_declaration`,
      );
    }
    if (record.layout !== undefined) {
      detail.layout = requireRecord(record.layout, `${path}.layout`);
    }
    return detail;
  }

  if (kind === 'materialization-plan') {
    const detail: MaterializationPlanDetailRecord = {
      kind,
      source_path: requireString(record.source_path, `${path}.source_path`),
      id: requireString(record.id, `${path}.id`),
      benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
      materialization_declaration: parseReferenceRecord(
        record.materialization_declaration,
        `${path}.materialization_declaration`,
      ),
      scale_assignment: parseAxisAssignment(record.scale_assignment, `${path}.scale_assignment`),
      complexity_assignment: parseAxisAssignment(
        record.complexity_assignment,
        `${path}.complexity_assignment`,
      ),
      resolution_assignment: parseAxisAssignment(
        record.resolution_assignment,
        `${path}.resolution_assignment`,
      ),
      seed: requireNumber(record.seed, `${path}.seed`),
    };
    if (record.latent_factor_declaration !== undefined) {
      detail.latent_factor_declaration = parseReferenceRecord(
        record.latent_factor_declaration,
        `${path}.latent_factor_declaration`,
      );
    }
    return detail;
  }

  if (kind === 'observation-formation-declaration') {
    return {
      kind,
      source_path: requireString(record.source_path, `${path}.source_path`),
      id: requireString(record.id, `${path}.id`),
      benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
      interpreter: requireString(record.interpreter, `${path}.interpreter`),
      output_field: parseOutputField(record.output_field, `${path}.output_field`),
      slot_composition: parseSlotComposition(record.slot_composition, `${path}.slot_composition`),
      component_count: requireNumber(record.component_count, `${path}.component_count`),
      mark_count: requireNumber(record.mark_count, `${path}.mark_count`),
      components: requireArray(record.components, `${path}.components`).map((component, index) =>
        parseObservationComponent(component, `${path}.components.${index}`),
      ),
    };
  }

  if (kind === 'observation-showcase') {
    return {
      kind,
      source_path: requireString(record.source_path, `${path}.source_path`),
      id: requireString(record.id, `${path}.id`),
      benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
      formation_declaration: parseReferenceRecord(
        record.formation_declaration,
        `${path}.formation_declaration`,
      ),
      materialization_declaration: parseReferenceRecord(
        record.materialization_declaration,
        `${path}.materialization_declaration`,
      ),
      samples: requireArray(record.samples, `${path}.samples`).map((sample, index) =>
        parseObservationShowcaseSample(sample, `${path}.samples.${index}`),
      ),
    };
  }

  throw new ConsoleArtifactDetailError(`${path}.kind: unsupported artifact detail kind`);
}

function parseLayerSummary(value: unknown, path: string): LayerSummaryRecord {
  const record = requireRecord(value, path);
  const layer: LayerSummaryRecord = {
    kind: requireString(record.kind, `${path}.kind`),
  };
  if (record.parameters !== undefined) {
    layer.parameters = parseParameters(record.parameters, `${path}.parameters`);
  }
  return layer;
}

function parseOutcomeSpace(value: unknown, path: string): OutcomeSpaceSummaryRecord {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    outcomes: requireArray(record.outcomes, `${path}.outcomes`).map((outcome, index) => {
      const outcomeRecord = requireRecord(outcome, `${path}.outcomes.${index}`);
      return { id: requireString(outcomeRecord.id, `${path}.outcomes.${index}.id`) };
    }),
  };
}

function parseOutcomeSequence(value: unknown, path: string): OutcomeSequenceSummaryRecord {
  const record = requireRecord(value, path);
  return {
    atom_count: requireNumber(record.atom_count, `${path}.atom_count`),
    atom_name: requireString(record.atom_name, `${path}.atom_name`),
    length_parameter: requireString(record.length_parameter, `${path}.length_parameter`),
  };
}

function parseScaleParameter(value: unknown, path: string): ScaleParameterSummaryRecord {
  const record = requireRecord(value, path);
  const parameter: ScaleParameterSummaryRecord = {
    symbol: requireString(record.symbol, `${path}.symbol`),
    minimum: requireNumber(record.minimum, `${path}.minimum`),
  };
  if (record.description !== undefined) {
    parameter.description = requireString(record.description, `${path}.description`);
  }
  return parameter;
}

function parseAcceptedEvent(
  value: unknown,
  path: string,
): MeasurementDetailRecord['accepted_event'] {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    outcome_space_id: requireString(record.outcome_space_id, `${path}.outcome_space_id`),
    outcomes: parseStringArray(record.outcomes, `${path}.outcomes`),
  };
}

function parseProbabilityMeasure(
  value: unknown,
  path: string,
): MeasurementDetailRecord['probability_measure'] {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    outcome_space_id: requireString(record.outcome_space_id, `${path}.outcome_space_id`),
    probabilities: requireArray(record.probabilities, `${path}.probabilities`).map(
      (probability, index) => parseProbability(probability, `${path}.probabilities.${index}`),
    ),
  };
}

function parseProbability(value: unknown, path: string): ProbabilitySummaryRecord {
  const record = requireRecord(value, path);
  return {
    outcome_id: requireString(record.outcome_id, `${path}.outcome_id`),
    probability: requireNumber(record.probability, `${path}.probability`),
  };
}

function parseLatentFactor(value: unknown, path: string): LatentFactorSummaryRecord {
  const record = requireRecord(value, path);
  const factor: LatentFactorSummaryRecord = {
    name: requireString(record.name, `${path}.name`),
    degree_measure: parseDegreeMeasure(record.degree_measure, `${path}.degree_measure`),
  };
  if (record.description !== undefined) {
    factor.description = requireString(record.description, `${path}.description`);
  }
  return factor;
}

function parseSampleLatentFactor(value: unknown, path: string): SampleLatentFactorSummaryRecord {
  const record = requireRecord(value, path);
  const factor: SampleLatentFactorSummaryRecord = {
    ...parseLatentFactor(value, path),
    role: requireString(record.role, `${path}.role`),
  };
  if (record.multiplicity !== undefined) {
    factor.multiplicity = requireNumber(record.multiplicity, `${path}.multiplicity`);
  }
  return factor;
}

function parseDegreeMeasure(value: unknown, path: string): DegreeMeasureSummaryRecord {
  const record = requireRecord(value, path);
  const measure: DegreeMeasureSummaryRecord = {
    kind: requireString(record.kind, `${path}.kind`),
    count: requireNumber(record.count, `${path}.count`),
  };
  if (record.domain_size !== undefined) {
    measure.domain_size = requireNumber(record.domain_size, `${path}.domain_size`);
  }
  return measure;
}

function parseComplexityProjection(value: unknown, path: string): ComplexityProjectionSummaryRecord {
  const record = requireRecord(value, path);
  const projection: ComplexityProjectionSummaryRecord = {
    name: requireString(record.name, `${path}.name`),
    coordinate: requireString(record.coordinate, `${path}.coordinate`),
    included_roles: parseStringArray(record.included_roles, `${path}.included_roles`),
  };
  if (record.description !== undefined) {
    projection.description = requireString(record.description, `${path}.description`);
  }
  return projection;
}

function parseResolutionRequirement(value: unknown, path: string): ResolutionRequirementSummaryRecord {
  const record = requireRecord(value, path);
  const requirement: ResolutionRequirementSummaryRecord = {
    name: requireString(record.name, `${path}.name`),
    resolution_axis: requireString(record.resolution_axis, `${path}.resolution_axis`),
    content_coordinate: requireString(record.content_coordinate, `${path}.content_coordinate`),
    content_complexity: requireNumber(record.content_complexity, `${path}.content_complexity`),
    minimum_resolution: requireNumber(record.minimum_resolution, `${path}.minimum_resolution`),
    basis: requireString(record.basis, `${path}.basis`),
  };
  if (record.description !== undefined) {
    requirement.description = requireString(record.description, `${path}.description`);
  }
  return requirement;
}

function parseLinearResolutionRequirement(
  value: unknown,
  path: string,
): LinearResolutionRequirementSummaryRecord {
  const record = requireRecord(value, path);
  const requirement: LinearResolutionRequirementSummaryRecord = {
    name: requireString(record.name, `${path}.name`),
    source_axis: requireString(record.source_axis, `${path}.source_axis`),
    resolution_axis: requireString(record.resolution_axis, `${path}.resolution_axis`),
    coefficient: requireNumber(record.coefficient, `${path}.coefficient`),
    basis: requireString(record.basis, `${path}.basis`),
  };
  if (record.intercept !== undefined) {
    requirement.intercept = requireNumber(record.intercept, `${path}.intercept`);
  }
  if (record.minimum !== undefined) {
    requirement.minimum = requireNumber(record.minimum, `${path}.minimum`);
  }
  if (record.description !== undefined) {
    requirement.description = requireString(record.description, `${path}.description`);
  }
  return requirement;
}

function parseAxisAssignment(value: unknown, path: string): AxisAssignmentSummaryRecord {
  const record = requireRecord(value, path);
  return {
    values: requireArray(record.values, `${path}.values`).map((item, index) => {
      const assignment = requireRecord(item, `${path}.values.${index}`);
      return {
        axis: requireString(assignment.axis, `${path}.values.${index}.axis`),
        value: requireNumber(assignment.value, `${path}.values.${index}.value`),
      };
    }),
  };
}

function parseOutputField(value: unknown, path: string): OutputFieldSummaryRecord {
  const record = requireRecord(value, path);
  return {
    channel_count: requireNumber(record.channel_count, `${path}.channel_count`),
    resolution_axis: requireString(record.resolution_axis, `${path}.resolution_axis`),
  };
}

function parseSlotComposition(value: unknown, path: string): SlotCompositionSummaryRecord {
  const record = requireRecord(value, path);
  return {
    count_axis: requireString(record.count_axis, `${path}.count_axis`),
    resolution_axis: requireString(record.resolution_axis, `${path}.resolution_axis`),
    slot_axis: requireString(record.slot_axis, `${path}.slot_axis`),
  };
}

function parseObservationComponent(value: unknown, path: string): ObservationComponentSummaryRecord {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    marks: requireArray(record.marks, `${path}.marks`).map((mark, index) =>
      parseObservationMark(mark, `${path}.marks.${index}`),
    ),
  };
}

function parseObservationMark(value: unknown, path: string): ObservationMarkSummaryRecord {
  const record = requireRecord(value, path);
  const mark: ObservationMarkSummaryRecord = {
    kind: requireString(record.kind, `${path}.kind`),
    channel: requireNumber(record.channel, `${path}.channel`),
    degree: requireNumber(record.degree, `${path}.degree`),
    control_points: requireArray(record.control_points, `${path}.control_points`).map(
      (point, index) => parseNumberArray(point, `${path}.control_points.${index}`),
    ),
    width: requireNumber(record.width, `${path}.width`),
  };
  if (record.value !== undefined) {
    mark.value = requireNumber(record.value, `${path}.value`);
  }
  return mark;
}

function parseObservationShowcaseSample(
  value: unknown,
  path: string,
): ObservationShowcaseSampleSummaryRecord {
  const record = requireRecord(value, path);
  const sample: ObservationShowcaseSampleSummaryRecord = {
    id: requireString(record.id, `${path}.id`),
    label: requireString(record.label, `${path}.label`),
    sample_index: requireNumber(record.sample_index, `${path}.sample_index`),
    scale_assignment: parseAxisAssignment(record.scale_assignment, `${path}.scale_assignment`),
    complexity_assignment: parseAxisAssignment(
      record.complexity_assignment,
      `${path}.complexity_assignment`,
    ),
    seed: requireNumber(record.seed, `${path}.seed`),
    component_sequence: parseNumberArray(record.component_sequence, `${path}.component_sequence`),
  };
  if (record.outcome_id !== undefined) {
    sample.outcome_id = requireString(record.outcome_id, `${path}.outcome_id`);
  }
  return sample;
}

function parseReferenceRecord(value: unknown, path: string): ArtifactReferenceRecord {
  const record = requireRecord(value, path);
  const reference: ArtifactReferenceRecord = {
    kind: requireString(record.kind, `${path}.kind`),
  };
  if (record.protocol_id !== undefined) {
    reference.protocol_id = requireString(record.protocol_id, `${path}.protocol_id`);
  }
  if (record.content_digest !== undefined) {
    reference.content_digest = requireString(record.content_digest, `${path}.content_digest`);
  }
  if (record.record_digest !== undefined) {
    reference.record_digest = requireString(record.record_digest, `${path}.record_digest`);
  }
  if (record.external_uri !== undefined) {
    reference.external_uri = requireString(record.external_uri, `${path}.external_uri`);
  }
  return reference;
}

function parseParameters(value: unknown, path: string): Record<string, string | number | boolean> {
  const record = requireRecord(value, path);
  const parameters: Record<string, string | number | boolean> = {};
  for (const [key, parameterValue] of Object.entries(record)) {
    if (
      typeof parameterValue !== 'string' &&
      typeof parameterValue !== 'number' &&
      typeof parameterValue !== 'boolean'
    ) {
      throw new ConsoleArtifactDetailError(`${path}.${key}: expected scalar parameter`);
    }
    parameters[key] = parameterValue;
  }
  return parameters;
}

function parseNumberArray(value: unknown, path: string): number[] {
  return requireArray(value, path).map((item, index) => requireNumber(item, `${path}.${index}`));
}

function parseStringArray(value: unknown, path: string): string[] {
  return requireArray(value, path).map((item, index) => requireString(item, `${path}.${index}`));
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ConsoleArtifactDetailError(`${path}: expected record`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new ConsoleArtifactDetailError(`${path}: expected array`);
  }
  return value;
}

function requireNumber(value: unknown, path: string): number {
  if (typeof value !== 'number') {
    throw new ConsoleArtifactDetailError(`${path}: expected number`);
  }
  return value;
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string') {
    throw new ConsoleArtifactDetailError(`${path}: expected string`);
  }
  return value;
}
