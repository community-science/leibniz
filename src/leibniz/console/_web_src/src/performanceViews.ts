import type { ArtifactReferenceRecord } from './artifactIndex.ts';

export type PerformanceViewRecord = {
  id: string;
  source_path: string;
  manifest: PerformanceViewManifestRecord;
  measurement_dataset: MeasurementDatasetRecord;
  materialization_plans: MaterializationPlanRecord[];
  competence_integral_view: CompetenceIntegralViewRecord;
};

export type PerformanceViewManifestRecord = {
  id: string;
  benchmark_manifest: ArtifactReferenceRecord;
  materialization_declaration: ArtifactReferenceRecord;
  observation_formation_declaration: ArtifactReferenceRecord;
  view_id: string;
  complexity_axis: string;
  expected_complexities: number[];
  measurement_cases: PerformanceMeasurementCaseRecord[];
};

export type PerformanceMeasurementCaseRecord = {
  id: string;
  component_sequence: number[];
  accepted_outcome_sequence: number[];
  scale_assignment: AxisAssignmentRecord;
  complexity_assignment: AxisAssignmentRecord;
  seed: number;
  probabilities: PerformanceProbabilityMassRecord[];
};

export type PerformanceProbabilityMassRecord =
  | { outcome_id: string; probability: number; outcome_sequence?: never }
  | { outcome_sequence: number[]; probability: number; outcome_id?: never };

export type MeasurementDatasetRecord = {
  measurements: MeasurementRecord[];
};

export type MeasurementRecord = {
  benchmark_id: string;
  accepted_event: {
    id: string;
    outcome_space_id: string;
    outcomes: string[];
  };
  probability_measure: {
    id: string;
    outcome_space_id: string;
    probabilities: { outcome_id: string; probability: number }[];
  };
  raw_scoring_evidence: {
    id: string;
    observation_id: string;
    outcome_space_id: string;
    accepted_event_id: string;
    probability_measure_id: string;
    accepted_mass: number;
    negative_log_score: number | 'infinity';
  };
};

export type MaterializationPlanRecord = {
  id: string;
  benchmark_id: string;
  scale_assignment: AxisAssignmentRecord;
  complexity_assignment: AxisAssignmentRecord;
  resolution_assignment: AxisAssignmentRecord;
  seed: number;
};

export type CompetenceIntegralViewRecord = {
  id: string;
  source_dataset_digest: string;
  projection_rule: string;
  complexity_axis: string;
  expected_complexities: number[];
  entries: CompetenceIntegralEntryRecord[];
};

export type CompetenceIntegralEntryRecord = {
  benchmark_id: string;
  integral: number;
  coverage: number;
  observed_complexities: number[];
  missing_complexities: number[];
  points: CompetenceIntegralPointRecord[];
};

export type CompetenceIntegralPointRecord = {
  measurement_id: string;
  materialization_plan: ArtifactReferenceRecord;
  complexity: number;
  competence: number;
};

type AxisAssignmentRecord = {
  values: { axis: string; value: number }[];
};

export class PerformanceViewError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PerformanceViewError';
  }
}

export function parsePerformanceViewRecords(value: unknown): PerformanceViewRecord[] {
  return requireArray(value, 'performance views').map((item, index) =>
    parsePerformanceView(item, `performance views.${index}`),
  );
}

function parsePerformanceView(value: unknown, path: string): PerformanceViewRecord {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    source_path: requireString(record.source_path, `${path}.source_path`),
    manifest: parseManifest(record.manifest, `${path}.manifest`),
    measurement_dataset: parseMeasurementDataset(
      record.measurement_dataset,
      `${path}.measurement_dataset`,
    ),
    materialization_plans: requireArray(
      record.materialization_plans,
      `${path}.materialization_plans`,
    ).map((plan, index) => parseMaterializationPlan(plan, `${path}.materialization_plans.${index}`)),
    competence_integral_view: parseCompetenceIntegralView(
      record.competence_integral_view,
      `${path}.competence_integral_view`,
    ),
  };
}

function parseManifest(value: unknown, path: string): PerformanceViewManifestRecord {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    benchmark_manifest: parseReference(record.benchmark_manifest, `${path}.benchmark_manifest`),
    materialization_declaration: parseReference(
      record.materialization_declaration,
      `${path}.materialization_declaration`,
    ),
    observation_formation_declaration: parseReference(
      record.observation_formation_declaration,
      `${path}.observation_formation_declaration`,
    ),
    view_id: requireString(record.view_id, `${path}.view_id`),
    complexity_axis: requireString(record.complexity_axis, `${path}.complexity_axis`),
    expected_complexities: parseNumberArray(
      record.expected_complexities,
      `${path}.expected_complexities`,
    ),
    measurement_cases: requireArray(record.measurement_cases, `${path}.measurement_cases`).map(
      (measurementCase, index) =>
        parseMeasurementCase(measurementCase, `${path}.measurement_cases.${index}`),
    ),
  };
}

function parseMeasurementCase(value: unknown, path: string): PerformanceMeasurementCaseRecord {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    component_sequence: parseNumberArray(record.component_sequence, `${path}.component_sequence`),
    accepted_outcome_sequence: parseNumberArray(
      record.accepted_outcome_sequence,
      `${path}.accepted_outcome_sequence`,
    ),
    scale_assignment: parseAxisAssignment(record.scale_assignment, `${path}.scale_assignment`),
    complexity_assignment: parseAxisAssignment(
      record.complexity_assignment,
      `${path}.complexity_assignment`,
    ),
    seed: requireNumber(record.seed, `${path}.seed`),
    probabilities: requireArray(record.probabilities, `${path}.probabilities`).map(
      (probability, index) => parseProbabilityMass(probability, `${path}.probabilities.${index}`),
    ),
  };
}

function parseProbabilityMass(value: unknown, path: string): PerformanceProbabilityMassRecord {
  const record = requireRecord(value, path);
  const probability = requireNumber(record.probability, `${path}.probability`);
  const hasOutcomeId = record.outcome_id !== undefined;
  const hasOutcomeSequence = record.outcome_sequence !== undefined;
  if (hasOutcomeId === hasOutcomeSequence) {
    throw new PerformanceViewError(`${path}: expected exactly one outcome identity`);
  }
  if (hasOutcomeId) {
    return { outcome_id: requireString(record.outcome_id, `${path}.outcome_id`), probability };
  }
  return {
    outcome_sequence: parseNumberArray(record.outcome_sequence, `${path}.outcome_sequence`),
    probability,
  };
}

function parseMeasurementDataset(value: unknown, path: string): MeasurementDatasetRecord {
  const record = requireRecord(value, path);
  return {
    measurements: requireArray(record.measurements, `${path}.measurements`).map(
      (measurement, index) => parseMeasurement(measurement, `${path}.measurements.${index}`),
    ),
  };
}

function parseMeasurement(value: unknown, path: string): MeasurementRecord {
  const record = requireRecord(value, path);
  return {
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
    accepted_event: parseAcceptedEvent(record.accepted_event, `${path}.accepted_event`),
    probability_measure: parseProbabilityMeasure(
      record.probability_measure,
      `${path}.probability_measure`,
    ),
    raw_scoring_evidence: parseRawScoringEvidence(
      record.raw_scoring_evidence,
      `${path}.raw_scoring_evidence`,
    ),
  };
}

function parseAcceptedEvent(value: unknown, path: string): MeasurementRecord['accepted_event'] {
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
): MeasurementRecord['probability_measure'] {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    outcome_space_id: requireString(record.outcome_space_id, `${path}.outcome_space_id`),
    probabilities: requireArray(record.probabilities, `${path}.probabilities`).map(
      (probability, index) => {
        const probabilityRecord = requireRecord(probability, `${path}.probabilities.${index}`);
        return {
          outcome_id: requireString(
            probabilityRecord.outcome_id,
            `${path}.probabilities.${index}.outcome_id`,
          ),
          probability: requireNumber(
            probabilityRecord.probability,
            `${path}.probabilities.${index}.probability`,
          ),
        };
      },
    ),
  };
}

function parseRawScoringEvidence(
  value: unknown,
  path: string,
): MeasurementRecord['raw_scoring_evidence'] {
  const record = requireRecord(value, path);
  const negativeLogScore =
    record.negative_log_score === 'infinity'
      ? 'infinity'
      : requireNumber(record.negative_log_score, `${path}.negative_log_score`);
  return {
    id: requireString(record.id, `${path}.id`),
    observation_id: requireString(record.observation_id, `${path}.observation_id`),
    outcome_space_id: requireString(record.outcome_space_id, `${path}.outcome_space_id`),
    accepted_event_id: requireString(record.accepted_event_id, `${path}.accepted_event_id`),
    probability_measure_id: requireString(
      record.probability_measure_id,
      `${path}.probability_measure_id`,
    ),
    accepted_mass: requireNumber(record.accepted_mass, `${path}.accepted_mass`),
    negative_log_score: negativeLogScore,
  };
}

function parseMaterializationPlan(value: unknown, path: string): MaterializationPlanRecord {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
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
}

function parseCompetenceIntegralView(value: unknown, path: string): CompetenceIntegralViewRecord {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    source_dataset_digest: requireString(
      record.source_dataset_digest,
      `${path}.source_dataset_digest`,
    ),
    projection_rule: requireString(record.projection_rule, `${path}.projection_rule`),
    complexity_axis: requireString(record.complexity_axis, `${path}.complexity_axis`),
    expected_complexities: parseNumberArray(
      record.expected_complexities,
      `${path}.expected_complexities`,
    ),
    entries: requireArray(record.entries, `${path}.entries`).map((entry, index) =>
      parseCompetenceIntegralEntry(entry, `${path}.entries.${index}`),
    ),
  };
}

function parseCompetenceIntegralEntry(value: unknown, path: string): CompetenceIntegralEntryRecord {
  const record = requireRecord(value, path);
  return {
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
    integral: requireNumber(record.integral, `${path}.integral`),
    coverage: requireNumber(record.coverage, `${path}.coverage`),
    observed_complexities: parseNumberArray(
      record.observed_complexities,
      `${path}.observed_complexities`,
    ),
    missing_complexities: parseNumberArray(
      record.missing_complexities,
      `${path}.missing_complexities`,
    ),
    points: requireArray(record.points, `${path}.points`).map((point, index) =>
      parseCompetenceIntegralPoint(point, `${path}.points.${index}`),
    ),
  };
}

function parseCompetenceIntegralPoint(value: unknown, path: string): CompetenceIntegralPointRecord {
  const record = requireRecord(value, path);
  return {
    measurement_id: requireString(record.measurement_id, `${path}.measurement_id`),
    materialization_plan: parseReference(record.materialization_plan, `${path}.materialization_plan`),
    complexity: requireNumber(record.complexity, `${path}.complexity`),
    competence: requireNumber(record.competence, `${path}.competence`),
  };
}

function parseAxisAssignment(value: unknown, path: string): AxisAssignmentRecord {
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

function parseReference(value: unknown, path: string): ArtifactReferenceRecord {
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

function parseNumberArray(value: unknown, path: string): number[] {
  return requireArray(value, path).map((item, index) => requireNumber(item, `${path}.${index}`));
}

function parseStringArray(value: unknown, path: string): string[] {
  return requireArray(value, path).map((item, index) => requireString(item, `${path}.${index}`));
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new PerformanceViewError(`${path}: expected record`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new PerformanceViewError(`${path}: expected array`);
  }
  return value;
}

function requireNumber(value: unknown, path: string): number {
  if (typeof value !== 'number') {
    throw new PerformanceViewError(`${path}: expected number`);
  }
  return value;
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string') {
    throw new PerformanceViewError(`${path}: expected string`);
  }
  return value;
}
