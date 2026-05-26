export type ObservationInspectionRecord = {
  id: string;
  label: string;
  benchmark_id: string;
  formed_observation: ArtifactReferenceRecord;
  formation_declaration: ArtifactReferenceRecord;
  materialization_plan: ArtifactReferenceRecord;
  sample_index: number;
  component_sequence: number[];
  scale_assignment: AxisAssignmentRecord;
  complexity_assignment: AxisAssignmentRecord;
  resolution_assignment: AxisAssignmentRecord;
  field_shape: number[];
  field_digest: string;
  field_preview: FieldPreviewRecord;
  outcome_id?: string;
  showcase: {
    id: string;
    source_path: string;
  };
};

export type FieldPreviewRecord = {
  encoding: 'uint8-rle';
  shape: number[];
  runs: FieldPreviewRunRecord[];
};

export type FieldPreviewRunRecord = {
  value: number;
  count: number;
};

type ArtifactReferenceRecord = {
  kind: string;
  protocol_id?: string;
  content_digest?: string;
  record_digest?: string;
  external_uri?: string;
};

type AxisAssignmentRecord = {
  values: { axis: string; value: number }[];
};

export class ObservationInspectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ObservationInspectionError';
  }
}

export function parseObservationInspectionRecords(value: unknown): ObservationInspectionRecord[] {
  return requireArray(value, 'observation inspections').map((item, index) =>
    parseObservationInspection(item, `observation inspections.${index}`),
  );
}

export function decodeFieldPreview(preview: FieldPreviewRecord): Uint8ClampedArray {
  const expected = preview.shape.reduce((product, axis) => product * axis, 1);
  const values = new Uint8ClampedArray(expected);
  let offset = 0;
  for (const run of preview.runs) {
    values.fill(run.value, offset, offset + run.count);
    offset += run.count;
  }
  if (offset !== expected) {
    throw new ObservationInspectionError('field preview runs do not match shape');
  }
  return values;
}

function parseObservationInspection(value: unknown, path: string): ObservationInspectionRecord {
  const record = requireRecord(value, path);
  const inspection: ObservationInspectionRecord = {
    id: requireString(record.id, `${path}.id`),
    label: requireString(record.label, `${path}.label`),
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
    formed_observation: parseReference(record.formed_observation, `${path}.formed_observation`),
    formation_declaration: parseReference(
      record.formation_declaration,
      `${path}.formation_declaration`,
    ),
    materialization_plan: parseReference(record.materialization_plan, `${path}.materialization_plan`),
    sample_index: requireNumber(record.sample_index, `${path}.sample_index`),
    component_sequence: parseNumberArray(record.component_sequence, `${path}.component_sequence`),
    scale_assignment: parseAxisAssignment(record.scale_assignment, `${path}.scale_assignment`),
    complexity_assignment: parseAxisAssignment(
      record.complexity_assignment,
      `${path}.complexity_assignment`,
    ),
    resolution_assignment: parseAxisAssignment(
      record.resolution_assignment,
      `${path}.resolution_assignment`,
    ),
    field_shape: parseNumberArray(record.field_shape, `${path}.field_shape`),
    field_digest: requireString(record.field_digest, `${path}.field_digest`),
    field_preview: parseFieldPreview(record.field_preview, `${path}.field_preview`),
    showcase: parseShowcase(record.showcase, `${path}.showcase`),
  };
  if (record.outcome_id !== undefined) {
    inspection.outcome_id = requireString(record.outcome_id, `${path}.outcome_id`);
  }
  return inspection;
}

function parseShowcase(value: unknown, path: string): ObservationInspectionRecord['showcase'] {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    source_path: requireString(record.source_path, `${path}.source_path`),
  };
}

function parseFieldPreview(value: unknown, path: string): FieldPreviewRecord {
  const record = requireRecord(value, path);
  const encoding = requireLiteral(record.encoding, `${path}.encoding`, 'uint8-rle');
  const shape = parseNumberArray(record.shape, `${path}.shape`);
  const runs = requireArray(record.runs, `${path}.runs`).map((run, index) =>
    parseFieldPreviewRun(run, `${path}.runs.${index}`),
  );
  const expected = shape.reduce((product, axis) => product * axis, 1);
  const actual = runs.reduce((sum, run) => sum + run.count, 0);
  if (actual !== expected) {
    throw new ObservationInspectionError(`${path}.runs: length does not match shape`);
  }
  return { encoding, shape, runs };
}

function parseFieldPreviewRun(value: unknown, path: string): FieldPreviewRunRecord {
  const record = requireRecord(value, path);
  const run = {
    value: requireNumber(record.value, `${path}.value`),
    count: requireNumber(record.count, `${path}.count`),
  };
  if (!Number.isInteger(run.value) || run.value < 0 || run.value > 255) {
    throw new ObservationInspectionError(`${path}.value: expected byte`);
  }
  if (!Number.isInteger(run.count) || run.count < 1) {
    throw new ObservationInspectionError(`${path}.count: expected positive integer`);
  }
  return run;
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

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ObservationInspectionError(`${path}: expected record`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new ObservationInspectionError(`${path}: expected array`);
  }
  return value;
}

function requireNumber(value: unknown, path: string): number {
  if (typeof value !== 'number') {
    throw new ObservationInspectionError(`${path}: expected number`);
  }
  return value;
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string') {
    throw new ObservationInspectionError(`${path}: expected string`);
  }
  return value;
}

function requireLiteral<const Literal extends string>(
  value: unknown,
  path: string,
  expected: Literal,
): Literal {
  if (value !== expected) {
    throw new ObservationInspectionError(`${path}: expected ${expected}`);
  }
  return expected;
}
