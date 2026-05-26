export type BenchmarkTaskRecord = {
  kind: string;
  benchmark_id: string;
  label: string;
  source_path: string;
  scale_axis: string;
  complexity_axis: string;
  outcome_atom_name: string;
  outcome_atom_count: number;
  batches: GeneratedObservationBatchRecord[];
};

export type GeneratedObservationBatchRecord = {
  mode: string;
  label: string;
  scale: number;
  seed: number;
  sample_count: number;
  presentation: GeneratedObservationBatchPresentationRecord;
  samples: GeneratedObservationSampleRecord[];
};

export type GeneratedObservationBatchPresentationRecord = {
  sample_card_density: 'standard' | 'compact';
  aggregate_mode: boolean;
};

export type GeneratedObservationSampleRecord = {
  index: number;
  outcome_id: string;
  component_sequence: number[];
  complexity: number;
  field_shape: number[];
  image_data_url: string;
  materialization_plan: Record<string, unknown>;
  latent_coordinates: GeneratedLatentCoordinateRecord[];
};

export type GeneratedLatentCoordinateRecord = {
  name: string;
  role: string;
  degree_measure: Record<string, unknown>;
  multiplicity: number;
  values: unknown;
};

export class BenchmarkTaskTransportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'BenchmarkTaskTransportError';
  }
}

export function parseBenchmarkTaskRecords(value: unknown): BenchmarkTaskRecord[] {
  return requireArray(value, 'benchmark tasks').map((item, index) =>
    parseBenchmarkTask(item, `benchmark tasks.${index}`),
  );
}

function parseBenchmarkTask(value: unknown, path: string): BenchmarkTaskRecord {
  const record = requireRecord(value, path);
  const kind = requireString(record.kind, `${path}.kind`);
  if (kind.length === 0) {
    throw new BenchmarkTaskTransportError(`${path}.kind: expected nonempty string`);
  }
  const task: BenchmarkTaskRecord = {
    kind,
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
    label: requireString(record.label, `${path}.label`),
    source_path: requireString(record.source_path, `${path}.source_path`),
    scale_axis: requireString(record.scale_axis, `${path}.scale_axis`),
    complexity_axis: requireString(record.complexity_axis, `${path}.complexity_axis`),
    outcome_atom_name: requireString(record.outcome_atom_name, `${path}.outcome_atom_name`),
    outcome_atom_count: requireNumber(record.outcome_atom_count, `${path}.outcome_atom_count`),
    batches: requireArray(record.batches, `${path}.batches`).map((batch, index) =>
      parseBatch(batch, `${path}.batches.${index}`),
    ),
  };
  if (!Number.isInteger(task.outcome_atom_count) || task.outcome_atom_count < 1) {
    throw new BenchmarkTaskTransportError(`${path}.outcome_atom_count: expected positive integer`);
  }
  if (task.batches.length === 0) {
    throw new BenchmarkTaskTransportError(`${path}.batches: expected at least one batch`);
  }
  return task;
}

function parseBatch(value: unknown, path: string): GeneratedObservationBatchRecord {
  const record = requireRecord(value, path);
  const batch = {
    mode: requireString(record.mode, `${path}.mode`),
    label: requireString(record.label, `${path}.label`),
    scale: requireNumber(record.scale, `${path}.scale`),
    seed: requireNumber(record.seed, `${path}.seed`),
    sample_count: requireNumber(record.sample_count, `${path}.sample_count`),
    presentation: parsePresentation(record.presentation, `${path}.presentation`),
    samples: requireArray(record.samples, `${path}.samples`).map((sample, index) =>
      parseSample(sample, `${path}.samples.${index}`),
    ),
  };
  if (!Number.isInteger(batch.scale) || batch.scale < 1) {
    throw new BenchmarkTaskTransportError(`${path}.scale: expected positive integer`);
  }
  if (!Number.isInteger(batch.seed) || batch.seed < 0) {
    throw new BenchmarkTaskTransportError(`${path}.seed: expected nonnegative integer`);
  }
  if (!Number.isInteger(batch.sample_count) || batch.sample_count !== batch.samples.length) {
    throw new BenchmarkTaskTransportError(`${path}.sample_count: expected sample length`);
  }
  return batch;
}

function parsePresentation(
  value: unknown,
  path: string,
): GeneratedObservationBatchPresentationRecord {
  const record = requireRecord(value, path);
  return {
    sample_card_density: requireDensity(
      record.sample_card_density,
      `${path}.sample_card_density`,
    ),
    aggregate_mode: requireBoolean(record.aggregate_mode, `${path}.aggregate_mode`),
  };
}

function parseSample(value: unknown, path: string): GeneratedObservationSampleRecord {
  const record = requireRecord(value, path);
  const sample = {
    index: requireNumber(record.index, `${path}.index`),
    outcome_id: requireString(record.outcome_id, `${path}.outcome_id`),
    component_sequence: parseNumberArray(record.component_sequence, `${path}.component_sequence`),
    complexity: requireNumber(record.complexity, `${path}.complexity`),
    field_shape: parseNumberArray(record.field_shape, `${path}.field_shape`),
    image_data_url: requireString(record.image_data_url, `${path}.image_data_url`),
    materialization_plan: requireRecord(record.materialization_plan, `${path}.materialization_plan`),
    latent_coordinates: requireArray(record.latent_coordinates, `${path}.latent_coordinates`).map(
      (coordinate, index) => parseLatentCoordinate(coordinate, `${path}.latent_coordinates.${index}`),
    ),
  };
  if (!Number.isInteger(sample.index) || sample.index < 0) {
    throw new BenchmarkTaskTransportError(`${path}.index: expected nonnegative integer`);
  }
  if (sample.component_sequence.length === 0) {
    throw new BenchmarkTaskTransportError(`${path}.component_sequence: expected at least one item`);
  }
  if (sample.field_shape.length !== 3) {
    throw new BenchmarkTaskTransportError(`${path}.field_shape: expected channel-first shape`);
  }
  if (!sample.image_data_url.startsWith('data:image/png;base64,')) {
    throw new BenchmarkTaskTransportError(`${path}.image_data_url: expected PNG data URL`);
  }
  return sample;
}

function parseLatentCoordinate(value: unknown, path: string): GeneratedLatentCoordinateRecord {
  const record = requireRecord(value, path);
  return {
    name: requireString(record.name, `${path}.name`),
    role: requireString(record.role, `${path}.role`),
    degree_measure: requireRecord(record.degree_measure, `${path}.degree_measure`),
    multiplicity: requireNumber(record.multiplicity, `${path}.multiplicity`),
    values: record.values,
  };
}

function parseNumberArray(value: unknown, path: string): number[] {
  return requireArray(value, path).map((item, index) => requireNumber(item, `${path}.${index}`));
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new BenchmarkTaskTransportError(`${path}: expected record`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new BenchmarkTaskTransportError(`${path}: expected array`);
  }
  return value;
}

function requireNumber(value: unknown, path: string): number {
  if (typeof value !== 'number') {
    throw new BenchmarkTaskTransportError(`${path}: expected number`);
  }
  return value;
}

function requireBoolean(value: unknown, path: string): boolean {
  if (typeof value !== 'boolean') {
    throw new BenchmarkTaskTransportError(`${path}: expected boolean`);
  }
  return value;
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string') {
    throw new BenchmarkTaskTransportError(`${path}: expected string`);
  }
  return value;
}

function requireDensity(
  value: unknown,
  path: string,
): GeneratedObservationBatchPresentationRecord['sample_card_density'] {
  if (value !== 'standard' && value !== 'compact') {
    throw new BenchmarkTaskTransportError(`${path}: expected supported sample card density`);
  }
  return value;
}
