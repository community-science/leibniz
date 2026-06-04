import { requireArray, requireNumber, requireRecord, requireString } from './transport.ts';

export type BenchmarkTaskRecord = {
  kind: string;
  benchmark_id: string;
  label: string;
  source_path: string;
  complexity_axis: string;
  outcome_atom_name: string;
  outcome_atom_count: number;
  code_surfaces: BenchmarkCodeSurfaceRecord[];
  batches: GeneratedObservationBatchRecord[];
};

export type BenchmarkCodeSurfaceRecord = {
  label: string;
  role: string;
  source_path: string;
  symbol: string;
  start_line: number;
  end_line: number;
  call_path: string[];
  code: string;
};

export type GeneratedObservationBatchRecord = {
  mode: string;
  label: string;
  component_count: number;
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

const error = (message: string) => new BenchmarkTaskTransportError(message);

export function parseBenchmarkTaskRecords(value: unknown): BenchmarkTaskRecord[] {
  return requireArray(value, 'benchmark tasks', error).map((item, index) =>
    parseBenchmarkTask(item, `benchmark tasks.${index}`),
  );
}

function parseBenchmarkTask(value: unknown, path: string): BenchmarkTaskRecord {
  const record = requireRecord(value, path, error);
  const batches = requireArray(record.batches, `${path}.batches`, error);
  const codeSurfaces = requireArray(record.code_surfaces ?? [], `${path}.code_surfaces`, error);
  batches.forEach((batch, index) => validateBatch(batch, `${path}.batches.${index}`));
  codeSurfaces.forEach((surface, index) =>
    validateCodeSurface(surface, `${path}.code_surfaces.${index}`),
  );
  return record as unknown as BenchmarkTaskRecord;
}

function validateBatch(value: unknown, path: string): void {
  const record = requireRecord(value, path, error);
  const sampleCount = requireNumber(record.sample_count, `${path}.sample_count`, error);
  const samples = requireArray(record.samples, `${path}.samples`, error);
  if (!Number.isInteger(sampleCount) || sampleCount !== samples.length) {
    throw error(`${path}.sample_count: expected sample length`);
  }
}

function validateCodeSurface(value: unknown, path: string): void {
  const record = requireRecord(value, path, error);
  requireString(record.label, `${path}.label`, error);
  requireString(record.role, `${path}.role`, error);
  requireString(record.source_path, `${path}.source_path`, error);
  requireString(record.symbol, `${path}.symbol`, error);
  requireString(record.code, `${path}.code`, error);
  const startLine = requireNumber(record.start_line, `${path}.start_line`, error);
  const endLine = requireNumber(record.end_line, `${path}.end_line`, error);
  if (!Number.isInteger(startLine) || startLine < 1) {
    throw error(`${path}.start_line: expected positive integer`);
  }
  if (!Number.isInteger(endLine) || endLine < startLine) {
    throw error(`${path}.end_line: expected integer after start_line`);
  }
  requireArray(record.call_path, `${path}.call_path`, error).forEach((entry, index) =>
    requireString(entry, `${path}.call_path.${index}`, error),
  );
}
