import { requireArray, requireNumber, requireRecord } from './transport.ts';

export type BenchmarkTaskRecord = {
  kind: string;
  benchmark_id: string;
  label: string;
  source_path: string;
  complexity_axis: string;
  outcome_atom_name: string;
  outcome_atom_count: number;
  batches: GeneratedObservationBatchRecord[];
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
  preview_crop: GeneratedObservationPreviewCropRecord;
  materialization_plan: Record<string, unknown>;
  latent_coordinates: GeneratedLatentCoordinateRecord[];
};

export type GeneratedObservationPreviewCropRecord = {
  left: number;
  top: number;
  size: number;
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
  batches.forEach((batch, index) => validateBatch(batch, `${path}.batches.${index}`));
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
