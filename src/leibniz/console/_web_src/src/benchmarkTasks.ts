import { requireArray, requireNumber, requireRecord, requireString } from './transport.ts';
import {
  parseGenerationRequestOutcomeRecord,
  parseStateSpaceRegionRecord,
  type GenerationRequestOutcomeRecord,
  type StateSpaceRegionRecord,
} from './stateSpaceRecords.ts';

export type BenchmarkTaskRecord = {
  kind: string;
  benchmark_id: string;
  label: string;
  source_path: string;
  volume_axis: string;
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
  seed: number;
  sample_count: number;
  volume_window?: GeneratedVolumeWindowRecord;
  volumes?: number[];
  region?: StateSpaceRegionRecord;
  request_outcome?: GenerationRequestOutcomeRecord;
  presentation: GeneratedObservationBatchPresentationRecord;
  samples: GeneratedObservationSampleRecord[];
};

export type GeneratedVolumeWindowRecord = {
  measure_id: string;
  minimum: number;
  maximum: number;
};

export type GeneratedObservationBatchPresentationRecord = {
  sample_card_density: 'standard' | 'compact';
  aggregate_mode: boolean;
};

export type GeneratedObservationSampleRecord = {
  index: number;
  outcome_id: string;
  component_index?: number;
  region_component_index?: number;
  axis_coordinates?: Record<string, unknown>;
  available_outcome_ids?: string[];
  field_shape?: number[];
  image_data_url?: string;
  image_overlay?: GeneratedImageOverlayRecord;
  materialization_plan?: Record<string, unknown>;
  observable_state_id?: string;
  target_distribution?: GeneratedTargetProbabilityRecord[];
  latent_coordinates: GeneratedLatentCoordinateRecord[];
};

export type GeneratedImageOverlayRecord = {
  kind: 'grid-move-highlights';
  columns: number;
  rows: number;
  moves: GeneratedGridMoveHighlightRecord[];
};

export type GeneratedGridMoveHighlightRecord = {
  from: [number, number];
  target_probability?: number;
  to: [number, number];
};

export type GeneratedTargetProbabilityRecord = {
  outcome_id: string;
  probability: number;
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
  const volumeWindow = record.volume_window;
  if (volumeWindow !== undefined) {
    validateVolumeWindow(volumeWindow, `${path}.volume_window`);
  }
  if (record.region !== undefined) {
    parseStateSpaceRegionRecord(record.region, `${path}.region`, error);
  }
  if (record.request_outcome !== undefined) {
    parseGenerationRequestOutcomeRecord(record.request_outcome, `${path}.request_outcome`, error);
  }
  if (record.volumes !== undefined) {
    requireArray(record.volumes, `${path}.volumes`, error).forEach(
      (size, index) => {
        const value = requireNumber(size, `${path}.volumes.${index}`, error);
        if (!Number.isInteger(value) || value < 1) {
          throw error(`${path}.volumes.${index}: expected positive integer`);
        }
      },
    );
  }
  if (!Number.isInteger(sampleCount) || sampleCount !== samples.length) {
    throw error(`${path}.sample_count: expected sample length`);
  }
  samples.forEach((sample, index) => validateSample(sample, `${path}.samples.${index}`));
}

function validateSample(value: unknown, path: string): void {
  const record = requireRecord(value, path, error);
  if (record.region_component_index !== undefined) {
    const componentIndex = requireNumber(
      record.region_component_index,
      `${path}.region_component_index`,
      error,
    );
    if (!Number.isInteger(componentIndex) || componentIndex < 0) {
      throw error(`${path}.region_component_index: expected nonnegative integer`);
    }
  }
  if (record.axis_coordinates !== undefined) {
    requireRecord(record.axis_coordinates, `${path}.axis_coordinates`, error);
  }
}

function validateVolumeWindow(value: unknown, path: string): void {
  const record = requireRecord(value, path, error);
  requireString(record.measure_id, `${path}.measure_id`, error);
  requireNumber(record.minimum, `${path}.minimum`, error);
  requireNumber(record.maximum, `${path}.maximum`, error);
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
