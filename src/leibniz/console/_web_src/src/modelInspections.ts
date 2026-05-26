import type { ArtifactReferenceRecord } from './artifactIndex.ts';

export type ModelInspectionRecord = {
  id: string;
  source_path: string;
  architecture: ArtifactReferenceRecord;
  input_shape: number[];
  output_shape: number[];
  layers: ModelInspectionLayerRecord[];
  cost_summary: ModelInspectionCostSummaryRecord;
  model_manifest?: ArtifactReferenceRecord;
  submission_package?: ArtifactReferenceRecord;
  benchmark_manifest?: ArtifactReferenceRecord;
  measurement_dataset?: ArtifactReferenceRecord;
  model_artifacts: ArtifactReferenceRecord[];
  training_provenance: ArtifactReferenceRecord[];
};

export type ModelInspectionLayerRecord = {
  index: number;
  kind: string;
  parameters: Record<string, unknown>;
  input_shape?: number[];
  output_shape?: number[];
  parameter_count?: number;
};

export type ModelInspectionCostSummaryRecord = {
  layer_count: number;
  parameter_count?: number;
  unknown_parameter_layers: number[];
};

export class ModelInspectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelInspectionError';
  }
}

export function parseModelInspectionRecords(value: unknown): ModelInspectionRecord[] {
  return requireArray(value, 'model inspections').map((item, index) =>
    parseModelInspection(item, `model inspections.${index}`),
  );
}

function parseModelInspection(value: unknown, path: string): ModelInspectionRecord {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    source_path: requireString(record.source_path, `${path}.source_path`),
    architecture: parseReference(record.architecture, `${path}.architecture`),
    input_shape: parseIntegerArray(record.input_shape, `${path}.input_shape`),
    output_shape: parseIntegerArray(record.output_shape, `${path}.output_shape`),
    layers: requireArray(record.layers, `${path}.layers`).map((layer, index) =>
      parseLayer(layer, `${path}.layers.${index}`),
    ),
    cost_summary: parseCostSummary(record.cost_summary, `${path}.cost_summary`),
    model_manifest: parseOptionalReference(record.model_manifest, `${path}.model_manifest`),
    submission_package: parseOptionalReference(
      record.submission_package,
      `${path}.submission_package`,
    ),
    benchmark_manifest: parseOptionalReference(
      record.benchmark_manifest,
      `${path}.benchmark_manifest`,
    ),
    measurement_dataset: parseOptionalReference(
      record.measurement_dataset,
      `${path}.measurement_dataset`,
    ),
    model_artifacts: parseOptionalReferenceArray(record.model_artifacts, `${path}.model_artifacts`),
    training_provenance: parseOptionalReferenceArray(
      record.training_provenance,
      `${path}.training_provenance`,
    ),
  };
}

function parseLayer(value: unknown, path: string): ModelInspectionLayerRecord {
  const record = requireRecord(value, path);
  const layer: ModelInspectionLayerRecord = {
    index: requireInteger(record.index, `${path}.index`),
    kind: requireString(record.kind, `${path}.kind`),
    parameters: requireRecord(record.parameters, `${path}.parameters`),
  };
  if (record.input_shape !== undefined) {
    layer.input_shape = parseIntegerArray(record.input_shape, `${path}.input_shape`);
  }
  if (record.output_shape !== undefined) {
    layer.output_shape = parseIntegerArray(record.output_shape, `${path}.output_shape`);
  }
  if (record.parameter_count !== undefined) {
    layer.parameter_count = requireInteger(record.parameter_count, `${path}.parameter_count`);
  }
  return layer;
}

function parseCostSummary(value: unknown, path: string): ModelInspectionCostSummaryRecord {
  const record = requireRecord(value, path);
  const summary: ModelInspectionCostSummaryRecord = {
    layer_count: requireInteger(record.layer_count, `${path}.layer_count`),
    unknown_parameter_layers: parseIntegerArray(
      record.unknown_parameter_layers,
      `${path}.unknown_parameter_layers`,
    ),
  };
  if (record.parameter_count !== undefined) {
    summary.parameter_count = requireInteger(record.parameter_count, `${path}.parameter_count`);
  }
  return summary;
}

function parseOptionalReference(value: unknown, path: string): ArtifactReferenceRecord | undefined {
  if (value === undefined) {
    return undefined;
  }
  return parseReference(value, path);
}

function parseOptionalReferenceArray(value: unknown, path: string): ArtifactReferenceRecord[] {
  if (value === undefined) {
    return [];
  }
  return requireArray(value, path).map((reference, index) =>
    parseReference(reference, `${path}.${index}`),
  );
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

function parseIntegerArray(value: unknown, path: string): number[] {
  return requireArray(value, path).map((item, index) => requireInteger(item, `${path}.${index}`));
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ModelInspectionError(`${path}: expected record`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new ModelInspectionError(`${path}: expected array`);
  }
  return value;
}

function requireNumber(value: unknown, path: string): number {
  if (typeof value !== 'number') {
    throw new ModelInspectionError(`${path}: expected number`);
  }
  return value;
}

function requireInteger(value: unknown, path: string): number {
  const number = requireNumber(value, path);
  if (!Number.isInteger(number)) {
    throw new ModelInspectionError(`${path}: expected integer`);
  }
  return number;
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string') {
    throw new ModelInspectionError(`${path}: expected string`);
  }
  return value;
}
