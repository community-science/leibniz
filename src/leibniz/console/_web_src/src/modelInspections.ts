import type { ArtifactReferenceRecord } from './artifactIndex.ts';

export type ModelInspectionRecord = {
  id: string;
  source_path: string;
  architecture: ArtifactReferenceRecord;
  input_shape: number[];
  output_shape: number[];
  layers: ModelInspectionLayerRecord[];
  cost_summary: ModelInspectionCostSummaryRecord;
  architecture_trace: ModelInspectionTraceRecord;
  architecture_graph: ModelInspectionArchitectureGraphRecord;
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
  operator?: Record<string, unknown>;
  parameter_count?: number;
  parameter_bytes?: number;
  inference_flops?: number;
};

export type ModelInspectionArchitectureGraphRecord = {
  nodes: ModelInspectionArchitectureGraphNodeRecord[];
  edges: ModelInspectionArchitectureGraphEdgeRecord[];
  input_node_ids: string[];
  output_node_ids: string[];
};

export type ModelInspectionArchitectureGraphNodeRecord = {
  id: string;
  component: {
    kind: string;
    parameters?: Record<string, string | number | boolean>;
  };
};

export type ModelInspectionArchitectureGraphEdgeRecord = {
  source_node_id: string;
  target_node_id: string;
  kind: string;
};

export type ModelInspectionCostSummaryRecord = {
  layer_count: number;
  parameter_count?: number;
  parameter_bytes?: number;
  inference_flops?: number;
  unknown_parameter_layers: number[];
  unknown_flop_layers: number[];
};

export type ModelInspectionTraceRecord = {
  input_shape: number[];
  output_shape: number[];
  stages: ModelInspectionTraceStageRecord[];
  program_effects: Record<string, unknown>[];
};

export type ModelInspectionTraceStageRecord = {
  index: number;
  kind: 'operator';
  syntax_alias: string;
  operator_kind: string;
  input_shape: number[];
  output_shape: number[];
  descriptor_axes: Record<string, string>;
  shape_law: string;
  cost_law: string;
  parameter_count?: number;
  inference_flops?: number;
};

export class ModelInspectionError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ModelInspectionError';
  }
}

export function parseModelInspectionRecords(value: unknown): ModelInspectionRecord[] {
  return requireArray(value, 'model inspections').map((item, index) =>
    parseModelInspectionRecord(item, `model inspections.${index}`),
  );
}

export function parseModelInspectionRecord(
  value: unknown,
  path: string,
): ModelInspectionRecord {
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
    architecture_trace: parseTrace(record.architecture_trace, `${path}.architecture_trace`),
    architecture_graph: parseArchitectureGraph(
      record.architecture_graph,
      `${path}.architecture_graph`,
    ),
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

function parseTrace(value: unknown, path: string): ModelInspectionTraceRecord {
  const record = requireRecord(value, path);
  return {
    input_shape: parseIntegerArray(record.input_shape, `${path}.input_shape`),
    output_shape: parseIntegerArray(record.output_shape, `${path}.output_shape`),
    stages: requireArray(record.stages, `${path}.stages`).map((stage, index) =>
      parseTraceStage(stage, `${path}.stages.${index}`),
    ),
    program_effects:
      record.program_effects === undefined
        ? []
        : requireArray(record.program_effects, `${path}.program_effects`).map((effect, index) =>
            requireRecord(effect, `${path}.program_effects.${index}`),
          ),
  };
}

function parseTraceStage(value: unknown, path: string): ModelInspectionTraceStageRecord {
  const record = requireRecord(value, path);
  const stage: ModelInspectionTraceStageRecord = {
    index: requireInteger(record.index, `${path}.index`),
    kind: requireLiteral(record.kind, `${path}.kind`, 'operator'),
    syntax_alias: requireString(record.syntax_alias, `${path}.syntax_alias`),
    operator_kind: requireString(record.operator_kind, `${path}.operator_kind`),
    input_shape: parseIntegerArray(record.input_shape, `${path}.input_shape`),
    output_shape: parseIntegerArray(record.output_shape, `${path}.output_shape`),
    descriptor_axes: parseStringRecord(record.descriptor_axes, `${path}.descriptor_axes`),
    shape_law: requireString(record.shape_law, `${path}.shape_law`),
    cost_law: requireString(record.cost_law, `${path}.cost_law`),
  };
  if (record.parameter_count !== undefined) {
    stage.parameter_count = requireInteger(record.parameter_count, `${path}.parameter_count`);
  }
  if (record.inference_flops !== undefined) {
    stage.inference_flops = requireInteger(record.inference_flops, `${path}.inference_flops`);
  }
  return stage;
}

function parseArchitectureGraph(
  value: unknown,
  path: string,
): ModelInspectionArchitectureGraphRecord {
  const record = requireRecord(value, path);
  return {
    nodes: requireArray(record.nodes, `${path}.nodes`).map((node, index) =>
      parseArchitectureGraphNode(node, `${path}.nodes.${index}`),
    ),
    edges: requireArray(record.edges, `${path}.edges`).map((edge, index) =>
      parseArchitectureGraphEdge(edge, `${path}.edges.${index}`),
    ),
    input_node_ids: parseStringArray(record.input_node_ids, `${path}.input_node_ids`),
    output_node_ids: parseStringArray(record.output_node_ids, `${path}.output_node_ids`),
  };
}

function parseArchitectureGraphNode(
  value: unknown,
  path: string,
): ModelInspectionArchitectureGraphNodeRecord {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    component: parseArchitectureGraphComponent(record.component, `${path}.component`),
  };
}

function parseArchitectureGraphComponent(
  value: unknown,
  path: string,
): ModelInspectionArchitectureGraphNodeRecord['component'] {
  const record = requireRecord(value, path);
  const component: ModelInspectionArchitectureGraphNodeRecord['component'] = {
    kind: requireString(record.kind, `${path}.kind`),
  };
  if (record.parameters !== undefined) {
    component.parameters = parseParameters(record.parameters, `${path}.parameters`);
  }
  return component;
}

function parseArchitectureGraphEdge(
  value: unknown,
  path: string,
): ModelInspectionArchitectureGraphEdgeRecord {
  const record = requireRecord(value, path);
  return {
    source_node_id: requireString(record.source_node_id, `${path}.source_node_id`),
    target_node_id: requireString(record.target_node_id, `${path}.target_node_id`),
    kind: requireString(record.kind, `${path}.kind`),
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
  if (record.operator !== undefined) {
    layer.operator = requireRecord(record.operator, `${path}.operator`);
  }
  if (record.parameter_count !== undefined) {
    layer.parameter_count = requireInteger(record.parameter_count, `${path}.parameter_count`);
  }
  if (record.parameter_bytes !== undefined) {
    layer.parameter_bytes = requireInteger(record.parameter_bytes, `${path}.parameter_bytes`);
  }
  if (record.inference_flops !== undefined) {
    layer.inference_flops = requireInteger(record.inference_flops, `${path}.inference_flops`);
  }
  return layer;
}

function parseParameters(
  value: unknown,
  path: string,
): Record<string, string | number | boolean> {
  const record = requireRecord(value, path);
  const parsed: Record<string, string | number | boolean> = {};
  for (const [key, item] of Object.entries(record)) {
    if (typeof item === 'string' || typeof item === 'number' || typeof item === 'boolean') {
      parsed[key] = item;
      continue;
    }
    throw new ModelInspectionError(`${path}.${key}: expected string, number, or boolean`);
  }
  return parsed;
}

function parseStringRecord(value: unknown, path: string): Record<string, string> {
  const record = requireRecord(value, path);
  return Object.fromEntries(
    Object.entries(record).map(([key, item]) => [key, requireString(item, `${path}.${key}`)]),
  );
}

function parseCostSummary(value: unknown, path: string): ModelInspectionCostSummaryRecord {
  const record = requireRecord(value, path);
  const summary: ModelInspectionCostSummaryRecord = {
    layer_count: requireInteger(record.layer_count, `${path}.layer_count`),
    unknown_parameter_layers: parseIntegerArray(
      record.unknown_parameter_layers,
      `${path}.unknown_parameter_layers`,
    ),
    unknown_flop_layers:
      record.unknown_flop_layers === undefined
        ? []
        : parseIntegerArray(record.unknown_flop_layers, `${path}.unknown_flop_layers`),
  };
  if (record.parameter_count !== undefined) {
    summary.parameter_count = requireInteger(record.parameter_count, `${path}.parameter_count`);
  }
  if (record.parameter_bytes !== undefined) {
    summary.parameter_bytes = requireInteger(record.parameter_bytes, `${path}.parameter_bytes`);
  }
  if (record.inference_flops !== undefined) {
    summary.inference_flops = requireInteger(record.inference_flops, `${path}.inference_flops`);
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

function parseStringArray(value: unknown, path: string): string[] {
  return requireArray(value, path).map((item, index) => requireString(item, `${path}.${index}`));
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

function requireLiteral<const Literal extends string>(
  value: unknown,
  path: string,
  expected: Literal,
): Literal {
  if (value !== expected) {
    throw new ModelInspectionError(`${path}: expected ${expected}`);
  }
  return expected;
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string') {
    throw new ModelInspectionError(`${path}: expected string`);
  }
  return value;
}
