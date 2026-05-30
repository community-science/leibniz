import type { ArtifactReferenceRecord } from './artifactReferences.ts';
import { requireArray, requireNumber, requireRecord } from './transport.ts';

export type ModelInspectionRecord = {
  id: string;
  source_path: string;
  architecture: ArtifactReferenceRecord;
  input_shape: number[];
  output_shape: number[];
  components: ModelInspectionComponentRecord[];
  cost_summary: ModelInspectionCostSummaryRecord;
  architecture_trace: ModelInspectionTraceRecord;
  architecture_graph: ModelInspectionArchitectureGraphRecord;
  architecture_summary: ModelInspectionGraphSummaryRecord;
  node_evidence: ModelGraphNodeEvidenceRecord[];
  model_manifest?: ArtifactReferenceRecord;
  submission_package?: ArtifactReferenceRecord;
  benchmark_manifest?: ArtifactReferenceRecord;
  measurement_dataset?: ArtifactReferenceRecord;
  model_artifacts: ArtifactReferenceRecord[];
  training_provenance: ArtifactReferenceRecord[];
};

export type ModelGraphNodeEvidenceRecord = {
  node_path: string[];
  claim_kinds: string[];
  evidence_artifacts: ArtifactReferenceRecord[];
};

export type ModelInspectionComponentRecord = {
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
  component_count: number;
  parameter_count?: number;
  parameter_bytes?: number;
  inference_flops?: number;
  unknown_parameter_components: number[];
  unknown_flop_components: number[];
};

export type ModelInspectionGraphSummaryRecord = {
  component_count: number;
  edge_count: number;
  input_count: number;
  output_count: number;
  input_node_ids: string[];
  output_node_ids: string[];
  component_kinds: string[];
  unsupported_parameter_components: number[];
  unsupported_flop_components: number[];
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

const error = (message: string) => new ModelInspectionError(message);

export function parseModelInspectionRecords(value: unknown): ModelInspectionRecord[] {
  return requireArray(value, 'model inspections', error).map((item, index) =>
    parseModelInspectionRecord(item, `model inspections.${index}`),
  );
}

export function parseModelInspectionRecord(
  value: unknown,
  path: string,
): ModelInspectionRecord {
  const record = requireRecord(value, path, error);
  requireArray(record.components, `${path}.components`, error).forEach((component, index) => {
    const componentRecord = requireRecord(component, `${path}.components.${index}`, error);
    requireNumber(componentRecord.index, `${path}.components.${index}.index`, error);
  });
  return record as unknown as ModelInspectionRecord;
}
