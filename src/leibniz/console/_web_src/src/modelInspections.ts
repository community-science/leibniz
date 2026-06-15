import type { ArtifactReferenceRecord } from './artifactReferences.ts';
import { requireArray, requireNumber, requireRecord } from './transport.ts';

export type ModelInspectionRecord = {
  id: string;
  source_path: string;
  program: ArtifactReferenceRecord;
  input_shape: number[];
  output_shape: number[];
  components: ModelInspectionComponentRecord[];
  cost_summary: ModelInspectionCostSummaryRecord;
  program_graph: ModelInspectionProgramGraphRecord;
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
  storage_bytes?: number;
};

export type ModelInspectionProgramGraphRecord = {
  nodes: ModelInspectionProgramGraphNodeRecord[];
  edges: ModelInspectionProgramGraphEdgeRecord[];
  inputs: ModelInspectionProgramGraphTensorContractRecord[];
  outputs: ModelInspectionProgramGraphTensorContractRecord[];
  contract_kind: string;
};

export type ModelInspectionProgramGraphNodeRecord = {
  id: string;
  kind: string;
  parameters?: Record<string, string | number | boolean>;
};

export type ModelInspectionProgramGraphEdgeRecord = {
  source_id: string;
  target_id: string;
  target_input_index: number;
};

export type ModelInspectionProgramGraphTensorContractRecord = {
  name: string;
  axes: Array<string | number>;
};

export type ModelInspectionCostSummaryRecord = {
  component_count: number;
  parameter_count?: number;
  storage_bytes?: number;
  inference_cost_measurement?: {
    execution_mode?: string;
    operation_stream_source?: string;
    operations_executed?: boolean;
    abstract_flops?: number;
  };
  inference_cost_sample_count?: number;
  training_cost_measurement?: {
    execution_mode?: string;
    operation_stream_source?: string;
    operations_executed?: boolean;
    abstract_flops?: number;
  };
  training_cost_sample_count?: number;
  unknown_parameter_components: number[];
  unknown_cost_components: number[];
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
  return {
    ...record,
    model_artifacts: artifactReferences(record.model_artifacts ?? [], `${path}.model_artifacts`),
    training_provenance: artifactReferences(
      record.training_provenance ?? [],
      `${path}.training_provenance`,
    ),
  } as unknown as ModelInspectionRecord;
}

function artifactReferences(value: unknown, path: string): ArtifactReferenceRecord[] {
  return requireArray(value, path, error).map((item, index) =>
    requireRecord(item, `${path}.${index}`, error) as ArtifactReferenceRecord,
  );
}
