import type { ArtifactReferenceRecord, ConsoleArtifactIndexEntryRecord } from './artifactIndex';
import { artifactKey } from './artifactBrowserModel.ts';
import { requireArray, requireRecord, requireString } from './transport.ts';

export type ArchitectureManifestDetailRecord = ArtifactDetailBase & {
  kind: 'architecture-manifest';
  input_shape: number[];
  output_shape: number[];
  layers: LayerSummaryRecord[];
  architecture_graph: ArchitectureGraphRecord;
};

export type BenchmarkManifestDetailRecord = ArtifactDetailBase & {
  kind: 'benchmark-manifest';
  id: string;
  outcome_space?: OutcomeSpaceSummaryRecord;
  outcome_sequence?: OutcomeSequenceSummaryRecord;
  scale_parameter?: ScaleParameterSummaryRecord;
  observation_ids?: string[];
  latent_factor_declaration?: ArtifactReferenceRecord;
  complexity_coordinate?: string;
};

export type MeasurementDetailRecord = ArtifactDetailBase & {
  kind: 'measurement';
  id: string;
  benchmark_id: string;
  observation_id: string;
  outcome_space: OutcomeSpaceSummaryRecord;
  accepted_event: {
    id: string;
    outcome_space_id: string;
    outcomes: string[];
  };
  probability_measure: {
    id: string;
    outcome_space_id: string;
    probabilities: ProbabilitySummaryRecord[];
  };
};

export type LatentFactorDeclarationDetailRecord = ArtifactDetailBase & {
  kind: 'latent-factor-declaration';
  id: string;
  construction_factors: LatentFactorSummaryRecord[];
  sample_factors: SampleLatentFactorSummaryRecord[];
  complexity_projections: ComplexityProjectionSummaryRecord[];
  resolution_requirements: ResolutionRequirementSummaryRecord[];
};

export type MaterializationDeclarationDetailRecord = ArtifactDetailBase & {
  kind: 'materialization-declaration';
  id: string;
  benchmark_id: string;
  requirements: LinearResolutionRequirementSummaryRecord[];
  latent_factor_declaration?: ArtifactReferenceRecord;
  layout?: Record<string, unknown>;
};

export type MaterializationPlanDetailRecord = ArtifactDetailBase & {
  kind: 'materialization-plan';
  id: string;
  benchmark_id: string;
  materialization_declaration: ArtifactReferenceRecord;
  scale_assignment: AxisAssignmentSummaryRecord;
  complexity_assignment: AxisAssignmentSummaryRecord;
  resolution_assignment: AxisAssignmentSummaryRecord;
  seed: number;
  latent_factor_declaration?: ArtifactReferenceRecord;
};

export type ObservationFormationDeclarationDetailRecord = ArtifactDetailBase & {
  kind: 'observation-formation-declaration';
  id: string;
  benchmark_id: string;
  interpreter: string;
  output_field: OutputFieldSummaryRecord;
  slot_composition: SlotCompositionSummaryRecord;
  component_count: number;
  mark_count: number;
  components: ObservationComponentSummaryRecord[];
};

export type ObservationShowcaseDetailRecord = ArtifactDetailBase & {
  kind: 'observation-showcase';
  id: string;
  benchmark_id: string;
  formation_declaration: ArtifactReferenceRecord;
  materialization_declaration: ArtifactReferenceRecord;
  samples: ObservationShowcaseSampleSummaryRecord[];
};

export type ConsoleArtifactDetailRecord =
  | ArchitectureManifestDetailRecord
  | BenchmarkManifestDetailRecord
  | LatentFactorDeclarationDetailRecord
  | MaterializationDeclarationDetailRecord
  | MaterializationPlanDetailRecord
  | MeasurementDetailRecord
  | ObservationFormationDeclarationDetailRecord
  | ObservationShowcaseDetailRecord;

export type ConsoleArtifactDetailMap = ReadonlyMap<string, ConsoleArtifactDetailRecord>;

type ArtifactDetailBase = {
  source_path: string;
};

type LayerSummaryRecord = {
  kind: string;
  parameters?: Record<string, string | number | boolean>;
};

type ArchitectureGraphRecord = {
  nodes: { id: string; component: LayerSummaryRecord }[];
  edges: { source_node_id: string; target_node_id: string; kind: string }[];
  input_node_ids: string[];
  output_node_ids: string[];
};

type OutcomeSpaceSummaryRecord = { id: string; outcomes: { id: string }[] };
type OutcomeSequenceSummaryRecord = { atom_count: number; atom_name: string; length_parameter: string };
type ScaleParameterSummaryRecord = { symbol: string; minimum: number; description?: string };
type ProbabilitySummaryRecord = { outcome_id: string; probability: number };
type DegreeMeasureSummaryRecord = { kind: string; count: number; domain_size?: number };
type LatentFactorSummaryRecord = { name: string; degree_measure: DegreeMeasureSummaryRecord; description?: string };
type SampleLatentFactorSummaryRecord = LatentFactorSummaryRecord & { role: string; multiplicity?: number };
type ComplexityProjectionSummaryRecord = { name: string; coordinate: string; included_roles: string[]; description?: string };
type ResolutionRequirementSummaryRecord = {
  name: string;
  resolution_axis: string;
  content_coordinate: string;
  content_complexity: number;
  minimum_resolution: number;
  basis: string;
  description?: string;
};
type LinearResolutionRequirementSummaryRecord = {
  name: string;
  source_axis: string;
  resolution_axis: string;
  coefficient: number;
  intercept?: number;
  minimum?: number;
  basis: string;
  description?: string;
};
type AxisAssignmentSummaryRecord = { values: { axis: string; value: number }[] };
type OutputFieldSummaryRecord = { channel_count: number; resolution_axis: string };
type SlotCompositionSummaryRecord = { count_axis: string; resolution_axis: string; slot_axis: string };
type ObservationComponentSummaryRecord = { id: string; marks: ObservationMarkSummaryRecord[] };
type ObservationMarkSummaryRecord = {
  kind: string;
  channel: number;
  degree: number;
  control_points: number[][];
  width: number;
  value?: number;
};
type ObservationShowcaseSampleSummaryRecord = {
  id: string;
  label: string;
  sample_index: number;
  scale_assignment: AxisAssignmentSummaryRecord;
  complexity_assignment: AxisAssignmentSummaryRecord;
  seed: number;
  component_sequence: number[];
  outcome_id?: string;
};

const supportedDetailKinds = new Set([
  'architecture-manifest',
  'benchmark-manifest',
  'latent-factor-declaration',
  'materialization-declaration',
  'materialization-plan',
  'measurement',
  'observation-formation-declaration',
  'observation-showcase',
]);

export class ConsoleArtifactDetailError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConsoleArtifactDetailError';
  }
}

const error = (message: string) => new ConsoleArtifactDetailError(message);

export function artifactDetailKey(detail: ConsoleArtifactDetailRecord): string {
  return `${detail.kind}:${detail.source_path}`;
}

export function artifactDetailKeyForEntry(artifact: ConsoleArtifactIndexEntryRecord): string {
  return `${artifact.kind}:${artifactKey(artifact)}`;
}

export function detailForArtifact(
  details: ConsoleArtifactDetailMap | undefined,
  artifact: ConsoleArtifactIndexEntryRecord,
): ConsoleArtifactDetailRecord | undefined {
  return details?.get(artifactDetailKeyForEntry(artifact));
}

export function parseConsoleArtifactDetailRecords(value: unknown): ConsoleArtifactDetailMap {
  const records = requireArray(value, 'artifact details', error).map((detail, index) =>
    parseDetailRecord(detail, `artifact details.${index}`),
  );
  return new Map(records.map((detail) => [artifactDetailKey(detail), detail]));
}

function parseDetailRecord(value: unknown, path: string): ConsoleArtifactDetailRecord {
  const record = requireRecord(value, path, error);
  const kind = requireString(record.kind, `${path}.kind`, error);
  if (!supportedDetailKinds.has(kind)) {
    throw error(`${path}.kind: unsupported artifact detail kind`);
  }
  requireString(record.source_path, `${path}.source_path`, error);
  return record as unknown as ConsoleArtifactDetailRecord;
}
