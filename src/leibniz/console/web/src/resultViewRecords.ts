import {
  parseModelInspectionRecord,
  type ModelInspectionRecord,
} from './modelInspections.ts';
import {
  requireArray,
  requireLiteral,
  requireNumber,
  requireRecord,
  requireString,
  optionalNumber,
} from './transport.ts';
import {
  consoleProtocolFormats,
  consoleProtocolFormatVersions,
} from './protocolVocabulary.ts';
import {
  parseStateSpaceRegionRecord,
  type StateSpaceRegionRecord,
} from './stateSpaceRecords.ts';

const resultViewFormats = consoleProtocolFormats.resultViews;
const resultViewFormatVersion = consoleProtocolFormatVersions.resultView;

type ResultViewBaseRecord = {
  format_version: typeof resultViewFormatVersion;
  source_path: string;
  source_mtime_ms?: number;
  source_size_bytes?: number;
};

export type ResultViewRecord = BenchmarkResultViewRecord;

export type BenchmarkResultViewRecord = ResultViewBaseRecord & {
  format: typeof resultViewFormats.benchmarkResults;
  benchmark_results: BenchmarkResultRecord[];
};

export type BenchmarkResultRecord = {
  benchmark_id: string;
  volume_axis?: string;
  leaderboard: ModelResultRecord[];
  model_candidates: ModelResultRecord[];
  frontiers: Record<string, ModelResultRecord[]>;
  reference_curves?: ReferenceCurveRecord[];
  training_history: RunResultRecord[];
  plot_runs: RunResultRecord[];
  model_inspections: ModelInspectionRecord[];
};

export type ReferenceCurvePointRecord = {
  log2_volume: number;
  score: number;
  cost: number;
  metadata?: Record<string, unknown>;
};

export type ReferenceCurveRecord = {
  kind: 'oracle-cost-measurement-reference-v1';
  key: string;
  label: string;
  x_axis: string;
  y_axis: string;
  points: ReferenceCurvePointRecord[];
};

export type CompetencePointRecord = {
  log2_volume: number;
  log2_volume_minimum?: number;
  log2_volume_maximum?: number;
  score: number;
  sample_count?: number;
  run_ids: string[];
  competence_value_kind?: string;
  predictability_boundary?: number;
  time_points?: CompetenceTimePointRecord[];
};

export type CompetenceTimePointRecord = {
  time: number;
  bits: number;
  certified_epsilon?: number;
  evolution_scale?: number;
};

export type TrainingEstimateComparisonPointRecord = {
  log2_volume: number;
  log2_volume_minimum?: number;
  log2_volume_maximum?: number;
  status: 'matched' | 'accepted-only' | 'training-only';
  accepted_score?: number;
  training_score?: number;
  score_delta?: number;
  accepted_sample_count?: number;
  training_sample_count?: number;
};

export type TrainingEstimateComparisonRecord = {
  kind: 'training-vs-accepted-sampled-competence-v1';
  accepted_score: number;
  training_score: number;
  score_delta: number;
  accepted_sample_count: number;
  training_sample_count: number;
  point_count: number;
  matched_point_count: number;
  points: TrainingEstimateComparisonPointRecord[];
};

export type StateSpaceIntegralTermRecord = {
  kind: string;
  log2_volume_minimum: number;
  log2_volume_maximum: number;
  width_in_bits: number;
  contribution: number;
  representative_log2_volume?: number;
  sample_count?: number;
  confidence_half_width?: number;
  confidence_method_id?: string;
  region?: StateSpaceRegionRecord;
};

export type StateSpaceIntegralRecord = {
  kind: string;
  value: number;
  terms: StateSpaceIntegralTermRecord[];
};

export type CapabilityMapNodeRecord = {
  kind: 'partition-capability-node-v1';
  label: string;
  measure: number;
  sample_count: number;
  competence: number;
  confidence_half_width: number;
  region?: StateSpaceRegionRecord;
  children: CapabilityMapNodeRecord[];
};

export type CapabilityMapRefinementStepRecord = {
  kind: 'partition-refinement-step-v1';
  depth: number;
  leaf_count: number;
  value: number;
  confidence_half_width: number;
  movement?: number;
};

export type CapabilityMapRecord = {
  kind: 'partition-capability-map-v1';
  value: number;
  confidence_half_width: number;
  confidence_method_id: string;
  sample_count: number;
  total_measure: number;
  score_width_bits?: number;
  mean_competence?: number;
  mean_competence_confidence_half_width?: number;
  leaf_count: number;
  refinement_ladder: CapabilityMapRefinementStepRecord[];
  root: CapabilityMapNodeRecord;
  diagnostics?: Record<string, unknown>;
};

export type CostSummaryRecord = {
  component_count: number;
  parameter_count?: number;
  cost?: number;
  storage_bytes?: number;
  inference_cost_measurement?: Record<string, unknown>;
  inference_cost_sample_count?: number;
  training_cost_measurement?: Record<string, unknown>;
  training_cost_sample_count?: number;
  unknown_parameter_components?: number[];
};

export type ModelResultRecord = {
  model_key: string;
  result_status: 'accepted' | 'provisional';
  program_digest: string;
  benchmark_id: string;
  score: number;
  score_integral: StateSpaceIntegralRecord;
  capability_map?: CapabilityMapRecord;
  cost_integral?: StateSpaceIntegralRecord;
  points: CompetencePointRecord[];
  cost_summary: CostSummaryRecord;
  run_ids: string[];
  measurement_count: number;
  source_kinds: string[];
  training_estimate_comparison?: TrainingEstimateComparisonRecord;
};

export type RunResultRecord = {
  source_kind: string;
  result_status: 'accepted' | 'provisional';
  source_path: string;
  run_id: string;
  run_slug: string;
  benchmark_id: string;
  program_digest: string;
  model_key: string;
  log2_volume?: number;
  measurement_count: number;
  score: number;
  cost_summary: CostSummaryRecord;
  program: Record<string, unknown>;
  program_graph: Record<string, unknown>;
  model_inspection_digest?: string;
  model_inspection_path?: string;
  measurement_dataset_digest: string;
  sampled_competence?: Record<string, unknown>;
  training_diagnostics?: TrainingDiagnosticsRecord;
};

export type TrainingDiagnosticsRecord = {
  status: string;
  stop_reason: string;
  steps_run: number;
  validation_checks: number;
  final_validation_loss: number;
  validation_loss_reference?: number;
  final_validation_step: number;
  final_validation_check: number;
  validation_history_sample_count?: number;
  validation_history_total_count?: number;
  protocol: TrainingProtocolRecord;
  validation_history: TrainingHistoryPointRecord[];
  artifacts: TrainingArtifactReferenceRecord[];
  throughput?: Record<string, unknown>;
  evaluation_curriculum?: Record<string, unknown>;
};

export type TrainingProtocolRecord = {
  kind: string;
  objective: string;
  optimizer: string;
  learning_rate?: number;
  schedule: string;
  seed: number;
  max_steps?: number;
  gate_check_interval: number;
  gate_decision_rule: string;
  min_delta: number;
  patience: number;
  validation_source: string;
};

export type TrainingHistoryPointRecord = {
  step: number;
  validation_check: number;
  validation_loss: number;
  stale_checks: number;
  learning_rates?: number[];
};

export type TrainingArtifactReferenceRecord = {
  kind: string;
  digest: string;
  path?: string;
};

const transportError = (message: string) => new Error(message);

export function isBenchmarkResultView(
  view: ResultViewRecord,
): view is BenchmarkResultViewRecord {
  return view.format === resultViewFormats.benchmarkResults;
}

export function parseResultViewRecords(value: unknown): ResultViewRecord[] {
  return requireArray(value, 'result_views', transportError).map((view, index) =>
    parseResultViewRecord(view, `result_views.${index}`),
  );
}

function parseResultViewRecord(value: unknown, path: string): ResultViewRecord {
  const record = requireRecord(value, path, transportError);
  requireLiteral(record.format_version, `${path}.format_version`, resultViewFormatVersion, transportError);
  requireString(record.source_path, `${path}.source_path`, transportError);
  optionalNumber(record.source_mtime_ms, `${path}.source_mtime_ms`, transportError);
  optionalNumber(record.source_size_bytes, `${path}.source_size_bytes`, transportError);
  requireLiteral(record.format, `${path}.format`, resultViewFormats.benchmarkResults, transportError);
  return parseBenchmarkResultViewRecord(record, path);
}

function parseBenchmarkResultViewRecord(
  record: Record<string, unknown>,
  path: string,
): BenchmarkResultViewRecord {
  return withFields(record, {
    benchmark_results: arrayOf(record.benchmark_results, `${path}.benchmark_results`, parseBenchmarkResult),
  }) as BenchmarkResultViewRecord;
}

function parseBenchmarkResult(value: unknown, path: string): BenchmarkResultRecord {
  const record = requireRecord(value, path, transportError);
  return {
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`, transportError),
    volume_axis: optional(record.volume_axis, `${path}.volume_axis`, (item, itemPath) => requireString(item, itemPath, transportError)),
    leaderboard: arrayOf(record.leaderboard, `${path}.leaderboard`, parseModelResult),
    model_candidates: arrayOf(record.model_candidates, `${path}.model_candidates`, parseModelResult),
    frontiers: parseFrontiers(record.frontiers, `${path}.frontiers`),
    reference_curves: optional(record.reference_curves, `${path}.reference_curves`, (value, valuePath) => arrayOf(value, valuePath, parseReferenceCurve)),
    training_history: arrayOf(record.training_history, `${path}.training_history`, parseRunResult),
    plot_runs: arrayOf(record.plot_runs, `${path}.plot_runs`, parseRunResult),
    model_inspections: arrayOf(record.model_inspections ?? [], `${path}.model_inspections`, parseModelInspectionRecord),
  };
}

function parseReferenceCurve(value: unknown, path: string): ReferenceCurveRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, ['kind', 'key', 'label', 'x_axis', 'y_axis']);
  if (record.kind !== 'oracle-cost-measurement-reference-v1') {
    throw transportError(`${path}.kind is invalid`);
  }
  return {
    kind: requireString(record.kind, `${path}.kind`, transportError) as 'oracle-cost-measurement-reference-v1',
    key: requireString(record.key, `${path}.key`, transportError),
    label: requireString(record.label, `${path}.label`, transportError),
    x_axis: requireString(record.x_axis, `${path}.x_axis`, transportError),
    y_axis: requireString(record.y_axis, `${path}.y_axis`, transportError),
    points: arrayOf(record.points, `${path}.points`, parseReferenceCurvePoint),
  };
}

function parseReferenceCurvePoint(value: unknown, path: string): ReferenceCurvePointRecord {
  const record = requireRecord(value, path, transportError);
  return {
    log2_volume: requireNumber(record.log2_volume, `${path}.log2_volume`, transportError),
    score: requireNumber(record.score, `${path}.score`, transportError),
    cost: requireNumber(record.cost, `${path}.cost`, transportError),
    metadata: optional(record.metadata, `${path}.metadata`, parseRecordMetadata),
  };
}

function parseModelResult(value: unknown, path: string): ModelResultRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, ['model_key', 'result_status', 'program_digest', 'benchmark_id']);
  if (record.result_status !== 'accepted' && record.result_status !== 'provisional') {
    throw transportError(`${path}.result_status must be accepted or provisional`);
  }
  return withFields(record, {
    model_key: requireString(record.model_key, `${path}.model_key`, transportError),
    result_status: requireString(record.result_status, `${path}.result_status`, transportError) as 'accepted' | 'provisional',
    program_digest: requireString(record.program_digest, `${path}.program_digest`, transportError),
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`, transportError),
    score: requireNumber(record.score, `${path}.score`, transportError),
    score_integral: parseStateSpaceIntegral(record.score_integral, `${path}.score_integral`),
    capability_map: optional(record.capability_map, `${path}.capability_map`, parseCapabilityMap),
    cost_integral: optional(record.cost_integral, `${path}.cost_integral`, parseStateSpaceIntegral),
    points: arrayOf(record.points, `${path}.points`, parseCompetencePoint),
    cost_summary: parseCostSummary(record.cost_summary, `${path}.cost_summary`),
    run_ids: stringArray(record.run_ids, `${path}.run_ids`),
    measurement_count: requireNumber(record.measurement_count, `${path}.measurement_count`, transportError),
    source_kinds: stringArray(record.source_kinds, `${path}.source_kinds`),
    training_estimate_comparison: optional(record.training_estimate_comparison, `${path}.training_estimate_comparison`, parseTrainingEstimateComparison),
  }) as ModelResultRecord;
}

function parseTrainingEstimateComparison(value: unknown, path: string): TrainingEstimateComparisonRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, ['kind']);
  if (record.kind !== 'training-vs-accepted-sampled-competence-v1') {
    throw transportError(`${path}.kind is invalid`);
  }
  return withFields(record, {
    kind: requireString(record.kind, `${path}.kind`, transportError) as 'training-vs-accepted-sampled-competence-v1',
    accepted_score: requireNumber(record.accepted_score, `${path}.accepted_score`, transportError),
    training_score: requireNumber(record.training_score, `${path}.training_score`, transportError),
    score_delta: requireNumber(record.score_delta, `${path}.score_delta`, transportError),
    accepted_sample_count: requireNumber(record.accepted_sample_count, `${path}.accepted_sample_count`, transportError),
    training_sample_count: requireNumber(record.training_sample_count, `${path}.training_sample_count`, transportError),
    point_count: requireNumber(record.point_count, `${path}.point_count`, transportError),
    matched_point_count: requireNumber(record.matched_point_count, `${path}.matched_point_count`, transportError),
    points: arrayOf(record.points, `${path}.points`, parseTrainingEstimateComparisonPoint),
  }) as TrainingEstimateComparisonRecord;
}

function parseStateSpaceIntegral(value: unknown, path: string): StateSpaceIntegralRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, ['kind']);
  return {
    kind: requireString(record.kind, `${path}.kind`, transportError),
    value: requireNumber(record.value, `${path}.value`, transportError),
    terms: arrayOf(record.terms, `${path}.terms`, parseStateSpaceIntegralTerm),
  };
}

function parseStateSpaceIntegralTerm(value: unknown, path: string): StateSpaceIntegralTermRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, ['kind']);
  requireNumbers(record, path, [
    'log2_volume_minimum',
    'log2_volume_maximum',
    'width_in_bits',
    'contribution',
  ]);
  return {
    kind: requireString(record.kind, `${path}.kind`, transportError),
    log2_volume_minimum: requireNumber(record.log2_volume_minimum, `${path}.log2_volume_minimum`, transportError),
    log2_volume_maximum: requireNumber(record.log2_volume_maximum, `${path}.log2_volume_maximum`, transportError),
    width_in_bits: requireNumber(record.width_in_bits, `${path}.width_in_bits`, transportError),
    contribution: requireNumber(record.contribution, `${path}.contribution`, transportError),
    representative_log2_volume: optionalNumber(record.representative_log2_volume, `${path}.representative_log2_volume`, transportError),
    sample_count: optionalNumber(record.sample_count, `${path}.sample_count`, transportError),
    confidence_half_width: optionalNumber(record.confidence_half_width, `${path}.confidence_half_width`, transportError),
    confidence_method_id: optional(record.confidence_method_id, `${path}.confidence_method_id`, parseString),
    region: optional(record.region, `${path}.region`, parseStateSpaceRegionRecord),
  };
}

function parseCapabilityMap(value: unknown, path: string): CapabilityMapRecord {
  const record = requireRecord(value, path, transportError);
  if (record.kind !== 'partition-capability-map-v1') {
    throw transportError(`${path}.kind is invalid`);
  }
  return {
    kind: requireString(record.kind, `${path}.kind`, transportError) as 'partition-capability-map-v1',
    value: requireNumber(record.value, `${path}.value`, transportError),
    confidence_half_width: requireNumber(record.confidence_half_width, `${path}.confidence_half_width`, transportError),
    confidence_method_id: requireString(record.confidence_method_id, `${path}.confidence_method_id`, transportError),
    sample_count: requireNumber(record.sample_count, `${path}.sample_count`, transportError),
    total_measure: requireNumber(record.total_measure, `${path}.total_measure`, transportError),
    score_width_bits: optionalNumber(record.score_width_bits, `${path}.score_width_bits`, transportError),
    mean_competence: optionalNumber(record.mean_competence, `${path}.mean_competence`, transportError),
    mean_competence_confidence_half_width: optionalNumber(
      record.mean_competence_confidence_half_width,
      `${path}.mean_competence_confidence_half_width`,
      transportError,
    ),
    leaf_count: requireNumber(record.leaf_count, `${path}.leaf_count`, transportError),
    refinement_ladder: arrayOf(record.refinement_ladder, `${path}.refinement_ladder`, parseCapabilityMapRefinementStep),
    root: parseCapabilityMapNode(record.root, `${path}.root`),
    diagnostics: optional(record.diagnostics, `${path}.diagnostics`, parseRecordMetadata),
  };
}

function parseCapabilityMapRefinementStep(value: unknown, path: string): CapabilityMapRefinementStepRecord {
  const record = requireRecord(value, path, transportError);
  if (record.kind !== 'partition-refinement-step-v1') {
    throw transportError(`${path}.kind is invalid`);
  }
  return {
    kind: requireString(record.kind, `${path}.kind`, transportError) as 'partition-refinement-step-v1',
    depth: requireNumber(record.depth, `${path}.depth`, transportError),
    leaf_count: requireNumber(record.leaf_count, `${path}.leaf_count`, transportError),
    value: requireNumber(record.value, `${path}.value`, transportError),
    confidence_half_width: requireNumber(record.confidence_half_width, `${path}.confidence_half_width`, transportError),
    movement: optionalNumber(record.movement, `${path}.movement`, transportError),
  };
}

function parseCapabilityMapNode(value: unknown, path: string): CapabilityMapNodeRecord {
  const record = requireRecord(value, path, transportError);
  if (record.kind !== 'partition-capability-node-v1') {
    throw transportError(`${path}.kind is invalid`);
  }
  return {
    kind: requireString(record.kind, `${path}.kind`, transportError) as 'partition-capability-node-v1',
    label: requireString(record.label, `${path}.label`, transportError),
    measure: requireNumber(record.measure, `${path}.measure`, transportError),
    sample_count: requireNumber(record.sample_count, `${path}.sample_count`, transportError),
    competence: requireNumber(record.competence, `${path}.competence`, transportError),
    confidence_half_width: requireNumber(record.confidence_half_width, `${path}.confidence_half_width`, transportError),
    region: optional(record.region, `${path}.region`, parseStateSpaceRegionRecord),
    children: arrayOf(record.children, `${path}.children`, parseCapabilityMapNode),
  };
}

function parseTrainingEstimateComparisonPoint(value: unknown, path: string): TrainingEstimateComparisonPointRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, ['status']);
  if (record.status !== 'matched' && record.status !== 'accepted-only' && record.status !== 'training-only') {
    throw transportError(`${path}.status is invalid`);
  }
  return withFields(record, {
    log2_volume: requireNumber(record.log2_volume, `${path}.log2_volume`, transportError),
    log2_volume_minimum: optional(record.log2_volume_minimum, `${path}.log2_volume_minimum`, parseNumber),
    log2_volume_maximum: optional(record.log2_volume_maximum, `${path}.log2_volume_maximum`, parseNumber),
    status: requireString(record.status, `${path}.status`, transportError) as 'matched' | 'accepted-only' | 'training-only',
    accepted_score: optional(record.accepted_score, `${path}.accepted_score`, parseNumber),
    training_score: optional(record.training_score, `${path}.training_score`, parseNumber),
    score_delta: optional(record.score_delta, `${path}.score_delta`, parseNumber),
    accepted_sample_count: optional(record.accepted_sample_count, `${path}.accepted_sample_count`, parseNumber),
    training_sample_count: optional(record.training_sample_count, `${path}.training_sample_count`, parseNumber),
  }) as TrainingEstimateComparisonPointRecord;
}

function parseRunResult(value: unknown, path: string): RunResultRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, [
    'source_kind',
    'result_status',
    'source_path',
    'run_id',
    'run_slug',
    'benchmark_id',
    'program_digest',
    'model_key',
    'measurement_dataset_digest',
  ]);
  if (record.result_status !== 'accepted' && record.result_status !== 'provisional') {
    throw transportError(`${path}.result_status must be accepted or provisional`);
  }
  return withFields(record, {
    measurement_count: requireNumber(record.measurement_count, `${path}.measurement_count`, transportError),
    score: requireNumber(record.score, `${path}.score`, transportError),
    cost_summary: parseCostSummary(record.cost_summary, `${path}.cost_summary`),
    program: requireRecord(record.program, `${path}.program`, transportError),
    program_graph: requireRecord(record.program_graph, `${path}.program_graph`, transportError),
    training_diagnostics: optional(record.training_diagnostics, `${path}.training_diagnostics`, parseTrainingDiagnostics),
  }) as RunResultRecord;
}

function parseCompetencePoint(value: unknown, path: string): CompetencePointRecord {
  const record = requireRecord(value, path, transportError);
  return {
    log2_volume: requireNumber(record.log2_volume, `${path}.log2_volume`, transportError),
    score: requireNumber(record.score, `${path}.score`, transportError),
    sample_count: optionalNumber(record.sample_count, `${path}.sample_count`, transportError),
    run_ids: stringArray(record.run_ids, `${path}.run_ids`),
    competence_value_kind: optional(record.competence_value_kind, `${path}.competence_value_kind`, (item, itemPath) => requireString(item, itemPath, transportError)),
    predictability_boundary: optionalNumber(record.predictability_boundary, `${path}.predictability_boundary`, transportError),
    time_points: optional(record.time_points, `${path}.time_points`, (item, itemPath) => arrayOf(item, itemPath, parseCompetenceTimePoint)),
  };
}

function parseCompetenceTimePoint(value: unknown, path: string): CompetenceTimePointRecord {
  const record = requireRecord(value, path, transportError);
  return {
    time: requireNumber(record.time, `${path}.time`, transportError),
    bits: requireNumber(record.bits, `${path}.bits`, transportError),
    certified_epsilon: optionalNumber(record.certified_epsilon, `${path}.certified_epsilon`, transportError),
    evolution_scale: optionalNumber(record.evolution_scale, `${path}.evolution_scale`, transportError),
  };
}

function parseTrainingDiagnostics(value: unknown, path: string): TrainingDiagnosticsRecord {
  const record = requireRecord(value, path, transportError);
  requireStrings(record, path, ['status', 'stop_reason']);
  requireNumbers(record, path, [
    'steps_run',
    'validation_checks',
    'final_validation_loss',
    'final_validation_step',
    'final_validation_check',
  ]);
  return withFields(record, {
    validation_loss_reference: optionalNumber(record.validation_loss_reference, `${path}.validation_loss_reference`, transportError),
    validation_history_sample_count: optionalNumber(record.validation_history_sample_count, `${path}.validation_history_sample_count`, transportError),
    validation_history_total_count: optionalNumber(record.validation_history_total_count, `${path}.validation_history_total_count`, transportError),
    protocol: requireRecord(record.protocol, `${path}.protocol`, transportError) as TrainingProtocolRecord,
    validation_history: requireArray(record.validation_history, `${path}.validation_history`, transportError) as TrainingHistoryPointRecord[],
    artifacts: requireArray(record.artifacts, `${path}.artifacts`, transportError) as TrainingArtifactReferenceRecord[],
    throughput:
      record.throughput === undefined
        ? undefined
        : requireRecord(record.throughput, `${path}.throughput`, transportError),
    evaluation_curriculum:
      record.evaluation_curriculum === undefined
        ? undefined
        : requireRecord(record.evaluation_curriculum, `${path}.evaluation_curriculum`, transportError),
  }) as TrainingDiagnosticsRecord;
}

function parseCostSummary(value: unknown, path: string): CostSummaryRecord {
  const record = requireRecord(value, path, transportError);
  return {
    ...record,
    component_count: requireNumber(record.component_count, `${path}.component_count`, transportError),
    parameter_count: optionalNumber(record.parameter_count, `${path}.parameter_count`, transportError),
    cost: optionalNumber(record.cost, `${path}.cost`, transportError),
    storage_bytes: optionalNumber(record.storage_bytes, `${path}.storage_bytes`, transportError),
    inference_cost_measurement:
      record.inference_cost_measurement === undefined
        ? undefined
        : requireRecord(record.inference_cost_measurement, `${path}.inference_cost_measurement`, transportError),
    inference_cost_sample_count: optionalNumber(record.inference_cost_sample_count, `${path}.inference_cost_sample_count`, transportError),
    training_cost_measurement:
      record.training_cost_measurement === undefined
        ? undefined
        : requireRecord(record.training_cost_measurement, `${path}.training_cost_measurement`, transportError),
    training_cost_sample_count: optionalNumber(record.training_cost_sample_count, `${path}.training_cost_sample_count`, transportError),
    unknown_parameter_components:
      record.unknown_parameter_components === undefined
        ? undefined
        : numberArray(record.unknown_parameter_components, `${path}.unknown_parameter_components`),
  };
}

function parseRecordMetadata(value: unknown, path: string): Record<string, unknown> {
  return requireRecord(value, path, transportError);
}

function parseFrontiers(value: unknown, path: string): Record<string, ModelResultRecord[]> {
  const record = requireRecord(value, path, transportError);
  return Object.fromEntries(
    Object.entries(record).map(([key, models]) => [
      key,
      arrayOf(models, `${path}.${key}`, parseModelResult),
    ]),
  );
}

function arrayOf<T>(value: unknown, path: string, parse: (item: unknown, path: string) => T): T[] {
  return requireArray(value, path, transportError).map((item, index) => parse(item, `${path}.${index}`));
}

function optional<T>(value: unknown, path: string, parse: (item: unknown, path: string) => T): T | undefined {
  return value === undefined ? undefined : parse(value, path);
}

function parseNumber(value: unknown, path: string): number {
  return requireNumber(value, path, transportError);
}

function parseString(value: unknown, path: string): string {
  return requireString(value, path, transportError);
}

function requireStrings(record: Record<string, unknown>, path: string, fields: string[]): void {
  fields.forEach((field) => requireString(record[field], `${path}.${field}`, transportError));
}

function requireNumbers(record: Record<string, unknown>, path: string, fields: string[]): void {
  fields.forEach((field) => requireNumber(record[field], `${path}.${field}`, transportError));
}

function withFields<T extends Record<string, unknown>>(
  record: Record<string, unknown>,
  fields: T,
): Record<string, unknown> & T {
  return { ...record, ...fields };
}

function stringArray(value: unknown, path: string): string[] {
  return arrayOf(value, path, (item, itemPath) => requireString(item, itemPath, transportError));
}

function numberArray(value: unknown, path: string): number[] {
  return arrayOf(value, path, (item, itemPath) => requireNumber(item, itemPath, transportError));
}
