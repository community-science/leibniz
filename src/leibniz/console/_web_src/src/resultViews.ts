import {
  parseModelInspectionRecord,
  type ModelInspectionRecord,
} from './modelInspections.ts';

export type ResultViewRecord =
  | ImportedResultViewRecord
  | BenchmarkResultViewRecord
  | WorkQueueViewRecord;

export type ImportedResultViewRecord = {
  format: 'leibniz.console.imported-results';
  format_version: 1;
  source_path: string;
  source_mtime_ms?: number;
  source_size_bytes?: number;
  publication_bundles: ImportedPublicationBundleRecord[];
};

class ResultViewTransportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ResultViewTransportError';
  }
}

export type ImportedPublicationBundleRecord = {
  id: string;
  digest: string;
  source_path: string;
  submission_package_id: string;
  benchmark_ids: string[];
  measurement_count: number;
  measurement_dataset: Record<string, unknown>;
  measurement_score_view: Record<string, unknown>;
};

export type BenchmarkResultViewRecord = {
  format: 'leibniz.console.benchmark-results';
  format_version: 1;
  source_path: string;
  source_mtime_ms?: number;
  source_size_bytes?: number;
  benchmark_results: BenchmarkResultRecord[];
};

export type WorkQueueViewRecord = {
  format: 'leibniz.console.work-queue';
  format_version: 1;
  source_path: string;
  source_mtime_ms?: number;
  source_size_bytes?: number;
  queue_items: WorkQueueItemRecord[];
};

export type WorkQueueItemStatus = 'pending' | 'reserved' | 'completed' | 'failed';

export type WorkQueueItemRecord = {
  format: 'leibniz.work-queue-item';
  format_version: 1;
  id: string;
  benchmark_id: string;
  proposal_id: string;
  candidate_id?: string;
  proposal_set_path: string;
  command: string[];
  status: WorkQueueItemStatus;
  sequence: number;
  run_id?: string;
  measurement_dataset_path?: string;
  error?: string;
};

export type BenchmarkResultRecord = {
  benchmark_id: string;
  complexity_axis?: string;
  scale_axis?: string;
  cost_axes: CostAxisRecord[];
  leaderboard: ModelResultRecord[];
  frontiers: Record<string, ModelResultRecord[]>;
  training_history: RunResultRecord[];
  model_inspections: ModelInspectionRecord[];
  proposals: ProposalRecord[];
};

export type CostAxisRecord = {
  key: string;
  label: string;
};

export type ModelResultRecord = {
  model_key: string;
  architecture_digest: string;
  benchmark_id: string;
  score: number;
  observed_complexities: number[];
  points: CompetencePointRecord[];
  cost_summary: CostSummaryRecord;
  run_ids: string[];
  measurement_count: number;
  source_kinds: string[];
};

export type CompetencePointRecord = {
  complexity: number;
  score: number;
  sample_count?: number;
  run_ids: string[];
};

export type RunResultRecord = {
  source_kind: string;
  source_path: string;
  run_id: string;
  run_slug: string;
  benchmark_id: string;
  architecture_digest: string;
  model_key: string;
  scale?: number;
  complexity?: number;
  measurement_count: number;
  score: number;
  cost_summary: CostSummaryRecord;
  architecture: Record<string, unknown>;
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
  best_validation_loss: number;
  best_validation_step: number;
  best_validation_check: number;
  final_validation_loss: number;
  final_validation_step: number;
  final_validation_check: number;
  protocol: TrainingProtocolRecord;
  validation_history: TrainingHistoryPointRecord[];
  artifacts: TrainingArtifactReferenceRecord[];
};

export type TrainingProtocolRecord = {
  kind: string;
  objective: string;
  optimizer: string;
  learning_rate: number;
  schedule: string;
  seed: number;
  batch_size: number;
  max_steps: number;
  validation_interval: number;
  validation_sample_count: number;
  min_delta: number;
  patience: number;
  validation_source: string;
};

export type TrainingHistoryPointRecord = {
  step: number;
  validation_check: number;
  validation_loss: number;
  best_validation_loss: number;
  best_validation_step: number;
  best_validation_check: number;
  stale_checks: number;
  learning_rates?: number[];
};

export type TrainingArtifactReferenceRecord = {
  kind: string;
  digest: string;
  path?: string;
};

export type ProposalRecord = {
  id: string;
  rank: number;
  candidate_kind: string;
  candidate_id: string;
  rationale: string;
  predicted_score?: number;
  uncertainty?: number;
  acquisition_value?: number;
  acquisition_model?: string;
  acquisition_components?: Record<string, unknown>;
  search_diagnostics?: Record<string, unknown>;
  novelty?: number;
  expected_frontier_improvement?: number;
  selector_name?: string;
  source_candidate_rank?: number;
  comparable_cost_best_score?: number;
  resource_stratum_index?: number;
  resource_stratum_count?: number;
  command: string[];
};

export type CostSummaryRecord = {
  layer_count: number;
  parameter_count: number;
  parameter_bytes: number;
  inference_flops: number;
  unknown_parameter_layers?: string[];
};

export function isBenchmarkResultView(
  view: ResultViewRecord,
): view is BenchmarkResultViewRecord {
  return view.format === 'leibniz.console.benchmark-results';
}

export function isImportedResultView(
  view: ResultViewRecord,
): view is ImportedResultViewRecord {
  return view.format === 'leibniz.console.imported-results';
}

export function isWorkQueueView(
  view: ResultViewRecord,
): view is WorkQueueViewRecord {
  return view.format === 'leibniz.console.work-queue';
}

export function parseResultViewRecords(value: unknown): ResultViewRecord[] {
  return requireArray(value, 'result_views').map((view, index) =>
    parseResultViewRecord(view, `result_views.${index}`),
  );
}

function parseResultViewRecord(value: unknown, path: string): ResultViewRecord {
  const record = requireRecord(value, path);
  if (record.format === 'leibniz.console.benchmark-results') {
    return parseBenchmarkResultViewRecord(record, path);
  }
  if (record.format === 'leibniz.console.work-queue') {
    return parseWorkQueueViewRecord(record, path);
  }
  return parseImportedResultViewRecord(record, path);
}

function parseImportedResultViewRecord(
  record: Record<string, unknown>,
  path: string,
): ImportedResultViewRecord {
  const format = requireLiteral(record.format, `${path}.format`, 'leibniz.console.imported-results');
  const formatVersion = requireLiteral(record.format_version, `${path}.format_version`, 1);
  const sourcePath = requireString(record.source_path, `${path}.source_path`);
  const sourceMtimeMs = optionalNumber(record.source_mtime_ms, `${path}.source_mtime_ms`);
  const sourceSizeBytes = optionalNumber(record.source_size_bytes, `${path}.source_size_bytes`);
  const publicationBundles = requireArray(
    record.publication_bundles,
    `${path}.publication_bundles`,
  ).map((bundle, index) =>
    parseImportedPublicationBundleRecord(bundle, `${path}.publication_bundles.${index}`),
  );
  return {
    format,
    format_version: formatVersion,
    source_path: sourcePath,
    source_mtime_ms: sourceMtimeMs,
    source_size_bytes: sourceSizeBytes,
    publication_bundles: publicationBundles,
  };
}

function parseBenchmarkResultViewRecord(
  record: Record<string, unknown>,
  path: string,
): BenchmarkResultViewRecord {
  const format = requireLiteral(record.format, `${path}.format`, 'leibniz.console.benchmark-results');
  const formatVersion = requireLiteral(record.format_version, `${path}.format_version`, 1);
  const sourcePath = requireString(record.source_path, `${path}.source_path`);
  const sourceMtimeMs = optionalNumber(record.source_mtime_ms, `${path}.source_mtime_ms`);
  const sourceSizeBytes = optionalNumber(record.source_size_bytes, `${path}.source_size_bytes`);
  const benchmarkResults = requireArray(
    record.benchmark_results,
    `${path}.benchmark_results`,
  ).map((result, index) => parseBenchmarkResult(result, `${path}.benchmark_results.${index}`));
  return {
    format,
    format_version: formatVersion,
    source_path: sourcePath,
    source_mtime_ms: sourceMtimeMs,
    source_size_bytes: sourceSizeBytes,
    benchmark_results: benchmarkResults,
  };
}

function parseWorkQueueViewRecord(
  record: Record<string, unknown>,
  path: string,
): WorkQueueViewRecord {
  const format = requireLiteral(record.format, `${path}.format`, 'leibniz.console.work-queue');
  const formatVersion = requireLiteral(record.format_version, `${path}.format_version`, 1);
  const sourcePath = requireString(record.source_path, `${path}.source_path`);
  const sourceMtimeMs = optionalNumber(record.source_mtime_ms, `${path}.source_mtime_ms`);
  const sourceSizeBytes = optionalNumber(record.source_size_bytes, `${path}.source_size_bytes`);
  const queueItems = requireArray(record.queue_items, `${path}.queue_items`).map((item, index) =>
    parseWorkQueueItem(item, `${path}.queue_items.${index}`),
  );
  return {
    format,
    format_version: formatVersion,
    source_path: sourcePath,
    source_mtime_ms: sourceMtimeMs,
    source_size_bytes: sourceSizeBytes,
    queue_items: queueItems,
  };
}

function parseWorkQueueItem(value: unknown, path: string): WorkQueueItemRecord {
  const record = requireRecord(value, path);
  return {
    format: requireLiteral(record.format, `${path}.format`, 'leibniz.work-queue-item'),
    format_version: requireLiteral(record.format_version, `${path}.format_version`, 1),
    id: requireString(record.id, `${path}.id`),
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
    proposal_id: requireString(record.proposal_id, `${path}.proposal_id`),
    candidate_id:
      record.candidate_id === undefined
        ? undefined
        : requireString(record.candidate_id, `${path}.candidate_id`),
    proposal_set_path: requireString(record.proposal_set_path, `${path}.proposal_set_path`),
    command: parseStringArray(record.command, `${path}.command`),
    status: parseWorkQueueStatus(record.status, `${path}.status`),
    sequence: requireNumber(record.sequence, `${path}.sequence`),
    run_id:
      record.run_id === undefined ? undefined : requireString(record.run_id, `${path}.run_id`),
    measurement_dataset_path:
      record.measurement_dataset_path === undefined
        ? undefined
        : requireString(record.measurement_dataset_path, `${path}.measurement_dataset_path`),
    error:
      record.error === undefined ? undefined : requireString(record.error, `${path}.error`),
  };
}

function parseWorkQueueStatus(value: unknown, path: string): WorkQueueItemStatus {
  const status = requireString(value, path);
  if (
    status !== 'pending' &&
    status !== 'reserved' &&
    status !== 'completed' &&
    status !== 'failed'
  ) {
    throw new ResultViewTransportError(`${path}: expected work queue status`);
  }
  return status;
}

function parseBenchmarkResult(value: unknown, path: string): BenchmarkResultRecord {
  const record = requireRecord(value, path);
  const costAxes = requireArray(record.cost_axes, `${path}.cost_axes`).map((axis, index) => {
    const axisRecord = requireRecord(axis, `${path}.cost_axes.${index}`);
    return {
      key: requireString(axisRecord.key, `${path}.cost_axes.${index}.key`),
      label: requireString(axisRecord.label, `${path}.cost_axes.${index}.label`),
    };
  });
  const frontiersRecord = requireRecord(record.frontiers, `${path}.frontiers`);
  const frontiers = Object.fromEntries(
    Object.entries(frontiersRecord).map(([key, value]) => [
      key,
      requireArray(value, `${path}.frontiers.${key}`).map((model, index) =>
        parseModelResult(model, `${path}.frontiers.${key}.${index}`),
      ),
    ]),
  );
  return {
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
    complexity_axis:
      record.complexity_axis === undefined
        ? undefined
        : requireString(record.complexity_axis, `${path}.complexity_axis`),
    scale_axis:
      record.scale_axis === undefined
        ? undefined
        : requireString(record.scale_axis, `${path}.scale_axis`),
    cost_axes: costAxes,
    leaderboard: requireArray(record.leaderboard, `${path}.leaderboard`).map((model, index) =>
      parseModelResult(model, `${path}.leaderboard.${index}`),
    ),
    frontiers,
    training_history: requireArray(record.training_history, `${path}.training_history`).map(
      (run, index) => parseRunResult(run, `${path}.training_history.${index}`),
    ),
    model_inspections:
      record.model_inspections === undefined
        ? []
        : requireArray(record.model_inspections, `${path}.model_inspections`).map(
            (inspection, index) =>
              parseModelInspectionRecord(inspection, `${path}.model_inspections.${index}`),
          ),
    proposals:
      record.proposals === undefined
        ? []
        : requireArray(record.proposals, `${path}.proposals`).map((proposal, index) =>
            parseProposal(proposal, `${path}.proposals.${index}`),
          ),
  };
}

function parseModelResult(value: unknown, path: string): ModelResultRecord {
  const record = requireRecord(value, path);
  return {
    model_key: requireString(record.model_key, `${path}.model_key`),
    architecture_digest: requireString(record.architecture_digest, `${path}.architecture_digest`),
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
    score: requireNumber(record.score, `${path}.score`),
    observed_complexities: parseNumberArray(
      record.observed_complexities,
      `${path}.observed_complexities`,
    ),
    points: requireArray(record.points, `${path}.points`).map((point, index) =>
      parseCompetencePoint(point, `${path}.points.${index}`),
    ),
    cost_summary: parseCostSummary(record.cost_summary, `${path}.cost_summary`),
    run_ids: parseStringArray(record.run_ids, `${path}.run_ids`),
    measurement_count: requireNumber(record.measurement_count, `${path}.measurement_count`),
    source_kinds: parseStringArray(record.source_kinds, `${path}.source_kinds`),
  };
}

function parseCompetencePoint(value: unknown, path: string): CompetencePointRecord {
  const record = requireRecord(value, path);
  return {
    complexity: requireNumber(record.complexity, `${path}.complexity`),
    score: requireNumber(record.score, `${path}.score`),
    sample_count:
      record.sample_count === undefined
        ? undefined
        : requireNumber(record.sample_count, `${path}.sample_count`),
    run_ids: parseStringArray(record.run_ids, `${path}.run_ids`),
  };
}

function parseRunResult(value: unknown, path: string): RunResultRecord {
  const record = requireRecord(value, path);
  return {
    source_kind: requireString(record.source_kind, `${path}.source_kind`),
    source_path: requireString(record.source_path, `${path}.source_path`),
    run_id: requireString(record.run_id, `${path}.run_id`),
    run_slug: requireString(record.run_slug, `${path}.run_slug`),
    benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
    architecture_digest: requireString(record.architecture_digest, `${path}.architecture_digest`),
    model_key: requireString(record.model_key, `${path}.model_key`),
    scale:
      record.scale === undefined ? undefined : requireNumber(record.scale, `${path}.scale`),
    complexity:
      record.complexity === undefined
        ? undefined
        : requireNumber(record.complexity, `${path}.complexity`),
    measurement_count: requireNumber(record.measurement_count, `${path}.measurement_count`),
    score: requireNumber(record.score, `${path}.score`),
    cost_summary: parseCostSummary(record.cost_summary, `${path}.cost_summary`),
    architecture: requireRecord(record.architecture, `${path}.architecture`),
    model_inspection_digest:
      record.model_inspection_digest === undefined
        ? undefined
        : requireString(record.model_inspection_digest, `${path}.model_inspection_digest`),
    model_inspection_path:
      record.model_inspection_path === undefined
        ? undefined
        : requireString(record.model_inspection_path, `${path}.model_inspection_path`),
    measurement_dataset_digest: requireString(
      record.measurement_dataset_digest,
      `${path}.measurement_dataset_digest`,
    ),
    sampled_competence:
      record.sampled_competence === undefined
        ? undefined
        : requireRecord(record.sampled_competence, `${path}.sampled_competence`),
    training_diagnostics:
      record.training_diagnostics === undefined
        ? undefined
        : parseTrainingDiagnostics(
            record.training_diagnostics,
            `${path}.training_diagnostics`,
          ),
  };
}

function parseTrainingDiagnostics(value: unknown, path: string): TrainingDiagnosticsRecord {
  const record = requireRecord(value, path);
  return {
    status: requireString(record.status, `${path}.status`),
    stop_reason: requireString(record.stop_reason, `${path}.stop_reason`),
    steps_run: requireNumber(record.steps_run, `${path}.steps_run`),
    validation_checks: requireNumber(record.validation_checks, `${path}.validation_checks`),
    best_validation_loss: requireNumber(
      record.best_validation_loss,
      `${path}.best_validation_loss`,
    ),
    best_validation_step: requireNumber(
      record.best_validation_step,
      `${path}.best_validation_step`,
    ),
    best_validation_check: requireNumber(
      record.best_validation_check,
      `${path}.best_validation_check`,
    ),
    final_validation_loss: requireNumber(
      record.final_validation_loss,
      `${path}.final_validation_loss`,
    ),
    final_validation_step: requireNumber(
      record.final_validation_step,
      `${path}.final_validation_step`,
    ),
    final_validation_check: requireNumber(
      record.final_validation_check,
      `${path}.final_validation_check`,
    ),
    protocol: parseTrainingProtocol(record.protocol, `${path}.protocol`),
    validation_history: requireArray(record.validation_history, `${path}.validation_history`).map(
      (point, index) => parseTrainingHistoryPoint(point, `${path}.validation_history.${index}`),
    ),
    artifacts: requireArray(record.artifacts, `${path}.artifacts`).map((artifact, index) =>
      parseTrainingArtifactReference(artifact, `${path}.artifacts.${index}`),
    ),
  };
}

function parseTrainingProtocol(value: unknown, path: string): TrainingProtocolRecord {
  const record = requireRecord(value, path);
  return {
    kind: requireString(record.kind, `${path}.kind`),
    objective: requireString(record.objective, `${path}.objective`),
    optimizer: requireString(record.optimizer, `${path}.optimizer`),
    learning_rate: requireNumber(record.learning_rate, `${path}.learning_rate`),
    schedule: requireString(record.schedule, `${path}.schedule`),
    seed: requireNumber(record.seed, `${path}.seed`),
    batch_size: requireNumber(record.batch_size, `${path}.batch_size`),
    max_steps: requireNumber(record.max_steps, `${path}.max_steps`),
    validation_interval: requireNumber(
      record.validation_interval,
      `${path}.validation_interval`,
    ),
    validation_sample_count: requireNumber(
      record.validation_sample_count,
      `${path}.validation_sample_count`,
    ),
    min_delta: requireNumber(record.min_delta, `${path}.min_delta`),
    patience: requireNumber(record.patience, `${path}.patience`),
    validation_source: requireString(record.validation_source, `${path}.validation_source`),
  };
}

function parseTrainingHistoryPoint(value: unknown, path: string): TrainingHistoryPointRecord {
  const record = requireRecord(value, path);
  return {
    step: requireNumber(record.step, `${path}.step`),
    validation_check: requireNumber(record.validation_check, `${path}.validation_check`),
    validation_loss: requireNumber(record.validation_loss, `${path}.validation_loss`),
    best_validation_loss: requireNumber(
      record.best_validation_loss,
      `${path}.best_validation_loss`,
    ),
    best_validation_step: requireNumber(
      record.best_validation_step,
      `${path}.best_validation_step`,
    ),
    best_validation_check: requireNumber(
      record.best_validation_check,
      `${path}.best_validation_check`,
    ),
    stale_checks: requireNumber(record.stale_checks, `${path}.stale_checks`),
    learning_rates:
      record.learning_rates === undefined
        ? undefined
        : parseNumberArray(record.learning_rates, `${path}.learning_rates`),
  };
}

function parseTrainingArtifactReference(
  value: unknown,
  path: string,
): TrainingArtifactReferenceRecord {
  const record = requireRecord(value, path);
  return {
    kind: requireString(record.kind, `${path}.kind`),
    digest: requireString(record.digest, `${path}.digest`),
    path: record.path === undefined ? undefined : requireString(record.path, `${path}.path`),
  };
}

function parseProposal(value: unknown, path: string): ProposalRecord {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    rank: requireNumber(record.rank, `${path}.rank`),
    candidate_kind: requireString(record.candidate_kind, `${path}.candidate_kind`),
    candidate_id: requireString(record.candidate_id, `${path}.candidate_id`),
    rationale: requireString(record.rationale, `${path}.rationale`),
    predicted_score: optionalNumber(record.predicted_score, `${path}.predicted_score`),
    uncertainty: optionalNumber(record.uncertainty, `${path}.uncertainty`),
    acquisition_value: optionalNumber(record.acquisition_value, `${path}.acquisition_value`),
    acquisition_model:
      record.acquisition_model === undefined
        ? undefined
        : requireString(record.acquisition_model, `${path}.acquisition_model`),
    acquisition_components:
      record.acquisition_components === undefined
        ? undefined
        : requireRecord(record.acquisition_components, `${path}.acquisition_components`),
    search_diagnostics:
      record.search_diagnostics === undefined
        ? undefined
        : requireRecord(record.search_diagnostics, `${path}.search_diagnostics`),
    novelty: optionalNumber(record.novelty, `${path}.novelty`),
    expected_frontier_improvement: optionalNumber(
      record.expected_frontier_improvement,
      `${path}.expected_frontier_improvement`,
    ),
    selector_name:
      record.selector_name === undefined
        ? undefined
        : requireString(record.selector_name, `${path}.selector_name`),
    source_candidate_rank: optionalNumber(
      record.source_candidate_rank,
      `${path}.source_candidate_rank`,
    ),
    comparable_cost_best_score: optionalNumber(
      record.comparable_cost_best_score,
      `${path}.comparable_cost_best_score`,
    ),
    resource_stratum_index: optionalNumber(
      record.resource_stratum_index,
      `${path}.resource_stratum_index`,
    ),
    resource_stratum_count: optionalNumber(
      record.resource_stratum_count,
      `${path}.resource_stratum_count`,
    ),
    command:
      record.command === undefined ? [] : parseStringArray(record.command, `${path}.command`),
  };
}

function parseCostSummary(value: unknown, path: string): CostSummaryRecord {
  const record = requireRecord(value, path);
  const costSummary: CostSummaryRecord = {
    layer_count: requireNumber(record.layer_count, `${path}.layer_count`),
    parameter_count: requireNumber(record.parameter_count, `${path}.parameter_count`),
    parameter_bytes: requireNumber(record.parameter_bytes, `${path}.parameter_bytes`),
    inference_flops: requireNumber(record.inference_flops, `${path}.inference_flops`),
  };
  if (record.unknown_parameter_layers !== undefined) {
    costSummary.unknown_parameter_layers = parseStringArray(
      record.unknown_parameter_layers,
      `${path}.unknown_parameter_layers`,
    );
  }
  return costSummary;
}

function parseImportedPublicationBundleRecord(
  value: unknown,
  path: string,
): ImportedPublicationBundleRecord {
  const record = requireRecord(value, path);
  const benchmarkIds = requireArray(record.benchmark_ids, `${path}.benchmark_ids`).map(
    (item, index) => requireString(item, `${path}.benchmark_ids.${index}`),
  );
  return {
    id: requireString(record.id, `${path}.id`),
    digest: requireString(record.digest, `${path}.digest`),
    source_path: requireString(record.source_path, `${path}.source_path`),
    submission_package_id: requireString(
      record.submission_package_id,
      `${path}.submission_package_id`,
    ),
    benchmark_ids: benchmarkIds,
    measurement_count: requireNumber(record.measurement_count, `${path}.measurement_count`),
    measurement_dataset: requireRecord(record.measurement_dataset, `${path}.measurement_dataset`),
    measurement_score_view: requireRecord(
      record.measurement_score_view,
      `${path}.measurement_score_view`,
    ),
  };
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ResultViewTransportError(`${path}: expected record`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new ResultViewTransportError(`${path}: expected array`);
  }
  return value;
}

function parseNumberArray(value: unknown, path: string): number[] {
  return requireArray(value, path).map((item, index) => requireNumber(item, `${path}.${index}`));
}

function parseStringArray(value: unknown, path: string): string[] {
  return requireArray(value, path).map((item, index) => requireString(item, `${path}.${index}`));
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string' || value.length === 0) {
    throw new ResultViewTransportError(`${path}: expected string`);
  }
  return value;
}

function requireNumber(value: unknown, path: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ResultViewTransportError(`${path}: expected number`);
  }
  return value;
}

function optionalNumber(value: unknown, path: string): number | undefined {
  if (value === undefined) {
    return undefined;
  }
  return requireNumber(value, path);
}

function requireLiteral<const Literal extends string | number>(
  value: unknown,
  path: string,
  expected: Literal,
): Literal {
  if (value !== expected) {
    throw new ResultViewTransportError(`${path}: expected ${String(expected)}`);
  }
  return expected;
}
