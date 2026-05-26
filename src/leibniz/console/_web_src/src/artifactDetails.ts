import type { ConsoleArtifactIndexEntryRecord } from './artifactIndex';
import { artifactKey } from './artifactBrowserModel.ts';

export type ArchitectureManifestDetailRecord = {
  kind: 'architecture-manifest';
  source_path: string;
  input_shape: number[];
  output_shape: number[];
  layers: LayerSummaryRecord[];
};

export type BenchmarkManifestDetailRecord = {
  kind: 'benchmark-manifest';
  source_path: string;
  id: string;
  outcome_space: OutcomeSpaceSummaryRecord;
  observation_ids?: string[];
};

export type MeasurementDetailRecord = {
  kind: 'measurement';
  source_path: string;
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

export type ConsoleArtifactDetailRecord =
  | ArchitectureManifestDetailRecord
  | BenchmarkManifestDetailRecord
  | MeasurementDetailRecord;

export type ConsoleArtifactDetailMap = ReadonlyMap<string, ConsoleArtifactDetailRecord>;

type LayerSummaryRecord = {
  kind: string;
  parameters?: Record<string, string | number | boolean>;
};

type OutcomeSpaceSummaryRecord = {
  id: string;
  outcomes: { id: string }[];
};

type ProbabilitySummaryRecord = {
  outcome_id: string;
  probability: number;
};

export class ConsoleArtifactDetailError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConsoleArtifactDetailError';
  }
}

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
  if (details === undefined) {
    return undefined;
  }
  return details.get(artifactDetailKeyForEntry(artifact));
}

export function parseConsoleArtifactDetailRecords(value: unknown): ConsoleArtifactDetailMap {
  const records = requireArray(value, 'artifact details').map((detail, index) =>
    parseDetailRecord(detail, `artifact details.${index}`),
  );
  return new Map(records.map((detail) => [artifactDetailKey(detail), detail]));
}

function parseDetailRecord(value: unknown, path: string): ConsoleArtifactDetailRecord {
  const record = requireRecord(value, path);
  const kind = requireString(record.kind, `${path}.kind`);

  if (kind === 'architecture-manifest') {
    return {
      kind,
      source_path: requireString(record.source_path, `${path}.source_path`),
      input_shape: parseNumberArray(record.input_shape, `${path}.input_shape`),
      output_shape: parseNumberArray(record.output_shape, `${path}.output_shape`),
      layers: requireArray(record.layers, `${path}.layers`).map((layer, index) =>
        parseLayerSummary(layer, `${path}.layers.${index}`),
      ),
    };
  }

  if (kind === 'benchmark-manifest') {
    const detail: BenchmarkManifestDetailRecord = {
      kind,
      source_path: requireString(record.source_path, `${path}.source_path`),
      id: requireString(record.id, `${path}.id`),
      outcome_space: parseOutcomeSpace(record.outcome_space, `${path}.outcome_space`),
    };
    if (record.observation_ids !== undefined) {
      detail.observation_ids = parseStringArray(record.observation_ids, `${path}.observation_ids`);
    }
    return detail;
  }

  if (kind === 'measurement') {
    return {
      kind,
      source_path: requireString(record.source_path, `${path}.source_path`),
      id: requireString(record.id, `${path}.id`),
      benchmark_id: requireString(record.benchmark_id, `${path}.benchmark_id`),
      observation_id: requireString(record.observation_id, `${path}.observation_id`),
      outcome_space: parseOutcomeSpace(record.outcome_space, `${path}.outcome_space`),
      accepted_event: parseAcceptedEvent(record.accepted_event, `${path}.accepted_event`),
      probability_measure: parseProbabilityMeasure(
        record.probability_measure,
        `${path}.probability_measure`,
      ),
    };
  }

  throw new ConsoleArtifactDetailError(`${path}.kind: unsupported artifact detail kind`);
}

function parseLayerSummary(value: unknown, path: string): LayerSummaryRecord {
  const record = requireRecord(value, path);
  const layer: LayerSummaryRecord = {
    kind: requireString(record.kind, `${path}.kind`),
  };
  if (record.parameters !== undefined) {
    layer.parameters = parseParameters(record.parameters, `${path}.parameters`);
  }
  return layer;
}

function parseOutcomeSpace(value: unknown, path: string): OutcomeSpaceSummaryRecord {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    outcomes: requireArray(record.outcomes, `${path}.outcomes`).map((outcome, index) => {
      const outcomeRecord = requireRecord(outcome, `${path}.outcomes.${index}`);
      return { id: requireString(outcomeRecord.id, `${path}.outcomes.${index}.id`) };
    }),
  };
}

function parseAcceptedEvent(
  value: unknown,
  path: string,
): MeasurementDetailRecord['accepted_event'] {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    outcome_space_id: requireString(record.outcome_space_id, `${path}.outcome_space_id`),
    outcomes: parseStringArray(record.outcomes, `${path}.outcomes`),
  };
}

function parseProbabilityMeasure(
  value: unknown,
  path: string,
): MeasurementDetailRecord['probability_measure'] {
  const record = requireRecord(value, path);
  return {
    id: requireString(record.id, `${path}.id`),
    outcome_space_id: requireString(record.outcome_space_id, `${path}.outcome_space_id`),
    probabilities: requireArray(record.probabilities, `${path}.probabilities`).map(
      (probability, index) => parseProbability(probability, `${path}.probabilities.${index}`),
    ),
  };
}

function parseProbability(value: unknown, path: string): ProbabilitySummaryRecord {
  const record = requireRecord(value, path);
  return {
    outcome_id: requireString(record.outcome_id, `${path}.outcome_id`),
    probability: requireNumber(record.probability, `${path}.probability`),
  };
}

function parseParameters(value: unknown, path: string): Record<string, string | number | boolean> {
  const record = requireRecord(value, path);
  const parameters: Record<string, string | number | boolean> = {};
  for (const [key, parameterValue] of Object.entries(record)) {
    if (
      typeof parameterValue !== 'string' &&
      typeof parameterValue !== 'number' &&
      typeof parameterValue !== 'boolean'
    ) {
      throw new ConsoleArtifactDetailError(`${path}.${key}: expected scalar parameter`);
    }
    parameters[key] = parameterValue;
  }
  return parameters;
}

function parseNumberArray(value: unknown, path: string): number[] {
  return requireArray(value, path).map((item, index) => requireNumber(item, `${path}.${index}`));
}

function parseStringArray(value: unknown, path: string): string[] {
  return requireArray(value, path).map((item, index) => requireString(item, `${path}.${index}`));
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ConsoleArtifactDetailError(`${path}: expected record`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new ConsoleArtifactDetailError(`${path}: expected array`);
  }
  return value;
}

function requireNumber(value: unknown, path: string): number {
  if (typeof value !== 'number') {
    throw new ConsoleArtifactDetailError(`${path}: expected number`);
  }
  return value;
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string') {
    throw new ConsoleArtifactDetailError(`${path}: expected string`);
  }
  return value;
}
