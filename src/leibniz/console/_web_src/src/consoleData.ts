import { parseModelInspectionRecords, type ModelInspectionRecord } from './modelInspections.ts';
import {
  parseOperatorVocabularyRecord,
  type OperatorVocabularyRecord,
} from './operatorVocabulary.ts';
import {
  parseResultViewRecords,
  type ResultViewRecord,
} from './resultViews.ts';
import { parseBenchmarkTaskRecords, type BenchmarkTaskRecord } from './benchmarkTasks.ts';
import {
  consoleProtocolFormats,
  consoleProtocolFormatVersions,
} from './generated/protocolVocabulary.ts';
import { requireArray, requireLiteral, requireRecord } from './transport.ts';

export type ArtifactIndexRecord = {
  format: typeof consoleProtocolFormats.artifactIndex;
  format_version: typeof consoleProtocolFormatVersions.artifactIndex;
  artifacts: Record<string, unknown>[];
};

export type ArtifactDetailRecord = Record<string, unknown> & {
  kind: string;
  source_path: string;
};

export type ConsoleDataRecord = {
  format: typeof consoleProtocolFormats.consoleData;
  format_version: typeof consoleProtocolFormatVersions.consoleData;
  artifact_index: ArtifactIndexRecord;
  artifact_details: ArtifactDetailRecord[];
  result_views: ResultViewRecord[];
  model_inspections: ModelInspectionRecord[];
  benchmark_tasks: BenchmarkTaskRecord[];
  operator_vocabulary: OperatorVocabularyRecord;
};

export class ConsoleDataTransportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConsoleDataTransportError';
  }
}

const error = (message: string) => new ConsoleDataTransportError(message);

export function parseConsoleDataRecord(value: unknown): ConsoleDataRecord {
  const record = requireRecord(value, 'console data', error);
  const format = requireLiteral(record.format, 'format', consoleProtocolFormats.consoleData, error);
  const formatVersion = requireLiteral(
    record.format_version,
    'format_version',
    consoleProtocolFormatVersions.consoleData,
    error,
  );
  const artifactIndex = parseArtifactIndexRecord(record.artifact_index);
  const artifactDetails = parseArtifactDetails(record.artifact_details);
  const resultViews = parseResultViewRecords(record.result_views);
  const modelInspections = parseModelInspectionRecords(record.model_inspections);
  const benchmarkTasks = parseBenchmarkTaskRecords(record.benchmark_tasks);
  const operatorVocabulary = parseOperatorVocabularyRecord(record.operator_vocabulary);

  return {
    format,
    format_version: formatVersion,
    artifact_index: artifactIndex,
    artifact_details: artifactDetails,
    result_views: resultViews,
    model_inspections: modelInspections,
    benchmark_tasks: benchmarkTasks,
    operator_vocabulary: operatorVocabulary,
  };
}

function parseArtifactIndexRecord(value: unknown): ArtifactIndexRecord {
  const record = requireRecord(value, 'artifact_index', error);
  const format = requireLiteral(
    record.format,
    'artifact_index.format',
    consoleProtocolFormats.artifactIndex,
    error,
  );
  const formatVersion = requireLiteral(
    record.format_version,
    'artifact_index.format_version',
    consoleProtocolFormatVersions.artifactIndex,
    error,
  );
  const artifacts = requireArray(record.artifacts, 'artifact_index.artifacts', error).map(
    (artifact, index) =>
      requireRecord(artifact, `artifact_index.artifacts.${index}`, error),
  );
  return {
    format,
    format_version: formatVersion,
    artifacts,
  };
}

function parseArtifactDetails(value: unknown): ArtifactDetailRecord[] {
  return requireArray(value, 'artifact_details', error).map((detail, index) => {
    const record = requireRecord(detail, `artifact_details.${index}`, error);
    if (typeof record.kind !== 'string' || record.kind === '') {
      throw error(`artifact_details.${index}.kind: expected string`);
    }
    if (typeof record.source_path !== 'string' || record.source_path === '') {
      throw error(`artifact_details.${index}.source_path: expected string`);
    }
    return record as ArtifactDetailRecord;
  });
}
