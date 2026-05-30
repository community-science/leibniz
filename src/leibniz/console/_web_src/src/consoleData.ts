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
import { requireLiteral, requireRecord } from './transport.ts';

export type ConsoleDataRecord = {
  format: typeof consoleProtocolFormats.consoleData;
  format_version: typeof consoleProtocolFormatVersions.consoleData;
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
  const resultViews = parseResultViewRecords(record.result_views);
  const modelInspections = parseModelInspectionRecords(record.model_inspections);
  const benchmarkTasks = parseBenchmarkTaskRecords(record.benchmark_tasks);
  const operatorVocabulary = parseOperatorVocabularyRecord(record.operator_vocabulary);

  return {
    format,
    format_version: formatVersion,
    result_views: resultViews,
    model_inspections: modelInspections,
    benchmark_tasks: benchmarkTasks,
    operator_vocabulary: operatorVocabulary,
  };
}
