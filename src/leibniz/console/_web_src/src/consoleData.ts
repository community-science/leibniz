import {
  parseConsoleArtifactIndexRecord,
  type ConsoleArtifactIndexRecord,
} from './artifactIndex.ts';
import {
  parseConsoleArtifactDetailRecords,
  type ConsoleArtifactDetailMap,
} from './artifactDetails.ts';
import {
  parseObservationInspectionRecords,
  type ObservationInspectionRecord,
} from './observationInspections.ts';
import { parsePerformanceViewRecords, type PerformanceViewRecord } from './performanceViews.ts';
import { parseModelInspectionRecords, type ModelInspectionRecord } from './modelInspections.ts';
import {
  parseImportedResultViewRecords,
  type ImportedResultViewRecord,
} from './resultViews.ts';
import { parseSourceModuleRecords, type SourceModuleRecord } from './sourceModules.ts';

export type ConsoleDataRecord = {
  format: 'leibniz.console-data';
  format_version: 1;
  artifact_index: ConsoleArtifactIndexRecord;
  artifact_details: ConsoleArtifactDetailMap;
  observation_inspections: ObservationInspectionRecord[];
  performance_views: PerformanceViewRecord[];
  result_views: ImportedResultViewRecord[];
  model_inspections: ModelInspectionRecord[];
  source_modules: SourceModuleRecord[];
};

export class ConsoleDataTransportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConsoleDataTransportError';
  }
}

export function parseConsoleDataRecord(value: unknown): ConsoleDataRecord {
  const record = requireRecord(value, 'console data');
  const format = requireLiteral(record.format, 'format', 'leibniz.console-data');
  const formatVersion = requireLiteral(record.format_version, 'format_version', 1);
  const artifactIndex = parseConsoleArtifactIndexRecord(record.artifact_index);
  const artifactDetails = parseConsoleArtifactDetailRecords(record.artifact_details);
  const observationInspections = parseObservationInspectionRecords(record.observation_inspections);
  const performanceViews = parsePerformanceViewRecords(record.performance_views);
  const resultViews = parseImportedResultViewRecords(record.result_views);
  const modelInspections = parseModelInspectionRecords(record.model_inspections);
  const sourceModules = parseSourceModuleRecords(record.source_modules);

  return {
    format,
    format_version: formatVersion,
    artifact_index: artifactIndex,
    artifact_details: artifactDetails,
    observation_inspections: observationInspections,
    performance_views: performanceViews,
    result_views: resultViews,
    model_inspections: modelInspections,
    source_modules: sourceModules,
  };
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new ConsoleDataTransportError(`${path}: expected record`);
  }
  return value as Record<string, unknown>;
}

function requireLiteral<const Literal extends string | number>(
  value: unknown,
  path: string,
  expected: Literal,
): Literal {
  if (value !== expected) {
    throw new ConsoleDataTransportError(`${path}: expected ${String(expected)}`);
  }
  return expected;
}
