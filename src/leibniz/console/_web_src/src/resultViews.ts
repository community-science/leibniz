export type ImportedResultViewRecord = {
  format: 'leibniz.console.imported-results';
  format_version: 1;
  source_path: string;
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

export function parseImportedResultViewRecords(value: unknown): ImportedResultViewRecord[] {
  return requireArray(value, 'result_views').map((view, index) =>
    parseImportedResultViewRecord(view, `result_views.${index}`),
  );
}

function parseImportedResultViewRecord(value: unknown, path: string): ImportedResultViewRecord {
  const record = requireRecord(value, path);
  const format = requireLiteral(record.format, `${path}.format`, 'leibniz.console.imported-results');
  const formatVersion = requireLiteral(record.format_version, `${path}.format_version`, 1);
  const sourcePath = requireString(record.source_path, `${path}.source_path`);
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
    publication_bundles: publicationBundles,
  };
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
