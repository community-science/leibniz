import {
  consoleProtocolFormats,
  consoleProtocolFormatVersions,
} from './generated/protocolVocabulary.ts';
import { requireArray, requireLiteral, requireRecord, requireString } from './transport.ts';

export type ArtifactReferenceRecord = {
  kind: string;
  protocol_id?: string;
  content_digest?: string;
  record_digest?: string;
  external_uri?: string;
};

export type ConsoleArtifactIndexEntryRecord = {
  kind: string;
  source_path: string;
  digest: string;
  reference: ArtifactReferenceRecord;
  dependencies: ArtifactReferenceRecord[];
  validation_status: 'valid';
  validation_command: string;
  protocol_id?: string;
};

export type ConsoleArtifactIndexRecord = {
  format: typeof consoleProtocolFormats.artifactIndex;
  format_version: typeof consoleProtocolFormatVersions.artifactIndex;
  artifacts: ConsoleArtifactIndexEntryRecord[];
};

export class ConsoleArtifactIndexTransportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ConsoleArtifactIndexTransportError';
  }
}

const error = (message: string) => new ConsoleArtifactIndexTransportError(message);

export function parseConsoleArtifactIndexRecord(value: unknown): ConsoleArtifactIndexRecord {
  const record = requireRecord(value, 'console artifact index', error);
  const format = requireLiteral(
    record.format,
    'format',
    consoleProtocolFormats.artifactIndex,
    error,
  );
  const formatVersion = requireLiteral(
    record.format_version,
    'format_version',
    consoleProtocolFormatVersions.artifactIndex,
    error,
  );
  const artifacts = requireArray(record.artifacts, 'artifacts', error).map((artifact, index) =>
    parseEntryRecord(artifact, `artifacts.${index}`),
  );

  return {
    format,
    format_version: formatVersion,
    artifacts,
  };
}

function parseEntryRecord(value: unknown, path: string): ConsoleArtifactIndexEntryRecord {
  const record = requireRecord(value, path, error);
  const entry: ConsoleArtifactIndexEntryRecord = {
    kind: requireString(record.kind, `${path}.kind`, error),
    source_path: requireString(record.source_path, `${path}.source_path`, error),
    digest: requireString(record.digest, `${path}.digest`, error),
    reference: parseReferenceRecord(record.reference, `${path}.reference`),
    dependencies: requireArray(record.dependencies, `${path}.dependencies`, error).map(
      (dependency, index) => parseReferenceRecord(dependency, `${path}.dependencies.${index}`),
    ),
    validation_status: requireLiteral(
      record.validation_status,
      `${path}.validation_status`,
      'valid',
      error,
    ),
    validation_command: requireString(record.validation_command, `${path}.validation_command`, error),
  };

  if (record.protocol_id !== undefined) {
    entry.protocol_id = requireString(record.protocol_id, `${path}.protocol_id`, error);
  }

  return entry;
}

function parseReferenceRecord(value: unknown, path: string): ArtifactReferenceRecord {
  const record = requireRecord(value, path, error);
  const reference: ArtifactReferenceRecord = {
    kind: requireString(record.kind, `${path}.kind`, error),
  };

  assignOptionalString(reference, 'protocol_id', record.protocol_id, path);
  assignOptionalString(reference, 'content_digest', record.content_digest, path);
  assignOptionalString(reference, 'record_digest', record.record_digest, path);
  assignOptionalString(reference, 'external_uri', record.external_uri, path);

  return reference;
}

function assignOptionalString(
  target: ArtifactReferenceRecord,
  key: keyof Omit<ArtifactReferenceRecord, 'kind'>,
  value: unknown,
  path: string,
) {
  if (value !== undefined) {
    target[key] = requireString(value, `${path}.${key}`, error);
  }
}
