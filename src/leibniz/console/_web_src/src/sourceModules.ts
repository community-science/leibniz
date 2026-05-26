export type SourceModuleRecord = {
  module_name: string;
  source_path: string;
  public_exports: string[];
  validation_commands: string[];
};

export class SourceModuleTransportError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SourceModuleTransportError';
  }
}

export function parseSourceModuleRecords(value: unknown): SourceModuleRecord[] {
  return requireArray(value, 'source_modules').map((record, index) =>
    parseSourceModuleRecord(record, `source_modules.${index}`),
  );
}

function parseSourceModuleRecord(value: unknown, path: string): SourceModuleRecord {
  const record = requireRecord(value, path);
  return {
    module_name: requireString(record.module_name, `${path}.module_name`),
    source_path: requireString(record.source_path, `${path}.source_path`),
    public_exports: parseStringArray(record.public_exports, `${path}.public_exports`),
    validation_commands: parseStringArray(
      record.validation_commands,
      `${path}.validation_commands`,
    ),
  };
}

function parseStringArray(value: unknown, path: string): string[] {
  return requireArray(value, path).map((item, index) => requireString(item, `${path}.${index}`));
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new SourceModuleTransportError(`${path}: expected record`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new SourceModuleTransportError(`${path}: expected array`);
  }
  return value;
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string') {
    throw new SourceModuleTransportError(`${path}: expected string`);
  }
  return value;
}
