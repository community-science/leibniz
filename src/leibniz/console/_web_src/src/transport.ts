export type ErrorFactory = (message: string) => Error;

export function requireRecord(
  value: unknown,
  path: string,
  error: ErrorFactory,
): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw error(`${path}: expected record`);
  }
  return value as Record<string, unknown>;
}

export function requireArray(
  value: unknown,
  path: string,
  error: ErrorFactory,
): unknown[] {
  if (!Array.isArray(value)) {
    throw error(`${path}: expected array`);
  }
  return value;
}

export function requireString(value: unknown, path: string, error: ErrorFactory): string {
  if (typeof value !== 'string') {
    throw error(`${path}: expected string`);
  }
  return value;
}

export function requireNumber(value: unknown, path: string, error: ErrorFactory): number {
  if (typeof value !== 'number') {
    throw error(`${path}: expected number`);
  }
  return value;
}

export function requireLiteral<const Literal extends string | number>(
  value: unknown,
  path: string,
  expected: Literal,
  error: ErrorFactory,
): Literal {
  if (value !== expected) {
    throw error(`${path}: expected ${String(expected)}`);
  }
  return expected;
}

export function optionalString(
  value: unknown,
  path: string,
  error: ErrorFactory,
): string | undefined {
  return value === undefined ? undefined : requireString(value, path, error);
}

export function optionalNumber(
  value: unknown,
  path: string,
  error: ErrorFactory,
): number | undefined {
  return value === undefined ? undefined : requireNumber(value, path, error);
}
