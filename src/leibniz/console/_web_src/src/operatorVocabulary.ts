export type OperatorVocabularyRecord = {
  format: 'leibniz.model-operator-vocabulary';
  format_version: 1;
  operators: OperatorVocabularyEntry[];
  descriptor_axis_descriptors: OperatorDescriptorAxisDescriptor[];
  descriptor_axes: Record<string, OperatorDescriptorAxisValue[]>;
  syntax_aliases: OperatorSyntaxAlias[];
  coordinate_descriptors: OperatorCoordinateDescriptor[];
  program_effects: ProgramEffectVocabularyEntry[];
};

export type OperatorVocabularyEntry = {
  kind: string;
  display_name: string;
  descriptor: Record<string, unknown>;
  syntax_aliases: string[];
  parameter_roles: OperatorParameterRole[];
};

export type OperatorParameterRole = {
  name: string;
  display_name: string;
  description: string;
  value_kind: string;
};

export type OperatorDescriptorAxisValue = {
  value: string;
  display_name: string;
};

export type OperatorDescriptorAxisDescriptor = {
  name: string;
  display_name: string;
};

export type OperatorSyntaxAlias = {
  alias: string;
  operator_kind: string;
  display_name: string;
};

export type OperatorCoordinateDescriptor = {
  name: string;
  display_name: string;
  value_kind: string;
};

export type ProgramEffectVocabularyEntry = {
  kind: string;
  input_arity_law: string;
  output_arity_law: string;
  shape_law: string;
  cost_law: string;
  trace_law: string;
};

export class OperatorVocabularyError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'OperatorVocabularyError';
  }
}

export function parseOperatorVocabularyRecord(value: unknown): OperatorVocabularyRecord {
  const record = requireRecord(value, 'operator vocabulary');
  return {
    format: requireLiteral(
      record.format,
      'operator vocabulary.format',
      'leibniz.model-operator-vocabulary',
    ),
    format_version: requireLiteral(record.format_version, 'operator vocabulary.format_version', 1),
    operators: requireArray(record.operators, 'operator vocabulary.operators').map(
      (operator, index) => parseOperator(operator, `operator vocabulary.operators.${index}`),
    ),
    descriptor_axis_descriptors: requireArray(
      record.descriptor_axis_descriptors,
      'operator vocabulary.descriptor_axis_descriptors',
    ).map((descriptor, index) =>
      parseDescriptorAxisDescriptor(
        descriptor,
        `operator vocabulary.descriptor_axis_descriptors.${index}`,
      ),
    ),
    descriptor_axes: parseDescriptorAxes(
      record.descriptor_axes,
      'operator vocabulary.descriptor_axes',
    ),
    syntax_aliases: requireArray(record.syntax_aliases, 'operator vocabulary.syntax_aliases').map(
      (alias, index) => parseSyntaxAlias(alias, `operator vocabulary.syntax_aliases.${index}`),
    ),
    coordinate_descriptors: requireArray(
      record.coordinate_descriptors,
      'operator vocabulary.coordinate_descriptors',
    ).map((descriptor, index) =>
      parseCoordinateDescriptor(descriptor, `operator vocabulary.coordinate_descriptors.${index}`),
    ),
    program_effects: requireArray(record.program_effects, 'operator vocabulary.program_effects').map(
      (effect, index) => parseProgramEffect(effect, `operator vocabulary.program_effects.${index}`),
    ),
  };
}

export function operatorDisplayName(
  vocabulary: OperatorVocabularyRecord,
  operatorKind: string | undefined,
): string {
  if (operatorKind === undefined) {
    return 'unknown';
  }
  return (
    vocabulary.operators.find((operator) => operator.kind === operatorKind)?.display_name ??
    operatorKind
  );
}

export function syntaxAliasDisplayName(
  vocabulary: OperatorVocabularyRecord,
  alias: string,
): string {
  return vocabulary.syntax_aliases.find((entry) => entry.alias === alias)?.display_name ?? alias;
}

export function parameterDisplayName(
  vocabulary: OperatorVocabularyRecord,
  operatorKind: string | undefined,
  parameterName: string,
): string {
  if (operatorKind === undefined) {
    return parameterName;
  }
  const operator = vocabulary.operators.find((entry) => entry.kind === operatorKind);
  return (
    operator?.parameter_roles.find((role) => role.name === parameterName)?.display_name ??
    parameterName
  );
}

export function descriptorValueDisplayName(
  vocabulary: OperatorVocabularyRecord,
  axis: string,
  value: string,
): string {
  return (
    vocabulary.descriptor_axes[axis]?.find((entry) => entry.value === value)?.display_name ??
    value
  );
}

export function descriptorAxisDisplayName(
  vocabulary: OperatorVocabularyRecord,
  axis: string,
): string {
  return (
    vocabulary.descriptor_axis_descriptors.find((descriptor) => descriptor.name === axis)
      ?.display_name ?? axis
  );
}

export function coordinateDisplayName(
  vocabulary: OperatorVocabularyRecord,
  coordinateName: string,
): string {
  const exact = vocabulary.coordinate_descriptors.find(
    (descriptor) => descriptor.name === coordinateName,
  );
  if (exact !== undefined) {
    return exact.display_name;
  }
  const normalized = coordinateName.replace(/operator\.\d+\./, 'operator.{index}.');
  return (
    vocabulary.coordinate_descriptors.find((descriptor) => descriptor.name === normalized)
      ?.display_name ?? coordinateName
  );
}

function parseOperator(value: unknown, path: string): OperatorVocabularyEntry {
  const record = requireRecord(value, path);
  return {
    kind: requireString(record.kind, `${path}.kind`),
    display_name: requireString(record.display_name, `${path}.display_name`),
    descriptor: requireRecord(record.descriptor, `${path}.descriptor`),
    syntax_aliases: parseStringArray(record.syntax_aliases, `${path}.syntax_aliases`),
    parameter_roles: requireArray(record.parameter_roles, `${path}.parameter_roles`).map(
      (role, index) => parseParameterRole(role, `${path}.parameter_roles.${index}`),
    ),
  };
}

function parseParameterRole(value: unknown, path: string): OperatorParameterRole {
  const record = requireRecord(value, path);
  return {
    name: requireString(record.name, `${path}.name`),
    display_name: requireString(record.display_name, `${path}.display_name`),
    description: requireString(record.description, `${path}.description`),
    value_kind: requireString(record.value_kind, `${path}.value_kind`),
  };
}

function parseDescriptorAxes(value: unknown, path: string): Record<string, OperatorDescriptorAxisValue[]> {
  const record = requireRecord(value, path);
  return Object.fromEntries(
    Object.entries(record).map(([axis, values]) => [
      axis,
      requireArray(values, `${path}.${axis}`).map((entry, index) =>
        parseAxisValue(entry, `${path}.${axis}.${index}`),
      ),
    ]),
  );
}

function parseAxisValue(value: unknown, path: string): OperatorDescriptorAxisValue {
  const record = requireRecord(value, path);
  return {
    value: requireString(record.value, `${path}.value`),
    display_name: requireString(record.display_name, `${path}.display_name`),
  };
}

function parseDescriptorAxisDescriptor(
  value: unknown,
  path: string,
): OperatorDescriptorAxisDescriptor {
  const record = requireRecord(value, path);
  return {
    name: requireString(record.name, `${path}.name`),
    display_name: requireString(record.display_name, `${path}.display_name`),
  };
}

function parseSyntaxAlias(value: unknown, path: string): OperatorSyntaxAlias {
  const record = requireRecord(value, path);
  return {
    alias: requireString(record.alias, `${path}.alias`),
    operator_kind: requireString(record.operator_kind, `${path}.operator_kind`),
    display_name: requireString(record.display_name, `${path}.display_name`),
  };
}

function parseCoordinateDescriptor(value: unknown, path: string): OperatorCoordinateDescriptor {
  const record = requireRecord(value, path);
  return {
    name: requireString(record.name, `${path}.name`),
    display_name: requireString(record.display_name, `${path}.display_name`),
    value_kind: requireString(record.value_kind, `${path}.value_kind`),
  };
}

function parseProgramEffect(value: unknown, path: string): ProgramEffectVocabularyEntry {
  const record = requireRecord(value, path);
  return {
    kind: requireString(record.kind, `${path}.kind`),
    input_arity_law: requireString(record.input_arity_law, `${path}.input_arity_law`),
    output_arity_law: requireString(record.output_arity_law, `${path}.output_arity_law`),
    shape_law: requireString(record.shape_law, `${path}.shape_law`),
    cost_law: requireString(record.cost_law, `${path}.cost_law`),
    trace_law: requireString(record.trace_law, `${path}.trace_law`),
  };
}

function parseStringArray(value: unknown, path: string): string[] {
  return requireArray(value, path).map((item, index) => requireString(item, `${path}.${index}`));
}

function requireRecord(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new OperatorVocabularyError(`${path}: expected record`);
  }
  return value as Record<string, unknown>;
}

function requireArray(value: unknown, path: string): unknown[] {
  if (!Array.isArray(value)) {
    throw new OperatorVocabularyError(`${path}: expected array`);
  }
  return value;
}

function requireLiteral<const Literal extends string | number>(
  value: unknown,
  path: string,
  expected: Literal,
): Literal {
  if (value !== expected) {
    throw new OperatorVocabularyError(`${path}: expected ${String(expected)}`);
  }
  return expected;
}

function requireString(value: unknown, path: string): string {
  if (typeof value !== 'string') {
    throw new OperatorVocabularyError(`${path}: expected string`);
  }
  return value;
}
