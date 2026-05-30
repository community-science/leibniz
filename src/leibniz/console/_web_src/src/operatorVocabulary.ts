import { requireArray, requireLiteral, requireRecord } from './transport.ts';

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

const error = (message: string) => new OperatorVocabularyError(message);

export function parseOperatorVocabularyRecord(value: unknown): OperatorVocabularyRecord {
  const record = requireRecord(value, 'operator vocabulary', error);
  requireLiteral(
    record.format,
    'operator vocabulary.format',
    'leibniz.model-operator-vocabulary',
    error,
  );
  requireLiteral(record.format_version, 'operator vocabulary.format_version', 1, error);
  requireArray(record.operators, 'operator vocabulary.operators', error);
  requireArray(record.syntax_aliases, 'operator vocabulary.syntax_aliases', error);
  requireArray(record.coordinate_descriptors, 'operator vocabulary.coordinate_descriptors', error);
  requireArray(record.program_effects, 'operator vocabulary.program_effects', error).forEach(
    (effect, index) =>
      parseProgramEffect(effect, `operator vocabulary.program_effects.${index}`),
  );
  requireRecord(record.descriptor_axes, 'operator vocabulary.descriptor_axes', error);
  return record as unknown as OperatorVocabularyRecord;
}

export function parseProgramEffect(value: unknown, path: string): ProgramEffectVocabularyEntry {
  return requireRecord(value, path, error) as unknown as ProgramEffectVocabularyEntry;
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
