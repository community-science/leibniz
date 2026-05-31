import { parseConsoleDataRecord } from '../src/leibniz/console/_web_src/src/consoleData.ts';
import {
  coordinateDisplayName,
  descriptorAxisDisplayName,
  descriptorValueDisplayName,
  operatorDisplayName,
  parameterDisplayName,
  syntaxAliasDisplayName,
} from '../src/leibniz/console/_web_src/src/operatorVocabulary.ts';

declare const consoleDataPayload: unknown;

const parsed = parseConsoleDataRecord(consoleDataPayload);
const rawConsoleData = consoleDataPayload as Record<string, unknown>;
const modelInspection = parsed.model_inspections[0];
if (modelInspection === undefined) {
  throw new Error('expected model inspection fixture');
}

assertEqual(parsed.format, 'leibniz.console-data', 'format');
assertEqual(parsed.format_version, 1, 'format version');
assertEqual(parsed.result_views.length, 0, 'result view count');
assertEqual(parsed.model_inspections.length, 1, 'model inspection count');
assertEqual(parsed.benchmark_tasks.length, 1, 'benchmark task count');
assertEqual(
  parsed.operator_vocabulary.operators.map((operator) => operator.kind).join(','),
  'local-aggregation,rank-collapse,affine-readout',
  'operator vocabulary order',
);
assertEqual(
  parsed.operator_vocabulary.program_effects.map((effect) => effect.kind).join(','),
  'branch,merge,route,repeat,identity-path,parameter-sharing',
  'program effect vocabulary order',
);
assertEqual(
  operatorDisplayName(parsed.operator_vocabulary, 'local-aggregation'),
  'Local aggregation',
  'operator vocabulary operator display',
);
assertEqual(
  syntaxAliasDisplayName(parsed.operator_vocabulary, 'adaptive-pooling'),
  'Local aggregation',
  'operator vocabulary syntax display',
);
assertEqual(
  parameterDisplayName(parsed.operator_vocabulary, 'local-aggregation', 'size'),
  'Output support size',
  'operator vocabulary parameter display',
);
assertEqual(
  descriptorValueDisplayName(parsed.operator_vocabulary, 'support', 'local-window'),
  'Local window',
  'operator vocabulary descriptor display',
);
assertEqual(
  descriptorAxisDisplayName(parsed.operator_vocabulary, 'support'),
  'Support',
  'operator vocabulary descriptor axis display',
);
assertEqual(
  coordinateDisplayName(parsed.operator_vocabulary, 'operator.0.local_support_size'),
  'Local support size',
  'operator vocabulary coordinate display',
);
assertEqual(
  modelInspection.cost_summary.parameter_count,
  50,
  'model inspection parameter count',
);
assertEqual(
  modelInspection.cost_summary.parameter_bytes,
  200,
  'model inspection parameter bytes',
);
assertEqual(
  modelInspection.cost_summary.inference_flops,
  1104,
  'model inspection flops',
);
assertEqual(
  modelInspection.components.map((component) => component.kind).join(','),
  'adaptive-pooling,flatten,dense',
  'model inspection components',
);
assertEqual(
  modelInspection.components.map((component) => component.operator?.kind).join(','),
  'local-aggregation,rank-collapse,affine-readout',
  'model operator summaries',
);
assertEqual(
  `${modelInspection.architecture_graph.nodes.length}:${modelInspection.architecture_graph.edges.length}`,
  '3:2',
  'model inspection architecture graph',
);
assertEqual(
  `${modelInspection.architecture_summary.component_count}:${modelInspection.architecture_summary.edge_count}`,
  '3:2',
  'model inspection graph summary',
);
assertEqual(
  modelInspection.architecture_summary.component_kinds.join(','),
  'adaptive-pooling,flatten,dense',
  'model inspection graph summary components',
);
assertEqual(
  modelInspection.node_evidence.map((evidence) => evidence.node_path.join('/')).join(','),
  'component-0,component-1,component-2',
  'model inspection node evidence paths',
);
assertEqual(
  modelInspection.architecture_trace.stages.map((stage) => stage.operator_kind).join(','),
  'local-aggregation,rank-collapse,affine-readout',
  'model architecture trace stages',
);
assertEqual(
  modelInspection.architecture_trace.stages[0]?.descriptor_axes.support,
  'local-window',
  'model architecture trace support axis',
);
const benchmarkTask = parsed.benchmark_tasks[0];
if (benchmarkTask === undefined) {
  throw new Error('expected benchmark task');
}
assertEqual(benchmarkTask?.kind, 'generated-observations', 'benchmark task kind');
assertEqual(benchmarkTask?.batches.length, 1, 'benchmark batch count');
assertEqual(
  benchmarkTask?.batches.map((batch) => `${batch.mode}:${batch.scale}:${batch.sample_count}`).join('|'),
  'balanced:1:40',
  'generated benchmark batches',
);
const generatedSamples = benchmarkTask?.batches[0]?.samples ?? [];
assertEqual(
  scaleCounts(generatedSamples).join(','),
  '5,5,5,5,5,5,5,5',
  'balanced scale samples',
);
assertEqual(
  digitCounts(generatedSamples).join(','),
  '18,18,18,18,18,18,18,18,18,18',
  'balanced digit counts',
);
assertEqual(
  new Set(generatedSamples.map((sample) => sample.field_shape.join('x'))).size,
  generatedSamples.length,
  'independent sample canvas shapes',
);
assertEqual(
  benchmarkTask?.batches[0]?.presentation.sample_card_density,
  'standard',
  'sample presentation density',
);
assertEqual(
  benchmarkTask?.batches[0]?.presentation.aggregate_mode,
  false,
  'sample presentation aggregate mode',
);
const generatedSample = benchmarkTask?.batches[0]?.samples[0];
if (generatedSample === undefined) {
  throw new Error('expected generated sample');
}
assertEqual(generatedSample.outcome_id.startsWith('digit-'), true, 'sample outcome id');
assertEqual(generatedSample.field_shape.join('x'), '1x63x45', 'sample field shape');
assertEqual(
  [
    generatedSample.preview_crop.left,
    generatedSample.preview_crop.top,
    generatedSample.preview_crop.size,
  ].join(','),
  '10,23,27',
  'sample preview crop',
);
assertEqual(
  generatedSample.latent_coordinates.map((coordinate) => coordinate.role).join(','),
  'content,variation,materialization',
  'sample latent roles',
);
assertEqual(
  generatedSample.latent_coordinates.find((coordinate) => coordinate.role === 'content')?.multiplicity,
  1,
  'sample content multiplicity',
);
const variationCoordinate = generatedSample.latent_coordinates.find(
  (coordinate) => coordinate.role === 'variation',
);
const variationValues = variationCoordinate?.values as Record<string, unknown> | undefined;
assertEqual(
  variationValues?.kind,
  'field-variation-transform-samples',
  'sample variation values kind',
);
assertEqual(
  (variationValues?.bounds as Record<string, unknown> | undefined)?.kind,
  'field-variation-transform',
  'sample variation bounds kind',
);
assertEqual(
  Array.isArray(variationValues?.coordinates) ? variationValues.coordinates.length : 0,
  1,
  'sample variation coordinate count',
);
const materializationPlan = generatedSample.materialization_plan as Record<string, unknown>;
assertEqual(assignmentLabel(materializationPlan.scale_assignment), 'L=1', 'sample scale assignment');
assertEqual(
  assignmentLabel(materializationPlan.complexity_assignment),
  'C=1',
  'sample complexity assignment',
);
assertEqual(
  assignmentLabel(materializationPlan.resolution_assignment),
  'H=63,W=45',
  'sample resolution assignment',
);
assertDataError(
  () => parseConsoleDataRecord({ ...parsed, format_version: 2 }),
  'format_version: expected 1',
);
assertDataError(
  () =>
    parseConsoleDataRecord({
      ...rawConsoleData,
      model_inspections: [{ ...modelInspection, components: [{ index: '0' }] }],
    }),
  'model inspections.0.components.0.index: expected number',
);
assertDataError(
  () =>
    parseConsoleDataRecord({
      ...rawConsoleData,
      benchmark_tasks: [
        {
          ...benchmarkTask,
          batches: [{ ...benchmarkTask.batches[0], samples: [] }],
        },
      ],
    }),
  'benchmark tasks.0.batches.0.sample_count: expected sample length',
);

function assertEqual(actual: unknown, expected: unknown, label: string) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

function assertDataError(callback: () => void, expectedMessage: string) {
  try {
    callback();
  } catch (error) {
    if (error instanceof Error && error.message === expectedMessage) {
      return;
    }
    throw error;
  }

  throw new Error(`expected console data transport error: ${expectedMessage}`);
}

function digitCounts(samples: { component_sequence: number[] }[]): number[] {
  const counts = new Map<number, number>();
  for (const sample of samples) {
    for (const digit of sample.component_sequence) {
      counts.set(digit, (counts.get(digit) ?? 0) + 1);
    }
  }
  return Array.from(counts.entries())
    .sort(([left], [right]) => left - right)
    .map(([, count]) => count);
}

function scaleCounts(samples: { component_sequence: number[] }[]): number[] {
  const counts = new Map<number, number>();
  for (const sample of samples) {
    const scale = sample.component_sequence.length;
    counts.set(scale, (counts.get(scale) ?? 0) + 1);
  }
  return Array.from(counts.entries())
    .sort(([left], [right]) => left - right)
    .map(([, count]) => count);
}

function assignmentLabel(value: unknown): string {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    return 'unknown';
  }
  const values = (value as Record<string, unknown>).values;
  if (!Array.isArray(values)) {
    return 'unknown';
  }
  return values
    .map((entry) => {
      if (typeof entry !== 'object' || entry === null || Array.isArray(entry)) {
        return null;
      }
      const record = entry as Record<string, unknown>;
      return `${String(record.axis)}=${String(record.value)}`;
    })
    .filter((entry): entry is string => entry !== null)
    .join(',');
}
