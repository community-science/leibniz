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
const modelInspection = parsed.model_inspections.find(
  (inspection) => inspection.source_path === 'tests/fixtures/architecture/digits_pool.json',
);
if (modelInspection === undefined) {
  throw new Error('expected model inspection fixture');
}

assertEqual(parsed.format, 'leibniz.console-data', 'format');
assertEqual(parsed.format_version, 1, 'format version');
assertEqual(parsed.artifact_index.format, 'leibniz.console.artifact-index', 'artifact index format');
assertEqual(parsed.artifact_index.format_version, 1, 'artifact index format version');
assertEqual(parsed.artifact_index.artifacts.length, 8, 'artifact index count');
assertEqual(parsed.artifact_details.length, 8, 'artifact detail count');
assertEqual(
  parsed.artifact_details.map((detail) => `${detail.kind}:${detail.source_path}`).join('|'),
  parsed.artifact_index.artifacts.map((artifact) => `${artifact.kind}:${artifact.source_path}`).join('|'),
  'artifact details align with artifact index',
);
assertEqual(parsed.result_views.length, 0, 'result view count');
assertEqual(parsed.model_inspections.length, 3, 'model inspection count');
assertEqual(parsed.benchmark_tasks.length, 2, 'benchmark task count');
assertEqual(
  parsed.operator_vocabulary.operators.map((operator) => operator.kind).join(','),
  'local-aggregation,local-affine,fixed-support-affine,rank-collapse,affine-readout',
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
  operatorDisplayName(parsed.operator_vocabulary, 'local-affine'),
  'Local affine',
  'operator vocabulary local affine display',
);
assertEqual(
  syntaxAliasDisplayName(parsed.operator_vocabulary, 'convolution'),
  'Local affine',
  'operator vocabulary convolution syntax display',
);
assertEqual(
  (parsed.operator_vocabulary.syntax_aliases.find((entry) => entry.alias === 'convolution')
    ?.specialization as Record<string, unknown> | undefined)?.kind,
  'local-affine',
  'operator vocabulary convolution specialization',
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
  modelInspection.cost_summary.storage_bytes,
  200,
  'model inspection storage bytes',
);
assertEqual(
  modelInspection.cost_summary.inference_compute,
  656,
  'model inspection compute',
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
const benchmarkTask = parsed.benchmark_tasks.find(
  (task) => task.benchmark_id === 'benchmarks.digits@0.1.0',
);
if (benchmarkTask === undefined) {
  throw new Error('expected digits benchmark task');
}
const chessBenchmarkTask = parsed.benchmark_tasks.find(
  (task) => task.benchmark_id === 'benchmarks.chess@0.1.0',
);
if (chessBenchmarkTask === undefined) {
  throw new Error('expected chess benchmark task');
}
assertEqual(benchmarkTask?.kind, 'generated-observations', 'benchmark task kind');
assertEqual(benchmarkTask?.batches.length, 3, 'benchmark batch count');
assertEqual(
  benchmarkTask?.batches.map((batch) => `${batch.mode}:${batch.sample_count}`).join('|'),
  'complexity-window:8|complexity-window:32|complexity-window:50',
  'generated benchmark batches',
);
const generatedSamples = benchmarkTask?.batches[0]?.samples ?? [];
const generatedComponentIndices = generatedSamples.map(requiredComponentIndex);
assertEqual(
  generatedComponentIndices.length,
  8,
  'generated digit sample count',
);
assertEqual(
  new Set(generatedComponentIndices.map((sample) => sample.component_index)).size,
  generatedComponentIndices.length,
  'generated digit label coverage',
);
assertEqual(
  new Set(generatedSamples.map((sample) => requiredFieldShape(sample).join('x'))).size,
  1,
  'canonical sample canvas shape',
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
assertEqual(
  benchmarkTask?.batches[0]?.region?.ambient.field_domain_kind,
  'lattice-2d',
  'digits batch region ambient',
);
assertEqual(
  benchmarkTask?.batches[0]?.request_outcome?.kind,
  'realized',
  'digits batch request outcome',
);
const generatedSample = benchmarkTask?.batches[1]?.samples[0];
if (generatedSample === undefined) {
  throw new Error('expected generated sample');
}
assertEqual(generatedSample.outcome_id.startsWith('digit-'), true, 'sample outcome id');
assertEqual(requiredFieldShape(generatedSample).join('x'), '1x16x16', 'sample field shape');
assertEqual(
  Object.hasOwn(generatedSample, 'preview_crop'),
  false,
  'sample preview crop omitted',
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
  'constructed-field-variation-transform-samples',
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
const generatedVariationCoordinate = Array.isArray(variationValues?.coordinates)
  ? variationValues.coordinates[0] as Record<string, unknown>
  : undefined;
assertEqual(
  Object.hasOwn(generatedVariationCoordinate ?? {}, 'constructed_affine_indices'),
  true,
  'sample variation constructed affine indices',
);
assertEqual(
  Object.hasOwn(generatedVariationCoordinate ?? {}, 'constructed_affine_parameters'),
  true,
  'sample variation constructed affine parameters',
);
assertEqual(
  Object.hasOwn(variationValues ?? {}, 'observable_state_id'),
  false,
  'sample omits observable state id',
);
assertEqual(
  Object.hasOwn(variationValues ?? {}, 'target_distribution'),
  false,
  'sample omits target distribution',
);
const materializationPlan = generatedSample.materialization_plan as Record<string, unknown>;
assertEqual(
  assignmentLabel(materializationPlan.resolution_assignment),
  'H=16,W=16',
  'sample resolution assignment',
);
const chessSample = chessBenchmarkTask.batches[0]?.samples[0];
if (chessSample === undefined) {
  throw new Error('expected chess generated sample');
}
assertEqual(
  chessBenchmarkTask.batches[0]?.complexity_cardinalities?.[0],
  8,
  'chess sample cardinality',
);
assertEqual(
  chessBenchmarkTask.batches[0]?.region?.components.length,
  8,
  'chess region component count',
);
assertEqual(
  chessBenchmarkTask.batches[0]?.region?.ambient.field_codomain_id,
  'piece-occupancy',
  'chess region codomain',
);
assertEqual(
  chessSample.available_outcome_ids?.length,
  2,
  'chess sample legal move count',
);
assertEqual(
  chessSample.region_component_index,
  0,
  'chess sample region component index',
);
assertEqual(
  typeof chessSample.axis_coordinates,
  'object',
  'chess sample axis coordinates',
);
assertEqual(
  chessSample.image_data_url?.startsWith('data:image/svg+xml;base64,'),
  true,
  'chess sample image data url',
);
assertEqual(
  chessSample.image_overlay?.kind,
  'grid-move-highlights',
  'chess sample image overlay kind',
);
assertEqual(
  chessSample.image_overlay?.moves.length,
  chessSample.available_outcome_ids?.length,
  'chess sample legal move overlay count',
);
assertEqual(
  chessSample.image_overlay?.moves.every(
    (move) => move.from.length === 2 && move.to.length === 2,
  ),
  true,
  'chess sample legal move overlay coordinates',
);
assertEqual(
  chessSample.image_overlay?.moves.some((move) => (move.target_probability ?? 0) > 0),
  true,
  'chess sample target move overlay',
);
assertEqual(
  chessSample.field_shape,
  undefined,
  'chess sample omits field shape',
);
assertEqual(
  chessSample.observable_state_id?.startsWith('fen:'),
  true,
  'chess sample observable state id',
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
          batches: benchmarkTask.batches.map((batch, index) =>
            index === 2 ? { ...batch, samples: [] } : batch,
          ),
        },
      ],
    }),
  'benchmark tasks.0.batches.2.sample_count: expected sample length',
);

function assertEqual(actual: unknown, expected: unknown, label: string) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

function requiredFieldShape(sample: { field_shape?: number[] }): number[] {
  if (sample.field_shape === undefined) {
    throw new Error('expected generated field shape');
  }
  return sample.field_shape;
}

function requiredComponentIndex(sample: { component_index?: number }): { component_index: number } {
  if (sample.component_index === undefined) {
    throw new Error('expected generated component index');
  }
  return { component_index: sample.component_index };
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
