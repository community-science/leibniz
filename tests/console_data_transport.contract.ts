import { parseConsoleDataRecord } from '../src/leibniz/console/_web_src/src/consoleData.ts';
import { coordinateDisplayName } from '../src/leibniz/console/_web_src/src/operatorVocabulary.ts';

declare const consoleDataPayload: unknown;

const parsed = parseConsoleDataRecord(consoleDataPayload);
const rawConsoleData = consoleDataPayload as Record<string, unknown>;

assertEqual(parsed.format, 'leibniz.console-data', 'format');
assertEqual(parsed.format_version, 1, 'format version');
assertEqual(parsed.artifact_index.format, 'leibniz.console.artifact-index', 'artifact index format');
assertEqual(parsed.artifact_index.format_version, 1, 'artifact index format version');
assertEqual(parsed.artifact_index.artifacts.length, 5, 'artifact index count');
assertEqual(parsed.artifact_details.length, 5, 'artifact detail count');
assertEqual(
  parsed.artifact_details.map((detail) => `${detail.kind}:${detail.source_path}`).join('|'),
  parsed.artifact_index.artifacts.map((artifact) => `${artifact.kind}:${artifact.source_path}`).join('|'),
  'artifact details align with artifact index',
);
assertEqual(parsed.result_views.length, 0, 'result view count');
assertEqual(parsed.model_inspections.length, 0, 'model inspection count');
assertEqual(parsed.benchmark_tasks.length, 3, 'benchmark task count');
assertEqual(
  parsed.operator_vocabulary.operators.length,
  0,
  'operator vocabulary order',
);
assertEqual(
  coordinateDisplayName(parsed.operator_vocabulary, 'operator.0.local_support_size'),
  'operator.0.local_support_size',
  'operator vocabulary coordinate display',
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
const ksBenchmarkTask = parsed.benchmark_tasks.find(
  (task) => task.benchmark_id === 'benchmarks.ks@0.1.0',
);
if (ksBenchmarkTask === undefined) {
  throw new Error('expected KS benchmark task');
}
assertEqual(
  ksBenchmarkTask.label,
  'Kuramoto-Sivashinsky',
  'KS benchmark label uses full name',
);
assertEqual(benchmarkTask?.kind, 'generated-observations', 'benchmark task kind');
assertEqual(benchmarkTask?.batches.length, 9, 'benchmark batch count');
assertEqual(
  benchmarkTask?.batches.map((batch) => `${batch.mode}:${batch.sample_count}`).join('|'),
  [
    'volume-window:1',
    'volume-window:2',
    'volume-window:4',
    'volume-window:8',
    'volume-window:16',
    'volume-window:32',
    'volume-window:50',
    'volume-window:50',
    'volume-window:50',
  ].join('|'),
  'generated benchmark batches',
);
const generatedSamples = benchmarkTask?.batches[3]?.samples ?? [];
const generatedComponentIndices = generatedSamples.map(requiredComponentIndex);
assertEqual(
  generatedComponentIndices.length,
  8,
  'generated digit sample count',
);
if (
  new Set(generatedComponentIndices.map((sample) => sample.component_index)).size >=
  generatedComponentIndices.length
) {
  throw new Error('generated digit sampled label coverage: expected at least one repeat');
}
if (
  !generatedComponentIndices.every(
    (sample) => sample.component_index >= 0 && sample.component_index < 10,
  )
) {
  throw new Error('generated digit labels stay in vocabulary');
}
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
const generatedSample = benchmarkTask?.batches[3]?.samples[0];
if (generatedSample === undefined) {
  throw new Error('expected generated sample');
}
assertEqual(generatedSample.outcome_id.startsWith('digit-'), true, 'sample outcome id');
assertEqual(requiredFieldShape(generatedSample).join('x'), '1x36x36', 'sample field shape');
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
  Object.hasOwn(generatedVariationCoordinate ?? {}, 'transform_cell'),
  true,
  'sample variation transform cell',
);
assertEqual(
  Object.hasOwn(generatedVariationCoordinate ?? {}, 'normalized_transform'),
  true,
  'sample variation normalized transform',
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
  'H=36,W=36',
  'sample resolution assignment',
);
const chessBatch = chessBenchmarkTask.batches[3];
const chessSample = chessBatch?.samples[0];
if (chessSample === undefined) {
  throw new Error('expected chess generated sample');
}
assertEqual(
  chessBenchmarkTask.batches
    .map((batch) => batch.volumes?.[0])
    .join(','),
  '1,2,4,8,16,32,64,128,256',
  'chess sample cardinalities',
);
assertEqual(
  chessBatch?.volumes?.[0],
  8,
  'chess sample cardinality',
);
assertEqual(
  chessBatch?.region?.components.length,
  8,
  'chess region component count',
);
assertEqual(
  chessBatch?.region?.ambient.field_codomain_id,
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
const ksBatch = ksBenchmarkTask.batches[3];
const ksSample = ksBatch?.samples[0];
if (ksSample === undefined) {
  throw new Error('expected KS generated sample');
}
assertEqual(ksBenchmarkTask.kind, 'generated-observations', 'KS benchmark task kind');
assertEqual(ksBenchmarkTask.outcome_atom_count, 1, 'KS outcome atom count');
assertEqual(
  ksBenchmarkTask.batches.map((batch) => batch.volumes?.[0]).join(','),
  '1,2,4,8,16,32,64,128,256',
  'KS sample cardinalities',
);
assertEqual(
  ksBatch?.region?.ambient.field_domain_kind,
  'box-2d',
  'KS region domain kind',
);
assertEqual(
  ksBatch?.region?.ambient.field_codomain_id,
  'scalar-field',
  'KS region codomain',
);
assertEqual(
  (ksBatch?.region?.ambient.field_domain as Record<string, unknown> | undefined)?.boundary_id,
  'periodic-space-initial-time',
  'KS region boundary',
);
assertEqual(
  ksSample.outcome_id,
  'field',
  'KS sample outcome',
);
assertEqual(
  ksSample.region_component_index,
  0,
  'KS sample region component index',
);
assertEqual(
  typeof ksSample.axis_coordinates?.['ks-space-time-log2-window'],
  'number',
  'KS sample axis coordinate',
);
assertEqual(
  (ksSample.latent_coordinates[0] as Record<string, unknown> | undefined)?.chart,
  'cartesian-fourier',
  'KS sample latent chart',
);
assertEqual(
  ksSample.field_shape,
  undefined,
  'KS sample omits inline field shape',
);
assertDataError(
  () => parseConsoleDataRecord({ ...parsed, format_version: 2 }),
  'format_version: expected 1',
);
assertDataError(
  () =>
    parseConsoleDataRecord({
      ...rawConsoleData,
      model_inspections: [
        {
          components: [{ index: '0' }],
          cost_summary: {
            component_count: 1,
            unknown_cost_components: [],
            unknown_parameter_components: [],
          },
          id: 'invalid-program-inspection',
          input_shape: [1],
          model_artifacts: [],
          node_evidence: [],
          output_shape: [1],
          program: { kind: 'program-graph', record_digest: 'sha256:invalid' },
          program_graph: {
            contract_kind: 'single-input-single-output',
            edges: [],
            inputs: [{ axes: [1], name: 'input' }],
            nodes: [{ id: 'component-0', kind: 'identity', parameters: {} }],
            outputs: [{ axes: [1], name: 'output' }],
          },
          source_path: 'tests/fixtures/programs/invalid.py',
          training_provenance: [],
        },
      ],
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
