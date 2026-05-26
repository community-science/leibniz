import { parseConsoleDataRecord } from '../src/leibniz/console/_web_src/src/consoleData.ts';
import { detailForArtifact } from '../src/leibniz/console/_web_src/src/artifactDetails.ts';

declare const consoleDataPayload: unknown;

const parsed = parseConsoleDataRecord(consoleDataPayload);
const rawConsoleData = consoleDataPayload as Record<string, unknown>;
const artifacts = parsed.artifact_index.artifacts;
const detailCoverage = artifacts.map((artifact) =>
  detailForArtifact(parsed.artifact_details, artifact),
);
const consoleDataSource = parsed.source_modules.find(
  (sourceModule) => sourceModule.module_name === 'leibniz.console.data',
);
const modelInspection = parsed.model_inspections[0];
if (modelInspection === undefined) {
  throw new Error('expected model inspection fixture');
}

assertEqual(parsed.format, 'leibniz.console-data', 'format');
assertEqual(parsed.format_version, 1, 'format version');
assertEqual(artifacts.length, 13, 'artifact count');
assertEqual(detailCoverage.every((detail) => detail !== undefined), true, 'detail coverage');
assertEqual(parsed.performance_views.length, 1, 'performance view count');
assertEqual(parsed.result_views.length, 0, 'result view count');
assertEqual(parsed.model_inspections.length, 1, 'model inspection count');
assertEqual(parsed.benchmark_tasks.length, 1, 'benchmark task count');
assertEqual(parsed.source_modules.length > 20, true, 'source module count');
assertEqual(consoleDataSource?.source_path, 'src/leibniz/console/data.py', 'console data source path');
assertEqual(
  consoleDataSource?.public_exports.join(','),
  'ConsoleData,ConsoleDataBuilder,ConsoleDataValidationError',
  'console data exports',
);
assertEqual(
  consoleDataSource?.validation_commands.includes('python -m pytest tests/test_console_data.py'),
  true,
  'console data validation command',
);
assertEqual(
  artifacts.map((artifact) => `${artifact.kind}:${artifact.source_path}`).join('|'),
  [
    'architecture-manifest:tests/fixtures/architecture/digits_pool/manifest.json',
    'benchmark-manifest:src/leibniz/benchmarks/digits/manifest.json',
    'benchmark-manifest:tests/fixtures/chess/mate_in_one/manifest.json',
    'benchmark-manifest:tests/fixtures/finite_outcome/manifest.json',
    'latent-factor-declaration:src/leibniz/benchmarks/digits/latent_factors.json',
    'materialization-declaration:src/leibniz/benchmarks/digits/materialization.json',
    'materialization-plan:tests/fixtures/digits/materialization_plan_l1.json',
    'materialization-plan:tests/fixtures/digits/materialization_plan_l3.json',
    'measurement:tests/fixtures/chess/mate_in_one/measurement.json',
    'measurement:tests/fixtures/finite_outcome/measurement.json',
    'observation-formation-declaration:src/leibniz/benchmarks/digits/observation_formation.json',
    'observation-showcase:src/leibniz/benchmarks/digits/inspection_showcase.json',
    'performance-view-bundle:src/leibniz/benchmarks/digits/performance_view_bundle.json',
  ].join('|'),
  'artifact order',
);
assertEqual(
  parsed.performance_views[0]?.competence_integral_view.entries[0]?.integral,
  0.25,
  'performance view integral',
);
assertEqual(
  parsed.performance_views[0]?.measurement_dataset.measurements.length,
  2,
  'performance measurement count',
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
  modelInspection.layers.map((layer) => layer.kind).join(','),
  'adaptive-pooling,flatten,dense',
  'model inspection layers',
);
assertEqual(
  modelInspection.layers.map((layer) => layer.operator?.kind).join(','),
  'local-aggregation,rank-collapse,affine-readout',
  'model operator summaries',
);
const benchmarkTask = parsed.benchmark_tasks[0];
if (benchmarkTask === undefined) {
  throw new Error('expected benchmark task');
}
assertEqual(benchmarkTask?.kind, 'generated-observations', 'benchmark task kind');
assertEqual(benchmarkTask?.batches.length, 9, 'benchmark batch count');
assertEqual(
  benchmarkTask?.batches.map((batch) => `${batch.mode}:${batch.scale}:${batch.sample_count}`).join('|'),
  'canonical:1:4|canonical:2:4|canonical:3:4|canonical:4:4|symbol-probe:1:10|complexity-sweep:1:1|complexity-sweep:2:1|complexity-sweep:3:1|complexity-sweep:4:1',
  'generated benchmark batches',
);
assertEqual(
  benchmarkTask?.batches[4]?.samples.map((sample) => sample.component_sequence.join('')).join(','),
  '0,1,2,3,4,5,6,7,8,9',
  'symbol probe batch',
);
assertEqual(
  benchmarkTask?.batches[4]?.presentation.sample_card_density,
  'compact',
  'compact batch presentation',
);
assertEqual(
  benchmarkTask?.batches[5]?.presentation.aggregate_mode,
  true,
  'aggregate batch presentation',
);
assertEqual(
  artifacts
    .filter((artifact) => artifact.kind === 'measurement')
    .map((artifact) => artifact.dependencies[0]?.protocol_id)
    .join(','),
  'benchmarks.chess@0.1.0,core.boolean-benchmark@0.1.0',
  'measurement dependencies',
);
assertDataError(
  () => parseConsoleDataRecord({ ...parsed, format_version: 2 }),
  'format_version: expected 1',
);
assertDataError(
  () =>
    parseConsoleDataRecord({
      ...rawConsoleData,
      model_inspections: [{ ...modelInspection, layers: [{ index: '0' }] }],
    }),
  'model inspections.0.layers.0.index: expected number',
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
