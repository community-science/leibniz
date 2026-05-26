import {
  ConsoleDataTransportError,
  parseConsoleDataRecord,
} from '../src/leibniz/console/_web_src/src/consoleData.ts';
import { detailForArtifact } from '../src/leibniz/console/_web_src/src/artifactDetails.ts';

declare const consoleDataPayload: unknown;

const parsed = parseConsoleDataRecord(consoleDataPayload);
const artifacts = parsed.artifact_index.artifacts;
const detailCoverage = artifacts.map((artifact) =>
  detailForArtifact(parsed.artifact_details, artifact),
);
const consoleDataSource = parsed.source_modules.find(
  (sourceModule) => sourceModule.module_name === 'leibniz.console.data',
);

assertEqual(parsed.format, 'leibniz.console-data', 'format');
assertEqual(parsed.format_version, 1, 'format version');
assertEqual(artifacts.length, 11, 'artifact count');
assertEqual(detailCoverage.every((detail) => detail !== undefined), true, 'detail coverage');
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
  ].join('|'),
  'artifact order',
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

function assertEqual(actual: unknown, expected: unknown, label: string) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

function assertDataError(callback: () => void, expectedMessage: string) {
  try {
    callback();
  } catch (error) {
    if (error instanceof ConsoleDataTransportError && error.message === expectedMessage) {
      return;
    }
    throw error;
  }

  throw new Error(`expected console data transport error: ${expectedMessage}`);
}
