import {
  ConsoleArtifactIndexTransportError,
  parseConsoleArtifactIndexRecord,
  type ConsoleArtifactIndexRecord,
} from '../src/leibniz/console/_web_src/src/artifactIndex.ts';

const fixture = {
  format: 'leibniz.console.artifact-index',
  format_version: 1,
  artifacts: [
    {
      kind: 'measurement',
      source_path: 'tests/fixtures/finite_outcome/measurement.json',
      digest: 'sha256:d91a31bac6332478a9f11f73764d036e40b1826c928579dc17e132ff6f9bd133',
      protocol_id: 'core.boolean-evidence@0.1.0',
      reference: {
        kind: 'measurement',
        protocol_id: 'core.boolean-evidence@0.1.0',
        record_digest: 'sha256:d91a31bac6332478a9f11f73764d036e40b1826c928579dc17e132ff6f9bd133',
      },
      dependencies: [
        {
          kind: 'benchmark-manifest',
          protocol_id: 'core.boolean-benchmark@0.1.0',
        },
      ],
      validation_status: 'valid',
      validation_command: 'python -m pytest tests/test_console_artifact_index.py',
    },
  ],
} satisfies ConsoleArtifactIndexRecord;

const parsed = parseConsoleArtifactIndexRecord(fixture);

assertEqual(parsed.artifacts[0]?.kind, 'measurement', 'fixture artifact kind');
assertEqual(
  parsed.artifacts[0]?.dependencies[0]?.protocol_id,
  'core.boolean-benchmark@0.1.0',
  'fixture dependency protocol id',
);
assertTransportError(
  () => parseConsoleArtifactIndexRecord({ ...fixture, format_version: 2 }),
  'format_version: expected 1',
);
assertTransportError(
  () => parseConsoleArtifactIndexRecord({ ...fixture, artifacts: [{}] }),
  'artifacts.0.kind: expected string',
);

function assertEqual(actual: unknown, expected: unknown, label: string) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

function assertTransportError(callback: () => void, expectedMessage: string) {
  try {
    callback();
  } catch (error) {
    if (
      error instanceof ConsoleArtifactIndexTransportError &&
      error.message === expectedMessage
    ) {
      return;
    }
    throw error;
  }

  throw new Error(`expected transport error: ${expectedMessage}`);
}
