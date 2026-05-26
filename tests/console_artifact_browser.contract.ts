import { demoArtifactIndex } from '../src/leibniz/console/_web_src/src/demoArtifactIndex.ts';

const kinds = Array.from(new Set(demoArtifactIndex.artifacts.map((artifact) => artifact.kind))).sort();
const dependencyCount = demoArtifactIndex.artifacts.reduce(
  (count, artifact) => count + artifact.dependencies.length,
  0,
);
const measurementArtifacts = demoArtifactIndex.artifacts.filter(
  (artifact) => artifact.kind === 'measurement',
);

assertEqual(kinds.join(','), 'architecture-manifest,benchmark-manifest,measurement', 'kinds');
assertEqual(dependencyCount, 1, 'dependency count');
assertEqual(measurementArtifacts.length, 1, 'measurement filter count');
assertEqual(
  measurementArtifacts[0]?.dependencies[0]?.protocol_id,
  'core.boolean-benchmark@0.1.0',
  'measurement dependency protocol id',
);

function assertEqual(actual: unknown, expected: unknown, label: string) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}
