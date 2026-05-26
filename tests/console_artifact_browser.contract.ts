import { demoArtifactIndex } from '../src/leibniz/console/_web_src/src/demoArtifactIndex.ts';
import {
  allArtifactKinds,
  artifactKey,
  artifactKinds,
  filterArtifacts,
  referenceLabel,
  resolveSelectedArtifact,
  shortDigest,
} from '../src/leibniz/console/_web_src/src/artifactBrowserModel.ts';
import {
  detailForArtifact,
  parseConsoleArtifactDetailRecords,
} from '../src/leibniz/console/_web_src/src/artifactDetails.ts';
import { demoArtifactDetails } from '../src/leibniz/console/_web_src/src/demoArtifactDetails.ts';

const kinds = artifactKinds(demoArtifactIndex.artifacts);
const dependencyCount = demoArtifactIndex.artifacts.reduce(
  (count, artifact) => count + artifact.dependencies.length,
  0,
);
const measurementArtifacts = filterArtifacts(demoArtifactIndex.artifacts, 'measurement');
const selectedMeasurement = resolveSelectedArtifact(
  measurementArtifacts,
  artifactKey(demoArtifactIndex.artifacts[0]),
);
const selectedMeasurementDetail =
  selectedMeasurement === undefined
    ? undefined
    : detailForArtifact(demoArtifactDetails, selectedMeasurement);

assertEqual(
  kinds.join(','),
  'all,architecture-manifest,benchmark-manifest,measurement',
  'kinds',
);
assertEqual(kinds[0], allArtifactKinds, 'all kind first');
assertEqual(dependencyCount, 1, 'dependency count');
assertEqual(measurementArtifacts.length, 1, 'measurement filter count');
assertEqual(
  measurementArtifacts[0]?.dependencies[0]?.protocol_id,
  'core.boolean-benchmark@0.1.0',
  'measurement dependency protocol id',
);
assertEqual(
  selectedMeasurement?.protocol_id,
  'core.boolean-evidence@0.1.0',
  'fallback selection after filter change',
);
assertEqual(
  referenceLabel(measurementArtifacts[0].dependencies[0]),
  'core.boolean-benchmark@0.1.0',
  'dependency reference label',
);
assertEqual(
  shortDigest(measurementArtifacts[0].digest),
  'sha256:d91a31bac63324',
  'short digest',
);
assertEqual(selectedMeasurementDetail?.kind, 'measurement', 'selected measurement detail kind');
assertEqual(
  selectedMeasurementDetail?.kind === 'measurement'
    ? selectedMeasurementDetail.accepted_event.outcomes.join(',')
    : '',
  'yes',
  'measurement accepted outcomes',
);
assertEqual(
  selectedMeasurementDetail?.kind === 'measurement'
    ? selectedMeasurementDetail.probability_measure.probabilities.length
    : 0,
  2,
  'measurement probability count',
);
assertEqual(
  detailForArtifact(demoArtifactDetails, demoArtifactIndex.artifacts[0])?.kind,
  'architecture-manifest',
  'architecture detail lookup',
);
assertThrows(
  () => parseConsoleArtifactDetailRecords([{ kind: 'private-roadmap', source_path: 'README.md' }]),
  'unsupported artifact detail kind',
);

function assertEqual(actual: unknown, expected: unknown, label: string) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

function assertThrows(action: () => void, expectedMessage: string) {
  try {
    action();
  } catch (error) {
    if (error instanceof Error && error.message.includes(expectedMessage)) {
      return;
    }
    throw error;
  }
  throw new Error(`expected error including ${expectedMessage}`);
}
