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
import { parseConsoleDataRecord } from '../src/leibniz/console/_web_src/src/consoleData.ts';

declare const consoleDataPayload: unknown;

const consoleData = parseConsoleDataRecord(consoleDataPayload);
const artifacts = consoleData.artifact_index.artifacts;
const details = consoleData.artifact_details;
const kinds = artifactKinds(artifacts);
const dependencyCount = artifacts.reduce(
  (count, artifact) => count + artifact.dependencies.length,
  0,
);
const measurementArtifacts = filterArtifacts(artifacts, 'measurement');
const selectedMeasurement = resolveSelectedArtifact(
  measurementArtifacts,
  artifactKey(artifacts[0]),
);
const selectedMeasurementDetail =
  selectedMeasurement === undefined
    ? undefined
    : detailForArtifact(details, selectedMeasurement);
const detailsByArtifact = artifacts.map((artifact) => detailForArtifact(details, artifact));
const chessBenchmark = artifacts.find(
  (artifact) => artifact.protocol_id === 'benchmarks.chess@0.1.0',
);
const chessMeasurement = artifacts.find(
  (artifact) => artifact.protocol_id === 'benchmarks.chess.fixture.mate-in-one-evidence@0.1.0',
);
const chessMeasurementDetail =
  chessMeasurement === undefined ? undefined : detailForArtifact(details, chessMeasurement);

assertEqual(
  kinds.join(','),
  'all,architecture-manifest,benchmark-manifest,measurement',
  'kinds',
);
assertEqual(kinds[0], allArtifactKinds, 'all kind first');
assertEqual(dependencyCount, 2, 'dependency count');
assertEqual(measurementArtifacts.length, 2, 'measurement filter count');
assertEqual(
  measurementArtifacts.map((artifact) => artifact.dependencies[0]?.protocol_id).join(','),
  'benchmarks.chess@0.1.0,core.boolean-benchmark@0.1.0',
  'measurement dependency protocol ids',
);
assertEqual(
  selectedMeasurement?.protocol_id,
  'benchmarks.chess.fixture.mate-in-one-evidence@0.1.0',
  'fallback selection after filter change',
);
assertEqual(
  referenceLabel(measurementArtifacts[0].dependencies[0]),
  'benchmarks.chess@0.1.0',
  'dependency reference label',
);
assertEqual(
  shortDigest(measurementArtifacts[0].digest),
  'sha256:07b9ca6e8603a2',
  'short digest',
);
assertEqual(selectedMeasurementDetail?.kind, 'measurement', 'selected measurement detail kind');
assertEqual(
  selectedMeasurementDetail?.kind === 'measurement'
    ? selectedMeasurementDetail.accepted_event.outcomes.join(',')
    : '',
  'g7f8',
  'measurement accepted outcomes',
);
assertEqual(
  selectedMeasurementDetail?.kind === 'measurement'
    ? selectedMeasurementDetail.probability_measure.probabilities.length
    : 0,
  3,
  'measurement probability count',
);
assertEqual(detailsByArtifact.every((detail) => detail !== undefined), true, 'all artifacts have details');
assertEqual(chessBenchmark?.source_path, 'tests/fixtures/chess/mate_in_one/manifest.json', 'chess benchmark path');
assertEqual(
  chessMeasurementDetail?.kind === 'measurement'
    ? chessMeasurementDetail.probability_measure.probabilities
        .map((probability) => `${probability.outcome_id}:${probability.probability}`)
        .join(',')
    : '',
  'g6f7:0.1,g7f8:0.7,g7g8:0.2',
  'chess probabilities',
);
assertEqual(
  detailForArtifact(details, artifacts[0])?.kind,
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
