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
const digitsBenchmark = artifacts.find(
  (artifact) => artifact.protocol_id === 'benchmarks.digits@0.1.0',
);
const digitsBenchmarkDetail =
  digitsBenchmark === undefined ? undefined : detailForArtifact(details, digitsBenchmark);
const chessMeasurement = artifacts.find(
  (artifact) => artifact.protocol_id === 'benchmarks.chess.fixture.mate-in-one-evidence@0.1.0',
);
const chessMeasurementDetail =
  chessMeasurement === undefined ? undefined : detailForArtifact(details, chessMeasurement);
const latentFactors = artifacts.find(
  (artifact) => artifact.protocol_id === 'benchmarks.digits.latent-factors@0.1.0',
);
const latentFactorsDetail =
  latentFactors === undefined ? undefined : detailForArtifact(details, latentFactors);
const materialization = artifacts.find(
  (artifact) => artifact.protocol_id === 'benchmarks.digits.materialization@0.1.0',
);
const materializationDetail =
  materialization === undefined ? undefined : detailForArtifact(details, materialization);
const materializationPlan = artifacts.find(
  (artifact) =>
    artifact.protocol_id === 'benchmarks.digits.materialization-plan.l3.seed101@0.1.0',
);
const materializationPlanDetail =
  materializationPlan === undefined ? undefined : detailForArtifact(details, materializationPlan);
const observationFormation = artifacts.find(
  (artifact) => artifact.protocol_id === 'benchmarks.digits.observation-formation@0.1.0',
);
const observationFormationDetail =
  observationFormation === undefined ? undefined : detailForArtifact(details, observationFormation);
const observationShowcase = artifacts.find(
  (artifact) => artifact.protocol_id === 'benchmarks.digits.inspection-showcase@0.1.0',
);
const observationShowcaseDetail =
  observationShowcase === undefined ? undefined : detailForArtifact(details, observationShowcase);

assertEqual(
  kinds.join(','),
  [
    'all',
    'architecture-manifest',
    'benchmark-manifest',
    'latent-factor-declaration',
    'materialization-declaration',
    'materialization-plan',
    'measurement',
    'observation-formation-declaration',
    'observation-showcase',
  ].join(','),
  'kinds',
);
assertEqual(kinds[0], allArtifactKinds, 'all kind first');
assertEqual(dependencyCount, 15, 'dependency count');
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
  digitsBenchmark?.dependencies[0]?.protocol_id,
  'benchmarks.digits.latent-factors@0.1.0',
  'digits latent factor dependency',
);
assertEqual(
  digitsBenchmarkDetail?.kind === 'benchmark-manifest'
    ? digitsBenchmarkDetail.complexity_coordinate
    : '',
  'C',
  'digits complexity coordinate',
);
assertEqual(
  digitsBenchmarkDetail?.kind === 'benchmark-manifest'
    ? digitsBenchmarkDetail.scale_parameter?.symbol
    : '',
  'L',
  'digits scale parameter',
);
assertEqual(
  digitsBenchmarkDetail?.kind === 'benchmark-manifest'
    ? digitsBenchmarkDetail.outcome_sequence?.atom_count
    : 0,
  10,
  'digits atom count',
);
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
  latentFactorsDetail?.kind === 'latent-factor-declaration'
    ? latentFactorsDetail.sample_factors.map((factor) => `${factor.name}:${factor.role}`).join(',')
    : '',
  [
    'benchmarks.digits.sample.digit-identity:content',
    'benchmarks.digits.sample.field-nuisance-transform:nuisance',
    'benchmarks.digits.materialization.canvas-side:materialization',
  ].join(','),
  'digits latent sample factors',
);
assertEqual(
  latentFactorsDetail?.kind === 'latent-factor-declaration'
    ? latentFactorsDetail.complexity_projections[0]?.coordinate
    : '',
  'C',
  'digits latent complexity coordinate',
);
assertEqual(
  materializationDetail?.kind === 'materialization-declaration'
    ? materializationDetail.requirements[0]?.coefficient
    : 0,
  32,
  'digits materialization resolution coefficient',
);
assertEqual(
  materializationPlanDetail?.kind === 'materialization-plan'
    ? materializationPlanDetail.resolution_assignment.values[0]?.value
    : 0,
  96,
  'digits materialization plan resolution',
);
assertEqual(
  observationFormationDetail?.kind === 'observation-formation-declaration'
    ? `${observationFormationDetail.component_count}:${observationFormationDetail.mark_count}`
    : '',
  '10:38',
  'digits observation formation coverage',
);
assertEqual(
  observationShowcaseDetail?.kind === 'observation-showcase'
    ? observationShowcaseDetail.samples.map((sample) => sample.label).join(',')
    : '',
  'Single digit 7,Three digit sequence 123',
  'digits observation showcase samples',
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
