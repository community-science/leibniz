import { parseConsoleArtifactDetailRecords } from './artifactDetails.ts';

export const demoArtifactDetails = parseConsoleArtifactDetailRecords([
  {
    kind: 'architecture-manifest',
    source_path: 'tests/fixtures/architecture/digits_pool/manifest.json',
    input_shape: [1, 32, 32],
    output_shape: [10],
    layers: [
      {
        kind: 'adaptive-pooling',
        parameters: {
          dimension: 2,
          size: 2,
        },
      },
      {
        kind: 'flatten',
      },
      {
        kind: 'dense',
        parameters: {
          out: 10,
        },
      },
    ],
  },
  {
    kind: 'benchmark-manifest',
    source_path: 'tests/fixtures/finite_outcome/manifest.json',
    id: 'core.boolean-benchmark@0.1.0',
    outcome_space: {
      id: 'core.boolean-outcome@0.1.0',
      outcomes: [{ id: 'yes' }, { id: 'no' }],
    },
  },
  {
    kind: 'measurement',
    source_path: 'tests/fixtures/finite_outcome/measurement.json',
    id: 'core.boolean-evidence@0.1.0',
    benchmark_id: 'core.boolean-benchmark@0.1.0',
    observation_id: 'observation-1',
    outcome_space: {
      id: 'core.boolean-outcome@0.1.0',
      outcomes: [{ id: 'yes' }, { id: 'no' }],
    },
    accepted_event: {
      id: 'core.boolean-accepted@0.1.0',
      outcome_space_id: 'core.boolean-outcome@0.1.0',
      outcomes: ['yes'],
    },
    probability_measure: {
      id: 'core.boolean-prediction@0.1.0',
      outcome_space_id: 'core.boolean-outcome@0.1.0',
      probabilities: [
        {
          outcome_id: 'yes',
          probability: 0.25,
        },
        {
          outcome_id: 'no',
          probability: 0.75,
        },
      ],
    },
  },
]);
