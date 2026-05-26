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
    source_path: 'tests/fixtures/chess/mate_in_one/manifest.json',
    id: 'benchmarks.chess@0.1.0',
    outcome_space: {
      id: 'benchmarks.chess.uci-moves@0.1.0',
      outcomes: [{ id: 'g7f8' }, { id: 'g7g8' }, { id: 'g6f7' }],
    },
    observation_ids: ['fen:7k/6Q1/6K1/8/8/8/8/8 w - - 0 1'],
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
    source_path: 'tests/fixtures/chess/mate_in_one/measurement.json',
    id: 'benchmarks.chess.fixture.mate-in-one-evidence@0.1.0',
    benchmark_id: 'benchmarks.chess@0.1.0',
    observation_id: 'fen:7k/6Q1/6K1/8/8/8/8/8 w - - 0 1',
    outcome_space: {
      id: 'benchmarks.chess.uci-moves@0.1.0',
      outcomes: [{ id: 'g7f8' }, { id: 'g7g8' }, { id: 'g6f7' }],
    },
    accepted_event: {
      id: 'benchmarks.chess.fixture.mate-in-one-accepted@0.1.0',
      outcome_space_id: 'benchmarks.chess.uci-moves@0.1.0',
      outcomes: ['g7f8'],
    },
    probability_measure: {
      id: 'benchmarks.chess.fixture.mate-in-one-prediction@0.1.0',
      outcome_space_id: 'benchmarks.chess.uci-moves@0.1.0',
      probabilities: [
        {
          outcome_id: 'g7f8',
          probability: 0.7,
        },
        {
          outcome_id: 'g7g8',
          probability: 0.2,
        },
        {
          outcome_id: 'g6f7',
          probability: 0.1,
        },
      ],
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
