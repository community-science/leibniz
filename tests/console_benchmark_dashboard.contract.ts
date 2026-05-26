import {
  benchmarkPlotModel,
  benchmarkResultsForTask,
  modelComparisonRows,
} from '../src/leibniz/console/_web_src/src/benchmarkDashboardModel.ts';
import type { ModelInspectionRecord } from '../src/leibniz/console/_web_src/src/modelInspections.ts';
import type {
  BenchmarkResultRecord,
  ResultViewRecord,
} from '../src/leibniz/console/_web_src/src/resultViews.ts';

const targetBenchmark = 'benchmarks.target@0.1.0';
const otherBenchmark = 'benchmarks.other@0.1.0';
const architectureDigest = 'sha256:abcdef1234567890';

const result: BenchmarkResultRecord = {
  benchmark_id: targetBenchmark,
  cost_axes: [{ key: 'parameter_count', label: 'Parameters' }],
  frontiers: {
    parameter_count: [
      {
        architecture_digest: architectureDigest,
        benchmark_id: targetBenchmark,
        cost_summary: {
          inference_flops: 20,
          layer_count: 1,
          parameter_bytes: 40,
          parameter_count: 10,
        },
        measurement_count: 2,
        model_key: 'model-a',
        observed_complexities: [1, 2],
        points: [],
        run_ids: ['run-a'],
        score: 0.75,
        source_kinds: ['local'],
      },
    ],
  },
  leaderboard: [
    {
      architecture_digest: architectureDigest,
      benchmark_id: targetBenchmark,
      cost_summary: {
        inference_flops: 20,
        layer_count: 1,
        parameter_bytes: 40,
        parameter_count: 10,
      },
      measurement_count: 2,
      model_key: 'model-a',
      observed_complexities: [1, 2],
      points: [],
      run_ids: ['run-a'],
      score: 0.75,
      source_kinds: ['local'],
    },
    {
      architecture_digest: 'sha256:fedcba9876543210',
      benchmark_id: targetBenchmark,
      cost_summary: {
        inference_flops: 80,
        layer_count: 2,
        parameter_bytes: 160,
        parameter_count: 40,
      },
      measurement_count: 1,
      model_key: 'model-b',
      observed_complexities: [1],
      points: [],
      run_ids: ['run-b'],
      score: 0.5,
      source_kinds: ['local'],
    },
  ],
  proposals: [
    {
      acquisition_value: 0.2,
      candidate_id: 'model-a',
      candidate_kind: 'architecture',
      command: [],
      expected_frontier_improvement: 0.1,
      id: 'proposal-a',
      novelty: 0.3,
      predicted_score: 0.8,
      rank: 1,
      rationale: 'probe nearby candidate',
      uncertainty: 0.05,
    },
  ],
  training_history: [],
};
const resultViews: ResultViewRecord[] = [
  {
    benchmark_results: [
      result,
      {
        ...result,
        benchmark_id: otherBenchmark,
        leaderboard: [],
      },
    ],
    format: 'leibniz.console.benchmark-results',
    format_version: 1,
    source_path: 'results/benchmark_results.json',
  },
];
const inspections: ModelInspectionRecord[] = [
  {
    architecture: {
      kind: 'architecture-manifest',
      record_digest: architectureDigest,
    },
    cost_summary: {
      inference_flops: 20,
      layer_count: 1,
      parameter_bytes: 40,
      parameter_count: 10,
      unknown_flop_layers: [],
      unknown_parameter_layers: [],
    },
    id: 'inspection-a',
    input_shape: [1],
    layers: [
      {
        index: 0,
        kind: 'operator',
        parameters: {},
      },
    ],
    model_artifacts: [],
    output_shape: [2],
    source_path: 'results/model.json',
    training_provenance: [],
  },
];
assertEqual(benchmarkResultsForTask(resultViews, targetBenchmark).length, 1, 'target results');
assertEqual(benchmarkResultsForTask(resultViews, otherBenchmark).length, 1, 'other results');
assertEqual(
  modelComparisonRows(result, inspections)[0]?.inspection?.id,
  'inspection-a',
  'model inspection match',
);
const plotModel = benchmarkPlotModel(result, 'parameter_count');
assertEqual(plotModel.points.length, 2, 'plot point count');
assertEqual(plotModel.frontierPoints.length, 1, 'plot frontier count');
assertEqual(plotModel.proposals.length, 1, 'plot proposal count');
assertEqual(plotModel.proposals[0]?.cost, 10, 'plot proposal cost');
assertEqual(plotModel.staircase.length, 1, 'plot staircase point count');
assertEqual(plotModel.xTicks.includes(16), true, 'plot log ticks');
assertEqual(plotModel.yDomain[0], 0, 'plot y starts at zero');
const emptyPlotModel = benchmarkPlotModel(
  {
    ...result,
    frontiers: {},
    leaderboard: [],
    proposals: [],
  },
  'parameter_count',
);
assertEqual(emptyPlotModel.points.length, 0, 'empty plot point count');
assertEqual(emptyPlotModel.xTicks.length > 0, true, 'empty plot has x ticks');
assertEqual(emptyPlotModel.yTicks.length > 0, true, 'empty plot has y ticks');

function assertEqual(actual: unknown, expected: unknown, label: string) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}
