import {
  benchmarkResultsForTask,
  modelComparisonRows,
  performanceViewsForTask,
} from '../src/leibniz/console/_web_src/src/benchmarkDashboardModel.ts';
import type { ModelInspectionRecord } from '../src/leibniz/console/_web_src/src/modelInspections.ts';
import type { PerformanceViewRecord } from '../src/leibniz/console/_web_src/src/performanceViews.ts';
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
  frontiers: {},
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
  ],
  proposals: [],
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
const performanceViews: PerformanceViewRecord[] = [
  {
    competence_integral_view: {
      complexity_axis: 'C',
      entries: [
        {
          benchmark_id: targetBenchmark,
          coverage: 1,
          integral: 0.75,
          missing_complexities: [],
          observed_complexities: [1],
          points: [],
        },
      ],
      expected_complexities: [1],
      id: 'integral-view',
      projection_rule: 'mean',
      source_dataset_digest: 'sha256:1234',
    },
    id: 'performance-view',
    manifest: {
      benchmark_manifest: {
        kind: 'benchmark-manifest',
        protocol_id: targetBenchmark,
      },
      complexity_axis: 'C',
      expected_complexities: [1],
      id: 'manifest',
      materialization_declaration: {
        kind: 'materialization-declaration',
      },
      measurement_cases: [],
      observation_formation_declaration: {
        kind: 'observation-formation-declaration',
      },
      view_id: 'views.target.performance@0.1.0',
    },
    materialization_plans: [],
    measurement_dataset: { measurements: [] },
    source_path: 'benchmarks/target/performance_view_bundle.json',
  },
];

assertEqual(benchmarkResultsForTask(resultViews, targetBenchmark).length, 1, 'target results');
assertEqual(benchmarkResultsForTask(resultViews, otherBenchmark).length, 1, 'other results');
assertEqual(performanceViewsForTask(performanceViews, targetBenchmark).length, 1, 'target views');
assertEqual(performanceViewsForTask(performanceViews, otherBenchmark).length, 0, 'other views');
assertEqual(
  modelComparisonRows(result, inspections)[0]?.inspection?.id,
  'inspection-a',
  'model inspection match',
);

function assertEqual(actual: unknown, expected: unknown, label: string) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}
