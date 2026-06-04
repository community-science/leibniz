import {
  benchmarkCostAxes,
  benchmarkCostAxis,
  benchmarkPlotModel,
  benchmarkResultsForTask,
  benchmarkScoreAxes,
  benchmarkScoreAxis,
  emptyFrontiersForCostAxes,
  modelComparisonRows,
  nextModelResultSort,
  runSelectionId,
  scoreTickLabel,
  selectionForId,
  sortedModelResults,
} from '../src/leibniz/console/_web_src/src/benchmarkDashboardModel.ts';
import type { ModelInspectionRecord } from '../src/leibniz/console/_web_src/src/modelInspections.ts';
import type {
  BenchmarkResultRecord,
  ResultViewRecord,
} from '../src/leibniz/console/_web_src/src/resultViews.ts';
import { parseResultViewRecords } from '../src/leibniz/console/_web_src/src/resultViews.ts';

const targetBenchmark = 'benchmarks.target@0.1.0';
const otherBenchmark = 'benchmarks.other@0.1.0';
const architectureDigest = 'sha256:abcdef1234567890';

const standardAxes = benchmarkCostAxes(undefined);
assertEqual(standardAxes.map((axis) => axis.key).join(','),
  'storage_bytes,inference_compute,training_compute',
  'standard cost axes',
);
assertEqual(
  Object.keys(emptyFrontiersForCostAxes(standardAxes)).join(','),
  'storage_bytes,inference_compute,training_compute',
  'empty frontier axes',
);
assertEqual(
  benchmarkCostAxis('missing_axis', standardAxes),
  'storage_bytes',
  'missing cost axis fallback',
);
assertEqual(
  benchmarkScoreAxes(undefined).map((axis) => axis.key).join(','),
  'absolute,relative',
  'default score axes',
);
assertEqual(
  benchmarkScoreAxis('missing_axis', benchmarkScoreAxes(undefined)),
  'absolute',
  'missing score axis fallback',
);

const result: BenchmarkResultRecord = {
  benchmark_id: targetBenchmark,
  cost_axes: [{ key: 'storage_bytes', label: 'Model Size' }],
  score_axes: [
    { key: 'absolute', label: 'Absolute Score' },
    { key: 'relative', label: 'Relative Score' },
  ],
  frontiers: {
    storage_bytes: [
      {
        architecture_digest: architectureDigest,
        benchmark_id: targetBenchmark,
        cost_summary: {
          component_count: 1,
          inference_compute: 20,
          storage_bytes: 40,
          training_compute: 360,
        },
        measurement_count: 2,
        model_key: 'model-a',
        result_status: 'accepted',
        observed_complexities: [1, 2],
        points: [],
        run_ids: ['run-a'],
        score: 0.75,
        score_views: {
          absolute: { key: 'absolute', label: 'Absolute Score', score: 0.75 },
          relative: { key: 'relative', label: 'Relative Score', score: 1200 },
        },
        source_kinds: ['local'],
      },
    ],
  },
  leaderboard: [
    {
      architecture_digest: architectureDigest,
      benchmark_id: targetBenchmark,
      cost_summary: {
        component_count: 1,
        inference_compute: 20,
        storage_bytes: 40,
        training_compute: 360,
      },
      measurement_count: 2,
      model_key: 'model-a',
      result_status: 'accepted',
      observed_complexities: [1, 2],
      points: [],
      run_ids: ['run-a'],
      score: 0.75,
      score_views: {
        absolute: { key: 'absolute', label: 'Absolute Score', score: 0.75 },
        relative: { key: 'relative', label: 'Relative Score', score: 1200 },
      },
      source_kinds: ['local'],
    },
    {
      architecture_digest: 'sha256:fedcba9876543210',
      benchmark_id: targetBenchmark,
      cost_summary: {
        component_count: 2,
        inference_compute: 80,
        storage_bytes: 160,
        training_compute: 1440,
      },
      measurement_count: 1,
      model_key: 'model-b',
      result_status: 'accepted',
      observed_complexities: [1],
      points: [],
      run_ids: ['run-b'],
      score: 0.5,
      score_views: {
        absolute: { key: 'absolute', label: 'Absolute Score', score: 0.5 },
        relative: { key: 'relative', label: 'Relative Score', score: 900 },
      },
      source_kinds: ['local'],
    },
  ],
  model_candidates: [
    {
      architecture_digest: architectureDigest,
      benchmark_id: targetBenchmark,
      cost_summary: {
        component_count: 1,
        inference_compute: 20,
        storage_bytes: 40,
        training_compute: 360,
      },
      measurement_count: 2,
      model_key: 'model-a',
      result_status: 'accepted',
      observed_complexities: [1, 2],
      points: [],
      run_ids: ['run-a'],
      score: 0.75,
      score_views: {
        absolute: { key: 'absolute', label: 'Absolute Score', score: 0.75 },
        relative: { key: 'relative', label: 'Relative Score', score: 1200 },
      },
      source_kinds: ['local'],
    },
    {
      architecture_digest: 'sha256:fedcba9876543210',
      benchmark_id: targetBenchmark,
      cost_summary: {
        component_count: 2,
        inference_compute: 80,
        storage_bytes: 160,
        training_compute: 1440,
      },
      measurement_count: 1,
      model_key: 'model-b',
      result_status: 'accepted',
      observed_complexities: [1],
      points: [],
      run_ids: ['run-b'],
      score: 0.5,
      score_views: {
        absolute: { key: 'absolute', label: 'Absolute Score', score: 0.5 },
        relative: { key: 'relative', label: 'Relative Score', score: 900 },
      },
      source_kinds: ['local'],
    },
  ],
  model_inspections: [],
  training_history: [
    {
      architecture: { layers: [] },
      architecture_digest: architectureDigest,
      benchmark_id: targetBenchmark,
      cost_summary: {
        component_count: 1,
        inference_compute: 20,
        storage_bytes: 40,
        training_compute: 360,
      },
      measurement_count: 2,
      measurement_dataset_digest: 'sha256:dataset1234',
      model_key: 'model-a',
      run_id: 'run-a',
      run_slug: 'train-a',
      complexity: 10,
      score: 0.75,
      result_status: 'accepted',
      source_kind: 'local',
      source_path: 'results/training/run-a.json',
      training_diagnostics: {
        artifacts: [
          { digest: 'sha256:dataset1234', kind: 'measurement-dataset' },
          { digest: architectureDigest, kind: 'model-inspection', path: 'results/models/a.json' },
          { digest: 'sha256:training1234', kind: 'training-summary', path: 'results/training/run-a.json' },
        ],
        final_validation_check: 1,
        final_validation_loss: 0.4,
        final_validation_step: 3,
        protocol: {
          batch_size: 2,
          kind: 'fixed-step-local-batch',
          learning_rate: 0.01,
          max_steps: 3,
          min_delta: 0,
          objective: 'cross-entropy',
          optimizer: 'sgd',
          patience: 0,
          schedule: 'none',
          seed: 101,
          gate_check_interval: 1,
          gate_sample_count: 2,
          gate_decision_rule: 'validation-loss-plateau',
          validation_source: 'generator-resample',
        },
        status: 'budget-exhausted',
        steps_run: 3,
        stop_reason: 'max-steps',
        validation_checks: 2,
        validation_history: [
          {
            stale_checks: 0,
            step: 0,
            validation_check: 0,
            validation_loss: 0.6,
          },
          {
            stale_checks: 0,
            step: 3,
            validation_check: 1,
            validation_loss: 0.4,
          },
        ],
      },
    },
  ],
  plot_runs: [
    {
      architecture: { layers: [] },
      architecture_digest: architectureDigest,
      benchmark_id: targetBenchmark,
      cost_summary: {
        component_count: 1,
        inference_compute: 20,
        storage_bytes: 40,
        training_compute: 360,
      },
      measurement_count: 2,
      measurement_dataset_digest: 'sha256:dataset1234',
      model_key: 'model-a',
      run_id: 'run-a',
      run_slug: 'train-a',
      complexity: 10,
      score: 0.75,
      result_status: 'accepted',
      source_kind: 'local',
      source_path: 'results/training/run-a.json',
    },
    {
      architecture: { layers: [] },
      architecture_digest: 'sha256:fedcba9876543210',
      benchmark_id: targetBenchmark,
      cost_summary: {
        component_count: 2,
        inference_compute: 80,
        storage_bytes: 160,
        training_compute: 1440,
      },
      measurement_count: 1,
      measurement_dataset_digest: 'sha256:dataset5678',
      model_key: 'model-b',
      run_id: 'run-b',
      run_slug: 'train-b',
      complexity: 10,
      score: 0.5,
      result_status: 'accepted',
      source_kind: 'local',
      source_path: 'results/training/run-b.json',
    },
  ],
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
    source_mtime_ms: 123456789,
    source_path: 'results/views/digits/benchmark_results.json',
    source_size_bytes: 4096,
  },
];
const inspections: ModelInspectionRecord[] = [
  {
    architecture: {
      kind: 'architecture-manifest',
      record_digest: architectureDigest,
    },
    cost_summary: {
      inference_compute: 20,
      component_count: 1,
      storage_bytes: 40,
      unknown_compute_components: [],
      unknown_parameter_components: [],
    },
    id: 'inspection-a',
    input_shape: [1],
    architecture_graph: {
      edges: [],
      input_node_ids: ['component-0'],
      nodes: [
        {
          component: {
            kind: 'operator',
            parameters: {},
          },
          id: 'component-0',
        },
      ],
      output_node_ids: ['component-0'],
    },
    architecture_summary: {
      component_count: 1,
      component_kinds: ['operator'],
      edge_count: 0,
      input_count: 1,
      input_node_ids: ['component-0'],
      output_count: 1,
      output_node_ids: ['component-0'],
      unsupported_compute_components: [],
      unsupported_parameter_components: [],
    },
    architecture_trace: {
      input_shape: [1],
      output_shape: [2],
      stages: [
        {
          cost_law: 'zero-arithmetic',
          descriptor_axes: {
            aggregation_law: 'none',
            parameter_sharing: 'none',
            projection_law: 'identity',
            state: 'fixed',
            support: 'global',
            tensor_relation: 'identity',
          },
          index: 0,
          inference_compute: 0,
          input_shape: [1],
          kind: 'operator',
          operator_kind: 'identity',
          output_shape: [2],
          shape_law: 'fixture-shape',
          syntax_alias: 'operator',
        },
      ],
      program_effects: [],
    },
    components: [
      {
        index: 0,
        kind: 'operator',
        parameters: {},
      },
    ],
    model_artifacts: [],
    node_evidence: [
      {
        claim_kinds: ['architecture-structure', 'operator-semantics', 'resource-accounting'],
        evidence_artifacts: [
          {
            kind: 'architecture-manifest',
            record_digest: architectureDigest,
          },
        ],
        node_path: ['component-0'],
      },
    ],
    output_shape: [2],
    source_path: 'results/model.json',
    training_provenance: [],
  },
];
assertEqual(benchmarkResultsForTask(resultViews, targetBenchmark).length, 1, 'target results');
assertEqual(benchmarkResultsForTask(resultViews, otherBenchmark).length, 1, 'other results');
assertEqual(
  benchmarkResultsForTask(resultViews, targetBenchmark)[0]?.sourceMtimeMs,
  123456789,
  'result view source mtime',
);
assertEqual(
  benchmarkResultsForTask(resultViews, targetBenchmark)[0]?.sourceSizeBytes,
  4096,
  'result view source size',
);
assertEqual(
  modelComparisonRows(result, inspections)[0]?.inspection?.id,
  'inspection-a',
  'model inspection match',
);
assertEqual(
  modelComparisonRows({ ...result, model_inspections: inspections }, [])[0]?.inspection?.id,
  'inspection-a',
  'result-local model inspection match',
);
const plotModel = benchmarkPlotModel(result, 'storage_bytes');
const relativePlotModel = benchmarkPlotModel(result, 'storage_bytes', 'relative');
assertEqual(plotModel.points.length, 2, 'plot point count');
assertEqual(benchmarkScoreAxes(result).map((axis) => axis.key).join(','), 'absolute,relative', 'result score axes');
assertEqual(benchmarkScoreAxis('relative', benchmarkScoreAxes(result)), 'relative', 'relative score axis');
assertEqual(plotModel.frontierPoints.length, 1, 'plot frontier count');
assertEqual(relativePlotModel.points[0]?.score, 1200, 'relative plot score');
assertEqual(plotModel.staircase.length, 1, 'plot staircase point count');
assertEqual(plotModel.xTicks.includes(16), true, 'plot log ticks');
assertEqual(plotModel.xDomain[0], 0, 'plot default x minimum');
assertEqual(plotModel.xDomain[1], 20, 'plot default x maximum');
assertEqual(plotModel.xMajorTicks.includes(1), true, 'plot major x ticks');
assertEqual(plotModel.xMinorTicks.includes(2), true, 'plot minor x ticks');
const trainingComputePlotModel = benchmarkPlotModel(result, 'training_compute');
assertEqual(trainingComputePlotModel.xMajorTicks.includes(10), true, 'training compute uses base-10 ticks');
assertEqual(trainingComputePlotModel.xMajorTicks.includes(16), false, 'training compute omits base-2 ticks');
const inferenceComputePlotModel = benchmarkPlotModel(result, 'inference_compute');
assertEqual(inferenceComputePlotModel.xMajorTicks.includes(10), true, 'inference compute uses base-10 ticks');
assertEqual(inferenceComputePlotModel.xMajorTicks.includes(16), false, 'inference compute omits base-2 ticks');
assertEqual(plotModel.yDomain[0], 0, 'plot y starts at zero');
assertEqual(plotModel.yDomain[1], 1.05, 'plot y ceiling follows score scale');
assertEqual(plotModel.yTicks.join(','), '0,0.2,0.4,0.6,0.8,1', 'absolute plot y ticks');
assertEqual(relativePlotModel.yTicks.join(','), '0,200,400,600,800,1000,1200', 'relative plot y ticks');
assertEqual(scoreTickLabel(1200), '1,200', 'relative score tick label');
assertEqual(scoreTickLabel(0.2), '0.2', 'fractional score tick label');
const acceptedDominatedRelativeScaleResult: BenchmarkResultRecord = {
  ...result,
  plot_runs: [
    {
      ...result.plot_runs[1]!,
    },
  ],
};
assertEqual(
  benchmarkPlotModel(acceptedDominatedRelativeScaleResult, 'storage_bytes', 'relative').yDomain[1],
  1260,
  'plot y ceiling includes accepted frontier points as well as tentative runs',
);
assertEqual(
  sortedModelResults(result.leaderboard, 'storage_bytes', 'absolute', {
    key: 'cost',
    direction: 'descending',
  })[0]?.model_key,
  'model-b',
  'model cost sort',
);
assertEqual(
  sortedModelResults(result.leaderboard, 'storage_bytes', 'relative', {
    key: 'score',
    direction: 'ascending',
  })[0]?.model_key,
  'model-b',
  'relative score sort',
);
assertEqual(
  nextModelResultSort({ key: 'score', direction: 'descending' }, 'score').direction,
  'ascending',
  'sort toggle',
);
assertEqual(
  selectionForId(result, runSelectionId(result.training_history[0]!)).selectedRun?.run_slug,
  'train-a',
  'run selection',
);
const emptyPlotModel = benchmarkPlotModel(
  {
    ...result,
    frontiers: {},
    leaderboard: [],
  },
  'storage_bytes',
);
assertEqual(emptyPlotModel.points.length, 0, 'empty plot point count');
assertEqual(emptyPlotModel.xTicks.length > 0, true, 'empty plot has x ticks');
assertEqual(emptyPlotModel.yTicks.length > 0, true, 'empty plot has y ticks');
const expandedPlotModel = benchmarkPlotModel(
  {
    ...result,
    leaderboard: [
      ...result.leaderboard,
      {
        ...result.leaderboard[1]!,
        cost_summary: {
          ...result.leaderboard[1]!.cost_summary,
          storage_bytes: 2 ** 22,
        },
        model_key: 'model-c',
      },
    ],
    plot_runs: [
      ...result.plot_runs,
      {
        ...result.plot_runs[1]!,
        cost_summary: {
          ...result.plot_runs[1]!.cost_summary,
          storage_bytes: 2 ** 22,
        },
        model_key: 'model-c',
        run_id: 'run-c',
        run_slug: 'train-c',
      },
    ],
  },
  'storage_bytes',
);
assertEqual(expandedPlotModel.xDomain[1], 22, 'plot x maximum expands by log2 step');

const parsedResultViews = parseResultViewRecords([
  {
    benchmark_results: [
      {
        ...result,
        model_inspections: [
          {
            ...inspections[0]!,
            model_artifacts: undefined,
            training_provenance: undefined,
          },
        ],
      },
    ],
    format: 'leibniz.console.benchmark-results',
    format_version: 1,
    source_path: 'results/views/digits/benchmark_results.json',
  },
]);
const parsedBenchmarkResult = parsedResultViews[0];
if (parsedBenchmarkResult?.format !== 'leibniz.console.benchmark-results') {
  throw new Error('parsed benchmark result view must keep its discriminant');
}
assertEqual(
  parsedBenchmarkResult.benchmark_results[0]?.model_inspections[0]?.model_artifacts.length,
  0,
  'parser defaults missing model artifacts',
);
assertEqual(
  parsedBenchmarkResult.benchmark_results[0]?.model_inspections[0]?.training_provenance.length,
  0,
  'parser defaults missing training provenance',
);
assertThrows(
  () =>
    parseResultViewRecords([
      {
        format: 'leibniz.console.unknown-results',
        format_version: 1,
        source_path: 'results/unknown_results.json',
      },
    ]),
  'parser rejects unsupported result views',
);
assertThrows(
  () =>
    parseResultViewRecords([
      {
        benchmark_results: [
          {
            ...result,
            leaderboard: [
              {
                ...result.leaderboard[0]!,
                console_view_model: {},
              },
            ],
          },
        ],
        format: 'leibniz.console.benchmark-results',
        format_version: 1,
        source_path: 'results/views/digits/benchmark_results.json',
      },
    ]),
  'parser rejects malformed console view models',
);

function assertEqual(actual: unknown, expected: unknown, label: string) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

function assertThrows(callback: () => unknown, label: string) {
  try {
    callback();
  } catch {
    return;
  }
  throw new Error(`${label}: expected exception`);
}
