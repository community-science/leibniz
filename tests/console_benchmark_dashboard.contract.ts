import {
  benchmarkCostAxes,
  benchmarkCostAxis,
  benchmarkPlotModel,
  benchmarkResultsForTask,
  emptyFrontiersForCostAxes,
  modelComparisonRows,
  nextModelResultSort,
  proposalAssociations,
  runDetails,
  runSelectionId,
  selectionForId,
  sortedModelResults,
} from '../src/leibniz/console/_web_src/src/benchmarkDashboardModel.ts';
import type { ModelInspectionRecord } from '../src/leibniz/console/_web_src/src/modelInspections.ts';
import type {
  BenchmarkResultRecord,
  ResultViewRecord,
} from '../src/leibniz/console/_web_src/src/resultViews.ts';

const targetBenchmark = 'benchmarks.target@0.1.0';
const otherBenchmark = 'benchmarks.other@0.1.0';
const architectureDigest = 'sha256:abcdef1234567890';

const standardAxes = benchmarkCostAxes(undefined);
assertEqual(standardAxes.map((axis) => axis.key).join(','),
  'parameter_count,inference_flops,parameter_bytes',
  'standard cost axes',
);
assertEqual(
  Object.keys(emptyFrontiersForCostAxes(standardAxes)).join(','),
  'parameter_count,inference_flops,parameter_bytes',
  'empty frontier axes',
);
assertEqual(
  benchmarkCostAxis('missing_axis', standardAxes),
  'parameter_count',
  'missing cost axis fallback',
);

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
  model_inspections: [],
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
      search_diagnostics: {
        nearest_measured_support: {
          architecture_digest: 'sha256:abcd',
          log_parameter_distance: 0.5,
          parameter_count: 10,
          score: 0.75,
        },
        sampled_resource_stratum: { count: 2, index: 0 },
        search_distribution_id: 'architecture-search-distributions.sha-abc@0.1.0',
        semantic_coordinates: [
          { name: 'operator.0.support', value: 'local-window' },
          { name: 'operator.0.local_support_size', value: 2 },
        ],
      },
      uncertainty: 0.05,
    },
  ],
  training_history: [
    {
      architecture: { layers: [] },
      architecture_digest: architectureDigest,
      benchmark_id: targetBenchmark,
      cost_summary: {
        inference_flops: 20,
        layer_count: 1,
        parameter_bytes: 40,
        parameter_count: 10,
      },
      measurement_count: 2,
      measurement_dataset_digest: 'sha256:dataset1234',
      model_key: 'model-a',
      run_id: 'run-a',
      run_slug: 'train-a',
      scale: 1,
      score: 0.75,
      source_kind: 'local',
      source_path: '.runs/training/run-a.json',
      training_diagnostics: {
        artifacts: [
          { digest: 'sha256:dataset1234', kind: 'measurement-dataset' },
          { digest: architectureDigest, kind: 'model-inspection', path: '.runs/models/a.json' },
          { digest: 'sha256:training1234', kind: 'training-summary', path: '.runs/training/run-a.json' },
        ],
        best_validation_check: 1,
        best_validation_loss: 0.4,
        best_validation_step: 3,
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
          validation_interval: 1,
          validation_sample_count: 2,
          validation_source: 'generator-resample',
        },
        status: 'budget-exhausted',
        steps_run: 3,
        stop_reason: 'max-steps',
        validation_checks: 2,
        validation_history: [
          {
            best_validation_check: 0,
            best_validation_loss: 0.6,
            best_validation_step: 0,
            stale_checks: 0,
            step: 0,
            validation_check: 0,
            validation_loss: 0.6,
          },
          {
            best_validation_check: 1,
            best_validation_loss: 0.4,
            best_validation_step: 3,
            stale_checks: 0,
            step: 3,
            validation_check: 1,
            validation_loss: 0.4,
          },
        ],
      },
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
    source_path: 'results/benchmark_results.json',
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
      inference_flops: 20,
      component_count: 1,
      parameter_bytes: 40,
      parameter_count: 10,
      unknown_flop_components: [],
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
          inference_flops: 0,
          input_shape: [1],
          kind: 'operator',
          operator_kind: 'identity',
          output_shape: [2],
          parameter_count: 0,
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
const plotModel = benchmarkPlotModel(result, 'parameter_count');
assertEqual(plotModel.points.length, 2, 'plot point count');
assertEqual(plotModel.frontierPoints.length, 1, 'plot frontier count');
assertEqual(plotModel.proposals.length, 1, 'plot proposal count');
assertEqual(plotModel.proposals[0]?.cost, 10, 'plot proposal cost');
assertEqual(plotModel.staircase.length, 1, 'plot staircase point count');
assertEqual(plotModel.xTicks.includes(16), true, 'plot log ticks');
assertEqual(plotModel.xDomain[0], 0, 'plot default x minimum');
assertEqual(plotModel.xDomain[1], 20, 'plot default x maximum');
assertEqual(plotModel.xMajorTicks.includes(1), true, 'plot major x ticks');
assertEqual(plotModel.xMinorTicks.includes(2), true, 'plot minor x ticks');
assertEqual(plotModel.yDomain[0], 0, 'plot y starts at zero');
assertEqual(plotModel.yDomain[1], 1.05, 'plot y ceiling follows score scale');
assertEqual(
  sortedModelResults(result.leaderboard, 'parameter_count', {
    key: 'cost',
    direction: 'descending',
  })[0]?.model_key,
  'model-b',
  'model cost sort',
);
assertEqual(
  nextModelResultSort({ key: 'score', direction: 'descending' }, 'score').direction,
  'ascending',
  'sort toggle',
);
assertEqual(proposalAssociations(result)[0]?.model?.model_key, 'model-a', 'proposal model match');
assertEqual(runDetails(result)[0]?.model?.model_key, 'model-a', 'run model match');
assertEqual(
  selectionForId(result, 'proposal-a').selectedProposal?.candidate_id,
  'model-a',
  'proposal selection',
);
assertEqual(
  selectionForId(result, 'proposal-a').selectedProposal?.search_diagnostics
    ?.search_distribution_id,
  'architecture-search-distributions.sha-abc@0.1.0',
  'proposal search diagnostics',
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
    proposals: [],
  },
  'parameter_count',
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
          parameter_count: 2 ** 22,
        },
        model_key: 'model-c',
      },
    ],
  },
  'parameter_count',
);
assertEqual(expandedPlotModel.xDomain[1], 22, 'plot x maximum expands by log2 step');

function assertEqual(actual: unknown, expected: unknown, label: string) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}
