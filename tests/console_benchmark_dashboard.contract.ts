import {
  benchmarkCostAxisKey,
  benchmarkCostAxisLabel,
  benchmarkPlotModel,
  benchmarkResultsForTask,
  emptyFrontiersForCostAxis,
  modelComparisonRows,
  nextModelResultSort,
  runSelectionId,
  scoreTickLabel,
  selectionForId,
  sortedModelResults,
} from '../src/leibniz/console/web/src/benchmarkDashboardModel.ts';
import type { ModelInspectionRecord } from '../src/leibniz/console/web/src/modelInspections.ts';
import type {
  BenchmarkResultRecord,
  ModelResultRecord,
  ResultViewRecord,
} from '../src/leibniz/console/web/src/resultViews.ts';
import { parseResultViewRecords } from '../src/leibniz/console/web/src/resultViews.ts';
import type { StateSpaceRegionRecord } from '../src/leibniz/console/web/src/stateSpaceRecords.ts';

const targetBenchmark = 'benchmarks.target@0.1.0';
const otherBenchmark = 'benchmarks.other@0.1.0';
const architectureDigest = 'sha256:abcdef1234567890';
const programGraph = {
  contract_kind: 'single-input-single-output',
  edges: [],
  inputs: [{ axes: ['batch', 1], name: 'input' }],
  nodes: [{ id: 'component-0', kind: 'identity', parameters: {} }],
  outputs: [{ axes: ['batch', 2], name: 'output' }],
};
const fixtureRegion: StateSpaceRegionRecord = {
  id: 'test.region',
  ambient: {
    field_domain_kind: 'lattice-2d',
    field_domain: { height: 2, width: 2 },
    field_codomain_id: 'unit-intensity',
    distinguishability: {
      kind: 'exact',
      certificate_id: 'test-certificate',
    },
  },
  components: [
    {
      axis_regions: [
        {
          axis: {
            id: 'x',
            domain: { kind: 'integer-range', lower: 0, upper: 1 },
          },
          coordinate_region: [0, 1],
          count: 2,
          log2_count: 1,
        },
      ],
      measure_rule: 'product-of-counts',
      volume: 2,
      log2_volume: 1,
      stratum_id: 'fixture',
      stratum_target: { label: 'fixture' },
    },
  ],
  union_rule: 'disjoint-union',
  volume: 2,
  log2_volume: 1,
};

function modelResult({
  architectureDigest,
  cost,
  inferenceCompute,
  measurementCount,
  modelKey,
  runIds,
  score,
  storageBytes,
  trainingCompute,
}: {
  architectureDigest: string;
  cost: number;
  inferenceCompute: number;
  measurementCount: number;
  modelKey: string;
  runIds: string[];
  score: number;
  storageBytes: number;
  trainingCompute: number;
}): ModelResultRecord {
  return {
    benchmark_id: targetBenchmark,
    cost_integral: {
      kind: 'compute-cost-integral',
      value: cost,
      terms: [
        {
          kind: 'measured-compute-cost',
          log2_volume_minimum: 0,
          log2_volume_maximum: score,
          width_in_bits: score,
          contribution: cost,
          representative_log2_volume: score,
        },
      ],
    },
    cost_summary: {
      component_count: 1,
      cost,
      inference_cost_measurement: { abstract_flops: inferenceCompute },
      inference_cost_sample_count: 1,
      storage_bytes: storageBytes,
    },
    measurement_count: measurementCount,
    model_key: modelKey,
    capability_map: {
      kind: 'partition-capability-map-v1',
      value: score,
      confidence_half_width: 0,
      confidence_method_id: 'integral-term-propagated-confidence',
      sample_count: measurementCount,
      total_measure: 4,
      score_width_bits: score,
      mean_competence: 0.75,
      mean_competence_confidence_half_width: 0,
      leaf_count: 2,
      refinement_ladder: [
        {
          kind: 'partition-refinement-step-v1',
          depth: 0,
          leaf_count: 1,
          value: score / 2,
          confidence_half_width: 0,
        },
        {
          kind: 'partition-refinement-step-v1',
          depth: 1,
          leaf_count: 2,
          value: score,
          confidence_half_width: 0,
          movement: score / 2,
        },
      ],
      root: {
        kind: 'partition-capability-node-v1',
        label: 'Capability map',
        measure: 4,
        sample_count: measurementCount,
        competence: 0.75,
        confidence_half_width: 0,
        children: [
          {
            kind: 'partition-capability-node-v1',
            label: 'test.region.parent',
            measure: 4,
            sample_count: measurementCount,
            competence: 0.75,
            confidence_half_width: 0,
            region: fixtureRegion,
            children: [
              {
                kind: 'partition-capability-node-v1',
                label: 'test.region.left',
                measure: 1,
                sample_count: 1,
                competence: 0,
                confidence_half_width: 0,
                region: fixtureRegion,
                children: [],
              },
              {
                kind: 'partition-capability-node-v1',
                label: 'test.region.right',
                measure: 3,
                sample_count: 1,
                competence: 1,
                confidence_half_width: 0,
                region: fixtureRegion,
                children: [],
              },
            ],
          },
        ],
      },
      diagnostics: {
        adaptive_sampling: 'deferred',
      },
    },
    points: [
      {
        competence_value_kind: 'validated-bits',
        log2_volume: score,
        predictability_boundary: 0.5,
        score,
        run_ids: runIds,
        time_points: [
          { bits: score / 2, certified_epsilon: 0.02, evolution_scale: 0.2, time: 0.25 },
          { bits: score, certified_epsilon: 0.03, evolution_scale: 0.3, time: 0.5 },
        ],
      },
    ],
    program_digest: architectureDigest,
    result_status: 'accepted',
    run_ids: runIds,
    score,
    score_integral: {
      kind: 'sampled-competence-integral',
      value: score,
      terms: [
        {
          kind: 'measured-state-space-competence',
          log2_volume_minimum: 0,
          log2_volume_maximum: score,
          width_in_bits: score,
          contribution: score,
          representative_log2_volume: score,
          confidence_half_width: 0,
          region: fixtureRegion,
        },
      ],
    },
    source_kinds: ['local'],
  };
}

const modelA = modelResult({
  architectureDigest,
  cost: 640,
  inferenceCompute: 20,
  measurementCount: 2,
  modelKey: 'model-a',
  runIds: ['run-a'],
  score: 0.75,
  storageBytes: 40,
  trainingCompute: 360,
});
const modelB = modelResult({
  architectureDigest: 'sha256:fedcba9876543210',
  cost: 2560,
  inferenceCompute: 80,
  measurementCount: 1,
  modelKey: 'model-b',
  runIds: ['run-b'],
  score: 0.5,
  storageBytes: 160,
  trainingCompute: 1440,
});

assertEqual(
  benchmarkCostAxisKey,
  'cost',
  'cost axis key',
);
assertEqual(benchmarkCostAxisLabel, 'Cost', 'cost axis label');
assertEqual(
  Object.keys(emptyFrontiersForCostAxis()).join(','),
  'cost',
  'empty frontier axis',
);
const result: BenchmarkResultRecord = {
  benchmark_id: targetBenchmark,
  frontiers: {
    cost: [modelA],
  },
  leaderboard: [modelA, modelB],
  model_candidates: [modelA, modelB],
  reference_curves: [
    {
      kind: 'oracle-cost-measurement-reference-v1',
      key: 'oracle_cost_measurement',
      label: 'Oracle Reference',
      x_axis: 'cost',
      y_axis: 'score',
      points: [
        { log2_volume: 1, score: 1, cost: 16 },
        { log2_volume: 2, score: 2, cost: 64 },
      ],
    },
  ],
  model_inspections: [],
  training_history: [
    {
      benchmark_id: targetBenchmark,
      cost_summary: {
        component_count: 1,
        cost: 640,
        inference_cost_measurement: { abstract_flops: 20 },
        inference_cost_sample_count: 1,
        storage_bytes: 40,
      },
      measurement_count: 2,
      measurement_dataset_digest: 'sha256:dataset1234',
      model_key: 'model-a',
      program: { kind: 'program-source' },
      program_digest: architectureDigest,
      program_graph: programGraph,
      run_id: 'run-a',
      run_slug: 'train-a',
      log2_volume: 10,
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
        validation_loss_reference: Math.log(10),
        final_validation_step: 3,
        protocol: {
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
          gate_decision_rule: 'score-estimate-plateau',
          validation_source: 'generator-resample',
        },
        status: 'budget-exhausted',
        steps_run: 3,
        stop_reason: 'max-steps',
        validation_checks: 2,
        validation_history_sample_count: 2,
        validation_history_total_count: 2,
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
      benchmark_id: targetBenchmark,
      cost_summary: {
        component_count: 1,
        cost: 640,
        inference_cost_measurement: { abstract_flops: 20 },
        inference_cost_sample_count: 1,
        storage_bytes: 40,
      },
      measurement_count: 2,
      measurement_dataset_digest: 'sha256:dataset1234',
      model_key: 'model-a',
      program: { kind: 'program-source' },
      program_digest: architectureDigest,
      program_graph: programGraph,
      run_id: 'run-a',
      run_slug: 'train-a',
      log2_volume: 10,
      score: 0.75,
      result_status: 'accepted',
      source_kind: 'local',
      source_path: 'results/training/run-a.json',
    },
    {
      benchmark_id: targetBenchmark,
      cost_summary: {
        component_count: 2,
        cost: 2560,
        inference_cost_measurement: { abstract_flops: 80 },
        inference_cost_sample_count: 1,
        storage_bytes: 160,
      },
      measurement_count: 1,
      measurement_dataset_digest: 'sha256:dataset5678',
      model_key: 'model-b',
      program: { kind: 'program-source' },
      program_digest: 'sha256:fedcba9876543210',
      program_graph: {
        ...programGraph,
        nodes: [{ id: 'component-0', kind: 'dense', parameters: { width: 2 } }],
      },
      run_id: 'run-b',
      run_slug: 'train-b',
      log2_volume: 10,
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
    program: {
      kind: 'program-graph',
      record_digest: architectureDigest,
    },
    cost_summary: {
      inference_cost_measurement: { abstract_flops: 20 },
      inference_cost_sample_count: 1,
      component_count: 1,
      storage_bytes: 40,
      unknown_cost_components: [],
      unknown_parameter_components: [],
    },
    id: 'inspection-a',
    input_shape: [1],
    program_graph: programGraph,
    components: [
      {
        index: 0,
        kind: 'component-0',
        parameters: { program_node_kind: 'identity' },
      },
    ],
    model_artifacts: [],
    node_evidence: [
      {
        claim_kinds: ['program-structure', 'resource-accounting'],
        evidence_artifacts: [
          {
            kind: 'program-graph',
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
const plotModel = benchmarkPlotModel(result);
assertEqual(plotModel.points.length, 2, 'plot point count');
assertEqual(plotModel.frontierPoints.length, 1, 'plot frontier count');
assertEqual(plotModel.staircase.length, 1, 'plot staircase point count');
const provisionalScalePlotModel = benchmarkPlotModel(
  {
    ...result,
    leaderboard: [
      result.leaderboard[0]!,
    ],
    model_candidates: [
      result.model_candidates[0]!,
      {
        ...result.model_candidates[1]!,
        score: 6,
      },
    ],
    plot_runs: [
      result.plot_runs[0]!,
      {
        ...result.plot_runs[1]!,
        result_status: 'provisional',
        score: 0.5,
      },
    ],
  },
);
assertEqual(
  provisionalScalePlotModel.points.find((point) => point.resultStatus === 'provisional')?.score,
  6,
  'provisional plot points use score-axis scale',
);
assertEqual(
  provisionalScalePlotModel.points.find((point) => point.resultStatus === 'provisional')?.frontier,
  false,
  'provisional plot points are not frontier points',
);
const repeatedProgramResult: BenchmarkResultRecord = {
  ...result,
  leaderboard: [
    {
      ...result.leaderboard[0]!,
      model_key: 'checkpoint-a',
      run_ids: ['run-a'],
      score: 8,
    },
    {
      ...result.leaderboard[0]!,
      model_key: 'checkpoint-b',
      run_ids: ['run-b'],
      score: 9,
    },
  ],
  model_candidates: [],
  plot_runs: [
    {
      ...result.plot_runs[0]!,
      model_key: 'checkpoint-a',
      run_id: 'run-a',
      run_slug: 'train-a',
      score: 0.2,
    },
    {
      ...result.plot_runs[0]!,
      model_key: 'checkpoint-b',
      run_id: 'run-b',
      run_slug: 'train-b',
      score: 0.3,
    },
  ],
};
const repeatedProgramPlotModel = benchmarkPlotModel(repeatedProgramResult);
assertEqual(
  repeatedProgramPlotModel.points.find((point) => point.run?.run_id === 'run-a')?.score,
  8,
  'same-program run keeps first checkpoint score',
);
assertEqual(
  repeatedProgramPlotModel.points.find((point) => point.run?.run_id === 'run-b')?.score,
  9,
  'same-program run keeps second checkpoint score',
);
assertEqual(
  repeatedProgramPlotModel.points.find((point) => point.run?.run_id === 'run-a')?.frontier,
  false,
  'same-program run does not inherit another checkpoint frontier status',
);
assertEqual(plotModel.xTicks.includes(10), true, 'plot log ticks');
assertEqual(plotModel.xDomain[0], 0, 'plot default x minimum');
assertEqual(plotModel.xDomain[1], 10, 'plot default x maximum');
assertEqual(plotModel.xMajorTicks.includes(1), true, 'plot major x ticks');
assertEqual(plotModel.xMajorTicks.includes(10), true, 'cost uses base-10 ticks');
assertEqual(plotModel.xMajorTicks.includes(16), false, 'cost omits base-2 ticks');
const inferenceComputePlotModel = benchmarkPlotModel(result);
assertEqual(inferenceComputePlotModel.xMajorTicks.includes(10), true, 'inference compute uses base-10 ticks');
assertEqual(inferenceComputePlotModel.xMajorTicks.includes(16), false, 'inference compute omits base-2 ticks');
assertEqual(inferenceComputePlotModel.referenceCurves.length, 1, 'inference compute shows oracle reference');
assertEqual(inferenceComputePlotModel.referenceCurves[0]?.points.length, 2, 'oracle reference point count');
assertEqual(inferenceComputePlotModel.xDomain[1], 10, 'oracle reference does not expand x domain');
assertEqual(inferenceComputePlotModel.yDomain[1], 1, 'oracle reference does not expand y domain');
assertEqual(plotModel.referenceCurves.length, 1, 'cost axis shows inference oracle reference');
assertEqual(plotModel.yDomain[0], 0, 'plot y starts at zero');
assertEqual(plotModel.yDomain[1], 1, 'score plot y ceiling defaults to one');
assertEqual(plotModel.yTicks.join(','), '0,0.2,0.4,0.6,0.8,1', 'score plot y ticks');
assertEqual(scoreTickLabel(0.2), '0.2', 'fractional score tick label');
const expandedScoreScaleResult: BenchmarkResultRecord = {
  ...result,
  frontiers: {
    ...result.frontiers,
    cost: [
      {
        ...result.frontiers.cost[0]!,
        score: 2.6,
      },
    ],
  },
  leaderboard: [
    {
      ...result.leaderboard[0]!,
      score: 2.6,
    },
    result.leaderboard[1]!,
  ],
};
const expandedScorePlotModel = benchmarkPlotModel(expandedScoreScaleResult);
assertEqual(
  expandedScorePlotModel.yDomain[1],
  5.5,
  'score plot y ceiling expands to twice the maximum rounded to a grid tick',
);
assertEqual(
  expandedScorePlotModel.yTicks.join(','),
  '0,1,2,3,4,5',
  'expanded score plot y ticks',
);
assertEqual(
  sortedModelResults(result.leaderboard, {
    key: 'cost',
    direction: 'descending',
  })[0]?.model_key,
  'model-b',
  'model cost sort',
);
assertEqual(
  sortedModelResults(result.leaderboard, {
    key: 'score',
    direction: 'ascending',
  })[0]?.model_key,
  'model-b',
  'score sort',
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
);
assertEqual(emptyPlotModel.points.length, 0, 'empty plot point count');
assertEqual(emptyPlotModel.xTicks.length > 0, true, 'empty plot has x ticks');
assertEqual(emptyPlotModel.yTicks.length > 0, true, 'empty plot has y ticks');
const referenceOnlyPlotModel = benchmarkPlotModel(
  {
    ...result,
    frontiers: {},
    leaderboard: [],
    model_candidates: [],
    plot_runs: [],
    training_history: [],
  },
);
assertEqual(referenceOnlyPlotModel.points.length, 0, 'reference-only plot has no model points');
assertEqual(referenceOnlyPlotModel.referenceCurves.length, 1, 'reference-only plot keeps oracle curve');
assertEqual(referenceOnlyPlotModel.xDomain[1], 10, 'reference-only plot uses measured-data x default');
assertEqual(referenceOnlyPlotModel.yDomain[1], 1, 'reference-only plot uses measured-data y default');
const expandedPlotModel = benchmarkPlotModel(
  {
    ...result,
    leaderboard: [
      ...result.leaderboard,
      {
        ...result.leaderboard[1]!,
        cost_summary: {
          ...result.leaderboard[1]!.cost_summary,
          cost: 10 ** 22,
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
          cost: 10 ** 22,
        },
        model_key: 'model-c',
        run_id: 'run-c',
        run_slug: 'train-c',
      },
    ],
  },
);
assertEqual(expandedPlotModel.xDomain[1], 22, 'plot x maximum expands by log10 step');

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
const parsedPoint = parsedResultViews[0].benchmark_results[0].leaderboard[0].points[0];
assertEqual(parsedPoint.competence_value_kind, 'validated-bits', 'competence value kind');
assertEqual(parsedPoint.predictability_boundary, 0.5, 'predictability boundary');
assertEqual(parsedPoint.time_points?.[1]?.bits, 0.75, 'time point bits');
assertEqual(parsedPoint.time_points?.[1]?.certified_epsilon, 0.03, 'time point epsilon');
assertEqual(parsedPoint.time_points?.[1]?.evolution_scale, 0.3, 'time point scale');
assertEqual(
  parsedResultViews[0].benchmark_results[0]?.leaderboard[0]?.capability_map?.root.children[0]?.children[1]?.label,
  'test.region.right',
  'parser keeps capability map tree depth',
);
assertEqual(
  parsedResultViews[0].benchmark_results[0]?.leaderboard[0]?.capability_map?.refinement_ladder.length,
  2,
  'parser keeps multi-rung capability map ladder',
);
assertEqual(
  parsedResultViews[0].benchmark_results[0]?.leaderboard[0]?.capability_map?.refinement_ladder[1]?.leaf_count,
  2,
  'parser keeps refined capability map leaf count',
);
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
assertEqual(
  parsedBenchmarkResult.benchmark_results[0]?.leaderboard[0]?.score_integral.terms[0]?.region?.id,
  'test.region',
  'parser keeps integral term region',
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
