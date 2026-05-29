import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { delimiter, dirname, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import {
  consoleResultRoots,
  consoleResultWatchRoots,
  resultRootArguments,
} from '../src/leibniz/console/_web_src/vite.config.mjs';

const testsRoot = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(testsRoot, '..');
const consoleWebBinPath = resolve(
  repositoryRoot,
  'src/leibniz/console/_web_src/node_modules/.bin',
);
const generatedPayloadPath = resolve(tmpdir(), 'leibniz-console-data.contract.json');
const pythonPath = [resolve(repositoryRoot, 'src'), process.env.PYTHONPATH]
  .filter((path) => path !== undefined && path !== '')
  .join(delimiter);

const contracts = [
  'tests/console_artifact_browser.contract.ts',
  'tests/console_benchmark_dashboard.contract.ts',
  'tests/console_data_transport.contract.ts',
  'tests/console_artifact_index_transport.contract.ts',
];
const generatedDataContracts = new Set([
  'tests/console_artifact_browser.contract.ts',
  'tests/console_data_transport.contract.ts',
]);

const generatedPayload = run(
  'python',
  ['-m', 'leibniz.console.data', 'tests/fixtures', 'src/leibniz/benchmarks'],
  {
    captureOutput: true,
    env: { PYTHONPATH: pythonPath },
  },
);
writeFileSync(generatedPayloadPath, generatedPayload);
assertShellUsesGeneratedConsoleData();
assertConsoleShellSurfaceIsConsolidated();
assertBenchmarkWorkbenchStructure();
assertBenchmarkSamplePaneStructure();
assertBenchmarkFrontierPlotStructure();
assertBenchmarkModelWorkbenchStructure();
assertConsoleTextIsUseful();
assertBenchmarkWebSourceIsDataDriven();
assertConsoleResultRootPolicy();

for (const contract of contracts) {
  run('tsc', [
    '--ignoreConfig',
    '--noEmit',
    '--target',
    'ES2023',
    '--module',
    'ESNext',
    '--moduleResolution',
    'Bundler',
    '--allowImportingTsExtensions',
    '--strict',
    '--verbatimModuleSyntax',
    '--skipLibCheck',
    contract,
  ]);
  if (generatedDataContracts.has(contract)) {
    const payload = readFileSync(generatedPayloadPath, 'utf8');
    const script = [
      `globalThis.consoleDataPayload = ${payload};`,
      `await import('./${contract}');`,
    ].join('\n');
    run('node', ['--experimental-strip-types', '--eval', script]);
  } else {
    run('node', ['--experimental-strip-types', contract]);
  }
}

function assertShellUsesGeneratedConsoleData() {
  const shell = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/ConsoleShell.tsx'),
    'utf8',
  );
  if (!shell.includes("from 'virtual:leibniz-console-data'")) {
    throw new Error('ConsoleShell must import generated console data');
  }
  if (!shell.includes('subscribeConsoleData')) {
    throw new Error('ConsoleShell must subscribe to console data updates');
  }
  if (shell.includes('demoArtifact')) {
    throw new Error('ConsoleShell must not import handwritten demo artifact data');
  }
  if (!shell.includes("{ id: 'benchmarks', label: 'Benchmarks' }")) {
    throw new Error('ConsoleShell must expose a Benchmarks tab');
  }
}

function assertConsoleShellSurfaceIsConsolidated() {
  const shell = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/ConsoleShell.tsx'),
    'utf8',
  );
  const consoleData = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/consoleData.ts'),
    'utf8',
  );
  const bannedShellMarkers = [
    "{ id: 'home'",
    "{ id: 'data'",
    "{ id: 'performance'",
    "{ id: 'models'",
    "{ id: 'source'",
    'console-grid',
    'console-section',
    'ModelInspectionPanel',
    'SourceModuleInventory',
  ];
  for (const marker of bannedShellMarkers) {
    if (shell.includes(marker)) {
      throw new Error(`ConsoleShell must not expose retired surface marker: ${marker}`);
    }
  }
  if (!shell.includes("useState<TabId>('benchmarks')")) {
    throw new Error('ConsoleShell must default to the Benchmarks tab');
  }
  if (consoleData.includes('source_modules') || consoleData.includes('SourceModule')) {
    throw new Error('consoleData transport must not include Source-only module inventory');
  }
}

function assertBenchmarkWebSourceIsDataDriven() {
  const sourceRoot = resolve(repositoryRoot, 'src/leibniz/console/_web_src/src');
  const bannedPatterns = [
    /\bDigitsBenchmark\b/,
    /\bDigitsTask\b/,
    /\bDigitsSample\b/,
    /benchmarks\.digits/,
    /kind\s*!==\s*['"]digits['"]/,
    /kind\s*:\s*['"]digits['"]/,
    /\.digits-/,
    /['"]symbol-probe['"]/,
    /['"]complexity-sweep['"]/,
  ];
  for (const path of webSourceFiles(sourceRoot)) {
    const relativePath = path.slice(sourceRoot.length + 1);
    const source = readFileSync(path, 'utf8');
    for (const pattern of bannedPatterns) {
      if (pattern.test(source)) {
        throw new Error(`${relativePath} hard-codes benchmark-specific presentation: ${pattern}`);
      }
    }
  }
}

function assertBenchmarkWorkbenchStructure() {
  const panel = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/BenchmarksPanel.tsx'),
    'utf8',
  );
  const styles = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/styles.css'),
    'utf8',
  );
  const requiredPanelMarkers = [
    'benchmark-workbench',
    'benchmark-title-select',
    'benchmark-section-summary',
    'workQueueItemsForTask',
    'workQueueStatusLabel',
    'resultUpdatedLabel',
    'resultSizeLabel',
  ];
  for (const marker of requiredPanelMarkers) {
    if (!panel.includes(marker)) {
      throw new Error(`BenchmarksPanel must expose benchmark workbench marker: ${marker}`);
    }
  }
  if (panel.includes('benchmark-sidebar') || panel.includes('benchmark-list-item')) {
    throw new Error('BenchmarksPanel must not use the old sidebar benchmark layout');
  }
  if (panel.includes('benchmark-selector-row')) {
    throw new Error('BenchmarksPanel must keep benchmark selection in the header');
  }

  const requiredThemeTokens = [
    '--frontier-accent',
    '--proposal-accent',
    '--measured-accent',
    '--benchmark-workbench-wide',
  ];
  for (const token of requiredThemeTokens) {
    if (!styles.includes(token)) {
      throw new Error(`Console styles must define benchmark workbench token: ${token}`);
    }
  }
}

function assertBenchmarkSamplePaneStructure() {
  const panel = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/BenchmarksPanel.tsx'),
    'utf8',
  );
  const styles = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/styles.css'),
    'utf8',
  );
  const requiredPanelMarkers = [
    'BenchmarkSampleCoordinateInspector',
    'benchmark-sample-coordinate-inspector',
    'benchmark-sample-caption',
  ];
  for (const marker of requiredPanelMarkers) {
    if (!panel.includes(marker)) {
      throw new Error(`BenchmarksPanel must expose sample pane marker: ${marker}`);
    }
  }
  if (panel.includes('benchmark-sample-card-body') || panel.includes('benchmark-sample-card-title')) {
    throw new Error('Benchmark sample cards must remain tile/caption oriented');
  }

  const requiredStyleMarkers = [
    '--benchmark-sample-tile-size',
    '--benchmark-sample-caption-height',
    '.benchmark-sample-coordinate-inspector',
    'text-overflow: ellipsis',
  ];
  for (const marker of requiredStyleMarkers) {
    if (!styles.includes(marker)) {
      throw new Error(`Console styles must preserve sample pane marker: ${marker}`);
    }
  }
}

function assertBenchmarkFrontierPlotStructure() {
  const dashboard = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/BenchmarkResultDashboard.tsx'),
    'utf8',
  );
  const panel = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/BenchmarksPanel.tsx'),
    'utf8',
  );
  const styles = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/styles.css'),
    'utf8',
  );
  const requiredDashboardMarkers = [
    'frontier-chart-legend',
    'frontier-chart-proposal-guide',
    'frontier-chart-proposal-cap',
    'frontier-chart-tooltip-kicker',
    'ProposalAcquisitionComponents',
    'acquisitionComponentRows',
    'proposal-card-command',
  ];
  for (const marker of requiredDashboardMarkers) {
    if (!dashboard.includes(marker)) {
      throw new Error(`BenchmarkResultDashboard must expose frontier plot marker: ${marker}`);
    }
  }
  const requiredControlMarkers = [
    'aria-label="Pan left"',
    'aria-label="Pan right"',
    'aria-label="Zoom in"',
    'aria-label="Zoom out"',
    'aria-label="Reset plot"',
    'pannedView',
  ];
  for (const marker of requiredControlMarkers) {
    if (!dashboard.includes(marker)) {
      throw new Error(`BenchmarkResultDashboard must expose plot control marker: ${marker}`);
    }
  }
  if (panel.includes('Reset Zoom') || panel.includes('resetToken')) {
    throw new Error('BenchmarksPanel must not route frontier controls through section actions');
  }
  for (const marker of ['Frontier Plot', 'performance-metrics', 'title="Frontier"']) {
    if (dashboard.includes(marker)) {
      throw new Error(`BenchmarkResultDashboard must not preserve retired frontier summary UI: ${marker}`);
    }
  }

  const requiredStyleMarkers = [
    '--frontier-grid-major',
    '--frontier-frame-bg',
    '--frontier-tooltip-bg',
    '.frontier-chart-legend',
    '.frontier-chart-proposal-guide',
    '.proposal-acquisition-components',
  ];
  for (const marker of requiredStyleMarkers) {
    if (!styles.includes(marker)) {
      throw new Error(`Console styles must preserve frontier plot marker: ${marker}`);
    }
  }
}

function assertBenchmarkModelWorkbenchStructure() {
  const panel = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/BenchmarksPanel.tsx'),
    'utf8',
  );
  const styles = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/styles.css'),
    'utf8',
  );
  const requiredPanelMarkers = [
    'benchmark-model-workbench',
    'benchmark-model-rail',
    'benchmark-model-artifact-hero',
    'benchmark-model-artifact-flow',
    'benchmark-model-lineage-graph',
    'benchmark-model-training-grid',
    'benchmark-model-validation-chart',
    'benchmark-model-cost-grid',
    'benchmark-model-layer-shape-grid',
  ];
  for (const marker of requiredPanelMarkers) {
    if (!panel.includes(marker)) {
      throw new Error(`BenchmarksPanel must expose model workbench marker: ${marker}`);
    }
  }
  if (panel.includes('benchmark-model-grid')) {
    throw new Error('BenchmarksPanel must not use the old model card grid');
  }

  const requiredStyleMarkers = [
    '--measured-accent-bg',
    '--measured-accent-border',
    '.benchmark-model-artifact-flow',
    '.benchmark-model-lineage-graph',
    '.benchmark-model-validation-chart',
    '.benchmark-model-layer-shape-grid',
  ];
  for (const marker of requiredStyleMarkers) {
    if (!styles.includes(marker)) {
      throw new Error(`Console styles must preserve model workbench marker: ${marker}`);
    }
  }
}

function assertConsoleTextIsUseful() {
  const webRoot = resolve(repositoryRoot, 'src/leibniz/console/_web_src/src');
  const bannedPatterns = [
    /fixture/i,
    /Best-known score by model cost/,
    /No active proposals are available/,
    /No training history is available/,
    /No [^.\n]+ records are available/,
    /Read-only architecture/,
    /Protocol documents, digests/,
    /Typed views over already-public/,
  ];
  for (const path of webSourceFiles(webRoot)) {
    const relativePath = path.slice(webRoot.length + 1);
    const source = readFileSync(path, 'utf8');
    for (const pattern of bannedPatterns) {
      if (pattern.test(source)) {
        throw new Error(`${relativePath} contains retired explanatory UI copy: ${pattern}`);
      }
    }
  }
}

function assertConsoleResultRootPolicy() {
  const resultViewRecords = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/generated/resultViewRecords.ts'),
    'utf8',
  );
  for (const marker of [
    'WorkQueueViewRecord',
    'WorkQueueItemRecord',
    'parseWorkQueueViewRecord',
    'parseWorkQueueItem',
    'isWorkQueueView',
    'acquisition_model',
    'acquisition_components',
  ]) {
    if (!resultViewRecords.includes(marker)) {
      throw new Error(`Result view transport must support work queue marker: ${marker}`);
    }
  }

  const viteConfig = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/vite.config.mjs'),
    'utf8',
  );
  if (viteConfig.includes('addWatchFile')) {
    throw new Error('Console result roots must be watched through the dev-server watcher');
  }
  if (viteConfig.includes("type: 'full-reload'")) {
    throw new Error('Console result polling must update console data without a full page reload');
  }
  if (
    !viteConfig.includes("type: 'custom'") ||
    !viteConfig.includes('leibniz-console-data:update')
  ) {
    throw new Error('Console result polling must send a custom console data update event');
  }

  const tempRoot = mkdtempSync(resolve(tmpdir(), 'leibniz-console-roots-'));
  try {
    assertEqual(consoleResultRoots({}, tempRoot).length, 0, 'missing default result root');
    assertEqual(consoleResultWatchRoots({}, tempRoot).length, 0, 'missing default watch root');
    mkdirSync(resolve(tempRoot, '.runs'), { recursive: true });
    assertEqual(
      consoleResultWatchRoots({}, tempRoot)[0],
      resolve(tempRoot, '.runs'),
      'default watch root',
    );
    const defaultRoot = resolve(tempRoot, '.runs/views');
    mkdirSync(defaultRoot, { recursive: true });
    assertEqual(consoleResultRoots({}, tempRoot)[0], defaultRoot, 'default result root');

    const explicitRoot = resolve(tempRoot, 'publication-results');
    const explicitMissingRoot = resolve(tempRoot, 'missing-results');
    const env = {
      LEIBNIZ_CONSOLE_RESULT_ROOTS: [explicitRoot, explicitMissingRoot].join(delimiter),
    };
    assertEqual(
      consoleResultRoots(env, tempRoot).join('|'),
      [explicitRoot, explicitMissingRoot].join('|'),
      'explicit result roots',
    );
    assertEqual(
      consoleResultWatchRoots(env, tempRoot).join('|'),
      [explicitRoot, explicitMissingRoot].join('|'),
      'explicit watch roots',
    );
    assertEqual(
      resultRootArguments([explicitRoot]).join('|'),
      ['--result-root', explicitRoot].join('|'),
      'result root arguments',
    );
  } finally {
    rmSync(tempRoot, { force: true, recursive: true });
  }
}

function webSourceFiles(root) {
  return readdirSync(root)
    .flatMap((entry) => {
      const path = resolve(root, entry);
      if (statSync(path).isDirectory()) {
        return webSourceFiles(path);
      }
      return [path];
    })
    .filter((path) => /\.(css|ts|tsx)$/.test(path));
}

function assertEqual(actual, expected, label) {
  if (actual !== expected) {
    throw new Error(`${label}: expected ${String(expected)}, got ${String(actual)}`);
  }
}

function run(command, args, options = {}) {
  const result = spawnSync(command, args, {
    cwd: repositoryRoot,
    env: {
      ...process.env,
      PATH: [consoleWebBinPath, process.env.PATH].filter(Boolean).join(delimiter),
      ...options.env,
    },
    stdio: options.captureOutput === true ? ['ignore', 'pipe', 'inherit'] : 'inherit',
  });
  if (result.error !== undefined) {
    throw result.error;
  }
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
  if (options.captureOutput === true) {
    return result.stdout.toString('utf8');
  }
  return '';
}
