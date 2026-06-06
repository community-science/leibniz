import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { delimiter, dirname, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import {
  consoleDataPayloadPath,
  consoleBasePath,
  consoleResultWatchIgnoredPaths,
  consoleResultRoots,
  consoleResultWatchPaths,
  consoleResultWatchRoots,
  isModelArtifactEvent,
  isMaterializedResultViewEvent,
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
  'tests/console_benchmark_dashboard.contract.ts',
  'tests/console_data_transport.contract.ts',
];
const generatedDataContracts = new Set([
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
assertConsoleShellNavigation();
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
    const script = [
      "import { readFileSync } from 'node:fs';",
      "globalThis.consoleDataPayload = JSON.parse(readFileSync(process.argv[1], 'utf8'));",
      "await import(`./${process.argv[2]}`);",
    ].join('\n');
    run('node', ['--experimental-strip-types', '--eval', script, generatedPayloadPath, contract]);
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
}

function assertConsoleShellNavigation() {
  const shell = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/ConsoleShell.tsx'),
    'utf8',
  );
  for (const marker of [
    'const tabs:',
    "usePersistentState<TabId>(",
    "'leibniz.console.currentTab'",
    "const activeTab = tabs.some((tab) => tab.id === currentTab) ? currentTab : 'benchmarks'",
    'setCurrentTab(tab.id)',
    'aria-current={activeTab === tab.id',
    "hidden={activeTab !== 'benchmarks'}",
    'Benchmarks',
  ]) {
    if (!shell.includes(marker)) {
      throw new Error(`ConsoleShell must keep functional Benchmarks navigation: ${marker}`);
    }
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
    'benchmark-image-fit',
    'benchmark-sample-coordinate-inspector',
  ];
  for (const marker of requiredPanelMarkers) {
    if (!panel.includes(marker)) {
      throw new Error(`BenchmarksPanel must expose sample pane marker: ${marker}`);
    }
  }
  const requiredStyleMarkers = [
    '--benchmark-sample-tile-size',
    '.benchmark-image-fit',
    'box-sizing: border-box',
    'inset: 0',
    '.benchmark-sample-coordinate-inspector',
    'object-fit: contain',
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
    'frontier-chart-tooltip-kicker',
    'models={frontierModels}',
    'const renderedPoints = [...visiblePoints].sort(comparePlotPointRenderOrder)',
    'left.frontier ? 1 : -1',
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
  if (dashboard.includes('2<tspan')) {
    throw new Error('BenchmarkResultDashboard must render x-axis tick labels from the plot log base');
  }
  if (!dashboard.includes('{model.xLogBase}<tspan')) {
    throw new Error('BenchmarkResultDashboard must display the plot model x log base');
  }
  if (panel.includes('Reset Zoom') || panel.includes('resetToken')) {
    throw new Error('BenchmarksPanel must not route frontier controls through section actions');
  }
  for (const marker of ['Frontier Plot', 'performance-metrics', 'title="Frontier"']) {
    if (dashboard.includes(marker)) {
      throw new Error(`BenchmarkResultDashboard must not expose unsupported frontier summary UI: ${marker}`);
    }
  }

  const requiredStyleMarkers = [
    '--frontier-grid-major',
    '--frontier-frame-bg',
    '--frontier-tooltip-bg',
    '.frontier-chart-legend',
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
    'selectedModelKey',
    'onModelSelect',
    'benchmark-model-artifact-hero',
    'benchmark-model-artifact-flow',
    'benchmark-model-lineage-graph',
    'benchmark-model-training-grid',
    'benchmark-model-validation-chart',
    'benchmark-model-cost-grid',
    'Graph Operations',
    'Provenance',
    'modelProvenanceReferences',
    'benchmark-model-operation-shape-grid',
  ];
  for (const marker of requiredPanelMarkers) {
    if (!panel.includes(marker)) {
      throw new Error(`BenchmarksPanel must expose model workbench marker: ${marker}`);
    }
  }
  if (panel.includes('benchmark-model-rail') || panel.includes('benchmark-model-card')) {
    throw new Error('BenchmarksPanel must not expose unsupported model selector rail');
  }
  if (panel.includes("label: 'Artifacts'")) {
    throw new Error('BenchmarksPanel must label provenance by the records it actually renders');
  }
  if (panel.includes('run.architecture_digest === model.architecture_digest')) {
    throw new Error('BenchmarksPanel must not group repeated training runs by architecture digest');
  }

  const requiredStyleMarkers = [
    '--measured-accent-bg',
    '--measured-accent-border',
    'grid-template-columns: minmax(0, 1fr)',
    '.benchmark-model-artifact-flow',
    '.benchmark-model-lineage-graph',
    '.benchmark-model-validation-chart',
    '.benchmark-model-operation-shape-grid',
  ];
  for (const marker of requiredStyleMarkers) {
    if (!styles.includes(marker)) {
      throw new Error(`Console styles must preserve model workbench marker: ${marker}`);
    }
  }
  if (styles.includes('.benchmark-model-rail') || styles.includes('.benchmark-model-card')) {
    throw new Error('Console styles must not expose unsupported model selector rail');
  }
}

function assertConsoleTextIsUseful() {
  const webRoot = resolve(repositoryRoot, 'src/leibniz/console/_web_src/src');
  const bannedPatterns = [
    /fixture/i,
    /Best-known score by model cost/,
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
        throw new Error(`${relativePath} contains unsupported explanatory UI copy: ${pattern}`);
      }
    }
  }
}

function assertConsoleResultRootPolicy() {
  const resultViewRecords = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/src/generated/resultViewRecords.ts'),
    'utf8',
  );
  for (const marker of ['delete resultRecord']) {
    if (!resultViewRecords.includes(marker)) {
      continue;
    }
    throw new Error(`Result view transport must not expose local mutation marker: ${marker}`);
  }
  const viteConfig = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/vite.config.mjs'),
    'utf8',
  );
  if (viteConfig.includes('addWatchFile')) {
    throw new Error('Console result roots must be watched through the dev-server watcher');
  }
  if (!viteConfig.includes('readConsoleDataPayload()')) {
    throw new Error('Console dev startup must read prepared console data before page load');
  }
  if (!viteConfig.includes('refreshConsoleDataPayload()')) {
    throw new Error('Console result polling must refresh the prepared console data payload');
  }
  if (!viteConfig.includes('refreshConsoleDataPayloadAsync()')) {
    throw new Error('Console result polling must refresh asynchronously during development');
  }
  if (
    !viteConfig.includes('consoleDataPayloadMaxBuffer') ||
    !viteConfig.includes('maxBuffer: consoleDataPayloadMaxBuffer')
  ) {
    throw new Error('Console data refresh must size the Python stdout buffer explicitly');
  }
  if (!viteConfig.includes('consoleResultWatchPaths(')) {
    throw new Error('Console result polling must watch stable parent paths for deleted roots');
  }
  if (!viteConfig.includes('server.watcher.add(existingDirectories(resultRoots))')) {
    throw new Error('Console result polling must re-add recreated result roots');
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
    assertEqual(
      consoleResultWatchRoots({}, tempRoot)[0],
      resolve(tempRoot, 'results'),
      'missing default watch root',
    );
    assertEqual(
      consoleResultWatchPaths({}, tempRoot).join('|'),
      tempRoot,
      'missing default result root watches parent',
    );
    mkdirSync(resolve(tempRoot, 'results'), { recursive: true });
    assertEqual(
      consoleResultWatchRoots({}, tempRoot)[0],
      resolve(tempRoot, 'results'),
      'default watch root',
    );
    const defaultRoot = resolve(tempRoot, 'results');
    mkdirSync(defaultRoot, { recursive: true });
    mkdirSync(resolve(defaultRoot, 'views'), { recursive: true });
    assertEqual(consoleResultRoots({}, tempRoot)[0], defaultRoot, 'default result root');
    assertEqual(
      consoleResultWatchPaths({}, tempRoot).join('|'),
      [tempRoot, defaultRoot, resolve(defaultRoot, 'views')].join('|'),
      'default result root watches parent, root, and materialized views',
    );
    assertEqual(
      consoleResultWatchIgnoredPaths({}, tempRoot).join('|'),
      `${defaultRoot.replaceAll('\\', '/')}/models/**`,
      'default result root ignores raw model artifacts',
    );
    assertEqual(consoleBasePath({}), '/', 'default console base path');
    assertEqual(
      consoleBasePath({ LEIBNIZ_CONSOLE_BASE_PATH: '/leibniz-pages/' }),
      '/leibniz-pages/',
      'explicit console base path',
    );

    const explicitRoot = resolve(tempRoot, 'external-results');
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
      consoleResultWatchPaths(env, tempRoot).join('|'),
      tempRoot,
      'missing explicit result roots watch shared parent',
    );
    mkdirSync(explicitRoot, { recursive: true });
    assertEqual(
      consoleResultWatchPaths(env, tempRoot).join('|'),
      [tempRoot, explicitRoot].join('|'),
      'explicit result roots watch parent and existing root',
    );
    assertEqual(
      resultRootArguments([explicitRoot]).join('|'),
      ['--result-root', explicitRoot].join('|'),
      'result root arguments',
    );
    assertEqual(
      isMaterializedResultViewEvent(
        resolve(defaultRoot, 'views', 'digits', 'benchmark_results.json'),
        [defaultRoot],
      ),
      true,
      'default result root ignores materialized view events',
    );
    const explicitViewsRoot = resolve(defaultRoot, 'views');
    assertEqual(
      isMaterializedResultViewEvent(
        resolve(explicitViewsRoot, 'digits', 'benchmark_results.json'),
        [explicitViewsRoot],
      ),
      false,
      'explicit views root still refreshes on view events',
    );
    assertEqual(
      isModelArtifactEvent(
        resolve(defaultRoot, 'models', 'digits', 'run', 'gate0001-step00000032.pt'),
        [defaultRoot],
      ),
      true,
      'default result root ignores raw model checkpoint events',
    );
    assertEqual(
      isModelArtifactEvent(resolve(defaultRoot, 'training', 'digits', 'run.json'), [
        defaultRoot,
      ]),
      false,
      'training summaries still refresh console data',
    );
    assertEqual(
      consoleDataPayloadPath().endsWith('src/leibniz/console/_web_src/src/generated/consoleDataPayload.json'),
      true,
      'console data payload path',
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
