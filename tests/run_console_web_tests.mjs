import { spawnSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { delimiter, dirname, resolve } from 'node:path';
import { tmpdir } from 'node:os';
import {
  consoleResultRoots,
  resultRootArguments,
} from '../src/leibniz/console/_web_src/vite.config.mjs';

const testsRoot = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(testsRoot, '..');
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
assertBenchmarkWorkbenchStructure();
assertBenchmarkSamplePaneStructure();
assertBenchmarkFrontierPlotStructure();
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
  if (shell.includes('demoArtifact')) {
    throw new Error('ConsoleShell must not import handwritten demo artifact data');
  }
  if (!shell.includes("{ id: 'benchmarks', label: 'Benchmarks' }")) {
    throw new Error('ConsoleShell must expose a Benchmarks tab');
  }
  if (shell.includes("{ id: 'data', label: 'Data' }")) {
    throw new Error('ConsoleShell must not expose a top-level Data tab');
  }
  if (shell.includes("{ id: 'performance', label: 'Performance' }")) {
    throw new Error('ConsoleShell must not expose a top-level Performance tab');
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
    'benchmark-selector-row',
    'benchmark-status-row',
    'benchmark-section-summary',
    'benchmark-section-actions',
  ];
  for (const marker of requiredPanelMarkers) {
    if (!panel.includes(marker)) {
      throw new Error(`BenchmarksPanel must expose benchmark workbench marker: ${marker}`);
    }
  }
  if (panel.includes('benchmark-sidebar') || panel.includes('benchmark-list-item')) {
    throw new Error('BenchmarksPanel must not use the old sidebar benchmark layout');
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
  ];
  for (const marker of requiredDashboardMarkers) {
    if (!dashboard.includes(marker)) {
      throw new Error(`BenchmarkResultDashboard must expose frontier plot marker: ${marker}`);
    }
  }
  if (dashboard.includes('Reset</button>')) {
    throw new Error('Frontier plot reset must live in the benchmark section action slot');
  }
  if (!panel.includes('Reset Zoom') || !panel.includes('resetToken')) {
    throw new Error('BenchmarksPanel must route frontier reset through section actions');
  }

  const requiredStyleMarkers = [
    '--frontier-grid-major',
    '--frontier-frame-bg',
    '--frontier-tooltip-bg',
    '.frontier-chart-legend',
    '.frontier-chart-proposal-guide',
  ];
  for (const marker of requiredStyleMarkers) {
    if (!styles.includes(marker)) {
      throw new Error(`Console styles must preserve frontier plot marker: ${marker}`);
    }
  }
}

function assertConsoleResultRootPolicy() {
  const viteConfig = readFileSync(
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/vite.config.mjs'),
    'utf8',
  );
  if (viteConfig.includes('addWatchFile')) {
    throw new Error('Console result roots must be watched through the dev-server watcher');
  }

  const tempRoot = mkdtempSync(resolve(tmpdir(), 'leibniz-console-roots-'));
  try {
    assertEqual(consoleResultRoots({}, tempRoot).length, 0, 'missing default result root');
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
