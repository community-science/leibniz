import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { tmpdir } from 'node:os';

const testsRoot = dirname(fileURLToPath(import.meta.url));
const repositoryRoot = resolve(testsRoot, '..');
const generatedPayloadPath = resolve(tmpdir(), 'leibniz-console-data.contract.json');

const contracts = [
  'tests/console_artifact_browser.contract.ts',
  'tests/console_data_transport.contract.ts',
  'tests/console_artifact_index_transport.contract.ts',
];
const generatedDataContracts = new Set([
  'tests/console_artifact_browser.contract.ts',
  'tests/console_data_transport.contract.ts',
]);

const generatedPayload = run('python', ['-m', 'leibniz.console.data', 'tests/fixtures'], {
  captureOutput: true,
});
writeFileSync(generatedPayloadPath, generatedPayload);
assertShellUsesGeneratedConsoleData();

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
