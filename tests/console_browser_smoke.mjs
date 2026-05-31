import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync } from 'node:fs';
import { createServer } from 'node:net';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';

const require = createRequire(
  new URL('../src/leibniz/console/_web_src/package.json', import.meta.url),
);
const { chromium } = require('playwright');

const host = '127.0.0.1';
const port = await freePort();
const consolePackageRoot = new URL('../src/leibniz/console/_web_src', import.meta.url);
const resultRoot = mkdtempSync(resolve(tmpdir(), 'leibniz-console-browser-results-'));
const smokeTimeout = setTimeout(() => {
  console.error('headless console browser smoke test timed out');
  process.exit(1);
}, 60_000);
const testEnv = {
  ...process.env,
  LEIBNIZ_CONSOLE_RESULT_ROOTS: resultRoot,
};
let preview = undefined;

try {
  await runConsoleCommand('npm', ['run', 'build'], { env: testEnv });
  preview = spawnConsoleCommand(
    'npx',
    ['vite', 'preview', '--host', host, '--port', String(port), '--strictPort'],
    { detached: process.platform !== 'win32', env: testEnv },
  );
  await waitForHttp(`http://${host}:${port}/`);
  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage();
    const browserFailures = [];
    page.on('console', (message) => {
      if (message.type() === 'error') {
        browserFailures.push(`console.error: ${message.text()}`);
      }
    });
    page.on('pageerror', (error) => {
      browserFailures.push(`pageerror: ${error.message}`);
    });
    await page.goto(`http://${host}:${port}/`, { waitUntil: 'networkidle' });
    await page.locator('#root').waitFor({ state: 'attached' });
    try {
      await page.locator('.benchmark-workbench').waitFor({ state: 'visible', timeout: 5_000 });
    } catch (error) {
      const rootText = await page.locator('#root').innerText().catch(() => '');
      const details = [
        `console root did not render the Benchmarks workbench: ${error}`,
        rootText === '' ? 'root text was empty' : `root text: ${rootText.slice(0, 500)}`,
        ...browserFailures,
      ];
      throw new Error(details.join('\n'));
    }
    if (browserFailures.length > 0) {
      throw new Error(browserFailures.join('\n'));
    }
    const samplesToggle = page.getByRole('button', { name: 'Samples' }).first();
    await samplesToggle.waitFor({ state: 'visible' });
    await samplesToggle.click();
    await page.reload({ waitUntil: 'networkidle' });
    await page.locator('.benchmark-workbench').waitFor({ state: 'visible', timeout: 5_000 });
    const persistedExpanded = await page
      .getByRole('button', { name: 'Samples' })
      .first()
      .getAttribute('aria-expanded');
    if (persistedExpanded !== 'false') {
      throw new Error(
        `expected collapsed Samples section to persist after reload, found ${persistedExpanded}`,
      );
    }
  } finally {
    await browser.close();
  }
} finally {
  if (preview !== undefined) {
    await stopPreview(preview);
  }
  rmSync(resultRoot, { force: true, recursive: true });
}

clearTimeout(smokeTimeout);
process.exit(0);

function spawnConsoleCommand(command, args, options = {}) {
  const executable = process.platform === 'win32' ? `${command}.cmd` : command;
  return spawn(executable, args, {
    cwd: consolePackageRoot,
    stdio: ['ignore', 'pipe', 'pipe'],
    ...options,
  });
}

async function runConsoleCommand(command, args, options = {}) {
  const child = spawnConsoleCommand(command, args, options);
  const output = [];
  child.stdout.on('data', (chunk) => output.push(String(chunk)));
  child.stderr.on('data', (chunk) => output.push(String(chunk)));
  const exitCode = await new Promise((resolve) => child.once('exit', resolve));
  if (exitCode !== 0) {
    throw new Error(`${command} ${args.join(' ')} failed:\n${output.join('')}`);
  }
}

async function stopPreview(child) {
  if (child.exitCode !== null || child.signalCode !== null) {
    return;
  }
  killProcessGroup(child);
  const exited = await Promise.race([
    new Promise((resolve) => child.once('exit', () => resolve(true))),
    new Promise((resolve) => setTimeout(() => resolve(false), 2_000)),
  ]);
  if (exited) {
    return;
  }
  killProcessGroup(child, 'SIGKILL');
  await Promise.race([
    new Promise((resolve) => child.once('exit', resolve)),
    new Promise((resolve) => setTimeout(resolve, 2_000)),
  ]);
}

function killProcessGroup(child, signal = 'SIGTERM') {
  if (process.platform !== 'win32' && child.pid !== undefined) {
    try {
      process.kill(-child.pid, signal);
      return;
    } catch (_error) {
      child.kill(signal);
      return;
    }
  }
  child.kill(signal);
}

async function freePort() {
  return await new Promise((resolve, reject) => {
    const server = createServer();
    server.on('error', reject);
    server.listen(0, host, () => {
      const address = server.address();
      if (address === null || typeof address === 'string') {
        server.close(() => reject(new Error('could not allocate TCP port')));
        return;
      }
      const selectedPort = address.port;
      server.close(() => resolve(selectedPort));
    });
  });
}

async function waitForHttp(url) {
  const deadline = Date.now() + 20_000;
  let lastError = undefined;
  while (Date.now() < deadline) {
    if (preview === undefined) {
      throw new Error('vite preview did not start');
    }
    if (preview.exitCode !== null) {
      throw new Error('vite preview exited early');
    }
    try {
      const response = await fetch(url);
      if (response.ok) {
        return;
      }
      lastError = new Error(`HTTP ${response.status}`);
    } catch (error) {
      lastError = error;
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`timed out waiting for ${url}: ${lastError}`);
}
