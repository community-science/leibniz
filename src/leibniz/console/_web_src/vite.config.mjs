import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { execFile, execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { delimiter, relative } from 'node:path';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const consoleDataModuleId = 'virtual:leibniz-console-data';
const resolvedConsoleDataModuleId = `\0${consoleDataModuleId}`;
const consoleDataUpdateEvent = 'leibniz-console-data:update';
const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const defaultResultRoot = 'results';
export const consoleDataPayloadMaxBuffer = 64 * 1024 * 1024;
const consoleDataRefreshDebounceMs = 250;
const consoleDataCachePath = resolve(
  repositoryRoot,
  'local-cache/console/consoleDataPayload.json',
);
const consoleDataCacheMetadataPath = `${consoleDataCachePath}.metadata.json`;

export default defineConfig({
  base: consoleBasePath(),
  plugins: [leibnizConsoleData(), react()],
  server: {
    watch: {
      ignored: consoleResultWatchIgnoredPaths(),
    },
  },
});

function leibnizConsoleData() {
  return {
    name: 'leibniz-console-data',
    resolveId(id) {
      if (id === consoleDataModuleId) {
        return resolvedConsoleDataModuleId;
      }
      return null;
    },
    load(id) {
      if (id !== resolvedConsoleDataModuleId) {
        return null;
      }
      const payload = readConsoleDataPayload();
      return [
        "import { parseConsoleDataRecord } from '/src/consoleData.ts';",
        `const payload = ${payload};`,
        'const consoleData = parseConsoleDataRecord(payload);',
        'export default consoleData;',
        'export function subscribeConsoleData(callback) {',
        '  if (import.meta.hot === undefined) {',
        '    return () => {};',
        '  }',
        `  const listener = (payload) => callback(parseConsoleDataRecord(payload));`,
        `  import.meta.hot.on('${consoleDataUpdateEvent}', listener);`,
        `  return () => import.meta.hot?.off('${consoleDataUpdateEvent}', listener);`,
        '}',
      ].join('\n');
    },
    configureServer(server) {
      const resultRoots = consoleResultWatchRoots();
      const watchRoots = consoleResultWatchPaths(process.env, repositoryRoot, resultRoots);
      if (watchRoots.length === 0) {
        return;
      }
      let refreshTimer = undefined;
      let refreshInFlight = false;
      let refreshPending = false;
      const runRefresh = () => {
        if (refreshInFlight || !refreshPending) {
          return;
        }
        refreshPending = false;
        refreshInFlight = true;
        refreshConsoleDataPayloadAsync()
          .then((payload) => {
            server.ws.send({
              type: 'custom',
              event: consoleDataUpdateEvent,
              data: JSON.parse(payload),
            });
          })
          .catch((error) => {
            server.config.logger.error(`failed to refresh Leibniz console data: ${error}`);
          })
          .finally(() => {
            refreshInFlight = false;
            if (refreshPending) {
              runRefresh();
            }
          });
      };
      const scheduleRefresh = () => {
        refreshPending = true;
        if (refreshTimer !== undefined) {
          clearTimeout(refreshTimer);
        }
        refreshTimer = setTimeout(() => {
          refreshTimer = undefined;
          runRefresh();
        }, consoleDataRefreshDebounceMs);
      };
      server.watcher.add(watchRoots);
      server.watcher.on('all', (_event, path) => {
        server.watcher.add(existingDirectories(resultRoots));
        if (
          !isInsideAnyRoot(path, resultRoots) ||
          isMaterializedResultViewEvent(path, resultRoots) ||
          isModelArtifactEvent(path, resultRoots)
        ) {
          return;
        }
        const module = server.moduleGraph.getModuleById(resolvedConsoleDataModuleId);
        if (module !== undefined) {
          server.moduleGraph.invalidateModule(module);
        }
        scheduleRefresh();
      });
    },
  };
}

function readConsoleDataPayload() {
  if (!isConsoleDataPayloadCurrent()) {
    return refreshConsoleDataPayload();
  }
  return readFileSync(consoleDataCachePath, 'utf8');
}

export function refreshConsoleDataPayload() {
  const payload = loadConsoleDataPayload();
  mkdirSync(dirname(consoleDataCachePath), { recursive: true });
  writeFileSync(consoleDataCachePath, payload);
  writeConsoleDataPayloadMetadata();
  return payload;
}

export function refreshConsoleDataPayloadAsync() {
  return loadConsoleDataPayloadAsync().then((payload) => {
    mkdirSync(dirname(consoleDataCachePath), { recursive: true });
    writeFileSync(consoleDataCachePath, payload);
    writeConsoleDataPayloadMetadata();
    return payload;
  });
}

export function loadConsoleDataPayload() {
  const roots = consoleResultRoots();
  const resultRootArgs = resultRootArguments(roots);
  return execFileSync(
    'python',
    [
      '-m',
      'leibniz.console.data',
      ...resultRootArgs,
      'tests/fixtures',
      'src/leibniz/benchmarks',
    ],
    {
      cwd: repositoryRoot,
      encoding: 'utf8',
      maxBuffer: consoleDataPayloadMaxBuffer,
    },
  );
}

export function loadConsoleDataPayloadAsync() {
  const roots = consoleResultRoots();
  const resultRootArgs = resultRootArguments(roots);
  return new Promise((resolvePayload, reject) => {
    execFile(
      'python',
      [
        '-m',
        'leibniz.console.data',
        ...resultRootArgs,
        'tests/fixtures',
        'src/leibniz/benchmarks',
      ],
      {
        cwd: repositoryRoot,
        encoding: 'utf8',
        maxBuffer: consoleDataPayloadMaxBuffer,
      },
      (error, stdout) => {
        if (error !== null) {
          reject(error);
          return;
        }
        resolvePayload(stdout);
      },
    );
  });
}

export function consoleDataPayloadPath() {
  return consoleDataCachePath;
}

export function consoleDataPayloadMetadataPath() {
  return consoleDataCacheMetadataPath;
}

export function isConsoleDataPayloadCurrent() {
  if (!existsSync(consoleDataCachePath) || !existsSync(consoleDataCacheMetadataPath)) {
    return false;
  }
  try {
    const metadata = JSON.parse(readFileSync(consoleDataCacheMetadataPath, 'utf8'));
    return metadata.fingerprint === consoleDataInputFingerprint();
  } catch (_error) {
    return false;
  }
}

export function consoleDataInputFingerprint() {
  const hash = createHash('sha256');
  for (const path of consoleDataInputFiles()) {
    hash.update(relative(repositoryRoot, path));
    hash.update('\0');
    hash.update(readFileSync(path));
    hash.update('\0');
  }
  return hash.digest('hex');
}

function writeConsoleDataPayloadMetadata() {
  writeFileSync(
    consoleDataCacheMetadataPath,
    `${JSON.stringify({ fingerprint: consoleDataInputFingerprint() }, null, 2)}\n`,
  );
}

function consoleDataInputFiles() {
  return Array.from(new Set([
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/package.json'),
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/vite.config.mjs'),
    resolve(repositoryRoot, 'src/leibniz/console/_web_src/scripts/prepareConsoleData.mjs'),
    ...sourceFiles(resolve(repositoryRoot, 'src/leibniz')),
    ...sourceFiles(resolve(repositoryRoot, 'tests/fixtures')),
    ...consoleResultRoots().flatMap((root) => sourceFiles(root)),
  ])).sort();
}

function sourceFiles(root) {
  if (!existsSync(root)) {
    return [];
  }
  const stat = statSync(root);
  if (stat.isFile()) {
    return isFingerprintSource(root) ? [root] : [];
  }
  return readdirSync(root)
    .flatMap((entry) => {
      const path = resolve(root, entry);
      if (ignoredSourcePath(path)) {
        return [];
      }
      return sourceFiles(path);
    })
    .sort();
}

function ignoredSourcePath(path) {
  const normalized = relative(repositoryRoot, path).replaceAll('\\', '/');
  return (
    normalized.includes('/__pycache__/') ||
    normalized.includes('/node_modules/') ||
    normalized.startsWith('local-cache/')
  );
}

function isFingerprintSource(path) {
  return (
    path.endsWith('.py') ||
    path.endsWith('.json') ||
    path.endsWith('.mjs') ||
    path.endsWith('.ts')
  );
}

export function consoleResultWatchRoots(env = process.env, root = repositoryRoot) {
  const raw = env.LEIBNIZ_CONSOLE_RESULT_ROOTS ?? '';
  if (raw.trim() === '') {
    const runsRoot = resolve(root, 'results');
    return [runsRoot];
  }
  return consoleResultRoots(env, root);
}

export function consoleResultRoots(env = process.env, root = repositoryRoot) {
  const raw = env.LEIBNIZ_CONSOLE_RESULT_ROOTS ?? '';
  if (raw.trim() === '') {
    const path = resolve(root, defaultResultRoot);
    return existingDirectories([path]);
  }
  return raw
    .split(delimiter)
    .filter((root) => root.trim() !== '')
    .map((path) => resolve(root, path));
}

export function resultRootArguments(roots = consoleResultRoots()) {
  return roots.flatMap((root) => ['--result-root', root]);
}

export function consoleResultWatchPaths(
  env = process.env,
  root = repositoryRoot,
  roots = consoleResultWatchRoots(env, root),
) {
  return uniquePaths(
    existingDirectories(
      roots.flatMap((path) => [
        dirname(path),
        path,
        materializedResultViewPath(path),
      ]),
    ),
  );
}

export function consoleResultWatchIgnoredPaths(
  env = process.env,
  root = repositoryRoot,
  roots = consoleResultWatchRoots(env, root),
) {
  return roots.map((path) => `${path.replaceAll('\\', '/')}/models/**`);
}

export function consoleBasePath(env = process.env) {
  return env.LEIBNIZ_CONSOLE_BASE_PATH ?? '/';
}

function existingDirectories(paths) {
  return paths.filter((path) => existsSync(path) && statSync(path).isDirectory());
}

function uniquePaths(paths) {
  return [...new Set(paths)];
}

function isInsideAnyRoot(path, roots) {
  const resolvedPath = resolve(path);
  return roots.some((root) => {
    const relativePath = relative(root, resolvedPath);
    return relativePath === '' || (!relativePath.startsWith('..') && !relativePath.startsWith('/'));
  });
}

export function isMaterializedResultViewEvent(path, roots) {
  const resolvedPath = resolve(path);
  return roots.some((root) => {
    if (root.name === 'views') {
      return false;
    }
    const relativePath = relative(root, resolvedPath);
    const parts = relativePath.split(/[\\/]+/);
    return parts[0] === 'views' && parts.length > 1;
  });
}

export function isModelArtifactEvent(path, roots) {
  const resolvedPath = resolve(path);
  return roots.some((root) => {
    const relativePath = relative(root, resolvedPath);
    const parts = relativePath.split(/[\\/]+/);
    return parts[0] === 'models' && parts.length > 1;
  });
}

function materializedResultViewPath(root) {
  return root.name === 'views' ? root : resolve(root, 'views');
}
