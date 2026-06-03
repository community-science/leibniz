import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { delimiter, relative } from 'node:path';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const consoleDataModuleId = 'virtual:leibniz-console-data';
const resolvedConsoleDataModuleId = `\0${consoleDataModuleId}`;
const consoleDataUpdateEvent = 'leibniz-console-data:update';
const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const defaultResultRoot = 'results';
const consoleDataCachePath = resolve(
  repositoryRoot,
  'src/leibniz/console/_web_src/src/generated/consoleDataPayload.json',
);

export default defineConfig({
  base: consoleBasePath(),
  plugins: [leibnizConsoleData(), react()],
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
      server.watcher.add(watchRoots);
      server.watcher.on('all', (_event, path) => {
        server.watcher.add(existingDirectories(resultRoots));
        if (
          !isInsideAnyRoot(path, resultRoots) ||
          isMaterializedResultViewEvent(path, resultRoots)
        ) {
          return;
        }
        const module = server.moduleGraph.getModuleById(resolvedConsoleDataModuleId);
        if (module !== undefined) {
          server.moduleGraph.invalidateModule(module);
        }
        try {
          const payload = refreshConsoleDataPayload();
          server.ws.send({
            type: 'custom',
            event: consoleDataUpdateEvent,
            data: JSON.parse(payload),
          });
        } catch (error) {
          server.config.logger.error(`failed to refresh Leibniz console data: ${error}`);
        }
      });
    },
  };
}

function readConsoleDataPayload() {
  if (!existsSync(consoleDataCachePath)) {
    return refreshConsoleDataPayload();
  }
  return readFileSync(consoleDataCachePath, 'utf8');
}

export function refreshConsoleDataPayload() {
  const payload = loadConsoleDataPayload();
  mkdirSync(dirname(consoleDataCachePath), { recursive: true });
  writeFileSync(consoleDataCachePath, payload);
  return payload;
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
    },
  );
}

export function consoleDataPayloadPath() {
  return consoleDataCachePath;
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
  return uniquePaths(existingDirectories(roots.flatMap((path) => [dirname(path), path])));
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
    const relativePath = relative(root, resolvedPath);
    const parts = relativePath.split(/[\\/]+/);
    return parts[0] === 'views' && parts.length > 1;
  });
}
