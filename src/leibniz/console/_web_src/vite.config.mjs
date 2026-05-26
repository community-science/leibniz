import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { execFileSync } from 'node:child_process';
import { existsSync, statSync } from 'node:fs';
import { delimiter, relative } from 'node:path';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const consoleDataModuleId = 'virtual:leibniz-console-data';
const resolvedConsoleDataModuleId = `\0${consoleDataModuleId}`;
const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');
const defaultResultRoot = '.runs/views';

export default defineConfig({
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
      const roots = consoleResultRoots();
      for (const root of roots) {
        this.addWatchFile(root);
      }
      const resultRootArgs = resultRootArguments(roots);
      const payload = execFileSync(
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
      return [
        "import { parseConsoleDataRecord } from '/src/consoleData.ts';",
        `const payload = ${payload};`,
        'export default parseConsoleDataRecord(payload);',
      ].join('\n');
    },
    configureServer(server) {
      const roots = consoleResultRoots();
      if (roots.length === 0) {
        return;
      }
      server.watcher.add(roots);
      server.watcher.on('all', (_event, path) => {
        if (!isInsideAnyRoot(path, roots)) {
          return;
        }
        const module = server.moduleGraph.getModuleById(resolvedConsoleDataModuleId);
        if (module !== undefined) {
          server.moduleGraph.invalidateModule(module);
        }
        server.ws.send({ type: 'full-reload' });
      });
    },
  };
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

function existingDirectories(paths) {
  return paths.filter((path) => existsSync(path) && statSync(path).isDirectory());
}

function isInsideAnyRoot(path, roots) {
  const resolvedPath = resolve(path);
  return roots.some((root) => {
    const relativePath = relative(root, resolvedPath);
    return relativePath === '' || (!relativePath.startsWith('..') && !relativePath.startsWith('/'));
  });
}
