import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { execFileSync } from 'node:child_process';
import { delimiter } from 'node:path';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const consoleDataModuleId = 'virtual:leibniz-console-data';
const resolvedConsoleDataModuleId = `\0${consoleDataModuleId}`;
const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../../../..');

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
      const resultRootArgs = resultRootArguments();
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
  };
}

function resultRootArguments() {
  const raw = process.env.LEIBNIZ_CONSOLE_RESULT_ROOTS ?? '';
  if (raw.trim() === '') {
    return [];
  }
  return raw
    .split(delimiter)
    .filter((root) => root.trim() !== '')
    .flatMap((root) => ['--result-root', root]);
}
