import react from '@vitejs/plugin-react';
import { defineConfig } from 'vite';
import { execFileSync } from 'node:child_process';
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
      const payload = execFileSync('python', ['-m', 'leibniz.console.data', 'tests/fixtures'], {
        cwd: repositoryRoot,
        encoding: 'utf8',
      });
      return [
        "import { parseConsoleDataRecord } from '/src/consoleData.ts';",
        `const payload = ${payload};`,
        'export default parseConsoleDataRecord(payload);',
      ].join('\n');
    },
  };
}
