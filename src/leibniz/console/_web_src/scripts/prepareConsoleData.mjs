import { createHash } from 'node:crypto';
import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { consoleDataPayloadPath, refreshConsoleDataPayload } from '../vite.config.mjs';

const scriptRoot = dirname(fileURLToPath(import.meta.url));
const packageRoot = resolve(scriptRoot, '..');
const repositoryRoot = resolve(packageRoot, '../../../..');
const metadataPath = `${consoleDataPayloadPath()}.metadata.json`;
const requiredArtifacts = [
  consoleDataPayloadPath(),
  resolve(packageRoot, 'src/generated/protocolVocabulary.ts'),
  resolve(packageRoot, 'src/generated/resultViewRecords.ts'),
];

const fingerprint = generatorInputFingerprint();
if (isPrepared(fingerprint)) {
  console.log(
    `Leibniz console data is current at ${relative(process.cwd(), consoleDataPayloadPath())}`,
  );
} else {
  refreshConsoleDataPayload();
  writeFileSync(metadataPath, `${JSON.stringify({ fingerprint }, null, 2)}\n`);
  console.log(
    `Prepared Leibniz console data at ${relative(process.cwd(), consoleDataPayloadPath())}`,
  );
}

function isPrepared(fingerprint) {
  if (!requiredArtifacts.every((path) => existsSync(path)) || !existsSync(metadataPath)) {
    return false;
  }
  try {
    const metadata = JSON.parse(readFileSync(metadataPath, 'utf8'));
    return metadata.fingerprint === fingerprint;
  } catch (_error) {
    return false;
  }
}

function generatorInputFingerprint() {
  const hash = createHash('sha256');
  for (const path of generatorInputFiles()) {
    hash.update(relative(repositoryRoot, path));
    hash.update('\0');
    hash.update(readFileSync(path));
    hash.update('\0');
  }
  return hash.digest('hex');
}

function generatorInputFiles() {
  return Array.from(new Set([
    resolve(packageRoot, 'package.json'),
    resolve(packageRoot, 'vite.config.mjs'),
    resolve(packageRoot, 'scripts/prepareConsoleData.mjs'),
    ...sourceFiles(resolve(repositoryRoot, 'src/leibniz')),
    ...sourceFiles(resolve(repositoryRoot, 'tests/fixtures')),
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
    normalized.startsWith('src/leibniz/console/_web_src/src/generated/')
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
