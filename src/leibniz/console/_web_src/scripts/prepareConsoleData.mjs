import { relative } from 'node:path';

import {
  consoleDataPayloadPath,
  isConsoleDataPayloadCurrent,
  refreshConsoleDataPayload,
} from '../vite.config.mjs';

if (isConsoleDataPayloadCurrent()) {
  console.log(
    `Leibniz console data is current at ${relative(process.cwd(), consoleDataPayloadPath())}`,
  );
} else {
  refreshConsoleDataPayload();
  console.log(
    `Prepared Leibniz console data at ${relative(process.cwd(), consoleDataPayloadPath())}`,
  );
}
