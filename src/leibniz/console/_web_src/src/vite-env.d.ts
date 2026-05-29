/// <reference types="vite/client" />

declare module 'virtual:leibniz-console-data' {
  import type { ConsoleDataRecord } from './consoleData.ts';

  const consoleData: ConsoleDataRecord;
  export function subscribeConsoleData(
    callback: (consoleData: ConsoleDataRecord) => void,
  ): () => void;
  export default consoleData;
}
