/// <reference types="vite/client" />

declare module 'virtual:leibniz-console-data' {
  import type { ConsoleDataRecord } from './consoleData.ts';

  const consoleData: ConsoleDataRecord;
  export default consoleData;
}
