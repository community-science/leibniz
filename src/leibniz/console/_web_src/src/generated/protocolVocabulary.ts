export const consoleProtocolFormats = {
  'artifactIndex': 'leibniz.console.artifact-index',
  'consoleData': 'leibniz.console-data',
  'resultViews': {
    'benchmarkResults': 'leibniz.console.benchmark-results',
    'importedResults': 'leibniz.console.imported-results',
    'workQueue': 'leibniz.console.work-queue'
  },
  'workQueueItem': 'leibniz.work-queue-item'
} as const;

export const consoleProtocolFormatVersions = {
  'artifactIndex': 1,
  'consoleData': 1,
  'resultView': 1,
  'workQueueItem': 1
} as const;
