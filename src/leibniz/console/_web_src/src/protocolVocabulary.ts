export const consoleProtocolFormats = {
  'artifactIndex': 'leibniz.console.artifact-index',
  'consoleData': 'leibniz.console-data',
  'resultViews': {
    'benchmarkResults': 'leibniz.console.benchmark-results'
  }
} as const;

export const consoleProtocolFormatVersions = {
  'artifactIndex': 1,
  'consoleData': 1,
  'resultView': 1
} as const;
