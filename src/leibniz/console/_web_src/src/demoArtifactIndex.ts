import { parseConsoleArtifactIndexRecord } from './artifactIndex.ts';

export const demoArtifactIndex = parseConsoleArtifactIndexRecord({
  format: 'leibniz.console.artifact-index',
  format_version: 1,
  artifacts: [
    {
      kind: 'architecture-manifest',
      source_path: 'tests/fixtures/architecture/digits_pool/manifest.json',
      digest: 'sha256:bb0dde9254dcde8bc71ca5f4746e6f282751e24c51c5f29b4a7d77b2c7622162',
      protocol_id:
        'architecture.sha-d695a59610f59ce2b61a20b7114b42da8692ffd9a55e4093431e3c00a932e693@0.1.0',
      reference: {
        kind: 'architecture-manifest',
        protocol_id:
          'architecture.sha-d695a59610f59ce2b61a20b7114b42da8692ffd9a55e4093431e3c00a932e693@0.1.0',
        record_digest: 'sha256:bb0dde9254dcde8bc71ca5f4746e6f282751e24c51c5f29b4a7d77b2c7622162',
      },
      dependencies: [],
      validation_status: 'valid',
      validation_command: 'python -m pytest tests/test_console_artifact_index.py',
    },
    {
      kind: 'benchmark-manifest',
      source_path: 'tests/fixtures/finite_outcome/manifest.json',
      digest: 'sha256:b9297f71ed09f10d69c6e46e1ef779951d99a7c31c2b41912fefe38d8968a0fe',
      protocol_id: 'core.boolean-benchmark@0.1.0',
      reference: {
        kind: 'benchmark-manifest',
        protocol_id: 'core.boolean-benchmark@0.1.0',
        record_digest: 'sha256:b9297f71ed09f10d69c6e46e1ef779951d99a7c31c2b41912fefe38d8968a0fe',
      },
      dependencies: [],
      validation_status: 'valid',
      validation_command: 'python -m pytest tests/test_console_artifact_index.py',
    },
    {
      kind: 'measurement',
      source_path: 'tests/fixtures/finite_outcome/measurement.json',
      digest: 'sha256:d91a31bac6332478a9f11f73764d036e40b1826c928579dc17e132ff6f9bd133',
      protocol_id: 'core.boolean-evidence@0.1.0',
      reference: {
        kind: 'measurement',
        protocol_id: 'core.boolean-evidence@0.1.0',
        record_digest: 'sha256:d91a31bac6332478a9f11f73764d036e40b1826c928579dc17e132ff6f9bd133',
      },
      dependencies: [
        {
          kind: 'benchmark-manifest',
          protocol_id: 'core.boolean-benchmark@0.1.0',
        },
      ],
      validation_status: 'valid',
      validation_command: 'python -m pytest tests/test_console_artifact_index.py',
    },
  ],
});
