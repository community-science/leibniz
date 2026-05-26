import { CheckCircle2, FileJson, Network } from 'lucide-react';
import { useMemo, useState } from 'react';

import type { ConsoleArtifactIndexEntryRecord, ConsoleArtifactIndexRecord } from './artifactIndex';

type ArtifactBrowserProps = {
  index: ConsoleArtifactIndexRecord;
};

const allKinds = 'all';

export function ArtifactBrowser({ index }: ArtifactBrowserProps) {
  const [selectedKind, setSelectedKind] = useState<string>(allKinds);
  const kinds = useMemo(
    () => [
      allKinds,
      ...Array.from(new Set(index.artifacts.map((artifact) => artifact.kind))).sort(),
    ],
    [index.artifacts],
  );
  const filteredArtifacts = useMemo(
    () =>
      selectedKind === allKinds
        ? index.artifacts
        : index.artifacts.filter((artifact) => artifact.kind === selectedKind),
    [index.artifacts, selectedKind],
  );
  const dependencyCount = index.artifacts.reduce(
    (count, artifact) => count + artifact.dependencies.length,
    0,
  );

  return (
    <section className="artifact-browser" aria-label="Artifact browser">
      <header className="artifact-browser-header">
        <div>
          <p className="section-label">Artifact Index</p>
          <h2>Protocol Artifacts</h2>
        </div>
        <dl className="artifact-metrics" aria-label="Artifact index summary">
          <div>
            <dt>Artifacts</dt>
            <dd>{index.artifacts.length}</dd>
          </div>
          <div>
            <dt>Dependencies</dt>
            <dd>{dependencyCount}</dd>
          </div>
          <div>
            <dt>Format</dt>
            <dd>v{index.format_version}</dd>
          </div>
        </dl>
      </header>

      <div className="artifact-browser-controls" role="group" aria-label="Artifact kind filter">
        {kinds.map((kind) => (
          <button
            className={`artifact-filter ${selectedKind === kind ? 'active' : ''}`}
            key={kind}
            onClick={() => setSelectedKind(kind)}
            type="button"
          >
            {kind === allKinds ? 'All' : kind}
          </button>
        ))}
      </div>

      <div className="artifact-list" aria-label="Indexed artifacts">
        {filteredArtifacts.map((artifact) => (
          <ArtifactRow artifact={artifact} key={artifact.source_path} />
        ))}
      </div>
    </section>
  );
}

function ArtifactRow({ artifact }: { artifact: ConsoleArtifactIndexEntryRecord }) {
  return (
    <article className="artifact-row">
      <div className="artifact-row-icon" aria-hidden="true">
        <FileJson size={18} />
      </div>
      <div className="artifact-row-main">
        <div className="artifact-row-title">
          <h3>{artifact.protocol_id ?? artifact.kind}</h3>
          <span className="artifact-kind">{artifact.kind}</span>
        </div>
        <p className="artifact-path">{artifact.source_path}</p>
        <dl className="artifact-identity">
          <div>
            <dt>Digest</dt>
            <dd>{shortDigest(artifact.digest)}</dd>
          </div>
          <div>
            <dt>Reference</dt>
            <dd>{referenceLabel(artifact.reference)}</dd>
          </div>
        </dl>
        {artifact.dependencies.length > 0 ? (
          <div className="artifact-dependencies">
            <Network size={14} />
            <span>{artifact.dependencies.map(referenceLabel).join(', ')}</span>
          </div>
        ) : null}
      </div>
      <div className="artifact-status" title={artifact.validation_command}>
        <CheckCircle2 size={16} />
        {artifact.validation_status}
      </div>
    </article>
  );
}

function referenceLabel(reference: ConsoleArtifactIndexEntryRecord['reference']) {
  return (
    reference.protocol_id ??
    reference.record_digest ??
    reference.content_digest ??
    reference.kind
  );
}

function shortDigest(digest: string) {
  const prefix = 'sha256:';
  if (digest.startsWith(prefix) && digest.length > prefix.length + 14) {
    return `${prefix}${digest.slice(prefix.length, prefix.length + 14)}`;
  }
  return digest;
}
