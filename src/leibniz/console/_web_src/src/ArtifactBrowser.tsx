import { CheckCircle2, FileJson, Link2, Network } from 'lucide-react';
import type { KeyboardEvent } from 'react';
import { useMemo, useState } from 'react';

import { DetailItem, DetailSection } from './ArtifactDetailPrimitives';
import { ArtifactTypedDetail } from './ArtifactTypedDetail';
import type { ConsoleArtifactIndexEntryRecord, ConsoleArtifactIndexRecord } from './artifactIndex';
import {
  allArtifactKinds,
  artifactKey,
  artifactKinds,
  filterArtifacts,
  referenceLabel,
  resolveSelectedArtifact,
  shortDigest,
} from './artifactBrowserModel';
import type { ConsoleArtifactDetailMap, ConsoleArtifactDetailRecord } from './artifactDetails';
import { detailForArtifact } from './artifactDetails';

export function ArtifactBrowser({ details, index }: ArtifactBrowserProps) {
  const [selectedKind, setSelectedKind] = useState<string>(allArtifactKinds);
  const [selectedArtifactKey, setSelectedArtifactKey] = useState<string | null>(null);
  const kinds = useMemo(() => artifactKinds(index.artifacts), [index.artifacts]);
  const filteredArtifacts = useMemo(
    () => filterArtifacts(index.artifacts, selectedKind),
    [index.artifacts, selectedKind],
  );
  const selectedArtifact = resolveSelectedArtifact(filteredArtifacts, selectedArtifactKey);
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
            {kind === allArtifactKinds ? 'All' : kind}
          </button>
        ))}
      </div>

      <div className="artifact-workspace">
        <div className="artifact-list" aria-label="Indexed artifacts" role="listbox">
          {filteredArtifacts.map((artifact) => {
            const isSelected =
              selectedArtifact !== undefined &&
              artifactKey(artifact) === artifactKey(selectedArtifact);
            return (
              <ArtifactRow
                artifact={artifact}
                isSelected={isSelected}
                key={artifact.source_path}
                onSelect={() => setSelectedArtifactKey(artifactKey(artifact))}
              />
            );
          })}
        </div>
        <ArtifactDetailPanel
          artifact={selectedArtifact}
          detail={
            selectedArtifact === undefined ? undefined : detailForArtifact(details, selectedArtifact)
          }
        />
      </div>
    </section>
  );
}

type ArtifactBrowserProps = {
  details?: ConsoleArtifactDetailMap;
  index: ConsoleArtifactIndexRecord;
};

type ArtifactRowProps = {
  artifact: ConsoleArtifactIndexEntryRecord;
  isSelected: boolean;
  onSelect: () => void;
};

function ArtifactRow({ artifact, isSelected, onSelect }: ArtifactRowProps) {
  const handleKeyDown = (event: KeyboardEvent<HTMLElement>) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      onSelect();
    }
  };

  return (
    <article
      aria-selected={isSelected}
      className={`artifact-row ${isSelected ? 'selected' : ''}`}
      onClick={onSelect}
      onKeyDown={handleKeyDown}
      role="option"
      tabIndex={0}
    >
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

function ArtifactDetailPanel({
  artifact,
  detail,
}: {
  artifact: ConsoleArtifactIndexEntryRecord | undefined;
  detail: ConsoleArtifactDetailRecord | undefined;
}) {
  if (artifact === undefined) {
    return (
      <aside className="artifact-detail empty" aria-label="Artifact detail">
        <p className="section-label">Selection</p>
        <h3>No artifact selected</h3>
        <p className="artifact-detail-empty">
          Choose a filter with matching artifacts to inspect an index entry.
        </p>
      </aside>
    );
  }

  return (
    <aside className="artifact-detail" aria-label="Artifact detail">
      <header className="artifact-detail-header">
        <p className="section-label">Selected Artifact</p>
        <h3>{artifact.protocol_id ?? artifact.kind}</h3>
        <p>{artifact.source_path}</p>
      </header>

      <DetailSection title="Identity">
        <DetailItem label="Kind" value={artifact.kind} />
        <DetailItem label="Protocol ID" value={artifact.protocol_id ?? 'Not declared'} />
        <DetailItem label="Digest" value={shortDigest(artifact.digest)} />
      </DetailSection>

      <DetailSection title="Reference">
        <ReferenceDetail reference={artifact.reference} />
      </DetailSection>

      <ArtifactTypedDetail artifact={artifact} detail={detail} />

      <DetailSection title="Dependencies">
        {artifact.dependencies.length > 0 ? (
          <ul className="artifact-reference-list">
            {artifact.dependencies.map((dependency) => (
              <li key={`${dependency.kind}:${referenceLabel(dependency)}`}>
                <Link2 size={14} aria-hidden="true" />
                <span>{referenceLabel(dependency)}</span>
              </li>
            ))}
          </ul>
        ) : (
          <p className="artifact-detail-note">No dependencies declared.</p>
        )}
      </DetailSection>

      <DetailSection title="Validation">
        <DetailItem label="Status" value={artifact.validation_status} />
        <DetailItem label="Command" value={artifact.validation_command} />
      </DetailSection>
    </aside>
  );
}

function ReferenceDetail({ reference }: { reference: ConsoleArtifactIndexEntryRecord['reference'] }) {
  const fields = [
    ['Kind', reference.kind],
    ['Protocol ID', reference.protocol_id],
    ['Record Digest', reference.record_digest],
    ['Content Digest', reference.content_digest],
    ['External URI', reference.external_uri],
  ].filter((field): field is [string, string] => field[1] !== undefined);

  return (
    <div className="artifact-reference-fields">
      {fields.map(([label, value]) => (
        <DetailItem
          key={label}
          label={label}
          value={label.includes('Digest') ? shortDigest(value) : value}
        />
      ))}
    </div>
  );
}
