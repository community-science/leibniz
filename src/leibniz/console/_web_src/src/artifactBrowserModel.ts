import type { ConsoleArtifactIndexEntryRecord } from './artifactIndex';

export const allArtifactKinds = 'all';

export function artifactKey(artifact: ConsoleArtifactIndexEntryRecord): string {
  return artifact.source_path;
}

export function artifactKinds(artifacts: ConsoleArtifactIndexEntryRecord[]): string[] {
  return [
    allArtifactKinds,
    ...Array.from(new Set(artifacts.map((artifact) => artifact.kind))).sort(),
  ];
}

export function filterArtifacts(
  artifacts: ConsoleArtifactIndexEntryRecord[],
  selectedKind: string,
): ConsoleArtifactIndexEntryRecord[] {
  if (selectedKind === allArtifactKinds) {
    return artifacts;
  }
  return artifacts.filter((artifact) => artifact.kind === selectedKind);
}

export function resolveSelectedArtifact(
  artifacts: ConsoleArtifactIndexEntryRecord[],
  selectedArtifactKey: string | null,
): ConsoleArtifactIndexEntryRecord | undefined {
  if (artifacts.length === 0) {
    return undefined;
  }

  if (selectedArtifactKey === null) {
    return artifacts[0];
  }

  return (
    artifacts.find((artifact) => artifactKey(artifact) === selectedArtifactKey) ?? artifacts[0]
  );
}

export function referenceLabel(reference: ConsoleArtifactIndexEntryRecord['reference']): string {
  return (
    reference.protocol_id ??
    reference.record_digest ??
    reference.content_digest ??
    reference.external_uri ??
    reference.kind
  );
}

export function shortDigest(digest: string): string {
  const prefix = 'sha256:';
  const visibleDigestLength = 14;
  if (digest.startsWith(prefix) && digest.length > prefix.length + visibleDigestLength) {
    return `${prefix}${digest.slice(prefix.length, prefix.length + visibleDigestLength)}`;
  }
  return digest;
}
