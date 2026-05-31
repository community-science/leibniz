"""Build the browser console's explicit public artifact index."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from leibniz.architectures import ArchitectureManifestDocument
from leibniz.artifacts import ArtifactReference
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.console.protocol import (
    console_protocol_format_versions,
    console_protocol_formats,
)
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.latent_factors import LatentFactorDeclarationDocument
from leibniz.materialization import (
    MaterializationDeclarationDocument,
    MaterializationPlanDocument,
)
from leibniz.measurements import MeasurementDocument
from leibniz.observation_formation import ObservationFormationDeclarationDocument
from leibniz.observation_showcases import ObservationShowcaseDocument

__all__ = [
    "ConsoleArtifactIndex",
    "ConsoleArtifactIndexBuilder",
    "ConsoleArtifactIndexEntry",
    "ConsoleArtifactIndexSource",
    "ConsoleArtifactIndexValidationError",
]

_ValidationStatus = Literal["valid"]
_LoadedArtifact = tuple[
    ProtocolIdentifier | None,
    Mapping[str, object],
    ContentDigest,
    tuple[ArtifactReference, ...],
]
_ArtifactLoader = Callable[[bytes], _LoadedArtifact]

_protocol_formats = console_protocol_formats()
_protocol_format_versions = console_protocol_format_versions()
_format = _protocol_formats.artifact_index
_format_version = _protocol_format_versions.artifact_index
_validation_command = "python -m pytest tests/test_console_artifact_index.py"


class ConsoleArtifactIndexValidationError(ValueError):
    """Raised when a console artifact index cannot be generated."""


@dataclass(frozen=True, slots=True)
class ConsoleArtifactIndexSource:
    """One explicit public document source for the console artifact index."""

    kind: str
    source_path: PurePosixPath

    def __post_init__(self) -> None:
        if self.source_path.is_absolute():
            raise ConsoleArtifactIndexValidationError("source_path must be repository-relative")
        if "results" in self.source_path.parts:
            raise ConsoleArtifactIndexValidationError("source_path must not reference local state")
        if self.kind not in _artifact_loaders:
            raise ConsoleArtifactIndexValidationError(f"unsupported document kind: {self.kind}")


@dataclass(frozen=True, slots=True)
class ConsoleArtifactIndexEntry:
    """One validated public artifact document available to the console."""

    kind: str
    source_path: PurePosixPath
    digest: ContentDigest
    reference: ArtifactReference
    dependencies: tuple[ArtifactReference, ...]
    validation_status: _ValidationStatus = "valid"
    validation_command: str = _validation_command

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": self.kind,
            "source_path": self.source_path.as_posix(),
            "digest": str(self.digest),
            "reference": self.reference.to_record(),
            "dependencies": [dependency.to_record() for dependency in self.dependencies],
            "validation_status": self.validation_status,
            "validation_command": self.validation_command,
        }
        if self.reference.protocol_id is not None:
            record["protocol_id"] = str(self.reference.protocol_id)
        return record


@dataclass(frozen=True, slots=True)
class ConsoleArtifactIndex:
    """A deterministic browser-console index over explicit public artifacts."""

    entries: tuple[ConsoleArtifactIndexEntry, ...]

    def __post_init__(self) -> None:
        if not self.entries:
            raise ConsoleArtifactIndexValidationError("entries must contain at least one artifact")
        duplicate_path = _first_duplicate(tuple(entry.source_path for entry in self.entries))
        if duplicate_path is not None:
            raise ConsoleArtifactIndexValidationError(f"duplicate source_path: {duplicate_path}")
        ordered = tuple(sorted(self.entries, key=_entry_sort_key))
        object.__setattr__(self, "entries", ordered)

    def to_record(self) -> dict[str, object]:
        return {
            "format": _format,
            "format_version": _format_version,
            "artifacts": [entry.to_record() for entry in self.entries],
        }

    def to_bytes(self) -> bytes:
        return canonical_document_bytes(self.to_record()) + b"\n"


class ConsoleArtifactIndexBuilder:
    """Build a console artifact index from explicit public document sources."""

    def __init__(self, repository_root: Path) -> None:
        self._repository_root = repository_root.resolve()

    def build(
        self,
        sources: Iterable[ConsoleArtifactIndexSource],
    ) -> ConsoleArtifactIndex:
        source_tuple = tuple(sources)
        return ConsoleArtifactIndex(tuple(self._entry_for(source) for source in source_tuple))

    @staticmethod
    def supported_kinds() -> tuple[str, ...]:
        """Return the public document kinds supported by the console artifact index."""

        return tuple(_artifact_loaders)

    @staticmethod
    def load_supported_artifact(kind: str, data: bytes) -> _LoadedArtifact:
        """Load a document for one of the console artifact index's supported kinds."""

        try:
            loader = _artifact_loaders[kind]
        except KeyError as error:
            raise ConsoleArtifactIndexValidationError(
                f"unsupported document kind: {kind}"
            ) from error
        return loader(data)

    def _entry_for(self, source: ConsoleArtifactIndexSource) -> ConsoleArtifactIndexEntry:
        path = self._repository_path(source.source_path)
        if not path.is_file():
            raise ConsoleArtifactIndexValidationError(
                f"source_path does not name a file: {source.source_path}"
            )
        protocol_id, record, digest, dependencies = self.load_supported_artifact(
            source.kind,
            path.read_bytes(),
        )
        reference = ArtifactReference(
            kind=source.kind,
            protocol_id=protocol_id,
            record_digest=ContentDigest.from_value(record),
        )
        return ConsoleArtifactIndexEntry(
            kind=source.kind,
            source_path=source.source_path,
            digest=digest,
            reference=reference,
            dependencies=dependencies,
        )

    def _repository_path(self, source_path: PurePosixPath) -> Path:
        if source_path.is_absolute():
            raise ConsoleArtifactIndexValidationError("source_path must be repository-relative")
        if "results" in source_path.parts:
            raise ConsoleArtifactIndexValidationError("source_path must not reference local state")
        path = (self._repository_root / Path(source_path)).resolve()
        if not path.is_relative_to(self._repository_root):
            raise ConsoleArtifactIndexValidationError(
                "source_path must stay inside repository root"
            )
        return path


def _load_architecture_manifest(data: bytes) -> _LoadedArtifact:
    document = ArchitectureManifestDocument.from_bytes(data)
    record = document.manifest.to_record()
    return document.manifest.id, record, document.digest, ()


def _load_benchmark_manifest(data: bytes) -> _LoadedArtifact:
    document = BenchmarkManifestDocument.from_bytes(data)
    record = document.manifest.to_record()
    dependencies: tuple[ArtifactReference, ...]
    if document.manifest.latent_factor_declaration is None:
        dependencies = ()
    else:
        dependencies = (document.manifest.latent_factor_declaration,)
    return document.manifest.id, record, document.digest, dependencies


def _load_latent_factor_declaration(data: bytes) -> _LoadedArtifact:
    document = LatentFactorDeclarationDocument.from_bytes(data)
    record = document.declaration.to_record()
    return document.declaration.id, record, document.digest, ()


def _load_materialization_declaration(data: bytes) -> _LoadedArtifact:
    document = MaterializationDeclarationDocument.from_bytes(data)
    declaration = document.declaration
    record = declaration.to_record()
    dependencies = [
        ArtifactReference(kind="benchmark-manifest", protocol_id=declaration.benchmark_id)
    ]
    if declaration.latent_factor_declaration is not None:
        dependencies.append(declaration.latent_factor_declaration)
    return declaration.id, record, document.digest, tuple(dependencies)


def _load_materialization_plan(data: bytes) -> _LoadedArtifact:
    document = MaterializationPlanDocument.from_bytes(data)
    plan = document.plan
    record = plan.to_record()
    dependencies = [
        ArtifactReference(kind="benchmark-manifest", protocol_id=plan.benchmark_id),
        plan.materialization_declaration,
    ]
    if plan.latent_factor_declaration is not None:
        dependencies.append(plan.latent_factor_declaration)
    return plan.id, record, document.digest, tuple(dependencies)


def _load_measurement(data: bytes) -> _LoadedArtifact:
    document = MeasurementDocument.from_bytes(data)
    measurement = document.measurement
    record = measurement.to_record()
    benchmark_reference = ArtifactReference(
        kind="benchmark-manifest",
        protocol_id=measurement.benchmark_id,
    )
    dependencies = (benchmark_reference, *measurement.evidence_artifacts)
    return measurement.raw_scoring_evidence.id, record, document.digest, dependencies


def _load_observation_formation_declaration(data: bytes) -> _LoadedArtifact:
    document = ObservationFormationDeclarationDocument.from_bytes(data)
    declaration = document.declaration
    record = declaration.to_record()
    dependencies = (
        ArtifactReference(kind="benchmark-manifest", protocol_id=declaration.benchmark_id),
    )
    return declaration.id, record, document.digest, dependencies


def _load_observation_showcase(data: bytes) -> _LoadedArtifact:
    document = ObservationShowcaseDocument.from_bytes(data)
    manifest = document.manifest
    record = manifest.to_record()
    dependencies = (
        ArtifactReference(kind="benchmark-manifest", protocol_id=manifest.benchmark_id),
        manifest.formation_declaration,
        manifest.materialization_declaration,
    )
    return manifest.id, record, document.digest, dependencies


_artifact_loaders: Mapping[str, _ArtifactLoader] = {
    "architecture-manifest": _load_architecture_manifest,
    "benchmark-manifest": _load_benchmark_manifest,
    "latent-factor-declaration": _load_latent_factor_declaration,
    "materialization-declaration": _load_materialization_declaration,
    "materialization-plan": _load_materialization_plan,
    "measurement": _load_measurement,
    "observation-formation-declaration": _load_observation_formation_declaration,
    "observation-showcase": _load_observation_showcase,
}


def _entry_sort_key(entry: ConsoleArtifactIndexEntry) -> tuple[str, str]:
    return (entry.kind, entry.source_path.as_posix())


def _first_duplicate(values: tuple[PurePosixPath, ...]) -> PurePosixPath | None:
    seen: set[PurePosixPath] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
