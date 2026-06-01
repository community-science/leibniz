"""Authority scan for source-controlled protocol declaration records."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

from leibniz.artifacts import ArtifactReference
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.content import ContentDigest
from leibniz.documents import document_filename_suffix, load_object_document
from leibniz.latent_factors import LatentFactorDeclarationDocument
from leibniz.materialization import MaterializationDeclarationDocument
from leibniz.observation_formation import ObservationFormationDeclarationDocument
from leibniz.observation_showcases import ObservationShowcaseDocument

__all__ = [
    "ProtocolAuthorityArtifact",
    "ProtocolAuthorityDependency",
    "ProtocolAuthorityError",
    "ProtocolAuthorityIndex",
    "ProtocolArtifactRoute",
    "discover_protocol_authority_index",
    "discover_protocol_artifacts",
    "route_protocol_record",
]

_ValidationStatus: TypeAlias = Literal["invalid", "valid"]
_DependencyStatus: TypeAlias = Literal["dangling", "resolved"]
_DocumentLoader = Callable[[bytes], object]
_DocumentIdentity = Callable[[object], str]

_state_artifact_kinds = frozenset(
    (
        "benchmark-result-view",
        "measurement",
        "measurement-dataset",
        "model-inspection",
        "publication-bundle",
        "submission-publication",
        "training-summary",
    )
)


class ProtocolAuthorityError(ValueError):
    """Raised when source-controlled protocol authority records are invalid."""


@dataclass(frozen=True, slots=True)
class ProtocolArtifactRoute:
    """Interpreter route for one source-controlled protocol declaration kind."""

    kind: str
    semantic_kind: str
    interpreter: str
    required_fields: frozenset[str]
    load_document: _DocumentLoader
    document_identity: _DocumentIdentity

    def matches(self, record: Mapping[str, object]) -> bool:
        return self.required_fields <= record.keys()

    def identity_for(self, data: bytes) -> str:
        return self.document_identity(self.load_document(data))


@dataclass(frozen=True, slots=True)
class ProtocolAuthorityArtifact:
    """One source-controlled protocol artifact discovered by the authority scan."""

    path: str
    kind: str | None
    semantic_kind: str | None
    protocol_id: str | None
    interpreter: str | None
    source_sha256: str
    canonical_sha256: str | None
    validation_status: _ValidationStatus
    validation_error: str | None = None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "path": self.path,
            "source_sha256": self.source_sha256,
            "validation_status": self.validation_status,
        }
        if self.kind is not None:
            record["kind"] = self.kind
        if self.semantic_kind is not None:
            record["semantic_kind"] = self.semantic_kind
        if self.protocol_id is not None:
            record["protocol_id"] = self.protocol_id
        if self.interpreter is not None:
            record["interpreter"] = self.interpreter
        if self.canonical_sha256 is not None:
            record["canonical_sha256"] = self.canonical_sha256
        if self.validation_error is not None:
            record["validation_error"] = self.validation_error
        return record


@dataclass(frozen=True, slots=True)
class ProtocolAuthorityDependency:
    """A derived reference edge between source-controlled protocol artifacts."""

    source_path: str
    source_kind: str
    source_protocol_id: str
    source_field: str
    target_kind: str
    target_protocol_id: str
    status: _DependencyStatus
    target_path: str | None = None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "source_path": self.source_path,
            "source_kind": self.source_kind,
            "source_protocol_id": self.source_protocol_id,
            "source_field": self.source_field,
            "target_kind": self.target_kind,
            "target_protocol_id": self.target_protocol_id,
            "status": self.status,
        }
        if self.target_path is not None:
            record["target_path"] = self.target_path
        return record


@dataclass(frozen=True, slots=True)
class ProtocolAuthorityIndex:
    """Authority report derived from source-controlled protocol declarations."""

    artifacts: tuple[ProtocolAuthorityArtifact, ...]
    dependencies: tuple[ProtocolAuthorityDependency, ...]

    @property
    def valid_artifacts(self) -> tuple[ProtocolAuthorityArtifact, ...]:
        return tuple(
            artifact
            for artifact in self.artifacts
            if artifact.validation_status == "valid"
        )

    @property
    def invalid_artifacts(self) -> tuple[ProtocolAuthorityArtifact, ...]:
        return tuple(
            artifact
            for artifact in self.artifacts
            if artifact.validation_status == "invalid"
        )

    @property
    def dangling_dependencies(self) -> tuple[ProtocolAuthorityDependency, ...]:
        return tuple(
            dependency
            for dependency in self.dependencies
            if dependency.status == "dangling"
        )

    def require_valid(self) -> None:
        if self.invalid_artifacts:
            artifact = self.invalid_artifacts[0]
            raise ProtocolAuthorityError(
                f"{artifact.path}: {artifact.validation_error or 'invalid artifact'}"
            )
        if self.dangling_dependencies:
            dependency = self.dangling_dependencies[0]
            raise ProtocolAuthorityError(
                f"{dependency.source_path}: dangling reference to "
                f"{dependency.target_kind} {dependency.target_protocol_id}"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "artifact_count": len(self.artifacts),
            "valid_artifact_count": len(self.valid_artifacts),
            "invalid_artifact_count": len(self.invalid_artifacts),
            "dependency_count": len(self.dependencies),
            "dangling_dependency_count": len(self.dangling_dependencies),
            "artifacts": [artifact.to_record() for artifact in self.artifacts],
            "dependencies": [
                dependency.to_record() for dependency in self.dependencies
            ],
        }


def route_protocol_record(record: Mapping[str, object]) -> ProtocolArtifactRoute:
    """Return the interpreter route for a source-controlled declaration record."""

    declared_kind = record.get("kind")
    if isinstance(declared_kind, str) and declared_kind in _state_artifact_kinds:
        raise ProtocolAuthorityError(
            f"{declared_kind} artifacts must not be source-controlled declarations"
        )
    matching_routes = tuple(route for route in _routes() if route.matches(record))
    if not matching_routes:
        raise ProtocolAuthorityError("record does not match a known declaration route")
    if len(matching_routes) > 1:
        route_names = ", ".join(route.kind for route in matching_routes)
        raise ProtocolAuthorityError(f"record matches multiple declaration routes: {route_names}")
    route = next(iter(matching_routes))
    return route


def discover_protocol_artifacts(
    *,
    root: Path | str | None = None,
) -> tuple[ProtocolAuthorityArtifact, ...]:
    """Discover and validate source-controlled protocol declaration artifacts."""

    scan_root = _protocol_scan_root(root)
    artifacts = tuple(
        _artifact_for_path(path=path, base=scan_root)
        for path in sorted(scan_root.rglob(f"*{document_filename_suffix()}"))
        if path.is_file()
    )
    return tuple(sorted(artifacts, key=lambda artifact: artifact.path))


def discover_protocol_authority_index(
    *,
    root: Path | str | None = None,
    strict: bool = False,
) -> ProtocolAuthorityIndex:
    """Discover declarations and derive their explicit protocol dependency edges."""

    artifacts = discover_protocol_artifacts(root=root)
    target_paths = {
        (artifact.kind, artifact.protocol_id): artifact.path
        for artifact in artifacts
        if artifact.validation_status == "valid"
        and artifact.kind is not None
        and artifact.protocol_id is not None
    }
    dependencies = tuple(
        sorted(
            _dependencies_for_artifacts(
                artifacts=artifacts,
                root=_protocol_scan_root(root),
                target_paths=target_paths,
            ),
            key=_dependency_sort_key,
        )
    )
    index = ProtocolAuthorityIndex(artifacts=artifacts, dependencies=dependencies)
    if strict:
        index.require_valid()
    return index


def _artifact_for_path(*, path: Path, base: Path) -> ProtocolAuthorityArtifact:
    relative_path = path.relative_to(base).as_posix()
    data = path.read_bytes()
    source_sha256 = hashlib.sha256(data).hexdigest()
    try:
        record = load_object_document(data, description=relative_path)
        route = route_protocol_record(record)
        protocol_id = route.identity_for(data)
    except Exception as error:
        return ProtocolAuthorityArtifact(
            path=relative_path,
            kind=None,
            semantic_kind=None,
            protocol_id=None,
            interpreter=None,
            source_sha256=source_sha256,
            canonical_sha256=None,
            validation_status="invalid",
            validation_error=str(error),
        )
    return ProtocolAuthorityArtifact(
        path=relative_path,
        kind=route.kind,
        semantic_kind=route.semantic_kind,
        protocol_id=protocol_id,
        interpreter=route.interpreter,
        source_sha256=source_sha256,
        canonical_sha256=ContentDigest.from_value(record).hex,
        validation_status="valid",
    )


def _dependencies_for_artifacts(
    *,
    artifacts: Iterable[ProtocolAuthorityArtifact],
    root: Path,
    target_paths: Mapping[tuple[str, str], str],
) -> Iterable[ProtocolAuthorityDependency]:
    for artifact in artifacts:
        if artifact.validation_status != "valid":
            continue
        if artifact.kind is None or artifact.protocol_id is None:
            continue
        record = load_object_document(
            (root / artifact.path).read_bytes(),
            description=artifact.path,
        )
        for source_field, reference in _artifact_references(record):
            target_protocol_id = str(reference.protocol_id)
            target_path = target_paths.get((reference.kind, target_protocol_id))
            yield ProtocolAuthorityDependency(
                source_path=artifact.path,
                source_kind=artifact.kind,
                source_protocol_id=artifact.protocol_id,
                source_field=source_field,
                target_kind=reference.kind,
                target_protocol_id=target_protocol_id,
                target_path=target_path,
                status="resolved" if target_path is not None else "dangling",
            )


def _artifact_references(
    value: object,
    *,
    path: tuple[str, ...] = (),
) -> Iterable[tuple[str, ArtifactReference]]:
    if isinstance(value, Mapping):
        record = cast(Mapping[object, object], value)
        if "kind" in record and "protocol_id" in record:
            reference = ArtifactReference.from_record(_object_mapping(record))
            if reference.protocol_id is not None:
                yield ".".join(path), reference
        for key, item in record.items():
            if isinstance(key, str):
                yield from _artifact_references(item, path=(*path, key))
    elif isinstance(value, list):
        items = cast(list[object], value)
        for index, item in enumerate(items):
            yield from _artifact_references(item, path=(*path, str(index)))


def _protocol_scan_root(root: Path | str | None) -> Path:
    repository_root = Path.cwd() if root is None else Path(root)
    return repository_root / "src" / "leibniz" / "benchmarks"


def _routes() -> tuple[ProtocolArtifactRoute, ...]:
    return (
        ProtocolArtifactRoute(
            kind="benchmark-manifest",
            semantic_kind="benchmark family",
            interpreter="leibniz.benchmarks.BenchmarkManifestDocument",
            required_fields=frozenset(("id", "name")),
            load_document=BenchmarkManifestDocument.from_bytes,
            document_identity=lambda document: str(
                cast(BenchmarkManifestDocument, document).manifest.id
            ),
        ),
        ProtocolArtifactRoute(
            kind="latent-factor-declaration",
            semantic_kind="latent factor declaration",
            interpreter="leibniz.latent_factors.LatentFactorDeclarationDocument",
            required_fields=frozenset(("id", "construction_factors", "sample_factors")),
            load_document=LatentFactorDeclarationDocument.from_bytes,
            document_identity=lambda document: str(
                cast(LatentFactorDeclarationDocument, document).declaration.id
            ),
        ),
        ProtocolArtifactRoute(
            kind="materialization-declaration",
            semantic_kind="materialization declaration",
            interpreter="leibniz.materialization.MaterializationDeclarationDocument",
            required_fields=frozenset(("id", "benchmark_id", "requirements")),
            load_document=MaterializationDeclarationDocument.from_bytes,
            document_identity=lambda document: str(
                cast(MaterializationDeclarationDocument, document).declaration.id
            ),
        ),
        ProtocolArtifactRoute(
            kind="observation-formation-declaration",
            semantic_kind="observation formation declaration",
            interpreter=(
                "leibniz.observation_formation."
                "ObservationFormationDeclarationDocument"
            ),
            required_fields=frozenset(("id", "interpreter", "sequence_layout", "components")),
            load_document=ObservationFormationDeclarationDocument.from_bytes,
            document_identity=lambda document: str(
                cast(ObservationFormationDeclarationDocument, document).declaration.id
            ),
        ),
        ProtocolArtifactRoute(
            kind="observation-showcase",
            semantic_kind="observation showcase",
            interpreter="leibniz.observation_showcases.ObservationShowcaseDocument",
            required_fields=frozenset(
                (
                    "id",
                    "formation_declaration",
                    "materialization_declaration",
                    "samples",
                )
            ),
            load_document=ObservationShowcaseDocument.from_bytes,
            document_identity=lambda document: str(
                cast(ObservationShowcaseDocument, document).manifest.id
            ),
        ),
    )


def _dependency_sort_key(
    dependency: ProtocolAuthorityDependency,
) -> tuple[str, str, str, str, str]:
    return (
        dependency.source_path,
        dependency.source_field,
        dependency.target_kind,
        dependency.target_protocol_id,
        dependency.status,
    )


def _object_mapping(value: Mapping[object, object]) -> Mapping[str, object]:
    return {str(key): item for key, item in value.items()}
