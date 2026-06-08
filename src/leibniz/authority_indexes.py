"""Local authority indexes over explicit protocol artifact references."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from leibniz.artifacts import (
    ArtifactIndex,
    ArtifactReference,
    first_duplicate_reference,
    reference_for_record,
    reference_sort_key,
)
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec

__all__ = [
    "AuthorityDependency",
    "AuthorityIndex",
    "AuthorityIndexDocument",
    "AuthorityIndexValidationEntry",
    "AuthorityIndexValidationError",
]

_ValidationStatus: TypeAlias = Literal["dangling", "invalid", "valid"]

_relation = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_statuses = frozenset(("dangling", "invalid", "valid"))

_dependency_record = RecordSpec(
    fields={
        "source": FieldSpec(kind="record"),
        "target": FieldSpec(kind="record"),
        "relation": FieldSpec(kind="string"),
    }
)
_validation_entry_record = RecordSpec(
    fields={
        "artifact": FieldSpec(kind="record"),
        "status": FieldSpec(kind="string"),
        "message": FieldSpec(kind="string"),
    }
)
_authority_index_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "artifacts": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "dependencies": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "validations": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
    }
)


class AuthorityIndexValidationError(ValueError):
    """Raised when an authority index is invalid."""


_extract = RecordExtractor(error_type=AuthorityIndexValidationError)


@dataclass(frozen=True, slots=True)
class AuthorityDependency:
    """A declared dependency edge between two artifact references."""

    source: ArtifactReference
    target: ArtifactReference
    relation: str

    def __post_init__(self) -> None:
        if _relation.fullmatch(self.relation) is None:
            raise AuthorityIndexValidationError("relation must be a valid relation name")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> AuthorityDependency:
        try:
            validated = _dependency_record.validate(record)
        except ValueError as error:
            raise AuthorityIndexValidationError(str(error)) from error
        return cls(
            source=ArtifactReference.from_record(
                _extract.mapping(validated["source"], "source")
            ),
            target=ArtifactReference.from_record(
                _extract.mapping(validated["target"], "target")
            ),
            relation=_extract.string(validated["relation"], "relation"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "source": self.source.to_record(),
            "target": self.target.to_record(),
            "relation": self.relation,
        }


@dataclass(frozen=True, slots=True)
class AuthorityIndexValidationEntry:
    """Validation status for one referenced artifact."""

    artifact: ArtifactReference
    status: _ValidationStatus
    message: str

    def __post_init__(self) -> None:
        if self.status not in _statuses:
            raise AuthorityIndexValidationError(f"unsupported status: {self.status}")
        if not self.message:
            raise AuthorityIndexValidationError("message must be nonempty")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> AuthorityIndexValidationEntry:
        try:
            validated = _validation_entry_record.validate(record)
        except ValueError as error:
            raise AuthorityIndexValidationError(str(error)) from error
        return cls(
            artifact=ArtifactReference.from_record(
                _extract.mapping(validated["artifact"], "artifact")
            ),
            status=cast(_ValidationStatus, _extract.string(validated["status"], "status")),
            message=_extract.string(validated["message"], "message"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "artifact": self.artifact.to_record(),
            "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class AuthorityIndex:
    """A local audit report over explicit artifact references and dependencies."""

    id: ProtocolIdentifier
    artifacts: tuple[ArtifactReference, ...]
    dependencies: tuple[AuthorityDependency, ...]
    validations: tuple[AuthorityIndexValidationEntry, ...]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise AuthorityIndexValidationError(str(error)) from error
        if not str(self.id.name).startswith("authority-indexes."):
            raise AuthorityIndexValidationError("id must be a valid authority index id")
        if not self.artifacts:
            raise AuthorityIndexValidationError(
                "artifacts must contain at least one artifact reference"
            )
        if not self.validations:
            raise AuthorityIndexValidationError(
                "validations must contain at least one validation entry"
            )
        duplicate_artifact = first_duplicate_reference(self.artifacts)
        if duplicate_artifact is not None:
            raise AuthorityIndexValidationError(f"duplicate artifact: {duplicate_artifact}")
        duplicate_dependency = _first_duplicate_dependency(self.dependencies)
        if duplicate_dependency is not None:
            raise AuthorityIndexValidationError(
                f"duplicate dependency: {duplicate_dependency}"
            )
        duplicate_validation = _first_duplicate_validation(self.validations)
        if duplicate_validation is not None:
            raise AuthorityIndexValidationError(
                f"duplicate validation entry: {duplicate_validation}"
            )
        object.__setattr__(
            self,
            "artifacts",
            tuple(sorted(self.artifacts, key=reference_sort_key)),
        )
        object.__setattr__(
            self,
            "dependencies",
            tuple(sorted(self.dependencies, key=_dependency_sort_key)),
        )
        object.__setattr__(
            self,
            "validations",
            tuple(sorted(self.validations, key=_validation_sort_key)),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> AuthorityIndex:
        try:
            validated = _authority_index_record.validate(record)
            artifacts = tuple(
                ArtifactReference.from_record(_extract.mapping(item, "artifacts"))
                for item in _extract.sequence(validated["artifacts"], "artifacts")
            )
            dependencies = tuple(
                AuthorityDependency.from_record(_extract.mapping(item, "dependencies"))
                for item in _extract.sequence(validated["dependencies"], "dependencies")
            )
            validations = tuple(
                AuthorityIndexValidationEntry.from_record(
                    _extract.mapping(item, "validations")
                )
                for item in _extract.sequence(validated["validations"], "validations")
            )
        except ValueError as error:
            raise AuthorityIndexValidationError(str(error)) from error
        return cls(
            id=_extract.identifier(validated["id"], "id"),
            artifacts=artifacts,
            dependencies=dependencies,
            validations=validations,
        )

    @classmethod
    def from_artifacts(
        cls,
        *,
        id: ProtocolIdentifier,
        artifacts: tuple[ArtifactReference, ...],
        dependencies: tuple[AuthorityDependency, ...] = (),
    ) -> AuthorityIndex:
        """Build an index and mark dependency endpoints outside artifacts as dangling."""

        validations = _validation_entries_for(
            artifacts=artifacts,
            dependencies=dependencies,
        )
        return cls(
            id=id,
            artifacts=artifacts,
            dependencies=dependencies,
            validations=validations,
        )

    @classmethod
    def from_records(
        cls,
        *,
        id: ProtocolIdentifier,
        artifact_indexes: tuple[ArtifactIndex, ...] = (),
    ) -> AuthorityIndex:
        """Build an index from explicit already-loaded public records."""

        artifacts = list[ArtifactReference]()
        dependencies = list[AuthorityDependency]()
        for index in artifact_indexes:
            index_reference = reference_for_record(
                kind="artifact-index",
                record=index.to_record(),
            )
            artifacts.append(index_reference)
            artifacts.extend(index.artifacts)
            dependencies.extend(
                AuthorityDependency(
                    source=index_reference,
                    target=artifact,
                    relation="contains",
                )
                for artifact in index.artifacts
            )
        return cls.from_artifacts(
            id=id,
            artifacts=tuple(artifacts),
            dependencies=tuple(dependencies),
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    @property
    def dangling_references(self) -> tuple[ArtifactReference, ...]:
        return tuple(
            entry.artifact for entry in self.validations if entry.status == "dangling"
        )

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "artifacts": [artifact.to_record() for artifact in self.artifacts],
            "dependencies": [dependency.to_record() for dependency in self.dependencies],
            "validations": [validation.to_record() for validation in self.validations],
        }


@dataclass(frozen=True, slots=True)
class AuthorityIndexDocument:
    """A loaded authority index and its canonical digest."""

    index: AuthorityIndex
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> AuthorityIndexDocument:
        try:
            record = load_object_document(data, description="authority index document")
        except ContentEncodingError as error:
            raise AuthorityIndexValidationError(str(error)) from error
        index = AuthorityIndex.from_record(record)
        return cls(index=index, digest=index.digest)


def _validation_entries_for(
    *,
    artifacts: tuple[ArtifactReference, ...],
    dependencies: tuple[AuthorityDependency, ...],
) -> tuple[AuthorityIndexValidationEntry, ...]:
    artifact_keys = {reference_sort_key(artifact) for artifact in artifacts}
    entries = [
        AuthorityIndexValidationEntry(
            artifact=artifact,
            status="valid",
            message="artifact reference was supplied explicitly",
        )
        for artifact in artifacts
    ]
    dependency_references = _unique_references(
        reference
        for dependency in dependencies
        for reference in (dependency.source, dependency.target)
    )
    entries.extend(
        AuthorityIndexValidationEntry(
            artifact=reference,
            status="dangling",
            message="dependency endpoint was not supplied as an indexed artifact",
        )
        for reference in dependency_references
        if reference_sort_key(reference) not in artifact_keys
    )
    return tuple(entries)


def _unique_references(references: Iterable[ArtifactReference]) -> tuple[ArtifactReference, ...]:
    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[ArtifactReference] = []
    for reference in references:
        key = reference_sort_key(reference)
        if key in seen:
            continue
        seen.add(key)
        unique.append(reference)
    return tuple(unique)


def _dependency_sort_key(dependency: AuthorityDependency) -> tuple[str, str, str]:
    return (
        _reference_identity(dependency.source),
        _reference_identity(dependency.target),
        dependency.relation,
    )


def _validation_sort_key(
    validation: AuthorityIndexValidationEntry,
) -> tuple[str, str, str]:
    return (
        _reference_identity(validation.artifact),
        validation.status,
        validation.message,
    )


def _first_duplicate_dependency(
    dependencies: tuple[AuthorityDependency, ...],
) -> str | None:
    seen: set[tuple[str, str, str]] = set()
    for dependency in dependencies:
        key = _dependency_sort_key(dependency)
        if key in seen:
            return "/".join(key)
        seen.add(key)
    return None


def _first_duplicate_validation(
    validations: tuple[AuthorityIndexValidationEntry, ...],
) -> str | None:
    seen: set[tuple[str, str, str]] = set()
    for validation in validations:
        key = _validation_sort_key(validation)
        if key in seen:
            return "/".join(key)
        seen.add(key)
    return None


def _reference_identity(reference: ArtifactReference) -> str:
    return "/".join(reference_sort_key(reference))
