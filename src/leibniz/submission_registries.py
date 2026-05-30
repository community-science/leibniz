"""Read-only submission registry records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.records import (
    RecordExtractor,
    record_specs_from_package_contract,
)

__all__ = [
    "SubmissionRegistry",
    "SubmissionRegistryDocument",
    "SubmissionRegistrySource",
    "SubmissionRegistryValidationError",
]

_RepositoryType: TypeAlias = Literal["dataset", "model", "space"]
_repo_part = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_record_contracts = record_specs_from_package_contract(
    "leibniz.contract_artifacts",
    "submission_registries",
    description="submission registry record contracts",
)
_submission_registry_source_record = _record_contracts["submission_registry_source"]
_submission_registry_record = _record_contracts["submission_registry"]


class SubmissionRegistryValidationError(ValueError):
    """Raised when a submission registry record is invalid."""


_record = RecordExtractor(SubmissionRegistryValidationError)


@dataclass(frozen=True, slots=True)
class SubmissionRegistrySource:
    """One configured external repository source."""

    repository: str
    repository_type: _RepositoryType = "dataset"
    enabled: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository", _normalize_repository(self.repository))
        if self.repository_type not in {"dataset", "model", "space"}:
            raise SubmissionRegistryValidationError(
                f"unsupported repository_type: {self.repository_type}"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SubmissionRegistrySource:
        try:
            validated = _submission_registry_source_record.validate(record)
        except ValueError as error:
            raise SubmissionRegistryValidationError(str(error)) from error
        return cls(
            repository=_record.string(validated["repository"], "repository"),
            repository_type=cast(
                _RepositoryType,
                _record.string(validated["repository_type"], "repository_type"),
            ),
            enabled=_record.boolean(validated["enabled"], "enabled"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "repository": self.repository,
            "repository_type": self.repository_type,
            "enabled": self.enabled,
        }


@dataclass(frozen=True, slots=True)
class SubmissionRegistry:
    """Operator-owned read-only configuration for known submission sources."""

    id: ProtocolIdentifier
    sources: tuple[SubmissionRegistrySource, ...]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise SubmissionRegistryValidationError(str(error)) from error
        if not str(self.id.name).startswith("submission-registries."):
            raise SubmissionRegistryValidationError("id must be a valid submission registry id")
        if not self.sources:
            raise SubmissionRegistryValidationError(
                "sources must contain at least one submission source"
            )
        duplicate = _first_duplicate(
            tuple((source.repository, source.repository_type) for source in self.sources)
        )
        if duplicate is not None:
            repository, repository_type = duplicate
            raise SubmissionRegistryValidationError(
                f"duplicate repository source: {repository} ({repository_type})"
            )
        object.__setattr__(
            self,
            "sources",
            tuple(
                sorted(
                    self.sources,
                    key=lambda source: (source.repository, source.repository_type),
                )
            ),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SubmissionRegistry:
        try:
            validated = _submission_registry_record.validate(record)
            sources = tuple(
                SubmissionRegistrySource.from_record(_record.mapping(item, "sources"))
                for item in _record.sequence(validated["sources"], "sources")
            )
        except ValueError as error:
            raise SubmissionRegistryValidationError(str(error)) from error
        return cls(
            id=_record.identifier(validated["id"], "id"),
            sources=sources,
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "sources": [source.to_record() for source in self.sources],
        }


@dataclass(frozen=True, slots=True)
class SubmissionRegistryDocument:
    """A loaded submission registry and its canonical digest."""

    registry: SubmissionRegistry
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> SubmissionRegistryDocument:
        try:
            record = load_object_document(data, description="submission registry document")
        except ContentEncodingError as error:
            raise SubmissionRegistryValidationError(str(error)) from error
        registry = SubmissionRegistry.from_record(record)
        return cls(registry=registry, digest=registry.digest)


def _normalize_repository(repository: str) -> str:
    if not repository:
        raise SubmissionRegistryValidationError("repository must be nonempty")
    if "://" in repository:
        raise SubmissionRegistryValidationError("repository must not include a URI scheme")
    if repository.startswith((".", "/")):
        raise SubmissionRegistryValidationError("repository must not be a local path")
    if "@" in repository:
        raise SubmissionRegistryValidationError("repository must not include credentials")
    parts = tuple(part for part in repository.strip().split("/") if part)
    if len(parts) != 2:
        raise SubmissionRegistryValidationError("repository must be owner/name")
    owner, name = parts
    if owner in {".", ".."} or name in {".", ".."}:
        raise SubmissionRegistryValidationError("repository must be a stable owner/name")
    if _repo_part.fullmatch(owner) is None or _repo_part.fullmatch(name) is None:
        raise SubmissionRegistryValidationError("repository must be a stable owner/name")
    return f"{owner.lower()}/{name.lower()}"


def _first_duplicate(values: tuple[tuple[str, str], ...]) -> tuple[str, str] | None:
    seen: set[tuple[str, str]] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
