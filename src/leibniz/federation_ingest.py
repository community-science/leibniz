"""Credential-free federation ingest planning records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from leibniz.artifacts import (
    ArtifactReference,
    first_duplicate_reference,
    reference_sort_key,
)
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec
from leibniz.submission_registries import SubmissionRegistry, SubmissionRegistrySource

__all__ = [
    "FederationIngestPlan",
    "FederationIngestPlanDocument",
    "FederationIngestPlanEntry",
    "FederationIngestValidationError",
    "plan_federation_ingest",
]

_IngestStatus: TypeAlias = Literal["would-inspect", "inactive"]
_ingest_plan_entry_record = RecordSpec(
    fields={
        "repository": FieldSpec(kind="string"),
        "repository_type": FieldSpec(kind="string"),
        "enabled": FieldSpec(kind="boolean"),
        "status": FieldSpec(kind="string"),
        "expected_publication_bundles": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "discovered_publication_bundles": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
            required=False,
        ),
    }
)
_ingest_plan_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "registry_digest": FieldSpec(kind="string"),
        "dry_run": FieldSpec(kind="boolean"),
        "entries": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
    }
)


class FederationIngestValidationError(ValueError):
    """Raised when a federation ingest plan is invalid."""


_extract = RecordExtractor(error_type=FederationIngestValidationError)


@dataclass(frozen=True, slots=True)
class FederationIngestPlanEntry:
    """One dry-run ingest planning entry for a registry source."""

    source: SubmissionRegistrySource
    status: _IngestStatus
    expected_publication_bundles: tuple[ArtifactReference, ...]
    discovered_publication_bundles: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in {"would-inspect", "inactive"}:
            raise FederationIngestValidationError(f"unsupported status: {self.status}")
        if self.source.enabled and self.status != "would-inspect":
            raise FederationIngestValidationError("enabled sources must be marked would-inspect")
        if not self.source.enabled and self.status != "inactive":
            raise FederationIngestValidationError("disabled sources must be marked inactive")
        if self.source.enabled and not self.expected_publication_bundles:
            raise FederationIngestValidationError(
                "enabled entries must contain expected publication bundles"
            )
        for reference in self.expected_publication_bundles:
            _validate_publication_bundle_reference(reference, field="expected_publication_bundles")
        for reference in self.discovered_publication_bundles:
            _validate_publication_bundle_reference(
                reference,
                field="discovered_publication_bundles",
            )
        duplicate_expected = first_duplicate_reference(self.expected_publication_bundles)
        if duplicate_expected is not None:
            raise FederationIngestValidationError(
                f"duplicate expected publication bundle reference: {duplicate_expected}"
            )
        duplicate_discovered = first_duplicate_reference(self.discovered_publication_bundles)
        if duplicate_discovered is not None:
            raise FederationIngestValidationError(
                f"duplicate discovered publication bundle reference: {duplicate_discovered}"
            )
        object.__setattr__(
            self,
            "expected_publication_bundles",
            tuple(sorted(self.expected_publication_bundles, key=reference_sort_key)),
        )
        object.__setattr__(
            self,
            "discovered_publication_bundles",
            tuple(sorted(self.discovered_publication_bundles, key=reference_sort_key)),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> FederationIngestPlanEntry:
        try:
            validated = _ingest_plan_entry_record.validate(record)
            expected = tuple(
                ArtifactReference.from_record(
                    _extract.mapping(item, "expected_publication_bundles")
                )
                for item in _extract.sequence(
                    validated["expected_publication_bundles"],
                    "expected_publication_bundles",
                )
            )
            discovered = tuple(
                ArtifactReference.from_record(
                    _extract.mapping(item, "discovered_publication_bundles")
                )
                for item in _extract.sequence(
                    validated.get("discovered_publication_bundles", ()),
                    "discovered_publication_bundles",
                )
            )
        except ValueError as error:
            raise FederationIngestValidationError(str(error)) from error
        return cls(
            source=SubmissionRegistrySource(
                repository=_extract.string(validated["repository"], "repository"),
                repository_type=cast(
                    Literal["dataset", "model", "space"],
                    _extract.string(validated["repository_type"], "repository_type"),
                ),
                enabled=_extract.boolean(validated["enabled"], "enabled"),
            ),
            status=cast(_IngestStatus, _extract.string(validated["status"], "status")),
            expected_publication_bundles=expected,
            discovered_publication_bundles=discovered,
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "repository": self.source.repository,
            "repository_type": self.source.repository_type,
            "enabled": self.source.enabled,
            "status": self.status,
            "expected_publication_bundles": [
                reference.to_record() for reference in self.expected_publication_bundles
            ],
        }
        if self.discovered_publication_bundles:
            record["discovered_publication_bundles"] = [
                reference.to_record() for reference in self.discovered_publication_bundles
            ]
        return record


@dataclass(frozen=True, slots=True)
class FederationIngestPlan:
    """A credential-free dry-run plan for inspecting federation sources."""

    id: ProtocolIdentifier
    registry_digest: ContentDigest
    entries: tuple[FederationIngestPlanEntry, ...]
    dry_run: bool = True

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise FederationIngestValidationError(str(error)) from error
        if not str(self.id.name).startswith("federation-ingest-plans."):
            raise FederationIngestValidationError("id must be a valid federation ingest plan id")
        if not self.dry_run:
            raise FederationIngestValidationError("federation ingest plans must be dry-run")
        if not self.entries:
            raise FederationIngestValidationError("entries must contain at least one ingest entry")
        duplicate_source = _first_duplicate_source(self.entries)
        if duplicate_source is not None:
            repository, repository_type = duplicate_source
            raise FederationIngestValidationError(
                f"duplicate ingest source: {repository} ({repository_type})"
            )
        object.__setattr__(
            self,
            "entries",
            tuple(
                sorted(
                    self.entries,
                    key=lambda entry: (entry.source.repository, entry.source.repository_type),
                )
            ),
        )

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        registry: SubmissionRegistry | None = None,
    ) -> FederationIngestPlan:
        try:
            validated = _ingest_plan_record.validate(record)
            entries = tuple(
                FederationIngestPlanEntry.from_record(_extract.mapping(item, "entries"))
                for item in _extract.sequence(validated["entries"], "entries")
            )
        except ValueError as error:
            raise FederationIngestValidationError(str(error)) from error
        plan = cls(
            id=_extract.identifier(validated["id"], "id"),
            registry_digest=_as_digest(validated["registry_digest"], field="registry_digest"),
            dry_run=_extract.boolean(validated["dry_run"], "dry_run"),
            entries=entries,
        )
        if registry is not None:
            plan.validate_registry(registry)
        return plan

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_registry(self, registry: SubmissionRegistry) -> None:
        if self.registry_digest != registry.digest:
            raise FederationIngestValidationError("registry_digest does not match registry")
        expected_sources = tuple(
            sorted(
                (source.repository, source.repository_type, source.enabled)
                for source in registry.sources
            )
        )
        actual_sources = tuple(
            sorted(
                (entry.source.repository, entry.source.repository_type, entry.source.enabled)
                for entry in self.entries
            )
        )
        if actual_sources != expected_sources:
            raise FederationIngestValidationError("entries do not match registry sources")

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "registry_digest": str(self.registry_digest),
            "dry_run": self.dry_run,
            "entries": [entry.to_record() for entry in self.entries],
        }


@dataclass(frozen=True, slots=True)
class FederationIngestPlanDocument:
    """A loaded federation ingest plan and its canonical digest."""

    plan: FederationIngestPlan
    digest: ContentDigest

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        registry: SubmissionRegistry | None = None,
    ) -> FederationIngestPlanDocument:
        try:
            record = load_object_document(data, description="federation ingest plan document")
        except ContentEncodingError as error:
            raise FederationIngestValidationError(str(error)) from error
        plan = FederationIngestPlan.from_record(record, registry=registry)
        return cls(plan=plan, digest=plan.digest)


def plan_federation_ingest(
    *,
    id: ProtocolIdentifier,
    registry: SubmissionRegistry,
    expected_publication_bundles: tuple[ArtifactReference, ...],
) -> FederationIngestPlan:
    """Create a dry-run ingest plan without contacting external services."""

    if not expected_publication_bundles:
        raise FederationIngestValidationError(
            "expected_publication_bundles must contain at least one reference"
        )
    entries = tuple(
        FederationIngestPlanEntry(
            source=source,
            status="would-inspect" if source.enabled else "inactive",
            expected_publication_bundles=expected_publication_bundles if source.enabled else (),
        )
        for source in registry.sources
    )
    return FederationIngestPlan(
        id=id,
        registry_digest=registry.digest,
        entries=entries,
        dry_run=True,
    )


def _validate_publication_bundle_reference(reference: ArtifactReference, *, field: str) -> None:
    if reference.kind != "publication-bundle":
        raise FederationIngestValidationError(
            f"{field} references must have kind publication-bundle"
        )


def _as_digest(value: object, *, field: str) -> ContentDigest:
    return ContentDigest.from_string(
        value,
        field=field,
        error_type=FederationIngestValidationError,
    )


def _first_duplicate_source(
    entries: tuple[FederationIngestPlanEntry, ...],
) -> tuple[str, str] | None:
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry.source.repository, entry.source.repository_type)
        if key in seen:
            return key
        seen.add(key)
    return None
