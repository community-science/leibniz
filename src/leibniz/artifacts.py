"""Durable references to protocol artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast
from urllib.parse import urlparse

from leibniz._documents import ContentEncodingError, load_object_document
from leibniz.content import ContentDigest
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "ArtifactIndex",
    "ArtifactIndexDocument",
    "ArtifactReference",
    "ArtifactReferenceDocument",
    "ArtifactReferenceValidationError",
    "reference_for_record",
]

_kind = re.compile(
    r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*(?:/[a-z][a-z0-9]*(?:-[a-z0-9]+)*)*$"
)
_artifact_reference_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="string"),
        "protocol_id": FieldSpec(kind="identifier", required=False),
        "content_digest": FieldSpec(kind="string", required=False),
        "record_digest": FieldSpec(kind="string", required=False),
        "external_uri": FieldSpec(kind="string", required=False),
    }
)
_artifact_index_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "source_kind": FieldSpec(kind="string", required=False),
        "source_digest": FieldSpec(kind="string", required=False),
        "artifacts": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
    }
)


class ArtifactReferenceValidationError(ValueError):
    """Raised when an artifact reference is invalid."""


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """A durable reference to an artifact without lookup behavior."""

    kind: str
    protocol_id: ProtocolIdentifier | None = None
    content_digest: ContentDigest | None = None
    record_digest: ContentDigest | None = None
    external_uri: str | None = None

    def __post_init__(self) -> None:
        _validate_kind(self.kind)
        if self.protocol_id is not None:
            try:
                self.protocol_id.require_unreleased()
            except IdentifierSyntaxError as error:
                raise ArtifactReferenceValidationError(str(error)) from error
        if self.external_uri is not None:
            _validate_external_uri(self.external_uri)
        if (
            self.protocol_id is None
            and self.content_digest is None
            and self.record_digest is None
            and self.external_uri is None
        ):
            raise ArtifactReferenceValidationError(
                "artifact reference must include at least one durable identity"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ArtifactReference:
        try:
            validated = _artifact_reference_record.validate(record)
        except ValueError as error:
            raise ArtifactReferenceValidationError(str(error)) from error
        return cls(
            kind=_as_string(validated["kind"], field="kind"),
            protocol_id=_as_optional_identifier(validated.get("protocol_id")),
            content_digest=_as_optional_digest(
                validated.get("content_digest"),
                field="content_digest",
            ),
            record_digest=_as_optional_digest(
                validated.get("record_digest"),
                field="record_digest",
            ),
            external_uri=_as_optional_string(validated.get("external_uri"), field="external_uri"),
        )

    def matches_record(self, record: Mapping[str, object]) -> bool:
        """Return whether comparable identity fields match an embedded record."""

        if self.protocol_id is not None:
            record_id = record.get("id")
            if not isinstance(record_id, str):
                return False
            try:
                if ProtocolIdentifier.parse(record_id) != self.protocol_id:
                    return False
            except IdentifierSyntaxError:
                return False
        if (
            self.record_digest is not None
            and ContentDigest.from_value(record) != self.record_digest
        ):
            return False
        return self.protocol_id is not None or self.record_digest is not None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {"kind": self.kind}
        if self.protocol_id is not None:
            record["protocol_id"] = str(self.protocol_id)
        if self.content_digest is not None:
            record["content_digest"] = str(self.content_digest)
        if self.record_digest is not None:
            record["record_digest"] = str(self.record_digest)
        if self.external_uri is not None:
            record["external_uri"] = self.external_uri
        return record


@dataclass(frozen=True, slots=True)
class ArtifactReferenceDocument:
    """A loaded artifact reference and its canonical digest."""

    reference: ArtifactReference
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> ArtifactReferenceDocument:
        try:
            record = load_object_document(data, description="artifact reference document")
        except ContentEncodingError as error:
            raise ArtifactReferenceValidationError(str(error)) from error
        reference = ArtifactReference.from_record(record)
        return cls(reference=reference, digest=ContentDigest.from_value(reference.to_record()))


@dataclass(frozen=True, slots=True)
class ArtifactIndex:
    """A canonical local inventory over explicit artifact references."""

    id: ProtocolIdentifier
    artifacts: tuple[ArtifactReference, ...]
    source_kind: str | None = None
    source_digest: ContentDigest | None = None

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise ArtifactReferenceValidationError(str(error)) from error
        if not str(self.id.name).startswith("artifact-indexes."):
            raise ArtifactReferenceValidationError("id must be a valid artifact index id")
        if (self.source_kind is None) != (self.source_digest is None):
            raise ArtifactReferenceValidationError(
                "source_kind and source_digest must be supplied together"
            )
        if self.source_kind is not None:
            _validate_kind(self.source_kind)
        if not self.artifacts:
            raise ArtifactReferenceValidationError(
                "artifacts must contain at least one artifact reference"
            )
        expected_artifacts = tuple(sorted(self.artifacts, key=_reference_sort_key))
        if self.artifacts != expected_artifacts:
            object.__setattr__(self, "artifacts", expected_artifacts)
        duplicate = _first_duplicate_reference(self.artifacts)
        if duplicate is not None:
            raise ArtifactReferenceValidationError(f"duplicate artifact reference: {duplicate}")

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        source_record: Mapping[str, object] | None = None,
    ) -> ArtifactIndex:
        try:
            validated = _artifact_index_record.validate(record)
            artifacts = tuple(
                ArtifactReference.from_record(_as_mapping(item, field="artifacts"))
                for item in _as_sequence(validated["artifacts"], field="artifacts")
            )
        except ValueError as error:
            raise ArtifactReferenceValidationError(str(error)) from error
        index = cls(
            id=_as_identifier(validated["id"], field="id"),
            artifacts=artifacts,
            source_kind=_as_optional_string(validated.get("source_kind"), field="source_kind"),
            source_digest=_as_optional_digest(
                validated.get("source_digest"),
                field="source_digest",
            ),
        )
        if source_record is not None:
            index.validate_source(source_record)
        return index

    @classmethod
    def from_source_record(
        cls,
        *,
        id: ProtocolIdentifier,
        source_kind: str,
        source_record: Mapping[str, object],
        artifacts: tuple[ArtifactReference, ...],
    ) -> ArtifactIndex:
        return cls(
            id=id,
            source_kind=source_kind,
            source_digest=ContentDigest.from_value(source_record),
            artifacts=artifacts,
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_source(self, source_record: Mapping[str, object]) -> None:
        if self.source_digest is None:
            raise ArtifactReferenceValidationError(
                "source_digest is required when validating a source record"
            )
        if self.source_digest != ContentDigest.from_value(source_record):
            raise ArtifactReferenceValidationError("source_digest does not match source record")

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "artifacts": [artifact.to_record() for artifact in self.artifacts],
        }
        if self.source_kind is not None:
            record["source_kind"] = self.source_kind
        if self.source_digest is not None:
            record["source_digest"] = str(self.source_digest)
        return record


@dataclass(frozen=True, slots=True)
class ArtifactIndexDocument:
    """A loaded artifact index and its canonical digest."""

    index: ArtifactIndex
    digest: ContentDigest

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        source_record: Mapping[str, object] | None = None,
    ) -> ArtifactIndexDocument:
        try:
            record = load_object_document(data, description="artifact index document")
        except ContentEncodingError as error:
            raise ArtifactReferenceValidationError(str(error)) from error
        index = ArtifactIndex.from_record(record, source_record=source_record)
        return cls(index=index, digest=index.digest)


def reference_for_record(
    *,
    kind: str,
    record: Mapping[str, object],
) -> ArtifactReference:
    """Summarize an embedded public record by its durable identities."""

    record_id = record.get("id")
    protocol_id = None
    if record_id is not None:
        if not isinstance(record_id, str):
            raise ArtifactReferenceValidationError("id: expected identifier string")
        try:
            protocol_id = ProtocolIdentifier.parse(record_id)
        except IdentifierSyntaxError as error:
            raise ArtifactReferenceValidationError(str(error)) from error
    return ArtifactReference(
        kind=kind,
        protocol_id=protocol_id,
        record_digest=ContentDigest.from_value(record),
    )


def _validate_kind(kind: str) -> None:
    if _kind.fullmatch(kind) is None:
        raise ArtifactReferenceValidationError("kind must be a stable lowercase artifact kind")


def _validate_external_uri(uri: str) -> None:
    if not uri:
        raise ArtifactReferenceValidationError("external_uri must be nonempty")
    if uri.startswith((".", "/")):
        raise ArtifactReferenceValidationError("external_uri must not be a local path")
    parsed = urlparse(uri)
    if not parsed.scheme:
        raise ArtifactReferenceValidationError("external_uri must include a URI scheme")
    if parsed.scheme == "file":
        raise ArtifactReferenceValidationError("external_uri must not use file URI scheme")
    if parsed.scheme in {"http", "https"} and not parsed.netloc:
        raise ArtifactReferenceValidationError("external_uri must include a URI authority")
    if parsed.username is not None or parsed.password is not None:
        raise ArtifactReferenceValidationError("external_uri must not include credentials")


def _as_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ArtifactReferenceValidationError(f"{field}: expected string")
    return value


def _as_optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _as_string(value, field=field)


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise ArtifactReferenceValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArtifactReferenceValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ArtifactReferenceValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_optional_identifier(value: object) -> ProtocolIdentifier | None:
    if value is None:
        return None
    if not isinstance(value, ProtocolIdentifier):
        raise ArtifactReferenceValidationError("protocol_id: expected parsed identifier")
    return value


def _as_optional_digest(value: object, *, field: str) -> ContentDigest | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ArtifactReferenceValidationError(f"{field}: expected digest string")
    algorithm, separator, digest_hex = value.partition(":")
    if separator == "":
        raise ArtifactReferenceValidationError(f"{field}: expected algorithm:digest")
    try:
        return ContentDigest(algorithm=algorithm, hex=digest_hex)
    except ContentEncodingError as error:
        raise ArtifactReferenceValidationError(str(error)) from error


def _reference_sort_key(reference: ArtifactReference) -> tuple[str, str, str, str, str]:
    return (
        reference.kind,
        str(reference.protocol_id) if reference.protocol_id is not None else "",
        str(reference.content_digest) if reference.content_digest is not None else "",
        str(reference.record_digest) if reference.record_digest is not None else "",
        reference.external_uri if reference.external_uri is not None else "",
    )


def _first_duplicate_reference(references: tuple[ArtifactReference, ...]) -> str | None:
    seen: set[str] = set()
    for reference in references:
        key = str(ContentDigest.from_value(reference.to_record()))
        if key in seen:
            return key
        seen.add(key)
    return None
