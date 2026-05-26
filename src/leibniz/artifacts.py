"""Durable references to protocol artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

from leibniz._documents import ContentEncodingError, load_object_document
from leibniz.content import ContentDigest
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
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
