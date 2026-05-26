"""Auditable scientific projection records over protocol artifacts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "ProjectionRecord",
    "ProjectionRecordDocument",
    "ProjectionRecordValidationError",
]

_Modality: TypeAlias = Literal[
    "convention",
    "derivation",
    "estimate",
    "hypothesis",
    "measurement",
    "theorem",
]
_Status: TypeAlias = Literal["proposed", "refuted", "superseded", "validated"]

_name = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_modalities = frozenset(
    ("convention", "derivation", "estimate", "hypothesis", "measurement", "theorem")
)
_statuses = frozenset(("proposed", "refuted", "superseded", "validated"))

_projection_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "subject": FieldSpec(kind="record"),
        "predicate": FieldSpec(kind="string"),
        "object": FieldSpec(kind="record"),
        "scope": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "evidence": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "modality": FieldSpec(kind="string"),
        "status": FieldSpec(kind="string"),
        "statement": FieldSpec(kind="string"),
        "assumptions": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="string"),
        ),
        "limitations": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="string"),
        ),
    }
)


class ProjectionRecordValidationError(ValueError):
    """Raised when a projection record is invalid."""


@dataclass(frozen=True, slots=True)
class ProjectionRecord:
    """A scoped scientific claim with explicit evidence references."""

    id: ProtocolIdentifier
    subject: ArtifactReference
    predicate: str
    object: ArtifactReference
    scope: tuple[ArtifactReference, ...]
    evidence: tuple[ArtifactReference, ...]
    modality: _Modality
    status: _Status
    statement: str
    assumptions: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise ProjectionRecordValidationError(str(error)) from error
        if not str(self.id.name).startswith("projection-records."):
            raise ProjectionRecordValidationError("id must be a valid projection record id")
        _validate_name(self.predicate, field="predicate")
        if not self.scope:
            raise ProjectionRecordValidationError(
                "scope must contain at least one artifact reference"
            )
        if not self.evidence:
            raise ProjectionRecordValidationError(
                "evidence must contain at least one artifact reference"
            )
        if self.modality not in _modalities:
            raise ProjectionRecordValidationError(
                f"unsupported modality: {self.modality}"
            )
        if self.status not in _statuses:
            raise ProjectionRecordValidationError(f"unsupported status: {self.status}")
        _validate_nonempty_text(self.statement, field="statement")
        _validate_nonempty_texts(self.assumptions, field="assumptions")
        _validate_nonempty_texts(self.limitations, field="limitations")
        duplicate_scope = _first_duplicate_reference(self.scope)
        if duplicate_scope is not None:
            raise ProjectionRecordValidationError(
                f"duplicate scope artifact reference: {duplicate_scope}"
            )
        duplicate_evidence = _first_duplicate_reference(self.evidence)
        if duplicate_evidence is not None:
            raise ProjectionRecordValidationError(
                f"duplicate evidence artifact reference: {duplicate_evidence}"
            )
        object.__setattr__(self, "scope", tuple(sorted(self.scope, key=_reference_sort_key)))
        object.__setattr__(
            self,
            "evidence",
            tuple(sorted(self.evidence, key=_reference_sort_key)),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ProjectionRecord:
        try:
            validated = _projection_record.validate(record)
            scope = tuple(
                ArtifactReference.from_record(_as_mapping(item, field="scope"))
                for item in _as_sequence(validated["scope"], field="scope")
            )
            evidence = tuple(
                ArtifactReference.from_record(_as_mapping(item, field="evidence"))
                for item in _as_sequence(validated["evidence"], field="evidence")
            )
            assumptions = tuple(
                _as_string(item, field="assumptions")
                for item in _as_sequence(validated["assumptions"], field="assumptions")
            )
            limitations = tuple(
                _as_string(item, field="limitations")
                for item in _as_sequence(validated["limitations"], field="limitations")
            )
        except ValueError as error:
            raise ProjectionRecordValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            subject=ArtifactReference.from_record(
                _as_mapping(validated["subject"], field="subject")
            ),
            predicate=_as_string(validated["predicate"], field="predicate"),
            object=ArtifactReference.from_record(
                _as_mapping(validated["object"], field="object")
            ),
            scope=scope,
            evidence=evidence,
            modality=cast(_Modality, _as_string(validated["modality"], field="modality")),
            status=cast(_Status, _as_string(validated["status"], field="status")),
            statement=_as_string(validated["statement"], field="statement"),
            assumptions=assumptions,
            limitations=limitations,
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_references(
        self,
        *,
        subject_record: Mapping[str, object] | None = None,
        object_record: Mapping[str, object] | None = None,
        scope_records: tuple[Mapping[str, object], ...] = (),
        evidence_records: tuple[Mapping[str, object], ...] = (),
    ) -> None:
        """Validate comparable references against supplied records."""

        if subject_record is not None and not self.subject.matches_record(subject_record):
            raise ProjectionRecordValidationError("subject reference does not match record")
        if object_record is not None and not self.object.matches_record(object_record):
            raise ProjectionRecordValidationError("object reference does not match record")
        _validate_supplied_records(
            references=self.scope,
            records=scope_records,
            field="scope",
        )
        _validate_supplied_records(
            references=self.evidence,
            records=evidence_records,
            field="evidence",
        )

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "subject": self.subject.to_record(),
            "predicate": self.predicate,
            "object": self.object.to_record(),
            "scope": [reference.to_record() for reference in self.scope],
            "evidence": [reference.to_record() for reference in self.evidence],
            "modality": self.modality,
            "status": self.status,
            "statement": self.statement,
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class ProjectionRecordDocument:
    """A loaded projection record and its canonical digest."""

    record: ProjectionRecord
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> ProjectionRecordDocument:
        try:
            record = load_object_document(data, description="projection record document")
        except ContentEncodingError as error:
            raise ProjectionRecordValidationError(str(error)) from error
        projection = ProjectionRecord.from_record(record)
        return cls(record=projection, digest=projection.digest)


def _validate_supplied_records(
    *,
    references: tuple[ArtifactReference, ...],
    records: tuple[Mapping[str, object], ...],
    field: str,
) -> None:
    if not records:
        return
    unmatched = tuple(
        reference
        for reference in references
        if not any(reference.matches_record(record) for record in records)
    )
    if unmatched:
        raise ProjectionRecordValidationError(
            f"{field} reference does not match supplied records: "
            f"{_reference_identity(unmatched[0])}"
        )


def _validate_name(value: str, *, field: str) -> None:
    if _name.fullmatch(value) is None:
        raise ProjectionRecordValidationError(f"{field} must be a valid predicate name")


def _validate_nonempty_text(value: str, *, field: str) -> None:
    if not value.strip():
        raise ProjectionRecordValidationError(f"{field} must be nonempty")


def _validate_nonempty_texts(values: tuple[str, ...], *, field: str) -> None:
    if not values:
        raise ProjectionRecordValidationError(f"{field} must contain at least one item")
    for value in values:
        _validate_nonempty_text(value, field=field)


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise ProjectionRecordValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ProjectionRecordValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ProjectionRecordValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ProjectionRecordValidationError(f"{field}: expected string")
    if not value:
        raise ProjectionRecordValidationError(f"{field}: expected nonempty string")
    return value


def _reference_sort_key(reference: ArtifactReference) -> tuple[str, str, str, str, str]:
    return (
        reference.kind,
        str(reference.protocol_id) if reference.protocol_id is not None else "",
        str(reference.content_digest) if reference.content_digest is not None else "",
        str(reference.record_digest) if reference.record_digest is not None else "",
        reference.external_uri if reference.external_uri is not None else "",
    )


def _first_duplicate_reference(
    references: tuple[ArtifactReference, ...],
) -> str | None:
    seen: set[tuple[str, str, str, str, str]] = set()
    for reference in references:
        key = _reference_sort_key(reference)
        if key in seen:
            return _reference_identity(reference)
        seen.add(key)
    return None


def _reference_identity(reference: ArtifactReference) -> str:
    return "/".join(_reference_sort_key(reference))
