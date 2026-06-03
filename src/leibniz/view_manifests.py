"""Metadata manifests for derived artifact views."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec

__all__ = [
    "ViewManifest",
    "ViewManifestDocument",
    "ViewManifestValidationError",
]

_ProjectionKind: TypeAlias = Literal["comparison", "frontier", "ranking", "summary"]
_ScoreDirection: TypeAlias = Literal["higher", "lower"]

_name = re.compile(r"^[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*$")
_subject_kinds = frozenset(
    (
        "competition-bundle",
        "evaluation-bundle",
        "measurement-score-view",
        "model-lineage",
        "resource-report-set",
    )
)
_projection_kinds = frozenset(("comparison", "frontier", "ranking", "summary"))
_score_directions = frozenset(("higher", "lower"))

_view_manifest_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "subject_kind": FieldSpec(kind="string"),
        "subject": FieldSpec(kind="record"),
        "projection_kind": FieldSpec(kind="string"),
        "source_artifacts": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "metric_name": FieldSpec(kind="string"),
        "score_direction": FieldSpec(kind="string"),
        "cost_axes": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="string"),
            required=False,
        ),
    }
)


class ViewManifestValidationError(ValueError):
    """Raised when a view manifest is invalid."""


_extract = RecordExtractor(error_type=ViewManifestValidationError)


@dataclass(frozen=True, slots=True)
class ViewManifest:
    """Declared metadata for a derived projection over protocol artifacts."""

    id: ProtocolIdentifier
    subject_kind: str
    subject: ArtifactReference
    projection_kind: _ProjectionKind
    source_artifacts: tuple[ArtifactReference, ...]
    metric_name: str
    score_direction: _ScoreDirection
    cost_axes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise ViewManifestValidationError(str(error)) from error
        if not str(self.id.name).startswith("view-manifests."):
            raise ViewManifestValidationError("id must be a valid view manifest id")
        if self.subject_kind not in _subject_kinds:
            raise ViewManifestValidationError(
                f"unsupported subject_kind: {self.subject_kind}"
            )
        if self.subject.kind != self.subject_kind:
            raise ViewManifestValidationError("subject kind must match subject_kind")
        if self.projection_kind not in _projection_kinds:
            raise ViewManifestValidationError(
                f"unsupported projection_kind: {self.projection_kind}"
            )
        _validate_name(self.metric_name, field="metric_name")
        if self.score_direction not in _score_directions:
            raise ViewManifestValidationError(
                f"unsupported score_direction: {self.score_direction}"
            )
        if not self.source_artifacts:
            raise ViewManifestValidationError(
                "source_artifacts must contain at least one artifact reference"
            )
        duplicate_source = _first_duplicate_reference(self.source_artifacts)
        if duplicate_source is not None:
            raise ViewManifestValidationError(
                f"duplicate source artifact reference: {duplicate_source}"
            )
        for axis in self.cost_axes:
            _validate_name(axis, field="cost_axes")
        duplicate_axis = _first_duplicate(self.cost_axes)
        if duplicate_axis is not None:
            raise ViewManifestValidationError(f"duplicate cost axis: {duplicate_axis}")
        object.__setattr__(
            self,
            "source_artifacts",
            tuple(sorted(self.source_artifacts, key=_reference_sort_key)),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ViewManifest:
        try:
            validated = _view_manifest_record.validate(record)
            source_artifacts = tuple(
                ArtifactReference.from_record(_extract.mapping(item, "source_artifacts"))
                for item in _extract.sequence(
                    validated["source_artifacts"],
                    "source_artifacts",
                )
            )
            cost_axes = tuple(
                _extract.string(item, "cost_axes")
                for item in _extract.sequence(validated.get("cost_axes", ()), "cost_axes")
            )
        except ValueError as error:
            raise ViewManifestValidationError(str(error)) from error
        return cls(
            id=_extract.identifier(validated["id"], "id"),
            subject_kind=_extract.string(validated["subject_kind"], "subject_kind"),
            subject=ArtifactReference.from_record(
                _extract.mapping(validated["subject"], "subject")
            ),
            projection_kind=cast(
                _ProjectionKind,
                _extract.string(validated["projection_kind"], "projection_kind"),
            ),
            source_artifacts=source_artifacts,
            metric_name=_extract.string(validated["metric_name"], "metric_name"),
            score_direction=cast(
                _ScoreDirection,
                _extract.string(validated["score_direction"], "score_direction"),
            ),
            cost_axes=cost_axes,
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_subject(self, subject_record: Mapping[str, object]) -> None:
        """Validate that the declared subject reference identifies a supplied record."""

        if not self.subject.matches_record(subject_record):
            raise ViewManifestValidationError("subject reference does not match record")

    def validate_source_artifacts(
        self,
        source_records: tuple[Mapping[str, object], ...],
    ) -> None:
        """Validate comparable source references against supplied source records."""

        unmatched = tuple(
            source
            for source in self.source_artifacts
            if not any(source.matches_record(record) for record in source_records)
        )
        if unmatched:
            raise ViewManifestValidationError(
                f"source artifact reference does not match supplied records: "
                f"{_reference_identity(unmatched[0])}"
            )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "subject_kind": self.subject_kind,
            "subject": self.subject.to_record(),
            "projection_kind": self.projection_kind,
            "source_artifacts": [
                artifact.to_record() for artifact in self.source_artifacts
            ],
            "metric_name": self.metric_name,
            "score_direction": self.score_direction,
        }
        if self.cost_axes:
            record["cost_axes"] = list(self.cost_axes)
        return record


@dataclass(frozen=True, slots=True)
class ViewManifestDocument:
    """A loaded view manifest and its canonical digest."""

    manifest: ViewManifest
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> ViewManifestDocument:
        try:
            record = load_object_document(data, description="view manifest document")
        except ContentEncodingError as error:
            raise ViewManifestValidationError(str(error)) from error
        manifest = ViewManifest.from_record(record)
        return cls(manifest=manifest, digest=manifest.digest)


def _validate_name(value: str, *, field: str) -> None:
    if _name.fullmatch(value) is None:
        raise ViewManifestValidationError(f"{field} must be a valid metric or axis name")
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


def _first_duplicate(values: tuple[str, ...]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _reference_identity(reference: ArtifactReference) -> str:
    return "/".join(_reference_sort_key(reference))
