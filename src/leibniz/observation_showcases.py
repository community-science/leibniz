"""Benchmark-owned declarations of observations worth inspecting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec

__all__ = [
    "ObservationShowcaseDocument",
    "ObservationShowcaseManifest",
    "ObservationShowcaseSample",
    "ObservationShowcaseValidationError",
]

_sample_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "label": FieldSpec(kind="string"),
        "sample_index": FieldSpec(kind="integer"),
        "seed": FieldSpec(kind="integer"),
        "component_sequence": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "outcome_id": FieldSpec(kind="string", required=False),
    }
)
_showcase_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "benchmark_id": FieldSpec(kind="identifier"),
        "formation_declaration": FieldSpec(kind="record"),
        "materialization_declaration": FieldSpec(kind="record"),
        "samples": FieldSpec(kind="sequence", item=FieldSpec(kind="record")),
    }
)


class ObservationShowcaseValidationError(ValueError):
    """Raised when an observation showcase declaration is invalid."""


_extract = RecordExtractor(error_type=ObservationShowcaseValidationError)


@dataclass(frozen=True, slots=True)
class ObservationShowcaseSample:
    """One benchmark-owned sample requested for data inspection."""

    id: ProtocolIdentifier
    label: str
    sample_index: int
    seed: int
    component_sequence: tuple[int, ...]
    outcome_id: str | None = None

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise ObservationShowcaseValidationError(str(error)) from error
        if not self.label:
            raise ObservationShowcaseValidationError("sample label must be nonempty")
        if type(self.sample_index) is not int:
            raise ObservationShowcaseValidationError("sample_index must be an integer")
        if self.sample_index < 0:
            raise ObservationShowcaseValidationError("sample_index must be nonnegative")
        if type(self.seed) is not int:
            raise ObservationShowcaseValidationError("seed must be an integer")
        if self.seed < 0:
            raise ObservationShowcaseValidationError("seed must be nonnegative")
        if not self.component_sequence:
            raise ObservationShowcaseValidationError("component_sequence must not be empty")
        if any(type(index) is not int or index < 0 for index in self.component_sequence):
            raise ObservationShowcaseValidationError(
                "component_sequence values must be nonnegative integers"
            )
        if self.outcome_id is not None and self.outcome_id == "":
            raise ObservationShowcaseValidationError("outcome_id must be nonempty")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ObservationShowcaseSample:
        try:
            validated = _sample_record.validate(record)
        except ValueError as error:
            raise ObservationShowcaseValidationError(str(error)) from error
        return cls(
            id=_extract.identifier(validated["id"], "id"),
            label=_extract.string(validated["label"], "label"),
            sample_index=_extract.integer(validated["sample_index"], "sample_index"),
            seed=_extract.integer(validated["seed"], "seed"),
            component_sequence=tuple(
                _extract.integer(index, "component_sequence")
                for index in _extract.sequence(
                    validated["component_sequence"],
                    "component_sequence",
                )
            ),
            outcome_id=_extract.optional_string(validated.get("outcome_id"), "outcome_id"),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "label": self.label,
            "sample_index": self.sample_index,
            "seed": self.seed,
            "component_sequence": list(self.component_sequence),
        }
        if self.outcome_id is not None:
            record["outcome_id"] = self.outcome_id
        return record


@dataclass(frozen=True, slots=True)
class ObservationShowcaseManifest:
    """A benchmark-owned declaration of observations to inspect."""

    id: ProtocolIdentifier
    benchmark_id: ProtocolIdentifier
    formation_declaration: ArtifactReference
    materialization_declaration: ArtifactReference
    samples: tuple[ObservationShowcaseSample, ...]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
            self.benchmark_id.require_unreleased()
        except ValueError as error:
            raise ObservationShowcaseValidationError(str(error)) from error
        if self.formation_declaration.kind != "observation-formation-declaration":
            raise ObservationShowcaseValidationError(
                "formation_declaration reference must have kind "
                "observation-formation-declaration"
            )
        if self.materialization_declaration.kind != "materialization-declaration":
            raise ObservationShowcaseValidationError(
                "materialization_declaration reference must have kind "
                "materialization-declaration"
            )
        if not self.samples:
            raise ObservationShowcaseValidationError("samples must not be empty")
        duplicate = _first_duplicate(tuple(sample.id for sample in self.samples))
        if duplicate is not None:
            raise ObservationShowcaseValidationError(f"duplicate sample id: {duplicate}")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ObservationShowcaseManifest:
        try:
            validated = _showcase_record.validate(record)
        except ValueError as error:
            raise ObservationShowcaseValidationError(str(error)) from error
        return cls(
            id=_extract.identifier(validated["id"], "id"),
            benchmark_id=_extract.identifier(validated["benchmark_id"], "benchmark_id"),
            formation_declaration=ArtifactReference.from_record(
                _extract.mapping(validated["formation_declaration"], "formation_declaration")
            ),
            materialization_declaration=ArtifactReference.from_record(
                _extract.mapping(
                    validated["materialization_declaration"],
                    "materialization_declaration",
                )
            ),
            samples=tuple(
                ObservationShowcaseSample.from_record(_extract.mapping(sample, "samples"))
                for sample in _extract.sequence(validated["samples"], "samples")
            ),
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "benchmark_id": str(self.benchmark_id),
            "formation_declaration": self.formation_declaration.to_record(),
            "materialization_declaration": self.materialization_declaration.to_record(),
            "samples": [sample.to_record() for sample in self.samples],
        }


@dataclass(frozen=True, slots=True)
class ObservationShowcaseDocument:
    """A loaded observation showcase manifest and its canonical digest."""

    manifest: ObservationShowcaseManifest
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> ObservationShowcaseDocument:
        try:
            record = load_object_document(data, description="observation showcase")
        except ContentEncodingError as error:
            raise ObservationShowcaseValidationError(str(error)) from error
        manifest = ObservationShowcaseManifest.from_record(record)
        return cls(manifest=manifest, digest=manifest.digest)
def _first_duplicate(values: tuple[object, ...]) -> object | None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
