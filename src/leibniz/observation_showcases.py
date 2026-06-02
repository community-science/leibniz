"""Benchmark-owned declarations of observations worth inspecting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.records import FieldSpec, RecordSpec

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
            id=_as_identifier(validated["id"], field="id"),
            label=_as_string(validated["label"], field="label"),
            sample_index=_as_int(validated["sample_index"], field="sample_index"),
            seed=_as_int(validated["seed"], field="seed"),
            component_sequence=tuple(
                _as_int(index, field="component_sequence")
                for index in _as_sequence(
                    validated["component_sequence"],
                    field="component_sequence",
                )
            ),
            outcome_id=_optional_string(validated.get("outcome_id"), field="outcome_id"),
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
            id=_as_identifier(validated["id"], field="id"),
            benchmark_id=_as_identifier(validated["benchmark_id"], field="benchmark_id"),
            formation_declaration=ArtifactReference.from_record(
                _as_mapping(validated["formation_declaration"], field="formation_declaration")
            ),
            materialization_declaration=ArtifactReference.from_record(
                _as_mapping(
                    validated["materialization_declaration"],
                    field="materialization_declaration",
                )
            ),
            samples=tuple(
                ObservationShowcaseSample.from_record(_as_mapping(sample, field="samples"))
                for sample in _as_sequence(validated["samples"], field="samples")
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


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise ObservationShowcaseValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ObservationShowcaseValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ObservationShowcaseValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ObservationShowcaseValidationError(f"{field}: expected string")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _as_string(value, field=field)


def _as_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise ObservationShowcaseValidationError(f"{field}: expected integer")
    return value


def _first_duplicate(values: tuple[object, ...]) -> object | None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
