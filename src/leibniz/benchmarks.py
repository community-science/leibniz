"""Benchmark manifests for finite-outcome scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz._documents import ContentEncodingError, load_object_document
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "BenchmarkManifestDocument",
    "BenchmarkManifest",
    "BenchmarkManifestValidationError",
]

_benchmark_manifest_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "name": FieldSpec(kind="name", required=False),
        "outcome_space_id": FieldSpec(kind="identifier"),
        "observation_ids": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="string"),
            required=False,
        ),
    }
)


class BenchmarkManifestValidationError(ValueError):
    """Raised when a benchmark manifest is invalid."""


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """A finite-outcome benchmark manifest."""

    id: ProtocolIdentifier
    name: ProtocolName
    outcome_space_id: ProtocolIdentifier
    observation_ids: frozenset[str] | None = None

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise BenchmarkManifestValidationError(str(error)) from error
        if self.id.name != self.name:
            raise BenchmarkManifestValidationError(
                f"name {self.name} does not match id name {self.id.name}"
            )
        if self.observation_ids is not None:
            if not self.observation_ids:
                raise BenchmarkManifestValidationError(
                    "observation_ids must contain at least one observation id"
                )
            if any(not observation_id for observation_id in self.observation_ids):
                raise BenchmarkManifestValidationError("observation_ids must be nonempty")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BenchmarkManifest:
        try:
            validated = _benchmark_manifest_record.validate(record)
        except ValueError as error:
            raise BenchmarkManifestValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            name=_manifest_name(validated),
            outcome_space_id=_as_identifier(
                validated["outcome_space_id"],
                field="outcome_space_id",
            ),
            observation_ids=_manifest_observation_ids(validated),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "name": str(self.name),
            "outcome_space_id": str(self.outcome_space_id),
        }
        if self.observation_ids is not None:
            record["observation_ids"] = sorted(self.observation_ids)
        return record


@dataclass(frozen=True, slots=True)
class BenchmarkManifestDocument:
    """A loaded benchmark manifest and the digest of its canonical record."""

    manifest: BenchmarkManifest
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> BenchmarkManifestDocument:
        try:
            record = load_object_document(data, description="manifest document")
        except ContentEncodingError as error:
            raise BenchmarkManifestValidationError(str(error)) from error
        manifest = BenchmarkManifest.from_record(record)
        return cls(manifest=manifest, digest=ContentDigest.from_value(manifest.to_record()))


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise BenchmarkManifestValidationError(f"{field}: expected parsed identifier")
    return value


def _as_name(value: object, *, field: str) -> ProtocolName:
    if not isinstance(value, ProtocolName):
        raise BenchmarkManifestValidationError(f"{field}: expected parsed name")
    return value


def _manifest_name(validated: Mapping[str, object]) -> ProtocolName:
    identifier = _as_identifier(validated["id"], field="id")
    value = validated.get("name")
    if value is None:
        return identifier.name
    return _as_name(value, field="name")


def _manifest_observation_ids(validated: Mapping[str, object]) -> frozenset[str] | None:
    value = validated.get("observation_ids")
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise BenchmarkManifestValidationError("observation_ids: expected parsed sequence")
    observation_ids = tuple(
        str(observation_id) for observation_id in cast(tuple[object, ...], value)
    )
    if len(set(observation_ids)) != len(observation_ids):
        raise BenchmarkManifestValidationError("observation_ids must be unique")
    return frozenset(observation_ids)
