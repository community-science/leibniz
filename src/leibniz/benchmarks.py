"""Benchmark declarations for finite-outcome scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz._documents import ContentEncodingError, load_object_document
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.records import FieldSpec, RecordSpec, RecordValidationError

__all__ = [
    "BenchmarkDeclaration",
    "BenchmarkDeclarationValidationError",
    "BenchmarkManifestDocument",
    "BenchmarkManifest",
    "BenchmarkManifestValidationError",
]

_oracle_acceptance_id = ProtocolIdentifier.parse("core.finite-outcome-accepted-event@0.1.0")
_prediction_interface_id = ProtocolIdentifier.parse(
    "core.finite-probability-measure-prediction@0.1.0"
)
_score_functional_id = ProtocolIdentifier.parse("core.negative-log-accepted-mass@0.1.0")
_benchmark_declaration_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "outcome_space_id": FieldSpec(kind="identifier"),
        "oracle_acceptance_id": FieldSpec(kind="identifier", required=False),
        "prediction_interface_id": FieldSpec(kind="identifier", required=False),
        "score_functional_id": FieldSpec(kind="identifier", required=False),
    }
)
_benchmark_manifest_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "name": FieldSpec(kind="name", required=False),
        "outcome_space_id": FieldSpec(kind="identifier", required=False),
        "observation_ids": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="string"),
            required=False,
        ),
        "declaration": FieldSpec(kind="record", required=False),
    }
)


class BenchmarkDeclarationValidationError(ValueError):
    """Raised when a benchmark declaration is invalid."""


class BenchmarkManifestValidationError(ValueError):
    """Raised when a benchmark manifest is invalid."""


@dataclass(frozen=True, slots=True)
class BenchmarkDeclaration:
    """A finite-outcome benchmark scoring declaration."""

    id: ProtocolIdentifier
    outcome_space_id: ProtocolIdentifier
    oracle_acceptance_id: ProtocolIdentifier = _oracle_acceptance_id
    prediction_interface_id: ProtocolIdentifier = _prediction_interface_id
    score_functional_id: ProtocolIdentifier = _score_functional_id

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise BenchmarkDeclarationValidationError(str(error)) from error
        _require_identifier(
            field="oracle_acceptance_id",
            actual=self.oracle_acceptance_id,
            expected=_oracle_acceptance_id,
        )
        _require_identifier(
            field="prediction_interface_id",
            actual=self.prediction_interface_id,
            expected=_prediction_interface_id,
        )
        _require_identifier(
            field="score_functional_id",
            actual=self.score_functional_id,
            expected=_score_functional_id,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BenchmarkDeclaration:
        try:
            validated = _benchmark_declaration_record.validate(record)
        except RecordValidationError as error:
            raise BenchmarkDeclarationValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            outcome_space_id=_as_identifier(
                validated["outcome_space_id"],
                field="outcome_space_id",
            ),
            oracle_acceptance_id=_as_identifier_or_default(
                validated.get("oracle_acceptance_id"),
                field="oracle_acceptance_id",
                default=_oracle_acceptance_id,
            ),
            prediction_interface_id=_as_identifier_or_default(
                validated.get("prediction_interface_id"),
                field="prediction_interface_id",
                default=_prediction_interface_id,
            ),
            score_functional_id=_as_identifier_or_default(
                validated.get("score_functional_id"),
                field="score_functional_id",
                default=_score_functional_id,
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "outcome_space_id": str(self.outcome_space_id),
            "oracle_acceptance_id": str(self.oracle_acceptance_id),
            "prediction_interface_id": str(self.prediction_interface_id),
            "score_functional_id": str(self.score_functional_id),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """A benchmark manifest with identity metadata and one declaration."""

    id: ProtocolIdentifier
    name: ProtocolName
    declaration: BenchmarkDeclaration
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
        if self.declaration.id != self.id:
            raise BenchmarkManifestValidationError(
                f"declaration id {self.declaration.id} does not match {self.id}"
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
        try:
            declaration = _manifest_declaration(validated)
        except ValueError as error:
            raise BenchmarkManifestValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            name=_manifest_name(validated),
            declaration=declaration,
            observation_ids=_manifest_observation_ids(validated),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "name": str(self.name),
            "declaration": self.declaration.to_record(),
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
        raise BenchmarkDeclarationValidationError(f"{field}: expected parsed identifier")
    return value


def _as_identifier_or_default(
    value: object,
    *,
    field: str,
    default: ProtocolIdentifier,
) -> ProtocolIdentifier:
    if value is None:
        return default
    return _as_identifier(value, field=field)


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


def _manifest_declaration(validated: Mapping[str, object]) -> BenchmarkDeclaration:
    declaration_value = validated.get("declaration")
    outcome_space_id_value = validated.get("outcome_space_id")
    if declaration_value is None and outcome_space_id_value is None:
        raise BenchmarkManifestValidationError("outcome_space_id: missing required field")

    declaration: BenchmarkDeclaration | None = None
    if declaration_value is not None:
        declaration = BenchmarkDeclaration.from_record(
            _manifest_mapping(declaration_value, field="declaration")
        )
    if outcome_space_id_value is None:
        if declaration is None:
            raise BenchmarkManifestValidationError("declaration: expected record")
        return declaration

    outcome_space_id = _as_identifier(outcome_space_id_value, field="outcome_space_id")
    if declaration is None:
        return BenchmarkDeclaration(
            id=_as_identifier(validated["id"], field="id"),
            outcome_space_id=outcome_space_id,
        )
    if declaration.outcome_space_id != outcome_space_id:
        raise BenchmarkManifestValidationError(
            f"outcome_space_id {outcome_space_id} does not match declaration "
            f"{declaration.outcome_space_id}"
        )
    return declaration


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


def _manifest_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BenchmarkManifestValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _require_identifier(
    *,
    field: str,
    actual: ProtocolIdentifier,
    expected: ProtocolIdentifier,
) -> None:
    if actual != expected:
        raise BenchmarkDeclarationValidationError(f"{field} {actual} does not match {expected}")
