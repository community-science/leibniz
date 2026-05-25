"""Benchmark declarations for finite-answer scoring."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz.answers import FiniteAnswerScoringBundle
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier, ProtocolName, require_unreleased_identifier
from leibniz.records import RecordSpec, RecordValidationError, optional, required, validate_record

__all__ = [
    "BenchmarkDeclaration",
    "BenchmarkDeclarationValidationError",
    "BenchmarkManifestDocument",
    "BenchmarkManifest",
    "BenchmarkManifestValidationError",
]

_oracle_acceptance_id = ProtocolIdentifier.parse("core.finite-answer-accepted-event@0.1.0")
_prediction_interface_id = ProtocolIdentifier.parse(
    "core.finite-probability-measure-prediction@0.1.0"
)
_score_functional_id = ProtocolIdentifier.parse("core.negative-log-accepted-mass@0.1.0")
_evidence_bundle_id = ProtocolIdentifier.parse("core.finite-answer-scoring-bundle@0.1.0")

_benchmark_declaration_record = RecordSpec(
    fields={
        "id": required("identifier"),
        "answer_space_id": required("identifier"),
        "oracle_acceptance_id": optional("identifier"),
        "prediction_interface_id": optional("identifier"),
        "score_functional_id": optional("identifier"),
        "evidence_bundle_id": optional("identifier"),
    }
)
_benchmark_manifest_record = RecordSpec(
    fields={
        "id": required("identifier"),
        "name": required("name"),
        "answer_space_id": optional("identifier"),
        "declaration": optional("record"),
    }
)


class BenchmarkDeclarationValidationError(ValueError):
    """Raised when a benchmark declaration is invalid."""


class BenchmarkManifestValidationError(ValueError):
    """Raised when a benchmark manifest is invalid."""


@dataclass(frozen=True, slots=True)
class BenchmarkDeclaration:
    """A finite-answer benchmark scoring declaration."""

    id: ProtocolIdentifier
    answer_space_id: ProtocolIdentifier
    oracle_acceptance_id: ProtocolIdentifier = _oracle_acceptance_id
    prediction_interface_id: ProtocolIdentifier = _prediction_interface_id
    score_functional_id: ProtocolIdentifier = _score_functional_id
    evidence_bundle_id: ProtocolIdentifier = _evidence_bundle_id

    def __post_init__(self) -> None:
        try:
            require_unreleased_identifier(self.id)
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
        _require_identifier(
            field="evidence_bundle_id",
            actual=self.evidence_bundle_id,
            expected=_evidence_bundle_id,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BenchmarkDeclaration:
        try:
            validated = validate_record(record, _benchmark_declaration_record)
        except RecordValidationError as error:
            raise BenchmarkDeclarationValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            answer_space_id=_as_identifier(
                validated["answer_space_id"],
                field="answer_space_id",
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
            evidence_bundle_id=_as_identifier_or_default(
                validated.get("evidence_bundle_id"),
                field="evidence_bundle_id",
                default=_evidence_bundle_id,
            ),
        )

    def validate_bundle(self, bundle: FiniteAnswerScoringBundle) -> None:
        if bundle.answer_space.id != self.answer_space_id:
            raise BenchmarkDeclarationValidationError(
                f"bundle answer_space_id {bundle.answer_space.id} does not match "
                f"{self.answer_space_id}"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "answer_space_id": str(self.answer_space_id),
            "oracle_acceptance_id": str(self.oracle_acceptance_id),
            "prediction_interface_id": str(self.prediction_interface_id),
            "score_functional_id": str(self.score_functional_id),
            "evidence_bundle_id": str(self.evidence_bundle_id),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    """A benchmark manifest with identity metadata and one declaration."""

    id: ProtocolIdentifier
    name: ProtocolName
    declaration: BenchmarkDeclaration

    def __post_init__(self) -> None:
        try:
            require_unreleased_identifier(self.id)
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

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BenchmarkManifest:
        try:
            validated = validate_record(record, _benchmark_manifest_record)
        except ValueError as error:
            raise BenchmarkManifestValidationError(str(error)) from error
        try:
            declaration = _manifest_declaration(validated)
        except ValueError as error:
            raise BenchmarkManifestValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            name=_as_name(validated["name"], field="name"),
            declaration=declaration,
        )

    def validate_bundle(self, bundle: FiniteAnswerScoringBundle) -> None:
        try:
            self.declaration.validate_bundle(bundle)
        except BenchmarkDeclarationValidationError as error:
            raise BenchmarkManifestValidationError(str(error)) from error

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "name": str(self.name),
            "declaration": self.declaration.to_record(),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkManifestDocument:
    """A loaded benchmark manifest and the digest of its canonical record."""

    manifest: BenchmarkManifest
    digest: ContentDigest

    @classmethod
    def from_json_bytes(cls, data: bytes) -> BenchmarkManifestDocument:
        try:
            value = json.loads(data.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise BenchmarkManifestValidationError("manifest JSON must be UTF-8") from error
        except json.JSONDecodeError as error:
            raise BenchmarkManifestValidationError(f"invalid manifest JSON: {error.msg}") from error
        if not isinstance(value, Mapping):
            raise BenchmarkManifestValidationError("manifest JSON must be an object")

        record = cast(Mapping[str, object], value)
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


def _manifest_declaration(validated: Mapping[str, object]) -> BenchmarkDeclaration:
    declaration_value = validated.get("declaration")
    answer_space_id_value = validated.get("answer_space_id")
    if declaration_value is None and answer_space_id_value is None:
        raise BenchmarkManifestValidationError("answer_space_id: missing required field")

    declaration: BenchmarkDeclaration | None = None
    if declaration_value is not None:
        declaration = BenchmarkDeclaration.from_record(
            _manifest_mapping(declaration_value, field="declaration")
        )
    if answer_space_id_value is None:
        if declaration is None:
            raise BenchmarkManifestValidationError("declaration: expected record")
        return declaration

    answer_space_id = _as_identifier(answer_space_id_value, field="answer_space_id")
    if declaration is None:
        return BenchmarkDeclaration(
            id=_as_identifier(validated["id"], field="id"),
            answer_space_id=answer_space_id,
        )
    if declaration.answer_space_id != answer_space_id:
        raise BenchmarkManifestValidationError(
            f"answer_space_id {answer_space_id} does not match declaration "
            f"{declaration.answer_space_id}"
        )
    return declaration


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
