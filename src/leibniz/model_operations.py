"""Declarative model operation records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "ModelOperation",
    "ModelOperationArtifact",
    "ModelOperationDocument",
    "ModelOperationValidationError",
]

_role = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_model_operation_artifact_record = RecordSpec(
    fields={
        "role": FieldSpec(kind="string"),
        "artifact": FieldSpec(kind="record"),
    }
)
_model_operation_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier", required=False),
        "operator_id": FieldSpec(kind="identifier"),
        "inputs": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "outputs": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "reports": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
            required=False,
        ),
        "observed_at": FieldSpec(kind="string", required=False),
    }
)


class ModelOperationValidationError(ValueError):
    """Raised when a model operation record is invalid."""


@dataclass(frozen=True, slots=True)
class ModelOperationArtifact:
    """One role-bound artifact reference in a model operation."""

    role: str
    artifact: ArtifactReference

    def __post_init__(self) -> None:
        if _role.fullmatch(self.role) is None:
            raise ModelOperationValidationError("artifact role must be a stable lowercase name")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelOperationArtifact:
        try:
            validated = _model_operation_artifact_record.validate(record)
        except ValueError as error:
            raise ModelOperationValidationError(str(error)) from error
        return cls(
            role=_as_string(validated["role"], field="role"),
            artifact=ArtifactReference.from_record(
                _as_mapping(validated["artifact"], field="artifact")
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "role": self.role,
            "artifact": self.artifact.to_record(),
        }


@dataclass(frozen=True, slots=True)
class ModelOperation:
    """An auditable operation over model artifact references."""

    id: ProtocolIdentifier
    operator_id: ProtocolIdentifier
    inputs: tuple[ModelOperationArtifact, ...]
    outputs: tuple[ModelOperationArtifact, ...]
    reports: tuple[ArtifactReference, ...] = ()
    observed_at: str | None = None

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
            self.operator_id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise ModelOperationValidationError(str(error)) from error
        if self.id != self.derived_id():
            raise ModelOperationValidationError("id must be derived from operation content")
        if not self.inputs:
            raise ModelOperationValidationError("inputs must contain at least one artifact")
        if not self.outputs:
            raise ModelOperationValidationError("outputs must contain at least one artifact")
        if self.observed_at == "":
            raise ModelOperationValidationError("observed_at must be nonempty")
        duplicate_input = _first_duplicate(tuple(item.role for item in self.inputs))
        if duplicate_input is not None:
            raise ModelOperationValidationError(f"duplicate input role: {duplicate_input}")
        duplicate_output = _first_duplicate(tuple(item.role for item in self.outputs))
        if duplicate_output is not None:
            raise ModelOperationValidationError(f"duplicate output role: {duplicate_output}")
        duplicate_report = _first_duplicate_reference(self.reports)
        if duplicate_report is not None:
            raise ModelOperationValidationError(f"duplicate report reference: {duplicate_report}")
        object.__setattr__(self, "inputs", tuple(sorted(self.inputs, key=lambda item: item.role)))
        object.__setattr__(self, "outputs", tuple(sorted(self.outputs, key=lambda item: item.role)))
        object.__setattr__(self, "reports", tuple(sorted(self.reports, key=_reference_sort_key)))

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelOperation:
        try:
            validated = _model_operation_record.validate(record)
            inputs = tuple(
                ModelOperationArtifact.from_record(_as_mapping(item, field="inputs"))
                for item in _as_sequence(validated["inputs"], field="inputs")
            )
            outputs = tuple(
                ModelOperationArtifact.from_record(_as_mapping(item, field="outputs"))
                for item in _as_sequence(validated["outputs"], field="outputs")
            )
            reports = tuple(
                ArtifactReference.from_record(_as_mapping(item, field="reports"))
                for item in _as_sequence(validated.get("reports", ()), field="reports")
            )
        except ValueError as error:
            raise ModelOperationValidationError(str(error)) from error
        content_record = _operation_content_record(
            operator_id=_as_identifier(validated["operator_id"], field="operator_id"),
            inputs=inputs,
            outputs=outputs,
            reports=reports,
            observed_at=_as_optional_string(validated.get("observed_at"), field="observed_at"),
        )
        operation_id = validated.get("id", _operation_id(content_record))
        return cls(
            id=_as_identifier(operation_id, field="id"),
            operator_id=_as_identifier(validated["operator_id"], field="operator_id"),
            inputs=inputs,
            outputs=outputs,
            reports=reports,
            observed_at=_as_optional_string(validated.get("observed_at"), field="observed_at"),
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def derived_id(self) -> ProtocolIdentifier:
        return _operation_id(self._content_record())

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            **self._content_record(),
        }

    def _content_record(self) -> dict[str, object]:
        return _operation_content_record(
            operator_id=self.operator_id,
            inputs=self.inputs,
            outputs=self.outputs,
            reports=self.reports,
            observed_at=self.observed_at,
        )


@dataclass(frozen=True, slots=True)
class ModelOperationDocument:
    """A loaded model operation record and its canonical digest."""

    operation: ModelOperation
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> ModelOperationDocument:
        try:
            record = load_object_document(data, description="model operation document")
        except ContentEncodingError as error:
            raise ModelOperationValidationError(str(error)) from error
        operation = ModelOperation.from_record(record)
        return cls(operation=operation, digest=operation.digest)


def _operation_content_record(
    *,
    operator_id: ProtocolIdentifier,
    inputs: tuple[ModelOperationArtifact, ...],
    outputs: tuple[ModelOperationArtifact, ...],
    reports: tuple[ArtifactReference, ...],
    observed_at: str | None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "operator_id": str(operator_id),
        "inputs": [item.to_record() for item in sorted(inputs, key=lambda item: item.role)],
        "outputs": [item.to_record() for item in sorted(outputs, key=lambda item: item.role)],
    }
    if reports:
        record["reports"] = [
            report.to_record() for report in sorted(reports, key=_reference_sort_key)
        ]
    if observed_at is not None:
        record["observed_at"] = observed_at
    return record


def _operation_id(content_record: Mapping[str, object]) -> ProtocolIdentifier:
    digest = ContentDigest.from_value(content_record)
    return ProtocolIdentifier.parse(f"model-operations.sha-{digest.hex}@0.1.0")


def _as_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ModelOperationValidationError(f"{field}: expected string")
    return value


def _as_optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _as_string(value, field=field)


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise ModelOperationValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ModelOperationValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ModelOperationValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _reference_sort_key(reference: ArtifactReference) -> tuple[str, str, str, str, str]:
    return (
        reference.kind,
        str(reference.protocol_id) if reference.protocol_id is not None else "",
        str(reference.content_digest) if reference.content_digest is not None else "",
        str(reference.record_digest) if reference.record_digest is not None else "",
        reference.external_uri if reference.external_uri is not None else "",
    )


def _first_duplicate(values: tuple[object, ...]) -> object | None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def _first_duplicate_reference(references: tuple[ArtifactReference, ...]) -> str | None:
    seen: set[str] = set()
    for reference in references:
        key = str(ContentDigest.from_value(reference.to_record()))
        if key in seen:
            return key
        seen.add(key)
    return None
