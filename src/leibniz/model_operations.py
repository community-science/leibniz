"""Declarative model operation records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from leibniz.artifacts import (
    ArtifactReference,
    first_duplicate,
    first_duplicate_reference,
    reference_sort_key,
)
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec

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


_extract = RecordExtractor(error_type=ModelOperationValidationError)


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
            role=_extract.string(validated["role"], "role"),
            artifact=ArtifactReference.from_record(
                _extract.mapping(validated["artifact"], "artifact")
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
        duplicate_input = first_duplicate(tuple(item.role for item in self.inputs))
        if duplicate_input is not None:
            raise ModelOperationValidationError(f"duplicate input role: {duplicate_input}")
        duplicate_output = first_duplicate(tuple(item.role for item in self.outputs))
        if duplicate_output is not None:
            raise ModelOperationValidationError(f"duplicate output role: {duplicate_output}")
        duplicate_report = first_duplicate_reference(self.reports)
        if duplicate_report is not None:
            raise ModelOperationValidationError(f"duplicate report reference: {duplicate_report}")
        object.__setattr__(self, "inputs", tuple(sorted(self.inputs, key=lambda item: item.role)))
        object.__setattr__(self, "outputs", tuple(sorted(self.outputs, key=lambda item: item.role)))
        object.__setattr__(self, "reports", tuple(sorted(self.reports, key=reference_sort_key)))

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelOperation:
        try:
            validated = _model_operation_record.validate(record)
            inputs = tuple(
                ModelOperationArtifact.from_record(_extract.mapping(item, "inputs"))
                for item in _extract.sequence(validated["inputs"], "inputs")
            )
            outputs = tuple(
                ModelOperationArtifact.from_record(_extract.mapping(item, "outputs"))
                for item in _extract.sequence(validated["outputs"], "outputs")
            )
            reports = tuple(
                ArtifactReference.from_record(_extract.mapping(item, "reports"))
                for item in _extract.sequence(validated.get("reports", ()), "reports")
            )
        except ValueError as error:
            raise ModelOperationValidationError(str(error)) from error
        content_record = _operation_content_record(
            operator_id=_extract.identifier(validated["operator_id"], "operator_id"),
            inputs=inputs,
            outputs=outputs,
            reports=reports,
            observed_at=_extract.optional_string(validated.get("observed_at"), "observed_at"),
        )
        operation_id = validated.get("id", _operation_id(content_record))
        return cls(
            id=_extract.identifier(operation_id, "id"),
            operator_id=_extract.identifier(validated["operator_id"], "operator_id"),
            inputs=inputs,
            outputs=outputs,
            reports=reports,
            observed_at=_extract.optional_string(validated.get("observed_at"), "observed_at"),
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
            report.to_record() for report in sorted(reports, key=reference_sort_key)
        ]
    if observed_at is not None:
        record["observed_at"] = observed_at
    return record


def _operation_id(content_record: Mapping[str, object]) -> ProtocolIdentifier:
    digest = ContentDigest.from_value(content_record)
    return ProtocolIdentifier.parse(f"model-operations.sha-{digest.hex}@0.1.0")
