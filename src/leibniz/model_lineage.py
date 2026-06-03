"""Declarative model lineage documents."""

from __future__ import annotations

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
from leibniz.model_operations import ModelOperation
from leibniz.records import (
    FieldSpec,
    RecordExtractor,
    RecordSpec,
)

__all__ = [
    "ModelLineageDocument",
    "ModelLineageGraph",
    "ModelLineageValidationError",
]

_model_lineage_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "artifacts": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "operations": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
    }
)


class ModelLineageValidationError(ValueError):
    """Raised when a model lineage document is invalid."""


_record = RecordExtractor(ModelLineageValidationError)


@dataclass(frozen=True, slots=True)
class ModelLineageGraph:
    """A local directed acyclic graph over artifact references and operations."""

    id: ProtocolIdentifier
    artifacts: tuple[ArtifactReference, ...]
    operations: tuple[ModelOperation, ...]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise ModelLineageValidationError(str(error)) from error
        if not str(self.id.name).startswith("model-lineages."):
            raise ModelLineageValidationError("id must be a valid model lineage id")
        if not self.artifacts:
            raise ModelLineageValidationError("artifacts must contain at least one artifact")
        if not self.operations:
            raise ModelLineageValidationError("operations must contain at least one operation")

        artifact_duplicate = first_duplicate_reference(self.artifacts)
        if artifact_duplicate is not None:
            raise ModelLineageValidationError(f"duplicate artifact reference: {artifact_duplicate}")
        operation_duplicate = first_duplicate(tuple(operation.id for operation in self.operations))
        if operation_duplicate is not None:
            raise ModelLineageValidationError(f"duplicate operation id: {operation_duplicate}")

        object.__setattr__(
            self,
            "artifacts",
            tuple(sorted(self.artifacts, key=reference_sort_key)),
        )
        object.__setattr__(
            self,
            "operations",
            tuple(sorted(self.operations, key=lambda operation: str(operation.id))),
        )
        self._validate_operation_references()
        self._validate_acyclic()

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelLineageGraph:
        try:
            validated = _model_lineage_record.validate(record)
            artifacts = tuple(
                ArtifactReference.from_record(_record.mapping(item, "artifacts"))
                for item in _record.sequence(validated["artifacts"], "artifacts")
            )
            operations = tuple(
                ModelOperation.from_record(_record.mapping(item, "operations"))
                for item in _record.sequence(validated["operations"], "operations")
            )
        except ValueError as error:
            raise ModelLineageValidationError(str(error)) from error
        return cls(
            id=_record.identifier(validated["id"], "id"),
            artifacts=artifacts,
            operations=operations,
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "artifacts": [artifact.to_record() for artifact in self.artifacts],
            "operations": [operation.to_record() for operation in self.operations],
        }

    def _validate_operation_references(self) -> None:
        artifact_keys = {_reference_key(artifact) for artifact in self.artifacts}
        for operation in self.operations:
            for input_artifact in operation.inputs:
                key = _reference_key(input_artifact.artifact)
                if key not in artifact_keys:
                    raise ModelLineageValidationError(
                        f"operation input {operation.id}.{input_artifact.role} "
                        "does not resolve to a declared artifact"
                    )
            for output_artifact in operation.outputs:
                key = _reference_key(output_artifact.artifact)
                if key not in artifact_keys:
                    raise ModelLineageValidationError(
                        f"operation output {operation.id}.{output_artifact.role} "
                        "does not resolve to a declared artifact"
                    )

    def _validate_acyclic(self) -> None:
        edges: dict[str, set[str]] = {
            _reference_key(artifact): set() for artifact in self.artifacts
        }
        for operation in self.operations:
            input_keys = tuple(_reference_key(item.artifact) for item in operation.inputs)
            output_keys = tuple(_reference_key(item.artifact) for item in operation.outputs)
            for input_key in input_keys:
                edges[input_key].update(output_keys)

        visiting: set[str] = set()
        visited: set[str] = set()
        for key in edges:
            if _has_cycle(key, edges=edges, visiting=visiting, visited=visited):
                raise ModelLineageValidationError("lineage graph must be acyclic")


@dataclass(frozen=True, slots=True)
class ModelLineageDocument:
    """A loaded model lineage graph and its canonical digest."""

    lineage: ModelLineageGraph
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> ModelLineageDocument:
        try:
            record = load_object_document(data, description="model lineage document")
        except ContentEncodingError as error:
            raise ModelLineageValidationError(str(error)) from error
        lineage = ModelLineageGraph.from_record(record)
        return cls(lineage=lineage, digest=lineage.digest)


def _has_cycle(
    key: str,
    *,
    edges: Mapping[str, set[str]],
    visiting: set[str],
    visited: set[str],
) -> bool:
    if key in visited:
        return False
    if key in visiting:
        return True
    visiting.add(key)
    for next_key in edges[key]:
        if _has_cycle(next_key, edges=edges, visiting=visiting, visited=visited):
            return True
    visiting.remove(key)
    visited.add(key)
    return False


def _reference_key(reference: ArtifactReference) -> str:
    return str(ContentDigest.from_value(reference.to_record()))
