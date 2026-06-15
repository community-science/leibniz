"""Declared manifests for model artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from leibniz.artifacts import (
    ArtifactReference,
    first_duplicate_reference,
    reference_sort_key,
)
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.model_interfaces import ModelInterface
from leibniz.records import (
    FieldSpec,
    RecordExtractor,
    RecordSpec,
)

__all__ = [
    "ModelArtifactManifest",
    "ModelArtifactManifestDocument",
    "ModelArtifactManifestValidationError",
    "ModelExecutionFamily",
]

_model_execution_family_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="string"),
        "runtime": FieldSpec(kind="string"),
        "program_family": FieldSpec(kind="string"),
    }
)
_model_artifact_manifest_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "program": FieldSpec(kind="record"),
        "interface": FieldSpec(kind="record"),
        "execution_family": FieldSpec(kind="record"),
        "model_artifacts": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "training_provenance": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
            required=False,
        ),
    }
)


class ModelArtifactManifestValidationError(ValueError):
    """Raised when a model artifact manifest is invalid."""


_record = RecordExtractor(ModelArtifactManifestValidationError)


@dataclass(frozen=True, slots=True)
class ModelExecutionFamily:
    """Declared executable family for model artifacts, separate from model semantics."""

    kind: str
    runtime: str
    program_family: str

    def __post_init__(self) -> None:
        if not self.kind:
            raise ModelArtifactManifestValidationError("execution_family kind must be nonempty")
        if not self.runtime:
            raise ModelArtifactManifestValidationError("execution_family runtime must be nonempty")
        if not self.program_family:
            raise ModelArtifactManifestValidationError(
                "execution_family program_family must be nonempty"
            )
        if self.kind == "submitted-program-graph":
            if self.runtime != "pytorch":
                raise ModelArtifactManifestValidationError(
                    "submitted-program-graph requires runtime pytorch"
                )
            if self.program_family != "open-node-program-graph":
                raise ModelArtifactManifestValidationError(
                    "submitted-program-graph requires program_family open-node-program-graph"
                )

    @classmethod
    def submitted_program_graph(cls) -> ModelExecutionFamily:
        return cls(
            kind="submitted-program-graph",
            runtime="pytorch",
            program_family="open-node-program-graph",
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelExecutionFamily:
        try:
            validated = _model_execution_family_record.validate(record)
        except ValueError as error:
            raise ModelArtifactManifestValidationError(str(error)) from error
        return cls(
            kind=str(validated["kind"]),
            runtime=str(validated["runtime"]),
            program_family=str(validated["program_family"]),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "runtime": self.runtime,
            "program_family": self.program_family,
        }


@dataclass(frozen=True, slots=True)
class ModelArtifactManifest:
    """Durable metadata for a trained or submitted model artifact."""

    id: ProtocolIdentifier
    program: ArtifactReference
    interface: ArtifactReference
    execution_family: ModelExecutionFamily
    model_artifacts: tuple[ArtifactReference, ...]
    training_provenance: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise ModelArtifactManifestValidationError(str(error)) from error
        if not str(self.id.name).startswith("model-manifests."):
            raise ModelArtifactManifestValidationError("id must be a valid model manifest id")
        if self.program.kind != "program-graph":
            raise ModelArtifactManifestValidationError(
                "program reference must have kind program-graph"
            )
        if self.interface.kind != "model-interface":
            raise ModelArtifactManifestValidationError(
                "interface reference must have kind model-interface"
            )
        if not self.model_artifacts:
            raise ModelArtifactManifestValidationError(
                "model_artifacts must contain at least one artifact reference"
            )
        duplicate = first_duplicate_reference(self.model_artifacts)
        if duplicate is not None:
            raise ModelArtifactManifestValidationError(
                f"duplicate model artifact reference: {duplicate}"
            )
        duplicate_provenance = first_duplicate_reference(self.training_provenance)
        if duplicate_provenance is not None:
            raise ModelArtifactManifestValidationError(
                f"duplicate training provenance reference: {duplicate_provenance}"
            )
        object.__setattr__(
            self,
            "model_artifacts",
            tuple(sorted(self.model_artifacts, key=reference_sort_key)),
        )
        object.__setattr__(
            self,
            "training_provenance",
            tuple(sorted(self.training_provenance, key=reference_sort_key)),
        )

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        program_record: Mapping[str, object] | None = None,
        model_interface: ModelInterface | None = None,
    ) -> ModelArtifactManifest:
        try:
            validated = _model_artifact_manifest_record.validate(record)
            model_artifacts = tuple(
                ArtifactReference.from_record(_record.mapping(item, "model_artifacts"))
                for item in _record.sequence(validated["model_artifacts"], "model_artifacts")
            )
            training_provenance = tuple(
                ArtifactReference.from_record(_record.mapping(item, "training_provenance"))
                for item in _record.sequence(
                    validated.get("training_provenance", ()),
                    "training_provenance",
                )
            )
        except ValueError as error:
            raise ModelArtifactManifestValidationError(str(error)) from error
        manifest = cls(
            id=_record.identifier(validated["id"], "id"),
            program=ArtifactReference.from_record(
                _record.mapping(validated["program"], "program")
            ),
            interface=ArtifactReference.from_record(
                _record.mapping(validated["interface"], "interface")
            ),
            execution_family=ModelExecutionFamily.from_record(
                _record.mapping(validated["execution_family"], "execution_family")
            ),
            model_artifacts=model_artifacts,
            training_provenance=training_provenance,
        )
        if program_record is not None:
            manifest.validate_program(program_record)
        if model_interface is not None:
            manifest.validate_interface(model_interface)
        return manifest

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_program(self, program_record: Mapping[str, object]) -> None:
        if not self.program.matches_record(program_record):
            raise ModelArtifactManifestValidationError(
                "program reference does not match program graph"
            )

    def validate_interface(self, model_interface: ModelInterface) -> None:
        if not self.interface.matches_record(model_interface.to_record()):
            raise ModelArtifactManifestValidationError(
                "interface reference does not match model interface"
            )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "program": self.program.to_record(),
            "interface": self.interface.to_record(),
            "execution_family": self.execution_family.to_record(),
            "model_artifacts": [artifact.to_record() for artifact in self.model_artifacts],
        }
        if self.training_provenance:
            record["training_provenance"] = [
                reference.to_record() for reference in self.training_provenance
            ]
        return record


@dataclass(frozen=True, slots=True)
class ModelArtifactManifestDocument:
    """A loaded model artifact manifest and its canonical digest."""

    manifest: ModelArtifactManifest
    digest: ContentDigest

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        program_record: Mapping[str, object] | None = None,
        model_interface: ModelInterface | None = None,
    ) -> ModelArtifactManifestDocument:
        try:
            record = load_object_document(data, description="model artifact manifest document")
        except ContentEncodingError as error:
            raise ModelArtifactManifestValidationError(str(error)) from error
        manifest = ModelArtifactManifest.from_record(
            record,
            program_record=program_record,
            model_interface=model_interface,
        )
        return cls(manifest=manifest, digest=manifest.digest)
