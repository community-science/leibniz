"""Declarative model derivation compatibility reports."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from leibniz.architectures import ArchitectureManifest
from leibniz.artifacts import (
    ArtifactReference,
    first_duplicate,
    first_duplicate_reference,
    reference_for_record,
    reference_sort_key,
)
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.model_interfaces import ModelInterface
from leibniz.model_manifests import ModelArtifactManifest
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec

__all__ = [
    "ModelDerivationCompatibilityReport",
    "ModelDerivationCompatibilityReportDocument",
    "ModelDerivationCompatibilityValidationError",
    "ParameterMappingSummary",
]

_Status: TypeAlias = Literal["compatible", "incompatible", "unknown"]
_name = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_parameter_mapping_record = RecordSpec(
    fields={
        "name": FieldSpec(kind="string"),
        "source": FieldSpec(kind="string"),
        "target": FieldSpec(kind="string"),
        "summary": FieldSpec(kind="string"),
    }
)
_compatibility_report_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "source_model": FieldSpec(kind="record"),
        "target_architecture": FieldSpec(kind="record"),
        "target_interface": FieldSpec(kind="record"),
        "operator_id": FieldSpec(kind="identifier"),
        "status": FieldSpec(kind="string"),
        "parameter_mappings": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "preservation_laws": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="string"),
        ),
        "operation": FieldSpec(kind="record", required=False),
        "resource_reports": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
            required=False,
        ),
    }
)


class ModelDerivationCompatibilityValidationError(ValueError):
    """Raised when a model derivation compatibility report is invalid."""


_extract = RecordExtractor(error_type=ModelDerivationCompatibilityValidationError)


@dataclass(frozen=True, slots=True)
class ParameterMappingSummary:
    """A non-executable summary of a source-to-target parameter mapping."""

    name: str
    source: str
    target: str
    summary: str

    def __post_init__(self) -> None:
        _validate_name(self.name, field="mapping name")
        if not self.source:
            raise ModelDerivationCompatibilityValidationError("mapping source must be nonempty")
        if not self.target:
            raise ModelDerivationCompatibilityValidationError("mapping target must be nonempty")
        if not self.summary:
            raise ModelDerivationCompatibilityValidationError("mapping summary must be nonempty")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ParameterMappingSummary:
        try:
            validated = _parameter_mapping_record.validate(record)
        except ValueError as error:
            raise ModelDerivationCompatibilityValidationError(str(error)) from error
        return cls(
            name=_extract.string(validated["name"], "name"),
            source=_extract.string(validated["source"], "source"),
            target=_extract.string(validated["target"], "target"),
            summary=_extract.string(validated["summary"], "summary"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "target": self.target,
            "summary": self.summary,
        }


@dataclass(frozen=True, slots=True)
class ModelDerivationCompatibilityReport:
    """A report-only model-to-model derivation compatibility claim."""

    id: ProtocolIdentifier
    source_model: ArtifactReference
    target_architecture: ArtifactReference
    target_interface: ArtifactReference
    operator_id: ProtocolIdentifier
    status: _Status
    parameter_mappings: tuple[ParameterMappingSummary, ...]
    preservation_laws: tuple[str, ...]
    operation: ArtifactReference | None = None
    resource_reports: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
            self.operator_id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise ModelDerivationCompatibilityValidationError(str(error)) from error
        if not str(self.id.name).startswith("model-derivations."):
            raise ModelDerivationCompatibilityValidationError(
                "id must be a valid model derivation compatibility report id"
            )
        if self.status not in {"compatible", "incompatible", "unknown"}:
            raise ModelDerivationCompatibilityValidationError(
                f"unsupported status: {self.status}"
            )
        if self.source_model.kind != "model-manifest":
            raise ModelDerivationCompatibilityValidationError(
                "source_model reference must have kind model-manifest"
            )
        if self.target_architecture.kind != "architecture-manifest":
            raise ModelDerivationCompatibilityValidationError(
                "target_architecture reference must have kind architecture-manifest"
            )
        if self.target_interface.kind != "model-interface":
            raise ModelDerivationCompatibilityValidationError(
                "target_interface reference must have kind model-interface"
            )
        if self.operation is not None and self.operation.kind != "model-operation":
            raise ModelDerivationCompatibilityValidationError(
                "operation reference must have kind model-operation"
            )
        if not self.parameter_mappings:
            raise ModelDerivationCompatibilityValidationError(
                "parameter_mappings must contain at least one mapping summary"
            )
        if not self.preservation_laws:
            raise ModelDerivationCompatibilityValidationError(
                "preservation_laws must contain at least one law name"
            )
        for law in self.preservation_laws:
            _validate_name(law, field="preservation law")
        duplicate_mapping = first_duplicate(
            tuple(mapping.name for mapping in self.parameter_mappings)
        )
        if duplicate_mapping is not None:
            raise ModelDerivationCompatibilityValidationError(
                f"duplicate parameter mapping name: {duplicate_mapping}"
            )
        duplicate_law = first_duplicate(self.preservation_laws)
        if duplicate_law is not None:
            raise ModelDerivationCompatibilityValidationError(
                f"duplicate preservation law: {duplicate_law}"
            )
        duplicate_resource = first_duplicate_reference(self.resource_reports)
        if duplicate_resource is not None:
            raise ModelDerivationCompatibilityValidationError(
                f"duplicate resource report reference: {duplicate_resource}"
            )
        object.__setattr__(
            self,
            "parameter_mappings",
            tuple(sorted(self.parameter_mappings, key=lambda mapping: mapping.name)),
        )
        object.__setattr__(self, "preservation_laws", tuple(sorted(self.preservation_laws)))
        object.__setattr__(
            self,
            "resource_reports",
            tuple(sorted(self.resource_reports, key=reference_sort_key)),
        )

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        source_model_manifest: ModelArtifactManifest | None = None,
        target_architecture_manifest: ArchitectureManifest | None = None,
        target_model_interface: ModelInterface | None = None,
    ) -> ModelDerivationCompatibilityReport:
        try:
            validated = _compatibility_report_record.validate(record)
            mappings = tuple(
                ParameterMappingSummary.from_record(
                    _extract.mapping(item, "parameter_mappings")
                )
                for item in _extract.sequence(
                    validated["parameter_mappings"],
                    "parameter_mappings",
                )
            )
            resource_reports = tuple(
                ArtifactReference.from_record(_extract.mapping(item, "resource_reports"))
                for item in _extract.sequence(
                    validated.get("resource_reports", ()),
                    "resource_reports",
                )
            )
            operation = (
                ArtifactReference.from_record(
                    _extract.mapping(validated["operation"], "operation")
                )
                if "operation" in validated
                else None
            )
        except ValueError as error:
            raise ModelDerivationCompatibilityValidationError(str(error)) from error
        report = cls(
            id=_extract.identifier(validated["id"], "id"),
            source_model=ArtifactReference.from_record(
                _extract.mapping(validated["source_model"], "source_model")
            ),
            target_architecture=ArtifactReference.from_record(
                _extract.mapping(validated["target_architecture"], "target_architecture")
            ),
            target_interface=ArtifactReference.from_record(
                _extract.mapping(validated["target_interface"], "target_interface")
            ),
            operator_id=_extract.identifier(validated["operator_id"], "operator_id"),
            status=cast(_Status, _extract.string(validated["status"], "status")),
            parameter_mappings=mappings,
            preservation_laws=tuple(
                _extract.string(item, "preservation_laws")
                for item in _extract.sequence(
                    validated["preservation_laws"],
                    "preservation_laws",
                )
            ),
            operation=operation,
            resource_reports=resource_reports,
        )
        if source_model_manifest is not None:
            report.validate_source_model(source_model_manifest)
        if target_architecture_manifest is not None:
            report.validate_target_architecture(target_architecture_manifest)
        if target_model_interface is not None:
            report.validate_target_interface(target_model_interface)
        return report

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_source_model(self, source_model_manifest: ModelArtifactManifest) -> None:
        if not self.source_model.matches_record(source_model_manifest.to_record()):
            raise ModelDerivationCompatibilityValidationError(
                "source_model reference does not match source model manifest"
            )

    def validate_target_architecture(
        self,
        target_architecture_manifest: ArchitectureManifest,
    ) -> None:
        target_reference = reference_for_record(
            kind="architecture-manifest",
            record=target_architecture_manifest.to_record(),
        )
        if self.target_architecture != target_reference:
            raise ModelDerivationCompatibilityValidationError(
                "target_architecture reference does not match target architecture manifest"
            )

    def validate_target_interface(self, target_model_interface: ModelInterface) -> None:
        target_reference = reference_for_record(
            kind="model-interface",
            record=target_model_interface.to_record(),
        )
        if self.target_interface != target_reference:
            raise ModelDerivationCompatibilityValidationError(
                "target_interface reference does not match target model interface"
            )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "source_model": self.source_model.to_record(),
            "target_architecture": self.target_architecture.to_record(),
            "target_interface": self.target_interface.to_record(),
            "operator_id": str(self.operator_id),
            "status": self.status,
            "parameter_mappings": [mapping.to_record() for mapping in self.parameter_mappings],
            "preservation_laws": list(self.preservation_laws),
        }
        if self.operation is not None:
            record["operation"] = self.operation.to_record()
        if self.resource_reports:
            record["resource_reports"] = [report.to_record() for report in self.resource_reports]
        return record


@dataclass(frozen=True, slots=True)
class ModelDerivationCompatibilityReportDocument:
    """A loaded model derivation compatibility report and canonical digest."""

    report: ModelDerivationCompatibilityReport
    digest: ContentDigest

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        source_model_manifest: ModelArtifactManifest | None = None,
        target_architecture_manifest: ArchitectureManifest | None = None,
        target_model_interface: ModelInterface | None = None,
    ) -> ModelDerivationCompatibilityReportDocument:
        try:
            record = load_object_document(
                data,
                description="model derivation compatibility report document",
            )
        except ContentEncodingError as error:
            raise ModelDerivationCompatibilityValidationError(str(error)) from error
        report = ModelDerivationCompatibilityReport.from_record(
            record,
            source_model_manifest=source_model_manifest,
            target_architecture_manifest=target_architecture_manifest,
            target_model_interface=target_model_interface,
        )
        return cls(report=report, digest=report.digest)


def _validate_name(value: str, *, field: str) -> None:
    if _name.fullmatch(value) is None:
        raise ModelDerivationCompatibilityValidationError(
            f"{field} must be a stable lowercase name"
        )
