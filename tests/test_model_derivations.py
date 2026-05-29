from collections.abc import Callable
from pathlib import Path

from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.artifacts import ArtifactReference, reference_for_record
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.model_derivations import (
    ModelDerivationCompatibilityReport,
    ModelDerivationCompatibilityReportDocument,
    ModelDerivationCompatibilityValidationError,
    ParameterMappingSummary,
)
from leibniz.model_interfaces import ModelInterface
from leibniz.model_manifests import ModelArtifactManifest, ModelExecutionFamily
from leibniz.outcomes import OutcomeSpace

_fixtures_root = Path(__file__).parent / "fixtures"


def test_model_derivation_compatibility_report_parses_and_canonicalizes() -> None:
    report = ModelDerivationCompatibilityReport.from_record(
        _compatibility_report_record(),
        source_model_manifest=_source_model_manifest(),
        target_architecture_manifest=_target_architecture_manifest(),
        target_model_interface=_target_model_interface(),
    )

    assert report == ModelDerivationCompatibilityReport(
        id=ProtocolIdentifier.parse("model-derivations.boolean-compression@0.1.0"),
        source_model=_source_model_reference(),
        target_architecture=_target_architecture_reference(),
        target_interface=_target_interface_reference(),
        operator_id=ProtocolIdentifier.parse("model-operators.compress@0.1.0"),
        status="compatible",
        parameter_mappings=(
            ParameterMappingSummary(
                name="dense-kernel",
                source="dense.weight",
                target="dense.weight",
                summary="target keeps the declared dense parameter block",
            ),
        ),
        preservation_laws=("outcome-space-preserved",),
        operation=_operation_reference(),
        resource_reports=(_resource_report_reference(),),
    )
    assert report.to_record() == _compatibility_report_record()
    assert report.digest == ContentDigest.from_value(report.to_record())


def test_model_derivation_compatibility_document_loads_bytes_with_digest() -> None:
    record = _compatibility_report_record()

    document = ModelDerivationCompatibilityReportDocument.from_bytes(
        canonical_document_bytes(record),
        source_model_manifest=_source_model_manifest(),
        target_architecture_manifest=_target_architecture_manifest(),
        target_model_interface=_target_model_interface(),
    )

    assert document.report == ModelDerivationCompatibilityReport.from_record(
        record,
        source_model_manifest=_source_model_manifest(),
        target_architecture_manifest=_target_architecture_manifest(),
        target_model_interface=_target_model_interface(),
    )
    assert document.digest == ContentDigest.from_value(document.report.to_record())


def test_model_derivation_compatibility_sorts_mappings_laws_and_reports() -> None:
    record = _compatibility_report_record()
    record["parameter_mappings"] = [
        {
            "name": "output-bias",
            "source": "dense.bias",
            "target": "dense.bias",
            "summary": "target keeps the declared output bias",
        },
        _mapping_record(),
    ]
    record["preservation_laws"] = ["resource-nonincreasing", "outcome-space-preserved"]
    second_resource = ArtifactReference.from_record(
        {
            "kind": "resource-report",
            "content_digest": str(ContentDigest.from_value({"resource": 2})),
        }
    )
    record["resource_reports"] = [
        second_resource.to_record(),
        _resource_report_reference().to_record(),
    ]

    report = ModelDerivationCompatibilityReport.from_record(record)

    assert [mapping.name for mapping in report.parameter_mappings] == [
        "dense-kernel",
        "output-bias",
    ]
    assert report.preservation_laws == ("outcome-space-preserved", "resource-nonincreasing")
    assert report.resource_reports == tuple(
        sorted(
            (second_resource, _resource_report_reference()),
            key=lambda reference: (
                reference.kind,
                str(reference.protocol_id) if reference.protocol_id is not None else "",
                str(reference.content_digest) if reference.content_digest is not None else "",
                str(reference.record_digest) if reference.record_digest is not None else "",
                reference.external_uri if reference.external_uri is not None else "",
            ),
        )
    )


def test_model_derivation_compatibility_validates_referenced_public_records() -> None:
    report = ModelDerivationCompatibilityReport.from_record(_compatibility_report_record())

    report.validate_source_model(_source_model_manifest())
    report.validate_target_architecture(_target_architecture_manifest())
    report.validate_target_interface(_target_model_interface())

    assert str(
        capture_derivation_error(
            lambda: report.validate_source_model(_alternate_source_model_manifest())
        )
    ) == "source_model reference does not match source model manifest"

    assert str(
        capture_derivation_error(
            lambda: report.validate_target_architecture(_alternate_architecture_manifest())
        )
    ) == "target_architecture reference does not match target architecture manifest"

    assert str(
        capture_derivation_error(lambda: report.validate_target_interface(_alternate_interface()))
    ) == "target_interface reference does not match target model interface"


def test_model_derivation_compatibility_rejects_malformed_mapping_and_status() -> None:
    record = _compatibility_report_record()
    record["status"] = "executed"
    assert str(
        capture_derivation_error(
            lambda: ModelDerivationCompatibilityReport.from_record(record)
        )
    ) == "unsupported status: executed"

    record = _compatibility_report_record()
    record["parameter_mappings"] = []
    assert str(
        capture_derivation_error(
            lambda: ModelDerivationCompatibilityReport.from_record(record)
        )
    ) == "parameter_mappings must contain at least one mapping summary"

    record = _compatibility_report_record()
    mappings = [_mapping_record(), _mapping_record()]
    record["parameter_mappings"] = mappings
    assert str(
        capture_derivation_error(
            lambda: ModelDerivationCompatibilityReport.from_record(record)
        )
    ) == "duplicate parameter mapping name: dense-kernel"

    record = _compatibility_report_record()
    mapping = _mapping_record()
    mapping["tensor_transform"] = "resample"
    record["parameter_mappings"] = [mapping]
    assert str(
        capture_derivation_error(
            lambda: ModelDerivationCompatibilityReport.from_record(record)
        )
    ) == "tensor_transform: unknown field"


def test_model_derivation_compatibility_rejects_bad_references_and_execution_fields() -> None:
    record = _compatibility_report_record()
    record["source_model"] = _target_architecture_reference().to_record()
    assert str(
        capture_derivation_error(
            lambda: ModelDerivationCompatibilityReport.from_record(record)
        )
    ) == "source_model reference must have kind model-manifest"

    record = _compatibility_report_record()
    record["operation"] = {"kind": "model-operation"}
    assert str(
        capture_derivation_error(
            lambda: ModelDerivationCompatibilityReport.from_record(record)
        )
    ) == "artifact reference must include at least one durable identity"

    record = _compatibility_report_record()
    record["operation"] = _resource_report_reference().to_record()
    assert str(
        capture_derivation_error(
            lambda: ModelDerivationCompatibilityReport.from_record(record)
        )
    ) == "operation reference must have kind model-operation"

    record = _compatibility_report_record()
    record["checkpoint_output"] = ".leibniz/derived/model.pt"
    assert str(
        capture_derivation_error(
            lambda: ModelDerivationCompatibilityReport.from_record(record)
        )
    ) == "checkpoint_output: unknown field"

    assert str(
        capture_derivation_error(
            lambda: ModelDerivationCompatibilityReportDocument.from_bytes(b"[]")
        )
    ) == "model derivation compatibility report document must contain an object"


def _compatibility_report_record() -> dict[str, object]:
    return {
        "id": "model-derivations.boolean-compression@0.1.0",
        "source_model": _source_model_reference().to_record(),
        "target_architecture": _target_architecture_reference().to_record(),
        "target_interface": _target_interface_reference().to_record(),
        "operator_id": "model-operators.compress@0.1.0",
        "status": "compatible",
        "parameter_mappings": [_mapping_record()],
        "preservation_laws": ["outcome-space-preserved"],
        "operation": _operation_reference().to_record(),
        "resource_reports": [_resource_report_reference().to_record()],
    }


def _mapping_record() -> dict[str, object]:
    return {
        "name": "dense-kernel",
        "source": "dense.weight",
        "target": "dense.weight",
        "summary": "target keeps the declared dense parameter block",
    }


def _source_model_reference() -> ArtifactReference:
    return reference_for_record(
        kind="model-manifest",
        record=_source_model_manifest().to_record(),
    )


def _target_architecture_reference() -> ArtifactReference:
    return reference_for_record(
        kind="architecture-manifest",
        record=_target_architecture_manifest().to_record(),
    )


def _target_interface_reference() -> ArtifactReference:
    return reference_for_record(
        kind="model-interface",
        record=_target_model_interface().to_record(),
    )


def _operation_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "model-operation",
            "content_digest": str(ContentDigest.from_value({"operation": "compress"})),
        }
    )


def _resource_report_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "resource-report",
            "content_digest": str(ContentDigest.from_value({"resource": 1})),
        }
    )


def _source_model_manifest() -> ModelArtifactManifest:
    return ModelArtifactManifest.from_record(
        {
            "id": "model-manifests.boolean-source@0.1.0",
            "architecture": _target_architecture_reference().to_record(),
            "interface": _target_interface_reference().to_record(),
            "execution_family": (
                ModelExecutionFamily.reference_runner_pytorch_sequential().to_record()
            ),
            "model_artifacts": [
                {
                    "kind": "model-checkpoint",
                    "content_digest": str(ContentDigest.from_value({"weights": [1, 2]})),
                }
            ],
        }
    )


def _alternate_source_model_manifest() -> ModelArtifactManifest:
    return ModelArtifactManifest.from_record(
        {
            "id": "model-manifests.boolean-other@0.1.0",
            "architecture": _target_architecture_reference().to_record(),
            "interface": _target_interface_reference().to_record(),
            "execution_family": (
                ModelExecutionFamily.reference_runner_pytorch_sequential().to_record()
            ),
            "model_artifacts": [
                {
                    "kind": "model-checkpoint",
                    "content_digest": str(ContentDigest.from_value({"weights": [3, 4]})),
                }
            ],
        }
    )


def _target_architecture_manifest() -> ArchitectureManifest:
    return ArchitectureManifestDocument.from_bytes(
        (_fixtures_root / "architecture" / "digits_pool" / "manifest.json").read_bytes()
    ).manifest


def _alternate_architecture_manifest() -> ArchitectureManifest:
    return ArchitectureManifest.from_record(
        {
            "input_shape": [1, 16, 16],
            "output_shape": [10],
            "layers": [{"kind": "dense", "parameters": {"out": 10}}],
        }
    )


def _target_model_interface() -> ModelInterface:
    return ModelInterface.from_outcome_space(
        id=ProtocolIdentifier.parse("model-interfaces.boolean@0.1.0"),
        outcome_space=_outcome_space(),
    )


def _alternate_interface() -> ModelInterface:
    return ModelInterface.from_outcome_space(
        id=ProtocolIdentifier.parse("model-interfaces.other@0.1.0"),
        outcome_space=_outcome_space(),
    )


def _outcome_space() -> OutcomeSpace:
    return OutcomeSpace.from_record(
        {
            "id": "core.boolean-outcome@0.1.0",
            "outcomes": [{"id": "yes"}, {"id": "no"}],
        }
    )


def capture_derivation_error(
    action: Callable[[], object],
) -> ModelDerivationCompatibilityValidationError:
    try:
        action()
    except ModelDerivationCompatibilityValidationError as error:
        return error
    raise AssertionError("expected ModelDerivationCompatibilityValidationError")
