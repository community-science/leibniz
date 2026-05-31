from collections.abc import Callable

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.model_operations import (
    ModelOperation,
    ModelOperationArtifact,
    ModelOperationDocument,
    ModelOperationValidationError,
)


def test_model_operation_derives_content_addressed_id() -> None:
    operation = ModelOperation.from_record(_operation_record(include_id=False))

    assert str(operation.id).startswith("model-operations.sha-")
    assert operation.id.version.is_unreleased
    assert operation == ModelOperation(
        id=operation.derived_id(),
        operator_id=ProtocolIdentifier.parse("model-operators.train@0.1.0"),
        inputs=(ModelOperationArtifact(role="architecture", artifact=_architecture_reference()),),
        outputs=(ModelOperationArtifact(role="model", artifact=_model_manifest_reference()),),
        reports=(_resource_report_reference(),),
        observed_at="2026-05-25T12:00:00Z",
    )
    assert operation.to_record() == {
        "id": str(operation.id),
        **_operation_content_record(),
    }
    assert operation.digest == ContentDigest.from_value(operation.to_record())


def test_model_operation_accepts_matching_explicit_id() -> None:
    operation = ModelOperation.from_record(_operation_record(include_id=False))
    record = _operation_record(include_id=False)
    record["id"] = str(operation.id)

    assert ModelOperation.from_record(record) == operation


def test_model_operation_document_loads_bytes_with_digest() -> None:
    operation = ModelOperation.from_record(_operation_record(include_id=False))

    document = ModelOperationDocument.from_bytes(canonical_document_bytes(operation.to_record()))

    assert document.operation == operation
    assert document.digest == ContentDigest.from_value(operation.to_record())


def test_model_operation_sorts_inputs_outputs_and_reports() -> None:
    architecture = ModelOperationArtifact(role="architecture", artifact=_architecture_reference())
    dataset = ModelOperationArtifact(role="dataset", artifact=_dataset_reference())
    model = ModelOperationArtifact(role="model", artifact=_model_manifest_reference())
    sidecar = ModelOperationArtifact(role="sidecar", artifact=_sidecar_reference())
    first_report = _resource_report_reference()
    second_report = ArtifactReference.from_record(
        {
            "kind": "resource-report",
            "content_digest": str(ContentDigest.from_value({"report": 2})),
        }
    )
    record = _operation_record(include_id=False)
    record["inputs"] = [dataset.to_record(), architecture.to_record()]
    record["outputs"] = [sidecar.to_record(), model.to_record()]
    record["reports"] = [second_report.to_record(), first_report.to_record()]

    operation = ModelOperation.from_record(record)

    assert operation.inputs == (architecture, dataset)
    assert operation.outputs == (model, sidecar)
    assert operation.reports == (second_report, first_report)


def test_model_operation_rejects_content_id_mismatch() -> None:
    record = _operation_record(include_id=False)
    record["id"] = (
        "model-operations.sha-"
        "0000000000000000000000000000000000000000000000000000000000000000@0.1.0"
    )

    assert str(capture_operation_error(lambda: ModelOperation.from_record(record))) == (
        "id must be derived from operation content"
    )


def test_model_operation_rejects_missing_inputs_outputs_and_duplicate_roles() -> None:
    record = _operation_record(include_id=False)
    record["inputs"] = []
    assert str(capture_operation_error(lambda: ModelOperation.from_record(record))) == (
        "inputs must contain at least one artifact"
    )

    record = _operation_record(include_id=False)
    record["outputs"] = []
    assert str(capture_operation_error(lambda: ModelOperation.from_record(record))) == (
        "outputs must contain at least one artifact"
    )

    record = _operation_record(include_id=False)
    record["inputs"] = [
        {"role": "source", "artifact": _architecture_reference().to_record()},
        {"role": "source", "artifact": _dataset_reference().to_record()},
    ]
    assert str(capture_operation_error(lambda: ModelOperation.from_record(record))) == (
        "duplicate input role: source"
    )


def test_model_operation_rejects_malformed_and_local_artifact_refs() -> None:
    record = _operation_record(include_id=False)
    record["operator_id"] = "not an id"
    assert str(capture_operation_error(lambda: ModelOperation.from_record(record))) == (
        "operator_id: invalid protocol identifier: 'not an id'"
    )

    record = _operation_record(include_id=False)
    record["inputs"] = [
        {
            "role": "Architecture",
            "artifact": _architecture_reference().to_record(),
        }
    ]
    assert str(capture_operation_error(lambda: ModelOperation.from_record(record))) == (
        "artifact role must be a stable lowercase name"
    )

    record = _operation_record(include_id=False)
    record["outputs"] = [
        {
            "role": "model",
            "artifact": {"kind": "model-checkpoint", "external_uri": "./results/model.pt"},
        }
    ]
    assert str(capture_operation_error(lambda: ModelOperation.from_record(record))) == (
        "external_uri must not be a local path"
    )


def test_model_operation_rejects_execution_fields_and_invalid_documents() -> None:
    record = _operation_record(include_id=False)
    record["command"] = "train"
    assert str(capture_operation_error(lambda: ModelOperation.from_record(record))) == (
        "command: unknown field"
    )

    record = _operation_record(include_id=False)
    record["scheduler"] = "local"
    assert str(capture_operation_error(lambda: ModelOperation.from_record(record))) == (
        "scheduler: unknown field"
    )

    record = _operation_record(include_id=False)
    record["observed_at"] = ""
    assert str(capture_operation_error(lambda: ModelOperation.from_record(record))) == (
        "observed_at must be nonempty"
    )

    assert str(
        capture_operation_error(lambda: ModelOperationDocument.from_bytes(b"[]"))
    ) == "model operation document must contain an object"


def _operation_record(*, include_id: bool) -> dict[str, object]:
    record = _operation_content_record()
    if include_id:
        record["id"] = str(ModelOperation.from_record(record).id)
    return record


def _operation_content_record() -> dict[str, object]:
    return {
        "operator_id": "model-operators.train@0.1.0",
        "inputs": [
            {
                "role": "architecture",
                "artifact": _architecture_reference().to_record(),
            }
        ],
        "outputs": [
            {
                "role": "model",
                "artifact": _model_manifest_reference().to_record(),
            }
        ],
        "reports": [_resource_report_reference().to_record()],
        "observed_at": "2026-05-25T12:00:00Z",
    }


def _architecture_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "architecture-manifest",
            "protocol_id": (
                "architecture.sha-"
                "d695a59610f59ce2b61a20b7114b42da8692ffd9a55e4093431e3c00a932e693@0.1.0"
            ),
        }
    )


def _dataset_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "measurement-dataset",
            "content_digest": str(ContentDigest.from_value({"measurements": []})),
        }
    )


def _model_manifest_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "model-manifest",
            "protocol_id": "model-manifests.boolean-digits-pool@0.1.0",
        }
    )


def _sidecar_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "model-sidecar",
            "content_digest": str(ContentDigest.from_value({"sidecar": True})),
        }
    )


def _resource_report_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "resource-report",
            "content_digest": str(ContentDigest.from_value({"report": 1})),
        }
    )


def capture_operation_error(action: Callable[[], object]) -> ModelOperationValidationError:
    try:
        action()
    except ModelOperationValidationError as error:
        return error
    raise AssertionError("expected ModelOperationValidationError")
