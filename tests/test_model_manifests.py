from collections.abc import Callable
from pathlib import Path

from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.artifacts import ArtifactReference, reference_for_record
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.model_interfaces import ModelInterface
from leibniz.model_manifests import (
    ModelArtifactManifest,
    ModelArtifactManifestDocument,
    ModelArtifactManifestValidationError,
    ModelExecutionFamily,
)
from leibniz.outcomes import OutcomeSpace

_fixtures_root = Path(__file__).parent / "fixtures"


def test_model_artifact_manifest_parses_and_canonicalizes() -> None:
    manifest = ModelArtifactManifest.from_record(
        _model_manifest_record(),
        architecture_manifest=_architecture_manifest(),
        model_interface=_model_interface(),
    )

    assert manifest == ModelArtifactManifest(
        id=ProtocolIdentifier.parse("model-manifests.boolean-digits-pool@0.1.0"),
        architecture=_architecture_reference(),
        interface=_interface_reference(),
        execution_family=ModelExecutionFamily.reference_runner_pytorch_sequential(),
        model_artifacts=(_checkpoint_reference(),),
        training_provenance=(_training_reference(),),
    )
    assert manifest.to_record() == _model_manifest_record()
    assert manifest.digest == ContentDigest.from_value(manifest.to_record())


def test_model_artifact_manifest_document_loads_bytes_with_digest() -> None:
    record = _model_manifest_record()

    document = ModelArtifactManifestDocument.from_bytes(
        canonical_document_bytes(record),
        architecture_manifest=_architecture_manifest(),
        model_interface=_model_interface(),
    )

    assert document.manifest == ModelArtifactManifest.from_record(
        record,
        architecture_manifest=_architecture_manifest(),
        model_interface=_model_interface(),
    )
    assert document.digest == ContentDigest.from_value(document.manifest.to_record())


def test_model_artifact_manifest_sorts_artifacts_and_provenance() -> None:
    checkpoint = _checkpoint_reference()
    sidecar = ArtifactReference.from_record(
        {
            "kind": "model-sidecar",
            "content_digest": str(ContentDigest.from_value({"metadata": True})),
        }
    )
    first_training = _training_reference()
    second_training = ArtifactReference.from_record(
        {
            "kind": "training-provenance",
            "content_digest": str(ContentDigest.from_value({"run": 2})),
        }
    )
    record = _model_manifest_record()
    record["model_artifacts"] = [sidecar.to_record(), checkpoint.to_record()]
    record["training_provenance"] = [second_training.to_record(), first_training.to_record()]

    manifest = ModelArtifactManifest.from_record(record)

    assert manifest.model_artifacts == (checkpoint, sidecar)
    assert manifest.training_provenance == (first_training, second_training)


def test_model_artifact_manifest_validates_embedded_public_sources() -> None:
    manifest = ModelArtifactManifest.from_record(_model_manifest_record())

    manifest.validate_architecture(_architecture_manifest())
    manifest.validate_interface(_model_interface())

    altered_architecture = ArchitectureManifest.from_record(
        {
            "input_shape": [1, 16, 16],
            "output_shape": [10],
            "layers": [{"kind": "dense", "parameters": {"out": 10}}],
        }
    )
    assert str(
        capture_model_manifest_error(lambda: manifest.validate_architecture(altered_architecture))
    ) == "architecture reference does not match architecture manifest"

    altered_interface = ModelInterface.from_outcome_space(
        id=ProtocolIdentifier.parse("model-interfaces.other@0.1.0"),
        outcome_space=_outcome_space(),
    )
    assert str(
        capture_model_manifest_error(lambda: manifest.validate_interface(altered_interface))
    ) == "interface reference does not match model interface"


def test_model_artifact_manifest_rejects_missing_and_malformed_references() -> None:
    record = _model_manifest_record()
    del record["architecture"]
    assert str(
        capture_model_manifest_error(lambda: ModelArtifactManifest.from_record(record))
    ) == "architecture: missing required field"

    record = _model_manifest_record()
    record["architecture"] = _interface_reference().to_record()
    assert str(
        capture_model_manifest_error(lambda: ModelArtifactManifest.from_record(record))
    ) == "architecture reference must have kind architecture-manifest"

    record = _model_manifest_record()
    record["interface"] = _architecture_reference().to_record()
    assert str(
        capture_model_manifest_error(lambda: ModelArtifactManifest.from_record(record))
    ) == "interface reference must have kind model-interface"

    record = _model_manifest_record()
    del record["execution_family"]
    assert str(
        capture_model_manifest_error(lambda: ModelArtifactManifest.from_record(record))
    ) == "execution_family: missing required field"

    record = _model_manifest_record()
    record["execution_family"] = {
        "kind": "reference-runner-pytorch-sequential",
        "runtime": "onnx",
        "architecture_family": "sequential-architecture-components",
    }
    assert str(
        capture_model_manifest_error(lambda: ModelArtifactManifest.from_record(record))
    ) == "reference-runner-pytorch-sequential requires runtime pytorch"

    record = _model_manifest_record()
    record["model_artifacts"] = []
    assert str(
        capture_model_manifest_error(lambda: ModelArtifactManifest.from_record(record))
    ) == "model_artifacts must contain at least one artifact reference"


def test_model_artifact_manifest_rejects_local_or_duplicate_artifacts() -> None:
    record = _model_manifest_record()
    record["model_artifacts"] = [
        {
            "kind": "model-checkpoint",
            "external_uri": ".leibniz/checkpoints/model.pt",
        }
    ]
    assert str(
        capture_model_manifest_error(lambda: ModelArtifactManifest.from_record(record))
    ) == "external_uri must not be a local path"

    record = _model_manifest_record()
    duplicate = _checkpoint_reference().to_record()
    record["model_artifacts"] = [duplicate, duplicate]
    assert str(
        capture_model_manifest_error(lambda: ModelArtifactManifest.from_record(record))
    ).startswith("duplicate model artifact reference: sha256:")


def test_model_artifact_manifest_rejects_execution_fields_and_invalid_documents() -> None:
    record = _model_manifest_record()
    record["builder"] = "torch.load"
    assert str(
        capture_model_manifest_error(lambda: ModelArtifactManifest.from_record(record))
    ) == "builder: unknown field"

    record = _model_manifest_record()
    record["prediction_adapter"] = "softmax"
    assert str(
        capture_model_manifest_error(lambda: ModelArtifactManifest.from_record(record))
    ) == "prediction_adapter: unknown field"

    assert str(
        capture_model_manifest_error(lambda: ModelArtifactManifestDocument.from_bytes(b"[]"))
    ) == "model artifact manifest document must contain an object"


def _model_manifest_record() -> dict[str, object]:
    return {
        "id": "model-manifests.boolean-digits-pool@0.1.0",
        "architecture": _architecture_reference().to_record(),
        "interface": _interface_reference().to_record(),
        "execution_family": ModelExecutionFamily.reference_runner_pytorch_sequential().to_record(),
        "model_artifacts": [_checkpoint_reference().to_record()],
        "training_provenance": [_training_reference().to_record()],
    }


def _architecture_reference() -> ArtifactReference:
    return reference_for_record(
        kind="architecture-manifest",
        record=_architecture_manifest().to_record(),
    )


def _interface_reference() -> ArtifactReference:
    return reference_for_record(
        kind="model-interface",
        record=_model_interface().to_record(),
    )


def _checkpoint_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "model-checkpoint",
            "content_digest": str(ContentDigest.from_value({"weights": [1, 2, 3]})),
        }
    )


def _training_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "training-provenance",
            "record_digest": str(ContentDigest.from_value({"optimizer": "declared"})),
        }
    )


def _architecture_manifest() -> ArchitectureManifest:
    return ArchitectureManifestDocument.from_bytes(
        (_fixtures_root / "architecture" / "digits_pool" / "manifest.json").read_bytes()
    ).manifest


def _model_interface() -> ModelInterface:
    return ModelInterface.from_outcome_space(
        id=ProtocolIdentifier.parse("model-interfaces.boolean@0.1.0"),
        outcome_space=_outcome_space(),
    )


def _outcome_space() -> OutcomeSpace:
    return OutcomeSpace.from_record(
        {
            "id": "core.boolean-outcome@0.1.0",
            "outcomes": [{"id": "yes"}, {"id": "no"}],
        }
    )


def capture_model_manifest_error(
    action: Callable[[], object],
) -> ModelArtifactManifestValidationError:
    try:
        action()
    except ModelArtifactManifestValidationError as error:
        return error
    raise AssertionError("expected ModelArtifactManifestValidationError")
