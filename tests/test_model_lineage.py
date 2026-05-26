from collections.abc import Callable

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.model_lineage import (
    ModelLineageDocument,
    ModelLineageGraph,
    ModelLineageValidationError,
)
from leibniz.model_operations import ModelOperation


def test_model_lineage_graph_parses_and_canonicalizes_one_parent() -> None:
    lineage = ModelLineageGraph.from_record(_lineage_record())
    operation = ModelOperation.from_record(_training_operation_record())

    assert lineage == ModelLineageGraph(
        id=ProtocolIdentifier.parse("model-lineages.boolean@0.1.0"),
        artifacts=(_architecture_reference(), _dataset_reference(), _model_reference()),
        operations=(operation,),
    )
    assert lineage.to_record() == {
        "id": "model-lineages.boolean@0.1.0",
        "artifacts": [
            _architecture_reference().to_record(),
            _dataset_reference().to_record(),
            _model_reference().to_record(),
        ],
        "operations": [operation.to_record()],
    }
    assert lineage.digest == ContentDigest.from_value(lineage.to_record())


def test_model_lineage_document_loads_bytes_with_digest() -> None:
    record = _lineage_record()

    document = ModelLineageDocument.from_bytes(canonical_document_bytes(record))

    assert document.lineage == ModelLineageGraph.from_record(record)
    assert document.digest == ContentDigest.from_value(document.lineage.to_record())


def test_model_lineage_accepts_multi_operation_dag() -> None:
    trained = _model_reference()
    compressed = _compressed_model_reference()
    training = ModelOperation.from_record(_training_operation_record(output=trained))
    compression = ModelOperation.from_record(
        _compression_operation_record(source=trained, output=compressed)
    )

    lineage = ModelLineageGraph.from_record(
        {
            "id": "model-lineages.boolean@0.1.0",
            "artifacts": [
                compressed.to_record(),
                _dataset_reference().to_record(),
                trained.to_record(),
                _architecture_reference().to_record(),
            ],
            "operations": [compression.to_record(), training.to_record()],
        }
    )

    assert lineage.artifacts == (
        _architecture_reference(),
        _dataset_reference(),
        compressed,
        trained,
    )
    assert lineage.operations == tuple(
        sorted((training, compression), key=lambda item: str(item.id))
    )


def test_model_lineage_rejects_dangling_operation_references() -> None:
    record = _lineage_record()
    record["artifacts"] = [
        _architecture_reference().to_record(),
        _model_reference().to_record(),
    ]

    assert str(capture_lineage_error(lambda: ModelLineageGraph.from_record(record))) == (
        f"operation input {ModelOperation.from_record(_training_operation_record()).id}.dataset "
        "does not resolve to a declared artifact"
    )

    record = _lineage_record()
    record["artifacts"] = [
        _architecture_reference().to_record(),
        _dataset_reference().to_record(),
    ]

    assert str(capture_lineage_error(lambda: ModelLineageGraph.from_record(record))) == (
        f"operation output {ModelOperation.from_record(_training_operation_record()).id}.model "
        "does not resolve to a declared artifact"
    )


def test_model_lineage_rejects_cycles() -> None:
    first_model = _model_reference()
    second_model = _compressed_model_reference()
    first = ModelOperation.from_record(
        _compression_operation_record(source=first_model, output=second_model)
    )
    second = ModelOperation.from_record(
        _compression_operation_record(source=second_model, output=first_model)
    )

    assert str(
        capture_lineage_error(
            lambda: ModelLineageGraph.from_record(
                {
                    "id": "model-lineages.boolean@0.1.0",
                    "artifacts": [first_model.to_record(), second_model.to_record()],
                    "operations": [first.to_record(), second.to_record()],
                }
            )
        )
    ) == "lineage graph must be acyclic"


def test_model_lineage_rejects_duplicate_nodes_and_operations() -> None:
    record = _lineage_record()
    record["artifacts"] = [
        _architecture_reference().to_record(),
        _architecture_reference().to_record(),
        _dataset_reference().to_record(),
        _model_reference().to_record(),
    ]
    assert str(capture_lineage_error(lambda: ModelLineageGraph.from_record(record))).startswith(
        "duplicate artifact reference: sha256:"
    )

    operation = ModelOperation.from_record(_training_operation_record())
    record = _lineage_record()
    record["operations"] = [operation.to_record(), operation.to_record()]
    assert str(capture_lineage_error(lambda: ModelLineageGraph.from_record(record))) == (
        f"duplicate operation id: {operation.id}"
    )


def test_model_lineage_rejects_malformed_and_execution_fields() -> None:
    record = _lineage_record()
    record["id"] = "core.boolean-lineage@0.1.0"
    assert str(capture_lineage_error(lambda: ModelLineageGraph.from_record(record))) == (
        "id must be a valid model lineage id"
    )

    record = _lineage_record()
    record["layout"] = "dagre"
    assert str(capture_lineage_error(lambda: ModelLineageGraph.from_record(record))) == (
        "layout: unknown field"
    )

    record = _lineage_record()
    record["operations"] = []
    assert str(capture_lineage_error(lambda: ModelLineageGraph.from_record(record))) == (
        "operations must contain at least one operation"
    )

    assert str(
        capture_lineage_error(lambda: ModelLineageDocument.from_bytes(b"[]"))
    ) == "model lineage document must contain an object"


def _lineage_record() -> dict[str, object]:
    return {
        "id": "model-lineages.boolean@0.1.0",
        "artifacts": [
            _model_reference().to_record(),
            _architecture_reference().to_record(),
            _dataset_reference().to_record(),
        ],
        "operations": [_training_operation_record()],
    }


def _training_operation_record(
    *,
    output: ArtifactReference | None = None,
) -> dict[str, object]:
    return ModelOperation.from_record(
        {
            "operator_id": "model-operators.train@0.1.0",
            "inputs": [
                {"role": "architecture", "artifact": _architecture_reference().to_record()},
                {"role": "dataset", "artifact": _dataset_reference().to_record()},
            ],
            "outputs": [
                {"role": "model", "artifact": (output or _model_reference()).to_record()}
            ],
        }
    ).to_record()


def _compression_operation_record(
    *,
    source: ArtifactReference,
    output: ArtifactReference,
) -> dict[str, object]:
    return ModelOperation.from_record(
        {
            "operator_id": "model-operators.compress@0.1.0",
            "inputs": [{"role": "source", "artifact": source.to_record()}],
            "outputs": [{"role": "model", "artifact": output.to_record()}],
        }
    ).to_record()


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


def _model_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "model-manifest",
            "protocol_id": "model-manifests.boolean-digits-pool@0.1.0",
        }
    )


def _compressed_model_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "model-manifest",
            "protocol_id": "model-manifests.boolean-digits-compressed@0.1.0",
        }
    )


def capture_lineage_error(action: Callable[[], object]) -> ModelLineageValidationError:
    try:
        action()
    except ModelLineageValidationError as error:
        return error
    raise AssertionError("expected ModelLineageValidationError")
