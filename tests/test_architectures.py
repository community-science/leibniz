from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from leibniz.architectures import (
    ArchitectureComponent,
    ArchitectureGraph,
    ArchitectureGraphEdge,
    ArchitectureGraphNode,
    ArchitectureLayer,
    ArchitectureManifest,
    ArchitectureManifestDocument,
    ArchitectureManifestValidationError,
)
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.model_scale_contracts import ModelScaleContract

_fixtures_root = Path(__file__).parent / "fixtures"


def test_architecture_manifest_derives_content_addressed_id() -> None:
    manifest = ArchitectureManifest.from_record(_architecture_record())

    assert str(manifest.id).startswith("architecture.sha-")
    assert manifest.id.version.is_unreleased
    assert manifest == ArchitectureManifest(
        id=manifest.derived_id(),
        input_shape=(1, 32, 32),
        output_shape=(10,),
        layers=(
            ArchitectureComponent(
                kind="adaptive-pooling",
                parameters={"dimension": 2, "size": 2},
            ),
            ArchitectureComponent(kind="flatten", parameters={}),
            ArchitectureComponent(kind="dense", parameters={"out": 10}),
        ),
    )
    assert ArchitectureLayer is ArchitectureComponent
    assert manifest.components == manifest.layers
    assert manifest.graph == ArchitectureGraph.sequential(manifest.components)
    assert manifest.to_record() == {
        "id": str(manifest.id),
        **_expanded_architecture_body(),
    }
    assert manifest.digest == ContentDigest.from_value(manifest.to_record())


def test_architecture_manifest_accepts_matching_explicit_id() -> None:
    manifest = ArchitectureManifest.from_record(_architecture_record())
    record = _architecture_record()
    record["id"] = str(manifest.id)

    assert ArchitectureManifest.from_record(record) == manifest


def test_architecture_manifest_accepts_model_scale_contract() -> None:
    record = _architecture_record()
    contract = ModelScaleContract.variable_input_shape(
        (1, 32, 32),
        minimum=32,
        axis_symbol="W",
        scale_axis_indices=(2,),
    )
    record["model_scale_contract"] = contract.to_record()

    manifest = ArchitectureManifest.from_record(record)

    assert manifest.model_scale_contract == contract
    assert manifest.to_record()["model_scale_contract"] == contract.to_record()


def test_architecture_graph_lowers_sequential_layers_to_single_path() -> None:
    manifest = ArchitectureManifest.from_record(_architecture_record())
    graph = manifest.graph

    assert graph.nodes == (
        ArchitectureGraphNode(id="component-0", component=manifest.components[0]),
        ArchitectureGraphNode(id="component-1", component=manifest.components[1]),
        ArchitectureGraphNode(id="component-2", component=manifest.components[2]),
    )
    assert graph.edges == (
        ArchitectureGraphEdge(
            source_node_id="component-0",
            target_node_id="component-1",
        ),
        ArchitectureGraphEdge(
            source_node_id="component-1",
            target_node_id="component-2",
        ),
    )
    assert graph.input_node_ids == ("component-0",)
    assert graph.output_node_ids == ("component-2",)
    assert graph.to_record() == {
        "nodes": [
            {
                "id": "component-0",
                "component": {
                    "kind": "adaptive-pooling",
                    "parameters": {"dimension": 2, "size": 2},
                },
            },
            {
                "id": "component-1",
                "component": {"kind": "flatten", "parameters": {}},
            },
            {
                "id": "component-2",
                "component": {"kind": "dense", "parameters": {"out": 10}},
            },
        ],
        "edges": [
            {
                "source_node_id": "component-0",
                "target_node_id": "component-1",
                "kind": "data-flow",
            },
            {
                "source_node_id": "component-1",
                "target_node_id": "component-2",
                "kind": "data-flow",
            },
        ],
        "input_node_ids": ["component-0"],
        "output_node_ids": ["component-2"],
    }
    assert ArchitectureGraph.from_record(graph.to_record()) == graph


def test_architecture_graph_rejects_invalid_references_and_cycles() -> None:
    graph = ArchitectureManifest.from_record(_architecture_record()).graph
    record = graph.to_record()
    record["edges"] = [
        {
            "source_node_id": "component-0",
            "target_node_id": "missing",
            "kind": "data-flow",
        }
    ]
    assert str(capture_architecture_error(lambda: ArchitectureGraph.from_record(record))) == (
        "edge target_node_id 'missing' is not a graph node"
    )

    record = graph.to_record()
    record["input_node_ids"] = ["missing"]
    assert str(capture_architecture_error(lambda: ArchitectureGraph.from_record(record))) == (
        "input_node_ids contains unknown node id 'missing'"
    )

    record = graph.to_record()
    graph_edges = cast(list[dict[str, object]], graph.to_record()["edges"])
    record["edges"] = [
        *graph_edges,
        {
            "source_node_id": "component-2",
            "target_node_id": "component-0",
            "kind": "data-flow",
        },
    ]
    assert str(capture_architecture_error(lambda: ArchitectureGraph.from_record(record))) == (
        "architecture graph must be acyclic"
    )


def test_architecture_manifest_rejects_invalid_ids_shapes_and_layers() -> None:
    record = _architecture_record()
    record["id"] = "architecture.sha-wrong@0.1.0"
    assert str(capture_architecture_error(lambda: ArchitectureManifest.from_record(record))) == (
        "id must be derived from architecture content"
    )

    record = _architecture_record()
    record["input_shape"] = [1, 0, 32]
    assert str(capture_architecture_error(lambda: ArchitectureManifest.from_record(record))) == (
        "input_shape axes must be positive integers"
    )

    record = _architecture_record()
    record["layers"] = []
    assert str(capture_architecture_error(lambda: ArchitectureManifest.from_record(record))) == (
        "layers must contain at least one layer"
    )

    record = _architecture_record()
    contract = ModelScaleContract.variable_input_shape(
        (1, 32, 64),
        minimum=64,
        axis_symbol="W",
        scale_axis_indices=(2,),
    )
    record["model_scale_contract"] = contract.to_record()
    assert str(capture_architecture_error(lambda: ArchitectureManifest.from_record(record))) == (
        "model_scale_contract anchor_shape must match input_shape"
    )

    record = _architecture_record()
    layers = list(_layers())
    layers[0] = {"kind": ""}
    record["layers"] = layers
    assert str(capture_architecture_error(lambda: ArchitectureManifest.from_record(record))) == (
        "component kind must be nonempty"
    )

    record = _architecture_record()
    layers = list(_layers())
    layers[0] = {"kind": "dense", "weights_path": ".leibniz/checkpoints/model.pt"}
    record["layers"] = layers
    assert str(capture_architecture_error(lambda: ArchitectureManifest.from_record(record))) == (
        "weights_path: unknown field"
    )


def test_architecture_manifest_document_loads_fixture_with_digest() -> None:
    document = ArchitectureManifestDocument.from_bytes(
        (_fixtures_root / "architecture" / "digits_pool" / "manifest.json").read_bytes()
    )

    assert document.manifest.input_shape == (1, 32, 32)
    assert document.manifest.output_shape == (10,)
    assert tuple(layer.kind for layer in document.manifest.layers) == (
        "adaptive-pooling",
        "flatten",
        "dense",
    )
    assert document.manifest.to_record()["input_shape"] == [1, 32, 32]
    assert document.manifest.to_record()["output_shape"] == [10]
    assert document.digest == ContentDigest.from_value(document.manifest.to_record())


def test_architecture_manifest_document_rejects_invalid_bytes() -> None:
    assert str(
        capture_architecture_error(lambda: ArchitectureManifestDocument.from_bytes(b"[]"))
    ) == "architecture manifest document must contain an object"
    assert str(
        capture_architecture_error(
            lambda: ArchitectureManifestDocument.from_bytes(
                canonical_document_bytes({"input_shape": [1]})
            )
        )
    ) == "output_shape: missing required field; layers: missing required field"


def test_architecture_manifest_rejects_runtime_or_weight_specific_layer_data() -> None:
    record = _architecture_record()
    layers = list(_layers())
    layers[0] = {
        "kind": "dense",
        "parameters": {
            "module": object(),
        },
    }
    record["layers"] = layers

    assert str(capture_architecture_error(lambda: ArchitectureManifest.from_record(record))) == (
        "parameters.module: unsupported JSON value"
    )


def _architecture_record() -> dict[str, object]:
    return {
        "input_shape": [1, 32, 32],
        "output_shape": [10],
        "layers": list(_layers()),
    }


def _expanded_architecture_body() -> dict[str, object]:
    return {
        "input_shape": [1, 32, 32],
        "output_shape": [10],
        "layers": [
            {
                "kind": "adaptive-pooling",
                "parameters": {
                    "dimension": 2,
                    "size": 2,
                },
            },
            {
                "kind": "flatten",
                "parameters": {},
            },
            {
                "kind": "dense",
                "parameters": {
                    "out": 10,
                },
            },
        ],
    }


def _layers() -> tuple[Mapping[str, object], ...]:
    return (
        {
            "kind": "adaptive-pooling",
            "parameters": {
                "dimension": 2,
                "size": 2,
            },
        },
        {
            "kind": "flatten",
        },
        {
            "kind": "dense",
            "parameters": {
                "out": 10,
            },
        },
    )


def capture_architecture_error(
    action: Callable[[], object],
) -> ArchitectureManifestValidationError:
    try:
        action()
    except ArchitectureManifestValidationError as error:
        return error
    raise AssertionError("expected ArchitectureManifestValidationError")
