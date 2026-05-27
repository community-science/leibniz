from collections.abc import Callable, Mapping
from pathlib import Path

from leibniz.architectures import (
    ArchitectureLayer,
    ArchitectureManifest,
    ArchitectureManifestDocument,
    ArchitectureManifestValidationError,
)
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes

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
            ArchitectureLayer(
                kind="adaptive-pooling",
                parameters={"dimension": 2, "size": 2},
            ),
            ArchitectureLayer(kind="flatten", parameters={}),
            ArchitectureLayer(kind="dense", parameters={"out": 10}),
        ),
    )
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
    layers = list(_layers())
    layers[0] = {"kind": ""}
    record["layers"] = layers
    assert str(capture_architecture_error(lambda: ArchitectureManifest.from_record(record))) == (
        "layer kind must be nonempty"
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
