from collections.abc import Callable, Mapping

from leibniz.architecture_semantics import (
    ArchitectureSemanticValidationError,
    validate_architecture_semantics,
)
from leibniz.architectures import ArchitectureManifest


def test_architecture_semantic_validation_accepts_public_fixture() -> None:
    manifest = ArchitectureManifest.from_record(_architecture_record())
    plan = validate_architecture_semantics(manifest)

    assert plan.output_shape == (10,)
    assert [operator.output_shape for operator in plan.operators] == [
        (1, 2, 2),
        (4,),
        (10,),
    ]
    assert plan.parameter_count == 50
    assert plan.inference_compute == 1104


def test_architecture_semantic_validation_rejects_unknown_operator_kind() -> None:
    record = _architecture_record()
    record["layers"] = [{"kind": "named-layer", "parameters": {}}]
    manifest = ArchitectureManifest.from_record(record)

    assert str(
        capture_semantic_error(lambda: validate_architecture_semantics(manifest))
    ) == "layer 0 (named-layer): unsupported operator kind"


def test_architecture_semantic_validation_reports_required_parameters() -> None:
    record = _architecture_record()
    layers = list(_layers())
    layers[0] = {"kind": "adaptive-pooling", "parameters": {"dimension": 2}}
    record["layers"] = layers
    manifest = ArchitectureManifest.from_record(record)

    assert str(
        capture_semantic_error(lambda: validate_architecture_semantics(manifest))
    ) == "layer 0 (adaptive-pooling): missing required parameter size"

    record = _architecture_record()
    layers = list(_layers())
    layers[2] = {"kind": "dense", "parameters": {"out": 0}}
    record["layers"] = layers
    manifest = ArchitectureManifest.from_record(record)

    assert str(
        capture_semantic_error(lambda: validate_architecture_semantics(manifest))
    ) == "layer 2 (dense): parameter out must be a positive integer"


def test_architecture_semantic_validation_reports_unresolved_shape_law() -> None:
    record = _architecture_record()
    layers = list(_layers())
    layers[0] = {
        "kind": "adaptive-pooling",
        "parameters": {"dimension": 4, "size": 2},
    }
    record["layers"] = layers
    manifest = ArchitectureManifest.from_record(record)

    assert str(
        capture_semantic_error(lambda: validate_architecture_semantics(manifest))
    ) == (
        "layer 0 (adaptive-pooling): "
        "semantic interpretation could not resolve output_shape"
    )


def test_architecture_semantic_validation_reports_declared_output_mismatch() -> None:
    record = _architecture_record()
    record["output_shape"] = [11]
    manifest = ArchitectureManifest.from_record(record)

    assert str(
        capture_semantic_error(lambda: validate_architecture_semantics(manifest))
    ) == "resolved operator output shape does not match architecture output_shape"


def test_architecture_manifest_structure_still_accepts_opaque_layers() -> None:
    record: dict[str, object] = {
        "input_shape": [1],
        "output_shape": [1],
        "layers": [{"kind": "opaque-layer", "parameters": {}}],
    }
    manifest = ArchitectureManifest.from_record(record)

    assert manifest.layers[0].kind == "opaque-layer"


def _architecture_record() -> dict[str, object]:
    return {
        "input_shape": [1, 32, 32],
        "output_shape": [10],
        "layers": list(_layers()),
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


def capture_semantic_error(
    action: Callable[[], object],
) -> ArchitectureSemanticValidationError:
    try:
        action()
    except ArchitectureSemanticValidationError as error:
        return error
    raise AssertionError("expected ArchitectureSemanticValidationError")
