import ast
import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.model_operators import (
    ExecutableModelOperator,
    ModelOperatorExecutionError,
    ModelOperatorSearchPoint,
    materialize_model_operator_search_point,
    summarize_architecture_operators,
)

_fixtures_root = Path(__file__).parent / "fixtures"
_src_root = Path(__file__).parents[1] / "src" / "leibniz"


def test_model_operator_summary_classifies_formal_semantics() -> None:
    plan = summarize_architecture_operators(_architecture_manifest())

    assert plan.input_shape == (1, 32, 32)
    assert plan.output_shape == (10,)
    assert [operator.descriptor.kind for operator in plan.operators] == [
        "local-aggregation",
        "rank-collapse",
        "affine-readout",
    ]
    assert [operator.descriptor.aliases for operator in plan.operators] == [
        ("adaptive-pooling",),
        ("flatten",),
        ("dense",),
    ]
    assert plan.operators[0].descriptor.to_record() == {
        "kind": "local-aggregation",
        "tensor_relation": "aggregation",
        "state": "fixed",
        "support": "local-window",
        "projection_law": "equal-output-partition",
        "aggregation_law": "mean",
        "parameter_sharing": "none",
        "shape_law": "preserve-prefix-replace-trailing-axes",
        "cost_law": "input-elements",
        "aliases": ["adaptive-pooling"],
    }
    assert [operator.output_shape for operator in plan.operators] == [
        (1, 2, 2),
        (4,),
        (10,),
    ]
    assert [operator.parameter_count for operator in plan.operators] == [0, 0, 50]
    assert [operator.parameter_bytes for operator in plan.operators] == [0, 0, 200]
    assert [operator.inference_flops for operator in plan.operators] == [1024, 0, 80]
    assert plan.parameter_count == 50
    assert plan.parameter_bytes == 200
    assert plan.inference_flops == 1104


def test_model_operator_summary_rejects_unknown_operator_kind() -> None:
    record: dict[str, object] = {
        "input_shape": [1, 32, 32],
        "output_shape": [10],
        "layers": [{"kind": "named-layer", "parameters": {}}],
    }
    manifest = ArchitectureManifest.from_record(
        record
    )

    assert str(
        capture_operator_error(lambda: summarize_architecture_operators(manifest))
    ) == "unsupported operator kind: named-layer"


def test_model_operator_summary_rejects_shape_law_mismatch() -> None:
    record: dict[str, object] = {
        "input_shape": [1, 32, 32],
        "output_shape": [11],
        "layers": [
            {"kind": "adaptive-pooling", "parameters": {"dimension": 2, "size": 2}},
            {"kind": "flatten", "parameters": {}},
            {"kind": "dense", "parameters": {"out": 10}},
        ],
    }
    manifest = ArchitectureManifest.from_record(record)

    assert str(
        capture_operator_error(lambda: summarize_architecture_operators(manifest))
    ) == "resolved operator output shape does not match architecture output_shape"


def test_torch_instantiation_is_a_minimal_sequential_specialization() -> None:
    torch = cast(Any, importlib.import_module("torch"))

    module = ExecutableModelOperator(_architecture_manifest()).torch_module()
    output = module(torch.zeros(2, 1, 32, 32))

    assert len(module) == 3
    assert output.shape == (2, 10)


def test_semantic_search_point_materialization_routes_aliases_through_operator_registry() -> None:
    manifest = materialize_model_operator_search_point(
        input_shape=(1, 32, 32),
        output_count=10,
        point=ModelOperatorSearchPoint(
            local_support_dimension=2,
            local_support_size=3,
        ),
    )
    plan = summarize_architecture_operators(manifest)

    assert manifest.input_shape == (1, 32, 32)
    assert manifest.output_shape == (10,)
    assert [operator.descriptor.kind for operator in plan.operators] == [
        "local-aggregation",
        "rank-collapse",
        "affine-readout",
    ]
    assert plan.parameter_count == 100


def test_layer_alias_literals_are_defined_only_in_the_operator_registry() -> None:
    observed_locations: dict[str, list[str]] = {
        "adaptive-pooling": [],
        "flatten": [],
        "dense": [],
    }
    for path in _src_root.rglob("*.py"):
        module = ast.parse(path.read_text(), filename=path.as_posix())
        for node in ast.walk(module):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in observed_locations
            ):
                observed_locations[node.value].append(
                    f"{path.relative_to(_src_root)}:{node.lineno}"
                )

    assert {
        alias: [location.split(":", maxsplit=1)[0] for location in locations]
        for alias, locations in observed_locations.items()
    } == {
        "adaptive-pooling": ["model_operators.py"],
        "flatten": ["model_operators.py"],
        "dense": ["model_operators.py"],
    }


def _architecture_manifest() -> ArchitectureManifest:
    return ArchitectureManifestDocument.from_bytes(
        (_fixtures_root / "architecture" / "digits_pool" / "manifest.json").read_bytes()
    ).manifest


def capture_operator_error(
    action: Callable[[], object],
) -> ModelOperatorExecutionError:
    try:
        action()
    except ModelOperatorExecutionError as error:
        return error
    raise AssertionError("expected ModelOperatorExecutionError")
