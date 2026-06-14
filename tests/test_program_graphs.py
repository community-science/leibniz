from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from leibniz.program_graphs import (
    ProgramAdd,
    ProgramConcat,
    ProgramGraph,
    ProgramGraphEdge,
    ProgramGraphError,
    ProgramGraphNode,
    ProgramIdentity,
    ProgramResampleLike,
    ProgramTensorContract,
)
from leibniz.tensor_runtime import resolve_tensor_runtime


def test_classification_program_graph_composes_trainable_open_node() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    graph = ProgramGraph(
        contract_kind="classification",
        inputs=(ProgramTensorContract("image", (1, 4)),),
        outputs=(ProgramTensorContract("class_logits", (3,)),),
        nodes=(
            ProgramGraphNode(
                id="readout",
                kind="submitted-linear-readout",
                operation=torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 3)),
            ),
        ),
        edges=(
            ProgramGraphEdge("image", "readout"),
            ProgramGraphEdge("readout", "class_logits"),
        ),
    )

    report = graph.validate(runtime, input_shapes=((1, 4),))
    module = graph.build_module(runtime)
    output = module(torch.zeros(2, 1, 4))

    assert report.contract_kind == "classification"
    assert report.topological_order == ("readout",)
    assert report.output_shapes == (((3,),),)
    assert output.shape == (2, 3)
    assert graph.digest == graph.digest


def test_prediction_program_graph_validates_skip_add_across_scales() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    graph = ProgramGraph(
        contract_kind="prediction",
        inputs=(ProgramTensorContract("field", (1, "N")),),
        outputs=(ProgramTensorContract("future_field", (1, "N")),),
        nodes=(
            ProgramGraphNode(
                id="local",
                kind="submitted-local-affine",
                operation=torch.nn.Conv1d(1, 1, kernel_size=1),
            ),
            ProgramGraphNode(id="skip", kind="structural-identity", operation=ProgramIdentity()),
            ProgramGraphNode(id="merge", kind="structural-add", operation=ProgramAdd()),
        ),
        edges=(
            ProgramGraphEdge("field", "local"),
            ProgramGraphEdge("field", "skip"),
            ProgramGraphEdge("local", "merge", target_input_index=0),
            ProgramGraphEdge("skip", "merge", target_input_index=1),
            ProgramGraphEdge("merge", "future_field"),
        ),
    )

    report = graph.validate(
        runtime,
        input_shapes=((1, 8),),
        additional_input_shapes=(((1, 13),),),
    )

    assert report.contract_kind == "prediction"
    assert report.input_shapes == (((1, 8),), ((1, 13),))
    assert report.output_shapes == (((1, 8),), ((1, 13),))
    assert report.topological_order == ("local", "skip", "merge")


def test_structural_concat_and_resample_like_are_scale_general() -> None:
    runtime = resolve_tensor_runtime("cpu")
    graph = ProgramGraph(
        contract_kind="prediction",
        inputs=(
            ProgramTensorContract("coarse", (1, "M")),
            ProgramTensorContract("reference", (1, "N")),
        ),
        outputs=(ProgramTensorContract("stacked", (2, "N")),),
        nodes=(
            ProgramGraphNode(
                id="coarse_at_reference",
                kind="structural-resample-like",
                operation=ProgramResampleLike(),
            ),
            ProgramGraphNode(
                id="stack",
                kind="structural-concat",
                operation=ProgramConcat(),
            ),
        ),
        edges=(
            ProgramGraphEdge("coarse", "coarse_at_reference", target_input_index=0),
            ProgramGraphEdge("reference", "coarse_at_reference", target_input_index=1),
            ProgramGraphEdge("reference", "stack", target_input_index=0),
            ProgramGraphEdge("coarse_at_reference", "stack", target_input_index=1),
            ProgramGraphEdge("stack", "stacked"),
        ),
    )

    report = graph.validate(
        runtime,
        input_shapes=((1, 5), (1, 11)),
        additional_input_shapes=(((1, 7), (1, 17)),),
    )

    assert report.output_shapes == (((2, 11),), ((2, 17),))


def test_program_graph_rejects_cycles_unknown_edges_and_duplicate_inputs() -> None:
    assert str(
        capture_program_error(
            lambda: ProgramGraph(
                contract_kind="prediction",
                inputs=(ProgramTensorContract("x", (1, "N")),),
                outputs=(ProgramTensorContract("y", (1, "N")),),
                nodes=(
                    ProgramGraphNode("a", ProgramIdentity(), "identity"),
                    ProgramGraphNode("b", ProgramIdentity(), "identity"),
                ),
                edges=(
                    ProgramGraphEdge("x", "a"),
                    ProgramGraphEdge("a", "b"),
                    ProgramGraphEdge("b", "a"),
                    ProgramGraphEdge("b", "y"),
                ),
            )
        )
    ) == "program graph must be acyclic"

    assert str(
        capture_program_error(
            lambda: ProgramGraph(
                contract_kind="classification",
                inputs=(ProgramTensorContract("x", (2,)),),
                outputs=(ProgramTensorContract("y", (2,)),),
                nodes=(ProgramGraphNode("a", ProgramIdentity(), "identity"),),
                edges=(
                    ProgramGraphEdge("missing", "a"),
                    ProgramGraphEdge("a", "y"),
                ),
            )
        )
    ) == "edge source_id 'missing' is not known"

    assert str(
        capture_program_error(
            lambda: ProgramGraph(
                contract_kind="classification",
                inputs=(ProgramTensorContract("x", (2,)),),
                outputs=(ProgramTensorContract("y", (2,)),),
                nodes=(ProgramGraphNode("a", ProgramIdentity(), "identity"),),
                edges=(
                    ProgramGraphEdge("x", "a"),
                    ProgramGraphEdge("x", "a"),
                    ProgramGraphEdge("a", "y"),
                ),
            )
        )
    ) == "target 'a' has duplicate input indices"


def test_program_graph_rejects_shape_and_scale_contract_violations() -> None:
    runtime = resolve_tensor_runtime("cpu")
    graph = ProgramGraph(
        contract_kind="prediction",
        inputs=(ProgramTensorContract("x", (1, "N")),),
        outputs=(ProgramTensorContract("y", (1, "N")),),
        nodes=(
            ProgramGraphNode("skip_a", ProgramIdentity(), "identity"),
            ProgramGraphNode("skip_b", ProgramIdentity(), "identity"),
            ProgramGraphNode("stack", ProgramConcat(), "concat"),
        ),
        edges=(
            ProgramGraphEdge("x", "skip_a"),
            ProgramGraphEdge("x", "skip_b"),
            ProgramGraphEdge("skip_a", "stack", target_input_index=0),
            ProgramGraphEdge("skip_b", "stack", target_input_index=1),
            ProgramGraphEdge("stack", "y"),
        ),
    )

    assert "sample 0 output y axis 0 extent 2 does not match contract 1" in str(
        capture_program_error(
            lambda: graph.validate(
                runtime,
                input_shapes=((1, 8),),
                additional_input_shapes=(((1, 13),),),
            )
        )
    )
    assert str(
        capture_program_error(lambda: graph.validate(runtime, input_shapes=((1, 8),)))
    ) == "symbolic tensor contracts require at least two scale probes"
    assert str(
        capture_program_error(
            lambda: graph.validate(
                runtime,
                input_shapes=((1, 8),),
                additional_input_shapes=(((1, 8),),),
            )
        )
    ) == "symbolic tensor contracts require varying scale probes"


def test_program_graph_rejects_nondifferentiable_submitted_programs() -> None:
    runtime = resolve_tensor_runtime("cpu")
    graph = ProgramGraph(
        contract_kind="classification",
        inputs=(ProgramTensorContract("x", (4,)),),
        outputs=(ProgramTensorContract("y", (4,)),),
        nodes=(ProgramGraphNode("detach", _Detach(), "submitted-detach"),),
        edges=(ProgramGraphEdge("x", "detach"), ProgramGraphEdge("detach", "y")),
    )

    assert str(
        capture_program_error(lambda: graph.validate(runtime, input_shapes=((4,),)))
    ) == "sample 0 output is not differentiable"


class _Detach:
    def __call__(self, value: Any) -> Any:
        return value.detach()


def capture_program_error(action: Callable[[], object]) -> ProgramGraphError:
    with pytest.raises(ProgramGraphError) as error:
        action()
    return error.value
