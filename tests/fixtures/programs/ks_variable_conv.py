from __future__ import annotations

from typing import Any

from leibniz.program_graphs import (
    ProgramGraph,
    ProgramGraphEdge,
    ProgramGraphNode,
    ProgramTensorContract,
)


def build_program_graph(runtime: Any) -> ProgramGraph:
    torch = runtime.torch
    return ProgramGraph(
        contract_kind="prediction",
        inputs=(ProgramTensorContract("field", (1, "S")),),
        outputs=(ProgramTensorContract("future_field", (1, "S")),),
        nodes=(
            ProgramGraphNode(
                id="conv_0",
                kind="submitted-conv1d",
                operation=torch.nn.Conv1d(
                    1,
                    8,
                    kernel_size=3,
                    padding=1,
                    padding_mode="circular",
                ),
            ),
            ProgramGraphNode(
                id="activation",
                kind="submitted-relu",
                operation=torch.nn.ReLU(),
            ),
            ProgramGraphNode(
                id="conv_1",
                kind="submitted-conv1d",
                operation=torch.nn.Conv1d(
                    8,
                    1,
                    kernel_size=3,
                    padding=1,
                    padding_mode="circular",
                ),
            ),
        ),
        edges=(
            ProgramGraphEdge("field", "conv_0"),
            ProgramGraphEdge("conv_0", "activation"),
            ProgramGraphEdge("activation", "conv_1"),
            ProgramGraphEdge("conv_1", "future_field"),
        ),
    )
