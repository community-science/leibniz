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
        contract_kind="classification",
        inputs=(ProgramTensorContract("image", (1, 24, 24)),),
        outputs=(ProgramTensorContract("class_logits", (10,)),),
        nodes=(
            ProgramGraphNode(
                id="pool",
                kind="submitted-adaptive-pooling",
                operation=torch.nn.AdaptiveAvgPool2d((2, 2)),
            ),
            ProgramGraphNode(
                id="flatten",
                kind="submitted-flatten",
                operation=torch.nn.Flatten(),
            ),
            ProgramGraphNode(
                id="readout",
                kind="submitted-linear-readout",
                operation=torch.nn.Linear(4, 10),
            ),
        ),
        edges=(
            ProgramGraphEdge("image", "pool"),
            ProgramGraphEdge("pool", "flatten"),
            ProgramGraphEdge("flatten", "readout"),
            ProgramGraphEdge("readout", "class_logits"),
        ),
    )
