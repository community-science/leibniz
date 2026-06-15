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
        inputs=(ProgramTensorContract("board", (18, 8, 8)),),
        outputs=(ProgramTensorContract("move_logits", (1968,)),),
        nodes=(
            ProgramGraphNode(
                id="flatten",
                kind="submitted-flatten",
                operation=torch.nn.Flatten(),
            ),
            ProgramGraphNode(
                id="readout",
                kind="submitted-linear-readout",
                operation=torch.nn.Linear(18 * 8 * 8, 1968),
            ),
        ),
        edges=(
            ProgramGraphEdge("board", "flatten"),
            ProgramGraphEdge("flatten", "readout"),
            ProgramGraphEdge("readout", "move_logits"),
        ),
    )
