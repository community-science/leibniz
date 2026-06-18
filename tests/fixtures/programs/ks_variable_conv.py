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

    class EulerResidualStep(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()  # pyright: ignore[reportUnknownMemberType]
            self.net = torch.nn.Sequential(
                torch.nn.Conv1d(
                    1,
                    8,
                    kernel_size=3,
                    padding=1,
                    padding_mode="circular",
                ),
                torch.nn.ReLU(),
                torch.nn.Conv1d(
                    8,
                    1,
                    kernel_size=3,
                    padding=1,
                    padding_mode="circular",
                ),
            )

        def forward(self, field: Any, dt: Any) -> Any:
            if len(tuple(dt.shape)) == 1:
                dt = dt.reshape((-1, 1, 1))
            return field + (dt * self.net(field))

    return ProgramGraph(
        contract_kind="prediction",
        inputs=(
            ProgramTensorContract("field", (1, "S")),
            ProgramTensorContract("dt", (), nonnegative=True),
        ),
        outputs=(ProgramTensorContract("future_field", (1, "S")),),
        nodes=(
            ProgramGraphNode(
                id="step",
                kind="submitted-euler-residual-step",
                operation=EulerResidualStep(),
            ),
        ),
        edges=(
            ProgramGraphEdge("field", "step", 0),
            ProgramGraphEdge("dt", "step", 1),
            ProgramGraphEdge("step", "future_field"),
        ),
    )
