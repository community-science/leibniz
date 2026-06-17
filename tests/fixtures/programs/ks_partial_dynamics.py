from __future__ import annotations

from typing import Any

from leibniz.program_graphs import (
    ProgramGraph,
    ProgramGraphEdge,
    ProgramGraphNode,
    ProgramTensorContract,
)

_box_length = 22.0


def build_program_graph(runtime: Any) -> ProgramGraph:
    torch = runtime.torch

    class PartialKSStep(torch.nn.Module):
        """A cheap explicit step using only the unstable linear KS terms."""

        def forward(self, field: Any, dt: Any) -> Any:
            step_dt = float(dt[0]) if hasattr(dt, "shape") and len(tuple(dt.shape)) else float(dt)
            spatial_points = int(field.shape[-1])
            dx = _box_length / float(spatial_points)
            u_xx = (
                field.roll(shifts=-1, dims=-1)
                - (2.0 * field)
                + field.roll(shifts=1, dims=-1)
            ) / (dx * dx)
            return field - (step_dt * u_xx)

    return ProgramGraph(
        contract_kind="prediction",
        inputs=(
            ProgramTensorContract("field", (1, "S")),
            ProgramTensorContract("dt", ()),
        ),
        outputs=(ProgramTensorContract("future_field", (1, "S")),),
        nodes=(
            ProgramGraphNode(
                id="step",
                kind="submitted-ks-partial-linear-step",
                operation=PartialKSStep(),
                parameters={"box_length": _box_length},
            ),
        ),
        edges=(
            ProgramGraphEdge("field", "step", 0),
            ProgramGraphEdge("dt", "step", 1),
            ProgramGraphEdge("step", "future_field"),
        ),
    )
