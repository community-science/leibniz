from __future__ import annotations

import math
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
        """A partial KS step: exact linear dynamics, no nonlinear transport."""

        def forward(self, field: Any, dt: Any) -> Any:
            step_dt = float(dt[0]) if hasattr(dt, "shape") and len(tuple(dt.shape)) else float(dt)
            spatial_points = int(field.shape[-1])
            frequencies = tuple(
                index if index <= spatial_points // 2 else index - spatial_points
                for index in range(spatial_points)
            )
            growth = tuple(
                ((2.0 * math.pi * frequency / _box_length) ** 2)
                - ((2.0 * math.pi * frequency / _box_length) ** 4)
                for frequency in frequencies
            )
            linear = torch.tensor(
                tuple(complex(math.exp(step_dt * value), 0.0) for value in growth),
                dtype=torch.complex128,
                device=field.device,
            ).reshape((1, 1, spatial_points))
            spectrum = torch.fft.fft(field.to(dtype=torch.float64), dim=-1)
            return torch.fft.ifft(linear * spectrum, dim=-1).real.to(dtype=field.dtype)

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
