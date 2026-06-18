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

    class LearnedResidualKSStep(torch.nn.Module):
        """Partial linear dynamics plus a learned residual correction."""

        def __init__(self) -> None:
            super().__init__()  # pyright: ignore[reportUnknownMemberType]
            self.correction = torch.nn.Sequential(
                torch.nn.Conv1d(
                    1,
                    16,
                    kernel_size=5,
                    padding=2,
                    padding_mode="circular",
                ),
                torch.nn.Tanh(),
                torch.nn.Conv1d(
                    16,
                    16,
                    kernel_size=5,
                    padding=2,
                    padding_mode="circular",
                ),
                torch.nn.Tanh(),
                torch.nn.Conv1d(
                    16,
                    1,
                    kernel_size=5,
                    padding=2,
                    padding_mode="circular",
                ),
            )
            torch.nn.init.zeros_(self.correction[-1].weight)
            torch.nn.init.zeros_(self.correction[-1].bias)

        def forward(self, field: Any, dt: Any) -> Any:
            step_dt = float(dt[0]) if hasattr(dt, "shape") and len(tuple(dt.shape)) else float(dt)
            linear_state = _linear_ks_step(
                torch=torch,
                field=field,
                dt=step_dt,
            )
            correction = self.correction(field)
            return linear_state + (step_dt * correction)

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
                kind="submitted-ks-learned-residual-correction-step",
                operation=LearnedResidualKSStep(),
                parameters={
                    "box_length": _box_length,
                    "base_dynamics": "linear_spectral_ks",
                    "learned_component": "local_residual_correction",
                    "training_signal": "ks_residual_loss",
                },
            ),
        ),
        edges=(
            ProgramGraphEdge("field", "step", 0),
            ProgramGraphEdge("dt", "step", 1),
            ProgramGraphEdge("step", "future_field"),
        ),
    )


def _linear_ks_step(*, torch: Any, field: Any, dt: float) -> Any:
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
        tuple(complex(math.exp(dt * value), 0.0) for value in growth),
        dtype=torch.complex128,
        device=field.device,
    ).reshape((1, 1, spatial_points))
    spectrum = torch.fft.fft(field.to(dtype=torch.float64), dim=-1)
    return torch.fft.ifft(linear * spectrum, dim=-1).real.to(dtype=field.dtype)
