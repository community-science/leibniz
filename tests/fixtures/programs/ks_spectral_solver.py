from __future__ import annotations

import cmath
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

    class SpectralKSStep(torch.nn.Module):
        def forward(self, field: Any, dt: Any) -> Any:
            step_dt = float(dt[0]) if hasattr(dt, "shape") and len(tuple(dt.shape)) else float(dt)
            state = field.to(dtype=torch.float64)
            spatial_points = int(state.shape[-1])
            (
                linear,
                half_linear,
                q,
                f1,
                f2,
                f3,
                derivative,
                dealias,
            ) = _solver_tensors(torch, state.device, spatial_points, step_dt)
            spectrum = torch.fft.fft(state, dim=-1)
            nonlinear = _nonlinear_spectrum(torch, spectrum, derivative, dealias)
            a_spectrum = (half_linear * spectrum) + (q * nonlinear)
            a_nonlinear = _nonlinear_spectrum(torch, a_spectrum, derivative, dealias)
            b_spectrum = (half_linear * spectrum) + (q * a_nonlinear)
            b_nonlinear = _nonlinear_spectrum(torch, b_spectrum, derivative, dealias)
            c_spectrum = (half_linear * a_spectrum) + (q * ((2.0 * b_nonlinear) - nonlinear))
            c_nonlinear = _nonlinear_spectrum(torch, c_spectrum, derivative, dealias)
            next_spectrum = (
                linear * spectrum
                + f1 * nonlinear
                + (2.0 * f2 * (a_nonlinear + b_nonlinear))
                + f3 * c_nonlinear
            )
            return torch.fft.ifft(next_spectrum, dim=-1).real.to(dtype=field.dtype)

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
                kind="submitted-ks-spectral-etdrk4-step",
                operation=SpectralKSStep(),
                parameters={"box_length": _box_length, "method": "ETDRK4"},
            ),
        ),
        edges=(
            ProgramGraphEdge("field", "step", 0),
            ProgramGraphEdge("dt", "step", 1),
            ProgramGraphEdge("step", "future_field"),
        ),
    )


def _solver_tensors(
    torch: Any,
    device: Any,
    spatial_points: int,
    dt: float,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    frequencies = tuple(
        index if index <= spatial_points // 2 else index - spatial_points
        for index in range(spatial_points)
    )
    wave_numbers = tuple(
        2.0 * math.pi * frequency / _box_length for frequency in frequencies
    )
    linear_values = tuple(
        (wave_number * wave_number) - (wave_number**4) for wave_number in wave_numbers
    )
    coefficient_rows = tuple(
        _etdrk4_coefficients(value, dt=dt) for value in linear_values
    )
    shape = (1, 1, spatial_points)
    return (
        torch.tensor(
            tuple(complex(math.exp(dt * value), 0.0) for value in linear_values),
            dtype=torch.complex128,
            device=device,
        ).reshape(shape),
        torch.tensor(
            tuple(complex(math.exp(0.5 * dt * value), 0.0) for value in linear_values),
            dtype=torch.complex128,
            device=device,
        ).reshape(shape),
        torch.tensor(
            tuple(row[0] for row in coefficient_rows),
            dtype=torch.complex128,
            device=device,
        ).reshape(shape),
        torch.tensor(
            tuple(row[1] for row in coefficient_rows),
            dtype=torch.complex128,
            device=device,
        ).reshape(shape),
        torch.tensor(
            tuple(row[2] for row in coefficient_rows),
            dtype=torch.complex128,
            device=device,
        ).reshape(shape),
        torch.tensor(
            tuple(row[3] for row in coefficient_rows),
            dtype=torch.complex128,
            device=device,
        ).reshape(shape),
        torch.tensor(
            tuple(complex(0.0, wave_number) for wave_number in wave_numbers),
            dtype=torch.complex128,
            device=device,
        ).reshape(shape),
        torch.tensor(
            tuple(
                1.0 if abs(frequency) <= spatial_points // 3 else 0.0
                for frequency in frequencies
            ),
            dtype=torch.float64,
            device=device,
        ).reshape(shape),
    )


def _nonlinear_spectrum(torch: Any, spectrum: Any, derivative: Any, dealias: Any) -> Any:
    field = torch.fft.ifft(spectrum, dim=-1).real
    nonlinear = -0.5 * torch.fft.fft(field * field, dim=-1) * derivative
    return nonlinear * dealias


def _etdrk4_coefficients(
    linear_value: float,
    *,
    dt: float,
) -> tuple[complex, complex, complex, complex]:
    roots = tuple(
        complex(
            math.cos(math.pi * ((index + 0.5) / 16.0)),
            math.sin(math.pi * ((index + 0.5) / 16.0)),
        )
        for index in range(16)
    )
    lr_values = tuple((dt * linear_value) + root for root in roots)
    q = dt * sum((cmath.exp(lr / 2.0) - 1.0) / lr for lr in lr_values) / len(
        lr_values
    )
    f1 = (
        dt
        * sum(
            (-4.0 - lr + cmath.exp(lr) * (4.0 - (3.0 * lr) + (lr * lr)))
            / (lr * lr * lr)
            for lr in lr_values
        )
        / len(lr_values)
    )
    f2 = (
        dt
        * sum(
            (2.0 + lr + cmath.exp(lr) * (-2.0 + lr)) / (lr * lr * lr)
            for lr in lr_values
        )
        / len(lr_values)
    )
    f3 = (
        dt
        * sum(
            (-4.0 - (3.0 * lr) - (lr * lr) + cmath.exp(lr) * (4.0 - lr))
            / (lr * lr * lr)
            for lr in lr_values
        )
        / len(lr_values)
    )
    return (q, f1, f2, f3)
