from __future__ import annotations

import cmath
import math
from typing import Any

from leibniz.tensor_runtime import TensorRuntime

_box_length = 22.0
_default_space_count = 32
_horizon = 1.0
_time_count = 9
_initial_condition_mode_count = 4


def ks_reference_trajectory(
    *,
    runtime: TensorRuntime,
    sample_count: int,
    seed: int,
    sample_indices: tuple[int, ...],
    window: int,
    spatial_points: int = _default_space_count,
    horizon: float = _horizon,
    time_count: int = _time_count,
) -> Any:
    if time_count < 2:
        raise ValueError("KS reference trajectory requires at least two time samples")
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("KS reference trajectory horizon must be positive and finite")
    state = _ks_initial_fields(
        runtime=runtime,
        sample_count=sample_count,
        seed=seed,
        sample_indices=sample_indices,
        window=window,
        spatial_points=spatial_points,
    )
    dt = horizon / float(time_count - 1)
    states = [state]
    for _index in range(time_count - 1):
        state = ks_reference_step(runtime=runtime, fields=state, dt=dt)
        states.append(state)
    return runtime.torch.stack(states, dim=2)


def _ks_initial_fields(
    *,
    runtime: TensorRuntime,
    sample_count: int,
    seed: int,
    sample_indices: tuple[int, ...],
    window: int,
    spatial_points: int,
) -> Any:
    _ = window
    torch = runtime.torch
    samples = torch.tensor(sample_indices, dtype=torch.float64, device=runtime.device).reshape(
        (-1, 1, 1)
    )
    if int(samples.shape[0]) != sample_count:
        raise ValueError("sample_count must match sample_indices")
    modes = torch.arange(
        1,
        _initial_condition_mode_count + 1,
        dtype=torch.float64,
        device=runtime.device,
    ).reshape((1, -1, 1))
    decay = 1.0 / (modes * modes)
    spatial = torch.arange(
        spatial_points,
        dtype=torch.float64,
        device=runtime.device,
    ).reshape((1, 1, -1)) * (2.0 * math.pi / float(spatial_points))
    random_key = float(seed) + (0.173 * samples)
    sine_coefficients = (
        (random_key * 12.9898 + modes * 78.233 + 0.37).sin()
        * decay
    )
    cosine_coefficients = (
        (random_key * 4.1414 + modes * 31.416 + 1.91).sin()
        * decay
    )
    energy = (
        (sine_coefficients * sine_coefficients)
        + (cosine_coefficients * cosine_coefficients)
    ).sum(dim=1, keepdim=True).sqrt().clamp_min(math.ulp(1.0))
    field = (
        (sine_coefficients * (modes * spatial).sin())
        + (cosine_coefficients * (modes * spatial).cos())
    ).sum(dim=1, keepdim=True)
    return 0.15 * field / energy


def ks_reference_step(*, runtime: TensorRuntime, fields: Any, dt: float) -> Any:
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("KS reference step dt must be positive and finite")
    spatial_points = int(fields.shape[-1])
    torch = runtime.torch
    state = fields.to(dtype=torch.float64)
    (
        linear_factors,
        half_linear_factors,
        q_coefficients,
        f1_coefficients,
        f2_coefficients,
        f3_coefficients,
        derivative_coefficients,
        dealias_mask,
    ) = _ks_solver_tensors(runtime=runtime, spatial_points=spatial_points, dt=dt)
    spectrum = torch.fft.fft(state, dim=-1)
    nonlinear_spectrum = _ks_nonlinear_spectrum(
        runtime=runtime,
        spectrum=spectrum,
        derivative_coefficients=derivative_coefficients,
        dealias_mask=dealias_mask,
    )
    half_linear = half_linear_factors.reshape((1, 1, -1))
    q = q_coefficients.reshape((1, 1, -1))
    a_spectrum = (half_linear * spectrum) + (q * nonlinear_spectrum)
    a_nonlinear = _ks_nonlinear_spectrum(
        runtime=runtime,
        spectrum=a_spectrum,
        derivative_coefficients=derivative_coefficients,
        dealias_mask=dealias_mask,
    )
    b_spectrum = (half_linear * spectrum) + (q * a_nonlinear)
    b_nonlinear = _ks_nonlinear_spectrum(
        runtime=runtime,
        spectrum=b_spectrum,
        derivative_coefficients=derivative_coefficients,
        dealias_mask=dealias_mask,
    )
    c_spectrum = (half_linear * a_spectrum) + (
        q * ((2.0 * b_nonlinear) - nonlinear_spectrum)
    )
    c_nonlinear = _ks_nonlinear_spectrum(
        runtime=runtime,
        spectrum=c_spectrum,
        derivative_coefficients=derivative_coefficients,
        dealias_mask=dealias_mask,
    )
    next_spectrum = (
        linear_factors.reshape((1, 1, -1)) * spectrum
        + f1_coefficients.reshape((1, 1, -1)) * nonlinear_spectrum
        + 2.0
        * f2_coefficients.reshape((1, 1, -1))
        * (a_nonlinear + b_nonlinear)
        + f3_coefficients.reshape((1, 1, -1)) * c_nonlinear
    )
    return torch.fft.ifft(next_spectrum, dim=-1).real


def _ks_solver_tensors(
    *,
    runtime: TensorRuntime,
    spatial_points: int,
    dt: float,
) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    torch = runtime.torch
    frequencies = tuple(
        index if index <= spatial_points // 2 else index - spatial_points
        for index in range(spatial_points)
    )
    wave_numbers = tuple(2.0 * math.pi * frequency / _box_length for frequency in frequencies)
    linear_values = tuple(
        (wave_number * wave_number) - (wave_number**4)
        for wave_number in wave_numbers
    )
    coefficient_rows = tuple(
        _etdrk4_coefficients(value, dt=dt)
        for value in linear_values
    )
    return (
        torch.tensor(
            tuple(complex(math.exp(dt * value), 0.0) for value in linear_values),
            dtype=torch.complex128,
            device=runtime.device,
        ),
        torch.tensor(
            tuple(complex(math.exp(0.5 * dt * value), 0.0) for value in linear_values),
            dtype=torch.complex128,
            device=runtime.device,
        ),
        torch.tensor(
            tuple(row[0] for row in coefficient_rows),
            dtype=torch.complex128,
            device=runtime.device,
        ),
        torch.tensor(
            tuple(row[1] for row in coefficient_rows),
            dtype=torch.complex128,
            device=runtime.device,
        ),
        torch.tensor(
            tuple(row[2] for row in coefficient_rows),
            dtype=torch.complex128,
            device=runtime.device,
        ),
        torch.tensor(
            tuple(row[3] for row in coefficient_rows),
            dtype=torch.complex128,
            device=runtime.device,
        ),
        torch.tensor(
            tuple(complex(0.0, wave_number) for wave_number in wave_numbers),
            dtype=torch.complex128,
            device=runtime.device,
        ),
        torch.tensor(
            tuple(
                1.0 if abs(frequency) <= spatial_points // 3 else 0.0
                for frequency in frequencies
            ),
            dtype=torch.float64,
            device=runtime.device,
        ),
    )


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
    lr_values = tuple(dt * linear_value + root for root in roots)
    q = dt * sum((cmath.exp(lr / 2.0) - 1.0) / lr for lr in lr_values) / len(lr_values)
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
    return q, f1, f2, f3


def _ks_nonlinear_spectrum(
    *,
    runtime: TensorRuntime,
    spectrum: Any,
    derivative_coefficients: Any,
    dealias_mask: Any,
) -> Any:
    torch = runtime.torch
    state = torch.fft.ifft(spectrum, dim=-1).real
    gradient = torch.fft.ifft(
        spectrum * derivative_coefficients.reshape((1, 1, -1)),
        dim=-1,
    ).real
    nonlinear_spectrum = torch.fft.fft(-state * gradient, dim=-1)
    return nonlinear_spectrum * dealias_mask.reshape((1, 1, -1))
