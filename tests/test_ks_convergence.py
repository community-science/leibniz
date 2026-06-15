import math
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from leibniz.benchmark_implementations import load_benchmark
from leibniz.tensor_runtime import resolve_tensor_runtime

_ks_benchmark_root = Path(__file__).parents[1] / "src" / "leibniz" / "benchmarks" / "ks"


def _ks_module() -> Any:
    loaded = cast(Any, load_benchmark(_ks_benchmark_root))
    return sys.modules[type(loaded.implementation).__module__]


def test_ambient_evolution_entropy_is_resolution_independent_for_bandlimited_field() -> None:
    module = _ks_module()
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    precision = 1.0e-3
    values: list[float] = []
    resolved_modes: list[int] = []
    for spatial_points in (32, 64, 128):
        x = (
            torch.arange(spatial_points, dtype=torch.float64, device=runtime.device)
            * (2.0 * math.pi / float(spatial_points))
        )
        initial = torch.sin(x).reshape(1, 1, spatial_points)
        delta = (
            0.125
            * torch.sin(2.0 * x).reshape(1, 1, spatial_points)
            * torch.arange(3, dtype=torch.float64, device=runtime.device).reshape(1, 3, 1)
        )
        trajectory = initial + delta

        entropy = module._ambient_evolution_entropy(
            runtime=runtime,
            trajectory=trajectory,
            precision=torch.full((1,), precision, dtype=torch.float64),
        )
        values.append(float(entropy.bits[0]))
        resolved_modes.append(int(entropy.resolved_mode_count[0]))

    assert max(values) - min(values) < 1.0e-9
    assert resolved_modes == [2, 2, 2]


def test_certified_ambient_bits_decrease_continuously_with_precision() -> None:
    module = _ks_module()
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    x = torch.arange(32, dtype=torch.float64, device=runtime.device) * (2.0 * math.pi / 32.0)
    trajectory = (
        torch.sin(x).reshape(1, 1, 32)
        + torch.arange(3, dtype=torch.float64, device=runtime.device).reshape(1, 3, 1)
        * 0.2
        * torch.cos(3.0 * x).reshape(1, 1, 32)
    )

    fine = module._ambient_evolution_entropy(
        runtime=runtime,
        trajectory=trajectory,
        precision=torch.full((1,), 1.0e-4, dtype=torch.float64),
    )
    coarse = module._ambient_evolution_entropy(
        runtime=runtime,
        trajectory=trajectory,
        precision=torch.full((1,), 1.0e-2, dtype=torch.float64),
    )

    assert float(fine.bits[0]) > float(coarse.bits[0]) > 0.0


def test_law_amplification_is_stable_for_smooth_field_and_grows_when_underresolved() -> None:
    module = _ks_module()
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    smooth_values: list[float] = []
    underresolved_values: list[float] = []
    for spatial_points in (32, 64, 128):
        x = torch.arange(
            spatial_points,
            dtype=torch.float64,
            device=runtime.device,
        ).reshape(1, 1, spatial_points)
        smooth_phase = x * (2.0 * math.pi / float(spatial_points))
        smooth = torch.sin(2.0 * smooth_phase).repeat(1, 3, 1)
        underresolved_phase = x * (
            2.0 * math.pi * float(spatial_points // 4) / float(spatial_points)
        )
        underresolved = torch.sin(underresolved_phase).repeat(1, 3, 1)

        smooth_values.append(
            float(module._per_sample_law_amplification(smooth, horizon=0.25)[0])
        )
        underresolved_values.append(
            float(module._per_sample_law_amplification(underresolved, horizon=0.25)[0])
        )

    assert max(smooth_values) / min(smooth_values) < 1.01
    assert underresolved_values[0] < underresolved_values[1] < underresolved_values[2]
    assert underresolved_values[-1] / underresolved_values[0] > 2.0


def test_ks_space_time_residual_uses_central_time_derivative() -> None:
    module = _ks_module()
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    dt = 0.25
    times = torch.arange(5, dtype=torch.float64, device=runtime.device) * dt
    trajectory = (times * times).reshape(1, 5, 1).repeat(1, 1, 8)

    residual = module.ks_space_time_residual(trajectory, dx=1.0, dt=dt)

    expected_interior_time_derivative = (2.0 * times[1:-1]).reshape(1, 3, 1).repeat(
        1,
        1,
        8,
    )
    assert residual.shape == trajectory.shape
    assert residual[:, 1:-1, :].allclose(expected_interior_time_derivative)


def test_ks_space_time_residual_rejects_invalid_shapes_and_spacing() -> None:
    module = _ks_module()
    runtime = resolve_tensor_runtime("cpu")
    trajectory = runtime.torch.zeros((1, 2, 8), dtype=runtime.torch.float32)

    with pytest.raises(ValueError, match="dx must be positive"):
        module.ks_space_time_residual(trajectory, dx=0.0, dt=1.0)
    with pytest.raises(ValueError, match="shape"):
        module.ks_space_time_residual(trajectory.reshape(2, 8), dx=1.0, dt=1.0)
    with pytest.raises(ValueError, match="at least five space"):
        module.ks_space_time_residual(trajectory[:, :, :4], dx=1.0, dt=1.0)
