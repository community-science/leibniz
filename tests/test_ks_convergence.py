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


def test_richardson_recovers_planted_order_and_bounds_true_error() -> None:
    module = _ks_module()
    true_limit = 3.0
    planted_order = 2.0
    factor = 2.0
    coefficient = 8.0
    sequence = tuple(
        true_limit + coefficient / (factor ** (planted_order * index))
        for index in range(4)
    )

    estimate = module.richardson(sequence, factor=factor)

    assert math.isclose(estimate.observed_order, planted_order)
    assert math.isclose(estimate.limit, true_limit)
    assert estimate.uncertainty >= abs(sequence[-1] - true_limit)


def test_richardson_rejects_degenerate_sequences() -> None:
    module = _ks_module()
    with pytest.raises(ValueError, match="at least three"):
        module.richardson((1.0, 0.5), factor=2.0)
    with pytest.raises(ValueError, match="nonzero"):
        module.richardson((1.0, 1.0, 1.0), factor=2.0)
    with pytest.raises(ValueError, match="greater than one"):
        module.richardson((1.0, 0.5, 0.25), factor=1.0)


def test_richardson_field_decimates_nested_ladder_and_bounds_error() -> None:
    module = _ks_module()
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    true_field = torch.arange(12, dtype=torch.float64, device=runtime.device).reshape(
        1,
        3,
        4,
    )
    planted_order = 2.0
    factor = 2
    coefficient = 8.0
    ladder: list[Any] = []
    for index in range(3):
        scale = factor**index
        field = true_field.repeat_interleave(scale, dim=-2).repeat_interleave(
            scale,
            dim=-1,
        )
        field = field + coefficient / (factor ** (planted_order * index))
        ladder.append(field)

    estimate = module.richardson_field(tuple(ladder), factor=factor)

    assert math.isclose(estimate.observed_order, planted_order)
    assert estimate.error >= module.grid_l2_norm(ladder[-1][..., ::4, ::4] - true_field)
    assert estimate.extrapolated_field.allclose(true_field)


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
