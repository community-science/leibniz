import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from ks_oracle import ks_reference_trajectory

from leibniz.benchmark_implementations import load_benchmark
from leibniz.benchmark_runner import (
    BenchmarkRunnerError,
    _field_valued_model_trajectory,  # pyright: ignore[reportPrivateUsage]
)
from leibniz.observation_generation import ObservationGenerationError, StateSpaceVolumeRequest
from leibniz.tensor_runtime import resolve_tensor_runtime

_repository_root = Path(__file__).parents[1]
_ks_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "ks"


def test_ks_generator_emits_initial_fields_and_space_time_targets() -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = load_benchmark(_ks_benchmark_root).generator

    batch = generator(
        seed=101,
        shape=3,
        volume_request=StateSpaceVolumeRequest(2.0, 3.0),
        runtime=runtime,
    )
    fields, targets = batch.require_tensors()

    assert batch.region is not None
    assert batch.region.measure_estimate is not None
    assert batch.region.measure_estimate.kind == "estimated"
    assert batch.region.log2_volume == 2.0
    assert fields.shape == (3, 1, 128)
    assert targets.shape == (3, 1, 128)
    assert fields.dtype == runtime.torch.float32
    assert targets.dtype == runtime.torch.float32
    assert targets.allclose(fields.to(dtype=targets.dtype))


def test_ks_generator_threads_spatial_resolution() -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = cast(Any, load_benchmark(_ks_benchmark_root).generator)

    coarse = generator(
        seed=101,
        shape=2,
        sample_indices=(0, 1),
        spatial_points=32,
        runtime=runtime,
    )
    fine = generator(
        seed=101,
        shape=2,
        sample_indices=(0, 1),
        spatial_points=64,
        runtime=runtime,
    )
    coarse_fields, coarse_targets = coarse.require_tensors()
    fine_fields, fine_targets = fine.require_tensors()

    assert coarse_fields.shape == (2, 1, 32)
    assert fine_fields.shape == (2, 1, 64)
    assert fine.region is not None
    assert fine.region.ambient == coarse.region.ambient
    assert fine_fields[:, :, ::2].allclose(coarse_fields, atol=1e-6)
    assert fine_targets[:, :, ::2].allclose(coarse_targets, atol=1e-6)


def test_ks_generator_maps_volume_window_to_spatial_resolution() -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = cast(Any, load_benchmark(_ks_benchmark_root).generator)

    coarse = generator(
        seed=101,
        shape=2,
        volume_request=StateSpaceVolumeRequest(0.0, 1.0),
        runtime=runtime,
    )
    refined = generator(
        seed=101,
        shape=2,
        volume_request=StateSpaceVolumeRequest(1.0, 2.0),
        runtime=runtime,
    )
    coarse_fields, _coarse_targets = coarse.require_tensors()
    refined_fields, _refined_targets = refined.require_tensors()

    assert coarse_fields.shape == (2, 1, 32)
    assert refined_fields.shape == (2, 1, 64)
    assert refined.region is not None
    assert refined.region.ambient == coarse.region.ambient
    assert refined_fields[:, :, ::2].allclose(coarse_fields, atol=1e-6)


def test_ks_generator_samples_distinct_band_limited_mode_content() -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = cast(Any, load_benchmark(_ks_benchmark_root).generator)

    batch = generator(
        seed=101,
        shape=4,
        sample_indices=(0, 1, 2, 3),
        spatial_points=64,
        runtime=runtime,
    )
    fields, _targets = batch.require_tensors()
    spectrum = runtime.torch.fft.rfft(fields[:, 0, :], dim=-1).abs()
    low_mode_magnitudes = spectrum[:, 1:5]
    high_mode_magnitudes = spectrum[:, 5:]

    assert not low_mode_magnitudes[0].allclose(low_mode_magnitudes[1], atol=1e-4)
    assert high_mode_magnitudes.max() < low_mode_magnitudes.max() * 1e-5


def test_ks_generator_rejects_non_ladder_spatial_resolution() -> None:
    generator = cast(Any, load_benchmark(_ks_benchmark_root).generator)

    try:
        generator(seed=101, shape=1, spatial_points=48)
    except ObservationGenerationError:
        pass
    else:
        raise AssertionError("expected invalid spatial_points to be rejected")


def test_ks_generation_does_not_run_reference_solver(
) -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = load_benchmark(_ks_benchmark_root).generator
    module = sys.modules[type(generator).__module__]

    batch = generator(seed=101, shape=2, runtime=runtime)

    assert batch.fields is not None
    assert batch.targets is not None
    assert not hasattr(module, "tensor_runtime_solve_tensor_trajectory")


def test_ks_generator_samples_cartesian_fourier_chart_metadata() -> None:
    generator = load_benchmark(_ks_benchmark_root).generator

    batch = generator(
        seed=101,
        shape=2,
        volume_request=StateSpaceVolumeRequest(1.0, 2.0),
    )

    assert [sample.outcome_id for sample in batch.samples] == ["field", "field"]
    assert all(
        sample.latent_coordinates[0]["chart"] == "cartesian-fourier"
        for sample in batch.samples
    )
    coordinates = tuple(
        cast(float, sample.axis_coordinates["ks-space-time-log2-window"])
        for sample in batch.samples
        if sample.axis_coordinates is not None
    )
    assert len(coordinates) == 2
    assert all(1.0 <= coordinate < 2.0 for coordinate in coordinates)


def test_ks_benchmark_builds_residual_training_loss() -> None:
    runtime = resolve_tensor_runtime("cpu")
    loaded = load_benchmark(_ks_benchmark_root)
    batch = loaded.generator(seed=17, shape=2, runtime=runtime)
    fields, targets = batch.require_tensors()
    loss = cast(Any, loaded).build_training_loss(runtime, loaded.target_contract)

    trajectory = ks_reference_trajectory(
        runtime=runtime,
        sample_count=2,
        seed=17,
        sample_indices=(0, 1),
        window=0,
    )[:, 0, :, :].float()
    exact_loss = float(loss(trajectory, targets))
    perturbed = trajectory.clone()
    perturbed[:, 0:1, :] = fields + 0.5
    perturbed_loss = float(loss(perturbed, targets))

    assert math.isfinite(exact_loss)
    assert perturbed_loss > exact_loss


def test_ks_reference_residual_uses_resolution_dependent_dx() -> None:
    runtime = resolve_tensor_runtime("cpu")
    loaded = load_benchmark(_ks_benchmark_root)
    loss = cast(Any, loaded).build_training_loss(runtime, loaded.target_contract)
    losses: list[float] = []
    for spatial_points in (32, 64, 128):
        batch = cast(Any, loaded.generator)(
            seed=17,
            shape=1,
            sample_indices=(0,),
            runtime=runtime,
            spatial_points=spatial_points,
        )
        _fields, targets = batch.require_tensors()
        reference = ks_reference_trajectory(
            runtime=runtime,
            sample_count=1,
            seed=17,
            sample_indices=(0,),
            window=0,
            spatial_points=spatial_points,
        )[:, 0, :, :].float()
        losses.append(float(loss(reference, targets)))

    assert losses[1] <= losses[0]
    assert losses[2] <= losses[1]


def test_field_valued_runner_queries_operator_at_horizons() -> None:
    runtime = resolve_tensor_runtime("cpu")
    fields = runtime.torch.zeros((2, 1, 32), dtype=runtime.torch.float32)
    labels = runtime.torch.zeros((2, 4, 32), dtype=runtime.torch.float32)
    dts: list[float] = []

    class HorizonModule:
        def __call__(self, state: Any, dt: float) -> Any:
            dts.append(dt)
            return state + dt

    trajectory = _field_valued_model_trajectory(
        runtime=runtime,
        module=HorizonModule(),
        fields=fields,
        labels=labels,
        horizons=(1 / 3, 2 / 3, 1.0),
    )

    assert all(math.isclose(dt, 1 / 3) for dt in dts)
    assert trajectory.shape == labels.shape
    assert trajectory[:, 0:1, :].allclose(fields)
    assert trajectory[:, 1, :].allclose(fields[:, 0, :] + (1 / 3))
    assert trajectory[:, 3, :].allclose(fields[:, 0, :] + 1.0)


def test_field_valued_runner_rejects_length_changing_operator() -> None:
    runtime = resolve_tensor_runtime("cpu")
    fields = runtime.torch.zeros((2, 1, 32), dtype=runtime.torch.float32)
    labels = runtime.torch.zeros((2, 4, 32), dtype=runtime.torch.float32)

    class BadModule:
        def __call__(self, state: Any, _horizon: float) -> Any:
            return state[:, :, :-1]

    try:
        _field_valued_model_trajectory(
            runtime=runtime,
            module=BadModule(),
            fields=fields,
            labels=labels,
            horizons=(1 / 3, 2 / 3, 1.0),
        )
    except BenchmarkRunnerError as error:
        assert "must return state shape" in str(error)
    else:
        raise AssertionError("expected length-changing field operator to be rejected")


def test_field_valued_runner_presents_dt_to_real_operator() -> None:
    runtime = resolve_tensor_runtime("cpu")

    class StepModule:
        def __call__(self, state: Any, dt: float) -> Any:
            return state + dt

    fields = runtime.torch.zeros((2, 1, 32), dtype=runtime.torch.float32)
    labels = runtime.torch.zeros((2, 4, 32), dtype=runtime.torch.float32)

    trajectory = _field_valued_model_trajectory(
        runtime=runtime,
        module=StepModule(),
        fields=fields,
        labels=labels,
        horizons=(0.25, 0.5, 1.0),
    )

    assert trajectory.shape == labels.shape
    assert trajectory[:, 0:1, :].allclose(fields)
    assert trajectory[:, 1, :].allclose(fields[:, 0, :] + 0.25)
    assert trajectory[:, 2, :].allclose(fields[:, 0, :] + 0.5)
    assert trajectory[:, 3, :].allclose(fields[:, 0, :] + 1.0)


def test_field_valued_runner_rejects_operator_without_dt_argument() -> None:
    runtime = resolve_tensor_runtime("cpu")

    class BadModule:
        def __call__(self, state: Any) -> Any:
            return state

    fields = runtime.torch.zeros((2, 1, 32), dtype=runtime.torch.float32)
    labels = runtime.torch.zeros((2, 4, 32), dtype=runtime.torch.float32)

    try:
        _field_valued_model_trajectory(
            runtime=runtime,
            module=BadModule(),
            fields=fields,
            labels=labels,
            horizons=(0.25, 0.5, 1.0),
        )
    except BenchmarkRunnerError as error:
        assert "must accept an input state and dt" in str(error)
    else:
        raise AssertionError("expected field operator without dt argument to fail")


def test_ks_convergence_bits_rejects_persistence() -> None:
    runtime = resolve_tensor_runtime("cpu")
    loaded = load_benchmark(_ks_benchmark_root)
    batch = cast(Any, loaded.generator)(seed=17, shape=1, runtime=runtime)
    fields, targets = batch.require_tensors()
    competence = cast(Any, loaded).build_training_competence(
        runtime,
        loaded.target_contract,
    )

    bits = competence(
        SimpleNamespace(
            runtime=runtime,
            module=None,
            generator=None,
            batch=batch,
            sample_keys=tuple(sample.to_record() for sample in batch.samples),
            predictions=fields.repeat(1, targets.shape[1], 1),
            targets=targets,
            horizons=(1.0,),
        )
    )

    assert float(bits[0]) == 0.0


def test_ks_convergence_bits_reward_planted_convergent_ladder(monkeypatch: Any) -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _loaded_ks_module()
    ladder = _planted_field_ladder(runtime, coefficient=0.25)
    residual_by_space = {
        int(trajectory.shape[-1]): value
        for trajectory, value in zip(ladder, (4.0, 1.0, 0.25), strict=True)
    }

    def planted_residual(trajectory: Any, *, dx: float, dt: float) -> Any:
        _ = dx
        _ = dt
        return trajectory * 0.0 + residual_by_space[int(trajectory.shape[-1])]

    monkeypatch.setattr(module, "ks_space_time_residual", planted_residual)

    bits = module._ks_ladder_convergence_bits(
        runtime=runtime,
        ladder=ladder,
        horizon=1.0,
    )
    diagnostics = bits.leibniz_competence_diagnostics

    assert float(bits[0]) > 0.0
    assert diagnostics[0]["kind"] == "ks-convergence-diagnostics"
    assert diagnostics[0]["gate_decision"] == "passed"
    assert len(diagnostics[0]["k_sensitivity"]) >= 2
    assert diagnostics[0]["predictability_boundary"] == 1.0
    expected_bits = cast(int, diagnostics[0]["node_count"]) * math.log2(
        cast(float, diagnostics[0]["evolution_scale"])
        / cast(float, diagnostics[0]["field_error"])
    )
    assert math.isclose(float(bits[0]), expected_bits)


def test_ks_convergence_bits_reward_finer_field_resolution(monkeypatch: Any) -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _loaded_ks_module()

    def planted_residual(trajectory: Any, *, dx: float, dt: float) -> Any:
        _ = dx
        _ = dt
        return trajectory * 0.0 + {4: 4.0, 8: 1.0, 16: 0.25}[int(trajectory.shape[-1])]

    monkeypatch.setattr(module, "ks_space_time_residual", planted_residual)

    coarse_bits = module._ks_ladder_convergence_bits(
        runtime=runtime,
        ladder=_planted_field_ladder(runtime, coefficient=1.0),
        horizon=1.0,
    )
    fine_bits = module._ks_ladder_convergence_bits(
        runtime=runtime,
        ladder=_planted_field_ladder(runtime, coefficient=0.125),
        horizon=1.0,
    )

    assert float(fine_bits[0]) > float(coarse_bits[0])


def test_ks_convergence_bits_stop_at_first_time_gate_failure(monkeypatch: Any) -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _loaded_ks_module()
    ladder = _planted_field_ladder(runtime, coefficient=0.25)
    residual_by_time_and_space = {
        (2, 4): 4.0,
        (3, 8): 1.0,
        (5, 16): 0.25,
        (3, 4): 4.0,
        (5, 8): 2.0,
        (9, 16): 1.0,
    }

    def planted_residual(trajectory: Any, *, dx: float, dt: float) -> Any:
        _ = dx
        _ = dt
        key = (int(trajectory.shape[1]), int(trajectory.shape[-1]))
        return trajectory * 0.0 + residual_by_time_and_space[key]

    monkeypatch.setattr(module, "ks_space_time_residual", planted_residual)

    bits = module._ks_ladder_convergence_bits(
        runtime=runtime,
        ladder=ladder,
        horizon=1.0,
    )
    diagnostics = bits.leibniz_competence_diagnostics
    time_points = diagnostics[0]["time_points"]

    assert float(bits[0]) > 0.0
    assert diagnostics[0]["predictability_boundary"] == 0.5
    assert time_points[0]["gate_decision"] == "passed"
    assert time_points[1]["gate_decision"] == "failed"
    assert time_points[1]["bits"] == 0.0


def test_ks_convergence_bits_require_evolution_beyond_initial_state(
    monkeypatch: Any,
) -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _loaded_ks_module()
    torch = runtime.torch
    initial = torch.linspace(0.0, 1.0, 4, dtype=torch.float32, device=runtime.device).reshape(
        1,
        1,
        4,
    )
    ladder = tuple(
        initial.repeat_interleave(2**rung, dim=-1).repeat(1, 3 * (2**rung), 1)
        for rung in range(3)
    )

    def planted_residual(trajectory: Any, *, dx: float, dt: float) -> Any:
        _ = dx
        _ = dt
        return trajectory * 0.0 + {4: 4.0, 8: 1.0, 16: 0.25}[int(trajectory.shape[-1])]

    monkeypatch.setattr(module, "ks_space_time_residual", planted_residual)

    bits = module._ks_ladder_convergence_bits(
        runtime=runtime,
        ladder=ladder,
        horizon=1.0,
    )
    diagnostics = bits.leibniz_competence_diagnostics

    assert float(bits[0]) == 0.0
    assert diagnostics[0]["gate_decision"] == "failed"
    assert diagnostics[0]["evolution_scale"] == 0.0


def test_ks_convergence_bits_rejects_wrong_observed_order(monkeypatch: Any) -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _loaded_ks_module()

    def planted_residual(trajectory: Any, *, dx: float, dt: float) -> Any:
        _ = dx
        _ = dt
        return trajectory * 0.0 + {4: 4.0, 8: 2.0, 16: 1.0}[int(trajectory.shape[-1])]

    monkeypatch.setattr(module, "ks_space_time_residual", planted_residual)

    bits = module._ks_ladder_convergence_bits(
        runtime=runtime,
        ladder=_planted_field_ladder(runtime, coefficient=0.25),
        horizon=1.0,
    )
    diagnostics = bits.leibniz_competence_diagnostics

    assert float(bits[0]) == 0.0
    assert diagnostics[0]["gate_decision"] == "failed"
    assert diagnostics[0]["expected_observed_order"] == 2.0
    assert diagnostics[0]["rung_count"] == 3


def test_ks_scoring_path_has_no_residual_certificate_constants() -> None:
    source = (_ks_benchmark_root / "benchmark.py").read_text()

    assert "_ks_truncation_floor" not in source
    assert "_ks_epsilon_residual_image_bound" not in source
    assert "_ks_residual_acceptance_level" not in source
    assert "_ks_residual_certificate_mass" not in source


def _loaded_ks_module() -> Any:
    loaded = cast(Any, load_benchmark(_ks_benchmark_root))
    return sys.modules[type(loaded.implementation).__module__]


def _planted_field_ladder(runtime: Any, *, coefficient: float) -> tuple[Any, ...]:
    torch = runtime.torch
    base = torch.arange(12, dtype=torch.float32, device=runtime.device).reshape(1, 3, 4)
    ladder: list[Any] = []
    for rung in range(3):
        scale = 2**rung
        field = base.repeat_interleave(scale, dim=1).repeat_interleave(scale, dim=2)
        field = field + coefficient / (4.0**rung)
        ladder.append(field)
    return tuple(ladder)
