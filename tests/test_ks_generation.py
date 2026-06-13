import math
import sys
from pathlib import Path
from typing import Any, cast

from leibniz.benchmark_implementations import load_benchmark
from leibniz.observation_generation import StateSpaceVolumeRequest
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
    assert fields.shape == (3, 1, 32)
    assert targets.shape == (3, 9, 32)
    assert fields.dtype == runtime.torch.float32
    assert targets.dtype == runtime.torch.float32
    assert targets[:, 0:1, :].allclose(fields.to(dtype=targets.dtype))
    assert targets[:, 1:, :].allclose(
        fields.to(dtype=targets.dtype).repeat(1, targets.shape[1] - 1, 1)
    )


def test_ks_generation_does_not_run_reference_solver(
    monkeypatch: Any,
) -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = load_benchmark(_ks_benchmark_root).generator
    module = sys.modules[type(generator).__module__]

    def fail_solver(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("generation must not run the reference solver")

    monkeypatch.setattr(module, "tensor_runtime_solve_tensor_trajectory", fail_solver)

    batch = generator(seed=101, shape=2, runtime=runtime)

    assert batch.fields is not None
    assert batch.targets is not None


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

    exact_loss = float(loss(targets, targets))
    perturbed = targets.clone()
    perturbed[:, 0:1, :] = fields + 0.5
    perturbed_loss = float(loss(perturbed, targets))

    assert math.isfinite(exact_loss)
    assert perturbed_loss > exact_loss


def test_ks_residual_certificate_scores_reference_and_rejects_bad_solutions() -> None:
    runtime = resolve_tensor_runtime("cpu")
    loaded = load_benchmark(_ks_benchmark_root)
    batch = loaded.generator(seed=17, shape=1, runtime=runtime)
    fields, targets = batch.require_tensors()
    competence = cast(Any, loaded).build_training_competence(
        runtime,
        loaded.target_contract,
    )
    reference = cast(Any, loaded).reference_trajectory(
        runtime=runtime,
        sample_count=1,
        seed=17,
        sample_indices=(0,),
        window=0,
    )[:, 0, :, :].float()
    zero = reference * 0.0

    assert float(competence(reference, targets)[0]) == 1.0
    assert float(competence(reference + (_epsilon() / 2.0), targets)[0]) == 1.0
    assert float(competence(reference + (2.0 * _epsilon()), targets)[0]) == 0.0
    assert float(competence(zero, targets)[0]) == 0.0
    assert float(competence(targets, targets)[0]) == 0.0
    assert fields.shape == (1, 1, 32)


def test_ks_residual_certificate_does_not_compare_against_reference_target() -> None:
    runtime = resolve_tensor_runtime("cpu")
    loaded = load_benchmark(_ks_benchmark_root)
    first = loaded.generator(seed=17, shape=1, sample_indices=(0,), runtime=runtime)
    second = loaded.generator(seed=18, shape=1, sample_indices=(0,), runtime=runtime)
    first_reference = cast(Any, loaded).reference_trajectory(
        runtime=runtime,
        sample_count=1,
        seed=17,
        sample_indices=(0,),
        window=0,
    )[:, 0, :, :].float()
    second_reference = cast(Any, loaded).reference_trajectory(
        runtime=runtime,
        sample_count=1,
        seed=18,
        sample_indices=(0,),
        window=0,
    )[:, 0, :, :].float()
    predictions = runtime.torch.cat((first_reference, second_reference), dim=0)
    _first_fields, first_targets = first.require_tensors()
    _second_fields, second_targets = second.require_tensors()
    targets = runtime.torch.cat((first_targets, second_targets), dim=0)
    competence = cast(Any, loaded).build_training_competence(
        runtime,
        loaded.target_contract,
    )
    reference_relative_error = (
        (first_reference - second_reference).pow(2).sum().sqrt()
        / second_reference.pow(2).sum().sqrt().clamp_min(1e-12)
    )

    masses = competence(predictions, targets)

    assert float(reference_relative_error) > 0.0
    assert [float(value) for value in masses] == [1.0, 1.0]


def _epsilon() -> float:
    return 0.05
