from pathlib import Path
from typing import cast

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
    assert targets.shape == (3, 1, 9, 32)
    assert fields.dtype == runtime.torch.float32
    assert targets.dtype == runtime.torch.float32
    assert targets[:, :, 0, :].allclose(fields)


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
