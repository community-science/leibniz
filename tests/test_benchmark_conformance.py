import math
from pathlib import Path
from typing import cast

import pytest

from leibniz.benchmark_implementations import discover_benchmark_roots, load_benchmark
from leibniz.observation_generation import GeneratedSampleSet, StateSpaceVolumeRequest
from leibniz.state_space import state_space_region_from_record, state_space_regions_are_disjoint

_repository_root = Path(__file__).parents[1]
_benchmark_parent = _repository_root / "src" / "leibniz" / "benchmarks"


def _benchmark_roots() -> tuple[Path, ...]:
    return discover_benchmark_roots(_benchmark_parent)


@pytest.mark.parametrize("benchmark_root", _benchmark_roots(), ids=lambda path: path.name)
@pytest.mark.parametrize(
    "volume_request",
    (StateSpaceVolumeRequest(0.0, 1.0), StateSpaceVolumeRequest(3.0, 4.0)),
)
def test_packaged_benchmarks_emit_region_authoritative_realized_windows(
    benchmark_root: Path,
    volume_request: StateSpaceVolumeRequest,
) -> None:
    generator = load_benchmark(benchmark_root).generator

    batch = generator(seed=407, shape=4, volume_request=volume_request)

    assert batch.request_outcome is not None
    assert batch.request_outcome.kind == "realized"
    assert batch.region is not None
    assert batch.request_outcome.region == batch.region
    assert math.isclose(batch.log2_volume, batch.region.log2_volume)
    assert volume_request.contains(batch.log2_volume)
    assert _state_count(batch) == batch.region.volume
    assert batch.samples
    for sample in batch.samples:
        assert sample.region_component_index is not None
        assert sample.axis_coordinates is not None
        assert batch.region.contains(sample.region_component_index, sample.axis_coordinates)

    record = batch.to_record(include_fields=True)
    assert record["log2_volume"] == batch.region.log2_volume
    region_record = cast(dict[str, object], record["region"])
    assert state_space_region_from_record(region_record) == batch.region
    for sample_record in cast(list[dict[str, object]], record["samples"]):
        assert "log2_volume" not in sample_record
        assert "volume_value" not in sample_record
        assert "region_component_index" in sample_record
        assert "axis_coordinates" in sample_record


@pytest.mark.parametrize("benchmark_root", _benchmark_roots(), ids=lambda path: path.name)
def test_packaged_benchmarks_realize_disjoint_integer_window_increments(
    benchmark_root: Path,
) -> None:
    generator = load_benchmark(benchmark_root).generator
    batches = tuple(
        generator(
            seed=407 + index,
            shape=4,
            volume_request=StateSpaceVolumeRequest(float(index), float(index + 1)),
        )
        for index in range(4)
    )
    regions = tuple(batch.region for batch in batches)

    for batch in batches:
        assert batch.request_outcome is not None
        assert batch.request_outcome.kind == "realized"
    assert all(region is not None for region in regions)
    realized_regions = tuple(region for region in regions if region is not None)
    for left_index, left in enumerate(realized_regions):
        for right in realized_regions[left_index + 1 :]:
            assert state_space_regions_are_disjoint(left, right)

    cumulative_volume = 0
    for index, region in enumerate(realized_regions):
        assert region.volume == _integer_window_increment_volume(index)
        cumulative_volume += region.volume
        assert cumulative_volume == _integer_window_cumulative_volume(index + 1)


@pytest.mark.parametrize("benchmark_root", _benchmark_roots(), ids=lambda path: path.name)
def test_packaged_benchmarks_distinguish_unrealized_window_outcomes(
    benchmark_root: Path,
) -> None:
    generator = load_benchmark(benchmark_root).generator

    batch = generator(
        seed=407,
        shape=4,
        volume_request=StateSpaceVolumeRequest(1.5, 1.5),
    )

    assert batch.shape == (0,)
    assert batch.samples == ()
    assert batch.request_outcome is not None
    assert batch.request_outcome.kind in {
        "exhausted-capacity",
        "unrepresentable-below-minimum",
    }
    if batch.request_outcome.kind == "exhausted-capacity":
        assert batch.request_outcome.capacity_region is not None


def _state_count(batch: GeneratedSampleSet) -> int:
    state_count = round(2**batch.log2_volume)
    assert state_count > 0
    return state_count


def _integer_window_increment_volume(index: int) -> int:
    if index == 0:
        return 1
    return 2**index


def _integer_window_cumulative_volume(upper_index: int) -> int:
    if upper_index <= 0:
        return 0
    return 2**upper_index - 1
