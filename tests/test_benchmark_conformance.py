import math
from pathlib import Path
from typing import cast

import pytest

from leibniz.benchmark_implementations import discover_benchmark_roots, load_benchmark
from leibniz.observation_generation import ComplexityRequest, GeneratedSampleSet
from leibniz.state_space import state_space_region_from_record

_repository_root = Path(__file__).parents[1]
_benchmark_parent = _repository_root / "src" / "leibniz" / "benchmarks"


def _benchmark_roots() -> tuple[Path, ...]:
    return discover_benchmark_roots(_benchmark_parent)


@pytest.mark.parametrize("benchmark_root", _benchmark_roots(), ids=lambda path: path.name)
@pytest.mark.parametrize(
    "complexity_request",
    (ComplexityRequest(0.0, 1.0), ComplexityRequest(3.0, 4.0)),
)
def test_packaged_benchmarks_emit_region_authoritative_realized_windows(
    benchmark_root: Path,
    complexity_request: ComplexityRequest,
) -> None:
    generator = load_benchmark(benchmark_root).generator

    batch = generator(seed=407, shape=4, complexity_request=complexity_request)

    assert batch.request_outcome is not None
    assert batch.request_outcome.kind == "realized"
    assert batch.region is not None
    assert batch.request_outcome.region == batch.region
    assert math.isclose(batch.complexity, batch.region.log2_volume)
    assert complexity_request.contains(batch.complexity)
    assert _state_count(batch) == batch.region.volume
    assert batch.samples
    for sample in batch.samples:
        assert sample.region_component_index is not None
        assert sample.axis_coordinates is not None
        assert batch.region.contains(sample.region_component_index, sample.axis_coordinates)

    record = batch.to_record(include_fields=True)
    assert record["complexity"] == batch.region.log2_volume
    region_record = cast(dict[str, object], record["region"])
    assert state_space_region_from_record(region_record) == batch.region
    for sample_record in cast(list[dict[str, object]], record["samples"]):
        assert "complexity" not in sample_record
        assert "complexity_value" not in sample_record
        assert "region_component_index" in sample_record
        assert "axis_coordinates" in sample_record


@pytest.mark.parametrize("benchmark_root", _benchmark_roots(), ids=lambda path: path.name)
def test_packaged_benchmarks_distinguish_unrealized_window_outcomes(
    benchmark_root: Path,
) -> None:
    generator = load_benchmark(benchmark_root).generator

    batch = generator(
        seed=407,
        shape=4,
        complexity_request=ComplexityRequest(1.5, 1.5),
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
    state_count = round(2**batch.complexity)
    assert state_count > 0
    return state_count
