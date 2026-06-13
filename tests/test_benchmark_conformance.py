import math
from pathlib import Path
from typing import cast

import pytest

from leibniz.benchmark_implementations import discover_benchmark_roots, load_benchmark
from leibniz.observation_generation import GeneratedSampleSet, StateSpaceVolumeRequest
from leibniz.state_space import (
    DiscreteAxisRegion,
    Distinguishability,
    IntegerRangeDomain,
    MeasureEstimate,
    ProductRegion,
    StateSpaceAmbient,
    StateSpaceAxis,
    StateSpaceRegion,
    state_space_region_contains,
    state_space_region_from_record,
    state_space_regions_are_disjoint,
)

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
    """Assert the cross-benchmark integer-window volume law.

    Every packaged benchmark must realize the integer bit window ``[k, k+1]``
    as a disjoint increment. Exact measures must realize exactly ``2**k`` new
    states for ``k >= 1``, with window ``[0, 1]`` realizing the single origin
    state, anchoring the cumulative realized state count at
    ``N(L) = 2**L - 1`` after level ``L``. Estimated measures must instead
    declare a bracket containing ``k`` bits and keep cumulative declared volume
    inside the summed bracket interval. This is protocol law, not a benchmark
    convention: it makes integer windows mean the same realized volumes or
    estimated volume brackets on every benchmark, so scores integrated along
    the bits axis are comparable across benchmarks.
    """

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
    _assert_integer_window_law(realized_regions)


@pytest.mark.parametrize("benchmark_root", _benchmark_roots(), ids=lambda path: path.name)
def test_packaged_benchmarks_realized_windows_lie_in_accessible_subspace(
    benchmark_root: Path,
) -> None:
    benchmark = load_benchmark(benchmark_root)
    batches = tuple(
        benchmark.generator(
            seed=407 + index,
            shape=4,
            volume_request=StateSpaceVolumeRequest(float(index), float(index + 1)),
        )
        for index in range(4)
    )
    realized_regions = tuple(batch.region for batch in batches if batch.region is not None)

    assert realized_regions
    for region in realized_regions:
        assert state_space_region_contains(
            benchmark.accessible_subspace.per_configuration_capacity,
            region,
        )
        for exclusion in benchmark.accessible_subspace.exclusions:
            assert state_space_regions_are_disjoint(region, exclusion)


@pytest.mark.parametrize("benchmark_root", _benchmark_roots(), ids=lambda path: path.name)
def test_exact_packaged_benchmark_windows_have_census_saturation(
    benchmark_root: Path,
) -> None:
    benchmark = load_benchmark(benchmark_root)
    batch = benchmark.generator(
        seed=407,
        shape=4,
        volume_request=StateSpaceVolumeRequest(0.0, 1.0),
    )

    assert batch.region is not None
    if batch.region.ambient.distinguishability.kind != "exact":
        return
    protocol = benchmark.sampling_protocol
    assert protocol.census_budget is not None
    assert protocol.census_budget >= batch.region.volume


def test_integer_window_law_accepts_estimated_fixture_branch() -> None:
    regions = (
        _fixture_exact_window_region("fixture-window-0", lower=0, upper=0),
        _fixture_estimated_window_region(
            "fixture-window-1",
            lower=1,
            upper=4,
            volume=3,
            log2_lower=1.0,
            log2_upper=2.0,
        ),
    )

    _assert_integer_window_law(regions)


def test_integer_window_law_rejects_estimated_fixture_excluding_window_bits() -> None:
    regions = (
        _fixture_exact_window_region("fixture-window-0", lower=0, upper=0),
        _fixture_estimated_window_region(
            "fixture-window-1",
            lower=1,
            upper=4,
            volume=3,
            log2_lower=1.1,
            log2_upper=2.0,
        ),
    )

    with pytest.raises(AssertionError):
        _assert_integer_window_law(regions)


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


def _assert_integer_window_law(regions: tuple[StateSpaceRegion, ...]) -> None:
    for left_index, left in enumerate(regions):
        for right in regions[left_index + 1 :]:
            assert state_space_regions_are_disjoint(left, right)

    cumulative_volume = 0
    cumulative_lower = 0.0
    cumulative_upper = 0.0
    seen_estimated = False
    for index, region in enumerate(regions):
        interval_lower, interval_upper = _integer_window_measure_interval(region)
        cumulative_lower += interval_lower
        cumulative_upper += interval_upper
        cumulative_volume += region.volume
        if _region_has_estimated_measure(region):
            seen_estimated = True
            estimate = region.measure_estimate
            assert estimate is not None
            assert estimate.log2_lower is not None
            assert estimate.log2_upper is not None
            assert estimate.log2_lower <= float(index) <= estimate.log2_upper
        else:
            assert region.volume == _integer_window_increment_volume(index)
            assert math.isclose(
                region.log2_volume,
                math.log2(region.volume),
                rel_tol=0.0,
                abs_tol=1e-9,
            )

        if seen_estimated:
            assert cumulative_lower <= cumulative_volume <= cumulative_upper
        else:
            assert cumulative_volume == _integer_window_cumulative_volume(index + 1)


def _integer_window_measure_interval(region: StateSpaceRegion) -> tuple[float, float]:
    if _region_has_estimated_measure(region):
        estimate = region.measure_estimate
        assert estimate is not None
        assert estimate.log2_lower is not None
        assert estimate.log2_upper is not None
        return 2.0**estimate.log2_lower, 2.0**estimate.log2_upper
    return float(region.volume), float(region.volume)


def _region_has_estimated_measure(region: StateSpaceRegion) -> bool:
    return region.measure_estimate is not None and region.measure_estimate.kind == "estimated"


def _fixture_ambient() -> StateSpaceAmbient:
    return StateSpaceAmbient(
        field_domain_kind="lattice-2d",
        field_domain={"height": 1, "width": 1},
        field_codomain_id="scalar-field",
        distinguishability=Distinguishability(kind="exact"),
    )


def _fixture_axis_region(*, lower: int, upper: int) -> DiscreteAxisRegion:
    axis = StateSpaceAxis(id="fixture-index", domain=IntegerRangeDomain(lower=0, upper=8))
    count = upper - lower + 1
    return DiscreteAxisRegion(
        axis=axis,
        coordinate_region=(lower, upper),
        count=count,
        log2_count=math.log2(count),
    )


def _fixture_exact_window_region(
    region_id: str,
    *,
    lower: int,
    upper: int,
) -> StateSpaceRegion:
    axis_region = _fixture_axis_region(lower=lower, upper=upper)
    component = ProductRegion(
        axis_regions=(axis_region,),
        measure_rule="product-of-counts",
        volume=axis_region.count,
        log2_volume=axis_region.log2_count,
    )
    return StateSpaceRegion(
        id=region_id,
        ambient=_fixture_ambient(),
        components=(component,),
        union_rule="disjoint-union",
        volume=axis_region.count,
        log2_volume=axis_region.log2_count,
    )


def _fixture_estimated_window_region(
    region_id: str,
    *,
    lower: int,
    upper: int,
    volume: int,
    log2_lower: float,
    log2_upper: float,
) -> StateSpaceRegion:
    estimate = MeasureEstimate(
        kind="estimated",
        method_id="covering-number-grid-bound",
        log2_lower=log2_lower,
        log2_upper=log2_upper,
    )
    axis_region = _fixture_axis_region(lower=lower, upper=upper)
    component = ProductRegion(
        axis_regions=(axis_region,),
        measure_rule="benchmark-computed-finite-count",
        volume=volume,
        log2_volume=math.log2(volume),
        measure_estimate=estimate,
    )
    return StateSpaceRegion(
        id=region_id,
        ambient=_fixture_ambient(),
        components=(component,),
        union_rule="disjoint-union",
        volume=volume,
        log2_volume=math.log2(volume),
        measure_estimate=estimate,
    )


def _integer_window_increment_volume(index: int) -> int:
    if index == 0:
        return 1
    return 2**index


def _integer_window_cumulative_volume(upper_index: int) -> int:
    if upper_index <= 0:
        return 0
    return 2**upper_index - 1
