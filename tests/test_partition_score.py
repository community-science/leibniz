import math
from pathlib import Path

from leibniz.observation_generation import StateSpaceVolumeRequest, load_generator
from leibniz.partition_score import (
    PartitionSample,
    fixed_partition_competence_integral,
    partition_samples_from_generated,
)
from leibniz.state_space import (
    DiscreteAxisRegion,
    Distinguishability,
    IntegerRangeDomain,
    ProductRegion,
    StateSpaceAmbient,
    StateSpaceAxis,
    StateSpaceRegion,
)


def test_fixed_partition_integral_matches_analytic_measure_weighted_mean() -> None:
    root = _grid_region("grid.root", x=(0, 1), y=(0, 1))
    left = _grid_region("grid.x0", x=(0, 0), y=(0, 1))
    right = _grid_region("grid.x1", x=(1, 1), y=(0, 1))
    samples = _grid_samples(root)

    score = fixed_partition_competence_integral(
        root_region=root,
        samples=samples,
        partition=(left, right),
    )

    assert score.sample_count == 4
    assert score.unassigned_sample_count == 0
    assert math.isclose(score.value, 1.0)
    assert [child.estimate.sample_count for child in score.root.children] == [2, 2]


def test_fixed_partition_integral_is_partition_independent_for_census_samples() -> None:
    root = _grid_region("grid.root", x=(0, 1), y=(0, 1))
    by_x = (
        _grid_region("grid.x0", x=(0, 0), y=(0, 1)),
        _grid_region("grid.x1", x=(1, 1), y=(0, 1)),
    )
    by_y = (
        _grid_region("grid.y0", x=(0, 1), y=(0, 0)),
        _grid_region("grid.y1", x=(0, 1), y=(1, 1)),
    )
    samples = _grid_samples(root)

    score_by_x = fixed_partition_competence_integral(
        root_region=root,
        samples=samples,
        partition=by_x,
    )
    score_by_y = fixed_partition_competence_integral(
        root_region=root,
        samples=samples,
        partition=by_y,
    )

    assert math.isclose(score_by_x.value, score_by_y.value)
    assert math.isclose(score_by_x.value, 1.0)


def test_fixed_partition_accepts_ks_generated_answer_coordinates() -> None:
    generator = load_generator(Path("src/leibniz/benchmarks/ks"))
    batch = generator(
        seed=17,
        shape=8,
        include_metadata=True,
        volume_request=StateSpaceVolumeRequest(minimum=1.0, maximum=1.0),
    )
    assert batch.region is not None
    competence = {sample.index: 0.25 for sample in batch.samples}

    score = fixed_partition_competence_integral(
        root_region=batch.region,
        samples=partition_samples_from_generated(batch.samples, competence),
        partition=(batch.region,),
    )

    assert score.sample_count == 8
    assert score.unassigned_sample_count == 0
    assert math.isclose(score.value, 0.25)


def test_fixed_partition_accepts_inverse_digits_natural_components() -> None:
    generator = load_generator(Path("src/leibniz/benchmarks/digits"))
    batch = generator(
        seed=23,
        shape=2,
        include_metadata=True,
        volume_request=StateSpaceVolumeRequest(minimum=1.0, maximum=2.0),
        sample_indices=(0, 1),
    )
    assert batch.region is not None
    partition = _component_partition(batch.region)
    competence = {
        sample.index: float(sample.component_index or 0) for sample in batch.samples
    }

    score = fixed_partition_competence_integral(
        root_region=batch.region,
        samples=partition_samples_from_generated(batch.samples, competence),
        partition=partition,
    )

    expected = math.fsum(
        child.estimate.measure * child.estimate.competence
        for child in score.root.children
    ) / score.total_measure
    assert score.sample_count == 2
    assert score.unassigned_sample_count == 0
    assert math.isclose(score.value, expected)


def _grid_samples(root: StateSpaceRegion) -> tuple[PartitionSample, ...]:
    samples: list[PartitionSample] = []
    index = 0
    for x in (0, 1):
        for y in (0, 1):
            samples.append(
                PartitionSample(
                    sample_index=index,
                    competence=float(x + y),
                    region_component_index=0,
                    axis_coordinates={"x": x, "y": y},
                )
            )
            index += 1
    assert all(root.contains(sample.region_component_index, sample.axis_coordinates) for sample in samples)
    return tuple(samples)


def _grid_region(
    region_id: str,
    *,
    x: tuple[int, int],
    y: tuple[int, int],
) -> StateSpaceRegion:
    x_axis = StateSpaceAxis(id="x", domain=IntegerRangeDomain(lower=0, upper=1))
    y_axis = StateSpaceAxis(id="y", domain=IntegerRangeDomain(lower=0, upper=1))
    x_count = x[1] - x[0] + 1
    y_count = y[1] - y[0] + 1
    volume = x_count * y_count
    return StateSpaceRegion(
        id=region_id,
        ambient=_grid_ambient(),
        components=(
            ProductRegion(
                axis_regions=(
                    DiscreteAxisRegion(
                        axis=x_axis,
                        coordinate_region=x,
                        count=x_count,
                        log2_count=math.log2(x_count),
                    ),
                    DiscreteAxisRegion(
                        axis=y_axis,
                        coordinate_region=y,
                        count=y_count,
                        log2_count=math.log2(y_count),
                    ),
                ),
                measure_rule="product-of-counts",
                volume=volume,
                log2_volume=math.log2(volume),
            ),
        ),
        union_rule="disjoint-union",
        volume=volume,
        log2_volume=math.log2(volume),
    )


def _grid_ambient() -> StateSpaceAmbient:
    return StateSpaceAmbient(
        field_domain_kind="lattice-2d",
        field_domain={"height": 2, "width": 2},
        field_codomain_id="fixture",
        distinguishability=Distinguishability(kind="exact"),
    )


def _component_partition(region: StateSpaceRegion) -> tuple[StateSpaceRegion, ...]:
    return tuple(
        StateSpaceRegion(
            id=f"{region.id}.component-{index}",
            ambient=region.ambient,
            components=(component,),
            union_rule=region.union_rule,
            volume=component.volume,
            log2_volume=component.log2_volume,
            measure_estimate=component.measure_estimate,
        )
        for index, component in enumerate(region.components)
    )
