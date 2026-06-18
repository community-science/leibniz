import math
from pathlib import Path

from leibniz.benchmark_implementations import Generator
from leibniz.observation_generation import StateSpaceVolumeRequest, load_generator
from leibniz.partition_score import (
    PartitionSample,
    PartitionScore,
    adversarial_partition_competence_integral,
    fixed_partition_competence_integral,
    partition_samples_from_generated,
)
from leibniz.state_space import (
    ContinuousAxisRegion,
    DiscreteAxisRegion,
    Distinguishability,
    IntegerRangeDomain,
    MeasureEstimate,
    ProductRegion,
    RealIntervalDomain,
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
    assert math.isclose(score.mean_competence, 1.0)
    assert math.isclose(score.value, 2.0)
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

    assert math.isclose(score_by_x.mean_competence, score_by_y.mean_competence)
    assert math.isclose(score_by_x.value, score_by_y.value)
    assert math.isclose(score_by_x.mean_competence, 1.0)
    assert math.isclose(score_by_x.value, 2.0)


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
    assert math.isclose(score.mean_competence, 0.25)
    assert math.isclose(score.value, 0.25 * score.score_width_bits)


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
    assert math.isclose(score.mean_competence, expected)
    assert math.isclose(score.value, expected * score.score_width_bits)


def test_adversarial_partition_isolates_planted_failure_pocket() -> None:
    root = _line_region("line.root", lower=0, upper=15)
    samples = _line_samples(root, repetitions=4, pocket=range(0, 2))

    score = adversarial_partition_competence_integral(
        root_region=root,
        samples=samples,
    )

    leaves = score.root.leaves()
    pocket_leaves = [
        leaf
        for leaf in leaves
        if leaf.estimate.region.volume == 2
        and math.isclose(leaf.estimate.competence, 0.0)
    ]
    assert len(pocket_leaves) == 1
    assert math.isclose(score.mean_competence, 14 / 16)
    assert math.isclose(score.value, 3.5)
    assert len(score.refinement_ladder) >= 2
    assert score.refinement_ladder[-1].movement == 0.0


def test_adversarial_partition_does_not_split_unstructured_noise() -> None:
    root = _line_region("line.root", lower=0, upper=15)
    samples: list[PartitionSample] = []
    index = 0
    for coordinate in range(16):
        for competence in (0.0, 1.0, 0.0, 1.0):
            samples.append(
                PartitionSample(
                    sample_index=index,
                    competence=competence,
                    region_component_index=0,
                    axis_coordinates={"x": coordinate},
                )
            )
            index += 1

    score = adversarial_partition_competence_integral(
        root_region=root,
        samples=tuple(samples),
    )

    assert score.root.children == ()
    assert len(score.refinement_ladder) == 1
    assert math.isclose(score.mean_competence, 0.5)
    assert math.isclose(score.value, 2.0)


def test_adversarial_partition_runs_on_real_digits_tree_and_ladder() -> None:
    generator = load_generator(Path("src/leibniz/benchmarks/digits"))
    batch = generator(
        seed=31,
        shape=32,
        include_metadata=True,
        volume_request=StateSpaceVolumeRequest(minimum=3.0, maximum=4.0),
        sample_indices=tuple(range(32)),
    )
    assert batch.region is not None
    samples = partition_samples_from_generated(
        batch.samples,
        {
            sample.index: (
                0.0
                if sample.region_component_index is not None
                and sample.region_component_index < 2
                else 1.0
            )
            for sample in batch.samples
        },
    )

    score = adversarial_partition_competence_integral(
        root_region=batch.region,
        samples=samples,
        score_width_bits=1.0,
    )

    assert len(score.refinement_ladder) > 1
    assert len(score.root.leaves()) > 1
    assert math.isclose(score.value, 6 / 8)
    assert math.isclose(score.mean_competence, 6 / 8)
    child_regions = [child.estimate.region for child in score.root.children]
    assert child_regions
    assert all(region.measure_estimate is not None for region in child_regions)
    assert all(
        component.measure_estimate is not None
        for region in child_regions
        for component in region.components
    )


def test_validated_bits_partition_converges_on_real_digits_region() -> None:
    generator = load_generator(Path("src/leibniz/benchmarks/digits"))

    coarse = _digits_component_score(generator=generator, sample_count=16)
    fine = _digits_component_score(generator=generator, sample_count=32)
    finer = _digits_component_score(generator=generator, sample_count=64)

    assert len(coarse.root.leaves()) > 1
    assert len(fine.root.leaves()) > 1
    assert len(finer.root.leaves()) > 1
    assert math.isclose(fine.value, finer.value, abs_tol=1e-12)
    assert math.isclose(fine.mean_competence, finer.mean_competence, abs_tol=1e-12)

    uniform = _digits_component_score(
        generator=generator,
        sample_count=64,
        structured=False,
    )
    assert len(uniform.root.leaves()) == 1
    assert len(uniform.refinement_ladder) == 1
    assert math.isclose(uniform.mean_competence, 1.0)


def test_axis_split_preserves_estimated_measure_inside_mixed_product() -> None:
    root = _mixed_estimated_region()
    samples: list[PartitionSample] = []
    index = 0
    for x in range(4):
        for _ in range(4):
            samples.append(
                PartitionSample(
                    sample_index=index,
                    competence=0.0 if x < 2 else 1.0,
                    region_component_index=0,
                    axis_coordinates={"x": x, "y": 0.5},
                )
            )
            index += 1

    score = adversarial_partition_competence_integral(
        root_region=root,
        samples=tuple(samples),
        score_width_bits=1.0,
    )

    assert len(score.root.children) == 2
    for child in score.root.children:
        product = child.estimate.region.components[0]
        assert product.volume == 8
        assert product.measure_estimate is not None
        assert math.isclose(product.measure_estimate.log2_lower or 0.0, 3.0)
        assert math.isclose(product.measure_estimate.log2_upper or 0.0, 3.0)
        assert child.estimate.region.measure_estimate is not None


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
    assert all(
        root.contains(sample.region_component_index, sample.axis_coordinates)
        for sample in samples
    )
    return tuple(samples)


def _line_samples(
    root: StateSpaceRegion,
    *,
    repetitions: int,
    pocket: range,
) -> tuple[PartitionSample, ...]:
    samples: list[PartitionSample] = []
    index = 0
    for coordinate in range(16):
        for _ in range(repetitions):
            competence = 0.0 if coordinate in pocket else 1.0
            samples.append(
                PartitionSample(
                    sample_index=index,
                    competence=competence,
                    region_component_index=0,
                    axis_coordinates={"x": coordinate},
                )
            )
            index += 1
    assert all(
        root.contains(sample.region_component_index, sample.axis_coordinates)
        for sample in samples
    )
    return tuple(samples)


def _line_region(region_id: str, *, lower: int, upper: int) -> StateSpaceRegion:
    axis = StateSpaceAxis(id="x", domain=IntegerRangeDomain(lower=0, upper=15))
    count = upper - lower + 1
    return StateSpaceRegion(
        id=region_id,
        ambient=_grid_ambient(),
        components=(
            ProductRegion(
                axis_regions=(
                    DiscreteAxisRegion(
                        axis=axis,
                        coordinate_region=(lower, upper),
                        count=count,
                        log2_count=math.log2(count),
                    ),
                ),
                measure_rule="product-of-counts",
                volume=count,
                log2_volume=math.log2(count),
            ),
        ),
        union_rule="disjoint-union",
        volume=count,
        log2_volume=math.log2(count),
    )


def _mixed_estimated_region() -> StateSpaceRegion:
    x_axis = StateSpaceAxis(id="x", domain=IntegerRangeDomain(lower=0, upper=3))
    y_axis = StateSpaceAxis(id="y", domain=RealIntervalDomain(lower=0.0, upper=1.0))
    y_measure = MeasureEstimate(
        kind="estimated",
        method_id="fixture-y",
        log2_lower=2.0,
        log2_upper=2.0,
    )
    product_measure = MeasureEstimate(
        kind="estimated",
        method_id="fixture-product",
        log2_lower=4.0,
        log2_upper=4.0,
    )
    product = ProductRegion(
        axis_regions=(
            DiscreteAxisRegion(
                axis=x_axis,
                coordinate_region=(0, 3),
                count=4,
                log2_count=2.0,
            ),
            ContinuousAxisRegion(
                axis=y_axis,
                coordinate_region=(0.0, 1.0),
                measure_estimate=y_measure,
            ),
        ),
        measure_rule="product-of-counts",
        volume=16,
        log2_volume=4.0,
        measure_estimate=product_measure,
    )
    return StateSpaceRegion(
        id="mixed-estimated.root",
        ambient=_grid_ambient(),
        components=(product,),
        union_rule="disjoint-union",
        volume=16,
        log2_volume=4.0,
        measure_estimate=product_measure,
    )


def _digits_component_score(
    *,
    generator: Generator,
    sample_count: int,
    structured: bool = True,
) -> PartitionScore:
    batch = generator(
        seed=31,
        shape=sample_count,
        include_metadata=True,
        volume_request=StateSpaceVolumeRequest(minimum=3.0, maximum=4.0),
        sample_indices=tuple(range(sample_count)),
    )
    assert batch.region is not None
    return adversarial_partition_competence_integral(
        root_region=batch.region,
        samples=partition_samples_from_generated(
            batch.samples,
            {
                sample.index: (
                    0.0
                    if structured
                    and sample.region_component_index is not None
                    and sample.region_component_index < 2
                    else 1.0
                )
                for sample in batch.samples
            },
        ),
        score_width_bits=1.0,
    )


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
