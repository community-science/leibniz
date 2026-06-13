import math
from collections.abc import Callable
from typing import cast

import pytest

from leibniz.documents import canonical_document_bytes, load_object_document
from leibniz.state_space import (
    BinaryVectorDomain,
    ContinuousAxisRegion,
    DiscreteAxisRegion,
    Distinguishability,
    EnumeratedCellsDomain,
    IntegerRangeDomain,
    MeasureEstimate,
    ProductRegion,
    RealGridDomain,
    RealIntervalDomain,
    RegionFiltration,
    SamplingProtocol,
    StateSpaceAmbient,
    StateSpaceAxis,
    StateSpaceError,
    StateSpaceRegion,
    axis_region_from_record,
    axis_regions_are_disjoint,
    measure_estimate_from_record,
    product_region_from_record,
    product_regions_are_disjoint,
    region_filtration_from_record,
    sampling_protocol_from_record,
    state_space_ambient_from_record,
    state_space_region_from_record,
    state_space_regions_are_disjoint,
)

_digits_axis_specs = (
    ("x_translation", -0.15, 0.15, 5),
    ("y_translation", -0.15, 0.15, 5),
    ("scale", 0.92, 1.08, 2),
    ("rotation", -0.03, 0.03, 2),
    ("x_shear", -0.03, 0.03, 1),
)

_chess_mechanisms = (
    "queen-adjacent-capture",
    "queen-file-capture",
    "queen-diagonal-capture",
    "supported-queen-adjacent-capture",
    "supported-queen-file-capture",
    "supported-queen-diagonal-capture",
)

_chess_transforms = (
    "identity",
    "mirror-file",
    "mirror-rank",
    "rotate-180",
    "transpose",
    "anti-transpose",
    "rotate-90",
    "rotate-270",
)


def _metric_ambient() -> StateSpaceAmbient:
    return StateSpaceAmbient(
        field_domain_kind="lattice-2d",
        field_domain={"height": 16, "width": 16},
        field_codomain_id="unit-intensity",
        distinguishability=Distinguishability(
            kind="metric-resolution",
            metric_id="digits-component-discriminability",
            resolution=20.0,
            certificate_id="component-discriminability-margin",
        ),
    )


def _exact_ambient() -> StateSpaceAmbient:
    return StateSpaceAmbient(
        field_domain_kind="lattice-2d",
        field_domain={"height": 8, "width": 8},
        field_codomain_id="chess-piece-occupancy",
        distinguishability=Distinguishability(kind="exact"),
    )


def _full_grid_region(
    name: str,
    *,
    lower: float,
    upper: float,
    count: int,
) -> DiscreteAxisRegion:
    axis = StateSpaceAxis(id=name, domain=RealGridDomain(lower=lower, upper=upper, count=count))
    return DiscreteAxisRegion(
        axis=axis,
        coordinate_region=(0, count - 1),
        count=count,
        log2_count=math.log2(count),
    )


def _digits_grid_region() -> StateSpaceRegion:
    axis_regions = tuple(
        _full_grid_region(name, lower=lower, upper=upper, count=count)
        for name, lower, upper, count in _digits_axis_specs
    )
    component = ProductRegion(
        axis_regions=axis_regions,
        measure_rule="product-of-counts",
        volume=100,
        log2_volume=math.log2(100),
        stratum_id="digit-7",
        stratum_target={"outcome_id": "digit-7"},
    )
    return StateSpaceRegion(
        id="digits-grid",
        ambient=_metric_ambient(),
        components=(component,),
        union_rule="disjoint-union",
        volume=100,
        log2_volume=math.log2(100),
    )


def _truncated_window_region() -> StateSpaceRegion:
    pose_axis = StateSpaceAxis(
        id="pose-transform-index", domain=IntegerRangeDomain(lower=0, upper=1)
    )
    components: list[ProductRegion] = []
    for digit in range(10):
        pose_count = 2 if digit < 3 else 1
        pose_region = DiscreteAxisRegion(
            axis=pose_axis,
            coordinate_region=(0, pose_count - 1),
            count=pose_count,
            log2_count=math.log2(pose_count),
        )
        components.append(
            ProductRegion(
                axis_regions=(pose_region,),
                measure_rule="product-of-counts",
                volume=pose_count,
                log2_volume=math.log2(pose_count),
                stratum_id=f"digit-{digit}",
                stratum_target={"outcome_id": f"digit-{digit}"},
            )
        )
    return StateSpaceRegion(
        id="digits-truncated-window",
        ambient=_metric_ambient(),
        components=tuple(components),
        union_rule="disjoint-union",
        volume=13,
        log2_volume=math.log2(13),
    )


def _preset_region() -> StateSpaceRegion:
    cells = tuple(f"preset-{index}" for index in range(8))
    axis = StateSpaceAxis(id="pose-preset", domain=EnumeratedCellsDomain(cells=cells))
    axis_region = DiscreteAxisRegion(
        axis=axis,
        coordinate_region=("preset-0", "preset-2", "preset-5", "preset-7"),
        count=4,
        log2_count=2.0,
    )
    component = ProductRegion(
        axis_regions=(axis_region,),
        measure_rule="product-of-counts",
        volume=4,
        log2_volume=2.0,
        stratum_id="digit-0",
    )
    return StateSpaceRegion(
        id="digits-presets",
        ambient=_metric_ambient(),
        components=(component,),
        union_rule="disjoint-union",
        volume=4,
        log2_volume=2.0,
    )


def _singleton_axis_region(name: str, *, coordinate: int) -> DiscreteAxisRegion:
    axis = StateSpaceAxis(id=name, domain=IntegerRangeDomain(lower=0, upper=7))
    return DiscreteAxisRegion(
        axis=axis,
        coordinate_region=(coordinate, coordinate),
        count=1,
        log2_count=0.0,
    )


def _chess_region() -> StateSpaceRegion:
    spectator_axis = StateSpaceAxis(
        id="spectator-occupancy", domain=BinaryVectorDomain(dimension=51)
    )
    spectator_region = DiscreteAxisRegion(
        axis=spectator_axis,
        coordinate_region=(0, 1, 2),
        count=8,
        log2_count=3.0,
    )
    components = tuple(
        ProductRegion(
            axis_regions=(
                _singleton_axis_region("white-king-file", coordinate=2),
                _singleton_axis_region("white-king-rank", coordinate=0),
                spectator_region,
            ),
            measure_rule="benchmark-computed-finite-count",
            volume=5,
            log2_volume=math.log2(5),
            stratum_id=f"{mechanism}/{transform}",
        )
        for mechanism in _chess_mechanisms
        for transform in _chess_transforms
    )
    return StateSpaceRegion(
        id="chess-mate-in-one",
        ambient=_exact_ambient(),
        components=components,
        union_rule="disjoint-union",
        volume=240,
        log2_volume=math.log2(240),
    )


def test_digits_grid_region_volume_is_product_of_axis_counts() -> None:
    region = _digits_grid_region()
    assert region.volume == 100
    assert region.components[0].volume == 5 * 5 * 2 * 2 * 1


def test_product_of_counts_log2_volume_adds_axis_bits() -> None:
    component = _digits_grid_region().components[0]
    axis_regions = cast(tuple[DiscreteAxisRegion, ...], component.axis_regions)
    axis_bits = sum(axis_region.log2_count for axis_region in axis_regions)
    assert math.isclose(component.log2_volume, axis_bits, rel_tol=0.0, abs_tol=1e-9)


def test_truncated_window_region_has_unequal_strata_and_exact_volume() -> None:
    region = _truncated_window_region()
    assert region.volume == 13
    assert len(region.components) == 10
    assert [component.volume for component in region.components] == [2, 2, 2, 1, 1, 1, 1, 1, 1, 1]
    assert {component.stratum_id for component in region.components} == {
        f"digit-{digit}" for digit in range(10)
    }


def test_preset_region_selects_enumerated_cells() -> None:
    region = _preset_region()
    axis_region = region.components[0].axis_regions[0]
    assert region.volume == 4
    assert axis_region.contains("preset-2")
    assert not axis_region.contains("preset-1")


def test_chess_region_decomposes_over_mechanism_transform_strata() -> None:
    region = _chess_region()
    assert len(region.components) == 48
    assert region.volume == 240
    component = region.components[0]
    axis_regions = cast(tuple[DiscreteAxisRegion, ...], component.axis_regions)
    box = math.prod(axis_region.count for axis_region in axis_regions)
    assert component.volume < box


def test_empty_binary_vector_region_is_singleton_zero_mask() -> None:
    axis = StateSpaceAxis(id="spectator-occupancy", domain=BinaryVectorDomain(dimension=51))
    axis_region = DiscreteAxisRegion(axis=axis, coordinate_region=(), count=1, log2_count=0.0)
    assert axis_region.contains(())
    assert not axis_region.contains((0,))


def test_integer_range_region_contains_only_in_range_integers() -> None:
    axis = StateSpaceAxis(id="pose-transform-index", domain=IntegerRangeDomain(lower=0, upper=9))
    region = DiscreteAxisRegion(axis=axis, coordinate_region=(2, 5), count=4, log2_count=2.0)
    assert region.contains(2)
    assert region.contains(5)
    assert not region.contains(1)
    assert not region.contains(6)
    assert not region.contains("2")
    assert not region.contains(True)


def test_real_grid_region_contains_grid_indices() -> None:
    axis = StateSpaceAxis(id="scale", domain=RealGridDomain(lower=0.92, upper=1.08, count=5))
    region = DiscreteAxisRegion(
        axis=axis,
        coordinate_region=(1, 3),
        count=3,
        log2_count=math.log2(3),
    )
    assert region.contains(1)
    assert region.contains(3)
    assert not region.contains(0)
    assert not region.contains(4)
    assert not region.contains(1.0)


def test_binary_vector_region_contains_subsets_of_enabled_indices() -> None:
    axis = StateSpaceAxis(id="spectator-occupancy", domain=BinaryVectorDomain(dimension=8))
    region = DiscreteAxisRegion(axis=axis, coordinate_region=(1, 4, 6), count=8, log2_count=3.0)
    assert region.contains(())
    assert region.contains((4,))
    assert region.contains((1, 6))
    assert not region.contains((2,))
    assert not region.contains((1, 1))
    assert not region.contains(4)


def test_product_region_requires_exact_axis_coordinate_keys() -> None:
    component = _truncated_window_region().components[0]
    assert component.contains({"pose-transform-index": 1})
    assert not component.contains({})
    assert not component.contains({"pose-transform-index": 1, "extra": 0})
    assert not component.contains({"pose-transform-index": 5})


def test_state_space_region_contains_delegates_to_component() -> None:
    region = _truncated_window_region()
    assert region.contains(0, {"pose-transform-index": 1})
    assert not region.contains(9, {"pose-transform-index": 1})
    with pytest.raises(StateSpaceError):
        region.contains(10, {"pose-transform-index": 0})
    with pytest.raises(StateSpaceError):
        region.contains(-1, {"pose-transform-index": 0})


@pytest.mark.parametrize(
    "build_region",
    [_digits_grid_region, _truncated_window_region, _preset_region, _chess_region],
)
def test_region_records_round_trip(build_region: Callable[[], StateSpaceRegion]) -> None:
    region = build_region()
    record = region.to_record()
    assert state_space_region_from_record(record) == region
    document = canonical_document_bytes(record)
    loaded = load_object_document(document, description="state-space region record")
    assert state_space_region_from_record(loaded) == region


def test_distinguishability_invariants() -> None:
    with pytest.raises(StateSpaceError):
        Distinguishability(kind="approximate")
    with pytest.raises(StateSpaceError):
        Distinguishability(kind="exact", metric_id="l2")
    with pytest.raises(StateSpaceError):
        Distinguishability(kind="exact", resolution=1.0)
    with pytest.raises(StateSpaceError):
        Distinguishability(kind="metric-resolution", metric_id="l2")
    with pytest.raises(StateSpaceError):
        Distinguishability(kind="metric-resolution", resolution=1.0)
    with pytest.raises(StateSpaceError):
        Distinguishability(kind="metric-resolution", metric_id="l2", resolution=0.0)
    with pytest.raises(StateSpaceError):
        Distinguishability(kind="exact", certificate_id="")


def test_ambient_invariants() -> None:
    distinguishability = Distinguishability(kind="exact")
    with pytest.raises(StateSpaceError):
        StateSpaceAmbient(
            field_domain_kind="",
            field_domain={"height": 8},
            field_codomain_id="unit-intensity",
            distinguishability=distinguishability,
        )
    with pytest.raises(StateSpaceError):
        StateSpaceAmbient(
            field_domain_kind="lattice-2d",
            field_domain={},
            field_codomain_id="unit-intensity",
            distinguishability=distinguishability,
        )
    with pytest.raises(StateSpaceError):
        StateSpaceAmbient(
            field_domain_kind="lattice-2d",
            field_domain={"height": [8]},
            field_codomain_id="unit-intensity",
            distinguishability=distinguishability,
        )
    with pytest.raises(StateSpaceError):
        StateSpaceAmbient(
            field_domain_kind="lattice-2d",
            field_domain={"height": 8},
            field_codomain_id="",
            distinguishability=distinguishability,
        )


def test_box_ambient_domains_validate_extent_count_and_boundary() -> None:
    distinguishability = Distinguishability(
        kind="metric-resolution",
        metric_id="periodic-l2",
        resolution=0.01,
    )
    ambient = StateSpaceAmbient(
        field_domain_kind="box-2d",
        field_domain={
            "length_x": math.tau,
            "length_y": 2.0,
            "boundary_id": "periodic",
            "units": "radian",
        },
        field_codomain_id="scalar-field",
        distinguishability=distinguishability,
    )

    record = load_object_document(
        canonical_document_bytes(ambient.to_record()),
        description="ambient",
    )
    parsed = state_space_ambient_from_record(record)

    assert parsed == ambient


@pytest.mark.parametrize(
    "field_domain_kind,field_domain",
    [
        ("box-1d", {"length_y": 1.0, "boundary_id": "periodic"}),
        ("box-2d", {"length_x": 1.0, "boundary_id": "periodic"}),
        (
            "box-3d",
            {
                "length_x": 1.0,
                "length_y": 1.0,
                "length_z": math.inf,
                "boundary_id": "periodic",
            },
        ),
        (
            "box-3d",
            {
                "length_x": 1.0,
                "length_y": 1.0,
                "length_z": 0.0,
                "boundary_id": "periodic",
            },
        ),
        ("box-1d", {"length_x": 1.0}),
        ("box-1d", {"length_x": 1.0, "boundary_id": ""}),
    ],
)
def test_box_ambient_domains_reject_malformed_domains(
    field_domain_kind: str,
    field_domain: dict[str, object],
) -> None:
    with pytest.raises(StateSpaceError):
        StateSpaceAmbient(
            field_domain_kind=field_domain_kind,
            field_domain=field_domain,
            field_codomain_id="scalar-field",
            distinguishability=Distinguishability(kind="exact"),
        )


def test_unknown_ambient_domain_kinds_remain_free_form() -> None:
    ambient = StateSpaceAmbient(
        field_domain_kind="benchmark-specific-domain",
        field_domain={"opaque": "value"},
        field_codomain_id="vector-field-3",
        distinguishability=Distinguishability(kind="exact"),
    )

    assert ambient.to_record()["field_domain"] == {"opaque": "value"}


def test_axis_domain_invariants() -> None:
    with pytest.raises(StateSpaceError):
        IntegerRangeDomain(lower=3, upper=2)
    with pytest.raises(StateSpaceError):
        IntegerRangeDomain(lower=True, upper=True)
    with pytest.raises(StateSpaceError):
        RealGridDomain(lower=0.0, upper=1.0, count=0)
    with pytest.raises(StateSpaceError):
        RealGridDomain(lower=1.0, upper=1.0, count=2)
    with pytest.raises(StateSpaceError):
        RealGridDomain(lower=math.inf, upper=1.0, count=1)
    with pytest.raises(StateSpaceError):
        RealIntervalDomain(lower=1.0, upper=1.0)
    with pytest.raises(StateSpaceError):
        RealIntervalDomain(lower=-math.inf, upper=1.0)
    with pytest.raises(StateSpaceError):
        EnumeratedCellsDomain(cells=())
    with pytest.raises(StateSpaceError):
        EnumeratedCellsDomain(cells=("preset-0", "preset-0"))
    with pytest.raises(StateSpaceError):
        EnumeratedCellsDomain(cells=("",))
    with pytest.raises(StateSpaceError):
        BinaryVectorDomain(dimension=0)
    with pytest.raises(StateSpaceError):
        StateSpaceAxis(id="", domain=IntegerRangeDomain(lower=0, upper=1))


def test_axis_region_invariants() -> None:
    axis = StateSpaceAxis(id="pose", domain=IntegerRangeDomain(lower=0, upper=9))
    with pytest.raises(StateSpaceError):
        DiscreteAxisRegion(axis=axis, coordinate_region=(0, 3), count=3, log2_count=math.log2(3))
    with pytest.raises(StateSpaceError):
        DiscreteAxisRegion(axis=axis, coordinate_region=(0, 3), count=4, log2_count=1.9)
    with pytest.raises(StateSpaceError):
        DiscreteAxisRegion(axis=axis, coordinate_region=(8, 10), count=3, log2_count=math.log2(3))
    with pytest.raises(StateSpaceError):
        DiscreteAxisRegion(axis=axis, coordinate_region=(5, 2), count=4, log2_count=2.0)
    with pytest.raises(StateSpaceError):
        DiscreteAxisRegion(axis=axis, coordinate_region=(1, 2, 3), count=3, log2_count=math.log2(3))
    grid_axis = StateSpaceAxis(id="scale", domain=RealGridDomain(lower=0.92, upper=1.08, count=3))
    with pytest.raises(StateSpaceError):
        DiscreteAxisRegion(axis=grid_axis, coordinate_region=(0, 3), count=4, log2_count=2.0)
    cells_axis = StateSpaceAxis(
        id="preset", domain=EnumeratedCellsDomain(cells=("preset-0", "preset-1"))
    )
    with pytest.raises(StateSpaceError):
        DiscreteAxisRegion(
            axis=cells_axis,
            coordinate_region=("preset-2",),
            count=1,
            log2_count=0.0,
        )
    with pytest.raises(StateSpaceError):
        DiscreteAxisRegion(
            axis=cells_axis,
            coordinate_region=("preset-0", "preset-0"),
            count=2,
            log2_count=1.0,
        )
    mask_axis = StateSpaceAxis(id="mask", domain=BinaryVectorDomain(dimension=4))
    with pytest.raises(StateSpaceError):
        DiscreteAxisRegion(axis=mask_axis, coordinate_region=(4,), count=2, log2_count=1.0)
    with pytest.raises(StateSpaceError):
        DiscreteAxisRegion(axis=mask_axis, coordinate_region=(1, 1), count=4, log2_count=2.0)
    continuous_axis = StateSpaceAxis(
        id="phase",
        domain=RealIntervalDomain(lower=0.0, upper=math.tau),
    )
    with pytest.raises(StateSpaceError):
        DiscreteAxisRegion(
            axis=continuous_axis,
            coordinate_region=(0, 1),
            count=2,
            log2_count=1.0,
        )
    with pytest.raises(StateSpaceError):
        ContinuousAxisRegion(
            axis=continuous_axis,
            coordinate_region=(0.0, 1.0),
            measure_estimate=MeasureEstimate(kind="exact"),
        )
    with pytest.raises(StateSpaceError):
        ContinuousAxisRegion(
            axis=continuous_axis,
            coordinate_region=(0.0, math.tau + 1.0),
            measure_estimate=MeasureEstimate(
                kind="estimated",
                method_id="covering-number-grid-bound",
                log2_lower=0.0,
                log2_upper=1.0,
            ),
        )


def test_continuous_axis_region_is_half_open_and_round_trips() -> None:
    axis = StateSpaceAxis(id="phase", domain=RealIntervalDomain(lower=0.0, upper=math.tau))
    region = ContinuousAxisRegion(
        axis=axis,
        coordinate_region=(0.5, 2.5),
        measure_estimate=MeasureEstimate(
            kind="estimated",
            method_id="covering-number-grid-bound",
            log2_lower=1.0,
            log2_upper=2.0,
        ),
    )

    assert region.contains(0.5)
    assert region.contains(1.25)
    assert not region.contains(2.5)
    assert not region.contains(0.49)
    assert axis_region_from_record(region.to_record()) == region


def test_product_regions_with_continuous_axes_require_estimated_measure() -> None:
    axis = StateSpaceAxis(id="phase", domain=RealIntervalDomain(lower=0.0, upper=math.tau))
    axis_region = ContinuousAxisRegion(
        axis=axis,
        coordinate_region=(0.0, 1.0),
        measure_estimate=MeasureEstimate(
            kind="estimated",
            method_id="covering-number-grid-bound",
            log2_lower=0.0,
            log2_upper=1.0,
        ),
    )

    with pytest.raises(StateSpaceError):
        ProductRegion(
            axis_regions=(axis_region,),
            measure_rule="benchmark-computed-finite-count",
            volume=1,
            log2_volume=0.0,
        )

    component = ProductRegion(
        axis_regions=(axis_region,),
        measure_rule="benchmark-computed-finite-count",
        volume=2,
        log2_volume=1.0,
        measure_estimate=MeasureEstimate(
            kind="estimated",
            method_id="covering-number-grid-bound",
            log2_lower=0.0,
            log2_upper=2.0,
        ),
    )

    assert product_region_from_record(component.to_record()) == component


def test_product_region_invariants() -> None:
    axis = StateSpaceAxis(id="pose", domain=IntegerRangeDomain(lower=0, upper=3))
    axis_region = DiscreteAxisRegion(axis=axis, coordinate_region=(0, 3), count=4, log2_count=2.0)
    with pytest.raises(StateSpaceError):
        ProductRegion(
            axis_regions=(),
            measure_rule="product-of-counts",
            volume=1,
            log2_volume=0.0,
        )
    with pytest.raises(StateSpaceError):
        ProductRegion(
            axis_regions=(axis_region, axis_region),
            measure_rule="product-of-counts",
            volume=16,
            log2_volume=4.0,
        )
    with pytest.raises(StateSpaceError):
        ProductRegion(
            axis_regions=(axis_region,),
            measure_rule="lebesgue",
            volume=4,
            log2_volume=2.0,
        )
    with pytest.raises(StateSpaceError):
        ProductRegion(
            axis_regions=(axis_region,),
            measure_rule="product-of-counts",
            volume=3,
            log2_volume=math.log2(3),
        )
    with pytest.raises(StateSpaceError):
        ProductRegion(
            axis_regions=(axis_region,),
            measure_rule="benchmark-computed-finite-count",
            volume=5,
            log2_volume=math.log2(5),
        )
    with pytest.raises(StateSpaceError):
        ProductRegion(
            axis_regions=(axis_region,),
            measure_rule="benchmark-computed-finite-count",
            volume=0,
            log2_volume=0.0,
        )
    with pytest.raises(StateSpaceError):
        ProductRegion(
            axis_regions=(axis_region,),
            measure_rule="product-of-counts",
            volume=4,
            log2_volume=2.0,
            stratum_id="",
        )
    with pytest.raises(StateSpaceError):
        ProductRegion(
            axis_regions=(axis_region,),
            measure_rule="product-of-counts",
            volume=4,
            log2_volume=2.0,
            stratum_target={"outcome_id": "digit-0"},
        )


def test_measure_estimate_invariants_and_round_trip() -> None:
    estimate = MeasureEstimate(
        kind="estimated",
        method_id="covering-number-grid-bound",
        log2_lower=1.0,
        log2_upper=2.0,
    )

    record = load_object_document(
        canonical_document_bytes(estimate.to_record()),
        description="measure estimate",
    )
    parsed = measure_estimate_from_record(record)

    assert parsed == estimate
    assert MeasureEstimate(kind="exact").to_record() == {"kind": "exact"}
    with pytest.raises(StateSpaceError):
        MeasureEstimate(kind="exact", method_id="method")
    with pytest.raises(StateSpaceError):
        MeasureEstimate(kind="estimated", log2_lower=1.0, log2_upper=2.0)
    with pytest.raises(StateSpaceError):
        MeasureEstimate(kind="estimated", method_id="method", log2_lower=2.0, log2_upper=1.0)
    with pytest.raises(StateSpaceError):
        MeasureEstimate(
            kind="estimated",
            method_id="method",
            log2_lower=1.0,
            log2_upper=math.inf,
        )


def test_sampling_protocol_records_round_trip() -> None:
    monte_carlo = SamplingProtocol(
        kind="uniform-monte-carlo",
        estimator_id="sample-mean",
        confidence_method_id="hoeffding",
        census_budget=128,
    )
    census = SamplingProtocol(kind="census", census_budget=32)

    monte_carlo_record = load_object_document(
        canonical_document_bytes(monte_carlo.to_record()),
        description="sampling protocol",
    )
    census_record = load_object_document(
        canonical_document_bytes(census.to_record()),
        description="sampling protocol",
    )

    assert sampling_protocol_from_record(monte_carlo_record) == monte_carlo
    assert sampling_protocol_from_record(census_record) == census


def test_sampling_protocol_invariants() -> None:
    with pytest.raises(StateSpaceError):
        SamplingProtocol(kind="latin-hypercube")
    with pytest.raises(StateSpaceError):
        SamplingProtocol(
            kind="uniform-monte-carlo",
            confidence_method_id="hoeffding",
        )
    with pytest.raises(StateSpaceError):
        SamplingProtocol(
            kind="uniform-monte-carlo",
            estimator_id="sample-mean",
        )
    with pytest.raises(StateSpaceError):
        SamplingProtocol(kind="census")
    with pytest.raises(StateSpaceError):
        SamplingProtocol(
            kind="census",
            estimator_id="sample-mean",
            census_budget=32,
        )
    with pytest.raises(StateSpaceError):
        SamplingProtocol(
            kind="uniform-monte-carlo",
            estimator_id="sample-mean",
            confidence_method_id="hoeffding",
            census_budget=0,
        )


def test_estimated_product_region_accepts_bracketed_volume() -> None:
    axis = StateSpaceAxis(id="pose", domain=IntegerRangeDomain(lower=0, upper=3))
    axis_region = DiscreteAxisRegion(axis=axis, coordinate_region=(0, 3), count=4, log2_count=2.0)
    component = ProductRegion(
        axis_regions=(axis_region,),
        measure_rule="product-of-counts",
        volume=3,
        log2_volume=1.7,
        measure_estimate=MeasureEstimate(
            kind="estimated",
            method_id="covering-number-grid-bound",
            log2_lower=1.0,
            log2_upper=2.0,
        ),
    )

    assert product_region_from_record(component.to_record()) == component


def test_estimated_product_region_rejects_invalid_brackets() -> None:
    axis = StateSpaceAxis(id="pose", domain=IntegerRangeDomain(lower=0, upper=3))
    axis_region = DiscreteAxisRegion(axis=axis, coordinate_region=(0, 3), count=4, log2_count=2.0)
    with pytest.raises(StateSpaceError):
        ProductRegion(
            axis_regions=(axis_region,),
            measure_rule="product-of-counts",
            volume=3,
            log2_volume=2.1,
            measure_estimate=MeasureEstimate(
                kind="estimated",
                method_id="covering-number-grid-bound",
                log2_lower=1.0,
                log2_upper=2.0,
            ),
        )
    with pytest.raises(StateSpaceError):
        ProductRegion(
            axis_regions=(axis_region,),
            measure_rule="product-of-counts",
            volume=3,
            log2_volume=1.7,
            measure_estimate=MeasureEstimate(
                kind="estimated",
                method_id="covering-number-grid-bound",
                log2_lower=1.0,
                log2_upper=3.0,
            ),
        )


def test_state_space_region_invariants() -> None:
    base = _truncated_window_region()
    with pytest.raises(StateSpaceError):
        StateSpaceRegion(
            id="",
            ambient=_metric_ambient(),
            components=base.components,
            union_rule="disjoint-union",
            volume=13,
            log2_volume=math.log2(13),
        )
    with pytest.raises(StateSpaceError):
        StateSpaceRegion(
            id="digits-truncated-window",
            ambient=_metric_ambient(),
            components=(),
            union_rule="disjoint-union",
            volume=1,
            log2_volume=0.0,
        )
    with pytest.raises(StateSpaceError):
        StateSpaceRegion(
            id="digits-truncated-window",
            ambient=_metric_ambient(),
            components=base.components,
            union_rule="overlapping-union",
            volume=13,
            log2_volume=math.log2(13),
        )
    with pytest.raises(StateSpaceError):
        StateSpaceRegion(
            id="digits-truncated-window",
            ambient=_metric_ambient(),
            components=base.components,
            union_rule="disjoint-union",
            volume=14,
            log2_volume=math.log2(14),
        )
    with pytest.raises(StateSpaceError):
        StateSpaceRegion(
            id="digits-truncated-window",
            ambient=_metric_ambient(),
            components=base.components,
            union_rule="disjoint-union",
            volume=13,
            log2_volume=math.log2(14),
        )


def test_estimated_state_space_region_contains_component_brackets() -> None:
    axis = StateSpaceAxis(id="pose", domain=IntegerRangeDomain(lower=0, upper=3))
    exact_axis_region = DiscreteAxisRegion(
        axis=axis,
        coordinate_region=(0, 1),
        count=2,
        log2_count=1.0,
    )
    estimated_axis_region = DiscreteAxisRegion(
        axis=axis,
        coordinate_region=(0, 3),
        count=4,
        log2_count=2.0,
    )
    exact_component = ProductRegion(
        axis_regions=(exact_axis_region,),
        measure_rule="product-of-counts",
        volume=2,
        log2_volume=1.0,
    )
    estimated_component = ProductRegion(
        axis_regions=(estimated_axis_region,),
        measure_rule="benchmark-computed-finite-count",
        volume=3,
        log2_volume=1.6,
        measure_estimate=MeasureEstimate(
            kind="estimated",
            method_id="covering-number-grid-bound",
            log2_lower=1.0,
            log2_upper=2.0,
        ),
    )
    region = StateSpaceRegion(
        id="estimated-region",
        ambient=_metric_ambient(),
        components=(exact_component, estimated_component),
        union_rule="disjoint-union",
        volume=5,
        log2_volume=math.log2(5),
        measure_estimate=MeasureEstimate(
            kind="estimated",
            method_id="covering-number-grid-bound",
            log2_lower=2.0,
            log2_upper=math.log2(6),
        ),
    )

    assert state_space_region_from_record(region.to_record()) == region


def test_estimated_state_space_region_rejects_component_interval_violation() -> None:
    axis = StateSpaceAxis(id="pose", domain=IntegerRangeDomain(lower=0, upper=3))
    axis_region = DiscreteAxisRegion(axis=axis, coordinate_region=(0, 3), count=4, log2_count=2.0)
    component = ProductRegion(
        axis_regions=(axis_region,),
        measure_rule="product-of-counts",
        volume=4,
        log2_volume=2.0,
    )

    with pytest.raises(StateSpaceError):
        StateSpaceRegion(
            id="estimated-region",
            ambient=_metric_ambient(),
            components=(component,),
            union_rule="disjoint-union",
            volume=4,
            log2_volume=2.0,
            measure_estimate=MeasureEstimate(
                kind="estimated",
                method_id="covering-number-grid-bound",
                log2_lower=1.0,
                log2_upper=1.5,
            ),
        )


def test_shared_axis_ids_must_match_across_components() -> None:
    first_axis = StateSpaceAxis(id="pose", domain=IntegerRangeDomain(lower=0, upper=1))
    second_axis = StateSpaceAxis(id="pose", domain=IntegerRangeDomain(lower=0, upper=3))
    first = ProductRegion(
        axis_regions=(
            DiscreteAxisRegion(axis=first_axis, coordinate_region=(0, 1), count=2, log2_count=1.0),
        ),
        measure_rule="product-of-counts",
        volume=2,
        log2_volume=1.0,
        stratum_id="digit-0",
    )
    second = ProductRegion(
        axis_regions=(
            DiscreteAxisRegion(axis=second_axis, coordinate_region=(0, 3), count=4, log2_count=2.0),
        ),
        measure_rule="product-of-counts",
        volume=4,
        log2_volume=2.0,
        stratum_id="digit-1",
    )
    with pytest.raises(StateSpaceError):
        StateSpaceRegion(
            id="digits-conflicting-axes",
            ambient=_metric_ambient(),
            components=(first, second),
            union_rule="disjoint-union",
            volume=6,
            log2_volume=math.log2(6),
        )


def test_region_parsing_rejects_malformed_records() -> None:
    record = _preset_region().to_record()
    with pytest.raises(StateSpaceError):
        state_space_region_from_record("not-a-mapping")
    with pytest.raises(StateSpaceError):
        state_space_region_from_record(
            {key: value for key, value in record.items() if key != "ambient"}
        )
    document = canonical_document_bytes(record)
    broken = dict(load_object_document(document, description="state-space region record"))
    components = cast(list[dict[str, object]], broken["components"])
    axis_regions = cast(list[dict[str, object]], components[0]["axis_regions"])
    domain = cast(dict[str, object], cast(dict[str, object], axis_regions[0]["axis"])["domain"])
    domain["kind"] = "qualitative-labels"
    with pytest.raises(StateSpaceError):
        state_space_region_from_record(broken)


_filtration_pose_axis = StateSpaceAxis(
    id="pose-transform-index", domain=IntegerRangeDomain(lower=0, upper=3)
)


def _pose_axis_region(*, lower: int, upper: int) -> DiscreteAxisRegion:
    count = upper - lower + 1
    return DiscreteAxisRegion(
        axis=_filtration_pose_axis,
        coordinate_region=(lower, upper),
        count=count,
        log2_count=math.log2(count),
    )


def _pose_product(stratum_id: str, *, lower: int, upper: int) -> ProductRegion:
    pose_region = _pose_axis_region(lower=lower, upper=upper)
    return ProductRegion(
        axis_regions=(pose_region,),
        measure_rule="product-of-counts",
        volume=pose_region.count,
        log2_volume=pose_region.log2_count,
        stratum_id=stratum_id,
    )


def _digit_shell_region(region_id: str, *, lower: int, upper: int) -> StateSpaceRegion:
    count = upper - lower + 1
    components = tuple(
        _pose_product(f"digit-{digit}", lower=lower, upper=upper) for digit in range(10)
    )
    volume = 10 * count
    return StateSpaceRegion(
        id=region_id,
        ambient=_metric_ambient(),
        components=components,
        union_rule="disjoint-union",
        volume=volume,
        log2_volume=math.log2(volume),
    )


def _digits_filtration() -> RegionFiltration:
    increments = (
        _digit_shell_region("digits-shell-0", lower=0, upper=0),
        _digit_shell_region("digits-shell-1", lower=1, upper=1),
        _digit_shell_region("digits-shell-2", lower=2, upper=3),
    )
    return RegionFiltration(
        id="digits-curriculum",
        increments=increments,
        volume=40,
        log2_volume=math.log2(40),
    )


def test_axis_regions_disjoint_for_interval_kinds() -> None:
    axis = StateSpaceAxis(id="pose", domain=IntegerRangeDomain(lower=0, upper=9))
    low = DiscreteAxisRegion(axis=axis, coordinate_region=(0, 2), count=3, log2_count=math.log2(3))
    high = DiscreteAxisRegion(axis=axis, coordinate_region=(3, 5), count=3, log2_count=math.log2(3))
    touching = DiscreteAxisRegion(
        axis=axis,
        coordinate_region=(2, 4),
        count=3,
        log2_count=math.log2(3),
    )
    assert axis_regions_are_disjoint(low, high)
    assert not axis_regions_are_disjoint(low, touching)

    grid_axis = StateSpaceAxis(id="scale", domain=RealGridDomain(lower=0.0, upper=1.0, count=6))
    grid_low = DiscreteAxisRegion(axis=grid_axis, coordinate_region=(0, 1), count=2, log2_count=1.0)
    grid_high = DiscreteAxisRegion(
        axis=grid_axis,
        coordinate_region=(2, 3),
        count=2,
        log2_count=1.0,
    )
    grid_overlap = DiscreteAxisRegion(
        axis=grid_axis,
        coordinate_region=(1, 2),
        count=2,
        log2_count=1.0,
    )
    assert axis_regions_are_disjoint(grid_low, grid_high)
    assert not axis_regions_are_disjoint(grid_low, grid_overlap)


def test_axis_regions_disjoint_for_enumerated_cells() -> None:
    axis = StateSpaceAxis(
        id="preset", domain=EnumeratedCellsDomain(cells=tuple(f"preset-{i}" for i in range(6)))
    )
    left = DiscreteAxisRegion(
        axis=axis, coordinate_region=("preset-0", "preset-1"), count=2, log2_count=1.0
    )
    right = DiscreteAxisRegion(
        axis=axis, coordinate_region=("preset-2", "preset-3"), count=2, log2_count=1.0
    )
    overlap = DiscreteAxisRegion(
        axis=axis, coordinate_region=("preset-1", "preset-4"), count=2, log2_count=1.0
    )
    assert axis_regions_are_disjoint(left, right)
    assert not axis_regions_are_disjoint(left, overlap)


def test_binary_vector_axis_regions_are_never_disjoint() -> None:
    axis = StateSpaceAxis(id="spectator-occupancy", domain=BinaryVectorDomain(dimension=8))
    left = DiscreteAxisRegion(axis=axis, coordinate_region=(0, 1), count=4, log2_count=2.0)
    right = DiscreteAxisRegion(axis=axis, coordinate_region=(2, 3, 4), count=8, log2_count=3.0)
    # The all-zeros vector is a subset of every enabled set, so it lies in both.
    assert not axis_regions_are_disjoint(left, right)


def test_axis_regions_over_different_axes_are_not_comparable() -> None:
    left_axis = StateSpaceAxis(id="pose", domain=IntegerRangeDomain(lower=0, upper=3))
    right_axis = StateSpaceAxis(id="scale", domain=IntegerRangeDomain(lower=0, upper=3))
    left = DiscreteAxisRegion(axis=left_axis, coordinate_region=(0, 0), count=1, log2_count=0.0)
    right = DiscreteAxisRegion(axis=right_axis, coordinate_region=(0, 0), count=1, log2_count=0.0)
    with pytest.raises(StateSpaceError):
        axis_regions_are_disjoint(left, right)


def test_product_regions_disjoint_by_stratum_or_shared_axis() -> None:
    # Different strata are disjoint even when their axis regions coincide.
    assert product_regions_are_disjoint(
        _pose_product("digit-0", lower=0, upper=1),
        _pose_product("digit-1", lower=0, upper=1),
    )
    # Same stratum, disjoint shared axis.
    assert product_regions_are_disjoint(
        _pose_product("digit-0", lower=0, upper=0),
        _pose_product("digit-0", lower=1, upper=1),
    )
    # Same stratum, overlapping shared axis: not disjoint.
    assert not product_regions_are_disjoint(
        _pose_product("digit-0", lower=0, upper=2),
        _pose_product("digit-0", lower=1, upper=3),
    )


def test_product_regions_without_distinguishing_evidence_are_not_certified_disjoint() -> None:
    pose_axis = StateSpaceAxis(id="pose", domain=IntegerRangeDomain(lower=0, upper=3))
    scale_axis = StateSpaceAxis(id="scale", domain=IntegerRangeDomain(lower=0, upper=3))
    pose_component = ProductRegion(
        axis_regions=(
            DiscreteAxisRegion(axis=pose_axis, coordinate_region=(0, 0), count=1, log2_count=0.0),
        ),
        measure_rule="product-of-counts",
        volume=1,
        log2_volume=0.0,
    )
    scale_component = ProductRegion(
        axis_regions=(
            DiscreteAxisRegion(axis=scale_axis, coordinate_region=(0, 0), count=1, log2_count=0.0),
        ),
        measure_rule="product-of-counts",
        volume=1,
        log2_volume=0.0,
    )
    # No shared axis and no strata: cannot certify disjointness, so report False.
    assert not product_regions_are_disjoint(pose_component, scale_component)


def test_state_space_regions_disjoint_requires_all_pairs_and_shared_ambient() -> None:
    base = _digit_shell_region("digits-shell-0", lower=0, upper=0)
    disjoint = _digit_shell_region("digits-shell-1", lower=1, upper=1)
    overlapping = _digit_shell_region("digits-shell-0-again", lower=0, upper=2)
    assert state_space_regions_are_disjoint(base, disjoint)
    assert not state_space_regions_are_disjoint(base, overlapping)
    with pytest.raises(StateSpaceError):
        state_space_regions_are_disjoint(base, _chess_region())


def test_region_filtration_reports_cumulative_volumes() -> None:
    filtration = _digits_filtration()
    assert filtration.volume == 40
    assert math.isclose(filtration.log2_volume, math.log2(40), rel_tol=0.0, abs_tol=1e-9)
    assert filtration.cumulative_volumes == (10, 20, 40)
    assert filtration.cumulative_log2_volumes == (
        math.log2(10),
        math.log2(20),
        math.log2(40),
    )
    assert filtration.ambient == _metric_ambient()


def test_region_filtration_accepts_mixed_estimated_increment_bounds() -> None:
    exact = _digit_shell_region("digits-shell-0", lower=0, upper=0)
    estimated = StateSpaceRegion(
        id="digits-shell-estimated",
        ambient=_metric_ambient(),
        components=_digit_shell_region("digits-shell-1", lower=1, upper=1).components,
        union_rule="disjoint-union",
        volume=12,
        log2_volume=math.log2(12),
        measure_estimate=MeasureEstimate(
            kind="estimated",
            method_id="covering-number-grid-bound",
            log2_lower=math.log2(9),
            log2_upper=math.log2(13),
        ),
    )
    filtration = RegionFiltration(
        id="digits-estimated-curriculum",
        increments=(exact, estimated),
        volume=20,
        log2_volume=math.log2(20),
    )

    assert region_filtration_from_record(filtration.to_record()) == filtration


def test_region_filtration_rejects_estimated_cumulative_interval_violations() -> None:
    exact = _digit_shell_region("digits-shell-0", lower=0, upper=0)
    estimated = StateSpaceRegion(
        id="digits-shell-estimated",
        ambient=_metric_ambient(),
        components=_digit_shell_region("digits-shell-1", lower=1, upper=1).components,
        union_rule="disjoint-union",
        volume=12,
        log2_volume=math.log2(12),
        measure_estimate=MeasureEstimate(
            kind="estimated",
            method_id="covering-number-grid-bound",
            log2_lower=math.log2(9),
            log2_upper=math.log2(13),
        ),
    )

    with pytest.raises(StateSpaceError):
        RegionFiltration(
            id="digits-estimated-curriculum",
            increments=(exact, estimated),
            volume=24,
            log2_volume=math.log2(20),
        )
    with pytest.raises(StateSpaceError):
        RegionFiltration(
            id="digits-estimated-curriculum",
            increments=(exact, estimated),
            volume=20,
            log2_volume=math.log2(24),
        )


def test_region_filtration_rejects_overlapping_increments() -> None:
    with pytest.raises(StateSpaceError):
        RegionFiltration(
            id="digits-overlapping",
            increments=(
                _digit_shell_region("digits-shell-0", lower=0, upper=1),
                _digit_shell_region("digits-shell-1", lower=1, upper=2),
            ),
            volume=40,
            log2_volume=math.log2(40),
        )


def test_region_filtration_invariants() -> None:
    increments = _digits_filtration().increments
    with pytest.raises(StateSpaceError):
        RegionFiltration(id="", increments=increments, volume=40, log2_volume=math.log2(40))
    with pytest.raises(StateSpaceError):
        RegionFiltration(id="empty", increments=(), volume=1, log2_volume=0.0)
    with pytest.raises(StateSpaceError):
        RegionFiltration(
            id="digits-curriculum",
            increments=increments,
            volume=39,
            log2_volume=math.log2(39),
        )
    with pytest.raises(StateSpaceError):
        RegionFiltration(
            id="digits-curriculum",
            increments=increments,
            volume=40,
            log2_volume=math.log2(39),
        )


def test_region_filtration_rejects_mismatched_ambient_increments() -> None:
    with pytest.raises(StateSpaceError):
        RegionFiltration(
            id="mixed-ambient",
            increments=(
                _digit_shell_region("digits-shell-0", lower=0, upper=0),
                _chess_region(),
            ),
            volume=250,
            log2_volume=math.log2(250),
        )


def test_region_filtration_rejects_conflicting_shared_axes() -> None:
    other_pose_axis = StateSpaceAxis(
        id="pose-transform-index", domain=IntegerRangeDomain(lower=0, upper=7)
    )
    conflicting = StateSpaceRegion(
        id="digits-shell-conflict",
        ambient=_metric_ambient(),
        components=(
            ProductRegion(
                axis_regions=(
                    DiscreteAxisRegion(
                        axis=other_pose_axis,
                        coordinate_region=(4, 4),
                        count=1,
                        log2_count=0.0,
                    ),
                ),
                measure_rule="product-of-counts",
                volume=1,
                log2_volume=0.0,
                stratum_id="digit-0",
            ),
        ),
        union_rule="disjoint-union",
        volume=1,
        log2_volume=0.0,
    )
    with pytest.raises(StateSpaceError):
        RegionFiltration(
            id="digits-curriculum",
            increments=(_digit_shell_region("digits-shell-0", lower=0, upper=0), conflicting),
            volume=11,
            log2_volume=math.log2(11),
        )


def test_region_filtration_record_round_trips() -> None:
    filtration = _digits_filtration()
    record = filtration.to_record()
    assert region_filtration_from_record(record) == filtration
    document = canonical_document_bytes(record)
    loaded = load_object_document(document, description="region filtration record")
    assert region_filtration_from_record(loaded) == filtration


def test_region_filtration_parsing_rejects_malformed_records() -> None:
    record = _digits_filtration().to_record()
    with pytest.raises(StateSpaceError):
        region_filtration_from_record("not-a-mapping")
    with pytest.raises(StateSpaceError):
        region_filtration_from_record(
            {key: value for key, value in record.items() if key != "increments"}
        )
    broken = dict(record)
    broken["volume"] = 41
    with pytest.raises(StateSpaceError):
        region_filtration_from_record(broken)
