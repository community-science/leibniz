import math
from collections.abc import Callable
from typing import cast

import pytest

from leibniz.documents import canonical_document_bytes, load_object_document
from leibniz.state_space import (
    AxisRegion,
    BinaryVectorDomain,
    Distinguishability,
    EnumeratedCellsDomain,
    IntegerRangeDomain,
    ProductRegion,
    RealGridDomain,
    StateSpaceAmbient,
    StateSpaceAxis,
    StateSpaceError,
    StateSpaceRegion,
    state_space_region_from_record,
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


def _full_grid_region(name: str, *, lower: float, upper: float, count: int) -> AxisRegion:
    axis = StateSpaceAxis(id=name, domain=RealGridDomain(lower=lower, upper=upper, count=count))
    return AxisRegion(
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
        pose_region = AxisRegion(
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
    axis_region = AxisRegion(
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


def _singleton_axis_region(name: str, *, coordinate: int) -> AxisRegion:
    axis = StateSpaceAxis(id=name, domain=IntegerRangeDomain(lower=0, upper=7))
    return AxisRegion(
        axis=axis,
        coordinate_region=(coordinate, coordinate),
        count=1,
        log2_count=0.0,
    )


def _chess_region() -> StateSpaceRegion:
    spectator_axis = StateSpaceAxis(
        id="spectator-occupancy", domain=BinaryVectorDomain(dimension=51)
    )
    spectator_region = AxisRegion(
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
    axis_bits = sum(axis_region.log2_count for axis_region in component.axis_regions)
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
    box = math.prod(axis_region.count for axis_region in component.axis_regions)
    assert component.volume < box


def test_empty_binary_vector_region_is_singleton_zero_mask() -> None:
    axis = StateSpaceAxis(id="spectator-occupancy", domain=BinaryVectorDomain(dimension=51))
    axis_region = AxisRegion(axis=axis, coordinate_region=(), count=1, log2_count=0.0)
    assert axis_region.contains(())
    assert not axis_region.contains((0,))


def test_integer_range_region_contains_only_in_range_integers() -> None:
    axis = StateSpaceAxis(id="pose-transform-index", domain=IntegerRangeDomain(lower=0, upper=9))
    region = AxisRegion(axis=axis, coordinate_region=(2, 5), count=4, log2_count=2.0)
    assert region.contains(2)
    assert region.contains(5)
    assert not region.contains(1)
    assert not region.contains(6)
    assert not region.contains("2")
    assert not region.contains(True)


def test_real_grid_region_contains_grid_indices() -> None:
    axis = StateSpaceAxis(id="scale", domain=RealGridDomain(lower=0.92, upper=1.08, count=5))
    region = AxisRegion(axis=axis, coordinate_region=(1, 3), count=3, log2_count=math.log2(3))
    assert region.contains(1)
    assert region.contains(3)
    assert not region.contains(0)
    assert not region.contains(4)
    assert not region.contains(1.0)


def test_binary_vector_region_contains_subsets_of_enabled_indices() -> None:
    axis = StateSpaceAxis(id="spectator-occupancy", domain=BinaryVectorDomain(dimension=8))
    region = AxisRegion(axis=axis, coordinate_region=(1, 4, 6), count=8, log2_count=3.0)
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
        AxisRegion(axis=axis, coordinate_region=(0, 3), count=3, log2_count=math.log2(3))
    with pytest.raises(StateSpaceError):
        AxisRegion(axis=axis, coordinate_region=(0, 3), count=4, log2_count=1.9)
    with pytest.raises(StateSpaceError):
        AxisRegion(axis=axis, coordinate_region=(8, 10), count=3, log2_count=math.log2(3))
    with pytest.raises(StateSpaceError):
        AxisRegion(axis=axis, coordinate_region=(5, 2), count=4, log2_count=2.0)
    with pytest.raises(StateSpaceError):
        AxisRegion(axis=axis, coordinate_region=(1, 2, 3), count=3, log2_count=math.log2(3))
    grid_axis = StateSpaceAxis(id="scale", domain=RealGridDomain(lower=0.92, upper=1.08, count=3))
    with pytest.raises(StateSpaceError):
        AxisRegion(axis=grid_axis, coordinate_region=(0, 3), count=4, log2_count=2.0)
    cells_axis = StateSpaceAxis(
        id="preset", domain=EnumeratedCellsDomain(cells=("preset-0", "preset-1"))
    )
    with pytest.raises(StateSpaceError):
        AxisRegion(axis=cells_axis, coordinate_region=("preset-2",), count=1, log2_count=0.0)
    with pytest.raises(StateSpaceError):
        AxisRegion(
            axis=cells_axis,
            coordinate_region=("preset-0", "preset-0"),
            count=2,
            log2_count=1.0,
        )
    mask_axis = StateSpaceAxis(id="mask", domain=BinaryVectorDomain(dimension=4))
    with pytest.raises(StateSpaceError):
        AxisRegion(axis=mask_axis, coordinate_region=(4,), count=2, log2_count=1.0)
    with pytest.raises(StateSpaceError):
        AxisRegion(axis=mask_axis, coordinate_region=(1, 1), count=4, log2_count=2.0)


def test_product_region_invariants() -> None:
    axis = StateSpaceAxis(id="pose", domain=IntegerRangeDomain(lower=0, upper=3))
    axis_region = AxisRegion(axis=axis, coordinate_region=(0, 3), count=4, log2_count=2.0)
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


def test_shared_axis_ids_must_match_across_components() -> None:
    first_axis = StateSpaceAxis(id="pose", domain=IntegerRangeDomain(lower=0, upper=1))
    second_axis = StateSpaceAxis(id="pose", domain=IntegerRangeDomain(lower=0, upper=3))
    first = ProductRegion(
        axis_regions=(
            AxisRegion(axis=first_axis, coordinate_region=(0, 1), count=2, log2_count=1.0),
        ),
        measure_rule="product-of-counts",
        volume=2,
        log2_volume=1.0,
        stratum_id="digit-0",
    )
    second = ProductRegion(
        axis_regions=(
            AxisRegion(axis=second_axis, coordinate_region=(0, 3), count=4, log2_count=2.0),
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
