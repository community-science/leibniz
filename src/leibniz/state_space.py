"""Measured state-space regions over benchmark field spaces.

Informally: every observation a benchmark can show a model -- a rendered glyph
image, a board-game position, a fluid field on a mesh -- is a point in a space
of possible observations. A benchmark generates observations by turning a
small number of knobs, so the knobs chart a structured island of meaningful
observations inside that space. Two knob settings count as different states
only when the observations they produce are distinguishable at the benchmark's
declared resolution. The volume of a region is the number of genuinely
distinguishable states in it, reported in bits as ``log2(volume)``; volumes
multiply across independent knobs, so bits add. A model's score is the area
under its competence-density curve along that bits axis.

Formally: a benchmark declares an ambient state space of fields on a geometric
domain with a distinguishability metric and resolution, and charts realizable
regions of it through measured latent coordinate axes. A region is a finite
disjoint union of products of per-axis regions; its volume is the
distinguishability-certified state count, defined in ambient field space and
computed exactly through the chart, so it is invariant to reparameterization
of the generator. Qualitative labels are strata -- typed annotations on union
components -- never axes. Field codomain identifiers are benchmark-declared
nonempty strings; common conventions are ``scalar-field`` and
``vector-field-<n>``. These records carry geometry and measure only; requests,
sampling outcomes, and integral terms build on them in later layers.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

__all__ = [
    "AccessibleSubspace",
    "AxisCoordinateRegion",
    "AxisDomain",
    "AxisRegion",
    "BinaryVectorDomain",
    "ContinuousAxisRegion",
    "ContinuousAxisCoordinateRegion",
    "DiscreteAxisRegion",
    "DiscreteAxisCoordinateRegion",
    "Distinguishability",
    "EnumeratedCellsDomain",
    "IntegerRangeDomain",
    "MeasureEstimate",
    "ProductRegion",
    "RealGridDomain",
    "RealIntervalDomain",
    "RegionFiltration",
    "SamplingProtocol",
    "StateSpaceAmbient",
    "StateSpaceAxis",
    "StateSpaceError",
    "StateSpaceRegion",
    "accessible_subspace_from_record",
    "axis_domain_from_record",
    "axis_region_from_record",
    "axis_regions_are_disjoint",
    "distinguishability_from_record",
    "measure_estimate_from_record",
    "product_region_from_record",
    "product_regions_are_disjoint",
    "region_filtration_from_record",
    "sampling_protocol_from_record",
    "state_space_ambient_from_record",
    "state_space_axis_from_record",
    "state_space_region_from_record",
    "state_space_region_contains",
    "state_space_regions_are_disjoint",
]

_log2_tolerance = 1e-9

_exact_distinguishability_kind = "exact"
_metric_resolution_distinguishability_kind = "metric-resolution"
_distinguishability_kinds = frozenset(
    {_exact_distinguishability_kind, _metric_resolution_distinguishability_kind}
)

_integer_range_kind = "integer-range"
_real_grid_kind = "real-grid"
_real_interval_kind = "real-interval"
_enumerated_cells_kind = "enumerated-cells"
_binary_vector_kind = "binary-vector"

_box_field_domain_extents = {
    "box-1d": ("length_x",),
    "box-2d": ("length_x", "length_y"),
    "box-3d": ("length_x", "length_y", "length_z"),
}

_product_of_counts_measure_rule = "product-of-counts"
_benchmark_computed_measure_rule = "benchmark-computed-finite-count"
_measure_rules = frozenset({_product_of_counts_measure_rule, _benchmark_computed_measure_rule})

_disjoint_union_rule = "disjoint-union"
_union_rules = frozenset({_disjoint_union_rule})

_exact_measure_estimate_kind = "exact"
_estimated_measure_estimate_kind = "estimated"
_measure_estimate_kinds = frozenset(
    {_exact_measure_estimate_kind, _estimated_measure_estimate_kind}
)

_uniform_monte_carlo_sampling_protocol_kind = "uniform-monte-carlo"
_census_sampling_protocol_kind = "census"
_sampling_protocol_kinds = frozenset(
    {_uniform_monte_carlo_sampling_protocol_kind, _census_sampling_protocol_kind}
)


class StateSpaceError(ValueError):
    """Raised when a state-space record violates its contract."""


@dataclass(frozen=True, slots=True)
class Distinguishability:
    """How ambient states are declared operationally distinguishable."""

    kind: str
    metric_id: str | None = None
    resolution: float | None = None
    certificate_id: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in _distinguishability_kinds:
            raise StateSpaceError("distinguishability kind is not a core kind")
        if self.certificate_id is not None and not self.certificate_id:
            raise StateSpaceError("distinguishability certificate_id must be nonempty")
        if self.kind == _exact_distinguishability_kind:
            if self.metric_id is not None or self.resolution is not None:
                raise StateSpaceError(
                    "exact distinguishability does not declare a metric or resolution"
                )
            return
        if not self.metric_id:
            raise StateSpaceError("metric-resolution distinguishability requires a metric_id")
        if self.resolution is None or type(self.resolution) not in (int, float):
            raise StateSpaceError("metric-resolution distinguishability requires a resolution")
        if not math.isfinite(float(self.resolution)) or float(self.resolution) <= 0.0:
            raise StateSpaceError("distinguishability resolution must be finite and positive")

    def to_record(self) -> dict[str, object]:
        """Return a record for this declaration."""

        record: dict[str, object] = {"kind": self.kind}
        if self.metric_id is not None:
            record["metric_id"] = self.metric_id
        if self.resolution is not None:
            record["resolution"] = self.resolution
        if self.certificate_id is not None:
            record["certificate_id"] = self.certificate_id
        return record


@dataclass(frozen=True, slots=True)
class StateSpaceAmbient:
    """The ambient field space that generated observations live in."""

    field_domain_kind: str
    field_domain: Mapping[str, object]
    field_codomain_id: str
    distinguishability: Distinguishability

    def __post_init__(self) -> None:
        if not self.field_domain_kind:
            raise StateSpaceError("ambient field_domain_kind must be nonempty")
        if not self.field_codomain_id:
            raise StateSpaceError("ambient field_codomain_id must be nonempty")
        _validate_scalar_mapping(self.field_domain, label="ambient field_domain")
        _validate_box_field_domain(
            self.field_domain_kind,
            self.field_domain,
            label="ambient field_domain",
        )

    def to_record(self) -> dict[str, object]:
        """Return a record for this ambient declaration."""

        return {
            "field_domain_kind": self.field_domain_kind,
            "field_domain": dict(self.field_domain),
            "field_codomain_id": self.field_codomain_id,
            "distinguishability": self.distinguishability.to_record(),
        }


@dataclass(frozen=True, slots=True)
class MeasureEstimate:
    """Whether a declared region volume is exact or an estimated bracket."""

    kind: str
    method_id: str | None = None
    log2_lower: float | None = None
    log2_upper: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _measure_estimate_kinds:
            raise StateSpaceError("measure estimate kind is not a core kind")
        if self.kind == _exact_measure_estimate_kind:
            if (
                self.method_id is not None
                or self.log2_lower is not None
                or self.log2_upper is not None
            ):
                raise StateSpaceError("exact measure estimates do not declare a method or bounds")
            return
        if not self.method_id:
            raise StateSpaceError("estimated measure estimates require a method_id")
        if self.log2_lower is None or self.log2_upper is None:
            raise StateSpaceError("estimated measure estimates require log2 bounds")
        for label, value in (
            ("log2_lower", self.log2_lower),
            ("log2_upper", self.log2_upper),
        ):
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise StateSpaceError(f"measure estimate {label} must be finite")
        if float(self.log2_lower) > float(self.log2_upper):
            raise StateSpaceError("measure estimate lower bound must not exceed upper bound")

    @property
    def estimated(self) -> bool:
        """Return whether this estimate declares an interval bracket."""

        return self.kind == _estimated_measure_estimate_kind

    def to_record(self) -> dict[str, object]:
        """Return a record for this measure estimate."""

        record: dict[str, object] = {"kind": self.kind}
        if self.method_id is not None:
            record["method_id"] = self.method_id
        if self.log2_lower is not None:
            record["log2_lower"] = self.log2_lower
        if self.log2_upper is not None:
            record["log2_upper"] = self.log2_upper
        return record


@dataclass(frozen=True, slots=True)
class SamplingProtocol:
    """Sampling protocol declared for density estimates over a region.

    Consumers must treat a protocol as a census whenever a sample budget reaches
    the region's upper volume bracket; this record owns only the declared
    estimator shape and saturation threshold.
    """

    kind: str
    estimator_id: str | None = None
    confidence_method_id: str | None = None
    census_budget: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in _sampling_protocol_kinds:
            raise StateSpaceError("sampling protocol kind is not a core kind")
        if self.census_budget is not None and (
            type(self.census_budget) is not int or self.census_budget < 1
        ):
            raise StateSpaceError("sampling protocol census_budget must be a positive integer")
        if self.kind == _census_sampling_protocol_kind:
            if self.estimator_id is not None or self.confidence_method_id is not None:
                raise StateSpaceError("census sampling does not declare estimator ids")
            if self.census_budget is None:
                raise StateSpaceError("census sampling requires a census_budget")
            return
        if not self.estimator_id:
            raise StateSpaceError("uniform-monte-carlo sampling requires estimator_id")
        if not self.confidence_method_id:
            raise StateSpaceError(
                "uniform-monte-carlo sampling requires confidence_method_id"
            )

    def to_record(self) -> dict[str, object]:
        """Return a record for this sampling protocol."""

        record: dict[str, object] = {"kind": self.kind}
        if self.estimator_id is not None:
            record["estimator_id"] = self.estimator_id
        if self.confidence_method_id is not None:
            record["confidence_method_id"] = self.confidence_method_id
        if self.census_budget is not None:
            record["census_budget"] = self.census_budget
        return record


@dataclass(frozen=True, slots=True)
class IntegerRangeDomain:
    """A contiguous integer coordinate domain."""

    lower: int
    upper: int

    def __post_init__(self) -> None:
        if type(self.lower) is not int or type(self.upper) is not int:
            raise StateSpaceError("integer-range domain bounds must be integers")
        if self.upper < self.lower:
            raise StateSpaceError("integer-range domain upper bound must be at least the lower")

    @property
    def count(self) -> int:
        """Return the number of coordinates in this domain."""

        return self.upper - self.lower + 1

    def to_record(self) -> dict[str, object]:
        """Return a record for this domain."""

        return {"kind": _integer_range_kind, "lower": self.lower, "upper": self.upper}


@dataclass(frozen=True, slots=True)
class RealGridDomain:
    """A finite grid over a real interval; a single point sits at the midpoint."""

    lower: float
    upper: float
    count: int

    def __post_init__(self) -> None:
        for bound in (self.lower, self.upper):
            if type(bound) not in (int, float) or not math.isfinite(float(bound)):
                raise StateSpaceError("real-grid domain bounds must be finite numbers")
        if type(self.count) is not int or self.count < 1:
            raise StateSpaceError("real-grid domain count must be a positive integer")
        if self.upper < self.lower:
            raise StateSpaceError("real-grid domain upper bound must be at least the lower")
        if self.count > 1 and self.upper <= self.lower:
            raise StateSpaceError("real-grid domains with multiple points need a nonempty interval")

    def to_record(self) -> dict[str, object]:
        """Return a record for this domain."""

        return {
            "kind": _real_grid_kind,
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class RealIntervalDomain:
    """A continuous real interval coordinate domain."""

    lower: float
    upper: float

    def __post_init__(self) -> None:
        for bound in (self.lower, self.upper):
            if type(bound) not in (int, float) or not math.isfinite(float(bound)):
                raise StateSpaceError("real-interval domain bounds must be finite numbers")
        if self.upper <= self.lower:
            raise StateSpaceError("real-interval domain upper bound must exceed the lower")

    def to_record(self) -> dict[str, object]:
        """Return a record for this domain."""

        return {
            "kind": _real_interval_kind,
            "lower": self.lower,
            "upper": self.upper,
        }


@dataclass(frozen=True, slots=True)
class EnumeratedCellsDomain:
    """A finite coordinate domain of explicitly enumerated cells."""

    cells: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.cells:
            raise StateSpaceError("enumerated-cells domain must declare at least one cell")
        for cell in self.cells:
            if type(cell) is not str or not cell:
                raise StateSpaceError("enumerated-cells domain cell ids must be nonempty strings")
        if len(set(self.cells)) != len(self.cells):
            raise StateSpaceError("enumerated-cells domain cell ids must be unique")

    @property
    def count(self) -> int:
        """Return the number of coordinates in this domain."""

        return len(self.cells)

    def to_record(self) -> dict[str, object]:
        """Return a record for this domain."""

        return {"kind": _enumerated_cells_kind, "cells": list(self.cells)}


@dataclass(frozen=True, slots=True)
class BinaryVectorDomain:
    """A binary-vector coordinate domain over a fixed dimension."""

    dimension: int

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension < 1:
            raise StateSpaceError("binary-vector domain dimension must be a positive integer")

    @property
    def count(self) -> int:
        """Return the number of coordinates in this domain."""

        return 2**self.dimension

    def to_record(self) -> dict[str, object]:
        """Return a record for this domain."""

        return {"kind": _binary_vector_kind, "dimension": self.dimension}


AxisDomain = (
    IntegerRangeDomain
    | RealGridDomain
    | RealIntervalDomain
    | EnumeratedCellsDomain
    | BinaryVectorDomain
)
DiscreteAxisCoordinateRegion = tuple[int, ...] | tuple[str, ...]
ContinuousAxisCoordinateRegion = tuple[float, float]
AxisCoordinateRegion = DiscreteAxisCoordinateRegion | ContinuousAxisCoordinateRegion


@dataclass(frozen=True, slots=True)
class StateSpaceAxis:
    """A measured latent coordinate axis charting ambient state."""

    id: str
    domain: AxisDomain

    def __post_init__(self) -> None:
        if not self.id:
            raise StateSpaceError("axis id must be nonempty")
        domain_types = (
            IntegerRangeDomain,
            RealGridDomain,
            RealIntervalDomain,
            EnumeratedCellsDomain,
            BinaryVectorDomain,
        )
        if type(self.domain) not in domain_types:
            raise StateSpaceError("axis domain must be a core axis domain")

    @property
    def coordinate_kind(self) -> str:
        """Return the coordinate kind declared by this axis domain."""

        return _domain_kind(self.domain)

    def to_record(self) -> dict[str, object]:
        """Return a record for this axis."""

        return {"id": self.id, "domain": self.domain.to_record()}


@dataclass(frozen=True, slots=True)
class DiscreteAxisRegion:
    """A measurable subset of one axis with its certified state count."""

    axis: StateSpaceAxis
    coordinate_region: DiscreteAxisCoordinateRegion
    count: int
    log2_count: float

    def __post_init__(self) -> None:
        recomputed = _axis_region_count(self.axis, self.coordinate_region)
        if type(self.count) is not int or self.count != recomputed:
            raise StateSpaceError("axis region count must equal its recomputed coordinate count")
        _validate_log2(self.log2_count, self.count, label="axis region log2_count")

    @property
    def axis_id(self) -> str:
        """Return the identifier of this region's axis."""

        return self.axis.id

    def contains(self, value: object) -> bool:
        """Return whether a coordinate value lies in this axis region."""

        domain = self.axis.domain
        if isinstance(domain, IntegerRangeDomain | RealGridDomain):
            if type(value) is not int:
                return False
            lower, upper = cast(tuple[int, int], self.coordinate_region)
            return lower <= value <= upper
        if isinstance(domain, EnumeratedCellsDomain):
            return type(value) is str and value in self.coordinate_region
        if not isinstance(value, tuple):
            return False
        items = cast(tuple[object, ...], value)
        if any(type(item) is not int for item in items):
            return False
        indices = cast(tuple[int, ...], items)
        if len(set(indices)) != len(indices):
            return False
        enabled = set(cast(tuple[int, ...], self.coordinate_region))
        return all(index in enabled for index in indices)

    def to_record(self) -> dict[str, object]:
        """Return a record for this axis region."""

        return {
            "axis": self.axis.to_record(),
            "coordinate_region": list(self.coordinate_region),
            "count": self.count,
            "log2_count": self.log2_count,
        }


@dataclass(frozen=True, slots=True)
class ContinuousAxisRegion:
    """A continuous half-open interval over a real-interval axis."""

    axis: StateSpaceAxis
    coordinate_region: ContinuousAxisCoordinateRegion
    measure_estimate: MeasureEstimate

    def __post_init__(self) -> None:
        if not isinstance(self.axis.domain, RealIntervalDomain):
            raise StateSpaceError("continuous axis regions require a real-interval domain")
        lower, upper = _coordinate_real_interval(self.coordinate_region)
        domain = self.axis.domain
        if lower < float(domain.lower) or upper > float(domain.upper):
            raise StateSpaceError("real-interval coordinate region must lie within its axis domain")
        if not self.measure_estimate.estimated:
            raise StateSpaceError("continuous axis regions require an estimated measure")

    @property
    def axis_id(self) -> str:
        """Return the identifier of this region's axis."""

        return self.axis.id

    def contains(self, value: object) -> bool:
        """Return whether a coordinate value lies in this half-open interval."""

        if type(value) not in (int, float):
            return False
        coordinate = float(cast(int | float, value))
        if not math.isfinite(coordinate):
            return False
        lower, upper = self.coordinate_region
        return lower <= coordinate < upper

    def to_record(self) -> dict[str, object]:
        """Return a record for this axis region."""

        return {
            "axis": self.axis.to_record(),
            "coordinate_region": list(self.coordinate_region),
            "measure_estimate": self.measure_estimate.to_record(),
        }


AxisRegion = DiscreteAxisRegion | ContinuousAxisRegion


@dataclass(frozen=True, slots=True)
class ProductRegion:
    """A product of axis regions, optionally annotated with a label stratum."""

    axis_regions: tuple[AxisRegion, ...]
    measure_rule: str
    volume: int
    log2_volume: float
    stratum_id: str | None = None
    stratum_target: Mapping[str, object] | None = None
    measure_estimate: MeasureEstimate | None = None

    def __post_init__(self) -> None:
        if not self.axis_regions:
            raise StateSpaceError("product region must declare at least one axis region")
        axis_ids = [axis_region.axis_id for axis_region in self.axis_regions]
        if len(set(axis_ids)) != len(axis_ids):
            raise StateSpaceError("product region axis ids must be unique")
        if self.measure_rule not in _measure_rules:
            raise StateSpaceError("product region measure rule is not a core measure rule")
        if type(self.volume) is not int or self.volume < 1:
            raise StateSpaceError("product region volume must be a positive integer")
        has_continuous_axis = any(
            isinstance(axis_region, ContinuousAxisRegion)
            for axis_region in self.axis_regions
        )
        box = math.prod(
            axis_region.count
            for axis_region in self.axis_regions
            if isinstance(axis_region, DiscreteAxisRegion)
        )
        estimated = _measure_estimate_is_estimated(self.measure_estimate)
        if has_continuous_axis and not estimated:
            raise StateSpaceError("product regions with continuous axes require a measure estimate")
        if (
            self.measure_rule == _product_of_counts_measure_rule
            and self.volume != box
            and not estimated
            and not has_continuous_axis
        ):
            raise StateSpaceError("product-of-counts volume must equal the product of axis counts")
        if not has_continuous_axis and self.volume > box:
            raise StateSpaceError("product region volume must not exceed its product box")
        if estimated:
            estimate = cast(MeasureEstimate, self.measure_estimate)
            _validate_estimated_log2_volume(
                self.log2_volume,
                estimate,
                label="product region log2_volume",
            )
            _validate_estimated_volume(self.volume, estimate, label="product region")
            if (
                not has_continuous_axis
                and cast(float, estimate.log2_upper) > math.log2(box) + _log2_tolerance
            ):
                raise StateSpaceError("product region measure estimate exceeds its product box")
        else:
            _validate_log2(self.log2_volume, self.volume, label="product region log2_volume")
        if self.stratum_id is not None and not self.stratum_id:
            raise StateSpaceError("product region stratum_id must be nonempty")
        if self.stratum_target is not None:
            if self.stratum_id is None:
                raise StateSpaceError("product region stratum_target requires a stratum_id")
            _validate_scalar_mapping(self.stratum_target, label="product region stratum_target")

    def contains(self, coordinates: Mapping[str, object]) -> bool:
        """Return whether axis coordinates lie in this product region."""

        regions_by_axis = {axis_region.axis_id: axis_region for axis_region in self.axis_regions}
        if set(coordinates) != set(regions_by_axis):
            return False
        return all(
            regions_by_axis[axis_id].contains(value) for axis_id, value in coordinates.items()
        )

    def to_record(self) -> dict[str, object]:
        """Return a record for this product region."""

        record: dict[str, object] = {
            "axis_regions": [axis_region.to_record() for axis_region in self.axis_regions],
            "measure_rule": self.measure_rule,
            "volume": self.volume,
            "log2_volume": self.log2_volume,
        }
        if self.stratum_id is not None:
            record["stratum_id"] = self.stratum_id
        if self.stratum_target is not None:
            record["stratum_target"] = dict(self.stratum_target)
        if self.measure_estimate is not None:
            record["measure_estimate"] = self.measure_estimate.to_record()
        return record


@dataclass(frozen=True, slots=True)
class StateSpaceRegion:
    """A finite disjoint union of product regions with its certified volume."""

    id: str
    ambient: StateSpaceAmbient
    components: tuple[ProductRegion, ...]
    union_rule: str
    volume: int
    log2_volume: float
    measure_estimate: MeasureEstimate | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise StateSpaceError("state-space region id must be nonempty")
        if not self.components:
            raise StateSpaceError("state-space region must declare at least one component")
        if self.union_rule not in _union_rules:
            raise StateSpaceError("state-space region union rule is not a core union rule")
        if type(self.volume) is not int or self.volume < 1:
            raise StateSpaceError("state-space region volume must be a positive integer")
        estimated = _measure_estimate_is_estimated(self.measure_estimate)
        if self.volume != sum(component.volume for component in self.components) and not estimated:
            raise StateSpaceError("disjoint-union volume must equal the sum of component volumes")
        axes_by_id: dict[str, StateSpaceAxis] = {}
        for component in self.components:
            for axis_region in component.axis_regions:
                declared = axes_by_id.setdefault(axis_region.axis_id, axis_region.axis)
                if axis_region.axis != declared:
                    raise StateSpaceError(
                        "shared axis ids must declare identical axes across components"
                    )
        if estimated:
            estimate = cast(MeasureEstimate, self.measure_estimate)
            _validate_estimated_log2_volume(
                self.log2_volume,
                estimate,
                label="state-space region log2_volume",
            )
            _validate_estimated_volume(self.volume, estimate, label="state-space region")
            component_lower, component_upper = _sum_log2_measure_intervals(
                _product_region_log2_interval(component) for component in self.components
            )
            estimate_lower, estimate_upper = _measure_estimate_log2_interval(estimate)
            if (
                component_lower < estimate_lower - _log2_tolerance
                or component_upper > estimate_upper + _log2_tolerance
            ):
                raise StateSpaceError(
                    "state-space region measure estimate must contain component measures"
                )
        else:
            _validate_log2(
                self.log2_volume,
                self.volume,
                label="state-space region log2_volume",
            )

    def contains(self, component_index: int, coordinates: Mapping[str, object]) -> bool:
        """Return whether axis coordinates lie in the indexed component."""

        if type(component_index) is not int or not 0 <= component_index < len(self.components):
            raise StateSpaceError("component index is outside this region's components")
        return self.components[component_index].contains(coordinates)

    def to_record(self) -> dict[str, object]:
        """Return a record for this state-space region."""

        record: dict[str, object] = {
            "id": self.id,
            "ambient": self.ambient.to_record(),
            "components": [component.to_record() for component in self.components],
            "union_rule": self.union_rule,
            "volume": self.volume,
            "log2_volume": self.log2_volume,
        }
        if self.measure_estimate is not None:
            record["measure_estimate"] = self.measure_estimate.to_record()
        return record


def axis_regions_are_disjoint(left: AxisRegion, right: AxisRegion) -> bool:
    """Return whether two regions over the same axis share no coordinate state.

    The two regions must chart the same axis. Integer-range and real-grid
    regions are disjoint when their index intervals do not overlap;
    enumerated-cells regions are disjoint when their selected cell sets are
    disjoint. Binary-vector regions are never disjoint: the all-zeros vector is
    a subset of every enabled set, so it lies in both regions whatever their
    enabled coordinates are.
    """

    if left.axis != right.axis:
        raise StateSpaceError("axis regions over different axes are not comparable")
    domain = left.axis.domain
    if isinstance(domain, IntegerRangeDomain | RealGridDomain):
        left_lower, left_upper = cast(tuple[int, int], left.coordinate_region)
        right_lower, right_upper = cast(tuple[int, int], right.coordinate_region)
        return left_upper < right_lower or right_upper < left_lower
    if isinstance(domain, RealIntervalDomain):
        left_lower, left_upper = cast(tuple[float, float], left.coordinate_region)
        right_lower, right_upper = cast(tuple[float, float], right.coordinate_region)
        return left_upper <= right_lower or right_upper <= left_lower
    if isinstance(domain, EnumeratedCellsDomain):
        return set(cast(tuple[str, ...], left.coordinate_region)).isdisjoint(
            cast(tuple[str, ...], right.coordinate_region)
        )
    return False


def product_regions_are_disjoint(left: ProductRegion, right: ProductRegion) -> bool:
    """Return whether two product regions share no state.

    They are disjoint when their strata differ -- a labelled partition admits no
    shared state -- or when some shared axis carries disjoint axis regions. Two
    product regions with no distinguishing stratum and no disjoint shared axis
    are not certified disjoint, so this returns ``False`` for them.
    """

    if (
        left.stratum_id is not None
        and right.stratum_id is not None
        and left.stratum_id != right.stratum_id
    ):
        return True
    right_by_axis = {axis_region.axis_id: axis_region for axis_region in right.axis_regions}
    for axis_region in left.axis_regions:
        counterpart = right_by_axis.get(axis_region.axis_id)
        if counterpart is not None and axis_regions_are_disjoint(axis_region, counterpart):
            return True
    return False


def state_space_regions_are_disjoint(left: StateSpaceRegion, right: StateSpaceRegion) -> bool:
    """Return whether two regions in the same ambient share no state.

    A finite disjoint union is disjoint from another when every component pair
    is disjoint. The two regions must declare the same ambient field space;
    comparing regions across different ambients is a category error.
    """

    if left.ambient != right.ambient:
        raise StateSpaceError("regions in different ambients are not comparable")
    return all(
        product_regions_are_disjoint(left_component, right_component)
        for left_component in left.components
        for right_component in right.components
    )


def state_space_region_contains(container: StateSpaceRegion, contained: StateSpaceRegion) -> bool:
    """Return whether every state in ``contained`` lies in ``container``.

    The predicate is conservative over the explicit region grammar: compared
    regions must share one ambient, contained components must be covered by a
    target component with the same stratum, and compared product components
    must chart the same axes with each contained axis region a subset of the
    target axis region.
    """

    if container.ambient != contained.ambient:
        raise StateSpaceError("regions in different ambients are not comparable")
    return all(
        any(
            _product_region_contains(container_component, contained_component)
            for container_component in container.components
        )
        for contained_component in contained.components
    )


def _product_region_contains(container: ProductRegion, contained: ProductRegion) -> bool:
    if container.stratum_id != contained.stratum_id:
        return False
    container_by_axis = {
        axis_region.axis_id: axis_region for axis_region in container.axis_regions
    }
    contained_by_axis = {
        axis_region.axis_id: axis_region for axis_region in contained.axis_regions
    }
    if set(container_by_axis) != set(contained_by_axis):
        return False
    return all(
        _axis_region_contains(container_by_axis[axis_id], contained_axis_region)
        for axis_id, contained_axis_region in contained_by_axis.items()
    )


def _axis_region_contains(container: AxisRegion, contained: AxisRegion) -> bool:
    if container.axis != contained.axis:
        raise StateSpaceError("axis regions over different axes are not comparable")
    domain = container.axis.domain
    if isinstance(domain, IntegerRangeDomain | RealGridDomain):
        container_lower, container_upper = cast(
            tuple[int, int],
            container.coordinate_region,
        )
        contained_lower, contained_upper = cast(
            tuple[int, int],
            contained.coordinate_region,
        )
        return container_lower <= contained_lower and contained_upper <= container_upper
    if isinstance(domain, RealIntervalDomain):
        container_lower, container_upper = cast(
            tuple[float, float],
            container.coordinate_region,
        )
        contained_lower, contained_upper = cast(
            tuple[float, float],
            contained.coordinate_region,
        )
        return container_lower <= contained_lower and contained_upper <= container_upper
    if isinstance(domain, EnumeratedCellsDomain):
        return set(cast(tuple[str, ...], contained.coordinate_region)).issubset(
            cast(tuple[str, ...], container.coordinate_region)
        )
    return set(cast(tuple[int, ...], contained.coordinate_region)).issubset(
        cast(tuple[int, ...], container.coordinate_region)
    )


@dataclass(frozen=True, slots=True)
class RegionFiltration:
    """An ordered chain of pairwise-disjoint region increments over one ambient.

    Each increment ``A_i`` is the new region a curriculum step adds; the
    cumulative regions ``S_i = A_0 union ... union A_i`` form the score
    filtration of the direction document. Increments are pairwise disjoint,
    share one ambient field space, and declare identical axes wherever they
    share an axis id, so the cumulative volume ``mu(S_i)`` is the exact running
    sum of increment volumes and no distinguishable state is ever counted twice.
    """

    id: str
    increments: tuple[StateSpaceRegion, ...]
    volume: int
    log2_volume: float

    def __post_init__(self) -> None:
        if not self.id:
            raise StateSpaceError("region filtration id must be nonempty")
        if not self.increments:
            raise StateSpaceError("region filtration must declare at least one increment")
        ambient = self.increments[0].ambient
        for increment in self.increments:
            if increment.ambient != ambient:
                raise StateSpaceError("region filtration increments must share one ambient")
        axes_by_id: dict[str, StateSpaceAxis] = {}
        for increment in self.increments:
            for component in increment.components:
                for axis_region in component.axis_regions:
                    declared = axes_by_id.setdefault(axis_region.axis_id, axis_region.axis)
                    if axis_region.axis != declared:
                        raise StateSpaceError(
                            "shared axis ids must declare identical axes across increments"
                        )
        for earlier in range(len(self.increments)):
            for later in range(earlier + 1, len(self.increments)):
                if not state_space_regions_are_disjoint(
                    self.increments[earlier], self.increments[later]
                ):
                    raise StateSpaceError(
                        "region filtration increments must be pairwise disjoint"
                    )
        if type(self.volume) is not int or self.volume < 1:
            raise StateSpaceError("region filtration volume must be a positive integer")
        estimated = any(
            _measure_estimate_is_estimated(increment.measure_estimate)
            for increment in self.increments
        )
        if self.volume != sum(increment.volume for increment in self.increments) and not estimated:
            raise StateSpaceError(
                "region filtration volume must equal the sum of increment volumes"
            )
        if estimated:
            if not math.isfinite(float(self.log2_volume)):
                raise StateSpaceError("region filtration log2_volume must be finite")
            lower, upper = _sum_log2_measure_intervals(
                _state_space_region_log2_interval(increment)
                for increment in self.increments
            )
            if not lower - _log2_tolerance <= math.log2(self.volume) <= upper + _log2_tolerance:
                raise StateSpaceError(
                    "region filtration volume must lie within increment measure bounds"
                )
            if not lower - _log2_tolerance <= float(self.log2_volume) <= upper + _log2_tolerance:
                raise StateSpaceError(
                    "region filtration log2_volume must lie within increment measure bounds"
                )
        else:
            _validate_log2(self.log2_volume, self.volume, label="region filtration log2_volume")

    @property
    def ambient(self) -> StateSpaceAmbient:
        """Return the shared ambient field space of the filtration."""

        return self.increments[0].ambient

    @property
    def cumulative_volumes(self) -> tuple[int, ...]:
        """Return the cumulative volume ``mu(S_i)`` after each increment."""

        cumulative: list[int] = []
        running = 0
        for increment in self.increments:
            running += increment.volume
            cumulative.append(running)
        return tuple(cumulative)

    @property
    def cumulative_log2_volumes(self) -> tuple[float, ...]:
        """Return ``log2 mu(S_i)`` after each increment."""

        return tuple(math.log2(volume) for volume in self.cumulative_volumes)

    def to_record(self) -> dict[str, object]:
        """Return a record for this region filtration."""

        return {
            "id": self.id,
            "increments": [increment.to_record() for increment in self.increments],
            "volume": self.volume,
            "log2_volume": self.log2_volume,
        }


@dataclass(frozen=True, slots=True)
class AccessibleSubspace:
    """A benchmark-declared accessible region family over an unbounded ladder."""

    ladder_id: str
    per_configuration_capacity: StateSpaceRegion
    exclusions: tuple[StateSpaceRegion, ...] = ()
    frontier_rationale: str = ""

    def __post_init__(self) -> None:
        if not self.ladder_id:
            raise StateSpaceError("accessible subspace ladder_id must be nonempty")
        if not self.frontier_rationale:
            raise StateSpaceError("accessible subspace frontier_rationale must be nonempty")
        capacity_ambient = self.per_configuration_capacity.ambient
        for exclusion in self.exclusions:
            if exclusion.ambient != capacity_ambient:
                raise StateSpaceError(
                    "accessible subspace exclusions must share the capacity ambient"
                )
        for earlier in range(len(self.exclusions)):
            for later in range(earlier + 1, len(self.exclusions)):
                if not state_space_regions_are_disjoint(
                    self.exclusions[earlier],
                    self.exclusions[later],
                ):
                    raise StateSpaceError(
                        "accessible subspace exclusions must be pairwise disjoint"
                    )

    def to_record(self) -> dict[str, object]:
        """Return a record for this accessible subspace declaration."""

        record: dict[str, object] = {
            "ladder_id": self.ladder_id,
            "per_configuration_capacity": self.per_configuration_capacity.to_record(),
            "frontier_rationale": self.frontier_rationale,
        }
        if self.exclusions:
            record["exclusions"] = [exclusion.to_record() for exclusion in self.exclusions]
        return record


def distinguishability_from_record(value: object) -> Distinguishability:
    """Parse a distinguishability declaration from a record."""

    record = _record_mapping(value, label="distinguishability record")
    resolution_value = record.get("resolution")
    resolution: float | None = None
    if resolution_value is not None:
        if type(resolution_value) not in (int, float):
            raise StateSpaceError("distinguishability record resolution must be a number")
        resolution = float(cast(float, resolution_value))
    return Distinguishability(
        kind=_record_str(record, "kind", label="distinguishability record"),
        metric_id=_record_optional_str(record, "metric_id", label="distinguishability record"),
        resolution=resolution,
        certificate_id=_record_optional_str(
            record, "certificate_id", label="distinguishability record"
        ),
    )


def measure_estimate_from_record(value: object) -> MeasureEstimate:
    """Parse a measure estimate from a record."""

    record = _record_mapping(value, label="measure estimate record")
    return MeasureEstimate(
        kind=_record_str(record, "kind", label="measure estimate record"),
        method_id=_record_optional_str(record, "method_id", label="measure estimate record"),
        log2_lower=_record_optional_float(
            record,
            "log2_lower",
            label="measure estimate record",
        ),
        log2_upper=_record_optional_float(
            record,
            "log2_upper",
            label="measure estimate record",
        ),
    )


def sampling_protocol_from_record(value: object) -> SamplingProtocol:
    """Parse a sampling protocol from a record."""

    record = _record_mapping(value, label="sampling protocol record")
    return SamplingProtocol(
        kind=_record_str(record, "kind", label="sampling protocol record"),
        estimator_id=_record_optional_str(
            record,
            "estimator_id",
            label="sampling protocol record",
        ),
        confidence_method_id=_record_optional_str(
            record,
            "confidence_method_id",
            label="sampling protocol record",
        ),
        census_budget=_record_optional_int(
            record,
            "census_budget",
            label="sampling protocol record",
        ),
    )


def state_space_ambient_from_record(value: object) -> StateSpaceAmbient:
    """Parse an ambient declaration from a record."""

    record = _record_mapping(value, label="ambient record")
    field_domain = dict(_record_mapping(record.get("field_domain"), label="ambient field_domain"))
    return StateSpaceAmbient(
        field_domain_kind=_record_str(record, "field_domain_kind", label="ambient record"),
        field_domain=field_domain,
        field_codomain_id=_record_str(record, "field_codomain_id", label="ambient record"),
        distinguishability=distinguishability_from_record(record.get("distinguishability")),
    )


def axis_domain_from_record(value: object) -> AxisDomain:
    """Parse an axis domain from a record."""

    record = _record_mapping(value, label="axis domain record")
    kind = _record_str(record, "kind", label="axis domain record")
    if kind == _integer_range_kind:
        return IntegerRangeDomain(
            lower=_record_int(record, "lower", label="axis domain record"),
            upper=_record_int(record, "upper", label="axis domain record"),
        )
    if kind == _real_grid_kind:
        return RealGridDomain(
            lower=_record_float(record, "lower", label="axis domain record"),
            upper=_record_float(record, "upper", label="axis domain record"),
            count=_record_int(record, "count", label="axis domain record"),
        )
    if kind == _real_interval_kind:
        return RealIntervalDomain(
            lower=_record_float(record, "lower", label="axis domain record"),
            upper=_record_float(record, "upper", label="axis domain record"),
        )
    if kind == _enumerated_cells_kind:
        cells = _record_sequence(record, "cells", label="axis domain record")
        if any(type(cell) is not str for cell in cells):
            raise StateSpaceError("axis domain record cells must be strings")
        return EnumeratedCellsDomain(cells=cast(tuple[str, ...], cells))
    if kind == _binary_vector_kind:
        return BinaryVectorDomain(
            dimension=_record_int(record, "dimension", label="axis domain record")
        )
    raise StateSpaceError("axis domain record kind is not a core coordinate kind")


def state_space_axis_from_record(value: object) -> StateSpaceAxis:
    """Parse a state-space axis from a record."""

    record = _record_mapping(value, label="axis record")
    return StateSpaceAxis(
        id=_record_str(record, "id", label="axis record"),
        domain=axis_domain_from_record(record.get("domain")),
    )


def axis_region_from_record(value: object) -> AxisRegion:
    """Parse an axis region from a record."""

    record = _record_mapping(value, label="axis region record")
    axis = state_space_axis_from_record(record.get("axis"))
    items = _record_sequence(record, "coordinate_region", label="axis region record")
    if isinstance(axis.domain, RealIntervalDomain):
        if len(items) != 2 or any(type(item) not in (int, float) for item in items):
            raise StateSpaceError(
                "real-interval axis region coordinate_region must be a numeric pair"
            )
        measure_estimate_value = record.get("measure_estimate")
        if measure_estimate_value is None:
            raise StateSpaceError("continuous axis region record requires measure_estimate")
        return ContinuousAxisRegion(
            axis=axis,
            coordinate_region=(
                float(cast(int | float, items[0])),
                float(cast(int | float, items[1])),
            ),
            measure_estimate=measure_estimate_from_record(measure_estimate_value),
        )
    coordinate_region: DiscreteAxisCoordinateRegion
    if any(type(item) is str for item in items):
        if any(type(item) is not str for item in items):
            raise StateSpaceError("axis region record coordinate region must not mix value types")
        coordinate_region = cast(tuple[str, ...], items)
    else:
        if any(type(item) is not int for item in items):
            raise StateSpaceError(
                "axis region record coordinate region values must be integers or strings"
            )
        coordinate_region = cast(tuple[int, ...], items)
    return DiscreteAxisRegion(
        axis=axis,
        coordinate_region=coordinate_region,
        count=_record_int(record, "count", label="axis region record"),
        log2_count=_record_float(record, "log2_count", label="axis region record"),
    )


def product_region_from_record(value: object) -> ProductRegion:
    """Parse a product region from a record."""

    record = _record_mapping(value, label="product region record")
    axis_regions = tuple(
        axis_region_from_record(item)
        for item in _record_sequence(record, "axis_regions", label="product region record")
    )
    stratum_target_value = record.get("stratum_target")
    stratum_target = (
        None
        if stratum_target_value is None
        else dict(_record_mapping(stratum_target_value, label="product region stratum_target"))
    )
    measure_estimate_value = record.get("measure_estimate")
    measure_estimate = (
        None
        if measure_estimate_value is None
        else measure_estimate_from_record(measure_estimate_value)
    )
    return ProductRegion(
        axis_regions=axis_regions,
        measure_rule=_record_str(record, "measure_rule", label="product region record"),
        volume=_record_int(record, "volume", label="product region record"),
        log2_volume=_record_float(record, "log2_volume", label="product region record"),
        stratum_id=_record_optional_str(record, "stratum_id", label="product region record"),
        stratum_target=stratum_target,
        measure_estimate=measure_estimate,
    )


def state_space_region_from_record(value: object) -> StateSpaceRegion:
    """Parse a state-space region from a record."""

    record = _record_mapping(value, label="state-space region record")
    components = tuple(
        product_region_from_record(item)
        for item in _record_sequence(record, "components", label="state-space region record")
    )
    return StateSpaceRegion(
        id=_record_str(record, "id", label="state-space region record"),
        ambient=state_space_ambient_from_record(record.get("ambient")),
        components=components,
        union_rule=_record_str(record, "union_rule", label="state-space region record"),
        volume=_record_int(record, "volume", label="state-space region record"),
        log2_volume=_record_float(record, "log2_volume", label="state-space region record"),
        measure_estimate=(
            None
            if record.get("measure_estimate") is None
            else measure_estimate_from_record(record.get("measure_estimate"))
        ),
    )


def region_filtration_from_record(value: object) -> RegionFiltration:
    """Parse a region filtration from a record."""

    record = _record_mapping(value, label="region filtration record")
    increments = tuple(
        state_space_region_from_record(item)
        for item in _record_sequence(record, "increments", label="region filtration record")
    )
    return RegionFiltration(
        id=_record_str(record, "id", label="region filtration record"),
        increments=increments,
        volume=_record_int(record, "volume", label="region filtration record"),
        log2_volume=_record_float(record, "log2_volume", label="region filtration record"),
    )


def accessible_subspace_from_record(value: object) -> AccessibleSubspace:
    """Parse an accessible subspace declaration from a record."""

    record = _record_mapping(value, label="accessible subspace record")
    exclusions_value = record.get("exclusions")
    exclusions = (
        ()
        if exclusions_value is None
        else tuple(
            state_space_region_from_record(item)
            for item in _record_sequence(
                record,
                "exclusions",
                label="accessible subspace record",
            )
        )
    )
    return AccessibleSubspace(
        ladder_id=_record_str(record, "ladder_id", label="accessible subspace record"),
        per_configuration_capacity=state_space_region_from_record(
            record.get("per_configuration_capacity")
        ),
        exclusions=exclusions,
        frontier_rationale=_record_str(
            record,
            "frontier_rationale",
            label="accessible subspace record",
        ),
    )


def _domain_kind(domain: AxisDomain) -> str:
    if isinstance(domain, IntegerRangeDomain):
        return _integer_range_kind
    if isinstance(domain, RealGridDomain):
        return _real_grid_kind
    if isinstance(domain, RealIntervalDomain):
        return _real_interval_kind
    if isinstance(domain, EnumeratedCellsDomain):
        return _enumerated_cells_kind
    return _binary_vector_kind


def _axis_region_count(
    axis: StateSpaceAxis,
    coordinate_region: DiscreteAxisCoordinateRegion,
) -> int:
    domain = axis.domain
    if isinstance(domain, RealIntervalDomain):
        raise StateSpaceError("discrete axis regions cannot use real-interval domains")
    if isinstance(domain, IntegerRangeDomain):
        lower, upper = _coordinate_index_pair(
            coordinate_region, label="integer-range coordinate region"
        )
        if lower < domain.lower or upper > domain.upper:
            raise StateSpaceError("integer-range coordinate region must lie within its axis domain")
        return upper - lower + 1
    if isinstance(domain, RealGridDomain):
        lower, upper = _coordinate_index_pair(
            coordinate_region, label="real-grid coordinate region"
        )
        if lower < 0 or upper >= domain.count:
            raise StateSpaceError("real-grid coordinate region must lie within its axis grid")
        return upper - lower + 1
    if isinstance(domain, EnumeratedCellsDomain):
        cells = _coordinate_cells(coordinate_region)
        for cell in cells:
            if cell not in domain.cells:
                raise StateSpaceError(
                    "enumerated-cells coordinate region must select declared cells"
                )
        return len(cells)
    indices = _coordinate_bit_indices(coordinate_region)
    for index in indices:
        if index < 0 or index >= domain.dimension:
            raise StateSpaceError(
                "binary-vector coordinate region indices must lie within the axis dimension"
            )
    return 2 ** len(indices)


def _coordinate_index_pair(
    coordinate_region: DiscreteAxisCoordinateRegion, *, label: str
) -> tuple[int, int]:
    if len(coordinate_region) != 2 or any(type(value) is not int for value in coordinate_region):
        raise StateSpaceError(f"{label} must be a (lower, upper) integer pair")
    lower, upper = cast(tuple[int, int], coordinate_region)
    if upper < lower:
        raise StateSpaceError(f"{label} upper bound must be at least the lower bound")
    return lower, upper


def _coordinate_real_interval(
    coordinate_region: ContinuousAxisCoordinateRegion,
) -> tuple[float, float]:
    if len(coordinate_region) != 2:
        raise StateSpaceError("real-interval coordinate region must be a numeric pair")
    lower, upper = coordinate_region
    for value in (lower, upper):
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise StateSpaceError("real-interval coordinate region bounds must be finite")
    if upper <= lower:
        raise StateSpaceError("real-interval coordinate region upper must exceed lower")
    return float(lower), float(upper)


def _coordinate_cells(coordinate_region: DiscreteAxisCoordinateRegion) -> tuple[str, ...]:
    if not coordinate_region or any(
        type(value) is not str or not value for value in coordinate_region
    ):
        raise StateSpaceError("enumerated-cells coordinate region must select nonempty cell ids")
    cells = cast(tuple[str, ...], coordinate_region)
    if len(set(cells)) != len(cells):
        raise StateSpaceError("enumerated-cells coordinate region cells must be unique")
    return cells


def _coordinate_bit_indices(coordinate_region: DiscreteAxisCoordinateRegion) -> tuple[int, ...]:
    if any(type(value) is not int for value in coordinate_region):
        raise StateSpaceError("binary-vector coordinate region must enable integer indices")
    indices = cast(tuple[int, ...], coordinate_region)
    if len(set(indices)) != len(indices):
        raise StateSpaceError("binary-vector coordinate region indices must be unique")
    return indices


def _validate_log2(declared: float, count: int, *, label: str) -> None:
    if type(declared) not in (int, float) or not math.isfinite(float(declared)):
        raise StateSpaceError(f"{label} must be a finite number")
    if abs(float(declared) - math.log2(count)) > _log2_tolerance:
        raise StateSpaceError(f"{label} must equal log2 of the declared count")


def _measure_estimate_is_estimated(estimate: MeasureEstimate | None) -> bool:
    return estimate is not None and estimate.estimated


def _validate_estimated_log2_volume(
    declared: float,
    estimate: MeasureEstimate,
    *,
    label: str,
) -> None:
    if type(declared) not in (int, float) or not math.isfinite(float(declared)):
        raise StateSpaceError(f"{label} must be finite")
    lower = cast(float, estimate.log2_lower)
    upper = cast(float, estimate.log2_upper)
    if not lower - _log2_tolerance <= float(declared) <= upper + _log2_tolerance:
        raise StateSpaceError(f"{label} must lie within the measure estimate bounds")


def _validate_estimated_volume(volume: int, estimate: MeasureEstimate, *, label: str) -> None:
    """Require an estimated region's integer volume to lie within its bracket.

    ``volume`` and ``log2_volume`` are independent representative points of an
    estimated region; both must lie inside the declared bracket. The check is
    in bits (``log2(volume)``), so it never materializes ``2**bits``.
    """

    lower = cast(float, estimate.log2_lower)
    upper = cast(float, estimate.log2_upper)
    log2_volume = math.log2(volume)
    if not lower - _log2_tolerance <= log2_volume <= upper + _log2_tolerance:
        raise StateSpaceError(f"{label} volume must lie within the measure estimate bounds")


def _log2_sum_exp(log2_values: Sequence[float]) -> float:
    """Return ``log2(sum(2**v for v in log2_values))`` without overflow.

    Summing covering-measure brackets is addition of state counts -- a sum in
    linear space -- but evaluating it as ``2**log2`` overflows and loses
    precision once bit counts are large. Shifting by the maximum exponent keeps
    every term in ``(0, 1]`` and recovers the sum in bits, so the bounds stay in
    the same unit the rest of the module measures in.
    """

    highest = max(log2_values)
    if not math.isfinite(highest):
        return highest
    shifted = math.fsum(2.0 ** (value - highest) for value in log2_values)
    return highest + math.log2(shifted)


def _measure_estimate_log2_interval(estimate: MeasureEstimate) -> tuple[float, float]:
    return (cast(float, estimate.log2_lower), cast(float, estimate.log2_upper))


def _product_region_log2_interval(region: ProductRegion) -> tuple[float, float]:
    if _measure_estimate_is_estimated(region.measure_estimate):
        return _measure_estimate_log2_interval(cast(MeasureEstimate, region.measure_estimate))
    return (math.log2(region.volume), math.log2(region.volume))


def _state_space_region_log2_interval(region: StateSpaceRegion) -> tuple[float, float]:
    if _measure_estimate_is_estimated(region.measure_estimate):
        return _measure_estimate_log2_interval(cast(MeasureEstimate, region.measure_estimate))
    return (math.log2(region.volume), math.log2(region.volume))


def _sum_log2_measure_intervals(
    intervals: Iterable[tuple[float, float]],
) -> tuple[float, float]:
    materialized = tuple(intervals)
    lower = _log2_sum_exp([interval[0] for interval in materialized])
    upper = _log2_sum_exp([interval[1] for interval in materialized])
    return lower, upper


def _validate_scalar_mapping(mapping: Mapping[str, object], *, label: str) -> None:
    if not mapping:
        raise StateSpaceError(f"{label} must not be empty")
    for key, value in mapping.items():
        if type(key) is not str or not key:
            raise StateSpaceError(f"{label} keys must be nonempty strings")
        if type(value) not in (int, float, str):
            raise StateSpaceError(f"{label} values must be integers, floats, or strings")


def _validate_box_field_domain(
    field_domain_kind: str,
    field_domain: Mapping[str, object],
    *,
    label: str,
) -> None:
    extent_keys = _box_field_domain_extents.get(field_domain_kind)
    if extent_keys is None:
        return
    for key in extent_keys:
        value = field_domain.get(key)
        if type(value) not in (int, float):
            raise StateSpaceError(f"{label} {key} must be a finite positive number")
        extent = float(cast(int | float, value))
        if not math.isfinite(extent) or extent <= 0.0:
            raise StateSpaceError(f"{label} {key} must be a finite positive number")
    boundary_id = field_domain.get("boundary_id")
    if type(boundary_id) is not str or not boundary_id:
        raise StateSpaceError(f"{label} boundary_id must be a nonempty string")


def _record_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StateSpaceError(f"{label} must be a mapping")
    for key in cast(Mapping[object, object], value):
        if type(key) is not str:
            raise StateSpaceError(f"{label} keys must be strings")
    return cast(Mapping[str, object], value)


def _record_sequence(record: Mapping[str, object], key: str, *, label: str) -> tuple[object, ...]:
    value = record.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise StateSpaceError(f"{label} {key} must be a sequence")
    return tuple(cast(Sequence[object], value))


def _record_str(record: Mapping[str, object], key: str, *, label: str) -> str:
    value = record.get(key)
    if type(value) is not str or not value:
        raise StateSpaceError(f"{label} {key} must be a nonempty string")
    return value


def _record_optional_str(record: Mapping[str, object], key: str, *, label: str) -> str | None:
    value = record.get(key)
    if value is None:
        return None
    if type(value) is not str or not value:
        raise StateSpaceError(f"{label} {key} must be a nonempty string when present")
    return value


def _record_optional_int(
    record: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if type(value) is not int:
        raise StateSpaceError(f"{label} {key} must be an integer when present")
    return value


def _record_optional_float(
    record: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> float | None:
    value = record.get(key)
    if value is None:
        return None
    if type(value) not in (int, float):
        raise StateSpaceError(f"{label} {key} must be a number when present")
    return float(cast(float, value))


def _record_int(record: Mapping[str, object], key: str, *, label: str) -> int:
    value = record.get(key)
    if type(value) is not int:
        raise StateSpaceError(f"{label} {key} must be an integer")
    return value


def _record_float(record: Mapping[str, object], key: str, *, label: str) -> float:
    value = record.get(key)
    if type(value) not in (int, float):
        raise StateSpaceError(f"{label} {key} must be a number")
    return float(cast(float, value))
