"""Measure-weighted competence integrals over state-space partitions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from leibniz.observation_generation import GeneratedSample
from leibniz.state_space import (
    AxisRegion,
    ContinuousAxisRegion,
    DiscreteAxisRegion,
    IntegerRangeDomain,
    MeasureEstimate,
    ProductRegion,
    RealGridDomain,
    StateSpaceError,
    StateSpaceRegion,
    state_space_region_contains,
    state_space_regions_are_disjoint,
)

__all__ = [
    "PartitionCompetenceEstimate",
    "PartitionRefinementStep",
    "PartitionSample",
    "PartitionScore",
    "PartitionScoreNode",
    "adversarial_partition_competence_integral",
    "fixed_partition_competence_integral",
    "partition_samples_from_generated",
]


@dataclass(frozen=True, slots=True)
class PartitionSample:
    """A per-sample competence value located in the state-space region grammar."""

    sample_index: int
    competence: float
    region_component_index: int
    axis_coordinates: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.sample_index) is not int or self.sample_index < 0:
            raise ValueError("partition sample index must be a nonnegative integer")
        if type(self.region_component_index) is not int or self.region_component_index < 0:
            raise ValueError("partition sample region_component_index must be nonnegative")
        if type(self.competence) not in (int, float):
            raise ValueError("partition sample competence must be numeric")
        if not math.isfinite(float(self.competence)) or float(self.competence) < 0.0:
            raise ValueError("partition sample competence must be finite and nonnegative")
        if not self.axis_coordinates:
            raise ValueError("partition sample axis_coordinates must be nonempty")


@dataclass(frozen=True, slots=True)
class PartitionCompetenceEstimate:
    """A region-local competence estimate from samples assigned to that region."""

    region: StateSpaceRegion
    measure: float
    sample_count: int
    competence: float
    confidence_half_width: float
    confidence_method_id: str

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "partition-region-competence-estimate-v1",
            "region": self.region.to_record(),
            "measure": self.measure,
            "sample_count": self.sample_count,
            "competence": self.competence,
            "confidence_half_width": self.confidence_half_width,
            "confidence_method_id": self.confidence_method_id,
        }


@dataclass(frozen=True, slots=True)
class PartitionScoreNode:
    """One node in a competence partition tree."""

    estimate: PartitionCompetenceEstimate
    children: tuple[PartitionScoreNode, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> tuple[PartitionScoreNode, ...]:
        if self.is_leaf:
            return (self,)
        return tuple(leaf for child in self.children for leaf in child.leaves())

    def to_record(self) -> dict[str, object]:
        record = self.estimate.to_record()
        record["children"] = [child.to_record() for child in self.children]
        return record


@dataclass(frozen=True, slots=True)
class PartitionRefinementStep:
    """One value on the partition-refinement convergence ladder."""

    depth: int
    leaf_count: int
    value: float
    confidence_half_width: float
    movement: float | None = None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": "partition-refinement-step-v1",
            "depth": self.depth,
            "leaf_count": self.leaf_count,
            "value": self.value,
            "confidence_half_width": self.confidence_half_width,
        }
        if self.movement is not None:
            record["movement"] = self.movement
        return record


@dataclass(frozen=True, slots=True)
class PartitionScore:
    """An extensive competence integral over partition leaves."""

    root: PartitionScoreNode
    value: float
    confidence_half_width: float
    confidence_method_id: str
    sample_count: int
    total_measure: float
    score_width_bits: float
    mean_competence: float
    mean_competence_confidence_half_width: float
    unassigned_sample_count: int = 0
    refinement_ladder: tuple[PartitionRefinementStep, ...] = ()

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "measure-weighted-partition-competence-integral-v1",
            "value": self.value,
            "confidence_half_width": self.confidence_half_width,
            "confidence_method_id": self.confidence_method_id,
            "sample_count": self.sample_count,
            "total_measure": self.total_measure,
            "score_width_bits": self.score_width_bits,
            "mean_competence": self.mean_competence,
            "mean_competence_confidence_half_width": (
                self.mean_competence_confidence_half_width
            ),
            "unassigned_sample_count": self.unassigned_sample_count,
            "refinement_ladder": [step.to_record() for step in self.refinement_ladder],
            "root": self.root.to_record(),
        }


def partition_samples_from_generated(
    samples: Sequence[GeneratedSample],
    competence_by_sample_index: Mapping[int, float],
) -> tuple[PartitionSample, ...]:
    """Join generated sample coordinates to per-sample competence by sample index."""

    partition_samples: list[PartitionSample] = []
    for sample in samples:
        if sample.index not in competence_by_sample_index:
            raise ValueError(f"missing competence for sample index {sample.index}")
        if sample.region_component_index is None or sample.axis_coordinates is None:
            raise ValueError("partition scoring requires region sample coordinates")
        partition_samples.append(
            PartitionSample(
                sample_index=sample.index,
                competence=float(competence_by_sample_index[sample.index]),
                region_component_index=sample.region_component_index,
                axis_coordinates=sample.axis_coordinates,
            )
        )
    return tuple(partition_samples)


def fixed_partition_competence_integral(
    *,
    root_region: StateSpaceRegion,
    samples: Sequence[PartitionSample],
    partition: Sequence[StateSpaceRegion],
    score_width_bits: float | None = None,
    confidence_z: float = 1.96,
) -> PartitionScore:
    """Estimate a competence integral over a caller-supplied state-space partition."""

    if not partition:
        raise ValueError("partition must contain at least one region")
    _validate_partition(root_region=root_region, partition=partition)
    root_estimate = _estimate_region(
        root_region=root_region,
        region=root_region,
        samples=samples,
        confidence_z=confidence_z,
        allow_empty=False,
    )
    child_nodes = tuple(
        PartitionScoreNode(
            estimate=_estimate_region(
                root_region=root_region,
                region=region,
                samples=samples,
                confidence_z=confidence_z,
                allow_empty=False,
            )
        )
        for region in partition
    )
    total_measure = math.fsum(child.estimate.measure for child in child_nodes)
    if total_measure <= 0.0:
        raise ValueError("partition measure must be positive")
    assigned = {
        sample.sample_index
        for sample in samples
        if any(_sample_in_region(root_region, sample, region) for region in partition)
    }
    root_node = PartitionScoreNode(estimate=root_estimate, children=child_nodes)
    bit_width = _score_width_bits(root_region, score_width_bits)
    value, confidence_half_width = _integral_for_nodes(
        child_nodes,
        score_width_bits=bit_width,
    )
    mean_competence, mean_half_width = _mean_competence_for_nodes(child_nodes)
    return PartitionScore(
        root=root_node,
        value=value,
        confidence_half_width=confidence_half_width,
        confidence_method_id="normal-sample-mean-1.96se",
        sample_count=len(samples),
        total_measure=total_measure,
        score_width_bits=bit_width,
        mean_competence=mean_competence,
        mean_competence_confidence_half_width=mean_half_width,
        unassigned_sample_count=len(samples) - len(assigned),
    )


def adversarial_partition_competence_integral(
    *,
    root_region: StateSpaceRegion,
    samples: Sequence[PartitionSample],
    score_width_bits: float | None = None,
    confidence_z: float = 1.96,
) -> PartitionScore:
    """Refine a partition while child disparity exceeds measured sampling noise."""

    root_node = _adversarial_node(
        root_region=root_region,
        region=root_region,
        samples=samples,
        confidence_z=confidence_z,
    )
    leaves = root_node.leaves()
    bit_width = _score_width_bits(root_region, score_width_bits)
    value, confidence_half_width = _integral_for_nodes(
        leaves,
        score_width_bits=bit_width,
    )
    mean_competence, mean_half_width = _mean_competence_for_nodes(leaves)
    assigned = {
        sample.sample_index
        for sample in samples
        if _sample_in_region(root_region, sample, root_region)
    }
    ladder = _refinement_ladder(root_node, score_width_bits=bit_width)
    return PartitionScore(
        root=root_node,
        value=value,
        confidence_half_width=confidence_half_width,
        confidence_method_id="normal-sample-mean-1.96se",
        sample_count=len(samples),
        total_measure=math.fsum(leaf.estimate.measure for leaf in leaves),
        score_width_bits=bit_width,
        mean_competence=mean_competence,
        mean_competence_confidence_half_width=mean_half_width,
        unassigned_sample_count=len(samples) - len(assigned),
        refinement_ladder=ladder,
    )


def _validate_partition(
    *,
    root_region: StateSpaceRegion,
    partition: Sequence[StateSpaceRegion],
) -> None:
    for index, region in enumerate(partition):
        try:
            if not state_space_region_contains(root_region, region):
                raise ValueError("partition regions must be contained in the root region")
        except StateSpaceError as error:
            raise ValueError(f"partition[{index}] is not comparable to root: {error}") from error
    for left_index, left in enumerate(partition):
        for right_index in range(left_index + 1, len(partition)):
            try:
                disjoint = state_space_regions_are_disjoint(left, partition[right_index])
            except StateSpaceError as error:
                raise ValueError(
                    f"partition[{left_index}] is not comparable to partition[{right_index}]: "
                    f"{error}"
                ) from error
            if not disjoint:
                raise ValueError("partition regions must be pairwise disjoint")


def _estimate_region(
    *,
    root_region: StateSpaceRegion,
    region: StateSpaceRegion,
    samples: Sequence[PartitionSample],
    confidence_z: float,
    allow_empty: bool,
) -> PartitionCompetenceEstimate:
    values = tuple(
        sample.competence for sample in samples if _sample_in_region(root_region, sample, region)
    )
    if not values and not allow_empty:
        raise ValueError(f"region {region.id} has no assigned samples")
    mean = math.fsum(values) / len(values) if values else 0.0
    half_width = _sample_mean_half_width(values, confidence_z=confidence_z)
    return PartitionCompetenceEstimate(
        region=region,
        measure=float(region.volume),
        sample_count=len(values),
        competence=mean,
        confidence_half_width=half_width,
        confidence_method_id="normal-sample-mean-1.96se",
    )


def _adversarial_node(
    *,
    root_region: StateSpaceRegion,
    region: StateSpaceRegion,
    samples: Sequence[PartitionSample],
    confidence_z: float,
) -> PartitionScoreNode:
    estimate = _estimate_region(
        root_region=root_region,
        region=region,
        samples=samples,
        confidence_z=confidence_z,
        allow_empty=False,
    )
    split = _best_significant_split(
        root_region=root_region,
        region=region,
        samples=samples,
        confidence_z=confidence_z,
    )
    if split is None:
        return PartitionScoreNode(estimate=estimate)
    left, right = split
    return PartitionScoreNode(
        estimate=estimate,
        children=(
            _adversarial_node(
                root_region=root_region,
                region=left,
                samples=samples,
                confidence_z=confidence_z,
            ),
            _adversarial_node(
                root_region=root_region,
                region=right,
                samples=samples,
                confidence_z=confidence_z,
            ),
        ),
    )


def _best_significant_split(
    *,
    root_region: StateSpaceRegion,
    region: StateSpaceRegion,
    samples: Sequence[PartitionSample],
    confidence_z: float,
) -> tuple[StateSpaceRegion, StateSpaceRegion] | None:
    best: tuple[float, tuple[StateSpaceRegion, StateSpaceRegion]] | None = None
    for left, right in _candidate_splits(region):
        left_estimate = _estimate_region(
            root_region=root_region,
            region=left,
            samples=samples,
            confidence_z=confidence_z,
            allow_empty=True,
        )
        right_estimate = _estimate_region(
            root_region=root_region,
            region=right,
            samples=samples,
            confidence_z=confidence_z,
            allow_empty=True,
        )
        if left_estimate.sample_count == 0 or right_estimate.sample_count == 0:
            continue
        disparity = abs(left_estimate.competence - right_estimate.competence)
        noise = _measured_split_noise(left_estimate, right_estimate)
        excess = disparity - noise
        if excess <= 0.0:
            continue
        if best is None or excess > best[0]:
            best = (excess, (left, right))
    if best is None:
        return None
    return best[1]


def _measured_split_noise(
    left: PartitionCompetenceEstimate,
    right: PartitionCompetenceEstimate,
) -> float:
    if left.sample_count < 2 or right.sample_count < 2:
        return math.inf
    return math.sqrt(left.confidence_half_width**2 + right.confidence_half_width**2)


def _candidate_splits(
    region: StateSpaceRegion,
) -> tuple[tuple[StateSpaceRegion, StateSpaceRegion], ...]:
    candidates: list[tuple[StateSpaceRegion, StateSpaceRegion]] = []
    if len(region.components) > 1:
        for index in range(len(region.components)):
            left_components = (region.components[index],)
            right_components = (
                region.components[:index] + region.components[index + 1 :]
            )
            candidates.append(
                (
                    _region_from_components(
                        region,
                        f"{region.id}.component-{index}",
                        left_components,
                    ),
                    _region_from_components(
                        region,
                        f"{region.id}.not-component-{index}",
                        right_components,
                    ),
                )
            )
    if len(region.components) == 1:
        component = region.components[0]
        for axis_index, axis_region in enumerate(component.axis_regions):
            axis_split = _split_axis_region(axis_region)
            if axis_split is None:
                continue
            left_axis, right_axis = axis_split
            left_component = _product_with_axis_region(component, axis_index, left_axis)
            right_component = _product_with_axis_region(component, axis_index, right_axis)
            candidates.append(
                (
                    _region_from_components(
                        region,
                        f"{region.id}.{left_axis.axis_id}-low",
                        (left_component,),
                    ),
                    _region_from_components(
                        region,
                        f"{region.id}.{right_axis.axis_id}-high",
                        (right_component,),
                    ),
                )
            )
    return tuple(candidates)


def _region_from_components(
    parent: StateSpaceRegion,
    region_id: str,
    components: tuple[ProductRegion, ...],
) -> StateSpaceRegion:
    volume = sum(component.volume for component in components)
    estimate = _summed_measure_estimate(
        method_id=f"{region_id}.component-measure-sum",
        intervals=tuple(_product_measure_interval(component) for component in components),
        force_estimated=any(component.measure_estimate is not None for component in components),
    )
    return StateSpaceRegion(
        id=region_id,
        ambient=parent.ambient,
        components=components,
        union_rule=parent.union_rule,
        volume=volume,
        log2_volume=math.log2(volume),
        measure_estimate=estimate,
    )


def _split_axis_region(axis_region: AxisRegion) -> tuple[AxisRegion, AxisRegion] | None:
    if isinstance(axis_region, ContinuousAxisRegion):
        return None
    domain = axis_region.axis.domain
    if not isinstance(domain, IntegerRangeDomain | RealGridDomain):
        return None
    lower, upper = cast(tuple[int, int], axis_region.coordinate_region)
    if lower >= upper:
        return None
    midpoint = (lower + upper) // 2
    left_count = midpoint - lower + 1
    right_count = upper - midpoint
    return (
        DiscreteAxisRegion(
            axis=axis_region.axis,
            coordinate_region=(lower, midpoint),
            count=left_count,
            log2_count=math.log2(left_count),
        ),
        DiscreteAxisRegion(
            axis=axis_region.axis,
            coordinate_region=(midpoint + 1, upper),
            count=right_count,
            log2_count=math.log2(right_count),
        ),
    )


def _product_with_axis_region(
    product: ProductRegion,
    axis_index: int,
    axis_region: AxisRegion,
) -> ProductRegion:
    axis_regions = tuple(
        axis_region if index == axis_index else existing
        for index, existing in enumerate(product.axis_regions)
    )
    volume = _scaled_product_volume(
        product=product,
        original_axis=product.axis_regions[axis_index],
        replacement_axis=axis_region,
    )
    estimate = _scaled_product_measure_estimate(
        product=product,
        original_axis=product.axis_regions[axis_index],
        replacement_axis=axis_region,
        volume=volume,
    )
    return ProductRegion(
        axis_regions=axis_regions,
        measure_rule=product.measure_rule,
        volume=volume,
        log2_volume=math.log2(volume),
        stratum_id=product.stratum_id,
        stratum_target=product.stratum_target,
        measure_estimate=estimate,
    )


def _scaled_product_measure_estimate(
    *,
    product: ProductRegion,
    original_axis: AxisRegion,
    replacement_axis: AxisRegion,
    volume: int,
) -> MeasureEstimate | None:
    if product.measure_estimate is None:
        return None
    original_lower, original_upper = _axis_region_measure_interval(original_axis)
    replacement_lower, replacement_upper = _axis_region_measure_interval(replacement_axis)
    lower = _log2_measure_ratio(replacement_lower, original_upper)
    upper = _log2_measure_ratio(replacement_upper, original_lower)
    product_lower, product_upper = _product_measure_interval(product)
    return MeasureEstimate(
        kind="estimated",
        method_id=f"{product.measure_estimate.method_id}.axis-split",
        log2_lower=product_lower + lower,
        log2_upper=product_upper + upper,
    )


def _scaled_product_volume(
    *,
    product: ProductRegion,
    original_axis: AxisRegion,
    replacement_axis: AxisRegion,
) -> int:
    if isinstance(original_axis, DiscreteAxisRegion) and isinstance(
        replacement_axis,
        DiscreteAxisRegion,
    ):
        scaled = product.volume * replacement_axis.count / original_axis.count
        return max(1, round(scaled))
    return product.volume


def _summed_measure_estimate(
    *,
    method_id: str,
    intervals: tuple[tuple[float, float], ...],
    force_estimated: bool = False,
) -> MeasureEstimate | None:
    if not force_estimated and all(
        math.isclose(lower, upper, rel_tol=0.0, abs_tol=1e-12)
        for lower, upper in intervals
    ):
        return None
    lower = math.log2(math.fsum(2**interval[0] for interval in intervals))
    upper = math.log2(math.fsum(2**interval[1] for interval in intervals))
    return MeasureEstimate(
        kind="estimated",
        method_id=method_id,
        log2_lower=lower,
        log2_upper=upper,
    )


def _product_measure_interval(product: ProductRegion) -> tuple[float, float]:
    if product.measure_estimate is None:
        return (product.log2_volume, product.log2_volume)
    assert product.measure_estimate.log2_lower is not None
    assert product.measure_estimate.log2_upper is not None
    return (
        float(product.measure_estimate.log2_lower),
        float(product.measure_estimate.log2_upper),
    )


def _axis_region_measure_interval(axis_region: AxisRegion) -> tuple[float, float]:
    if isinstance(axis_region, DiscreteAxisRegion):
        return (axis_region.log2_count, axis_region.log2_count)
    assert axis_region.measure_estimate.log2_lower is not None
    assert axis_region.measure_estimate.log2_upper is not None
    return (
        float(axis_region.measure_estimate.log2_lower),
        float(axis_region.measure_estimate.log2_upper),
    )


def _log2_measure_ratio(numerator_log2: float, denominator_log2: float) -> float:
    return numerator_log2 - denominator_log2


def _score_width_bits(root_region: StateSpaceRegion, requested: float | None) -> float:
    value = root_region.log2_volume if requested is None else requested
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError("score_width_bits must be finite and nonnegative")
    return float(value)


def _mean_competence_for_nodes(nodes: Sequence[PartitionScoreNode]) -> tuple[float, float]:
    total_measure = math.fsum(node.estimate.measure for node in nodes)
    if total_measure <= 0.0:
        raise ValueError("partition measure must be positive")
    value = math.fsum(
        node.estimate.measure * node.estimate.competence for node in nodes
    ) / total_measure
    confidence_half_width = math.sqrt(
        math.fsum(
            ((node.estimate.measure / total_measure) * node.estimate.confidence_half_width)
            ** 2
            for node in nodes
        )
    )
    return (value, confidence_half_width)


def _integral_for_nodes(
    nodes: Sequence[PartitionScoreNode],
    *,
    score_width_bits: float,
) -> tuple[float, float]:
    mean, mean_confidence_half_width = _mean_competence_for_nodes(nodes)
    return (
        score_width_bits * mean,
        score_width_bits * mean_confidence_half_width,
    )


def _refinement_ladder(
    root: PartitionScoreNode,
    *,
    score_width_bits: float,
) -> tuple[PartitionRefinementStep, ...]:
    max_depth = _node_depth(root)
    steps: list[PartitionRefinementStep] = []
    previous_value: float | None = None
    for depth in range(max_depth + 1):
        nodes = _nodes_at_depth(root, depth)
        value, confidence_half_width = _integral_for_nodes(
            nodes,
            score_width_bits=score_width_bits,
        )
        movement = None if previous_value is None else abs(value - previous_value)
        steps.append(
            PartitionRefinementStep(
                depth=depth,
                leaf_count=len(nodes),
                value=value,
                confidence_half_width=confidence_half_width,
                movement=movement,
            )
        )
        previous_value = value
    return tuple(steps)


def _node_depth(node: PartitionScoreNode) -> int:
    if not node.children:
        return 0
    return 1 + max(_node_depth(child) for child in node.children)


def _nodes_at_depth(
    node: PartitionScoreNode,
    depth: int,
) -> tuple[PartitionScoreNode, ...]:
    if depth <= 0 or not node.children:
        return (node,)
    return tuple(
        child_node
        for child in node.children
        for child_node in _nodes_at_depth(child, depth - 1)
    )


def _sample_mean_half_width(values: Sequence[float], *, confidence_z: float) -> float:
    if len(values) < 2:
        return 0.0
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return confidence_z * math.sqrt(variance / len(values))


def _sample_in_region(
    root_region: StateSpaceRegion,
    sample: PartitionSample,
    region: StateSpaceRegion,
) -> bool:
    if sample.region_component_index >= len(root_region.components):
        raise ValueError("sample region_component_index is outside the root region")
    root_component = root_region.components[sample.region_component_index]
    return any(
        _component_accepts_sample(
            leaf_component=component,
            root_component=root_component,
            coordinates=sample.axis_coordinates,
        )
        for component in region.components
    )


def _component_accepts_sample(
    *,
    leaf_component: ProductRegion,
    root_component: ProductRegion,
    coordinates: Mapping[str, object],
) -> bool:
    if (
        leaf_component.stratum_id is not None
        and leaf_component.stratum_id != root_component.stratum_id
    ):
        return False
    if root_component.stratum_id is not None and leaf_component.stratum_id is None:
        return False
    return leaf_component.contains(coordinates)
