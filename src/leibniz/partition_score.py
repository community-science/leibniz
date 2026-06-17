"""Measure-weighted competence integrals over state-space partitions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from leibniz.observation_generation import GeneratedSample
from leibniz.state_space import (
    ProductRegion,
    StateSpaceError,
    StateSpaceRegion,
    state_space_region_contains,
    state_space_regions_are_disjoint,
)

__all__ = [
    "PartitionCompetenceEstimate",
    "PartitionSample",
    "PartitionScore",
    "PartitionScoreNode",
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
    children: tuple["PartitionScoreNode", ...] = ()

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def leaves(self) -> tuple["PartitionScoreNode", ...]:
        if self.is_leaf:
            return (self,)
        return tuple(leaf for child in self.children for leaf in child.leaves())

    def to_record(self) -> dict[str, object]:
        record = self.estimate.to_record()
        record["children"] = [child.to_record() for child in self.children]
        return record


@dataclass(frozen=True, slots=True)
class PartitionScore:
    """A normalized measure-weighted competence integral over partition leaves."""

    root: PartitionScoreNode
    value: float
    confidence_half_width: float
    confidence_method_id: str
    sample_count: int
    total_measure: float
    unassigned_sample_count: int = 0

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "measure-weighted-partition-competence-integral-v1",
            "value": self.value,
            "confidence_half_width": self.confidence_half_width,
            "confidence_method_id": self.confidence_method_id,
            "sample_count": self.sample_count,
            "total_measure": self.total_measure,
            "unassigned_sample_count": self.unassigned_sample_count,
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
    value = math.fsum(
        child.estimate.measure * child.estimate.competence for child in child_nodes
    ) / total_measure
    confidence_half_width = math.sqrt(
        math.fsum(
            ((child.estimate.measure / total_measure) * child.estimate.confidence_half_width)
            ** 2
            for child in child_nodes
        )
    )
    assigned = {
        sample.sample_index
        for sample in samples
        if any(_sample_in_region(root_region, sample, region) for region in partition)
    }
    return PartitionScore(
        root=PartitionScoreNode(estimate=root_estimate, children=child_nodes),
        value=value,
        confidence_half_width=confidence_half_width,
        confidence_method_id="normal-sample-mean-1.96se",
        sample_count=len(samples),
        total_measure=total_measure,
        unassigned_sample_count=len(samples) - len(assigned),
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
    if leaf_component.stratum_id is not None and leaf_component.stratum_id != root_component.stratum_id:
        return False
    if root_component.stratum_id is not None and leaf_component.stratum_id is None:
        return False
    return leaf_component.contains(cast(Mapping[str, object], coordinates))
