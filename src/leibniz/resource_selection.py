"""Deterministic resource-axis selectors for architecture candidates."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from leibniz.candidate_observations import ArchitectureCandidateObservation
from leibniz.content import ContentDigest

__all__ = [
    "ResourceBootstrapSelection",
    "ResourceBootstrapSelectionError",
    "select_resource_bootstrap_candidates",
]


class ResourceBootstrapSelectionError(ValueError):
    """Raised when resource bootstrap selection cannot proceed."""


@dataclass(frozen=True, slots=True)
class ResourceBootstrapSelection:
    """One candidate selected by the deterministic resource bootstrap."""

    observation: ArchitectureCandidateObservation
    selector_name: str
    resource_stratum_index: int
    resource_stratum_count: int
    measured_count_in_stratum: int
    selected_count_in_stratum: int

    def __post_init__(self) -> None:
        if not self.selector_name:
            raise ResourceBootstrapSelectionError("selector_name must be nonempty")
        if type(self.resource_stratum_index) is not int or self.resource_stratum_index < 0:
            raise ResourceBootstrapSelectionError("resource_stratum_index must be nonnegative")
        if type(self.resource_stratum_count) is not int or self.resource_stratum_count < 1:
            raise ResourceBootstrapSelectionError("resource_stratum_count must be positive")
        if self.resource_stratum_index >= self.resource_stratum_count:
            raise ResourceBootstrapSelectionError(
                "resource_stratum_index must be less than resource_stratum_count"
            )
        if type(self.measured_count_in_stratum) is not int or (
            self.measured_count_in_stratum < 0
        ):
            raise ResourceBootstrapSelectionError("measured_count_in_stratum must be nonnegative")
        if type(self.selected_count_in_stratum) is not int or (
            self.selected_count_in_stratum < 0
        ):
            raise ResourceBootstrapSelectionError("selected_count_in_stratum must be nonnegative")


def select_resource_bootstrap_candidates(
    observations: tuple[ArchitectureCandidateObservation, ...],
    *,
    budget: int,
) -> tuple[ResourceBootstrapSelection, ...]:
    """Select unmeasured candidates to cover the candidate resource axis."""

    if type(budget) is not int or budget < 1:
        raise ResourceBootstrapSelectionError("budget must be positive")
    if not observations:
        raise ResourceBootstrapSelectionError("observations must contain at least one item")
    unmeasured = tuple(observation for observation in observations if not observation.is_measured)
    if not unmeasured:
        return ()

    stratum_count = min(budget, len(unmeasured))
    coordinates = _resource_coordinates(observations)
    stratum_by_digest = {
        observation.architecture_digest: _stratum_index(
            coordinate=coordinates[observation.architecture_digest],
            minimum=min(coordinates.values()),
            maximum=max(coordinates.values()),
            stratum_count=stratum_count,
        )
        for observation in observations
    }
    measured_counts = tuple(
        sum(
            1
            for observation in observations
            if observation.is_measured
            and stratum_by_digest[observation.architecture_digest] == stratum_index
        )
        for stratum_index in range(stratum_count)
    )
    selected_counts = [0 for _index in range(stratum_count)]
    selected: list[ResourceBootstrapSelection] = []
    remaining: list[ArchitectureCandidateObservation] = list(unmeasured)
    while remaining and len(selected) < budget:
        stratum_index = min(
            range(stratum_count),
            key=lambda index: (
                measured_counts[index] + selected_counts[index],
                index,
            ),
        )
        candidates: tuple[ArchitectureCandidateObservation, ...] = tuple(
            observation
            for observation in remaining
            if stratum_by_digest[observation.architecture_digest] == stratum_index
        )
        if not candidates:
            selected_counts[stratum_index] += 1
            continue
        observation = min(
            candidates,
            key=_candidate_sort_key(
                coordinates=coordinates,
                stratum_index=stratum_index,
                stratum_count=stratum_count,
            ),
        )
        selected.append(
            ResourceBootstrapSelection(
                observation=observation,
                selector_name="resource-bootstrap",
                resource_stratum_index=stratum_index,
                resource_stratum_count=stratum_count,
                measured_count_in_stratum=measured_counts[stratum_index],
                selected_count_in_stratum=selected_counts[stratum_index],
            )
        )
        selected_counts[stratum_index] += 1
        remaining = [item for item in remaining if item.candidate_id != observation.candidate_id]
    return tuple(selected)


def _resource_coordinates(
    observations: tuple[ArchitectureCandidateObservation, ...],
) -> dict[ContentDigest, float]:
    return {
        observation.architecture_digest: math.log1p(observation.parameter_count)
        for observation in observations
    }


def _stratum_index(
    *,
    coordinate: float,
    minimum: float,
    maximum: float,
    stratum_count: int,
) -> int:
    if maximum == minimum:
        return 0
    position = (coordinate - minimum) / (maximum - minimum)
    return min(stratum_count - 1, int(position * stratum_count))


def _distance_to_stratum_center(
    *,
    coordinate: float,
    stratum_index: int,
    stratum_count: int,
    minimum: float,
    maximum: float,
) -> float:
    if maximum == minimum:
        return 0.0
    width = (maximum - minimum) / stratum_count
    center = minimum + width * (stratum_index + 0.5)
    return abs(coordinate - center)


def _candidate_sort_key(
    *,
    coordinates: dict[ContentDigest, float],
    stratum_index: int,
    stratum_count: int,
) -> Callable[[ArchitectureCandidateObservation], tuple[float, int, str]]:
    minimum = min(coordinates.values())
    maximum = max(coordinates.values())

    def sort_key(item: ArchitectureCandidateObservation) -> tuple[float, int, str]:
        return (
            _distance_to_stratum_center(
                coordinate=coordinates[item.architecture_digest],
                stratum_index=stratum_index,
                stratum_count=stratum_count,
                minimum=minimum,
                maximum=maximum,
            ),
            item.source_candidate_rank,
            str(item.candidate_id),
        )

    return sort_key
