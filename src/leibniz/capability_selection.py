"""Deterministic capability selectors for architecture candidates."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from leibniz.candidate_observations import ArchitectureCandidateObservation
from leibniz.identifiers import ProtocolIdentifier

__all__ = [
    "CapabilityBootstrapSelection",
    "CapabilityKey",
    "CapabilitySelectionError",
    "capability_key_for_observation",
    "select_capability_bootstrap_candidates",
]


class CapabilitySelectionError(ValueError):
    """Raised when capability bootstrap selection cannot proceed."""


@dataclass(frozen=True, slots=True, order=True)
class CapabilityKey:
    """Generic capability key derived from formal candidate metadata."""

    family_kind: str
    operator_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.family_kind:
            raise CapabilitySelectionError("family_kind must be nonempty")
        if not self.operator_kinds or any(not kind for kind in self.operator_kinds):
            raise CapabilitySelectionError("operator_kinds must be nonempty")


@dataclass(frozen=True, slots=True)
class CapabilityBootstrapSelection:
    """One candidate selected by deterministic capability coverage."""

    observation: ArchitectureCandidateObservation
    selector_name: str
    capability_key: CapabilityKey
    measured_count_for_key: int
    selected_count_for_key: int

    def __post_init__(self) -> None:
        if not self.selector_name:
            raise CapabilitySelectionError("selector_name must be nonempty")
        if type(self.measured_count_for_key) is not int or self.measured_count_for_key < 0:
            raise CapabilitySelectionError("measured_count_for_key must be nonnegative")
        if type(self.selected_count_for_key) is not int or self.selected_count_for_key < 0:
            raise CapabilitySelectionError("selected_count_for_key must be nonnegative")


def capability_key_for_observation(
    observation: ArchitectureCandidateObservation,
) -> CapabilityKey:
    """Return the formal capability key for a candidate observation."""

    return CapabilityKey(
        family_kind=observation.family_kind,
        operator_kinds=observation.operator_kinds,
    )


def select_capability_bootstrap_candidates(
    observations: tuple[ArchitectureCandidateObservation, ...],
    *,
    budget: int,
    excluded_candidate_ids: tuple[ProtocolIdentifier, ...] = (),
) -> tuple[CapabilityBootstrapSelection, ...]:
    """Select unmeasured candidates to cover formal capability keys."""

    if type(budget) is not int or budget < 1:
        raise CapabilitySelectionError("budget must be positive")
    if not observations:
        raise CapabilitySelectionError("observations must contain at least one item")

    excluded = set(excluded_candidate_ids)
    remaining: list[ArchitectureCandidateObservation] = [
        observation
        for observation in observations
        if not observation.is_measured and observation.candidate_id not in excluded
    ]
    if not remaining:
        return ()

    measured_counts = _measured_counts_by_key(observations)
    selected_counts: dict[CapabilityKey, int] = {}
    selected: list[CapabilityBootstrapSelection] = []
    while remaining and len(selected) < budget:
        observation = min(
            remaining,
            key=_candidate_sort_key(
                measured_counts=measured_counts,
                selected_counts=selected_counts,
            ),
        )
        key = capability_key_for_observation(observation)
        selected.append(
            CapabilityBootstrapSelection(
                observation=observation,
                selector_name="capability-bootstrap",
                capability_key=key,
                measured_count_for_key=measured_counts.get(key, 0),
                selected_count_for_key=selected_counts.get(key, 0),
            )
        )
        selected_counts[key] = selected_counts.get(key, 0) + 1
        remaining = [item for item in remaining if item.candidate_id != observation.candidate_id]
    return tuple(selected)


def _measured_counts_by_key(
    observations: tuple[ArchitectureCandidateObservation, ...],
) -> dict[CapabilityKey, int]:
    counts: dict[CapabilityKey, int] = {}
    for observation in observations:
        if not observation.is_measured:
            continue
        key = capability_key_for_observation(observation)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _candidate_sort_key(
    *,
    measured_counts: dict[CapabilityKey, int],
    selected_counts: dict[CapabilityKey, int],
) -> Callable[[ArchitectureCandidateObservation], tuple[int, CapabilityKey, int, str]]:
    def sort_key(
        observation: ArchitectureCandidateObservation,
    ) -> tuple[int, CapabilityKey, int, str]:
        key = capability_key_for_observation(observation)
        return (
            measured_counts.get(key, 0) + selected_counts.get(key, 0),
            key,
            observation.source_candidate_rank,
            str(observation.candidate_id),
        )

    return sort_key
