"""Deterministic composition of proposal selectors."""

from __future__ import annotations

from dataclasses import dataclass

from leibniz.candidate_observations import ArchitectureCandidateObservation
from leibniz.resource_selection import select_resource_bootstrap_candidates

__all__ = [
    "CandidateProposalSelection",
    "ProposalSelectionError",
    "select_candidate_proposals",
]


class ProposalSelectionError(ValueError):
    """Raised when deterministic proposal selector composition cannot proceed."""


@dataclass(frozen=True, slots=True)
class CandidateProposalSelection:
    """One candidate selected for proposal, with compact selector reasoning."""

    observation: ArchitectureCandidateObservation
    selector_name: str
    source_candidate_rank: int
    comparable_cost_best_score: float
    resource_stratum_index: int | None = None
    resource_stratum_count: int | None = None

    def __post_init__(self) -> None:
        if not self.selector_name:
            raise ProposalSelectionError("selector_name must be nonempty")
        if type(self.source_candidate_rank) is not int or self.source_candidate_rank < 1:
            raise ProposalSelectionError("source_candidate_rank must be positive")
        if self.comparable_cost_best_score < 0 or self.comparable_cost_best_score > 1:
            raise ProposalSelectionError("comparable_cost_best_score must be a probability")
        if (self.resource_stratum_index is None) != (self.resource_stratum_count is None):
            raise ProposalSelectionError("resource stratum fields must be provided together")
        if self.resource_stratum_index is not None:
            if type(self.resource_stratum_index) is not int or self.resource_stratum_index < 0:
                raise ProposalSelectionError("resource_stratum_index must be nonnegative")
            if type(self.resource_stratum_count) is not int or self.resource_stratum_count < 1:
                raise ProposalSelectionError("resource_stratum_count must be positive")
            if self.resource_stratum_index >= self.resource_stratum_count:
                raise ProposalSelectionError(
                    "resource_stratum_index must be less than resource_stratum_count"
                )


def select_candidate_proposals(
    observations: tuple[ArchitectureCandidateObservation, ...],
    *,
    budget: int,
) -> tuple[CandidateProposalSelection, ...]:
    """Compose deterministic selectors into one proposal candidate list."""

    if type(budget) is not int or budget < 1:
        raise ProposalSelectionError("budget must be positive")
    if not observations:
        raise ProposalSelectionError("observations must contain at least one item")
    resource_selections = select_resource_bootstrap_candidates(
        observations,
        budget=budget,
    )
    return tuple(
        CandidateProposalSelection(
            observation=selection.observation,
            selector_name=selection.selector_name,
            source_candidate_rank=selection.observation.source_candidate_rank,
            comparable_cost_best_score=(
                selection.observation.best_measured_score_at_or_below_cost
            ),
            resource_stratum_index=selection.resource_stratum_index,
            resource_stratum_count=selection.resource_stratum_count,
        )
        for selection in resource_selections
    )
