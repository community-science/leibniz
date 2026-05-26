"""Deterministic active selection over declared artifacts."""

from __future__ import annotations

from dataclasses import dataclass

from leibniz.architectures import ArchitectureManifest
from leibniz.content import ContentDigest
from leibniz.measurements import MeasurementDataset
from leibniz.proposals import ExperimentProposal, ExperimentProposalSet
from leibniz.relationships import RelationshipFitRecord
from leibniz.submissions import SubmissionPackageManifest
from leibniz.surrogates import ArchitectureSurrogateRecord

__all__ = [
    "ActiveSelectionResult",
    "ActiveSelectionValidationError",
    "select_experiments",
]


class ActiveSelectionValidationError(ValueError):
    """Raised when active selection inputs are invalid."""


@dataclass(frozen=True, slots=True)
class ActiveSelectionResult:
    """Selected proposals with the source artifact identities used."""

    selected_proposals: tuple[ExperimentProposal, ...]
    selection_rule: str
    source_dataset_digest: ContentDigest
    source_proposal_set_digest: ContentDigest
    relationship_fit_digest: ContentDigest | None = None
    surrogate_digest: ContentDigest | None = None

    def __post_init__(self) -> None:
        if not self.selected_proposals:
            raise ActiveSelectionValidationError(
                "selected_proposals must contain at least one proposal"
            )
        if not self.selection_rule:
            raise ActiveSelectionValidationError("selection_rule must be nonempty")
        ordered = tuple(sorted(self.selected_proposals, key=lambda proposal: proposal.rank))
        if self.selected_proposals != ordered:
            raise ActiveSelectionValidationError("selected_proposals must be sorted by rank")

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "selection_rule": self.selection_rule,
            "source_dataset_digest": str(self.source_dataset_digest),
            "source_proposal_set_digest": str(self.source_proposal_set_digest),
            "selected_proposals": [
                proposal.to_record()
                for proposal in self.selected_proposals
            ],
        }
        if self.relationship_fit_digest is not None:
            record["relationship_fit_digest"] = str(self.relationship_fit_digest)
        if self.surrogate_digest is not None:
            record["surrogate_digest"] = str(self.surrogate_digest)
        return record


def select_experiments(
    proposal_set: ExperimentProposalSet,
    *,
    dataset: MeasurementDataset,
    relationship_fit: RelationshipFitRecord | None = None,
    surrogate: ArchitectureSurrogateRecord | None = None,
    architectures: tuple[ArchitectureManifest, ...] = (),
    submission_packages: tuple[SubmissionPackageManifest, ...] = (),
    limit: int = 1,
    min_surrogate_observations: int = 1,
) -> ActiveSelectionResult:
    """Select the next declared experiment proposals from existing records."""

    if limit <= 0:
        raise ActiveSelectionValidationError("limit must be positive")
    if min_surrogate_observations < 0:
        raise ActiveSelectionValidationError("min_surrogate_observations must be nonnegative")

    try:
        proposal_set.validate_sources(
            dataset=dataset,
            relationship_fit=relationship_fit,
            architectures=architectures,
            submission_packages=submission_packages,
        )
        if relationship_fit is not None:
            relationship_fit.validate_sources(
                dataset=dataset,
                architecture=_relationship_architecture(
                    relationship_fit=relationship_fit,
                    architectures=architectures,
                ),
            )
        if surrogate is not None:
            surrogate.validate_sources(dataset=dataset)
    except ValueError as error:
        raise ActiveSelectionValidationError(str(error)) from error

    selected = proposal_set.proposals[:limit]
    if not selected:
        raise ActiveSelectionValidationError("no proposals are available for selection")

    return ActiveSelectionResult(
        selected_proposals=selected,
        selection_rule=_selection_rule(
            surrogate=surrogate,
            min_surrogate_observations=min_surrogate_observations,
        ),
        source_dataset_digest=dataset.digest,
        source_proposal_set_digest=proposal_set.digest,
        relationship_fit_digest=relationship_fit.digest if relationship_fit is not None else None,
        surrogate_digest=surrogate.digest if surrogate is not None else None,
    )


def _selection_rule(
    *,
    surrogate: ArchitectureSurrogateRecord | None,
    min_surrogate_observations: int,
) -> str:
    if surrogate is None:
        return "deterministic-bootstrap"
    if surrogate.training.status != "fit":
        return "deterministic-bootstrap"
    if surrogate.training.observation_count < min_surrogate_observations:
        return "deterministic-bootstrap"
    return "surrogate-informed-declared-rank"


def _relationship_architecture(
    *,
    relationship_fit: RelationshipFitRecord,
    architectures: tuple[ArchitectureManifest, ...],
) -> ArchitectureManifest | None:
    if relationship_fit.architecture_id is None:
        return None
    for architecture in architectures:
        if architecture.id == relationship_fit.architecture_id:
            return architecture
    return None
