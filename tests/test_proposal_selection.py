from leibniz.acquisition import score_candidate_acquisition
from leibniz.architecture_candidates import (
    default_architecture_search_distribution,
    generate_architecture_candidates,
)
from leibniz.candidate_observations import (
    ArchitectureCandidateObservation,
    project_architecture_candidate_observations,
)
from leibniz.proposal_selection import select_candidate_proposals


def test_proposal_selection_covers_resource_axis_without_category_balancing() -> None:
    observations = _observations()
    selections = select_candidate_proposals(
        observations,
        budget=3,
        acquisition_scores=score_candidate_acquisition(observations, output_count=3),
    )

    assert [selection.selector_name for selection in selections] == [
        "resource-bootstrap",
        "resource-bootstrap",
        "resource-bootstrap",
    ]
    assert [selection.resource_stratum_index for selection in selections] == [0, 1, 2]


def test_proposal_selection_does_not_balance_operator_categories() -> None:
    observations = _observations()

    selections = select_candidate_proposals(
        observations,
        budget=3,
        acquisition_scores=score_candidate_acquisition(observations, output_count=3),
    )

    assert [selection.selector_name for selection in selections] == [
        "resource-bootstrap",
        "resource-bootstrap",
        "resource-bootstrap",
    ]
    assert [selection.resource_stratum_index for selection in selections] == [0, 1, 2]
    assert len({selection.observation.candidate_id for selection in selections}) == 3


def _observations() -> tuple[ArchitectureCandidateObservation, ...]:
    candidates = generate_architecture_candidates(
        default_architecture_search_distribution(),
        input_shape=(1, 8, 8),
        output_count=3,
    )[:6]
    return project_architecture_candidate_observations(candidates)
