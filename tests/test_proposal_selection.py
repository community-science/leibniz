from dataclasses import replace

from leibniz.architecture_candidates import (
    default_architecture_candidate_space,
    generate_architecture_candidates,
)
from leibniz.candidate_observations import (
    ArchitectureCandidateObservation,
    project_architecture_candidate_observations,
)
from leibniz.proposal_selection import select_candidate_proposals


def test_proposal_selection_degrades_to_resource_selector_for_one_capability() -> None:
    selections = select_candidate_proposals(_observations(), budget=3)

    assert [selection.selector_name for selection in selections] == [
        "resource-bootstrap",
        "resource-bootstrap",
        "resource-bootstrap",
    ]
    assert [selection.resource_stratum_index for selection in selections] == [0, 1, 2]
    assert all(
        selection.capability_key.family_kind == "local-aggregation-readout"
        for selection in selections
    )


def test_proposal_selection_composes_resource_and_capability_selectors() -> None:
    observations = _with_alternating_capabilities(_observations())

    selections = select_candidate_proposals(observations, budget=3)

    assert [selection.selector_name for selection in selections] == [
        "resource-bootstrap",
        "resource-bootstrap",
        "capability-bootstrap",
    ]
    assert selections[0].resource_stratum_index == 0
    assert selections[1].resource_stratum_index == 1
    assert selections[2].resource_stratum_index is None
    assert len({selection.observation.candidate_id for selection in selections}) == 3
    assert selections[2].capability_key.family_kind in {"family-a", "family-b"}


def _observations() -> tuple[ArchitectureCandidateObservation, ...]:
    candidates = generate_architecture_candidates(
        default_architecture_candidate_space(),
        input_shape=(1, 8, 8),
        output_count=3,
    )[:6]
    return project_architecture_candidate_observations(candidates)


def _with_alternating_capabilities(
    observations: tuple[ArchitectureCandidateObservation, ...],
) -> tuple[ArchitectureCandidateObservation, ...]:
    alternated: list[ArchitectureCandidateObservation] = []
    for index, observation in enumerate(observations):
        alternated.append(
            replace(
                observation,
                family_kind="family-a" if index % 2 == 0 else "family-b",
                operator_kinds=("operator-a",) if index % 2 == 0 else ("operator-b",),
            )
        )
    return tuple(alternated)
