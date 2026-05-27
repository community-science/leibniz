import pytest

from leibniz.architecture_candidates import (
    default_architecture_search_distribution,
    generate_architecture_candidates,
)
from leibniz.candidate_observations import (
    ArchitectureMeasurementEvidence,
    project_architecture_candidate_observations,
)
from leibniz.resource_selection import (
    ResourceBootstrapSelectionError,
    select_resource_bootstrap_candidates,
)


def test_resource_bootstrap_selects_unmeasured_cost_axis_coverage() -> None:
    observations = _observations()

    selections = select_resource_bootstrap_candidates(observations, budget=3)

    assert [selection.selector_name for selection in selections] == [
        "resource-bootstrap",
        "resource-bootstrap",
        "resource-bootstrap",
    ]
    assert [selection.resource_stratum_index for selection in selections] == [0, 1, 2]
    assert [selection.resource_stratum_count for selection in selections] == [3, 3, 3]
    assert len({selection.observation.candidate_id for selection in selections}) == 3
    assert all(not selection.observation.is_measured for selection in selections)
    assert [
        selection.observation.source_candidate_rank
        for selection in selections
    ] == [2, 3, 6]


def test_resource_bootstrap_prefers_unmeasured_strata_with_less_evidence() -> None:
    base_observations = _observations()
    measured = (
        ArchitectureMeasurementEvidence(
            architecture_digest=base_observations[1].architecture_digest,
            score=0.4,
            parameter_count=base_observations[1].parameter_count,
        ),
    )
    observations = _observations(measured=measured)

    selections = select_resource_bootstrap_candidates(observations, budget=3)

    assert [selection.resource_stratum_index for selection in selections] == [1, 2, 0]
    assert [selection.measured_count_in_stratum for selection in selections] == [0, 0, 1]
    assert all(
        selection.observation.architecture_digest != measured[0].architecture_digest
        for selection in selections
    )


def test_resource_bootstrap_returns_empty_when_every_candidate_is_measured() -> None:
    base_observations = _observations(candidate_count=2)
    measured = tuple(
        ArchitectureMeasurementEvidence(
            architecture_digest=observation.architecture_digest,
            score=0.5,
            parameter_count=observation.parameter_count,
        )
        for observation in base_observations
    )
    observations = _observations(candidate_count=2, measured=measured)

    assert select_resource_bootstrap_candidates(observations, budget=2) == ()


def test_resource_bootstrap_rejects_empty_inputs_and_invalid_budget() -> None:
    with pytest.raises(ResourceBootstrapSelectionError, match="budget must be positive"):
        select_resource_bootstrap_candidates(_observations(), budget=0)

    with pytest.raises(
        ResourceBootstrapSelectionError,
        match="observations must contain at least one item",
    ):
        select_resource_bootstrap_candidates((), budget=1)


def _observations(
    *,
    candidate_count: int = 8,
    measured: tuple[ArchitectureMeasurementEvidence, ...] = (),
):
    candidates = generate_architecture_candidates(
        default_architecture_search_distribution(),
        input_shape=(1, 8, 8),
        output_count=3,
    )[:candidate_count]
    return project_architecture_candidate_observations(candidates, measured=measured)
