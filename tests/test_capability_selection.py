from dataclasses import replace

import pytest

from leibniz.architecture_candidates import (
    default_architecture_candidate_space,
    generate_architecture_candidates,
)
from leibniz.candidate_observations import (
    ArchitectureCandidateObservation,
    ArchitectureMeasurementEvidence,
    project_architecture_candidate_observations,
)
from leibniz.capability_selection import (
    CapabilityKey,
    CapabilitySelectionError,
    capability_key_for_observation,
    select_capability_bootstrap_candidates,
)


def test_capability_key_uses_formal_candidate_metadata() -> None:
    observations = _observations(candidate_count=1)

    key = capability_key_for_observation(observations[0])

    assert key == CapabilityKey(
        family_kind="local-aggregation-readout",
        operator_kinds=("local-aggregation", "rank-collapse", "affine-readout"),
    )


def test_capability_bootstrap_degrades_to_stable_order_for_one_capability() -> None:
    observations = _observations(candidate_count=4)

    selections = select_capability_bootstrap_candidates(observations, budget=3)

    assert [selection.selector_name for selection in selections] == [
        "capability-bootstrap",
        "capability-bootstrap",
        "capability-bootstrap",
    ]
    assert [selection.observation.source_candidate_rank for selection in selections] == [1, 2, 3]
    assert [selection.selected_count_for_key for selection in selections] == [0, 1, 2]


def test_capability_bootstrap_prefers_less_measured_capability_keys() -> None:
    base = _observations(candidate_count=6)
    measured = (
        ArchitectureMeasurementEvidence(
            architecture_digest=base[0].architecture_digest,
            score=0.5,
            parameter_count=base[0].parameter_count,
        ),
    )
    observations = _with_alternating_capabilities(
        _observations(candidate_count=6, measured=measured)
    )

    selections = select_capability_bootstrap_candidates(observations, budget=3)

    assert [selection.capability_key.family_kind for selection in selections] == [
        "family-b",
        "family-a",
        "family-b",
    ]
    assert [selection.measured_count_for_key for selection in selections] == [0, 1, 0]
    assert all(not selection.observation.is_measured for selection in selections)


def test_capability_bootstrap_deduplicates_against_existing_selections() -> None:
    observations = _observations(candidate_count=4)

    selections = select_capability_bootstrap_candidates(
        observations,
        budget=2,
        excluded_candidate_ids=(observations[0].candidate_id, observations[1].candidate_id),
    )

    assert [selection.observation.source_candidate_rank for selection in selections] == [3, 4]


def test_capability_bootstrap_rejects_empty_inputs_and_invalid_budget() -> None:
    with pytest.raises(CapabilitySelectionError, match="budget must be positive"):
        select_capability_bootstrap_candidates(_observations(), budget=0)

    with pytest.raises(
        CapabilitySelectionError,
        match="observations must contain at least one item",
    ):
        select_capability_bootstrap_candidates((), budget=1)


def _observations(
    *,
    candidate_count: int = 6,
    measured: tuple[ArchitectureMeasurementEvidence, ...] = (),
) -> tuple[ArchitectureCandidateObservation, ...]:
    candidates = generate_architecture_candidates(
        default_architecture_candidate_space(),
        input_shape=(1, 8, 8),
        output_count=3,
    )[:candidate_count]
    return project_architecture_candidate_observations(candidates, measured=measured)


def _with_alternating_capabilities(
    observations: tuple[ArchitectureCandidateObservation, ...],
) -> tuple[ArchitectureCandidateObservation, ...]:
    alternated: list[ArchitectureCandidateObservation] = []
    for index, observation in enumerate(observations):
        if index % 2 == 0:
            alternated.append(
                replace(
                    observation,
                    family_kind="family-a",
                    operator_kinds=("operator-a",),
                )
            )
        else:
            alternated.append(
                replace(
                    observation,
                    family_kind="family-b",
                    operator_kinds=("operator-b",),
                )
            )
    return tuple(alternated)
