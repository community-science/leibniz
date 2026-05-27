import math

import pytest

from leibniz.acquisition import (
    AcquisitionScoringError,
    score_candidate_acquisition,
)
from leibniz.architecture_candidates import (
    default_architecture_search_distribution,
    generate_architecture_candidates,
)
from leibniz.candidate_observations import (
    ArchitectureMeasurementEvidence,
    project_architecture_candidate_observations,
)


def test_acquisition_scores_candidate_observations_without_architecture_categories() -> None:
    base_observations = _observations()
    measured = (
        ArchitectureMeasurementEvidence(
            architecture_digest=base_observations[1].architecture_digest,
            score=0.6,
            parameter_count=base_observations[1].parameter_count,
        ),
    )
    observations = _observations(measured=measured)

    scores = score_candidate_acquisition(observations, output_count=3)

    assert [score.model_name for score in scores] == [
        "frontier-resource-gap",
        "frontier-resource-gap",
        "frontier-resource-gap",
        "frontier-resource-gap",
    ]
    assert math.isclose(scores[0].chance_score, 1 / 3)
    assert scores[0].best_observed_score == 0.6
    assert math.isclose(scores[1].estimated_score, 0.6)
    assert scores[1].resource_novelty == 0.0
    assert math.isclose(
        scores[2].expected_frontier_improvement,
        scores[2].estimated_score - 0.6,
    )
    assert scores[2].to_component_record()["acquisition_value"] == scores[2].acquisition_value


def test_acquisition_scores_sparse_evidence_from_chance_and_resource_novelty() -> None:
    scores = score_candidate_acquisition(_observations(), output_count=5)

    assert all(score.chance_score == 0.2 for score in scores)
    assert all(score.resource_novelty == 1.0 for score in scores)
    assert all(score.estimated_score == 0.25 for score in scores)
    assert all(score.expected_frontier_improvement == 0.25 for score in scores)


def test_acquisition_rejects_empty_observations_and_invalid_output_count() -> None:
    observations = _observations()

    with pytest.raises(AcquisitionScoringError, match="output_count must be positive"):
        score_candidate_acquisition(observations, output_count=0)

    with pytest.raises(
        AcquisitionScoringError,
        match="observations must contain at least one item",
    ):
        score_candidate_acquisition((), output_count=1)


def _observations(
    *,
    measured: tuple[ArchitectureMeasurementEvidence, ...] = (),
):
    candidates = generate_architecture_candidates(
        default_architecture_search_distribution(),
        input_shape=(1, 4, 4),
        output_count=3,
    )
    return project_architecture_candidate_observations(candidates, measured=measured)
