import pytest

from leibniz.architecture_candidates import (
    ArchitectureSearchDistribution,
    default_architecture_search_distribution,
    generate_architecture_candidates,
)
from leibniz.candidate_observations import (
    ArchitectureMeasurementEvidence,
    CandidateObservationProjectionError,
    project_architecture_candidate_observations,
)


def test_candidate_observations_mark_measured_candidates_by_digest() -> None:
    candidates = generate_architecture_candidates(
        default_architecture_search_distribution(),
        input_shape=(1, 4, 4),
        output_count=3,
    )
    measured = ArchitectureMeasurementEvidence(
        architecture_digest=candidates[1].architecture.digest,
        score=0.75,
        parameter_count=candidates[1].parameter_count,
    )

    observations = project_architecture_candidate_observations(
        candidates,
        measured=(measured,),
    )

    assert [observation.source_candidate_rank for observation in observations] == [1, 2, 3, 4]
    assert [observation.parameter_count for observation in observations] == [6, 15, 30, 51]
    assert [observation.is_measured for observation in observations] == [
        False,
        True,
        False,
        False,
    ]
    assert observations[1].measured_score == 0.75
    assert observations[0].measured_score is None
    assert observations[0].best_measured_score_at_or_below_cost == 0.0
    assert observations[2].best_measured_score_at_or_below_cost == 0.75
    operator_kinds = {
        operator
        for observation in observations
        for operator in observation.operator_kinds
    }
    assert operator_kinds == {
        "affine-readout",
        "local-aggregation",
        "rank-collapse",
    }


def test_candidate_observations_deduplicate_repeated_measurements_by_best_score() -> None:
    candidates = generate_architecture_candidates(
        default_architecture_search_distribution(),
        input_shape=(1, 4, 4),
        output_count=3,
    )
    measured = (
        ArchitectureMeasurementEvidence(
            architecture_digest=candidates[1].architecture.digest,
            score=0.25,
            parameter_count=candidates[1].parameter_count,
        ),
        ArchitectureMeasurementEvidence(
            architecture_digest=candidates[1].architecture.digest,
            score=0.5,
            parameter_count=candidates[1].parameter_count,
        ),
    )

    observations = project_architecture_candidate_observations(candidates, measured=measured)

    assert observations[1].measured_score == 0.5
    assert observations[2].best_measured_score_at_or_below_cost == 0.5


def test_candidate_observations_reject_conflicting_measurement_costs() -> None:
    candidates = generate_architecture_candidates(
        default_architecture_search_distribution(),
        input_shape=(1, 4, 4),
        output_count=3,
    )
    measured = (
        ArchitectureMeasurementEvidence(
            architecture_digest=candidates[1].architecture.digest,
            score=0.25,
            parameter_count=candidates[1].parameter_count,
        ),
        ArchitectureMeasurementEvidence(
            architecture_digest=candidates[1].architecture.digest,
            score=0.5,
            parameter_count=candidates[1].parameter_count + 1,
        ),
    )

    with pytest.raises(
        CandidateObservationProjectionError,
        match="conflicting parameter_count",
    ):
        project_architecture_candidate_observations(candidates, measured=measured)


def test_candidate_observations_project_sparse_unmeasured_candidates_without_fake_scores() -> None:
    distribution = ArchitectureSearchDistribution(
        local_support_dimension=2,
        local_support_size_minimum=1,
        local_support_size_maximum=2,
    )
    candidates = generate_architecture_candidates(
        distribution,
        input_shape=(1, 4, 4),
        output_count=3,
    )

    observations = project_architecture_candidate_observations(candidates)

    assert len(observations) == 2
    assert all(not observation.is_measured for observation in observations)
    assert all(observation.measured_score is None for observation in observations)
    assert all(observation.best_measured_score == 0.0 for observation in observations)
    assert all(
        observation.best_measured_score_at_or_below_cost == 0.0
        for observation in observations
    )
