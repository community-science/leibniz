import math

from leibniz.benchmark_evaluation import (
    CompetencePoint,
    ComputeCostPoint,
    ValidationCompetencePoint,
    integrated_compute_cost,
    sampled_competence_frontier_score,
    validation_competence_frontier_advances,
)


def test_integrated_compute_cost_uses_observed_intervals_without_competence_weighting() -> None:
    assert math.isclose(
        integrated_compute_cost(
            (
                ComputeCostPoint(
                    complexity=2.0,
                    complexity_minimum=1.5,
                    complexity_maximum=2.0,
                    compute_per_sample=10.0,
                    bit_length_per_op=8.0,
                ),
                ComputeCostPoint(
                    complexity=4.0,
                    complexity_minimum=3.0,
                    complexity_maximum=4.0,
                    compute_per_sample=20.0,
                    bit_length_per_op=4.0,
                ),
            )
        ),
        120.0,
    )


def test_sampled_competence_score_counts_exact_complexity_gaps_once() -> None:
    assert math.isclose(
        sampled_competence_frontier_score(
            (
                CompetencePoint(
                    complexity=1.0,
                    accepted_mass=1.0,
                    complexity_minimum=1.0,
                    complexity_maximum=1.0,
                ),
                CompetencePoint(
                    complexity=2.0,
                    accepted_mass=1.0,
                    complexity_minimum=2.0,
                    complexity_maximum=2.0,
                ),
                CompetencePoint(
                    complexity=3.0,
                    accepted_mass=1.0,
                    complexity_minimum=3.0,
                    complexity_maximum=3.0,
                ),
            ),
            chance_mass=0.1,
        ),
        3.0,
    )


def test_frontier_advancement_uses_current_rung_competence_not_integrated_score() -> None:
    assert validation_competence_frontier_advances(
        frontier_point=ValidationCompetencePoint(
            complexity=0.0,
            accepted_mass=0.95,
            sample_count=64,
            complexity_minimum=0.0,
            complexity_maximum=0.0,
        ),
        previous_frontier_points=(),
        chance_mass=0.1,
    )


def test_frontier_advancement_rejects_chance_competence() -> None:
    assert not validation_competence_frontier_advances(
        frontier_point=ValidationCompetencePoint(
            complexity=0.0,
            accepted_mass=0.1,
            sample_count=64,
            complexity_minimum=0.0,
            complexity_maximum=0.0,
        ),
        previous_frontier_points=(),
        chance_mass=0.1,
    )
