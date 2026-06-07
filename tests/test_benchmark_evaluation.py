import math

from leibniz.benchmark_evaluation import ComputeCostPoint, integrated_compute_cost


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
