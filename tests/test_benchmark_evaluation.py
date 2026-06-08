import math

from leibniz.benchmark_evaluation import (
    CompetencePoint,
    ValidationCompetencePoint,
    finite_measurements_for_predictions,
    sampled_competence_frontier_integral,
    validation_competence_frontier_advances,
)
from leibniz.identifiers import ProtocolIdentifier
from leibniz.observation_generation import GeneratedSample, GeneratedSampleSet
from leibniz.outcomes import Outcome, OutcomeSpace


def test_prediction_measurements_accept_declared_target_distribution() -> None:
    outcome_space = OutcomeSpace(
        id=ProtocolIdentifier.parse("tests.outcomes@0.1.0"),
        outcomes=(
            Outcome(id="first"),
            Outcome(id="second"),
            Outcome(id="third"),
        ),
    )
    batch = GeneratedSampleSet(
        benchmark_id=ProtocolIdentifier.parse("tests.benchmark@0.1.0"),
        generator_id=ProtocolIdentifier.parse("tests.generator@0.1.0"),
        generator_version="0",
        seed=1,
        shape=(1,),
        samples=(
            GeneratedSample(
                index=0,
                outcome_id="first",
                complexity=0.0,
                target_distribution={"first": 0.5, "second": 0.5},
            ),
        ),
    )

    measurements = finite_measurements_for_predictions(
        batch=batch,
        outcome_space=outcome_space,
        probabilities=((0.25, 0.75, 0.0),),
        run_slug="target-distribution-test",
    )

    event = measurements[0].accepted_event
    evidence = measurements[0].raw_scoring_evidence
    assert event.outcomes == frozenset({"first", "second"})
    assert math.isclose(evidence.accepted_mass, 1.0)


def test_sampled_competence_score_counts_exact_complexity_gaps_once() -> None:
    assert math.isclose(
        sampled_competence_frontier_integral(
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
        ).value,
        3.0,
    )


def test_sampled_competence_score_charges_representative_intervals() -> None:
    assert math.isclose(
        sampled_competence_frontier_integral(
            (
                CompetencePoint(
                    complexity=0.0,
                    accepted_mass=1.0,
                    complexity_minimum=0.0,
                    complexity_maximum=0.0,
                ),
                CompetencePoint(
                    complexity=1.0,
                    accepted_mass=0.1,
                    complexity_minimum=0.0,
                    complexity_maximum=1.0,
                ),
                CompetencePoint(
                    complexity=4.0,
                    accepted_mass=1.0,
                    complexity_minimum=4.0,
                    complexity_maximum=4.0,
                ),
            ),
            chance_mass=0.1,
        ).value,
        3.0,
    )


def test_sampled_competence_integral_exposes_human_readable_terms() -> None:
    integral = sampled_competence_frontier_integral(
        (
            CompetencePoint(
                complexity=2.0,
                accepted_mass=0.55,
                sample_count=8,
                complexity_minimum=1.0,
                complexity_maximum=2.0,
            ),
        ),
        chance_mass=0.1,
    )

    gap, measured = [term.to_record() for term in integral.terms]
    assert gap == {
        "kind": "unrepresentable-gap",
        "complexity_minimum": 0.0,
        "complexity_maximum": 1.0,
        "complexity_width": 1.0,
        "density": 1.0,
        "contribution": 1.0,
    }
    measured_density = measured.pop("density")
    measured_contribution = measured.pop("contribution")
    assert measured == {
        "kind": "measured-competence",
        "complexity_minimum": 1.0,
        "complexity_maximum": 2.0,
        "complexity_width": 1.0,
        "representative_complexity": 2.0,
        "sample_count": 8,
    }
    assert isinstance(measured_density, int | float)
    assert isinstance(measured_contribution, int | float)
    assert math.isclose(measured_density, 0.5)
    assert math.isclose(measured_contribution, 0.5)
    assert math.isclose(integral.value, 1.5)


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
