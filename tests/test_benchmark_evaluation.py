import math
from typing import cast

from leibniz.benchmark_evaluation import (
    CompetencePoint,
    ValidationCompetencePoint,
    finite_measurements_for_predictions,
    sampled_competence_curriculum_record,
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


def test_sampled_competence_score_counts_exact_volume_gaps_once() -> None:
    assert math.isclose(
        sampled_competence_frontier_integral(
            (
                CompetencePoint(
                    log2_volume=1.0,
                    accepted_mass=1.0,
                    log2_volume_minimum=1.0,
                    log2_volume_maximum=1.0,
                ),
                CompetencePoint(
                    log2_volume=2.0,
                    accepted_mass=1.0,
                    log2_volume_minimum=2.0,
                    log2_volume_maximum=2.0,
                ),
                CompetencePoint(
                    log2_volume=3.0,
                    accepted_mass=1.0,
                    log2_volume_minimum=3.0,
                    log2_volume_maximum=3.0,
                ),
            ),
            chance_mass=0.1,
        ).value,
        0.0,
    )


def test_sampled_competence_score_charges_representative_intervals() -> None:
    assert math.isclose(
        sampled_competence_frontier_integral(
            (
                CompetencePoint(
                    log2_volume=0.0,
                    accepted_mass=1.0,
                    log2_volume_minimum=0.0,
                    log2_volume_maximum=0.0,
                ),
                CompetencePoint(
                    log2_volume=1.0,
                    accepted_mass=0.1,
                    log2_volume_minimum=0.0,
                    log2_volume_maximum=1.0,
                ),
                CompetencePoint(
                    log2_volume=4.0,
                    accepted_mass=1.0,
                    log2_volume_minimum=4.0,
                    log2_volume_maximum=4.0,
                ),
            ),
            chance_mass=0.1,
        ).value,
        0.0,
    )


def test_sampled_competence_integral_exposes_human_readable_terms() -> None:
    integral = sampled_competence_frontier_integral(
        (
            CompetencePoint(
                log2_volume=2.0,
                accepted_mass=0.55,
                sample_count=8,
                log2_volume_minimum=1.0,
                log2_volume_maximum=2.0,
            ),
        ),
        chance_mass=0.1,
    )

    [measured] = [term.to_record() for term in integral.terms]
    measured_density = measured.pop("competence_density")
    measured_contribution = measured.pop("contribution")
    assert measured == {
        "kind": "measured-state-space-competence",
        "log2_volume_minimum": 1.0,
        "log2_volume_maximum": 2.0,
        "width_in_bits": 1.0,
        "representative_log2_volume": 2.0,
        "sample_count": 8,
    }
    assert isinstance(measured_density, int | float)
    assert isinstance(measured_contribution, int | float)
    assert math.isclose(measured_density, 0.5)
    assert math.isclose(measured_contribution, 0.5)
    assert math.isclose(integral.value, 0.5)


def test_validated_bit_competence_curriculum_allows_unbounded_bit_values() -> None:
    record = sampled_competence_curriculum_record(
        (
            {
                "kind": "sampled-state-space-volume-window",
                "sampling_rule": "generator-uniform-component-index-v1",
                "difficulty_assumption": "approximately-uniform-within-volume-window",
                "benchmark_id": "tests.benchmark@0.1.0",
                "volume_axis": None,
                "log2_volume": 1.0,
                "sample_count": 2,
                "mean_accepted_mass": 3.0,
                "competence_value_kind": "validated-bits",
            },
            {
                "kind": "sampled-state-space-volume-window",
                "sampling_rule": "generator-uniform-component-index-v1",
                "difficulty_assumption": "approximately-uniform-within-volume-window",
                "benchmark_id": "tests.benchmark@0.1.0",
                "volume_axis": None,
                "log2_volume": 2.0,
                "sample_count": 1,
                "mean_accepted_mass": 6.0,
                "competence_value_kind": "validated-bits",
            },
        )
    )

    assert record["competence_value_kind"] == "validated-bits"
    assert math.isclose(cast(float, record["mean_accepted_mass"]), 4.0)


def test_sampled_competence_record_regions_flow_into_integral_terms() -> None:
    point = CompetencePoint.from_sampled_record(
        {
            "log2_volume": 2.0,
            "log2_volume_minimum": 1.0,
            "log2_volume_maximum": 2.0,
            "mean_accepted_mass": 0.55,
            "sample_count": 8,
            "region": _minimal_state_space_region_record(),
        },
        field_prefix="point",
    )

    integral = sampled_competence_frontier_integral((point,), chance_mass=0.1)

    [term] = integral.terms
    assert term.region is not None
    assert term.region.id == "test.region"
    assert term.to_record()["region"] == _minimal_state_space_region_record()


def test_frontier_advancement_uses_current_rung_competence_not_integrated_score() -> None:
    assert validation_competence_frontier_advances(
        frontier_point=ValidationCompetencePoint(
            log2_volume=0.0,
            accepted_mass=0.95,
            sample_count=64,
            log2_volume_minimum=0.0,
            log2_volume_maximum=0.0,
        ),
        previous_frontier_points=(),
        chance_mass=0.1,
    )


def test_frontier_advancement_rejects_chance_competence() -> None:
    assert not validation_competence_frontier_advances(
        frontier_point=ValidationCompetencePoint(
            log2_volume=0.0,
            accepted_mass=0.1,
            sample_count=64,
            log2_volume_minimum=0.0,
            log2_volume_maximum=0.0,
        ),
        previous_frontier_points=(),
        chance_mass=0.1,
    )


def _minimal_state_space_region_record() -> dict[str, object]:
    return {
        "id": "test.region",
        "ambient": {
            "field_domain_kind": "lattice-2d",
            "field_domain": {"height": 2, "width": 2},
            "field_codomain_id": "unit-intensity",
            "distinguishability": {
                "kind": "exact",
                "certificate_id": "test-certificate",
            },
        },
        "components": [
            {
                "axis_regions": [
                    {
                        "axis": {
                            "id": "x",
                            "domain": {"kind": "integer-range", "lower": 0, "upper": 1},
                        },
                        "coordinate_region": [0, 1],
                        "count": 2,
                        "log2_count": 1.0,
                    },
                ],
                "measure_rule": "product-of-counts",
                "volume": 2,
                "log2_volume": 1.0,
                "stratum_id": "fixture",
                "stratum_target": {"label": "fixture"},
            },
        ],
        "union_rule": "disjoint-union",
        "volume": 2,
        "log2_volume": 1.0,
    }
