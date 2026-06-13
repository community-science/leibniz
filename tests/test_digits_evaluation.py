import math
from pathlib import Path

from benchmark_typing import load_digits_benchmark

from leibniz.artifacts import ArtifactReference
from leibniz.benchmarks import BenchmarkManifest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import MaterializationPlanDocument
from leibniz.measurements import MeasurementRecord
from leibniz.observation_generation import StateSpaceVolumeRequest
from leibniz.outcomes import (
    AcceptedEvent,
    FiniteProbabilityMeasure,
    ProbabilityMass,
    RawScoringEvidence,
)
from leibniz.state_space import ContinuousAxisRegion, RealIntervalDomain

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_digits_fixture_root = _repository_root / "tests" / "fixtures" / "digits"


def test_digits_length_one_perfect_measurement_scores_full_accepted_mass() -> None:
    measurement = _measurement_for_sequence(
        sequence=(7,),
        plan_name="materialization_plan_l1.json",
        measure_kind="perfect",
    )

    measurement.validate_manifest(_digits_manifest())

    assert measurement.raw_scoring_evidence.observation_id == (
        "benchmarks.digits.observations.l1.digit-7@0.1.0"
    )
    assert measurement.raw_scoring_evidence.accepted_mass == 1.0
    assert measurement.raw_scoring_evidence.negative_log_score == 0.0
    assert [artifact.kind for artifact in measurement.evidence_artifacts] == [
        "observation-formation-declaration",
        "materialization-plan",
    ]


def test_digits_length_one_uniform_and_wrong_measurements_score_expected_mass() -> None:
    uniform = _measurement_for_sequence(
        sequence=(7,),
        plan_name="materialization_plan_l1.json",
        measure_kind="uniform",
    )
    wrong = _measurement_for_sequence(
        sequence=(7,),
        plan_name="materialization_plan_l1.json",
        measure_kind="wrong",
    )

    assert math.isclose(uniform.raw_scoring_evidence.accepted_mass, 0.1)
    assert math.isclose(uniform.raw_scoring_evidence.negative_log_score, math.log(10))
    assert wrong.raw_scoring_evidence.accepted_mass == 0.0
    assert wrong.raw_scoring_evidence.negative_log_score == math.inf


def test_digits_manifest_declares_single_digit_outcomes() -> None:
    manifest = _digits_manifest()

    assert manifest.outcome_space is not None
    assert [outcome.id for outcome in manifest.outcome_space.outcomes] == [
        f"digit-{index}" for index in range(10)
    ]


def test_digits_realized_regions_claim_continuous_transform_cells() -> None:
    benchmark = load_digits_benchmark(_digits_benchmark_root)

    batch = benchmark.generator(
        seed=407,
        shape=4,
        volume_request=StateSpaceVolumeRequest(2.0, 3.0),
    )

    assert batch.region is not None
    assert batch.region.measure_estimate is not None
    assert batch.region.measure_estimate.kind == "estimated"
    assert batch.region.components
    for component in batch.region.components:
        assert component.measure_estimate is not None
        assert component.measure_estimate.kind == "estimated"
        assert {axis_region.axis_id for axis_region in component.axis_regions} == {
            "x_translation",
            "y_translation",
            "scale",
        }
        for axis_region in component.axis_regions:
            assert isinstance(axis_region, ContinuousAxisRegion)
            assert isinstance(axis_region.axis.domain, RealIntervalDomain)
    for sample in batch.samples:
        assert sample.axis_coordinates is not None
        assert sample.region_component_index is not None
        assert set(sample.axis_coordinates) == {
            "x_translation",
            "y_translation",
            "scale",
        }
        assert "transform-ordinal" not in sample.axis_coordinates
        assert batch.region.contains(sample.region_component_index, sample.axis_coordinates)


def _measurement_for_sequence(
    *,
    sequence: tuple[int, ...],
    plan_name: str,
    measure_kind: str,
) -> MeasurementRecord:
    manifest = _digits_manifest()
    declaration = load_digits_benchmark(_digits_benchmark_root).formation
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / plan_name).read_bytes()
    ).plan
    scale = 1
    outcome_space = manifest.resolve_outcome_space()
    assert manifest.outcome_space is not None
    assert len(sequence) == 1
    accepted_outcome = f"digit-{sequence[0]}"
    sequence_label = accepted_outcome.removeprefix("digit-")
    observation_id = ProtocolIdentifier.parse(
        f"benchmarks.digits.observations.l{scale}.digit-{sequence_label}@0.1.0"
    )
    accepted_event = AcceptedEvent.from_record(
        {
            "id": f"benchmarks.digits.events.l{scale}.digit-{sequence_label}@0.1.0",
            "outcome_space_id": str(outcome_space.id),
            "outcomes": [accepted_outcome],
        },
        outcome_space=outcome_space,
    )
    probability_measure = FiniteProbabilityMeasure(
        id=ProtocolIdentifier.parse(
            f"benchmarks.digits.measures.l{scale}.digit-{sequence_label}.{measure_kind}@0.1.0"
        ),
        outcome_space_id=outcome_space.id,
        probabilities=_probabilities(
            outcome_ids=tuple(outcome.id for outcome in outcome_space.outcomes),
            accepted_outcome=accepted_outcome,
            measure_kind=measure_kind,
        ),
    )
    return MeasurementRecord(
        benchmark_id=manifest.id,
        outcome_space=outcome_space,
        accepted_event=accepted_event,
        probability_measure=probability_measure,
        raw_scoring_evidence=RawScoringEvidence.from_event_and_measure(
            id=ProtocolIdentifier.parse(
                f"benchmarks.digits.evidence.l{scale}.digit-{sequence_label}.{measure_kind}@0.1.0"
            ),
            observation_id=str(observation_id),
            event=accepted_event,
            measure=probability_measure,
        ),
        evidence_artifacts=(
            ArtifactReference(
                kind="observation-formation-declaration",
                protocol_id=declaration.id,
                record_digest=declaration.digest,
            ),
            ArtifactReference(
                kind="materialization-plan",
                protocol_id=plan.id,
                record_digest=plan.digest,
            ),
        ),
    )


def _probabilities(
    *,
    outcome_ids: tuple[str, ...],
    accepted_outcome: str,
    measure_kind: str,
) -> tuple[ProbabilityMass, ...]:
    if measure_kind == "perfect":
        return (ProbabilityMass(accepted_outcome, 1.0),)
    if measure_kind == "wrong":
        wrong_outcome = next(
            outcome_id for outcome_id in outcome_ids if outcome_id != accepted_outcome
        )
        return (ProbabilityMass(wrong_outcome, 1.0),)
    if measure_kind == "uniform":
        probability = 1.0 / len(outcome_ids)
        return tuple(ProbabilityMass(outcome_id, probability) for outcome_id in outcome_ids)
    raise AssertionError(f"unknown measure kind: {measure_kind}")


def _digits_manifest() -> BenchmarkManifest:
    return load_digits_benchmark(_digits_benchmark_root).manifest
