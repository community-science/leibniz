import math
from pathlib import Path

from leibniz.artifacts import ArtifactReference
from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestDocument
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import MaterializationPlanDocument
from leibniz.measurements import MeasurementDataset, MeasurementRecord
from leibniz.observation_formation import ObservationFormationDeclarationDocument
from leibniz.outcomes import (
    AcceptedEvent,
    FiniteProbabilityMeasure,
    ProbabilityMass,
    RawScoringEvidence,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_digits_fixture_root = _repository_root / "tests" / "fixtures" / "digits"


def test_digits_length_one_perfect_measurement_scores_full_accepted_mass() -> None:
    measurement = _measurement_for_sequence(
        sequence=(7,),
        plan_name="materialization_plan_l1.json",
        measure_kind="perfect",
    )

    measurement.validate_manifest(_digits_manifest(), scale=1)

    assert measurement.raw_scoring_evidence.observation_id == (
        "benchmarks.digits.observations.l1.digit-7@0.1.0"
    )
    assert measurement.raw_scoring_evidence.accepted_mass == 1.0
    assert measurement.raw_scoring_evidence.negative_log_score == 0.0
    assert [artifact.kind for artifact in measurement.evidence_artifacts] == [
        "observation-formation-declaration",
        "materialization-plan",
        "formed-observation",
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


def test_digits_length_three_measurements_use_resolved_sequence_outcome_space() -> None:
    perfect = _measurement_for_sequence(
        sequence=(1, 2, 3),
        plan_name="materialization_plan_l3.json",
        measure_kind="perfect",
    )
    uniform = _measurement_for_sequence(
        sequence=(1, 2, 3),
        plan_name="materialization_plan_l3.json",
        measure_kind="uniform",
    )
    dataset = MeasurementDataset(measurements=(uniform, perfect))

    dataset.validate_manifest(_digits_manifest(), scale=3)

    assert perfect.outcome_space.id == ProtocolIdentifier.parse(
        "benchmarks.digits.outcomes.l3@0.1.0"
    )
    assert len(perfect.outcome_space.outcomes) == 1000
    assert perfect.accepted_event.outcomes == frozenset({"digit-1-2-3"})
    assert perfect.raw_scoring_evidence.accepted_mass == 1.0
    assert math.isclose(uniform.raw_scoring_evidence.accepted_mass, 0.001)
    assert math.isclose(uniform.raw_scoring_evidence.negative_log_score, math.log(1000))
    assert [str(measurement.raw_scoring_evidence.id) for measurement in dataset.measurements] == [
        "benchmarks.digits.evidence.l3.digit-1-2-3.perfect@0.1.0",
        "benchmarks.digits.evidence.l3.digit-1-2-3.uniform@0.1.0",
    ]


def _measurement_for_sequence(
    *,
    sequence: tuple[int, ...],
    plan_name: str,
    measure_kind: str,
) -> MeasurementRecord:
    manifest = _digits_manifest()
    declaration = ObservationFormationDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "observation_formation.json").read_bytes()
    ).declaration
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / plan_name).read_bytes()
    ).plan
    scale = len(sequence)
    outcome_space = manifest.resolve_outcome_space(scale=scale)
    assert manifest.outcome_sequence is not None
    accepted_outcome = manifest.outcome_sequence.outcome_id(sequence)
    sequence_label = accepted_outcome.removeprefix("digit-")
    observation_id = ProtocolIdentifier.parse(
        f"benchmarks.digits.observations.l{scale}.digit-{sequence_label}@0.1.0"
    )
    observation = declaration.form_observation(
        id=observation_id,
        plan=plan,
        component_sequence=sequence,
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
            observation_id=str(observation.id),
            event=accepted_event,
            measure=probability_measure,
        ),
        evidence_artifacts=(
            observation.formation_declaration,
            observation.materialization_plan,
            ArtifactReference(
                kind="formed-observation",
                protocol_id=observation.id,
                record_digest=observation.digest,
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
    return BenchmarkManifestDocument.from_bytes(
        (_digits_benchmark_root / "manifest.json").read_bytes()
    ).manifest
