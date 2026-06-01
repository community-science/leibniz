import math
from collections.abc import Callable
from typing import cast

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import AxisAssignment, MaterializationPlan
from leibniz.measurements import MeasurementDataset
from leibniz.views import (
    CompetenceIntegralSource,
    CompetenceIntegralView,
    CompetenceIntegralViewDocument,
    MeasurementScoreView,
    MeasurementScoreViewDocument,
    MeasurementScoreViewValidationError,
)


def test_measurement_score_view_sorts_scores_and_ties_deterministically() -> None:
    dataset = _measurement_dataset()
    view = MeasurementScoreView.from_dataset(
        id=ProtocolIdentifier.parse("views.measurement-scores.boolean@0.1.0"),
        dataset=dataset,
    )

    assert [str(entry.measurement_id) for entry in view.entries] == [
        "core.boolean-evidence-a@0.1.0",
        "core.boolean-evidence-c@0.1.0",
        "core.boolean-evidence-b@0.1.0",
    ]
    assert [entry.negative_log_score for entry in view.entries] == [
        -math.log(0.8),
        -math.log(0.8),
        -math.log(0.2),
    ]
    assert view.to_record() == {
        "id": "views.measurement-scores.boolean@0.1.0",
        "source_dataset_digest": str(dataset.digest),
        "projection_rule": "measurement-score-ascending",
        "entries": [entry.to_record() for entry in view.entries],
    }
    assert view.digest == ContentDigest.from_value(view.to_record())


def test_measurement_score_view_validates_against_source_dataset() -> None:
    dataset = _measurement_dataset()
    view = MeasurementScoreView.from_dataset(
        id=ProtocolIdentifier.parse("views.measurement-scores.boolean@0.1.0"),
        dataset=dataset,
    )

    parsed = MeasurementScoreView.from_record(view.to_record(), dataset=dataset)

    assert parsed == view


def test_measurement_score_view_document_loads_bytes_with_digest() -> None:
    dataset = _measurement_dataset()
    view = MeasurementScoreView.from_dataset(
        id=ProtocolIdentifier.parse("views.measurement-scores.boolean@0.1.0"),
        dataset=dataset,
    )

    document = MeasurementScoreViewDocument.from_bytes(
        canonical_document_bytes(view.to_record()),
        dataset=dataset,
    )

    assert document.view == view
    assert document.digest == ContentDigest.from_value(view.to_record())


def test_measurement_score_view_rejects_source_digest_and_entry_mismatches() -> None:
    dataset = _measurement_dataset()
    view = MeasurementScoreView.from_dataset(
        id=ProtocolIdentifier.parse("views.measurement-scores.boolean@0.1.0"),
        dataset=dataset,
    )

    record = view.to_record()
    record["source_dataset_digest"] = str(ContentDigest.from_value({"other": True}))
    assert str(
        capture_view_error(lambda: MeasurementScoreView.from_record(record, dataset=dataset))
    ) == (
        "source_dataset_digest does not match dataset"
    )

    record = view.to_record()
    entries = _entry_records(record)
    first = dict(_entry_record(entries[0]))
    first["negative_log_score"] = 9.0
    entries[0] = first
    record["entries"] = entries
    assert str(
        capture_view_error(lambda: MeasurementScoreView.from_record(record, dataset=dataset))
    ) == (
        "entries must be sorted by score"
    )


def test_measurement_score_view_rejects_malformed_records() -> None:
    dataset = _measurement_dataset()
    view = MeasurementScoreView.from_dataset(
        id=ProtocolIdentifier.parse("views.measurement-scores.boolean@0.1.0"),
        dataset=dataset,
    )

    record = view.to_record()
    record["id"] = "core.boolean-view@0.1.0"
    assert str(
        capture_view_error(lambda: MeasurementScoreView.from_record(record, dataset=dataset))
    ) == (
        "id must be a valid measurement score view id"
    )

    record = view.to_record()
    entries = _entry_records(record)
    first = dict(_entry_record(entries[0]))
    del first["negative_log_score"]
    entries[0] = first
    record["entries"] = entries
    assert str(
        capture_view_error(lambda: MeasurementScoreView.from_record(record, dataset=dataset))
    ) == (
        "negative_log_score: missing required field"
    )

    record = view.to_record()
    entries = _entry_records(record)
    first = dict(_entry_record(entries[0]))
    first["local_path"] = "results/views/scores.json"
    entries[0] = first
    record["entries"] = entries
    assert str(
        capture_view_error(lambda: MeasurementScoreView.from_record(record, dataset=dataset))
    ) == (
        "local_path: unknown field"
    )


def test_measurement_score_view_preserves_infinite_scores() -> None:
    dataset = MeasurementDataset.from_record(
        {
            "measurements": [
                _measurement_record(
                    evidence_id="core.boolean-evidence-zero@0.1.0",
                    probability_measure_id="core.boolean-prediction-zero@0.1.0",
                    yes_probability=0.0,
                )
            ]
        }
    )

    view = MeasurementScoreView.from_dataset(
        id=ProtocolIdentifier.parse("views.measurement-scores.boolean@0.1.0"),
        dataset=dataset,
    )

    assert view.entries[0].negative_log_score == math.inf
    assert view.to_record()["entries"] == [
        {
            "measurement_id": "core.boolean-evidence-zero@0.1.0",
            "benchmark_id": "core.boolean-benchmark@0.1.0",
            "observation_id": "observation-1",
            "accepted_mass": 0.0,
            "negative_log_score": "infinity",
        }
    ]
    assert MeasurementScoreView.from_record(view.to_record(), dataset=dataset) == view


def test_measurement_score_view_document_rejects_invalid_bytes() -> None:
    assert str(
        capture_view_error(
            lambda: MeasurementScoreViewDocument.from_bytes(b"[]", dataset=_measurement_dataset())
        )
    ) == "measurement score view document must contain an object"


def test_competence_integral_view_uses_declared_complexity_spacing() -> None:
    dataset, sources = _competence_dataset(
        first_mass=1.0,
        second_mass=0.0,
        first_complexity=1,
        second_complexity=2,
    )
    close_view = CompetenceIntegralView.from_sources(
        id=ProtocolIdentifier.parse("views.competence-integrals.boolean@0.1.0"),
        dataset=dataset,
        sources=sources,
        complexity_axis="C",
        expected_complexities=(1.0, 2.0),
    )
    wide_view = CompetenceIntegralView.from_sources(
        id=ProtocolIdentifier.parse("views.competence-integrals.boolean@0.1.0"),
        dataset=dataset,
        sources=sources,
        complexity_axis="C",
        expected_complexities=(1.0, 2.0, 10.0),
    )

    assert close_view.entries[0].integral == 0.5
    assert wide_view.entries[0].integral == 0.5 / 9.0
    assert close_view.entries[0].missing_complexities == ()
    assert wide_view.entries[0].missing_complexities == (10.0,)


def test_competence_integral_view_validates_against_sources() -> None:
    dataset, sources = _competence_dataset(
        first_mass=1.0,
        second_mass=0.0,
        first_complexity=1,
        second_complexity=2,
    )
    view = CompetenceIntegralView.from_sources(
        id=ProtocolIdentifier.parse("views.competence-integrals.boolean@0.1.0"),
        dataset=dataset,
        sources=sources,
        complexity_axis="C",
        expected_complexities=(1.0, 2.0),
    )

    parsed = CompetenceIntegralView.from_record(
        view.to_record(),
        dataset=dataset,
        sources=sources,
    )
    document = CompetenceIntegralViewDocument.from_bytes(
        canonical_document_bytes(view.to_record()),
        dataset=dataset,
        sources=sources,
    )

    assert parsed == view
    assert document.view == view
    assert document.digest == ContentDigest.from_value(view.to_record())


def test_competence_integral_view_reports_missing_levels_without_mutating_measurements() -> None:
    dataset, sources = _competence_dataset(
        first_mass=1.0,
        second_mass=1.0,
        first_complexity=1,
        second_complexity=3,
    )
    view = CompetenceIntegralView.from_sources(
        id=ProtocolIdentifier.parse("views.competence-integrals.boolean@0.1.0"),
        dataset=dataset,
        sources=sources,
        complexity_axis="C",
        expected_complexities=(1.0, 2.0, 3.0),
    )

    assert view.entries[0].coverage == 2 / 3
    assert view.entries[0].missing_complexities == (2.0,)
    accepted_masses = [
        measurement.raw_scoring_evidence.accepted_mass
        for measurement in dataset.measurements
    ]
    assert accepted_masses == [
        1.0,
        1.0,
    ]


def test_competence_integral_view_rejects_malformed_records() -> None:
    dataset, sources = _competence_dataset(
        first_mass=1.0,
        second_mass=0.0,
        first_complexity=1,
        second_complexity=2,
    )
    view = CompetenceIntegralView.from_sources(
        id=ProtocolIdentifier.parse("views.competence-integrals.boolean@0.1.0"),
        dataset=dataset,
        sources=sources,
        complexity_axis="C",
        expected_complexities=(1.0, 2.0),
    )

    record = view.to_record()
    record["id"] = "core.boolean-view@0.1.0"
    assert str(
        capture_view_error(
            lambda: CompetenceIntegralView.from_record(
                record,
                dataset=dataset,
                sources=sources,
            )
        )
    ) == "id must be a valid competence integral view id"

    record = view.to_record()
    record["expected_complexities"] = [2.0, 1.0]
    assert str(
        capture_view_error(
            lambda: CompetenceIntegralView.from_record(
                record,
                dataset=dataset,
                sources=sources,
            )
        )
    ) == "expected_complexities must be sorted"


def test_competence_integral_view_rejects_sources_outside_dataset() -> None:
    dataset, sources = _competence_dataset(
        first_mass=1.0,
        second_mass=0.0,
        first_complexity=1,
        second_complexity=2,
    )
    other_dataset, _other_sources = _competence_dataset(
        first_mass=1.0,
        second_mass=0.0,
        first_complexity=1,
        second_complexity=2,
        prefix="other",
    )

    assert str(
        capture_view_error(
            lambda: CompetenceIntegralView.from_sources(
                id=ProtocolIdentifier.parse("views.competence-integrals.boolean@0.1.0"),
                dataset=other_dataset,
                sources=sources,
                complexity_axis="C",
                expected_complexities=(1.0, 2.0),
            )
        )
    ) == "source measurement core.boolean-evidence-a@0.1.0 is not in dataset"

    assert str(
        capture_view_error(
            lambda: CompetenceIntegralView.from_sources(
                id=ProtocolIdentifier.parse("views.competence-integrals.boolean@0.1.0"),
                dataset=dataset,
                sources=(sources[0], sources[0]),
                complexity_axis="C",
                expected_complexities=(1.0, 2.0),
            )
        )
    ) == "duplicate source measurement id: core.boolean-evidence-a@0.1.0"


def _measurement_dataset() -> MeasurementDataset:
    return MeasurementDataset.from_record(
        {
            "measurements": [
                _measurement_record(
                    evidence_id="core.boolean-evidence-b@0.1.0",
                    observation_id="observation-b",
                    probability_measure_id="core.boolean-prediction-b@0.1.0",
                    yes_probability=0.2,
                ),
                _measurement_record(
                    evidence_id="core.boolean-evidence-c@0.1.0",
                    observation_id="observation-c",
                    probability_measure_id="core.boolean-prediction-c@0.1.0",
                    yes_probability=0.8,
                ),
                _measurement_record(
                    evidence_id="core.boolean-evidence-a@0.1.0",
                    observation_id="observation-a",
                    probability_measure_id="core.boolean-prediction-a@0.1.0",
                    yes_probability=0.8,
                ),
            ]
        }
    )


def _competence_dataset(
    *,
    first_mass: float,
    second_mass: float,
    first_complexity: int,
    second_complexity: int,
    prefix: str = "core",
) -> tuple[MeasurementDataset, tuple[CompetenceIntegralSource, ...]]:
    benchmark_id = f"{prefix}.boolean-benchmark@0.1.0"
    first = _measurement_record(
        evidence_id=f"{prefix}.boolean-evidence-a@0.1.0",
        probability_measure_id=f"{prefix}.boolean-prediction-a@0.1.0",
        yes_probability=first_mass,
        observation_id="observation-a",
        benchmark_id=benchmark_id,
    )
    second = _measurement_record(
        evidence_id=f"{prefix}.boolean-evidence-b@0.1.0",
        probability_measure_id=f"{prefix}.boolean-prediction-b@0.1.0",
        yes_probability=second_mass,
        observation_id="observation-b",
        benchmark_id=benchmark_id,
    )
    first_plan = _materialization_plan("a", first_complexity, benchmark_id=benchmark_id)
    second_plan = _materialization_plan("b", second_complexity, benchmark_id=benchmark_id)
    first["evidence_artifacts"] = [_materialization_reference(first_plan)]
    second["evidence_artifacts"] = [_materialization_reference(second_plan)]
    dataset = MeasurementDataset.from_record({"measurements": [first, second]})
    measurements = {str(item.raw_scoring_evidence.id): item for item in dataset.measurements}
    return (
        dataset,
        (
            CompetenceIntegralSource(
                measurement=measurements[f"{prefix}.boolean-evidence-a@0.1.0"],
                materialization_plan=first_plan,
                complexity=first_complexity,
            ),
            CompetenceIntegralSource(
                measurement=measurements[f"{prefix}.boolean-evidence-b@0.1.0"],
                materialization_plan=second_plan,
                complexity=second_complexity,
            ),
        ),
    )


def _materialization_plan(
    label: str,
    complexity: int,
    *,
    benchmark_id: str,
) -> MaterializationPlan:
    return MaterializationPlan(
        id=ProtocolIdentifier.parse(f"core.boolean.materialization-plan.{label}@0.1.0"),
        benchmark_id=ProtocolIdentifier.parse(benchmark_id),
        materialization_declaration=ArtifactReference(
            kind="materialization-declaration",
            protocol_id=ProtocolIdentifier.parse("core.boolean.materialization@0.1.0"),
        ),
        resolution_assignment=AxisAssignment(values={"N": complexity}),
        seed=101,
    )


def _materialization_reference(plan: MaterializationPlan) -> dict[str, object]:
    return {
        "kind": "materialization-plan",
        "protocol_id": str(plan.id),
        "record_digest": str(plan.digest),
    }


def _measurement_record(
    *,
    evidence_id: str,
    probability_measure_id: str,
    yes_probability: float,
    observation_id: str = "observation-1",
    benchmark_id: str = "core.boolean-benchmark@0.1.0",
) -> dict[str, object]:
    no_probability = 1.0 - yes_probability
    return {
        "benchmark_id": benchmark_id,
        "id": evidence_id,
        "observation_id": observation_id,
        "outcome_space": {
            "id": "core.boolean-outcome@0.1.0",
            "outcomes": [{"id": "yes"}, {"id": "no"}],
        },
        "accepted_event": {
            "id": "core.boolean-accepted@0.1.0",
            "outcome_space_id": "core.boolean-outcome@0.1.0",
            "outcomes": ["yes"],
        },
        "probability_measure": {
            "id": probability_measure_id,
            "outcome_space_id": "core.boolean-outcome@0.1.0",
            "probabilities": [
                {"outcome_id": "yes", "probability": yes_probability},
                {"outcome_id": "no", "probability": no_probability},
            ],
        },
    }


def _entry_record(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _entry_records(record: dict[str, object]) -> list[dict[str, object]]:
    value = record["entries"]
    assert isinstance(value, list)
    items = cast(list[object], value)
    return [_entry_record(item) for item in items]


def capture_view_error(
    action: Callable[[], object],
) -> MeasurementScoreViewValidationError:
    try:
        action()
    except MeasurementScoreViewValidationError as error:
        return error
    raise AssertionError("expected MeasurementScoreViewValidationError")
