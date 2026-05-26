import math
from collections.abc import Callable
from typing import cast

from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset
from leibniz.views import (
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
    first["local_path"] = ".leibniz/views/scores.json"
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


def _measurement_record(
    *,
    evidence_id: str,
    probability_measure_id: str,
    yes_probability: float,
    observation_id: str = "observation-1",
) -> dict[str, object]:
    no_probability = 1.0 - yes_probability
    return {
        "benchmark_id": "core.boolean-benchmark@0.1.0",
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
