import json
from collections.abc import Callable, Mapping
from typing import cast

from leibniz.answers import FiniteAnswerScoringBundle
from leibniz.benchmarks import BenchmarkManifest
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import (
    MeasurementDocument,
    MeasurementRecord,
    MeasurementRecordValidationError,
)


def test_measurement_record_parses_finite_answer_scoring_evidence() -> None:
    measurement = MeasurementRecord.from_record(_measurement_record())

    assert measurement == MeasurementRecord(
        benchmark_id=ProtocolIdentifier.parse("core.boolean-benchmark@0.1.0"),
        scoring_bundle=FiniteAnswerScoringBundle.from_record(_boolean_bundle_record()),
    )
    assert measurement.to_record() == _expanded_measurement_record()


def test_measurement_record_validates_against_matching_manifest() -> None:
    measurement = MeasurementRecord.from_record(_measurement_record())
    manifest = BenchmarkManifest.from_record(_benchmark_manifest_record())

    measurement.validate_manifest(manifest)


def test_measurement_record_rejects_mismatched_manifest() -> None:
    measurement = MeasurementRecord.from_record(_measurement_record())
    manifest_record = _benchmark_manifest_record()
    manifest_record["id"] = "core.other-benchmark@0.1.0"
    manifest_record["answer_space_id"] = "core.boolean-answer@0.1.0"
    manifest = BenchmarkManifest.from_record(manifest_record)

    error = capture_measurement_error(lambda: measurement.validate_manifest(manifest))

    assert str(error) == (
        "benchmark_id core.boolean-benchmark@0.1.0 does not match manifest "
        "core.other-benchmark@0.1.0"
    )


def test_measurement_record_rejects_bundle_outside_manifest_answer_space() -> None:
    measurement_record = _measurement_record()
    bundle_record = dict(cast(Mapping[str, object], measurement_record["scoring_bundle"]))
    answer_space = dict(cast(Mapping[str, object], bundle_record["answer_space"]))
    answer_space["id"] = "core.other-answer@0.1.0"
    accepted_event = dict(cast(Mapping[str, object], bundle_record["accepted_event"]))
    accepted_event["answer_space_id"] = "core.other-answer@0.1.0"
    probability_measure = dict(cast(Mapping[str, object], bundle_record["probability_measure"]))
    probability_measure["answer_space_id"] = "core.other-answer@0.1.0"
    bundle_record["answer_space"] = answer_space
    bundle_record["accepted_event"] = accepted_event
    bundle_record["probability_measure"] = probability_measure
    measurement_record["scoring_bundle"] = bundle_record
    measurement = MeasurementRecord.from_record(measurement_record)
    manifest = BenchmarkManifest.from_record(_benchmark_manifest_record())

    error = capture_measurement_error(lambda: measurement.validate_manifest(manifest))

    assert str(error) == (
        "bundle answer_space_id core.other-answer@0.1.0 does not match "
        "core.boolean-answer@0.1.0"
    )


def test_measurement_record_digest_is_stable_for_minimal_and_expanded_records() -> None:
    measurement = MeasurementRecord.from_record(_measurement_record())
    expanded = MeasurementRecord.from_record(_expanded_measurement_record())

    assert measurement == expanded
    assert measurement.digest == ContentDigest.from_value(_expanded_measurement_record())
    assert measurement.digest == expanded.digest


def test_measurement_record_rejects_malformed_records_and_state_paths() -> None:
    assert str(
        capture_measurement_error(
            lambda: MeasurementRecord.from_record(
                {"scoring_bundle": _boolean_bundle_record()}
            )
        )
    ) == "benchmark_id: missing required field"
    assert str(
        capture_measurement_error(
            lambda: MeasurementRecord.from_record(
                {
                    "benchmark_id": "core.boolean-benchmark@1.0.0",
                    "scoring_bundle": _boolean_bundle_record(),
                }
            )
        )
    ) == (
        "identifier must use a pre-1.0.0 version before release policy exists: "
        "core.boolean-benchmark@1.0.0"
    )
    assert str(
        capture_measurement_error(
            lambda: MeasurementRecord.from_record(
                {
                    "benchmark_id": "core.boolean-benchmark@0.1.0",
                    "scoring_bundle": _boolean_bundle_record(),
                    "local_path": ".leibniz/measurement_records/boolean.json",
                }
            )
        )
    ) == "local_path: unknown field"


def test_measurement_document_loads_json_bytes_with_digest() -> None:
    document = MeasurementDocument.from_json_bytes(_json_bytes(_measurement_record()))

    assert document.measurement.to_record() == _expanded_measurement_record()
    assert document.digest == ContentDigest.from_value(_expanded_measurement_record())


def test_measurement_document_digest_is_stable_for_minimal_and_expanded_records() -> None:
    minimal = MeasurementDocument.from_json_bytes(_json_bytes(_measurement_record()))
    expanded = MeasurementDocument.from_json_bytes(_json_bytes(_expanded_measurement_record()))

    assert minimal.measurement.to_record() == expanded.measurement.to_record()
    assert minimal.digest == expanded.digest


def test_measurement_document_rejects_invalid_json_input() -> None:
    assert (
        str(capture_measurement_error(lambda: MeasurementDocument.from_json_bytes(b"[]")))
        == "measurement JSON file must contain an object"
    )
    assert str(
        capture_measurement_error(
            lambda: MeasurementDocument.from_json_bytes(b'{"benchmark_id": false}')
        )
    ) == "benchmark_id: expected identifier string; scoring_bundle: missing required field"


def _measurement_record() -> dict[str, object]:
    return {
        "benchmark_id": "core.boolean-benchmark@0.1.0",
        "scoring_bundle": _minimal_boolean_bundle_record(),
    }


def _expanded_measurement_record() -> dict[str, object]:
    return {
        "benchmark_id": "core.boolean-benchmark@0.1.0",
        "scoring_bundle": _expanded_boolean_bundle_record(),
    }


def _benchmark_manifest_record() -> dict[str, object]:
    return {
        "id": "core.boolean-benchmark@0.1.0",
        "answer_space_id": "core.boolean-answer@0.1.0",
    }


def _minimal_boolean_bundle_record() -> dict[str, object]:
    return {
        "id": "core.boolean-evidence@0.1.0",
        "observation_id": "observation-1",
        "answer_space": {
            "id": "core.boolean-answer@0.1.0",
            "elements": [{"id": "yes"}, {"id": "no"}],
        },
        "accepted_event": {
            "id": "core.boolean-accepted@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "elements": ["yes"],
        },
        "probability_measure": {
            "id": "core.boolean-prediction@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "probabilities": [
                {"element_id": "no", "probability": 0.75},
                {"element_id": "yes", "probability": 0.25},
            ],
        },
    }


def _boolean_bundle_record() -> dict[str, object]:
    record = _minimal_boolean_bundle_record()
    record["raw_scoring_evidence"] = {
        "id": "core.boolean-evidence@0.1.0",
        "observation_id": "observation-1",
        "answer_space_id": "core.boolean-answer@0.1.0",
        "accepted_event_id": "core.boolean-accepted@0.1.0",
        "probability_measure_id": "core.boolean-prediction@0.1.0",
        "accepted_mass": 0.25,
        "negative_log_score": 1.3862943611198906,
    }
    return record


def _expanded_boolean_bundle_record() -> dict[str, object]:
    record = _boolean_bundle_record()
    del record["id"]
    del record["observation_id"]
    return record


def _json_bytes(record: Mapping[str, object]) -> bytes:
    return json.dumps(record).encode("utf-8")


def capture_measurement_error(
    action: Callable[[], object],
) -> MeasurementRecordValidationError:
    try:
        action()
    except MeasurementRecordValidationError as error:
        return error
    raise AssertionError("expected MeasurementRecordValidationError")
