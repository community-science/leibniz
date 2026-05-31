from collections.abc import Callable, Mapping
from pathlib import Path

from leibniz.artifacts import ArtifactReference
from leibniz.benchmarks import BenchmarkManifest
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import (
    MeasurementDataset,
    MeasurementDatasetDocument,
    MeasurementDocument,
    MeasurementRecord,
    MeasurementRecordValidationError,
)
from leibniz.outcomes import (
    AcceptedEvent,
    FiniteProbabilityMeasure,
    OutcomeSpace,
    ProbabilityMass,
    RawScoringEvidence,
)

_fixtures_root = Path(__file__).parent / "fixtures"


def test_measurement_record_parses_finite_outcome_scoring_evidence() -> None:
    measurement = MeasurementRecord.from_record(_measurement_record())

    assert measurement == MeasurementRecord(
        benchmark_id=ProtocolIdentifier.parse("core.boolean-benchmark@0.1.0"),
        outcome_space=OutcomeSpace.from_record(_outcome_space_record()),
        accepted_event=AcceptedEvent.from_record(
            _accepted_event_record(),
            outcome_space=OutcomeSpace.from_record(_outcome_space_record()),
        ),
        probability_measure=FiniteProbabilityMeasure(
            id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
            outcome_space_id=ProtocolIdentifier.parse("core.boolean-outcome@0.1.0"),
            probabilities=(ProbabilityMass("no", 0.75), ProbabilityMass("yes", 0.25)),
        ),
        raw_scoring_evidence=RawScoringEvidence.from_record(_raw_scoring_evidence_record()),
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
    manifest = BenchmarkManifest.from_record(manifest_record)

    error = capture_measurement_error(lambda: measurement.validate_manifest(manifest))

    assert str(error) == (
        "benchmark_id core.boolean-benchmark@0.1.0 does not match manifest "
        "core.other-benchmark@0.1.0"
    )


def test_measurement_record_rejects_result_outside_manifest_outcome_space() -> None:
    measurement_record = _measurement_record()
    measurement_record["outcome_space"] = _other_outcome_space_record()
    measurement_record["accepted_event"] = _other_accepted_event_record()
    measurement_record["probability_measure"] = _other_probability_measure_record()
    measurement_record["raw_scoring_evidence"] = _other_raw_scoring_evidence_record()
    measurement = MeasurementRecord.from_record(measurement_record)
    manifest = BenchmarkManifest.from_record(_benchmark_manifest_record())

    error = capture_measurement_error(lambda: measurement.validate_manifest(manifest))

    assert str(error) == (
        "measurement outcome_space does not match manifest outcome_space "
        "core.boolean-outcome@0.1.0"
    )


def test_measurement_record_rejects_same_outcome_space_id_with_different_outcomes() -> None:
    measurement = MeasurementRecord.from_record(_measurement_record())
    manifest_record = _benchmark_manifest_record()
    manifest_record["outcome_space"] = {
        "id": "core.boolean-outcome@0.1.0",
        "outcomes": [{"id": "yes"}, {"id": "maybe"}],
    }
    manifest = BenchmarkManifest.from_record(manifest_record)

    error = capture_measurement_error(lambda: measurement.validate_manifest(manifest))

    assert str(error) == (
        "measurement outcome_space does not match manifest outcome_space "
        "core.boolean-outcome@0.1.0"
    )


def test_measurement_record_digest_is_stable_for_minimal_and_expanded_records() -> None:
    measurement = MeasurementRecord.from_record(_measurement_record())
    expanded = MeasurementRecord.from_record(_expanded_measurement_record())

    assert measurement == expanded
    assert measurement.digest == ContentDigest.from_value(_expanded_measurement_record())
    assert measurement.digest == expanded.digest


def test_measurement_record_preserves_evidence_artifact_references() -> None:
    record = _measurement_record()
    record["evidence_artifacts"] = [
        {
            "kind": "materialization-plan",
            "protocol_id": "core.boolean-materialization.plan-one@0.1.0",
        }
    ]

    measurement = MeasurementRecord.from_record(record)

    assert measurement.evidence_artifacts == (
        ArtifactReference(
            kind="materialization-plan",
            protocol_id=ProtocolIdentifier.parse(
                "core.boolean-materialization.plan-one@0.1.0"
            ),
        ),
    )
    assert measurement.to_record()["evidence_artifacts"] == record["evidence_artifacts"]


def test_measurement_record_rejects_malformed_records_and_state_paths() -> None:
    assert str(
        capture_measurement_error(
            lambda: MeasurementRecord.from_record(
                {
                    "id": "core.boolean-evidence@0.1.0",
                    "observation_id": "observation-1",
                    "outcome_space": _outcome_space_record(),
                    "accepted_event": _accepted_event_record(),
                    "probability_measure": _probability_measure_record(),
                }
            )
        )
    ) == "benchmark_id: missing required field"
    assert str(
        capture_measurement_error(
            lambda: MeasurementRecord.from_record(
                {
                    "benchmark_id": "core.boolean-benchmark@1.0.0",
                    **_minimal_scoring_record(),
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
                {"benchmark_id": "core.boolean-benchmark@0.1.0"}
            )
        )
    ) == "measurement scoring fields are missing"
    assert str(
        capture_measurement_error(
            lambda: MeasurementRecord.from_record(
                {
                    "benchmark_id": "core.boolean-benchmark@0.1.0",
                    **_minimal_scoring_record(),
                    "local_path": "results/measurement_records/boolean.json",
                }
            )
        )
    ) == "local_path: unknown field"

    record = _measurement_record()
    artifact = {
        "kind": "materialization-plan",
        "protocol_id": "core.boolean-materialization.plan-one@0.1.0",
    }
    record["evidence_artifacts"] = [artifact, artifact]

    assert str(capture_measurement_error(lambda: MeasurementRecord.from_record(record))) == (
        "duplicate evidence artifact"
    )


def test_measurement_record_rejects_conflicting_raw_scoring_evidence() -> None:
    record = _expanded_measurement_record()
    raw_scoring_evidence = _raw_scoring_evidence_record()
    raw_scoring_evidence["accepted_event_id"] = "core.other-accepted@0.1.0"
    record["raw_scoring_evidence"] = raw_scoring_evidence

    assert str(capture_measurement_error(lambda: MeasurementRecord.from_record(record))) == (
        "raw_scoring_evidence must equal derived scoring evidence"
    )


def test_measurement_record_rejects_missing_minimal_evidence_identity() -> None:
    record = _measurement_record()
    del record["id"]

    assert str(capture_measurement_error(lambda: MeasurementRecord.from_record(record))) == (
        "id: missing required field"
    )

    record = _measurement_record()
    del record["observation_id"]

    assert str(capture_measurement_error(lambda: MeasurementRecord.from_record(record))) == (
        "observation_id: missing required field"
    )


def test_measurement_document_loads_bytes_with_digest() -> None:
    document = MeasurementDocument.from_bytes(_json_bytes(_measurement_record()))

    assert document.measurement.to_record() == _expanded_measurement_record()
    assert document.digest == ContentDigest.from_value(_expanded_measurement_record())


def test_measurement_document_digest_is_stable_for_minimal_and_expanded_records() -> None:
    minimal = MeasurementDocument.from_bytes(_json_bytes(_measurement_record()))
    expanded = MeasurementDocument.from_bytes(_json_bytes(_expanded_measurement_record()))

    assert minimal.measurement.to_record() == expanded.measurement.to_record()
    assert minimal.digest == expanded.digest


def test_measurement_document_rejects_invalid_document_bytes() -> None:
    assert (
        str(capture_measurement_error(lambda: MeasurementDocument.from_bytes(b"[]")))
        == "measurement document must contain an object"
    )
    assert str(
        capture_measurement_error(
            lambda: MeasurementDocument.from_bytes(b'{"benchmark_id": false}')
        )
    ) == "benchmark_id: expected identifier string"


def test_measurement_dataset_loads_records_with_stable_canonical_order() -> None:
    first_record = _measurement_record()
    second_record = _measurement_record(
        evidence_id="core.second-evidence@0.1.0",
        observation_id="observation-2",
    )

    dataset = MeasurementDataset.from_record(
        {"measurements": [second_record, first_record]}
    )

    assert [str(measurement.raw_scoring_evidence.id) for measurement in dataset.measurements] == [
        "core.boolean-evidence@0.1.0",
        "core.second-evidence@0.1.0",
    ]


def test_measurement_dataset_document_digest_is_independent_of_authoring_order() -> None:
    first_record = _measurement_record()
    second_record = _measurement_record(
        evidence_id="core.second-evidence@0.1.0",
        observation_id="observation-2",
    )

    forward = MeasurementDatasetDocument.from_bytes(
        _json_bytes({"measurements": [first_record, second_record]})
    )
    reverse = MeasurementDatasetDocument.from_bytes(
        _json_bytes({"measurements": [second_record, first_record]})
    )

    assert forward.dataset == reverse.dataset
    assert forward.digest == reverse.digest
    assert forward.digest == ContentDigest.from_value(forward.dataset.to_record())


def test_measurement_dataset_rejects_duplicate_measurement_ids() -> None:
    duplicate_record = _measurement_record(observation_id="observation-2")

    error = capture_measurement_error(
        lambda: MeasurementDataset.from_record(
            {"measurements": [_measurement_record(), duplicate_record]}
        )
    )

    assert str(error) == "duplicate measurement id: core.boolean-evidence@0.1.0"


def test_measurement_dataset_validates_every_measurement_against_manifest() -> None:
    manifest = BenchmarkManifest.from_record(_benchmark_manifest_record())
    dataset = MeasurementDataset.from_record(
        {
            "measurements": [
                _measurement_record(),
                _measurement_record(
                    evidence_id="core.second-evidence@0.1.0",
                    observation_id="observation-2",
                ),
            ]
        }
    )

    dataset.validate_manifest(manifest)

    invalid_dataset = MeasurementDataset.from_record(
        {
            "measurements": [
                _measurement_record(),
                _measurement_record(
                    benchmark_id="core.other-benchmark@0.1.0",
                    evidence_id="core.second-evidence@0.1.0",
                    observation_id="observation-2",
                ),
            ]
        }
    )
    error = capture_measurement_error(lambda: invalid_dataset.validate_manifest(manifest))

    assert str(error) == (
        "benchmark_id core.other-benchmark@0.1.0 does not match manifest "
        "core.boolean-benchmark@0.1.0"
    )


def test_measurement_dataset_document_rejects_invalid_document_bytes() -> None:
    assert (
        str(capture_measurement_error(lambda: MeasurementDatasetDocument.from_bytes(b"[]")))
        == "measurement dataset document must contain an object"
    )
    assert str(
        capture_measurement_error(
            lambda: MeasurementDatasetDocument.from_bytes(b'{"measurements": [{}]}')
        )
    ) == "benchmark_id: missing required field"


def test_measurement_dataset_round_trips_one_record_fixture() -> None:
    manifest = BenchmarkManifest.from_record(_benchmark_manifest_record())
    measurement_bytes = (_fixtures_root / "finite_outcome" / "measurement.json").read_bytes()
    single = MeasurementDocument.from_bytes(measurement_bytes)
    document = MeasurementDatasetDocument.from_bytes(
        _json_bytes({"measurements": [single.measurement.to_record()]})
    )

    assert document.dataset.to_record() == {
        "measurements": [single.measurement.to_record()]
    }
    assert document.digest == ContentDigest.from_value(document.dataset.to_record())
    document.dataset.validate_manifest(manifest)


def _measurement_record(
    *,
    benchmark_id: str = "core.boolean-benchmark@0.1.0",
    evidence_id: str = "core.boolean-evidence@0.1.0",
    observation_id: str = "observation-1",
) -> dict[str, object]:
    return {
        "benchmark_id": benchmark_id,
        **_minimal_scoring_record(
            evidence_id=evidence_id,
            observation_id=observation_id,
        ),
    }


def _expanded_measurement_record() -> dict[str, object]:
    return {
        "benchmark_id": "core.boolean-benchmark@0.1.0",
        **_expanded_scoring_record(),
    }


def _benchmark_manifest_record() -> dict[str, object]:
    return {
        "id": "core.boolean-benchmark@0.1.0",
        "outcome_space": _outcome_space_record(),
    }


def _minimal_scoring_record(
    *,
    evidence_id: str = "core.boolean-evidence@0.1.0",
    observation_id: str = "observation-1",
) -> dict[str, object]:
    return {
        "id": evidence_id,
        "observation_id": observation_id,
        "outcome_space": _outcome_space_record(),
        "accepted_event": _accepted_event_record(),
        "probability_measure": _probability_measure_record(),
    }


def _expanded_scoring_record() -> dict[str, object]:
    record = _minimal_scoring_record()
    del record["id"]
    del record["observation_id"]
    record["raw_scoring_evidence"] = {
        "id": "core.boolean-evidence@0.1.0",
        "observation_id": "observation-1",
        "outcome_space_id": "core.boolean-outcome@0.1.0",
        "accepted_event_id": "core.boolean-accepted@0.1.0",
        "probability_measure_id": "core.boolean-prediction@0.1.0",
        "accepted_mass": 0.25,
        "negative_log_score": 1.3862943611198906,
    }
    return record


def _raw_scoring_evidence_record() -> dict[str, object]:
    return {
        "id": "core.boolean-evidence@0.1.0",
        "observation_id": "observation-1",
        "outcome_space_id": "core.boolean-outcome@0.1.0",
        "accepted_event_id": "core.boolean-accepted@0.1.0",
        "probability_measure_id": "core.boolean-prediction@0.1.0",
        "accepted_mass": 0.25,
        "negative_log_score": 1.3862943611198906,
    }


def _outcome_space_record() -> dict[str, object]:
    return {
        "id": "core.boolean-outcome@0.1.0",
        "outcomes": [{"id": "yes"}, {"id": "no"}],
    }


def _accepted_event_record() -> dict[str, object]:
    return {
        "id": "core.boolean-accepted@0.1.0",
        "outcome_space_id": "core.boolean-outcome@0.1.0",
        "outcomes": ["yes"],
    }


def _probability_measure_record() -> dict[str, object]:
    return {
        "id": "core.boolean-prediction@0.1.0",
        "outcome_space_id": "core.boolean-outcome@0.1.0",
        "probabilities": [
            {"outcome_id": "no", "probability": 0.75},
            {"outcome_id": "yes", "probability": 0.25},
        ],
    }


def _other_outcome_space_record() -> dict[str, object]:
    return {
        "id": "core.other-outcome@0.1.0",
        "outcomes": [{"id": "yes"}, {"id": "no"}],
    }


def _other_accepted_event_record() -> dict[str, object]:
    return {
        "id": "core.boolean-accepted@0.1.0",
        "outcome_space_id": "core.other-outcome@0.1.0",
        "outcomes": ["yes"],
    }


def _other_probability_measure_record() -> dict[str, object]:
    return {
        "id": "core.boolean-prediction@0.1.0",
        "outcome_space_id": "core.other-outcome@0.1.0",
        "probabilities": [
            {"outcome_id": "no", "probability": 0.75},
            {"outcome_id": "yes", "probability": 0.25},
        ],
    }


def _other_raw_scoring_evidence_record() -> dict[str, object]:
    record = _raw_scoring_evidence_record()
    record["outcome_space_id"] = "core.other-outcome@0.1.0"
    return record


def _json_bytes(record: Mapping[str, object]) -> bytes:
    return canonical_document_bytes(record)


def capture_measurement_error(
    action: Callable[[], object],
) -> MeasurementRecordValidationError:
    try:
        action()
    except MeasurementRecordValidationError as error:
        return error
    raise AssertionError("expected MeasurementRecordValidationError")
