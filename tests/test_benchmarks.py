import math
from collections.abc import Callable, Mapping
from typing import cast

from leibniz.benchmarks import (
    BenchmarkDeclaration,
    BenchmarkDeclarationValidationError,
    BenchmarkManifest,
    BenchmarkManifestDocument,
    BenchmarkManifestValidationError,
)
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.outcomes import FiniteOutcomeScoringBundle


def test_benchmark_declaration_parses_finite_outcome_scoring_contract() -> None:
    declaration = BenchmarkDeclaration.from_record(_benchmark_declaration_record())

    assert declaration == BenchmarkDeclaration(
        id=ProtocolIdentifier.parse("core.boolean-benchmark@0.1.0"),
        outcome_space_id=ProtocolIdentifier.parse("core.boolean-outcome@0.1.0"),
    )
    assert declaration.to_record() == _benchmark_declaration_record()


def test_benchmark_declaration_defaults_finite_outcome_protocol_bindings() -> None:
    declaration = BenchmarkDeclaration.from_record(
        {
            "id": "core.boolean-benchmark@0.1.0",
            "outcome_space_id": "core.boolean-outcome@0.1.0",
        }
    )

    assert declaration.to_record() == _benchmark_declaration_record()


def test_benchmark_declaration_validates_matching_scoring_bundle() -> None:
    declaration = BenchmarkDeclaration.from_record(_benchmark_declaration_record())
    bundle = FiniteOutcomeScoringBundle.from_record(_boolean_bundle_record())

    declaration.validate_bundle(bundle)


def test_benchmark_declaration_rejects_mismatched_bundle_outcome_space() -> None:
    declaration = BenchmarkDeclaration.from_record(_benchmark_declaration_record())
    record = _boolean_bundle_record()
    outcome_space = dict(cast(Mapping[str, object], record["outcome_space"]))
    outcome_space["id"] = "core.other-outcome@0.1.0"
    event = dict(cast(Mapping[str, object], record["accepted_event"]))
    event["outcome_space_id"] = "core.other-outcome@0.1.0"
    measure = dict(cast(Mapping[str, object], record["probability_measure"]))
    measure["outcome_space_id"] = "core.other-outcome@0.1.0"
    evidence = dict(cast(Mapping[str, object], record["raw_scoring_evidence"]))
    evidence["outcome_space_id"] = "core.other-outcome@0.1.0"
    record["outcome_space"] = outcome_space
    record["accepted_event"] = event
    record["probability_measure"] = measure
    record["raw_scoring_evidence"] = evidence
    bundle = FiniteOutcomeScoringBundle.from_record(record)

    error = capture_benchmark_error(lambda: declaration.validate_bundle(bundle))

    assert str(error) == (
        "bundle outcome_space_id core.other-outcome@0.1.0 does not match "
        "core.boolean-outcome@0.1.0"
    )


def test_benchmark_declaration_rejects_invalid_protocol_binding_identifiers() -> None:
    record = _benchmark_declaration_record()
    record["score_functional_id"] = "core.mean-accuracy@0.1.0"

    error = capture_benchmark_error(lambda: BenchmarkDeclaration.from_record(record))

    assert str(error) == (
        "score_functional_id core.mean-accuracy@0.1.0 does not match "
        "core.negative-log-accepted-mass@0.1.0"
    )


def test_benchmark_declaration_rejects_malformed_records() -> None:
    assert str(
        capture_benchmark_error(
            lambda: BenchmarkDeclaration.from_record(
                {
                    "id": "core.boolean-benchmark@1.0.0",
                    "outcome_space_id": "core.boolean-outcome@0.1.0",
                    "oracle_acceptance_id": "core.finite-outcome-accepted-event@0.1.0",
                    "prediction_interface_id": (
                        "core.finite-probability-measure-prediction@0.1.0"
                    ),
                    "score_functional_id": "core.negative-log-accepted-mass@0.1.0",
                    "evidence_bundle_id": "core.finite-outcome-scoring-bundle@0.1.0",
                }
            )
        )
    ) == (
        "identifier must use a pre-1.0.0 version before release policy exists: "
        "core.boolean-benchmark@1.0.0"
    )
    record = _benchmark_declaration_record()
    record["summary"] = "boolean finite-outcome benchmark"

    error = capture_benchmark_error(lambda: BenchmarkDeclaration.from_record(record))

    assert str(error) == "summary: unknown field"


def test_benchmark_declaration_digest_is_stable() -> None:
    record = _benchmark_declaration_record()
    reordered = {
        "score_functional_id": record["score_functional_id"],
        "prediction_interface_id": record["prediction_interface_id"],
        "oracle_acceptance_id": record["oracle_acceptance_id"],
        "id": record["id"],
        "evidence_bundle_id": record["evidence_bundle_id"],
        "outcome_space_id": record["outcome_space_id"],
    }

    assert ContentDigest.from_value(record) == ContentDigest.from_value(reordered)


def test_benchmark_manifest_parses_declaration_container() -> None:
    manifest = BenchmarkManifest.from_record(_benchmark_manifest_record())

    assert manifest == BenchmarkManifest(
        id=ProtocolIdentifier.parse("core.boolean-benchmark@0.1.0"),
        name=ProtocolIdentifier.parse("core.boolean-benchmark@0.1.0").name,
        declaration=BenchmarkDeclaration.from_record(_benchmark_declaration_record()),
    )
    assert manifest.to_record() == _benchmark_manifest_record()


def test_benchmark_manifest_parses_minimal_authoring_record() -> None:
    manifest = BenchmarkManifest.from_record(_minimal_benchmark_manifest_record())

    assert manifest == BenchmarkManifest.from_record(_benchmark_manifest_record())
    assert manifest.to_record() == _benchmark_manifest_record()


def test_benchmark_manifest_defaults_name_from_id() -> None:
    manifest = BenchmarkManifest.from_record(_two_field_benchmark_manifest_record())

    assert manifest == BenchmarkManifest.from_record(_benchmark_manifest_record())
    assert manifest.to_record() == _benchmark_manifest_record()


def test_benchmark_manifest_validates_matching_scoring_bundle() -> None:
    manifest = BenchmarkManifest.from_record(_benchmark_manifest_record())
    bundle = FiniteOutcomeScoringBundle.from_record(_boolean_bundle_record())

    manifest.validate_bundle(bundle)


def test_benchmark_manifest_rejects_mismatched_name_and_declaration_id() -> None:
    name_record = _benchmark_manifest_record()
    name_record["name"] = "core.other-benchmark"

    assert str(
        capture_manifest_error(lambda: BenchmarkManifest.from_record(name_record))
    ) == "name core.other-benchmark does not match id name core.boolean-benchmark"

    declaration_record = _benchmark_declaration_record()
    declaration_record["id"] = "core.other-benchmark@0.1.0"
    declaration_record["outcome_space_id"] = "core.boolean-outcome@0.1.0"
    declaration_record["oracle_acceptance_id"] = "core.finite-outcome-accepted-event@0.1.0"
    declaration_record["prediction_interface_id"] = (
        "core.finite-probability-measure-prediction@0.1.0"
    )
    declaration_record["score_functional_id"] = "core.negative-log-accepted-mass@0.1.0"
    declaration_record["evidence_bundle_id"] = "core.finite-outcome-scoring-bundle@0.1.0"
    declaration_id_record = _benchmark_manifest_record()
    declaration_id_record["declaration"] = declaration_record

    assert str(
        capture_manifest_error(lambda: BenchmarkManifest.from_record(declaration_id_record))
    ) == (
        "declaration id core.other-benchmark@0.1.0 does not match "
        "core.boolean-benchmark@0.1.0"
    )


def test_benchmark_manifest_rejects_malformed_records() -> None:
    assert str(
        capture_manifest_error(
            lambda: BenchmarkManifest.from_record(
                {
                    "id": "core.boolean-benchmark@1.0.0",
                    "name": "core.boolean-benchmark",
                    "declaration": _benchmark_declaration_record(),
                }
            )
        )
    ) == (
        "identifier must use a pre-1.0.0 version before release policy exists: "
        "core.boolean-benchmark@1.0.0"
    )
    record = _benchmark_manifest_record()
    record["summary"] = "boolean finite-outcome benchmark"

    error = capture_manifest_error(lambda: BenchmarkManifest.from_record(record))

    assert str(error) == "summary: unknown field"


def test_benchmark_manifest_rejects_missing_or_conflicting_declaration_inputs() -> None:
    assert str(
        capture_manifest_error(
            lambda: BenchmarkManifest.from_record(
                {
                    "id": "core.boolean-benchmark@0.1.0",
                    "name": "core.boolean-benchmark",
                }
            )
        )
    ) == "outcome_space_id: missing required field"

    record = _minimal_benchmark_manifest_record()
    declaration = _benchmark_declaration_record()
    declaration["outcome_space_id"] = "core.other-outcome@0.1.0"
    record["declaration"] = declaration

    assert str(capture_manifest_error(lambda: BenchmarkManifest.from_record(record))) == (
        "outcome_space_id core.boolean-outcome@0.1.0 does not match declaration "
        "core.other-outcome@0.1.0"
    )


def test_benchmark_manifest_digest_is_stable() -> None:
    record = _benchmark_manifest_record()
    reordered = {
        "declaration": record["declaration"],
        "name": record["name"],
        "id": record["id"],
    }

    assert ContentDigest.from_value(record) == ContentDigest.from_value(reordered)


def test_benchmark_manifest_document_loads_bytes_with_digest() -> None:
    document = BenchmarkManifestDocument.from_bytes(
        b"""{
            "name": "core.boolean-benchmark",
            "id": "core.boolean-benchmark@0.1.0",
            "declaration": {
                "id": "core.boolean-benchmark@0.1.0",
                "outcome_space_id": "core.boolean-outcome@0.1.0",
                "oracle_acceptance_id": "core.finite-outcome-accepted-event@0.1.0",
                "prediction_interface_id": "core.finite-probability-measure-prediction@0.1.0",
                "score_functional_id": "core.negative-log-accepted-mass@0.1.0",
                "evidence_bundle_id": "core.finite-outcome-scoring-bundle@0.1.0"
            }
        }"""
    )

    assert document.manifest == BenchmarkManifest.from_record(_benchmark_manifest_record())
    assert document.digest == ContentDigest.from_value(_benchmark_manifest_record())


def test_benchmark_manifest_document_expands_minimal_authoring_record() -> None:
    document = BenchmarkManifestDocument.from_bytes(
        _json_bytes(_two_field_benchmark_manifest_record())
    )

    assert document.manifest == BenchmarkManifest.from_record(_benchmark_manifest_record())
    assert document.digest == ContentDigest.from_value(_benchmark_manifest_record())


def test_benchmark_manifest_document_rejects_invalid_document_bytes() -> None:
    assert str(
        capture_manifest_error(lambda: BenchmarkManifestDocument.from_bytes(b"\xff"))
    ) == "manifest document must be UTF-8"
    assert str(
        capture_manifest_error(lambda: BenchmarkManifestDocument.from_bytes(b"{"))
    ) == "invalid manifest document: Expecting property name enclosed in double quotes"
    assert str(
        capture_manifest_error(lambda: BenchmarkManifestDocument.from_bytes(b"[]"))
    ) == "manifest document must contain an object"


def test_benchmark_manifest_document_rejects_invalid_manifest_record() -> None:
    record = _benchmark_manifest_record()
    record["name"] = "core.other-benchmark"

    error = capture_manifest_error(
        lambda: BenchmarkManifestDocument.from_bytes(_json_bytes(record))
    )

    assert str(error) == "name core.other-benchmark does not match id name core.boolean-benchmark"


def test_benchmark_manifest_document_digest_is_stable() -> None:
    left = BenchmarkManifestDocument.from_bytes(_json_bytes(_benchmark_manifest_record()))
    right = BenchmarkManifestDocument.from_bytes(
        _json_bytes(_two_field_benchmark_manifest_record())
    )

    assert left.digest == right.digest


def capture_benchmark_error(
    call: Callable[[], object],
) -> BenchmarkDeclarationValidationError:
    try:
        call()
    except BenchmarkDeclarationValidationError as error:
        return error
    raise AssertionError("expected BenchmarkDeclarationValidationError")


def capture_manifest_error(
    call: Callable[[], object],
) -> BenchmarkManifestValidationError:
    try:
        call()
    except BenchmarkManifestValidationError as error:
        return error
    raise AssertionError("expected BenchmarkManifestValidationError")


def _benchmark_declaration_record() -> dict[str, object]:
    return {
        "id": "core.boolean-benchmark@0.1.0",
        "outcome_space_id": "core.boolean-outcome@0.1.0",
        "oracle_acceptance_id": "core.finite-outcome-accepted-event@0.1.0",
        "prediction_interface_id": "core.finite-probability-measure-prediction@0.1.0",
        "score_functional_id": "core.negative-log-accepted-mass@0.1.0",
        "evidence_bundle_id": "core.finite-outcome-scoring-bundle@0.1.0",
    }


def _benchmark_manifest_record() -> dict[str, object]:
    return {
        "id": "core.boolean-benchmark@0.1.0",
        "name": "core.boolean-benchmark",
        "declaration": _benchmark_declaration_record(),
    }


def _minimal_benchmark_manifest_record() -> dict[str, object]:
    return {
        "id": "core.boolean-benchmark@0.1.0",
        "name": "core.boolean-benchmark",
        "outcome_space_id": "core.boolean-outcome@0.1.0",
    }


def _two_field_benchmark_manifest_record() -> dict[str, object]:
    return {
        "id": "core.boolean-benchmark@0.1.0",
        "outcome_space_id": "core.boolean-outcome@0.1.0",
    }


def _json_bytes(record: Mapping[str, object]) -> bytes:
    import json

    return json.dumps(record).encode("utf-8")


def _boolean_bundle_record() -> dict[str, object]:
    return {
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
            "id": "core.boolean-prediction@0.1.0",
            "outcome_space_id": "core.boolean-outcome@0.1.0",
            "probabilities": [
                {"outcome_id": "no", "probability": 0.25},
                {"outcome_id": "yes", "probability": 0.75},
            ],
        },
        "raw_scoring_evidence": {
            "id": "core.boolean-evidence@0.1.0",
            "observation_id": "observation-1",
            "outcome_space_id": "core.boolean-outcome@0.1.0",
            "accepted_event_id": "core.boolean-accepted@0.1.0",
            "probability_measure_id": "core.boolean-prediction@0.1.0",
            "accepted_mass": 0.75,
            "negative_log_score": -math.log(0.75),
        },
    }
