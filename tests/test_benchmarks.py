import math
from collections.abc import Callable, Mapping
from pathlib import Path

from benchmark_typing import load_digits_benchmark

from leibniz.benchmarks import (
    BenchmarkManifest,
    BenchmarkManifestDocument,
    BenchmarkManifestValidationError,
)
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementRecord, MeasurementRecordValidationError
from leibniz.outcomes import OutcomeSpace

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def test_benchmark_manifest_parses_finite_outcome_contract() -> None:
    manifest = BenchmarkManifest.from_record(_benchmark_manifest_record())

    assert manifest == BenchmarkManifest(
        id=ProtocolIdentifier.parse("core.boolean-benchmark@0.1.0"),
        name=ProtocolIdentifier.parse("core.boolean-benchmark@0.1.0").name,
        outcome_space=OutcomeSpace.from_record(_outcome_space_record()),
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


def test_benchmark_manifest_accepts_declared_observation_ids() -> None:
    record = _two_field_benchmark_manifest_record()
    record["observation_ids"] = ["fen:7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"]

    manifest = BenchmarkManifest.from_record(record)

    assert manifest.observation_ids == frozenset({"fen:7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"})
    assert manifest.to_record() == {
        "id": "core.boolean-benchmark@0.1.0",
        "name": "core.boolean-benchmark",
        "outcome_space": _outcome_space_record(),
        "observation_ids": ["fen:7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"],
    }


def test_benchmark_manifest_validates_matching_measurement() -> None:
    manifest = BenchmarkManifest.from_record(_benchmark_manifest_record())
    measurement = MeasurementRecord.from_record(_measurement_record())

    measurement.validate_manifest(manifest)


def test_benchmark_manifest_validates_declared_observation_ids() -> None:
    record = _two_field_benchmark_manifest_record()
    record["observation_ids"] = ["observation-1"]
    manifest = BenchmarkManifest.from_record(record)
    measurement = MeasurementRecord.from_record(_measurement_record())

    measurement.validate_manifest(manifest)

    record["observation_ids"] = ["observation-2"]
    manifest = BenchmarkManifest.from_record(record)

    assert str(capture_measurement_error(lambda: measurement.validate_manifest(manifest))) == (
        "observation_id 'observation-1' is not declared by core.boolean-benchmark@0.1.0"
    )


def test_benchmark_manifest_observation_ids_are_order_independent() -> None:
    left_record = _two_field_benchmark_manifest_record()
    left_record["observation_ids"] = ["observation-2", "observation-1"]
    right_record = _two_field_benchmark_manifest_record()
    right_record["observation_ids"] = ["observation-1", "observation-2"]

    left = BenchmarkManifest.from_record(left_record)
    right = BenchmarkManifest.from_record(right_record)

    assert left == right
    assert left.to_record()["observation_ids"] == ["observation-1", "observation-2"]
    assert ContentDigest.from_value(left.to_record()) == ContentDigest.from_value(right.to_record())


def test_benchmark_manifest_rejects_mismatched_name() -> None:
    name_record = _benchmark_manifest_record()
    name_record["name"] = "core.other-benchmark"

    assert (
        str(capture_manifest_error(lambda: BenchmarkManifest.from_record(name_record)))
        == "name core.other-benchmark does not match id name core.boolean-benchmark"
    )


def test_benchmark_manifest_rejects_malformed_records() -> None:
    assert str(
        capture_manifest_error(
            lambda: BenchmarkManifest.from_record(
                {
                    "id": "core.boolean-benchmark@1.0.0",
                    "name": "core.boolean-benchmark",
                    "outcome_space": _outcome_space_record(),
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


def test_benchmark_manifest_rejects_invalid_observation_ids() -> None:
    record = _two_field_benchmark_manifest_record()
    record["observation_ids"] = []

    assert str(capture_manifest_error(lambda: BenchmarkManifest.from_record(record))) == (
        "observation_ids must contain at least one observation id"
    )

    record["observation_ids"] = ["observation-1", "observation-1"]
    assert str(capture_manifest_error(lambda: BenchmarkManifest.from_record(record))) == (
        "observation_ids must be unique"
    )

    record["observation_ids"] = [""]
    assert str(capture_manifest_error(lambda: BenchmarkManifest.from_record(record))) == (
        "observation_ids must be nonempty"
    )


def test_benchmark_manifest_accepts_inverse_manifest_without_outcome_declaration() -> None:
    manifest = BenchmarkManifest.from_record(
        {
            "id": "core.inverse-benchmark@0.1.0",
            "name": "core.inverse-benchmark",
        }
    )

    assert manifest.outcome_space is None
    assert "outcome_space" not in manifest.to_record()
    assert str(capture_manifest_error(manifest.resolve_outcome_space)) == (
        "manifest does not declare outcome_space"
    )


def test_benchmark_manifest_rejects_unknown_fields() -> None:
    record = _benchmark_manifest_record()
    record["unsupported_field"] = {"atom_count": 10, "atom_name": "digit"}

    assert str(capture_manifest_error(lambda: BenchmarkManifest.from_record(record))) == (
        "unsupported_field: unknown field"
    )


def test_benchmark_manifest_digest_is_stable() -> None:
    record = _benchmark_manifest_record()
    reordered = {
        "outcome_space": record["outcome_space"],
        "name": record["name"],
        "id": record["id"],
    }

    assert ContentDigest.from_value(record) == ContentDigest.from_value(reordered)


def test_benchmark_manifest_document_loads_bytes_with_digest() -> None:
    document = BenchmarkManifestDocument.from_bytes(
        b"""{
            "name": "core.boolean-benchmark",
            "id": "core.boolean-benchmark@0.1.0",
            "outcome_space": {
                "id": "core.boolean-outcome@0.1.0",
                "outcomes": [{"id": "yes"}, {"id": "no"}]
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
    assert (
        str(capture_manifest_error(lambda: BenchmarkManifestDocument.from_bytes(b"\xff")))
        == "manifest document must be UTF-8"
    )
    assert (
        str(capture_manifest_error(lambda: BenchmarkManifestDocument.from_bytes(b"{")))
        == "invalid manifest document: Expecting property name enclosed in double quotes"
    )
    assert (
        str(capture_manifest_error(lambda: BenchmarkManifestDocument.from_bytes(b"[]")))
        == "manifest document must contain an object"
    )


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


def test_digits_benchmark_manifest_is_python_owned() -> None:
    manifest = load_digits_benchmark(_digits_benchmark_root).manifest

    assert manifest.id == ProtocolIdentifier.parse("benchmarks.digits@0.1.0")
    assert manifest.outcome_space is None
    assert manifest.latent_factor_declaration is not None
    assert manifest.latent_factor_declaration.kind == "latent-factor-declaration"
    assert str(manifest.latent_factor_declaration.protocol_id) == (
        "benchmarks.digits.latent-factors@0.2.0"
    )
    assert manifest.resolution_analysis == {
        "kind": "component-discriminability-margin",
        "discriminability_margin": 20.0,
        "description": (
            "Native-footprint component separation at the fixed render pitch."
        ),
        "render_unit_side": 28,
        "translation_step_pixels": 4,
        "scale_ratio_per_level": 0.12,
        "volume_value": {
            "kind": "domain-growth-setup-window",
            "measure_id": "log2-state-space-volume",
            "formula": "log2(realized_cardinality)",
            "digit_count": 10,
            "transform_axes": ["x_translation", "y_translation", "scale"],
            "target_policy": "contiguous-global-address-increment",
            "description": (
                "Score-bearing Digits volume windows are finite increments "
                "of one global address walk over digit identity and a "
                "shell-ordered translation/scale transform lattice. The "
                "output task remains 10-way classification; volume counts "
                "problem setups. Canvas size is materialization metadata "
                "derived from the deepest realized transform ordinal."
            ),
        },
    }
    assert manifest.resolution_discriminability_margin() == 20.0
    assert manifest.affine_acceptance_thresholds() == {}
    assert ContentDigest.from_value(manifest.to_record()) == ContentDigest.from_value(
        BenchmarkManifest.from_record(manifest.to_record()).to_record()
    )


def test_digits_benchmark_manifest_validates_volume_declaration() -> None:
    benchmark = load_digits_benchmark(_digits_benchmark_root)

    benchmark.manifest.validate_latent_factor_declaration(benchmark.latent_factors)


def test_benchmark_manifest_rejects_wrong_latent_factor_reference_kind() -> None:
    record = _two_field_benchmark_manifest_record()
    record["latent_factor_declaration"] = {
        "kind": "benchmark-manifest",
        "protocol_id": "benchmarks.digits.latent-factors@0.2.0",
    }

    assert str(capture_manifest_error(lambda: BenchmarkManifest.from_record(record))) == (
        "latent_factor_declaration reference must have kind latent-factor-declaration"
    )


def capture_manifest_error(
    call: Callable[[], object],
) -> BenchmarkManifestValidationError:
    try:
        call()
    except BenchmarkManifestValidationError as error:
        return error
    raise AssertionError("expected BenchmarkManifestValidationError")


def capture_measurement_error(
    call: Callable[[], object],
) -> MeasurementRecordValidationError:
    try:
        call()
    except MeasurementRecordValidationError as error:
        return error
    raise AssertionError("expected MeasurementRecordValidationError")


def _benchmark_manifest_record() -> dict[str, object]:
    return {
        "id": "core.boolean-benchmark@0.1.0",
        "name": "core.boolean-benchmark",
        "outcome_space": _outcome_space_record(),
    }


def _minimal_benchmark_manifest_record() -> dict[str, object]:
    return {
        "id": "core.boolean-benchmark@0.1.0",
        "name": "core.boolean-benchmark",
        "outcome_space": _outcome_space_record(),
    }


def _two_field_benchmark_manifest_record() -> dict[str, object]:
    return {
        "id": "core.boolean-benchmark@0.1.0",
        "outcome_space": _outcome_space_record(),
    }


def _outcome_space_record() -> dict[str, object]:
    return {
        "id": "core.boolean-outcome@0.1.0",
        "outcomes": [{"id": "yes"}, {"id": "no"}],
    }


def _json_bytes(record: Mapping[str, object]) -> bytes:
    return canonical_document_bytes(record)


def _measurement_record() -> dict[str, object]:
    return {
        "benchmark_id": "core.boolean-benchmark@0.1.0",
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
