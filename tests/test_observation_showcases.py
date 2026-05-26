from pathlib import Path

import pytest

from leibniz.identifiers import ProtocolIdentifier
from leibniz.observation_showcases import (
    ObservationShowcaseDocument,
    ObservationShowcaseManifest,
    ObservationShowcaseValidationError,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def test_digits_observation_showcase_loads_benchmark_owned_samples() -> None:
    document = ObservationShowcaseDocument.from_bytes(
        (_digits_benchmark_root / "inspection_showcase.json").read_bytes()
    )
    manifest = document.manifest

    assert manifest.id == ProtocolIdentifier.parse(
        "benchmarks.digits.inspection-showcase@0.1.0"
    )
    assert manifest.benchmark_id == ProtocolIdentifier.parse("benchmarks.digits@0.1.0")
    assert manifest.formation_declaration.kind == "observation-formation-declaration"
    assert manifest.materialization_declaration.kind == "materialization-declaration"
    assert [sample.label for sample in manifest.samples] == [
        "Single digit 7",
        "Three digit sequence 123",
    ]
    assert manifest.samples[0].component_sequence == (7,)
    assert manifest.samples[1].component_sequence == (1, 2, 3)
    assert manifest.samples[1].scale_assignment.values == {"L": 3}
    assert manifest.samples[1].complexity_assignment.values == {"C": 3}


def test_observation_showcase_round_trips_canonically() -> None:
    document = ObservationShowcaseDocument.from_bytes(
        (_digits_benchmark_root / "inspection_showcase.json").read_bytes()
    )

    assert ObservationShowcaseManifest.from_record(
        document.manifest.to_record()
    ) == document.manifest
    assert document.digest == document.manifest.digest


def test_observation_showcase_rejects_wrong_reference_kind() -> None:
    record = ObservationShowcaseDocument.from_bytes(
        (_digits_benchmark_root / "inspection_showcase.json").read_bytes()
    ).manifest.to_record()
    formation = record["formation_declaration"]
    assert isinstance(formation, dict)
    formation["kind"] = "benchmark-manifest"

    with pytest.raises(
        ObservationShowcaseValidationError,
        match="formation_declaration reference must have kind",
    ):
        ObservationShowcaseManifest.from_record(record)


def test_observation_showcase_rejects_duplicate_sample_ids() -> None:
    record = ObservationShowcaseDocument.from_bytes(
        (_digits_benchmark_root / "inspection_showcase.json").read_bytes()
    ).manifest.to_record()
    samples = record["samples"]
    assert isinstance(samples, list)
    assert isinstance(samples[0], dict)
    assert isinstance(samples[1], dict)
    samples[1]["id"] = samples[0]["id"]

    with pytest.raises(ObservationShowcaseValidationError, match="duplicate sample id"):
        ObservationShowcaseManifest.from_record(record)
