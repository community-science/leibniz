from pathlib import Path

import pytest
from benchmark_typing import load_digits_benchmark

from leibniz.identifiers import ProtocolIdentifier
from leibniz.observation_showcases import (
    ObservationShowcaseManifest,
    ObservationShowcaseValidationError,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def test_digits_observation_showcase_loads_benchmark_owned_samples() -> None:
    manifest = load_digits_benchmark(_digits_benchmark_root).showcase

    assert manifest.id == ProtocolIdentifier.parse(
        "benchmarks.digits.inspection-showcase@0.1.0"
    )
    assert manifest.benchmark_id == ProtocolIdentifier.parse("benchmarks.digits@0.1.0")
    assert manifest.formation_declaration.kind == "observation-formation-declaration"
    assert manifest.materialization_declaration.kind == "materialization-declaration"
    assert [sample.label for sample in manifest.samples] == [
        "Single digit 7",
        "Single digit 3",
    ]
    assert manifest.samples[0].component_index == 7
    assert manifest.samples[1].component_index == 3


def test_observation_showcase_round_trips_canonically() -> None:
    manifest = load_digits_benchmark(_digits_benchmark_root).showcase

    assert ObservationShowcaseManifest.from_record(manifest.to_record()) == manifest
    assert manifest.digest == ObservationShowcaseManifest.from_record(
        manifest.to_record()
    ).digest


def test_observation_showcase_rejects_wrong_reference_kind() -> None:
    record = load_digits_benchmark(_digits_benchmark_root).showcase.to_record()
    formation = record["formation_declaration"]
    assert isinstance(formation, dict)
    formation["kind"] = "benchmark-manifest"

    with pytest.raises(
        ObservationShowcaseValidationError,
        match="formation_declaration reference must have kind",
    ):
        ObservationShowcaseManifest.from_record(record)


def test_observation_showcase_rejects_duplicate_sample_ids() -> None:
    record = load_digits_benchmark(_digits_benchmark_root).showcase.to_record()
    samples = record["samples"]
    assert isinstance(samples, list)
    assert isinstance(samples[0], dict)
    assert isinstance(samples[1], dict)
    samples[1]["id"] = samples[0]["id"]

    with pytest.raises(ObservationShowcaseValidationError, match="duplicate sample id"):
        ObservationShowcaseManifest.from_record(record)
