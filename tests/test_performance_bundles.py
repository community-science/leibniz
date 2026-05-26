from pathlib import Path
from typing import cast

import pytest

from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import MaterializationDeclarationDocument
from leibniz.observation_formation import ObservationFormationDeclarationDocument
from leibniz.performance_bundles import (
    PerformanceViewBundle,
    PerformanceViewBundleDocument,
    PerformanceViewBundleManifest,
    PerformanceViewBundleValidationError,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_performance_bundle_fixture = Path(__file__).with_name("performance_view_bundle_fixture.json")


def test_digits_performance_view_bundle_loads_validated_sources() -> None:
    bundle = _digits_bundle()
    entry = bundle.competence_integral_view.entries[0]

    assert bundle.id == ProtocolIdentifier.parse("performance-view-bundles.digits@0.1.0")
    assert len(bundle.manifest.measurement_cases) == 2
    assert len(bundle.measurement_dataset.measurements) == 2
    assert len(bundle.materialization_plans) == 2
    assert bundle.competence_integral_view.source_dataset_digest == (
        bundle.measurement_dataset.digest
    )
    assert entry.benchmark_id == ProtocolIdentifier.parse("benchmarks.digits@0.1.0")
    assert entry.observed_complexities == (1.0, 2.0)
    assert entry.missing_complexities == (3.0,)
    assert entry.coverage == 2 / 3
    assert entry.integral == 0.25


def test_performance_view_bundle_round_trips_canonically() -> None:
    document = PerformanceViewBundleDocument.from_bytes(
        _performance_bundle_fixture.read_bytes()
    )

    assert PerformanceViewBundleManifest.from_record(
        document.manifest.to_record()
    ) == document.manifest
    assert document.digest == document.manifest.digest


def test_performance_view_bundle_rejects_source_reference_mismatch() -> None:
    record = PerformanceViewBundleDocument.from_bytes(
        _performance_bundle_fixture.read_bytes()
    ).manifest.to_record()
    benchmark_reference = record["benchmark_manifest"]
    assert isinstance(benchmark_reference, dict)
    benchmark_reference["protocol_id"] = "benchmarks.other@0.1.0"

    with pytest.raises(
        PerformanceViewBundleValidationError,
        match="benchmark_manifest reference does not match source manifest",
    ):
        _digits_bundle(manifest=PerformanceViewBundleManifest.from_record(record))


def test_performance_view_bundle_rejects_malformed_probability_cases() -> None:
    record = PerformanceViewBundleDocument.from_bytes(
        _performance_bundle_fixture.read_bytes()
    ).manifest.to_record()
    cases = record["measurement_cases"]
    assert isinstance(cases, list)
    first = cast(dict[str, object], cases[0])
    assert isinstance(first, dict)
    probabilities = cast(list[object], first["probabilities"])
    assert isinstance(probabilities, list)
    probability = cast(dict[str, object], probabilities[0])
    assert isinstance(probability, dict)
    probability["outcome_id"] = "digit-7"

    with pytest.raises(
        PerformanceViewBundleValidationError,
        match="probability mass must declare exactly one outcome identity",
    ):
        PerformanceViewBundleManifest.from_record(record)


def _digits_bundle(
    *,
    manifest: PerformanceViewBundleManifest | None = None,
) -> PerformanceViewBundle:
    if manifest is None:
        manifest = PerformanceViewBundleDocument.from_bytes(
            _performance_bundle_fixture.read_bytes()
        ).manifest
    benchmark_manifest = BenchmarkManifestDocument.from_bytes(
        (_digits_benchmark_root / "manifest.json").read_bytes()
    ).manifest
    materialization_declaration = MaterializationDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "materialization.json").read_bytes()
    ).declaration
    observation_formation_declaration = ObservationFormationDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "observation_formation.json").read_bytes()
    ).declaration
    return PerformanceViewBundle.from_manifest(
        manifest,
        benchmark_manifest=benchmark_manifest,
        materialization_declaration=materialization_declaration,
        observation_formation_declaration=observation_formation_declaration,
    )
