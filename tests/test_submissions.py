from collections.abc import Callable
from pathlib import Path

from leibniz._documents import canonical_document_bytes
from leibniz.architectures import ArchitectureManifestDocument
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.content import ContentDigest
from leibniz.measurements import MeasurementDatasetDocument, MeasurementDocument
from leibniz.submissions import (
    SubmissionArtifact,
    SubmissionPackageDocument,
    SubmissionPackageManifest,
    SubmissionPackageValidationError,
)

_fixtures_root = Path(__file__).parent / "fixtures"


def test_submission_package_manifest_parses_and_canonicalizes() -> None:
    manifest = SubmissionPackageManifest.from_record(_submission_package_record())

    assert manifest == SubmissionPackageManifest(
        id=manifest.id,
        benchmark_manifest=_benchmark_document().manifest,
        architecture_manifest=_architecture_document().manifest,
        measurement_dataset=_dataset_document().dataset,
        artifacts=(
            SubmissionArtifact(
                id=manifest.artifacts[0].id,
                digest=manifest.artifacts[0].digest,
                description="checkpoint metadata only",
            ),
        ),
    )
    assert manifest.to_record() == _expanded_submission_package_record(manifest)
    assert manifest.digest == ContentDigest.from_value(manifest.to_record())


def test_submission_package_document_loads_bytes_with_digest() -> None:
    document = SubmissionPackageDocument.from_bytes(
        canonical_document_bytes(_submission_package_record())
    )

    assert document.digest == ContentDigest.from_value(document.manifest.to_record())
    assert document.manifest.measurement_dataset.digest == _dataset_document().digest


def test_submission_package_rejects_inconsistent_measurements() -> None:
    record = _submission_package_record()
    benchmark = _benchmark_document().manifest.to_record()
    benchmark["id"] = "core.other-benchmark@0.1.0"
    benchmark["name"] = "core.other-benchmark"
    record["benchmark_manifest"] = benchmark

    error = capture_submission_error(lambda: SubmissionPackageManifest.from_record(record))

    assert str(error) == (
        "benchmark_id core.boolean-benchmark@0.1.0 does not match manifest "
        "core.other-benchmark@0.1.0"
    )


def test_submission_package_rejects_missing_or_malformed_references() -> None:
    record = _submission_package_record()
    del record["architecture_manifest"]
    assert str(capture_submission_error(lambda: SubmissionPackageManifest.from_record(record))) == (
        "architecture_manifest: missing required field"
    )

    record = _submission_package_record()
    record["architecture_manifest"] = {"input_shape": [1]}
    assert str(capture_submission_error(lambda: SubmissionPackageManifest.from_record(record))) == (
        "output_shape: missing required field; layers: missing required field"
    )

    record = _submission_package_record()
    record["measurement_dataset"] = {"measurements": []}
    assert str(capture_submission_error(lambda: SubmissionPackageManifest.from_record(record))) == (
        "measurement_dataset must contain at least one measurement"
    )

    record = _submission_package_record()
    record["id"] = "core.boolean-digits-pool@0.1.0"
    assert str(capture_submission_error(lambda: SubmissionPackageManifest.from_record(record))) == (
        "id must be a valid submission package id"
    )


def test_submission_package_artifact_metadata_is_durable_only() -> None:
    record = _submission_package_record()
    record["artifacts"] = [
        {
            "id": "artifacts.model-weights@0.1.0",
            "digest": "sha256:not-a-digest",
        }
    ]
    assert str(capture_submission_error(lambda: SubmissionPackageManifest.from_record(record))) == (
        "sha256 digest must be 64 lowercase hexadecimal characters"
    )

    record = _submission_package_record()
    record["artifacts"] = [
        {
            "id": "artifacts.model-weights@0.1.0",
            "digest": str(_architecture_document().digest),
            "path": ".leibniz/checkpoints/model.pt",
        }
    ]
    assert str(capture_submission_error(lambda: SubmissionPackageManifest.from_record(record))) == (
        "path: unknown field"
    )

    record = _submission_package_record()
    artifact = _artifact_record()
    record["artifacts"] = [artifact, artifact]
    assert str(capture_submission_error(lambda: SubmissionPackageManifest.from_record(record))) == (
        "duplicate artifact id: artifacts.model-weights@0.1.0"
    )


def test_submission_package_document_rejects_invalid_bytes() -> None:
    assert str(
        capture_submission_error(lambda: SubmissionPackageDocument.from_bytes(b"[]"))
    ) == "submission package document must contain an object"
    assert str(
        capture_submission_error(
            lambda: SubmissionPackageDocument.from_bytes(
                canonical_document_bytes({"id": "submissions.boolean@0.1.0"})
            )
        )
    ) == (
        "benchmark_manifest: missing required field; architecture_manifest: "
        "missing required field; measurement_dataset: missing required field"
    )


def _submission_package_record() -> dict[str, object]:
    return {
        "id": "submissions.boolean-digits-pool@0.1.0",
        "benchmark_manifest": _benchmark_document().manifest.to_record(),
        "architecture_manifest": _architecture_document().manifest.to_record(),
        "measurement_dataset": _dataset_document().dataset.to_record(),
        "artifacts": [_artifact_record()],
    }


def _expanded_submission_package_record(
    manifest: SubmissionPackageManifest,
) -> dict[str, object]:
    return {
        "id": "submissions.boolean-digits-pool@0.1.0",
        "benchmark_manifest": _benchmark_document().manifest.to_record(),
        "architecture_manifest": _architecture_document().manifest.to_record(),
        "measurement_dataset": _dataset_document().dataset.to_record(),
        "artifacts": [manifest.artifacts[0].to_record()],
    }


def _artifact_record() -> dict[str, object]:
    return {
        "id": "artifacts.model-weights@0.1.0",
        "digest": str(_architecture_document().digest),
        "description": "checkpoint metadata only",
    }


def _benchmark_document() -> BenchmarkManifestDocument:
    return BenchmarkManifestDocument.from_bytes(
        (_fixtures_root / "finite_outcome" / "manifest.json").read_bytes()
    )


def _architecture_document() -> ArchitectureManifestDocument:
    return ArchitectureManifestDocument.from_bytes(
        (_fixtures_root / "architecture" / "digits_pool" / "manifest.json").read_bytes()
    )


def _dataset_document() -> MeasurementDatasetDocument:
    measurement = MeasurementDocument.from_bytes(
        (_fixtures_root / "finite_outcome" / "measurement.json").read_bytes()
    ).measurement
    return MeasurementDatasetDocument.from_bytes(
        canonical_document_bytes({"measurements": [measurement.to_record()]})
    )


def capture_submission_error(
    action: Callable[[], object],
) -> SubmissionPackageValidationError:
    try:
        action()
    except SubmissionPackageValidationError as error:
        return error
    raise AssertionError("expected SubmissionPackageValidationError")
