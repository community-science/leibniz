from pathlib import Path
from typing import cast

import pytest

from leibniz.architectures import ArchitectureManifestDocument
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.local_results import (
    LocalResultImportError,
    import_submission_publications,
    load_console_result_view,
)
from leibniz.measurements import MeasurementDataset, MeasurementDocument
from leibniz.publications import SubmissionPublicationDocument
from leibniz.views import MeasurementScoreView

_repository_root = Path(__file__).parents[1]


def test_import_submission_publications_materializes_runs_views(tmp_path: Path) -> None:
    source_root = tmp_path / "hf-checkout"
    source_root.mkdir()
    bundle_path = source_root / "digits_publication.json"
    bundle_path.write_bytes(canonical_document_bytes(_digits_publication_bundle_record()))

    summary = import_submission_publications(
        (source_root,),
        repository_root=_repository_root,
        runs_root=tmp_path / ".runs",
    )

    assert summary.publication_bundle_count == 1
    assert summary.measurement_count == 1
    assert summary.view_file == tmp_path / ".runs" / "views" / "imported_results.json"
    assert len(summary.import_files) == 1

    imported_bundle = SubmissionPublicationDocument.from_bytes(
        summary.import_files[0].read_bytes()
    ).bundle
    assert imported_bundle.id == ProtocolIdentifier.parse("publication-bundles.digits@0.1.0")

    view = load_console_result_view(summary.view_file.read_bytes())
    assert view["format"] == "leibniz.console.imported-results"
    bundles = cast(list[dict[str, object]], view["publication_bundles"])
    assert bundles[0]["id"] == "publication-bundles.digits@0.1.0"
    assert bundles[0]["benchmark_ids"] == ["benchmarks.digits@0.1.0"]
    assert bundles[0]["measurement_count"] == 1


def test_import_submission_publications_ignores_non_publication_json(tmp_path: Path) -> None:
    source_root = tmp_path / "hf-checkout"
    source_root.mkdir()
    (source_root / "not-a-publication.json").write_bytes(
        canonical_document_bytes({"measurement_dataset": {"measurements": []}})
    )

    with pytest.raises(LocalResultImportError, match="no publication bundle"):
        import_submission_publications(
            (source_root,),
            repository_root=_repository_root,
            runs_root=tmp_path / ".runs",
        )


def test_console_result_view_rejects_wrong_format() -> None:
    with pytest.raises(LocalResultImportError, match="unsupported format"):
        load_console_result_view(canonical_document_bytes({"format": "other", "format_version": 1}))


def _digits_publication_bundle_record() -> dict[str, object]:
    dataset = _digits_dataset()
    return {
        "id": "publication-bundles.digits@0.1.0",
        "submission_package": {
            "id": "submissions.digits-pool@0.1.0",
            "benchmark_manifest": _digits_benchmark().manifest.to_record(),
            "architecture_manifest": _architecture().manifest.to_record(),
            "measurement_dataset": dataset.to_record(),
            "artifacts": [
                {
                    "id": "artifacts.digits-weights@0.1.0",
                    "digest": str(ContentDigest.from_value({"checkpoint": "metadata"})),
                    "description": "checkpoint metadata only",
                }
            ],
        },
        "measurement_dataset": dataset.to_record(),
        "measurement_score_view": MeasurementScoreView.from_dataset(
            id=ProtocolIdentifier.parse("views.measurement-scores.digits@0.1.0"),
            dataset=dataset,
        ).to_record(),
    }


def _digits_dataset() -> MeasurementDataset:
    return MeasurementDataset.from_record({"measurements": [_digits_measurement().to_record()]})


def _digits_measurement():
    return MeasurementDocument.from_bytes(
        canonical_document_bytes(_digits_measurement_record())
    ).measurement


def _digits_measurement_record() -> dict[str, object]:
    outcome_space = _digits_benchmark().manifest.resolve_outcome_space(scale=1)
    return {
        "benchmark_id": "benchmarks.digits@0.1.0",
        "outcome_space": outcome_space.to_record(),
        "accepted_event": {
            "id": "benchmarks.digits.accepted.digit-7@0.1.0",
            "outcome_space_id": str(outcome_space.id),
            "outcomes": ["digit-7"],
        },
        "probability_measure": {
            "id": "benchmarks.digits.prediction.digit-7@0.1.0",
            "outcome_space_id": str(outcome_space.id),
            "probabilities": [
                {"outcome_id": f"digit-{digit}", "probability": 1.0 if digit == 7 else 0.0}
                for digit in range(10)
            ],
        },
        "raw_scoring_evidence": {
            "id": "benchmarks.digits.measurements.digit-7@0.1.0",
            "observation_id": "digits-l1-seed-7",
            "outcome_space_id": str(outcome_space.id),
            "accepted_event_id": "benchmarks.digits.accepted.digit-7@0.1.0",
            "probability_measure_id": "benchmarks.digits.prediction.digit-7@0.1.0",
            "accepted_mass": 1.0,
            "negative_log_score": 0.0,
        },
    }


def _digits_benchmark() -> BenchmarkManifestDocument:
    manifest_path = _repository_root / "src" / "leibniz" / "benchmarks" / "digits" / "manifest.json"
    return BenchmarkManifestDocument.from_bytes(
        manifest_path.read_bytes()
    )


def _architecture() -> ArchitectureManifestDocument:
    manifest_path = (
        _repository_root / "tests" / "fixtures" / "architecture" / "digits_pool" / "manifest.json"
    )
    return ArchitectureManifestDocument.from_bytes(
        manifest_path.read_bytes()
    )
