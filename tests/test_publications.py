from collections.abc import Callable
from pathlib import Path
from typing import cast

from leibniz.architectures import ArchitectureManifestDocument
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset, MeasurementDatasetDocument, MeasurementDocument
from leibniz.publications import (
    SubmissionPublicationBundle,
    SubmissionPublicationDocument,
    SubmissionPublicationValidationError,
)
from leibniz.submissions import SubmissionPackageManifest
from leibniz.views import MeasurementScoreView

_fixtures_root = Path(__file__).parent / "fixtures"


def test_submission_publication_bundle_parses_and_canonicalizes() -> None:
    dataset = _dataset_document().dataset
    submission = _submission_package()
    view = _measurement_score_view(dataset=dataset)

    bundle = SubmissionPublicationBundle.from_record(_publication_bundle_record())

    assert bundle == SubmissionPublicationBundle(
        id=ProtocolIdentifier.parse("publication-bundles.boolean@0.1.0"),
        submission_package=submission,
        measurement_dataset=dataset,
        measurement_score_view=view,
        proposal_set=bundle.proposal_set,
        architecture_surrogate=bundle.architecture_surrogate,
    )
    assert bundle.to_record() == _publication_bundle_record()
    assert bundle.digest == ContentDigest.from_value(bundle.to_record())


def test_submission_publication_document_loads_bytes_with_digest() -> None:
    document = SubmissionPublicationDocument.from_bytes(
        canonical_document_bytes(_publication_bundle_record())
    )

    assert document.bundle.measurement_dataset.digest == _dataset_document().dataset.digest
    assert document.digest == ContentDigest.from_value(document.bundle.to_record())


def test_submission_publication_rejects_inconsistent_dataset_sources() -> None:
    record = _publication_bundle_record()
    bundle_dataset = _dataset_document().dataset
    package = SubmissionPackageManifest.from_record(_submission_package_record())
    altered_package = SubmissionPackageManifest(
        id=package.id,
        benchmark_manifest=package.benchmark_manifest,
        architecture_manifest=package.architecture_manifest,
        measurement_dataset=MeasurementDataset.from_record(
            {
                "measurements": [
                    package.measurement_dataset.measurements[0].to_record(),
                    _alternate_measurement_record(),
                ]
            }
        ),
        artifacts=package.artifacts,
    )
    record["submission_package"] = altered_package.to_record()

    assert str(
        capture_publication_error(lambda: SubmissionPublicationBundle.from_record(record))
    ) == "submission_package measurement_dataset does not match bundle dataset"

    record = _publication_bundle_record()
    view = MeasurementScoreView.from_dataset(
        id=ProtocolIdentifier.parse("views.measurement-scores.boolean@0.1.0"),
        dataset=bundle_dataset,
    ).to_record()
    view["source_dataset_digest"] = str(ContentDigest.from_value({"other": True}))
    record["measurement_score_view"] = view

    assert str(
        capture_publication_error(lambda: SubmissionPublicationBundle.from_record(record))
    ) == "source_dataset_digest does not match dataset"


def test_submission_publication_rejects_optional_artifact_mismatches() -> None:
    record = _publication_bundle_record()
    proposal = _proposal_set_record(dataset=_dataset_document().dataset)
    proposal["source_dataset_digest"] = str(ContentDigest.from_value({"other": True}))
    record["proposal_set"] = proposal

    assert str(
        capture_publication_error(lambda: SubmissionPublicationBundle.from_record(record))
    ) == "source_dataset_digest does not match dataset"

    record = _publication_bundle_record()
    surrogate = _surrogate_record(dataset=_dataset_document().dataset)
    surrogate["source_dataset_digest"] = str(ContentDigest.from_value({"other": True}))
    record["architecture_surrogate"] = surrogate

    assert str(
        capture_publication_error(lambda: SubmissionPublicationBundle.from_record(record))
    ) == "source_dataset_digest does not match dataset"


def test_submission_publication_rejects_nonlocal_publication_fields() -> None:
    record = _publication_bundle_record()
    record["target_repo"] = "operator/submissions"

    assert str(
        capture_publication_error(lambda: SubmissionPublicationBundle.from_record(record))
    ) == "target_repo: unknown field"


def test_submission_publication_document_rejects_invalid_bytes() -> None:
    assert str(
        capture_publication_error(lambda: SubmissionPublicationDocument.from_bytes(b"[]"))
    ) == "submission publication document must contain an object"


def _publication_bundle_record() -> dict[str, object]:
    dataset = _dataset_document().dataset
    return {
        "id": "publication-bundles.boolean@0.1.0",
        "submission_package": _submission_package_record(),
        "measurement_dataset": dataset.to_record(),
        "measurement_score_view": _measurement_score_view(dataset=dataset).to_record(),
        "proposal_set": _proposal_set_record(dataset=dataset),
        "architecture_surrogate": _surrogate_record(dataset=dataset),
    }


def _submission_package_record() -> dict[str, object]:
    return {
        "id": "submissions.boolean-digits-pool@0.1.0",
        "benchmark_manifest": _benchmark_document().manifest.to_record(),
        "architecture_manifest": _architecture_document().manifest.to_record(),
        "measurement_dataset": _dataset_document().dataset.to_record(),
        "artifacts": [
            {
                "id": "artifacts.model-weights@0.1.0",
                "digest": str(_architecture_document().digest),
                "description": "checkpoint metadata only",
            }
        ],
    }


def _proposal_set_record(*, dataset: MeasurementDataset) -> dict[str, object]:
    architecture = _architecture_document().manifest
    submission = _submission_package()
    return {
        "id": "experiment-proposal-sets.boolean@0.1.0",
        "source_dataset_digest": str(dataset.digest),
        "proposals": [
            {
                "id": "experiment-proposals.boolean.rank-1@0.1.0",
                "rank": 1,
                "candidate_kind": "architecture",
                "candidate_id": str(architecture.id),
                "rationale": "first unmeasured architecture declaration",
            },
            {
                "id": "experiment-proposals.boolean.rank-2@0.1.0",
                "rank": 2,
                "candidate_kind": "submission-package",
                "candidate_id": str(submission.id),
                "rationale": "complete submission package declaration",
            },
        ],
    }


def _surrogate_record(*, dataset: MeasurementDataset) -> dict[str, object]:
    return {
        "id": "architecture-surrogates.boolean@0.1.0",
        "source_dataset_digest": str(dataset.digest),
        "model_kind": "neural-empirical",
        "target_name": "negative_log_accepted_mass",
        "features": [
            {
                "name": "layer_count",
                "mean": 2.0,
                "scale": 1.0,
                "sensitivity": 0.1,
            }
        ],
        "training": {
            "status": "fit",
            "observation_count": 1,
        },
        "state": {
            "format": "dense-regressor-summary",
            "input_width": 1,
            "output_width": 1,
            "parameter_count": 3,
            "state_digest": str(ContentDigest.from_value({"weights": [0.2]})),
        },
    }


def _measurement_score_view(*, dataset: MeasurementDataset) -> MeasurementScoreView:
    return MeasurementScoreView.from_dataset(
        id=ProtocolIdentifier.parse("views.measurement-scores.boolean@0.1.0"),
        dataset=dataset,
    )


def _submission_package() -> SubmissionPackageManifest:
    return SubmissionPackageManifest.from_record(_submission_package_record())


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


def _alternate_measurement_record() -> dict[str, object]:
    measurement = _dataset_document().dataset.measurements[0].to_record()
    scoring_evidence = _dict_record(measurement["raw_scoring_evidence"])
    scoring_evidence["id"] = "core.boolean-evidence-alt@0.1.0"
    measurement["raw_scoring_evidence"] = scoring_evidence
    return measurement


def _dict_record(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def capture_publication_error(
    action: Callable[[], object],
) -> SubmissionPublicationValidationError:
    try:
        action()
    except SubmissionPublicationValidationError as error:
        return error
    raise AssertionError("expected SubmissionPublicationValidationError")
