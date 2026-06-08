from collections.abc import Mapping

import pytest

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.view_manifests import (
    ViewManifest,
    ViewManifestDocument,
    ViewManifestValidationError,
)


def test_view_manifest_parses_and_canonicalizes_measurement_score_view() -> None:
    manifest = ViewManifest.from_record(_measurement_score_view_manifest_record())

    assert manifest.subject_kind == "measurement-score-view"
    assert manifest.projection_kind == "ranking"
    assert manifest.metric_name == "negative_log_score"
    assert manifest.score_direction == "lower"
    assert manifest.to_record() == _measurement_score_view_manifest_record()
    assert manifest.digest == ContentDigest.from_value(manifest.to_record())


def test_view_manifest_document_loads_bytes_with_digest() -> None:
    document = ViewManifestDocument.from_bytes(
        canonical_document_bytes(_measurement_score_view_manifest_record())
    )

    assert str(document.manifest.id.name) == "view-manifests.measurement-scores"
    assert document.digest == ContentDigest.from_value(document.manifest.to_record())


def test_view_manifest_validates_subject_and_sources() -> None:
    subject_record: dict[str, object] = {
        "id": "views.measurement-scores.boolean@0.1.0",
        "source_dataset_digest": str(ContentDigest.from_value({"measurements": []})),
    }
    source_record: dict[str, object] = {
        "id": "measurement-datasets.boolean@0.1.0",
        "measurements": list[object](),
    }
    manifest = ViewManifest.from_record(
        {
            **_measurement_score_view_manifest_record(),
            "subject": _reference_for("measurement-score-view", subject_record).to_record(),
            "source_artifacts": [
                _reference_for("measurement-dataset", source_record).to_record()
            ],
        }
    )

    manifest.validate_subject(subject_record)
    manifest.validate_source_artifacts((source_record,))


def test_view_manifest_parses_evaluation_bundle_subject() -> None:
    evaluation = ViewManifest.from_record(
        {
            "id": "view-manifests.evaluations@0.1.0",
            "subject_kind": "evaluation-bundle",
            "subject": _protocol_reference(
                "evaluation-bundle",
                "benchmark-evaluations.boolean@0.1.0",
            ).to_record(),
            "projection_kind": "summary",
            "source_artifacts": [
                _protocol_reference(
                    "evaluation-bundle",
                    "benchmark-evaluations.boolean@0.1.0",
                ).to_record()
            ],
            "metric_name": "accepted_mass",
            "score_direction": "higher",
        }
    )

    assert evaluation.subject_kind == "evaluation-bundle"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("subject_kind", "thing", "unsupported subject_kind"),
        ("projection_kind", "layout", "unsupported projection_kind"),
        ("metric_name", "Not A Metric", "metric_name must be"),
        ("score_direction", "middle", "unsupported score_direction"),
        ("source_artifacts", [], "source_artifacts must contain"),
    ],
)
def test_view_manifest_rejects_invalid_records(
    field: str,
    value: object,
    message: str,
) -> None:
    record = _measurement_score_view_manifest_record()
    record[field] = value

    with pytest.raises(ViewManifestValidationError, match=message):
        ViewManifest.from_record(record)


def test_view_manifest_rejects_subject_kind_mismatch() -> None:
    record = _measurement_score_view_manifest_record()
    record["subject"] = _protocol_reference(
        "evaluation-bundle",
        "benchmark-evaluations.boolean@0.1.0",
    ).to_record()

    with pytest.raises(ViewManifestValidationError, match="subject kind must match"):
        ViewManifest.from_record(record)


def test_view_manifest_document_rejects_invalid_bytes() -> None:
    with pytest.raises(
        ViewManifestValidationError,
        match="view manifest document must contain an object",
    ):
        ViewManifestDocument.from_bytes(b"[]")


def _measurement_score_view_manifest_record() -> dict[str, object]:
    return {
        "id": "view-manifests.measurement-scores@0.1.0",
        "subject_kind": "measurement-score-view",
        "subject": _protocol_reference(
            "measurement-score-view",
            "views.measurement-scores.boolean@0.1.0",
        ).to_record(),
        "projection_kind": "ranking",
        "source_artifacts": [
            _protocol_reference(
                "measurement-dataset",
                "measurement-datasets.boolean@0.1.0",
            ).to_record()
        ],
        "metric_name": "negative_log_score",
        "score_direction": "lower",
    }


def _protocol_reference(kind: str, protocol_id: str) -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": kind,
            "protocol_id": protocol_id,
        }
    )


def _reference_for(kind: str, record: Mapping[str, object]) -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": kind,
            "protocol_id": record["id"],
            "record_digest": str(ContentDigest.from_value(record)),
        }
    )
