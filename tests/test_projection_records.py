from collections.abc import Mapping

import pytest

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.projection_records import (
    ProjectionRecord,
    ProjectionRecordDocument,
    ProjectionRecordValidationError,
)


def test_projection_record_parses_and_canonicalizes() -> None:
    record = ProjectionRecord.from_record(_projection_record())

    assert record.predicate == "orders_by_negative_log_score"
    assert record.modality == "measurement"
    assert record.status == "validated"
    assert record.assumptions == (
        "The score view was derived from the named measurement dataset.",
    )
    assert record.to_record() == _projection_record()
    assert record.digest == ContentDigest.from_value(record.to_record())


def test_projection_record_document_loads_bytes_with_digest() -> None:
    document = ProjectionRecordDocument.from_bytes(
        canonical_document_bytes(_projection_record())
    )

    assert str(document.record.id) == "projection-records.measurement-ranking@0.1.0"
    assert document.digest == ContentDigest.from_value(document.record.to_record())


def test_projection_record_validates_supplied_references() -> None:
    subject = {
        "id": "views.measurement-scores.boolean@0.1.0",
        "source_dataset_digest": str(ContentDigest.from_value({"measurements": []})),
    }
    metric = {
        "id": "metrics.negative-log-score@0.1.0",
        "name": "negative_log_score",
    }
    dataset = {
        "id": "measurement-datasets.boolean@0.1.0",
        "measurements": list[object](),
    }
    report = {
        "id": "resource-reports.boolean@0.1.0",
        "parameter_count": 1,
    }
    record = ProjectionRecord.from_record(
        {
            **_projection_record(),
            "subject": _reference_for("measurement-score-view", subject).to_record(),
            "object": _reference_for("metric", metric).to_record(),
            "scope": [_reference_for("measurement-dataset", dataset).to_record()],
            "evidence": [_reference_for("resource-report", report).to_record()],
        }
    )

    record.validate_references(
        subject_record=subject,
        object_record=metric,
        scope_records=(dataset,),
        evidence_records=(report,),
    )


def test_projection_record_rejects_reference_mismatches() -> None:
    record = ProjectionRecord.from_record(_projection_record())
    mismatched = {
        "id": "measurement-datasets.other@0.1.0",
        "measurements": list[object](),
    }

    with pytest.raises(ProjectionRecordValidationError, match="scope reference"):
        record.validate_references(scope_records=(mismatched,))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("predicate", "Bad Predicate", "predicate must be"),
        ("scope", [], "scope must contain"),
        ("evidence", [], "evidence must contain"),
        ("modality", "opinion", "unsupported modality"),
        ("status", "reviewing", "unsupported status"),
        ("statement", " ", "statement must be nonempty"),
        ("assumptions", [], "assumptions must contain"),
        ("limitations", [""], "limitations: expected nonempty string"),
    ],
)
def test_projection_record_rejects_invalid_records(
    field: str,
    value: object,
    message: str,
) -> None:
    record = _projection_record()
    record[field] = value

    with pytest.raises(ProjectionRecordValidationError, match=message):
        ProjectionRecord.from_record(record)


def test_projection_record_rejects_duplicate_evidence() -> None:
    record = _projection_record()
    record["evidence"] = [
        _protocol_reference("resource-report", "resource-reports.boolean@0.1.0").to_record(),
        _protocol_reference("resource-report", "resource-reports.boolean@0.1.0").to_record(),
    ]

    with pytest.raises(ProjectionRecordValidationError, match="duplicate evidence"):
        ProjectionRecord.from_record(record)


def test_projection_record_document_rejects_invalid_bytes() -> None:
    with pytest.raises(
        ProjectionRecordValidationError,
        match="projection record document must contain an object",
    ):
        ProjectionRecordDocument.from_bytes(b"[]")


def _projection_record() -> dict[str, object]:
    return {
        "id": "projection-records.measurement-ranking@0.1.0",
        "subject": _protocol_reference(
            "measurement-score-view",
            "views.measurement-scores.boolean@0.1.0",
        ).to_record(),
        "predicate": "orders_by_negative_log_score",
        "object": _protocol_reference(
            "metric",
            "metrics.negative-log-score@0.1.0",
        ).to_record(),
        "scope": [
            _protocol_reference(
                "measurement-dataset",
                "measurement-datasets.boolean@0.1.0",
            ).to_record()
        ],
        "evidence": [
            _protocol_reference(
                "resource-report",
                "resource-reports.boolean@0.1.0",
            ).to_record()
        ],
        "modality": "measurement",
        "status": "validated",
        "statement": "The supplied score view orders measurements by negative log score.",
        "assumptions": [
            "The score view was derived from the named measurement dataset.",
        ],
        "limitations": [
            "This record does not rank models outside the supplied dataset.",
        ],
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
