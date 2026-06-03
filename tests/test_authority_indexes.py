import pytest

from leibniz.artifacts import ArtifactIndex, ArtifactReference
from leibniz.authority_indexes import (
    AuthorityDependency,
    AuthorityIndex,
    AuthorityIndexDocument,
    AuthorityIndexValidationEntry,
    AuthorityIndexValidationError,
)
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.projection_records import ProjectionRecord
from leibniz.view_manifests import ViewManifest


def test_authority_index_from_records_reports_dangling_dependencies() -> None:
    artifact_index = ArtifactIndex.from_record(_artifact_index_record())
    projection = ProjectionRecord.from_record(_projection_record())
    view_manifest = ViewManifest.from_record(_view_manifest_record())

    index = AuthorityIndex.from_records(
        id=ProtocolIdentifier.parse("authority-indexes.review@0.1.0"),
        artifact_indexes=(artifact_index,),
        projection_records=(projection,),
        view_manifests=(view_manifest,),
    )

    assert index.id == ProtocolIdentifier.parse("authority-indexes.review@0.1.0")
    assert any(dependency.relation == "evidence" for dependency in index.dependencies)
    assert any(dependency.relation == "source-artifact" for dependency in index.dependencies)
    assert any(entry.status == "valid" for entry in index.validations)
    assert any(entry.status == "dangling" for entry in index.validations)
    assert index.validations == tuple(
        sorted(
            index.validations,
            key=lambda entry: (entry.artifact.kind, entry.status, entry.message),
        )
    )


def test_authority_index_parses_and_canonicalizes() -> None:
    record = _authority_index_record()

    index = AuthorityIndex.from_record(record)

    assert index.to_record() == record
    assert index.digest == ContentDigest.from_value(record)


def test_authority_index_document_loads_bytes_with_digest() -> None:
    document = AuthorityIndexDocument.from_bytes(
        canonical_document_bytes(_authority_index_record())
    )

    assert str(document.index.id) == "authority-indexes.review@0.1.0"
    assert document.digest == ContentDigest.from_value(document.index.to_record())


def test_authority_index_from_artifacts_reports_dangling_dependency_target() -> None:
    source = _protocol_reference("view-manifest", "view-manifests.review@0.1.0")
    target = _protocol_reference("projection-record", "projection-records.review@0.1.0")

    index = AuthorityIndex.from_artifacts(
        id=ProtocolIdentifier.parse("authority-indexes.dangling@0.1.0"),
        artifacts=(source,),
        dependencies=(
            AuthorityDependency(
                source=source,
                target=target,
                relation="claims",
            ),
        ),
    )

    assert index.dangling_references == (target,)
    assert AuthorityIndexValidationEntry(
        artifact=target,
        status="dangling",
        message="dependency endpoint was not supplied as an indexed artifact",
    ) in index.validations


def test_authority_index_rejects_duplicate_artifacts() -> None:
    artifact = _evaluation_bundle_reference()

    with pytest.raises(AuthorityIndexValidationError, match="duplicate artifact"):
        AuthorityIndex.from_artifacts(
            id=ProtocolIdentifier.parse("authority-indexes.duplicates@0.1.0"),
            artifacts=(artifact, artifact),
        )


def test_authority_index_rejects_invalid_dependency_relation() -> None:
    with pytest.raises(AuthorityIndexValidationError, match="relation must be"):
        AuthorityDependency(
            source=_evaluation_bundle_reference(),
            target=_evaluation_bundle_reference(),
            relation="Bad Relation",
        )


def test_authority_index_document_rejects_invalid_bytes() -> None:
    with pytest.raises(
        AuthorityIndexValidationError,
        match="authority index document must contain an object",
    ):
        AuthorityIndexDocument.from_bytes(b"[]")


def _authority_index_record() -> dict[str, object]:
    artifact = _evaluation_bundle_reference().to_record()
    return {
        "id": "authority-indexes.review@0.1.0",
        "artifacts": [artifact],
        "dependencies": [],
        "validations": [
            {
                "artifact": artifact,
                "status": "valid",
                "message": "artifact reference was supplied explicitly",
            }
        ],
    }


def _artifact_index_record() -> dict[str, object]:
    return {
        "id": "artifact-indexes.review@0.1.0",
        "artifacts": [_evaluation_bundle_reference().to_record()],
    }


def _projection_record() -> dict[str, object]:
    return {
        "id": "projection-records.review@0.1.0",
        "subject": _protocol_reference(
            "view-manifest",
            "view-manifests.review@0.1.0",
        ).to_record(),
        "predicate": "declares_projection",
        "object": _protocol_reference("metric", "metrics.review@0.1.0").to_record(),
        "scope": [_evaluation_bundle_reference().to_record()],
        "evidence": [
            _protocol_reference(
                "resource-report",
                "resource-reports.review@0.1.0",
            ).to_record()
        ],
        "modality": "measurement",
        "status": "proposed",
        "statement": "The referenced view manifest declares the reviewed projection.",
        "assumptions": ["The referenced artifacts validate independently."],
        "limitations": ["No verifier is executed by this record."],
    }


def _view_manifest_record() -> dict[str, object]:
    return {
        "id": "view-manifests.review@0.1.0",
        "subject_kind": "evaluation-bundle",
        "subject": _evaluation_bundle_reference().to_record(),
        "projection_kind": "summary",
        "source_artifacts": [_evaluation_bundle_reference().to_record()],
        "metric_name": "accepted_mass",
        "score_direction": "higher",
    }


def _evaluation_bundle_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "evaluation-bundle",
            "content_digest": str(ContentDigest.from_value({"evaluation": "bundle"})),
        }
    )


def _protocol_reference(kind: str, protocol_id: str) -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": kind,
            "protocol_id": protocol_id,
        }
    )
