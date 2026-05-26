from collections.abc import Callable
from pathlib import Path

from leibniz._documents import canonical_document_bytes
from leibniz.architectures import ArchitectureManifestDocument
from leibniz.artifacts import (
    ArtifactReference,
    ArtifactReferenceDocument,
    ArtifactReferenceValidationError,
    reference_for_record,
)
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier

_fixtures_root = Path(__file__).parent / "fixtures"


def test_artifact_reference_parses_and_canonicalizes() -> None:
    architecture_record = _architecture_record()
    reference = ArtifactReference.from_record(
        {
            "kind": "architecture-manifest",
            "protocol_id": architecture_record["id"],
            "content_digest": str(ContentDigest.from_value({"weights": [1, 2, 3]})),
            "record_digest": str(ContentDigest.from_value(architecture_record)),
            "external_uri": "https://example.org/artifacts/digits-pool.json",
        }
    )

    assert reference == ArtifactReference(
        kind="architecture-manifest",
        protocol_id=ProtocolIdentifier.parse(str(architecture_record["id"])),
        content_digest=ContentDigest.from_value({"weights": [1, 2, 3]}),
        record_digest=ContentDigest.from_value(architecture_record),
        external_uri="https://example.org/artifacts/digits-pool.json",
    )
    assert reference.to_record() == {
        "kind": "architecture-manifest",
        "protocol_id": architecture_record["id"],
        "content_digest": str(ContentDigest.from_value({"weights": [1, 2, 3]})),
        "record_digest": str(ContentDigest.from_value(architecture_record)),
        "external_uri": "https://example.org/artifacts/digits-pool.json",
    }


def test_artifact_reference_accepts_each_durable_identity() -> None:
    architecture_record = _architecture_record()

    assert ArtifactReference.from_record(
        {
            "kind": "architecture-manifest",
            "protocol_id": architecture_record["id"],
        }
    ).protocol_id == ProtocolIdentifier.parse(str(architecture_record["id"]))
    assert ArtifactReference.from_record(
        {
            "kind": "architecture-manifest",
            "content_digest": str(ContentDigest.from_value({"payload": True})),
        }
    ).content_digest == ContentDigest.from_value({"payload": True})
    assert ArtifactReference.from_record(
        {
            "kind": "architecture-manifest",
            "record_digest": str(ContentDigest.from_value(architecture_record)),
        }
    ).matches_record(architecture_record)
    assert ArtifactReference.from_record(
        {
            "kind": "architecture-manifest",
            "external_uri": "urn:leibniz:architecture:digits-pool",
        }
    ).external_uri == "urn:leibniz:architecture:digits-pool"


def test_artifact_reference_document_loads_bytes_with_digest() -> None:
    record = {
        "kind": "architecture-manifest",
        "record_digest": str(ContentDigest.from_value(_architecture_record())),
    }

    document = ArtifactReferenceDocument.from_bytes(canonical_document_bytes(record))

    assert document.reference == ArtifactReference.from_record(record)
    assert document.digest == ContentDigest.from_value(document.reference.to_record())


def test_reference_for_record_summarizes_embedded_public_record() -> None:
    architecture_record = _architecture_record()

    reference = reference_for_record(
        kind="architecture-manifest",
        record=architecture_record,
    )

    assert reference.to_record() == {
        "kind": "architecture-manifest",
        "protocol_id": architecture_record["id"],
        "record_digest": str(ContentDigest.from_value(architecture_record)),
    }
    assert reference.matches_record(architecture_record)

    altered_record = dict(architecture_record)
    altered_record["layers"] = []
    assert not reference.matches_record(altered_record)


def test_artifact_reference_rejects_non_durable_identity() -> None:
    assert str(
        capture_artifact_error(
            lambda: ArtifactReference.from_record({"kind": "architecture-manifest"})
        )
    ) == "artifact reference must include at least one durable identity"

    assert str(
        capture_artifact_error(
            lambda: ArtifactReference.from_record(
                {
                    "kind": "architecture-manifest",
                    "record_digest": "sha256:not-a-digest",
                }
            )
        )
    ) == "sha256 digest must be 64 lowercase hexadecimal characters"

    assert str(
        capture_artifact_error(
            lambda: ArtifactReference.from_record(
                {
                    "kind": "architecture-manifest",
                    "external_uri": ".leibniz/cache/model.json",
                }
            )
        )
    ) == "external_uri must not be a local path"

    assert str(
        capture_artifact_error(
            lambda: ArtifactReference.from_record(
                {
                    "kind": "architecture-manifest",
                    "external_uri": "file:///tmp/model.json",
                }
            )
        )
    ) == "external_uri must not use file URI scheme"


def test_artifact_reference_rejects_malformed_fields() -> None:
    assert str(
        capture_artifact_error(
            lambda: ArtifactReference.from_record(
                {
                    "kind": "ArchitectureManifest",
                    "record_digest": str(ContentDigest.from_value(_architecture_record())),
                }
            )
        )
    ) == "kind must be a stable lowercase artifact kind"

    assert str(
        capture_artifact_error(
            lambda: ArtifactReference.from_record(
                {
                    "kind": "architecture-manifest",
                    "protocol_id": "architectures.digits-pool@1.0.0",
                }
            )
        )
    ) == (
        "identifier must use a pre-1.0.0 version before release policy exists: "
        "architectures.digits-pool@1.0.0"
    )

    assert str(
        capture_artifact_error(
            lambda: ArtifactReference.from_record(
                {
                    "kind": "architecture-manifest",
                    "record_digest": str(ContentDigest.from_value(_architecture_record())),
                    "path": ".leibniz/cache/model.json",
                }
            )
        )
    ) == "path: unknown field"


def test_reference_for_record_rejects_malformed_embedded_id() -> None:
    architecture_record = _architecture_record()
    architecture_record["id"] = "not a valid id"

    assert str(
        capture_artifact_error(
            lambda: reference_for_record(
                kind="architecture-manifest",
                record=architecture_record,
            )
        )
    ) == "invalid protocol identifier: 'not a valid id'"


def test_artifact_reference_document_rejects_invalid_bytes() -> None:
    assert str(
        capture_artifact_error(lambda: ArtifactReferenceDocument.from_bytes(b"[]"))
    ) == "artifact reference document must contain an object"


def _architecture_record() -> dict[str, object]:
    return ArchitectureManifestDocument.from_bytes(
        (_fixtures_root / "architecture" / "digits_pool" / "manifest.json").read_bytes()
    ).manifest.to_record()


def capture_artifact_error(
    action: Callable[[], object],
) -> ArtifactReferenceValidationError:
    try:
        action()
    except ArtifactReferenceValidationError as error:
        return error
    raise AssertionError("expected ArtifactReferenceValidationError")
