from collections.abc import Callable
from pathlib import Path

from leibniz.architectures import ArchitectureManifestDocument
from leibniz.artifacts import (
    ArtifactIndex,
    ArtifactIndexDocument,
    ArtifactReference,
    ArtifactReferenceDocument,
    ArtifactReferenceValidationError,
    reference_for_record,
)
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
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
                    "external_uri": "./results/cache/model.json",
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
                    "path": "./results/cache/model.json",
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


def test_artifact_index_canonicalizes_explicit_references() -> None:
    architecture = reference_for_record(
        kind="architecture-manifest",
        record=_architecture_record(),
    )
    weights = ArtifactReference.from_record(
        {
            "kind": "model-weights",
            "content_digest": str(ContentDigest.from_value({"weights": [1, 2, 3]})),
        }
    )

    index = ArtifactIndex.from_record(
        {
            "id": "artifact-indexes.boolean-evaluation@0.1.0",
            "artifacts": [weights.to_record(), architecture.to_record()],
        }
    )

    assert index == ArtifactIndex(
        id=ProtocolIdentifier.parse("artifact-indexes.boolean-evaluation@0.1.0"),
        artifacts=(architecture, weights),
    )
    assert index.to_record() == {
        "id": "artifact-indexes.boolean-evaluation@0.1.0",
        "artifacts": [architecture.to_record(), weights.to_record()],
    }
    assert index.digest == ContentDigest.from_value(index.to_record())


def test_artifact_index_records_equivalent_orderings_with_same_digest() -> None:
    references = _index_references()
    forward = ArtifactIndex.from_record(
        {
            "id": "artifact-indexes.boolean-evaluation@0.1.0",
            "artifacts": [reference.to_record() for reference in references],
        }
    )
    reverse = ArtifactIndex.from_record(
        {
            "id": "artifact-indexes.boolean-evaluation@0.1.0",
            "artifacts": [reference.to_record() for reference in reversed(references)],
        }
    )

    assert forward.to_record() == reverse.to_record()
    assert forward.digest == reverse.digest


def test_artifact_index_validates_source_digest_when_source_is_supplied() -> None:
    source_record = {"evaluation": True, "version": 1}
    index = ArtifactIndex.from_source_record(
        id=ProtocolIdentifier.parse("artifact-indexes.boolean-evaluation@0.1.0"),
        source_kind="evaluation-bundle",
        source_record=source_record,
        artifacts=_index_references(),
    )

    parsed = ArtifactIndex.from_record(index.to_record(), source_record=source_record)
    document = ArtifactIndexDocument.from_bytes(
        canonical_document_bytes(index.to_record()),
        source_record=source_record,
    )

    assert parsed == index
    assert document.index == index
    assert document.digest == ContentDigest.from_value(index.to_record())
    assert index.to_record()["source_digest"] == str(ContentDigest.from_value(source_record))

    altered_source = {"evaluation": True, "version": 2}
    assert str(
        capture_artifact_error(
            lambda: ArtifactIndex.from_record(index.to_record(), source_record=altered_source)
        )
    ) == "source_digest does not match source record"


def test_artifact_index_rejects_duplicate_and_malformed_records() -> None:
    reference = reference_for_record(
        kind="architecture-manifest",
        record=_architecture_record(),
    )

    assert str(
        capture_artifact_error(
            lambda: ArtifactIndex.from_record(
                {
                    "id": "artifact-indexes.boolean-evaluation@0.1.0",
                    "artifacts": [reference.to_record(), reference.to_record()],
                }
            )
        )
    ).startswith("duplicate artifact reference: sha256:")

    assert str(
        capture_artifact_error(
            lambda: ArtifactIndex.from_record(
                {
                    "id": "artifact-indexes.boolean-evaluation@0.1.0",
                    "artifacts": [],
                }
            )
        )
    ) == "artifacts must contain at least one artifact reference"

    assert str(
        capture_artifact_error(
            lambda: ArtifactIndex.from_record(
                {
                    "id": "core.boolean-index@0.1.0",
                    "artifacts": [reference.to_record()],
                }
            )
        )
    ) == "id must be a valid artifact index id"

    assert str(
        capture_artifact_error(
            lambda: ArtifactIndex.from_record(
                {
                    "id": "artifact-indexes.boolean-evaluation@0.1.0",
                    "source_digest": str(ContentDigest.from_value({"source": True})),
                    "artifacts": [reference.to_record()],
                }
            )
        )
    ) == "source_kind and source_digest must be supplied together"

    assert str(
        capture_artifact_error(
            lambda: ArtifactIndex.from_record(
                {
                    "id": "artifact-indexes.boolean-evaluation@0.1.0",
                    "source_kind": "evaluation-bundle",
                    "artifacts": [reference.to_record()],
                }
            )
        )
    ) == "source_kind and source_digest must be supplied together"

    assert str(
        capture_artifact_error(lambda: ArtifactIndexDocument.from_bytes(b"[]"))
    ) == "artifact index document must contain an object"


def _architecture_record() -> dict[str, object]:
    return ArchitectureManifestDocument.from_bytes(
        (_fixtures_root / "architecture" / "digits_pool.json").read_bytes()
    ).manifest.to_record()


def _index_references() -> tuple[ArtifactReference, ArtifactReference]:
    return (
        reference_for_record(
            kind="architecture-manifest",
            record=_architecture_record(),
        ),
        ArtifactReference.from_record(
            {
                "kind": "model-weights",
                "content_digest": str(ContentDigest.from_value({"weights": [1, 2, 3]})),
            }
        ),
    )


def capture_artifact_error(
    action: Callable[[], object],
) -> ArtifactReferenceValidationError:
    try:
        action()
    except ArtifactReferenceValidationError as error:
        return error
    raise AssertionError("expected ArtifactReferenceValidationError")
