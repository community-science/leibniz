from collections.abc import Callable

from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.submission_registries import (
    SubmissionRegistry,
    SubmissionRegistryDocument,
    SubmissionRegistrySource,
    SubmissionRegistryValidationError,
)


def test_submission_registry_parses_normalizes_and_canonicalizes() -> None:
    registry = SubmissionRegistry.from_record(
        {
            "id": "submission-registries.public-sources@0.1.0",
            "sources": [
                {
                    "repository": "Example-Owner/Leibniz-Submissions",
                    "repository_type": "dataset",
                    "enabled": True,
                },
                {
                    "repository": "example-owner/model-zoo",
                    "repository_type": "model",
                    "enabled": False,
                },
            ],
        }
    )

    assert registry == SubmissionRegistry(
        id=ProtocolIdentifier.parse("submission-registries.public-sources@0.1.0"),
        sources=(
            SubmissionRegistrySource(
                repository="example-owner/leibniz-submissions",
                repository_type="dataset",
                enabled=True,
            ),
            SubmissionRegistrySource(
                repository="example-owner/model-zoo",
                repository_type="model",
                enabled=False,
            ),
        ),
    )
    assert registry.to_record() == {
        "id": "submission-registries.public-sources@0.1.0",
        "sources": [
            {
                "repository": "example-owner/leibniz-submissions",
                "repository_type": "dataset",
                "enabled": True,
            },
            {
                "repository": "example-owner/model-zoo",
                "repository_type": "model",
                "enabled": False,
            },
        ],
    }
    assert registry.digest == ContentDigest.from_value(registry.to_record())


def test_submission_registry_document_loads_bytes_with_digest() -> None:
    record = _registry_record()

    document = SubmissionRegistryDocument.from_bytes(canonical_document_bytes(record))

    assert document.registry == SubmissionRegistry.from_record(record)
    assert document.digest == ContentDigest.from_value(document.registry.to_record())


def test_submission_registry_sorts_sources_deterministically() -> None:
    registry = SubmissionRegistry.from_record(
        {
            "id": "submission-registries.public-sources@0.1.0",
            "sources": [
                {
                    "repository": "example-zeta/source",
                    "repository_type": "dataset",
                    "enabled": True,
                },
                {
                    "repository": "example-alpha/source",
                    "repository_type": "space",
                    "enabled": False,
                },
                {
                    "repository": "example-alpha/source",
                    "repository_type": "dataset",
                    "enabled": True,
                },
            ],
        }
    )

    assert registry.to_record()["sources"] == [
        {
            "repository": "example-alpha/source",
            "repository_type": "dataset",
            "enabled": True,
        },
        {
            "repository": "example-alpha/source",
            "repository_type": "space",
            "enabled": False,
        },
        {
            "repository": "example-zeta/source",
            "repository_type": "dataset",
            "enabled": True,
        },
    ]


def test_submission_registry_rejects_duplicates() -> None:
    record = _registry_record()
    record["sources"] = [
        {
            "repository": "Example-Owner/Leibniz-Submissions",
            "repository_type": "dataset",
            "enabled": True,
        },
        {
            "repository": "example-owner/leibniz-submissions",
            "repository_type": "dataset",
            "enabled": False,
        },
    ]

    assert str(capture_registry_error(lambda: SubmissionRegistry.from_record(record))) == (
        "duplicate repository source: example-owner/leibniz-submissions (dataset)"
    )


def test_submission_registry_rejects_malformed_or_local_repositories() -> None:
    assert str(
        capture_registry_error(
            lambda: SubmissionRegistrySource.from_record(
                {
                    "repository": ".leibniz/submissions",
                    "repository_type": "dataset",
                    "enabled": True,
                }
            )
        )
    ) == "repository must not be a local path"

    assert str(
        capture_registry_error(
            lambda: SubmissionRegistrySource.from_record(
                {
                    "repository": "https://example.org/owner/repo",
                    "repository_type": "dataset",
                    "enabled": True,
                }
            )
        )
    ) == "repository must not include a URI scheme"

    assert str(
        capture_registry_error(
            lambda: SubmissionRegistrySource.from_record(
                {
                    "repository": "token@owner/repo",
                    "repository_type": "dataset",
                    "enabled": True,
                }
            )
        )
    ) == "repository must not include credentials"

    assert str(
        capture_registry_error(
            lambda: SubmissionRegistrySource.from_record(
                {
                    "repository": "owner",
                    "repository_type": "dataset",
                    "enabled": True,
                }
            )
        )
    ) == "repository must be owner/name"


def test_submission_registry_rejects_unsupported_fields_and_types() -> None:
    source = {
        "repository": "owner/repo",
        "repository_type": "git",
        "enabled": True,
    }
    assert str(capture_registry_error(lambda: SubmissionRegistrySource.from_record(source))) == (
        "unsupported repository_type: git"
    )

    source = {
        "repository": "owner/repo",
        "repository_type": "dataset",
        "enabled": True,
        "token": "secret",
    }
    assert str(capture_registry_error(lambda: SubmissionRegistrySource.from_record(source))) == (
        "token: unknown field"
    )

    record = _registry_record()
    record["cache_dir"] = ".leibniz/submission-cache"
    assert str(capture_registry_error(lambda: SubmissionRegistry.from_record(record))) == (
        "cache_dir: unknown field"
    )

    record = _registry_record()
    record["id"] = "core.registry@0.1.0"
    assert str(capture_registry_error(lambda: SubmissionRegistry.from_record(record))) == (
        "id must be a valid submission registry id"
    )

    assert str(
        capture_registry_error(lambda: SubmissionRegistryDocument.from_bytes(b"[]"))
    ) == "submission registry document must contain an object"


def _registry_record() -> dict[str, object]:
    return {
        "id": "submission-registries.public-sources@0.1.0",
        "sources": [
            {
                "repository": "example-owner/leibniz-submissions",
                "repository_type": "dataset",
                "enabled": True,
            }
        ],
    }


def capture_registry_error(
    action: Callable[[], object],
) -> SubmissionRegistryValidationError:
    try:
        action()
    except SubmissionRegistryValidationError as error:
        return error
    raise AssertionError("expected SubmissionRegistryValidationError")
