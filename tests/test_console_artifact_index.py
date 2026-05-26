from pathlib import Path, PurePosixPath
from typing import cast

import pytest

from leibniz.console.artifact_index import (
    ConsoleArtifactIndexBuilder,
    ConsoleArtifactIndexSource,
    ConsoleArtifactIndexValidationError,
)
from leibniz.documents import load_object_document

_repository_root = Path(__file__).parents[1]
_public_fixture_sources = (
    ConsoleArtifactIndexSource(
        kind="architecture-manifest",
        source_path=PurePosixPath("tests/fixtures/architecture/digits_pool/manifest.json"),
    ),
    ConsoleArtifactIndexSource(
        kind="benchmark-manifest",
        source_path=PurePosixPath("tests/fixtures/chess/mate_in_one/manifest.json"),
    ),
    ConsoleArtifactIndexSource(
        kind="measurement",
        source_path=PurePosixPath("tests/fixtures/chess/mate_in_one/measurement.json"),
    ),
    ConsoleArtifactIndexSource(
        kind="benchmark-manifest",
        source_path=PurePosixPath("tests/fixtures/finite_outcome/manifest.json"),
    ),
    ConsoleArtifactIndexSource(
        kind="measurement",
        source_path=PurePosixPath("tests/fixtures/finite_outcome/measurement.json"),
    ),
)


def test_console_artifact_index_is_deterministic() -> None:
    builder = ConsoleArtifactIndexBuilder(_repository_root)

    first = builder.build(_public_fixture_sources)
    second = builder.build(reversed(_public_fixture_sources))

    assert first.to_bytes() == second.to_bytes()


def test_console_artifact_index_validates_public_fixture_documents() -> None:
    index = ConsoleArtifactIndexBuilder(_repository_root).build(_public_fixture_sources)
    record = index.to_record()

    assert record["format"] == "leibniz.console.artifact-index"
    assert record["format_version"] == 1

    artifacts = cast(list[dict[str, object]], record["artifacts"])
    assert isinstance(artifacts, list)
    assert [(artifact["kind"], artifact["source_path"]) for artifact in artifacts] == sorted(
        (artifact["kind"], artifact["source_path"]) for artifact in artifacts
    )
    assert {artifact["validation_status"] for artifact in artifacts} == {"valid"}
    assert {artifact["kind"] for artifact in artifacts} == {
        "architecture-manifest",
        "benchmark-manifest",
        "measurement",
    }
    assert all("reference" in artifact for artifact in artifacts)
    assert all("digest" in artifact for artifact in artifacts)


def test_console_artifact_index_records_available_dependency_references() -> None:
    index = ConsoleArtifactIndexBuilder(_repository_root).build(_public_fixture_sources)
    artifacts = {
        entry["source_path"]: entry
        for entry in cast(
            list[dict[str, object]],
            load_object_document(index.to_bytes(), description="console artifact index")[
                "artifacts"
            ],
        )
    }

    finite_measurement = artifacts["tests/fixtures/finite_outcome/measurement.json"]

    assert finite_measurement["dependencies"] == [
        {
            "kind": "benchmark-manifest",
            "protocol_id": "core.boolean-benchmark@0.1.0",
        }
    ]


def test_console_artifact_index_rejects_missing_files() -> None:
    builder = ConsoleArtifactIndexBuilder(_repository_root)

    with pytest.raises(ConsoleArtifactIndexValidationError, match="does not name a file"):
        builder.build(
            (
                ConsoleArtifactIndexSource(
                    kind="benchmark-manifest",
                    source_path=PurePosixPath("tests/fixtures/missing.json"),
                ),
            )
        )


def test_console_artifact_index_rejects_local_state_paths() -> None:
    with pytest.raises(ConsoleArtifactIndexValidationError, match="local state"):
        ConsoleArtifactIndexSource(
            kind="benchmark-manifest",
            source_path=PurePosixPath(".leibniz/manifest.json"),
        )


def test_console_artifact_index_rejects_unsupported_document_kinds() -> None:
    with pytest.raises(ConsoleArtifactIndexValidationError, match="unsupported document kind"):
        ConsoleArtifactIndexSource(
            kind="private-roadmap",
            source_path=PurePosixPath("README.md"),
        )
