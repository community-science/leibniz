from pathlib import Path, PurePosixPath
from typing import cast

import pytest

from leibniz.console.data import ConsoleDataBuilder, ConsoleDataValidationError
from leibniz.documents import load_object_document

_repository_root = Path(__file__).parents[1]


def test_console_data_discovery_is_deterministic() -> None:
    builder = ConsoleDataBuilder(_repository_root)

    first = builder.discover((PurePosixPath("tests/fixtures"),))
    second = builder.discover(
        (
            PurePosixPath("tests/fixtures/finite_outcome"),
            PurePosixPath("tests/fixtures/chess"),
            PurePosixPath("tests/fixtures/architecture"),
        )
    )

    assert first.to_bytes() == second.to_bytes()


def test_console_data_discovers_supported_public_fixture_documents() -> None:
    data = ConsoleDataBuilder(_repository_root).discover((PurePosixPath("tests/fixtures"),))
    record = data.to_record()

    assert record["format"] == "leibniz.console-data"
    assert record["format_version"] == 1

    artifact_index = cast(dict[str, object], record["artifact_index"])
    artifacts = cast(list[dict[str, object]], artifact_index["artifacts"])
    details = cast(list[dict[str, object]], record["artifact_details"])

    assert [(artifact["kind"], artifact["source_path"]) for artifact in artifacts] == [
        ("architecture-manifest", "tests/fixtures/architecture/digits_pool/manifest.json"),
        ("benchmark-manifest", "tests/fixtures/chess/mate_in_one/manifest.json"),
        ("benchmark-manifest", "tests/fixtures/finite_outcome/manifest.json"),
        ("measurement", "tests/fixtures/chess/mate_in_one/measurement.json"),
        ("measurement", "tests/fixtures/finite_outcome/measurement.json"),
    ]
    assert [(detail["kind"], detail["source_path"]) for detail in details] == [
        (artifact["kind"], artifact["source_path"]) for artifact in artifacts
    ]
    assert {artifact["validation_status"] for artifact in artifacts} == {"valid"}


def test_console_data_payload_is_a_canonical_object_document() -> None:
    data = ConsoleDataBuilder(_repository_root).discover((PurePosixPath("tests/fixtures"),))

    record = load_object_document(data.to_bytes(), description="console data")

    assert record["format"] == "leibniz.console-data"


def test_console_data_rejects_local_state_roots() -> None:
    with pytest.raises(ConsoleDataValidationError, match="local state"):
        ConsoleDataBuilder(_repository_root).discover((PurePosixPath(".leibniz"),))


def test_console_data_rejects_missing_public_roots() -> None:
    with pytest.raises(ConsoleDataValidationError, match="does not name a directory"):
        ConsoleDataBuilder(_repository_root).discover((PurePosixPath("tests/missing"),))


def test_console_data_rejects_roots_without_supported_documents(tmp_path: Path) -> None:
    (tmp_path / "README.txt").write_text("not a supported document", encoding="utf-8")

    with pytest.raises(ConsoleDataValidationError, match="did not contain supported documents"):
        ConsoleDataBuilder(tmp_path).discover((PurePosixPath("."),))
