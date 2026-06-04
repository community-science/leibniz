from collections.abc import Mapping
from pathlib import Path

import pytest

from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.protocol_authority import (
    ProtocolAuthorityError,
    discover_protocol_artifacts,
    discover_protocol_authority_index,
    route_protocol_record,
)


def test_protocol_authority_finds_no_committed_benchmark_declarations() -> None:
    artifacts = discover_protocol_artifacts()

    assert artifacts == ()


def test_protocol_authority_routes_and_hashes_manifest_record(tmp_path: Path) -> None:
    record: dict[str, object] = {
        "id": "benchmarks.example@0.1.0",
        "name": "benchmarks.example",
        "outcome_space": {
            "id": "benchmarks.example.outcomes@0.1.0",
            "outcomes": [
                {"id": "bit-0"},
                {"id": "bit-1"},
            ],
        },
    }

    route = route_protocol_record(record)

    assert route.kind == "benchmark-manifest"
    artifact = discover_protocol_artifacts(
        root=_source_tree(tmp_path, {"manifest.json": record})
    )[0]
    assert artifact.protocol_id == "benchmarks.example@0.1.0"
    assert artifact.canonical_sha256 == ContentDigest.from_value(record).hex
    assert len(artifact.source_sha256) == 64


def test_protocol_authority_derives_empty_index_for_python_owned_benchmarks() -> None:
    index = discover_protocol_authority_index(strict=True)

    assert index.artifacts == ()
    assert index.dependencies == ()
    assert index.dangling_dependencies == ()
    assert index.to_record()["dangling_dependency_count"] == 0


def test_protocol_authority_reports_dangling_reference_edges(tmp_path: Path) -> None:
    root = _source_tree(
        tmp_path,
        {
            "manifest.json": {
                "id": "benchmarks.example@0.1.0",
                "name": "benchmarks.example",
                "outcome_space": {
                    "id": "benchmarks.example.outcomes@0.1.0",
                    "outcomes": [
                        {"id": "bit-0"},
                        {"id": "bit-1"},
                    ],
                },
                "latent_factor_declaration": {
                    "kind": "latent-factor-declaration",
                    "protocol_id": "latent-factors.missing@0.1.0",
                },
            }
        }
    )

    index = discover_protocol_authority_index(root=root)

    assert len(index.dangling_dependencies) == 1
    with pytest.raises(ProtocolAuthorityError, match="dangling reference"):
        index.require_valid()


def test_protocol_authority_rejects_source_controlled_state_artifacts(
    tmp_path: Path,
) -> None:
    root = _source_tree(
        tmp_path,
        {
            "training_summary.json": {
                "kind": "training-summary",
                "id": "training.digits.example@0.1.0",
            }
        }
    )

    artifacts = discover_protocol_artifacts(root=root)

    assert artifacts[0].validation_status == "invalid"
    assert "must not be source-controlled" in str(artifacts[0].validation_error)
    with pytest.raises(ProtocolAuthorityError, match="must not be source-controlled"):
        discover_protocol_authority_index(root=root, strict=True)


def _source_tree(root: Path, records: Mapping[str, Mapping[str, object]]) -> Path:
    scan_root = root / "src" / "leibniz" / "benchmarks" / "example"
    scan_root.mkdir(parents=True)
    for name, record in records.items():
        (scan_root / name).write_bytes(canonical_document_bytes(record))
    return root
