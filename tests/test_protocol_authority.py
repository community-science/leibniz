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


def test_protocol_authority_discovers_committed_benchmark_declarations() -> None:
    artifacts = discover_protocol_artifacts()

    paths = {artifact.path for artifact in artifacts}

    assert paths == {
        "digits/inspection_showcase.json",
        "digits/latent_factors.json",
        "digits/manifest.json",
        "digits/materialization.json",
        "digits/observation_formation.json",
    }
    assert {artifact.validation_status for artifact in artifacts} == {"valid"}
    assert {artifact.kind for artifact in artifacts} == {
        "benchmark-manifest",
        "latent-factor-declaration",
        "materialization-declaration",
        "observation-formation-declaration",
        "observation-showcase",
    }


def test_protocol_authority_routes_and_hashes_manifest_record(tmp_path: Path) -> None:
    record: dict[str, object] = {
        "id": "benchmarks.example@0.1.0",
        "name": "benchmarks.example",
        "outcome_sequence": {
            "atom_count": 2,
            "atom_name": "bit",
            "length_parameter": "L",
        },
        "scale_parameter": {"symbol": "L", "minimum": 1},
    }

    route = route_protocol_record(record)

    assert route.kind == "benchmark-manifest"
    artifact = discover_protocol_artifacts(
        root=_source_tree(tmp_path, {"manifest.json": record})
    )[0]
    assert artifact.protocol_id == "benchmarks.example@0.1.0"
    assert artifact.canonical_sha256 == ContentDigest.from_value(record).hex
    assert len(artifact.source_sha256) == 64


def test_protocol_authority_derives_resolved_reference_edges() -> None:
    index = discover_protocol_authority_index(strict=True)

    references = {
        (
            dependency.source_kind,
            dependency.source_field,
            dependency.target_kind,
            dependency.target_protocol_id,
            dependency.status,
        )
        for dependency in index.dependencies
    }

    assert (
        "benchmark-manifest",
        "latent_factor_declaration",
        "latent-factor-declaration",
        "benchmarks.digits.latent-factors@0.1.0",
        "resolved",
    ) in references
    assert (
        "materialization-declaration",
        "latent_factor_declaration",
        "latent-factor-declaration",
        "benchmarks.digits.latent-factors@0.1.0",
        "resolved",
    ) in references
    assert (
        "observation-showcase",
        "formation_declaration",
        "observation-formation-declaration",
        "benchmarks.digits.observation-formation@0.1.0",
        "resolved",
    ) in references
    assert index.dangling_dependencies == ()
    assert index.to_record()["dangling_dependency_count"] == 0


def test_protocol_authority_reports_dangling_reference_edges(tmp_path: Path) -> None:
    root = _source_tree(
        tmp_path,
        {
            "manifest.json": {
                "id": "benchmarks.example@0.1.0",
                "name": "benchmarks.example",
                "outcome_sequence": {
                    "atom_count": 2,
                    "atom_name": "bit",
                    "length_parameter": "L",
                },
                "scale_parameter": {"symbol": "L", "minimum": 1},
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
