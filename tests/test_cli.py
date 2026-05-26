from pathlib import Path

import pytest

from leibniz.artifacts import ArtifactReference, reference_for_record
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.cli import main
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.federation_ingest import plan_federation_ingest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDocument
from leibniz.model_operations import ModelOperation
from leibniz.submission_registries import SubmissionRegistry

_fixtures_root = Path(__file__).parent / "fixtures"
_finite_fixture = _fixtures_root / "finite_outcome"


def test_cli_help_text(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["validate", "--help"])

    assert exit_info.value.code == 0
    assert "validate artifact files" in capsys.readouterr().out


def test_cli_validate_help_lists_expanded_artifacts(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["validate", "--help"])

    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert "artifact-reference" in output
    assert "model-manifest" in output
    assert "model-lineage" in output
    assert "submission-registry" in output
    assert "federation-ingest-plan" in output


def test_cli_validates_manifest_file(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["validate", "manifest", str(_finite_fixture / "manifest.json")])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "valid manifest core.boolean-benchmark@0.1.0\n"
    assert captured.err == ""


def test_cli_validates_measurement_file(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["validate", "measurement", str(_finite_fixture / "measurement.json")])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "valid measurement core.boolean-evidence@0.1.0\n"
    assert captured.err == ""


def test_cli_validates_measurement_file_against_manifest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "validate",
            "measurement",
            str(_finite_fixture / "measurement.json"),
            "--manifest",
            str(_finite_fixture / "manifest.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "valid measurement core.boolean-evidence@0.1.0\n"
    assert captured.err == ""


def test_cli_validates_dataset_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = _dataset_path(tmp_path)

    exit_code = main(["validate", "dataset", str(dataset_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "valid measurement dataset (1 measurements)\n"
    assert captured.err == ""


def test_cli_validates_dataset_file_against_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = _dataset_path(tmp_path)

    exit_code = main(
        [
            "validate",
            "dataset",
            str(dataset_path),
            "--manifest",
            str(_finite_fixture / "manifest.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "valid measurement dataset (1 measurements)\n"
    assert captured.err == ""


def test_cli_reports_malformed_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("[]", encoding="utf-8")

    exit_code = main(["validate", "manifest", str(manifest_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "error: manifest document must contain an object\n"


def test_cli_reports_incompatible_manifest_pair(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = BenchmarkManifestDocument.from_bytes(
        (_finite_fixture / "manifest.json").read_bytes()
    ).manifest.to_record()
    manifest["id"] = "core.other-benchmark@0.1.0"
    manifest["name"] = "core.other-benchmark"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_document_bytes(manifest))

    exit_code = main(
        [
            "validate",
            "measurement",
            str(_finite_fixture / "measurement.json"),
            "--manifest",
            str(manifest_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "error: benchmark_id core.boolean-benchmark@0.1.0 does not match manifest "
        "core.other-benchmark@0.1.0\n"
    )


@pytest.mark.parametrize(
    ("artifact", "factory_name", "expected"),
    [
        (
            "artifact-reference",
            "_artifact_reference_record",
            "valid artifact reference sha256:",
        ),
        (
            "artifact-index",
            "_artifact_index_record",
            "valid artifact index artifact-indexes.cli@0.1.0",
        ),
        (
            "resource-report",
            "_resource_report_record",
            "valid resource report resource-reports.cli@0.1.0",
        ),
        (
            "resource-report-set",
            "_resource_report_set_record",
            "valid resource report set resource-report-sets.cli@0.1.0",
        ),
        (
            "model-manifest",
            "_model_manifest_record",
            "valid model manifest model-manifests.cli@0.1.0",
        ),
        (
            "model-operation",
            "_model_operation_record",
            "valid model operation model-operations.sha-",
        ),
        (
            "model-lineage",
            "_model_lineage_record",
            "valid model lineage model-lineages.cli@0.1.0",
        ),
        (
            "model-derivation",
            "_model_derivation_record",
            "valid model derivation compatibility report model-derivations.cli@0.1.0",
        ),
        (
            "submission-registry",
            "_submission_registry_record",
            "valid submission registry submission-registries.cli@0.1.0",
        ),
    ],
)
def test_cli_validates_expanded_artifact_files(
    artifact: str,
    factory_name: str,
    expected: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / f"{artifact}.json"
    record = _expanded_artifact_record(factory_name)
    path.write_bytes(canonical_document_bytes(record))

    exit_code = main(["validate", artifact, str(path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith(expected)
    assert captured.err == ""


def test_cli_validates_model_interface_with_outcome_space(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    interface_path = tmp_path / "interface.json"
    outcome_space_path = tmp_path / "outcome-space.json"
    interface_path.write_bytes(canonical_document_bytes(_model_interface_record()))
    outcome_space_path.write_bytes(canonical_document_bytes(_outcome_space_record()))

    exit_code = main(
        [
            "validate",
            "model-interface",
            str(interface_path),
            "--outcome-space",
            str(outcome_space_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "valid model interface model-interfaces.cli@0.1.0\n"
    assert captured.err == ""


def test_cli_validates_federation_ingest_plan_with_registry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = SubmissionRegistry.from_record(_submission_registry_record())
    plan = plan_federation_ingest(
        id=ProtocolIdentifier.parse("federation-ingest-plans.cli@0.1.0"),
        registry=registry,
        expected_publication_bundles=(_publication_bundle_reference(),),
    )
    registry_path = tmp_path / "registry.json"
    plan_path = tmp_path / "plan.json"
    registry_path.write_bytes(canonical_document_bytes(registry.to_record()))
    plan_path.write_bytes(canonical_document_bytes(plan.to_record()))

    exit_code = main(
        [
            "validate",
            "federation-ingest-plan",
            str(plan_path),
            "--registry",
            str(registry_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "valid federation ingest plan federation-ingest-plans.cli@0.1.0\n"
    assert captured.err == ""


@pytest.mark.parametrize(
    "artifact",
    [
        "artifact-reference",
        "artifact-index",
        "resource-report",
        "resource-report-set",
        "model-manifest",
        "model-operation",
        "model-lineage",
        "model-derivation",
        "publication-bundle",
        "submission-registry",
        "federation-ingest-plan",
    ],
)
def test_cli_reports_malformed_expanded_artifact_files(
    artifact: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / f"{artifact}.json"
    path.write_text("[]", encoding="utf-8")

    exit_code = main(["validate", artifact, str(path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err.startswith("error: ")


def test_cli_reports_model_interface_pairing_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    interface_path = tmp_path / "interface.json"
    outcome_space_path = tmp_path / "outcome-space.json"
    interface_path.write_bytes(canonical_document_bytes(_model_interface_record()))
    outcome_space = _outcome_space_record()
    outcome_space["id"] = "core.other-outcome@0.1.0"
    outcome_space_path.write_bytes(canonical_document_bytes(outcome_space))

    exit_code = main(
        [
            "validate",
            "model-interface",
            str(interface_path),
            "--outcome-space",
            str(outcome_space_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "error: outcome_space_id core.boolean-outcome@0.1.0 does not match "
        "core.other-outcome@0.1.0\n"
    )


def _dataset_path(tmp_path: Path) -> Path:
    measurement = MeasurementDocument.from_bytes(
        (_finite_fixture / "measurement.json").read_bytes()
    ).measurement.to_record()
    dataset_path = tmp_path / "measurements.json"
    dataset_path.write_bytes(canonical_document_bytes({"measurements": [measurement]}))
    return dataset_path


def _artifact_reference_record() -> dict[str, object]:
    return _publication_bundle_reference().to_record()


def _expanded_artifact_record(factory_name: str) -> dict[str, object]:
    if factory_name == "_artifact_reference_record":
        return _artifact_reference_record()
    if factory_name == "_artifact_index_record":
        return _artifact_index_record()
    if factory_name == "_resource_report_record":
        return _resource_report_record()
    if factory_name == "_resource_report_set_record":
        return _resource_report_set_record()
    if factory_name == "_model_manifest_record":
        return _model_manifest_record()
    if factory_name == "_model_operation_record":
        return _model_operation_record()
    if factory_name == "_model_lineage_record":
        return _model_lineage_record()
    if factory_name == "_model_derivation_record":
        return _model_derivation_record()
    if factory_name == "_submission_registry_record":
        return _submission_registry_record()
    raise ValueError(f"unsupported expanded artifact factory {factory_name!r}")


def _artifact_index_record() -> dict[str, object]:
    return {
        "id": "artifact-indexes.cli@0.1.0",
        "artifacts": [_publication_bundle_reference().to_record()],
    }


def _resource_report_record() -> dict[str, object]:
    return {
        "id": "resource-reports.cli@0.1.0",
        "artifact": _model_manifest_reference().to_record(),
        "parameter_count": 1,
        "parameter_bits": 8,
        "parameter_bytes": 1,
    }


def _resource_report_set_record() -> dict[str, object]:
    return {
        "id": "resource-report-sets.cli@0.1.0",
        "reports": [_resource_report_record()],
    }


def _model_interface_record() -> dict[str, object]:
    return {
        "id": "model-interfaces.cli@0.1.0",
        "outcome_space_id": "core.boolean-outcome@0.1.0",
        "prediction_semantics": "finite-probability-measure",
        "output_encoding": "probability-mass-sequence",
    }


def _model_manifest_record() -> dict[str, object]:
    return {
        "id": "model-manifests.cli@0.1.0",
        "architecture": _architecture_reference().to_record(),
        "interface": reference_for_record(
            kind="model-interface",
            record=_model_interface_record(),
        ).to_record(),
        "model_artifacts": [
            {
                "kind": "model-checkpoint",
                "content_digest": str(ContentDigest.from_value({"weights": [1]})),
            }
        ],
    }


def _model_operation_record() -> dict[str, object]:
    return ModelOperation.from_record(
        {
            "operator_id": "model-operators.train@0.1.0",
            "inputs": [
                {
                    "role": "architecture",
                    "artifact": _architecture_reference().to_record(),
                }
            ],
            "outputs": [
                {
                    "role": "model",
                    "artifact": _model_manifest_reference().to_record(),
                }
            ],
        }
    ).to_record()


def _model_lineage_record() -> dict[str, object]:
    return {
        "id": "model-lineages.cli@0.1.0",
        "artifacts": [
            _architecture_reference().to_record(),
            _model_manifest_reference().to_record(),
        ],
        "operations": [_model_operation_record()],
    }


def _model_derivation_record() -> dict[str, object]:
    return {
        "id": "model-derivations.cli@0.1.0",
        "source_model": _model_manifest_reference().to_record(),
        "target_architecture": _architecture_reference().to_record(),
        "target_interface": reference_for_record(
            kind="model-interface",
            record=_model_interface_record(),
        ).to_record(),
        "operator_id": "model-operators.compress@0.1.0",
        "status": "compatible",
        "parameter_mappings": [
            {
                "name": "dense-kernel",
                "source": "dense.weight",
                "target": "dense.weight",
                "summary": "declared summary only",
            }
        ],
        "preservation_laws": ["outcome-space-preserved"],
    }


def _submission_registry_record() -> dict[str, object]:
    return {
        "id": "submission-registries.cli@0.1.0",
        "sources": [
            {
                "repository": "maximumcats/leibniz-submissions",
                "repository_type": "dataset",
                "enabled": True,
            }
        ],
    }


def _outcome_space_record() -> dict[str, object]:
    return {
        "id": "core.boolean-outcome@0.1.0",
        "outcomes": [{"id": "yes"}, {"id": "no"}],
    }


def _publication_bundle_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "publication-bundle",
            "content_digest": str(ContentDigest.from_value({"publication": "bundle"})),
        }
    )


def _architecture_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "architecture-manifest",
            "protocol_id": (
                "architecture.sha-"
                "d695a59610f59ce2b61a20b7114b42da8692ffd9a55e4093431e3c00a932e693@0.1.0"
            ),
        }
    )


def _model_manifest_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "model-manifest",
            "protocol_id": "model-manifests.cli@0.1.0",
        }
    )
