import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest

import leibniz.cli as cli
from leibniz.artifacts import ArtifactReference, reference_for_record
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.cli import main
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes, load_object_document
from leibniz.measurements import MeasurementDocument
from leibniz.model_manifests import ModelExecutionFamily
from leibniz.model_operations import ModelOperation

_fixtures_root = Path(__file__).parent / "fixtures"
_finite_fixture = _fixtures_root / "finite_outcome"
_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def test_cli_help_text(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["validate", "--help"])

    assert exit_info.value.code == 0
    assert "validate artifact files" in capsys.readouterr().out


def test_cli_module_invokes_main() -> None:
    environment = {
        **os.environ,
        "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
    }
    result = subprocess.run(
        [sys.executable, "-m", "leibniz.cli", "validate", "--help"],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert result.returncode == 0
    assert "validate artifact files" in result.stdout
    assert result.stderr == ""


def test_cli_validate_help_lists_expanded_artifacts(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["validate", "--help"])

    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert "artifact-reference" in output
    assert "model-manifest" in output
    assert "model-lineage" in output
    assert "architecture" in output
    assert "submission-registry" in output
    assert "evaluation-bundle" in output


def test_cli_benchmark_evaluate_keeps_checkpoint_artifact_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["benchmark", "evaluate", "--help"])

    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert "--checkpoint-artifact" in output
    assert "--training-summary" not in output
    assert "--run-slug" not in output
    assert "--evaluation-rung-count" not in output
    assert "--training-compute" not in output


def test_cli_benchmark_train_discovers_architecture_manifests(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["benchmark", "train", "--help"])

    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert "--architecture ARCHITECTURE" in output
    assert "--benchmark-root BENCHMARK_ROOT" in output
    assert "benchmark ids or names" in output
    assert "may be" in output
    assert "repeated" in output
    assert "discovered under results/training" in output


def test_cli_benchmark_clean_removes_generated_state_and_keeps_architectures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = tmp_path / "results"
    architecture = results_root / "architectures" / "digits" / "digits_pool.json"
    training_record = results_root / "training" / "digits" / "run.json"
    model_artifact = results_root / "models" / "digits" / "checkpoint.pt"
    evaluation_record = results_root / "evaluations" / "digits" / "eval.json"
    view_record = results_root / "views" / "digits" / "benchmark_results.json"
    for path in (architecture, training_record, model_artifact, evaluation_record, view_record):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    for root_name in ("training", "models", "evaluations", "views"):
        (results_root / root_name / ".gitkeep").touch()

    assert (
        main(
            [
                "benchmark",
                "clean",
                "--results-root",
                str(results_root),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "removed 4 generated benchmark path(s)" in output
    assert architecture.is_file()
    assert not training_record.exists()
    assert not model_artifact.exists()
    assert not evaluation_record.exists()
    assert not view_record.exists()
    for root_name in ("training", "models", "evaluations", "views"):
        assert (results_root / root_name / ".gitkeep").is_file()


def test_cli_benchmark_clean_dry_run_does_not_remove_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = tmp_path / "results"
    architecture = results_root / "architectures" / "digits" / "digits_pool.json"
    training_record = results_root / "training" / "digits" / "run.json"
    architecture.parent.mkdir(parents=True)
    training_record.parent.mkdir(parents=True)
    architecture.write_text("{}", encoding="utf-8")
    training_record.write_text("{}", encoding="utf-8")

    assert (
        main(
            [
                "benchmark",
                "clean",
                "--results-root",
                str(results_root),
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "would remove 1 generated benchmark path(s)" in output
    assert architecture.is_file()
    assert training_record.is_file()


def test_cli_benchmark_train_scopes_architecture_subdirs_by_benchmark_name(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    architecture_root = tmp_path / "architectures"
    digits_architecture = architecture_root / "digits" / "development" / "digits_pool.json"
    chess_architecture = architecture_root / "chess" / "linear_board.json"
    digits_architecture.parent.mkdir(parents=True)
    chess_architecture.parent.mkdir(parents=True)
    digits_architecture.write_bytes(
        (_fixtures_root / "architecture" / "digits_pool.json").read_bytes()
    )
    chess_architecture.write_bytes(
        (_fixtures_root / "architecture" / "chess_board_linear.json").read_bytes()
    )

    assert (
        main(
            [
                "benchmark",
                "train",
                "--architecture",
                str(architecture_root),
                "--results-root",
                str(tmp_path / "results"),
                "--dry-run",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert output.count("planned benchmark training run ") == 2
    assert "planned benchmark training run chess-arch-" in output
    assert "planned benchmark training run digits-arch-" in output
    assert "training/chess/digits-" not in output
    assert "training/digits/chess-" not in output


def test_cli_benchmark_help_advertises_current_workflow_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["benchmark", "--help"])

    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert "init" in output
    assert "publish" in output
    assert "clean" in output
    assert "train" in output
    assert "evaluate" in output
    assert "profile" in output
    assert "compete" not in output
    assert "push" not in output
    assert "materialize" not in output
    assert "time-formation" not in output


def test_cli_benchmark_publish_push_is_opt_out(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["benchmark", "publish", "--help"])

    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert "--no-push" in output
    assert "  --push" not in output


def test_cli_root_exposes_benchmark_surface_without_result_management_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    output = capsys.readouterr().out
    assert exit_info.value.code == 0
    assert "results" not in output


def test_cli_console_dev_runs_npm_dev_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], Path]] = []

    def run_command(
        command: list[str],
        *,
        check: bool,
        cwd: Path,
    ) -> subprocess.CompletedProcess[str]:
        assert check is False
        calls.append((command, cwd))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", run_command)

    assert main(["console", "dev", "--host", "0.0.0.0", "--port", "5174"]) == 0

    assert calls == [
        (
            ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5174"],
            Path(cli.__file__).parent / "console" / "_web_src",
        )
    ]


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


def test_cli_validates_architecture_manifest_with_semantics(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "validate",
            "architecture",
            str(_fixtures_root / "architecture" / "digits_pool.json"),
            "--semantic",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.startswith("valid architecture architecture.sha-")
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
            "authority-index",
            "_authority_index_record",
            "valid authority index authority-indexes.cli@0.1.0",
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


@pytest.mark.parametrize(
    "artifact",
    [
        "artifact-reference",
        "artifact-index",
        "authority-index",
        "resource-report",
        "resource-report-set",
        "model-manifest",
        "model-operation",
        "model-lineage",
        "model-derivation",
        "evaluation-bundle",
        "submission-registry",
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


def test_cli_profiles_benchmark_formation_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "benchmark",
            "profile",
            "--benchmark-root",
            str(_digits_benchmark_root),
            "--batch-target",
            "1",
            "--repeats",
            "1",
            "--warmup-repeats",
            "0",
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    record = load_object_document(captured.out.encode("utf-8"), description="timing")
    assert exit_code == 0
    assert captured.err == ""
    assert record["format"] == "leibniz.formation-operator-profile"
    assert record["tensor_device"] == "cpu"
    assert record["sample_count"] == 1
    assert record["rows"] != []


def test_cli_benchmark_selection_loads_python_implementation_without_manifest_file(
    tmp_path: Path,
) -> None:
    benchmark_root = tmp_path / "digits"
    benchmark_root.mkdir()
    (benchmark_root / "benchmark.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "from leibniz.benchmark_implementations import load_benchmark",
                f"_source = Path({str(_digits_benchmark_root)!r})",
                "def benchmark(root: Path):",
                "    return load_benchmark(_source)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    roots = cast(Any, cli)._benchmark_roots_by_id(
        repository_root=tmp_path,
        explicit_roots=(benchmark_root,),
    )

    assert roots == {"benchmarks.digits@0.1.0": benchmark_root}


def _dataset_path(tmp_path: Path) -> Path:
    measurement = MeasurementDocument.from_bytes(
        (_finite_fixture / "measurement.json").read_bytes()
    ).measurement.to_record()
    dataset_path = tmp_path / "measurements.json"
    dataset_path.write_bytes(canonical_document_bytes({"measurements": [measurement]}))
    return dataset_path


def _artifact_reference_record() -> dict[str, object]:
    return _architecture_reference().to_record()


def _expanded_artifact_record(factory_name: str) -> dict[str, object]:
    if factory_name == "_artifact_reference_record":
        return _artifact_reference_record()
    if factory_name == "_artifact_index_record":
        return _artifact_index_record()
    if factory_name == "_authority_index_record":
        return _authority_index_record()
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
        "artifacts": [_architecture_reference().to_record()],
    }


def _authority_index_record() -> dict[str, object]:
    artifact = _architecture_reference().to_record()
    return {
        "id": "authority-indexes.cli@0.1.0",
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


def _resource_report_record() -> dict[str, object]:
    return {
        "id": "resource-reports.cli@0.1.0",
        "artifact": _model_manifest_reference().to_record(),
        "parameter_count": 1,
        "parameter_bits": 8,
        "storage_bytes": 1,
    }


def _resource_report_set_record() -> dict[str, object]:
    return {
        "id": "resource-report-sets.cli@0.1.0",
        "reports": [_resource_report_record()],
    }


def _model_interface_record() -> dict[str, object]:
    return {
        "id": "model-interfaces.cli@0.1.0",
        "prediction_space": {
            "kind": "finite-outcome-space",
            "outcome_space_id": "core.boolean-outcome@0.1.0",
            "outcome_count": 2,
        },
        "prediction_kind": "direct-finite-probability-measure",
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
        "execution_family": ModelExecutionFamily.reference_runner_pytorch_sequential().to_record(),
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
                "repository": "example-owner/leibniz-submissions",
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
