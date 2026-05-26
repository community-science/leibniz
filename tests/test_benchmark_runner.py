from pathlib import Path

import pytest

from leibniz.benchmark_runner import (
    BenchmarkRunnerError,
    BenchmarkRunPlan,
    run_benchmark,
)
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.cli import main
from leibniz.measurements import MeasurementDatasetDocument
from leibniz.model_inspection import ModelInspectionDocument

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_digits_architecture = (
    _repository_root / "tests" / "fixtures" / "architecture" / "digits_pool" / "manifest.json"
)


def test_digits_benchmark_runner_dry_run_does_not_write_state(tmp_path: Path) -> None:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            runs_root=tmp_path / ".runs",
            sample_count=2,
            dry_run=True,
        )
    )

    assert summary.dry_run is True
    assert summary.measurement_count == 2
    assert summary.measurement_dataset_path == (
        tmp_path / ".runs" / "measurements" / "digits" / "digits-l1-seed101-samples2-steps1.json"
    )
    assert not summary.measurement_dataset_path.exists()
    assert not (tmp_path / ".runs").exists()


def test_digits_benchmark_runner_writes_valid_tiny_cpu_outputs(tmp_path: Path) -> None:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            runs_root=tmp_path / ".runs",
            sample_count=2,
            seed=101,
            train_steps=1,
        )
    )

    dataset_document = MeasurementDatasetDocument.from_bytes(
        summary.measurement_dataset_path.read_bytes()
    )
    inspection_document = ModelInspectionDocument.from_bytes(
        summary.model_inspection_path.read_bytes()
    )
    manifest = BenchmarkManifestDocument.from_bytes(
        (_digits_benchmark_root / "manifest.json").read_bytes()
    ).manifest

    dataset_document.dataset.validate_manifest(manifest, scale=1)
    assert len(dataset_document.dataset.measurements) == 2
    assert inspection_document.inspection.cost_summary.parameter_count == 50
    assert inspection_document.inspection.cost_summary.inference_flops == 1104
    assert summary.training_summary_path.exists()


def test_digits_benchmark_runner_rejects_unmatched_architecture_shape(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkRunnerError, match="does not match generated observation shape"):
        run_benchmark(
            BenchmarkRunPlan(
                architecture_path=_digits_architecture,
                benchmark_root=_digits_benchmark_root,
                runs_root=tmp_path / ".runs",
                scale=2,
                sample_count=1,
            )
        )


def test_cli_runs_digits_benchmark_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "benchmark",
            "run",
            "--architecture",
            str(_digits_architecture),
            "--benchmark-root",
            str(_digits_benchmark_root),
            "--runs-root",
            str(tmp_path / ".runs"),
            "--sample-count",
            "2",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.startswith("planned benchmark run digits-l1-seed101-samples2-steps1")
    assert not (tmp_path / ".runs").exists()
