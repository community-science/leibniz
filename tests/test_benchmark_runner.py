from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from leibniz.benchmark_runner import (
    BenchmarkRunnerError,
    BenchmarkRunPlan,
    run_benchmark,
)
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.cli import main
from leibniz.documents import load_object_document
from leibniz.local_results import load_console_result_view, materialize_benchmark_result_views
from leibniz.measurements import MeasurementDatasetDocument
from leibniz.model_inspection import ModelInspectionDocument
from leibniz.training_runs import TrainingRunRecord

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
            evaluation_sample_count=3,
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
    assert summary.measurement_count == 3
    assert len(dataset_document.dataset.measurements) == 3
    assert inspection_document.inspection.cost_summary.parameter_count == 50
    assert inspection_document.inspection.cost_summary.inference_flops == 1104
    assert summary.training_summary_path.exists()
    training_summary = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )
    training_run = TrainingRunRecord.from_record(
        cast(Mapping[str, object], training_summary["training_run"])
    )
    assert training_run.protocol.optimizer == "sgd"
    assert training_run.protocol.objective == "cross-entropy"
    assert training_run.protocol.max_steps == 1
    assert training_run.protocol.validation_sample_count == 2
    assert training_run.steps_run == 1
    assert training_run.validation_checks == 2
    assert training_run.validation_history[0].step == 0
    assert training_run.validation_history[-1].step == 1
    sampled_competence = cast(dict[str, object], training_summary["sampled_competence"])
    assert sampled_competence["kind"] == "sampled-complexity-class"
    assert sampled_competence["sampling_rule"] == "generator-uniform-component-sequence-v1"
    assert (
        sampled_competence["difficulty_assumption"]
        == "approximately-uniform-within-complexity-class"
    )
    assert sampled_competence["complexity_axis"] == "C"
    assert sampled_competence["complexity"] == 1
    assert sampled_competence["sample_count"] == 3
    assert 0.0 <= cast(float, sampled_competence["mean_accepted_mass"]) <= 1.0


def test_digits_benchmark_runner_records_convergence_protocol_controls(
    tmp_path: Path,
) -> None:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            runs_root=tmp_path / ".runs",
            sample_count=2,
            seed=101,
            train_steps=3,
            learning_rate=0.005,
            optimizer="adam",
            schedule="cosine",
            validation_interval=2,
            convergence_patience=2,
            convergence_min_delta=0.001,
        )
    )

    training_summary = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )
    training_run = TrainingRunRecord.from_record(
        cast(Mapping[str, object], training_summary["training_run"])
    )

    assert training_run.protocol.optimizer == "adam"
    assert training_run.protocol.schedule == "cosine"
    assert training_run.protocol.learning_rate == 0.005
    assert training_run.protocol.validation_interval == 2
    assert training_run.protocol.patience == 2
    assert training_run.protocol.min_delta == 0.001
    assert training_run.protocol.validation_source == "generator-resample"
    assert [point.step for point in training_run.validation_history] == [0, 2, 3]
    assert training_run.validation_history[-1].learning_rates


def test_digits_benchmark_runner_outputs_feed_benchmark_result_views(tmp_path: Path) -> None:
    run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            runs_root=tmp_path / ".runs",
            sample_count=2,
            seed=101,
            train_steps=1,
        )
    )

    summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        runs_root=tmp_path / ".runs",
    )

    view = load_console_result_view(summary.view_file.read_bytes())
    results = cast(list[dict[str, object]], view["benchmark_results"])
    result = results[0]
    history = cast(list[dict[str, object]], result["training_history"])
    assert history[0]["source_kind"] == "local-run"
    assert "model_inspection_digest" in history[0]
    assert "model_inspection_path" in history[0]
    assert "sampled_competence" in history[0]
    training_summary = cast(dict[str, object], history[0]["training_summary"])
    training_run = cast(dict[str, object], training_summary["training_run"])
    protocol = cast(dict[str, object], training_run["protocol"])
    assert protocol["optimizer"] == "sgd"
    assert protocol["schedule"] == "none"
    assert len(cast(list[dict[str, object]], training_run["validation_history"])) == 2
    leaderboard = cast(list[dict[str, object]], result["leaderboard"])
    assert leaderboard[0]["observed_complexities"] == [1.0]
    points = cast(list[dict[str, object]], leaderboard[0]["points"])
    assert points[0]["sample_count"] == 2
    inspections = cast(list[dict[str, object]], result["model_inspections"])
    assert len(inspections) == 1
    assert inspections[0]["source_path"] == history[0]["model_inspection_path"]
    assert "measurement_dataset" in inspections[0]
    assert "training_provenance" in inspections[0]


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
