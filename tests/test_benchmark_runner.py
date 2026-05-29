from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

import leibniz.benchmark_runner as benchmark_runner
from leibniz.benchmark_runner import (
    BenchmarkRunnerError,
    BenchmarkRunPlan,
    BenchmarkRunSummary,
    run_benchmark,
)
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.cli import main
from leibniz.documents import load_object_document
from leibniz.local_results import load_console_result_view, materialize_benchmark_result_views
from leibniz.measurements import MeasurementDatasetDocument
from leibniz.model_inspection import ModelInspectionDocument
from leibniz.tensor_runtime import (
    TensorRuntime,
    TensorRuntimeDeviceKind,
    resolve_tensor_runtime,
)
from leibniz.training_runs import TrainingHistoryPoint, TrainingRunRecord

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
            train_steps=1,
            dry_run=True,
        )
    )

    assert summary.dry_run is True
    assert summary.measurement_count == 2
    assert summary.run_slug.startswith(
        "digits-arch-bb0dde9254dc-l1-seed101-samples2-steps1-train-"
    )
    assert summary.measurement_dataset_path == (
        tmp_path
        / ".runs"
        / "measurements"
        / "digits"
        / f"{summary.run_slug}.json"
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
            tensor_device="cpu",
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
    assert training_run.protocol.tensor_runtime == "pytorch"
    assert training_run.protocol.tensor_device == "cpu"
    assert training_run.protocol.max_steps == 1
    assert training_run.protocol.validation_sample_count == 2
    assert training_summary["tensor_runtime"] == "pytorch"
    assert training_summary["tensor_device"] == "cpu"
    throughput = cast(dict[str, object], training_summary["throughput"])
    training_throughput = cast(dict[str, object], throughput["training"])
    evaluation_throughput = cast(dict[str, object], throughput["evaluation"])
    phase_timing = cast(dict[str, object], throughput["phase_timing"])
    timing_phases = cast(dict[str, object], phase_timing["phases"])
    tensor_batch_timing = cast(dict[str, object], timing_phases["training_tensor_batch"])
    forward_timing = cast(dict[str, object], timing_phases["training_forward_loss"])
    materialization_timing = cast(
        dict[str, object],
        timing_phases["training_formation_generation.materialization_plan"],
    )
    variation_timing = cast(
        dict[str, object],
        timing_phases["training_formation_generation.variation_coordinates"],
    )
    roofline = cast(dict[str, object], throughput["roofline"])
    roofline_comparison = cast(dict[str, object], throughput["roofline_comparison"])
    phases = cast(dict[str, object], roofline_comparison["phases"])
    training_phase = cast(dict[str, object], phases["training"])
    assert throughput["tensor_runtime"] == "pytorch"
    assert throughput["tensor_device"] == "cpu"
    assert phase_timing["kind"] == "benchmark-phase-timing"
    assert tensor_batch_timing["sample_count"] == 2
    assert cast(float, tensor_batch_timing["seconds"]) > 0
    assert materialization_timing["sample_count"] == 2
    assert variation_timing["sample_count"] == 2
    assert forward_timing["sample_count"] == 2
    assert cast(float, forward_timing["seconds"]) > 0
    assert cast(float, roofline["peak_bytes_per_second"]) > 0
    assert training_throughput["sample_count"] == 2
    assert cast(float, training_throughput["samples_per_second"]) > 0
    assert evaluation_throughput["sample_count"] == 3
    assert cast(float, evaluation_throughput["samples_per_second"]) > 0
    assert roofline_comparison["status"] == "available"
    assert roofline_comparison["model"] == "operational-intensity"
    assert cast(float, roofline_comparison["training_fraction_of_roofline"]) > 0
    assert training_phase["limiting_resource"] in {"compute", "memory-bandwidth"}
    assert cast(float, training_phase["arithmetic_intensity_flops_per_byte"]) > 0
    assert cast(float, training_phase["expected_roofline_flops_per_second"]) > 0
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
            tensor_device="cpu",
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


def test_windowed_plateau_ignores_tiny_recent_best_loss_resets() -> None:
    history = (
        _history_point(check=0, step=0, loss=1.0, best=1.0),
        _history_point(check=1, step=250, loss=1.01, best=1.0, stale_checks=1),
        _history_point(check=2, step=500, loss=0.9995, best=0.9995),
    )

    assert benchmark_runner.has_windowed_validation_plateau(
        history,
        window_checks=2,
        min_delta=0.001,
    )


def test_windowed_plateau_continues_after_material_best_loss_improvement() -> None:
    history = (
        _history_point(check=0, step=0, loss=1.0, best=1.0),
        _history_point(check=1, step=250, loss=1.01, best=1.0, stale_checks=1),
        _history_point(check=2, step=500, loss=0.998, best=0.998),
    )

    assert not benchmark_runner.has_windowed_validation_plateau(
        history,
        window_checks=2,
        min_delta=0.001,
    )


def test_digits_benchmark_runner_auto_falls_back_after_runtime_compile_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_runtime = resolve_tensor_runtime("cpu")
    calls: list[str] = []
    module_calls = 0

    def fake_device_kinds(_requested: object) -> tuple[TensorRuntimeDeviceKind, ...]:
        return ("mps", "cpu")

    def fake_resolve_tensor_runtime(requested: object) -> TensorRuntime:
        device_kind = cast(str, requested)
        calls.append(device_kind)
        return TensorRuntime(
            torch=cpu_runtime.torch,
            device=cpu_runtime.device,
            device_kind=cast(Any, device_kind),
        )

    original_torch_module = benchmark_runner.ExecutableModelOperator.torch_module

    def flaky_torch_module(self: object) -> object:
        nonlocal module_calls
        module_calls += 1
        if module_calls == 1:
            raise RuntimeError("MPS backend failed to compile adaptive pooling")
        return original_torch_module(cast(Any, self))

    monkeypatch.setattr(benchmark_runner, "tensor_runtime_device_kinds", fake_device_kinds)
    monkeypatch.setattr(benchmark_runner, "resolve_tensor_runtime", fake_resolve_tensor_runtime)
    monkeypatch.setattr(
        benchmark_runner.ExecutableModelOperator,
        "torch_module",
        flaky_torch_module,
    )

    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            runs_root=tmp_path / ".runs",
            sample_count=2,
            train_steps=1,
            tensor_device="auto",
        )
    )

    training_summary = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )
    throughput = cast(dict[str, object], training_summary["throughput"])
    fallbacks = cast(list[dict[str, object]], throughput["runtime_fallbacks"])

    assert calls == ["mps", "cpu"]
    assert training_summary["tensor_device"] == "cpu"
    assert throughput["tensor_device"] == "cpu"
    assert fallbacks == [
        {
            "from_device": "mps",
            "to_device": "cpu",
            "reason": "MPS backend failed to compile adaptive pooling",
        }
    ]


def test_digits_benchmark_runner_run_slug_includes_training_controls() -> None:
    base_plan = BenchmarkRunPlan(
        architecture_path=_digits_architecture,
        benchmark_root=_digits_benchmark_root,
        sample_count=4,
        seed=401,
        train_steps=10,
        optimizer="sgd",
    )
    alternate_plan = BenchmarkRunPlan(
        architecture_path=_digits_architecture,
        benchmark_root=_digits_benchmark_root,
        sample_count=4,
        seed=401,
        train_steps=10,
        optimizer="adam",
    )

    assert base_plan.run_slug.startswith("l1-seed401-samples4-steps10-train-")
    assert alternate_plan.run_slug.startswith("l1-seed401-samples4-steps10-train-")
    assert base_plan.run_slug != alternate_plan.run_slug


def test_digits_benchmark_runner_outputs_feed_benchmark_result_views(tmp_path: Path) -> None:
    run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            runs_root=tmp_path / ".runs",
            sample_count=2,
            seed=101,
            train_steps=1,
            tensor_device="cpu",
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
    diagnostics = cast(dict[str, object], history[0]["training_diagnostics"])
    protocol = cast(dict[str, object], diagnostics["protocol"])
    throughput = cast(dict[str, object], diagnostics["throughput"])
    phase_timing = cast(dict[str, object], throughput["phase_timing"])
    roofline_comparison = cast(dict[str, object], throughput["roofline_comparison"])
    assert protocol["optimizer"] == "sgd"
    assert protocol["schedule"] == "none"
    assert diagnostics["stop_reason"] == "max-steps"
    assert diagnostics["steps_run"] == 1
    assert diagnostics["validation_checks"] == 2
    assert "final_validation_loss" in diagnostics
    assert "training_tensor_batch" in cast(dict[str, object], phase_timing["phases"])
    assert roofline_comparison["status"] == "available"
    assert roofline_comparison["model"] == "operational-intensity"
    assert len(cast(list[dict[str, object]], diagnostics["validation_history"])) == 2
    console_view_model = cast(dict[str, object], history[0]["console_view_model"])
    detail_sections = cast(list[dict[str, object]], console_view_model["detail_sections"])
    assert [section["title"] for section in detail_sections] == [
        "Sampled Competence",
        "Training Protocol",
        "Training Outcome",
        "Throughput",
        "Validation History",
    ]
    validation_table = cast(dict[str, object], detail_sections[-1]["table"])
    assert validation_table["columns"] == ["Step", "Loss", "Best", "Stale"]
    artifact_kinds = {
        artifact["kind"]
        for artifact in cast(list[dict[str, object]], diagnostics["artifacts"])
    }
    assert artifact_kinds == {
        "measurement-dataset",
        "model-inspection",
        "training-summary",
    }
    leaderboard = cast(list[dict[str, object]], result["leaderboard"])
    assert leaderboard[0]["observed_complexities"] == [1.0]
    points = cast(list[dict[str, object]], leaderboard[0]["points"])
    assert points[0]["sample_count"] == 2
    inspections = cast(list[dict[str, object]], result["model_inspections"])
    assert len(inspections) == 1
    assert inspections[0]["source_path"] == history[0]["model_inspection_path"]
    assert "measurement_dataset" in inspections[0]
    assert "training_provenance" in inspections[0]


def test_digits_benchmark_runner_materializes_running_training_history(
    tmp_path: Path,
) -> None:
    progress_views: list[Mapping[str, object]] = []

    def refresh_progress(_summary: BenchmarkRunSummary) -> None:
        view_summary = materialize_benchmark_result_views(
            repository_root=_repository_root,
            runs_root=tmp_path / ".runs",
        )
        progress_views.append(load_console_result_view(view_summary.view_file.read_bytes()))

    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            runs_root=tmp_path / ".runs",
            sample_count=2,
            evaluation_sample_count=2,
            seed=101,
            train_steps=2,
            validation_interval=1,
            convergence_patience=0,
            convergence_min_delta=0.0,
            convergence_min_steps=0,
            tensor_device="cpu",
        ),
        progress_callback=refresh_progress,
    )

    assert progress_views
    progress_view = progress_views[0]
    result = cast(list[dict[str, object]], progress_view["benchmark_results"])[0]
    history = cast(list[dict[str, object]], result["training_history"])
    running_run = history[0]
    diagnostics = cast(dict[str, object], running_run["training_diagnostics"])
    throughput = cast(dict[str, object], diagnostics["throughput"])
    training_throughput = cast(dict[str, object], throughput["training"])

    assert running_run["source_kind"] == "local-progress"
    assert running_run["measurement_count"] == 0
    assert diagnostics["status"] == "running"
    assert diagnostics["stop_reason"] == "validation-checkpoint"
    assert diagnostics["validation_checks"] == 1
    assert training_throughput["sample_count"] == 0
    assert cast(dict[str, object], throughput["roofline_comparison"])["status"] == "available"
    assert cast(list[dict[str, object]], result["leaderboard"])[0]["source_kinds"] == [
        "local-progress"
    ]

    final_view_summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        runs_root=tmp_path / ".runs",
    )
    final_view = load_console_result_view(final_view_summary.view_file.read_bytes())
    final_result = cast(list[dict[str, object]], final_view["benchmark_results"])[0]
    final_history = cast(list[dict[str, object]], final_result["training_history"])
    progress_path = (
        summary.training_summary_path.parent.parent.parent
        / "training-progress"
        / summary.training_summary_path.parent.name
        / summary.training_summary_path.name
    )
    assert final_history[0]["source_kind"] == "local-run"
    assert not progress_path.exists()


def test_digits_benchmark_runner_rejects_unmatched_architecture_shape(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkRunnerError, match="does not match generated observation shape"):
        run_benchmark(
            BenchmarkRunPlan(
                architecture_path=_digits_architecture,
                benchmark_root=_digits_benchmark_root,
                runs_root=tmp_path / ".runs",
                scale=2,
                sample_count=1,
                tensor_device="cpu",
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
    assert captured.out.startswith(
        "planned benchmark run "
        "digits-arch-bb0dde9254dc-l1-seed101-samples2-stepsconverge-train-"
    )
    assert not (tmp_path / ".runs").exists()


def _history_point(
    *,
    check: int,
    step: int,
    loss: float,
    best: float,
    stale_checks: int = 0,
) -> TrainingHistoryPoint:
    return TrainingHistoryPoint(
        step=step,
        validation_check=check,
        validation_loss=loss,
        best_validation_loss=best,
        best_validation_step=step,
        best_validation_check=check,
        stale_checks=stale_checks,
    )
