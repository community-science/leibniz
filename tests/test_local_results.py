import math
import subprocess
from collections.abc import Callable
from pathlib import Path
from textwrap import dedent
from typing import Any, cast

import pytest

import leibniz.local_results as local_results
from leibniz.architectures import ArchitectureManifestDocument
from leibniz.benchmark_evaluation import sampled_competence_compute_cost_integral
from leibniz.benchmark_runner import (
    BenchmarkEvaluationPlan,
    BenchmarkRunPlan,
    evaluate_benchmark_checkpoint,
    run_benchmark,
)
from leibniz.cli import main
from leibniz.documents import canonical_document_bytes, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.local_results import (
    LocalResultImportError,
    initialize_result_checkout,
    load_console_result_view,
    materialize_benchmark_result_views,
    publish_local_benchmark_results,
    push_result_checkout,
)
from leibniz.model_operators import (
    architecture_with_input_shape,
    summarize_architecture_operators,
)
from leibniz.training_runs import TrainingHistoryPoint

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_digits_architecture = (
    _repository_root / "tests" / "fixtures" / "architecture" / "digits_pool.json"
)


def _run_and_evaluate_digits_benchmark(results_root: Path, *, sample_count: int = 1) -> None:
    training_summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=results_root,
            train_steps=0,
            tensor_device="cpu",
        )
    )
    evaluate_benchmark_checkpoint(
        BenchmarkEvaluationPlan(
            checkpoint_artifact_path=_selected_checkpoint_artifact_path(
                training_summary.training_summary_path
            ),
            benchmark_root=_digits_benchmark_root,
            results_root=results_root,
            tensor_device="cpu",
        )
    )


def test_competence_integral_integrates_bits_above_chance() -> None:
    complexity = 20.0
    chance_mass = 0.1

    assert math.isclose(
        local_results.competence_integral(
            ({"complexity": complexity, "score": 1.0},),
            chance_mass=chance_mass,
        ).value,
        20.0,
    )
    assert math.isclose(
        local_results.competence_integral(
            ({"complexity": complexity * 2.0, "score": 1.0},),
            chance_mass=chance_mass,
        ).value,
        40.0,
    )
    assert math.isclose(
        local_results.competence_integral(
            ({"complexity": complexity, "score": 0.55},),
            chance_mass=chance_mass,
        ).value,
        19.5,
    )
    assert math.isclose(
        local_results.competence_integral(
            ({"complexity": complexity * 4.0, "score": chance_mass},),
            chance_mass=chance_mass,
        ).value,
        79.0,
    )
    assert math.isclose(
        local_results.competence_integral(
            (
                {"complexity": complexity, "score": chance_mass},
                {"complexity": complexity * 2.0, "score": 1.0},
            ),
            chance_mass=chance_mass,
        ).value,
        39.0,
    )


def test_console_result_view_rejects_wrong_format() -> None:
    with pytest.raises(LocalResultImportError, match="unsupported format"):
        load_console_result_view(canonical_document_bytes({"format": "other", "format_version": 1}))


def test_known_benchmark_manifests_loads_python_implementation_without_manifest_file(
    tmp_path: Path,
) -> None:
    benchmark_root = tmp_path / "src" / "leibniz" / "benchmarks" / "digits"
    benchmark_root.mkdir(parents=True)
    (benchmark_root.parent / "__pycache__").mkdir()
    (benchmark_root / "benchmark.py").write_text(
        dedent(
            """
            from pathlib import Path

            from leibniz.benchmarks import BenchmarkManifest
            from leibniz.identifiers import ProtocolIdentifier, ProtocolName
            from leibniz.outcomes import Outcome, OutcomeSpace


            class Impl:
                def __init__(self, root: Path) -> None:
                    self._root = root

                @property
                def root(self) -> Path:
                    return self._root

                @property
                def manifest(self):
                    return BenchmarkManifest(
                        id=ProtocolIdentifier.parse("benchmarks.digits@0.1.0"),
                        name=ProtocolName.parse("benchmarks.digits"),
                        outcome_space=OutcomeSpace(
                            id=ProtocolIdentifier.parse("benchmarks.digits.outcomes@0.1.0"),
                            outcomes=tuple(Outcome(id=f"digit-{index}") for index in range(10)),
                        ),
                    )

                @property
                def generator(self):
                    return lambda **kwargs: None


            def benchmark(root: Path):
                return Impl(root)
            """
        ),
        encoding="utf-8",
    )

    benchmarks = cast(Any, local_results)._known_benchmarks(tmp_path)

    benchmark = benchmarks[ProtocolIdentifier.parse("benchmarks.digits@0.1.0")]
    assert str(benchmark.manifest.id) == "benchmarks.digits@0.1.0"


def test_console_result_view_validates_embedded_model_inspections(tmp_path: Path) -> None:
    _run_and_evaluate_digits_benchmark(tmp_path / "results")
    summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )

    view = dict(load_console_result_view(summary.view_file.read_bytes()))
    results = cast(list[dict[str, object]], view["benchmark_results"])
    inspections = cast(list[dict[str, object]], results[0]["model_inspections"])
    inspections[0] = {key: value for key, value in inspections[0].items() if key != "components"}

    with pytest.raises(
        LocalResultImportError,
        match="model_inspections.0: invalid model inspection",
    ):
        load_console_result_view(canonical_document_bytes(view))


def test_console_result_view_validates_benchmark_leaderboard_models(tmp_path: Path) -> None:
    _run_and_evaluate_digits_benchmark(tmp_path / "results")
    summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )

    view = dict(load_console_result_view(summary.view_file.read_bytes()))
    results = cast(list[dict[str, object]], view["benchmark_results"])
    leaderboard = cast(list[dict[str, object]], results[0]["leaderboard"])
    leaderboard[0] = {key: value for key, value in leaderboard[0].items() if key != "model_key"}

    with pytest.raises(LocalResultImportError, match="model_key"):
        load_console_result_view(canonical_document_bytes(view))


def test_console_result_view_validates_model_detail_tables(tmp_path: Path) -> None:
    _run_and_evaluate_digits_benchmark(tmp_path / "results")
    summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )

    view = dict(load_console_result_view(summary.view_file.read_bytes()))
    results = cast(list[dict[str, object]], view["benchmark_results"])
    leaderboard = cast(list[dict[str, object]], results[0]["leaderboard"])
    model_view = cast(dict[str, object], leaderboard[0]["console_view_model"])
    sections = cast(list[dict[str, object]], model_view["detail_sections"])
    sections[0]["table"] = {
        "aria_label": "Malformed detail table",
        "columns": ["A", "B"],
        "rows": [["only one cell"]],
    }

    with pytest.raises(LocalResultImportError, match="table rows must match columns"):
        load_console_result_view(canonical_document_bytes(view))


def test_console_result_view_validates_training_diagnostics_records(tmp_path: Path) -> None:
    _run_and_evaluate_digits_benchmark(tmp_path / "results")
    summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )

    view = dict(load_console_result_view(summary.view_file.read_bytes()))
    results = cast(list[dict[str, object]], view["benchmark_results"])
    history = cast(list[dict[str, object]], results[0]["training_history"])
    history[0]["training_diagnostics"] = {
        "status": "not-a-training-status",
    }

    with pytest.raises(LocalResultImportError, match="unsupported training status"):
        load_console_result_view(canonical_document_bytes(view))


def test_console_validation_history_is_bounded_and_preserves_extrema() -> None:
    history = tuple(
        TrainingHistoryPoint(
            step=index,
            validation_check=index,
            validation_loss=100.0 if index == 250 else float(index % 11),
            stale_checks=0,
        )
        for index in range(1000)
    )

    sample_history = cast(
        Callable[
            [tuple[TrainingHistoryPoint, ...]],
            tuple[TrainingHistoryPoint, ...],
        ],
        local_results._sample_console_validation_history,  # pyright: ignore[reportPrivateUsage]
    )
    sampled = sample_history(history)

    assert len(sampled) <= 512
    assert sampled[0] == history[0]
    assert sampled[-1] == history[-1]
    assert any(point.validation_loss == 100.0 for point in sampled)


def test_console_result_view_validates_training_estimate_comparison(
    tmp_path: Path,
) -> None:
    _run_and_evaluate_digits_benchmark(tmp_path / "results")
    summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )

    view = dict(load_console_result_view(summary.view_file.read_bytes()))
    results = cast(list[dict[str, object]], view["benchmark_results"])
    leaderboard = cast(list[dict[str, object]], results[0]["leaderboard"])
    comparison = cast(dict[str, object], leaderboard[0]["training_estimate_comparison"])
    comparison["kind"] = "other"

    with pytest.raises(
        LocalResultImportError,
        match=r"training_estimate_comparison.kind is invalid",
    ):
        load_console_result_view(canonical_document_bytes(view))


def test_console_result_view_validates_training_protocol_gate_cadence(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=results_root,
            train_steps=0,
            tensor_device="cpu",
        )
    )
    summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=results_root,
    )

    view = dict(load_console_result_view(summary.view_file.read_bytes()))
    results = cast(list[dict[str, object]], view["benchmark_results"])
    history = cast(list[dict[str, object]], results[0]["training_history"])
    assert history[0]["result_status"] == "provisional"
    protocol = cast(dict[str, object], history[0]["training_diagnostics"])["protocol"]
    cast(dict[str, object], protocol)["gate_check_interval"] = 0

    with pytest.raises(
        LocalResultImportError,
        match="gate_check_interval",
    ):
        load_console_result_view(canonical_document_bytes(view))


def test_materialize_benchmark_result_views_rejects_evaluation_bundle_without_inference_compute(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    _run_and_evaluate_digits_benchmark(results_root)
    for path in results_root.rglob("*.json"):
        record = dict(load_object_document(path.read_bytes(), description="result record"))
        if record.get("format") != "leibniz.benchmark-evaluation":
            continue
        throughput = cast(dict[str, object], record["throughput"])
        for key in ("checkpoint_evaluation", "evaluation"):
            phase = cast(dict[str, object], throughput[key])
            phase.pop("max_inference_compute", None)
        path.write_bytes(canonical_document_bytes(record) + b"\n")

    with pytest.raises(
        LocalResultImportError,
        match="missing measured max_inference_compute",
    ):
        materialize_benchmark_result_views(
            repository_root=_repository_root,
            results_root=results_root,
        )


def test_materialize_benchmark_result_views_projects_evaluation_bundles(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    _run_and_evaluate_digits_benchmark(results_root)

    summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=results_root,
    )

    assert summary.benchmark_count == 2
    assert summary.model_count == 1
    assert summary.run_count == 1
    assert summary.view_file == (
        tmp_path / "results" / "views" / "digits" / "benchmark_results.json"
    )
    assert {
        path.relative_to(results_root).as_posix()
        for path in summary.benchmark_view_files
    } == {
        "views/chess/benchmark_results.json",
        "views/digits/benchmark_results.json",
    }

    view = load_console_result_view(summary.view_file.read_bytes())
    assert view["format"] == "leibniz.console.benchmark-results"
    results = cast(list[dict[str, object]], view["benchmark_results"])
    result = results[0]
    assert result["benchmark_id"] == "benchmarks.digits@0.1.0"
    leaderboard = cast(list[dict[str, object]], result["leaderboard"])
    measurement_count = cast(int, leaderboard[0]["measurement_count"])
    assert measurement_count >= 64 * 4
    assert measurement_count % 64 == 0
    cost_summary = cast(dict[str, object], leaderboard[0]["cost_summary"])
    assert isinstance(cost_summary["inference_compute"], int | float)
    assert isinstance(cost_summary["cost"], int | float)
    assert cost_summary["cost"] >= 0
    cost_integral = cast(dict[str, object], leaderboard[0]["cost_integral"])
    assert cost_integral["kind"] == "compute-cost-integral"
    assert math.isclose(cast(float, cost_integral["value"]), cast(float, cost_summary["cost"]))
    assert cast(list[dict[str, object]], cost_integral["terms"])
    frontiers = cast(dict[str, object], result["frontiers"])
    assert len(cast(list[object], frontiers["cost"])) == 1
    reference_curves = cast(list[dict[str, object]], result["reference_curves"])
    assert len(reference_curves) == 1
    oracle_curve = reference_curves[0]
    assert oracle_curve["kind"] == "oracle-inference-compute-reference-v1"
    assert oracle_curve["key"] == "oracle_inference_compute"
    assert oracle_curve["x_axis"] == "cost"
    assert oracle_curve["y_axis"] == "score"
    oracle_points = cast(list[dict[str, object]], oracle_curve["points"])
    assert len(oracle_points) >= 2
    assert all(cast(int | float, point["cost"]) >= 0 for point in oracle_points)
    assert any(cast(int | float, point["cost"]) > 0 for point in oracle_points)
    assert max(cast(int | float, point["cost"]) for point in oracle_points) >= 10_000_000_000
    model_view = cast(dict[str, object], leaderboard[0]["console_view_model"])
    model_sections = cast(list[dict[str, object]], model_view["detail_sections"])
    assert [section["title"] for section in model_sections] == [
        "Model Contract",
        "Architecture Graph",
        "Evidence",
        "Training Estimate",
        "Training Estimate Rungs",
        "Resources",
    ]
    comparison = cast(dict[str, object], leaderboard[0]["training_estimate_comparison"])
    assert comparison["kind"] == "training-vs-accepted-sampled-competence-v1"
    assert math.isclose(
        cast(float, comparison["score_delta"]),
        cast(float, comparison["training_score"])
        - cast(float, comparison["accepted_score"]),
    )
    comparison_points = cast(list[dict[str, object]], comparison["points"])
    assert comparison_points
    assert 0 < cast(int, comparison["matched_point_count"]) <= len(comparison_points)
    assert comparison_points[0]["status"] == "matched"
    assert "training_score" in comparison_points[0]
    assert "accepted_score" in comparison_points[0]
    assert "score_delta" in comparison_points[0]
    assert {point["status"] for point in comparison_points} >= {
        "accepted-only",
        "matched",
    }
    cost_summary = cast(dict[str, object], leaderboard[0]["cost_summary"])
    assert "parameter_count" not in cost_summary
    assert cost_summary["storage_bytes"] == 200
    frontiers = cast(dict[str, object], result["frontiers"])
    assert len(cast(list[dict[str, object]], frontiers["cost"])) == 1
    history = cast(list[dict[str, object]], result["training_history"])
    assert history[0]["source_kind"] == "local-run"
    diagnostics = cast(dict[str, object], history[0]["training_diagnostics"])
    assert math.isclose(cast(float, diagnostics["validation_loss_reference"]), math.log(10))
    validation_history = cast(list[dict[str, object]], diagnostics["validation_history"])
    assert validation_history
    assert diagnostics["validation_history_sample_count"] == len(validation_history)
    assert cast(int, diagnostics["validation_history_total_count"]) >= len(validation_history)
    inspections = cast(list[dict[str, object]], result["model_inspections"])
    assert len(inspections) == 1
    assert inspections[0]["source_path"] == history[0]["source_path"]
    assert "measurement_dataset" in inspections[0]
    artifact_kinds = {
        artifact["kind"]
        for artifact in cast(list[dict[str, object]], inspections[0]["artifacts"])
    }
    assert artifact_kinds == {
        "measurement-dataset",
        "model-checkpoint",
        "model-inspection",
        "model-manifest",
    }


def test_integrated_model_cost_reconstructs_point_density_from_input_shape() -> None:
    architecture = ArchitectureManifestDocument.from_bytes(
        _digits_architecture.read_bytes()
    ).manifest
    first_input_shape = (1, 16, 16)
    second_input_shape = (1, 32, 32)
    first_compute = summarize_architecture_operators(
        architecture_with_input_shape(architecture, first_input_shape)
    ).inference_compute
    second_compute = summarize_architecture_operators(
        architecture_with_input_shape(architecture, second_input_shape)
    ).inference_compute

    assert first_compute is not None
    assert second_compute is not None

    cost_integral = sampled_competence_compute_cost_integral(
        points=(
            {
                "complexity": 1.0,
                "input_shape": list(first_input_shape),
            },
            {
                "complexity": 2.0,
                "complexity_minimum": 2.0,
                "complexity_maximum": 2.0,
                "input_shape": list(second_input_shape),
            },
        ),
        architecture=architecture,
        error_type=local_results.LocalResultImportError,
        field_prefix="compute_cost_point",
    )

    assert math.isclose(cost_integral.value, 32.0 * (first_compute + second_compute))


def test_training_estimate_comparison_uses_selected_checkpoint_estimate(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"
    _run_and_evaluate_digits_benchmark(results_root)
    training_summary_path = next((results_root / "training" / "digits").glob("*.json"))
    training_summary = dict(
        load_object_document(
            training_summary_path.read_bytes(),
            description="training summary",
        )
    )
    selected_checkpoint = cast(
        dict[str, object],
        training_summary["selected_model_checkpoint"],
    )
    assert "score_estimate" not in selected_checkpoint
    selected_estimate = cast(
        dict[str, object],
        training_summary["selected_model_checkpoint_score_estimate"],
    )
    terminal_estimate = dict(selected_estimate)
    terminal_estimate["score"] = cast(float, selected_estimate["score"]) + 1.0
    training_summary["training_estimate"] = terminal_estimate
    training_summary_path.write_bytes(canonical_document_bytes(training_summary))

    summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=results_root,
    )
    view = load_console_result_view(summary.view_file.read_bytes())
    result = cast(list[dict[str, object]], view["benchmark_results"])[0]
    leaderboard = cast(list[dict[str, object]], result["leaderboard"])
    comparison = cast(dict[str, object], leaderboard[0]["training_estimate_comparison"])

    assert comparison["training_score"] == selected_estimate["score"]
    assert comparison["training_score"] != terminal_estimate["score"]


def test_cli_benchmark_evaluate_discovers_training_checkpoints(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = tmp_path / "results"
    pending_root = results_root / "training" / "digits" / "pending"
    pending_root.mkdir(parents=True)
    (pending_root / "queued_architecture.json").write_bytes(_digits_architecture.read_bytes())
    run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=results_root,
            train_steps=0,
            tensor_device="cpu",
        )
    )

    exit_code = main(
        [
            "benchmark",
            "evaluate",
            "digits",
            "--benchmark-root",
            str(_digits_benchmark_root),
            "--results-root",
            str(results_root),
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "completed benchmark evaluation" in captured.out
    assert "materialized 2 benchmark result view(s)" in captured.out
    assert len(tuple((results_root / "evaluations" / "digits").glob("*.json"))) == 1
    assert (results_root / "views" / "digits" / "benchmark_results.json").is_file()

    (results_root / "views" / "digits" / "benchmark_results.json").unlink()
    assert main(
        [
            "benchmark",
            "evaluate",
            "digits",
            "--benchmark-root",
            str(_digits_benchmark_root),
            "--results-root",
            str(results_root),
            "--device",
            "cpu",
        ]
    ) == 0
    rerun = capsys.readouterr()
    assert "no unevaluated benchmark checkpoints found" in rerun.out
    assert "materialized 2 benchmark result view(s)" in rerun.out
    assert len(tuple((results_root / "evaluations" / "digits").glob("*.json"))) == 1
    assert (results_root / "views" / "digits" / "benchmark_results.json").is_file()


def test_cli_benchmark_train_discovers_uncompleted_architecture_manifests(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = tmp_path / "results"
    pending_root = results_root / "training" / "digits" / "pending"
    pending_root.mkdir(parents=True)
    completed_architecture = pending_root / "digits_pool.json"
    uncompleted_architecture = pending_root / "digits_convnet.json"
    completed_architecture.write_bytes(_digits_architecture.read_bytes())
    uncompleted_architecture.write_bytes(
        (_repository_root / "tests" / "fixtures" / "architecture" / "digits_convnet.json")
        .read_bytes()
    )

    assert (
        main(
            [
                "benchmark",
                "train",
                "digits",
                "--architecture",
                str(completed_architecture),
                "--results-root",
                str(results_root),
                "--train-steps",
                "0",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    capsys.readouterr()

    exit_code = main(
        [
            "benchmark",
            "train",
            "digits",
            "--results-root",
            str(results_root),
            "--train-steps",
            "0",
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.count("completed benchmark training run ") == 1
    assert "skipped 1 completed benchmark training manifest(s)" in captured.out
    assert "moved 2 completed benchmark training manifest(s) out of pending" in captured.out
    completed_root = results_root / "training" / "digits" / "completed"
    assert not completed_architecture.exists()
    assert not uncompleted_architecture.exists()
    assert (completed_root / completed_architecture.name).is_file()
    assert (completed_root / uncompleted_architecture.name).is_file()
    assert len(tuple((results_root / "training" / "digits").glob("*.json"))) == 2
    for summary_path in (results_root / "training" / "digits").glob("*.json"):
        record = load_object_document(
            summary_path.read_bytes(),
            description=summary_path.as_posix(),
        )
        architecture_path = Path(cast(str, record["architecture_path"]))
        if architecture_path.is_absolute():
            resolved_architecture_path = architecture_path
        elif architecture_path.parts[:1] == (results_root.name,):
            resolved_architecture_path = results_root.parent / architecture_path
        else:
            resolved_architecture_path = _repository_root / architecture_path
        assert resolved_architecture_path.is_file()
        assert architecture_path.parts[-3:-1] == ("digits", "completed")

    assert (
        main(
            [
                "benchmark",
                "train",
                "digits",
                "--results-root",
                str(results_root),
                "--train-steps",
                "0",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    rerun = capsys.readouterr()
    assert "no uncompleted benchmark training manifests found" in rerun.out


def test_cli_benchmark_evaluate_runs_checkpoint_evaluations_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = tmp_path / "results"
    for seed in (101, 202):
        run_benchmark(
            BenchmarkRunPlan(
                architecture_path=_digits_architecture,
                benchmark_root=_digits_benchmark_root,
                results_root=results_root,
                seed=seed,
                train_steps=0,
                tensor_device="cpu",
            )
        )

    exit_code = main(
        [
            "benchmark",
            "evaluate",
            "digits",
            "--benchmark-root",
            str(_digits_benchmark_root),
            "--results-root",
            str(results_root),
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "completed benchmark evaluation" in captured.out
    assert "materialized 2 benchmark result view(s)" in captured.out
    assert len(tuple((results_root / "evaluations" / "digits").glob("*.json"))) == 2

    assert main(
        [
            "benchmark",
            "evaluate",
            "digits",
            "--benchmark-root",
            str(_digits_benchmark_root),
            "--results-root",
            str(results_root),
            "--device",
            "cpu",
        ]
    ) == 0
    rerun = capsys.readouterr()
    assert "no unevaluated benchmark checkpoints found" in rerun.out
    assert "benchmark result views already current" in rerun.out
    for path in results_root.rglob("*.json"):
        record = load_object_document(path.read_bytes(), description="result record")
        for value in _string_values(record):
            assert not Path(value).is_absolute()


def test_materialize_benchmark_result_views_projects_reference_curves_without_runs(
    tmp_path: Path,
) -> None:
    results_root = tmp_path / "results"

    summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=results_root,
    )

    assert summary.benchmark_count == 2
    assert summary.model_count == 0
    assert summary.run_count == 0
    assert {
        path.relative_to(results_root).as_posix()
        for path in summary.benchmark_view_files
    } == {
        "views/chess/benchmark_results.json",
        "views/digits/benchmark_results.json",
    }
    for view_file in summary.benchmark_view_files:
        view = load_console_result_view(view_file.read_bytes())
        result = cast(list[dict[str, object]], view["benchmark_results"])[0]
        assert result["leaderboard"] == []
        assert result["plot_runs"] == []
        reference_curves = cast(list[dict[str, object]], result["reference_curves"])
        assert reference_curves
        assert reference_curves[0]["x_axis"] == "cost"


def test_cli_publishes_local_benchmark_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = tmp_path / "results"
    _init_git(results_root, configure_identity=False)
    _run_and_evaluate_digits_benchmark(results_root)

    exit_code = main(
        [
            "benchmark",
            "publish",
            "--results-root",
            str(results_root),
            "--no-push",
            "--message",
            "Publish test results",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "published " in captured.out
    assert " measurement(s)" in captured.out
    assert "commit: " in captured.out
    assert len(tuple((results_root / "evaluations" / "digits").glob("*.json"))) == 1
    assert _git(results_root, "status", "--porcelain").stdout == ""


def test_publish_can_commit_results_root_checkout(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _init_git(results_root)
    _run_and_evaluate_digits_benchmark(results_root)
    assert not (results_root / "views").exists()

    summary = publish_local_benchmark_results(
        repository_root=_repository_root,
        results_root=results_root,
        push=False,
        commit_message="Publish test results",
    )

    assert summary.git_commit == _git(results_root, "rev-parse", "HEAD").stdout.strip()
    assert summary.git_pushed is False
    assert _git(results_root, "status", "--porcelain").stdout == ""
    tracked_files = _git(results_root, "ls-files").stdout.splitlines()
    assert "views/digits/benchmark_results.json" not in tracked_files
    assert any(path.startswith("evaluations/digits/") for path in tracked_files)


def test_publish_pushes_by_default(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    remote_root = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(remote_root))
    _init_git(results_root)
    _git(results_root, "remote", "add", "origin", str(remote_root))
    _run_and_evaluate_digits_benchmark(results_root)

    summary = publish_local_benchmark_results(
        repository_root=_repository_root,
        results_root=results_root,
        commit_message="Publish test results",
    )

    assert summary.git_pushed is True
    assert _git(remote_root, "rev-parse", "HEAD").stdout.strip() == summary.git_commit


def test_publish_prefers_hugging_face_api_when_token_is_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "results"
    uploaded_paths: list[str] = []

    class _CommitInfo:
        commit_id = "hf-commit"

    class _CommitOperationAdd:
        def __init__(self, *, path_in_repo: str, path_or_fileobj: str) -> None:
            del path_or_fileobj
            uploaded_paths.append(path_in_repo)

    class _HfApi:
        def create_commit(self, **kwargs: object) -> _CommitInfo:
            assert kwargs["repo_id"] == "operator/leibniz-results"
            assert kwargs["repo_type"] == "dataset"
            assert kwargs["token"] == "hf_test"
            return _CommitInfo()

    class _HfModule:
        CommitOperationAdd = _CommitOperationAdd
        HfApi = _HfApi

    monkeypatch.setattr(local_results, "_hf_api_module", lambda: _HfModule)
    _run_and_evaluate_digits_benchmark(results_root)

    summary = publish_local_benchmark_results(
        repository_root=_repository_root,
        results_root=results_root,
        repo_id="operator/leibniz-results",
        token="hf_test",
        commit_message="Publish test results",
    )

    assert summary.remote == "hf"
    assert summary.remote_commit == "hf-commit"
    assert summary.git_commit is None
    assert "views/digits/benchmark_results.json" not in uploaded_paths
    assert any(path.startswith("evaluations/digits/") for path in uploaded_paths)


def test_push_result_checkout_pushes_existing_commit(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    remote_root = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(remote_root))
    _init_git(results_root)
    _git(results_root, "remote", "add", "origin", str(remote_root))
    (results_root / "README.md").write_text("result checkout\n", encoding="utf-8")
    _git(results_root, "add", "README.md")
    _git(results_root, "commit", "-m", "Prepare checkout")

    summary = push_result_checkout(
        repository_root=_repository_root,
        results_root=results_root,
    )

    assert summary.pushed_commit == _git(results_root, "rev-parse", "HEAD").stdout.strip()
    assert _git(remote_root, "rev-parse", "HEAD").stdout.strip() == summary.pushed_commit


def test_initialize_result_checkout_scaffolds_existing_git_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "results"
    _init_git(results_root)
    calls: list[str] = []

    class _HfApi:
        def create_repo(self, **_kwargs: object) -> None:
            calls.append("create")

    class _HfModule:
        HfApi = _HfApi

        @staticmethod
        def get_token() -> str:
            return "hf_test"

    monkeypatch.setattr(local_results, "_hf_api_module", lambda: _HfModule)

    summary = initialize_result_checkout(
        repo_id="operator/leibniz-results",
        token="hf_test",
        repository_root=_repository_root,
        results_root=results_root,
    )

    assert calls == []
    assert summary.repo_url == "https://huggingface.co/datasets/operator/leibniz-results"
    assert summary.scaffold_commit == _git(results_root, "rev-parse", "HEAD").stdout.strip()
    assert summary.pushed is False
    assert _git(results_root, "status", "--porcelain").stdout == ""
    tracked_files = _git(results_root, "ls-files").stdout.splitlines()
    assert "README.md" in tracked_files
    assert "evaluations/.gitkeep" in tracked_files
    assert "models/.gitkeep" in tracked_files
    assert "training/.gitkeep" in tracked_files
    assert "views/.gitkeep" in tracked_files


def test_initialize_result_checkout_creates_hugging_face_repo_for_plain_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    results_root = tmp_path / "results"
    calls: list[str] = []

    class _HfApi:
        def create_repo(self, **kwargs: object) -> None:
            calls.append(str(kwargs["repo_id"]))

    class _HfModule:
        HfApi = _HfApi

    monkeypatch.setattr(local_results, "_hf_api_module", lambda: _HfModule)

    summary = initialize_result_checkout(
        repo_id="operator/leibniz-results",
        repository_root=_repository_root,
        results_root=results_root,
        token="hf_test",
    )

    assert calls == ["operator/leibniz-results"]
    assert summary.scaffold_commit is None
    assert summary.created_or_reused is True
    assert (results_root / "evaluations" / ".gitkeep").is_file()


def test_initialize_result_checkout_supports_local_only_fallback(tmp_path: Path) -> None:
    results_root = tmp_path / "results"

    summary = initialize_result_checkout(
        repo_id=None,
        token=None,
        repository_root=_repository_root,
        results_root=results_root,
        local_only=True,
    )

    assert summary.repo_id is None
    assert summary.repo_url is None
    assert summary.scaffold_commit is None
    assert summary.pushed is False
    assert (results_root / "evaluations" / ".gitkeep").is_file()


def test_cli_initializes_local_result_checkout_with_default_results_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["benchmark", "init", "--local-only"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "repository: local-only" in captured.out
    assert "results root: " in captured.out
    assert (tmp_path / "results" / "evaluations" / ".gitkeep").is_file()


def _selected_checkpoint_artifact_path(training_summary_path: Path) -> Path:
    training_summary = load_object_document(
        training_summary_path.read_bytes(),
        description="training summary",
    )
    checkpoint = cast(dict[str, object], training_summary["selected_model_checkpoint"])
    path = Path(cast(str, checkpoint["record_path"]))
    if path.parts[:1] == ("results",):
        return training_summary_path.parents[2] / path.relative_to("results")
    return path


def _string_values(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        return tuple(
            string
            for item in mapping.values()
            for string in _string_values(item)
        )
    if isinstance(value, list | tuple):
        sequence = cast(list[object] | tuple[object, ...], value)
        return tuple(string for item in sequence for string in _string_values(item))
    return ()


def _git(path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _init_git(path: Path, *, configure_identity: bool = True) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init")
    if configure_identity:
        _git(path, "config", "user.email", "operator@example.test")
        _git(path, "config", "user.name", "Leibniz Operator")
