import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

from leibniz.active_loop import ActiveTrainingLoopPlan, run_active_training_loop
from leibniz.benchmark_runner import BenchmarkRunPlan, run_benchmark
from leibniz.cli import main
from leibniz.console.data import ConsoleDataBuilder

_repository_root = Path(__file__).parents[1]
_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_architecture_path = (
    _repository_root / "tests" / "fixtures" / "architecture" / "digits_pool" / "manifest.json"
)


def test_active_training_loop_dry_run_plans_one_training_session(
    tmp_path: Path,
) -> None:
    summary = run_active_training_loop(
        ActiveTrainingLoopPlan(
            benchmark_root=_benchmark_root,
            results_root=tmp_path / "results",
            dry_run=True,
            sample_count=1,
        )
    )

    assert summary.dry_run is True
    assert summary.completed_run_count == 0
    assert len(summary.planned_commands) == 1
    assert summary.planned_commands[0][:3] == ("leibniz", "benchmark", "run")
    assert summary.proposal_set_paths[0].exists()
    assert summary.measurement_dataset_paths == ()
    assert not (tmp_path / "results" / "measurements").exists()


def test_active_training_loop_runs_one_training_session_and_refreshes_results(
    tmp_path: Path,
) -> None:
    summary = run_active_training_loop(
        ActiveTrainingLoopPlan(
            benchmark_root=_benchmark_root,
            results_root=tmp_path / "results",
            sample_count=1,
            train_steps=0,
            tensor_device="cpu",
        )
    )

    assert summary.completed_run_count == 1
    assert len(summary.planned_commands) == 1
    assert len(summary.measurement_dataset_paths) == 1
    assert summary.measurement_dataset_paths[0].exists()
    assert summary.result_view_path is not None
    assert summary.result_view_path.exists()


def test_active_training_loop_always_generates_one_proposal(
    tmp_path: Path,
) -> None:
    summary = run_active_training_loop(
        ActiveTrainingLoopPlan(
            benchmark_root=_benchmark_root,
            results_root=tmp_path / "results",
            candidate_sample_count=8,
            sample_count=1,
            train_steps=0,
            tensor_device="cpu",
        )
    )

    assert summary.completed_run_count == 1
    assert len(summary.planned_commands) == 1
    assert len(summary.measurement_dataset_paths) == 1


def test_cli_active_loop_outputs_feed_console_data(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    environment = {
        **os.environ,
        "PYTHONPATH": str(_repository_root / "src"),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "leibniz.cli",
            "benchmark",
            "loop",
            "--benchmark-root",
            str(_benchmark_root),
            "--results-root",
            str(results_root),
            "--sample-count",
            "1",
            "--train-steps",
            "0",
            "--device",
            "cpu",
        ],
        capture_output=True,
        check=False,
        cwd=_repository_root,
        env=environment,
        text=True,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert "training " in result.stdout
    assert "validation_loss=" in result.stdout
    assert "completed active benchmark loop" in result.stdout

    record = ConsoleDataBuilder(_repository_root).discover(
        (PurePosixPath("tests/fixtures"), PurePosixPath("src/leibniz/benchmarks")),
        result_roots=(results_root / "views",),
    ).to_record()
    result_views = cast(list[dict[str, object]], record["result_views"])
    benchmark_view = next(
        view for view in result_views if view["format"] == "leibniz.console.benchmark-results"
    )
    benchmark_results = cast(list[dict[str, object]], benchmark_view["benchmark_results"])
    digits_result = next(
        result
        for result in benchmark_results
        if result["benchmark_id"] == "benchmarks.digits@0.1.0"
    )
    assert benchmark_view["source_path"] == (
        results_root / "views" / "benchmark_results.json"
    ).as_posix()
    training_history = cast(list[dict[str, object]], digits_result["training_history"])
    proposals = cast(list[dict[str, object]], digits_result["proposals"])
    leaderboard = cast(list[dict[str, object]], digits_result["leaderboard"])
    proposal = proposals[0]
    assert len(training_history) == 1
    assert training_history[0]["source_kind"] == "local-run"
    assert len(proposals) == 1
    assert proposal["selector_name"] == "resource-bootstrap"
    assert proposal["source_candidate_rank"]
    assert proposal["command"]
    command = cast(list[str], proposal["command"])
    assert "--scale" not in command
    assert "--scale-curriculum" not in command
    assert "--curriculum-max-scale" not in command
    assert len(leaderboard) == 1
    observed_complexities = cast(list[float], leaderboard[0]["observed_complexities"])
    assert observed_complexities[0] == pytest.approx(26.981617448931395)
    assert observed_complexities == sorted(observed_complexities)
    assert results_root.joinpath("measurements").is_dir()
    assert results_root.joinpath("proposals").is_dir()
    assert results_root.joinpath("views", "benchmark_results.json").is_file()


def test_active_training_loop_preserves_existing_measurements_on_run_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_architecture_path,
            benchmark_root=_benchmark_root,
            results_root=tmp_path / "results",
            sample_count=1,
            train_steps=0,
            tensor_device="cpu",
        )
    )
    before = initial.measurement_dataset_path.read_bytes()

    def fail_run(
        _plan: BenchmarkRunPlan,
        *,
        progress_callback: object | None = None,
    ):
        del progress_callback
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr("leibniz.active_loop.run_benchmark", fail_run)

    with pytest.raises(RuntimeError, match="synthetic training failure"):
        run_active_training_loop(
            ActiveTrainingLoopPlan(
                benchmark_root=_benchmark_root,
                results_root=tmp_path / "results",
                sample_count=1,
                train_steps=0,
            )
        )

    assert initial.measurement_dataset_path.read_bytes() == before


def test_cli_runs_active_training_loop_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "benchmark",
            "loop",
            "--benchmark-root",
            str(_benchmark_root),
            "--results-root",
            str(tmp_path / "results"),
            "--sample-count",
            "1",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "planned active benchmark loop" in captured.out
    assert "command: leibniz benchmark run" in captured.out


def test_cli_runs_active_frontier_shakedown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = tmp_path / "results"

    exit_code = main(
        [
            "benchmark",
            "shakedown",
            "--benchmark-root",
            str(_benchmark_root),
            "--results-root",
            str(results_root),
            "--sample-count",
            "1",
            "--train-steps",
            "0",
            "--device",
            "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "completed active frontier shakedown for benchmarks.digits@0.1.0" in captured.out
    assert "runs: 0 -> 1 (+1)" in captured.out
    assert "models: 0 -> 1 (+1)" in captured.out
    assert "best score: n/a -> " in captured.out
    assert "view: " in captured.out
    assert results_root.joinpath("views", "benchmark_results.json").is_file()
