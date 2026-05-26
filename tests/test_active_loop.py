import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

from leibniz.active_loop import (
    ActiveTrainingLoopPlan,
    run_active_training_loop,
)
from leibniz.benchmark_runner import BenchmarkRunPlan, run_benchmark
from leibniz.cli import main
from leibniz.console.data import ConsoleDataBuilder

_repository_root = Path(__file__).parents[1]
_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_architecture_path = (
    _repository_root / "tests" / "fixtures" / "architecture" / "digits_pool" / "manifest.json"
)


def test_active_training_loop_dry_run_plans_without_training(tmp_path: Path) -> None:
    summary = run_active_training_loop(
        ActiveTrainingLoopPlan(
            benchmark_root=_benchmark_root,
            runs_root=tmp_path / ".runs",
            dry_run=True,
            candidate_budget=1,
            sample_count=1,
        )
    )

    assert summary.dry_run is True
    assert summary.completed_run_count == 0
    assert len(summary.planned_commands) == 1
    assert summary.planned_commands[0][:3] == ("leibniz", "benchmark", "run")
    assert summary.proposal_set_paths[0].exists()
    assert not (tmp_path / ".runs" / "measurements").exists()


def test_active_training_loop_runs_one_iteration_and_refreshes_results(tmp_path: Path) -> None:
    summary = run_active_training_loop(
        ActiveTrainingLoopPlan(
            benchmark_root=_benchmark_root,
            runs_root=tmp_path / ".runs",
            iterations=1,
            candidate_budget=1,
            sample_count=1,
            train_steps=0,
        )
    )

    assert summary.completed_run_count == 1
    assert summary.measurement_dataset_paths[0].exists()
    assert summary.result_view_path is not None
    assert summary.result_view_path.exists()


def test_cli_active_loop_outputs_feed_console_data(tmp_path: Path) -> None:
    runs_root = tmp_path / ".runs"
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
            "--runs-root",
            str(runs_root),
            "--iterations",
            "1",
            "--candidate-budget",
            "1",
            "--sample-count",
            "1",
            "--train-steps",
            "0",
        ],
        capture_output=True,
        check=False,
        cwd=_repository_root,
        env=environment,
        text=True,
    )

    assert result.returncode == 0
    assert result.stderr == ""
    assert "completed active benchmark loop" in result.stdout

    record = ConsoleDataBuilder(_repository_root).discover(
        (PurePosixPath("tests/fixtures"), PurePosixPath("src/leibniz/benchmarks")),
        result_roots=(runs_root / "views",),
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
        runs_root / "views" / "benchmark_results.json"
    ).as_posix()
    assert len(cast(list[dict[str, object]], digits_result["training_history"])) == 1
    assert len(cast(list[dict[str, object]], digits_result["proposals"])) == 1


def test_active_training_loop_preserves_existing_measurements_on_run_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_architecture_path,
            benchmark_root=_benchmark_root,
            runs_root=tmp_path / ".runs",
            sample_count=1,
            train_steps=0,
        )
    )
    before = initial.measurement_dataset_path.read_bytes()

    def fail_run(_plan: BenchmarkRunPlan):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr("leibniz.active_loop.run_benchmark", fail_run)

    with pytest.raises(RuntimeError, match="synthetic training failure"):
        run_active_training_loop(
            ActiveTrainingLoopPlan(
                benchmark_root=_benchmark_root,
                runs_root=tmp_path / ".runs",
                iterations=1,
                candidate_budget=1,
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
            "--runs-root",
            str(tmp_path / ".runs"),
            "--candidate-budget",
            "1",
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
