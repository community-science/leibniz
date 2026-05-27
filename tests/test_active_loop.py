import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

from leibniz.active_loop import (
    ActiveTrainingLoopError,
    ActiveTrainingLoopPlan,
    run_active_training_loop,
)
from leibniz.benchmark_runner import BenchmarkRunPlan, run_benchmark
from leibniz.cli import main
from leibniz.console.data import ConsoleDataBuilder
from leibniz.local_results import load_console_result_view
from leibniz.work_queues import load_work_queue_items, write_work_queue_item

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
    work_queue_view = load_console_result_view(
        (tmp_path / ".runs" / "views" / "work_queue.json").read_bytes()
    )
    queue_items = cast(list[dict[str, object]], work_queue_view["queue_items"])
    assert [item["status"] for item in queue_items] == ["pending"]


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
    work_queue_view = load_console_result_view(
        (tmp_path / ".runs" / "views" / "work_queue.json").read_bytes()
    )
    queue_items = cast(list[dict[str, object]], work_queue_view["queue_items"])
    assert [item["status"] for item in queue_items] == ["completed"]
    assert queue_items[0]["run_id"]
    assert queue_items[0]["measurement_dataset_path"] == summary.measurement_dataset_paths[
        0
    ].as_posix()
    assert queue_items[0]["candidate_id"]


def test_active_training_loop_runs_bounded_proposal_batch(tmp_path: Path) -> None:
    summary = run_active_training_loop(
        ActiveTrainingLoopPlan(
            benchmark_root=_benchmark_root,
            runs_root=tmp_path / ".runs",
            iterations=1,
            candidate_budget=2,
            candidate_sample_count=8,
            sample_count=1,
            train_steps=0,
        )
    )

    queue_view = load_console_result_view(
        (tmp_path / ".runs" / "views" / "work_queue.json").read_bytes()
    )
    queue_items = cast(list[dict[str, object]], queue_view["queue_items"])

    assert summary.completed_run_count == 2
    assert len(summary.planned_commands) == 2
    assert len(summary.measurement_dataset_paths) == 2
    assert [item["status"] for item in queue_items] == ["completed", "completed"]
    assert [item["sequence"] for item in queue_items] == [0, 1]
    assert len({item["candidate_id"] for item in queue_items}) == 2
    assert len({item["run_id"] for item in queue_items}) == 2


def test_active_training_loop_resumes_pending_dry_run_work(tmp_path: Path) -> None:
    dry_run_summary = run_active_training_loop(
        ActiveTrainingLoopPlan(
            benchmark_root=_benchmark_root,
            runs_root=tmp_path / ".runs",
            dry_run=True,
            candidate_budget=1,
            sample_count=1,
            train_steps=0,
        )
    )
    queue_view = load_console_result_view(
        (tmp_path / ".runs" / "views" / "work_queue.json").read_bytes()
    )
    pending_items = cast(list[dict[str, object]], queue_view["queue_items"])

    run_summary = run_active_training_loop(
        ActiveTrainingLoopPlan(
            benchmark_root=_benchmark_root,
            runs_root=tmp_path / ".runs",
            candidate_budget=1,
            sample_count=1,
            train_steps=0,
        )
    )

    queue_view = load_console_result_view(
        (tmp_path / ".runs" / "views" / "work_queue.json").read_bytes()
    )
    completed_items = cast(list[dict[str, object]], queue_view["queue_items"])
    assert dry_run_summary.planned_commands == run_summary.planned_commands
    assert [item["id"] for item in completed_items] == [item["id"] for item in pending_items]
    assert [item["status"] for item in completed_items] == ["completed"]


def test_active_training_loop_skips_completed_matching_work(tmp_path: Path) -> None:
    runs_root = tmp_path / ".runs"
    run_active_training_loop(
        ActiveTrainingLoopPlan(
            benchmark_root=_benchmark_root,
            runs_root=runs_root,
            dry_run=True,
            candidate_budget=1,
            sample_count=1,
            train_steps=0,
        )
    )
    pending_item = load_work_queue_items(runs_root)[0]
    write_work_queue_item(
        runs_root,
        replace(
            pending_item,
            measurement_dataset_path=runs_root / "measurements" / "existing.json",
            run_id="existing-run",
            status="completed",
        ),
    )

    summary = run_active_training_loop(
        ActiveTrainingLoopPlan(
            benchmark_root=_benchmark_root,
            runs_root=runs_root,
            candidate_budget=1,
            sample_count=1,
            train_steps=0,
        )
    )

    queue_view = load_console_result_view(
        (runs_root / "views" / "work_queue.json").read_bytes()
    )
    queue_items = cast(list[dict[str, object]], queue_view["queue_items"])
    assert summary.completed_run_count == 0
    assert not runs_root.joinpath("measurements").is_dir()
    assert [item["status"] for item in queue_items] == ["completed"]


def test_active_training_loop_blocks_failed_work_without_explicit_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_run(_plan: BenchmarkRunPlan):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr("leibniz.active_loop.run_benchmark", fail_run)
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        run_active_training_loop(
            ActiveTrainingLoopPlan(
                benchmark_root=_benchmark_root,
                runs_root=tmp_path / ".runs",
                candidate_budget=1,
                sample_count=1,
                train_steps=0,
            )
        )

    monkeypatch.setattr("leibniz.active_loop.run_benchmark", run_benchmark)
    with pytest.raises(ActiveTrainingLoopError, match="requires --retry-failed"):
        run_active_training_loop(
            ActiveTrainingLoopPlan(
                benchmark_root=_benchmark_root,
                runs_root=tmp_path / ".runs",
                candidate_budget=1,
                sample_count=1,
                train_steps=0,
            )
        )

    queue_view = load_console_result_view(
        (tmp_path / ".runs" / "views" / "work_queue.json").read_bytes()
    )
    queue_items = cast(list[dict[str, object]], queue_view["queue_items"])
    assert [item["status"] for item in queue_items] == ["failed"]


def test_active_training_loop_retries_failed_work_when_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_run(_plan: BenchmarkRunPlan):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr("leibniz.active_loop.run_benchmark", fail_run)
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        run_active_training_loop(
            ActiveTrainingLoopPlan(
                benchmark_root=_benchmark_root,
                runs_root=tmp_path / ".runs",
                candidate_budget=1,
                sample_count=1,
                train_steps=0,
            )
        )

    monkeypatch.setattr("leibniz.active_loop.run_benchmark", run_benchmark)
    summary = run_active_training_loop(
        ActiveTrainingLoopPlan(
            benchmark_root=_benchmark_root,
            runs_root=tmp_path / ".runs",
            candidate_budget=1,
            sample_count=1,
            train_steps=0,
            retry_failed=True,
        )
    )

    queue_view = load_console_result_view(
        (tmp_path / ".runs" / "views" / "work_queue.json").read_bytes()
    )
    queue_items = cast(list[dict[str, object]], queue_view["queue_items"])
    assert summary.completed_run_count == 1
    assert [item["status"] for item in queue_items] == ["completed"]
    assert "error" not in queue_items[0]


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
    work_queue_view = next(
        view for view in result_views if view["format"] == "leibniz.console.work-queue"
    )
    queue_items = cast(list[dict[str, object]], work_queue_view["queue_items"])
    assert work_queue_view["source_path"] == (
        runs_root / "views" / "work_queue.json"
    ).as_posix()
    assert [item["status"] for item in queue_items] == ["completed"]
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
    assert len(leaderboard) == 1
    assert leaderboard[0]["observed_complexities"] == [1.0]
    assert runs_root.joinpath("measurements").is_dir()
    assert runs_root.joinpath("proposals").is_dir()
    assert runs_root.joinpath("views", "benchmark_results.json").is_file()


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
    work_queue_view = load_console_result_view(
        (tmp_path / ".runs" / "views" / "work_queue.json").read_bytes()
    )
    queue_items = cast(list[dict[str, object]], work_queue_view["queue_items"])
    assert [item["status"] for item in queue_items] == ["failed"]
    assert queue_items[0]["error"] == "synthetic training failure"


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


def test_cli_runs_active_frontier_shakedown(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs_root = tmp_path / ".runs"

    exit_code = main(
        [
            "benchmark",
            "shakedown",
            "--benchmark-root",
            str(_benchmark_root),
            "--runs-root",
            str(runs_root),
            "--candidate-budget",
            "1",
            "--sample-count",
            "1",
            "--train-steps",
            "0",
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
    assert runs_root.joinpath("views", "benchmark_results.json").is_file()
    assert runs_root.joinpath("views", "work_queue.json").is_file()
