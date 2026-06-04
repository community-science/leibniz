import math
import subprocess
from pathlib import Path
from typing import Any, cast

import pytest

import leibniz.local_results as local_results
from leibniz.architectures import ArchitectureManifestDocument
from leibniz.benchmark_runner import (
    BenchmarkEvaluationPlan,
    BenchmarkRunPlan,
    evaluate_benchmark_checkpoint,
    run_benchmark,
)
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.cli import main
from leibniz.content import ContentDigest
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
from leibniz.measurements import MeasurementDataset, MeasurementDocument

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
            sample_count=sample_count,
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


def test_competent_complexity_score_integrates_bits_above_chance() -> None:
    complexity = 20.0
    chance_mass = 0.1

    assert math.isclose(
        local_results.competent_complexity_score(
            ({"complexity": complexity, "score": 1.0},),
            chance_mass=chance_mass,
        ),
        20.0,
    )
    assert math.isclose(
        local_results.competent_complexity_score(
            ({"complexity": complexity * 2.0, "score": 1.0},),
            chance_mass=chance_mass,
        ),
        40.0,
    )
    assert math.isclose(
        local_results.competent_complexity_score(
            ({"complexity": complexity, "score": 0.55},),
            chance_mass=chance_mass,
        ),
        10.0,
    )
    assert math.isclose(
        local_results.competent_complexity_score(
            ({"complexity": complexity * 4.0, "score": chance_mass},),
            chance_mass=chance_mass,
        ),
        0.0,
    )


def test_console_result_view_rejects_wrong_format() -> None:
    with pytest.raises(LocalResultImportError, match="unsupported format"):
        load_console_result_view(canonical_document_bytes({"format": "other", "format_version": 1}))


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


def test_relative_competition_does_not_read_absolute_measurements() -> None:
    run_a = _benchmark_run_record_for_competition(
        model_key="model-a",
        run_id="run-a",
    )
    run_b = _benchmark_run_record_for_competition(
        model_key="model-b",
        run_id="run-b",
    )

    assert cast(Any, local_results)._pairwise_competition_outcomes(
        {"model-a": run_a, "model-b": run_b},
        (),
    ) == ()

    outcomes = cast(Any, local_results)._pairwise_competition_outcomes(
        {"model-a": run_a, "model-b": run_b},
        (_competition_record(left_model_key="model-a", right_model_key="model-b"),),
    )

    assert len(outcomes) == 1
    assert outcomes[0].left_model_key == "model-a"
    assert outcomes[0].right_model_key == "model-b"
    assert outcomes[0].left_score == 1.0
    assert outcomes[0].right_score == 0.0


def test_relative_competition_scores_rank_undefeated_model_first() -> None:
    best_runs = {
        "model-a": _benchmark_run_record_for_competition(
            model_key="model-a",
            run_id="run-a",
        ),
        "model-b": _benchmark_run_record_for_competition(
            model_key="model-b",
            run_id="run-b",
        ),
        "model-c": _benchmark_run_record_for_competition(
            model_key="model-c",
            run_id="run-c",
        ),
    }
    outcomes = (
        cast(Any, local_results)._ModelCompetitionOutcome(
            left_model_key="model-a",
            right_model_key="model-b",
            left_score=0.64,
            right_score=0.36,
            sample_count=512,
        ),
        cast(Any, local_results)._ModelCompetitionOutcome(
            left_model_key="model-a",
            right_model_key="model-c",
            left_score=0.71,
            right_score=0.29,
            sample_count=512,
        ),
        cast(Any, local_results)._ModelCompetitionOutcome(
            left_model_key="model-b",
            right_model_key="model-c",
            left_score=0.87,
            right_score=0.13,
            sample_count=512,
        ),
    )

    ratings = cast(Any, local_results)._relative_rating_fit(
        best_runs,
        outcomes=outcomes,
    ).ratings

    assert ratings["model-a"].score > ratings["model-b"].score
    assert ratings["model-b"].score > ratings["model-c"].score


def test_relative_competition_batch_fit_aggregates_reversed_pairs() -> None:
    best_runs = {
        "model-a": _benchmark_run_record_for_competition(
            model_key="model-a",
            run_id="run-a",
        ),
        "model-b": _benchmark_run_record_for_competition(
            model_key="model-b",
            run_id="run-b",
        ),
    }
    outcome_type = cast(Any, local_results)._ModelCompetitionOutcome
    ratings = cast(Any, local_results)._relative_rating_fit(
        best_runs,
        outcomes=(
            outcome_type(
                left_model_key="model-a",
                right_model_key="model-b",
                left_score=0.75,
                right_score=0.25,
                sample_count=40,
            ),
            outcome_type(
                left_model_key="model-b",
                right_model_key="model-a",
                left_score=0.25,
                right_score=0.75,
                sample_count=24,
            ),
        ),
    ).ratings

    assert ratings["model-a"].score > ratings["model-b"].score
    assert ratings["model-a"].sample_count == 64
    assert ratings["model-a"].opponent_count == 1
    assert ratings["model-a"].competition_count == 2
    assert ratings["model-a"].provisional
    assert ratings["model-a"].uncertainty > 0.0


def test_relative_competition_batch_fit_uses_transitive_evidence() -> None:
    best_runs = {
        "model-a": _benchmark_run_record_for_competition(
            model_key="model-a",
            run_id="run-a",
        ),
        "model-b": _benchmark_run_record_for_competition(
            model_key="model-b",
            run_id="run-b",
        ),
        "model-c": _benchmark_run_record_for_competition(
            model_key="model-c",
            run_id="run-c",
        ),
    }
    outcome_type = cast(Any, local_results)._ModelCompetitionOutcome
    ratings = cast(Any, local_results)._relative_rating_fit(
        best_runs,
        outcomes=(
            outcome_type(
                left_model_key="model-a",
                right_model_key="model-b",
                left_score=0.60,
                right_score=0.40,
                sample_count=128,
            ),
            outcome_type(
                left_model_key="model-b",
                right_model_key="model-c",
                left_score=0.60,
                right_score=0.40,
                sample_count=128,
            ),
            outcome_type(
                left_model_key="model-a",
                right_model_key="model-c",
                left_score=0.70,
                right_score=0.30,
                sample_count=128,
            ),
        ),
    ).ratings

    assert ratings["model-a"].score > ratings["model-b"].score
    assert ratings["model-b"].score > ratings["model-c"].score
    assert not ratings["model-a"].provisional
    assert not ratings["model-b"].provisional
    assert not ratings["model-c"].provisional


def test_relative_score_view_exposes_batch_rating_evidence() -> None:
    best_runs = {
        "model-a": _benchmark_run_record_for_competition(
            model_key="model-a",
            run_id="run-a",
        ),
        "model-b": _benchmark_run_record_for_competition(
            model_key="model-b",
            run_id="run-b",
        ),
    }
    cost_summary = {
        "component_count": 1,
        "parameter_count": 1,
        "storage_bytes": 1,
        "inference_compute": 1,
        "training_compute": 1,
    }
    records: list[dict[str, object]] = [
        {
            "model_key": "model-a",
            "score_views": {},
            "cost_summary": cost_summary,
        },
        {
            "model_key": "model-b",
            "score_views": {},
            "cost_summary": cost_summary,
        },
    ]

    cast(Any, local_results)._add_relative_score_views(
        records,
        best_runs=best_runs,
        competitions=(
            _competition_record(left_model_key="model-a", right_model_key="model-b"),
        ),
    )

    score_views = cast(dict[str, object], records[0]["score_views"])
    relative = cast(dict[str, object], score_views["relative"])
    basis = cast(dict[str, object], relative["basis"])
    assert basis["kind"] == "model-competition-bradley-terry-batch-v1"
    assert basis["competition_count"] == 1
    assert basis["sample_count"] == 1
    assert basis["opponent_count"] == 1
    assert basis["provisional"] is True
    assert cast(float, basis["rating_uncertainty"]) > 0.0
    confidence = cast(dict[str, object], basis["frontier_confidence"])
    size_confidence = cast(dict[str, object], confidence["storage_bytes"])
    assert size_confidence["risk_threshold"] == 0.05


def test_relative_frontier_confidence_requests_uncertain_nearest_competition() -> None:
    best_runs = {
        "model-a": _benchmark_run_record_for_competition(
            model_key="model-a",
            run_id="run-a",
        ),
        "model-b": _benchmark_run_record_for_competition(
            model_key="model-b",
            run_id="run-b",
        ),
    }
    outcome_type = cast(Any, local_results)._ModelCompetitionOutcome

    requests = cast(Any, local_results)._relative_frontier_competition_requests(
        best_runs,
        outcomes=(
            outcome_type(
                left_model_key="model-a",
                right_model_key="model-b",
                left_score=0.55,
                right_score=0.45,
                sample_count=8,
            ),
        ),
    )

    assert requests == (("model-a", "model-b"),)


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


def test_console_result_view_validates_training_protocol_gate_cadence(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=results_root,
            sample_count=1,
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
    assert history[0]["result_status"] == "tentative"
    protocol = cast(dict[str, object], history[0]["training_diagnostics"])["protocol"]
    cast(dict[str, object], protocol)["gate_check_interval"] = 0

    with pytest.raises(
        LocalResultImportError,
        match="gate_check_interval",
    ):
        load_console_result_view(canonical_document_bytes(view))


def test_materialize_benchmark_result_views_projects_evaluation_bundles(
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

    summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=results_root,
    )

    assert summary.benchmark_count == 1
    assert summary.model_count == 1
    assert summary.run_count == 1
    assert summary.view_file == (
        tmp_path / "results" / "views" / "digits" / "benchmark_results.json"
    )
    assert summary.benchmark_view_files == (summary.view_file,)

    view = load_console_result_view(summary.view_file.read_bytes())
    assert view["format"] == "leibniz.console.benchmark-results"
    results = cast(list[dict[str, object]], view["benchmark_results"])
    result = results[0]
    assert result["benchmark_id"] == "benchmarks.digits@0.1.0"
    leaderboard = cast(list[dict[str, object]], result["leaderboard"])
    assert leaderboard[0]["measurement_count"] == 1
    cost_summary = cast(dict[str, object], leaderboard[0]["cost_summary"])
    assert isinstance(cost_summary["inference_compute"], int | float)
    frontiers = cast(dict[str, object], result["frontiers"])
    assert len(cast(list[object], frontiers["inference_compute"])) == 1
    model_view = cast(dict[str, object], leaderboard[0]["console_view_model"])
    model_sections = cast(list[dict[str, object]], model_view["detail_sections"])
    assert [section["title"] for section in model_sections] == [
        "Model Contract",
        "Architecture Graph",
        "Evidence",
        "Resources",
    ]
    cost_summary = cast(dict[str, object], leaderboard[0]["cost_summary"])
    assert "parameter_count" not in cost_summary
    assert cost_summary["storage_bytes"] == 200
    frontiers = cast(dict[str, object], result["frontiers"])
    assert len(cast(list[dict[str, object]], frontiers["storage_bytes"])) == 1
    history = cast(list[dict[str, object]], result["training_history"])
    assert history[0]["source_kind"] == "local-run"
    assert "training_diagnostics" not in history[0]
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
            sample_count=1,
            evaluation_sample_count=1,
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
    assert "completed benchmark absolute evaluation" in captured.out
    assert "materialized 1 benchmark result view(s)" in captured.out
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
    assert "materialized 1 benchmark result view(s)" in rerun.out
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
                "--sample-count",
                "1",
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
                "--sample-count",
                "1",
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


def test_cli_benchmark_evaluate_runs_absolute_and_relative_phases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import leibniz.cli as cli

    def no_confidence_requests(
        *,
        results_root: Path,
        benchmark_selectors: tuple[str, ...],
    ) -> tuple[tuple[str, str], ...]:
        return ()

    monkeypatch.setattr(cli, "relative_frontier_competition_requests", no_confidence_requests)
    results_root = tmp_path / "results"
    for seed in (101, 202):
        run_benchmark(
            BenchmarkRunPlan(
                architecture_path=_digits_architecture,
                benchmark_root=_digits_benchmark_root,
                results_root=results_root,
                sample_count=1,
                evaluation_sample_count=1,
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
    assert "completed benchmark absolute evaluation" in captured.out
    assert "completed benchmark relative evaluation" in captured.out
    assert "materialized 1 benchmark result view(s)" in captured.out
    assert len(tuple((results_root / "evaluations" / "digits").glob("*.json"))) == 2
    competition_paths = tuple(
        (results_root / "evaluations" / "digits" / "competitions").glob("*.json")
    )
    assert len(competition_paths) == 1

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
    assert "no missing benchmark relative evaluations found" in rerun.out
    assert "materialized 1 benchmark result view(s)" in rerun.out
    competition_paths = tuple(
        (results_root / "evaluations" / "digits" / "competitions").glob("*.json")
    )
    assert len(competition_paths) == 1
    for path in results_root.rglob("*.json"):
        record = load_object_document(path.read_bytes(), description="result record")
        for value in _string_values(record):
            assert not Path(value).is_absolute()


def test_materialize_benchmark_result_views_rejects_empty_results_root(tmp_path: Path) -> None:
    with pytest.raises(LocalResultImportError, match="no benchmark result records"):
        materialize_benchmark_result_views(
            repository_root=_repository_root,
            results_root=tmp_path / "results",
        )


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
    assert "published 1 measurement(s)" in captured.out
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


def _digits_dataset() -> MeasurementDataset:
    return MeasurementDataset.from_record({"measurements": [_digits_measurement().to_record()]})


def _selected_checkpoint_artifact_path(training_summary_path: Path) -> Path:
    training_summary = load_object_document(
        training_summary_path.read_bytes(),
        description="training summary",
    )
    checkpoint = cast(dict[str, object], training_summary["selected_model_checkpoint"])
    return Path(cast(str, checkpoint["record_path"]))


def _benchmark_run_record_for_competition(
    *,
    model_key: str,
    run_id: str,
) -> object:
    dataset = _digits_dataset()
    architecture = _architecture().manifest
    digest = ContentDigest.from_value({"model": model_key})
    return cast(Any, local_results)._BenchmarkRunRecord(
        source_kind="test",
        result_status="accepted",
        source_path=Path(f"results/training/{run_id}.json"),
        run_id=run_id,
        run_slug=run_id,
        benchmark_id=ProtocolIdentifier.parse("benchmarks.digits@0.1.0"),
        architecture_digest=digest,
        model_key=model_key,
        complexity=1.0,
        measurement_count=len(dataset.measurements),
        score=1.0,
        cost_summary={"component_count": 1, "storage_bytes": 1},
        architecture=architecture.to_record(),
        model_inspection={},
        model_inspection_digest=digest,
        model_inspection_path=None,
        measurement_dataset=dataset,
        measurement_dataset_digest=dataset.digest,
    )


def _competition_record(*, left_model_key: str, right_model_key: str) -> dict[str, object]:
    return {
        "format": "leibniz.model-competition",
        "format_version": 1,
        "benchmark_id": "benchmarks.digits@0.1.0",
        "competition_id": "model-a-vs-model-b",
        "mechanic": "paired-prediction-accepted-mass",
        "seed": 9000110,
        "sample_count": 1,
        "outcome_space_id": "benchmarks.digits.outcomes@0.1.0",
        "left_model_key": left_model_key,
        "right_model_key": right_model_key,
        "left_score": 1.0,
        "right_score": 0.0,
        "left_wins": 1,
        "right_wins": 0,
        "ties": 0,
        "entries": [
            {
                "id": "benchmarks.digits.competition.model-a-vs-model-b.sample-0@0.1.0",
                "observation_id": "benchmarks.digits.observations.competition.sample-0@0.1.0",
                "accepted_outcome_id": "digit-7",
                "left_score": 0.9,
                "right_score": 0.1,
                "winner": "left",
            }
        ],
    }


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


def _digits_measurement():
    return MeasurementDocument.from_bytes(
        canonical_document_bytes(_digits_measurement_record())
    ).measurement


def _digits_measurement_record() -> dict[str, object]:
    outcome_space = _digits_benchmark().manifest.resolve_outcome_space()
    return {
        "benchmark_id": "benchmarks.digits@0.1.0",
        "outcome_space": outcome_space.to_record(),
        "accepted_event": {
            "id": "benchmarks.digits.accepted.digit-7@0.1.0",
            "outcome_space_id": str(outcome_space.id),
            "outcomes": ["digit-7"],
        },
        "probability_measure": {
            "id": "benchmarks.digits.prediction.digit-7@0.1.0",
            "outcome_space_id": str(outcome_space.id),
            "probabilities": [
                {"outcome_id": f"digit-{digit}", "probability": 1.0 if digit == 7 else 0.0}
                for digit in range(10)
            ],
        },
        "raw_scoring_evidence": {
            "id": "benchmarks.digits.measurements.digit-7@0.1.0",
            "observation_id": "digits-l1-seed-7",
            "outcome_space_id": str(outcome_space.id),
            "accepted_event_id": "benchmarks.digits.accepted.digit-7@0.1.0",
            "probability_measure_id": "benchmarks.digits.prediction.digit-7@0.1.0",
            "accepted_mass": 1.0,
            "negative_log_score": 0.0,
        },
    }


def _digits_benchmark() -> BenchmarkManifestDocument:
    manifest_path = _repository_root / "src" / "leibniz" / "benchmarks" / "digits" / "manifest.json"
    return BenchmarkManifestDocument.from_bytes(
        manifest_path.read_bytes()
    )


def _architecture() -> ArchitectureManifestDocument:
    manifest_path = (
        _repository_root / "tests" / "fixtures" / "architecture" / "digits_pool.json"
    )
    return ArchitectureManifestDocument.from_bytes(
        manifest_path.read_bytes()
    )
