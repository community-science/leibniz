import math
import subprocess
from pathlib import Path
from typing import cast

import pytest

import leibniz.local_results as local_results
from leibniz.architectures import ArchitectureManifestDocument
from leibniz.benchmark_runner import BenchmarkRunPlan, run_benchmark
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.cli import main
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.local_results import (
    LocalResultImportError,
    import_submission_publications,
    initialize_publication_checkout,
    load_console_result_view,
    materialize_benchmark_result_views,
    publish_local_benchmark_results,
    push_publication_checkout,
)
from leibniz.measurements import MeasurementDataset, MeasurementDocument
from leibniz.publications import SubmissionPublicationDocument
from leibniz.views import MeasurementScoreView

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_digits_architecture = (
    _repository_root / "tests" / "fixtures" / "architecture" / "digits_pool" / "manifest.json"
)


def test_base_normalized_absolute_score_rewards_complexity_above_chance() -> None:
    base_complexity = 20.0
    chance_mass = 0.1

    assert math.isclose(
        local_results.base_normalized_absolute_score(
            ({"complexity": base_complexity, "score": 1.0},),
            base_complexity=base_complexity,
            chance_mass=chance_mass,
        ),
        1.0,
    )
    assert math.isclose(
        local_results.base_normalized_absolute_score(
            ({"complexity": base_complexity * 2.0, "score": 1.0},),
            base_complexity=base_complexity,
            chance_mass=chance_mass,
        ),
        2.0,
    )
    assert math.isclose(
        local_results.base_normalized_absolute_score(
            ({"complexity": base_complexity, "score": 0.55},),
            base_complexity=base_complexity,
            chance_mass=chance_mass,
        ),
        0.5,
    )
    assert math.isclose(
        local_results.base_normalized_absolute_score(
            ({"complexity": base_complexity * 4.0, "score": chance_mass},),
            base_complexity=base_complexity,
            chance_mass=chance_mass,
        ),
        0.0,
    )


def test_import_submission_publications_materializes_runs_views(tmp_path: Path) -> None:
    source_root = tmp_path / "hf-checkout"
    source_root.mkdir()
    bundle_path = source_root / "digits_publication.json"
    bundle_path.write_bytes(canonical_document_bytes(_digits_publication_bundle_record()))

    summary = import_submission_publications(
        (source_root,),
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )

    assert summary.publication_bundle_count == 1
    assert summary.measurement_count == 1
    assert summary.view_file == tmp_path / "results" / "views" / "imported_results.json"
    assert len(summary.import_files) == 1

    imported_bundle = SubmissionPublicationDocument.from_bytes(
        summary.import_files[0].read_bytes()
    ).bundle
    assert imported_bundle.id == ProtocolIdentifier.parse("publication-bundles.digits@0.1.0")

    view = load_console_result_view(summary.view_file.read_bytes())
    assert view["format"] == "leibniz.console.imported-results"
    bundles = cast(list[dict[str, object]], view["publication_bundles"])
    assert bundles[0]["id"] == "publication-bundles.digits@0.1.0"
    assert bundles[0]["benchmark_ids"] == ["benchmarks.digits@0.1.0"]
    assert bundles[0]["measurement_count"] == 1


def test_import_submission_publications_ignores_non_publication_json(tmp_path: Path) -> None:
    source_root = tmp_path / "hf-checkout"
    source_root.mkdir()
    (source_root / "not-a-publication.json").write_bytes(
        canonical_document_bytes({"measurement_dataset": {"measurements": []}})
    )

    with pytest.raises(LocalResultImportError, match="no publication bundle"):
        import_submission_publications(
            (source_root,),
            repository_root=_repository_root,
            results_root=tmp_path / "results",
        )


def test_console_result_view_rejects_wrong_format() -> None:
    with pytest.raises(LocalResultImportError, match="unsupported format"):
        load_console_result_view(canonical_document_bytes({"format": "other", "format_version": 1}))


def test_console_result_view_validates_embedded_model_inspections(tmp_path: Path) -> None:
    source_root = tmp_path / "hf-checkout"
    source_root.mkdir()
    bundle_path = source_root / "digits_publication.json"
    bundle_path.write_bytes(canonical_document_bytes(_digits_publication_bundle_record()))
    import_submission_publications(
        (source_root,),
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )
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
    source_root = tmp_path / "hf-checkout"
    source_root.mkdir()
    bundle_path = source_root / "digits_publication.json"
    bundle_path.write_bytes(canonical_document_bytes(_digits_publication_bundle_record()))
    import_submission_publications(
        (source_root,),
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )
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
    source_root = tmp_path / "hf-checkout"
    source_root.mkdir()
    bundle_path = source_root / "digits_publication.json"
    bundle_path.write_bytes(canonical_document_bytes(_digits_publication_bundle_record()))
    import_submission_publications(
        (source_root,),
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )
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
    source_root = tmp_path / "hf-checkout"
    source_root.mkdir()
    bundle_path = source_root / "digits_publication.json"
    bundle_path.write_bytes(canonical_document_bytes(_digits_publication_bundle_record()))
    import_submission_publications(
        (source_root,),
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )
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


def test_materialize_benchmark_result_views_projects_imported_publications(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "hf-checkout"
    source_root.mkdir()
    bundle_path = source_root / "digits_publication.json"
    bundle_path.write_bytes(canonical_document_bytes(_digits_publication_bundle_record()))
    import_submission_publications(
        (source_root,),
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )

    summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )

    assert summary.benchmark_count == 1
    assert summary.model_count == 1
    assert summary.run_count == 1
    assert summary.view_file == tmp_path / "results" / "views" / "benchmark_results.json"

    view = load_console_result_view(summary.view_file.read_bytes())
    assert view["format"] == "leibniz.console.benchmark-results"
    results = cast(list[dict[str, object]], view["benchmark_results"])
    result = results[0]
    assert result["benchmark_id"] == "benchmarks.digits@0.1.0"
    leaderboard = cast(list[dict[str, object]], result["leaderboard"])
    assert leaderboard[0]["score"] == 0.0
    assert leaderboard[0]["observed_complexities"] == []
    model_view = cast(dict[str, object], leaderboard[0]["console_view_model"])
    model_sections = cast(list[dict[str, object]], model_view["detail_sections"])
    assert [section["title"] for section in model_sections] == [
        "Model Contract",
        "Architecture Graph",
        "Evidence",
        "Resources",
    ]
    contract_entries = cast(list[dict[str, object]], model_sections[0]["entries"])
    assert contract_entries[1] == {
        "label": "Prediction Space",
        "value": "finite outcome space with 10 outcomes",
    }
    graph_entries = cast(list[dict[str, object]], model_sections[1]["entries"])
    assert graph_entries[0] == {"label": "Components", "value": "3"}
    evidence_entries = cast(list[dict[str, object]], model_sections[2]["entries"])
    assert evidence_entries[0] == {"label": "Node Evidence", "value": "3"}
    cost_summary = cast(dict[str, object], leaderboard[0]["cost_summary"])
    assert cost_summary["parameter_count"] == 50
    frontiers = cast(dict[str, object], result["frontiers"])
    assert len(cast(list[dict[str, object]], frontiers["parameter_count"])) == 1
    history = cast(list[dict[str, object]], result["training_history"])
    assert history[0]["source_kind"] == "imported-publication"
    inspections = cast(list[dict[str, object]], result["model_inspections"])
    assert len(inspections) == 1
    assert inspections[0]["source_path"] == history[0]["source_path"]
    assert "measurement_dataset" in inspections[0]


def test_materialize_imported_publications_accepts_numeric_architecture_digest(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "hf-checkout"
    source_root.mkdir()
    bundle_path = source_root / "digits_publication.json"
    bundle_path.write_bytes(
        canonical_document_bytes(
            _digits_publication_bundle_record(
                architecture_manifest=_numeric_digest_architecture_record()
            )
        )
    )
    import_submission_publications(
        (source_root,),
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )

    summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )

    view = load_console_result_view(summary.view_file.read_bytes())
    results = cast(list[dict[str, object]], view["benchmark_results"])
    inspections = cast(list[dict[str, object]], results[0]["model_inspections"])
    assert inspections[0]["id"] == "model-inspections.imported.sha-057e708d0a213627@0.1.0"


def test_materialize_benchmark_result_views_rejects_empty_results_root(tmp_path: Path) -> None:
    with pytest.raises(LocalResultImportError, match="no benchmark result records"):
        materialize_benchmark_result_views(
            repository_root=_repository_root,
            results_root=tmp_path / "results",
        )


def test_publish_import_materialize_local_frontier_round_trip(tmp_path: Path) -> None:
    local_results_root = tmp_path / "local-runs"
    imported_results_root = tmp_path / "imported-runs"
    _init_git(local_results_root)
    run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=local_results_root,
            sample_count=1,
            train_steps=0,
        )
    )
    local_result_summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=local_results_root,
    )

    publish_summary = publish_local_benchmark_results(
        repository_root=_repository_root,
        results_root=local_results_root,
    )
    imported_summary = import_submission_publications(
        (local_results_root / "publication_bundles",),
        repository_root=_repository_root,
        results_root=imported_results_root,
    )
    result_summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=imported_results_root,
    )

    assert publish_summary.publication_bundle_count == 1
    assert publish_summary.measurement_count >= 1
    publication_document = SubmissionPublicationDocument.from_bytes(
        publish_summary.publication_files[0].read_bytes()
    )
    assert publication_document.bundle.submission_package.id == ProtocolIdentifier.parse(
        "submissions.digits.digits-arch-bb0dde9254dc-c1-seed101-samples1-steps0"
        "-train-e4b2ff9c5722@0.1.0"
    )
    assert publication_document.bundle.submission_package.sampled_competence is not None
    assert imported_summary.publication_bundle_count == 1
    assert result_summary.run_count == 1
    local_view = load_console_result_view(local_result_summary.view_file.read_bytes())
    local_results = cast(list[dict[str, object]], local_view["benchmark_results"])
    local_leaderboard = cast(list[dict[str, object]], local_results[0]["leaderboard"])
    view = load_console_result_view(result_summary.view_file.read_bytes())
    results = cast(list[dict[str, object]], view["benchmark_results"])
    history = cast(list[dict[str, object]], results[0]["training_history"])
    leaderboard = cast(list[dict[str, object]], results[0]["leaderboard"])
    assert history[0]["source_kind"] == "imported-publication"
    assert history[0]["measurement_count"] == publish_summary.measurement_count
    assert "sampled_competence" in history[0]
    assert leaderboard[0]["score"] == local_leaderboard[0]["score"]
    assert leaderboard[0]["observed_complexities"] == local_leaderboard[0]["observed_complexities"]
    assert leaderboard[0]["run_ids"] == [history[0]["run_id"]]


def test_cli_publishes_local_benchmark_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = tmp_path / "results"
    _init_git(results_root, configure_identity=False)
    run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=results_root,
            sample_count=1,
            train_steps=0,
        )
    )

    exit_code = main(
        [
            "results",
            "publish",
            "--results-root",
            str(results_root),
            "--message",
            "Publish test results",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "wrote 1 publication bundle(s), 2 measurement(s)" in captured.out
    assert "publication: " in captured.out
    assert "commit: " in captured.out
    assert len(tuple((results_root / "publication_bundles").glob("*.json"))) == 1
    assert _git(results_root, "status", "--porcelain").stdout == ""


def test_publish_defaults_publication_output_to_results_root(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _init_git(results_root)
    run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=results_root,
            sample_count=1,
            train_steps=0,
        )
    )

    summary = publish_local_benchmark_results(
        repository_root=_repository_root,
        results_root=results_root,
    )

    assert len(summary.publication_files) == 1
    assert summary.publication_files[0].parent == results_root / "publication_bundles"


def test_publish_can_commit_results_root_checkout(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    _init_git(results_root)
    run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=results_root,
            sample_count=1,
            train_steps=0,
        )
    )

    summary = publish_local_benchmark_results(
        repository_root=_repository_root,
        results_root=results_root,
        commit_message="Publish test results",
    )

    assert summary.git_commit == _git(results_root, "rev-parse", "HEAD").stdout.strip()
    assert summary.git_pushed is False
    assert _git(results_root, "status", "--porcelain").stdout == ""
    tracked_files = _git(results_root, "ls-files").stdout.splitlines()
    assert "views/benchmark_results.json" in tracked_files
    assert any(path.startswith("publication_bundles/") for path in tracked_files)


def test_publish_pushes_only_when_requested(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    remote_root = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(remote_root))
    _init_git(results_root)
    _git(results_root, "remote", "add", "origin", str(remote_root))
    run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=results_root,
            sample_count=1,
            train_steps=0,
        )
    )

    summary = publish_local_benchmark_results(
        repository_root=_repository_root,
        results_root=results_root,
        push=True,
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
    run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=results_root,
            sample_count=1,
            train_steps=0,
        )
    )

    summary = publish_local_benchmark_results(
        repository_root=_repository_root,
        results_root=results_root,
        push=True,
        repo_id="operator/leibniz-results",
        token="hf_test",
        commit_message="Publish test results",
    )

    assert summary.remote == "hf"
    assert summary.remote_commit == "hf-commit"
    assert summary.git_commit is None
    assert "views/benchmark_results.json" in uploaded_paths
    assert any(path.startswith("publication_bundles/") for path in uploaded_paths)


def test_push_publication_checkout_pushes_existing_commit(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    remote_root = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(remote_root))
    _init_git(results_root)
    _git(results_root, "remote", "add", "origin", str(remote_root))
    (results_root / "README.md").write_text("result checkout\n", encoding="utf-8")
    _git(results_root, "add", "README.md")
    _git(results_root, "commit", "-m", "Prepare checkout")

    summary = push_publication_checkout(
        repository_root=_repository_root,
        results_root=results_root,
    )

    assert summary.pushed_commit == _git(results_root, "rev-parse", "HEAD").stdout.strip()
    assert _git(remote_root, "rev-parse", "HEAD").stdout.strip() == summary.pushed_commit


def test_cli_pushes_publication_checkout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    results_root = tmp_path / "results"
    remote_root = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(remote_root))
    _init_git(results_root)
    _git(results_root, "remote", "add", "origin", str(remote_root))
    (results_root / "README.md").write_text("result checkout\n", encoding="utf-8")
    _git(results_root, "add", "README.md")
    _git(results_root, "commit", "-m", "Prepare checkout")

    exit_code = main(
        [
            "results",
            "push",
            "--results-root",
            str(results_root),
            "--remote",
            "git",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "pushed: " in captured.out
    assert _git(remote_root, "rev-parse", "HEAD").stdout.strip() == _git(
        results_root, "rev-parse", "HEAD"
    ).stdout.strip()


def test_initialize_publication_checkout_scaffolds_existing_git_checkout(
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

    summary = initialize_publication_checkout(
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
    assert "publication_bundles/.gitkeep" in tracked_files


def test_initialize_publication_checkout_creates_hugging_face_repo_for_plain_results(
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

    summary = initialize_publication_checkout(
        repo_id="operator/leibniz-results",
        repository_root=_repository_root,
        results_root=results_root,
        token="hf_test",
    )

    assert calls == ["operator/leibniz-results"]
    assert summary.scaffold_commit is None
    assert summary.created_or_reused is True
    assert (results_root / "publication_bundles" / ".gitkeep").is_file()


def test_initialize_publication_checkout_supports_local_only_fallback(tmp_path: Path) -> None:
    results_root = tmp_path / "results"

    summary = initialize_publication_checkout(
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
    assert (results_root / "publication_bundles" / ".gitkeep").is_file()


def test_cli_initializes_local_publication_checkout_with_default_results_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    exit_code = main(["results", "init-publication", "--local-only"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "repository: local-only" in captured.out
    assert "results root: " in captured.out
    assert (tmp_path / "results" / "publication_bundles" / ".gitkeep").is_file()


def _digits_publication_bundle_record(
    *,
    architecture_manifest: dict[str, object] | None = None,
) -> dict[str, object]:
    dataset = _digits_dataset()
    return {
        "id": "publication-bundles.digits@0.1.0",
        "submission_package": {
            "id": "submissions.digits-pool@0.1.0",
            "benchmark_manifest": _digits_benchmark().manifest.to_record(),
            "architecture_manifest": (
                _architecture().manifest.to_record()
                if architecture_manifest is None
                else architecture_manifest
            ),
            "measurement_dataset": dataset.to_record(),
            "artifacts": [
                {
                    "id": "artifacts.digits-weights@0.1.0",
                    "digest": str(ContentDigest.from_value({"checkpoint": "metadata"})),
                    "description": "checkpoint metadata only",
                }
            ],
        },
        "measurement_dataset": dataset.to_record(),
        "measurement_score_view": MeasurementScoreView.from_dataset(
            id=ProtocolIdentifier.parse("views.measurement-scores.digits@0.1.0"),
            dataset=dataset,
        ).to_record(),
    }


def _digits_dataset() -> MeasurementDataset:
    return MeasurementDataset.from_record({"measurements": [_digits_measurement().to_record()]})


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
        _repository_root / "tests" / "fixtures" / "architecture" / "digits_pool" / "manifest.json"
    )
    return ArchitectureManifestDocument.from_bytes(
        manifest_path.read_bytes()
    )


def _numeric_digest_architecture_record() -> dict[str, object]:
    record: dict[str, object] = {
        "id": (
            "architecture."
            "sha-ec44c2854f8e8dae76bbdcc5316fb467ac6cf6613b58dfd1aa692f6d1d3c2f16"
            "@0.1.0"
        ),
        "input_shape": [1, 32, 32],
        "output_shape": [10],
        "layers": [
            {"kind": "adaptive-pooling", "parameters": {"dimension": 1, "size": 3}},
            {"kind": "flatten", "parameters": {}},
            {"kind": "dense", "parameters": {"out": 10}},
        ],
    }
    document = ArchitectureManifestDocument.from_bytes(canonical_document_bytes(record))
    assert str(document.manifest.digest).startswith("sha256:057e708d0a213627")
    return document.manifest.to_record()
