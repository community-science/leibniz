from pathlib import Path
from typing import Any, cast

from leibniz.architectures import ArchitectureManifestDocument
from leibniz.benchmark_runner import BenchmarkRunPlan, run_benchmark
from leibniz.cli import main
from leibniz.local_results import load_console_result_view, materialize_benchmark_result_views
from leibniz.proposal_generation import (
    ProposalGenerationPlan,
    generate_experiment_proposals,
)
from leibniz.proposals import ExperimentProposalDocument

_repository_root = Path(__file__).parents[1]
_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_architecture_path = (
    _repository_root / "tests" / "fixtures" / "architecture" / "digits_pool" / "manifest.json"
)


def test_generate_experiment_proposals_writes_unmeasured_architecture_candidates(
    tmp_path: Path,
) -> None:
    run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_architecture_path,
            benchmark_root=_benchmark_root,
            runs_root=tmp_path / ".runs",
            sample_count=2,
            seed=101,
            train_steps=1,
        )
    )
    measured = ArchitectureManifestDocument.from_bytes(_architecture_path.read_bytes()).manifest

    summary = generate_experiment_proposals(
        ProposalGenerationPlan(
            benchmark_root=_benchmark_root,
            runs_root=tmp_path / ".runs",
            candidate_budget=2,
        )
    )

    architectures = tuple(
        ArchitectureManifestDocument.from_bytes(path.read_bytes()).manifest
        for path in summary.architecture_paths
    )
    document = ExperimentProposalDocument.from_bytes(
        summary.proposal_set_path.read_bytes(),
        dataset=_local_measurement_dataset(tmp_path / ".runs"),
        architectures=architectures,
    )

    assert summary.proposal_count == 2
    assert measured.digest not in {architecture.digest for architecture in architectures}
    assert [proposal.rank for proposal in document.proposal_set.proposals] == [1, 2]
    assert all(proposal.command for proposal in document.proposal_set.proposals)
    assert document.proposal_set.proposals[0].rationale.startswith(
        "resource-bootstrap selected resource stratum "
    )
    assert document.proposal_set.proposals[0].selector_name == "resource-bootstrap"
    assert document.proposal_set.proposals[0].source_candidate_rank is not None
    assert document.proposal_set.proposals[0].comparable_cost_best_score is not None
    assert document.proposal_set.proposals[0].resource_stratum_index is not None
    assert document.proposal_set.proposals[0].resource_stratum_count is not None
    assert document.proposal_set.proposals[0].capability_family_kind is not None
    assert document.proposal_set.proposals[0].capability_operator_kinds
    assert document.proposal_set.proposals[0].candidate_id in {
        architecture.id for architecture in architectures
    }
    assert document.proposal_set.proposals[0].acquisition_value is not None
    assert "--optimizer" in document.proposal_set.proposals[0].command
    assert "--schedule" in document.proposal_set.proposals[0].command
    assert "--validation-interval" in document.proposal_set.proposals[0].command
    assert "--evaluation-sample-count" in document.proposal_set.proposals[0].command

    result_summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        runs_root=tmp_path / ".runs",
    )
    result_view = load_console_result_view(result_summary.view_file.read_bytes())
    benchmark_results = cast(list[dict[str, object]], result_view["benchmark_results"])
    proposals = cast(list[dict[str, object]], benchmark_results[0]["proposals"])
    assert len(proposals) == 2
    assert proposals[0]["command"]
    assert proposals[0]["selector_name"] == "resource-bootstrap"
    assert proposals[0]["source_candidate_rank"]


def test_cli_generates_experiment_proposals(
    tmp_path: Path,
    capsys: Any,
) -> None:
    exit_code = main(
        [
            "results",
            "propose",
            "--benchmark-root",
            str(_benchmark_root),
            "--runs-root",
            str(tmp_path / ".runs"),
            "--candidate-budget",
            "1",
            "--evaluation-sample-count",
            "3",
            "--optimizer",
            "adamw",
            "--schedule",
            "reduce-on-plateau",
            "--validation-interval",
            "2",
            "--convergence-patience",
            "3",
            "--convergence-min-delta",
            "0.001",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "generated 1 proposal(s)" in captured.out
    assert (tmp_path / ".runs" / "proposals").is_dir()


def _local_measurement_dataset(runs_root: Path):
    from leibniz.measurements import MeasurementDatasetDocument

    measurement_files = tuple((runs_root / "measurements").rglob("*.json"))
    assert len(measurement_files) == 1
    return MeasurementDatasetDocument.from_bytes(measurement_files[0].read_bytes()).dataset
