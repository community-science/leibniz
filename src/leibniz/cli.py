"""Command line artifact validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from leibniz.active_loop import ActiveTrainingLoopPlan, run_active_training_loop
from leibniz.architecture_semantics import validate_architecture_semantics
from leibniz.architectures import ArchitectureManifestDocument
from leibniz.artifacts import ArtifactIndexDocument, ArtifactReferenceDocument
from leibniz.authority_indexes import AuthorityIndexDocument
from leibniz.benchmark_runner import BenchmarkRunPlan, BenchmarkRunSummary, run_benchmark
from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestDocument
from leibniz.documents import (
    canonical_document_bytes,
    document_filename_suffix,
    load_object_document,
)
from leibniz.federation_ingest import FederationIngestPlanDocument
from leibniz.formation_timing import FormationTimingPlan, time_formation_paths
from leibniz.local_results import (
    LocalResultImportError,
    import_submission_publications,
    initialize_publication_checkout,
    load_console_result_view,
    materialize_benchmark_result_views,
    publish_local_benchmark_results,
    push_publication_checkout,
)
from leibniz.measurements import (
    MeasurementDataset,
    MeasurementDatasetDocument,
    MeasurementDocument,
    MeasurementRecord,
)
from leibniz.model_derivations import ModelDerivationCompatibilityReportDocument
from leibniz.model_interfaces import ModelInterfaceDocument
from leibniz.model_lineage import ModelLineageDocument
from leibniz.model_manifests import ModelArtifactManifestDocument
from leibniz.model_operations import ModelOperationDocument
from leibniz.outcomes import OutcomeSpace
from leibniz.projection_records import ProjectionRecordDocument
from leibniz.proposal_generation import ProposalGenerationPlan, generate_experiment_proposals
from leibniz.publications import SubmissionPublicationDocument
from leibniz.resources import ResourceReportDocument, ResourceReportSetDocument
from leibniz.submission_registries import SubmissionRegistry, SubmissionRegistryDocument
from leibniz.view_manifests import ViewManifestDocument

__all__ = ["main"]

_manifest_filename = "manifest" + document_filename_suffix()


@dataclass(frozen=True, slots=True)
class _FrontierSnapshot:
    benchmark_id: str
    best_score: float | None
    model_count: int
    run_count: int


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Leibniz command line interface."""

    parser = _parser()
    args = parser.parse_args(argv)
    command = getattr(args, "command", None)
    if command == "validate":
        return _validate(args)
    if command == "results":
        return _results(args)
    if command == "benchmark":
        return _benchmark(args)
    if command == "console":
        return _console(args)
    parser.print_help(sys.stderr)
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leibniz")
    subcommands = parser.add_subparsers(dest="command")

    validate = subcommands.add_parser(
        "validate",
        description="validate artifact files",
        help="validate artifact files",
    )
    validate_subcommands = validate.add_subparsers(dest="artifact", required=True)

    manifest = validate_subcommands.add_parser(
        "manifest",
        help="validate a benchmark manifest document",
    )
    manifest.add_argument("path", type=Path)

    measurement = validate_subcommands.add_parser(
        "measurement",
        help="validate a measurement document",
    )
    measurement.add_argument("path", type=Path)
    measurement.add_argument("--manifest", type=Path)

    dataset = validate_subcommands.add_parser(
        "dataset",
        help="validate a measurement dataset document",
    )
    dataset.add_argument("path", type=Path)
    dataset.add_argument("--manifest", type=Path)

    artifact_reference = validate_subcommands.add_parser(
        "artifact-reference",
        help="validate an artifact reference document",
    )
    artifact_reference.add_argument("path", type=Path)

    artifact_index = validate_subcommands.add_parser(
        "artifact-index",
        help="validate an artifact index document",
    )
    artifact_index.add_argument("path", type=Path)

    authority_index = validate_subcommands.add_parser(
        "authority-index",
        help="validate an authority index document",
    )
    authority_index.add_argument("path", type=Path)

    resource_report = validate_subcommands.add_parser(
        "resource-report",
        help="validate a resource report document",
    )
    resource_report.add_argument("path", type=Path)

    resource_report_set = validate_subcommands.add_parser(
        "resource-report-set",
        help="validate a resource report set document",
    )
    resource_report_set.add_argument("path", type=Path)

    model_interface = validate_subcommands.add_parser(
        "model-interface",
        help="validate a model interface document",
    )
    model_interface.add_argument("path", type=Path)
    model_interface.add_argument("--outcome-space", type=Path, required=True)

    model_manifest = validate_subcommands.add_parser(
        "model-manifest",
        help="validate a model artifact manifest document",
    )
    model_manifest.add_argument("path", type=Path)

    architecture = validate_subcommands.add_parser(
        "architecture",
        help="validate an architecture manifest document",
    )
    architecture.add_argument("path", type=Path)
    architecture.add_argument("--semantic", action="store_true")

    model_operation = validate_subcommands.add_parser(
        "model-operation",
        help="validate a model operation document",
    )
    model_operation.add_argument("path", type=Path)

    model_lineage = validate_subcommands.add_parser(
        "model-lineage",
        help="validate a model lineage document",
    )
    model_lineage.add_argument("path", type=Path)

    view_manifest = validate_subcommands.add_parser(
        "view-manifest",
        help="validate a view manifest document",
    )
    view_manifest.add_argument("path", type=Path)

    projection_record = validate_subcommands.add_parser(
        "projection-record",
        help="validate a projection record document",
    )
    projection_record.add_argument("path", type=Path)

    model_derivation = validate_subcommands.add_parser(
        "model-derivation",
        help="validate a model derivation compatibility report document",
    )
    model_derivation.add_argument("path", type=Path)

    publication_bundle = validate_subcommands.add_parser(
        "publication-bundle",
        help="validate a submission publication bundle document",
    )
    publication_bundle.add_argument("path", type=Path)

    submission_registry = validate_subcommands.add_parser(
        "submission-registry",
        help="validate a submission registry document",
    )
    submission_registry.add_argument("path", type=Path)

    federation_ingest_plan = validate_subcommands.add_parser(
        "federation-ingest-plan",
        help="validate a federation ingest plan document",
    )
    federation_ingest_plan.add_argument("path", type=Path)
    federation_ingest_plan.add_argument("--registry", type=Path)

    results = subcommands.add_parser(
        "results",
        description="manage operator-local result views",
        help="manage operator-local result views",
    )
    results_subcommands = results.add_subparsers(dest="results_command", required=True)

    init_publication = results_subcommands.add_parser(
        "init-publication",
        description="prepare the local result root for public Hugging Face publication",
        help="prepare a result-publication root",
    )
    init_publication.add_argument(
        "--repo",
        help="Hugging Face dataset repository id in owner/name form",
    )
    init_publication.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token; defaults to HF_TOKEN or hf auth login",
    )
    init_publication.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    init_publication.add_argument(
        "--remote",
        choices=("auto", "hf", "git"),
        default="auto",
        help="publication remote selection; defaults to auto",
    )
    init_publication.add_argument(
        "--push",
        action="store_true",
        help="push the scaffold commit after initialization",
    )
    init_publication.add_argument(
        "--local-only",
        action="store_true",
        help="prepare a local publication checkout without a Hugging Face account",
    )
    init_publication.add_argument(
        "--message",
        default="Initialize Leibniz result publication checkout",
        help="Git commit message for the scaffold commit",
    )

    import_results = results_subcommands.add_parser(
        "import",
        description="import local publication bundles into a result checkout",
        help="import local publication bundles",
    )
    import_results.add_argument(
        "--source",
        action="append",
        default=[],
        required=True,
        type=Path,
        help="local publication checkout path or publication bundle document file",
    )
    import_results.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    publish_results = results_subcommands.add_parser(
        "publish",
        description="commit local benchmark runs as publication state",
        help="publish local result checkout",
    )
    publish_results.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    publish_results.add_argument(
        "--repo",
        help="Hugging Face dataset repository id in owner/name form",
    )
    publish_results.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token; defaults to HF_TOKEN or hf auth login",
    )
    publish_results.add_argument(
        "--remote",
        choices=("auto", "hf", "git"),
        default="auto",
        help="publication remote selection; defaults to auto",
    )
    publish_results.add_argument(
        "--push",
        action="store_true",
        help="push the result checkout after publishing",
    )
    publish_results.add_argument(
        "--message",
        default="Publish Leibniz benchmark results",
        help="Git commit message used when publishing",
    )
    push_results = results_subcommands.add_parser(
        "push",
        description="push an existing result-publication checkout",
        help="push result checkout",
    )
    push_results.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    push_results.add_argument(
        "--repo",
        help="Hugging Face dataset repository id in owner/name form",
    )
    push_results.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token; defaults to HF_TOKEN or hf auth login",
    )
    push_results.add_argument(
        "--remote",
        choices=("auto", "hf", "git"),
        default="auto",
        help="publication remote selection; defaults to auto",
    )
    materialize_results = results_subcommands.add_parser(
        "materialize",
        description="derive console result views from a result checkout",
        help="materialize local result views",
    )
    materialize_results.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    propose_results = results_subcommands.add_parser(
        "propose",
        description="generate deterministic local benchmark proposals",
        help="generate local proposals",
    )
    propose_results.add_argument("--benchmark-root", type=Path, required=True)
    propose_results.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    propose_results.add_argument("--candidate-budget", default=3, type=int)
    propose_results.add_argument("--candidate-sample-count", default=64, type=int)
    propose_results.add_argument("--sample-count", default=512, type=int)
    propose_results.add_argument("--evaluation-sample-count", default=None, type=int)
    propose_results.add_argument("--seed", default=101, type=int)
    propose_results.add_argument("--train-steps", default=None, type=int)
    propose_results.add_argument("--learning-rate", default=0.01, type=float)
    propose_results.add_argument("--optimizer", default="adam", choices=("sgd", "adam", "adamw"))
    propose_results.add_argument(
        "--schedule",
        default="reduce-on-plateau",
        choices=("none", "cosine", "reduce-on-plateau"),
    )
    propose_results.add_argument("--checkpoint-interval", default=256, type=int)
    propose_results.add_argument("--gate-check-interval", default=32, type=int)
    propose_results.add_argument("--gate-sample-count", default=None, type=int)
    propose_results.add_argument("--gate-decision-rule", default="validation-loss-plateau")
    propose_results.add_argument("--convergence-patience", default=6, type=int)
    propose_results.add_argument("--convergence-min-delta", default=1e-3, type=float)
    propose_results.add_argument("--convergence-min-steps", default=500, type=int)
    propose_results.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="tensor runtime device; auto prefers CUDA, then MPS, then CPU",
    )

    benchmark = subcommands.add_parser(
        "benchmark",
        description="run local benchmark workflows",
        help="run local benchmark workflows",
    )
    benchmark_subcommands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    run = benchmark_subcommands.add_parser(
        "run",
        description="run a benchmark locally",
        help="run a benchmark locally",
    )
    run.add_argument("--architecture", type=Path, required=True)
    run.add_argument("--benchmark-root", type=Path, required=True)
    run.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    run.add_argument("--sample-count", default=512, type=int)
    run.add_argument("--evaluation-sample-count", default=None, type=int)
    run.add_argument("--seed", default=101, type=int)
    run.add_argument("--train-steps", default=None, type=int)
    run.add_argument("--learning-rate", default=0.01, type=float)
    run.add_argument("--optimizer", default="adam", choices=("sgd", "adam", "adamw"))
    run.add_argument(
        "--schedule",
        default="reduce-on-plateau",
        choices=("none", "cosine", "reduce-on-plateau"),
    )
    run.add_argument("--checkpoint-interval", default=256, type=int)
    run.add_argument("--gate-check-interval", default=32, type=int)
    run.add_argument("--gate-sample-count", default=None, type=int)
    run.add_argument("--gate-decision-rule", default="validation-loss-plateau")
    run.add_argument("--convergence-patience", default=6, type=int)
    run.add_argument("--convergence-min-delta", default=1e-3, type=float)
    run.add_argument("--convergence-min-steps", default=500, type=int)
    run.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="tensor runtime device; auto prefers CUDA, then MPS, then CPU",
    )
    run.add_argument("--dry-run", action="store_true")
    loop = benchmark_subcommands.add_parser(
        "loop",
        description="run an active benchmark proposal loop",
        help="run an active benchmark loop",
    )
    loop.add_argument("--benchmark-root", type=Path, required=True)
    loop.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    loop.add_argument("--candidate-sample-count", default=64, type=int)
    loop.add_argument("--sample-count", default=512, type=int)
    loop.add_argument("--evaluation-sample-count", default=None, type=int)
    loop.add_argument("--seed", default=101, type=int)
    loop.add_argument("--train-steps", default=None, type=int)
    loop.add_argument("--learning-rate", default=0.01, type=float)
    loop.add_argument("--optimizer", default="adam", choices=("sgd", "adam", "adamw"))
    loop.add_argument(
        "--schedule",
        default="reduce-on-plateau",
        choices=("none", "cosine", "reduce-on-plateau"),
    )
    loop.add_argument("--checkpoint-interval", default=256, type=int)
    loop.add_argument("--gate-check-interval", default=32, type=int)
    loop.add_argument("--gate-sample-count", default=None, type=int)
    loop.add_argument("--gate-decision-rule", default="validation-loss-plateau")
    loop.add_argument("--convergence-patience", default=6, type=int)
    loop.add_argument("--convergence-min-delta", default=1e-3, type=float)
    loop.add_argument("--convergence-min-steps", default=500, type=int)
    loop.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="tensor runtime device; auto prefers CUDA, then MPS, then CPU",
    )
    loop.add_argument("--dry-run", action="store_true")
    shakedown = benchmark_subcommands.add_parser(
        "shakedown",
        description="run a small active frontier shakedown",
        help="run an active frontier shakedown",
    )
    shakedown.add_argument("--benchmark-root", type=Path, required=True)
    shakedown.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    shakedown.add_argument("--candidate-sample-count", default=64, type=int)
    shakedown.add_argument("--sample-count", default=1, type=int)
    shakedown.add_argument("--evaluation-sample-count", default=None, type=int)
    shakedown.add_argument("--seed", default=101, type=int)
    shakedown.add_argument("--train-steps", default=0, type=int)
    shakedown.add_argument("--learning-rate", default=0.01, type=float)
    shakedown.add_argument("--optimizer", default="adam", choices=("sgd", "adam", "adamw"))
    shakedown.add_argument(
        "--schedule",
        default="reduce-on-plateau",
        choices=("none", "cosine", "reduce-on-plateau"),
    )
    shakedown.add_argument("--checkpoint-interval", default=1, type=int)
    shakedown.add_argument("--gate-check-interval", default=1, type=int)
    shakedown.add_argument("--gate-sample-count", default=None, type=int)
    shakedown.add_argument("--gate-decision-rule", default="validation-loss-plateau")
    shakedown.add_argument("--convergence-patience", default=0, type=int)
    shakedown.add_argument("--convergence-min-delta", default=0.0, type=float)
    shakedown.add_argument("--convergence-min-steps", default=0, type=int)
    shakedown.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="tensor runtime device; auto prefers CUDA, then MPS, then CPU",
    )
    time_formation = benchmark_subcommands.add_parser(
        "time-formation",
        description="time local benchmark observation formation paths",
        help="time benchmark formation paths",
    )
    time_formation.add_argument("--benchmark-root", type=Path, required=True)
    time_formation.add_argument("--component-count", default=1, type=int)
    time_formation.add_argument("--sample-count", default=64, type=int)
    time_formation.add_argument("--seed", default=101, type=int)
    time_formation.add_argument("--repeats", default=3, type=int)
    time_formation.add_argument("--warmup-repeats", default=1, type=int)
    time_formation.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="tensor runtime device; auto prefers CUDA, then MPS, then CPU",
    )

    console = subcommands.add_parser(
        "console",
        description="run console web workflows",
        help="run console web workflows",
    )
    console_subcommands = console.add_subparsers(dest="console_command", required=True)
    dev = console_subcommands.add_parser(
        "dev",
        description="start the console development server",
        help="start the console dev server",
    )
    dev.add_argument("--host", default="127.0.0.1")
    dev.add_argument("--port", default=None, type=int)

    return parser


def _console(args: argparse.Namespace) -> int:
    if str(args.console_command) == "dev":
        command = _console_dev_command(args)
        try:
            return subprocess.run(
                command,
                check=False,
                cwd=_console_web_source_root(),
            ).returncode
        except OSError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
    print(f"error: unsupported console command {args.console_command!r}", file=sys.stderr)
    return 2


def _console_dev_command(args: argparse.Namespace) -> list[str]:
    command = ["npm", "run", "dev", "--", "--host", str(args.host)]
    if args.port is not None:
        command.extend(("--port", str(args.port)))
    return command


def _console_web_source_root() -> Path:
    return Path(__file__).parent / "console" / "_web_src"


def _benchmark(args: argparse.Namespace) -> int:
    try:
        if str(args.benchmark_command) == "run":
            summary = run_benchmark(
                BenchmarkRunPlan(
                    architecture_path=args.architecture,
                    results_root=args.results_root,
                    benchmark_root=args.benchmark_root,
                    sample_count=args.sample_count,
                    evaluation_sample_count=args.evaluation_sample_count,
                    seed=args.seed,
                    train_steps=args.train_steps,
                    learning_rate=args.learning_rate,
                    optimizer=args.optimizer,
                    schedule=args.schedule,
                    checkpoint_interval=args.checkpoint_interval,
                    gate_check_interval=args.gate_check_interval,
                    gate_sample_count=args.gate_sample_count,
                    gate_decision_rule=args.gate_decision_rule,
                    convergence_patience=args.convergence_patience,
                    convergence_min_delta=args.convergence_min_delta,
                    convergence_min_steps=args.convergence_min_steps,
                    tensor_device=args.device,
                    dry_run=args.dry_run,
                )
            )
            prefix = "planned" if summary.dry_run else "completed"
            print(
                f"{prefix} benchmark run {summary.run_slug} "
                f"({summary.measurement_count} measurement(s))"
            )
            print(f"measurements: {summary.measurement_dataset_path}")
            print(f"model inspection: {summary.model_inspection_path}")
            print(f"training summary: {summary.training_summary_path}")
            return 0
        if str(args.benchmark_command) == "loop":
            summary = run_active_training_loop(
                _active_training_loop_plan(
                    args,
                    dry_run=args.dry_run,
                    progress_callback=_print_active_training_progress,
                )
            )
            prefix = "planned" if summary.dry_run else "completed"
            print(
                f"{prefix} active benchmark loop for {summary.benchmark_id}: "
                f"{summary.completed_run_count} run(s)"
            )
            for command in summary.planned_commands:
                print("command: " + " ".join(command))
            if summary.result_view_path is not None:
                print(f"view: {summary.result_view_path}")
            return 0
        if str(args.benchmark_command) == "shakedown":
            before = _frontier_snapshot(
                benchmark_root=args.benchmark_root,
                results_root=args.results_root,
            )
            summary = run_active_training_loop(
                _active_training_loop_plan(args, dry_run=False)
            )
            after = _frontier_snapshot(
                benchmark_root=args.benchmark_root,
                results_root=args.results_root,
            )
            print(f"completed active frontier shakedown for {summary.benchmark_id}")
            print(
                f"runs: {before.run_count} -> {after.run_count} "
                f"({_delta(after.run_count, before.run_count)})"
            )
            print(
                f"models: {before.model_count} -> {after.model_count} "
                f"({_delta(after.model_count, before.model_count)})"
            )
            print(
                "best score: "
                f"{_score_label(before.best_score)} -> {_score_label(after.best_score)} "
                f"({_score_delta(after.best_score, before.best_score)})"
            )
            if summary.result_view_path is not None:
                print(f"view: {summary.result_view_path}")
            return 0
        if str(args.benchmark_command) == "time-formation":
            summary = time_formation_paths(
                FormationTimingPlan(
                    benchmark_root=args.benchmark_root,
                    component_count=args.component_count,
                    sample_count=args.sample_count,
                    seed=args.seed,
                    repeats=args.repeats,
                    warmup_repeats=args.warmup_repeats,
                    tensor_device=args.device,
                )
            )
            print(canonical_document_bytes(summary.to_record()).decode("utf-8"))
            return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(
        "error: unsupported benchmark command "
        f"{args.benchmark_command!r}",
        file=sys.stderr,
    )
    return 2


def _active_training_loop_plan(
    args: argparse.Namespace,
    *,
    dry_run: bool,
    progress_callback: Callable[[BenchmarkRunSummary], None] | None = None,
) -> ActiveTrainingLoopPlan:
    return ActiveTrainingLoopPlan(
        benchmark_root=args.benchmark_root,
        results_root=args.results_root,
        candidate_sample_count=args.candidate_sample_count,
        sample_count=args.sample_count,
        evaluation_sample_count=args.evaluation_sample_count,
        seed=args.seed,
        train_steps=args.train_steps,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        schedule=args.schedule,
        checkpoint_interval=args.checkpoint_interval,
        gate_check_interval=args.gate_check_interval,
        gate_sample_count=args.gate_sample_count,
        gate_decision_rule=args.gate_decision_rule,
        convergence_patience=args.convergence_patience,
        convergence_min_delta=args.convergence_min_delta,
        convergence_min_steps=args.convergence_min_steps,
        tensor_device=args.device,
        dry_run=dry_run,
        progress_callback=progress_callback,
    )


def _print_active_training_progress(summary: BenchmarkRunSummary) -> None:
    progress_path = (
        summary.training_summary_path.parent.parent.parent
        / "training-progress"
        / summary.training_summary_path.parent.name
        / summary.training_summary_path.name
    )
    if not progress_path.exists():
        return
    progress_record = load_object_document(
        progress_path.read_bytes(),
        description="training progress",
    )
    training_run = cast(Mapping[str, object], progress_record.get("training_run", {}))
    history = cast(Sequence[Mapping[str, object]], training_run.get("validation_history", ()))
    if not history:
        return
    last = history[-1]
    protocol = cast(Mapping[str, object], training_run.get("protocol", {}))
    max_steps = protocol.get("max_steps", "convergence")
    print(
        f"training {summary.run_slug}: "
        f"step {last.get('step', '?')}/{max_steps} "
        f"validation_loss={_format_progress_number(last.get('validation_loss'))} "
        f"stale_checks={last.get('stale_checks', '?')}",
        flush=True,
    )


def _format_progress_number(value: object) -> str:
    if isinstance(value, int | float):
        return f"{float(value):.4f}"
    return "?"


def _frontier_snapshot(*, benchmark_root: Path, results_root: Path) -> _FrontierSnapshot:
    manifest = _load_manifest(benchmark_root / _manifest_filename)
    empty = _FrontierSnapshot(
        benchmark_id=str(manifest.id),
        best_score=None,
        model_count=0,
        run_count=0,
    )
    try:
        summary = materialize_benchmark_result_views(
            repository_root=Path.cwd(),
            results_root=results_root,
        )
    except LocalResultImportError as error:
        if "no benchmark result records found" in str(error):
            return empty
        raise
    view = load_console_result_view(summary.view_file.read_bytes())
    for result in _sequence(view.get("benchmark_results")):
        if not isinstance(result, dict):
            continue
        typed_result = cast(dict[str, object], result)
        if typed_result.get("benchmark_id") != str(manifest.id):
            continue
        leaderboard = tuple(
            cast(dict[str, object], item)
            for item in _sequence(typed_result.get("leaderboard"))
            if isinstance(item, dict)
        )
        scores = tuple(
            score
            for item in leaderboard
            for score in [_number(item.get("score"))]
            if score is not None
        )
        return _FrontierSnapshot(
            benchmark_id=str(manifest.id),
            best_score=max(scores, default=None),
            model_count=len(leaderboard),
            run_count=len(_sequence(typed_result.get("training_history"))),
        )
    return empty


def _sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return cast(tuple[object, ...], value)
    if isinstance(value, list):
        return tuple(cast(list[object], value))
    return ()


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _delta(after: int, before: int) -> str:
    difference = after - before
    return f"+{difference}" if difference >= 0 else str(difference)


def _score_label(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _score_delta(after: float | None, before: float | None) -> str:
    if after is None or before is None:
        return "n/a"
    difference = after - before
    return f"+{difference:.4f}" if difference >= 0 else f"{difference:.4f}"


def _results(args: argparse.Namespace) -> int:
    try:
        results_command = str(args.results_command)
        if results_command == "init-publication":
            summary = initialize_publication_checkout(
                repo_id=args.repo,
                repository_root=Path.cwd(),
                results_root=args.results_root,
                remote=args.remote,
                local_only=args.local_only,
                push=args.push,
                commit_message=args.message,
                token=args.token,
            )
            if summary.repo_url is not None:
                print(f"repository: {summary.repo_url}")
            else:
                print("repository: local-only")
            print(f"results root: {summary.results_root}")
            if summary.scaffold_commit is not None:
                print(f"commit: {summary.scaffold_commit}")
            else:
                print("commit: unchanged")
            if summary.pushed:
                print("pushed: yes")
            return 0
        if results_command == "import":
            summary = import_submission_publications(
                args.source,
                repository_root=Path.cwd(),
                results_root=args.results_root,
            )
            print(
                "imported "
                f"{summary.publication_bundle_count} publication bundle(s), "
                f"{summary.measurement_count} measurement(s)"
            )
            print(f"view: {summary.view_file}")
            return 0
        if results_command == "publish":
            summary = publish_local_benchmark_results(
                repository_root=Path.cwd(),
                results_root=args.results_root,
                push=args.push,
                repo_id=args.repo,
                remote=args.remote,
                token=args.token,
                commit_message=args.message,
            )
            print(
                "wrote "
                f"{summary.publication_bundle_count} publication bundle(s), "
                f"{summary.measurement_count} measurement(s)"
            )
            for publication_file in summary.publication_files:
                print(f"publication: {publication_file}")
            if summary.git_commit is not None:
                print(f"commit: {summary.git_commit}")
            if summary.git_pushed:
                print("pushed: yes")
            if summary.remote_commit is not None:
                print(f"remote: {summary.remote}")
                print(f"remote commit: {summary.remote_commit}")
            return 0
        if results_command == "push":
            summary = push_publication_checkout(
                repository_root=Path.cwd(),
                results_root=args.results_root,
                repo_id=args.repo,
                remote=args.remote,
                token=args.token,
            )
            print(f"pushed: {summary.pushed_commit}")
            print(f"results root: {summary.results_root}")
            return 0
        if results_command == "materialize":
            summary = materialize_benchmark_result_views(
                repository_root=Path.cwd(),
                results_root=args.results_root,
            )
            print(
                "materialized "
                f"{summary.benchmark_count} benchmark result view(s), "
                f"{summary.model_count} model(s), "
                f"{summary.run_count} run(s)"
            )
            print(f"view: {summary.view_file}")
            return 0
        if results_command == "propose":
            summary = generate_experiment_proposals(
                ProposalGenerationPlan(
                    benchmark_root=args.benchmark_root,
                    results_root=args.results_root,
                    candidate_budget=args.candidate_budget,
                    candidate_sample_count=args.candidate_sample_count,
                    sample_count=args.sample_count,
                    evaluation_sample_count=args.evaluation_sample_count,
                    seed=args.seed,
                    train_steps=args.train_steps,
                    learning_rate=args.learning_rate,
                    optimizer=args.optimizer,
                    schedule=args.schedule,
                    checkpoint_interval=args.checkpoint_interval,
                    gate_check_interval=args.gate_check_interval,
                    gate_sample_count=args.gate_sample_count,
                    gate_decision_rule=args.gate_decision_rule,
                    convergence_patience=args.convergence_patience,
                    convergence_min_delta=args.convergence_min_delta,
                    convergence_min_steps=args.convergence_min_steps,
                    tensor_device=args.device,
                )
            )
            print(
                f"generated {summary.proposal_count} proposal(s) "
                f"for {summary.benchmark_id}"
            )
            print(f"proposal set: {summary.proposal_set_path}")
            return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"error: unsupported results command {args.results_command!r}", file=sys.stderr)
    return 2


def _validate(args: argparse.Namespace) -> int:
    try:
        artifact = str(args.artifact)
        if artifact == "manifest":
            manifest = _load_manifest(args.path)
            print(f"valid manifest {manifest.id}")
            return 0
        if artifact == "measurement":
            measurement = _load_measurement(args.path)
            manifest_path = getattr(args, "manifest", None)
            if manifest_path is not None:
                measurement.validate_manifest(_load_manifest(manifest_path))
            print(f"valid measurement {measurement.raw_scoring_evidence.id}")
            return 0
        if artifact == "dataset":
            dataset = _load_dataset(args.path)
            manifest_path = getattr(args, "manifest", None)
            if manifest_path is not None:
                dataset.validate_manifest(_load_manifest(manifest_path))
            print(f"valid measurement dataset ({len(dataset.measurements)} measurements)")
            return 0
        if artifact == "artifact-reference":
            document = ArtifactReferenceDocument.from_bytes(args.path.read_bytes())
            print(f"valid artifact reference {document.digest}")
            return 0
        if artifact == "artifact-index":
            document = ArtifactIndexDocument.from_bytes(args.path.read_bytes())
            print(f"valid artifact index {document.index.id}")
            return 0
        if artifact == "authority-index":
            document = AuthorityIndexDocument.from_bytes(args.path.read_bytes())
            print(f"valid authority index {document.index.id}")
            return 0
        if artifact == "resource-report":
            document = ResourceReportDocument.from_bytes(args.path.read_bytes())
            print(f"valid resource report {document.report.id}")
            return 0
        if artifact == "resource-report-set":
            document = ResourceReportSetDocument.from_bytes(args.path.read_bytes())
            print(f"valid resource report set {document.report_set.id}")
            return 0
        if artifact == "model-interface":
            document = ModelInterfaceDocument.from_bytes(
                args.path.read_bytes(),
                outcome_space=_load_outcome_space(args.outcome_space),
            )
            print(f"valid model interface {document.interface.id}")
            return 0
        if artifact == "model-manifest":
            document = ModelArtifactManifestDocument.from_bytes(args.path.read_bytes())
            print(f"valid model manifest {document.manifest.id}")
            return 0
        if artifact == "architecture":
            document = ArchitectureManifestDocument.from_bytes(args.path.read_bytes())
            if bool(getattr(args, "semantic", False)):
                validate_architecture_semantics(document.manifest)
            print(f"valid architecture {document.manifest.id}")
            return 0
        if artifact == "model-operation":
            document = ModelOperationDocument.from_bytes(args.path.read_bytes())
            print(f"valid model operation {document.operation.id}")
            return 0
        if artifact == "model-lineage":
            document = ModelLineageDocument.from_bytes(args.path.read_bytes())
            print(f"valid model lineage {document.lineage.id}")
            return 0
        if artifact == "view-manifest":
            document = ViewManifestDocument.from_bytes(args.path.read_bytes())
            print(f"valid view manifest {document.manifest.id}")
            return 0
        if artifact == "projection-record":
            document = ProjectionRecordDocument.from_bytes(args.path.read_bytes())
            print(f"valid projection record {document.record.id}")
            return 0
        if artifact == "model-derivation":
            document = ModelDerivationCompatibilityReportDocument.from_bytes(args.path.read_bytes())
            print(f"valid model derivation compatibility report {document.report.id}")
            return 0
        if artifact == "publication-bundle":
            document = SubmissionPublicationDocument.from_bytes(args.path.read_bytes())
            print(f"valid publication bundle {document.bundle.id}")
            return 0
        if artifact == "submission-registry":
            document = SubmissionRegistryDocument.from_bytes(args.path.read_bytes())
            print(f"valid submission registry {document.registry.id}")
            return 0
        if artifact == "federation-ingest-plan":
            registry_path = getattr(args, "registry", None)
            registry = _load_submission_registry(registry_path) if registry_path else None
            document = FederationIngestPlanDocument.from_bytes(
                args.path.read_bytes(),
                registry=registry,
            )
            print(f"valid federation ingest plan {document.plan.id}")
            return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"error: unsupported validation artifact {args.artifact!r}", file=sys.stderr)
    return 2


def _load_manifest(path: Path) -> BenchmarkManifest:
    return BenchmarkManifestDocument.from_bytes(path.read_bytes()).manifest


def _load_measurement(path: Path) -> MeasurementRecord:
    return MeasurementDocument.from_bytes(path.read_bytes()).measurement


def _load_dataset(path: Path) -> MeasurementDataset:
    return MeasurementDatasetDocument.from_bytes(path.read_bytes()).dataset


def _load_outcome_space(path: Path) -> OutcomeSpace:
    return OutcomeSpace.from_record(
        load_object_document(path.read_bytes(), description="outcome space document")
    )


def _load_submission_registry(path: Path) -> SubmissionRegistry:
    return SubmissionRegistryDocument.from_bytes(path.read_bytes()).registry


if __name__ == "__main__":
    raise SystemExit(main())
