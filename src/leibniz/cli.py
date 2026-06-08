"""Command line artifact validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from leibniz.architecture_semantics import validate_architecture_semantics
from leibniz.architectures import (
    ArchitectureManifestDocument,
    ArchitectureManifestValidationError,
)
from leibniz.artifacts import ArtifactIndexDocument, ArtifactReferenceDocument
from leibniz.authority_indexes import AuthorityIndexDocument
from leibniz.benchmark_implementations import (
    discover_benchmark_roots,
    load_benchmark,
)
from leibniz.benchmark_runner import (
    BenchmarkEvaluationPlan,
    BenchmarkEvaluationSummary,
    BenchmarkRunPlan,
    BenchmarkRunSummary,
    evaluate_benchmark_checkpoint,
    run_benchmark,
)
from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestDocument
from leibniz.documents import (
    canonical_document_bytes,
    document_filename_suffix,
    load_object_document,
)
from leibniz.evaluation_bundles import BenchmarkEvaluationBundleDocument
from leibniz.formation_timing import FormationTimingPlan, time_formation_paths
from leibniz.local_results import (
    LocalResultImportError,
    initialize_result_checkout,
    materialize_benchmark_result_views,
    publish_local_benchmark_results,
    summarize_local_benchmark_results,
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
from leibniz.resources import ResourceReportDocument, ResourceReportSetDocument
from leibniz.submission_registries import SubmissionRegistryDocument
from leibniz.tensor_runtime import tensor_runtime_device_choices

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Leibniz command line interface."""

    parser = _parser()
    args = parser.parse_args(argv)
    command = getattr(args, "command", None)
    if command == "validate":
        return _validate(args)
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

    model_derivation = validate_subcommands.add_parser(
        "model-derivation",
        help="validate a model derivation compatibility report document",
    )
    model_derivation.add_argument("path", type=Path)

    evaluation_bundle = validate_subcommands.add_parser(
        "evaluation-bundle",
        help="validate a benchmark evaluation bundle document",
    )
    evaluation_bundle.add_argument("path", type=Path)

    submission_registry = validate_subcommands.add_parser(
        "submission-registry",
        help="validate a submission registry document",
    )
    submission_registry.add_argument("path", type=Path)

    benchmark = subcommands.add_parser(
        "benchmark",
        description="run local benchmark workflows",
        help="run local benchmark workflows",
    )
    benchmark_subcommands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    init_benchmark = benchmark_subcommands.add_parser(
        "init",
        description="prepare the local benchmark result root",
        help="prepare a result root",
    )
    init_benchmark.add_argument(
        "--repo",
        help="Hugging Face dataset repository id in owner/name form",
    )
    init_benchmark.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token; defaults to HF_TOKEN or hf auth login",
    )
    init_benchmark.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    init_benchmark.add_argument(
        "--remote",
        choices=("auto", "hf", "git"),
        default="auto",
        help="result remote selection; defaults to auto",
    )
    init_benchmark.add_argument(
        "--push",
        action="store_true",
        help="push the scaffold commit after initialization",
    )
    init_benchmark.add_argument(
        "--local-only",
        action="store_true",
        help="prepare a local result checkout without a Hugging Face account",
    )
    init_benchmark.add_argument(
        "--message",
        default="Initialize Leibniz result checkout",
        help="Git commit message for the scaffold commit",
    )
    publish_benchmark = benchmark_subcommands.add_parser(
        "publish",
        description="commit and push local benchmark result state",
        help="publish local result checkout",
    )
    publish_benchmark.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    publish_benchmark.add_argument(
        "--repo",
        help="Hugging Face dataset repository id in owner/name form",
    )
    publish_benchmark.add_argument(
        "--token",
        default=None,
        help="Hugging Face API token; defaults to HF_TOKEN or hf auth login",
    )
    publish_benchmark.add_argument(
        "--remote",
        choices=("auto", "hf", "git"),
        default="auto",
        help="result remote selection; defaults to auto",
    )
    publish_benchmark.add_argument(
        "--no-push",
        action="store_true",
        help="commit without pushing the result checkout",
    )
    publish_benchmark.add_argument(
        "--message",
        default="Publish Leibniz benchmark results",
        help="Git commit message used when publishing",
    )
    inspect_benchmark = benchmark_subcommands.add_parser(
        "inspect",
        description="print a compact summary of local benchmark result state",
        help="inspect local result state",
    )
    inspect_benchmark.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    train = benchmark_subcommands.add_parser(
        "train",
        description="train locally available architecture manifests",
        help="train architecture manifests",
    )
    train.add_argument(
        "--architecture",
        type=Path,
        action="append",
        default=[],
        help=(
            "architecture manifest or directory to train; may be repeated, "
            "defaults to architecture manifests discovered under results/training"
        ),
    )
    train.add_argument(
        "--benchmark-root",
        type=Path,
        action="append",
        default=[],
        help="benchmark root; may be repeated, defaults to packaged benchmarks",
    )
    train.add_argument(
        "benchmarks",
        nargs="*",
        help="optional benchmark ids or names; defaults to all local benchmarks",
    )
    train.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    train.add_argument("--seed", default=101, type=int)
    train.add_argument("--train-steps", default=None, type=int)
    train.add_argument("--gate-check-interval", default=32, type=int)
    train.add_argument("--model-checkpoint-gate-interval", default=1, type=int)
    train.add_argument("--gate-decision-rule", default="score-estimate-plateau")
    train.add_argument("--rung-competence-threshold", default=0.5, type=float)
    train.add_argument("--convergence-patience", default=6, type=int)
    train.add_argument("--convergence-min-delta", default=1e-3, type=float)
    train.add_argument(
        "--device",
        default="auto",
        choices=tensor_runtime_device_choices(),
        help="tensor runtime device; auto prefers accelerated runtimes before host",
    )
    train.add_argument("--dry-run", action="store_true")
    evaluate = benchmark_subcommands.add_parser(
        "evaluate",
        description="evaluate saved training checkpoints as benchmark evidence",
        help="evaluate saved checkpoints",
    )
    evaluate.add_argument("--checkpoint-artifact", type=Path)
    evaluate.add_argument(
        "--benchmark-root",
        type=Path,
        action="append",
        default=[],
        help="benchmark root; may be repeated, defaults to packaged benchmarks",
    )
    evaluate.add_argument(
        "benchmarks",
        nargs="*",
        help="optional benchmark ids or names; defaults to all local benchmarks",
    )
    evaluate.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    evaluate.add_argument(
        "--device",
        default="auto",
        choices=tensor_runtime_device_choices(),
        help="tensor runtime device; auto prefers accelerated runtimes before host",
    )
    profile = benchmark_subcommands.add_parser(
        "profile",
        description="profile local benchmark observation formation paths",
        help="profile benchmark formation paths",
    )
    profile.add_argument("--benchmark-root", type=Path, required=True)
    profile.add_argument("--batch-target", default=64, type=int)
    profile.add_argument("--seed", default=101, type=int)
    profile.add_argument("--repeats", default=3, type=int)
    profile.add_argument("--warmup-repeats", default=1, type=int)
    profile.add_argument(
        "--device",
        default="auto",
        choices=tensor_runtime_device_choices(),
        help="tensor runtime device; auto prefers accelerated runtimes before host",
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
        if str(args.benchmark_command) == "init":
            summary = initialize_result_checkout(
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
        if str(args.benchmark_command) == "publish":
            summary = publish_local_benchmark_results(
                repository_root=Path.cwd(),
                results_root=args.results_root,
                push=not args.no_push,
                repo_id=args.repo,
                remote=args.remote,
                token=args.token,
                commit_message=args.message,
            )
            print(
                "published "
                f"{summary.measurement_count} measurement(s)"
            )
            if summary.git_commit is not None:
                print(f"commit: {summary.git_commit}")
            if summary.git_pushed:
                print("pushed: yes")
            if summary.remote_commit is not None:
                print(f"remote: {summary.remote}")
                print(f"remote commit: {summary.remote_commit}")
            return 0
        if str(args.benchmark_command) == "inspect":
            print(
                canonical_document_bytes(
                    summarize_local_benchmark_results(
                        repository_root=Path.cwd(),
                        results_root=args.results_root,
                    )
                ).decode("utf-8")
            )
            return 0
        if str(args.benchmark_command) == "train":
            summaries, skipped, moved = _run_benchmark_training(args)
            if not summaries and not skipped:
                print("no uncompleted benchmark training manifests found")
            for summary in summaries:
                prefix = "planned" if summary.dry_run else "completed"
                print(
                    f"{prefix} benchmark training run {summary.run_slug} "
                    f"({summary.measurement_count} benchmark measurement(s) planned)"
                )
                print(f"training summary: {summary.training_summary_path}")
                print(f"model artifacts: {summary.model_artifact_root}")
            if skipped:
                print(f"skipped {skipped} completed benchmark training manifest(s)")
            if moved:
                print(f"moved {moved} completed benchmark training manifest(s) out of pending")
            return 0
        if str(args.benchmark_command) == "evaluate":
            benchmark_selectors = tuple(args.benchmarks)
            evaluation_summaries: list[BenchmarkEvaluationSummary] = []
            benchmark_roots = dict(
                _selected_benchmark_roots_by_id(
                    explicit_roots=tuple(args.benchmark_root),
                    benchmark_selectors=benchmark_selectors,
                )
            )
            for checkpoint_artifact in _evaluation_checkpoint_artifacts(
                results_root=args.results_root,
                checkpoint_artifact=args.checkpoint_artifact,
                benchmark_selectors=benchmark_selectors,
            ):
                checkpoint_record = _load_object_record(
                    checkpoint_artifact,
                    description="checkpoint artifact",
                )
                benchmark_root = _benchmark_root_for_record(
                    checkpoint_record,
                    benchmark_roots=benchmark_roots,
                    description="checkpoint_artifact",
                )
                summary = evaluate_benchmark_checkpoint(
                    BenchmarkEvaluationPlan(
                        checkpoint_artifact_path=checkpoint_artifact,
                        benchmark_root=benchmark_root,
                        results_root=args.results_root,
                        tensor_device=args.device,
                    )
                )
                evaluation_summaries.append(summary)
                print(
                    f"completed benchmark evaluation {summary.run_slug} "
                    f"({summary.measurement_count} measurement(s))"
                )
                print(f"evaluation bundle: {summary.evaluation_bundle_path}")
            if not evaluation_summaries and args.checkpoint_artifact is not None:
                raise ValueError("no checkpoint artifacts matched benchmark evaluation inputs")
            if not evaluation_summaries:
                print("no unevaluated benchmark checkpoints found")
            if evaluation_summaries or not _benchmark_views_present(
                results_root=args.results_root,
                benchmark_selectors=benchmark_selectors,
            ):
                _materialize_benchmark_views_if_present(results_root=args.results_root)
            else:
                print("benchmark result views already current")
            return 0
        if str(args.benchmark_command) == "profile":
            summary = time_formation_paths(
                FormationTimingPlan(
                    benchmark_root=args.benchmark_root,
                    sample_count=args.batch_target,
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


def _run_benchmark_training(args: argparse.Namespace) -> tuple[list[BenchmarkRunSummary], int, int]:
    summaries: list[BenchmarkRunSummary] = []
    skipped = 0
    moved = 0
    benchmark_roots = tuple(
        root
        for _benchmark_id, root in _selected_benchmark_roots_by_id(
            explicit_roots=tuple(args.benchmark_root),
            benchmark_selectors=tuple(args.benchmarks),
        )
    )
    for architecture_path in _training_architecture_manifests(
        architecture_inputs=tuple(args.architecture),
        results_root=args.results_root,
    ):
        moved_architecture_path = None
        if not args.architecture and not args.dry_run:
            moved_architecture_path = _move_training_manifest_out_of_pending(architecture_path)
            if moved_architecture_path is not None:
                architecture_path = moved_architecture_path
                moved += 1
        for benchmark_root in benchmark_roots:
            plan = _benchmark_run_plan(
                args,
                architecture_path=architecture_path,
                benchmark_root=benchmark_root,
            )
            if not args.architecture and _benchmark_training_completed(plan):
                skipped += 1
                continue
            summaries.append(run_benchmark(plan))
    return summaries, skipped, moved


def _benchmark_run_plan(
    args: argparse.Namespace,
    *,
    architecture_path: Path,
    benchmark_root: Path,
    dry_run: bool | None = None,
) -> BenchmarkRunPlan:
    return BenchmarkRunPlan(
        architecture_path=architecture_path,
        results_root=args.results_root,
        benchmark_root=benchmark_root,
        seed=args.seed,
        train_steps=args.train_steps,
        gate_check_interval=args.gate_check_interval,
        model_checkpoint_gate_interval=args.model_checkpoint_gate_interval,
        gate_decision_rule=args.gate_decision_rule,
        rung_competence_threshold=args.rung_competence_threshold,
        convergence_patience=args.convergence_patience,
        convergence_min_delta=args.convergence_min_delta,
        tensor_device=args.device,
        dry_run=args.dry_run if dry_run is None else dry_run,
    )


def _selected_benchmark_roots_by_id(
    *,
    explicit_roots: tuple[Path, ...],
    benchmark_selectors: tuple[str, ...],
) -> tuple[tuple[str, Path], ...]:
    by_id = _benchmark_roots_by_id(
        repository_root=Path.cwd(),
        explicit_roots=explicit_roots,
    )
    roots = tuple(
        (benchmark_id, root)
        for benchmark_id, root in sorted(by_id.items(), key=lambda item: str(item[0]))
        if _benchmark_selected(benchmark_id, benchmark_selectors)
    )
    if benchmark_selectors and not roots:
        raise ValueError("no benchmark roots matched benchmark selectors")
    return roots


def _training_architecture_manifests(
    *,
    architecture_inputs: tuple[Path, ...],
    results_root: Path,
) -> tuple[Path, ...]:
    if not architecture_inputs:
        return _pending_training_architecture_manifests(results_root=results_root)
    roots = architecture_inputs
    paths: list[Path] = []
    for root in roots:
        if root.is_file():
            if _is_architecture_manifest(root):
                paths.append(root)
            else:
                raise ValueError(f"architecture manifest is invalid: {root}")
            continue
        if root.is_dir():
            paths.extend(
                path
                for path in sorted(root.rglob("*" + document_filename_suffix()))
                if _is_architecture_manifest(path)
            )
            continue
        if architecture_inputs:
            raise ValueError(f"architecture path does not exist: {root}")
    return tuple(dict.fromkeys(paths))


def _pending_training_architecture_manifests(*, results_root: Path) -> tuple[Path, ...]:
    training_root = results_root / "training"
    if not training_root.is_dir():
        return ()
    paths: list[Path] = []
    for pending_root in sorted(training_root.rglob("pending")):
        if not pending_root.is_dir():
            continue
        paths.extend(
            path
            for path in sorted(pending_root.rglob("*" + document_filename_suffix()))
            if _is_architecture_manifest(path)
        )
    return tuple(dict.fromkeys(paths))


def _is_architecture_manifest(path: Path) -> bool:
    try:
        ArchitectureManifestDocument.from_bytes(path.read_bytes())
    except (OSError, ArchitectureManifestValidationError):
        return False
    return True


def _benchmark_training_completed(plan: BenchmarkRunPlan) -> bool:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=plan.architecture_path,
            benchmark_root=plan.benchmark_root,
            results_root=plan.results_root,
            seed=plan.seed,
            train_steps=plan.train_steps,
            learning_rate=plan.learning_rate,
            optimizer=plan.optimizer,
            schedule=plan.schedule,
            gate_check_interval=plan.gate_check_interval,
            model_checkpoint_gate_interval=plan.model_checkpoint_gate_interval,
            gate_decision_rule=plan.gate_decision_rule,
            convergence_patience=plan.convergence_patience,
            convergence_min_delta=plan.convergence_min_delta,
            tensor_device=plan.tensor_device,
            dry_run=True,
        )
    )
    if not summary.training_summary_path.is_file():
        return False
    record = _load_object_record(summary.training_summary_path, description="training summary")
    completed = record.get("run_status") == "completed"
    if completed:
        _rewrite_training_summary_architecture_path(
            summary.training_summary_path,
            architecture_path=summary.architecture_path,
            results_root=summary.results_root,
        )
    return completed


def _move_training_manifest_out_of_pending(path: Path) -> Path | None:
    if path.parent.name != "pending" or not path.is_file():
        return None
    completed_root = path.parent.parent / "completed"
    completed_root.mkdir(parents=True, exist_ok=True)
    target = completed_root / path.name
    if target.exists():
        if target.read_bytes() == path.read_bytes():
            path.unlink()
            return target
        manifest = ArchitectureManifestDocument.from_bytes(path.read_bytes()).manifest
        target = completed_root / f"{path.stem}-{manifest.digest.hex[:12]}{path.suffix}"
    path.replace(target)
    return target


def _rewrite_training_summary_architecture_path(
    summary_path: Path,
    *,
    architecture_path: Path,
    results_root: Path,
) -> None:
    record = _load_object_record(summary_path, description="training summary")
    architecture_path_record = _portable_record_path(
        architecture_path,
        results_root=results_root,
    )
    if record.get("architecture_path") == architecture_path_record:
        return
    record["architecture_path"] = architecture_path_record
    summary_path.write_bytes(canonical_document_bytes(record) + b"\n")


def _portable_record_path(path: Path, *, results_root: Path) -> str:
    if not path.is_absolute():
        return path.as_posix()
    resolved = path.resolve()
    resolved_results_root = results_root.resolve()
    if resolved.is_relative_to(resolved_results_root):
        return (Path(results_root.name) / resolved.relative_to(resolved_results_root)).as_posix()
    working_root = Path.cwd().resolve()
    if resolved.is_relative_to(working_root):
        return resolved.relative_to(working_root).as_posix()
    return path.as_posix()


def _materialize_benchmark_views_if_present(*, results_root: Path) -> None:
    try:
        started = time.perf_counter()
        summary = materialize_benchmark_result_views(
            repository_root=Path.cwd(),
            results_root=results_root,
        )
        seconds = time.perf_counter() - started
    except LocalResultImportError as error:
        if "no benchmark result records found" in str(error):
            return
        raise
    print(
        "materialized "
        f"{summary.benchmark_count} benchmark result view(s), "
        f"{summary.model_count} model(s), "
        f"{summary.run_count} run(s) "
        f"in {seconds:.3f}s"
    )
    for view_file in summary.benchmark_view_files or (summary.view_file,):
        print(f"view: {view_file}")


def _benchmark_views_present(
    *,
    results_root: Path,
    benchmark_selectors: tuple[str, ...],
) -> bool:
    paths = _evaluation_bundle_paths(
        results_root=results_root,
        benchmark_selectors=benchmark_selectors,
    )
    if not paths:
        return False
    benchmark_ids = {
        _benchmark_id_from_record(
            _load_evaluation_summary_record(path),
            description="evaluation",
        )
        for path in paths
    }
    return all(
        (
            results_root
            / "views"
            / _benchmark_atom(benchmark_id)
            / ("benchmark_results" + document_filename_suffix())
        ).is_file()
        for benchmark_id in benchmark_ids
    )


def _evaluation_checkpoint_artifacts(
    *,
    results_root: Path,
    checkpoint_artifact: Path | None,
    benchmark_selectors: tuple[str, ...],
) -> tuple[Path, ...]:
    if checkpoint_artifact is not None:
        return (
            _resolve_result_artifact_path(
                checkpoint_artifact,
                results_root=results_root,
            ),
        )
    training_root = results_root / "training"
    if not training_root.is_dir():
        return ()
    checkpoint_paths: list[Path] = []
    for path in sorted(training_root.rglob("*" + document_filename_suffix())):
        record = _load_object_record(path, description="training summary")
        if record.get("format") != "leibniz.benchmark-run":
            continue
        if record.get("run_status") != "completed":
            continue
        benchmark_id = _benchmark_id_from_record(record, description="training_summary")
        if not _benchmark_selected(benchmark_id, benchmark_selectors):
            continue
        run_slug = record.get("run_slug")
        if isinstance(run_slug, str) and _evaluation_bundle_exists(
            results_root=results_root,
            benchmark_id=benchmark_id,
            run_slug=run_slug,
        ):
            continue
        raw_artifact = record.get("evaluation_model_artifact")
        if not isinstance(raw_artifact, Mapping):
            continue
        artifact = cast(Mapping[str, object], raw_artifact)
        record_path = artifact.get("record_path")
        if not isinstance(record_path, str) or not record_path:
            continue
        checkpoint_paths.append(
            _resolve_result_artifact_path(Path(record_path), results_root=results_root)
        )
    return tuple(dict.fromkeys(checkpoint_paths))


def _resolve_result_artifact_path(path: Path, *, results_root: Path) -> Path:
    if path.is_absolute():
        return path
    if path.parts[:1] == (results_root.name,):
        return results_root.parent / path
    return path


def _evaluation_bundle_paths(
    *,
    results_root: Path,
    benchmark_selectors: tuple[str, ...],
) -> tuple[Path, ...]:
    evaluation_root = results_root / "evaluations"
    if not evaluation_root.is_dir():
        return ()
    paths: list[Path] = []
    for path in sorted(evaluation_root.rglob("*" + document_filename_suffix())):
        record = _load_evaluation_summary_record(path)
        benchmark_id = _benchmark_id_from_record(record, description="evaluation")
        if not _benchmark_selected(benchmark_id, benchmark_selectors):
            continue
        paths.append(path)
    return tuple(paths)


def _evaluation_bundle_exists(
    *,
    results_root: Path,
    benchmark_id: str,
    run_slug: str,
) -> bool:
    path = (
        results_root
        / "evaluations"
        / _benchmark_atom(benchmark_id)
        / (run_slug + document_filename_suffix())
    )
    return path.is_file()


def _benchmark_selected(benchmark_id: str, selectors: tuple[str, ...]) -> bool:
    if not selectors:
        return True
    benchmark_name = benchmark_id.split("@", maxsplit=1)[0]
    benchmark_atom = _benchmark_atom(benchmark_id)
    return any(
        selector in {benchmark_id, benchmark_name, benchmark_atom}
        for selector in selectors
    )


def _benchmark_atom(benchmark_id: str) -> str:
    return benchmark_id.split("@", maxsplit=1)[0].rsplit(".", maxsplit=1)[-1]


def _benchmark_roots_by_id(
    *,
    repository_root: Path,
    explicit_roots: tuple[Path, ...],
) -> dict[str, Path]:
    packaged_root = repository_root / "src" / "leibniz" / "benchmarks"
    roots = explicit_roots or discover_benchmark_roots(packaged_root)
    by_id: dict[str, Path] = {}
    for root in roots:
        manifest = load_benchmark(root).manifest
        by_id[str(manifest.id)] = root
    return by_id


def _benchmark_root_for_record(
    record: Mapping[str, object],
    *,
    benchmark_roots: Mapping[str, Path],
    description: str,
) -> Path:
    benchmark_id = _benchmark_id_from_record(record, description=description)
    benchmark_root = benchmark_roots.get(benchmark_id)
    if benchmark_root is None:
        raise ValueError(f"no benchmark root available for {benchmark_id}")
    return benchmark_root


def _load_evaluation_summary_record(path: Path) -> dict[str, object]:
    record = _load_object_record(path, description="benchmark evaluation bundle")
    if record.get("format") != "leibniz.benchmark-evaluation":
        raise ValueError(f"unsupported benchmark evaluation bundle: {path}")
    return record


def _load_object_record(path: Path, *, description: str) -> dict[str, object]:
    record = load_object_document(path.read_bytes(), description=description)
    return dict(record)


def _benchmark_id_from_record(record: Mapping[str, object], *, description: str) -> str:
    benchmark_id = record.get("benchmark_id")
    if isinstance(benchmark_id, str) and benchmark_id:
        return benchmark_id
    raw_benchmark_manifest = record.get("benchmark_manifest")
    if isinstance(raw_benchmark_manifest, Mapping):
        benchmark_manifest = cast(Mapping[str, object], raw_benchmark_manifest)
        manifest_id = benchmark_manifest.get("id")
        if isinstance(manifest_id, str) and manifest_id:
            return manifest_id
    raw_checkpoint = record.get("model_checkpoint")
    if isinstance(raw_checkpoint, Mapping):
        checkpoint = cast(Mapping[str, object], raw_checkpoint)
        checkpoint_benchmark_id = checkpoint.get("benchmark_id")
        if isinstance(checkpoint_benchmark_id, str) and checkpoint_benchmark_id:
            return checkpoint_benchmark_id
    raise ValueError(f"{description}.benchmark_id must be a non-empty string")


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
        if artifact == "model-derivation":
            document = ModelDerivationCompatibilityReportDocument.from_bytes(args.path.read_bytes())
            print(f"valid model derivation compatibility report {document.report.id}")
            return 0
        if artifact == "evaluation-bundle":
            document = BenchmarkEvaluationBundleDocument.from_bytes(args.path.read_bytes())
            print(f"valid evaluation bundle {document.bundle.id}")
            return 0
        if artifact == "submission-registry":
            document = SubmissionRegistryDocument.from_bytes(args.path.read_bytes())
            print(f"valid submission registry {document.registry.id}")
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


if __name__ == "__main__":
    raise SystemExit(main())
