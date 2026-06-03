"""Command line artifact validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from leibniz.architecture_semantics import validate_architecture_semantics
from leibniz.architectures import ArchitectureManifestDocument
from leibniz.artifacts import ArtifactIndexDocument, ArtifactReferenceDocument
from leibniz.authority_indexes import AuthorityIndexDocument
from leibniz.benchmark_runner import BenchmarkRunPlan, run_benchmark
from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestDocument
from leibniz.documents import (
    canonical_document_bytes,
    document_filename_suffix,
    load_object_document,
)
from leibniz.federation_ingest import FederationIngestPlanDocument
from leibniz.formation_timing import FormationTimingPlan, time_formation_paths
from leibniz.local_results import (
    import_submission_publications,
    initialize_publication_checkout,
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
from leibniz.publications import SubmissionPublicationDocument
from leibniz.resources import ResourceReportDocument, ResourceReportSetDocument
from leibniz.submission_registries import SubmissionRegistry, SubmissionRegistryDocument
from leibniz.view_manifests import ViewManifestDocument

__all__ = ["main"]

_manifest_filename = "manifest" + document_filename_suffix()


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
    benchmark = subcommands.add_parser(
        "benchmark",
        description="run local benchmark workflows",
        help="run local benchmark workflows",
    )
    benchmark_subcommands = benchmark.add_subparsers(dest="benchmark_command", required=True)
    train = benchmark_subcommands.add_parser(
        "train",
        description="train and evaluate an explicit architecture locally",
        help="train an explicit architecture",
    )
    train.add_argument("--architecture", type=Path, required=True)
    train.add_argument("--benchmark-root", type=Path, required=True)
    train.add_argument(
        "--results-root",
        default=Path("results"),
        type=Path,
        help="local result checkout; defaults to results",
    )
    train.add_argument("--sample-count", default=512, type=int)
    train.add_argument("--evaluation-sample-count", default=None, type=int)
    train.add_argument("--seed", default=101, type=int)
    train.add_argument("--train-steps", default=None, type=int)
    train.add_argument("--learning-rate", default=0.01, type=float)
    train.add_argument("--optimizer", default="adam", choices=("sgd", "adam", "adamw"))
    train.add_argument(
        "--schedule",
        default="reduce-on-plateau",
        choices=("none", "cosine", "reduce-on-plateau"),
    )
    train.add_argument("--gate-check-interval", default=32, type=int)
    train.add_argument("--model-checkpoint-gate-interval", default=1, type=int)
    train.add_argument("--gate-sample-count", default=None, type=int)
    train.add_argument("--gate-decision-rule", default="validation-loss-plateau")
    train.add_argument("--convergence-patience", default=6, type=int)
    train.add_argument("--convergence-min-delta", default=1e-3, type=float)
    train.add_argument("--convergence-min-steps", default=500, type=int)
    train.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="tensor runtime device; auto prefers CUDA, then MPS, then CPU",
    )
    train.add_argument("--dry-run", action="store_true")
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
        if str(args.benchmark_command) == "train":
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
                    gate_check_interval=args.gate_check_interval,
                    model_checkpoint_gate_interval=args.model_checkpoint_gate_interval,
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
                f"{prefix} benchmark training run {summary.run_slug} "
                f"({summary.measurement_count} measurement(s))"
            )
            print(f"measurements: {summary.measurement_dataset_path}")
            print(f"model inspection: {summary.model_inspection_path}")
            print(f"training summary: {summary.training_summary_path}")
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
