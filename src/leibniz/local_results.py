"""Operator-local result import and console view materialization."""

from __future__ import annotations

import hashlib
import importlib
import math
import os
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.benchmark_evaluation import (
    CompetencePoint,
    sampled_competence_frontier_score,
)
from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestDocument
from leibniz.console.protocol import (
    console_protocol_format_versions,
    console_protocol_formats,
)
from leibniz.content import ContentDigest
from leibniz.documents import (
    canonical_document_bytes,
    document_filename_suffix,
    load_object_document,
)
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import (
    MeasurementDataset,
    MeasurementDatasetDocument,
    MeasurementRecord,
)
from leibniz.model_inspection import (
    ModelInspectionDocument,
    ModelInspectionRecord,
    ModelInspectionValidationError,
)
from leibniz.observation_generation import load_observation_generator
from leibniz.publications import SubmissionPublicationBundle, SubmissionPublicationDocument
from leibniz.records import RecordExtractor
from leibniz.submissions import SubmissionArtifact, SubmissionPackageManifest
from leibniz.training_runs import TrainingRunRecord
from leibniz.views import MeasurementScoreView

__all__ = [
    "LocalBenchmarkResultViewSummary",
    "LocalPublicationExportSummary",
    "LocalPublicationCheckoutSummary",
    "LocalPublicationPushSummary",
    "LocalResultImportError",
    "LocalResultImportSummary",
    "competent_complexity_score",
    "import_submission_publications",
    "initialize_publication_checkout",
    "load_console_result_view",
    "materialize_benchmark_result_views",
    "publish_local_benchmark_results",
    "push_publication_checkout",
]

_protocol_formats = console_protocol_formats()
_protocol_format_versions = console_protocol_format_versions()
_console_result_view_format = _protocol_formats.imported_result_view
_benchmark_result_view_format = _protocol_formats.benchmark_result_view
_console_result_view_format_version = _protocol_format_versions.result_view
_document_suffix = document_filename_suffix()
_manifest_filename = "manifest" + _document_suffix
_default_results_root = Path("results")
_publication_directories = (
    "evaluations",
    "imports/publication_bundles",
    "measurements",
    "models",
    "publication_bundles",
    "training",
    "views",
)
_benchmark_cost_axes: tuple[tuple[str, str], ...] = (
    ("parameter_count", "Parameters"),
    ("storage_bytes", "Storage"),
    ("inference_compute", "Compute"),
    ("training_compute", "Compute"),
)
_benchmark_cost_axis_keys = tuple(axis for axis, _label in _benchmark_cost_axes)


class _SummaryRecordMixin:
    def to_record(self) -> dict[str, object]:
        return _summary_record(self)


class LocalResultImportError(ValueError):
    """Raised when local result import cannot produce a valid console view."""


_extract = RecordExtractor(error_type=LocalResultImportError)


@dataclass(frozen=True, slots=True)
class LocalResultImportSummary(_SummaryRecordMixin):
    source_files: tuple[Path, ...]
    import_files: tuple[Path, ...]
    view_file: Path
    publication_bundle_count: int
    measurement_count: int


@dataclass(frozen=True, slots=True)
class LocalBenchmarkResultViewSummary(_SummaryRecordMixin):
    source_files: tuple[Path, ...]
    view_file: Path
    benchmark_count: int
    model_count: int
    run_count: int


@dataclass(frozen=True, slots=True)
class LocalPublicationExportSummary(_SummaryRecordMixin):
    source_files: tuple[Path, ...]
    publication_files: tuple[Path, ...]
    publication_bundle_count: int
    measurement_count: int
    git_commit: str | None = None
    git_pushed: bool = False
    remote: str | None = None
    remote_commit: str | None = None


@dataclass(frozen=True, slots=True)
class LocalPublicationCheckoutSummary(_SummaryRecordMixin):
    repo_id: str | None
    results_root: Path
    repo_url: str | None
    created_or_reused: bool
    scaffold_commit: str | None
    pushed: bool


@dataclass(frozen=True, slots=True)
class LocalPublicationPushSummary(_SummaryRecordMixin):
    results_root: Path
    pushed_commit: str


def _summary_record(summary: Any) -> dict[str, object]:
    record: dict[str, object] = {}
    for field in fields(summary):
        value = getattr(summary, field.name)
        if value is None:
            continue
        if isinstance(value, Path):
            record[field.name] = value.as_posix()
        elif isinstance(value, tuple):
            values = cast(tuple[object, ...], value)
            record[field.name] = (
                [path.as_posix() for path in cast(tuple[Path, ...], values)]
                if all(isinstance(item, Path) for item in values)
                else values
            )
        else:
            record[field.name] = value
    return record


def initialize_publication_checkout(
    *,
    repo_id: str | None,
    repository_root: Path | None = None,
    results_root: Path = _default_results_root,
    remote: str = "auto",
    local_only: bool = False,
    push: bool = False,
    commit_message: str = "Initialize Leibniz result publication checkout",
    token: str | None = None,
) -> LocalPublicationCheckoutSummary:
    repository_root = Path.cwd().resolve() if repository_root is None else repository_root.resolve()
    results_root = _resolve_output_root(repository_root, results_root)
    selected_remote: str | None = None
    if local_only:
        if push:
            raise LocalResultImportError("--push cannot be used with --local-only")
        repo_id = None
        repo_url = None
        created_or_reused = False
        results_root.mkdir(parents=True, exist_ok=True)
    else:
        if repo_id is None:
            raise LocalResultImportError("--repo is required unless --local-only is used")
        repo_id = _validate_hf_repo_id(repo_id)
        selected_remote = _select_publication_remote(
            repo_id=repo_id,
            remote=remote,
            results_root=results_root,
            token=token,
        )
        repo_url = _hf_dataset_url(repo_id)
        created_or_reused = selected_remote == "hf"
        if results_root.exists():
            if selected_remote == "git" and not _is_git_checkout(results_root):
                raise LocalResultImportError("results root exists but is not a Git checkout")
        elif selected_remote == "git":
            _git_clone(source=_hf_dataset_ssh_url(repo_id), target=results_root)
        else:
            results_root.mkdir(parents=True, exist_ok=True)
            _create_hf_dataset_repo(repo_id=repo_id, token=token)
    _ensure_publication_checkout_structure(results_root)
    scaffold_commit = _commit_checkout_if_dirty(
        results_root=results_root,
        message=commit_message,
        push=push,
        repo_id=repo_id,
        remote=remote,
        token=token,
    )
    if push and selected_remote == "hf" and repo_id is not None:
        _push_hf_api(
            results_root=results_root,
            repo_id=repo_id,
            message=commit_message,
            token=token,
        )
    return LocalPublicationCheckoutSummary(
        repo_id=repo_id,
        results_root=results_root,
        repo_url=repo_url,
        created_or_reused=created_or_reused,
        scaffold_commit=scaffold_commit,
        pushed=push and (scaffold_commit is not None or selected_remote == "hf"),
    )


def push_publication_checkout(
    *,
    repository_root: Path | None = None,
    results_root: Path = _default_results_root,
    repo_id: str | None = None,
    remote: str = "auto",
    token: str | None = None,
) -> LocalPublicationPushSummary:
    """Push an existing result-publication checkout without creating a commit."""

    repository_root = Path.cwd().resolve() if repository_root is None else repository_root.resolve()
    results_root = _resolve_output_root(repository_root, results_root)
    selected_remote = _select_publication_remote(
        repo_id=repo_id,
        remote=remote,
        results_root=results_root,
        token=token,
    )
    if selected_remote == "hf":
        if repo_id is None:
            raise LocalResultImportError("Hugging Face API push requires --repo")
        pushed_commit = _push_hf_api(
            results_root=results_root,
            repo_id=repo_id,
            message="Publish Leibniz result checkout",
            token=token,
        )
    elif _is_git_checkout(results_root):
        pushed_commit = _push_checkout(results_root=results_root)
    else:
        raise LocalResultImportError("results root must be a Git checkout when pushing")
    return LocalPublicationPushSummary(
        results_root=results_root,
        pushed_commit=pushed_commit,
    )


def publish_local_benchmark_results(
    *,
    repository_root: Path | None = None,
    results_root: Path = _default_results_root,
    commit: bool = True,
    push: bool = False,
    repo_id: str | None = None,
    remote: str = "auto",
    token: str | None = None,
    commit_message: str = "Publish Leibniz benchmark results",
) -> LocalPublicationExportSummary:
    repository_root = Path.cwd().resolve() if repository_root is None else repository_root.resolve()
    results_root = _resolve_output_root(repository_root, results_root)
    output_root = results_root / "publication_bundles"
    if push and not commit:
        raise LocalResultImportError("push requires committing the result checkout")
    manifests = _known_benchmark_manifests(repository_root)
    runs = _local_run_records(results_root)
    if not runs:
        raise LocalResultImportError("no local benchmark result records found")
    output_root.mkdir(parents=True, exist_ok=True)
    publication_files: list[Path] = []
    measurement_count = 0
    for run in runs:
        manifest = manifests.get(run.benchmark_id)
        if manifest is None:
            raise LocalResultImportError(
                f"unknown benchmark id in local results: {run.benchmark_id}"
            )
        bundle = _publication_bundle_for_local_run(
            manifest=manifest,
            repository_root=repository_root,
            run=run,
        )
        publication_file = output_root / _bundle_filename(bundle.id, bundle.digest)
        publication_file.write_bytes(canonical_document_bytes(bundle.to_record()) + b"\n")
        publication_files.append(publication_file)
        measurement_count += len(bundle.measurement_dataset.measurements)
    materialize_benchmark_result_views(repository_root=repository_root, results_root=results_root)
    git_commit: str | None = None
    selected_remote: str | None = None
    remote_commit: str | None = None
    if commit:
        git_commit, selected_remote, remote_commit = _commit_runs_checkout(
            results_root=results_root,
            message=commit_message,
            push=push,
            repo_id=repo_id,
            remote=remote,
            token=token,
        )
    return LocalPublicationExportSummary(
        source_files=tuple(run.source_path for run in runs),
        publication_files=tuple(publication_files),
        publication_bundle_count=len(publication_files),
        measurement_count=measurement_count,
        git_commit=git_commit,
        git_pushed=push and selected_remote == "git",
        remote=selected_remote,
        remote_commit=remote_commit,
    )


def import_submission_publications(
    source_roots: Iterable[Path],
    *,
    repository_root: Path | None = None,
    results_root: Path = _default_results_root,
) -> LocalResultImportSummary:
    """Import local publication bundles into a result checkout and console views."""

    repository_root = Path.cwd().resolve() if repository_root is None else repository_root.resolve()
    results_root = _resolve_output_root(repository_root, results_root)
    known_benchmark_ids = _known_benchmark_ids(repository_root)
    documents = tuple(_publication_documents(source_roots))
    if not documents:
        raise LocalResultImportError("no publication bundle documents found")

    measurement_records: dict[ProtocolIdentifier, Mapping[str, object]] = {}
    publication_views: list[Mapping[str, object]] = []
    imported_files: list[Path] = []
    source_files: list[Path] = []

    import_root = results_root / "imports" / "publication_bundles"
    view_root = results_root / "views"
    import_root.mkdir(parents=True, exist_ok=True)
    view_root.mkdir(parents=True, exist_ok=True)

    for source_file, document in documents:
        bundle = document.bundle
        if bundle.submission_package.benchmark_manifest.id not in known_benchmark_ids:
            raise LocalResultImportError(
                "unknown benchmark id in imported submission package: "
                f"{bundle.submission_package.benchmark_manifest.id}"
            )
        _validate_known_benchmarks(bundle.measurement_dataset.measurements, known_benchmark_ids)
        for measurement in bundle.measurement_dataset.measurements:
            measurement_id = measurement.raw_scoring_evidence.id
            measurement_record = measurement.to_record()
            previous = measurement_records.get(measurement_id)
            if previous is not None and previous != measurement_record:
                raise LocalResultImportError(
                    f"conflicting measurement record for {measurement_id}"
                )
            measurement_records[measurement_id] = measurement_record

        import_file = import_root / _bundle_filename(bundle.id, document.digest)
        import_file.write_bytes(canonical_document_bytes(bundle.to_record()) + b"\n")
        imported_files.append(import_file)
        source_files.append(source_file)
        publication_views.append(
            {
                "id": str(bundle.id),
                "digest": str(document.digest),
                "source_path": source_file.as_posix(),
                "submission_package_id": str(bundle.submission_package.id),
                "benchmark_ids": sorted(
                    {
                        str(measurement.benchmark_id)
                        for measurement in bundle.measurement_dataset.measurements
                    }
                ),
                "measurement_count": len(bundle.measurement_dataset.measurements),
                "measurement_dataset": bundle.measurement_dataset.to_record(),
                "measurement_score_view": bundle.measurement_score_view.to_record(),
            }
        )

    view_file = view_root / ("imported_results" + _document_suffix)
    view_file.write_bytes(
        canonical_document_bytes(
            {
                "format": _console_result_view_format,
                "format_version": _console_result_view_format_version,
                "publication_bundles": publication_views,
            }
        )
        + b"\n"
    )
    return LocalResultImportSummary(
        source_files=tuple(source_files),
        import_files=tuple(imported_files),
        view_file=view_file,
        publication_bundle_count=len(publication_views),
        measurement_count=len(measurement_records),
    )


def materialize_benchmark_result_views(
    *,
    repository_root: Path | None = None,
    results_root: Path = _default_results_root,
) -> LocalBenchmarkResultViewSummary:
    """Derive console benchmark result views from ignored run/import state."""

    repository_root = Path.cwd().resolve() if repository_root is None else repository_root.resolve()
    results_root = _resolve_output_root(repository_root, results_root)
    manifests = _known_benchmark_manifests(repository_root)
    local_runs = _local_run_records(results_root)
    local_training_estimates = _local_training_estimate_records(
        results_root,
        repository_root=repository_root,
        accepted_run_slugs={run.run_slug for run in local_runs},
    )
    imported_runs = _imported_run_records(results_root)
    runs = tuple(
        sorted(
            (*local_runs, *local_training_estimates, *imported_runs),
            key=_run_sort_key,
        )
    )
    if not runs:
        raise LocalResultImportError("no benchmark result records found")

    benchmark_records: list[Mapping[str, object]] = []
    for benchmark_id in sorted({run.benchmark_id for run in runs}, key=str):
        manifest = manifests.get(benchmark_id)
        if manifest is None:
            raise LocalResultImportError(f"unknown benchmark id in local results: {benchmark_id}")
        benchmark_runs = tuple(run for run in runs if run.benchmark_id == benchmark_id)
        benchmark_competitions = tuple(
            competition
            for competition in _local_competition_records(results_root)
            if competition.get("benchmark_id") == str(benchmark_id)
        )
        benchmark_records.append(
            _benchmark_result_record(
                manifest=manifest,
                repository_root=repository_root,
                runs=benchmark_runs,
                competitions=benchmark_competitions,
            )
        )

    view_root = results_root / "views"
    view_root.mkdir(parents=True, exist_ok=True)
    view_file = view_root / ("benchmark_results" + _document_suffix)
    view_file.write_bytes(
        canonical_document_bytes(
            {
                "format": _benchmark_result_view_format,
                "format_version": _console_result_view_format_version,
                "benchmark_results": benchmark_records,
            }
        )
        + b"\n"
    )
    return LocalBenchmarkResultViewSummary(
        source_files=tuple(run.source_path for run in runs),
        view_file=view_file,
        benchmark_count=len(benchmark_records),
        model_count=len({(run.benchmark_id, run.model_key) for run in runs}),
        run_count=len(runs),
    )


def load_console_result_view(data: bytes) -> Mapping[str, object]:
    """Load a generated console result view document."""

    try:
        record = load_object_document(data, description="console result view")
    except ValueError as error:
        raise LocalResultImportError(str(error)) from error
    if record.get("format") == _benchmark_result_view_format:
        _validate_benchmark_result_view(record)
        return record
    if record.get("format") != _console_result_view_format:
        raise LocalResultImportError("console result view has unsupported format")
    if record.get("format_version") != _console_result_view_format_version:
        raise LocalResultImportError("console result view has unsupported format_version")
    publication_bundles = _as_sequence(record.get("publication_bundles"), "publication_bundles")
    for index, publication_bundle in enumerate(publication_bundles):
        _validate_publication_bundle_view(
            _extract.mapping(publication_bundle, f"publication_bundles.{index}")
        )
    return record


@dataclass(frozen=True, slots=True)
class _BenchmarkRunRecord:
    source_kind: str
    result_status: str
    source_path: Path
    run_id: str
    run_slug: str
    benchmark_id: ProtocolIdentifier
    architecture_digest: ContentDigest
    model_key: str
    complexity: float | None
    measurement_count: int
    score: float
    cost_summary: Mapping[str, object]
    architecture: Mapping[str, object]
    model_inspection: Mapping[str, object]
    model_inspection_digest: ContentDigest
    model_inspection_path: Path | None
    measurement_dataset: MeasurementDataset
    measurement_dataset_digest: ContentDigest
    sampled_competence: Mapping[str, object] | None = None
    training_summary: Mapping[str, object] | None = None

    def to_record(self, *, complexity_axis: str | None = None) -> dict[str, object]:
        record: dict[str, object] = {
            "source_kind": self.source_kind,
            "result_status": self.result_status,
            "source_path": self.source_path.as_posix(),
            "run_id": self.run_id,
            "run_slug": self.run_slug,
            "benchmark_id": str(self.benchmark_id),
            "architecture_digest": str(self.architecture_digest),
            "model_key": self.model_key,
            "measurement_count": self.measurement_count,
            "score": self.score,
            "cost_summary": _run_cost_summary(self),
            "architecture": dict(self.architecture),
            "model_inspection_digest": str(self.model_inspection_digest),
            "measurement_dataset_digest": str(self.measurement_dataset_digest),
            "console_view_model": _run_console_view_model(
                run=self,
                complexity_axis=complexity_axis,
            ),
        }
        if self.model_inspection_path is not None:
            record["model_inspection_path"] = self.model_inspection_path.as_posix()
        if self.complexity is not None:
            record["complexity"] = self.complexity
        if self.sampled_competence is not None:
            record["sampled_competence"] = dict(self.sampled_competence)
        if self.training_summary is not None:
            record["training_diagnostics"] = _training_diagnostics_record(run=self)
        return record


@dataclass(frozen=True, slots=True)
class _ModelCompetitionOutcome:
    left_model_key: str
    right_model_key: str
    left_score: float
    right_score: float
    sample_count: int


def _publication_documents(
    source_roots: Iterable[Path],
) -> tuple[tuple[Path, SubmissionPublicationDocument], ...]:
    documents: list[tuple[Path, SubmissionPublicationDocument]] = []
    for source_root in source_roots:
        for path in _candidate_document_files(source_root):
            try:
                document = SubmissionPublicationDocument.from_bytes(path.read_bytes())
            except ValueError:
                continue
            documents.append((path.resolve(), document))
    return tuple(sorted(documents, key=lambda item: item[0].as_posix()))


def _candidate_document_files(source_root: Path) -> tuple[Path, ...]:
    root = source_root.resolve()
    if root.is_file():
        return (root,) if root.suffix == _document_suffix else ()
    if not root.is_dir():
        raise LocalResultImportError(f"source root does not exist: {source_root}")
    return tuple(sorted(path for path in root.rglob("*" + _document_suffix) if path.is_file()))


def _known_benchmark_ids(repository_root: Path) -> frozenset[ProtocolIdentifier]:
    return frozenset(_known_benchmark_manifests(repository_root))


def _known_benchmark_manifests(
    repository_root: Path,
) -> dict[ProtocolIdentifier, BenchmarkManifest]:
    benchmark_root = repository_root / "src" / "leibniz" / "benchmarks"
    manifests: dict[ProtocolIdentifier, BenchmarkManifest] = {}
    for path in sorted(benchmark_root.rglob(_manifest_filename)):
        document = BenchmarkManifestDocument.from_bytes(path.read_bytes())
        manifests[document.manifest.id] = document.manifest
    if not manifests:
        raise LocalResultImportError("no known benchmark manifests found")
    return manifests


def _local_run_records(results_root: Path) -> tuple[_BenchmarkRunRecord, ...]:
    evaluation_root = results_root / "evaluations"
    if not evaluation_root.is_dir():
        return ()
    records: list[_BenchmarkRunRecord] = []
    for path in sorted(evaluation_root.rglob("*" + _document_suffix)):
        summary = load_object_document(path.read_bytes(), description="benchmark evaluation")
        if summary.get("format") != "leibniz.benchmark-evaluation":
            continue
        training_summary_path = _local_artifact_path(
            results_root=results_root,
            value=summary.get("training_summary_path"),
            field="training_summary_path",
        )
        training_summary = load_object_document(
            training_summary_path.read_bytes(),
            description="training summary",
        )
        measurement_dataset_path = _local_artifact_path(
            results_root=results_root,
            value=summary.get("measurement_dataset_path"),
            field="measurement_dataset_path",
        )
        model_inspection_path = _local_artifact_path(
            results_root=results_root,
            value=summary.get("model_inspection_path"),
            field="model_inspection_path",
        )
        dataset = MeasurementDatasetDocument.from_bytes(
            measurement_dataset_path.read_bytes()
        ).dataset
        inspection = ModelInspectionDocument.from_bytes(
            model_inspection_path.read_bytes()
        ).inspection
        _validate_model_checkpoint_artifacts(
            results_root=results_root,
            training_summary=training_summary,
        )
        sampled_competence = _sampled_competence(summary)
        records.append(
            _BenchmarkRunRecord(
                source_kind="local-run",
                result_status="accepted",
                source_path=path.resolve(),
                run_id=_extract.non_empty_string(summary.get("run_slug"), "run_slug"),
                run_slug=_extract.non_empty_string(summary.get("run_slug"), "run_slug"),
                benchmark_id=_as_identifier(summary.get("benchmark_id"), "benchmark_id"),
                architecture_digest=_record_digest(inspection.architecture.to_record()),
                model_key=str(inspection.architecture.record_digest),
                complexity=_sampled_competence_complexity(summary),
                measurement_count=len(dataset.measurements),
                score=_mean_accepted_mass(dataset),
                cost_summary=inspection.cost_summary.to_record(),
                architecture=inspection.architecture.to_record(),
                model_inspection=_model_inspection_view_record(
                    inspection=inspection.to_record(),
                    source_path=model_inspection_path.resolve(),
                    measurement_dataset_digest=dataset.digest,
                    training_summary=summary,
                ),
                model_inspection_digest=inspection.digest,
                model_inspection_path=model_inspection_path.resolve(),
                measurement_dataset=dataset,
                measurement_dataset_digest=dataset.digest,
                sampled_competence=sampled_competence,
                training_summary=training_summary,
            )
        )
    return tuple(records)


def _local_training_estimate_records(
    results_root: Path,
    *,
    repository_root: Path,
    accepted_run_slugs: set[str],
) -> tuple[_BenchmarkRunRecord, ...]:
    training_root = results_root / "training"
    if not training_root.is_dir():
        return ()
    records: list[_BenchmarkRunRecord] = []
    empty_dataset = MeasurementDataset(measurements=())
    for path in sorted(training_root.rglob("*" + _document_suffix)):
        summary = load_object_document(path.read_bytes(), description="training record")
        if summary.get("format") not in {
            "leibniz.benchmark-run",
            "leibniz.benchmark-training-progress",
        }:
            continue
        run_slug = _extract.non_empty_string(summary.get("run_slug"), "run_slug")
        if run_slug in accepted_run_slugs:
            continue
        estimate = _extract.optional_mapping(
            summary.get("training_estimate"),
            "training_estimate",
        )
        if estimate is None:
            continue
        architecture = _architecture_manifest_from_training_record(
            repository_root=repository_root,
            training_summary=summary,
        )
        inspection = _inspection_from_architecture(architecture)
        cost_summary = dict(_extract.mapping(summary.get("cost_summary"), "cost_summary"))
        records.append(
            _BenchmarkRunRecord(
                source_kind="local-training-estimate",
                result_status="tentative",
                source_path=path.resolve(),
                run_id=run_slug,
                run_slug=run_slug,
                benchmark_id=_as_identifier(summary.get("benchmark_id"), "benchmark_id"),
                architecture_digest=architecture.digest,
                model_key=str(architecture.digest),
                complexity=_sampled_competence_record_complexity(estimate),
                measurement_count=0,
                score=_as_nonnegative_number(
                    estimate.get("mean_accepted_mass"),
                    "training_estimate.mean_accepted_mass",
                ),
                cost_summary=cost_summary,
                architecture=architecture.to_record(),
                model_inspection=_model_inspection_view_record(
                    inspection=inspection.to_record(),
                    source_path=path.resolve(),
                    measurement_dataset_digest=empty_dataset.digest,
                    training_summary=summary,
                ),
                model_inspection_digest=inspection.digest,
                model_inspection_path=None,
                measurement_dataset=empty_dataset,
                measurement_dataset_digest=empty_dataset.digest,
                sampled_competence=estimate,
                training_summary=summary,
            )
        )
    return tuple(records)


def _imported_run_records(results_root: Path) -> tuple[_BenchmarkRunRecord, ...]:
    import_root = results_root / "imports" / "publication_bundles"
    if not import_root.is_dir():
        return ()
    records: list[_BenchmarkRunRecord] = []
    for path in sorted(import_root.rglob("*" + _document_suffix)):
        document = SubmissionPublicationDocument.from_bytes(path.read_bytes())
        bundle = document.bundle
        package = bundle.submission_package
        inspection = _inspection_from_architecture(package.architecture_manifest)
        inspection_record = inspection.to_record()
        cost_summary = inspection.cost_summary.to_record()
        if package.model_metadata is not None:
            metadata_cost_summary = package.model_metadata.get("cost_summary")
            if metadata_cost_summary is not None:
                cost_summary.update(
                    _extract.mapping(
                        metadata_cost_summary,
                        "submission_package.model_metadata.cost_summary",
                    )
                )
        records.append(
            _BenchmarkRunRecord(
                source_kind="imported-publication",
                result_status="accepted",
                source_path=path.resolve(),
                run_id=str(bundle.id),
                run_slug=str(bundle.id.name),
                benchmark_id=package.benchmark_manifest.id,
                architecture_digest=package.architecture_manifest.digest,
                model_key=str(package.architecture_manifest.digest),
                complexity=_sampled_competence_record_complexity(
                    package.sampled_competence,
                ),
                measurement_count=len(bundle.measurement_dataset.measurements),
                score=_mean_accepted_mass(bundle.measurement_dataset),
                cost_summary=cost_summary,
                architecture=inspection.architecture.to_record(),
                model_inspection=_model_inspection_view_record(
                    inspection=inspection_record,
                    source_path=path.resolve(),
                    measurement_dataset_digest=bundle.measurement_dataset.digest,
                    training_summary=None,
                ),
                model_inspection_digest=inspection.digest,
                model_inspection_path=None,
                measurement_dataset=bundle.measurement_dataset,
                measurement_dataset_digest=bundle.measurement_dataset.digest,
                sampled_competence=package.sampled_competence,
            )
        )
    return tuple(records)


def _local_competition_records(results_root: Path) -> tuple[Mapping[str, object], ...]:
    evaluation_root = results_root / "evaluations"
    if not evaluation_root.is_dir():
        return ()
    records: list[Mapping[str, object]] = []
    for path in sorted(evaluation_root.rglob("competitions/*" + _document_suffix)):
        record = load_object_document(path.read_bytes(), description="benchmark competition")
        _validate_competition_record(record, "competition")
        records.append(record)
    return tuple(records)


def _sampled_competence(summary: Mapping[str, object]) -> Mapping[str, object] | None:
    evidence = summary.get("sampled_competence")
    if evidence is None:
        return None
    return _extract.mapping(evidence, "sampled_competence")


def _sampled_competence_complexity(summary: Mapping[str, object]) -> float | None:
    record = _sampled_competence(summary)
    return _sampled_competence_record_complexity(record)


def _sampled_competence_record_complexity(
    record: Mapping[str, object] | None,
) -> float | None:
    if record is None:
        return None
    return _as_nonnegative_number(record.get("complexity"), "sampled_competence.complexity")


def _model_inspection_view_record(
    *,
    inspection: Mapping[str, object],
    source_path: Path,
    measurement_dataset_digest: ContentDigest,
    training_summary: Mapping[str, object] | None,
) -> Mapping[str, object]:
    record = dict(inspection)
    record["source_path"] = source_path.as_posix()
    record["measurement_dataset"] = {
        "kind": "measurement-dataset",
        "content_digest": str(measurement_dataset_digest),
    }
    if training_summary is not None:
        record["training_provenance"] = [
            {
                "kind": "training-run",
                "record_digest": str(ContentDigest.from_value(training_summary)),
            }
        ]
    return record


def _training_diagnostics_record(run: _BenchmarkRunRecord) -> Mapping[str, object]:
    if run.training_summary is None:
        raise LocalResultImportError("training summary is required for diagnostics")
    training_run = TrainingRunRecord.from_record(
        _extract.mapping(run.training_summary.get("training_run"), "training_run")
    )
    history = training_run.validation_history
    final = history[-1]
    record: dict[str, object] = {
        "status": training_run.status,
        "stop_reason": training_run.stop_reason,
        "steps_run": training_run.steps_run,
        "validation_checks": training_run.validation_checks,
        "final_validation_loss": final.validation_loss,
        "final_validation_step": final.step,
        "final_validation_check": final.validation_check,
        "protocol": training_run.protocol.to_record(),
        "validation_history": [point.to_record() for point in history],
        "artifacts": _training_artifact_references(run),
    }
    if training_run.training_compute is not None:
        record["training_compute"] = training_run.training_compute
    throughput = run.training_summary.get("throughput")
    if isinstance(throughput, Mapping):
        record["throughput"] = dict(cast(Mapping[str, object], throughput))
    evaluation_curriculum = run.training_summary.get("evaluation_curriculum")
    if isinstance(evaluation_curriculum, Mapping):
        record["evaluation_curriculum"] = dict(
            cast(Mapping[str, object], evaluation_curriculum)
        )
    training_curriculum = run.training_summary.get("training_curriculum")
    if isinstance(training_curriculum, Mapping):
        record["training_curriculum"] = dict(
            cast(Mapping[str, object], training_curriculum)
        )
    return record


def _training_artifact_references(run: _BenchmarkRunRecord) -> list[dict[str, object]]:
    if run.training_summary is None:
        raise LocalResultImportError("training summary is required for artifact references")
    references: list[dict[str, object]] = [
        {
            "kind": "measurement-dataset",
            "digest": str(run.measurement_dataset_digest),
        },
        {
            "kind": "model-inspection",
            "digest": str(run.model_inspection_digest),
        },
        {
            "kind": "training-summary",
            "digest": str(ContentDigest.from_value(run.training_summary)),
            "path": run.source_path.as_posix(),
        },
    ]
    if run.model_inspection_path is not None:
        references[1]["path"] = run.model_inspection_path.as_posix()
    for checkpoint in _model_checkpoint_records(run.training_summary):
        references.append(
            {
                "kind": "model-checkpoint",
                "digest": _extract.non_empty_string(
                    checkpoint.get("digest"),
                    "model_checkpoints.digest",
                ),
                "path": _extract.non_empty_string(
                    checkpoint.get("path"),
                    "model_checkpoints.path",
                ),
            }
        )
        references.append(
            {
                "kind": "model-manifest",
                "digest": _extract.non_empty_string(
                    checkpoint.get("manifest_digest"),
                    "model_checkpoints.manifest_digest",
                ),
                "path": _extract.non_empty_string(
                    checkpoint.get("manifest_path"),
                    "model_checkpoints.manifest_path",
                ),
            }
        )
    return references


def _model_checkpoint_records(
    training_summary: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        _extract.mapping(item, "model_checkpoints")
        for item in _as_sequence(
            training_summary.get("model_checkpoints", ()),
            "model_checkpoints",
        )
    )


def _validate_model_checkpoint_artifacts(
    *,
    results_root: Path,
    training_summary: Mapping[str, object],
) -> None:
    for checkpoint in _model_checkpoint_records(training_summary):
        checkpoint_path = _local_artifact_path(
            results_root=results_root,
            value=checkpoint.get("path"),
            field="model_checkpoints.path",
        )
        expected_digest = ContentDigest.from_string(
            checkpoint.get("digest"),
            field="model_checkpoints.digest",
            error_type=LocalResultImportError,
        )
        actual_digest = _file_content_digest(checkpoint_path)
        if actual_digest != expected_digest:
            raise LocalResultImportError(
                f"model_checkpoints.path digest mismatch: {checkpoint_path}"
            )
        manifest_path = _local_artifact_path(
            results_root=results_root,
            value=checkpoint.get("manifest_path"),
            field="model_checkpoints.manifest_path",
        )
        manifest_record = load_object_document(
            manifest_path.read_bytes(),
            description="model checkpoint manifest",
        )
        expected_manifest_digest = ContentDigest.from_string(
            checkpoint.get("manifest_digest"),
            field="model_checkpoints.manifest_digest",
            error_type=LocalResultImportError,
        )
        actual_manifest_digest = ContentDigest.from_value(manifest_record)
        if actual_manifest_digest != expected_manifest_digest:
            raise LocalResultImportError(
                f"model_checkpoints.manifest_path digest mismatch: {manifest_path}"
            )


def _file_content_digest(path: Path) -> ContentDigest:
    return ContentDigest(
        algorithm="sha256",
        hex=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def _run_console_view_model(
    *,
    run: _BenchmarkRunRecord,
    complexity_axis: str | None,
) -> Mapping[str, object]:
    sections: list[Mapping[str, object]] = []
    if run.sampled_competence is not None:
        sections.append(
            _console_detail_entries_section(
                title="Sampled Competence",
                entries=(
                    (
                        complexity_axis or "Complexity",
                        _console_number_value(run.sampled_competence.get("complexity")),
                    ),
                    ("Samples", _console_number_value(run.sampled_competence.get("sample_count"))),
                    (
                        "Mean Score",
                        _console_number_value(
                            run.sampled_competence.get("mean_accepted_mass"),
                            precision=4,
                        ),
                    ),
                    (
                        "Sampling",
                        _console_string_value(run.sampled_competence.get("sampling_rule")),
                    ),
                ),
            )
        )
    if run.training_summary is not None:
        diagnostics = _training_diagnostics_record(run)
        protocol = _extract.mapping(diagnostics.get("protocol"), "training_diagnostics.protocol")
        sections.append(
            _console_detail_entries_section(
                title="Training Protocol",
                entries=(
                    ("Objective", _console_string_value(protocol.get("objective"))),
                    ("Optimizer", _console_string_value(protocol.get("optimizer"))),
                    ("Schedule", _console_string_value(protocol.get("schedule"))),
                    (
                        "Learning Rate",
                        _console_number_value(protocol.get("learning_rate"), precision=4),
                    ),
                    ("Steps", _console_number_value(protocol.get("max_steps"))),
                    ("Batch", _console_number_value(protocol.get("batch_size"))),
                    ("Gate Check", _console_number_value(protocol.get("gate_check_interval"))),
                    (
                        "Checkpoint Gate",
                        _console_number_value(
                            run.training_summary.get("model_checkpoint_gate_interval")
                        ),
                    ),
                    ("Gate Samples", _console_number_value(protocol.get("gate_sample_count"))),
                    ("Gate Rule", _console_string_value(protocol.get("gate_decision_rule"))),
                    ("Validation", _console_string_value(protocol.get("validation_source"))),
                ),
            )
        )
        sections.append(
            _console_detail_entries_section(
                title="Training Outcome",
                entries=(
                    ("Status", _console_string_value(diagnostics.get("status"))),
                    ("Stop", _console_string_value(diagnostics.get("stop_reason"))),
                    (
                        "Final Loss",
                        _console_number_value(
                            diagnostics.get("final_validation_loss"),
                            precision=4,
                        ),
                    ),
                    ("Checks", _console_number_value(diagnostics.get("validation_checks"))),
                ),
            )
        )
        throughput = diagnostics.get("throughput")
        if isinstance(throughput, Mapping):
            throughput_record = cast(Mapping[str, object], throughput)
            training_throughput = _extract.mapping(
                throughput_record.get("training"),
                "training_diagnostics.throughput.training",
            )
            evaluation_throughput = _extract.mapping(
                throughput_record.get("evaluation"),
                "training_diagnostics.throughput.evaluation",
            )
            phase_timing = (
                _extract.mapping(
                    throughput_record.get("phase_timing"),
                    "training_diagnostics.throughput.phase_timing",
                )
                if "phase_timing" in throughput_record
                else None
            )
            roofline_comparison = _extract.mapping(
                throughput_record.get("roofline_comparison"),
                "training_diagnostics.throughput.roofline_comparison",
            )
            sections.append(
                _console_detail_entries_section(
                    title="Throughput",
                    entries=(
                        (
                            "Training",
                            _console_samples_per_second(training_throughput),
                        ),
                        (
                            "Evaluation",
                            _console_samples_per_second(evaluation_throughput),
                        ),
                        (
                            "Roofline",
                            _console_string_value(roofline_comparison.get("status")),
                        ),
                        (
                            "Slowest Phase",
                            _slowest_phase_label(phase_timing),
                        ),
                    ),
                )
            )
        history = _as_sequence(
            diagnostics.get("validation_history"),
            "training_diagnostics.validation_history",
        )
        if history:
            sections.append(
                {
                    "title": "Validation History",
                    "table": {
                        "aria_label": "Validation history",
                        "columns": ["Step", "Loss", "Stale"],
                        "rows": [_console_validation_history_row(point) for point in history],
                    },
                }
            )
    return {"detail_sections": sections}


def _console_detail_entries_section(
    *,
    title: str,
    entries: tuple[tuple[str, str], ...],
) -> Mapping[str, object]:
    return {
        "title": title,
        "entries": [{"label": label, "value": value} for label, value in entries],
    }


def _console_string_value(value: object) -> str:
    return value if isinstance(value, str) and value else "unknown"


def _console_number_value(value: object, *, precision: int = 0) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return "unknown"
    if precision == 0:
        return f"{value:,}" if isinstance(value, int) else f"{value:,.0f}"
    return f"{value:.{precision}f}"


def _console_samples_per_second(record: Mapping[str, object]) -> str:
    value = record.get("samples_per_second")
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return "unknown"
    return f"{float(value):,.1f}/s"


def _slowest_phase_label(record: Mapping[str, object] | None) -> str:
    if record is None:
        return "unknown"
    phases = record.get("phases")
    if not isinstance(phases, Mapping):
        return "unknown"
    phase_map = cast(Mapping[object, object], phases)
    slowest_name = ""
    slowest_seconds = -1.0
    for name, phase in phase_map.items():
        if not isinstance(name, str) or not isinstance(phase, Mapping):
            continue
        phase_record = cast(Mapping[str, object], phase)
        seconds = phase_record.get("seconds")
        if (
            isinstance(seconds, int | float)
            and not isinstance(seconds, bool)
            and math.isfinite(float(seconds))
            and float(seconds) > slowest_seconds
        ):
            slowest_name = name
            slowest_seconds = float(seconds)
    if not slowest_name:
        return "unknown"
    return slowest_name.replace("_", " ")


def _console_validation_history_row(point: object) -> list[str]:
    point_record = _extract.mapping(point, "validation_history")
    return [
        _console_number_value(point_record.get("step")),
        _console_number_value(point_record.get("validation_loss"), precision=4),
        _console_number_value(point_record.get("stale_checks")),
    ]


def _publication_bundle_for_local_run(
    *,
    manifest: BenchmarkManifest,
    repository_root: Path,
    run: _BenchmarkRunRecord,
) -> SubmissionPublicationBundle:
    identifier_stem = f"{_identifier_atom(run.benchmark_id)}.{run.run_slug}"
    package = SubmissionPackageManifest(
        id=ProtocolIdentifier.parse(f"submissions.{identifier_stem}@0.1.0"),
        benchmark_manifest=manifest,
        architecture_manifest=_local_run_architecture_manifest(
            repository_root=repository_root,
            run=run,
        ),
        measurement_dataset=run.measurement_dataset,
        sampled_competence=run.sampled_competence,
        artifacts=_submission_artifacts_for_local_run(run),
        model_metadata={
            "cost_summary": _run_cost_summary(run),
        },
    )
    score_view = MeasurementScoreView.from_dataset(
        id=ProtocolIdentifier.parse(f"views.measurement-scores.{identifier_stem}@0.1.0"),
        dataset=run.measurement_dataset,
    )
    return SubmissionPublicationBundle(
        id=ProtocolIdentifier.parse(f"publication-bundles.{identifier_stem}@0.1.0"),
        submission_package=package,
        measurement_dataset=run.measurement_dataset,
        measurement_score_view=score_view,
    )


def _local_run_architecture_manifest(
    *,
    repository_root: Path,
    run: _BenchmarkRunRecord,
) -> ArchitectureManifest:
    if run.training_summary is None:
        raise LocalResultImportError("training summary is required for publication")
    return _architecture_manifest_from_training_record(
        repository_root=repository_root,
        training_summary=run.training_summary,
    )


def _architecture_manifest_from_training_record(
    *,
    repository_root: Path,
    training_summary: Mapping[str, object],
) -> ArchitectureManifest:
    path = Path(
        _extract.non_empty_string(
            training_summary.get("architecture_path"), "architecture_path"
        )
    )
    resolved = path if path.is_absolute() else repository_root / path
    if not resolved.is_file():
        raise LocalResultImportError(f"architecture_path does not exist: {path}")
    return ArchitectureManifestDocument.from_bytes(resolved.read_bytes()).manifest


def _submission_artifacts_for_local_run(
    run: _BenchmarkRunRecord,
) -> tuple[SubmissionArtifact, ...]:
    identifier_stem = f"{_identifier_atom(run.benchmark_id)}.{run.run_slug}"
    artifacts = [
        SubmissionArtifact(
            id=ProtocolIdentifier.parse(f"artifacts.{identifier_stem}.measurement-dataset@0.1.0"),
            digest=run.measurement_dataset_digest,
            description="measurement dataset",
        ),
        SubmissionArtifact(
            id=ProtocolIdentifier.parse(f"artifacts.{identifier_stem}.model-inspection@0.1.0"),
            digest=run.model_inspection_digest,
            description="model inspection",
        ),
    ]
    if run.training_summary is not None:
        artifacts.append(
            SubmissionArtifact(
                id=ProtocolIdentifier.parse(
                    f"artifacts.{identifier_stem}.training-summary@0.1.0"
                ),
                digest=ContentDigest.from_value(run.training_summary),
                description="training summary",
            )
        )
    return tuple(artifacts)


def _inspection_from_architecture(architecture: ArchitectureManifest) -> ModelInspectionRecord:
    digest_atom = str(architecture.digest).split(":", maxsplit=1)[1][:16]
    return ModelInspectionRecord.from_architecture(
        id=ProtocolIdentifier.parse(
            f"model-inspections.imported.sha-{digest_atom}@0.1.0"
        ),
        architecture_manifest=architecture,
    )


def _benchmark_result_record(
    *,
    manifest: BenchmarkManifest,
    repository_root: Path,
    runs: tuple[_BenchmarkRunRecord, ...],
    competitions: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    accepted_runs = tuple(run for run in runs if run.result_status == "accepted")
    models = tuple(
        _model_result_records(
            accepted_runs,
            manifest=manifest,
            repository_root=repository_root,
            competitions=competitions,
        )
    )
    model_candidates = tuple(
        _model_result_records(
            runs,
            manifest=manifest,
            repository_root=repository_root,
            competitions=(),
        )
    )
    record: dict[str, object] = {
        "benchmark_id": str(runs[0].benchmark_id),
        "cost_axes": _benchmark_cost_axis_records(),
        "score_axes": _benchmark_score_axis_records(models),
        "leaderboard": list(models),
        "model_candidates": list(model_candidates),
        "frontiers": {
            axis: _frontier_records(models, cost_axis=axis)
            for axis in _benchmark_cost_axis_keys
        },
        "training_history": [run.to_record(complexity_axis=None) for run in runs],
        "plot_runs": [run.to_record(complexity_axis=None) for run in runs],
        "model_inspections": _model_inspection_records(accepted_runs),
    }
    return record


def _benchmark_cost_axis_records() -> list[dict[str, object]]:
    return [
        {
            "key": key,
            "label": label,
        }
        for key, label in _benchmark_cost_axes
    ]


def _benchmark_score_axis_records(
    models: tuple[Mapping[str, object], ...],
) -> list[dict[str, object]]:
    axes: list[dict[str, object]] = [{"key": "absolute", "label": "Absolute"}]
    if any(
        "relative" in _extract.mapping(model.get("score_views"), "score_views")
        for model in models
    ):
        axes.append({"key": "relative", "label": "Relative"})
    return axes


def _model_inspection_records(
    runs: tuple[_BenchmarkRunRecord, ...],
) -> list[Mapping[str, object]]:
    by_digest: dict[str, Mapping[str, object]] = {}
    for run in runs:
        by_digest.setdefault(str(run.model_inspection_digest), run.model_inspection)
    return [by_digest[digest] for digest in sorted(by_digest)]


def _model_result_records(
    runs: tuple[_BenchmarkRunRecord, ...],
    *,
    manifest: BenchmarkManifest,
    repository_root: Path,
    competitions: tuple[Mapping[str, object], ...] = (),
) -> tuple[dict[str, object], ...]:
    grouped: dict[str, list[_BenchmarkRunRecord]] = {}
    for run in runs:
        grouped.setdefault(run.model_key, []).append(run)

    reference_baseline_complexity = _benchmark_base_complexity(
        manifest=manifest,
        repository_root=repository_root,
    )
    chance_mass = _chance_mass(manifest)
    records: list[dict[str, object]] = []
    best_runs: dict[str, _BenchmarkRunRecord] = {}
    for model_key, model_runs in grouped.items():
        ordered_runs = tuple(sorted(model_runs, key=_run_sort_key))
        points = _competence_points(ordered_runs)
        score = competent_complexity_score(
            points,
            chance_mass=chance_mass,
        )
        best_run = max(
            ordered_runs,
            key=lambda run: (run.score, -_cost_value(run.cost_summary, "parameter_count")),
        )
        best_runs[model_key] = best_run
        result_status = (
            "accepted"
            if any(run.result_status == "accepted" for run in ordered_runs)
            else "tentative"
        )
        score_basis = {
            "kind": "competence-integral-over-complexity-v1",
            "score_unit": "bits",
            "complexity_axis": "log2-distinguishable-states",
            "reference_baseline_complexity": reference_baseline_complexity,
            "chance_mass": chance_mass,
            "point_score": "accepted-mass",
            "local_competence": "above-chance-accepted-mass",
            "integration": "trapezoid-over-observed-complexity",
        }
        record: dict[str, object] = {
            "model_key": model_key,
            "result_status": result_status,
            "architecture_digest": str(best_run.architecture_digest),
            "benchmark_id": str(best_run.benchmark_id),
            "score": score,
            "score_basis": score_basis,
            "score_views": {
                "absolute": {
                    "key": "absolute",
                    "label": "Absolute",
                    "score": score,
                    "basis": score_basis,
                }
            },
            "observed_complexities": [point["complexity"] for point in points],
            "points": list(points),
            "cost_summary": _model_cost_summary(ordered_runs, best_run=best_run),
            "run_ids": [run.run_id for run in ordered_runs],
            "measurement_count": sum(run.measurement_count for run in ordered_runs),
            "source_kinds": sorted({run.source_kind for run in ordered_runs}),
        }
        record["console_view_model"] = _model_console_view_model(
            manifest=manifest,
            model=record,
            runs=ordered_runs,
            inspection=best_run.model_inspection,
        )
        records.append(record)
    _add_relative_score_views(records, best_runs=best_runs, competitions=competitions)
    return tuple(sorted(records, key=_model_sort_key))


def _add_relative_score_views(
    records: list[dict[str, object]],
    *,
    best_runs: Mapping[str, _BenchmarkRunRecord],
    competitions: tuple[Mapping[str, object], ...],
) -> None:
    if len(records) < 2:
        return
    competition_outcomes = _pairwise_competition_outcomes(best_runs, competitions)
    if not competition_outcomes:
        return
    relative_scores = _relative_model_scores(best_runs, outcomes=competition_outcomes)
    for record in records:
        model_key = str(record["model_key"])
        score_views = cast(dict[str, object], record["score_views"])
        score_views["relative"] = {
            "key": "relative",
            "label": "Relative",
            "score": relative_scores.get(model_key, _relative_score_baseline),
            "basis": {
                "kind": "model-competition-elo-like-v1",
                "score_unit": "rating-points",
                "baseline": _relative_score_baseline,
                "rating_update": "repeated-fractional-elo",
                "competition_mechanic": "paired-prediction-accepted-mass",
                "pair_outcome": "higher-score-wins",
                "tie_value": 0.5,
            },
        }


_relative_score_baseline = 1000.0
_relative_score_scale = 400.0
_relative_score_k_factor = 32.0
_relative_score_update_epochs = 64


def _relative_model_scores(
    best_runs: Mapping[str, _BenchmarkRunRecord],
    *,
    outcomes: tuple[_ModelCompetitionOutcome, ...],
) -> dict[str, float]:
    ratings = dict.fromkeys(best_runs, _relative_score_baseline)
    ordered_outcomes = tuple(
        sorted(
            outcomes,
            key=lambda outcome: (outcome.left_model_key, outcome.right_model_key),
        )
    )
    for _epoch in range(_relative_score_update_epochs):
        for outcome in ordered_outcomes:
            left_actual = _normalized_pair_score(outcome.left_score, outcome.right_score)
            left_expected = _elo_expected_score(
                ratings[outcome.left_model_key],
                ratings[outcome.right_model_key],
            )
            delta = _relative_score_k_factor * (left_actual - left_expected)
            ratings[outcome.left_model_key] += delta
            ratings[outcome.right_model_key] -= delta
    return {model_key: ratings[model_key] for model_key in sorted(best_runs)}


def _pairwise_competition_outcomes(
    best_runs: Mapping[str, _BenchmarkRunRecord],
    competitions: tuple[Mapping[str, object], ...],
) -> tuple[_ModelCompetitionOutcome, ...]:
    known_models = set(best_runs)
    outcomes: list[_ModelCompetitionOutcome] = []
    for competition in competitions:
        left_model_key = _extract.non_empty_string(
            competition.get("left_model_key"),
            "competition.left_model_key",
        )
        right_model_key = _extract.non_empty_string(
            competition.get("right_model_key"),
            "competition.right_model_key",
        )
        if left_model_key not in known_models or right_model_key not in known_models:
            continue
        outcomes.append(
            _ModelCompetitionOutcome(
                left_model_key=left_model_key,
                right_model_key=right_model_key,
                left_score=_as_probability(
                    competition.get("left_score"),
                    "competition.left_score",
                ),
                right_score=_as_probability(
                    competition.get("right_score"),
                    "competition.right_score",
                ),
                sample_count=_as_positive_int(
                    competition.get("sample_count"),
                    "competition.sample_count",
                ),
            )
        )
    return tuple(outcomes)


def _normalized_pair_score(left_score: float, right_score: float) -> float:
    total = left_score + right_score
    if total <= 0.0:
        return 0.5
    return left_score / total


def _elo_expected_score(left_rating: float, right_rating: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((right_rating - left_rating) / _relative_score_scale))


def _model_cost_summary(
    runs: tuple[_BenchmarkRunRecord, ...],
    *,
    best_run: _BenchmarkRunRecord,
) -> dict[str, object]:
    cost_summary = _run_cost_summary(best_run)
    training_values = tuple(_run_training_compute_value(run) for run in runs)
    if any(value is None for value in training_values):
        raise LocalResultImportError(
            f"model {best_run.model_key} is missing reconstructible training compute"
        )
    cost_summary["training_compute"] = sum(cast(float, value) for value in training_values)
    return cost_summary


def _run_training_compute_value(run: _BenchmarkRunRecord) -> float | None:
    training_compute = _run_training_compute(run)
    if training_compute is not None:
        return training_compute
    return _optional_cost_value(run.cost_summary, "training_compute")


def _run_cost_summary(run: _BenchmarkRunRecord) -> dict[str, object]:
    cost_summary = dict(run.cost_summary)
    inference_compute = _optional_cost_value(cost_summary, "inference_compute")
    if inference_compute is not None:
        cost_summary["inference_compute"] = inference_compute
    training_compute = _run_training_compute(run)
    if training_compute is not None:
        cost_summary["training_compute"] = training_compute
    return cost_summary


def _run_training_compute(run: _BenchmarkRunRecord) -> float | None:
    if run.training_summary is None:
        return None
    training_run = TrainingRunRecord.from_record(
        _extract.mapping(run.training_summary.get("training_run"), "training_run")
    )
    return training_run.training_compute


def _model_console_view_model(
    *,
    manifest: BenchmarkManifest,
    model: Mapping[str, object],
    runs: tuple[_BenchmarkRunRecord, ...],
    inspection: Mapping[str, object],
) -> Mapping[str, object]:
    architecture_summary = _extract.mapping(
        inspection.get("architecture_summary"),
        "model_inspection.architecture_summary",
    )
    cost_summary = _extract.mapping(model.get("cost_summary"), "model.cost_summary")
    node_evidence = _as_sequence(
        inspection.get("node_evidence", ()),
        "model_inspection.node_evidence",
    )
    node_claim_kinds = sorted(
        {
            claim
            for evidence in node_evidence
            for claim in _as_sequence(
                _extract.mapping(evidence, "model_inspection.node_evidence").get("claim_kinds"),
                "model_inspection.node_evidence.claim_kinds",
            )
            if isinstance(claim, str) and claim
        }
    )
    source_kinds = _as_sequence(model.get("source_kinds"), "model.source_kinds")
    sections = [
        _console_detail_entries_section(
            title="Model Contract",
            entries=(
                ("Benchmark", str(manifest.id)),
                (
                    "Prediction Space",
                    _prediction_space_label(manifest),
                ),
                (
                    "Observed " + _model_complexity_label(manifest),
                    ", ".join(
                        _console_number_value(value, precision=2)
                        for value in _as_sequence(
                            model.get("observed_complexities"),
                            "model.observed_complexities",
                        )
                    )
                    or "none",
                ),
                ("Score", _console_number_value(model.get("score"), precision=4)),
            ),
        ),
        _console_detail_entries_section(
            title="Architecture Graph",
            entries=(
                ("Components", _console_number_value(architecture_summary.get("component_count"))),
                ("Edges", _console_number_value(architecture_summary.get("edge_count"))),
                ("Inputs", _node_list_label(architecture_summary.get("input_node_ids"))),
                ("Outputs", _node_list_label(architecture_summary.get("output_node_ids"))),
                (
                    "Component Kinds",
                    ", ".join(
                        str(kind)
                        for kind in _as_sequence(
                            architecture_summary.get("component_kinds"),
                            "architecture_summary.component_kinds",
                        )
                    )
                    or "unknown",
                ),
            ),
        ),
        _console_detail_entries_section(
            title="Evidence",
            entries=(
                ("Node Evidence", _console_number_value(len(node_evidence))),
                ("Claim Kinds", ", ".join(node_claim_kinds) or "none"),
                ("Runs", _console_number_value(len(runs))),
                ("Measurements", _console_number_value(model.get("measurement_count"))),
                (
                    "Sources",
                    ", ".join(str(kind) for kind in source_kinds if isinstance(kind, str))
                    or "unknown",
                ),
            ),
        ),
        _console_detail_entries_section(
            title="Resources",
            entries=(
                ("Parameters", _console_number_value(cost_summary.get("parameter_count"))),
                ("Storage", _console_number_value(cost_summary.get("storage_bytes"))),
                (
                    "Inference Compute",
                    _console_number_value(cost_summary.get("inference_compute")),
                ),
                (
                    "Training Compute",
                    _console_number_value(cost_summary.get("training_compute")),
                ),
            ),
        ),
    ]
    return {"detail_sections": sections}


def _prediction_space_label(manifest: BenchmarkManifest) -> str:
    return f"finite outcome space with {len(manifest.outcome_space.outcomes)} outcomes"


def _model_complexity_label(manifest: BenchmarkManifest) -> str:
    return "Complexity"


def _node_list_label(value: object) -> str:
    if not isinstance(value, list):
        return "unknown"
    labels = [str(item) for item in cast(list[object], value) if isinstance(item, str) and item]
    return ", ".join(labels) if labels else "none"


def _competence_points(
    runs: tuple[_BenchmarkRunRecord, ...],
) -> tuple[dict[str, object], ...]:
    by_complexity: dict[float, list[tuple[_BenchmarkRunRecord, float, int]]] = {}
    for run in runs:
        for complexity, score, sample_count in _run_competence_points(run):
            by_complexity.setdefault(complexity, []).append((run, score, sample_count))
    points: list[dict[str, object]] = []
    for complexity, evidence in by_complexity.items():
        total_samples = sum(sample_count for _run, _score, sample_count in evidence)
        score = (
            sum(score * sample_count for _run, score, sample_count in evidence)
            / total_samples
        )
        points.append(
            {
                "complexity": complexity,
                "score": score,
                "sample_count": total_samples,
                "run_ids": [
                    run.run_id
                    for run in sorted(
                        {
                            run.run_id: run
                            for run, _score, _sample_count in evidence
                        }.values(),
                        key=_run_sort_key,
                    )
                ],
            }
        )
    return tuple(sorted(points, key=_point_complexity))


def _run_competence_points(run: _BenchmarkRunRecord) -> tuple[tuple[float, float, int], ...]:
    if run.sampled_competence is not None:
        points = run.sampled_competence.get("points")
        if isinstance(points, list | tuple):
            return tuple(
                (
                    _as_nonnegative_number(
                        point.get("complexity"),
                        "sampled_competence.point.complexity",
                    ),
                    _as_nonnegative_number(
                        point.get("mean_accepted_mass"),
                        "sampled_competence.point.mean_accepted_mass",
                    ),
                    _as_positive_int(
                        point.get("sample_count"),
                        "sampled_competence.point.sample_count",
                    ),
                )
                for point in (
                    _extract.mapping(value, "sampled_competence.points")
                    for value in _as_sequence(
                        cast(object, points),
                        "sampled_competence.points",
                    )
                )
            )
        return (
            (
                _as_nonnegative_number(
                    run.sampled_competence.get("complexity"),
                    "sampled_competence.complexity",
                ),
                _as_nonnegative_number(
                    run.sampled_competence.get("mean_accepted_mass"),
                    "sampled_competence.mean_accepted_mass",
                ),
                _as_positive_int(
                    run.sampled_competence.get("sample_count"),
                    "sampled_competence.sample_count",
                ),
            ),
        )
    if run.complexity is None:
        return ()
    return ((run.complexity, run.score, run.measurement_count),)


def competent_complexity_score(
    points: tuple[dict[str, object], ...],
    *,
    chance_mass: float,
) -> float:
    return sampled_competence_frontier_score(
        tuple(
            CompetencePoint(
                complexity=_point_complexity(point),
                accepted_mass=_point_score(point),
            )
            for point in points
        ),
        chance_mass=chance_mass,
    )


def _benchmark_base_complexity(
    *,
    manifest: BenchmarkManifest,
    repository_root: Path,
) -> float:
    generator = load_observation_generator(
        repository_root / "src" / "leibniz" / "benchmarks" / _identifier_atom(manifest.id)
    )
    resolution = generator.minimum_discriminatable_resolution_assignment(
        component_count=1,
        minimum_assignment=generator.materialization.minimum_resolution(),
    )
    width = resolution.require_axis(generator.formation.width_axis)
    height = resolution.require_axis(generator.formation.height_axis)
    return generator.distinguishable_state_complexity(
        component_count=1,
        width=width,
        height=height,
        variation_extent=0.0,
    )


def _chance_mass(manifest: BenchmarkManifest) -> float:
    outcome_count = len(manifest.outcome_space.outcomes)
    if outcome_count < 1:
        return 0.0
    return 1.0 / outcome_count


def _frontier_records(
    models: tuple[dict[str, object], ...],
    *,
    cost_axis: str,
) -> list[dict[str, object]]:
    ordered = sorted(
        models,
        key=lambda model: (
            _cost_value(_extract.mapping(model["cost_summary"], "cost_summary"), cost_axis),
            -_as_nonnegative_number(model["score"], "score"),
            str(model["model_key"]),
        ),
    )
    frontier: list[dict[str, object]] = []
    best_score = -math.inf
    for model in ordered:
        score = _as_nonnegative_number(model["score"], "score")
        if score > best_score:
            frontier.append(model)
            best_score = score
    return frontier


def _validate_known_benchmarks(
    measurements: tuple[MeasurementRecord, ...],
    known_benchmark_ids: frozenset[ProtocolIdentifier],
) -> None:
    for measurement in measurements:
        if measurement.benchmark_id not in known_benchmark_ids:
            raise LocalResultImportError(
                f"unknown benchmark id in imported measurement: {measurement.benchmark_id}"
            )


def _local_artifact_path(*, results_root: Path, value: object, field: str) -> Path:
    path = Path(_extract.non_empty_string(value, field))
    resolved = path.resolve() if path.is_absolute() else (results_root.parent / path).resolve()
    if not resolved.is_relative_to(results_root.resolve()):
        raise LocalResultImportError(f"{field} must stay inside results root")
    if not resolved.is_file():
        raise LocalResultImportError(f"{field} does not exist: {path}")
    return resolved


def _mean_accepted_mass(dataset: MeasurementDataset) -> float:
    if not dataset.measurements:
        return 0.0
    return sum(
        measurement.raw_scoring_evidence.accepted_mass
        for measurement in dataset.measurements
    ) / len(dataset.measurements)


def _record_digest(record: Mapping[str, object]) -> ContentDigest:
    digest = record.get("record_digest")
    if digest is None:
        return ContentDigest.from_value(record)
    return _as_digest(digest, field="record_digest")


def _cost_value(cost_summary: Mapping[str, object], cost_axis: str) -> float:
    value = cost_summary.get(cost_axis)
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise LocalResultImportError(f"cost summary missing numeric {cost_axis}")
    return float(value)


def _optional_cost_value(cost_summary: Mapping[str, object], cost_axis: str) -> float | None:
    value = cost_summary.get(cost_axis)
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise LocalResultImportError(f"cost summary has nonnumeric {cost_axis}")
    return float(value)


def _point_complexity(point: Mapping[str, object]) -> float:
    return _as_nonnegative_number(point["complexity"], "complexity")


def _point_score(point: Mapping[str, object]) -> float:
    return _as_nonnegative_number(point["score"], "score")


def _model_sort_key(record: Mapping[str, object]) -> tuple[float, float, str]:
    cost_summary = _extract.mapping(record["cost_summary"], "cost_summary")
    return (
        -_as_nonnegative_number(record["score"], "score"),
        _cost_value(cost_summary, "parameter_count"),
        str(record["model_key"]),
    )


def _run_sort_key(run: _BenchmarkRunRecord) -> tuple[str, str, str]:
    return (str(run.benchmark_id), run.run_id, run.source_path.as_posix())


def _identifier_atom(identifier: ProtocolIdentifier) -> str:
    return str(identifier.name).rsplit(".", maxsplit=1)[-1]


def _resolve_output_root(repository_root: Path, results_root: Path) -> Path:
    if results_root.is_absolute():
        resolved = results_root.resolve()
    else:
        resolved = (repository_root / results_root).resolve()
    if resolved == repository_root:
        raise LocalResultImportError("results root must not be the repository root")
    return resolved


def _commit_runs_checkout(
    *,
    results_root: Path,
    message: str,
    push: bool,
    repo_id: str | None,
    remote: str,
    token: str | None,
) -> tuple[str | None, str | None, str | None]:
    if not message.strip():
        raise LocalResultImportError("commit message must not be empty")
    if not push and not _is_git_checkout(results_root):
        return None, None, None
    selected_remote = _select_publication_remote(
        repo_id=repo_id,
        remote=remote,
        results_root=results_root,
        token=token,
    )
    commit = _commit_checkout_if_dirty(
        results_root=results_root,
        message=message,
        push=push,
        repo_id=repo_id,
        remote=remote,
        token=token,
    )
    if commit is None and selected_remote == "git":
        raise LocalResultImportError("no dirty run-state changes to commit")
    remote_commit = None
    if push and selected_remote == "hf":
        if repo_id is None:
            raise LocalResultImportError("Hugging Face API push requires --repo")
        remote_commit = _push_hf_api(
            results_root=results_root,
            repo_id=repo_id,
            message=message,
            token=token,
        )
    return commit, selected_remote, remote_commit


def _commit_checkout_if_dirty(
    *,
    results_root: Path,
    message: str,
    push: bool,
    repo_id: str | None,
    remote: str,
    token: str | None,
) -> str | None:
    if not message.strip():
        raise LocalResultImportError("commit message must not be empty")
    if not push and not _is_git_checkout(results_root):
        return None
    selected_remote = _select_publication_remote(
        repo_id=repo_id,
        remote=remote,
        results_root=results_root,
        token=token,
    )
    if selected_remote == "hf" and not _is_git_checkout(results_root):
        return None
    if not _is_git_checkout(results_root):
        if push:
            raise LocalResultImportError("Git push requires results root to be a Git checkout")
        return None
    _git(results_root, "add", "-A")
    status = _git(results_root, "status", "--porcelain").stdout.strip()
    if not status:
        return None
    _ensure_git_identity(results_root)
    _git(results_root, "commit", "-m", message)
    commit = _git(results_root, "rev-parse", "HEAD").stdout.strip()
    if push and selected_remote == "git":
        _push_checkout(results_root=results_root)
    return commit


def _push_checkout(*, results_root: Path) -> str:
    commit = _git(results_root, "rev-parse", "HEAD").stdout.strip()
    _git(results_root, "push", "-u", "origin", "HEAD")
    return commit


def _is_git_checkout(path: Path) -> bool:
    try:
        result = _git(path, "rev-parse", "--is-inside-work-tree")
    except LocalResultImportError:
        return False
    return result.stdout.strip() == "true"


def _ensure_git_identity(results_root: Path) -> None:
    if not _git_config_value(results_root, "user.email"):
        _git(results_root, "config", "user.email", "leibniz@example.invalid")
    if not _git_config_value(results_root, "user.name"):
        _git(results_root, "config", "user.name", "Leibniz Operator")


def _git_config_value(results_root: Path, key: str) -> str | None:
    try:
        return _git(results_root, "config", "--get", key).stdout.strip() or None
    except LocalResultImportError:
        return None


def _git(
    path: Path,
    *args: str,
) -> subprocess.CompletedProcess[str]:
    try:
        return _run_git_process(["git", "-C", str(path), *args])
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        command = " ".join(("git", *args))
        message = f"{command} failed"
        if detail:
            message = f"{message}: {detail}"
        raise LocalResultImportError(message) from error


def _git_clone(
    *,
    source: str,
    target: Path,
) -> None:
    try:
        _run_git_process(["git", "clone", source, str(target)])
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        message = "git clone failed"
        if detail:
            message = f"{message}: {detail}"
        raise LocalResultImportError(message) from error


def _run_git_process(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, capture_output=True, text=True)


def _select_publication_remote(
    *,
    repo_id: str | None,
    remote: str,
    results_root: Path,
    token: str | None,
) -> str:
    if remote not in {"auto", "hf", "git"}:
        raise LocalResultImportError("remote must be auto, hf, or git")
    if remote == "hf":
        if repo_id is None:
            raise LocalResultImportError("Hugging Face API remote requires --repo")
        _require_hf_api_token(token)
        return "hf"
    if remote == "git":
        return "git"
    if repo_id is not None and _hf_api_token(token) is not None and _hf_api_module() is not None:
        return "hf"
    if _is_git_checkout(results_root):
        return "git"
    if repo_id is not None:
        return "git"
    raise LocalResultImportError(
        "no publication remote is available: provide --repo with Hugging Face auth "
        "or use a Git checkout for results"
    )


def _push_hf_api(
    *,
    results_root: Path,
    repo_id: str,
    message: str,
    token: str | None,
) -> str:
    api_module = _require_hf_api_module()
    resolved_token = _require_hf_api_token(token)
    operations = [
        api_module.CommitOperationAdd(
            path_in_repo=path.relative_to(results_root).as_posix(),
            path_or_fileobj=path.as_posix(),
        )
        for path in _publication_upload_files(results_root)
    ]
    if not operations:
        raise LocalResultImportError("no result files found to publish")
    api = api_module.HfApi()
    info = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=message,
        token=resolved_token,
    )
    commit_id = getattr(info, "commit_id", None)
    if isinstance(commit_id, str) and commit_id:
        return commit_id
    oid = getattr(info, "oid", None)
    return str(oid) if oid is not None else ""


def _create_hf_dataset_repo(*, repo_id: str, token: str | None) -> bool:
    api_module = _require_hf_api_module()
    api = api_module.HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        exist_ok=True,
        token=_require_hf_api_token(token),
    )
    return True


def _publication_upload_files(results_root: Path) -> tuple[Path, ...]:
    return tuple(
        sorted(
            path
            for path in results_root.rglob("*")
            if path.is_file() and ".git" not in path.relative_to(results_root).parts
        )
    )


def _hf_api_module() -> Any | None:
    try:
        return importlib.import_module("huggingface_hub")
    except ModuleNotFoundError:
        return None


def _require_hf_api_module() -> Any:
    module = _hf_api_module()
    if module is None:
        raise LocalResultImportError(
            "Hugging Face API remote requires the huggingface_hub package"
        )
    return module


def _hf_api_token(token: str | None) -> str | None:
    value = token or os.environ.get("HF_TOKEN")
    if value is not None and value.strip():
        return value.strip()
    module = _hf_api_module()
    if module is None or not hasattr(module, "get_token"):
        return None
    found = module.get_token()
    return found.strip() if isinstance(found, str) and found.strip() else None


def _require_hf_api_token(token: str | None) -> str:
    value = _hf_api_token(token)
    if value is None:
        raise LocalResultImportError(
            "Hugging Face API remote requires HF_TOKEN or a token from hf auth login"
        )
    return value


def _hf_dataset_url(repo_id: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}"


def _hf_dataset_ssh_url(repo_id: str) -> str:
    return f"git@hf.co:datasets/{repo_id}.git"


def _ensure_publication_checkout_structure(results_root: Path) -> None:
    readme = results_root / "README.md"
    if not readme.exists():
        readme.write_text(
            "---\n"
            "license: cc0-1.0\n"
            "---\n\n"
            "# Leibniz Result Publication Checkout\n\n"
            "This dataset repository stores Leibniz benchmark run state and "
            "publication bundle documents.\n",
            encoding="utf-8",
        )
    for directory in _publication_directories:
        marker = results_root / directory / ".gitkeep"
        marker.parent.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text("", encoding="utf-8")


def _validate_hf_repo_id(repo_id: str) -> str:
    value = repo_id.strip()
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise LocalResultImportError("Hugging Face repo id must have owner/name form")
    if any(part in {".", ".."} or any(char.isspace() for char in part) for part in parts):
        raise LocalResultImportError("Hugging Face repo id contains invalid path atoms")
    return value


def _bundle_filename(bundle_id: ProtocolIdentifier, digest: ContentDigest) -> str:
    name = str(bundle_id.name).replace(".", "_")
    version = str(bundle_id.version).replace(".", "_")
    return f"{name}_{version}_{str(digest)[:12]}{_document_suffix}"


def _validate_publication_bundle_view(record: Mapping[str, object]) -> None:
    _require_string_fields(
        record,
        "",
        ("id", "digest", "source_path", "submission_package_id"),
    )
    benchmark_ids = _as_sequence(record.get("benchmark_ids"), "benchmark_ids")
    if not all(isinstance(item, str) and item for item in benchmark_ids):
        raise LocalResultImportError("benchmark_ids must contain nonempty strings")
    measurement_count = record.get("measurement_count")
    if not isinstance(measurement_count, int) or isinstance(measurement_count, bool):
        raise LocalResultImportError("measurement_count must be an integer")
    if measurement_count < 0:
        raise LocalResultImportError("measurement_count must be nonnegative")
    _require_mapping_fields(record, "", ("measurement_dataset", "measurement_score_view"))


def _validate_benchmark_result_view(record: Mapping[str, object]) -> None:
    if record.get("format_version") != _console_result_view_format_version:
        raise LocalResultImportError("console result view has unsupported format_version")
    results = _as_sequence(record.get("benchmark_results"), "benchmark_results")
    for index, result in enumerate(results):
        _validate_benchmark_result(_extract.mapping(result, f"benchmark_results.{index}"))


def _validate_benchmark_result(record: Mapping[str, object]) -> None:
    _reject_unknown_fields(
        record,
        {
            "benchmark_id",
            "complexity_axis",
            "cost_axes",
            "score_axes",
            "leaderboard",
            "model_candidates",
            "frontiers",
            "training_history",
            "plot_runs",
            "model_inspections",
        },
        prefix="benchmark_results",
    )
    _extract.non_empty_string(record.get("benchmark_id"), "benchmark_id")
    cost_axes = _as_sequence(record.get("cost_axes"), "cost_axes")
    if not cost_axes:
        raise LocalResultImportError("cost_axes must not be empty")
    for index, axis in enumerate(cost_axes):
        _require_string_fields(
            _extract.mapping(axis, f"cost_axes.{index}"),
            f"cost_axes.{index}",
            ("key", "label"),
        )
    for index, axis in enumerate(_as_sequence(record.get("score_axes", ()), "score_axes")):
        _require_string_fields(
            _extract.mapping(axis, f"score_axes.{index}"),
            f"score_axes.{index}",
            ("key", "label"),
        )
    for index, model in enumerate(_record_sequence(record, "leaderboard")):
        _validate_model_result(model, f"leaderboard.{index}")
        if model.get("result_status") != "accepted":
            raise LocalResultImportError("leaderboard entries must be accepted")
    for index, model in enumerate(_record_sequence(record, "model_candidates")):
        _validate_model_result(model, f"model_candidates.{index}")
    frontiers = _extract.mapping(record.get("frontiers"), "frontiers")
    for axis in _benchmark_cost_axis_keys:
        for index, model in enumerate(_as_sequence(frontiers.get(axis), f"frontiers.{axis}")):
            field = f"frontiers.{axis}.{index}"
            _validate_model_result(_extract.mapping(model, field), field)
    for index, run in enumerate(_record_sequence(record, "training_history")):
        _validate_run_result(run, f"training_history.{index}")
    for index, run in enumerate(_record_sequence(record, "plot_runs")):
        _validate_run_result(run, f"plot_runs.{index}")
    inspections = _as_sequence(record.get("model_inspections", ()), "model_inspections")
    for index, inspection in enumerate(inspections):
        field = f"model_inspections.{index}"
        inspection_record = dict(_extract.mapping(inspection, field))
        inspection_record.pop("source_path", None)
        try:
            ModelInspectionRecord.from_record(inspection_record)
        except ModelInspectionValidationError as error:
            raise LocalResultImportError(f"{field}: invalid model inspection: {error}") from error


def _validate_model_result(record: Mapping[str, object], prefix: str) -> None:
    _require_string_fields(
        record,
        prefix,
        ("model_key", "result_status", "architecture_digest", "benchmark_id"),
    )
    if record.get("result_status") not in {"accepted", "tentative"}:
        raise LocalResultImportError(f"{_field_path(prefix, 'result_status')} is invalid")
    _as_nonnegative_number(record.get("score"), _field_path(prefix, "score"))
    _require_sequence_fields(
        record,
        prefix,
        ("observed_complexities", "points", "run_ids", "source_kinds"),
    )
    _require_mapping_fields(record, prefix, ("cost_summary",))
    _as_nonnegative_number(
        record.get("measurement_count"),
        _field_path(prefix, "measurement_count"),
    )
    if "console_view_model" in record:
        _validate_console_detail_view_model(
            _extract.mapping(
                record["console_view_model"],
                _field_path(prefix, "console_view_model"),
            ),
            _field_path(prefix, "console_view_model"),
        )
    if "score_views" in record:
        for key, view in _extract.mapping(
            record["score_views"],
            _field_path(prefix, "score_views"),
        ).items():
            view_path = _field_path(prefix, f"score_views.{key}")
            view_record = _extract.mapping(view, view_path)
            _require_string_fields(view_record, view_path, ("key", "label"))
            _as_nonnegative_number(view_record.get("score"), _field_path(view_path, "score"))
            if "basis" in view_record:
                _extract.mapping(view_record["basis"], _field_path(view_path, "basis"))


def _validate_competition_record(record: Mapping[str, object], prefix: str) -> None:
    if record.get("format") != "leibniz.model-competition":
        raise LocalResultImportError(f"{prefix}: unsupported format")
    if record.get("format_version") != 1:
        raise LocalResultImportError(f"{prefix}: unsupported format_version")
    _require_string_fields(
        record,
        prefix,
        (
            "benchmark_id",
            "competition_id",
            "mechanic",
            "outcome_space_id",
            "left_model_key",
            "right_model_key",
        ),
    )
    _as_positive_int(record.get("sample_count"), _field_path(prefix, "sample_count"))
    _as_probability(record.get("left_score"), _field_path(prefix, "left_score"))
    _as_probability(record.get("right_score"), _field_path(prefix, "right_score"))
    entries = _as_sequence(record.get("entries"), _field_path(prefix, "entries"))
    for index, entry in enumerate(entries):
        entry_path = _field_path(prefix, f"entries.{index}")
        entry_record = _extract.mapping(entry, entry_path)
        _require_string_fields(
            entry_record,
            entry_path,
            ("id", "observation_id", "accepted_outcome_id", "winner"),
        )
        _as_probability(entry_record.get("left_score"), _field_path(entry_path, "left_score"))
        _as_probability(entry_record.get("right_score"), _field_path(entry_path, "right_score"))
        if entry_record.get("winner") not in {"left", "right", "tie"}:
            raise LocalResultImportError(f"{_field_path(entry_path, 'winner')} is invalid")


def _validate_run_result(record: Mapping[str, object], prefix: str) -> None:
    _require_string_fields(
        record,
        prefix,
        (
            "source_kind",
            "result_status",
            "source_path",
            "run_id",
            "run_slug",
            "benchmark_id",
            "architecture_digest",
            "model_key",
            "measurement_dataset_digest",
        ),
    )
    if record.get("result_status") not in {"accepted", "tentative"}:
        raise LocalResultImportError(f"{_field_path(prefix, 'result_status')} is invalid")
    _as_nonnegative_number(
        record.get("measurement_count"),
        _field_path(prefix, "measurement_count"),
    )
    _as_nonnegative_number(record.get("score"), _field_path(prefix, "score"))
    _require_mapping_fields(record, prefix, ("cost_summary", "architecture"))
    for field in ("model_inspection_digest", "model_inspection_path"):
        if field in record:
            _extract.non_empty_string(record[field], _field_path(prefix, field))
    if "sampled_competence" in record:
        _extract.mapping(record["sampled_competence"], _field_path(prefix, "sampled_competence"))
    if "training_diagnostics" in record:
        diagnostics_path = _field_path(prefix, "training_diagnostics")
        _validate_training_diagnostics(
            _extract.mapping(record["training_diagnostics"], diagnostics_path),
            diagnostics_path,
        )
    if "console_view_model" in record:
        _validate_console_detail_view_model(
            _extract.mapping(
                record["console_view_model"],
                _field_path(prefix, "console_view_model"),
            ),
            _field_path(prefix, "console_view_model"),
        )


def _validate_console_detail_view_model(record: Mapping[str, object], prefix: str) -> None:
    sections = _as_sequence(
        record.get("detail_sections"),
        _field_path(prefix, "detail_sections"),
    )
    for section_index, section in enumerate(sections):
        section_prefix = f"{prefix}.detail_sections.{section_index}"
        section_record = _extract.mapping(section, section_prefix)
        _extract.non_empty_string(section_record.get("title"), _field_path(section_prefix, "title"))
        if "entries" in section_record:
            entries = _as_sequence(
                section_record["entries"],
                _field_path(section_prefix, "entries"),
            )
            for entry_index, entry in enumerate(entries):
                entry_record = _extract.mapping(entry, f"{section_prefix}.entries.{entry_index}")
                _require_string_fields(
                    entry_record,
                    f"{section_prefix}.entries.{entry_index}",
                    ("label", "value"),
                )
        if "table" in section_record:
            table_prefix = _field_path(section_prefix, "table")
            table = _extract.mapping(section_record["table"], table_prefix)
            _extract.non_empty_string(
                table.get("aria_label"), _field_path(table_prefix, "aria_label")
            )
            columns = _as_sequence(table.get("columns"), _field_path(table_prefix, "columns"))
            rows = _as_sequence(table.get("rows"), _field_path(table_prefix, "rows"))
            if not columns or not all(isinstance(column, str) and column for column in columns):
                raise LocalResultImportError("console_view_model table columns must be strings")
            for row in rows:
                values = _as_sequence(row, _field_path(table_prefix, "rows"))
                cells_are_strings = all(isinstance(value, str) and value for value in values)
                if len(values) != len(columns) or not cells_are_strings:
                    raise LocalResultImportError("console_view_model table rows must match columns")


def _validate_training_diagnostics(record: Mapping[str, object], prefix: str) -> None:
    status = _extract.non_empty_string(record.get("status"), _field_path(prefix, "status"))
    statuses = {"running", "completed", "converged", "budget-exhausted", "not-trainable", "failed"}
    if status not in statuses:
        raise LocalResultImportError(f"unsupported training status: {status}")
    _extract.non_empty_string(record.get("stop_reason"), _field_path(prefix, "stop_reason"))
    numeric_fields = (
        "steps_run",
        "validation_checks",
        "final_validation_loss",
        "final_validation_step",
        "final_validation_check",
    )
    for field in numeric_fields:
        _as_nonnegative_number(record.get(field), _field_path(prefix, field))
    if "training_compute" in record:
        _as_nonnegative_number(
            record.get("training_compute"),
            _field_path(prefix, "training_compute"),
        )
    protocol = _extract.mapping(record.get("protocol"), _field_path(prefix, "protocol"))
    protocol_path = _field_path(prefix, "protocol")
    _require_string_fields(
        protocol,
        protocol_path,
        ("kind", "objective", "optimizer", "schedule", "validation_source"),
    )
    protocol_numbers = ("learning_rate", "min_delta")
    for field in protocol_numbers:
        _as_nonnegative_number(protocol.get(field), f"{prefix}.protocol.{field}")
    protocol_positive_ints = (
        "seed",
        "batch_size",
        "gate_check_interval",
        "gate_sample_count",
    )
    for field in protocol_positive_ints:
        _as_positive_int(protocol.get(field), f"{prefix}.protocol.{field}")
    _as_nonnegative_number(protocol.get("patience"), f"{prefix}.protocol.patience")
    _extract.non_empty_string(
        protocol.get("gate_decision_rule"),
        f"{prefix}.protocol.gate_decision_rule",
    )
    for field in ("validation_history", "artifacts"):
        _as_sequence(record.get(field), _field_path(prefix, field))
    if "evaluation_curriculum" in record:
        curriculum = _extract.mapping(
            record.get("evaluation_curriculum"),
            _field_path(prefix, "evaluation_curriculum"),
        )
        _extract.non_empty_string(
            curriculum.get("kind"),
            _field_path(prefix, "evaluation_curriculum.kind"),
        )
        _as_sequence(
            curriculum.get("rungs"),
            _field_path(prefix, "evaluation_curriculum.rungs"),
        )
    if "training_curriculum" in record:
        curriculum = _extract.mapping(
            record.get("training_curriculum"),
            _field_path(prefix, "training_curriculum"),
        )
        _extract.non_empty_string(
            curriculum.get("kind"),
            _field_path(prefix, "training_curriculum.kind"),
        )
        _as_sequence(
            curriculum.get("rungs"),
            _field_path(prefix, "training_curriculum.rungs"),
        )


def _record_sequence(
    record: Mapping[str, object],
    field: str,
    *,
    default: tuple[object, ...] | None = None,
) -> tuple[Mapping[str, object], ...]:
    value = record.get(field, default) if default is not None else record.get(field)
    return tuple(
        _extract.mapping(item, f"{field}.{index}")
        for index, item in enumerate(_as_sequence(value, field))
    )


def _reject_unknown_fields(
    record: Mapping[str, object],
    allowed: set[str],
    *,
    prefix: str,
) -> None:
    unknown = sorted(str(field) for field in record if field not in allowed)
    if unknown:
        raise LocalResultImportError(f"{prefix}: unknown fields: {', '.join(unknown)}")


def _field_path(prefix: str, field: str) -> str:
    return field if not prefix else f"{prefix}.{field}"


def _require_string_fields(
    record: Mapping[str, object],
    prefix: str,
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        _extract.non_empty_string(record.get(field), _field_path(prefix, field))


def _require_mapping_fields(
    record: Mapping[str, object],
    prefix: str,
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        _extract.mapping(record.get(field), _field_path(prefix, field))


def _require_sequence_fields(
    record: Mapping[str, object],
    prefix: str,
    fields: tuple[str, ...],
) -> None:
    for field in fields:
        _as_sequence(record.get(field), _field_path(prefix, field))


def _as_sequence(value: object, field: str) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return cast(tuple[object, ...], value)
    if isinstance(value, list):
        return tuple(cast(list[object], value))
    raise LocalResultImportError(f"{field}: expected parsed sequence")


def _as_identifier(value: object, field: str) -> ProtocolIdentifier:
    if isinstance(value, ProtocolIdentifier):
        return value
    if isinstance(value, str):
        try:
            return ProtocolIdentifier.parse(value)
        except ValueError as error:
            raise LocalResultImportError(str(error)) from error
    raise LocalResultImportError(f"{field}: expected identifier")


def _as_digest(value: object, *, field: str) -> ContentDigest:
    return ContentDigest.from_string(value, field=field, error_type=LocalResultImportError)


def _as_nonnegative_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LocalResultImportError(f"{field}: expected number")
    numeric = float(value)
    if numeric < 0 or not math.isfinite(numeric):
        raise LocalResultImportError(f"{field}: expected finite nonnegative number")
    return numeric


def _as_probability(value: object, field: str) -> float:
    numeric = _as_nonnegative_number(value, field)
    if numeric > 1.0:
        raise LocalResultImportError(f"{field}: expected probability no greater than 1")
    return numeric


def _as_positive_int(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise LocalResultImportError(f"{field}: expected positive integer")
    return value
