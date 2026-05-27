"""Operator-local result import and console view materialization."""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestDocument
from leibniz.console.protocol import (
    console_protocol_format_versions,
    console_protocol_formats,
)
from leibniz.content import ContentDigest
from leibniz.documents import (
    canonical_document_bytes,
    document_filename_suffix,
    document_media_type,
    load_object_document,
)
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import (
    MeasurementDataset,
    MeasurementDatasetDocument,
    MeasurementRecord,
)
from leibniz.model_inspection import ModelInspectionDocument, ModelInspectionRecord
from leibniz.publications import SubmissionPublicationBundle, SubmissionPublicationDocument
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
_work_queue_view_format = _protocol_formats.work_queue_view
_console_result_view_format_version = _protocol_format_versions.result_view
_document_suffix = document_filename_suffix()
_manifest_filename = "manifest" + _document_suffix
_default_hf_endpoint = "https://huggingface.co"
_publication_directories = (
    "imports/publication_bundles",
    "measurements",
    "models",
    "proposals",
    "publication_bundles",
    "training",
    "views",
    "work-queues",
)


class LocalResultImportError(ValueError):
    """Raised when local result import cannot produce a valid console view."""


@dataclass(frozen=True, slots=True)
class LocalResultImportSummary:
    """Summary of one operator-local result import."""

    source_files: tuple[Path, ...]
    import_files: tuple[Path, ...]
    view_file: Path
    publication_bundle_count: int
    measurement_count: int

    def to_record(self) -> dict[str, object]:
        return {
            "source_files": [path.as_posix() for path in self.source_files],
            "import_files": [path.as_posix() for path in self.import_files],
            "view_file": self.view_file.as_posix(),
            "publication_bundle_count": self.publication_bundle_count,
            "measurement_count": self.measurement_count,
        }


@dataclass(frozen=True, slots=True)
class LocalBenchmarkResultViewSummary:
    """Summary of one operator-local benchmark result projection."""

    source_files: tuple[Path, ...]
    view_file: Path
    benchmark_count: int
    model_count: int
    run_count: int

    def to_record(self) -> dict[str, object]:
        return {
            "source_files": [path.as_posix() for path in self.source_files],
            "view_file": self.view_file.as_posix(),
            "benchmark_count": self.benchmark_count,
            "model_count": self.model_count,
            "run_count": self.run_count,
        }


@dataclass(frozen=True, slots=True)
class LocalPublicationExportSummary:
    """Summary of publication bundles exported from local benchmark results."""

    source_files: tuple[Path, ...]
    publication_files: tuple[Path, ...]
    publication_bundle_count: int
    measurement_count: int
    git_commit: str | None = None
    git_pushed: bool = False

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "source_files": [path.as_posix() for path in self.source_files],
            "publication_files": [path.as_posix() for path in self.publication_files],
            "publication_bundle_count": self.publication_bundle_count,
            "measurement_count": self.measurement_count,
            "git_pushed": self.git_pushed,
        }
        if self.git_commit is not None:
            record["git_commit"] = self.git_commit
        return record


@dataclass(frozen=True, slots=True)
class LocalPublicationCheckoutSummary:
    """Summary of a prepared public result-publication checkout."""

    repo_id: str | None
    runs_root: Path
    repo_url: str | None
    created_or_reused: bool
    scaffold_commit: str | None
    pushed: bool

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "runs_root": self.runs_root.as_posix(),
            "created_or_reused": self.created_or_reused,
            "pushed": self.pushed,
        }
        if self.repo_id is not None:
            record["repo_id"] = self.repo_id
        if self.repo_url is not None:
            record["repo_url"] = self.repo_url
        if self.scaffold_commit is not None:
            record["scaffold_commit"] = self.scaffold_commit
        return record


@dataclass(frozen=True, slots=True)
class LocalPublicationPushSummary:
    """Summary of pushing a result-publication checkout."""

    runs_root: Path
    pushed_commit: str

    def to_record(self) -> dict[str, object]:
        return {
            "runs_root": self.runs_root.as_posix(),
            "pushed_commit": self.pushed_commit,
        }


def initialize_publication_checkout(
    *,
    repo_id: str | None,
    token: str | None,
    repository_root: Path | None = None,
    runs_root: Path = Path(".runs"),
    endpoint: str = _default_hf_endpoint,
    local_only: bool = False,
    push: bool = False,
    commit_message: str = "Initialize Leibniz result publication checkout",
) -> LocalPublicationCheckoutSummary:
    """Create or reuse a public Hugging Face dataset checkout for run results."""

    repository_root = Path.cwd().resolve() if repository_root is None else repository_root.resolve()
    runs_root = _resolve_output_root(repository_root, runs_root)
    endpoint = endpoint.rstrip("/")
    if local_only:
        if push:
            raise LocalResultImportError("--push cannot be used with --local-only")
        repo_id = None
        repo_url = None
        token = None
        created_or_reused = False
        _ensure_local_git_checkout(runs_root)
    else:
        if repo_id is None:
            raise LocalResultImportError("--repo is required unless --local-only is used")
        repo_id = _validate_hf_repo_id(repo_id)
        token = _validate_token(token)
        repo_url = f"{endpoint}/datasets/{repo_id}"
        created_or_reused = _create_hf_dataset_repo(
            repo_id=repo_id,
            token=token,
            endpoint=endpoint,
        )
        if runs_root.exists():
            if not _is_git_checkout(runs_root):
                raise LocalResultImportError("runs root exists but is not a Git checkout")
        else:
            _git_clone(
                source=repo_url,
                target=runs_root,
                token=token,
                endpoint=endpoint,
            )
    _ensure_publication_checkout_structure(runs_root)
    scaffold_commit = _commit_checkout_if_dirty(
        runs_root=runs_root,
        message=commit_message,
        push=push,
        token=token,
        endpoint=endpoint,
    )
    return LocalPublicationCheckoutSummary(
        repo_id=repo_id,
        runs_root=runs_root,
        repo_url=repo_url,
        created_or_reused=created_or_reused,
        scaffold_commit=scaffold_commit,
        pushed=push and scaffold_commit is not None,
    )


def push_publication_checkout(
    *,
    repository_root: Path | None = None,
    runs_root: Path = Path(".runs"),
    token: str | None = None,
    endpoint: str = _default_hf_endpoint,
) -> LocalPublicationPushSummary:
    """Push an existing result-publication checkout without creating a commit."""

    repository_root = Path.cwd().resolve() if repository_root is None else repository_root.resolve()
    runs_root = _resolve_output_root(repository_root, runs_root)
    if not _is_git_checkout(runs_root):
        raise LocalResultImportError("runs root must be a Git checkout when pushing")
    pushed_commit = _push_checkout(runs_root=runs_root, token=token, endpoint=endpoint)
    return LocalPublicationPushSummary(
        runs_root=runs_root,
        pushed_commit=pushed_commit,
    )


def publish_local_benchmark_results(
    *,
    repository_root: Path | None = None,
    runs_root: Path = Path(".runs"),
    commit: bool = True,
    push: bool = False,
    token: str | None = None,
    commit_message: str = "Publish Leibniz benchmark results",
) -> LocalPublicationExportSummary:
    """Write local benchmark runs as publication bundles for explicit import.

    The resolved runs root is treated as the publication checkout. Publication
    bundles are generated under that checkout, and ``commit`` commits all dirty
    run-state changes there.
    """

    repository_root = Path.cwd().resolve() if repository_root is None else repository_root.resolve()
    runs_root = _resolve_output_root(repository_root, runs_root)
    output_root = runs_root / "publication_bundles"
    if push and not commit:
        raise LocalResultImportError("push requires committing the result checkout")
    manifests = _known_benchmark_manifests(repository_root)
    runs = _local_run_records(runs_root)
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
    materialize_benchmark_result_views(repository_root=repository_root, runs_root=runs_root)
    git_commit = _commit_runs_checkout(
        runs_root=runs_root,
        message=commit_message,
        push=push,
        token=token,
        endpoint=_default_hf_endpoint,
    ) if commit else None
    return LocalPublicationExportSummary(
        source_files=tuple(run.source_path for run in runs),
        publication_files=tuple(publication_files),
        publication_bundle_count=len(publication_files),
        measurement_count=measurement_count,
        git_commit=git_commit,
        git_pushed=push,
    )


def import_submission_publications(
    source_roots: Iterable[Path],
    *,
    repository_root: Path | None = None,
    runs_root: Path = Path(".runs"),
) -> LocalResultImportSummary:
    """Import local publication bundles into a result checkout and console views."""

    repository_root = Path.cwd().resolve() if repository_root is None else repository_root.resolve()
    runs_root = _resolve_output_root(repository_root, runs_root)
    known_benchmark_ids = _known_benchmark_ids(repository_root)
    documents = tuple(_publication_documents(source_roots))
    if not documents:
        raise LocalResultImportError("no publication bundle documents found")

    measurement_records: dict[ProtocolIdentifier, Mapping[str, object]] = {}
    publication_views: list[Mapping[str, object]] = []
    imported_files: list[Path] = []
    source_files: list[Path] = []

    import_root = runs_root / "imports" / "publication_bundles"
    view_root = runs_root / "views"
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
    runs_root: Path = Path(".runs"),
) -> LocalBenchmarkResultViewSummary:
    """Derive console benchmark result views from ignored run/import state."""

    repository_root = Path.cwd().resolve() if repository_root is None else repository_root.resolve()
    runs_root = _resolve_output_root(repository_root, runs_root)
    manifests = _known_benchmark_manifests(repository_root)
    local_runs = _local_run_records(runs_root)
    progress_runs = _local_progress_run_records(runs_root)
    imported_runs = _imported_run_records(runs_root)
    runs = tuple(sorted((*local_runs, *progress_runs, *imported_runs), key=_run_sort_key))
    if not runs:
        raise LocalResultImportError("no benchmark result records found")

    benchmark_records: list[Mapping[str, object]] = []
    for benchmark_id in sorted({run.benchmark_id for run in runs}, key=str):
        manifest = manifests.get(benchmark_id)
        if manifest is None:
            raise LocalResultImportError(f"unknown benchmark id in local results: {benchmark_id}")
        benchmark_runs = tuple(run for run in runs if run.benchmark_id == benchmark_id)
        benchmark_records.append(
            _benchmark_result_record(
                manifest=manifest,
                runs=benchmark_runs,
                proposals=_proposal_records(runs_root, benchmark_id=benchmark_id),
            )
        )

    view_root = runs_root / "views"
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
    if record.get("format") == _work_queue_view_format:
        _validate_work_queue_view(record)
        return record
    if record.get("format") != _console_result_view_format:
        raise LocalResultImportError("console result view has unsupported format")
    if record.get("format_version") != _console_result_view_format_version:
        raise LocalResultImportError("console result view has unsupported format_version")
    publication_bundles = _as_sequence(record.get("publication_bundles"), "publication_bundles")
    for index, publication_bundle in enumerate(publication_bundles):
        _validate_publication_bundle_view(
            _as_mapping(publication_bundle, f"publication_bundles.{index}")
        )
    return record


@dataclass(frozen=True, slots=True)
class _BenchmarkRunRecord:
    source_kind: str
    source_path: Path
    run_id: str
    run_slug: str
    benchmark_id: ProtocolIdentifier
    architecture_digest: ContentDigest
    model_key: str
    scale: int | None
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
            "source_path": self.source_path.as_posix(),
            "run_id": self.run_id,
            "run_slug": self.run_slug,
            "benchmark_id": str(self.benchmark_id),
            "architecture_digest": str(self.architecture_digest),
            "model_key": self.model_key,
            "measurement_count": self.measurement_count,
            "score": self.score,
            "cost_summary": dict(self.cost_summary),
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
        if self.scale is not None:
            record["scale"] = self.scale
        if self.complexity is not None:
            record["complexity"] = self.complexity
        if self.sampled_competence is not None:
            record["sampled_competence"] = dict(self.sampled_competence)
        if self.training_summary is not None:
            record["training_diagnostics"] = _training_diagnostics_record(run=self)
        return record


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


def _local_run_records(runs_root: Path) -> tuple[_BenchmarkRunRecord, ...]:
    training_root = runs_root / "training"
    if not training_root.is_dir():
        return ()
    records: list[_BenchmarkRunRecord] = []
    for path in sorted(training_root.rglob("*" + _document_suffix)):
        summary = load_object_document(path.read_bytes(), description="training summary")
        if summary.get("format") != "leibniz.benchmark-run":
            continue
        measurement_dataset_path = _local_artifact_path(
            runs_root=runs_root,
            value=summary.get("measurement_dataset_path"),
            field="measurement_dataset_path",
        )
        model_inspection_path = _local_artifact_path(
            runs_root=runs_root,
            value=summary.get("model_inspection_path"),
            field="model_inspection_path",
        )
        dataset = MeasurementDatasetDocument.from_bytes(
            measurement_dataset_path.read_bytes()
        ).dataset
        inspection = ModelInspectionDocument.from_bytes(
            model_inspection_path.read_bytes()
        ).inspection
        sampled_competence = _sampled_competence(summary)
        records.append(
            _BenchmarkRunRecord(
                source_kind="local-run",
                source_path=path.resolve(),
                run_id=_as_string(summary.get("run_slug"), "run_slug"),
                run_slug=_as_string(summary.get("run_slug"), "run_slug"),
                benchmark_id=_as_identifier(summary.get("benchmark_id"), "benchmark_id"),
                architecture_digest=_record_digest(inspection.architecture.to_record()),
                model_key=str(inspection.architecture.record_digest),
                scale=_optional_int(summary.get("scale"), "scale"),
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
                training_summary=summary,
            )
        )
    return tuple(records)


def _local_progress_run_records(runs_root: Path) -> tuple[_BenchmarkRunRecord, ...]:
    progress_root = runs_root / "training-progress"
    if not progress_root.is_dir():
        return ()
    records: list[_BenchmarkRunRecord] = []
    empty_dataset = MeasurementDataset(measurements=())
    for path in sorted(progress_root.rglob("*" + _document_suffix)):
        summary = load_object_document(path.read_bytes(), description="training progress")
        if summary.get("format") != "leibniz.benchmark-training-progress":
            continue
        if summary.get("format_version") != 1:
            raise LocalResultImportError("training progress has unsupported format_version")
        training_run = TrainingRunRecord.from_record(
            _as_mapping(summary.get("training_run"), "training_run")
        )
        if training_run.status != "running":
            raise LocalResultImportError("training progress must have running status")
        inspection = _as_mapping(summary.get("model_inspection"), "model_inspection")
        architecture = _as_mapping(summary.get("architecture"), "architecture")
        cost_summary = _as_mapping(summary.get("cost_summary"), "cost_summary")
        architecture_digest = _as_digest(
            summary.get("architecture_digest"),
            field="architecture_digest",
        )
        model_inspection_digest = _as_digest(
            summary.get("model_inspection_digest"),
            field="model_inspection_digest",
        )
        measurement_dataset_digest = empty_dataset.digest
        records.append(
            _BenchmarkRunRecord(
                source_kind="local-progress",
                source_path=path.resolve(),
                run_id=_as_string(summary.get("run_slug"), "run_slug"),
                run_slug=_as_string(summary.get("run_slug"), "run_slug"),
                benchmark_id=_as_identifier(summary.get("benchmark_id"), "benchmark_id"),
                architecture_digest=architecture_digest,
                model_key=str(architecture_digest),
                scale=_optional_int(summary.get("scale"), "scale"),
                complexity=None,
                measurement_count=0,
                score=_as_nonnegative_number(
                    summary.get("provisional_score"),
                    "provisional_score",
                ),
                cost_summary=cost_summary,
                architecture=architecture,
                model_inspection=_model_inspection_view_record(
                    inspection=inspection,
                    source_path=path.resolve(),
                    measurement_dataset_digest=measurement_dataset_digest,
                    training_summary=summary,
                ),
                model_inspection_digest=model_inspection_digest,
                model_inspection_path=None,
                measurement_dataset=empty_dataset,
                measurement_dataset_digest=measurement_dataset_digest,
                training_summary=summary,
            )
        )
    return tuple(records)


def _imported_run_records(runs_root: Path) -> tuple[_BenchmarkRunRecord, ...]:
    import_root = runs_root / "imports" / "publication_bundles"
    if not import_root.is_dir():
        return ()
    records: list[_BenchmarkRunRecord] = []
    for path in sorted(import_root.rglob("*" + _document_suffix)):
        document = SubmissionPublicationDocument.from_bytes(path.read_bytes())
        bundle = document.bundle
        package = bundle.submission_package
        inspection = _inspection_from_architecture(package.architecture_manifest)
        inspection_record = inspection.to_record()
        records.append(
            _BenchmarkRunRecord(
                source_kind="imported-publication",
                source_path=path.resolve(),
                run_id=str(bundle.id),
                run_slug=str(bundle.id.name),
                benchmark_id=package.benchmark_manifest.id,
                architecture_digest=package.architecture_manifest.digest,
                model_key=str(package.architecture_manifest.digest),
                scale=_dataset_scale(
                    manifest=package.benchmark_manifest,
                    dataset=bundle.measurement_dataset,
                ),
                complexity=None,
                measurement_count=len(bundle.measurement_dataset.measurements),
                score=_mean_accepted_mass(bundle.measurement_dataset),
                cost_summary=inspection.cost_summary.to_record(),
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
            )
        )
    return tuple(records)


def _sampled_competence(summary: Mapping[str, object]) -> Mapping[str, object] | None:
    evidence = summary.get("sampled_competence")
    if evidence is None:
        return None
    return _as_mapping(evidence, "sampled_competence")


def _sampled_competence_complexity(summary: Mapping[str, object]) -> float | None:
    record = _sampled_competence(summary)
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
        _as_mapping(run.training_summary.get("training_run"), "training_run")
    )
    history = training_run.validation_history
    final = history[-1]
    return {
        "status": training_run.status,
        "stop_reason": training_run.stop_reason,
        "steps_run": training_run.steps_run,
        "validation_checks": training_run.validation_checks,
        "best_validation_loss": training_run.best_validation_loss,
        "best_validation_step": training_run.best_validation_step,
        "best_validation_check": training_run.best_validation_check,
        "final_validation_loss": final.validation_loss,
        "final_validation_step": final.step,
        "final_validation_check": final.validation_check,
        "protocol": training_run.protocol.to_record(),
        "validation_history": [point.to_record() for point in history],
        "artifacts": _training_artifact_references(run),
    }


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
    return references


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
        protocol = _as_mapping(diagnostics.get("protocol"), "training_diagnostics.protocol")
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
                    ("Interval", _console_number_value(protocol.get("validation_interval"))),
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
                        "Best Loss",
                        _console_number_value(
                            diagnostics.get("best_validation_loss"),
                            precision=4,
                        ),
                    ),
                    ("Best Step", _console_number_value(diagnostics.get("best_validation_step"))),
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
                        "columns": ["Step", "Loss", "Best", "Stale"],
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


def _console_validation_history_row(point: object) -> list[str]:
    point_record = _as_mapping(point, "validation_history")
    return [
        _console_number_value(point_record.get("step")),
        _console_number_value(point_record.get("validation_loss"), precision=4),
        _console_number_value(point_record.get("best_validation_loss"), precision=4),
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
        artifacts=_submission_artifacts_for_local_run(run),
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
    path = Path(_as_string(run.training_summary.get("architecture_path"), "architecture_path"))
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
    runs: tuple[_BenchmarkRunRecord, ...],
    proposals: tuple[Mapping[str, object], ...] = (),
) -> dict[str, object]:
    models = tuple(_model_result_records(runs))
    record: dict[str, object] = {
        "benchmark_id": str(runs[0].benchmark_id),
        "cost_axes": [
            {"key": "parameter_count", "label": "Parameters"},
            {"key": "inference_flops", "label": "FLOPs"},
            {"key": "parameter_bytes", "label": "Storage"},
        ],
        "leaderboard": list(models),
        "frontiers": {
            axis: _frontier_records(models, cost_axis=axis)
            for axis in ("parameter_count", "inference_flops", "parameter_bytes")
        },
        "training_history": [
            run.to_record(complexity_axis=manifest.complexity_coordinate) for run in runs
        ],
        "model_inspections": _model_inspection_records(runs),
    }
    if proposals:
        record["proposals"] = list(proposals)
    if manifest.complexity_coordinate is not None:
        record["complexity_axis"] = manifest.complexity_coordinate
    if manifest.scale_parameter is not None:
        record["scale_axis"] = manifest.scale_parameter.symbol
    return record


def _model_inspection_records(
    runs: tuple[_BenchmarkRunRecord, ...],
) -> list[Mapping[str, object]]:
    by_digest: dict[str, Mapping[str, object]] = {}
    for run in runs:
        by_digest.setdefault(str(run.model_inspection_digest), run.model_inspection)
    return [by_digest[digest] for digest in sorted(by_digest)]


def _model_result_records(
    runs: tuple[_BenchmarkRunRecord, ...],
) -> tuple[dict[str, object], ...]:
    grouped: dict[str, list[_BenchmarkRunRecord]] = {}
    for run in runs:
        grouped.setdefault(run.model_key, []).append(run)

    records: list[dict[str, object]] = []
    for model_key, model_runs in grouped.items():
        ordered_runs = tuple(sorted(model_runs, key=_run_sort_key))
        points = _competence_points(ordered_runs)
        score = _competence_integral(points)
        best_run = max(
            ordered_runs,
            key=lambda run: (run.score, -_cost_value(run.cost_summary, "parameter_count")),
        )
        records.append(
            {
                "model_key": model_key,
                "architecture_digest": str(best_run.architecture_digest),
                "benchmark_id": str(best_run.benchmark_id),
                "score": score,
                "observed_complexities": [point["complexity"] for point in points],
                "points": list(points),
                "cost_summary": dict(best_run.cost_summary),
                "run_ids": [run.run_id for run in ordered_runs],
                "measurement_count": sum(run.measurement_count for run in ordered_runs),
                "source_kinds": sorted({run.source_kind for run in ordered_runs}),
            }
        )
    return tuple(sorted(records, key=_model_sort_key))


def _proposal_records(
    runs_root: Path,
    *,
    benchmark_id: ProtocolIdentifier,
) -> tuple[Mapping[str, object], ...]:
    proposal_path = (
        runs_root
        / "proposals"
        / _identifier_atom(benchmark_id)
        / ("proposal_set" + _document_suffix)
    )
    if not proposal_path.is_file():
        return ()
    record = load_object_document(proposal_path.read_bytes(), description="proposal set")
    proposals = _as_sequence(record.get("proposals"), "proposals")
    return tuple(_as_mapping(proposal, "proposals") for proposal in proposals)


def _competence_points(
    runs: tuple[_BenchmarkRunRecord, ...],
) -> tuple[dict[str, object], ...]:
    by_complexity: dict[float, list[_BenchmarkRunRecord]] = {}
    for run in runs:
        complexity = run.complexity
        if complexity is None and run.scale is not None:
            complexity = float(run.scale)
        if complexity is None:
            continue
        by_complexity.setdefault(complexity, []).append(run)
    points: list[dict[str, object]] = []
    for complexity, complexity_runs in by_complexity.items():
        score = sum(run.score for run in complexity_runs) / len(complexity_runs)
        points.append(
            {
                "complexity": complexity,
                "score": score,
                "sample_count": sum(run.measurement_count for run in complexity_runs),
                "run_ids": [run.run_id for run in sorted(complexity_runs, key=_run_sort_key)],
            }
        )
    return tuple(sorted(points, key=_point_complexity))


def _competence_integral(points: tuple[dict[str, object], ...]) -> float:
    if not points:
        return 0.0
    if len(points) == 1:
        return _point_score(points[0])
    total_width = _point_complexity(points[-1]) - _point_complexity(points[0])
    if total_width <= 0:
        return _point_score(points[-1])
    area = 0.0
    for left, right in zip(points, points[1:], strict=True):
        width = _point_complexity(right) - _point_complexity(left)
        area += width * (_point_score(left) + _point_score(right)) / 2.0
    return area / total_width


def _frontier_records(
    models: tuple[dict[str, object], ...],
    *,
    cost_axis: str,
) -> list[dict[str, object]]:
    ordered = sorted(
        models,
        key=lambda model: (
            _cost_value(_as_mapping(model["cost_summary"], "cost_summary"), cost_axis),
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


def _local_artifact_path(*, runs_root: Path, value: object, field: str) -> Path:
    path = Path(_as_string(value, field))
    resolved = path.resolve() if path.is_absolute() else (runs_root.parent / path).resolve()
    if not resolved.is_relative_to(runs_root.resolve()):
        raise LocalResultImportError(f"{field} must stay inside runs root")
    if not resolved.is_file():
        raise LocalResultImportError(f"{field} does not exist: {path}")
    return resolved


def _dataset_scale(*, manifest: BenchmarkManifest, dataset: MeasurementDataset) -> int | None:
    scales = {
        _scale_from_outcome_space_id(
            manifest=manifest,
            outcome_space_id=measurement.outcome_space.id,
        )
        for measurement in dataset.measurements
    }
    scales.discard(None)
    if len(scales) > 1:
        raise LocalResultImportError("measurement dataset spans multiple scales")
    return next(iter(scales), None)


def _scale_from_outcome_space_id(
    *,
    manifest: BenchmarkManifest,
    outcome_space_id: ProtocolIdentifier,
) -> int | None:
    if manifest.outcome_space is not None:
        return None
    prefix = f"{manifest.id.name}.outcomes.l"
    outcome_space_name = str(outcome_space_id.name)
    if not outcome_space_name.startswith(prefix):
        raise LocalResultImportError(
            "measurement outcome_space does not match scale-indexed benchmark"
        )
    scale_text = outcome_space_name.removeprefix(prefix)
    if not scale_text.isdecimal():
        raise LocalResultImportError("measurement outcome_space does not declare an integer scale")
    return int(scale_text)


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


def _point_complexity(point: Mapping[str, object]) -> float:
    return _as_nonnegative_number(point["complexity"], "complexity")


def _point_score(point: Mapping[str, object]) -> float:
    return _as_nonnegative_number(point["score"], "score")


def _model_sort_key(record: Mapping[str, object]) -> tuple[float, float, str]:
    cost_summary = _as_mapping(record["cost_summary"], "cost_summary")
    return (
        -_as_nonnegative_number(record["score"], "score"),
        _cost_value(cost_summary, "parameter_count"),
        str(record["model_key"]),
    )


def _run_sort_key(run: _BenchmarkRunRecord) -> tuple[str, str, str]:
    return (str(run.benchmark_id), run.run_id, run.source_path.as_posix())


def _identifier_atom(identifier: ProtocolIdentifier) -> str:
    return str(identifier.name).rsplit(".", maxsplit=1)[-1]


def _resolve_output_root(repository_root: Path, runs_root: Path) -> Path:
    if runs_root.is_absolute():
        resolved = runs_root.resolve()
    else:
        resolved = (repository_root / runs_root).resolve()
    if resolved == repository_root:
        raise LocalResultImportError("runs root must not be the repository root")
    return resolved


def _commit_runs_checkout(
    *,
    runs_root: Path,
    message: str,
    push: bool,
    token: str | None,
    endpoint: str,
) -> str:
    if not message.strip():
        raise LocalResultImportError("commit message must not be empty")
    if not _is_git_checkout(runs_root):
        raise LocalResultImportError("runs root must be a Git checkout when publishing")
    commit = _commit_checkout_if_dirty(
        runs_root=runs_root,
        message=message,
        push=push,
        token=token,
        endpoint=endpoint,
    )
    if commit is None:
        raise LocalResultImportError("no dirty run-state changes to commit")
    return commit


def _commit_checkout_if_dirty(
    *,
    runs_root: Path,
    message: str,
    push: bool,
    token: str | None,
    endpoint: str,
) -> str | None:
    if not message.strip():
        raise LocalResultImportError("commit message must not be empty")
    _git(runs_root, "add", "-A")
    status = _git(runs_root, "status", "--porcelain").stdout.strip()
    if not status:
        return None
    _ensure_git_identity(runs_root)
    _git(runs_root, "commit", "-m", message)
    commit = _git(runs_root, "rev-parse", "HEAD").stdout.strip()
    if push:
        _push_checkout(runs_root=runs_root, token=token, endpoint=endpoint)
    return commit


def _push_checkout(*, runs_root: Path, token: str | None, endpoint: str) -> str:
    commit = _git(runs_root, "rev-parse", "HEAD").stdout.strip()
    _git(
        runs_root,
        "push",
        "-u",
        "origin",
        "HEAD",
        token=token,
        endpoint=endpoint,
        username=_checkout_remote_owner(runs_root),
    )
    return commit


def _is_git_checkout(path: Path) -> bool:
    try:
        result = _git(path, "rev-parse", "--is-inside-work-tree")
    except LocalResultImportError:
        return False
    return result.stdout.strip() == "true"


def _ensure_git_identity(runs_root: Path) -> None:
    if not _git_config_value(runs_root, "user.email"):
        _git(runs_root, "config", "user.email", "leibniz@example.invalid")
    if not _git_config_value(runs_root, "user.name"):
        _git(runs_root, "config", "user.name", "Leibniz Operator")


def _ensure_local_git_checkout(runs_root: Path) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    if not _is_git_checkout(runs_root):
        _git(runs_root, "init")


def _git_config_value(runs_root: Path, key: str) -> str | None:
    try:
        return _git(runs_root, "config", "--get", key).stdout.strip() or None
    except LocalResultImportError:
        return None


def _git(
    path: Path,
    *args: str,
    token: str | None = None,
    endpoint: str = _default_hf_endpoint,
    username: str = "hf_user",
) -> subprocess.CompletedProcess[str]:
    try:
        return _run_git_process(
            ["git", "-C", str(path), *args],
            token=token,
            endpoint=endpoint,
            username=username,
        )
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
    token: str,
    endpoint: str,
) -> None:
    try:
        _run_git_process(
            ["git", "clone", source, str(target)],
            token=token,
            endpoint=endpoint,
            username=_repo_owner_from_url(source) or "hf_user",
        )
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        message = "git clone failed"
        if detail:
            message = f"{message}: {detail}"
        raise LocalResultImportError(message) from error


def _run_git_process(
    args: list[str],
    *,
    token: str | None,
    endpoint: str,
    username: str,
) -> subprocess.CompletedProcess[str]:
    env = _git_auth_env(token=token, endpoint=endpoint)
    if token is None:
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    with tempfile.TemporaryDirectory(prefix="leibniz-git-auth-") as auth_dir:
        askpass = Path(auth_dir) / "askpass.sh"
        askpass.write_text(
            "#!/bin/sh\n"
            "case \"$1\" in\n"
            "*Username*) printf '%s\\n' \"$LEIBNIZ_HF_USERNAME\" ;;\n"
            "*Password*) printf '%s\\n' \"$LEIBNIZ_HF_TOKEN\" ;;\n"
            "*) printf '\\n' ;;\n"
            "esac\n",
            encoding="utf-8",
        )
        askpass.chmod(0o700)
        assert env is not None
        env["GIT_ASKPASS"] = askpass.as_posix()
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["LEIBNIZ_HF_TOKEN"] = token
        env["LEIBNIZ_HF_USERNAME"] = username
        return subprocess.run(
            args,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )


def _git_auth_env(*, token: str | None, endpoint: str) -> dict[str, str] | None:
    if token is None:
        return None
    del endpoint
    env = os.environ.copy()
    return env


def _checkout_remote_owner(runs_root: Path) -> str:
    try:
        url = _git(runs_root, "remote", "get-url", "origin").stdout.strip()
    except LocalResultImportError:
        return "hf_user"
    return _repo_owner_from_url(url) or "hf_user"


def _repo_owner_from_url(url: str) -> str | None:
    marker = "/datasets/"
    if marker not in url:
        return None
    tail = url.split(marker, maxsplit=1)[1]
    owner = tail.split("/", maxsplit=1)[0].strip()
    return owner or None


def _create_hf_dataset_repo(*, repo_id: str, token: str, endpoint: str) -> bool:
    name, organization = _split_hf_repo_id(repo_id)
    payload = {
        "name": name,
        "organization": organization,
        "type": "dataset",
        "private": False,
        "exist_ok": True,
    }
    request = urllib.request.Request(
        f"{endpoint.rstrip('/')}/api/repos/create",
        data=canonical_document_bytes(payload),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": document_media_type(),
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(request, timeout=30).read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace").strip()
        message = f"Hugging Face repo create failed ({error.code})"
        if detail:
            message = f"{message}: {detail}"
        raise LocalResultImportError(message) from error
    except urllib.error.URLError as error:
        raise LocalResultImportError(f"Hugging Face repo create failed: {error}") from error
    return True


def _ensure_publication_checkout_structure(runs_root: Path) -> None:
    readme = runs_root / "README.md"
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
        marker = runs_root / directory / ".gitkeep"
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


def _split_hf_repo_id(repo_id: str) -> tuple[str, str]:
    organization, name = repo_id.split("/", maxsplit=1)
    return name, organization


def _validate_token(token: str | None) -> str:
    value = "" if token is None else token.strip()
    if not value:
        raise LocalResultImportError("Hugging Face token must not be empty")
    return value


def _bundle_filename(bundle_id: ProtocolIdentifier, digest: ContentDigest) -> str:
    name = str(bundle_id.name).replace(".", "_")
    version = str(bundle_id.version).replace(".", "_")
    return f"{name}_{version}_{str(digest)[:12]}{_document_suffix}"


def _validate_publication_bundle_view(record: Mapping[str, object]) -> None:
    _as_string(record.get("id"), "id")
    _as_string(record.get("digest"), "digest")
    _as_string(record.get("source_path"), "source_path")
    _as_string(record.get("submission_package_id"), "submission_package_id")
    benchmark_ids = _as_sequence(record.get("benchmark_ids"), "benchmark_ids")
    if not all(isinstance(item, str) and item for item in benchmark_ids):
        raise LocalResultImportError("benchmark_ids must contain nonempty strings")
    measurement_count = record.get("measurement_count")
    if not isinstance(measurement_count, int) or isinstance(measurement_count, bool):
        raise LocalResultImportError("measurement_count must be an integer")
    if measurement_count < 0:
        raise LocalResultImportError("measurement_count must be nonnegative")
    _as_mapping(record.get("measurement_dataset"), "measurement_dataset")
    _as_mapping(record.get("measurement_score_view"), "measurement_score_view")


def _validate_benchmark_result_view(record: Mapping[str, object]) -> None:
    if record.get("format_version") != _console_result_view_format_version:
        raise LocalResultImportError("console result view has unsupported format_version")
    results = _as_sequence(record.get("benchmark_results"), "benchmark_results")
    for index, result in enumerate(results):
        _validate_benchmark_result(_as_mapping(result, f"benchmark_results.{index}"))


def _validate_work_queue_view(record: Mapping[str, object]) -> None:
    if record.get("format_version") != _console_result_view_format_version:
        raise LocalResultImportError("console result view has unsupported format_version")
    for index, item in enumerate(_as_sequence(record.get("queue_items"), "queue_items")):
        _validate_work_queue_item(_as_mapping(item, f"queue_items.{index}"))


def _validate_work_queue_item(record: Mapping[str, object]) -> None:
    if record.get("format") != "leibniz.work-queue-item":
        raise LocalResultImportError("work queue item has unsupported format")
    if record.get("format_version") != _console_result_view_format_version:
        raise LocalResultImportError("work queue item has unsupported format_version")
    _as_string(record.get("id"), "queue_items.id")
    _as_string(record.get("benchmark_id"), "queue_items.benchmark_id")
    _as_string(record.get("proposal_id"), "queue_items.proposal_id")
    if "candidate_id" in record:
        _as_string(record.get("candidate_id"), "queue_items.candidate_id")
    _as_string(record.get("proposal_set_path"), "queue_items.proposal_set_path")
    command = _as_sequence(record.get("command"), "queue_items.command")
    if not all(isinstance(argument, str) and argument for argument in command):
        raise LocalResultImportError("queue_items.command must contain strings")
    status = _as_string(record.get("status"), "queue_items.status")
    if status not in {"pending", "reserved", "completed", "failed"}:
        raise LocalResultImportError(f"unsupported queue item status: {status}")
    sequence = _optional_int(record.get("sequence"), "queue_items.sequence")
    if sequence is None or sequence < 0:
        raise LocalResultImportError("queue_items.sequence must be nonnegative")
    if "run_id" in record:
        _as_string(record.get("run_id"), "queue_items.run_id")
    if "measurement_dataset_path" in record:
        _as_string(
            record.get("measurement_dataset_path"),
            "queue_items.measurement_dataset_path",
        )
    if "error" in record:
        _as_string(record.get("error"), "queue_items.error")


def _validate_benchmark_result(record: Mapping[str, object]) -> None:
    _as_string(record.get("benchmark_id"), "benchmark_id")
    cost_axes = _as_sequence(record.get("cost_axes"), "cost_axes")
    if not cost_axes:
        raise LocalResultImportError("cost_axes must not be empty")
    for index, axis in enumerate(cost_axes):
        axis_record = _as_mapping(axis, f"cost_axes.{index}")
        _as_string(axis_record.get("key"), f"cost_axes.{index}.key")
        _as_string(axis_record.get("label"), f"cost_axes.{index}.label")
    leaderboard = _as_sequence(record.get("leaderboard"), "leaderboard")
    for index, model in enumerate(leaderboard):
        _validate_model_result(_as_mapping(model, f"leaderboard.{index}"))
    frontiers = _as_mapping(record.get("frontiers"), "frontiers")
    for axis in ("parameter_count", "inference_flops", "parameter_bytes"):
        _as_sequence(frontiers.get(axis), f"frontiers.{axis}")
    history = _as_sequence(record.get("training_history"), "training_history")
    for index, run in enumerate(history):
        _validate_run_result(_as_mapping(run, f"training_history.{index}"))
    _as_sequence(record.get("model_inspections", ()), "model_inspections")
    for index, proposal in enumerate(_as_sequence(record.get("proposals", ()), "proposals")):
        _validate_proposal_result(_as_mapping(proposal, f"proposals.{index}"))


def _validate_model_result(record: Mapping[str, object]) -> None:
    _as_string(record.get("model_key"), "model_key")
    _as_string(record.get("architecture_digest"), "architecture_digest")
    _as_string(record.get("benchmark_id"), "benchmark_id")
    _as_nonnegative_number(record.get("score"), "score")
    _as_sequence(record.get("observed_complexities"), "observed_complexities")
    for index, point in enumerate(_as_sequence(record.get("points"), "points")):
        point_record = _as_mapping(point, f"points.{index}")
        _as_nonnegative_number(point_record.get("complexity"), f"points.{index}.complexity")
        _as_nonnegative_number(point_record.get("score"), f"points.{index}.score")
        if "sample_count" in point_record:
            _as_nonnegative_number(point_record.get("sample_count"), f"points.{index}.sample_count")
        _as_sequence(point_record.get("run_ids"), f"points.{index}.run_ids")
    _as_mapping(record.get("cost_summary"), "cost_summary")
    _as_sequence(record.get("run_ids"), "run_ids")
    _as_sequence(record.get("source_kinds"), "source_kinds")


def _validate_run_result(record: Mapping[str, object]) -> None:
    _as_string(record.get("source_kind"), "source_kind")
    _as_string(record.get("source_path"), "source_path")
    _as_string(record.get("run_id"), "run_id")
    _as_string(record.get("run_slug"), "run_slug")
    _as_string(record.get("benchmark_id"), "benchmark_id")
    _as_string(record.get("architecture_digest"), "architecture_digest")
    _as_string(record.get("model_key"), "model_key")
    _as_nonnegative_number(record.get("measurement_count"), "measurement_count")
    _as_nonnegative_number(record.get("score"), "score")
    _as_mapping(record.get("cost_summary"), "cost_summary")
    _as_mapping(record.get("architecture"), "architecture")
    if "model_inspection_digest" in record:
        _as_string(record.get("model_inspection_digest"), "model_inspection_digest")
    if "model_inspection_path" in record:
        _as_string(record.get("model_inspection_path"), "model_inspection_path")
    _as_string(record.get("measurement_dataset_digest"), "measurement_dataset_digest")
    if "sampled_competence" in record:
        _as_mapping(record.get("sampled_competence"), "sampled_competence")
    if "training_diagnostics" in record:
        _validate_training_diagnostics(
            _as_mapping(record.get("training_diagnostics"), "training_diagnostics")
        )
    if "console_view_model" in record:
        _validate_run_console_view_model(
            _as_mapping(record.get("console_view_model"), "console_view_model")
        )


def _validate_run_console_view_model(record: Mapping[str, object]) -> None:
    sections = _as_sequence(record.get("detail_sections"), "console_view_model.detail_sections")
    for section_index, section in enumerate(sections):
        section_record = _as_mapping(
            section,
            f"console_view_model.detail_sections.{section_index}",
        )
        _as_string(
            section_record.get("title"),
            f"console_view_model.detail_sections.{section_index}.title",
        )
        if "entries" in section_record:
            entries = _as_sequence(
                section_record["entries"],
                f"console_view_model.detail_sections.{section_index}.entries",
            )
            for entry_index, entry in enumerate(entries):
                entry_record = _as_mapping(
                    entry,
                    f"console_view_model.detail_sections.{section_index}.entries.{entry_index}",
                )
                _as_string(
                    entry_record.get("label"),
                    "console_view_model.detail_sections.entries.label",
                )
                _as_string(
                    entry_record.get("value"),
                    "console_view_model.detail_sections.entries.value",
                )
        if "table" in section_record:
            table = _as_mapping(
                section_record["table"],
                f"console_view_model.detail_sections.{section_index}.table",
            )
            _as_string(table.get("aria_label"), "console_view_model.detail_sections.table")
            columns = _as_sequence(
                table.get("columns"),
                f"console_view_model.detail_sections.{section_index}.table.columns",
            )
            if not all(isinstance(column, str) and column for column in columns):
                raise LocalResultImportError("console_view_model table columns must be strings")
            rows = _as_sequence(
                table.get("rows"),
                f"console_view_model.detail_sections.{section_index}.table.rows",
            )
            for row in rows:
                values = _as_sequence(row, "console_view_model.detail_sections.table.rows")
                if len(values) != len(columns) or not all(
                    isinstance(value, str) and value for value in values
                ):
                    raise LocalResultImportError(
                        "console_view_model table rows must match columns"
                    )


def _validate_training_diagnostics(record: Mapping[str, object]) -> None:
    status = _as_string(record.get("status"), "training_diagnostics.status")
    if status not in {
        "running",
        "completed",
        "converged",
        "budget-exhausted",
        "not-trainable",
        "failed",
    }:
        raise LocalResultImportError(f"unsupported training status: {status}")
    _as_string(record.get("stop_reason"), "training_diagnostics.stop_reason")
    _as_nonnegative_number(record.get("steps_run"), "training_diagnostics.steps_run")
    _as_nonnegative_number(
        record.get("validation_checks"),
        "training_diagnostics.validation_checks",
    )
    _as_nonnegative_number(
        record.get("best_validation_loss"),
        "training_diagnostics.best_validation_loss",
    )
    _as_nonnegative_number(
        record.get("best_validation_step"),
        "training_diagnostics.best_validation_step",
    )
    _as_nonnegative_number(
        record.get("best_validation_check"),
        "training_diagnostics.best_validation_check",
    )
    _as_nonnegative_number(
        record.get("final_validation_loss"),
        "training_diagnostics.final_validation_loss",
    )
    _as_nonnegative_number(
        record.get("final_validation_step"),
        "training_diagnostics.final_validation_step",
    )
    _as_nonnegative_number(
        record.get("final_validation_check"),
        "training_diagnostics.final_validation_check",
    )
    protocol = _as_mapping(record.get("protocol"), "training_diagnostics.protocol")
    for field in (
        "kind",
        "objective",
        "optimizer",
        "schedule",
        "validation_source",
    ):
        _as_string(protocol.get(field), f"training_diagnostics.protocol.{field}")
    for field in (
        "learning_rate",
        "seed",
        "batch_size",
        "max_steps",
        "validation_interval",
        "validation_sample_count",
        "min_delta",
        "patience",
    ):
        _as_nonnegative_number(
            protocol.get(field),
            f"training_diagnostics.protocol.{field}",
        )
    history = _as_sequence(
        record.get("validation_history"),
        "training_diagnostics.validation_history",
    )
    if not history:
        raise LocalResultImportError("training_diagnostics.validation_history must not be empty")
    for index, point in enumerate(history):
        point_record = _as_mapping(point, f"training_diagnostics.validation_history.{index}")
        for field in (
            "step",
            "validation_check",
            "validation_loss",
            "best_validation_loss",
            "best_validation_step",
            "best_validation_check",
            "stale_checks",
        ):
            _as_nonnegative_number(
                point_record.get(field),
                f"training_diagnostics.validation_history.{index}.{field}",
            )
        if "learning_rates" in point_record:
            for rate in _as_sequence(
                point_record.get("learning_rates"),
                f"training_diagnostics.validation_history.{index}.learning_rates",
            ):
                _as_nonnegative_number(
                    rate,
                    f"training_diagnostics.validation_history.{index}.learning_rates",
                )
    for index, artifact in enumerate(
        _as_sequence(record.get("artifacts"), "training_diagnostics.artifacts")
    ):
        artifact_record = _as_mapping(artifact, f"training_diagnostics.artifacts.{index}")
        _as_string(artifact_record.get("kind"), f"training_diagnostics.artifacts.{index}.kind")
        _as_string(
            artifact_record.get("digest"),
            f"training_diagnostics.artifacts.{index}.digest",
        )
        if "path" in artifact_record:
            _as_string(
                artifact_record.get("path"),
                f"training_diagnostics.artifacts.{index}.path",
            )


def _validate_proposal_result(record: Mapping[str, object]) -> None:
    _as_string(record.get("id"), "proposals.id")
    _as_nonnegative_number(record.get("rank"), "proposals.rank")
    _as_string(record.get("candidate_kind"), "proposals.candidate_kind")
    _as_string(record.get("candidate_id"), "proposals.candidate_id")
    _as_string(record.get("rationale"), "proposals.rationale")
    for field in (
        "predicted_score",
        "uncertainty",
        "acquisition_value",
        "novelty",
        "expected_frontier_improvement",
    ):
        if field in record:
            _as_nonnegative_number(record[field], f"proposals.{field}")
    if "acquisition_model" in record:
        _as_string(record.get("acquisition_model"), "proposals.acquisition_model")
    if "acquisition_components" in record:
        _as_mapping(record.get("acquisition_components"), "proposals.acquisition_components")
    if "search_diagnostics" in record:
        diagnostics = _as_mapping(record.get("search_diagnostics"), "proposals.search_diagnostics")
        if "search_distribution_id" in diagnostics:
            _as_string(
                diagnostics.get("search_distribution_id"),
                "proposals.search_diagnostics.search_distribution_id",
            )
        if "semantic_coordinates" in diagnostics:
            for index, coordinate in enumerate(
                _as_sequence(
                    diagnostics.get("semantic_coordinates"),
                    "proposals.search_diagnostics.semantic_coordinates",
                )
            ):
                coordinate_record = _as_mapping(
                    coordinate,
                    f"proposals.search_diagnostics.semantic_coordinates.{index}",
                )
                _as_string(
                    coordinate_record.get("name"),
                    f"proposals.search_diagnostics.semantic_coordinates.{index}.name",
                )
                value = coordinate_record.get("value")
                if not isinstance(value, str | int):
                    raise LocalResultImportError(
                        "proposals.search_diagnostics.semantic_coordinates.value "
                        "must be a string or integer"
                    )
        if "sampled_resource_stratum" in diagnostics:
            stratum = _as_mapping(
                diagnostics.get("sampled_resource_stratum"),
                "proposals.search_diagnostics.sampled_resource_stratum",
            )
            _as_nonnegative_number(
                stratum.get("index"),
                "proposals.search_diagnostics.sampled_resource_stratum.index",
            )
            _as_nonnegative_number(
                stratum.get("count"),
                "proposals.search_diagnostics.sampled_resource_stratum.count",
            )
        if "nearest_measured_support" in diagnostics:
            support = _as_mapping(
                diagnostics.get("nearest_measured_support"),
                "proposals.search_diagnostics.nearest_measured_support",
            )
            _as_string(
                support.get("architecture_digest"),
                "proposals.search_diagnostics.nearest_measured_support.architecture_digest",
            )
            _as_nonnegative_number(
                support.get("parameter_count"),
                "proposals.search_diagnostics.nearest_measured_support.parameter_count",
            )
            _as_nonnegative_number(
                support.get("score"),
                "proposals.search_diagnostics.nearest_measured_support.score",
            )
            _as_nonnegative_number(
                support.get("log_parameter_distance"),
                "proposals.search_diagnostics.nearest_measured_support.log_parameter_distance",
            )
    if "command" in record:
        command = _as_sequence(record["command"], "proposals.command")
        if not all(isinstance(argument, str) and argument for argument in command):
            raise LocalResultImportError("proposals.command must contain strings")


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LocalResultImportError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, field: str) -> tuple[object, ...]:
    if isinstance(value, tuple):
        return cast(tuple[object, ...], value)
    if isinstance(value, list):
        return tuple(cast(list[object], value))
    raise LocalResultImportError(f"{field}: expected parsed sequence")


def _as_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise LocalResultImportError(f"{field}: expected nonempty string")
    return value


def _as_identifier(value: object, field: str) -> ProtocolIdentifier:
    if isinstance(value, ProtocolIdentifier):
        return value
    if isinstance(value, str):
        try:
            return ProtocolIdentifier.parse(value)
        except ValueError as error:
            raise LocalResultImportError(str(error)) from error
    raise LocalResultImportError(f"{field}: expected identifier")


def _optional_int(value: object, field: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise LocalResultImportError(f"{field}: expected integer")
    return value


def _as_digest(value: object, *, field: str) -> ContentDigest:
    if not isinstance(value, str):
        raise LocalResultImportError(f"{field}: expected digest string")
    algorithm, separator, digest_hex = value.partition(":")
    if separator == "":
        raise LocalResultImportError(f"{field}: expected algorithm:digest")
    try:
        return ContentDigest(algorithm=algorithm, hex=digest_hex)
    except ValueError as error:
        raise LocalResultImportError(str(error)) from error


def _as_nonnegative_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LocalResultImportError(f"{field}: expected number")
    numeric = float(value)
    if numeric < 0 or not math.isfinite(numeric):
        raise LocalResultImportError(f"{field}: expected finite nonnegative number")
    return numeric
