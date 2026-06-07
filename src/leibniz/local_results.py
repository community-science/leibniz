"""Operator-local result import and console view materialization."""

from __future__ import annotations

import importlib
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.benchmark_evaluation import (
    CompetencePoint,
    ComputeCostPoint,
    integrated_compute_cost,
    sampled_competence_frontier_score,
)
from leibniz.benchmark_implementations import (
    Benchmark,
    discover_benchmark_roots,
    load_benchmark,
)
from leibniz.benchmarks import BenchmarkManifest
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
)
from leibniz.model_inspection import (
    ModelInspectionRecord,
)
from leibniz.model_operators import (
    architecture_with_input_shape,
    summarize_architecture_operators,
)
from leibniz.observation_generation import ComplexityCandidate, ComplexityRequest
from leibniz.records import RecordExtractor
from leibniz.training_runs import TrainingRunRecord

__all__ = [
    "LocalBenchmarkResultViewSummary",
    "LocalResultCheckoutSummary",
    "LocalResultPublishSummary",
    "LocalResultPushSummary",
    "LocalResultImportError",
    "competent_complexity_score",
    "initialize_result_checkout",
    "load_console_result_view",
    "materialize_benchmark_result_views",
    "publish_local_benchmark_results",
    "push_result_checkout",
]

_protocol_formats = console_protocol_formats()
_protocol_format_versions = console_protocol_format_versions()
_benchmark_result_view_format = _protocol_formats.benchmark_result_view
_console_result_view_format_version = _protocol_format_versions.result_view
_document_suffix = document_filename_suffix()
_default_results_root = Path("results")
_result_directories = (
    "evaluations",
    "models",
    "training",
    "views",
)
_benchmark_cost_axis_key = "cost"
_benchmark_cost_axis_keys = (_benchmark_cost_axis_key,)
_default_bit_length_per_op = 32.0
_console_validation_history_max_points = 512
_reference_curve_default_maximum_cost = 10_000_000_000
_reference_curve_initial_curriculum_count = 16
_component_count = 1
_reference_baseline_complexity = 1.0


class _SummaryRecordMixin:
    def to_record(self) -> dict[str, object]:
        return _summary_record(self)


class LocalResultImportError(ValueError):
    """Raised when local result import cannot produce a valid console view."""


_extract = RecordExtractor(error_type=LocalResultImportError)


@dataclass(frozen=True, slots=True)
class LocalBenchmarkResultViewSummary(_SummaryRecordMixin):
    source_files: tuple[Path, ...]
    view_file: Path
    benchmark_count: int
    model_count: int
    run_count: int
    benchmark_view_files: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class LocalResultPublishSummary(_SummaryRecordMixin):
    source_files: tuple[Path, ...]
    measurement_count: int
    git_commit: str | None = None
    git_pushed: bool = False
    remote: str | None = None
    remote_commit: str | None = None


@dataclass(frozen=True, slots=True)
class LocalResultCheckoutSummary(_SummaryRecordMixin):
    repo_id: str | None
    results_root: Path
    repo_url: str | None
    created_or_reused: bool
    scaffold_commit: str | None
    pushed: bool


@dataclass(frozen=True, slots=True)
class LocalResultPushSummary(_SummaryRecordMixin):
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


def initialize_result_checkout(
    *,
    repo_id: str | None,
    repository_root: Path | None = None,
    results_root: Path = _default_results_root,
    remote: str = "auto",
    local_only: bool = False,
    push: bool = False,
    commit_message: str = "Initialize Leibniz result checkout",
    token: str | None = None,
) -> LocalResultCheckoutSummary:
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
        selected_remote = _select_result_remote(
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
    _ensure_result_checkout_structure(results_root)
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
    return LocalResultCheckoutSummary(
        repo_id=repo_id,
        results_root=results_root,
        repo_url=repo_url,
        created_or_reused=created_or_reused,
        scaffold_commit=scaffold_commit,
        pushed=push and (scaffold_commit is not None or selected_remote == "hf"),
    )


def push_result_checkout(
    *,
    repository_root: Path | None = None,
    results_root: Path = _default_results_root,
    repo_id: str | None = None,
    remote: str = "auto",
    token: str | None = None,
) -> LocalResultPushSummary:
    """Push an existing result checkout without creating a commit."""

    repository_root = Path.cwd().resolve() if repository_root is None else repository_root.resolve()
    results_root = _resolve_output_root(repository_root, results_root)
    selected_remote = _select_result_remote(
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
    return LocalResultPushSummary(
        results_root=results_root,
        pushed_commit=pushed_commit,
    )


def publish_local_benchmark_results(
    *,
    repository_root: Path | None = None,
    results_root: Path = _default_results_root,
    commit: bool = True,
    push: bool = True,
    repo_id: str | None = None,
    remote: str = "auto",
    token: str | None = None,
    commit_message: str = "Publish Leibniz benchmark results",
) -> LocalResultPublishSummary:
    repository_root = Path.cwd().resolve() if repository_root is None else repository_root.resolve()
    results_root = _resolve_output_root(repository_root, results_root)
    if push and not commit:
        raise LocalResultImportError("push requires committing the result checkout")
    runs = _local_run_records(results_root)
    if not runs:
        raise LocalResultImportError("no local benchmark result records found")
    measurement_count = sum(run.measurement_count for run in runs)
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
    return LocalResultPublishSummary(
        source_files=tuple(run.source_path for run in runs),
        measurement_count=measurement_count,
        git_commit=git_commit,
        git_pushed=push and selected_remote == "git",
        remote=selected_remote,
        remote_commit=remote_commit,
    )


def materialize_benchmark_result_views(
    *,
    repository_root: Path | None = None,
    results_root: Path = _default_results_root,
) -> LocalBenchmarkResultViewSummary:
    """Derive console benchmark result views from local result state."""

    repository_root = Path.cwd().resolve() if repository_root is None else repository_root.resolve()
    results_root = _resolve_output_root(repository_root, results_root)
    benchmarks = _known_benchmarks(repository_root)
    local_runs = _local_run_records(results_root)
    local_training_estimates = _local_training_estimate_records(
        results_root,
        repository_root=repository_root,
        accepted_run_slugs={run.run_slug for run in local_runs},
    )
    runs = tuple(
        sorted(
            (*local_runs, *local_training_estimates),
            key=_run_sort_key,
        )
    )
    unknown_benchmark_ids = sorted(
        {run.benchmark_id for run in runs if run.benchmark_id not in benchmarks},
        key=str,
    )
    if unknown_benchmark_ids:
        raise LocalResultImportError(
            f"unknown benchmark id in local results: {unknown_benchmark_ids[0]}"
        )
    benchmark_records: list[Mapping[str, object]] = []
    benchmark_view_files: list[Path] = []
    primary_view_file: Path | None = None
    view_root = results_root / "views"
    view_root.mkdir(parents=True, exist_ok=True)
    for benchmark_id, benchmark in sorted(benchmarks.items(), key=lambda item: str(item[0])):
        benchmark_runs = tuple(run for run in runs if run.benchmark_id == benchmark_id)
        benchmark_record = _benchmark_result_record(
            benchmark=benchmark,
            repository_root=repository_root,
            runs=benchmark_runs,
        )
        benchmark_records.append(benchmark_record)
        benchmark_view_root = view_root / _identifier_atom(benchmark_id)
        benchmark_view_root.mkdir(parents=True, exist_ok=True)
        benchmark_view_file = benchmark_view_root / ("benchmark_results" + _document_suffix)
        benchmark_view_file.write_bytes(
            canonical_document_bytes(
                {
                    "format": _benchmark_result_view_format,
                    "format_version": _console_result_view_format_version,
                    "benchmark_results": [benchmark_record],
                }
            )
            + b"\n"
        )
        benchmark_view_files.append(benchmark_view_file)
        if primary_view_file is None or benchmark_runs:
            primary_view_file = benchmark_view_file

    return LocalBenchmarkResultViewSummary(
        source_files=tuple(run.source_path for run in runs),
        view_file=primary_view_file if primary_view_file is not None else benchmark_view_files[0],
        benchmark_view_files=tuple(benchmark_view_files),
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
    raise LocalResultImportError("console result view has unsupported format")


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
class _EvaluationBundleSummary:
    run_slug: str
    benchmark_manifest: Mapping[str, object]
    architecture_manifest: ArchitectureManifest
    model_checkpoint: Mapping[str, object]
    model_inspection: Mapping[str, object]
    measurement_score_view: Mapping[str, object]
    sampled_competence: Mapping[str, object]
    evaluation_protocol: Mapping[str, object]
    evaluation_curriculum: Mapping[str, object]
    throughput: Mapping[str, object]


def _known_benchmarks(
    repository_root: Path,
) -> dict[ProtocolIdentifier, Benchmark]:
    benchmark_root = repository_root / "src" / "leibniz" / "benchmarks"
    benchmarks: dict[ProtocolIdentifier, Benchmark] = {}
    for path in discover_benchmark_roots(benchmark_root):
        benchmark = load_benchmark(path)
        benchmarks[benchmark.manifest.id] = benchmark
    if not benchmarks:
        raise LocalResultImportError("no known benchmark manifests found")
    return benchmarks


def _local_run_records(
    results_root: Path,
    *,
    include_model_details: bool = True,
) -> tuple[_BenchmarkRunRecord, ...]:
    evaluation_root = results_root / "evaluations"
    if not evaluation_root.is_dir():
        return ()
    training_summaries = _training_summary_records_by_run_slug(results_root)
    records: list[_BenchmarkRunRecord] = []
    for path in sorted(evaluation_root.rglob("*" + _document_suffix)):
        record = load_object_document(path.read_bytes(), description="benchmark evaluation")
        if record.get("format") != "leibniz.benchmark-evaluation":
            continue
        source_path = _result_state_record_path(path, results_root=results_root)
        try:
            summary = _evaluation_bundle_summary_from_record(record)
        except LocalResultImportError as error:
            raise LocalResultImportError(f"{source_path}: {error}") from error
        model_key = _model_key_from_checkpoint_record(summary.model_checkpoint)
        benchmark_id = ProtocolIdentifier.parse(
            _extract.non_empty_string(
                summary.benchmark_manifest.get("id"),
                "benchmark_manifest.id",
            )
        )
        measurement_dataset_digest = ContentDigest.from_string(
            summary.measurement_score_view.get("source_dataset_digest"),
            field="measurement_score_view.source_dataset_digest",
            error_type=LocalResultImportError,
        )
        model_inspection_digest = ContentDigest.from_value(summary.model_inspection)
        model_inspection: Mapping[str, object]
        cost_summary: Mapping[str, object]
        try:
            cost_summary = _evaluation_summary_cost_summary(
                summary,
            )
        except LocalResultImportError as error:
            raise LocalResultImportError(f"{source_path}: {error}") from error
        training_summary_entry = training_summaries.get(summary.run_slug)
        training_summary = (
            None if training_summary_entry is None else training_summary_entry[0]
        )
        if include_model_details:
            model_inspection = _model_inspection_view_record(
                inspection=summary.model_inspection,
                source_path=source_path,
                measurement_dataset_digest=measurement_dataset_digest,
                training_summary=training_summary,
                artifact_references=_evaluation_summary_artifact_references(summary),
            )
            model_inspection_path = source_path
        else:
            model_inspection = {}
            model_inspection_path = None
        records.append(
            _BenchmarkRunRecord(
                source_kind="local-run",
                result_status=_evaluation_result_status(summary),
                source_path=source_path,
                run_id=summary.run_slug,
                run_slug=summary.run_slug,
                benchmark_id=benchmark_id,
                architecture_digest=summary.architecture_manifest.digest,
                model_key=model_key,
                complexity=_sampled_competence_record_complexity(summary.sampled_competence),
                measurement_count=_as_positive_int(
                    summary.sampled_competence.get("sample_count"),
                    "sampled_competence.sample_count",
                ),
                score=_as_probability(
                    summary.sampled_competence.get("mean_accepted_mass"),
                    "sampled_competence.mean_accepted_mass",
                ),
                cost_summary=cost_summary,
                architecture=summary.architecture_manifest.to_record(),
                model_inspection=model_inspection,
                model_inspection_digest=model_inspection_digest,
                model_inspection_path=model_inspection_path,
                measurement_dataset=MeasurementDataset(measurements=()),
                measurement_dataset_digest=measurement_dataset_digest,
                sampled_competence=summary.sampled_competence,
                training_summary=training_summary,
            )
        )
    return tuple(records)


def _training_summary_records_by_run_slug(
    results_root: Path,
) -> dict[str, tuple[Mapping[str, object], Path]]:
    training_root = results_root / "training"
    if not training_root.is_dir():
        return {}
    records: dict[str, tuple[Mapping[str, object], Path]] = {}
    for path in sorted(training_root.rglob("*" + _document_suffix)):
        summary = load_object_document(path.read_bytes(), description="training record")
        if summary.get("format") not in {
            "leibniz.benchmark-run",
            "leibniz.benchmark-training-progress",
        }:
            continue
        run_slug = _extract.non_empty_string(summary.get("run_slug"), "run_slug")
        records[run_slug] = (summary, path)
    return records


def _local_training_estimate_records(
    results_root: Path,
    *,
    repository_root: Path,
    accepted_run_slugs: set[str],
) -> tuple[_BenchmarkRunRecord, ...]:
    training_root = results_root / "training"
    if not training_root.is_dir():
        return ()
    training_summaries = _training_summary_records_by_run_slug(results_root)
    records: list[_BenchmarkRunRecord] = []
    empty_dataset = MeasurementDataset(measurements=())
    for summary, path in training_summaries.values():
        run_slug = _extract.non_empty_string(summary.get("run_slug"), "run_slug")
        if run_slug in accepted_run_slugs:
            continue
        estimate = _extract.optional_mapping(
            summary.get("training_estimate"),
            "training_estimate",
        )
        if estimate is None:
            continue
        sampled_competence = _extract.mapping(
            estimate.get("sampled_competence"),
            "training_estimate.sampled_competence",
        )
        architecture = _architecture_manifest_from_training_record(
            repository_root=repository_root,
            training_summary=summary,
        )
        inspection = _inspection_from_architecture(architecture)
        cost_summary = dict(_extract.mapping(summary.get("cost_summary"), "cost_summary"))
        records.append(
            _BenchmarkRunRecord(
                source_kind="local-training-estimate",
                result_status="provisional",
                source_path=_result_state_record_path(path, results_root=results_root),
                run_id=run_slug,
                run_slug=run_slug,
                benchmark_id=_as_identifier(summary.get("benchmark_id"), "benchmark_id"),
                architecture_digest=architecture.digest,
                model_key=str(architecture.digest),
                complexity=_sampled_competence_record_complexity(sampled_competence),
                measurement_count=0,
                score=_as_nonnegative_number(
                    estimate.get("score"),
                    "training_estimate.score",
                ),
                cost_summary=cost_summary,
                architecture=architecture.to_record(),
                model_inspection=_model_inspection_view_record(
                    inspection=inspection.to_record(),
                    source_path=_result_state_record_path(path, results_root=results_root),
                    measurement_dataset_digest=empty_dataset.digest,
                    training_summary=summary,
                    artifact_references=None,
                ),
                model_inspection_digest=inspection.digest,
                model_inspection_path=None,
                measurement_dataset=empty_dataset,
                measurement_dataset_digest=empty_dataset.digest,
                sampled_competence=sampled_competence,
                training_summary=summary,
            )
        )
    return tuple(records)


def _result_state_record_path(path: Path, *, results_root: Path) -> Path:
    resolved = path.resolve()
    resolved_results_root = results_root.resolve()
    if resolved.is_relative_to(resolved_results_root):
        return Path(results_root.name) / resolved.relative_to(resolved_results_root)
    return path


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
    artifact_references: tuple[Mapping[str, object], ...] | None,
) -> Mapping[str, object]:
    record = cast(dict[str, object], _view_record_without_parameter_counts(inspection))
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
    if artifact_references is not None:
        record["artifacts"] = [dict(reference) for reference in artifact_references]
    return record


def _view_record_without_parameter_counts(value: object) -> object:
    if isinstance(value, Mapping):
        record = cast(Mapping[object, object], value)
        return {
            str(key): _view_record_without_parameter_counts(item)
            for key, item in record.items()
            if key != "parameter_count"
        }
    if isinstance(value, list | tuple):
        return [
            _view_record_without_parameter_counts(item)
            for item in cast(tuple[object, ...] | list[object], value)
        ]
    return value


def _model_key_from_checkpoint_record(record: Mapping[str, object]) -> str:
    return str(
        ContentDigest.from_string(
            record.get("digest"),
            field="model_checkpoint.digest",
            error_type=LocalResultImportError,
        )
    )


def _evaluation_bundle_summary_from_record(
    record: Mapping[str, object],
) -> _EvaluationBundleSummary:
    if record.get("format") != "leibniz.benchmark-evaluation":
        raise LocalResultImportError("benchmark evaluation has unsupported format")
    if record.get("format_version") != 1:
        raise LocalResultImportError("benchmark evaluation has unsupported format_version")
    return _EvaluationBundleSummary(
        run_slug=_extract.non_empty_string(record.get("run_slug"), "run_slug"),
        benchmark_manifest=_extract.mapping(
            record.get("benchmark_manifest"),
            "benchmark_manifest",
        ),
        architecture_manifest=ArchitectureManifest.from_record(
            _extract.mapping(record.get("architecture_manifest"), "architecture_manifest")
        ),
        model_checkpoint=_extract.mapping(record.get("model_checkpoint"), "model_checkpoint"),
        model_inspection=_extract.mapping(record.get("model_inspection"), "model_inspection"),
        measurement_score_view=_extract.mapping(
            record.get("measurement_score_view"),
            "measurement_score_view",
        ),
        sampled_competence=_extract.mapping(
            record.get("sampled_competence"),
            "sampled_competence",
        ),
        evaluation_protocol=_extract.mapping(
            record.get("evaluation_protocol"),
            "evaluation_protocol",
        ),
        evaluation_curriculum=_extract.mapping(
            record.get("evaluation_curriculum"),
            "evaluation_curriculum",
        ),
        throughput=_extract.mapping(record.get("throughput"), "throughput"),
    )


def _evaluation_summary_cost_summary(
    summary: _EvaluationBundleSummary,
) -> Mapping[str, object]:
    inspection_cost = _extract.mapping(
        summary.model_inspection.get("cost_summary"),
        "model_inspection.cost_summary",
    )
    cost_summary = dict(inspection_cost)
    cost_summary.pop("parameter_count", None)
    cost_summary.pop("inference_compute", None)
    cost_summary["inference_compute"] = _evaluation_summary_max_inference_compute(summary)
    training_compute = summary.evaluation_protocol.get("training_compute")
    if training_compute is not None:
        cost_summary["training_compute"] = _as_nonnegative_number(
            training_compute,
            "evaluation_protocol.training_compute",
        )
    return cost_summary


def _evaluation_result_status(summary: _EvaluationBundleSummary) -> str:
    score_status = summary.evaluation_protocol.get("score_status")
    if score_status in {"accepted", "provisional"}:
        return cast(str, score_status)
    if score_status is not None:
        raise LocalResultImportError("evaluation_protocol.score_status is invalid")
    if _evaluation_summary_capacity_limited(summary):
        return "provisional"
    return "accepted"


def _evaluation_summary_capacity_limited(summary: _EvaluationBundleSummary) -> bool:
    for key in ("checkpoint_evaluation", "evaluation"):
        phase = summary.throughput.get(key)
        if not isinstance(phase, Mapping):
            continue
        phase_record = _extract.mapping(
            cast(Mapping[str, object], phase),
            f"throughput.{key}",
        )
        if phase_record.get("capacity_limited") is True:
            return True
    return False


def _evaluation_summary_max_inference_compute(
    summary: _EvaluationBundleSummary,
) -> float:
    checkpoint_value = _throughput_max_inference_compute(
        summary.throughput.get("checkpoint_evaluation"),
        "evaluation_bundle.throughput.checkpoint_evaluation",
    )
    if checkpoint_value is not None:
        return checkpoint_value
    evaluation_value = _throughput_max_inference_compute(
        summary.throughput.get("evaluation"),
        "evaluation_bundle.throughput.evaluation",
    )
    if evaluation_value is not None:
        return evaluation_value
    raise LocalResultImportError(
        "benchmark evaluation bundle is missing measured max_inference_compute"
    )


def _evaluation_summary_artifact_references(
    summary: _EvaluationBundleSummary,
) -> tuple[Mapping[str, object], ...]:
    checkpoint = dict(summary.model_checkpoint)
    model_artifacts = _as_sequence(
        checkpoint.get("model_artifacts", ()),
        "model_checkpoint.model_artifacts",
    )
    training_provenance = _as_sequence(
        checkpoint.get("training_provenance", ()),
        "model_checkpoint.training_provenance",
    )
    references: list[Mapping[str, object]] = [
        {
            "kind": "measurement-dataset",
            "digest": _extract.non_empty_string(
                summary.measurement_score_view.get("source_dataset_digest"),
                "measurement_score_view.source_dataset_digest",
            ),
        },
        {
            "kind": "model-inspection",
            "digest": str(ContentDigest.from_value(summary.model_inspection)),
        },
        {
            "kind": "model-checkpoint",
            "digest": _extract.non_empty_string(
                checkpoint.get("digest"),
                "model_checkpoint.digest",
            ),
            "path": _extract.non_empty_string(
                checkpoint.get("path"),
                "model_checkpoint.path",
            ),
        },
        {
            "kind": "model-manifest",
            "digest": _extract.non_empty_string(
                checkpoint.get("manifest_digest"),
                "model_checkpoint.manifest_digest",
            ),
            "path": _extract.non_empty_string(
                checkpoint.get("manifest_path"),
                "model_checkpoint.manifest_path",
            ),
        },
    ]
    for index, artifact in enumerate(model_artifacts):
        references.append(
            _extract.mapping(artifact, f"model_checkpoint.model_artifacts.{index}")
        )
    for index, provenance in enumerate(training_provenance):
        references.append(
            _extract.mapping(provenance, f"model_checkpoint.training_provenance.{index}")
        )
    return tuple(references)


def _training_diagnostics_record(run: _BenchmarkRunRecord) -> Mapping[str, object]:
    if run.training_summary is None:
        raise LocalResultImportError("training summary is required for diagnostics")
    training_run = TrainingRunRecord.from_record(
        _extract.mapping(run.training_summary.get("training_run"), "training_run")
    )
    history = training_run.validation_history
    console_history = _sample_console_validation_history(history)
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
        "validation_history": [
            _console_validation_history_point_record(point)
            for point in console_history
        ],
        "validation_history_sample_count": len(console_history),
        "validation_history_total_count": len(history),
        "artifacts": _training_artifact_references(run),
    }
    validation_loss_reference = _training_validation_loss_reference(
        training_summary=run.training_summary,
        history=history,
    )
    if validation_loss_reference is not None:
        record["validation_loss_reference"] = validation_loss_reference
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


def _sample_console_validation_history(
    history: Sequence[Any],
) -> tuple[Any, ...]:
    if len(history) <= _console_validation_history_max_points:
        return tuple(history)
    if _console_validation_history_max_points < 4:
        return (history[0], history[-1])

    bucket_count = max(1, (_console_validation_history_max_points - 2) // 2)
    interior = history[1:-1]
    selected: dict[tuple[int, int], Any] = {
        _validation_history_point_key(history[0]): history[0],
        _validation_history_point_key(history[-1]): history[-1],
    }
    for bucket_index in range(bucket_count):
        start = bucket_index * len(interior) // bucket_count
        end = (bucket_index + 1) * len(interior) // bucket_count
        bucket = interior[start:end]
        if not bucket:
            continue
        low = min(bucket, key=lambda point: (point.validation_loss, point.step))
        high = max(bucket, key=lambda point: (point.validation_loss, -point.step))
        selected[_validation_history_point_key(low)] = low
        selected[_validation_history_point_key(high)] = high
    return tuple(
        selected[key]
        for key in sorted(selected, key=lambda key: (key[0], key[1]))
    )


def _console_validation_history_point_record(point: Any) -> dict[str, object]:
    record = {
        "step": point.step,
        "validation_check": point.validation_check,
        "validation_loss": point.validation_loss,
        "stale_checks": point.stale_checks,
    }
    learning_rates = getattr(point, "learning_rates", ())
    if learning_rates:
        record["learning_rates"] = list(learning_rates)
    return record


def _validation_history_point_key(point: Any) -> tuple[int, int]:
    return (int(point.step), int(point.validation_check))


def _training_validation_loss_reference(
    *,
    training_summary: Mapping[str, object],
    history: Sequence[Any],
) -> float | None:
    estimate = training_summary.get("training_estimate")
    if isinstance(estimate, Mapping):
        estimate_record = cast(Mapping[str, object], estimate)
        chance_mass = estimate_record.get("chance_mass")
        if isinstance(chance_mass, int | float) and 0.0 < float(chance_mass) <= 1.0:
            return -math.log(float(chance_mass))
    for point in reversed(history):
        score_estimate = getattr(point, "score_estimate", None)
        if not isinstance(score_estimate, Mapping):
            continue
        score_estimate_record = cast(Mapping[str, object], score_estimate)
        chance_mass = score_estimate_record.get("chance_mass")
        if isinstance(chance_mass, int | float) and 0.0 < float(chance_mass) <= 1.0:
            return -math.log(float(chance_mass))
    return None


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
                    *(
                        (
                            (
                                "Learning Rate",
                                _console_number_value(
                                    protocol.get("learning_rate"),
                                    precision=4,
                                ),
                            ),
                        )
                        if "learning_rate" in protocol
                        else ()
                    ),
                    ("Steps", _console_number_value(protocol.get("max_steps"))),
                    (
                        "Training Batch Target",
                        _console_number_value(protocol.get("training_batch_target")),
                    ),
                    ("Gate Check", _console_number_value(protocol.get("gate_check_interval"))),
                    (
                        "Checkpoint Gate",
                        _console_number_value(
                            run.training_summary.get("model_checkpoint_gate_interval")
                        ),
                    ),
                    (
                        "Gate Batch Target",
                        _console_number_value(protocol.get("gate_batch_target")),
                    ),
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
    benchmark: Benchmark,
    repository_root: Path,
    runs: tuple[_BenchmarkRunRecord, ...],
) -> dict[str, object]:
    manifest = benchmark.manifest
    accepted_runs = tuple(run for run in runs if run.result_status == "accepted")
    models = tuple(
        _model_result_records(
            accepted_runs,
            manifest=manifest,
            repository_root=repository_root,
        )
    )
    model_candidates = tuple(
        _model_result_records(
            runs,
            manifest=manifest,
            repository_root=repository_root,
        )
    )
    record: dict[str, object] = {
        "benchmark_id": str(manifest.id),
        "leaderboard": list(models),
        "model_candidates": list(model_candidates),
        "frontiers": {
            axis: _frontier_records(models, cost_axis=axis)
            for axis in _benchmark_cost_axis_keys
        },
        "reference_curves": _benchmark_reference_curve_records(
            generator=benchmark.generator,
            runs=runs,
        ),
        "training_history": [run.to_record(complexity_axis=None) for run in runs],
        "plot_runs": [run.to_record(complexity_axis=None) for run in runs],
        "model_inspections": _model_inspection_records(accepted_runs),
    }
    return record


def _benchmark_reference_curve_records(
    *,
    generator: Any,
    runs: tuple[_BenchmarkRunRecord, ...],
) -> list[dict[str, object]]:
    points = _generator_oracle_inference_reference_points(generator)
    if not points:
        points = _oracle_inference_reference_points(
            generator=generator,
            runs=runs,
        )
    if not points:
        return []
    return [
        {
            "kind": "oracle-inference-compute-reference-v1",
            "key": "oracle_inference_compute",
            "label": "Oracle Reference",
            "x_axis": _benchmark_cost_axis_key,
            "y_axis": "score",
            "points": points,
        }
    ]


def _generator_oracle_inference_reference_points(
    generator: Any,
) -> list[dict[str, object]]:
    if not hasattr(generator, "oracle_inference_reference_points"):
        return []
    raw_points = generator.oracle_inference_reference_points(
        maximum_cost=_reference_curve_default_maximum_cost
    )
    reference_points: list[dict[str, object]] = []
    for index, raw_point in enumerate(
        _as_sequence(raw_points, "oracle_inference_reference_points")
    ):
        point = _extract.mapping(raw_point, f"oracle_inference_reference_points.{index}")
        complexity = _as_nonnegative_number(
            point.get("complexity"),
            f"oracle_inference_reference_points.{index}.complexity",
        )
        score = _as_nonnegative_number(
            point.get("score"),
            f"oracle_inference_reference_points.{index}.score",
        )
        cost = _as_nonnegative_number(
            point.get("cost"),
            f"oracle_inference_reference_points.{index}.cost",
        )
        if cost <= 0:
            raise LocalResultImportError(
                f"oracle_inference_reference_points.{index}.cost must be positive"
            )
        metadata = _extract.mapping(
            point.get("metadata"),
            f"oracle_inference_reference_points.{index}.metadata",
        )
        reference_points.append(
            {
                "complexity": complexity,
                "score": score,
                "cost_density": cost,
                "metadata": dict(metadata),
            }
        )
    return _integrated_reference_curve_points(reference_points)


def _reference_curve_candidates(
    *,
    generator: Any,
    runs: tuple[_BenchmarkRunRecord, ...],
) -> tuple[ComplexityCandidate, ...]:
    candidates: list[ComplexityCandidate] = []
    seen_complexities: set[float] = set()

    def append_candidate(candidate: ComplexityCandidate | None) -> None:
        if candidate is None:
            return
        if candidate.complexity in seen_complexities:
            return
        seen_complexities.add(candidate.complexity)
        candidates.append(candidate)

    for candidate in _generator_curriculum_candidates(
        generator=generator,
        count=_reference_curve_initial_curriculum_count,
    ):
        append_candidate(candidate)
    for run in runs:
        if run.sampled_competence is None:
            continue
        for point in _as_sequence(
            run.sampled_competence.get("points", ()),
            "sampled_competence.points",
        ):
            point_record = _extract.mapping(point, "sampled_competence.points")
            append_candidate(
                _candidate_for_complexity(
                    generator=generator,
                    complexity=_as_nonnegative_number(
                        point_record.get("complexity"),
                        "sampled_competence.points.complexity",
                    ),
                )
            )
    return tuple(candidates)


def _generator_curriculum_candidates(
    *,
    generator: Any,
    count: int,
) -> tuple[ComplexityCandidate, ...]:
    if not hasattr(generator, "complexity_curriculum_candidates"):
        return ()
    candidates = generator.complexity_curriculum_candidates(start_index=0, count=count)
    return tuple(
        candidate
        for candidate in candidates
        if isinstance(candidate, ComplexityCandidate)
    )


def _oracle_inference_reference_points(
    *,
    generator: Any,
    runs: tuple[_BenchmarkRunRecord, ...],
) -> list[dict[str, object]]:
    reference_points: list[dict[str, object]] = []
    for candidate in _reference_curve_candidates(generator=generator, runs=runs):
        reference = _candidate_oracle_inference_compute(candidate)
        if reference is None:
            continue
        score = candidate.complexity
        reference_points.append(
            {
                "complexity": score,
                "score": score,
                "cost_density": cast(float, reference["value"]),
                "metadata": reference,
            }
        )
    return _integrated_reference_curve_points(reference_points)


def _integrated_reference_curve_points(
    points: list[dict[str, object]],
) -> list[dict[str, object]]:
    ordered = sorted(
        points,
        key=lambda point: (
            _as_nonnegative_number(point.get("complexity"), "reference_curve.complexity"),
            _as_nonnegative_number(point.get("cost_density"), "reference_curve.cost_density"),
        ),
    )
    integrated_points: list[dict[str, object]] = []
    previous_complexity = 0.0
    cumulative_cost = 0.0
    for point in ordered:
        complexity = _as_nonnegative_number(
            point.get("complexity"),
            "reference_curve.complexity",
        )
        if complexity < previous_complexity:
            raise LocalResultImportError("reference curve complexities must be ordered")
        cost_density = _as_nonnegative_number(
            point.get("cost_density"),
            "reference_curve.cost_density",
        )
        cumulative_cost += _integrated_compute_cost_from_points(
            points=(
                ComputeCostPoint(
                    complexity=complexity,
                    compute_per_sample=cost_density,
                    bit_length_per_op=_default_bit_length_per_op,
                    complexity_minimum=previous_complexity,
                    complexity_maximum=complexity,
                ),
            )
        )
        integrated_points.append(
            {
                "complexity": complexity,
                "score": _as_nonnegative_number(point.get("score"), "reference_curve.score"),
                "cost": cumulative_cost,
                "metadata": _extract.mapping(point.get("metadata"), "reference_curve.metadata"),
            }
        )
        previous_complexity = complexity
    return integrated_points


def _candidate_for_complexity(
    *,
    generator: Any,
    complexity: float,
) -> ComplexityCandidate | None:
    if not hasattr(generator, "complexity_candidate_for_request"):
        return None
    candidate = generator.complexity_candidate_for_request(
        request=ComplexityRequest(minimum=complexity, maximum=complexity)
    )
    return candidate if isinstance(candidate, ComplexityCandidate) else None


def _candidate_oracle_inference_compute(
    candidate: ComplexityCandidate,
) -> dict[str, object] | None:
    reference_value = candidate.metadata.get("oracle_inference_compute")
    if not isinstance(reference_value, Mapping):
        return None
    reference: dict[str, object] = {
        key: value
        for key, value in cast(Mapping[object, object], reference_value).items()
        if isinstance(key, str)
    }
    if reference.get("kind") != "oracle-inference-compute-reference-v1":
        return None
    value = reference.get("value")
    if not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        return None
    reference["value"] = float(value)
    return reference


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
) -> tuple[dict[str, object], ...]:
    grouped: dict[str, list[_BenchmarkRunRecord]] = {}
    for run in runs:
        grouped.setdefault(run.model_key, []).append(run)

    reference_baseline_complexity = _reference_baseline_complexity
    chance_mass = _chance_mass(manifest)
    records: list[dict[str, object]] = []
    for model_key, model_runs in grouped.items():
        ordered_runs = tuple(sorted(model_runs, key=_run_sort_key))
        points = _competence_points(ordered_runs)
        score = competent_complexity_score(
            points,
            chance_mass=chance_mass,
        )
        best_run = max(
            ordered_runs,
            key=lambda run: (run.score, -_cost_value(run.cost_summary, "storage_bytes")),
        )
        result_status = (
            "accepted"
            if any(run.result_status == "accepted" for run in ordered_runs)
            else "provisional"
        )
        score_basis = {
            "kind": "competence-integral-over-complexity-v1",
            "score_unit": "bits",
            "complexity_axis": "log2-distinguishable-states",
            "reference_baseline_complexity": reference_baseline_complexity,
            "chance_mass": chance_mass,
            "point_score": "accepted-mass",
            "local_competence": "above-chance-accepted-mass",
            "integration": "free-empty-rungs-then-first-observed-competence",
        }
        record: dict[str, object] = {
            "model_key": model_key,
            "result_status": result_status,
            "architecture_digest": str(best_run.architecture_digest),
            "benchmark_id": str(best_run.benchmark_id),
            "score": score,
            "score_basis": score_basis,
            "observed_complexities": [point["complexity"] for point in points],
            "points": list(points),
            "cost_summary": _model_cost_summary(
                ordered_runs,
                best_run=best_run,
                points=points,
            ),
            "cost_basis": _integrated_compute_cost_basis(),
            "run_ids": [run.run_id for run in ordered_runs],
            "measurement_count": sum(run.measurement_count for run in ordered_runs),
            "source_kinds": sorted({run.source_kind for run in ordered_runs}),
        }
        training_estimate_comparison = _training_estimate_comparison_record(
            run=best_run,
            accepted_points=points,
            accepted_score=score,
        )
        if training_estimate_comparison is not None:
            record["training_estimate_comparison"] = training_estimate_comparison
        record["console_view_model"] = _model_console_view_model(
            manifest=manifest,
            model=record,
            runs=ordered_runs,
            inspection=best_run.model_inspection,
        )
        records.append(record)
    return tuple(sorted(records, key=_model_sort_key))


def _model_cost_summary(
    runs: tuple[_BenchmarkRunRecord, ...],
    *,
    best_run: _BenchmarkRunRecord,
    points: tuple[dict[str, object], ...],
) -> dict[str, object]:
    cost_summary = _run_cost_summary(best_run)
    cost_summary.pop("parameter_count", None)
    cost_summary.pop("cost", None)
    cost_summary.pop("inference_compute", None)
    inference_compute = _model_measured_inference_compute(runs)
    if inference_compute is not None:
        cost_summary["inference_compute"] = inference_compute
        cost_summary["cost"] = _integrated_model_compute_cost_from_points(
            points=points,
            architecture=_run_architecture_manifest(best_run),
        )
    training_values = tuple(_run_training_compute_value(run) for run in runs)
    if any(value is None for value in training_values):
        raise LocalResultImportError(
            f"model {best_run.model_key} is missing reconstructible training compute"
        )
    cost_summary["training_compute"] = sum(cast(float, value) for value in training_values)
    return cost_summary


def _model_measured_inference_compute(
    runs: tuple[_BenchmarkRunRecord, ...],
) -> float | None:
    values = [
        value
        for run in runs
        if (value := _optional_cost_value(run.cost_summary, "inference_compute")) is not None
    ]
    if not values:
        return None
    return max(values)


def _run_architecture_manifest(run: _BenchmarkRunRecord) -> ArchitectureManifest:
    return ArchitectureManifest.from_record(run.architecture)


def _throughput_max_inference_compute(
    value: object,
    field_path: str,
    *,
    field: str = "max_inference_compute",
) -> float | None:
    if not isinstance(value, Mapping):
        return None
    record = cast(Mapping[str, object], value)
    if field not in record:
        return None
    return _as_nonnegative_number(record[field], field_path)


def _run_training_compute_value(run: _BenchmarkRunRecord) -> float | None:
    training_compute = _run_training_compute(run)
    if training_compute is not None:
        return training_compute
    return _optional_cost_value(run.cost_summary, "training_compute")


def _run_cost_summary(run: _BenchmarkRunRecord) -> dict[str, object]:
    cost_summary = dict(run.cost_summary)
    cost_summary.pop("parameter_count", None)
    cost_summary.pop("training_compute_per_sample", None)
    cost_summary.pop("cost", None)
    if run.source_kind in {"local-run", "local-training-estimate"}:
        inference_compute = _optional_cost_value(cost_summary, "inference_compute")
        if inference_compute is not None:
            cost_summary["inference_compute"] = inference_compute
            cost_summary["cost"] = _integrated_model_compute_cost_from_points(
                points=_run_competence_points(run),
                architecture=_run_architecture_manifest(run),
            )
    else:
        cost_summary.pop("inference_compute", None)
    training_compute = _run_training_compute(run)
    if training_compute is not None:
        cost_summary["training_compute"] = training_compute
    return cost_summary


def _integrated_compute_cost_from_points(
    *,
    points: tuple[ComputeCostPoint, ...],
) -> float:
    return integrated_compute_cost(points)


def _integrated_model_compute_cost_from_points(
    *,
    points: tuple[dict[str, object], ...],
    architecture: ArchitectureManifest,
) -> float:
    return integrated_compute_cost(
        _compute_cost_points_from_sampled_competence(
            points=points,
            architecture=architecture,
        )
    )


def _compute_cost_points_from_sampled_competence(
    *,
    points: tuple[dict[str, object], ...],
    architecture: ArchitectureManifest,
) -> tuple[ComputeCostPoint, ...]:
    ordered = sorted(points, key=lambda point: _point_complexity(point))
    cursor = 0.0
    cost_points: list[ComputeCostPoint] = []
    for point in ordered:
        complexity = _point_complexity(point)
        minimum = _optional_nonnegative_number(
            point.get("complexity_minimum"),
            "compute_cost_point.complexity_minimum",
        )
        maximum = _optional_nonnegative_number(
            point.get("complexity_maximum"),
            "compute_cost_point.complexity_maximum",
        )
        if minimum is None:
            minimum = cursor
        if maximum is None:
            maximum = complexity
        if maximum <= minimum:
            minimum = cursor
            maximum = complexity
        cost_points.append(
            ComputeCostPoint(
                complexity=complexity,
                compute_per_sample=_point_inference_compute(
                    point,
                    architecture=architecture,
                ),
                bit_length_per_op=_default_bit_length_per_op,
                complexity_minimum=minimum,
                complexity_maximum=maximum,
            )
        )
        cursor = max(cursor, maximum)
    return tuple(cost_points)


def _point_inference_compute(
    point: Mapping[str, object],
    *,
    architecture: ArchitectureManifest,
) -> float:
    plan = summarize_architecture_operators(
        architecture_with_input_shape(architecture, _point_input_shape(point))
    )
    if plan.inference_compute is None:
        raise LocalResultImportError("compute_cost_point input_shape has unknown inference compute")
    return float(plan.inference_compute)


def _point_input_shape(point: Mapping[str, object]) -> tuple[int, ...]:
    value = point.get("input_shape")
    if not isinstance(value, list | tuple):
        raise LocalResultImportError("compute_cost_point.input_shape: expected shape")
    shape: list[int] = []
    for axis in cast(Sequence[object], value):
        if type(axis) is not int or axis < 1:
            raise LocalResultImportError(
                "compute_cost_point.input_shape: expected positive integer shape"
            )
        shape.append(axis)
    return tuple(shape)


def _optional_point_input_shape(
    point: Mapping[str, object],
    field: str,
) -> tuple[int, ...] | None:
    if "input_shape" not in point:
        return None
    value = point.get("input_shape")
    if not isinstance(value, list | tuple):
        raise LocalResultImportError(f"{field}: expected shape")
    shape: list[int] = []
    for axis in cast(Sequence[object], value):
        if type(axis) is not int or axis < 1:
            raise LocalResultImportError(f"{field}: expected positive integer shape")
        shape.append(axis)
    return tuple(shape)


def _integrated_compute_cost_basis() -> dict[str, object]:
    return {
        "kind": "compute-integral-over-observed-complexity-v1",
        "cost_unit": "bit-ops-per-sample complexity-bits",
        "complexity_axis": "log2-distinguishable-states",
        "density_source": "architecture+sampled_competence.points.input_shape",
        "density_unit": "ops-per-sample",
        "bit_length_per_op": _default_bit_length_per_op,
        "integration": "observed-complexity-intervals",
        "competence_weighted": False,
    }


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
        *_model_training_estimate_comparison_sections(model),
        _console_detail_entries_section(
            title="Resources",
            entries=(
                ("Cost", _console_number_value(cost_summary.get("cost"))),
                ("Model Size", _console_number_value(cost_summary.get("storage_bytes"))),
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


def _model_training_estimate_comparison_sections(
    model: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    comparison = _extract.optional_mapping(
        model.get("training_estimate_comparison"),
        "model.training_estimate_comparison",
    )
    if comparison is None:
        return ()
    points = _as_sequence(
        comparison.get("points"),
        "model.training_estimate_comparison.points",
    )
    return (
        _console_detail_entries_section(
            title="Training Estimate",
            entries=(
                (
                    "Training Score",
                    _console_number_value(comparison.get("training_score"), precision=4),
                ),
                (
                    "Accepted Score",
                    _console_number_value(comparison.get("accepted_score"), precision=4),
                ),
                (
                    "Delta",
                    _console_number_value(comparison.get("score_delta"), precision=4),
                ),
                (
                    "Matched Rungs",
                    (
                        f"{_console_number_value(comparison.get('matched_point_count'))}"
                        f" / {_console_number_value(comparison.get('point_count'))}"
                    ),
                ),
            ),
        ),
        {
            "title": "Training Estimate Rungs",
            "table": {
                "aria_label": "Training estimate compared with accepted evaluation by rung",
                "columns": [
                    "Range",
                    "Training",
                    "Accepted",
                    "Delta",
                    "Samples",
                    "Status",
                ],
                "rows": [
                    _console_training_estimate_comparison_row(
                        _extract.mapping(
                            point,
                            "model.training_estimate_comparison.points",
                        )
                    )
                    for point in points
                ],
            },
        },
    )


def _console_training_estimate_comparison_row(
    point: Mapping[str, object],
) -> list[str]:
    training_sample_count = point.get("training_sample_count")
    accepted_sample_count = point.get("accepted_sample_count")
    return [
        _console_interval_label(point),
        _console_number_value(point.get("training_score"), precision=4),
        _console_number_value(point.get("accepted_score"), precision=4),
        _console_number_value(point.get("score_delta"), precision=4),
        (
            f"{_console_number_value(training_sample_count)}"
            f" / {_console_number_value(accepted_sample_count)}"
        ),
        _console_string_value(point.get("status")),
    ]


def _console_interval_label(point: Mapping[str, object]) -> str:
    minimum = _optional_nonnegative_number(
        point.get("complexity_minimum"),
        "training_estimate_comparison.point.complexity_minimum",
    )
    maximum = _optional_nonnegative_number(
        point.get("complexity_maximum"),
        "training_estimate_comparison.point.complexity_maximum",
    )
    if minimum is None or maximum is None:
        return _console_number_value(point.get("complexity"), precision=2)
    return (
        f"[{_console_number_value(minimum, precision=2)}, "
        f"{_console_number_value(maximum, precision=2)}]"
    )


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
    by_interval: dict[
        tuple[float, float | None, float | None],
        list[tuple[_BenchmarkRunRecord, float, int, tuple[int, ...] | None]],
    ] = {}
    for run in runs:
        for point in _run_competence_points(run):
            complexity = _as_nonnegative_number(point.get("complexity"), "point.complexity")
            minimum = _optional_nonnegative_number(
                point.get("complexity_minimum"),
                "point.complexity_minimum",
            )
            maximum = _optional_nonnegative_number(
                point.get("complexity_maximum"),
                "point.complexity_maximum",
            )
            score = _as_nonnegative_number(point.get("score"), "point.score")
            sample_count = _as_positive_int(point.get("sample_count"), "point.sample_count")
            input_shape = _optional_point_input_shape(point, "point.input_shape")
            by_interval.setdefault((complexity, minimum, maximum), []).append(
                (run, score, sample_count, input_shape)
            )
    points: list[dict[str, object]] = []
    for (complexity, minimum, maximum), evidence in by_interval.items():
        total_samples = sum(
            sample_count for _run, _score, sample_count, _input_shape in evidence
        )
        score = (
            sum(
                score * sample_count
                for _run, score, sample_count, _input_shape in evidence
            )
            / total_samples
        )
        point: dict[str, object] = {
            "complexity": complexity,
            "score": score,
            "sample_count": total_samples,
            "run_ids": [
                run.run_id
                for run in sorted(
                    {
                        run.run_id: run
                        for run, _score, _sample_count, _input_shape in evidence
                    }.values(),
                    key=_run_sort_key,
                )
            ],
        }
        input_shapes = {
            input_shape for _run, _score, _sample_count, input_shape in evidence
            if input_shape is not None
        }
        if len(input_shapes) > 1:
            raise LocalResultImportError(
                "cannot merge competence points with different input_shape values"
            )
        if input_shapes:
            point["input_shape"] = list(next(iter(input_shapes)))
        if minimum is not None:
            point["complexity_minimum"] = minimum
        if maximum is not None:
            point["complexity_maximum"] = maximum
        points.append(point)
    return tuple(sorted(points, key=_point_complexity))


def _run_competence_points(run: _BenchmarkRunRecord) -> tuple[dict[str, object], ...]:
    if run.sampled_competence is not None:
        points = run.sampled_competence.get("points")
        if isinstance(points, list | tuple):
            return tuple(
                _competence_point_from_sampled_record(point)
                for point in (
                    _extract.mapping(value, "sampled_competence.points")
                    for value in _as_sequence(
                        cast(object, points),
                        "sampled_competence.points",
                    )
                )
            )
        return (_competence_point_from_sampled_record(run.sampled_competence),)
    if run.complexity is None:
        return ()
    return (
        {
            "complexity": run.complexity,
            "score": run.score,
            "sample_count": run.measurement_count,
        },
    )


def _competence_point_from_sampled_record(point: Mapping[str, object]) -> dict[str, object]:
    record: dict[str, object] = {
        "complexity": _as_nonnegative_number(
            point.get("complexity"),
            "sampled_competence.point.complexity",
        ),
        "score": _as_nonnegative_number(
            point.get("mean_accepted_mass"),
            "sampled_competence.point.mean_accepted_mass",
        ),
        "sample_count": _as_positive_int(
            point.get("sample_count"),
            "sampled_competence.point.sample_count",
        ),
    }
    minimum = _optional_nonnegative_number(
        point.get("complexity_minimum"),
        "sampled_competence.point.complexity_minimum",
    )
    maximum = _optional_nonnegative_number(
        point.get("complexity_maximum"),
        "sampled_competence.point.complexity_maximum",
    )
    input_shape = _optional_point_input_shape(
        point,
        "sampled_competence.point.input_shape",
    )
    if input_shape is not None:
        record["input_shape"] = list(input_shape)
    if minimum is not None:
        record["complexity_minimum"] = minimum
    if maximum is not None:
        record["complexity_maximum"] = maximum
    return record


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
                complexity_minimum=_optional_nonnegative_number(
                    point.get("complexity_minimum"),
                    "competence_point.complexity_minimum",
                ),
                complexity_maximum=_optional_nonnegative_number(
                    point.get("complexity_maximum"),
                    "competence_point.complexity_maximum",
                ),
            )
            for point in points
        ),
        chance_mass=chance_mass,
    )


def _training_estimate_comparison_record(
    *,
    run: _BenchmarkRunRecord,
    accepted_points: tuple[dict[str, object], ...],
    accepted_score: float,
) -> dict[str, object] | None:
    if run.result_status != "accepted" or run.training_summary is None:
        return None
    estimate = _selected_checkpoint_training_estimate(run.training_summary)
    if estimate is None:
        return None
    sampled_competence = _extract.mapping(
        estimate.get("sampled_competence"),
        "training_estimate.sampled_competence",
    )
    training_points = _training_estimate_competence_points(sampled_competence)
    training_score = _as_nonnegative_number(
        estimate.get("score"),
        "training_estimate.score",
    )
    accepted_by_interval = {
        _comparison_interval_key(point): point for point in accepted_points
    }
    training_by_interval = {
        _comparison_interval_key(point): point for point in training_points
    }
    comparison_points: list[dict[str, object]] = []
    for key in sorted(
        {*accepted_by_interval, *training_by_interval},
        key=_comparison_interval_sort_key,
    ):
        accepted_point = accepted_by_interval.get(key)
        training_point = training_by_interval.get(key)
        comparison_point: dict[str, object] = {
            "complexity": _comparison_interval_complexity(
                accepted_point if accepted_point is not None else training_point
            ),
            "status": _comparison_point_status(
                accepted_point=accepted_point,
                training_point=training_point,
            ),
        }
        _copy_optional_interval_fields(
            comparison_point,
            accepted_point if accepted_point is not None else training_point,
        )
        if accepted_point is not None:
            comparison_point["accepted_score"] = _point_score(accepted_point)
            comparison_point["accepted_sample_count"] = _point_sample_count(accepted_point)
        if training_point is not None:
            comparison_point["training_score"] = _point_score(training_point)
            comparison_point["training_sample_count"] = _point_sample_count(training_point)
        if accepted_point is not None and training_point is not None:
            comparison_point["score_delta"] = (
                _point_score(training_point) - _point_score(accepted_point)
            )
        comparison_points.append(comparison_point)
    accepted_sample_count = sum(_point_sample_count(point) for point in accepted_points)
    training_sample_count = sum(_point_sample_count(point) for point in training_points)
    return {
        "kind": "training-vs-accepted-sampled-competence-v1",
        "accepted_score": accepted_score,
        "training_score": training_score,
        "score_delta": training_score - accepted_score,
        "accepted_sample_count": accepted_sample_count,
        "training_sample_count": training_sample_count,
        "point_count": len(comparison_points),
        "matched_point_count": sum(
            1 for point in comparison_points if point["status"] == "matched"
        ),
        "points": comparison_points,
    }


def _selected_checkpoint_training_estimate(
    training_summary: Mapping[str, object],
) -> Mapping[str, object] | None:
    selected_estimate = _extract.optional_mapping(
        training_summary.get("selected_model_checkpoint_score_estimate"),
        "selected_model_checkpoint_score_estimate",
    )
    if selected_estimate is not None:
        return selected_estimate
    return _extract.optional_mapping(
        training_summary.get("training_estimate"),
        "training_estimate",
    )


def _training_estimate_competence_points(
    sampled_competence: Mapping[str, object],
) -> tuple[dict[str, object], ...]:
    points = sampled_competence.get("points")
    if not isinstance(points, list | tuple):
        return (_competence_point_from_sampled_record(sampled_competence),)
    return tuple(
        _competence_point_from_sampled_record(point)
        for point in (
            _extract.mapping(value, "training_estimate.sampled_competence.points")
            for value in _as_sequence(
                cast(object, points),
                "training_estimate.sampled_competence.points",
            )
        )
    )


def _comparison_interval_key(point: Mapping[str, object]) -> tuple[float, float]:
    complexity = _point_complexity(point)
    minimum = _optional_nonnegative_number(
        point.get("complexity_minimum"),
        "competence_point.complexity_minimum",
    )
    maximum = _optional_nonnegative_number(
        point.get("complexity_maximum"),
        "competence_point.complexity_maximum",
    )
    return (
        minimum if minimum is not None else complexity,
        maximum if maximum is not None else complexity,
    )


def _comparison_interval_sort_key(key: tuple[float, float]) -> tuple[float, float]:
    return key


def _comparison_interval_complexity(point: Mapping[str, object] | None) -> float:
    if point is None:
        raise LocalResultImportError("training comparison point is missing")
    return _point_complexity(point)


def _copy_optional_interval_fields(
    target: dict[str, object],
    point: Mapping[str, object] | None,
) -> None:
    if point is None:
        return
    minimum = _optional_nonnegative_number(
        point.get("complexity_minimum"),
        "competence_point.complexity_minimum",
    )
    maximum = _optional_nonnegative_number(
        point.get("complexity_maximum"),
        "competence_point.complexity_maximum",
    )
    if minimum is not None:
        target["complexity_minimum"] = minimum
    if maximum is not None:
        target["complexity_maximum"] = maximum


def _comparison_point_status(
    *,
    accepted_point: Mapping[str, object] | None,
    training_point: Mapping[str, object] | None,
) -> str:
    if accepted_point is not None and training_point is not None:
        return "matched"
    if accepted_point is not None:
        return "accepted-only"
    return "training-only"


def _point_sample_count(point: Mapping[str, object]) -> int:
    return _as_positive_int(point.get("sample_count"), "competence_point.sample_count")


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
        (
            model
            for model in models
            if _optional_cost_value(
                _extract.mapping(model["cost_summary"], "cost_summary"),
                cost_axis,
            )
            is not None
        ),
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
        _cost_value(cost_summary, "storage_bytes"),
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
    selected_remote = _select_result_remote(
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
    selected_remote = _select_result_remote(
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


def _select_result_remote(
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
        "no result remote is available: provide --repo with Hugging Face auth "
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
        for path in _result_upload_files(results_root)
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


def _result_upload_files(results_root: Path) -> tuple[Path, ...]:
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


def _ensure_result_checkout_structure(results_root: Path) -> None:
    readme = results_root / "README.md"
    if not readme.exists():
        readme.write_text(
            "---\n"
            "license: cc0-1.0\n"
            "---\n\n"
            "# Leibniz Result Checkout\n\n"
            "This dataset repository stores Leibniz benchmark result state.\n",
            encoding="utf-8",
        )
    for directory in _result_directories:
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
            "leaderboard",
            "model_candidates",
            "frontiers",
            "reference_curves",
            "training_history",
            "plot_runs",
            "model_inspections",
        },
        prefix="benchmark_results",
    )
    _extract.non_empty_string(record.get("benchmark_id"), "benchmark_id")
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
    for index, curve in enumerate(
        _as_sequence(record.get("reference_curves", ()), "reference_curves")
    ):
        _validate_reference_curve(
            _extract.mapping(curve, f"reference_curves.{index}"),
            f"reference_curves.{index}",
        )
    for index, run in enumerate(_record_sequence(record, "training_history")):
        _validate_run_result(run, f"training_history.{index}")
    for index, run in enumerate(_record_sequence(record, "plot_runs")):
        _validate_run_result(run, f"plot_runs.{index}")
    inspections = _as_sequence(record.get("model_inspections", ()), "model_inspections")
    for index, inspection in enumerate(inspections):
        field = f"model_inspections.{index}"
        inspection_record = dict(_extract.mapping(inspection, field))
        try:
            _require_string_fields(
                inspection_record,
                field,
                ("id", "source_path"),
            )
            _require_mapping_fields(
                inspection_record,
                field,
                (
                    "architecture",
                    "architecture_graph",
                    "architecture_summary",
                    "architecture_trace",
                    "cost_summary",
                ),
            )
            _require_sequence_fields(inspection_record, field, ("components", "node_evidence"))
        except LocalResultImportError as error:
            raise LocalResultImportError(
                f"{field}: invalid model inspection: {error}"
            ) from error


def _validate_reference_curve(record: Mapping[str, object], field: str) -> None:
    _require_string_fields(
        record,
        field,
        ("kind", "key", "label", "x_axis", "y_axis"),
    )
    if record.get("kind") != "oracle-inference-compute-reference-v1":
        raise LocalResultImportError(f"{field}.kind is invalid")
    points = _as_sequence(record.get("points"), f"{field}.points")
    for index, point in enumerate(points):
        point_field = f"{field}.points.{index}"
        point_record = _extract.mapping(point, point_field)
        _as_nonnegative_number(point_record.get("complexity"), f"{point_field}.complexity")
        _as_nonnegative_number(point_record.get("score"), f"{point_field}.score")
        _as_nonnegative_number(point_record.get("cost"), f"{point_field}.cost")
        if "metadata" in point_record:
            _extract.mapping(point_record["metadata"], f"{point_field}.metadata")


def _validate_model_result(record: Mapping[str, object], prefix: str) -> None:
    _require_string_fields(
        record,
        prefix,
        ("model_key", "result_status", "architecture_digest", "benchmark_id"),
    )
    if record.get("result_status") not in {"accepted", "provisional"}:
        raise LocalResultImportError(f"{_field_path(prefix, 'result_status')} is invalid")
    _as_nonnegative_number(record.get("score"), _field_path(prefix, "score"))
    _require_sequence_fields(
        record,
        prefix,
        ("observed_complexities", "points", "run_ids", "source_kinds"),
    )
    _require_mapping_fields(record, prefix, ("cost_summary",))
    if "cost_basis" in record:
        _extract.mapping(record["cost_basis"], _field_path(prefix, "cost_basis"))
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
    if "training_estimate_comparison" in record:
        _validate_training_estimate_comparison(
            _extract.mapping(
                record["training_estimate_comparison"],
                _field_path(prefix, "training_estimate_comparison"),
            ),
            _field_path(prefix, "training_estimate_comparison"),
        )


def _validate_training_estimate_comparison(
    record: Mapping[str, object],
    prefix: str,
) -> None:
    _require_string_fields(record, prefix, ("kind",))
    if record.get("kind") != "training-vs-accepted-sampled-competence-v1":
        raise LocalResultImportError(f"{_field_path(prefix, 'kind')} is invalid")
    for field in (
        "accepted_score",
        "training_score",
        "accepted_sample_count",
        "training_sample_count",
        "point_count",
        "matched_point_count",
    ):
        _as_nonnegative_number(record.get(field), _field_path(prefix, field))
    _as_finite_number(record.get("score_delta"), _field_path(prefix, "score_delta"))
    points = _as_sequence(record.get("points"), _field_path(prefix, "points"))
    for index, point in enumerate(points):
        point_path = f"{prefix}.points.{index}"
        point_record = _extract.mapping(point, point_path)
        _extract.non_empty_string(point_record.get("status"), _field_path(point_path, "status"))
        if point_record.get("status") not in {
            "matched",
            "accepted-only",
            "training-only",
        }:
            raise LocalResultImportError(f"{_field_path(point_path, 'status')} is invalid")
        _as_nonnegative_number(
            point_record.get("complexity"),
            _field_path(point_path, "complexity"),
        )
        for optional_number in (
            "complexity_minimum",
            "complexity_maximum",
            "accepted_score",
            "training_score",
            "accepted_sample_count",
            "training_sample_count",
        ):
            if optional_number in point_record:
                _as_nonnegative_number(
                    point_record[optional_number],
                    _field_path(point_path, optional_number),
                )
        if "score_delta" in point_record:
            _as_finite_number(
                point_record["score_delta"],
                _field_path(point_path, "score_delta"),
            )


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
    if record.get("result_status") not in {"accepted", "provisional"}:
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
    if "validation_loss_reference" in record:
        _as_nonnegative_number(
            record.get("validation_loss_reference"),
            _field_path(prefix, "validation_loss_reference"),
        )
    if "validation_history_sample_count" in record:
        _as_positive_int(
            record.get("validation_history_sample_count"),
            _field_path(prefix, "validation_history_sample_count"),
        )
    if "validation_history_total_count" in record:
        _as_positive_int(
            record.get("validation_history_total_count"),
            _field_path(prefix, "validation_history_total_count"),
        )
    protocol = _extract.mapping(record.get("protocol"), _field_path(prefix, "protocol"))
    protocol_path = _field_path(prefix, "protocol")
    _require_string_fields(
        protocol,
        protocol_path,
        ("kind", "objective", "optimizer", "schedule", "validation_source"),
    )
    if "learning_rate" in protocol:
        _as_nonnegative_number(
            protocol.get("learning_rate"),
            f"{prefix}.protocol.learning_rate",
        )
    _as_nonnegative_number(protocol.get("min_delta"), f"{prefix}.protocol.min_delta")
    protocol_positive_ints = (
        "seed",
        "training_batch_target",
        "gate_check_interval",
        "gate_batch_target",
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


def _as_nonnegative_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LocalResultImportError(f"{field}: expected number")
    numeric = float(value)
    if numeric < 0 or not math.isfinite(numeric):
        raise LocalResultImportError(f"{field}: expected finite nonnegative number")
    return numeric


def _as_finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LocalResultImportError(f"{field}: expected number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise LocalResultImportError(f"{field}: expected finite number")
    return numeric


def _optional_nonnegative_number(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _as_nonnegative_number(value, field)


def _as_probability(value: object, field: str) -> float:
    numeric = _as_nonnegative_number(value, field)
    if numeric > 1.0:
        raise LocalResultImportError(f"{field}: expected probability no greater than 1")
    return numeric


def _as_positive_int(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise LocalResultImportError(f"{field}: expected positive integer")
    return value
