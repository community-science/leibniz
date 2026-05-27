"""Operator-local result import and console view materialization."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from leibniz.architectures import ArchitectureManifest
from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestDocument
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
from leibniz.model_inspection import ModelInspectionDocument, ModelInspectionRecord
from leibniz.publications import SubmissionPublicationDocument
from leibniz.training_runs import TrainingRunRecord

__all__ = [
    "LocalBenchmarkResultViewSummary",
    "LocalResultImportError",
    "LocalResultImportSummary",
    "import_submission_publications",
    "load_console_result_view",
    "materialize_benchmark_result_views",
]

_console_result_view_format = "leibniz.console.imported-results"
_benchmark_result_view_format = "leibniz.console.benchmark-results"
_work_queue_view_format = "leibniz.console.work-queue"
_console_result_view_format_version = 1
_document_suffix = document_filename_suffix()
_manifest_filename = "manifest" + _document_suffix


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


def import_submission_publications(
    source_roots: Iterable[Path],
    *,
    repository_root: Path | None = None,
    runs_root: Path = Path(".runs"),
) -> LocalResultImportSummary:
    """Import local publication bundles into ignored run state and console views."""

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
    imported_runs = _imported_run_records(runs_root)
    runs = tuple(sorted((*local_runs, *imported_runs), key=_run_sort_key))
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

    def to_record(self) -> dict[str, object]:
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


def _inspection_from_architecture(architecture: ArchitectureManifest) -> ModelInspectionRecord:
    return ModelInspectionRecord.from_architecture(
        id=ProtocolIdentifier.parse(
            "model-inspections.imported."
            f"{str(architecture.digest).split(':', maxsplit=1)[1][:16]}@0.1.0"
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
        "training_history": [run.to_record() for run in runs],
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


def _validate_training_diagnostics(record: Mapping[str, object]) -> None:
    status = _as_string(record.get("status"), "training_diagnostics.status")
    if status not in {
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
