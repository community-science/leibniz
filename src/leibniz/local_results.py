"""Operator-local result import and console view materialization."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.content import ContentDigest
from leibniz.documents import (
    canonical_document_bytes,
    document_filename_suffix,
    load_object_document,
)
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementRecord
from leibniz.publications import SubmissionPublicationDocument

__all__ = [
    "LocalResultImportError",
    "LocalResultImportSummary",
    "import_submission_publications",
    "load_console_result_view",
]

_console_result_view_format = "leibniz.console.imported-results"
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


def load_console_result_view(data: bytes) -> Mapping[str, object]:
    """Load a generated console result view document."""

    try:
        record = load_object_document(data, description="console result view")
    except ValueError as error:
        raise LocalResultImportError(str(error)) from error
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
    benchmark_root = repository_root / "src" / "leibniz" / "benchmarks"
    ids: set[ProtocolIdentifier] = set()
    for path in sorted(benchmark_root.rglob(_manifest_filename)):
        document = BenchmarkManifestDocument.from_bytes(path.read_bytes())
        ids.add(document.manifest.id)
    if not ids:
        raise LocalResultImportError("no known benchmark manifests found")
    return frozenset(ids)


def _validate_known_benchmarks(
    measurements: tuple[MeasurementRecord, ...],
    known_benchmark_ids: frozenset[ProtocolIdentifier],
) -> None:
    for measurement in measurements:
        if measurement.benchmark_id not in known_benchmark_ids:
            raise LocalResultImportError(
                f"unknown benchmark id in imported measurement: {measurement.benchmark_id}"
            )


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
