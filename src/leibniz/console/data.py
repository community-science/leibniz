"""Build generated data payloads for the browser console."""

from __future__ import annotations

import argparse
import ast
import hashlib
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from leibniz.benchmark_implementations import (
    Generator as BenchmarkGenerator,
)
from leibniz.benchmark_implementations import (
    discover_benchmark_roots,
)
from leibniz.benchmarks import BenchmarkManifest
from leibniz.console.artifact_index import (
    ConsoleArtifactIndex,
    ConsoleArtifactIndexBuilder,
    ConsoleArtifactIndexEntry,
    ConsoleArtifactIndexSource,
    ConsoleArtifactIndexValidationError,
)
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
from leibniz.identifiers import ProtocolIdentifier, ProtocolName, SemanticVersion
from leibniz.local_results import (
    LocalResultImportError,
    load_console_result_view,
    materialize_benchmark_result_views,
)
from leibniz.model_inspection import ModelInspectionRecord
from leibniz.observation_generation import (
    StateSpaceVolumeRequest,
    load_generator,
    sample_indices_for_even_state_coverage,
)
from leibniz.program_graphs import ProgramGraphSpec

__all__ = [
    "ConsoleData",
    "ConsoleDataBuilder",
    "ConsoleDataValidationError",
]

_protocol_formats = console_protocol_formats()
_protocol_format_versions = console_protocol_format_versions()
_format = _protocol_formats.console_data
_format_version = _protocol_format_versions.console_data
_document_suffix = document_filename_suffix()
_generated_batch_cache_format = "leibniz.console.generated-sample-set-cache"
_generated_batch_cache_format_version = 2
_generated_batch_cache_path = (
    Path(__file__).parents[3]
    / ".local-cache"
    / "console"
    / ("generatedSampleSets" + _document_suffix)
)
_generated_batch_cache: dict[tuple[str, str, str], tuple[Mapping[str, object], ...]] = {}
_generated_preview_sample_limit = 50
_generated_preview_volume_windows = (
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.0),
    (3.0, 4.0),
    (4.0, 5.0),
    (5.0, 6.0),
    (6.0, 7.0),
    (7.0, 8.0),
    (8.0, 9.0),
)


@dataclass(frozen=True, slots=True)
class _PreviewWindow:
    label: str
    seed: int
    volume_request: StateSpaceVolumeRequest
    state_count: int


def _preview_windows_for_generator(
    generator: BenchmarkGenerator,
) -> tuple[_PreviewWindow, ...]:
    candidates = _preview_window_candidates(generator)
    return tuple(
        _PreviewWindow(
            label=window.label,
            seed=401 + index,
            volume_request=window.volume_request,
            state_count=window.state_count,
        )
        for index, window in enumerate(candidates)
    )


def _preview_window_candidates(
    generator: BenchmarkGenerator,
) -> tuple[_PreviewWindow, ...]:
    candidates: list[_PreviewWindow] = []
    seen_state_counts: set[int] = set()
    for minimum, maximum in _generated_preview_volume_windows:
        request = StateSpaceVolumeRequest(minimum=minimum, maximum=maximum)
        sample_set = generator(
            seed=401,
            shape=1,
            include_metadata=True,
            volume_request=request,
        )
        if not sample_set.samples:
            continue
        state_count = _state_count_from_log2_volume(sample_set.log2_volume)
        if state_count in seen_state_counts:
            continue
        seen_state_counts.add(state_count)
        candidates.append(
            _PreviewWindow(
                label=f"[{minimum:g}, {maximum:g}]",
                seed=401,
                volume_request=request,
                state_count=state_count,
            )
        )
    return tuple(candidates)


def _state_count_from_log2_volume(log2_volume: float) -> int:
    state_count = round(2**log2_volume)
    if state_count < 1:
        raise ConsoleDataValidationError("generated sample state count must be positive")
    return state_count


def _preview_batch_record(
    *,
    generator: BenchmarkGenerator,
    window: _PreviewWindow,
) -> Mapping[str, object]:
    sample_indices = sample_indices_for_even_state_coverage(
        state_count=window.state_count,
        seed=window.seed,
        sample_limit=_generated_preview_sample_limit,
    )
    sample_set = generator(
        seed=window.seed,
        shape=len(sample_indices),
        include_fields=True,
        include_artifacts=True,
        volume_request=window.volume_request,
        sample_indices=sample_indices,
    )
    samples = [sample.to_record() for sample in sample_set.samples]
    record: dict[str, object] = {
        "mode": "volume-window",
        "label": window.label,
        "seed": window.seed,
        "sample_count": len(samples),
        "volume_window": window.volume_request.to_record(),
        "volumes": [window.state_count],
        "presentation": {
            "sample_card_density": "compact" if len(samples) > 80 else "standard",
            "aggregate_mode": False,
        },
        "samples": samples,
    }
    if sample_set.region is not None:
        record["region"] = sample_set.region.to_record()
    if sample_set.request_outcome is not None:
        record["request_outcome"] = sample_set.request_outcome.to_record()
    return record


class ConsoleDataValidationError(ValueError):
    """Raised when console data cannot be discovered or generated."""


@dataclass(frozen=True, slots=True)
class ConsoleData:
    """A generated console data payload for the browser."""

    artifact_index: ConsoleArtifactIndex
    artifact_details: tuple[Mapping[str, object], ...]
    result_views: tuple[Mapping[str, object], ...]
    model_inspections: tuple[Mapping[str, object], ...]
    benchmark_tasks: tuple[Mapping[str, object], ...]
    operator_vocabulary: Mapping[str, object]

    def to_record(self) -> dict[str, object]:
        return {
            "format": _format,
            "format_version": _format_version,
            "artifact_index": self.artifact_index.to_record(),
            "artifact_details": list(self.artifact_details),
            "result_views": list(self.result_views),
            "model_inspections": list(self.model_inspections),
            "benchmark_tasks": list(self.benchmark_tasks),
            "operator_vocabulary": dict(self.operator_vocabulary),
        }

    def to_bytes(self) -> bytes:
        return canonical_document_bytes(self.to_record()) + b"\n"


class ConsoleDataBuilder:
    """Discover supported public documents and build a console data payload."""

    def __init__(self, repository_root: Path) -> None:
        self._repository_root = repository_root.resolve()
        self._artifact_builder = ConsoleArtifactIndexBuilder(self._repository_root)

    def discover(
        self,
        roots: Iterable[PurePosixPath],
        *,
        result_roots: Iterable[Path] = (),
    ) -> ConsoleData:
        sources = tuple(self._discover_sources(tuple(roots)))
        artifact_index = self._artifact_builder.build(sources)
        details = tuple(self._detail_for_source(source) for source in artifact_index.entries)
        result_views = tuple(self._result_views(tuple(result_roots)))
        model_inspections = tuple(self._model_inspections(artifact_index.entries))
        benchmark_tasks = tuple(self._benchmark_tasks(self._benchmark_roots(tuple(roots))))
        return ConsoleData(
            artifact_index=artifact_index,
            artifact_details=details,
            result_views=result_views,
            model_inspections=model_inspections,
            benchmark_tasks=benchmark_tasks,
            operator_vocabulary={},
        )

    def _discover_sources(
        self,
        roots: tuple[PurePosixPath, ...],
    ) -> tuple[ConsoleArtifactIndexSource, ...]:
        if not roots:
            raise ConsoleDataValidationError("at least one public root is required")

        sources: list[ConsoleArtifactIndexSource] = []
        seen_paths: set[PurePosixPath] = set()
        for root in roots:
            root_path = self._repository_path(root, description="public root")
            if not root_path.is_dir():
                raise ConsoleDataValidationError(f"public root does not name a directory: {root}")
            for path in sorted(item for item in root_path.rglob("*") if item.is_file()):
                source_path = PurePosixPath(path.relative_to(self._repository_root).as_posix())
                if source_path in seen_paths:
                    continue
                source = self._source_for_path(source_path)
                if source is not None:
                    sources.append(source)
                    seen_paths.add(source_path)

        if not sources:
            raise ConsoleDataValidationError("public roots did not contain supported documents")
        return tuple(sources)

    def _source_for_path(self, source_path: PurePosixPath) -> ConsoleArtifactIndexSource | None:
        data = self._repository_path(source_path, description="source document").read_bytes()
        matches: list[str] = []
        for kind in ConsoleArtifactIndexBuilder.supported_kinds():
            try:
                ConsoleArtifactIndexBuilder.load_supported_artifact(kind, data)
            except ValueError:
                continue
            matches.append(kind)

        if len(matches) > 1:
            kinds = ", ".join(matches)
            raise ConsoleDataValidationError(
                f"ambiguous supported document kind for {source_path}: {kinds}"
            )
        if not matches:
            return None
        return ConsoleArtifactIndexSource(kind=matches[0], source_path=source_path)

    def _detail_for_source(self, source: ConsoleArtifactIndexEntry) -> Mapping[str, object]:
        path = self._repository_path(source.source_path, description="source document")
        loaded = ConsoleArtifactIndexBuilder.load_supported_artifact(
            source.kind,
            path.read_bytes(),
        )
        _protocol_id, record, _digest, _dependencies = loaded
        summary = self._detail_summary(source.kind, record)
        return {
            "kind": source.kind,
            "source_path": source.source_path.as_posix(),
            **summary,
        }

    def _detail_summary(
        self,
        kind: str,
        record: Mapping[str, object],
    ) -> Mapping[str, object]:
        if kind == "program-graph":
            return {
                "contract_kind": record["contract_kind"],
                "inputs": record["inputs"],
                "outputs": record["outputs"],
                "nodes": record["nodes"],
                "edges": record["edges"],
            }
        if kind == "benchmark-manifest":
            summary: dict[str, object] = {
                "id": record["id"],
            }
            if "outcome_space" in record:
                summary["outcome_space"] = record["outcome_space"]
            if "observation_ids" in record:
                summary["observation_ids"] = record["observation_ids"]
            if "latent_factor_declaration" in record:
                summary["latent_factor_declaration"] = record["latent_factor_declaration"]
            return summary
        if kind == "latent-factor-declaration":
            return {
                "id": record["id"],
                "construction_factors": record["construction_factors"],
                "sample_factors": record["sample_factors"],
            }
        if kind == "materialization-declaration":
            summary = {
                "id": record["id"],
                "benchmark_id": record["benchmark_id"],
                "requirements": record["requirements"],
            }
            if "latent_factor_declaration" in record:
                summary["latent_factor_declaration"] = record["latent_factor_declaration"]
            if "layout" in record:
                summary["layout"] = record["layout"]
            return summary
        if kind == "materialization-plan":
            summary = {
                "id": record["id"],
                "benchmark_id": record["benchmark_id"],
                "materialization_declaration": record["materialization_declaration"],
                "resolution_assignment": record["resolution_assignment"],
                "seed": record["seed"],
            }
            if "latent_factor_declaration" in record:
                summary["latent_factor_declaration"] = record["latent_factor_declaration"]
            return summary
        if kind == "measurement":
            raw_scoring_evidence = self._required_mapping(
                record["raw_scoring_evidence"],
                "raw_scoring_evidence",
            )
            return {
                "id": raw_scoring_evidence["id"],
                "benchmark_id": record["benchmark_id"],
                "observation_id": raw_scoring_evidence["observation_id"],
                "outcome_space": record["outcome_space"],
                "accepted_event": record["accepted_event"],
                "probability_measure": record["probability_measure"],
            }
        if kind == "observation-formation-declaration":
            components = self._required_sequence(record["components"], "components")
            return {
                "id": record["id"],
                "benchmark_id": record["benchmark_id"],
                "interpreter": record["interpreter"],
                "output_field": record["output_field"],
                "sequence_layout": record["sequence_layout"],
                "component_count": len(components),
                "mark_count": sum(
                    len(
                        self._required_sequence(
                            self._required_mapping(component, "components")["marks"],
                            "marks",
                        )
                    )
                    for component in components
                ),
                "components": record["components"],
            }
        if kind == "observation-showcase":
            return {
                "id": record["id"],
                "benchmark_id": record["benchmark_id"],
                "formation_declaration": record["formation_declaration"],
                "materialization_declaration": record["materialization_declaration"],
                "samples": record["samples"],
            }
        raise ConsoleDataValidationError(f"unsupported document kind: {kind}")

    def _model_inspections(
        self,
        entries: tuple[ConsoleArtifactIndexEntry, ...],
    ) -> tuple[Mapping[str, object], ...]:
        inspections: list[Mapping[str, object]] = []
        for entry in entries:
            if entry.kind != "program-graph":
                continue
            _protocol_id, record, _digest, _dependencies = (
                ConsoleArtifactIndexBuilder.load_supported_artifact(
                    entry.kind,
                    self._repository_path(
                        entry.source_path,
                        description="source document",
                    ).read_bytes(),
                )
            )
            spec = ProgramGraphSpec.from_record(record)
            inspection = ModelInspectionRecord.from_program_graph(
                id=_model_inspection_identifier(entry.reference.record_digest),
                program_graph=record,
                input_shape=_representative_contract_shape(spec.inputs[0].axes),
                output_shape=_representative_contract_shape(spec.outputs[0].axes),
            )
            record = inspection.to_record()
            record["source_path"] = entry.source_path.as_posix()
            inspections.append(record)
        return tuple(inspections)

    def _benchmark_roots(
        self,
        roots: tuple[PurePosixPath, ...],
    ) -> tuple[Path, ...]:
        benchmark_roots: list[Path] = []
        seen_roots: set[Path] = set()
        for root in roots:
            root_path = self._repository_path(root, description="public root")
            if not _is_packaged_benchmark_parent(root_path, repository_root=self._repository_root):
                continue
            for benchmark_root in discover_benchmark_roots(root_path):
                resolved = benchmark_root.resolve()
                if resolved not in seen_roots:
                    benchmark_roots.append(benchmark_root)
                    seen_roots.add(resolved)
        return tuple(benchmark_roots)

    def _benchmark_tasks(
        self,
        benchmark_roots: tuple[Path, ...],
    ) -> tuple[Mapping[str, object], ...]:
        tasks: list[Mapping[str, object]] = []
        for benchmark_root in benchmark_roots:
            source_fingerprint = self._benchmark_task_cache_fingerprint(
                benchmark_root=benchmark_root,
            )
            generator = load_generator(benchmark_root)
            manifest = generator.manifest
            atom_count = len(manifest.outcome_space.outcomes)
            outcome_atom_name = _outcome_atom_name(
                tuple(outcome.id for outcome in manifest.outcome_space.outcomes)
            )
            tasks.append(
                {
                    "kind": "generated-observations",
                    "benchmark_id": str(manifest.id),
                    "label": _benchmark_task_label(manifest),
                    "source_path": _repository_relative_path(
                        benchmark_root,
                        repository_root=self._repository_root,
                    ),
                    "volume_axis": "log2-state-space-volume",
                    "outcome_atom_name": outcome_atom_name,
                    "outcome_atom_count": atom_count,
                    "code_surfaces": self._benchmark_code_surfaces(benchmark_root),
                    "batches": list(
                        self._sample_sets(
                            generator=generator,
                            atom_count=atom_count,
                            source_fingerprint=source_fingerprint,
                        )
                    ),
                }
            )
        return tuple(tasks)

    def _benchmark_code_surfaces(self, benchmark_root: Path) -> tuple[Mapping[str, object], ...]:
        surfaces: list[Mapping[str, object]] = []
        entrypoint = benchmark_root / "benchmark.py"
        if entrypoint.is_file():
            generator_symbol = (
                _python_method_symbol(entrypoint.read_text(encoding="utf-8"), "__call__")
                or "Generator.__call__"
            )
            surfaces.append(
                self._code_surface(
                    label="Generator",
                    role="data-generator",
                    path=entrypoint,
                    symbol=generator_symbol,
                    call_path=(
                        "benchmark(root)",
                        "benchmark.generator",
                        "generator(...)",
                    ),
                )
            )
        return tuple(surfaces)

    def _code_surface(
        self,
        *,
        label: str,
        role: str,
        path: Path,
        symbol: str,
        call_path: tuple[str, ...],
    ) -> Mapping[str, object]:
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        line_span = _python_symbol_line_span(source, symbol)
        if line_span is None:
            start_line = 1
            end_line = min(len(lines), 80)
        else:
            start_line, end_line = line_span
        return {
            "label": label,
            "role": role,
            "source_path": _repository_relative_path(
                path,
                repository_root=self._repository_root,
            ),
            "symbol": symbol,
            "start_line": start_line,
            "end_line": end_line,
            "call_path": list(call_path),
            "code": "\n".join(lines[start_line - 1 : end_line]),
        }

    def _benchmark_task_cache_fingerprint(
        self,
        *,
        benchmark_root: Path,
    ) -> str:
        hasher = hashlib.sha256()
        hasher.update(b"volume-window-sample-sets-v5\0")
        entrypoint = benchmark_root / "benchmark.py"
        if entrypoint.is_file():
            self._hash_file(hasher, entrypoint)
        else:
            for name in (
                "manifest",
                "latent_factors",
                "materialization",
                "observation_formation",
            ):
                self._hash_file(hasher, benchmark_root / (name + _document_suffix))
        self._hash_file(hasher, Path(__file__))
        for module_name in ("leibniz.observation_generation", "leibniz.observation_formation"):
            module = sys.modules.get(module_name)
            module_file = None if module is None else getattr(module, "__file__", None)
            if module_file is not None:
                self._hash_file(hasher, Path(module_file))
        return hasher.hexdigest()

    def _hash_file(self, hasher: Any, path: Path) -> None:
        resolved = path.resolve()
        try:
            display_path = resolved.relative_to(self._repository_root).as_posix()
        except ValueError:
            display_path = resolved.as_posix()
        hasher.update(display_path.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(resolved.read_bytes())
        hasher.update(b"\0")

    def _sample_sets(
        self,
        *,
        generator: BenchmarkGenerator,
        atom_count: int,
        source_fingerprint: str,
    ) -> tuple[Mapping[str, object], ...]:
        cache_key = (
            str(generator.manifest.id),
            "volume-window-samples",
            source_fingerprint,
        )
        cached = _generated_batch_cache.get(cache_key)
        if cached is not None:
            return cached
        persistent_cache = _load_generated_batch_cache()
        persistent_key = _generated_batch_cache_key(cache_key)
        cached = persistent_cache.get(persistent_key)
        if cached is not None:
            _generated_batch_cache[cache_key] = cached
            return cached
        preview_windows = _preview_windows_for_generator(generator)
        if not preview_windows:
            raise ConsoleDataValidationError(
                "benchmark generator does not expose volume-window preview batches"
            )
        try:
            records = tuple(
                _preview_batch_record(generator=generator, window=window)
                for window in preview_windows
            )
        except ValueError as error:
            raise ConsoleDataValidationError(str(error)) from error
        _generated_batch_cache[cache_key] = records
        persistent_cache[persistent_key] = records
        _store_generated_batch_cache(persistent_cache)
        return records

    def _required_mapping(self, value: object, description: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise ConsoleDataValidationError(f"{description} must be a record")
        return cast(Mapping[str, object], value)

    def _required_sequence(self, value: object, description: str) -> tuple[object, ...]:
        if isinstance(value, tuple):
            return cast(tuple[object, ...], value)
        if isinstance(value, list):
            return tuple(cast(list[object], value))
        raise ConsoleDataValidationError(f"{description} must be a sequence")

    def _repository_path(self, source_path: PurePosixPath, *, description: str) -> Path:
        if source_path.is_absolute():
            raise ConsoleDataValidationError(f"{description} must be repository-relative")
        if "results" in source_path.parts:
            raise ConsoleDataValidationError(f"{description} must not reference local state")
        path = (self._repository_root / Path(source_path)).resolve()
        if not path.is_relative_to(self._repository_root):
            raise ConsoleDataValidationError(f"{description} must stay inside repository root")
        return path

    def _result_views(self, roots: tuple[Path, ...]) -> tuple[Mapping[str, object], ...]:
        views: list[Mapping[str, object]] = []
        seen_paths: set[Path] = set()
        for root in roots:
            root_path = self._result_root_path(root)
            if root_path in seen_paths:
                continue
            seen_paths.add(root_path)
            for path in self._result_view_files(self._materialized_result_view_root(root_path)):
                try:
                    record = dict(load_console_result_view(path.read_bytes()))
                except LocalResultImportError as error:
                    raise ConsoleDataValidationError(
                        f"{path}: invalid console result view: {error}"
                    ) from error
                stat = path.stat()
                record["source_path"] = self._display_path(path)
                record["source_mtime_ms"] = int(stat.st_mtime * 1000)
                record["source_size_bytes"] = stat.st_size
                views.append(record)
        return tuple(sorted(views, key=lambda view: str(view["source_path"])))

    def _result_root_path(self, root: Path) -> Path:
        path = root.resolve() if root.is_absolute() else (self._repository_root / root).resolve()
        if path.is_relative_to(self._repository_root):
            relative = path.relative_to(self._repository_root)
            if relative.parts[:1] == ("results",) and relative.parts not in {
                ("results",),
                ("results", "views"),
            }:
                raise ConsoleDataValidationError(
                    "result root inside results must be results or results/views"
                )
        if not path.is_dir():
            raise ConsoleDataValidationError(f"result root does not name a directory: {root}")
        return path

    def _materialized_result_view_root(self, root: Path) -> Path:
        if root.name == "views":
            return root
        try:
            materialize_benchmark_result_views(
                repository_root=self._repository_root,
                results_root=root,
            )
        except LocalResultImportError as error:
            if str(error) == "no benchmark result records found":
                view_root = root / "views"
                return view_root if view_root.is_dir() else root
            raise ConsoleDataValidationError(
                f"{root}: could not materialize console result views: {error}"
            ) from error
        return root / "views"

    def _result_view_files(self, root: Path) -> tuple[Path, ...]:
        files = tuple(sorted(path for path in root.rglob("*" + _document_suffix) if path.is_file()))
        nested_files = tuple(path for path in files if path.parent != root)
        return nested_files or files

    def _display_path(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved.is_relative_to(self._repository_root):
            return resolved.relative_to(self._repository_root).as_posix()
        return resolved.as_posix()


def _generated_batch_cache_key(cache_key: tuple[str, str, str]) -> str:
    return "\0".join(cache_key)


def _load_generated_batch_cache() -> dict[str, tuple[Mapping[str, object], ...]]:
    if not _generated_batch_cache_path.is_file():
        return {}
    try:
        record = load_object_document(
            _generated_batch_cache_path.read_bytes(),
            description="generated sample set cache",
        )
    except ValueError:
        return {}
    if record.get("format") != _generated_batch_cache_format:
        return {}
    if record.get("format_version") != _generated_batch_cache_format_version:
        return {}
    batches = record.get("batches")
    if not isinstance(batches, Mapping):
        return {}
    typed_batches = cast(Mapping[object, object], batches)
    cache: dict[str, tuple[Mapping[str, object], ...]] = {}
    for key, value in typed_batches.items():
        if not isinstance(value, list):
            continue
        raw_entries = cast(list[object], value)
        entries = tuple(
            cast(Mapping[str, object], entry)
            for entry in raw_entries
            if isinstance(entry, Mapping)
        )
        if entries:
            cache[str(key)] = entries
    return cache


def _store_generated_batch_cache(
    cache: Mapping[str, tuple[Mapping[str, object], ...]],
) -> None:
    try:
        _generated_batch_cache_path.parent.mkdir(parents=True, exist_ok=True)
        _generated_batch_cache_path.write_bytes(
            canonical_document_bytes(
                {
                    "format": _generated_batch_cache_format,
                    "format_version": _generated_batch_cache_format_version,
                    "batches": {
                        key: list(value)
                        for key, value in cache.items()
                    },
                }
            )
            + b"\n"
        )
    except OSError:
        return


def _outcome_atom_name(outcome_ids: tuple[str, ...]) -> str:
    prefixes = {
        outcome_id.rsplit("-", 1)[0]
        for outcome_id in outcome_ids
        if "-" in outcome_id
    }
    if len(prefixes) == 1:
        return prefixes.pop()
    return "outcome"


def _repository_relative_path(path: Path, *, repository_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repository_root).as_posix()
    except ValueError:
        return resolved.as_posix()


def _python_symbol_line_span(source: str, symbol: str) -> tuple[int, int] | None:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None
    parts = symbol.split(".")
    if len(parts) == 1:
        node = _find_ast_child(module.body, ast.FunctionDef, parts[0])
    elif len(parts) == 2:
        class_node = _find_ast_child(module.body, ast.ClassDef, parts[0])
        node = None if class_node is None else _find_ast_child(
            class_node.body,
            ast.FunctionDef,
            parts[1],
        )
    else:
        return None
    if node is None or node.end_lineno is None:
        return None
    decorators = getattr(node, "decorator_list", ())
    start_line = min((decorator.lineno for decorator in decorators), default=node.lineno)
    return (start_line, node.end_lineno)


def _python_method_symbol(source: str, method_name: str) -> str | None:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return None
    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        method = _find_ast_child(node.body, ast.FunctionDef, method_name)
        if method is not None:
            return f"{node.name}.{method.name}"
    return None


def _find_ast_child(
    nodes: Iterable[ast.stmt],
    node_type: type[ast.ClassDef] | type[ast.FunctionDef],
    name: str,
) -> ast.ClassDef | ast.FunctionDef | None:
    for node in nodes:
        if isinstance(node, node_type) and node.name == name:
            return node
    return None


def _is_packaged_benchmark_parent(path: Path, *, repository_root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(repository_root)
    except ValueError:
        return False
    return relative.parts == ("src", "leibniz", "benchmarks")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        metavar="ROOT",
        nargs="+",
        help="repository-relative public roots to discover",
    )
    parser.add_argument(
        "--result-root",
        action="append",
        default=[],
        type=Path,
        help="explicit generated result-view root, such as results/views",
    )
    args = parser.parse_args(argv)

    try:
        roots = tuple(PurePosixPath(root) for root in args.roots)
        data = ConsoleDataBuilder(Path.cwd()).discover(roots, result_roots=args.result_root)
    except (ConsoleArtifactIndexValidationError, ConsoleDataValidationError) as error:
        parser.error(str(error))

    sys.stdout.buffer.write(data.to_bytes())
    return 0


def _model_inspection_identifier(program_digest: ContentDigest | None) -> ProtocolIdentifier:
    if program_digest is None:
        raise ConsoleDataValidationError("program graph index entry is missing record digest")
    return ProtocolIdentifier(
        name=ProtocolName.parse(f"model-inspections.programs.sha-{program_digest.hex[:16]}"),
        version=SemanticVersion.parse("0.1.0"),
    )


def _representative_contract_shape(axes: tuple[object, ...]) -> tuple[int, ...]:
    shape: list[int] = []
    for axis in axes:
        if type(axis) is int:
            shape.append(axis)
        else:
            shape.append(1)
    return tuple(shape)


def _benchmark_task_label(manifest: BenchmarkManifest) -> str:
    if manifest.resolution_analysis is not None:
        display_name = manifest.resolution_analysis.get("display_name")
        if isinstance(display_name, str) and display_name:
            return display_name
    return _title_from_protocol_name(str(manifest.name))


def _title_from_protocol_name(name: str) -> str:
    parts = name.split(".")
    label = parts[-1] if parts else name
    return " ".join(_title_word(word) for word in label.split("_") if word)


def _title_word(word: str) -> str:
    return "-".join(part.capitalize() for part in word.split("-") if part)


if __name__ == "__main__":
    raise SystemExit(_main())
