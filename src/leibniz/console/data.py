"""Build generated data payloads for the browser console."""

from __future__ import annotations

import argparse
import ast
import hashlib
import random
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.benchmark_implementations import (
    Generator as BenchmarkGenerator,
)
from leibniz.benchmark_implementations import (
    discover_benchmark_roots,
)
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
from leibniz.documents import (
    canonical_document_bytes,
    document_filename_suffix,
    load_object_document,
)
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.local_results import (
    LocalResultImportError,
    load_console_result_view,
    materialize_benchmark_result_views,
)
from leibniz.model_inspection import ModelInspectionRecord
from leibniz.model_operators import model_operator_vocabulary
from leibniz.observation_generation import field_to_png_data_url, load_generator

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
_generated_batch_cache_format_version = 1
_generated_batch_cache_path = (
    Path(__file__).parent
    / "_web_src"
    / "src"
    / "generated"
    / ("generatedSampleSets" + _document_suffix)
)
_generated_batch_cache: dict[tuple[str, str, str], Mapping[str, object]] = {}


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
            operator_vocabulary=model_operator_vocabulary(),
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
        if kind == "architecture-manifest":
            graph = ArchitectureManifest.from_record(record).graph
            return {
                "input_shape": record["input_shape"],
                "output_shape": record["output_shape"],
                "layers": record["layers"],
                "architecture_graph": graph.to_record(),
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
                "complexity_projections": record.get("complexity_projections", ()),
                "resolution_requirements": record.get("resolution_requirements", ()),
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
            if entry.kind != "architecture-manifest":
                continue
            document = ArchitectureManifestDocument.from_bytes(
                self._repository_path(entry.source_path, description="source document").read_bytes()
            )
            inspection = ModelInspectionRecord.from_architecture(
                id=_model_inspection_identifier(document.manifest.id),
                architecture_manifest=document.manifest,
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
            manifest = generator.benchmark_manifest
            atom_count = len(manifest.outcome_space.outcomes)
            outcome_atom_name = _outcome_atom_name(
                tuple(outcome.id for outcome in manifest.outcome_space.outcomes)
            )
            tasks.append(
                {
                    "kind": "generated-observations",
                    "benchmark_id": str(manifest.id),
                    "label": _title_from_protocol_name(str(manifest.name)),
                    "source_path": _repository_relative_path(
                        benchmark_root,
                        repository_root=self._repository_root,
                    ),
                    "complexity_axis": None,
                    "outcome_atom_name": outcome_atom_name,
                    "outcome_atom_count": atom_count,
                    "code_surfaces": self._benchmark_code_surfaces(benchmark_root),
                    "batches": [
                        self._balanced_sample_set(
                            generator=generator,
                            atom_count=atom_count,
                            source_fingerprint=source_fingerprint,
                        )
                    ],
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
        hasher.update(b"balanced-sample-set-v1\0")
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

    def _balanced_sample_set(
        self,
        *,
        generator: BenchmarkGenerator,
        atom_count: int,
        source_fingerprint: str,
    ) -> Mapping[str, object]:
        cache_key = (
            str(generator.benchmark_manifest.id),
            "balanced-component-samples",
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
        samples: list[Mapping[str, object]] = []
        samples_per_component_count = 40
        component_counts = (1,)
        component_sequences = _balanced_component_sequences(
            component_counts=component_counts,
            samples_per_component_count=samples_per_component_count,
            atom_count=atom_count,
            seed=f"{generator.benchmark_manifest.id}:balanced-console-samples",
        )
        used_field_shapes: set[tuple[int, ...]] = set()
        for component_count in component_counts:
            for sample_index, sequence in enumerate(component_sequences[component_count]):
                seed = 4000 + component_count * 100 + sample_index
                attempt_count = 0
                while True:
                    attempt_count += 1
                    if attempt_count > 512:
                        raise ConsoleDataValidationError(
                            "could not generate unique console sample canvas shapes"
                        )
                    sample_set = generator(
                        component_count=component_count,
                        shape=(),
                        seed=seed,
                        include_fields=True,
                        component_sequences=(sequence,),
                    )
                    if not sample_set.includes_fields:
                        raise ConsoleDataValidationError(
                            "generator did not include generated fields"
                        )
                    batch = sample_set
                    sample = batch.samples[0]
                    field_shape = tuple(sample.require_field().shape)
                    if field_shape not in used_field_shapes:
                        used_field_shapes.add(field_shape)
                        break
                    seed += 1000
                samples.append(
                    {
                        "index": len(samples),
                        "outcome_id": sample.outcome_id,
                        "component_sequence": list(
                            sample.field_record().component_sequence
                        ),
                        "complexity": sample.complexity,
                        "field_shape": list(sample.require_field().shape),
                        "image_data_url": field_to_png_data_url(sample.require_field()),
                        "materialization_plan": sample.materialization_plan.to_record(),
                        "latent_coordinates": [
                            dict(coordinate) for coordinate in sample.latent_coordinates
                        ],
                    }
                )
        sample_count = len(samples)
        samples.sort(key=lambda sample: _sample_display_key(sample, sample_count))
        record = {
            "mode": "balanced",
            "label": "Balanced samples",
            "component_count": 1,
            "seed": 401,
            "sample_count": len(samples),
            "presentation": {
                "sample_card_density": "standard",
                "aggregate_mode": False,
            },
            "samples": samples,
        }
        _generated_batch_cache[cache_key] = record
        persistent_cache[persistent_key] = record
        _store_generated_batch_cache(persistent_cache)
        return record

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


def _sample_display_key(sample: Mapping[str, object], sample_count: int) -> int:
    index = sample["index"]
    if not isinstance(index, int):
        raise ConsoleDataValidationError("generated sample index must be an integer")
    return (index * 17) % (sample_count + 1)


def _generated_batch_cache_key(cache_key: tuple[str, str, str]) -> str:
    return "\0".join(cache_key)


def _load_generated_batch_cache() -> dict[str, Mapping[str, object]]:
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
    return {
        str(key): cast(Mapping[str, object], value)
        for key, value in typed_batches.items()
        if isinstance(value, Mapping)
    }


def _store_generated_batch_cache(cache: Mapping[str, Mapping[str, object]]) -> None:
    try:
        _generated_batch_cache_path.parent.mkdir(parents=True, exist_ok=True)
        _generated_batch_cache_path.write_bytes(
            canonical_document_bytes(
                {
                    "format": _generated_batch_cache_format,
                    "format_version": _generated_batch_cache_format_version,
                    "batches": dict(cache),
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


def _balanced_component_sequences(
    *,
    component_counts: tuple[int, ...],
    samples_per_component_count: int,
    atom_count: int,
    seed: str,
) -> dict[int, tuple[tuple[int, ...], ...]]:
    total_tokens = samples_per_component_count * sum(component_counts)
    if total_tokens % atom_count != 0:
        raise ConsoleDataValidationError(
            "balanced console samples require total token count to divide atom count"
        )
    generator = random.Random(seed)
    tokens = [
        digit
        for digit in range(atom_count)
        for _occurrence in range(total_tokens // atom_count)
    ]
    generator.shuffle(tokens)
    token_iter = iter(tokens)
    return {
        component_count: tuple(
            tuple(next(token_iter) for _token in range(component_count))
            for _sample in range(samples_per_component_count)
        )
        for component_count in component_counts
    }


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


def _model_inspection_identifier(architecture_id: ProtocolIdentifier) -> ProtocolIdentifier:
    return ProtocolIdentifier(
        name=ProtocolName.parse(f"model-inspections.{architecture_id.name}"),
        version=architecture_id.version,
    )


def _title_from_protocol_name(name: str) -> str:
    parts = name.split(".")
    label = parts[-1] if parts else name
    return " ".join(word.capitalize() for word in label.replace("-", "_").split("_") if word)


if __name__ == "__main__":
    raise SystemExit(_main())
