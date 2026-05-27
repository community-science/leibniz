"""Build generated data payloads for the browser console."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from leibniz.architectures import ArchitectureManifestDocument
from leibniz.console.artifact_index import (
    ConsoleArtifactIndex,
    ConsoleArtifactIndexBuilder,
    ConsoleArtifactIndexEntry,
    ConsoleArtifactIndexSource,
    ConsoleArtifactIndexValidationError,
)
from leibniz.documents import canonical_document_bytes, document_filename_suffix
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.local_results import LocalResultImportError, load_console_result_view
from leibniz.model_inspection import ModelInspectionRecord
from leibniz.model_operators import model_operator_vocabulary
from leibniz.observation_generation import (
    ObservationGenerator,
    field_to_png_data_url,
    load_observation_generator,
)

__all__ = [
    "ConsoleData",
    "ConsoleDataBuilder",
    "ConsoleDataValidationError",
]

_format = "leibniz.console-data"
_format_version = 1
_document_suffix = document_filename_suffix()
_generated_batch_cache: dict[
    tuple[str, str, int, int, int, str, bool, tuple[tuple[int, ...], ...] | None],
    Mapping[str, object],
] = {}


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
        benchmark_tasks = tuple(self._benchmark_tasks(artifact_index.entries))
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
            except Exception:
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
            return {
                "input_shape": record["input_shape"],
                "output_shape": record["output_shape"],
                "layers": record["layers"],
            }
        if kind == "benchmark-manifest":
            summary: dict[str, object] = {
                "id": record["id"],
            }
            if "outcome_space" in record:
                summary["outcome_space"] = record["outcome_space"]
            if "outcome_sequence" in record:
                summary["outcome_sequence"] = record["outcome_sequence"]
            if "scale_parameter" in record:
                summary["scale_parameter"] = record["scale_parameter"]
            if "observation_ids" in record:
                summary["observation_ids"] = record["observation_ids"]
            if "latent_factor_declaration" in record:
                summary["latent_factor_declaration"] = record["latent_factor_declaration"]
            if "complexity_coordinate" in record:
                summary["complexity_coordinate"] = record["complexity_coordinate"]
            return summary
        if kind == "latent-factor-declaration":
            return {
                "id": record["id"],
                "construction_factors": record["construction_factors"],
                "sample_factors": record["sample_factors"],
                "complexity_projections": record["complexity_projections"],
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
                "scale_assignment": record["scale_assignment"],
                "complexity_assignment": record["complexity_assignment"],
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
                "slot_composition": record["slot_composition"],
                "component_count": len(components),
                "mark_count": sum(
                    len(self._required_sequence(
                        self._required_mapping(component, "components")["marks"],
                        "marks",
                    ))
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

    def _benchmark_tasks(
        self,
        entries: tuple[ConsoleArtifactIndexEntry, ...],
    ) -> tuple[Mapping[str, object], ...]:
        tasks: list[Mapping[str, object]] = []
        for entry in entries:
            if entry.kind != "benchmark-manifest":
                continue
            benchmark_root = self._repository_path(
                entry.source_path,
                description="benchmark manifest",
            ).parent
            required = (
                "manifest",
                "latent_factors",
                "materialization",
                "observation_formation",
            )
            if any(not (benchmark_root / (name + _document_suffix)).is_file() for name in required):
                continue
            generator = load_observation_generator(benchmark_root)
            manifest = generator.benchmark_manifest
            if manifest.outcome_sequence is None:
                continue
            if manifest.scale_parameter is None:
                raise ConsoleDataValidationError(
                    f"{entry.source_path}: benchmark task requires scale_parameter"
                )
            atom_count = manifest.outcome_sequence.atom_count
            scales = (1, 2, 3, 4)
            batches: list[Mapping[str, object]] = []
            for scale in scales:
                batches.append(
                    self._generated_observation_batch(
                        generator=generator,
                        mode="canonical",
                        label=f"Canonical L={scale}",
                        scale=scale,
                        sample_count=4,
                        seed=101,
                        sample_card_density="standard",
                        aggregate_mode=False,
                    )
                )
            batches.append(
                self._generated_observation_batch(
                    generator=generator,
                    mode="symbol-probe",
                    label="Symbol probe",
                    scale=1,
                    sample_count=atom_count,
                    seed=101,
                    component_sequences=tuple((digit,) for digit in range(atom_count)),
                    sample_card_density="compact",
                    aggregate_mode=False,
                )
            )
            for scale in scales:
                batches.append(
                    self._generated_observation_batch(
                        generator=generator,
                        mode="complexity-sweep",
                        label=f"Complexity C={scale}",
                        scale=scale,
                        sample_count=1,
                        seed=202,
                        component_sequences=(
                            tuple(index % atom_count for index in range(scale)),
                        ),
                        sample_card_density="standard",
                        aggregate_mode=True,
                    )
                )
            tasks.append(
                {
                    "kind": "generated-observations",
                    "benchmark_id": str(manifest.id),
                    "label": _title_from_protocol_name(str(manifest.name)),
                    "source_path": entry.source_path.as_posix(),
                    "scale_axis": manifest.scale_parameter.symbol,
                    "complexity_axis": manifest.complexity_coordinate,
                    "outcome_atom_name": manifest.outcome_sequence.atom_name,
                    "outcome_atom_count": atom_count,
                    "batches": batches,
                }
            )
        return tuple(tasks)

    def _generated_observation_batch(
        self,
        *,
        generator: ObservationGenerator,
        mode: str,
        label: str,
        scale: int,
        sample_count: int,
        seed: int,
        sample_card_density: str,
        aggregate_mode: bool,
        component_sequences: tuple[tuple[int, ...], ...] | None = None,
    ) -> Mapping[str, object]:
        cache_key = (
            str(generator.benchmark_manifest.id),
            mode,
            scale,
            sample_count,
            seed,
            sample_card_density,
            aggregate_mode,
            component_sequences,
        )
        cached = _generated_batch_cache.get(cache_key)
        if cached is not None:
            return cached
        batch = generator.sample_batch(
            scale=scale,
            sample_count=sample_count,
            seed=seed,
            component_sequences=component_sequences,
        )
        record = {
            "mode": mode,
            "label": label,
            "scale": batch.scale,
            "seed": batch.seed,
            "sample_count": len(batch.samples),
            "presentation": {
                "sample_card_density": sample_card_density,
                "aggregate_mode": aggregate_mode,
            },
            "samples": [
                {
                    "index": sample.index,
                    "outcome_id": sample.outcome_id,
                    "component_sequence": list(sample.observation.component_sequence),
                    "complexity": sample.complexity,
                    "field_shape": list(sample.field.shape),
                    "image_data_url": field_to_png_data_url(sample.field),
                    "materialization_plan": sample.materialization_plan.to_record(),
                    "latent_coordinates": [
                        dict(coordinate) for coordinate in sample.latent_coordinates
                    ],
                }
                for sample in batch.samples
            ],
        }
        _generated_batch_cache[cache_key] = record
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
        if ".leibniz" in source_path.parts or ".runs" in source_path.parts:
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
            for path in self._result_view_files(root_path):
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
        path = (
            root.resolve()
            if root.is_absolute()
            else (self._repository_root / root).resolve()
        )
        if path.is_relative_to(self._repository_root):
            relative = path.relative_to(self._repository_root)
            if ".leibniz" in relative.parts:
                raise ConsoleDataValidationError("result root must not reference .leibniz")
            if ".runs" in relative.parts and relative.parts[:2] != (".runs", "views"):
                raise ConsoleDataValidationError("result root inside .runs must be .runs/views")
        if not path.is_dir():
            raise ConsoleDataValidationError(f"result root does not name a directory: {root}")
        return path

    def _result_view_files(self, root: Path) -> tuple[Path, ...]:
        return tuple(
            sorted(path for path in root.rglob("*" + _document_suffix) if path.is_file())
        )

    def _display_path(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved.is_relative_to(self._repository_root):
            return resolved.relative_to(self._repository_root).as_posix()
        return resolved.as_posix()

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
        help="explicit generated result-view root, such as .runs/views",
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
