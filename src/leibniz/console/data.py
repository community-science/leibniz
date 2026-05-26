"""Build generated data payloads for the browser console."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from leibniz.architectures import ArchitectureManifestDocument
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.console.artifact_index import (
    ConsoleArtifactIndex,
    ConsoleArtifactIndexBuilder,
    ConsoleArtifactIndexEntry,
    ConsoleArtifactIndexSource,
    ConsoleArtifactIndexValidationError,
)
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.materialization import MaterializationDeclarationDocument, MaterializationPlan
from leibniz.model_inspection import ModelInspectionRecord
from leibniz.observation_formation import ObservationFormationDeclarationDocument
from leibniz.observation_inspection import ObservationInspectionRecord
from leibniz.observation_showcases import ObservationShowcaseDocument
from leibniz.performance_bundles import PerformanceViewBundle, PerformanceViewBundleDocument

__all__ = [
    "ConsoleData",
    "ConsoleDataBuilder",
    "ConsoleDataValidationError",
]

_format = "leibniz.console-data"
_format_version = 1


class ConsoleDataValidationError(ValueError):
    """Raised when console data cannot be discovered or generated."""


@dataclass(frozen=True, slots=True)
class ConsoleData:
    """A generated console data payload for the browser."""

    artifact_index: ConsoleArtifactIndex
    artifact_details: tuple[Mapping[str, object], ...]
    observation_inspections: tuple[Mapping[str, object], ...]
    performance_views: tuple[Mapping[str, object], ...]
    model_inspections: tuple[Mapping[str, object], ...]
    source_modules: tuple[Mapping[str, object], ...]

    def to_record(self) -> dict[str, object]:
        return {
            "format": _format,
            "format_version": _format_version,
            "artifact_index": self.artifact_index.to_record(),
            "artifact_details": list(self.artifact_details),
            "observation_inspections": list(self.observation_inspections),
            "performance_views": list(self.performance_views),
            "model_inspections": list(self.model_inspections),
            "source_modules": list(self.source_modules),
        }

    def to_bytes(self) -> bytes:
        return canonical_document_bytes(self.to_record()) + b"\n"


class ConsoleDataBuilder:
    """Discover supported public documents and build a console data payload."""

    def __init__(self, repository_root: Path) -> None:
        self._repository_root = repository_root.resolve()
        self._artifact_builder = ConsoleArtifactIndexBuilder(self._repository_root)

    def discover(self, roots: Iterable[PurePosixPath]) -> ConsoleData:
        sources = tuple(self._discover_sources(tuple(roots)))
        artifact_index = self._artifact_builder.build(sources)
        details = tuple(self._detail_for_source(source) for source in artifact_index.entries)
        observation_inspections = tuple(self._observation_inspections(artifact_index.entries))
        performance_views = tuple(self._performance_views(artifact_index.entries))
        model_inspections = tuple(self._model_inspections(artifact_index.entries))
        source_modules = tuple(self._source_modules())
        return ConsoleData(
            artifact_index=artifact_index,
            artifact_details=details,
            observation_inspections=observation_inspections,
            performance_views=performance_views,
            model_inspections=model_inspections,
            source_modules=source_modules,
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
        if kind == "performance-view-bundle":
            measurement_cases = self._required_sequence(
                record["measurement_cases"],
                "measurement_cases",
            )
            return {
                "id": record["id"],
                "benchmark_manifest": record["benchmark_manifest"],
                "materialization_declaration": record["materialization_declaration"],
                "observation_formation_declaration": (
                    record["observation_formation_declaration"]
                ),
                "view_id": record["view_id"],
                "complexity_axis": record["complexity_axis"],
                "expected_complexities": record["expected_complexities"],
                "measurement_cases": record["measurement_cases"],
                "measurement_count": len(measurement_cases),
            }
        raise ConsoleDataValidationError(f"unsupported document kind: {kind}")

    def _observation_inspections(
        self,
        entries: tuple[ConsoleArtifactIndexEntry, ...],
    ) -> tuple[Mapping[str, object], ...]:
        formation_declarations = {
            document.declaration.id: document.declaration
            for document in (
                ObservationFormationDeclarationDocument.from_bytes(
                    self._repository_path(
                        entry.source_path,
                        description="source document",
                    ).read_bytes()
                )
                for entry in entries
                if entry.kind == "observation-formation-declaration"
            )
        }
        materialization_declarations = {
            document.declaration.id: document.declaration
            for document in (
                MaterializationDeclarationDocument.from_bytes(
                    self._repository_path(
                        entry.source_path,
                        description="source document",
                    ).read_bytes()
                )
                for entry in entries
                if entry.kind == "materialization-declaration"
            )
        }
        inspections: list[Mapping[str, object]] = []
        for entry in entries:
            if entry.kind != "observation-showcase":
                continue
            document = ObservationShowcaseDocument.from_bytes(
                self._repository_path(entry.source_path, description="source document").read_bytes()
            )
            showcase = document.manifest
            if showcase.formation_declaration.protocol_id is None:
                raise ConsoleDataValidationError(
                    f"{entry.source_path}: formation_declaration must include protocol_id"
                )
            if showcase.materialization_declaration.protocol_id is None:
                raise ConsoleDataValidationError(
                    f"{entry.source_path}: materialization_declaration must include protocol_id"
                )
            try:
                formation = formation_declarations[showcase.formation_declaration.protocol_id]
                materialization = materialization_declarations[
                    showcase.materialization_declaration.protocol_id
                ]
            except KeyError as error:
                raise ConsoleDataValidationError(
                    f"{entry.source_path}: showcase references an undiscovered declaration"
                ) from error
            if not showcase.formation_declaration.matches_record(formation.to_record()):
                raise ConsoleDataValidationError(
                    f"{entry.source_path}: formation_declaration reference does not match"
                )
            if not showcase.materialization_declaration.matches_record(
                materialization.to_record()
            ):
                raise ConsoleDataValidationError(
                    f"{entry.source_path}: materialization_declaration reference does not match"
                )
            for sample in showcase.samples:
                plan = MaterializationPlan.resolve(
                    id=_child_identifier(sample.id, "materialization-plan"),
                    declaration=materialization,
                    scale_assignment=sample.scale_assignment,
                    complexity_assignment=sample.complexity_assignment,
                    seed=sample.seed,
                )
                observation = formation.form_observation(
                    id=_child_identifier(sample.id, "formed-observation"),
                    plan=plan,
                    component_sequence=sample.component_sequence,
                )
                inspection = ObservationInspectionRecord.from_formed_observation(
                    id=sample.id,
                    observation=observation,
                    materialization_plan=plan,
                    sample_index=sample.sample_index,
                    outcome_id=sample.outcome_id,
                )
                record = inspection.to_record()
                record["label"] = sample.label
                record["showcase"] = {
                    "id": str(showcase.id),
                    "source_path": entry.source_path.as_posix(),
                }
                inspections.append(record)
        return tuple(inspections)

    def _performance_views(
        self,
        entries: tuple[ConsoleArtifactIndexEntry, ...],
    ) -> tuple[Mapping[str, object], ...]:
        benchmark_manifests = {
            document.manifest.id: document.manifest
            for document in (
                BenchmarkManifestDocument.from_bytes(
                    self._repository_path(
                        entry.source_path,
                        description="source document",
                    ).read_bytes()
                )
                for entry in entries
                if entry.kind == "benchmark-manifest"
            )
        }
        materialization_declarations = {
            document.declaration.id: document.declaration
            for document in (
                MaterializationDeclarationDocument.from_bytes(
                    self._repository_path(
                        entry.source_path,
                        description="source document",
                    ).read_bytes()
                )
                for entry in entries
                if entry.kind == "materialization-declaration"
            )
        }
        formation_declarations = {
            document.declaration.id: document.declaration
            for document in (
                ObservationFormationDeclarationDocument.from_bytes(
                    self._repository_path(
                        entry.source_path,
                        description="source document",
                    ).read_bytes()
                )
                for entry in entries
                if entry.kind == "observation-formation-declaration"
            )
        }
        views: list[Mapping[str, object]] = []
        for entry in entries:
            if entry.kind != "performance-view-bundle":
                continue
            document = PerformanceViewBundleDocument.from_bytes(
                self._repository_path(entry.source_path, description="source document").read_bytes()
            )
            manifest = document.manifest
            if manifest.benchmark_manifest.protocol_id is None:
                raise ConsoleDataValidationError(
                    f"{entry.source_path}: benchmark_manifest must include protocol_id"
                )
            if manifest.materialization_declaration.protocol_id is None:
                raise ConsoleDataValidationError(
                    f"{entry.source_path}: materialization_declaration must include protocol_id"
                )
            if manifest.observation_formation_declaration.protocol_id is None:
                raise ConsoleDataValidationError(
                    f"{entry.source_path}: observation_formation_declaration must include "
                    "protocol_id"
                )
            try:
                benchmark_manifest = benchmark_manifests[manifest.benchmark_manifest.protocol_id]
                materialization_declaration = materialization_declarations[
                    manifest.materialization_declaration.protocol_id
                ]
                formation_declaration = formation_declarations[
                    manifest.observation_formation_declaration.protocol_id
                ]
            except KeyError as error:
                raise ConsoleDataValidationError(
                    f"{entry.source_path}: performance bundle references an undiscovered source"
                ) from error
            bundle = PerformanceViewBundle.from_manifest(
                manifest,
                benchmark_manifest=benchmark_manifest,
                materialization_declaration=materialization_declaration,
                observation_formation_declaration=formation_declaration,
            )
            record = bundle.to_record()
            record["source_path"] = entry.source_path.as_posix()
            views.append(record)
        return tuple(views)

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
        if ".leibniz" in source_path.parts:
            raise ConsoleDataValidationError(f"{description} must not reference local state")
        path = (self._repository_root / Path(source_path)).resolve()
        if not path.is_relative_to(self._repository_root):
            raise ConsoleDataValidationError(f"{description} must stay inside repository root")
        return path

    def _source_modules(self) -> tuple[Mapping[str, object], ...]:
        package_root = self._repository_root / "src" / "leibniz"
        records: list[Mapping[str, object]] = []
        for path in sorted(package_root.rglob("*.py")):
            relative_path = path.relative_to(self._repository_root)
            relative_module_path = path.relative_to(package_root)
            if any(part.startswith("_") for part in relative_module_path.parts):
                continue
            module_name = self._module_name(relative_module_path)
            if module_name is None:
                continue
            records.append(
                {
                    "module_name": module_name,
                    "source_path": relative_path.as_posix(),
                    "public_exports": list(self._public_exports(path)),
                    "validation_commands": list(self._validation_commands(path)),
                }
            )
        return tuple(records)

    def _module_name(self, relative_module_path: Path) -> str | None:
        if relative_module_path.name == "__init__.py":
            module_parts = relative_module_path.parent.parts
            if not module_parts:
                return "leibniz"
            return ".".join(("leibniz", *module_parts))
        if relative_module_path.suffix != ".py":
            return None
        return ".".join(("leibniz", *relative_module_path.with_suffix("").parts))

    def _public_exports(self, path: Path) -> tuple[str, ...]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        return self._literal_string_sequence(node.value, "__all__")
        return ()

    def _literal_string_sequence(self, value: ast.expr, description: str) -> tuple[str, ...]:
        if not isinstance(value, ast.List | ast.Tuple):
            raise ConsoleDataValidationError(f"{description} must be a literal sequence")
        names: list[str] = []
        for item in value.elts:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                raise ConsoleDataValidationError(f"{description} must contain literal strings")
            names.append(item.value)
        return tuple(names)

    def _validation_commands(self, path: Path) -> tuple[str, ...]:
        test_path = self._test_path_for_source(path)
        commands = ["python -m pytest tests/test_public_surface.py"]
        if test_path is not None:
            test_command_path = test_path.relative_to(self._repository_root).as_posix()
            commands.insert(0, f"python -m pytest {test_command_path}")
        return tuple(commands)

    def _test_path_for_source(self, path: Path) -> Path | None:
        package_root = self._repository_root / "src" / "leibniz"
        relative_path = path.relative_to(package_root)
        if relative_path.name == "__init__.py":
            module_parts = relative_path.parent.parts
        else:
            module_parts = relative_path.with_suffix("").parts

        candidates: list[Path] = []
        if module_parts:
            candidates.append(
                self._repository_root / "tests" / f"test_{'_'.join(module_parts)}.py"
            )
            candidates.append(self._repository_root / "tests" / f"test_{module_parts[-1]}.py")
        else:
            candidates.append(self._repository_root / "tests" / "test_package.py")

        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        metavar="ROOT",
        nargs="+",
        help="repository-relative public roots to discover",
    )
    args = parser.parse_args(argv)

    try:
        roots = tuple(PurePosixPath(root) for root in args.roots)
        data = ConsoleDataBuilder(Path.cwd()).discover(roots)
    except (ConsoleArtifactIndexValidationError, ConsoleDataValidationError) as error:
        parser.error(str(error))

    sys.stdout.buffer.write(data.to_bytes())
    return 0


def _child_identifier(parent: ProtocolIdentifier, suffix: str) -> ProtocolIdentifier:
    return ProtocolIdentifier(
        name=ProtocolName.parse(f"{parent.name}.{suffix}"),
        version=parent.version,
    )


def _model_inspection_identifier(architecture_id: ProtocolIdentifier) -> ProtocolIdentifier:
    return ProtocolIdentifier(
        name=ProtocolName.parse(f"model-inspections.{architecture_id.name}"),
        version=architecture_id.version,
    )


if __name__ == "__main__":
    raise SystemExit(_main())
