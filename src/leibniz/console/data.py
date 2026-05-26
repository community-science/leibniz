"""Build generated data payloads for the browser console."""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from leibniz.console.artifact_index import (
    ConsoleArtifactIndex,
    ConsoleArtifactIndexBuilder,
    ConsoleArtifactIndexEntry,
    ConsoleArtifactIndexSource,
    ConsoleArtifactIndexValidationError,
)
from leibniz.documents import canonical_document_bytes

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
    source_modules: tuple[Mapping[str, object], ...]

    def to_record(self) -> dict[str, object]:
        return {
            "format": _format,
            "format_version": _format_version,
            "artifact_index": self.artifact_index.to_record(),
            "artifact_details": list(self.artifact_details),
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
        source_modules = tuple(self._source_modules())
        return ConsoleData(
            artifact_index=artifact_index,
            artifact_details=details,
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
        raise ConsoleDataValidationError(f"unsupported document kind: {kind}")

    def _required_mapping(self, value: object, description: str) -> Mapping[str, object]:
        if not isinstance(value, Mapping):
            raise ConsoleDataValidationError(f"{description} must be a record")
        return cast(Mapping[str, object], value)

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


if __name__ == "__main__":
    raise SystemExit(_main())
