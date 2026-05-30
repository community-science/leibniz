#!/usr/bin/env python
"""Project the contract-generation inventory to a deterministic source graph."""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from leibniz.documents import canonical_document_bytes, load_object_document


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    kind: str
    label: str
    path: str | None = None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
        }
        if self.path is not None:
            record["path"] = self.path
        return record


@dataclass(frozen=True, slots=True)
class GraphEdge:
    kind: str
    source: str
    target: str

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "source": self.source,
            "target": self.target,
        }


@dataclass(frozen=True, slots=True)
class RecordSpecDeclaration:
    module_path: str
    name: str
    line: int

    @property
    def id(self) -> str:
        return _node_id("record-spec", f"{self.module_path}:{self.name}")


@dataclass(frozen=True, slots=True)
class ContractObjectDeclaration:
    module_path: str
    class_name: str
    contract_name: str
    line: int

    @property
    def id(self) -> str:
        return _node_id(
            "contract-object",
            f"{self.module_path}:{self.class_name}:{self.contract_name}",
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=Path("."), type=Path)
    parser.add_argument(
        "--status",
        default=Path("CONTRACT_GENERATION_STATUS.json"),
        type=Path,
    )
    args = parser.parse_args(argv)

    repository_root = args.repository_root.resolve()
    status_path = _resolve_status_path(repository_root=repository_root, status=args.status)
    graph = source_graph(repository_root=repository_root, status_path=status_path)
    sys.stdout.buffer.write(canonical_document_bytes(graph))
    sys.stdout.buffer.write(b"\n")
    return 0


def source_graph(*, repository_root: Path, status_path: Path) -> dict[str, object]:
    status = _load_status(status_path)
    inventory = _mapping(status["code_inventory"], field="code_inventory")
    projection = _mapping(inventory["graph_projection"], field="graph_projection")
    categories = _sequence(inventory["categories"], field="categories")
    surfaces = _sequence(status["surfaces"], field="surfaces")
    tracked_paths = _tracked_inventory_paths(
        repository_root=repository_root,
        tracked_roots=_strings(inventory["tracked_roots"], field="tracked_roots"),
    )
    record_specs_by_module = _record_specs_by_module(repository_root=repository_root)
    contract_objects_by_module = _contract_objects_by_module(repository_root=repository_root)

    node_kinds = frozenset(_strings(projection["node_kinds"], field="node_kinds"))
    edge_kinds = frozenset(_strings(projection["edge_kinds"], field="edge_kinds"))
    nodes: dict[str, GraphNode] = {}
    edges: set[GraphEdge] = set()

    def add_node(node: GraphNode) -> None:
        if node.kind not in node_kinds:
            raise ValueError(f"unsupported graph node kind: {node.kind}")
        existing = nodes.get(node.id)
        if existing is not None and existing != node:
            raise ValueError(f"conflicting graph node: {node.id}")
        nodes[node.id] = node

    def add_edge(edge: GraphEdge) -> None:
        if edge.kind not in edge_kinds:
            raise ValueError(f"unsupported graph edge kind: {edge.kind}")
        edges.add(edge)

    for category_value in categories:
        category = _mapping(category_value, field="categories")
        category_name = _string(category["name"], field="categories.name")
        category_id = _node_id("category", category_name)
        add_node(GraphNode(id=category_id, kind="category", label=category_name))
        marker = category.get("structural_marker")
        if marker is not None:
            marker_text = _string(marker, field=f"{category_name}.structural_marker")
            marker_id = _node_id("structural-marker", marker_text)
            add_node(GraphNode(id=marker_id, kind="structural-marker", label=marker_text))
            add_edge(GraphEdge(kind="has-structural-marker", source=category_id, target=marker_id))
        patterns = _strings(category["path_patterns"], field=f"{category_name}.path_patterns")
        for path in _matched_paths(paths=tracked_paths, patterns=patterns):
            path_id = _path_node_id(path)
            add_node(GraphNode(id=path_id, kind="path", label=path, path=path))
            add_edge(GraphEdge(kind="categorizes", source=category_id, target=path_id))

    declared_record_spec_modules: set[str] = set()
    declared_contract_object_modules: set[str] = set()
    for surface_value in surfaces:
        surface = _mapping(surface_value, field="surfaces")
        surface_name = _string(surface["name"], field="surfaces.name")
        surface_id = _node_id("contract-surface", surface_name)
        add_node(GraphNode(id=surface_id, kind="contract-surface", label=surface_name))
        for path in _strings(
            surface.get("authored_contracts", []),
            field=f"{surface_name}.authored_contracts",
        ):
            path_id = _path_node_id(path)
            add_node(GraphNode(id=path_id, kind="path", label=path, path=path))
            add_edge(GraphEdge(kind="uses-authored-contract", source=surface_id, target=path_id))
            for record_spec in _record_spec_contract_declarations(
                repository_root=repository_root,
                relative_path=path,
            ):
                add_node(
                    GraphNode(
                        id=record_spec.id,
                        kind="record-spec",
                        label=record_spec.name,
                        path=record_spec.module_path,
                    )
                )
                add_edge(
                    GraphEdge(
                        kind="declares-record-spec",
                        source=path_id,
                        target=record_spec.id,
                    )
                )
                add_edge(
                    GraphEdge(
                        kind="covers-record-spec",
                        source=surface_id,
                        target=record_spec.id,
                    )
                )
        for path in _strings(
            surface["record_spec_modules"],
            field=f"{surface_name}.record_spec_modules",
        ):
            add_node(GraphNode(id=_path_node_id(path), kind="path", label=path, path=path))
            add_edge(
                GraphEdge(
                    kind="owns-record-spec-module",
                    source=surface_id,
                    target=_path_node_id(path),
                )
            )
            declared_record_spec_modules.add(path)
            for record_spec in record_specs_by_module.get(path, ()):
                add_node(
                    GraphNode(
                        id=record_spec.id,
                        kind="record-spec",
                        label=record_spec.name,
                        path=record_spec.module_path,
                    )
                )
                add_edge(
                    GraphEdge(
                        kind="declares-record-spec",
                        source=_path_node_id(path),
                        target=record_spec.id,
                    )
                )
                add_edge(
                    GraphEdge(
                        kind="covers-record-spec",
                        source=surface_id,
                        target=record_spec.id,
                    )
                )
        for path in _strings(surface["python_runtime"], field=f"{surface_name}.python_runtime"):
            add_node(GraphNode(id=_path_node_id(path), kind="path", label=path, path=path))
            add_edge(GraphEdge(kind="uses-runtime", source=surface_id, target=_path_node_id(path)))
            declared_contract_object_modules.add(path)
            for contract_object in contract_objects_by_module.get(path, ()):
                add_node(
                    GraphNode(
                        id=contract_object.id,
                        kind="contract-object",
                        label=(
                            f"{contract_object.class_name}."
                            f"{contract_object.contract_name}"
                        ),
                        path=contract_object.module_path,
                    )
                )
                add_edge(
                    GraphEdge(
                        kind="declares-contract-object",
                        source=_path_node_id(path),
                        target=contract_object.id,
                    )
                )
                add_edge(
                    GraphEdge(
                        kind="covers-contract-object",
                        source=surface_id,
                        target=contract_object.id,
                    )
                )
        for path in _strings(
            surface["typescript_runtime"],
            field=f"{surface_name}.typescript_runtime",
        ):
            add_node(GraphNode(id=_path_node_id(path), kind="path", label=path, path=path))
            add_edge(GraphEdge(kind="uses-runtime", source=surface_id, target=_path_node_id(path)))
        for path in _strings(surface["tests"], field=f"{surface_name}.tests"):
            test_id = _node_id("test", path)
            add_node(GraphNode(id=test_id, kind="test", label=path, path=path))
            add_edge(GraphEdge(kind="tested-by", source=surface_id, target=test_id))
        for path in _strings(
            surface["generated_outputs"],
            field=f"{surface_name}.generated_outputs",
        ):
            generated_id = _node_id("generated-output", path)
            add_node(GraphNode(id=generated_id, kind="generated-output", label=path, path=path))
            add_edge(GraphEdge(kind="generated-by", source=generated_id, target=surface_id))

    _assert_record_spec_module_coverage(
        discovered_modules=frozenset(record_specs_by_module),
        declared_modules=frozenset(declared_record_spec_modules),
    )
    _assert_contract_object_module_coverage(
        discovered_modules=frozenset(contract_objects_by_module),
        declared_modules=frozenset(declared_contract_object_modules),
    )

    return {
        "format": "leibniz.contract-source-graph",
        "format_version": 1,
        "nodes": [
            node.to_record()
            for node in sorted(nodes.values(), key=lambda node: (node.kind, node.id))
        ],
        "edges": [
            edge.to_record()
            for edge in sorted(edges, key=lambda edge: (edge.kind, edge.source, edge.target))
        ],
    }


def _resolve_status_path(*, repository_root: Path, status: Path) -> Path:
    return status if status.is_absolute() else repository_root / status


def _load_status(path: Path) -> dict[str, Any]:
    return dict(load_object_document(path.read_bytes(), description=path.as_posix()))


def _record_specs_by_module(
    *,
    repository_root: Path,
) -> dict[str, tuple[RecordSpecDeclaration, ...]]:
    declarations_by_module: dict[str, tuple[RecordSpecDeclaration, ...]] = {}
    for path in sorted((repository_root / "src" / "leibniz").rglob("*.py")):
        relative_path = path.relative_to(repository_root).as_posix()
        if relative_path == "src/leibniz/records.py":
            continue
        declarations = _record_spec_declarations(path=path, relative_path=relative_path)
        if declarations:
            declarations_by_module[relative_path] = declarations
    return declarations_by_module


def _contract_objects_by_module(
    *,
    repository_root: Path,
) -> dict[str, tuple[ContractObjectDeclaration, ...]]:
    declarations_by_module: dict[str, tuple[ContractObjectDeclaration, ...]] = {}
    for path in sorted((repository_root / "src" / "leibniz").rglob("*.py")):
        relative_path = path.relative_to(repository_root).as_posix()
        declarations = _contract_object_declarations(path=path, relative_path=relative_path)
        if declarations:
            declarations_by_module[relative_path] = declarations
    return declarations_by_module


def _record_spec_declarations(
    *,
    path: Path,
    relative_path: str,
) -> tuple[RecordSpecDeclaration, ...]:
    syntax_tree = ast.parse(path.read_text(encoding="utf-8"))
    declarations: list[RecordSpecDeclaration] = []
    for node in ast.walk(syntax_tree):
        if not isinstance(node, ast.Assign):
            continue
        if not _is_record_spec_call(node.value):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                declarations.append(
                    RecordSpecDeclaration(
                        module_path=relative_path,
                        name=target.id,
                        line=node.lineno,
                    )
                )
    return tuple(sorted(declarations, key=lambda declaration: declaration.name))


def _contract_object_declarations(
    *,
    path: Path,
    relative_path: str,
) -> tuple[ContractObjectDeclaration, ...]:
    syntax_tree = ast.parse(path.read_text(encoding="utf-8"))
    declarations: list[ContractObjectDeclaration] = []
    for node in ast.walk(syntax_tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for child in node.body:
            if not isinstance(child, ast.FunctionDef):
                continue
            if child.name != "record_contract":
                continue
            contract_name = _returned_record_contract_name(child)
            if contract_name is None:
                continue
            declarations.append(
                ContractObjectDeclaration(
                    module_path=relative_path,
                    class_name=node.name,
                    contract_name=contract_name,
                    line=child.lineno,
                )
            )
    return tuple(
        sorted(
            declarations,
            key=lambda declaration: (declaration.module_path, declaration.class_name),
        )
    )


def _returned_record_contract_name(function: ast.FunctionDef) -> str | None:
    for node in ast.walk(function):
        if not isinstance(node, ast.Return):
            continue
        value = node.value
        if not isinstance(value, ast.Call):
            continue
        if not _is_name(value.func, "RecordContract"):
            continue
        for keyword in value.keywords:
            if keyword.arg != "name":
                continue
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                return keyword.value.value
    return None


def _record_spec_contract_declarations(
    *,
    repository_root: Path,
    relative_path: str,
) -> tuple[RecordSpecDeclaration, ...]:
    contract = _load_status(repository_root / relative_path)
    if contract.get("format") != "leibniz.record-contract-set":
        return ()
    records = _sequence(contract.get("records"), field=f"{relative_path}.records")
    declarations: list[RecordSpecDeclaration] = []
    for index, record_value in enumerate(records):
        record = _mapping(record_value, field=f"{relative_path}.records.{index}")
        declarations.append(
            RecordSpecDeclaration(
                module_path=relative_path,
                name=_string(record.get("name"), field=f"{relative_path}.records.{index}.name"),
                line=index + 1,
            )
        )
    return tuple(sorted(declarations, key=lambda declaration: declaration.name))


def _is_record_spec_call(value: ast.expr) -> bool:
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "RecordSpec"
    )


def _is_name(value: ast.expr, name: str) -> bool:
    return isinstance(value, ast.Name) and value.id == name


def _assert_record_spec_module_coverage(
    *,
    discovered_modules: frozenset[str],
    declared_modules: frozenset[str],
) -> None:
    missing = sorted(discovered_modules - declared_modules)
    unknown = sorted(declared_modules - discovered_modules)
    if missing or unknown:
        raise ValueError(
            "record spec module coverage drift: "
            f"missing={missing!r}; unknown={unknown!r}"
        )


def _assert_contract_object_module_coverage(
    *,
    discovered_modules: frozenset[str],
    declared_modules: frozenset[str],
) -> None:
    missing = sorted(discovered_modules - declared_modules)
    if missing:
        raise ValueError(f"contract object module coverage drift: missing={missing!r}")


def _tracked_inventory_paths(
    *,
    repository_root: Path,
    tracked_roots: tuple[str, ...],
) -> tuple[str, ...]:
    tracked = subprocess.run(
        ["git", "ls-files", *tracked_roots],
        check=True,
        cwd=repository_root,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    return tuple(
        path
        for path in tracked
        if "/node_modules/" not in path
        and "/dist/" not in path
        and "/generated/" not in path
    )


def _matched_paths(*, paths: tuple[str, ...], patterns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        path
        for path in paths
        if any(PurePosixPath(path).match(pattern) for pattern in patterns)
    )


def _node_id(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def _path_node_id(path: str) -> str:
    return _node_id("path", path)


def _mapping(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field}: expected record")
    return cast(dict[str, Any], value)


def _sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field}: expected sequence")
    return tuple(value)


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field}: expected string")
    return value


def _strings(value: object, *, field: str) -> tuple[str, ...]:
    return tuple(_string(item, field=field) for item in _sequence(value, field=field))


if __name__ == "__main__":
    raise SystemExit(main())
