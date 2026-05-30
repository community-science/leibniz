#!/usr/bin/env python
"""Project the contract-generation inventory to a deterministic source graph."""

from __future__ import annotations

import argparse
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

    for surface_value in surfaces:
        surface = _mapping(surface_value, field="surfaces")
        surface_name = _string(surface["name"], field="surfaces.name")
        surface_id = _node_id("contract-surface", surface_name)
        add_node(GraphNode(id=surface_id, kind="contract-surface", label=surface_name))
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
        for path in _strings(surface["python_runtime"], field=f"{surface_name}.python_runtime"):
            add_node(GraphNode(id=_path_node_id(path), kind="path", label=path, path=path))
            add_edge(GraphEdge(kind="uses-runtime", source=surface_id, target=_path_node_id(path)))
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
