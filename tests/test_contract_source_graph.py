import importlib.util
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from leibniz.documents import load_object_document

_repository_root = Path(__file__).parents[1]
_script_path = _repository_root / "scripts" / "contract_source_graph.py"


class _SourceGraphModule(Protocol):
    source_graph: Callable[..., dict[str, object]]


def test_contract_source_graph_projects_inventory_to_checked_graph() -> None:
    graph = _source_graph()
    graph_nodes = cast(list[dict[str, object]], graph["nodes"])
    graph_edges = cast(list[dict[str, object]], graph["edges"])
    nodes = {node["id"]: node for node in graph_nodes}
    edges = {
        (edge["kind"], edge["source"], edge["target"])
        for edge in graph_edges
    }

    assert graph["format"] == "leibniz.contract-source-graph"
    assert graph["format_version"] == 1
    assert nodes["category:python-contract-runtime"]["kind"] == "category"
    assert nodes["structural-marker:leibniz.records.ContractRuntimeSupport"]["kind"] == (
        "structural-marker"
    )
    assert (
        "has-structural-marker",
        "category:python-contract-runtime",
        "structural-marker:leibniz.records.ContractRuntimeSupport",
    ) in edges
    assert (
        "categorizes",
        "category:python-contract-runtime",
        "path:src/leibniz/records.py",
    ) in edges
    assert (
        "owns-record-spec-module",
        "contract-surface:model-architecture-and-semantics-records",
        "path:src/leibniz/model_interfaces.py",
    ) in edges
    assert (
        "tested-by",
        "contract-surface:console-result-view-records",
        "test:tests/test_console_codegen.py",
    ) in edges
    assert (
        "generated-by",
        "generated-output:src/leibniz/console/_web_src/src/generated/resultViewRecords.ts",
        "contract-surface:console-result-view-records",
    ) in edges


def test_contract_source_graph_cli_outputs_canonical_document() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(_script_path),
            "--repository-root",
            str(_repository_root),
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    graph = load_object_document(
        completed.stdout,
        description="contract source graph",
    )

    assert graph["format"] == "leibniz.contract-source-graph"


def _source_graph() -> dict[str, object]:
    module = _source_graph_module()
    return module.source_graph(
        repository_root=_repository_root,
        status_path=_repository_root / "CONTRACT_GENERATION_STATUS.json",
    )


def _source_graph_module() -> _SourceGraphModule:
    spec = importlib.util.spec_from_file_location("contract_source_graph", _script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return cast(_SourceGraphModule, module)
