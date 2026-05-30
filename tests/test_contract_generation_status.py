import ast
import importlib
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, cast

from leibniz.contracts import ContractRuntimeSupport
from leibniz.documents import load_object_document
from leibniz.records import (
    FieldSpec,
    RecordExtractor,
    RecordSpec,
    RecordViolation,
)

_repository_root = Path(__file__).parents[1]
_status_path = _repository_root / "CONTRACT_GENERATION_STATUS.json"
_generated_web_root = Path("src/leibniz/console/_web_src/src/generated")
_source_implementation_suffixes = frozenset({".css", ".html", ".mjs", ".py", ".ts", ".tsx"})
_excluded_line_budget_bases = frozenset(
    {
        "authored-data",
        "authored-contract",
        "build-configuration",
        "codegen-runtime",
        "contract-runtime",
        "inventory",
        "package-metadata",
        "repository-automation",
        "tests",
    }
)
_coverage_keys = frozenset(
    {
        "authored_contract",
        "generated_python_runtime",
        "generated_typescript_runtime",
        "generated_conformance_tests",
    }
)


def test_contract_generation_status_is_complete_and_well_formed() -> None:
    status = _load_status()

    assert status["format"] == "leibniz.contract-generation-status"
    assert status["format_version"] == 1
    assert set(status["coverage_keys"]) == _coverage_keys
    assert status["code_inventory"]["tracked_roots"]

    names: set[str] = set()
    covered_record_spec_modules: list[str] = []
    for surface in status["surfaces"]:
        name = surface["name"]
        assert name not in names
        names.add(name)
        assert surface["scope"]
        assert surface["contract_owner"] in {
            "mixed-python-and-typescript-runtime",
            "python-owned-codegen",
            "python-runtime-coupled",
        }
        assert set(surface["coverage"]) == _coverage_keys
        assert all(isinstance(value, bool) for value in surface["coverage"].values())
        assert surface["ratchet_next"]
        assert surface["tests"]

        for key in (
            "authored_contracts",
            "record_spec_modules",
            "python_runtime",
            "typescript_runtime",
            "tests",
        ):
            paths_value = surface.get(key, [])
            assert isinstance(paths_value, list)
            paths = cast(list[object], paths_value)
            for relative_path in [str(path) for path in paths]:
                _assert_existing_repository_path(relative_path)

        for relative_path in surface["record_spec_modules"]:
            assert relative_path in surface["python_runtime"]
        covered_record_spec_modules.extend(surface["record_spec_modules"])

        for relative_path in surface["generated_outputs"]:
            generated_path = Path(relative_path)
            assert _generated_web_root in (generated_path, *generated_path.parents)

    assert sorted(covered_record_spec_modules) == _record_spec_modules()
    assert _typescript_generation_surfaces(status) == {
        "console-protocol-vocabulary",
        "console-result-view-records",
        "work-queue-item-records",
    }


def test_contract_generation_status_categorizes_tracked_code() -> None:
    status = _load_status()
    code_inventory = status["code_inventory"]
    tracked_paths = _tracked_inventory_paths(code_inventory["tracked_roots"])
    line_budget = code_inventory["line_budget"]
    graph_projection = code_inventory["graph_projection"]

    assert line_budget["metric"] == "handwritten_implementation_lines"
    assert line_budget["maximum"] > 0
    assert line_budget["tracked_roots"] == ["src/leibniz"]
    assert line_budget["counting"]
    assert set(graph_projection["node_kinds"]) == {
        "category",
        "contract-surface",
        "generated-output",
        "path",
        "record-spec",
        "structural-marker",
        "test",
    }
    assert set(graph_projection["edge_kinds"]) == {
        "categorizes",
        "covers-record-spec",
        "declares-record-spec",
        "generated-by",
        "has-structural-marker",
        "owns-record-spec-module",
        "tested-by",
        "uses-authored-contract",
        "uses-runtime",
    }
    assert _contract_runtime_paths(code_inventory["categories"]) == [
        "src/leibniz/contracts.py",
        "src/leibniz/record_contracts.py",
        "src/leibniz/records.py",
    ]
    assert _contract_runtime_types() == {
        "FieldSpec",
        "RecordExtractor",
        "RecordSpec",
        "RecordViolation",
    }
    assert _contract_runtime_roles() == {
        "field-spec",
        "record-extractor",
        "record-spec",
        "record-violation",
    }

    category_names: set[str] = set()
    patterns_by_category: dict[str, tuple[str, ...]] = {}
    authored_contract_paths = _surface_paths(status, key="authored_contracts")
    generated_output_paths = _surface_paths(status, key="generated_outputs")
    for category in code_inventory["categories"]:
        name = category["name"]
        assert name not in category_names
        category_names.add(name)
        assert category["scope"]
        assert category["status"] in {
            "hand-authored-data",
            "hand-authored-contract-inventory",
            "hand-maintained",
            "mixed-hand-maintained-and-codegen",
            "mixed-hand-maintained-and-generated-consumer",
        }
        assert category["line_budget"] in {"excluded", "handwritten-implementation"}
        assert category["line_budget_basis"]
        if category["line_budget"] == "handwritten-implementation":
            assert category["line_budget_basis"] == "domain-implementation"
        else:
            assert category["line_budget_basis"] in _excluded_line_budget_bases
            _assert_excluded_category_basis(
                category=category,
                matched_paths=_paths_matching_patterns(
                    tracked_paths=tracked_paths,
                    patterns=tuple(category["path_patterns"]),
                ),
                authored_contract_paths=authored_contract_paths,
                generated_output_paths=generated_output_paths,
            )
        assert category["ratchet_next"]
        patterns = tuple(category["path_patterns"])
        assert patterns
        patterns_by_category[name] = patterns

    uncategorized: list[str] = []
    multiply_categorized: dict[str, list[str]] = {}
    for relative_path in tracked_paths:
        matches = [
            name
            for name, patterns in patterns_by_category.items()
            if any(PurePosixPath(relative_path).match(pattern) for pattern in patterns)
        ]
        if not matches:
            uncategorized.append(relative_path)
        elif len(matches) > 1:
            multiply_categorized[relative_path] = matches

    assert uncategorized == []
    assert multiply_categorized == {}
    budgeted_paths = _handwritten_implementation_paths(
        tracked_paths=_tracked_inventory_paths(line_budget["tracked_roots"]),
        categories=code_inventory["categories"],
    )
    assert all(Path(path).suffix in _source_implementation_suffixes for path in budgeted_paths)
    assert "src/leibniz/console/_web_src/package-lock.json" not in budgeted_paths
    assert "src/leibniz/console/_web_src/package.json" not in budgeted_paths
    assert _handwritten_implementation_lines(
        budgeted_paths=budgeted_paths,
    ) <= line_budget["maximum"]


def test_contract_generation_status_generated_outputs_are_build_artifacts() -> None:
    status = _load_status()
    generated_outputs = sorted(
        {
            output
            for surface in status["surfaces"]
            for output in surface["generated_outputs"]
        }
    )

    tracked_outputs = subprocess.run(
        ["git", "ls-files", *generated_outputs],
        check=True,
        cwd=_repository_root,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()

    assert tracked_outputs == []
    assert "src/leibniz/console/_web_src/src/generated/" in (
        _repository_root / ".gitignore"
    ).read_text(encoding="utf-8")


def _load_status() -> dict[str, Any]:
    return dict(
        load_object_document(
            _status_path.read_bytes(),
            description="contract generation status",
        )
    )


def _assert_existing_repository_path(relative_path: str) -> None:
    path = _repository_root / relative_path
    assert path.exists(), relative_path
    assert path.is_file(), relative_path


def _record_spec_modules() -> list[str]:
    return sorted(
        path.relative_to(_repository_root).as_posix()
        for path in (_repository_root / "src" / "leibniz").rglob("*.py")
        if path.name != "records.py" and _contains_record_spec_call(path)
    )


def _contains_record_spec_call(path: Path) -> bool:
    syntax_tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RecordSpec"
        for node in ast.walk(syntax_tree)
    )


def _tracked_inventory_paths(tracked_roots: list[str]) -> list[str]:
    tracked = subprocess.run(
        ["git", "ls-files", *tracked_roots],
        check=True,
        cwd=_repository_root,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    return [
        path
        for path in tracked
        if "/node_modules/" not in path
        and "/dist/" not in path
        and "/generated/" not in path
    ]


def _handwritten_implementation_lines(
    *,
    budgeted_paths: list[str],
) -> int:
    return sum(
        _nonblank_line_count(_repository_root / relative_path)
        for relative_path in budgeted_paths
    )


def _handwritten_implementation_paths(
    *,
    tracked_paths: list[str],
    categories: list[dict[str, Any]],
) -> list[str]:
    line_budget_by_pattern = {
        pattern: category["line_budget"]
        for category in categories
        for pattern in category["path_patterns"]
    }
    budgeted_paths: list[str] = []
    for relative_path in tracked_paths:
        path = PurePosixPath(relative_path)
        if not any(
            path.match(pattern) and budget == "handwritten-implementation"
            for pattern, budget in line_budget_by_pattern.items()
        ):
            continue
        budgeted_paths.append(relative_path)
    return budgeted_paths


def _contract_runtime_paths(categories: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for category in categories:
        if category["name"] == "python-contract-runtime":
            paths.extend(category["path_patterns"])
    return sorted(paths)


def _contract_runtime_types() -> set[str]:
    contract_runtime_types = {
        FieldSpec,
        RecordExtractor,
        RecordSpec,
        RecordViolation,
    }
    assert all(issubclass(type_, ContractRuntimeSupport) for type_ in contract_runtime_types)
    return {type_.__name__ for type_ in contract_runtime_types}


def _contract_runtime_roles() -> set[str]:
    return {
        FieldSpec(kind="string").contract_runtime_role,
        RecordExtractor().contract_runtime_role,
        RecordSpec(fields={}).contract_runtime_role,
        RecordViolation(path=(), message="example").contract_runtime_role,
    }


def _nonblank_line_count(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _typescript_generation_surfaces(status: dict[str, Any]) -> set[str]:
    return {
        surface["name"]
        for surface in status["surfaces"]
        if surface["coverage"]["generated_typescript_runtime"]
    }


def _surface_paths(status: dict[str, Any], *, key: str) -> set[str]:
    return {
        str(path)
        for surface in status["surfaces"]
        for path in cast(list[object], surface.get(key, []))
    }


def _assert_excluded_category_basis(
    *,
    category: dict[str, Any],
    matched_paths: set[str],
    authored_contract_paths: set[str],
    generated_output_paths: set[str],
) -> None:
    basis = category["line_budget_basis"]
    if basis in {"contract-runtime", "codegen-runtime"}:
        marker = str(category["structural_marker"])
        marker_object = _resolve_structural_marker(marker)
        if basis == "contract-runtime":
            assert isinstance(marker_object, type)
            assert issubclass(marker_object, ContractRuntimeSupport)
        else:
            assert callable(marker_object)
    elif basis == "authored-contract":
        assert matched_paths <= authored_contract_paths
        for relative_path in matched_paths:
            document = load_object_document(
                (_repository_root / relative_path).read_bytes(),
                description=relative_path,
            )
            assert document["format"] == "leibniz.record-contract-set"
            assert document["format_version"] == 1
    elif basis == "authored-data":
        assert matched_paths.isdisjoint(generated_output_paths)
    elif basis == "build-configuration":
        assert all(
            "package" in path or "config" in path or "tsconfig" in path
            for path in matched_paths
        )
    elif basis == "inventory":
        assert matched_paths == {"CONTRACT_GENERATION_STATUS.json"}
    elif basis == "package-metadata":
        assert all(
            path.endswith("__init__.py") or path.endswith("py.typed")
            for path in matched_paths
        )
    elif basis == "repository-automation":
        assert all(
            path.startswith((".github/", "scripts/")) or path == "pyproject.toml"
            for path in matched_paths
        )
    elif basis == "tests":
        assert all(path.startswith("tests/") for path in matched_paths)
    else:
        raise AssertionError(f"unsupported excluded line-budget basis: {basis}")


def _paths_matching_patterns(
    *,
    tracked_paths: list[str],
    patterns: tuple[str, ...],
) -> set[str]:
    return {
        relative_path
        for relative_path in tracked_paths
        if any(PurePosixPath(relative_path).match(pattern) for pattern in patterns)
    }


def _resolve_structural_marker(marker: str) -> object:
    module_name, _, attribute_name = marker.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attribute_name)
