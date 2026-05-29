import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

from leibniz.documents import load_object_document

_repository_root = Path(__file__).parents[1]
_status_path = _repository_root / "CONTRACT_GENERATION_STATUS.json"
_generated_web_root = Path("src/leibniz/console/_web_src/src/generated")
_source_implementation_suffixes = frozenset({".css", ".html", ".mjs", ".py", ".ts", ".tsx"})
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

        for key in ("record_spec_modules", "python_runtime", "typescript_runtime", "tests"):
            for relative_path in surface[key]:
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
    }


def test_contract_generation_status_categorizes_tracked_code() -> None:
    status = _load_status()
    code_inventory = status["code_inventory"]
    tracked_paths = _tracked_inventory_paths(code_inventory["tracked_roots"])
    line_budget = code_inventory["line_budget"]

    assert line_budget["metric"] == "handwritten_implementation_lines"
    assert line_budget["maximum"] > 0
    assert line_budget["tracked_roots"] == ["src/leibniz"]
    assert line_budget["counting"]

    category_names: set[str] = set()
    patterns_by_category: dict[str, tuple[str, ...]] = {}
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
        if "RecordSpec(" in path.read_text(encoding="utf-8")
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
