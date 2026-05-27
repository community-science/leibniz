import subprocess
from pathlib import Path

from leibniz.console.codegen import (
    generated_console_protocol_module,
    generated_console_result_view_records_module,
)

_repository_root = Path(__file__).parents[1]
_console_package = (
    _repository_root
    / "src"
    / "leibniz"
    / "console"
    / "_web_src"
)
_generated_source_root = (
    _console_package
    / "src"
    / "generated"
)
_web_source_root = _console_package / "src"


def test_generated_console_protocol_module_contains_python_owned_formats() -> None:
    generated = generated_console_protocol_module()
    assert "consoleProtocolFormats" in generated
    assert "leibniz.console.benchmark-results" in generated


def test_generated_console_result_view_records_module_contains_parser_surface() -> None:
    generated = generated_console_result_view_records_module()
    assert "export type BenchmarkResultRecord" in generated
    assert "export function parseResultViewRecords" in generated
    assert "function parseWorkQueueItem" in generated


def test_generated_console_web_modules_are_npm_build_artifacts() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", str(_generated_source_root.relative_to(_repository_root))],
        check=True,
        cwd=_repository_root,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    assert tracked == []

    package = (_console_package / "package.json").read_text(encoding="utf-8")
    assert '"generate": "python -m leibniz.console.codegen"' in package
    for lifecycle in ("prebuild", "precheck", "predev", "pretest", "pretypecheck"):
        assert f'"{lifecycle}": "npm run generate"' in package


def test_handwritten_web_source_uses_generated_protocol_formats() -> None:
    migrated_literals = (
        "leibniz.console-data",
        "leibniz.console.artifact-index",
        "leibniz.console.imported-results",
        "leibniz.console.benchmark-results",
        "leibniz.console.work-queue",
        "leibniz.work-queue-item",
    )

    offenders = tuple(
        path.relative_to(_repository_root).as_posix()
        for path in sorted(_web_source_root.rglob("*"))
        if path.is_file()
        and path.suffix in {".ts", ".tsx"}
        and "generated" not in path.parts
        and any(literal in path.read_text(encoding="utf-8") for literal in migrated_literals)
    )

    assert offenders == ()


def test_handwritten_result_view_source_uses_generated_record_parsers() -> None:
    migrated_markers = (
        "class ResultViewTransportError",
        "function parseResultViewRecord",
        "function parseImportedResultViewRecord",
        "function parseBenchmarkResultViewRecord",
        "function parseWorkQueueViewRecord",
        "function parseWorkQueueItem",
        "function parseBenchmarkResult",
        "function parseModelResult",
        "function parseRunResult",
        "function parseTrainingDiagnostics",
        "function parseProposal",
    )

    offenders = tuple(
        path.relative_to(_repository_root).as_posix()
        for path in sorted(_web_source_root.rglob("*"))
        if path.is_file()
        and path.suffix in {".ts", ".tsx"}
        and "generated" not in path.parts
        and any(marker in path.read_text(encoding="utf-8") for marker in migrated_markers)
    )

    assert offenders == ()
