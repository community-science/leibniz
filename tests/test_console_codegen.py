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
_web_source_suffixes = frozenset({".ts", ".tsx"})


def test_generated_console_protocol_module_contains_python_owned_formats() -> None:
    generated = generated_console_protocol_module()
    assert "consoleProtocolFormats" in generated
    assert "leibniz.console.benchmark-results" in generated


def test_generated_console_result_view_records_module_contains_parser_surface() -> None:
    generated = generated_console_result_view_records_module()
    assert "export type BenchmarkResultRecord" in generated
    assert "export function parseResultViewRecords" in generated
    assert "console_view_model?: RunDetailViewModelRecord;" in generated
    assert "parseRunDetailViewModel(record.console_view_model" in generated
    assert "component_count: requireNumber(record.component_count" in generated
    assert "layer_count" not in generated
    assert "function parseWorkQueueItem" in generated
    assert "export type RunDetailViewModelRecord" in generated
    assert "function parseRunDetailViewModel" in generated


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
    assert (
        '"test": "node ../../../../tests/run_console_web_tests.mjs && npm run browser-smoke"'
        in package
    )
    assert '"browser-smoke": "node ../../../../tests/console_browser_smoke.mjs"' in package
    assert '"playwright":' in package
    for lifecycle in ("prebuild", "precheck", "predev", "pretest", "pretypecheck"):
        assert f'"{lifecycle}": "npm run generate"' in package

    browser_smoke = (_repository_root / "tests" / "console_browser_smoke.mjs").read_text(
        encoding="utf-8"
    )
    assert "LEIBNIZ_CONSOLE_RESULT_ROOTS" in browser_smoke
    assert "await runConsoleCommand('npm', ['run', 'build']" in browser_smoke
    assert "chromium.launch({ headless: true })" in browser_smoke
    assert "async function stopPreview" in browser_smoke
    assert "detached: process.platform !== 'win32'" in browser_smoke
    assert "process.kill(-child.pid, signal)" in browser_smoke
    assert "killProcessGroup(child, 'SIGKILL')" in browser_smoke
    assert "headless console browser smoke test timed out" in browser_smoke
    assert "process.exit(0)" in browser_smoke

    console_contract_runner = (
        _repository_root / "tests" / "run_console_web_tests.mjs"
    ).read_text(encoding="utf-8")
    assert "globalThis.consoleDataPayload = JSON.parse(readFileSync(process.argv[1], 'utf8'))" in (
        console_contract_runner
    )
    assert "globalThis.consoleDataPayload = ${payload}" not in console_contract_runner


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


def test_benchmark_dashboard_renders_python_owned_run_detail_sections() -> None:
    dashboard = (
        _web_source_root / "BenchmarkResultDashboard.tsx"
    ).read_text(encoding="utf-8")
    migrated_markers = (
        "sampled_competence",
        "training_diagnostics",
        "validation_history",
        "best_validation_loss",
        "validation_source",
    )

    offenders = tuple(marker for marker in migrated_markers if marker in dashboard)

    assert "console_view_model?.detail_sections" in dashboard
    assert offenders == ()


def test_console_artifact_kind_literals_stay_inside_detail_boundary() -> None:
    artifact_kind_literals = (
        "architecture-manifest",
        "benchmark-manifest",
        "latent-factor-declaration",
        "materialization-declaration",
        "materialization-plan",
        "measurement",
        "observation-formation-declaration",
        "observation-showcase",
    )
    allowed = {
        "src/leibniz/console/_web_src/src/ArtifactTypedDetail.tsx",
        "src/leibniz/console/_web_src/src/artifactDetails.ts",
    }

    offenders = tuple(
        path.relative_to(_repository_root).as_posix()
        for path in _handwritten_web_source_files()
        if path.relative_to(_repository_root).as_posix() not in allowed
        and any(
            _typescript_string_literal(source=path.read_text(encoding="utf-8"), value=literal)
            for literal in artifact_kind_literals
        )
    )

    assert offenders == ()


def test_handwritten_console_source_avoids_migrated_protocol_literals() -> None:
    migrated_literals = (
        "operator.0.local_support_size",
        "operator.0.support",
        "local-aggregation",
        "rank-collapse",
        "affine-readout",
        "adaptive-pooling",
        "generated-observations",
        "benchmarks.digits",
        "benchmarks.digits@0.1.0",
    )
    allowed = {
        "src/leibniz/console/_web_src/src/operatorVocabulary.ts",
    }

    offenders = tuple(
        path.relative_to(_repository_root).as_posix()
        for path in _handwritten_web_source_files()
        if path.relative_to(_repository_root).as_posix() not in allowed
        and any(
            _typescript_string_literal(source=path.read_text(encoding="utf-8"), value=literal)
            for literal in migrated_literals
        )
    )

    assert offenders == ()


def _handwritten_web_source_files() -> tuple[Path, ...]:
    return tuple(
        path
        for path in sorted(_web_source_root.rglob("*"))
        if path.is_file()
        and path.suffix in _web_source_suffixes
        and "generated" not in path.parts
    )


def _typescript_string_literal(*, source: str, value: str) -> bool:
    return f"'{value}'" in source or f'"{value}"' in source
