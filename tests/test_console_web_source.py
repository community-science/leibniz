import re
import subprocess
from pathlib import Path

from leibniz.cli import _console_dependencies_missing
from leibniz.console.protocol import (
    console_protocol_format_versions,
    console_protocol_formats,
)

_repository_root = Path(__file__).parents[1]
_console_package = (
    _repository_root
    / "src"
    / "leibniz"
    / "console"
    / "web"
)
_generated_source_root = (
    _console_package
    / "src"
    / "generated"
)
_web_source_root = _console_package / "src"
_web_source_suffixes = frozenset({".ts", ".tsx"})


def test_console_web_source_is_handwritten_not_codegen_output() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", str(_generated_source_root.relative_to(_repository_root))],
        check=True,
        cwd=_repository_root,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    assert tracked == []
    assert "src/leibniz/console/web/src/generated/" in (
        _repository_root / ".gitignore"
    ).read_text(encoding="utf-8")
    assert ".local-cache/" in (_repository_root / ".gitignore").read_text(encoding="utf-8")
    assert not (_repository_root / "src" / "leibniz" / "console" / "codegen.py").exists()

    package = (_console_package / "package.json").read_text(encoding="utf-8")
    assert "leibniz.console.codegen" not in package
    assert '"generate"' not in package
    assert '"prebuild"' not in package
    assert '"precheck"' not in package
    assert '"pretest"' not in package
    assert '"pretypecheck"' not in package
    assert (
        '"test": "node ../../../../tests/run_console_web_tests.mjs && npm run browser-smoke"'
        in package
    )
    assert '"browser-smoke": "node ../../../../tests/console_browser_smoke.mjs"' in package
    assert '"playwright":' in package
    assert '"prepare-console-data": "node scripts/prepareConsoleData.mjs"' in package
    assert '"predev": "npm run prepare-console-data"' in package

    protocol_vocabulary = _web_source_root / "protocolVocabulary.ts"
    result_view_records = _web_source_root / "resultViewRecords.ts"
    assert protocol_vocabulary.exists()
    assert result_view_records.exists()
    assert "from './protocolVocabulary.ts'" in result_view_records.read_text(
        encoding="utf-8"
    )
    assert "from '../" not in result_view_records.read_text(encoding="utf-8")

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

    prepare_console_data = (
        _console_package / "scripts" / "prepareConsoleData.mjs"
    ).read_text(encoding="utf-8")
    vite_config = (_console_package / "vite.config.mjs").read_text(encoding="utf-8")
    assert "isConsoleDataPayloadCurrent()" in prepare_console_data
    assert "export function isConsoleDataPayloadCurrent()" in vite_config
    assert "function consoleDataInputFiles()" in vite_config
    assert ".local-cache/console/consoleDataPayload.json" in vite_config
    assert "src/leibniz/console/web/src/generated/consoleDataPayload.json" not in (
        vite_config
    )
    assert "Leibniz console data is current" in prepare_console_data
    assert "headless console browser smoke test timed out" in browser_smoke
    assert "process.exit(0)" in browser_smoke

    console_contract_runner = (
        _repository_root / "tests" / "run_console_web_tests.mjs"
    ).read_text(encoding="utf-8")
    assert "globalThis.consoleDataPayload = JSON.parse(readFileSync(process.argv[1], 'utf8'))" in (
        console_contract_runner
    )
    assert "globalThis.consoleDataPayload = ${payload}" not in console_contract_runner


def test_console_protocol_vocabulary_matches_python_constants() -> None:
    protocol_vocabulary = (_web_source_root / "protocolVocabulary.ts").read_text(
        encoding="utf-8"
    )
    formats = console_protocol_formats()
    versions = console_protocol_format_versions()

    assert formats.console_data in protocol_vocabulary
    assert formats.artifact_index in protocol_vocabulary
    assert formats.benchmark_result_view in protocol_vocabulary
    assert f"'consoleData': {versions.console_data}" in protocol_vocabulary
    assert f"'artifactIndex': {versions.artifact_index}" in protocol_vocabulary
    assert f"'resultView': {versions.result_view}" in protocol_vocabulary


def test_handwritten_web_source_uses_protocol_vocabulary_formats() -> None:
    migrated_literals = (
        "leibniz.console-data",
        "leibniz.console.artifact-index",
        "leibniz.console.benchmark-results",
    )
    allowed = {
        "src/leibniz/console/web/src/protocolVocabulary.ts",
    }

    offenders = tuple(
        path.relative_to(_repository_root).as_posix()
        for path in sorted(_web_source_root.rglob("*"))
        if path.is_file()
        and path.suffix in {".ts", ".tsx"}
        and "generated" not in path.parts
        and path.relative_to(_repository_root).as_posix() not in allowed
        and any(literal in path.read_text(encoding="utf-8") for literal in migrated_literals)
    )

    assert offenders == ()


def test_handwritten_result_view_source_centralizes_record_parsers() -> None:
    migrated_markers = (
        "class ResultViewTransportError",
        "function parseResultViewRecord",
        "function parseBenchmarkResultViewRecord",
        "function parseBenchmarkResult",
        "function parseModelResult",
        "function parseRunResult",
        "function parseTrainingDiagnostics",
    )

    offenders = tuple(
        path.relative_to(_repository_root).as_posix()
        for path in sorted(_web_source_root.rglob("*"))
        if path.is_file()
        and path.suffix in {".ts", ".tsx"}
        and "generated" not in path.parts
        and path.name != "resultViewRecords.ts"
        and any(marker in path.read_text(encoding="utf-8") for marker in migrated_markers)
    )

    assert offenders == ()


def test_console_transport_modules_share_boundary_helpers() -> None:
    transport = _web_source_root / "transport.ts"
    assert transport.exists()
    assert "export function requireRecord" in transport.read_text(encoding="utf-8")

    duplicated_helpers: list[str] = []
    for path in (
        _web_source_root / "benchmarkTasks.ts",
        _web_source_root / "modelInspections.ts",
        _web_source_root / "operatorVocabulary.ts",
    ):
        source = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(_repository_root).as_posix()
        if "from './transport.ts'" not in source:
            duplicated_helpers.append(f"{relative_path}: missing shared transport import")
        for marker in (
            "function requireRecord",
            "function requireArray",
            "function requireString",
            "function requireNumber",
            "function requireLiteral",
        ):
            if marker in source:
                duplicated_helpers.append(f"{relative_path}: {marker}")

    assert duplicated_helpers == []


def test_console_styles_do_not_keep_unused_class_selectors() -> None:
    styles = (_web_source_root / "styles.css").read_text(encoding="utf-8")
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in _handwritten_web_source_files()
        if path.name != "styles.css"
    )
    classes = sorted(set(re.findall(r"\.([A-Za-z_][A-Za-z0-9_-]*)", styles)))

    assert [class_name for class_name in classes if class_name not in source] == []


def test_console_styles_do_not_reference_undefined_custom_properties() -> None:
    styles = (_web_source_root / "styles.css").read_text(encoding="utf-8")
    definitions = set(re.findall(r"^\s*(--[A-Za-z0-9_-]+)\s*:", styles, flags=re.MULTILINE))
    references = set(re.findall(r"var\((--[A-Za-z0-9_-]+)", styles))

    assert sorted(references - definitions) == []


def test_benchmark_dashboard_omits_python_owned_run_detail_sections() -> None:
    dashboard = (
        _web_source_root / "BenchmarkResultDashboard.tsx"
    ).read_text(encoding="utf-8")
    model_inspector = (
        _web_source_root / "BenchmarksPanel.tsx"
    ).read_text(encoding="utf-8")
    offenders = tuple(
        marker
        for marker in ("console_view_model", "RunDetail")
        if marker in dashboard or marker in model_inspector
    )

    assert offenders == ()


def test_benchmark_panel_omits_legacy_sample_and_integral_tables() -> None:
    panel = (_web_source_root / "BenchmarksPanel.tsx").read_text(encoding="utf-8")
    result_view_records = (_web_source_root / "resultViewRecords.ts").read_text(
        encoding="utf-8"
    )
    removed_markers = (
        "BenchmarkTaskPane",
        "BenchmarkSampleCard",
        "BenchmarkSampleCoordinateInspector",
        "CombinedIntegralTermTable",
        "IntegralTermTable",
        "benchmark-sample-window",
        "competence_density",
    )

    assert [marker for marker in removed_markers if marker in panel] == []
    assert "competence_density" not in result_view_records


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
        "src/leibniz/console/web/src/operatorVocabulary.ts",
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


def test_console_dependency_check_reports_uninstalled_toolchain(tmp_path: Path) -> None:
    # No node_modules at all (fresh checkout / wrong directory).
    assert _console_dependencies_missing(tmp_path) is not None

    # node_modules exists but the vite dev toolchain was never installed
    # (e.g. an empty directory left after a source rename).
    (tmp_path / "node_modules").mkdir()
    assert _console_dependencies_missing(tmp_path) is not None

    # A populated install with the dev toolchain present passes.
    (tmp_path / "node_modules" / "vite").mkdir()
    assert _console_dependencies_missing(tmp_path) is None


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
