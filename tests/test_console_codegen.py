import re
import subprocess
from pathlib import Path

from leibniz.console.codegen import (
    generated_console_protocol_module,
    generated_console_result_view_records_module,
    generated_console_work_queue_records_module,
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
    assert "from '../transport.ts'" in generated
    assert "parseImportedPublicationBundleRecord" in generated
    assert "proposals: arrayOf(record.proposals ?? []" in generated
    assert "parseRunDetailViewModel" in generated
    assert "arrayOf(record.model_inspections ?? []" in generated
    assert "layer_count" not in generated
    assert "parseWorkQueueItem(item" in generated
    assert "from './workQueueRecords.ts'" in generated
    assert "export type RunDetailViewModelRecord" in generated


def test_generated_console_work_queue_records_module_uses_authored_contract() -> None:
    generated = generated_console_work_queue_records_module()
    assert "export type WorkQueueItemRecord" in generated
    assert "export function parseWorkQueueItem" in generated
    assert "export type WorkQueueItemStatus = 'pending' | 'reserved' | 'completed' | 'failed';" in (
        generated
    )
    assert "rejectUnknownFields(record, path" in generated
    assert "sequence: requireInteger(record.sequence" in generated
    assert "leibniz.work-queue-item" not in generated
    assert "function requireBoolean" not in generated
    assert "function requireNumber" not in generated


def test_generated_console_web_modules_are_npm_build_artifacts() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", str(_generated_source_root.relative_to(_repository_root))],
        check=True,
        cwd=_repository_root,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    assert tracked == []
    assert "src/leibniz/console/_web_src/src/generated/" in (
        _repository_root / ".gitignore"
    ).read_text(encoding="utf-8")

    package = (_console_package / "package.json").read_text(encoding="utf-8")
    assert '"generate": "python -m leibniz.console.codegen"' in package
    assert (
        '"test": "node ../../../../tests/run_console_web_tests.mjs && npm run browser-smoke"'
        in package
    )
    assert '"browser-smoke": "node ../../../../tests/console_browser_smoke.mjs"' in package
    assert '"playwright":' in package
    assert '"prepare-console-data": "node scripts/prepareConsoleData.mjs"' in package
    assert '"predev": "npm run generate && npm run prepare-console-data"' in package
    for lifecycle in ("prebuild", "precheck", "pretest", "pretypecheck"):
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

    prepare_console_data = (
        _console_package / "scripts" / "prepareConsoleData.mjs"
    ).read_text(encoding="utf-8")
    assert "function isPrepared(fingerprint)" in prepare_console_data
    assert (
        "const metadataPath = `${consoleDataPayloadPath()}.metadata.json`"
        in prepare_console_data
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
