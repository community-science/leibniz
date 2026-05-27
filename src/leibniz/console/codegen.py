"""Generate small console web-source modules from Python-owned protocol metadata."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from leibniz.console.protocol import (
    console_protocol_format_versions,
    console_protocol_formats,
)

__all__ = [
    "generated_console_protocol_module",
    "write_generated_console_protocol_module",
]

_generated_module_path = (
    Path(__file__).parent / "_web_src" / "src" / "generated" / "protocolVocabulary.ts"
)


def generated_console_protocol_module() -> str:
    """Return the generated TypeScript console protocol vocabulary module."""

    formats = _console_protocol_formats()
    versions = _console_protocol_format_versions()
    return (
        "export const consoleProtocolFormats = "
        f"{_typescript_literal(formats)} as const;\n\n"
        "export const consoleProtocolFormatVersions = "
        f"{_typescript_literal(versions)} as const;\n"
    )


def write_generated_console_protocol_module(
    path: Path = _generated_module_path,
) -> Path:
    """Write the generated TypeScript console protocol vocabulary module."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(generated_console_protocol_module(), encoding="utf-8")
    return path


def _console_protocol_formats() -> Mapping[str, object]:
    formats = console_protocol_formats()
    return {
        "consoleData": formats.console_data,
        "artifactIndex": formats.artifact_index,
        "resultViews": {
            "importedResults": formats.imported_result_view,
            "benchmarkResults": formats.benchmark_result_view,
            "workQueue": formats.work_queue_view,
        },
        "workQueueItem": formats.work_queue_item,
    }


def _console_protocol_format_versions() -> Mapping[str, object]:
    versions = console_protocol_format_versions()
    return {
        "consoleData": versions.console_data,
        "artifactIndex": versions.artifact_index,
        "resultView": versions.result_view,
        "workQueueItem": versions.work_queue_item,
    }


def _typescript_literal(value: object) -> str:
    return _typescript_literal_lines(value, indent=0)


def _typescript_literal_lines(value: object, *, indent: int) -> str:
    prefix = " " * indent
    child_indent = indent + 2
    child_prefix = " " * child_indent
    if isinstance(value, str):
        return _typescript_string(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        if not value:
            return "{}"
        lines = ["{"]
        items = sorted(mapping.items())
        for index, (key, item) in enumerate(items):
            suffix = "," if index < len(items) - 1 else ""
            rendered = _typescript_literal_lines(item, indent=child_indent)
            lines.append(f"{child_prefix}{_typescript_string(str(key))}: {rendered}{suffix}")
        lines.append(f"{prefix}}}")
        return "\n".join(lines)
    raise TypeError(f"unsupported generated TypeScript value: {type(value).__name__}")


def _typescript_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate console web-source modules.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the generated module is not up to date",
    )
    parser.add_argument(
        "--path",
        default=_generated_module_path,
        type=Path,
        help="generated TypeScript module path",
    )
    args = parser.parse_args(argv)

    expected = generated_console_protocol_module()
    if args.check:
        actual = args.path.read_text(encoding="utf-8")
        if actual != expected:
            raise SystemExit(f"{args.path}: generated console protocol module is out of date")
        return 0
    write_generated_console_protocol_module(args.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
