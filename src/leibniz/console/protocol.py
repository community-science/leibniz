"""Python-owned protocol metadata shared with the browser console."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ConsoleProtocolFormatVersions",
    "ConsoleProtocolFormats",
    "console_protocol_format_versions",
    "console_protocol_formats",
]


@dataclass(frozen=True, slots=True)
class ConsoleProtocolFormats:
    """Format literals for console transport records."""

    console_data: str = "leibniz.console-data"
    artifact_index: str = "leibniz.console.artifact-index"
    imported_result_view: str = "leibniz.console.imported-results"
    benchmark_result_view: str = "leibniz.console.benchmark-results"


@dataclass(frozen=True, slots=True)
class ConsoleProtocolFormatVersions:
    """Format versions for console transport records."""

    console_data: int = 1
    artifact_index: int = 1
    result_view: int = 1


def console_protocol_formats() -> ConsoleProtocolFormats:
    """Return Python-owned console transport format literals."""

    return ConsoleProtocolFormats()


def console_protocol_format_versions() -> ConsoleProtocolFormatVersions:
    """Return Python-owned console transport format versions."""

    return ConsoleProtocolFormatVersions()
