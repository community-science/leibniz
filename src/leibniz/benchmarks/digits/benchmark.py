"""Digits benchmark implementation entry point."""

from __future__ import annotations

from pathlib import Path

from leibniz.benchmark_implementations import (
    BenchmarkImplementation,
    DeclarationBackedBenchmarkImplementation,
)

__all__ = ["benchmark"]


def benchmark(root: Path) -> BenchmarkImplementation:
    """Return the Digits benchmark implementation."""

    return DeclarationBackedBenchmarkImplementation(root)
