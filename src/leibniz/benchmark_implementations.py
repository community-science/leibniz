"""Executable benchmark implementation interfaces and loaders."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Protocol, cast

from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestDocument
from leibniz.content import ContentDigest
from leibniz.documents import document_filename_suffix
from leibniz.latent_factors import (
    LatentFactorDeclaration,
    LatentFactorDeclarationDocument,
)
from leibniz.materialization import (
    MaterializationDeclaration,
    MaterializationDeclarationDocument,
)
from leibniz.observation_formation import (
    ObservationFormationDeclaration,
    ObservationFormationDeclarationDocument,
)

__all__ = [
    "BenchmarkImplementation",
    "BenchmarkImplementationError",
    "DeclarationBackedBenchmarkImplementation",
    "load_benchmark_implementation",
]

_document_suffix = document_filename_suffix()
_entrypoint_filename = "benchmark.py"
_entrypoint_factory = "benchmark"


class BenchmarkImplementationError(ValueError):
    """Raised when a benchmark implementation cannot be loaded."""


class BenchmarkImplementation(Protocol):
    """Benchmark-owned executable behavior used by generic Leibniz evaluators."""

    @property
    def root(self) -> Path: ...

    @property
    def benchmark_manifest(self) -> BenchmarkManifest: ...

    @property
    def latent_factors(self) -> LatentFactorDeclaration: ...

    @property
    def materialization(self) -> MaterializationDeclaration: ...

    @property
    def formation(self) -> ObservationFormationDeclaration: ...

    def observation_generator(self) -> object: ...


class DeclarationBackedBenchmarkImplementation:
    """Benchmark implementation backed by declaration documents in a benchmark root."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._benchmark_manifest: BenchmarkManifest | None = None
        self._latent_factors: LatentFactorDeclaration | None = None
        self._materialization: MaterializationDeclaration | None = None
        self._formation: ObservationFormationDeclaration | None = None

    @property
    def root(self) -> Path:
        return self._root

    @property
    def benchmark_manifest(self) -> BenchmarkManifest:
        if self._benchmark_manifest is None:
            self._benchmark_manifest = BenchmarkManifestDocument.from_bytes(
                self._document_bytes("manifest")
            ).manifest
        return self._benchmark_manifest

    @property
    def latent_factors(self) -> LatentFactorDeclaration:
        if self._latent_factors is None:
            self._latent_factors = LatentFactorDeclarationDocument.from_bytes(
                self._document_bytes("latent_factors")
            ).declaration
        return self._latent_factors

    @property
    def materialization(self) -> MaterializationDeclaration:
        if self._materialization is None:
            self._materialization = MaterializationDeclarationDocument.from_bytes(
                self._document_bytes("materialization")
            ).declaration
        return self._materialization

    @property
    def formation(self) -> ObservationFormationDeclaration:
        if self._formation is None:
            self._formation = ObservationFormationDeclarationDocument.from_bytes(
                self._document_bytes("observation_formation")
            ).declaration
        return self._formation

    def observation_generator(self) -> object:
        from leibniz.observation_generation import ObservationGenerator

        return ObservationGenerator(
            benchmark_manifest=self.benchmark_manifest,
            latent_factors=self.latent_factors,
            materialization=self.materialization,
            formation=self.formation,
        )

    def _document_bytes(self, stem: str) -> bytes:
        return (self.root / (stem + _document_suffix)).read_bytes()


def load_benchmark_implementation(benchmark_root: Path) -> BenchmarkImplementation:
    """Load a benchmark implementation from a benchmark package root."""

    entrypoint = benchmark_root / _entrypoint_filename
    if not entrypoint.is_file():
        return DeclarationBackedBenchmarkImplementation(benchmark_root)
    module = _load_entrypoint_module(entrypoint)
    factory = getattr(module, _entrypoint_factory, None)
    if not callable(factory):
        raise BenchmarkImplementationError(
            f"{entrypoint}: expected callable {_entrypoint_factory}"
        )
    implementation = factory(benchmark_root)
    _validate_benchmark_implementation(implementation, entrypoint=entrypoint)
    return cast(BenchmarkImplementation, implementation)


def _load_entrypoint_module(entrypoint: Path) -> ModuleType:
    module_name = "leibniz_benchmark_" + ContentDigest.from_value(
        {"entrypoint": entrypoint.resolve().as_posix()}
    ).hex
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
        raise BenchmarkImplementationError(f"{entrypoint}: could not load module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_benchmark_implementation(
    value: object,
    *,
    entrypoint: Path,
) -> None:
    for name in (
        "root",
        "benchmark_manifest",
        "latent_factors",
        "materialization",
        "formation",
    ):
        try:
            getattr(value, name)
        except Exception as error:
            raise BenchmarkImplementationError(
                f"{entrypoint}: benchmark implementation missing {name}"
            ) from error
    method = getattr(value, "observation_generator", None)
    if not callable(method):
        raise BenchmarkImplementationError(
            f"{entrypoint}: benchmark implementation missing observation_generator"
        )
