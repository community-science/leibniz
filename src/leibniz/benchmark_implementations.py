"""Executable benchmark implementation interfaces and loaders."""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable, Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, cast

from leibniz.benchmarks import BenchmarkManifest
from leibniz.content import ContentDigest
from leibniz.latent_factors import LatentFactorDeclaration
from leibniz.materialization import AxisAssignment, MaterializationDeclaration
from leibniz.observation_formation import ObservationFormationDeclaration
from leibniz.timing import TimingCollector

if TYPE_CHECKING:
    from leibniz.observation_generation import (
        GeneratedSampleSet,
        StateSpaceMeasureRequest,
    )

__all__ = [
    "Benchmark",
    "BenchmarkError",
    "Generator",
    "discover_benchmark_roots",
    "load_benchmark",
]

_entrypoint_filename = "benchmark.py"
_entrypoint_factory = "benchmark"


class BenchmarkError(ValueError):
    """Raised when a benchmark implementation cannot be loaded."""


class Generator(Protocol):
    """Callable benchmark data generator surface."""

    @property
    def id(self) -> object: ...

    @property
    def version(self) -> str: ...

    @property
    def benchmark_manifest(self) -> BenchmarkManifest: ...

    @property
    def latent_factors(self) -> LatentFactorDeclaration: ...

    @property
    def materialization(self) -> MaterializationDeclaration: ...

    @property
    def formation(self) -> ObservationFormationDeclaration: ...

    def minimum_discriminatable_resolution_assignment(
        self,
        *,
        minimum_assignment: AxisAssignment,
    ) -> AxisAssignment: ...

    def distinguishable_state_complexity(
        self,
        *,
        width: int,
        height: int,
        variation_extent: float = 1.0,
    ) -> float: ...

    def __call__(
        self,
        *,
        seed: int,
        shape: int | Sequence[int] | None = None,
        include_fields: bool = False,
        state_space_request: StateSpaceMeasureRequest | None = None,
        component_indices: Iterable[int] | None = None,
        memory_limit_bytes: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
        variation_extent: float = 1.0,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedSampleSet: ...


class Benchmark(Protocol):
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

    @property
    def generator(self) -> Generator: ...


def load_benchmark(benchmark_root: Path) -> Benchmark:
    """Load a benchmark implementation from a benchmark package root."""

    entrypoint = benchmark_root / _entrypoint_filename
    if not entrypoint.is_file():
        raise BenchmarkError(f"{benchmark_root}: missing {_entrypoint_filename}")
    module = _load_entrypoint_module(entrypoint)
    factory = getattr(module, _entrypoint_factory, None)
    if not callable(factory):
        raise BenchmarkError(
            f"{entrypoint}: expected callable {_entrypoint_factory}"
        )
    implementation = factory(benchmark_root)
    _validate_benchmark_implementation(implementation, entrypoint=entrypoint)
    return cast(Benchmark, implementation)


def discover_benchmark_roots(benchmark_parent: Path) -> tuple[Path, ...]:
    """Return benchmark roots beneath a benchmark package parent."""

    if not benchmark_parent.is_dir():
        return ()
    return tuple(
        path
        for path in sorted(benchmark_parent.iterdir())
        if path.is_dir() and _is_benchmark_root(path)
    )


def _is_benchmark_root(path: Path) -> bool:
    return (path / _entrypoint_filename).is_file()


def _load_entrypoint_module(entrypoint: Path) -> ModuleType:
    module_name = "leibniz_benchmark_" + ContentDigest.from_value(
        {"entrypoint": entrypoint.resolve().as_posix()}
    ).hex
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    if spec is None or spec.loader is None:
        raise BenchmarkError(f"{entrypoint}: could not load module spec")
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
            raise BenchmarkError(
                f"{entrypoint}: benchmark implementation missing {name}"
            ) from error
    try:
        _generator = cast(Any, value).generator
    except Exception as error:
        raise BenchmarkError(
            f"{entrypoint}: benchmark implementation missing generator"
        ) from error
    if not callable(_generator):
        raise BenchmarkError(
            f"{entrypoint}: benchmark implementation generator must be callable"
        )
