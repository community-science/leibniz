"""Executable benchmark implementation interfaces and loaders."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, cast

from leibniz.benchmarks import BenchmarkManifest
from leibniz.content import ContentDigest
from leibniz.timing import TimingCollector

if TYPE_CHECKING:
    from leibniz.observation_generation import (
        GeneratedSampleSet,
        StateSpaceCandidate,
        StateSpaceMeasureRequest,
        StateSpaceMeasureValue,
    )
    from leibniz.tensor_runtime import TensorRuntime

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
    def manifest(self) -> BenchmarkManifest: ...

    def __call__(
        self,
        *,
        seed: int,
        shape: int | Sequence[int] | None = None,
        include_fields: bool = False,
        include_metadata: bool = True,
        state_space_request: StateSpaceMeasureRequest | None = None,
        runtime: TensorRuntime | None = None,
        outcome_ids: tuple[str, ...] | None = None,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedSampleSet: ...

    def minimum_state_space_measure(self) -> StateSpaceMeasureValue: ...

    def complexity_rung_size(self) -> float:
        """Return the log2 state-space complexity width for curriculum rungs."""
        ...

    def state_space_for_request(
        self,
        *,
        request: StateSpaceMeasureRequest,
    ) -> StateSpaceCandidate | None: ...

    def state_spaces_for_request(
        self,
        *,
        request: StateSpaceMeasureRequest,
    ) -> Sequence[StateSpaceCandidate]: ...


class Benchmark(Protocol):
    """Benchmark-owned executable behavior used by generic Leibniz evaluators."""

    @property
    def root(self) -> Path: ...

    @property
    def manifest(self) -> BenchmarkManifest: ...

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
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _validate_benchmark_implementation(
    value: object,
    *,
    entrypoint: Path,
) -> None:
    for name in ("root", "manifest"):
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
