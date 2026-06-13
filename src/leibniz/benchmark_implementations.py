"""Executable benchmark implementation interfaces and loaders."""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol, cast

from leibniz.benchmarks import BenchmarkManifest
from leibniz.content import ContentDigest
from leibniz.state_space import AccessibleSubspace, SamplingProtocol
from leibniz.target_contracts import TargetContract
from leibniz.timing import TimingCollector

if TYPE_CHECKING:
    from leibniz.observation_generation import (
        GeneratedSampleSet,
        StateSpaceVolumeRequest,
        StateSpaceVolumeValue,
    )
    from leibniz.tensor_runtime import TensorRuntime

__all__ = [
    "Benchmark",
    "BenchmarkError",
    "Generator",
    "RawBenchmark",
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
        include_artifacts: bool = False,
        volume_request: StateSpaceVolumeRequest | None = None,
        sample_indices: Sequence[int] | None = None,
        runtime: TensorRuntime | None = None,
        outcome_ids: tuple[str, ...] | None = None,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedSampleSet: ...

    def minimum_log2_volume(self) -> StateSpaceVolumeValue: ...


class RawBenchmark(Protocol):
    """Benchmark-owned executable behavior before loader-derived contracts."""

    @property
    def root(self) -> Path: ...

    @property
    def manifest(self) -> BenchmarkManifest: ...

    @property
    def generator(self) -> Generator: ...

    @property
    def sampling_protocol(self) -> SamplingProtocol: ...

    @property
    def accessible_subspace(self) -> AccessibleSubspace: ...


class Benchmark(RawBenchmark, Protocol):
    """Benchmark-owned executable behavior used by generic Leibniz evaluators."""

    @property
    def target_contract(self) -> TargetContract: ...


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
    target_contract = getattr(implementation, "target_contract", None)
    return _BenchmarkWithContracts(
        implementation=cast(RawBenchmark, implementation),
        target_contract=(
            target_contract
            if isinstance(target_contract, TargetContract)
            else _finite_outcome_target_contract(cast(RawBenchmark, implementation).manifest)
        ),
    )


@dataclass(frozen=True, slots=True)
class _BenchmarkWithContracts:
    implementation: RawBenchmark
    target_contract: TargetContract

    @property
    def root(self) -> Path:
        return self.implementation.root

    @property
    def manifest(self) -> BenchmarkManifest:
        return self.implementation.manifest

    @property
    def generator(self) -> Generator:
        return self.implementation.generator

    @property
    def sampling_protocol(self) -> SamplingProtocol:
        return self.implementation.sampling_protocol

    @property
    def accessible_subspace(self) -> AccessibleSubspace:
        return self.implementation.accessible_subspace

    def __getattr__(self, name: str) -> object:
        return getattr(self.implementation, name)


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


def _finite_outcome_target_contract(manifest: BenchmarkManifest) -> TargetContract:
    return TargetContract.finite_outcome(
        tuple(outcome.id for outcome in manifest.resolve_outcome_space().outcomes)
    )


def _validate_benchmark_implementation(
    value: object,
    *,
    entrypoint: Path,
) -> None:
    for name in ("root", "manifest", "sampling_protocol", "accessible_subspace"):
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
