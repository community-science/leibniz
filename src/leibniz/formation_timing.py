"""Diagnostic timing for declaration-backed benchmark formation paths."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from leibniz.benchmark_implementations import Generator as BenchmarkGenerator
from leibniz.materialization import AxisAssignment
from leibniz.observation_formation import ObservationFormationDeclaration
from leibniz.observation_generation import (
    ComplexityRequest,
    GeneratedSampleSet,
    load_generator,
)
from leibniz.tensor_runtime import (
    TensorRuntime,
    TensorRuntimeDevice,
    TensorRuntimeError,
    resolve_tensor_runtime,
    runtime_roofline_record,
    synchronize_runtime,
    tensor_runtime_available_memory_bytes,
    tensor_runtime_profile_operator_rows,
    validate_tensor_runtime_device,
)
from leibniz.timing import TimingCollector

__all__ = [
    "FormationTimingError",
    "FormationOperatorProfilePlan",
    "FormationOperatorProfileSummary",
    "FormationTimingPlan",
    "FormationTimingSummary",
    "profile_formation_operators",
    "time_formation_paths",
]

_initial_generation_memory_limit_bytes = 1_024_000


class _FieldTimingGenerator(BenchmarkGenerator, Protocol):
    @property
    def formation(self) -> ObservationFormationDeclaration: ...

    def __call__(
        self,
        *,
        seed: int,
        shape: int | Sequence[int] | None = None,
        include_fields: bool = False,
        include_metadata: bool = True,
        complexity_request: ComplexityRequest | None = None,
        component_indices: Iterable[int] | None = None,
        memory_limit_bytes: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
        variation_extent: float = 1.0,
        runtime: TensorRuntime | None = None,
        outcome_ids: tuple[str, ...] | None = None,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedSampleSet: ...


class FormationTimingError(ValueError):
    """Raised when formation timing cannot be planned or executed."""


@dataclass(frozen=True, slots=True)
class FormationTimingPlan:
    """Plan for timing pure and tensor-backed formation paths."""

    benchmark_root: Path
    sample_count: int = 64
    seed: int = 101
    repeats: int = 3
    warmup_repeats: int = 1
    tensor_device: TensorRuntimeDevice = "auto"

    def __post_init__(self) -> None:
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise FormationTimingError("sample_count must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise FormationTimingError("seed must be a nonnegative integer")
        if type(self.repeats) is not int or self.repeats < 1:
            raise FormationTimingError("repeats must be a positive integer")
        if type(self.warmup_repeats) is not int or self.warmup_repeats < 0:
            raise FormationTimingError("warmup_repeats must be nonnegative")
        try:
            validate_tensor_runtime_device(self.tensor_device)
        except TensorRuntimeError as error:
            raise FormationTimingError(str(error)) from error


@dataclass(frozen=True, slots=True)
class FormationOperatorProfilePlan:
    """Plan for profiling tensor-backed formation operators."""

    benchmark_root: Path
    sample_count: int = 64
    seed: int = 101
    repeats: int = 3
    warmup_repeats: int = 1
    tensor_device: TensorRuntimeDevice = "auto"
    row_limit: int = 30

    def __post_init__(self) -> None:
        if type(self.sample_count) is not int or self.sample_count < 1:
            raise FormationTimingError("sample_count must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise FormationTimingError("seed must be a nonnegative integer")
        if type(self.repeats) is not int or self.repeats < 1:
            raise FormationTimingError("repeats must be a positive integer")
        if type(self.warmup_repeats) is not int or self.warmup_repeats < 0:
            raise FormationTimingError("warmup_repeats must be nonnegative")
        if type(self.row_limit) is not int or self.row_limit < 1:
            raise FormationTimingError("row_limit must be a positive integer")
        try:
            validate_tensor_runtime_device(self.tensor_device)
        except TensorRuntimeError as error:
            raise FormationTimingError(str(error)) from error


@dataclass(frozen=True, slots=True)
class FormationTimingSummary:
    """Measured wall-time summary for formation paths."""

    benchmark_id: str
    sample_count: int
    repeats: int
    seed: int
    tensor_runtime: str
    tensor_device: str
    pure_observation_seconds: float
    tensor_batch_seconds: float
    pure_phase_timing: Mapping[str, object]
    tensor_phase_timing: Mapping[str, object]
    roofline: Mapping[str, object]

    @property
    def pure_observation_samples_per_second(self) -> float:
        return self._samples_per_second(self.pure_observation_seconds)

    @property
    def tensor_batch_samples_per_second(self) -> float:
        return self._samples_per_second(self.tensor_batch_seconds)

    def to_record(self) -> dict[str, object]:
        """Return a document-friendly timing record."""

        return {
            "format": "leibniz.formation-timing",
            "format_version": 1,
            "benchmark_id": self.benchmark_id,
            "sample_count": self.sample_count,
            "repeats": self.repeats,
            "seed": self.seed,
            "tensor_runtime": self.tensor_runtime,
            "tensor_device": self.tensor_device,
            "pure_observation_seconds": self.pure_observation_seconds,
            "tensor_batch_seconds": self.tensor_batch_seconds,
            "pure_observation_samples_per_second": (
                self.pure_observation_samples_per_second
            ),
            "tensor_batch_samples_per_second": self.tensor_batch_samples_per_second,
            "pure_phase_timing": dict(self.pure_phase_timing),
            "tensor_phase_timing": dict(self.tensor_phase_timing),
            "roofline": dict(self.roofline),
        }

    def _samples_per_second(self, seconds: float) -> float:
        if seconds <= 0.0:
            return float("inf")
        return self.sample_count * self.repeats / seconds


@dataclass(frozen=True, slots=True)
class FormationOperatorProfileSummary:
    """Compact PyTorch operator profile for tensor-backed formation."""

    benchmark_id: str
    sample_count: int
    repeats: int
    seed: int
    tensor_runtime: str
    tensor_device: str
    row_limit: int
    rows: tuple[Mapping[str, object], ...]

    def to_record(self) -> dict[str, object]:
        """Return a document-friendly operator profile record."""

        return {
            "format": "leibniz.formation-operator-profile",
            "format_version": 1,
            "benchmark_id": self.benchmark_id,
            "sample_count": self.sample_count,
            "repeats": self.repeats,
            "seed": self.seed,
            "tensor_runtime": self.tensor_runtime,
            "tensor_device": self.tensor_device,
            "row_limit": self.row_limit,
            "rows": [dict(row) for row in self.rows],
        }


def time_formation_paths(plan: FormationTimingPlan) -> FormationTimingSummary:
    """Measure pure observation formation and tensor batch construction."""

    generator = cast(_FieldTimingGenerator, load_generator(plan.benchmark_root))
    try:
        runtime = resolve_tensor_runtime(plan.tensor_device)
    except TensorRuntimeError as error:
        raise FormationTimingError(str(error)) from error
    memory_limit_bytes = min(
        tensor_runtime_available_memory_bytes(runtime),
        _initial_generation_memory_limit_bytes,
    )
    outcome_ids = tuple(
        outcome.id for outcome in generator.manifest.resolve_outcome_space().outcomes
    )

    def pure_once(seed: int, timing: TimingCollector | None = None) -> None:
        sample_set = generator(
            shape=plan.sample_count,
            seed=seed,
            include_fields=True,
            memory_limit_bytes=memory_limit_bytes,
            timing=timing,
            timing_prefix="pure.",
        )
        if not sample_set.includes_fields:
            raise FormationTimingError("generator did not include generated fields")

    def tensor_once(seed: int, timing: TimingCollector | None = None) -> None:
        sample_set = generator(
            shape=plan.sample_count,
            seed=seed,
            include_metadata=False,
            memory_limit_bytes=memory_limit_bytes,
            runtime=runtime,
            outcome_ids=outcome_ids,
            timing=timing,
            timing_prefix="tensor.",
        )
        sample_set.require_tensors()

    for offset in range(plan.warmup_repeats):
        pure_once(plan.seed + offset)
        tensor_once(plan.seed + 100_003 + offset)
    _synchronize(runtime)
    pure_timing = TimingCollector()
    tensor_timing = TimingCollector()

    def timed_pure_once(offset: int) -> None:
        pure_once(plan.seed + 1_000_003 + offset, timing=pure_timing)

    def timed_tensor_once(offset: int) -> None:
        tensor_once(plan.seed + 2_000_003 + offset, timing=tensor_timing)

    pure_seconds = _time_repeats(
        timed_pure_once,
        repeats=plan.repeats,
        runtime=runtime,
    )
    tensor_seconds = _time_repeats(
        timed_tensor_once,
        repeats=plan.repeats,
        runtime=runtime,
    )
    return FormationTimingSummary(
        benchmark_id=str(generator.manifest.id),
        sample_count=plan.sample_count,
        repeats=plan.repeats,
        seed=plan.seed,
        tensor_runtime="pytorch",
        tensor_device=runtime.device_kind,
        pure_observation_seconds=pure_seconds,
        tensor_batch_seconds=tensor_seconds,
        pure_phase_timing=pure_timing.to_record(kind="pure-formation-phase-timing"),
        tensor_phase_timing=tensor_timing.to_record(kind="tensor-formation-phase-timing"),
        roofline=runtime_roofline_record(runtime),
    )


def profile_formation_operators(
    plan: FormationOperatorProfilePlan,
) -> FormationOperatorProfileSummary:
    """Profile tensor-backed formation operators after warmup."""

    generator = cast(_FieldTimingGenerator, load_generator(plan.benchmark_root))
    try:
        runtime = resolve_tensor_runtime(plan.tensor_device)
    except TensorRuntimeError as error:
        raise FormationTimingError(str(error)) from error
    memory_limit_bytes = min(
        tensor_runtime_available_memory_bytes(runtime),
        _initial_generation_memory_limit_bytes,
    )
    outcome_ids = tuple(
        outcome.id for outcome in generator.manifest.resolve_outcome_space().outcomes
    )

    def tensor_once(seed: int) -> None:
        sample_set = generator(
            shape=plan.sample_count,
            seed=seed,
            include_metadata=False,
            memory_limit_bytes=memory_limit_bytes,
            runtime=runtime,
            outcome_ids=outcome_ids,
        )
        fields, targets = sample_set.require_tensors()
        _ = fields.sum() + targets.sum()

    for offset in range(plan.warmup_repeats):
        tensor_once(plan.seed + offset)
    _synchronize(runtime)

    try:
        rows = tensor_runtime_profile_operator_rows(
            runtime,
            callback=lambda offset: tensor_once(plan.seed + 1_000_003 + offset),
            repeats=plan.repeats,
            row_limit=plan.row_limit,
            record_name="leibniz.formation.tensor_batch",
        )
    except TensorRuntimeError as error:
        raise FormationTimingError(str(error)) from error
    return FormationOperatorProfileSummary(
        benchmark_id=str(generator.manifest.id),
        sample_count=plan.sample_count,
        repeats=plan.repeats,
        seed=plan.seed,
        tensor_runtime="pytorch",
        tensor_device=runtime.device_kind,
        row_limit=plan.row_limit,
        rows=rows,
    )


def _time_repeats(
    callback: Callable[[int], None],
    *,
    repeats: int,
    runtime: TensorRuntime,
) -> float:
    _synchronize(runtime)
    start = time.perf_counter()
    for offset in range(repeats):
        callback(offset)
    _synchronize(runtime)
    return time.perf_counter() - start


def _synchronize(runtime: TensorRuntime) -> None:
    synchronize_runtime(runtime)
