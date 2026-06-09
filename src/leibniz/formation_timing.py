"""Operator profiling for tensor-backed benchmark formation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from leibniz.observation_generation import load_generator
from leibniz.tensor_runtime import (
    TensorRuntimeDevice,
    TensorRuntimeError,
    resolve_tensor_runtime,
    tensor_runtime_available_memory_bytes,
    tensor_runtime_profile_operator_rows,
    validate_tensor_runtime_device,
)

__all__ = [
    "FormationTimingError",
    "FormationOperatorProfilePlan",
    "profile_formation_operators",
]

_initial_generation_memory_limit_bytes = 1_024_000


class FormationTimingError(ValueError):
    """Raised when formation profiling cannot be planned or executed."""


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


def profile_formation_operators(plan: FormationOperatorProfilePlan) -> dict[str, object]:
    """Profile tensor-backed formation operators after warmup."""

    generator = cast(Any, load_generator(plan.benchmark_root))
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
    return {
        "format": "leibniz.formation-operator-profile",
        "format_version": 1,
        "benchmark_id": str(generator.manifest.id),
        "sample_count": plan.sample_count,
        "repeats": plan.repeats,
        "seed": plan.seed,
        "tensor_runtime": "pytorch",
        "tensor_device": runtime.device_kind,
        "row_limit": plan.row_limit,
        "rows": [dict(row) for row in rows],
    }
