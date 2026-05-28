"""Narrow tensor runtime adapter for local benchmark execution."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from leibniz.observation_formation import ObservationFormationDeclaration

__all__ = [
    "FormationTensorCache",
    "TensorRuntime",
    "TensorRuntimeError",
    "TensorRuntimeDevice",
    "resolve_tensor_runtime",
    "validate_tensor_runtime_device",
]

TensorRuntimeDevice = Literal["auto", "cpu", "cuda", "mps"]
_available_devices = frozenset({"auto", "cpu", "cuda", "mps"})


class TensorRuntimeError(ValueError):
    """Raised when a tensor runtime cannot be resolved."""


@dataclass(frozen=True, slots=True)
class TensorRuntime:
    """Resolved local tensor runtime for benchmark execution."""

    torch: Any
    device: Any
    device_kind: Literal["cpu", "cuda", "mps"]


@dataclass(slots=True)
class FormationTensorCache:
    """Cache declaration-backed unvaried component fields as runtime tensors."""

    runtime: TensorRuntime
    formation: ObservationFormationDeclaration
    _component_tensors: dict[tuple[int, int, int, int], Any] = field(
        default_factory=lambda: cast(dict[tuple[int, int, int, int], Any], {})
    )

    def component_sequence_tensor(
        self,
        *,
        resolution: int,
        component_sequence: Sequence[int],
    ) -> Any:
        """Return an unvaried formed-field tensor for a component sequence."""

        _require_positive_integer(resolution, "resolution")
        sequence = tuple(component_sequence)
        if not sequence:
            raise TensorRuntimeError("component_sequence must not be empty")
        tensors = [
            self.component_tensor(
                resolution=resolution,
                slot_count=len(sequence),
                slot_index=slot_index,
                component_index=component_index,
            )
            for slot_index, component_index in enumerate(sequence)
        ]
        return self.runtime.torch.stack(tensors).amax(dim=0)

    def component_tensor(
        self,
        *,
        resolution: int,
        slot_count: int,
        slot_index: int,
        component_index: int,
    ) -> Any:
        """Return a cached tensor for one component drawn into one slot."""

        _require_positive_integer(resolution, "resolution")
        _require_positive_integer(slot_count, "slot_count")
        _require_slot_index(slot_index=slot_index, slot_count=slot_count)
        if (
            type(component_index) is not int
            or component_index < 0
            or component_index >= len(self.formation.components)
        ):
            raise TensorRuntimeError("component_index is outside component vocabulary")
        key = (resolution, slot_count, slot_index, component_index)
        cached = self._component_tensors.get(key)
        if cached is not None:
            return cached
        tensor = self._build_component_tensor(
            resolution=resolution,
            slot_count=slot_count,
            slot_index=slot_index,
            component_index=component_index,
        )
        self._component_tensors[key] = tensor
        return tensor

    def _build_component_tensor(
        self,
        *,
        resolution: int,
        slot_count: int,
        slot_index: int,
        component_index: int,
    ) -> Any:
        field = self.formation.component_field(
            resolution=resolution,
            slot_count=slot_count,
            slot_index=slot_index,
            component_index=component_index,
        )
        tensor = self.runtime.torch.tensor(
            field.values,
            dtype=self.runtime.torch.float32,
            device=self.runtime.device,
        )
        return tensor.reshape(field.shape)


def validate_tensor_runtime_device(value: str) -> TensorRuntimeDevice:
    """Validate a requested tensor runtime device name."""

    if value not in _available_devices:
        raise TensorRuntimeError(
            "tensor runtime device must be one of: auto, cpu, cuda, mps"
        )
    return cast(TensorRuntimeDevice, value)


def resolve_tensor_runtime(requested_device: TensorRuntimeDevice = "auto") -> TensorRuntime:
    """Resolve the PyTorch-backed tensor runtime for local benchmark execution."""

    torch = _torch()
    device_kind = _resolve_device_kind(torch=torch, requested_device=requested_device)
    return TensorRuntime(
        torch=torch,
        device=torch.device(device_kind),
        device_kind=device_kind,
    )


def _resolve_device_kind(
    *,
    torch: Any,
    requested_device: TensorRuntimeDevice,
) -> Literal["cpu", "cuda", "mps"]:
    if requested_device == "cpu":
        return "cpu"
    if requested_device == "cuda":
        if _cuda_available(torch):
            return "cuda"
        raise TensorRuntimeError("requested tensor runtime device cuda is not available")
    if requested_device == "mps":
        if _mps_available(torch):
            return "mps"
        raise TensorRuntimeError("requested tensor runtime device mps is not available")
    if _cuda_available(torch):
        return "cuda"
    if _mps_available(torch):
        return "mps"
    return "cpu"


def _cuda_available(torch: Any) -> bool:
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(callable(is_available) and is_available())


def _mps_available(torch: Any) -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    return bool(callable(is_available) and is_available())


def _torch() -> Any:
    try:
        return cast(Any, importlib.import_module("torch"))
    except ImportError as error:
        raise TensorRuntimeError(
            "PyTorch is required to run benchmark training"
        ) from error


def _require_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise TensorRuntimeError(f"{name} must be a positive integer")


def _require_slot_index(*, slot_index: int, slot_count: int) -> None:
    if type(slot_index) is not int or slot_index < 0 or slot_index >= slot_count:
        raise TensorRuntimeError("slot_index must be within slot_count")
