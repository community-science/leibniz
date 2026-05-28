"""Narrow tensor runtime adapter for local benchmark execution."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal, cast

__all__ = [
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
