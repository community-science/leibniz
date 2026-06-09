"""Narrow tensor runtime adapter for local benchmark execution."""

from __future__ import annotations

import importlib
import math
import os
import time
from array import array
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal, cast

from leibniz.architectures import ArchitectureManifest
from leibniz.tensor_shapes import TensorShape

__all__ = [
    "architecture_tensor_runtime_issue",
    "architecture_supported_by_tensor_runtime",
    "build_architecture_modules",
    "build_architecture_sequential",
    "build_cosine_lr_schedule",
    "build_cross_entropy_loss",
    "build_optimizer",
    "build_plateau_lr_schedule",
    "make_float_tensor",
    "make_long_tensor",
    "no_grad_context",
    "optimizer_step",
    "OperationFallbackSequential",
    "preferred_tensor_runtime_device_kind",
    "seed_runtime",
    "softmax_prediction_rows",
    "softmax_target_masses",
    "synchronize_runtime",
    "TensorRuntime",
    "TensorElementParameter",
    "TensorElementProgram",
    "TensorElementRecipe",
    "TensorRuntimeError",
    "TensorRuntimeDevice",
    "TensorRuntimeDeviceKind",
    "tensor_runtime_available_memory_bytes",
    "tensor_runtime_construct_tensor",
    "tensor_runtime_default_device",
    "tensor_runtime_device_choices",
    "tensor_runtime_device_kinds",
    "tensor_runtime_has_fixed_device_memory",
    "tensor_runtime_total_memory_bytes",
    "tensor_runtime_used_memory_bytes",
    "tensor_value_to_host",
    "resolve_host_tensor_runtime",
    "resolve_tensor_runtime",
    "runtime_roofline_record",
    "save_tensor_runtime_state",
    "load_tensor_runtime_state",
    "runtime_capacity_error",
    "validate_tensor_runtime_device",
]

TensorRuntimeDevice = Literal["auto", "cpu", "cuda", "mps"]
TensorRuntimeDeviceKind = Literal["cpu", "cuda", "mps"]
TensorElementDType = Literal["float32", "int64"]
TensorElementParameterDType = Literal["float32", "int64"]
_available_devices = frozenset({"auto", "cpu", "cuda", "mps"})
_roofline_cache: dict[str, dict[str, object]] = {}
_tensor_element_kernel_cache: dict[tuple[object, ...], Any] = {}
_tensor_element_parameter_cache: dict[tuple[object, ...], Any] = {}
_tensor_element_tile_metadata_cache: dict[tuple[object, ...], Any] = {}
_tensor_element_tile_size = 131_072


class TensorRuntimeError(ValueError):
    """Raised when a tensor runtime cannot be resolved."""


@dataclass(frozen=True, slots=True)
class TensorRuntime:
    """Resolved local tensor runtime for benchmark execution."""

    torch: Any
    device: Any
    device_kind: Literal["cpu", "cuda", "mps"]


@dataclass(frozen=True, slots=True)
class TensorElementRecipe:
    """Elementwise tensor construction recipe supplied by benchmark implementations."""

    shape: tuple[int, ...]
    dtype: TensorElementDType
    program: TensorElementProgram


@dataclass(frozen=True, slots=True)
class TensorElementParameter:
    """Numeric parameter buffer referenced by a tensor element program."""

    dtype: TensorElementParameterDType
    shape: tuple[int, ...]
    values: Sequence[int | float]
    dynamic_axes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TensorElementProgram:
    """Rank-agnostic vectorized element function to lift over a tensor extent."""

    kernel: Callable[..., object]
    parameters: Mapping[str, TensorElementParameter]
    compile: bool = True
    cache_key: object | None = None


def tensor_runtime_construct_tensor(
    runtime: TensorRuntime,
    *,
    recipe: TensorElementRecipe,
) -> Any:
    """Construct a tensor by lifting benchmark element semantics over runtime coordinates."""

    torch = runtime.torch
    resolved_shape = tuple(_positive_tensor_extent(size) for size in recipe.shape)
    dtype = _tensor_element_dtype(runtime=runtime, dtype=recipe.dtype)
    total_elements = math.prod(resolved_shape)
    if total_elements == 0:
        return torch.empty(resolved_shape, dtype=dtype, device=runtime.device)
    return _construct_tensor_element_program(
        runtime=runtime,
        shape=resolved_shape,
        dtype=dtype,
        program=recipe.program,
    )


def tensor_runtime_device_choices() -> tuple[str, ...]:
    """Return the public tensor runtime device choices."""

    return ("auto", "cpu", "cuda", "mps")


def tensor_runtime_default_device() -> str:
    """Return the portable default tensor runtime device name."""

    return "cpu"


def resolve_host_tensor_runtime() -> TensorRuntime:
    """Resolve the portable host tensor runtime."""

    return resolve_tensor_runtime("cpu")


def tensor_value_to_host(value: Any) -> Any:
    """Detach a backend tensor-like value and move it to host memory when possible."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    move_to_host = getattr(value, "cpu", None)
    if callable(move_to_host):
        value = move_to_host()
    return value


def tensor_runtime_has_fixed_device_memory(runtime: TensorRuntime) -> bool:
    """Return whether the runtime exposes a fixed device memory budget."""

    return runtime.device_kind == "cuda"


def tensor_runtime_total_memory_bytes(runtime: TensorRuntime) -> int | None:
    """Return total fixed device memory bytes, if the runtime exposes it."""

    if runtime.device_kind == "cuda":
        try:
            _free_bytes, total_bytes = runtime.torch.cuda.mem_get_info(runtime.device)
        except Exception as error:  # pragma: no cover - backend-specific failure
            raise TensorRuntimeError(f"could not query device memory: {error}") from error
        return _positive_memory_bytes(total_bytes, field="device total memory")
    return None


def tensor_runtime_used_memory_bytes(runtime: TensorRuntime) -> int:
    """Return current fixed device memory usage bytes for diagnostics."""

    if runtime.device_kind == "cuda":
        try:
            free_bytes, total_bytes = runtime.torch.cuda.mem_get_info(runtime.device)
        except Exception as error:  # pragma: no cover - backend-specific failure
            raise TensorRuntimeError(f"could not query device memory: {error}") from error
        return max(0, int(total_bytes) - int(free_bytes))
    return 0


def runtime_capacity_error(error: RuntimeError) -> bool:
    """Return whether a backend runtime error represents memory capacity exhaustion."""

    message = str(error).lower()
    return any(
        phrase in message
        for phrase in (
            "out of memory",
            "can't allocate memory",
            "cannot allocate memory",
            "cuda error: out of memory",
        )
    )


@dataclass(slots=True)
class _OperationPlacement:
    device: Any
    device_kind: TensorRuntimeDeviceKind


class OperationFallbackSequential:
    """Sequential module with per-operation CPU fallback for backend failures."""

    def __new__(
        cls,
        *,
        runtime: TensorRuntime,
        operations: Sequence[Any],
    ) -> Any:
        torch = runtime.torch

        class Module(torch.nn.Module):  # type: ignore[misc]
            def __init__(self) -> None:
                torch.nn.Module.__init__(self)
                self._preferred = _OperationPlacement(
                    device=runtime.device,
                    device_kind=runtime.device_kind,
                )
                self._fallback = _OperationPlacement(
                    device=torch.device("cpu"),
                    device_kind="cpu",
                )
                self._operations = torch.nn.ModuleList(
                    operation.to(self._preferred.device) for operation in operations
                )
                self._placements = [
                    _OperationPlacement(
                        device=self._preferred.device,
                        device_kind=self._preferred.device_kind,
                    )
                    for _operation in self._operations
                ]
                self._optimizer: Any | None = None
                self._fallback_records: list[dict[str, object]] = []

            def attach_optimizer(self, optimizer: Any) -> None:
                self._optimizer = optimizer

            def operation_fallback_records(self) -> tuple[dict[str, object], ...]:
                return tuple(dict(record) for record in self._fallback_records)

            def forward(self, value: Any) -> Any:
                current = value
                for index, operation in enumerate(self._operations):
                    placement = self._placements[index]
                    current = current.to(placement.device)
                    try:
                        current = operation(current)
                    except RuntimeError as error:
                        if placement.device_kind == "cpu":
                            raise
                        reason = str(error)
                        self._move_operation_to_fallback(index=index, reason=reason)
                        current = self._operations[index](current.to(self._fallback.device))
                return current.to(self._preferred.device)

            def _move_operation_to_fallback(self, *, index: int, reason: str) -> None:
                operation = self._operations[index]
                operation.to(self._fallback.device)
                self._move_optimizer_state(operation=operation, device=self._fallback.device)
                previous = self._placements[index]
                self._placements[index] = _OperationPlacement(
                    device=self._fallback.device,
                    device_kind=self._fallback.device_kind,
                )
                self._fallback_records.append(
                    {
                        "operation_index": index,
                        "from_device": previous.device_kind,
                        "to_device": self._fallback.device_kind,
                        "reason": reason,
                    }
                )

            def _move_optimizer_state(self, *, operation: Any, device: Any) -> None:
                if self._optimizer is None:
                    return
                for parameter in operation.parameters():
                    state = self._optimizer.state.get(parameter)
                    if state is not None:
                        _optimizer_state_to_device(state, device=device)

        return Module()


def save_tensor_runtime_state(path: Any, state: Mapping[str, object]) -> None:
    """Persist a tensor-runtime state payload."""

    torch = importlib.import_module("torch")
    torch.save(dict(state), path)


def load_tensor_runtime_state(
    runtime: TensorRuntime,
    path: Any,
    *,
    weights_only: bool = False,
) -> object:
    """Load a tensor-runtime state payload onto the runtime device."""

    return runtime.torch.load(
        path,
        map_location=runtime.device,
        weights_only=weights_only,
    )


def _optimizer_state_to_device(
    value: dict[object, object] | list[object],
    *,
    device: Any,
) -> None:
    if isinstance(value, dict):
        for key, item in tuple(value.items()):
            value[key] = _optimizer_state_value_to_device(item, device=device)
    else:
        for index, item in enumerate(value):
            value[index] = _optimizer_state_value_to_device(item, device=device)


def _optimizer_state_value_to_device(value: object, *, device: Any) -> object:
    to_device = getattr(value, "to", None)
    if callable(to_device):
        return to_device(device)
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        _optimizer_state_to_device(mapping, device=device)
        return mapping
    if isinstance(value, list):
        sequence = cast(list[object], value)
        _optimizer_state_to_device(sequence, device=device)
        return sequence
    return value


def tensor_runtime_available_memory_bytes(runtime: TensorRuntime) -> int:
    """Return available memory bytes for the resolved runtime device."""

    if runtime.device_kind == "cuda":
        try:
            free_bytes, _total_bytes = runtime.torch.cuda.mem_get_info(runtime.device)
        except Exception as error:  # pragma: no cover - backend-specific failure
            raise TensorRuntimeError(f"could not query cuda memory: {error}") from error
        return _positive_memory_bytes(free_bytes, field="cuda free memory")
    if runtime.device_kind == "mps":
        mps = runtime.torch.mps
        recommended = (
            mps.recommended_max_memory()
            if hasattr(mps, "recommended_max_memory")
            else None
        )
        if recommended is None:
            return _host_memory_bytes()
        allocated = (
            mps.driver_allocated_memory()
            if hasattr(mps, "driver_allocated_memory")
            else (
                mps.current_allocated_memory()
                if hasattr(mps, "current_allocated_memory")
                else 0
            )
        )
        return _positive_memory_bytes(
            max(1, int(recommended) - int(allocated)),
            field="mps available memory",
        )
    return _host_memory_bytes()

def validate_tensor_runtime_device(value: str) -> TensorRuntimeDevice:
    """Validate a requested tensor runtime device name."""

    if value not in _available_devices:
        raise TensorRuntimeError("tensor runtime device must be one of: auto, cpu, cuda, mps")
    return cast(TensorRuntimeDevice, value)


def resolve_tensor_runtime(requested_device: TensorRuntimeDevice = "auto") -> TensorRuntime:
    """Resolve the PyTorch-backed tensor runtime for local benchmark execution."""

    torch = _torch()
    device_kind = _resolve_device_kinds(torch=torch, requested_device=requested_device)[0]
    return TensorRuntime(
        torch=torch,
        device=torch.device(device_kind),
        device_kind=device_kind,
    )


def tensor_runtime_device_kinds(
    requested_device: TensorRuntimeDevice = "auto",
) -> tuple[TensorRuntimeDeviceKind, ...]:
    """Return available runtime device kinds in fallback order."""

    return _resolve_device_kinds(torch=_torch(), requested_device=requested_device)


def preferred_tensor_runtime_device_kind(
    requested_device: TensorRuntimeDevice = "auto",
) -> TensorRuntimeDeviceKind:
    """Return the first device kind that would be used for a request."""

    return tensor_runtime_device_kinds(requested_device)[0]


def architecture_supported_by_tensor_runtime(
    architecture: ArchitectureManifest,
    *,
    device_kind: TensorRuntimeDeviceKind,
) -> bool:
    """Return whether an architecture is eligible for a resolved tensor device."""

    return (
        architecture_tensor_runtime_issue(
            architecture,
            device_kind=device_kind,
        )
        is None
    )


def architecture_tensor_runtime_issue(
    architecture: ArchitectureManifest,
    *,
    device_kind: TensorRuntimeDeviceKind,
) -> str | None:
    """Return the first known runtime-specific architecture incompatibility."""

    _ = architecture
    _ = device_kind
    return None


def runtime_roofline_record(runtime: TensorRuntime) -> dict[str, object]:
    """Return best-effort local hardware roofline metadata for a tensor runtime."""

    key = str(runtime.device)
    cached = _roofline_cache.get(key)
    if cached is not None:
        return dict(cached)
    record = _calibrated_roofline_record(runtime)
    _roofline_cache[key] = record
    return dict(record)


def build_architecture_modules(
    architecture: ArchitectureManifest,
    *,
    canonical_layer_kinds: tuple[str, ...],
) -> tuple[Any, ...]:
    """Build PyTorch operation modules for an architecture with pre-resolved layer kinds."""

    if len(canonical_layer_kinds) != len(architecture.layers):
        raise TensorRuntimeError("canonical_layer_kinds length must match architecture layers")
    torch = _torch()
    modules: list[Any] = []
    shape = architecture.input_shape
    for layer, kind in zip(architecture.layers, canonical_layer_kinds, strict=True):
        parameters = layer.parameters
        if kind == "local-aggregation":
            dimension = _require_int_parameter(parameters, "dimension")
            if dimension != 2:
                raise TensorRuntimeError("local aggregation currently supports dimension 2")
            out_height, out_width = _require_fixed_support(parameters)
            modules.append(torch.nn.AdaptiveAvgPool2d((out_height, out_width)))
            shape = (*shape[: len(shape) - dimension], out_height, out_width)
        elif kind == "fixed-support-affine":
            dimension = _require_int_parameter(parameters, "dimension")
            if dimension != 2:
                raise TensorRuntimeError("fixed support affine currently supports dimension 2")
            if len(shape) <= dimension:
                raise TensorRuntimeError(
                    "fixed support affine requires a channel axis before support axes"
                )
            channel_axis_index = len(shape) - dimension - 1
            out_channels = _require_int_parameter(parameters, "out_channels")
            out_height = _require_int_parameter(parameters, "out_height")
            out_width = _require_int_parameter(parameters, "out_width")
            modules.append(
                torch.nn.Sequential(
                    torch.nn.AdaptiveAvgPool2d((out_height, out_width)),
                    torch.nn.Conv2d(
                        in_channels=shape[channel_axis_index],
                        out_channels=out_channels,
                        kernel_size=1,
                    ),
                )
            )
            shape = (*shape[:channel_axis_index], out_channels, out_height, out_width)
        elif kind == "local-affine":
            dimension = _require_int_parameter(parameters, "dimension")
            if dimension != 2:
                raise TensorRuntimeError("local affine currently supports dimension 2")
            if len(shape) <= dimension:
                raise TensorRuntimeError(
                    "local affine requires a channel axis before local support axes"
                )
            channel_axis_index = len(shape) - dimension - 1
            spatial_axis_start = len(shape) - dimension
            size = _require_int_parameter(parameters, "size")
            out_channels = _require_int_parameter(parameters, "out_channels")
            stride = _require_int_parameter(parameters, "stride")
            padding = _require_nonneg_int_parameter(parameters, "padding")
            modules.append(
                torch.nn.Conv2d(
                    in_channels=shape[channel_axis_index],
                    out_channels=out_channels,
                    kernel_size=size,
                    stride=stride,
                    padding=padding,
                )
            )
            output_spatial_axes = tuple(
                _local_window_output_size(axis, size=size, stride=stride, padding=padding)
                for axis in shape[spatial_axis_start:]
            )
            shape = (*shape[:channel_axis_index], out_channels, *output_spatial_axes)
        elif kind == "rank-collapse":
            modules.append(torch.nn.Flatten())
            shape = (TensorShape.from_axes(shape).element_count,)
        elif kind == "affine-readout":
            if len(shape) != 1:
                raise TensorRuntimeError("affine readout requires rank-1 input")
            out = _require_int_parameter(parameters, "out")
            modules.append(torch.nn.Linear(shape[0], out))
            shape = (out,)
        else:
            raise TensorRuntimeError(f"unsupported operator kind: {kind}")
    return tuple(modules)


def build_architecture_sequential(
    architecture: ArchitectureManifest,
    *,
    canonical_layer_kinds: tuple[str, ...],
) -> Any:
    """Build a PyTorch Sequential module for an architecture with pre-resolved layer kinds."""

    torch = _torch()
    modules = build_architecture_modules(architecture, canonical_layer_kinds=canonical_layer_kinds)
    return torch.nn.Sequential(*modules)


def synchronize_runtime(runtime: TensorRuntime) -> None:
    """Synchronize the runtime device; no-op on CPU."""

    _synchronize_runtime(runtime)


def seed_runtime(runtime: TensorRuntime, *, seed: int) -> None:
    """Set the global random seed for reproducible training."""

    _ = runtime
    _torch().manual_seed(seed)


def build_cross_entropy_loss(runtime: TensorRuntime) -> Any:
    """Build a cross-entropy classification loss module."""

    _ = runtime
    return _torch().nn.CrossEntropyLoss()


def no_grad_context(runtime: TensorRuntime) -> Any:
    """Return a torch.no_grad() context manager for inference."""

    _ = runtime
    return _torch().no_grad()


def softmax_prediction_rows(runtime: TensorRuntime, logits: Any) -> list[list[float]]:
    _ = runtime
    return _torch().softmax(logits.detach(), dim=1).tolist()


def softmax_target_masses(
    runtime: TensorRuntime,
    logits: Any,
    targets: Any,
) -> list[float]:
    """Return each prediction row's probability mass assigned to its target."""

    _ = runtime
    torch = _torch()
    probabilities = torch.softmax(logits.detach(), dim=1)
    if targets.shape == probabilities.shape:
        return (probabilities * targets).sum(dim=1).detach().tolist()
    target_indexes = targets.reshape((-1, 1)).long()
    return probabilities.gather(1, target_indexes).reshape((-1,)).detach().tolist()


def build_optimizer(
    runtime: TensorRuntime,
    *,
    name: str,
    parameters: Any,
    learning_rate: float | None,
) -> Any:
    """Build a named optimizer for module parameters."""

    _ = runtime
    torch = _torch()
    if name == "loss-search":
        return _LossSearchOptimizer(parameters)
    if learning_rate is None:
        raise TensorRuntimeError(f"{name} optimizer requires a learning rate")
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    raise TensorRuntimeError(f"unsupported optimizer: {name}")


class _LossSearchOptimizer:
    requires_loss_closure = True
    _beta1 = 0.9
    _beta2 = 0.999
    _epsilon = 1e-8

    def __init__(self, parameters: Any) -> None:
        self._parameters = tuple(parameters)
        self.param_groups: list[dict[str, object]] = [{"params": list(self._parameters)}]
        self.state: dict[object, dict[str, object]] = {}
        self._step_size = 1.0

    def zero_grad(self, *, set_to_none: bool = False) -> None:
        for parameter in self._parameters:
            gradient = getattr(parameter, "grad", None)
            if gradient is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                gradient.detach_()
                gradient.zero_()

    def step(self, closure: Any | None = None) -> Any:
        if closure is None:
            raise TensorRuntimeError("loss-search optimizer requires a loss closure")
        torch = _torch()
        with torch.enable_grad():
            baseline_loss = closure()
        baseline_value = float(baseline_loss.detach())
        if not math.isfinite(baseline_value):
            return baseline_loss
        gradients = tuple(
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in self._parameters
        )
        next_step = self._next_step_index()
        direction_records = tuple(
            None
            if gradient is None
            else self._direction_record(
                parameter=parameter,
                gradient=gradient,
                step=next_step,
            )
            for parameter, gradient in zip(self._parameters, gradients, strict=True)
        )
        directional_derivative = sum(
            float((gradient * record["direction"]).sum().detach())
            for gradient, record in zip(gradients, direction_records, strict=True)
            if gradient is not None and record is not None
        )
        if directional_derivative <= 0.0 or not math.isfinite(directional_derivative):
            return baseline_loss
        originals = tuple(parameter.detach().clone() for parameter in self._parameters)
        step_size = min(self._step_size, max(1e-12, baseline_value / directional_derivative))
        for _attempt in range(12):
            with torch.no_grad():
                for parameter, original, record in zip(
                    self._parameters,
                    originals,
                    direction_records,
                    strict=True,
                ):
                    if record is not None:
                        parameter.copy_(
                            original - step_size * record["direction"]
                        )
            with torch.enable_grad():
                trial_loss = closure()
            trial_value = float(trial_loss.detach())
            if math.isfinite(trial_value) and trial_value <= baseline_value:
                for parameter, record in zip(
                    self._parameters,
                    direction_records,
                    strict=True,
                ):
                    if record is not None:
                        self.state[parameter] = {
                            "step": next_step,
                            "exp_avg": record["exp_avg"],
                            "exp_avg_sq": record["exp_avg_sq"],
                        }
                self._step_size = min(step_size * 2.0, 1.0)
                return trial_loss
            step_size *= 0.5
        with torch.no_grad():
            for parameter, original in zip(self._parameters, originals, strict=True):
                parameter.copy_(original)
        self._step_size = max(step_size, 1e-12)
        return baseline_loss

    def _next_step_index(self) -> int:
        steps = (
            step
            for state in self.state.values()
            for step in (state.get("step"),)
            if type(step) is int
        )
        return max(steps, default=0) + 1

    def _direction_record(
        self,
        *,
        parameter: Any,
        gradient: Any,
        step: int,
    ) -> dict[str, Any]:
        torch = _torch()
        state = self.state.get(parameter, {})
        exp_avg = cast(Any, state.get("exp_avg"))
        exp_avg_sq = cast(Any, state.get("exp_avg_sq"))
        if exp_avg is None:
            exp_avg = torch.zeros_like(parameter)
        if exp_avg_sq is None:
            exp_avg_sq = torch.zeros_like(parameter)
        exp_avg = self._beta1 * exp_avg + (1.0 - self._beta1) * gradient
        exp_avg_sq = self._beta2 * exp_avg_sq + (1.0 - self._beta2) * gradient * gradient
        bias_corrected_avg = exp_avg / (1.0 - self._beta1**step)
        bias_corrected_avg_sq = exp_avg_sq / (1.0 - self._beta2**step)
        direction = bias_corrected_avg / (bias_corrected_avg_sq.sqrt() + self._epsilon)
        return {
            "direction": direction,
            "exp_avg": exp_avg.detach().clone(),
            "exp_avg_sq": exp_avg_sq.detach().clone(),
        }


def optimizer_step(runtime: TensorRuntime, optimizer: Any, closure: Any) -> Any:
    _ = runtime
    if getattr(optimizer, "requires_loss_closure", False):
        return optimizer.step(closure)
    loss = closure()
    optimizer.step()
    return loss


def build_cosine_lr_schedule(runtime: TensorRuntime, optimizer: Any, *, T_max: int) -> Any:
    """Build a cosine annealing learning rate scheduler."""

    _ = runtime
    return _torch().optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)


def build_plateau_lr_schedule(
    runtime: TensorRuntime,
    optimizer: Any,
    *,
    factor: float,
    threshold: float,
    patience: int,
    eps: float,
) -> Any:
    """Build a reduce-on-plateau learning rate scheduler."""

    _ = runtime
    return _torch().optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=factor,
        threshold=threshold,
        threshold_mode="abs",
        patience=patience,
        eps=eps,
    )


def make_float_tensor(runtime: TensorRuntime, values: Any, *, device: Any) -> Any:
    """Create a float32 tensor on the given device."""

    _ = runtime
    torch = _torch()
    return torch.tensor(values, dtype=torch.float32, device=device)


def make_long_tensor(runtime: TensorRuntime, values: Any, *, device: Any) -> Any:
    """Create a long (int64) tensor on the given device."""

    _ = runtime
    torch = _torch()
    return torch.tensor(values, dtype=torch.long, device=device)


def _calibrated_roofline_record(runtime: TensorRuntime) -> dict[str, object]:
    record: dict[str, object] = {
        "kind": "system-roofline",
        "status": "unavailable",
        "tensor_runtime": "pytorch",
        "tensor_device": runtime.device_kind,
        "method": "dense-matmul-and-copy-calibration",
    }
    try:
        torch = runtime.torch
        matrix_size = 512
        first = torch.randn(
            (matrix_size, matrix_size),
            dtype=torch.float32,
            device=runtime.device,
        )
        second = torch.randn(
            (matrix_size, matrix_size),
            dtype=torch.float32,
            device=runtime.device,
        )
        _synchronize_runtime(runtime)
        with torch.no_grad():
            for _ in range(1):
                _ = first @ second
        _synchronize_runtime(runtime)
        matmul_repeats = 1
        matmul_seconds = 0.0
        while matmul_repeats <= 64 and matmul_seconds < 0.02:
            started = _monotonic_seconds()
            with torch.no_grad():
                for _ in range(matmul_repeats):
                    _ = first @ second
            _synchronize_runtime(runtime)
            matmul_seconds = _monotonic_seconds() - started
            if matmul_seconds < 0.02:
                matmul_repeats *= 2

        element_count = 8 * 1024 * 1024
        source = torch.randn((element_count,), dtype=torch.float32, device=runtime.device)
        target = torch.empty_like(source)
        _synchronize_runtime(runtime)
        with torch.no_grad():
            target.copy_(source)
        _synchronize_runtime(runtime)
        copy_repeats = 1
        copy_seconds = 0.0
        while copy_repeats <= 128 and copy_seconds < 0.02:
            started = _monotonic_seconds()
            with torch.no_grad():
                for _ in range(copy_repeats):
                    target.copy_(source)
            _synchronize_runtime(runtime)
            copy_seconds = _monotonic_seconds() - started
            if copy_seconds < 0.02:
                copy_repeats *= 2
    except Exception as error:  # pragma: no cover - hardware/runtime dependent
        record["reason"] = f"roofline calibration failed: {error}"
        return record
    if matmul_seconds <= 0 or copy_seconds <= 0:
        record["reason"] = "roofline calibration completed too quickly to measure"
        return record
    copy_bytes = 2.0 * element_count * 4 * copy_repeats
    record.update(
        {
            "status": "calibrated",
            "compute_calibration_seconds": matmul_seconds,
            "compute_calibration_matrix_size": matrix_size,
            "compute_calibration_repeats": matmul_repeats,
            "peak_compute_per_second": (
                2.0 * matrix_size * matrix_size * matrix_size * matmul_repeats
            )
            / matmul_seconds,
            "bandwidth_calibration_seconds": copy_seconds,
            "bandwidth_calibration_bytes": copy_bytes,
            "bandwidth_calibration_repeats": copy_repeats,
            "peak_bytes_per_second": copy_bytes / copy_seconds,
        }
    )
    if runtime.device_kind == "cuda":
        cuda = getattr(runtime.torch, "cuda", None)
        get_device_name = getattr(cuda, "get_device_name", None)
        if callable(get_device_name):
            record["device_name"] = str(get_device_name(runtime.device))
    return record


def _synchronize_runtime(runtime: TensorRuntime) -> None:
    if runtime.device_kind == "cuda":
        runtime.torch.cuda.synchronize(runtime.device)
    elif runtime.device_kind == "mps":
        synchronize = getattr(runtime.torch.mps, "synchronize", None)
        if callable(synchronize):
            synchronize()


def _monotonic_seconds() -> float:
    return time.perf_counter()


def _resolve_device_kinds(
    *,
    torch: Any,
    requested_device: TensorRuntimeDevice,
) -> tuple[TensorRuntimeDeviceKind, ...]:
    if requested_device == "cpu":
        return ("cpu",)
    if requested_device == "cuda":
        if _cuda_available(torch):
            return ("cuda",)
        raise TensorRuntimeError("requested tensor runtime device cuda is not available")
    if requested_device == "mps":
        if _mps_available(torch):
            return ("mps",)
        raise TensorRuntimeError("requested tensor runtime device mps is not available")
    device_kinds: list[TensorRuntimeDeviceKind] = []
    if _cuda_available(torch):
        device_kinds.append("cuda")
    if _mps_available(torch):
        device_kinds.append("mps")
    device_kinds.append("cpu")
    return tuple(device_kinds)


def _cuda_available(torch: Any) -> bool:
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    return bool(callable(is_available) and is_available())


def _mps_available(torch: Any) -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    is_available = getattr(mps, "is_available", None)
    return bool(callable(is_available) and is_available())


def _positive_tensor_extent(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TensorRuntimeError("tensor shape extents must be integers")
    if value < 0:
        raise TensorRuntimeError("tensor shape extents must be nonnegative")
    return value


def _tensor_element_dtype(*, runtime: TensorRuntime, dtype: TensorElementDType) -> Any:
    if dtype == "float32":
        return runtime.torch.float32
    if dtype == "int64":
        return runtime.torch.long
    raise TensorRuntimeError(f"unsupported tensor element dtype: {dtype}")


def _construct_tensor_element_program(
    *,
    runtime: TensorRuntime,
    shape: tuple[int, ...],
    dtype: Any,
    program: TensorElementProgram,
) -> Any:
    parameter_tensors = {
        name: _tensor_element_parameter(
            runtime=runtime,
            program=program,
            name=name,
            parameter=parameter,
        )
        for name, parameter in program.parameters.items()
    }
    if (
        runtime.device_kind in {"cuda", "mps"}
        and program.compile
        and _tensor_runtime_compile_available(runtime)
    ):
        return _construct_compiled_tensor_element_tiles(
            runtime=runtime,
            shape=shape,
            dtype=dtype,
            program=program,
            parameter_tensors=parameter_tensors,
        )
    return _construct_eager_tensor_element_program(
        runtime=runtime,
        shape=shape,
        dtype=dtype,
        program=program,
        parameter_tensors=parameter_tensors,
    )


def _construct_eager_tensor_element_program(
    *,
    runtime: TensorRuntime,
    shape: tuple[int, ...],
    dtype: Any,
    program: TensorElementProgram,
    parameter_tensors: Mapping[str, Any],
) -> Any:
    backend = runtime.torch
    total_elements = math.prod(shape)
    flat_indices = backend.arange(total_elements, dtype=backend.long, device=runtime.device)
    coordinate_tensors = _tensor_element_coordinate_tensors(
        shape=shape,
        flat_indices=flat_indices,
    )
    values = cast(Any, program.kernel(coordinate_tensors, flat_indices, **parameter_tensors))
    return values.to(dtype=dtype).reshape(shape)


def _construct_compiled_tensor_element_tiles(
    *,
    runtime: TensorRuntime,
    shape: tuple[int, ...],
    dtype: Any,
    program: TensorElementProgram,
    parameter_tensors: Mapping[str, Any],
) -> Any:
    with _tensor_runtime_profile_span(runtime, "leibniz.tensor_construct.compiled_tiles"):
        backend = runtime.torch
        total_elements = math.prod(shape)
        tile_positions = _tensor_element_tile_positions(runtime)
        extents, strides, total_tensor = _tensor_element_shape_metadata(
            runtime,
            shape=shape,
            total_elements=total_elements,
        )
        _mark_dynamic_tensor_element_parameters(
            runtime=runtime,
            parameter_declarations=program.parameters,
            parameter_tensors=parameter_tensors,
        )
        output = backend.empty((total_elements,), dtype=dtype, device=runtime.device)
        kernel = _compiled_tensor_element_tile_kernel(
            runtime=runtime,
            program=program,
            rank=len(shape),
            parameter_tensors=parameter_tensors,
        )
        for offset in range(0, total_elements, _tensor_element_tile_size):
            valid_count = min(_tensor_element_tile_size, total_elements - offset)
            offset_tensor = _tensor_element_offset_tensor(runtime, offset=offset)
            values = kernel(
                tile_positions,
                offset_tensor,
                extents,
                strides,
                total_tensor,
                **parameter_tensors,
            ).to(dtype=dtype)
            output[offset : offset + valid_count] = values[:valid_count]
        return output.reshape(shape)


def _tensor_element_parameter(
    *,
    runtime: TensorRuntime,
    program: TensorElementProgram,
    name: str,
    parameter: TensorElementParameter,
) -> Any:
    with _tensor_runtime_profile_span(runtime, f"leibniz.tensor_parameter.{name}"):
        expected_count = math.prod(
            tuple(_positive_tensor_extent(size) for size in parameter.shape)
        )
        if len(parameter.values) != expected_count:
            raise TensorRuntimeError("tensor element parameter value count does not match shape")
        dtype = _tensor_element_parameter_dtype(runtime=runtime, dtype=parameter.dtype)
        cache_key = _tensor_element_parameter_cache_key(
            runtime=runtime,
            program=program,
            name=name,
            parameter=parameter,
        )
        if cache_key is not None:
            cached = _tensor_element_parameter_cache.get(cache_key)
            if cached is not None:
                return cached
        tensor = runtime.torch.tensor(
            parameter.values,
            dtype=dtype,
            device=runtime.device,
        ).reshape(parameter.shape)
        if cache_key is not None:
            _tensor_element_parameter_cache[cache_key] = tensor
        return tensor


def _tensor_element_parameter_cache_key(
    *,
    runtime: TensorRuntime,
    program: TensorElementProgram,
    name: str,
    parameter: TensorElementParameter,
) -> tuple[object, ...] | None:
    if parameter.dynamic_axes:
        return None
    return (
        program.cache_key if program.cache_key is not None else id(program.kernel),
        runtime.device_kind,
        str(runtime.device),
        name,
        parameter.dtype,
        parameter.shape,
        _tensor_element_parameter_values_key(parameter),
    )


def _tensor_element_parameter_values_key(parameter: TensorElementParameter) -> bytes:
    if parameter.dtype == "float32":
        return array("f", (float(value) for value in parameter.values)).tobytes()
    if parameter.dtype == "int64":
        return array("q", (int(value) for value in parameter.values)).tobytes()
    raise TensorRuntimeError(f"unsupported tensor element parameter dtype: {parameter.dtype}")


def _tensor_element_parameter_dtype(
    *,
    runtime: TensorRuntime,
    dtype: TensorElementParameterDType,
) -> Any:
    if dtype == "float32":
        return runtime.torch.float32
    if dtype == "int64":
        return runtime.torch.long
    raise TensorRuntimeError(f"unsupported tensor element parameter dtype: {dtype}")


def _tensor_element_tile_positions(runtime: TensorRuntime) -> Any:
    key = (
        "tile-positions",
        runtime.device_kind,
        str(runtime.device),
        _tensor_element_tile_size,
    )
    cached = _tensor_element_tile_metadata_cache.get(key)
    if cached is not None:
        return cached
    tensor = runtime.torch.arange(
        _tensor_element_tile_size,
        dtype=runtime.torch.long,
        device=runtime.device,
    )
    _tensor_element_tile_metadata_cache[key] = tensor
    return tensor


def _tensor_element_shape_metadata(
    runtime: TensorRuntime,
    *,
    shape: tuple[int, ...],
    total_elements: int,
) -> tuple[Any, Any, Any]:
    key = (
        "shape-metadata",
        runtime.device_kind,
        str(runtime.device),
        shape,
        total_elements,
    )
    cached = _tensor_element_tile_metadata_cache.get(key)
    if cached is not None:
        return cast(tuple[Any, Any, Any], cached)
    extents = runtime.torch.tensor(shape, dtype=runtime.torch.long, device=runtime.device)
    strides = runtime.torch.tensor(
        _tensor_element_shape_strides(shape),
        dtype=runtime.torch.long,
        device=runtime.device,
    )
    total_tensor = runtime.torch.tensor(
        total_elements,
        dtype=runtime.torch.long,
        device=runtime.device,
    )
    result = (extents, strides, total_tensor)
    _tensor_element_tile_metadata_cache[key] = result
    return result


def _tensor_element_offset_tensor(runtime: TensorRuntime, *, offset: int) -> Any:
    key = (
        "offset",
        runtime.device_kind,
        str(runtime.device),
        offset,
    )
    cached = _tensor_element_tile_metadata_cache.get(key)
    if cached is not None:
        return cached
    tensor = runtime.torch.tensor(offset, dtype=runtime.torch.long, device=runtime.device)
    _tensor_element_tile_metadata_cache[key] = tensor
    return tensor


def _compiled_tensor_element_tile_kernel(
    *,
    runtime: TensorRuntime,
    program: TensorElementProgram,
    rank: int,
    parameter_tensors: Mapping[str, Any],
) -> Callable[..., Any]:
    key = (
        program.cache_key if program.cache_key is not None else id(program.kernel),
        runtime.device_kind,
        "tile",
        rank,
        _tensor_element_tile_size,
        tuple(
            (name, str(parameter_tensors[name].dtype))
            for name in sorted(parameter_tensors)
        ),
    )
    cached = _tensor_element_kernel_cache.get(key)
    if cached is not None:
        return cast(Callable[..., Any], cached)

    with _tensor_runtime_profile_span(runtime, "leibniz.tensor_construct.compile_lookup"):

        def tile_kernel(
            tile_positions: Any,
            offset: Any,
            extents: Any,
            strides: Any,
            total_elements: Any,
            **parameters: Any,
        ) -> Any:
            flat_indices = tile_positions + offset
            active = flat_indices < total_elements
            safe_flat_indices = flat_indices.clamp(max=total_elements - 1)
            coordinates: list[Any] = []
            remainder = safe_flat_indices
            for axis in range(rank):
                stride = strides[axis]
                coordinates.append(remainder.div(stride, rounding_mode="floor"))
                remainder = remainder.remainder(stride)
            values = cast(
                Any,
                program.kernel(tuple(coordinates), safe_flat_indices, **parameters),
            )
            return values.where(active, values * 0)

        try:
            compiled = runtime.torch.compile(tile_kernel)
        except Exception:
            compiled = tile_kernel
    _tensor_element_kernel_cache[key] = compiled
    return cast(Callable[..., Any], compiled)


def _mark_dynamic_tensor_element_parameters(
    *,
    runtime: TensorRuntime,
    parameter_declarations: Mapping[str, TensorElementParameter],
    parameter_tensors: Mapping[str, Any],
) -> None:
    mark_dynamic = getattr(getattr(runtime.torch, "_dynamo", None), "mark_dynamic", None)
    if not callable(mark_dynamic):
        return
    for name, parameter in parameter_declarations.items():
        tensor = parameter_tensors[name]
        for axis in parameter.dynamic_axes:
            mark_dynamic(tensor, axis)


def _tensor_runtime_compile_available(runtime: TensorRuntime) -> bool:
    compile_function = getattr(runtime.torch, "compile", None)
    if not callable(compile_function):
        return False
    if runtime.device_kind == "mps":
        return True
    try:
        compiler = importlib.import_module("triton.compiler.compiler")
    except ImportError:
        return False
    return hasattr(compiler, "triton_key")


def _tensor_runtime_profile_span(runtime: TensorRuntime, name: str) -> Any:
    profiler = getattr(runtime.torch, "profiler", None)
    record_function = getattr(profiler, "record_function", None)
    if callable(record_function):
        return record_function(name)
    return nullcontext()


def _tensor_element_coordinate_tensors(
    *,
    shape: tuple[int, ...],
    flat_indices: Any,
) -> tuple[Any, ...]:
    coordinates: list[Any] = []
    for stride in _tensor_element_shape_strides(shape):
        coordinates.append(flat_indices.div(stride, rounding_mode="floor"))
        flat_indices = flat_indices.remainder(stride)
    return tuple(coordinates)


def _tensor_element_shape_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()
    strides: list[int] = []
    stride = 1
    for extent in reversed(shape[1:]):
        stride *= extent
        strides.append(stride)
    strides.reverse()
    strides.append(1)
    return tuple(strides)


def _torch() -> Any:
    try:
        return cast(Any, importlib.import_module("torch"))
    except ImportError as error:
        raise TensorRuntimeError("PyTorch is required to run benchmark training") from error


def _positive_memory_bytes(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TensorRuntimeError(f"{field} must be numeric")
    bytes_value = int(value)
    if bytes_value < 1:
        raise TensorRuntimeError(f"{field} must be positive")
    return bytes_value


def _host_memory_bytes() -> int:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        page_size = page_count = -1
    if page_size > 0 and page_count > 0:
        return page_size * page_count
    return 1_073_741_824


def _require_int_parameter(parameters: Mapping[str, object], key: str) -> int:
    value = parameters.get(key)
    if type(value) is not int or value < 1:
        raise TensorRuntimeError(f"{key} must be a positive integer")
    return value


def _require_nonneg_int_parameter(parameters: Mapping[str, object], key: str) -> int:
    value = parameters.get(key)
    if type(value) is not int or value < 0:
        raise TensorRuntimeError(f"{key} must be a nonnegative integer")
    return value


def _require_fixed_support(parameters: Mapping[str, object]) -> tuple[int, int]:
    out_height = parameters.get("out_height")
    out_width = parameters.get("out_width")
    if type(out_height) is int and out_height > 0 and type(out_width) is int and out_width > 0:
        return out_height, out_width
    return _require_int_parameter(parameters, "size"), _require_int_parameter(parameters, "size")


def _local_window_output_size(axis: int, *, size: int, stride: int, padding: int) -> int:
    result = ((axis + 2 * padding - size) // stride) + 1
    if result < 1:
        raise TensorRuntimeError("local affine output axis must be positive")
    return result
