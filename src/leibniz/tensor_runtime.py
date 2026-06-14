"""Narrow tensor runtime adapter for local benchmark execution."""

from __future__ import annotations

import importlib
import inspect
import math
import os
import sys
import time
from array import array
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal, Self, cast

from leibniz.architectures import ArchitectureManifest
from leibniz.target_contracts import TargetContract
from leibniz.tensor_shapes import TensorShape

__all__ = [
    "architecture_tensor_runtime_issue",
    "architecture_supported_by_tensor_runtime",
    "build_architecture_modules",
    "build_architecture_sequential",
    "build_cosine_lr_schedule",
    "build_cross_entropy_loss",
    "build_loss",
    "build_mse_loss",
    "build_optimizer",
    "build_plateau_lr_schedule",
    "build_relative_l2_loss",
    "make_float_tensor",
    "make_empty_float_tensor",
    "make_long_tensor",
    "no_grad_context",
    "optimizer_step",
    "OperationFallbackSequential",
    "seed_runtime",
    "softmax_prediction_rows",
    "softmax_target_mass_tensor",
    "softmax_target_masses",
    "spatial_axis_names_for_dimension",
    "synchronize_runtime",
    "TensorRuntime",
    "TensorRuntimeOperationRecord",
    "TensorElementParameter",
    "TensorElementParameterDType",
    "TensorBatchProgram",
    "TensorKernelOps",
    "TensorFieldOps",
    "TensorSolverProgram",
    "TensorElementRecipe",
    "TensorElementDType",
    "TensorRuntimeError",
    "TensorRuntimeTensorSpec",
    "TensorRuntimeDevice",
    "TensorRuntimeDeviceKind",
    "tensor_element_compile_fallback_records",
    "tensor_runtime_available_memory_bytes",
    "tensor_runtime_capture_operations",
    "tensor_runtime_broadcast_zeros",
    "tensor_runtime_concat",
    "tensor_runtime_construct_tensor",
    "tensor_runtime_default_device",
    "tensor_runtime_device_choices",
    "tensor_runtime_device_kinds",
    "tensor_runtime_has_fixed_device_memory",
    "tensor_runtime_operation_capture",
    "tensor_runtime_project_operations",
    "tensor_runtime_profile_operator_rows",
    "tensor_runtime_solve_tensor",
    "tensor_runtime_solve_tensor_trajectory",
    "tensor_runtime_shape_element_count",
    "tensor_runtime_total_memory_bytes",
    "tensor_runtime_used_memory_bytes",
    "tensor_value_to_host",
    "tensor_value_to_host_values",
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
TensorElementDType = Literal["float32", "float64", "int64"]
TensorElementParameterDType = Literal["float32", "float64", "complex64", "complex128", "int64"]
_available_devices = frozenset({"auto", "cpu", "cuda", "mps"})
_roofline_cache: dict[str, dict[str, object]] = {}
_tensor_element_kernel_cache: dict[tuple[object, ...], Any] = {}
_tensor_element_compile_failure_cache: set[tuple[object, ...]] = set()
_tensor_element_compile_fallbacks: dict[tuple[object, ...], dict[str, object]] = {}
_tensor_element_parameter_cache: dict[tuple[object, ...], Any] = {}
_tensor_element_parameter_cache_capacity = 256
_tensor_element_parameter_values_key_cache: dict[
    tuple[int, TensorElementParameterDType],
    tuple[Sequence[int | float | complex], bytes],
] = {}
_tensor_batch_kernel_accepts_ops_cache: dict[Callable[..., object], bool] = {}
_tensor_element_parameter_values_key_cache_minimum_size = 64
_tensor_element_tile_size = 131_072
_require_tensor_compile_environment_variable = "LEIBNIZ_REQUIRE_TENSOR_COMPILE"
_require_device_residency_environment_variable = "LEIBNIZ_REQUIRE_DEVICE_RESIDENCY"


class TensorRuntimeError(ValueError):
    """Raised when a tensor runtime cannot be resolved."""


@dataclass(frozen=True, slots=True)
class TensorRuntime:
    """Resolved local tensor runtime for benchmark execution."""

    torch: Any
    device: Any
    device_kind: Literal["cpu", "cuda", "mps"]


@dataclass(frozen=True, slots=True)
class TensorRuntimeTensorSpec:
    """Backend tensor shape and dtype observed during runtime operation capture."""

    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True, slots=True)
class TensorRuntimeOperationRecord:
    """One captured tensor runtime operation with tensor inputs and outputs."""

    name: str
    arguments: tuple[object, ...]
    keyword_arguments: tuple[tuple[str, object], ...]
    input_tensors: tuple[TensorRuntimeTensorSpec, ...]
    output_tensors: tuple[TensorRuntimeTensorSpec, ...]


class _TensorRuntimeOperationCapture:
    """Context manager that captures a tensor runtime operation stream."""

    def __init__(self, runtime: TensorRuntime) -> None:
        self._runtime = runtime
        self._mode: object | None = None
        self._operations: list[TensorRuntimeOperationRecord] = []

    def __enter__(self) -> Self:
        if self._mode is not None:
            raise TensorRuntimeError("tensor runtime operation capture is already active")
        del self._runtime
        from torch.utils._python_dispatch import TorchDispatchMode

        parent = self

        class _OperationCaptureMode(TorchDispatchMode):
            def __torch_dispatch__(
                self,
                func: object,
                types: object,
                args: tuple[object, ...] = (),
                kwargs: Mapping[str, object] | None = None,
            ) -> object:
                del types
                keyword_arguments = dict(kwargs or {})
                input_specs = _tensor_runtime_tensor_specs_from_value(args)
                output = cast(Callable[..., object], func)(*args, **keyword_arguments)
                output_specs = _tensor_runtime_tensor_specs_from_value(output)
                if not output_specs:
                    return output
                parent._operations.append(
                    TensorRuntimeOperationRecord(
                        name=_tensor_runtime_operation_name(func),
                        arguments=tuple(
                            _tensor_runtime_operation_argument(arg) for arg in args
                        ),
                        keyword_arguments=tuple(
                            (key, _tensor_runtime_operation_argument(value))
                            for key, value in sorted(
                                keyword_arguments.items(),
                                key=lambda item: item[0],
                            )
                        ),
                        input_tensors=input_specs,
                        output_tensors=output_specs,
                    )
                )
                return output

        self._mode = _OperationCaptureMode()
        self._mode.__enter__()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._mode is None:
            return
        mode = self._mode
        self._mode = None
        mode.__exit__(exc_type, exc, traceback)  # type: ignore[attr-defined]

    def records(self) -> tuple[TensorRuntimeOperationRecord, ...]:
        return tuple(self._operations)


@dataclass(frozen=True, slots=True)
class TensorElementRecipe:
    """Elementwise tensor construction recipe supplied by benchmark implementations."""

    shape: tuple[int, ...]
    dtype: TensorElementDType
    program: TensorBatchProgram


@dataclass(frozen=True, slots=True)
class TensorElementParameter:
    """Numeric parameter buffer referenced by a tensor element program."""

    dtype: TensorElementParameterDType
    shape: tuple[int, ...]
    values: Sequence[int | float | complex]
    dynamic_axes: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class TensorBatchProgram:
    """Batch-rank tensor construction kernel over recipe axis coordinates."""

    kernel: Callable[..., object]
    parameters: Mapping[str, TensorElementParameter]
    compile: bool = True
    cache_key: object | None = None


@dataclass(frozen=True, slots=True)
class TensorKernelOps:
    """Backend-agnostic helpers passed to tensor batch kernels that accept ``ops``."""

    def broadcast_zeros(self, axis_coordinates: tuple[Any, ...]) -> Any:
        return tensor_runtime_broadcast_zeros(axis_coordinates)


@dataclass(frozen=True, slots=True)
class TensorFieldOps:
    """Small backend-owned field-operator namespace for solver step kernels."""

    torch: Any

    def fft(self, value: Any, axis: int) -> Any:
        return self.torch.fft.fft(value, dim=axis)

    def ifft(self, value: Any, axis: int) -> Any:
        return self.torch.fft.ifft(value, dim=axis)

    def real(self, value: Any) -> Any:
        return value.real


@dataclass(frozen=True, slots=True)
class TensorSolverProgram:
    """Sequential tensor program that evolves a field state by repeated steps."""

    initial_state: TensorBatchProgram
    step_kernel: Callable[..., object]
    step_count: int
    parameters: Mapping[str, TensorElementParameter]
    dtype: TensorElementDType = "float32"
    compile: bool = True
    cache_key: object | None = None


@dataclass(frozen=True, slots=True)
class _CompiledTensorElementParameters:
    tensors: Mapping[str, Any]
    scalar_aliases: Mapping[str, tuple[str, int]]


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


def tensor_runtime_solve_tensor(
    runtime: TensorRuntime,
    *,
    program: TensorSolverProgram,
    shape: Sequence[int],
) -> Any:
    """Construct an initial field and evolve it through a solver step program."""

    resolved_shape = tuple(_positive_tensor_extent(size) for size in shape)
    if type(program.step_count) is not int or program.step_count < 0:
        raise TensorRuntimeError("solver step_count must be a nonnegative integer")
    dtype = _tensor_element_dtype(runtime=runtime, dtype=program.dtype)
    state = tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(
            shape=resolved_shape,
            dtype=program.dtype,
            program=program.initial_state,
        ),
    )
    if program.step_count == 0:
        return state
    parameter_tensors = {
        name: _tensor_element_parameter(
            runtime=runtime,
            program=program,
            name=name,
            parameter=parameter,
        )
        for name, parameter in program.parameters.items()
    }
    return _solve_tensor_program(
        runtime=runtime,
        program=program,
        state=state,
        dtype=dtype,
        parameter_tensors=parameter_tensors,
        record_trajectory=False,
    )


def tensor_runtime_solve_tensor_trajectory(
    runtime: TensorRuntime,
    *,
    program: TensorSolverProgram,
    shape: Sequence[int],
) -> Any:
    """Construct and evolve a field, returning every state on a solver time axis.

    The returned tensor has shape ``(batch, channels, step_count + 1, *spatial)``
    for state tensors shaped ``(batch, channels, *spatial)``. The first time
    slice is the initial state and each following slice is one solver step.
    """

    resolved_shape = tuple(_positive_tensor_extent(size) for size in shape)
    if len(resolved_shape) < 2:
        raise TensorRuntimeError("solver trajectories require batch and channel axes")
    if type(program.step_count) is not int or program.step_count < 0:
        raise TensorRuntimeError("solver step_count must be a nonnegative integer")
    dtype = _tensor_element_dtype(runtime=runtime, dtype=program.dtype)
    state = tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(
            shape=resolved_shape,
            dtype=program.dtype,
            program=program.initial_state,
        ),
    )
    parameter_tensors = {
        name: _tensor_element_parameter(
            runtime=runtime,
            program=program,
            name=name,
            parameter=parameter,
        )
        for name, parameter in program.parameters.items()
    }
    return _solve_tensor_program(
        runtime=runtime,
        program=program,
        state=state,
        dtype=dtype,
        parameter_tensors=parameter_tensors,
        record_trajectory=True,
    )


def tensor_element_compile_fallback_records() -> tuple[dict[str, object], ...]:
    """Return process-wide records of element programs that fell back to eager construction.

    Compiled tile construction is the fast path for tensor element programs on
    accelerator runtimes; the eager fallback is orders of magnitude slower and
    must never be silent. Each record carries the runtime device kind, a stable
    program label, the first fallback reason, and the number of constructions
    served by the fallback. Setting the ``LEIBNIZ_REQUIRE_TENSOR_COMPILE``
    environment variable to a value other than ``0`` turns any fallback into a
    ``TensorRuntimeError`` instead.
    """

    return tuple(dict(record) for record in _tensor_element_compile_fallbacks.values())


def _note_tensor_element_compile_fallback(
    *,
    runtime: TensorRuntime,
    program: TensorBatchProgram | TensorSolverProgram,
    cache_key: tuple[object, ...],
    reason: str,
) -> None:
    if _tensor_element_compile_required():
        raise TensorRuntimeError(
            f"{_require_tensor_compile_environment_variable} is set but tensor "
            f"element program {_tensor_element_program_label(program)!r} fell "
            f"back to eager construction: {reason}"
        )
    record = _tensor_element_compile_fallbacks.get(cache_key)
    if record is None:
        _tensor_element_compile_fallbacks[cache_key] = {
            "kind": "tensor-element-compile-fallback",
            "tensor_device": runtime.device_kind,
            "program": _tensor_element_program_label(program),
            "reason": reason,
            "constructions": 1,
        }
        return
    record["constructions"] = cast(int, record["constructions"]) + 1


def _tensor_element_compile_required() -> bool:
    return os.environ.get(_require_tensor_compile_environment_variable, "") not in {"", "0"}


def _tensor_element_program_label(program: TensorBatchProgram | TensorSolverProgram) -> str:
    if program.cache_key is not None:
        return str(program.cache_key)
    kernel = (
        program.kernel
        if isinstance(program, TensorBatchProgram)
        else program.step_kernel
    )
    return getattr(kernel, "__qualname__", repr(kernel))


def tensor_runtime_device_choices() -> tuple[str, ...]:
    """Return the public tensor runtime device choices."""

    return ("auto", "cpu", "cuda", "mps")


def tensor_runtime_default_device() -> str:
    """Return the portable default tensor runtime device name."""

    return "cpu"


def resolve_host_tensor_runtime() -> TensorRuntime:
    """Resolve the portable host tensor runtime."""

    return resolve_tensor_runtime("cpu")


def tensor_runtime_shape_element_count(shape: tuple[int, ...]) -> int:
    """Return the tensor element count for a positive shape."""

    return TensorShape.from_axes(shape).element_count


def tensor_runtime_capture_operations(
    runtime: TensorRuntime,
    callback: Callable[[], object],
) -> tuple[TensorRuntimeOperationRecord, ...]:
    """Capture the tensor runtime operation stream produced by a callback."""

    with tensor_runtime_operation_capture(runtime) as capture:
        callback()
    return capture.records()


def tensor_runtime_operation_capture(runtime: TensorRuntime) -> _TensorRuntimeOperationCapture:
    """Create a context manager that captures tensor runtime operations."""

    return _TensorRuntimeOperationCapture(runtime)


def tensor_runtime_broadcast_zeros(axis_coordinates: tuple[Any, ...]) -> Any:
    """Return a broadcast-shaped zero tile from backend axis-coordinate tensors.

    Tensor batch kernels should use this helper when they need a zero-valued
    tile with the full coordinate broadcast shape. Prefer declaring a small
    number of packed parameter tensors over many scalar parameters; compiled
    backends cache and bind those consolidated buffers more predictably.
    """

    if not axis_coordinates:
        raise TensorRuntimeError("broadcast_zeros requires at least one axis coordinate")
    rank = len(axis_coordinates)
    result = None
    for axis, coordinate in enumerate(axis_coordinates):
        shape = (1,) * axis + (-1,) + (1,) * (rank - axis - 1)
        value = coordinate.reshape(shape) * 0
        result = value if result is None else result + value
    return result


def tensor_runtime_project_operations(
    runtime: TensorRuntime,
    callback: Callable[[], object],
) -> tuple[TensorRuntimeOperationRecord, ...]:
    """Project a tensor operation stream without executing tensor kernels."""

    try:
        from torch._subclasses.fake_tensor import FakeTensorMode
    except ImportError as error:
        raise TensorRuntimeError(
            "PyTorch FakeTensorMode is required for dry-run projection"
        ) from error

    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    with tensor_runtime_operation_capture(runtime) as capture, fake_mode:
        callback()
    return capture.records()


def tensor_value_to_host(value: Any) -> Any:
    """Detach a backend tensor-like value and move it to host memory when possible."""

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    move_to_host = getattr(value, "cpu", None)
    if callable(move_to_host):
        value = move_to_host()
    return value


def tensor_value_to_host_values(value: Any) -> list[float]:
    """Return a tensor-like value's elements as a flat host list of floats.

    This is the supported host-materialization path for benchmark
    implementations, which the repository policy bars from calling backend
    host-transfer methods directly.
    """

    host_value = tensor_value_to_host(value)
    reshape = getattr(host_value, "reshape", None)
    if callable(reshape):
        host_value = reshape(-1)
    to_list = getattr(host_value, "tolist", None)
    flat_values = cast(Any, to_list() if callable(to_list) else host_value)
    return [float(element) for element in flat_values]


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

        class Module(torch.nn.Module):
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
                        if _truthy_environment_flag(
                            _require_device_residency_environment_variable
                        ):
                            raise TensorRuntimeError(
                                "LEIBNIZ_REQUIRE_DEVICE_RESIDENCY blocked CPU "
                                f"fallback for operation {index}: {reason}"
                            ) from error
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
            pool_class = _adaptive_pool_class(torch, dimension=dimension)
            output_axes = _require_fixed_support(parameters, dimension=dimension)
            modules.append(pool_class(output_axes))
            shape = (*shape[: len(shape) - dimension], *output_axes)
        elif kind == "fixed-support-affine":
            dimension = _require_int_parameter(parameters, "dimension")
            pool_class = _adaptive_pool_class(torch, dimension=dimension)
            conv_class = _conv_class(torch, dimension=dimension)
            if len(shape) <= dimension:
                raise TensorRuntimeError(
                    "fixed support affine requires a channel axis before support axes"
                )
            channel_axis_index = len(shape) - dimension - 1
            out_channels = _require_int_parameter(parameters, "out_channels")
            output_axes = _require_fixed_support(parameters, dimension=dimension)
            modules.append(
                torch.nn.Sequential(
                    pool_class(output_axes),
                    conv_class(
                        in_channels=shape[channel_axis_index],
                        out_channels=out_channels,
                        kernel_size=1,
                    ),
                )
            )
            shape = (*shape[:channel_axis_index], out_channels, *output_axes)
        elif kind == "local-affine":
            dimension = _require_int_parameter(parameters, "dimension")
            conv_class = _conv_class(torch, dimension=dimension)
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
            padding_mode = _require_padding_mode(parameters)
            modules.append(
                conv_class(
                    in_channels=shape[channel_axis_index],
                    out_channels=out_channels,
                    kernel_size=size,
                    stride=stride,
                    padding=padding,
                    padding_mode=padding_mode,
                )
            )
            output_spatial_axes = tuple(
                _local_window_output_size(axis, size=size, stride=stride, padding=padding)
                for axis in shape[spatial_axis_start:]
            )
            shape = (*shape[:channel_axis_index], out_channels, *output_spatial_axes)
        elif kind == "rectified-linear-activation":
            modules.append(torch.nn.ReLU())
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


def tensor_runtime_profile_operator_rows(
    runtime: TensorRuntime,
    *,
    callback: Callable[[int], None],
    repeats: int,
    row_limit: int,
    record_name: str,
) -> tuple[dict[str, object], ...]:
    """Return compact backend operator profile rows for repeated runtime work."""

    profiler = getattr(runtime.torch, "profiler", None)
    if profiler is None:
        raise TensorRuntimeError("tensor runtime does not expose torch.profiler")
    _synchronize_runtime(runtime)
    activities = [profiler.ProfilerActivity.CPU]
    device_activity = _profiler_device_activity(runtime, profiler=profiler)
    if device_activity is not None:
        activities.append(device_activity)
    with profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profile:
        for offset in range(repeats):
            with profiler.record_function(record_name):
                callback(offset)
            profile.step()
    _synchronize_runtime(runtime)
    return tuple(
        _operator_profile_row(event)
        for event in sorted(
            profile.key_averages(),
            key=_operator_profile_sort_key,
            reverse=True,
        )[:row_limit]
    )


def seed_runtime(runtime: TensorRuntime, *, seed: int) -> None:
    """Set the global random seed for reproducible training."""

    _ = runtime
    _torch().manual_seed(seed)


def build_cross_entropy_loss(runtime: TensorRuntime) -> Any:
    """Build a cross-entropy classification loss module."""

    _ = runtime
    return _torch().nn.CrossEntropyLoss()


def build_mse_loss(runtime: TensorRuntime) -> Any:
    """Build a mean-squared-error loss module."""

    _ = runtime
    return _torch().nn.MSELoss()


def build_relative_l2_loss(runtime: TensorRuntime) -> Any:
    """Build a relative L2 loss closure."""

    _ = runtime
    torch = _torch()

    def relative_l2_loss(predictions: Any, targets: Any) -> Any:
        residual_norm = torch.linalg.vector_norm(predictions - targets)
        target_norm = torch.linalg.vector_norm(targets)
        return residual_norm / torch.maximum(
            target_norm,
            torch.tensor(1e-12, dtype=target_norm.dtype, device=target_norm.device),
        )

    return relative_l2_loss


def build_loss(runtime: TensorRuntime, contract: TargetContract) -> Any:
    """Build the tensor loss declared by a target contract."""

    if contract.loss_id == "cross-entropy":
        return build_cross_entropy_loss(runtime)
    if contract.loss_id == "mse":
        return build_mse_loss(runtime)
    if contract.loss_id == "relative-l2":
        return build_relative_l2_loss(runtime)
    raise TensorRuntimeError(f"unsupported tensor loss: {contract.loss_id}")


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

    return softmax_target_mass_tensor(runtime, logits, targets).detach().tolist()


def softmax_target_mass_tensor(
    runtime: TensorRuntime,
    logits: Any,
    targets: Any,
) -> Any:
    """Return target probability masses as a runtime tensor."""

    _ = runtime
    torch = _torch()
    probabilities = torch.softmax(logits.detach(), dim=1)
    if targets.shape == probabilities.shape:
        return (probabilities * targets).sum(dim=1).detach()
    target_indexes = targets.reshape((-1, 1)).long()
    return probabilities.gather(1, target_indexes).reshape((-1,)).detach()


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
    """Deterministic host-interactive reference optimizer.

    This optimizer intentionally evaluates candidate losses on the host during
    each step. Its line search is the default local reference path because it
    does not introduce a learning-rate knob; device hot-path gates must opt into
    a non-host-interactive optimizer such as Adam.
    """

    requires_loss_closure = True
    _beta1 = 0.9
    _beta2 = 0.999
    _epsilon = 1e-8
    _armijo_sufficient_decrease = 0.1
    _maximum_step_size = 1.0
    _minimum_step_size = 1e-8

    def __init__(self, parameters: Any) -> None:
        self._parameters = tuple(parameters)
        self.param_groups: list[dict[str, object]] = [{"params": list(self._parameters)}]
        self.state: dict[object, dict[str, object]] = {}
        self._accepted_step_size = 1e-3

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
        self._commit_direction_records(
            records=direction_records,
            step=next_step,
        )
        directional_derivative = sum(
            float((gradient * record["direction"]).sum().detach())
            for gradient, record in zip(gradients, direction_records, strict=True)
            if gradient is not None and record is not None
        )
        if directional_derivative <= 0.0 or not math.isfinite(directional_derivative):
            direction_records = tuple(
                None
                if gradient is None
                else {
                    "direction": gradient,
                    "exp_avg": record["exp_avg"] if record is not None else gradient,
                    "exp_avg_sq": (
                        record["exp_avg_sq"] if record is not None else gradient * gradient
                    ),
                }
                for gradient, record in zip(gradients, direction_records, strict=True)
            )
            directional_derivative = sum(
                float((gradient * record["direction"]).sum().detach())
                for gradient, record in zip(gradients, direction_records, strict=True)
                if gradient is not None and record is not None
            )
        if directional_derivative <= 0.0 or not math.isfinite(directional_derivative):
            return baseline_loss
        originals = tuple(parameter.detach().clone() for parameter in self._parameters)
        step_size = min(
            self._maximum_step_size,
            max(self._minimum_step_size, self._accepted_step_size * 2.0),
        )
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
            sufficient_decrease = (
                baseline_value
                - self._armijo_sufficient_decrease * step_size * directional_derivative
            )
            if math.isfinite(trial_value) and trial_value <= sufficient_decrease:
                self._accepted_step_size = min(step_size, self._maximum_step_size)
                return trial_loss
            step_size *= 0.5
            if step_size < self._minimum_step_size:
                break
        with torch.no_grad():
            for parameter, original in zip(self._parameters, originals, strict=True):
                parameter.copy_(original)
        self._accepted_step_size = max(
            min(self._accepted_step_size, step_size),
            self._minimum_step_size,
        )
        return baseline_loss

    def _commit_direction_records(
        self,
        *,
        records: Sequence[dict[str, Any] | None],
        step: int,
    ) -> None:
        for parameter, record in zip(self._parameters, records, strict=True):
            if record is not None:
                self.state[parameter] = {
                    "step": step,
                    "exp_avg": record["exp_avg"],
                    "exp_avg_sq": record["exp_avg_sq"],
                }

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


def make_empty_float_tensor(runtime: TensorRuntime, shape: Sequence[int], *, device: Any) -> Any:
    """Create an uninitialized float32 tensor on the given device."""

    _ = runtime
    torch = _torch()
    resolved_shape = tuple(_positive_tensor_extent(extent) for extent in shape)
    return torch.empty(resolved_shape, dtype=torch.float32, device=device)


def make_long_tensor(runtime: TensorRuntime, values: Any, *, device: Any) -> Any:
    """Create a long (int64) tensor on the given device."""

    _ = runtime
    torch = _torch()
    return torch.tensor(values, dtype=torch.long, device=device)


def tensor_runtime_concat(runtime: TensorRuntime, tensors: Sequence[Any], *, dim: int) -> Any:
    """Concatenate runtime tensors along an axis."""

    _ = runtime
    return _torch().cat(tuple(tensors), dim=dim)


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
        compute_points: list[dict[str, object]] = []
        peak_compute_per_second = 0.0
        chosen_matrix_size = 0
        matmul_repeats = 0
        matmul_seconds = 0.0
        for matrix_size in (512, 1024, 2048, 4096):
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
            repeats = 1
            seconds = 0.0
            while repeats <= 64 and seconds < 0.02:
                started = _monotonic_seconds()
                with torch.no_grad():
                    for _ in range(repeats):
                        _ = first @ second
                _synchronize_runtime(runtime)
                seconds = _monotonic_seconds() - started
                if seconds < 0.02:
                    repeats *= 2
            if seconds <= 0:
                continue
            flops = 2.0 * matrix_size * matrix_size * matrix_size * repeats
            compute_per_second = flops / seconds
            compute_points.append(
                {
                    "matrix_size": matrix_size,
                    "repeats": repeats,
                    "seconds": seconds,
                    "peak_compute_per_second": compute_per_second,
                }
            )
            previous_peak = peak_compute_per_second
            if compute_per_second > peak_compute_per_second:
                peak_compute_per_second = compute_per_second
                chosen_matrix_size = matrix_size
                matmul_repeats = repeats
                matmul_seconds = seconds
            if previous_peak > 0 and compute_per_second < previous_peak * 1.10:
                break

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
    if peak_compute_per_second <= 0 or matmul_seconds <= 0 or copy_seconds <= 0:
        record["reason"] = "roofline calibration completed too quickly to measure"
        return record
    copy_bytes = 2.0 * element_count * 4 * copy_repeats
    record.update(
        {
            "status": "calibrated",
            "compute_calibration_seconds": matmul_seconds,
            "compute_calibration_matrix_size": chosen_matrix_size,
            "compute_calibration_chosen_matrix_size": chosen_matrix_size,
            "compute_calibration_points": compute_points,
            "compute_calibration_repeats": matmul_repeats,
            "peak_compute_per_second": peak_compute_per_second,
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


def _profiler_device_activity(runtime: TensorRuntime, *, profiler: object) -> object | None:
    activities = cast(Any, profiler).ProfilerActivity
    if runtime.device_kind == "cuda":
        return getattr(activities, "CUDA", None)
    _ = profiler
    return None


def _operator_profile_sort_key(event: object) -> tuple[float, float]:
    return (
        float(getattr(event, "device_time_total", 0.0)),
        float(getattr(event, "cpu_time_total", 0.0)),
    )


def _operator_profile_row(event: object) -> dict[str, object]:
    typed_event = cast(Any, event)
    return {
        "name": str(typed_event.key),
        "calls": int(typed_event.count),
        "cpu_time_total_us": float(getattr(event, "cpu_time_total", 0.0)),
        "self_cpu_time_total_us": float(getattr(event, "self_cpu_time_total", 0.0)),
        "device_time_total_us": float(getattr(event, "device_time_total", 0.0)),
        "self_device_time_total_us": float(
            getattr(event, "self_device_time_total", 0.0)
        ),
        "cpu_memory_usage_bytes": int(getattr(event, "cpu_memory_usage", 0)),
        "self_cpu_memory_usage_bytes": int(
            getattr(event, "self_cpu_memory_usage", 0)
        ),
        "device_memory_usage_bytes": int(getattr(event, "device_memory_usage", 0)),
        "self_device_memory_usage_bytes": int(
            getattr(event, "self_device_memory_usage", 0)
        ),
    }


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


def _truthy_environment_flag(name: str) -> bool:
    return os.environ.get(name, "") not in {"", "0"}


def _positive_tensor_extent(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TensorRuntimeError("tensor shape extents must be integers")
    if value < 0:
        raise TensorRuntimeError("tensor shape extents must be nonnegative")
    return value


def _tensor_element_dtype(*, runtime: TensorRuntime, dtype: TensorElementDType) -> Any:
    if dtype == "float32":
        return runtime.torch.float32
    if dtype == "float64":
        return runtime.torch.float64
    if dtype == "int64":
        return runtime.torch.long
    raise TensorRuntimeError(f"unsupported tensor element dtype: {dtype}")


def _construct_tensor_element_program(
    *,
    runtime: TensorRuntime,
    shape: tuple[int, ...],
    dtype: Any,
    program: TensorBatchProgram,
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
    if runtime.device_kind in {"cuda", "mps"} and program.compile:
        compile_cache_key = _tensor_element_kernel_cache_key(
            runtime=runtime,
            program=program,
            rank=len(shape),
            parameter_tensors=parameter_tensors,
        )
        if not _tensor_runtime_compile_available(runtime):
            _note_tensor_element_compile_fallback(
                runtime=runtime,
                program=program,
                cache_key=compile_cache_key,
                reason="torch.compile is not available for this tensor runtime",
            )
        elif compile_cache_key in _tensor_element_compile_failure_cache:
            _note_tensor_element_compile_fallback(
                runtime=runtime,
                program=program,
                cache_key=compile_cache_key,
                reason="tensor element compile previously failed for this program",
            )
        else:
            try:
                return _construct_compiled_tensor_element_tiles(
                    runtime=runtime,
                    shape=shape,
                    dtype=dtype,
                    program=program,
                    parameter_tensors=parameter_tensors,
                    cache_key=compile_cache_key,
                )
            except Exception as error:
                _tensor_element_compile_failure_cache.add(compile_cache_key)
                _tensor_element_kernel_cache.pop(compile_cache_key, None)
                _note_tensor_element_compile_fallback(
                    runtime=runtime,
                    program=program,
                    cache_key=compile_cache_key,
                    reason=f"tensor element compile failed: {error}",
                )
    return _construct_eager_tensor_element_program(
        runtime=runtime,
        shape=shape,
        dtype=dtype,
        program=program,
        parameter_tensors=parameter_tensors,
    )


def _solve_tensor_program(
    *,
    runtime: TensorRuntime,
    program: TensorSolverProgram,
    state: Any,
    dtype: Any,
    parameter_tensors: Mapping[str, Any],
    record_trajectory: bool,
) -> Any:
    if runtime.device_kind in {"cuda", "mps"} and program.compile:
        compile_cache_key = _tensor_solver_kernel_cache_key(
            runtime=runtime,
            program=program,
            parameter_tensors=parameter_tensors,
        )
        if not _tensor_runtime_compile_available(runtime):
            _note_tensor_element_compile_fallback(
                runtime=runtime,
                program=program,
                cache_key=compile_cache_key,
                reason="torch.compile is not available for this tensor runtime",
            )
        elif compile_cache_key in _tensor_element_compile_failure_cache:
            _note_tensor_element_compile_fallback(
                runtime=runtime,
                program=program,
                cache_key=compile_cache_key,
                reason="tensor solver compile previously failed for this program",
            )
        else:
            try:
                step = _compiled_tensor_solver_step_kernel(
                    runtime=runtime,
                    program=program,
                    parameter_tensors=parameter_tensors,
                    cache_key=compile_cache_key,
                )
                return _run_tensor_solver_steps(
                    runtime=runtime,
                    program=program,
                    state=state,
                    dtype=dtype,
                    step=step,
                    record_trajectory=record_trajectory,
                )
            except Exception as error:
                _tensor_element_compile_failure_cache.add(compile_cache_key)
                _tensor_element_kernel_cache.pop(compile_cache_key, None)
                _note_tensor_element_compile_fallback(
                    runtime=runtime,
                    program=program,
                    cache_key=compile_cache_key,
                    reason=f"tensor solver compile failed: {error}",
                )
    step = _eager_tensor_solver_step_kernel(
        runtime=runtime,
        program=program,
        parameter_tensors=parameter_tensors,
    )
    return _run_tensor_solver_steps(
        runtime=runtime,
        program=program,
        state=state,
        dtype=dtype,
        step=step,
        record_trajectory=record_trajectory,
    )


def _run_tensor_solver_steps(
    *,
    runtime: TensorRuntime,
    program: TensorSolverProgram,
    state: Any,
    dtype: Any,
    step: Callable[[Any], Any],
    record_trajectory: bool,
) -> Any:
    expected_shape = tuple(int(extent) for extent in state.shape)
    states = [state] if record_trajectory else None
    for _step_index in range(program.step_count):
        state = step(state).to(dtype=dtype)
        if tuple(int(extent) for extent in state.shape) != expected_shape:
            raise TensorRuntimeError("solver step kernel must preserve state shape")
        if state.dtype != dtype:
            raise TensorRuntimeError("solver step kernel must preserve state dtype")
        if states is not None:
            states.append(state)
    if states is not None:
        return runtime.torch.stack(states, dim=2)
    return state


def _eager_tensor_solver_step_kernel(
    *,
    runtime: TensorRuntime,
    program: TensorSolverProgram,
    parameter_tensors: Mapping[str, Any],
) -> Callable[[Any], Any]:
    ops = TensorFieldOps(runtime.torch)

    def step(state: Any) -> Any:
        return cast(Any, program.step_kernel(state, ops, **parameter_tensors))

    return step


def _compiled_tensor_solver_step_kernel(
    *,
    runtime: TensorRuntime,
    program: TensorSolverProgram,
    parameter_tensors: Mapping[str, Any],
    cache_key: tuple[object, ...],
) -> Callable[[Any], Any]:
    cached = _tensor_element_kernel_cache.get(cache_key)
    if cached is not None:
        return cast(Callable[[Any], Any], cached)
    with _tensor_runtime_profile_span(runtime, "leibniz.tensor_solve.compile_lookup"):
        ops = TensorFieldOps(runtime.torch)

        def step(state: Any) -> Any:
            return cast(Any, program.step_kernel(state, ops, **parameter_tensors))

        _clear_stale_compile_module_alias(program.step_kernel)
        compiled = runtime.torch.compile(step)
    _tensor_element_kernel_cache[cache_key] = compiled
    return cast(Callable[[Any], Any], compiled)


def _construct_eager_tensor_element_program(
    *,
    runtime: TensorRuntime,
    shape: tuple[int, ...],
    dtype: Any,
    program: TensorBatchProgram,
    parameter_tensors: Mapping[str, Any],
) -> Any:
    output = runtime.torch.empty(shape, dtype=dtype, device=runtime.device)
    for start, stop, axis_coordinates in _tensor_batch_axis_coordinate_chunks(
        runtime=runtime,
        shape=shape,
    ):
        values = cast(
            Any,
            program.kernel(
                axis_coordinates,
                **_tensor_batch_kernel_arguments(
                    program=program,
                    parameter_tensors=parameter_tensors,
                ),
            ),
        )
        output[start:stop] = values.to(dtype=dtype)
    return output


def _construct_compiled_tensor_element_tiles(
    *,
    runtime: TensorRuntime,
    shape: tuple[int, ...],
    dtype: Any,
    program: TensorBatchProgram,
    parameter_tensors: Mapping[str, Any],
    cache_key: tuple[object, ...],
) -> Any:
    with _tensor_runtime_profile_span(runtime, "leibniz.tensor_construct.compiled_tiles"):
        backend = runtime.torch
        compiled_parameters = _compiled_tensor_element_parameters(
            runtime=runtime,
            declarations=program.parameters,
            parameter_tensors=parameter_tensors,
        )
        _mark_dynamic_tensor_element_parameters(
            runtime=runtime,
            parameter_declarations=program.parameters,
            parameter_tensors=compiled_parameters.tensors,
        )
        output = backend.empty(shape, dtype=dtype, device=runtime.device)
        kernel = _compiled_tensor_element_tile_kernel(
            runtime=runtime,
            program=program,
            rank=len(shape),
            parameter_tensors=compiled_parameters.tensors,
            scalar_aliases=compiled_parameters.scalar_aliases,
            cache_key=cache_key,
        )
        for start, stop, axis_coordinates in _tensor_batch_axis_coordinate_chunks(
            runtime=runtime,
            shape=shape,
        ):
            values = kernel(
                axis_coordinates,
                **compiled_parameters.tensors,
            ).to(dtype=dtype)
            output[start:stop] = values
        return output


def _compiled_tensor_element_parameters(
    *,
    runtime: TensorRuntime,
    declarations: Mapping[str, TensorElementParameter],
    parameter_tensors: Mapping[str, Any],
) -> _CompiledTensorElementParameters:
    packed_tensors: dict[str, Any] = {}
    scalar_aliases: dict[str, tuple[str, int]] = {}
    scalar_groups: dict[TensorElementParameterDType, list[tuple[str, Any]]] = {
        "float32": [],
        "float64": [],
        "complex64": [],
        "complex128": [],
        "int64": [],
    }
    for name, declaration in declarations.items():
        tensor = parameter_tensors[name]
        if declaration.shape == () and not declaration.dynamic_axes:
            scalar_groups[declaration.dtype].append((name, tensor.reshape(())))
        else:
            packed_tensors[name] = tensor
    for dtype, entries in scalar_groups.items():
        if not entries:
            continue
        packed_name = f"__scalar_{dtype}_parameters"
        packed_tensors[packed_name] = runtime.torch.stack(
            tuple(tensor for _name, tensor in entries)
        )
        for index, (name, _tensor) in enumerate(entries):
            scalar_aliases[name] = (packed_name, index)
    return _CompiledTensorElementParameters(
        tensors=packed_tensors,
        scalar_aliases=scalar_aliases,
    )


def _tensor_element_parameter(
    *,
    runtime: TensorRuntime,
    program: TensorBatchProgram | TensorSolverProgram,
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
            while len(_tensor_element_parameter_cache) >= _tensor_element_parameter_cache_capacity:
                _tensor_element_parameter_cache.pop(
                    next(iter(_tensor_element_parameter_cache))
                )
            _tensor_element_parameter_cache[cache_key] = tensor
        return tensor


def _tensor_element_parameter_cache_key(
    *,
    runtime: TensorRuntime,
    program: TensorBatchProgram | TensorSolverProgram,
    name: str,
    parameter: TensorElementParameter,
) -> tuple[object, ...] | None:
    kernel = (
        program.kernel
        if isinstance(program, TensorBatchProgram)
        else program.step_kernel
    )
    return (
        program.cache_key if program.cache_key is not None else id(kernel),
        runtime.device_kind,
        str(runtime.device),
        name,
        parameter.dtype,
        parameter.shape,
        _tensor_element_parameter_values_key(parameter),
    )


def _tensor_element_parameter_values_key(parameter: TensorElementParameter) -> bytes:
    cacheable = len(parameter.values) >= _tensor_element_parameter_values_key_cache_minimum_size
    if cacheable:
        cache_key = (id(parameter.values), parameter.dtype)
        cached = _tensor_element_parameter_values_key_cache.get(cache_key)
        if cached is not None and cached[0] is parameter.values:
            return cached[1]
    if parameter.dtype == "float32":
        real_values = cast(Sequence[int | float], parameter.values)
        values_key = array("f", (float(value) for value in real_values)).tobytes()
    elif parameter.dtype == "float64":
        real_values = cast(Sequence[int | float], parameter.values)
        values_key = array("d", (float(value) for value in real_values)).tobytes()
    elif parameter.dtype == "complex64":
        values_key = array(
            "f",
            (
                component
                for value in parameter.values
                for component in (complex(value).real, complex(value).imag)
            ),
        ).tobytes()
    elif parameter.dtype == "complex128":
        values_key = array(
            "d",
            (
                component
                for value in parameter.values
                for component in (complex(value).real, complex(value).imag)
            ),
        ).tobytes()
    elif parameter.dtype == "int64":
        integer_values = cast(Sequence[int], parameter.values)
        values_key = array("q", (int(value) for value in integer_values)).tobytes()
    else:
        raise TensorRuntimeError(f"unsupported tensor element parameter dtype: {parameter.dtype}")
    if cacheable:
        _tensor_element_parameter_values_key_cache[(id(parameter.values), parameter.dtype)] = (
            parameter.values,
            values_key,
        )
    return values_key


def _tensor_element_parameter_dtype(
    *,
    runtime: TensorRuntime,
    dtype: TensorElementParameterDType,
) -> Any:
    if dtype == "float32":
        return runtime.torch.float32
    if dtype == "float64":
        return runtime.torch.float64
    if dtype == "complex64":
        return runtime.torch.complex64
    if dtype == "complex128":
        return runtime.torch.complex128
    if dtype == "int64":
        return runtime.torch.long
    raise TensorRuntimeError(f"unsupported tensor element parameter dtype: {dtype}")


def _tensor_batch_axis_coordinate_chunks(
    *,
    runtime: TensorRuntime,
    shape: tuple[int, ...],
) -> tuple[tuple[int, int, tuple[Any, ...]], ...]:
    if not shape:
        scalar_coordinate = runtime.torch.arange(
            1,
            dtype=runtime.torch.long,
            device=runtime.device,
        )
        return ((0, 1, (scalar_coordinate,)),)
    leading_extent = shape[0]
    trailing_element_count = math.prod(shape[1:]) if len(shape) > 1 else 1
    chunk_extent = max(
        1,
        min(leading_extent, _tensor_element_tile_size // trailing_element_count),
    )
    trailing_coordinates = tuple(
        runtime.torch.arange(extent, dtype=runtime.torch.long, device=runtime.device)
        for extent in shape[1:]
    )
    chunks: list[tuple[int, int, tuple[Any, ...]]] = []
    for start in range(0, leading_extent, chunk_extent):
        stop = min(leading_extent, start + chunk_extent)
        leading_coordinates = runtime.torch.arange(
            start,
            stop,
            dtype=runtime.torch.long,
            device=runtime.device,
        )
        _mark_dynamic_leading_axis_coordinate(runtime, leading_coordinates)
        chunks.append((start, stop, (leading_coordinates, *trailing_coordinates)))
    return tuple(chunks)


def _mark_dynamic_leading_axis_coordinate(runtime: TensorRuntime, tensor: Any) -> None:
    mark_dynamic = getattr(getattr(runtime.torch, "_dynamo", None), "mark_dynamic", None)
    if callable(mark_dynamic):
        mark_dynamic(tensor, 0)


def _compiled_tensor_element_tile_kernel(
    *,
    runtime: TensorRuntime,
    program: TensorBatchProgram,
    rank: int,
    parameter_tensors: Mapping[str, Any],
    scalar_aliases: Mapping[str, tuple[str, int]],
    cache_key: tuple[object, ...],
) -> Callable[..., Any]:
    cached = _tensor_element_kernel_cache.get(cache_key)
    if cached is not None:
        return cast(Callable[..., Any], cached)

    with _tensor_runtime_profile_span(runtime, "leibniz.tensor_construct.compile_lookup"):
        def tile_kernel(
            axis_coordinates: tuple[Any, ...],
            **parameters: Any,
        ) -> Any:
            packed_names = {packed_name for packed_name, _index in scalar_aliases.values()}
            kernel_parameters = {
                name: value
                for name, value in parameters.items()
                if name not in packed_names
            }
            for name, (packed_name, index) in scalar_aliases.items():
                kernel_parameters[name] = parameters[packed_name][index]
            kernel_arguments = _tensor_batch_kernel_arguments(
                program=program,
                parameter_tensors=kernel_parameters,
            )
            _ = rank
            values = cast(
                Any,
                program.kernel(
                    axis_coordinates,
                    **kernel_arguments,
                ),
            )
            return values

        _clear_stale_compile_module_alias(program.kernel)
        compiled = runtime.torch.compile(tile_kernel)
    _tensor_element_kernel_cache[cache_key] = compiled
    return cast(Callable[..., Any], compiled)


def _tensor_batch_kernel_arguments(
    *,
    program: TensorBatchProgram,
    parameter_tensors: Mapping[str, Any],
) -> dict[str, Any]:
    arguments = dict(parameter_tensors)
    if not _tensor_batch_kernel_accepts_ops(program.kernel):
        return arguments
    if "ops" in arguments:
        raise TensorRuntimeError("tensor batch parameter name 'ops' is reserved")
    arguments["ops"] = TensorKernelOps()
    return arguments


def _tensor_batch_kernel_accepts_ops(kernel: Callable[..., object]) -> bool:
    cached = _tensor_batch_kernel_accepts_ops_cache.get(kernel)
    if cached is not None:
        return cached
    try:
        signature = inspect.signature(kernel)
    except (TypeError, ValueError):
        accepts = False
    else:
        accepts = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD or name == "ops"
            for name, parameter in signature.parameters.items()
        )
    _tensor_batch_kernel_accepts_ops_cache[kernel] = accepts
    return accepts


def _tensor_element_kernel_cache_key(
    *,
    runtime: TensorRuntime,
    program: TensorBatchProgram,
    rank: int,
    parameter_tensors: Mapping[str, Any],
) -> tuple[object, ...]:
    return (
        _tensor_element_program_scope_key(program.kernel),
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


def _tensor_solver_kernel_cache_key(
    *,
    runtime: TensorRuntime,
    program: TensorSolverProgram,
    parameter_tensors: Mapping[str, Any],
) -> tuple[object, ...]:
    return (
        _tensor_element_program_scope_key(program.step_kernel),
        program.cache_key if program.cache_key is not None else id(program.step_kernel),
        runtime.device_kind,
        "solver-step",
        tuple(
            (name, str(parameter_tensors[name].dtype))
            for name in sorted(parameter_tensors)
        ),
    )


def _tensor_element_program_scope_key(kernel: Callable[..., object]) -> int:
    globals_mapping = getattr(kernel, "__globals__", None)
    if isinstance(globals_mapping, dict):
        return id(cast(object, globals_mapping))
    return id(kernel)


def _clear_stale_compile_module_alias(kernel: Callable[..., object]) -> None:
    module_name = getattr(kernel, "__module__", None)
    if not isinstance(module_name, str):
        return
    current_module = sys.modules.get(module_name)
    if current_module is None:
        return
    alias = "__import_" + module_name.replace(".", "_dot_")
    existing_module = globals().get(alias)
    if existing_module is not None and existing_module is not current_module:
        del globals()[alias]


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
        tensor = parameter_tensors.get(name)
        if tensor is None:
            continue
        for axis in parameter.dynamic_axes:
            mark_dynamic(tensor, axis)


def _tensor_runtime_compile_available(runtime: TensorRuntime) -> bool:
    compile_function = getattr(runtime.torch, "compile", None)
    if not callable(compile_function):
        return False
    if runtime.device_kind == "mps":
        return True
    try:
        triton_support = importlib.import_module("torch.utils._triton")
    except ImportError:
        return False
    has_triton = getattr(triton_support, "has_triton", None)
    if not callable(has_triton):
        return False
    try:
        return bool(has_triton())
    except Exception:  # pragma: no cover - backend probe failure
        return False


def _tensor_runtime_profile_span(runtime: TensorRuntime, name: str) -> Any:
    profiler = getattr(runtime.torch, "profiler", None)
    record_function = getattr(profiler, "record_function", None)
    if callable(record_function):
        return record_function(name)
    return nullcontext()


def _torch() -> Any:
    try:
        return cast(Any, importlib.import_module("torch"))
    except ImportError as error:
        raise TensorRuntimeError("PyTorch is required to run benchmark training") from error


def _tensor_runtime_operation_name(func: object) -> str:
    raw = str(func)
    if raw.startswith("aten."):
        return raw
    name = getattr(func, "__name__", None)
    if isinstance(name, str) and name:
        return f"aten.{name}"
    return raw


def _tensor_runtime_tensor_specs_from_value(
    value: object,
) -> tuple[TensorRuntimeTensorSpec, ...]:
    specs: list[TensorRuntimeTensorSpec] = []
    _append_tensor_runtime_tensor_specs(specs, value)
    return tuple(specs)


def _append_tensor_runtime_tensor_specs(
    specs: list[TensorRuntimeTensorSpec],
    value: object,
) -> None:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            specs.append(
                TensorRuntimeTensorSpec(
                    shape=tuple(int(extent) for extent in shape),
                    dtype=str(dtype).removeprefix("torch."),
                )
            )
            return
        except TypeError:
            pass
    if isinstance(value, Mapping):
        for item in cast(Mapping[object, object], value).values():
            _append_tensor_runtime_tensor_specs(specs, item)
        return
    if isinstance(value, (tuple, list)):
        for item in cast(Sequence[object], value):
            _append_tensor_runtime_tensor_specs(specs, item)


def _tensor_runtime_operation_argument(value: object) -> object:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            return TensorRuntimeTensorSpec(
                shape=tuple(int(extent) for extent in shape),
                dtype=str(dtype).removeprefix("torch."),
            )
        except TypeError:
            pass
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _tensor_runtime_operation_argument(item))
            for key, item in sorted(
                cast(Mapping[object, object], value).items(),
                key=lambda item: str(item[0]),
            )
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            _tensor_runtime_operation_argument(item) for item in cast(Sequence[object], value)
        )
    return repr(value)


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


def _require_fixed_support(
    parameters: Mapping[str, object],
    *,
    dimension: int,
) -> tuple[int, ...]:
    axis_names = _spatial_axis_names(dimension)
    values = tuple(parameters.get(name) for name in axis_names)
    if any(value is not None for value in values):
        if any(type(value) is not int or value < 1 for value in values):
            raise TensorRuntimeError("fixed support axes must be positive integers")
        return cast(tuple[int, ...], values)
    size = _require_int_parameter(parameters, "size")
    return tuple(size for _index in range(dimension))


# Single source of truth for fixed-support output-axis parameter names per
# spatial dimension; the semantics layer (operator_interpretation) reads it
# through spatial_axis_names_for_dimension rather than keeping a second copy.
# Names exist for every interpretable dimension (1-3); which dimensions can
# actually be *built* is gated separately by the pool/conv class maps below
# ({1, 2} today).
_spatial_axis_names_by_dimension: dict[int, tuple[str, ...]] = {
    1: ("out_length",),
    2: ("out_height", "out_width"),
    3: ("out_depth", "out_height", "out_width"),
}


def spatial_axis_names_for_dimension(dimension: int) -> tuple[str, ...] | None:
    """Return fixed-support output-axis parameter names for a spatial dimension."""

    return _spatial_axis_names_by_dimension.get(dimension)


def _spatial_axis_names(dimension: int) -> tuple[str, ...]:
    names = _spatial_axis_names_by_dimension.get(dimension)
    if names is None:
        raise TensorRuntimeError(
            f"local support has no spatial-axis names for dimension {dimension}"
        )
    return names


def _adaptive_pool_class(torch: Any, *, dimension: int) -> Any:
    if dimension == 1:
        return torch.nn.AdaptiveAvgPool1d
    if dimension == 2:
        return torch.nn.AdaptiveAvgPool2d
    raise TensorRuntimeError("local aggregation currently supports dimensions 1 and 2")


def _conv_class(torch: Any, *, dimension: int) -> Any:
    if dimension == 1:
        return torch.nn.Conv1d
    if dimension == 2:
        return torch.nn.Conv2d
    raise TensorRuntimeError("local affine currently supports dimensions 1 and 2")


def _require_padding_mode(parameters: Mapping[str, object]) -> str:
    value = parameters.get("padding_mode", "zeros")
    if value == "zeros":
        return "zeros"
    if value == "periodic":
        return "circular"
    raise TensorRuntimeError("padding_mode must be one of: zeros, periodic")


def _local_window_output_size(axis: int, *, size: int, stride: int, padding: int) -> int:
    result = ((axis + 2 * padding - size) // stride) + 1
    if result < 1:
        raise TensorRuntimeError("local affine output axis must be positive")
    return result
