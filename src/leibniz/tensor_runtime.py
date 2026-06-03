"""Narrow tensor runtime adapter for local benchmark execution."""

from __future__ import annotations

import importlib
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from leibniz.architectures import ArchitectureManifest
from leibniz.observation_formation import (
    AffineMatrix2D,
    ObservationFormationDeclaration,
    VariationCoordinate,
    affine_translation,
    linear_affine_matrix,
    sequence_center,
    sequence_relative_translation,
)
from leibniz.observation_generation import GeneratedFormationBatch
from leibniz.tensor_shapes import TensorShape

__all__ = [
    "apply_softmax_predictions",
    "architecture_tensor_runtime_issue",
    "architecture_supported_by_tensor_runtime",
    "build_architecture_modules",
    "build_architecture_sequential",
    "build_cosine_lr_schedule",
    "build_cross_entropy_loss",
    "build_optimizer",
    "build_plateau_lr_schedule",
    "FormationTensorCache",
    "make_float_tensor",
    "make_long_tensor",
    "no_grad_context",
    "OperationFallbackSequential",
    "preferred_tensor_runtime_device_kind",
    "seed_runtime",
    "synchronize_runtime",
    "TensorRuntime",
    "TensorRuntimeError",
    "TensorRuntimeDevice",
    "TensorRuntimeDeviceKind",
    "tensor_runtime_available_memory_bytes",
    "tensor_runtime_device_kinds",
    "resolve_tensor_runtime",
    "runtime_roofline_record",
    "save_tensor_runtime_state",
    "load_tensor_runtime_state",
    "validate_tensor_runtime_device",
]

TensorRuntimeDevice = Literal["auto", "cpu", "cuda", "mps"]
TensorRuntimeDeviceKind = Literal["cpu", "cuda", "mps"]
_available_devices = frozenset({"auto", "cpu", "cuda", "mps"})
_roofline_cache: dict[str, dict[str, object]] = {}


class TensorRuntimeError(ValueError):
    """Raised when a tensor runtime cannot be resolved."""


@dataclass(frozen=True, slots=True)
class TensorRuntime:
    """Resolved local tensor runtime for benchmark execution."""

    torch: Any
    device: Any
    device_kind: Literal["cpu", "cuda", "mps"]


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


@dataclass(slots=True)
class FormationTensorCache:
    """Cache declaration-backed unvaried component fields as runtime tensors."""

    runtime: TensorRuntime
    formation: ObservationFormationDeclaration
    _component_tensors: dict[tuple[int, int, int, int, int], Any] = field(
        default_factory=lambda: cast(dict[tuple[int, int, int, int, int], Any], {})
    )

    def component_sequence_tensor(
        self,
        *,
        width: int,
        height: int,
        component_sequence: Sequence[int],
    ) -> Any:
        """Return an unvaried formed-field tensor for a component sequence."""

        _require_positive_integer(width, "width")
        _require_positive_integer(height, "height")
        sequence = tuple(component_sequence)
        if not sequence:
            raise TensorRuntimeError("component_sequence must not be empty")
        tensors = [
            self.component_tensor(
                width=width,
                height=height,
                sequence_length=len(sequence),
                sequence_index=sequence_index,
                component_index=component_index,
            )
            for sequence_index, component_index in enumerate(sequence)
        ]
        return self.runtime.torch.stack(tensors).amax(dim=0)

    def varied_component_sequence_tensor(
        self,
        *,
        width: int,
        height: int,
        component_sequence: Sequence[int],
        variation_coordinates: Sequence[Mapping[str, object]],
    ) -> Any:
        """Return a formed-field tensor with recorded per-position variation applied."""

        _require_positive_integer(width, "width")
        _require_positive_integer(height, "height")
        sequence = tuple(component_sequence)
        if not sequence:
            raise TensorRuntimeError("component_sequence must not be empty")
        coordinates = tuple(variation_coordinates)
        if len(coordinates) != len(sequence):
            raise TensorRuntimeError("variation_coordinates length must match sequence length")
        source_tensors: list[Any] = []
        affine_rows: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        for sequence_index, component_index in enumerate(sequence):
            coordinate = _variation_coordinate(
                coordinates[sequence_index],
                expected_sequence_index=sequence_index,
            )
            source_tensors.append(
                self.component_tensor(
                    width=width,
                    height=height,
                    sequence_length=len(sequence),
                    sequence_index=sequence_index,
                    component_index=component_index,
                )
            )
            affine_rows.append(
                _affine_grid_row(
                    coordinate=coordinate,
                    sequence_length=len(sequence),
                    sequence_index=sequence_index,
                    placement_axis=self.formation.sequence_layout.placement_axis,
                    width=width,
                    height=height,
                )
            )
        torch = self.runtime.torch
        sources = torch.stack(source_tensors)
        theta = torch.tensor(
            affine_rows,
            dtype=torch.float32,
            device=self.runtime.device,
        )
        grid = torch.nn.functional.affine_grid(
            theta,
            sources.shape,
            align_corners=False,
        )
        transformed = torch.nn.functional.grid_sample(
            sources,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return transformed.amax(dim=0)

    def batch_tensors(
        self,
        *,
        batch: GeneratedFormationBatch,
        outcome_ids: tuple[str, ...],
    ) -> tuple[Any, Any]:
        """Return field and label tensors for a generated formation batch."""

        if not outcome_ids:
            raise TensorRuntimeError("outcome_ids must not be empty")
        fields = self._varied_batch_tensor(batch=batch)
        labels = self.runtime.torch.tensor(
            [outcome_ids.index(sample.outcome_id) for sample in batch.samples],
            dtype=self.runtime.torch.long,
            device=self.runtime.device,
        )
        return fields, labels

    def _varied_batch_tensor(self, *, batch: GeneratedFormationBatch) -> Any:
        sample_count = len(batch.samples)
        if sample_count < 1:
            raise TensorRuntimeError("batch samples must not be empty")
        width = batch.samples[0].width
        height = batch.samples[0].height
        sequence_length = len(batch.samples[0].component_sequence)
        if sequence_length < 1:
            raise TensorRuntimeError("component_sequence must not be empty")
        source_tensors: list[Any] = []
        affine_rows: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        for sample in batch.samples:
            if sample.width != width or sample.height != height:
                raise TensorRuntimeError("batch sample canvas shapes must match")
            if len(sample.component_sequence) != sequence_length:
                raise TensorRuntimeError("batch sample sequence lengths must match")
            if len(sample.variation_coordinates) != sequence_length:
                raise TensorRuntimeError("variation_coordinates length must match sequence length")
            for sequence_index, component_index in enumerate(sample.component_sequence):
                source_tensors.append(
                    self.component_tensor(
                        width=width,
                        height=height,
                        sequence_length=sequence_length,
                        sequence_index=sequence_index,
                        component_index=component_index,
                    )
                )
                affine_rows.append(
                    _generated_affine_grid_row(
                        sample.variation_coordinates[sequence_index],
                        sequence_length=sequence_length,
                        sequence_index=sequence_index,
                        placement_axis=self.formation.sequence_layout.placement_axis,
                        width=width,
                        height=height,
                    )
                )
        torch = self.runtime.torch
        sources = torch.stack(source_tensors)
        theta = torch.tensor(
            affine_rows,
            dtype=torch.float32,
            device=self.runtime.device,
        )
        grid = torch.nn.functional.affine_grid(
            theta,
            sources.shape,
            align_corners=False,
        )
        transformed = torch.nn.functional.grid_sample(
            sources,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        channels, height, width = transformed.shape[1:]
        return transformed.reshape((sample_count, sequence_length, channels, height, width)).amax(
            dim=1
        )

    def component_tensor(
        self,
        *,
        width: int,
        height: int,
        sequence_length: int,
        sequence_index: int,
        component_index: int,
    ) -> Any:
        """Return a cached tensor for one component drawn at one sequence position."""

        _require_positive_integer(width, "width")
        _require_positive_integer(height, "height")
        _require_positive_integer(sequence_length, "sequence_length")
        _require_sequence_index(sequence_index=sequence_index, sequence_length=sequence_length)
        if (
            type(component_index) is not int
            or component_index < 0
            or component_index >= len(self.formation.components)
        ):
            raise TensorRuntimeError("component_index is outside component vocabulary")
        key = (width, height, sequence_length, sequence_index, component_index)
        cached = self._component_tensors.get(key)
        if cached is not None:
            return cached
        tensor = self._build_component_tensor(
            width=width,
            height=height,
            sequence_length=sequence_length,
            sequence_index=sequence_index,
            component_index=component_index,
        )
        self._component_tensors[key] = tensor
        return tensor

    def _build_component_tensor(
        self,
        *,
        width: int,
        height: int,
        sequence_length: int,
        sequence_index: int,
        component_index: int,
    ) -> Any:
        field = self.formation.component_field(
            width=width,
            height=height,
            sequence_length=sequence_length,
            sequence_index=sequence_index,
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


def apply_softmax_predictions(
    runtime: TensorRuntime, module: Any, fields: Any
) -> list[list[float]]:
    _ = runtime
    return _torch().softmax(module(fields), dim=1).tolist()


def build_optimizer(
    runtime: TensorRuntime,
    *,
    name: str,
    parameters: Any,
    learning_rate: float,
) -> Any:
    """Build a named optimizer for module parameters."""

    _ = runtime
    torch = _torch()
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate)
    raise TensorRuntimeError(f"unsupported optimizer: {name}")


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


def _torch() -> Any:
    try:
        return cast(Any, importlib.import_module("torch"))
    except ImportError as error:
        raise TensorRuntimeError("PyTorch is required to run benchmark training") from error


def _require_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise TensorRuntimeError(f"{name} must be a positive integer")


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


def _require_sequence_index(*, sequence_index: int, sequence_length: int) -> None:
    if type(sequence_index) is not int or sequence_index < 0 or sequence_index >= sequence_length:
        raise TensorRuntimeError("sequence_index must be within sequence_length")


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


def _variation_coordinate(
    record: Mapping[str, object],
    *,
    expected_sequence_index: int,
) -> VariationCoordinate:
    if str(record.get("kind")) != "field-variation-transform-coordinate":
        raise TensorRuntimeError(
            "variation coordinate kind must be field-variation-transform-coordinate"
        )
    sequence_index = _integer(record.get("sequence_index"), "sequence_index")
    if sequence_index != expected_sequence_index:
        raise TensorRuntimeError(
            "variation coordinate sequence_index must match coordinate position"
        )
    spatial = _mapping(record.get("spatial_affine"), "spatial_affine")
    if str(spatial.get("kind")) != "spatial-affine-coordinate":
        raise TensorRuntimeError("spatial_affine kind must be spatial-affine-coordinate")
    if str(spatial.get("coordinate_system")) != "normalized-sequence-element":
        raise TensorRuntimeError(
            "spatial_affine coordinate_system must be normalized-sequence-element"
        )
    matrix = _matrix(spatial.get("matrix"), "spatial_affine.matrix")
    return VariationCoordinate(
        sequence_index=sequence_index,
        matrix=matrix,
    )


def _generated_affine_grid_row(
    record: Mapping[str, object],
    *,
    width: int,
    height: int,
    sequence_length: int,
    sequence_index: int,
    placement_axis: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    spatial = cast(Mapping[str, object], record["spatial_affine"])
    return _affine_grid_row_from_values(
        matrix=_trusted_matrix(spatial["matrix"]),
        width=width,
        height=height,
        sequence_length=sequence_length,
        sequence_index=sequence_index,
        placement_axis=placement_axis,
    )


def _affine_grid_row(
    *,
    coordinate: VariationCoordinate,
    width: int,
    height: int,
    sequence_length: int,
    sequence_index: int,
    placement_axis: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return _affine_grid_row_from_values(
        matrix=coordinate.matrix,
        width=width,
        height=height,
        sequence_length=sequence_length,
        sequence_index=sequence_index,
        placement_axis=placement_axis,
    )


def _affine_grid_row_from_values(
    *,
    matrix: AffineMatrix2D,
    width: int,
    height: int,
    sequence_length: int,
    sequence_index: int,
    placement_axis: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    linear_matrix = linear_affine_matrix(matrix)
    inverse = _inverse_affine_matrix_from_values(matrix=linear_matrix)
    center = sequence_center(
        width=width,
        height=height,
        sequence_length=sequence_length,
        sequence_index=sequence_index,
        placement_axis=placement_axis,
    )
    center_x = 2.0 * center[0] - 1.0
    center_y = 2.0 * center[1] - 1.0
    field_translation = sequence_relative_translation(
        affine_translation(matrix),
        sequence_length=sequence_length,
        placement_axis=placement_axis,
    )
    translation_x = 2.0 * field_translation[0]
    translation_y = 2.0 * field_translation[1]
    return (
        (
            inverse[0][0],
            inverse[0][1],
            center_x
            - inverse[0][0] * (center_x + translation_x)
            - inverse[0][1] * (center_y + translation_y),
        ),
        (
            inverse[1][0],
            inverse[1][1],
            center_y
            - inverse[1][0] * (center_x + translation_x)
            - inverse[1][1] * (center_y + translation_y),
        ),
    )


def _inverse_affine_matrix_from_values(
    *,
    matrix: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    (a, b), (c, d) = matrix
    determinant = a * d - b * c
    if not math.isfinite(determinant) or determinant == 0.0:
        raise TensorRuntimeError("variation affine transform is singular")
    return ((d / determinant, -b / determinant), (-c / determinant, a / determinant))


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TensorRuntimeError(f"{name} must be a record")
    return cast(Mapping[str, object], value)


def _matrix(
    value: object,
    name: str,
) -> AffineMatrix2D:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TensorRuntimeError(f"{name} must contain three rows")
    sequence = cast(Sequence[object], value)
    if len(sequence) != 3:
        raise TensorRuntimeError(f"{name} must contain three rows")
    matrix = (
        _triple(sequence[0], f"{name}.0"),
        _triple(sequence[1], f"{name}.1"),
        _triple(sequence[2], f"{name}.2"),
    )
    if matrix[2] != (0.0, 0.0, 1.0):
        raise TensorRuntimeError(f"{name} final row must be fixed affine coordinates")
    return matrix


def _trusted_matrix(value: object) -> AffineMatrix2D:
    sequence = cast(Sequence[object], value)
    return (
        _trusted_triple(sequence[0]),
        _trusted_triple(sequence[1]),
        _trusted_triple(sequence[2]),
    )


def _triple(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TensorRuntimeError(f"{name} must contain three values")
    sequence = cast(Sequence[object], value)
    if len(sequence) != 3:
        raise TensorRuntimeError(f"{name} must contain three values")
    return (
        _number(sequence[0], f"{name}.0"),
        _number(sequence[1], f"{name}.1"),
        _number(sequence[2], f"{name}.2"),
    )


def _trusted_triple(value: object) -> tuple[float, float, float]:
    sequence = cast(Sequence[object], value)
    return (
        _trusted_float(sequence[0]),
        _trusted_float(sequence[1]),
        _trusted_float(sequence[2]),
    )


def _trusted_float(value: object) -> float:
    return float(cast(int | float, value))


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TensorRuntimeError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise TensorRuntimeError(f"{name} must be finite")
    return number


def _integer(value: object, name: str) -> int:
    if type(value) is not int:
        raise TensorRuntimeError(f"{name} must be an integer")
    return value
