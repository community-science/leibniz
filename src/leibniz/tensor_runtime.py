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
from leibniz.model_operators import summarize_architecture_operators
from leibniz.observation_formation import ObservationFormationDeclaration
from leibniz.observation_generation import GeneratedFormationBatch

__all__ = [
    "architecture_tensor_runtime_issue",
    "architecture_supported_by_tensor_runtime",
    "FormationTensorCache",
    "preferred_tensor_runtime_device_kind",
    "TensorRuntime",
    "TensorRuntimeError",
    "TensorRuntimeDevice",
    "TensorRuntimeDeviceKind",
    "tensor_runtime_available_memory_bytes",
    "tensor_runtime_device_kinds",
    "resolve_tensor_runtime",
    "runtime_roofline_record",
    "validate_tensor_runtime_device",
]

TensorRuntimeDevice = Literal["auto", "cpu", "cuda", "mps"]
TensorRuntimeDeviceKind = Literal["cpu", "cuda", "mps"]
_available_devices = frozenset({"auto", "cpu", "cuda", "mps"})
_roofline_cache: dict[str, dict[str, object]] = {}
_AffineMatrix2D = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


class TensorRuntimeError(ValueError):
    """Raised when a tensor runtime cannot be resolved."""


@dataclass(frozen=True, slots=True)
class TensorRuntime:
    """Resolved local tensor runtime for benchmark execution."""

    torch: Any
    device: Any
    device_kind: Literal["cpu", "cuda", "mps"]


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

    if device_kind != "mps":
        return None
    for operator in summarize_architecture_operators(architecture).operators:
        if operator.descriptor.kind != "local-aggregation":
            continue
        input_shape = operator.input_shape
        output_shape = operator.output_shape
        if input_shape is None or output_shape is None:
            continue
        if len(input_shape) < 2 or len(output_shape) < 2:
            continue
        input_height, input_width = input_shape[-2:]
        output_height, output_width = output_shape[-2:]
        if input_height % output_height != 0 or input_width % output_width != 0:
            return (
                "mps adaptive pooling requires trailing input axes to be divisible "
                "by the requested output axes"
            )
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


@dataclass(frozen=True, slots=True)
class _VariationCoordinate:
    sequence_index: int
    matrix: _AffineMatrix2D


def _variation_coordinate(
    record: Mapping[str, object],
    *,
    expected_sequence_index: int,
) -> _VariationCoordinate:
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
    return _VariationCoordinate(
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
    coordinate: _VariationCoordinate,
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
    matrix: _AffineMatrix2D,
    width: int,
    height: int,
    sequence_length: int,
    sequence_index: int,
    placement_axis: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    linear_matrix = _linear_affine_matrix(matrix)
    inverse = _inverse_affine_matrix_from_values(matrix=linear_matrix)
    center = _sequence_center(
        width=width,
        height=height,
        sequence_length=sequence_length,
        sequence_index=sequence_index,
        placement_axis=placement_axis,
    )
    center_x = 2.0 * center[0] - 1.0
    center_y = 2.0 * center[1] - 1.0
    field_translation = _sequence_relative_translation(
        _affine_translation(matrix),
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


def _linear_affine_matrix(
    matrix: _AffineMatrix2D,
) -> tuple[tuple[float, float], tuple[float, float]]:
    return ((matrix[0][0], matrix[0][1]), (matrix[1][0], matrix[1][1]))


def _affine_translation(
    matrix: _AffineMatrix2D,
) -> tuple[float, float]:
    return (matrix[0][2], matrix[1][2])


def _sequence_center(
    *,
    width: int,
    height: int,
    sequence_length: int,
    sequence_index: int,
    placement_axis: str,
) -> tuple[float, float]:
    if placement_axis == "x":
        return ((sequence_index + 0.5) / sequence_length, 0.5)
    return (0.5, (sequence_index + 0.5) / sequence_length)


def _sequence_relative_translation(
    translation: tuple[float, float],
    *,
    sequence_length: int,
    placement_axis: str,
) -> tuple[float, float]:
    if placement_axis == "x":
        return (translation[0] / sequence_length, translation[1])
    return (translation[0], translation[1] / sequence_length)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TensorRuntimeError(f"{name} must be a record")
    return cast(Mapping[str, object], value)


def _matrix(
    value: object,
    name: str,
) -> _AffineMatrix2D:
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


def _trusted_matrix(value: object) -> _AffineMatrix2D:
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
