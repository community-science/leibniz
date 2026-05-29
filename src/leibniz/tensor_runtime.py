"""Narrow tensor runtime adapter for local benchmark execution."""

from __future__ import annotations

import importlib
import math
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
    "tensor_runtime_device_kinds",
    "resolve_tensor_runtime",
    "runtime_roofline_record",
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

    def varied_component_sequence_tensor(
        self,
        *,
        resolution: int,
        component_sequence: Sequence[int],
        variation_coordinates: Sequence[Mapping[str, object]],
    ) -> Any:
        """Return a formed-field tensor with recorded per-slot variation applied."""

        _require_positive_integer(resolution, "resolution")
        sequence = tuple(component_sequence)
        if not sequence:
            raise TensorRuntimeError("component_sequence must not be empty")
        coordinates = tuple(variation_coordinates)
        if len(coordinates) != len(sequence):
            raise TensorRuntimeError("variation_coordinates length must match slot count")
        source_tensors: list[Any] = []
        affine_rows: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        value_scales: list[float] = []
        for slot_index, component_index in enumerate(sequence):
            coordinate = _variation_coordinate(
                coordinates[slot_index],
                expected_slot_index=slot_index,
            )
            source_tensors.append(
                self.component_tensor(
                    resolution=resolution,
                    slot_count=len(sequence),
                    slot_index=slot_index,
                    component_index=component_index,
                )
            )
            affine_rows.append(
                _affine_grid_row(
                    coordinate=coordinate,
                    slot_count=len(sequence),
                    slot_index=slot_index,
                    slot_axis=self.formation.slot_composition.slot_axis,
                )
            )
            value_scales.append(coordinate.value_scale)
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
        scales = torch.tensor(
            value_scales,
            dtype=torch.float32,
            device=self.runtime.device,
        ).reshape((len(value_scales), 1, 1, 1))
        return torch.clamp(transformed * scales, min=0.0, max=1.0).amax(dim=0)

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
        resolution = batch.samples[0].resolution
        slot_count = len(batch.samples[0].component_sequence)
        if slot_count < 1:
            raise TensorRuntimeError("component_sequence must not be empty")
        source_tensors: list[Any] = []
        affine_rows: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
        value_scales: list[float] = []
        for sample in batch.samples:
            if sample.resolution != resolution:
                raise TensorRuntimeError("batch sample resolutions must match")
            if len(sample.component_sequence) != slot_count:
                raise TensorRuntimeError("batch sample slot counts must match")
            if len(sample.variation_coordinates) != slot_count:
                raise TensorRuntimeError("variation_coordinates length must match slot count")
            for slot_index, component_index in enumerate(sample.component_sequence):
                source_tensors.append(
                    self.component_tensor(
                        resolution=resolution,
                        slot_count=slot_count,
                        slot_index=slot_index,
                        component_index=component_index,
                    )
                )
                row, value_scale = _generated_affine_grid_row_and_value_scale(
                    sample.variation_coordinates[slot_index],
                    slot_count=slot_count,
                    slot_index=slot_index,
                    slot_axis=self.formation.slot_composition.slot_axis,
                )
                affine_rows.append(row)
                value_scales.append(value_scale)
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
        scales = torch.tensor(
            value_scales,
            dtype=torch.float32,
            device=self.runtime.device,
        ).reshape((len(value_scales), 1, 1, 1))
        transformed = torch.clamp(transformed * scales, min=0.0, max=1.0)
        channels, height, width = transformed.shape[1:]
        return transformed.reshape(
            (sample_count, slot_count, channels, height, width)
        ).amax(dim=1)

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

    return architecture_tensor_runtime_issue(
        architecture,
        device_kind=device_kind,
    ) is None


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
            "flop_calibration_seconds": matmul_seconds,
            "flop_calibration_matrix_size": matrix_size,
            "flop_calibration_repeats": matmul_repeats,
            "peak_flops_per_second": (
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
        raise TensorRuntimeError(
            "PyTorch is required to run benchmark training"
        ) from error


def _require_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise TensorRuntimeError(f"{name} must be a positive integer")


def _require_slot_index(*, slot_index: int, slot_count: int) -> None:
    if type(slot_index) is not int or slot_index < 0 or slot_index >= slot_count:
        raise TensorRuntimeError("slot_index must be within slot_count")


@dataclass(frozen=True, slots=True)
class _VariationCoordinate:
    slot_index: int
    translation: tuple[float, float]
    scale: tuple[float, float]
    rotation_degrees: float
    shear_degrees: float
    value_scale: float


def _variation_coordinate(
    record: Mapping[str, object],
    *,
    expected_slot_index: int,
) -> _VariationCoordinate:
    if str(record.get("kind")) != "field-variation-transform-coordinate":
        raise TensorRuntimeError(
            "variation coordinate kind must be field-variation-transform-coordinate"
        )
    slot_index = _integer(record.get("slot_index"), "slot_index")
    if slot_index != expected_slot_index:
        raise TensorRuntimeError("variation coordinate slot_index must match coordinate position")
    spatial = _mapping(record.get("spatial_affine"), "spatial_affine")
    if str(spatial.get("kind")) != "spatial-affine-coordinate":
        raise TensorRuntimeError("spatial_affine kind must be spatial-affine-coordinate")
    if str(spatial.get("coordinate_system")) != "normalized-field":
        raise TensorRuntimeError("spatial_affine coordinate_system must be normalized-field")
    value_scale = _mapping(record.get("value_scale"), "value_scale")
    if str(value_scale.get("kind")) != "value-scale-coordinate":
        raise TensorRuntimeError("value_scale kind must be value-scale-coordinate")
    scale = _pair(spatial.get("scale"), "spatial_affine.scale")
    if scale[0] <= 0.0 or scale[1] <= 0.0:
        raise TensorRuntimeError("spatial_affine.scale values must be positive")
    scale_value = _number(value_scale.get("scale"), "value_scale.scale")
    if scale_value <= 0.0:
        raise TensorRuntimeError("value_scale.scale must be positive")
    return _VariationCoordinate(
        slot_index=slot_index,
        translation=_pair(spatial.get("translation"), "spatial_affine.translation"),
        scale=scale,
        rotation_degrees=_single_number(
            spatial.get("rotation_degrees"),
            "spatial_affine.rotation_degrees",
        ),
        shear_degrees=_single_number(
            spatial.get("shear_degrees"),
            "spatial_affine.shear_degrees",
        ),
        value_scale=scale_value,
    )


def _generated_affine_grid_row_and_value_scale(
    record: Mapping[str, object],
    *,
    slot_count: int,
    slot_index: int,
    slot_axis: str,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], float]:
    spatial = cast(Mapping[str, object], record["spatial_affine"])
    value_scale = cast(Mapping[str, object], record["value_scale"])
    return (
        _affine_grid_row_from_values(
            translation=_trusted_pair(spatial["translation"]),
            scale=_trusted_pair(spatial["scale"]),
            rotation_degrees=_trusted_single_number(spatial["rotation_degrees"]),
            shear_degrees=_trusted_single_number(spatial["shear_degrees"]),
            slot_count=slot_count,
            slot_index=slot_index,
            slot_axis=slot_axis,
        ),
        _trusted_float(value_scale["scale"]),
    )


def _affine_grid_row(
    *,
    coordinate: _VariationCoordinate,
    slot_count: int,
    slot_index: int,
    slot_axis: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return _affine_grid_row_from_values(
        translation=coordinate.translation,
        scale=coordinate.scale,
        rotation_degrees=coordinate.rotation_degrees,
        shear_degrees=coordinate.shear_degrees,
        slot_count=slot_count,
        slot_index=slot_index,
        slot_axis=slot_axis,
    )


def _affine_grid_row_from_values(
    *,
    translation: tuple[float, float],
    scale: tuple[float, float],
    rotation_degrees: float,
    shear_degrees: float,
    slot_count: int,
    slot_index: int,
    slot_axis: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    inverse = _inverse_affine_matrix_from_values(
        scale=scale,
        rotation_degrees=rotation_degrees,
        shear_degrees=shear_degrees,
    )
    center = _slot_center(slot_count=slot_count, slot_index=slot_index, axis=slot_axis)
    center_x = 2.0 * center[0] - 1.0
    center_y = 2.0 * center[1] - 1.0
    translation_x = 2.0 * translation[0]
    translation_y = 2.0 * translation[1]
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
    scale: tuple[float, float],
    rotation_degrees: float,
    shear_degrees: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    angle = math.radians(rotation_degrees)
    shear = math.tan(math.radians(shear_degrees))
    scale_x, scale_y = scale
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    a = cos_angle * scale_x
    b = (cos_angle * shear - sin_angle) * scale_y
    c = sin_angle * scale_x
    d = (sin_angle * shear + cos_angle) * scale_y
    determinant = a * d - b * c
    if not math.isfinite(determinant) or determinant == 0.0:
        raise TensorRuntimeError("variation affine transform is singular")
    return ((d / determinant, -b / determinant), (-c / determinant, a / determinant))


def _slot_center(
    *,
    slot_count: int,
    slot_index: int,
    axis: str,
) -> tuple[float, float]:
    if axis == "x":
        return ((slot_index + 0.5) / slot_count, 0.5)
    return (0.5, (slot_index + 0.5) / slot_count)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TensorRuntimeError(f"{name} must be a record")
    return cast(Mapping[str, object], value)


def _pair(value: object, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TensorRuntimeError(f"{name} must contain two values")
    sequence = cast(Sequence[object], value)
    if len(sequence) != 2:
        raise TensorRuntimeError(f"{name} must contain two values")
    return (_number(sequence[0], f"{name}.0"), _number(sequence[1], f"{name}.1"))


def _trusted_pair(value: object) -> tuple[float, float]:
    sequence = cast(Sequence[object], value)
    return (_trusted_float(sequence[0]), _trusted_float(sequence[1]))


def _single_number(value: object, name: str) -> float:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise TensorRuntimeError(f"{name} must contain one value")
    sequence = cast(Sequence[object], value)
    if len(sequence) != 1:
        raise TensorRuntimeError(f"{name} must contain one value")
    return _number(sequence[0], f"{name}.0")


def _trusted_single_number(value: object) -> float:
    sequence = cast(Sequence[object], value)
    return _trusted_float(sequence[0])


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
