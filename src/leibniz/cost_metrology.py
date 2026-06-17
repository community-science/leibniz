"""Measured tensor program cost with a declared abstract-FLOP model."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, Self, cast

from leibniz.tensor_runtime import (
    TensorRuntime,
    TensorRuntimeOperationRecord,
    TensorRuntimeTensorSpec,
    tensor_runtime_capture_operations,
    tensor_runtime_operation_capture,
    tensor_runtime_project_operations,
    tensor_runtime_shape_element_count,
)

__all__ = [
    "CostMeasurement",
    "CostMetrologyError",
    "CostMeter",
    "CostOperationTraceRecord",
    "DeviceCostProfile",
    "DeviceDType",
    "EnergyCostBreakdown",
    "MovementCostRecord",
    "OperationClass",
    "OperationCostRecord",
    "TensorValueSpec",
    "UnmodeledOperationRecord",
    "estimate_program_cost",
    "estimate_operation_stream_cost",
    "device_cost_profile",
    "device_cost_profiles",
    "operation_class_for_name",
    "measure_program_cost",
    "normalize_tensor_dtype",
    "price_cost_measurement_energy",
]

_tensor_runtime_cost_model_id = "leibniz.cost-model.tensor-runtime@0.1.0"
_CostExecutionMode = Literal["measured", "dry-run"]
OperationClass = Literal[
    "dense-matmul",
    "convolution",
    "elementwise",
    "transcendental",
    "reduction",
    "fft",
    "data-movement",
]
DeviceDType = Literal["fp64", "fp32", "tf32", "fp16", "bf16", "int8"]

_operation_classes: tuple[OperationClass, ...] = (
    "dense-matmul",
    "convolution",
    "elementwise",
    "transcendental",
    "reduction",
    "fft",
    "data-movement",
)
_device_dtypes: tuple[DeviceDType, ...] = (
    "fp64",
    "fp32",
    "tf32",
    "fp16",
    "bf16",
    "int8",
)


class CostMetrologyError(ValueError):
    """Raised when a cost measurement or record is invalid."""


@dataclass(frozen=True, slots=True)
class DeviceCostProfile:
    """Declared device energy coefficients for machine-independent counts."""

    profile_id: str
    label: str
    version: str
    provenance: tuple[str, ...]
    compute_energy_joules: Mapping[tuple[OperationClass, DeviceDType], float]
    bytes_moved_energy_joules: float
    bytes_resident_energy_joules: float
    unified_memory: bool = False
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty_string(self.profile_id, "profile_id")
        _require_nonempty_string(self.label, "profile label")
        _require_nonempty_string(self.version, "profile version")
        for index, item in enumerate(self.provenance):
            _require_nonempty_string(item, f"profile provenance.{index}")
        if type(self.unified_memory) is not bool:
            raise CostMetrologyError("profile unified_memory must be a boolean")
        for index, item in enumerate(self.notes):
            _require_nonempty_string(item, f"profile notes.{index}")
        for (operation_class, dtype), value in self.compute_energy_joules.items():
            if operation_class not in _operation_classes or dtype not in _device_dtypes:
                raise CostMetrologyError(
                    "compute_energy_joules keys must be (operation_class, dtype)"
                )
            _require_finite_nonnegative_float(
                value,
                f"compute_energy_joules.{operation_class}.{dtype}",
            )
        _require_finite_nonnegative_float(
            self.bytes_moved_energy_joules,
            "bytes_moved_energy_joules",
        )
        _require_finite_nonnegative_float(
            self.bytes_resident_energy_joules,
            "bytes_resident_energy_joules",
        )

    @classmethod
    def from_record(cls, record: object) -> Self:
        mapping = _record_mapping(record, "device cost profile record")
        compute_entries: dict[tuple[OperationClass, DeviceDType], float] = {}
        for index, item in enumerate(
            _record_sequence(mapping.get("compute_energy_joules"), "compute_energy_joules")
        ):
            entry = _record_mapping(item, f"compute_energy_joules.{index}")
            operation_class = _record_operation_class(
                entry.get("operation_class"),
                f"compute_energy_joules.{index}.operation_class",
            )
            dtype = _record_device_dtype(
                entry.get("dtype"),
                f"compute_energy_joules.{index}.dtype",
            )
            key = (operation_class, dtype)
            if key in compute_entries:
                raise CostMetrologyError(
                    f"duplicate compute energy entry for {operation_class}/{dtype}"
                )
            compute_entries[key] = _record_float(
                entry.get("joules"),
                f"compute_energy_joules.{index}.joules",
            )
        return cls(
            profile_id=_record_string(mapping.get("profile_id"), "profile_id"),
            label=_record_string(mapping.get("label"), "profile label"),
            version=_record_string(mapping.get("version"), "profile version"),
            provenance=tuple(
                _record_string(item, "profile provenance")
                for item in _record_sequence(mapping.get("provenance"), "provenance")
            ),
            compute_energy_joules=compute_entries,
            bytes_moved_energy_joules=_record_float(
                mapping.get("bytes_moved_energy_joules"),
                "bytes_moved_energy_joules",
            ),
            bytes_resident_energy_joules=_record_float(
                mapping.get("bytes_resident_energy_joules"),
                "bytes_resident_energy_joules",
            ),
            unified_memory=_record_bool(mapping.get("unified_memory", False), "unified_memory"),
            notes=tuple(
                _record_string(item, "profile notes")
                for item in _record_sequence(mapping.get("notes", ()), "notes")
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "label": self.label,
            "version": self.version,
            "provenance": list(self.provenance),
            "compute_energy_joules": [
                {
                    "operation_class": operation_class,
                    "dtype": dtype,
                    "joules": joules,
                }
                for (operation_class, dtype), joules in sorted(
                    self.compute_energy_joules.items()
                )
            ],
            "bytes_moved_energy_joules": self.bytes_moved_energy_joules,
            "bytes_resident_energy_joules": self.bytes_resident_energy_joules,
            "unified_memory": self.unified_memory,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class EnergyCostBreakdown:
    """Energy price of a machine-independent cost measurement under one profile."""

    profile_id: str
    compute_joules: float
    bytes_moved_joules: float
    bytes_resident_joules: float
    total_joules: float
    coefficient_overrides: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.profile_id, "energy profile_id")
        _require_finite_nonnegative_float(self.compute_joules, "compute_joules")
        _require_finite_nonnegative_float(self.bytes_moved_joules, "bytes_moved_joules")
        _require_finite_nonnegative_float(
            self.bytes_resident_joules,
            "bytes_resident_joules",
        )
        _require_finite_nonnegative_float(self.total_joules, "total_joules")

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "profile_id": self.profile_id,
            "compute_joules": self.compute_joules,
            "bytes_moved_joules": self.bytes_moved_joules,
            "bytes_resident_joules": self.bytes_resident_joules,
            "total_joules": self.total_joules,
            "residency_model": "per-evaluation-footprint",
            "residency_duration_deferred": True,
        }
        if self.coefficient_overrides:
            record["coefficient_overrides"] = dict(self.coefficient_overrides)
        return record


_default_device_cost_profile_id = "cost-model.device.nvidia-a100@0.1.0"


def device_cost_profiles() -> Mapping[str, DeviceCostProfile]:
    """Return the bundled declared device cost profiles."""

    return _device_cost_profiles


def device_cost_profile(profile_id: str = _default_device_cost_profile_id) -> DeviceCostProfile:
    """Return a bundled declared device cost profile by id."""

    try:
        return _device_cost_profiles[profile_id]
    except KeyError as error:
        raise CostMetrologyError(f"unknown device cost profile: {profile_id}") from error


def price_cost_measurement_energy(
    measurement: CostMeasurement,
    *,
    profile: DeviceCostProfile | str = _default_device_cost_profile_id,
    compute_energy_overrides: Mapping[tuple[OperationClass, DeviceDType], float] | None = None,
    bytes_moved_energy_joules: float | None = None,
    bytes_resident_energy_joules: float | None = None,
) -> EnergyCostBreakdown:
    """Price invariant cost counts under a declared device profile."""

    resolved_profile = device_cost_profile(profile) if isinstance(profile, str) else profile
    compute_table = dict(resolved_profile.compute_energy_joules)
    override_record: dict[str, float] = {}
    if compute_energy_overrides:
        for key, value in compute_energy_overrides.items():
            operation_class, dtype = key
            if operation_class not in _operation_classes or dtype not in _device_dtypes:
                raise CostMetrologyError("compute energy override key is not in the taxonomy")
            _require_finite_nonnegative_float(value, f"override.{operation_class}.{dtype}")
            compute_table[key] = value
            override_record[f"compute.{operation_class}.{dtype}"] = float(value)
    moved_coefficient = (
        resolved_profile.bytes_moved_energy_joules
        if bytes_moved_energy_joules is None
        else bytes_moved_energy_joules
    )
    resident_coefficient = (
        resolved_profile.bytes_resident_energy_joules
        if bytes_resident_energy_joules is None
        else bytes_resident_energy_joules
    )
    _require_finite_nonnegative_float(moved_coefficient, "bytes_moved_energy_joules")
    _require_finite_nonnegative_float(resident_coefficient, "bytes_resident_energy_joules")
    if bytes_moved_energy_joules is not None:
        override_record["bytes_moved"] = float(bytes_moved_energy_joules)
    if bytes_resident_energy_joules is not None:
        override_record["bytes_resident"] = float(bytes_resident_energy_joules)

    compute_joules = 0.0
    for record in measurement.per_op:
        if record.abstract_flops == 0:
            continue
        if record.operation_class is None or record.dtype is None:
            raise CostMetrologyError(
                f"operation {record.name} is missing operation_class or dtype"
            )
        try:
            coefficient = compute_table[(record.operation_class, record.dtype)]
        except KeyError as error:
            raise CostMetrologyError(
                "profile is missing compute coefficient "
                f"for {record.operation_class}/{record.dtype}"
            ) from error
        compute_joules += float(record.abstract_flops) * coefficient
    moved_joules = float(measurement.bytes_moved) * moved_coefficient
    resident_joules = float(measurement.bytes_resident) * resident_coefficient
    return EnergyCostBreakdown(
        profile_id=resolved_profile.profile_id,
        compute_joules=compute_joules,
        bytes_moved_joules=moved_joules,
        bytes_resident_joules=resident_joules,
        total_joules=compute_joules + moved_joules + resident_joules,
        coefficient_overrides=override_record or None,
    )


@dataclass(frozen=True, slots=True)
class TensorValueSpec:
    """Shape and dtype for a tensor value observed in an op stream."""

    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if type(self.dtype) is not str or not self.dtype:
            raise CostMetrologyError("tensor dtype must be a nonempty string")
        for extent in self.shape:
            if type(extent) is not int:
                raise CostMetrologyError("tensor shape extents must be integers")
            if extent < 0:
                raise CostMetrologyError("tensor shape extents must be nonnegative")

    @classmethod
    def from_record(cls, record: object) -> Self:
        mapping = _record_mapping(record, "tensor spec record")
        raw_shape = mapping.get("shape")
        raw_dtype = mapping.get("dtype")
        if not isinstance(raw_shape, Sequence) or isinstance(raw_shape, (str, bytes)):
            raise CostMetrologyError("tensor spec shape must be a sequence")
        if not isinstance(raw_dtype, str):
            raise CostMetrologyError("tensor spec dtype must be a string")
        return cls(
            shape=tuple(
                _record_int(value, "tensor shape extent")
                for value in cast(Sequence[object], raw_shape)
            ),
            dtype=raw_dtype,
        )

    def to_record(self) -> dict[str, object]:
        return {"shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True, slots=True)
class OperationCostRecord:
    """Aggregated abstract-FLOP cost for one modeled op name."""

    name: str
    calls: int
    abstract_flops: int
    output_elements: int
    operation_class: OperationClass | None = None
    dtype: DeviceDType | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, "operation name")
        _require_nonnegative_int(self.calls, "operation calls")
        _require_nonnegative_int(self.abstract_flops, "operation abstract_flops")
        _require_nonnegative_int(self.output_elements, "operation output_elements")
        if self.operation_class is not None and self.operation_class not in _operation_classes:
            raise CostMetrologyError("operation_class must be a known operation class")
        if self.dtype is not None and self.dtype not in _device_dtypes:
            raise CostMetrologyError("operation dtype must be a known device dtype")

    @classmethod
    def from_record(cls, record: object) -> Self:
        mapping = _record_mapping(record, "operation cost record")
        return cls(
            name=_record_string(mapping.get("name"), "operation name"),
            calls=_record_int(mapping.get("calls"), "operation calls"),
            abstract_flops=_record_int(mapping.get("abstract_flops"), "operation abstract_flops"),
            output_elements=_record_int(
                mapping.get("output_elements"),
                "operation output_elements",
            ),
            operation_class=(
                None
                if mapping.get("operation_class") is None
                else _record_operation_class(
                    mapping.get("operation_class"),
                    "operation operation_class",
                )
            ),
            dtype=(
                None
                if mapping.get("dtype") is None
                else _record_device_dtype(mapping.get("dtype"), "operation dtype")
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "name": self.name,
            "calls": self.calls,
            "abstract_flops": self.abstract_flops,
            "output_elements": self.output_elements,
        }
        if self.operation_class is not None:
            record["operation_class"] = self.operation_class
        if self.dtype is not None:
            record["dtype"] = self.dtype
        return record


@dataclass(frozen=True, slots=True)
class MovementCostRecord:
    """Aggregated element movement for one movement-class op name."""

    name: str
    calls: int
    moved_elements: int
    bytes_moved: int = 0

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, "movement operation name")
        _require_nonnegative_int(self.calls, "movement operation calls")
        _require_nonnegative_int(self.moved_elements, "movement operation moved_elements")
        _require_nonnegative_int(self.bytes_moved, "movement operation bytes_moved")

    @classmethod
    def from_record(cls, record: object) -> Self:
        mapping = _record_mapping(record, "movement cost record")
        return cls(
            name=_record_string(mapping.get("name"), "movement operation name"),
            calls=_record_int(mapping.get("calls"), "movement operation calls"),
            moved_elements=_record_int(
                mapping.get("moved_elements"),
                "movement operation moved_elements",
            ),
            bytes_moved=_record_int(mapping.get("bytes_moved", 0), "movement bytes_moved"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "calls": self.calls,
            "moved_elements": self.moved_elements,
            "bytes_moved": self.bytes_moved,
        }


@dataclass(frozen=True, slots=True)
class UnmodeledOperationRecord:
    """Aggregated record for dispatched ops that the declared model does not price."""

    name: str
    calls: int
    output_elements: int

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, "unmodeled operation name")
        _require_nonnegative_int(self.calls, "unmodeled operation calls")
        _require_nonnegative_int(self.output_elements, "unmodeled operation output_elements")

    @classmethod
    def from_record(cls, record: object) -> Self:
        mapping = _record_mapping(record, "unmodeled operation record")
        return cls(
            name=_record_string(mapping.get("name"), "unmodeled operation name"),
            calls=_record_int(mapping.get("calls"), "unmodeled operation calls"),
            output_elements=_record_int(
                mapping.get("output_elements"),
                "unmodeled operation output_elements",
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "calls": self.calls,
            "output_elements": self.output_elements,
        }


@dataclass(frozen=True, slots=True)
class CostOperationTraceRecord:
    """One dispatched operation with tensor shapes and dtypes for recounting."""

    name: str
    input_tensors: tuple[TensorValueSpec, ...]
    output_tensors: tuple[TensorValueSpec, ...]

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, "operation trace name")

    @classmethod
    def from_record(cls, record: object) -> Self:
        mapping = _record_mapping(record, "operation trace record")
        return cls(
            name=_record_string(mapping.get("name"), "operation trace name"),
            input_tensors=tuple(
                TensorValueSpec.from_record(item)
                for item in _record_sequence(mapping.get("input_tensors"), "input_tensors")
            ),
            output_tensors=tuple(
                TensorValueSpec.from_record(item)
                for item in _record_sequence(mapping.get("output_tensors"), "output_tensors")
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "input_tensors": [spec.to_record() for spec in self.input_tensors],
            "output_tensors": [spec.to_record() for spec in self.output_tensors],
        }


@dataclass(frozen=True, slots=True)
class CostMeasurement:
    """Abstract cost for one tensor-runtime operation stream."""

    cost_model_id: str
    abstract_flops: int
    per_op: tuple[OperationCostRecord, ...]
    moved_elements: int
    movement: tuple[MovementCostRecord, ...]
    unmodeled_operations: tuple[UnmodeledOperationRecord, ...]
    operation_count: int
    operation_trace: tuple[CostOperationTraceRecord, ...]
    wall_seconds: float
    tensor_device: str
    execution_mode: _CostExecutionMode = "measured"
    operation_stream_source: str = "runtime-executed"
    operations_executed: bool = True
    bytes_moved: int = 0
    bytes_resident: int = 0
    roofline: Mapping[str, object] | None = None

    @classmethod
    def tensor_runtime_cost_model_id(cls) -> str:
        """Return the cost model id used for tensor runtime operation streams."""

        return _tensor_runtime_cost_model_id

    def __post_init__(self) -> None:
        _require_nonempty_string(self.cost_model_id, "cost_model_id")
        _require_nonnegative_int(self.abstract_flops, "abstract_flops")
        _require_nonnegative_int(self.moved_elements, "moved_elements")
        _require_nonnegative_int(self.bytes_moved, "bytes_moved")
        _require_nonnegative_int(self.operation_count, "operation_count")
        _require_nonnegative_int(self.bytes_resident, "bytes_resident")
        _require_finite_nonnegative_float(self.wall_seconds, "wall_seconds")
        _require_nonempty_string(self.tensor_device, "tensor_device")
        if self.execution_mode not in {"measured", "dry-run"}:
            raise CostMetrologyError("execution_mode must be measured or dry-run")
        _require_nonempty_string(
            self.operation_stream_source,
            "operation_stream_source",
        )
        if type(self.operations_executed) is not bool:
            raise CostMetrologyError("operations_executed must be a boolean")
        if self.execution_mode == "measured" and not self.operations_executed:
            raise CostMetrologyError("measured cost must execute operations")
        if self.execution_mode == "dry-run" and self.operations_executed:
            raise CostMetrologyError("dry-run cost must not execute operations")
        if sum(record.abstract_flops for record in self.per_op) != self.abstract_flops:
            raise CostMetrologyError("abstract_flops must equal summed per_op abstract_flops")
        if sum(record.moved_elements for record in self.movement) != self.moved_elements:
            raise CostMetrologyError("moved_elements must equal summed movement moved_elements")
        if sum(record.bytes_moved for record in self.movement) != self.bytes_moved:
            raise CostMetrologyError("bytes_moved must equal summed movement bytes_moved")
        if self.operation_trace and len(self.operation_trace) != self.operation_count:
            raise CostMetrologyError("operation_count must match operation_trace length")

    @classmethod
    def from_record(cls, record: object) -> Self:
        mapping = _record_mapping(record, "cost measurement record")
        roofline = mapping.get("roofline")
        if roofline is not None and not isinstance(roofline, Mapping):
            raise CostMetrologyError("roofline must be an object when present")
        return cls(
            cost_model_id=_record_string(mapping.get("cost_model_id"), "cost_model_id"),
            abstract_flops=_record_int(mapping.get("abstract_flops"), "abstract_flops"),
            per_op=tuple(
                OperationCostRecord.from_record(item)
                for item in _record_sequence(mapping.get("per_op"), "per_op")
            ),
            moved_elements=_record_int(mapping.get("moved_elements"), "moved_elements"),
            movement=tuple(
                MovementCostRecord.from_record(item)
                for item in _record_sequence(mapping.get("movement"), "movement")
            ),
            unmodeled_operations=tuple(
                UnmodeledOperationRecord.from_record(item)
                for item in _record_sequence(
                    mapping.get("unmodeled_operations"),
                    "unmodeled_operations",
                )
            ),
            operation_count=_record_int(mapping.get("operation_count"), "operation_count"),
            operation_trace=tuple(
                CostOperationTraceRecord.from_record(item)
                for item in _record_sequence(
                    mapping.get("operation_trace", ()),
                    "operation_trace",
                )
            ),
            wall_seconds=_record_float(mapping.get("wall_seconds"), "wall_seconds"),
            tensor_device=_record_string(mapping.get("tensor_device"), "tensor_device"),
            execution_mode=_record_execution_mode(
                mapping.get("execution_mode", "measured")
            ),
            operation_stream_source=_record_string(
                mapping.get("operation_stream_source", "runtime-executed"),
                "operation_stream_source",
            ),
            operations_executed=_record_bool(
                mapping.get("operations_executed", True),
                "operations_executed",
            ),
            bytes_moved=_record_int(mapping.get("bytes_moved", 0), "bytes_moved"),
            bytes_resident=_record_int(mapping.get("bytes_resident", 0), "bytes_resident"),
            roofline=cast(Mapping[str, object] | None, roofline),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "cost_model_id": self.cost_model_id,
            "abstract_flops": self.abstract_flops,
            "per_op": [op.to_record() for op in self.per_op],
            "moved_elements": self.moved_elements,
            "movement": [op.to_record() for op in self.movement],
            "unmodeled_operations": [op.to_record() for op in self.unmodeled_operations],
            "operation_count": self.operation_count,
            "operation_trace": [op.to_record() for op in self.operation_trace],
            "wall_seconds": self.wall_seconds,
            "tensor_device": self.tensor_device,
            "execution_mode": self.execution_mode,
            "operation_stream_source": self.operation_stream_source,
            "operations_executed": self.operations_executed,
            "bytes_moved": self.bytes_moved,
            "bytes_resident": self.bytes_resident,
        }
        if self.roofline is not None:
            record["roofline"] = dict(self.roofline)
        return record

    def without_operation_trace(self) -> CostMeasurement:
        """Return this measurement with the per-operation trace dropped.

        Durable evidence records store trace-free measurements: third-party
        verification is re-run-and-recount, so an embedded trace adds bulk
        without adding verifiability, and solver-scale traces (step count
        times ops per step) would dominate evidence size. The in-memory
        measurement keeps its trace for direct inspection and tests.
        """

        if not self.operation_trace:
            return self
        return replace(self, operation_trace=())

    def abstract_flops_per_item(self, item_count: int) -> float:
        """Return this measurement's abstract-FLOP count normalized by item count."""

        return self.abstract_flops_per_item_value(self.abstract_flops, item_count)

    def abstract_flops_per_byte(self, byte_count: int | float) -> float:
        """Return this measurement's abstract-FLOP intensity per byte."""

        return self.abstract_flops_per_byte_value(self.abstract_flops, byte_count)

    def abstract_flops_rate(self, items_per_second: int | float, *, item_count: int = 1) -> float:
        """Return observed abstract FLOPs per second from this measurement and throughput."""

        return self.abstract_flops_rate_value(
            self.abstract_flops_per_item(item_count),
            items_per_second,
        )

    def bit_density(self, *, item_count: int = 1) -> float:
        """Return this measurement's declared bit-density cost unit."""

        return self.abstract_flops_bit_density(self.abstract_flops_per_item(item_count))

    @classmethod
    def abstract_flops_bit_density(cls, abstract_flops: int | float) -> float:
        """Convert abstract FLOPs into the declared bit-density cost unit."""

        return _nonnegative_number(abstract_flops, "abstract_flops") * 32.0

    @classmethod
    def abstract_flops_per_item_value(
        cls,
        abstract_flops: int | float,
        item_count: int,
    ) -> float:
        """Return an abstract-FLOP count normalized by item count."""

        return _nonnegative_number(abstract_flops, "abstract_flops") / _positive_item_count(
            item_count
        )

    @classmethod
    def abstract_flops_per_byte_value(
        cls,
        abstract_flops: int | float,
        byte_count: int | float,
    ) -> float:
        """Return abstract-FLOP intensity per byte."""

        bytes_value = _nonnegative_number(byte_count, "byte_count")
        if bytes_value <= 0.0:
            raise CostMetrologyError("byte_count must be positive")
        return _nonnegative_number(abstract_flops, "abstract_flops") / bytes_value

    @classmethod
    def abstract_flops_rate_value(
        cls,
        abstract_flops_per_item: int | float,
        items_per_second: int | float,
    ) -> float:
        """Return observed abstract FLOPs per second from cost and throughput."""

        return _nonnegative_number(
            abstract_flops_per_item,
            "abstract_flops_per_item",
        ) * _nonnegative_number(items_per_second, "items_per_second")


class CostMeter:
    """Context manager that records a tensor runtime operation stream."""

    def __init__(
        self,
        runtime: TensorRuntime,
        *,
        strict: bool = False,
        roofline: Mapping[str, object] | None = None,
        device_profile: DeviceCostProfile | str | None = None,
    ) -> None:
        self._runtime = runtime
        self._strict = strict
        self._roofline = roofline
        self._device_profile = device_profile
        self._capture: Any | None = None
        self._active = False
        self._started_at = 0.0
        self._wall_seconds = 0.0

    def __enter__(self) -> Self:
        if self._active:
            raise CostMetrologyError("cost meter is already active")
        if self._capture is not None:
            raise CostMetrologyError("cost meter has already captured operations")
        capture = tensor_runtime_operation_capture(self._runtime)
        self._started_at = time.perf_counter()
        capture.__enter__()
        self._capture = capture
        self._active = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        capture = self._capture
        if capture is None or not self._active:
            return
        self._wall_seconds += time.perf_counter() - self._started_at
        self._active = False
        capture.__exit__(exc_type, exc, traceback)

    def measurement(self) -> CostMeasurement:
        capture = self._capture
        if capture is None:
            raise CostMetrologyError("cost meter has not captured any operations")
        measurement = _measurement_from_operation_stream(
            runtime=self._runtime,
            operations=capture.records(),
            wall_seconds=self._wall_seconds
            if not self._active
            else time.perf_counter() - self._started_at,
            strict=self._strict,
            roofline=self._roofline,
            execution_mode="measured",
            operation_stream_source="runtime-executed",
            operations_executed=True,
        )
        return _measurement_with_energy_profile(measurement, self._device_profile)


def measure_program_cost(
    runtime: TensorRuntime,
    program: Callable[..., object],
    inputs: object = (),
    *,
    strict: bool = False,
    roofline: Mapping[str, object] | None = None,
    device_profile: DeviceCostProfile | str | None = None,
) -> CostMeasurement:
    """Measure one execution of a program under the declared tensor cost model."""

    args = cast(tuple[object, ...], inputs) if isinstance(inputs, tuple) else (inputs,)
    def callback() -> object:
        return program(*args)

    started = time.perf_counter()
    operations = tensor_runtime_capture_operations(runtime, callback)
    wall_seconds = time.perf_counter() - started
    measurement = _measurement_from_operation_stream(
        runtime=runtime,
        operations=operations,
        wall_seconds=wall_seconds,
        strict=strict,
        roofline=roofline,
        execution_mode="measured",
        operation_stream_source="runtime-executed",
        operations_executed=True,
    )
    return _measurement_with_energy_profile(measurement, device_profile)


def estimate_operation_stream_cost(
    runtime: TensorRuntime,
    operations: Iterable[TensorRuntimeOperationRecord],
    *,
    strict: bool = False,
    roofline: Mapping[str, object] | None = None,
    device_profile: DeviceCostProfile | str | None = None,
    operation_stream_source: str = "runtime-dry-run",
) -> CostMeasurement:
    """Estimate cost for a projected tensor-runtime operation stream."""

    measurement = _measurement_from_operation_stream(
        runtime=runtime,
        operations=tuple(operations),
        wall_seconds=0.0,
        strict=strict,
        roofline=roofline,
        execution_mode="dry-run",
        operation_stream_source=operation_stream_source,
        operations_executed=False,
    )
    return _measurement_with_energy_profile(measurement, device_profile)


def estimate_program_cost(
    runtime: TensorRuntime,
    program: Callable[..., object],
    inputs: object = (),
    *,
    strict: bool = False,
    roofline: Mapping[str, object] | None = None,
    device_profile: DeviceCostProfile | str | None = None,
    operation_stream_source: str = "runtime-dry-run",
) -> CostMeasurement:
    """Estimate a program's cost from a dry-run tensor-runtime operation stream."""

    args = cast(tuple[object, ...], inputs) if isinstance(inputs, tuple) else (inputs,)

    def callback() -> object:
        return program(*args)

    return estimate_operation_stream_cost(
        runtime,
        tensor_runtime_project_operations(runtime, callback),
        strict=strict,
        roofline=roofline,
        device_profile=device_profile,
        operation_stream_source=operation_stream_source,
    )


def _measurement_with_energy_profile(
    measurement: CostMeasurement,
    profile: DeviceCostProfile | str | None,
) -> CostMeasurement:
    if profile is None:
        return measurement
    resolved_profile = device_cost_profile(profile) if isinstance(profile, str) else profile
    energy = price_cost_measurement_energy(measurement, profile=resolved_profile)
    roofline = dict(measurement.roofline or {})
    roofline["device_cost_profile"] = resolved_profile.to_record()
    roofline["energy"] = energy.to_record()
    return replace(measurement, roofline=roofline)


def _measurement_from_operation_stream(
    *,
    runtime: TensorRuntime,
    operations: tuple[TensorRuntimeOperationRecord, ...],
    wall_seconds: float,
    strict: bool,
    roofline: Mapping[str, object] | None,
    execution_mode: _CostExecutionMode,
    operation_stream_source: str,
    operations_executed: bool,
) -> CostMeasurement:
    per_op: dict[tuple[str, OperationClass | None, DeviceDType | None], _OperationAccumulator] = {}
    movement: dict[str, _MovementAccumulator] = {}
    unmodeled: dict[str, _UnmodeledAccumulator] = {}
    operation_trace: list[CostOperationTraceRecord] = []
    bytes_resident = 0
    for operation in operations:
        trace = _operation_trace_from_runtime(operation)
        operation_trace.append(trace)
        bytes_resident = max(bytes_resident, _operation_resident_bytes(trace))
        output_elements = _specs_numel(trace.output_tensors)
        moved_elements = _movement_elements(
            name=trace.name,
            input_specs=trace.input_tensors,
            output_specs=trace.output_tensors,
        )
        if moved_elements is not None:
            record = movement.setdefault(trace.name, _MovementAccumulator())
            record.calls += 1
            record.moved_elements += moved_elements
            record.bytes_moved += _specs_bytes(trace.output_tensors or trace.input_tensors)
            continue
        abstract_flops = _abstract_flops(
            name=trace.name,
            arguments=operation.arguments,
            keyword_arguments=operation.keyword_arguments,
            input_specs=trace.input_tensors,
            output_specs=trace.output_tensors,
        )
        if abstract_flops is None:
            if strict:
                raise CostMetrologyError(
                    "unmodeled operation in cost model "
                    f"{_tensor_runtime_cost_model_id}: {trace.name}"
                )
            record = unmodeled.setdefault(trace.name, _UnmodeledAccumulator())
            record.calls += 1
            record.output_elements += output_elements
            continue
        operation_class = operation_class_for_name(trace.name)
        dtype = _operation_dtype(trace)
        record = per_op.setdefault((trace.name, operation_class, dtype), _OperationAccumulator())
        record.calls += 1
        record.abstract_flops += abstract_flops
        record.output_elements += output_elements
    return CostMeasurement(
        cost_model_id=_tensor_runtime_cost_model_id,
        abstract_flops=sum(record.abstract_flops for record in per_op.values()),
        per_op=tuple(
            OperationCostRecord(
                name=name,
                calls=record.calls,
                abstract_flops=record.abstract_flops,
                output_elements=record.output_elements,
                operation_class=operation_class,
                dtype=dtype,
            )
            for (name, operation_class, dtype), record in sorted(per_op.items())
        ),
        moved_elements=sum(record.moved_elements for record in movement.values()),
        bytes_moved=sum(record.bytes_moved for record in movement.values()),
        movement=tuple(
            MovementCostRecord(
                name=name,
                calls=record.calls,
                moved_elements=record.moved_elements,
                bytes_moved=record.bytes_moved,
            )
            for name, record in sorted(movement.items())
        ),
        unmodeled_operations=tuple(
            UnmodeledOperationRecord(
                name=name,
                calls=record.calls,
                output_elements=record.output_elements,
            )
            for name, record in sorted(unmodeled.items())
        ),
        operation_count=len(operations),
        operation_trace=tuple(operation_trace),
        wall_seconds=wall_seconds,
        tensor_device=runtime.device_kind,
        execution_mode=execution_mode,
        operation_stream_source=operation_stream_source,
        operations_executed=operations_executed,
        bytes_resident=bytes_resident,
        roofline=roofline,
    )


def _operation_trace_from_runtime(
    operation: TensorRuntimeOperationRecord,
) -> CostOperationTraceRecord:
    return CostOperationTraceRecord(
        name=operation.name,
        input_tensors=tuple(_tensor_spec_from_runtime(spec) for spec in operation.input_tensors),
        output_tensors=tuple(
            _tensor_spec_from_runtime(spec) for spec in operation.output_tensors
        ),
    )


def _tensor_spec_from_runtime(spec: TensorRuntimeTensorSpec) -> TensorValueSpec:
    return TensorValueSpec(shape=spec.shape, dtype=spec.dtype)


@dataclass(slots=True)
class _OperationAccumulator:
    calls: int = 0
    abstract_flops: int = 0
    output_elements: int = 0


@dataclass(slots=True)
class _MovementAccumulator:
    calls: int = 0
    moved_elements: int = 0
    bytes_moved: int = 0


@dataclass(slots=True)
class _UnmodeledAccumulator:
    calls: int = 0
    output_elements: int = 0


def _abstract_flops(
    *,
    name: str,
    arguments: tuple[object, ...],
    keyword_arguments: tuple[tuple[str, object], ...],
    input_specs: tuple[TensorValueSpec, ...],
    output_specs: tuple[TensorValueSpec, ...],
) -> int | None:
    if name in {"aten.mm.default", "aten.matmul.default"}:
        return _matmul_flops(input_specs)
    if name == "aten.addmm.default":
        return _addmm_flops(input_specs, output_specs)
    if name == "aten.bmm.default":
        return _bmm_flops(input_specs)
    if name in {"aten.convolution.default", "aten.conv2d.default"}:
        return _convolution_flops(input_specs, output_specs)
    if name.startswith("aten._fft_"):
        return _fft_flops(
            name=name,
            arguments=arguments,
            keyword_arguments=keyword_arguments,
            input_specs=input_specs,
        )
    if name in _pointwise_ops:
        return _specs_numel(output_specs)
    if name in _reduction_ops:
        return _spec_numel(input_specs[0]) if input_specs else 0
    return None


def _matmul_flops(input_specs: tuple[TensorValueSpec, ...]) -> int | None:
    if len(input_specs) < 2:
        return None
    left, right = input_specs[0].shape, input_specs[1].shape
    if len(left) == 2 and len(right) == 2:
        return _multiply_add_flops(left[0], right[1], left[1])
    if len(left) >= 3 and len(right) >= 3 and left[:-2] == right[:-2]:
        return _multiply_add_flops(
            math.prod(left[:-2]),
            left[-2],
            right[-1],
            left[-1],
        )
    return None


def _multiply_add_flops(*factors: int) -> int:
    if not factors:
        return 0
    for factor in factors:
        if type(factor) is not int or factor < 0:
            raise CostMetrologyError("multiply-add factors must be nonnegative integers")
    return 2 * math.prod(factors)


def _addmm_flops(
    input_specs: tuple[TensorValueSpec, ...],
    output_specs: tuple[TensorValueSpec, ...],
) -> int | None:
    if len(input_specs) < 3:
        return None
    left, right = input_specs[1].shape, input_specs[2].shape
    if len(left) == 2 and len(right) == 2:
        return _multiply_add_flops(left[0], right[1], left[1])
    return _specs_numel(output_specs)


def _bmm_flops(input_specs: tuple[TensorValueSpec, ...]) -> int | None:
    if len(input_specs) < 2:
        return None
    left, right = input_specs[0].shape, input_specs[1].shape
    if len(left) != 3 or len(right) != 3:
        return None
    return _multiply_add_flops(left[0], left[1], right[2], left[2])


def _convolution_flops(
    input_specs: tuple[TensorValueSpec, ...],
    output_specs: tuple[TensorValueSpec, ...],
) -> int | None:
    if len(input_specs) < 2 or not output_specs:
        return None
    weight = input_specs[1].shape
    if len(weight) < 3:
        return None
    kernel_elements_per_output = math.prod(weight[1:])
    return _multiply_add_flops(
        _spec_numel(output_specs[0]),
        kernel_elements_per_output,
    )


def _fft_flops(
    *,
    name: str,
    arguments: tuple[object, ...],
    keyword_arguments: tuple[tuple[str, object], ...],
    input_specs: tuple[TensorValueSpec, ...],
) -> int | None:
    if not input_specs:
        return None
    shape = input_specs[0].shape
    if not shape:
        return 0
    dims = _fft_dims(
        arguments=arguments,
        keyword_arguments=keyword_arguments,
        rank=len(shape),
    )
    real_factor = 0.5 if name in {"aten._fft_r2c.default", "aten._fft_c2r.default"} else 1.0
    return _fft_formula_flops(shape, dims=dims, real_factor=real_factor)


def _fft_formula_flops(
    shape: tuple[int, ...],
    *,
    dims: Sequence[int],
    real_factor: float = 1.0,
) -> int:
    tensor_runtime_shape_element_count(shape)
    if isinstance(real_factor, bool):
        raise CostMetrologyError("real_factor must be numeric")
    if not math.isfinite(float(real_factor)) or real_factor < 0:
        raise CostMetrologyError("real_factor must be finite and nonnegative")
    total = 0.0
    for dim in dims:
        if type(dim) is not int:
            raise CostMetrologyError("FFT dimensions must be integers")
        extent = shape[dim % len(shape)]
        if extent <= 1:
            continue
        transform_count = math.prod(shape) / extent
        total += float(real_factor) * 5.0 * transform_count * extent * math.log2(extent)
    return int(round(total))


def _fft_dims(
    *,
    arguments: tuple[object, ...],
    keyword_arguments: tuple[tuple[str, object], ...],
    rank: int,
) -> tuple[int, ...]:
    keyword_map = dict(keyword_arguments)
    raw_dims = keyword_map.get("dim", keyword_map.get("dims"))
    if raw_dims is None:
        raw_dims = arguments[1] if len(arguments) >= 2 else rank - 1
    if isinstance(raw_dims, int):
        dims = (raw_dims,)
    elif isinstance(raw_dims, Sequence) and not isinstance(raw_dims, (str, bytes)):
        dims = tuple(
            _record_signed_int(dim, "fft dimension") for dim in cast(Sequence[object], raw_dims)
        )
    else:
        dims = (rank - 1,)
    return tuple(dim % rank for dim in dims)


def _movement_elements(
    *,
    name: str,
    input_specs: tuple[TensorValueSpec, ...],
    output_specs: tuple[TensorValueSpec, ...],
) -> int | None:
    if name not in _movement_ops and not _is_indexing_op(name):
        return None
    if output_specs:
        return _specs_numel(output_specs)
    return _specs_numel(input_specs)


def _operation_dtype(trace: CostOperationTraceRecord) -> DeviceDType | None:
    for spec in (*trace.output_tensors, *trace.input_tensors):
        dtype = normalize_tensor_dtype(spec.dtype)
        if dtype is not None:
            return dtype
    return None


def operation_class_for_name(name: str) -> OperationClass | None:
    """Return the declared device-cost operation class for a runtime op name."""

    _require_nonempty_string(name, "operation name")
    if name in _dense_matmul_ops:
        return "dense-matmul"
    if name in _convolution_ops:
        return "convolution"
    if name in _transcendental_ops:
        return "transcendental"
    if name in _pointwise_ops:
        return "elementwise"
    if name in _reduction_ops:
        return "reduction"
    if name.startswith("aten._fft_"):
        return "fft"
    if name in _movement_ops or _is_indexing_op(name):
        return "data-movement"
    return None


def normalize_tensor_dtype(dtype: str) -> DeviceDType | None:
    """Map tensor runtime dtype strings into the declared device-profile dtype set."""

    _require_nonempty_string(dtype, "tensor dtype")
    normalized = dtype.removeprefix("torch.").lower()
    if normalized in {"float64", "double", "complex128"}:
        return "fp64"
    if normalized in {"float32", "float", "complex64"}:
        return "fp32"
    if normalized in {"float16", "half"}:
        return "fp16"
    if normalized == "bfloat16":
        return "bf16"
    if normalized in {"int8", "uint8", "qint8", "quint8"}:
        return "int8"
    return None


def _is_indexing_op(name: str) -> bool:
    return (
        name.startswith("aten.index.")
        or name.startswith("aten.slice.")
        or name.startswith("aten.select.")
    )


def _spec_numel(spec: TensorValueSpec) -> int:
    return tensor_runtime_shape_element_count(spec.shape) if spec.shape else 1


def _specs_numel(specs: Iterable[TensorValueSpec]) -> int:
    return sum(_spec_numel(spec) for spec in specs)


def _operation_resident_bytes(trace: CostOperationTraceRecord) -> int:
    return _specs_bytes((*trace.input_tensors, *trace.output_tensors))


def _spec_bytes(spec: TensorValueSpec) -> int:
    return _spec_numel(spec) * _dtype_size_bytes(spec.dtype)


def _specs_bytes(specs: Iterable[TensorValueSpec]) -> int:
    return sum(_spec_bytes(spec) for spec in specs)


def _dtype_size_bytes(dtype: str) -> int:
    normalized = dtype.removeprefix("torch.").lower()
    if normalized in {"complex128"}:
        return 16
    if normalized in {"float64", "double", "complex64", "int64", "long"}:
        return 8
    if normalized in {"float32", "float", "complex32", "int32"}:
        return 4
    if normalized in {"float16", "half", "bfloat16", "int16", "short"}:
        return 2
    if normalized in {"int8", "uint8", "qint8", "quint8", "bool"}:
        return 1
    return 0


_pointwise_ops = frozenset(
    {
        "aten.abs.default",
        "aten.acos.default",
        "aten.add.Tensor",
        "aten.asin.default",
        "aten.atan.default",
        "aten.ceil.default",
        "aten.clamp.default",
        "aten.cos.default",
        "aten.div.Tensor",
        "aten.eq.Tensor",
        "aten.erf.default",
        "aten.exp.default",
        "aten.floor.default",
        "aten.ge.Tensor",
        "aten.gt.Tensor",
        "aten.le.Tensor",
        "aten.log.default",
        "aten.lt.Tensor",
        "aten.maximum.default",
        "aten.minimum.default",
        "aten.mul.Tensor",
        "aten.neg.default",
        "aten.ne.Tensor",
        "aten.pow.Tensor_Scalar",
        "aten.reciprocal.default",
        "aten.relu.default",
        "aten.rsqrt.default",
        "aten.sigmoid.default",
        "aten.sin.default",
        "aten.sqrt.default",
        "aten.sub.Tensor",
        "aten.tanh.default",
        "aten.where.self",
    }
)

_transcendental_ops = frozenset(
    {
        "aten.acos.default",
        "aten.asin.default",
        "aten.atan.default",
        "aten.cos.default",
        "aten.erf.default",
        "aten.exp.default",
        "aten.log.default",
        "aten.pow.Tensor_Scalar",
        "aten.rsqrt.default",
        "aten.sigmoid.default",
        "aten.sin.default",
        "aten.sqrt.default",
        "aten.tanh.default",
    }
)

_reduction_ops = frozenset(
    {
        "aten._adaptive_avg_pool2d.default",
        "aten.amax.default",
        "aten.argmax.default",
        "aten.argmin.default",
        "aten.max.default",
        "aten.mean.dim",
        "aten.min.default",
        "aten.prod.default",
        "aten.sum.default",
        "aten.sum.dim_IntList",
    }
)

_dense_matmul_ops = frozenset(
    {
        "aten.addmm.default",
        "aten.bmm.default",
        "aten.matmul.default",
        "aten.mm.default",
    }
)

_convolution_ops = frozenset(
    {
        "aten.convolution.default",
        "aten.conv2d.default",
    }
)

_movement_ops = frozenset(
    {
        "aten.clone.default",
        "aten.cat.default",
        "aten.copy.default",
        "aten.copy_.default",
        "aten.detach.default",
        "aten.full.default",
        "aten.gather.default",
        "aten.index_select.default",
        "aten.lift_fresh.default",
        "aten.new_empty.default",
        "aten.reshape.default",
        "aten.scatter.default",
        "aten.scatter_.src",
        "aten.squeeze.dim",
        "aten.t.default",
        "aten.to.device",
        "aten.unsqueeze.default",
        "aten.view.default",
    }
)


def _compute_energy_table(
    *,
    dense_fp32: float,
    unified_memory: bool = False,
) -> dict[tuple[OperationClass, DeviceDType], float]:
    dtype_scale = {
        "fp64": 4.0,
        "fp32": 1.0,
        "tf32": 0.25,
        "fp16": 0.125,
        "bf16": 0.125,
        "int8": 0.0625,
    }
    class_scale = {
        "dense-matmul": 1.0,
        "convolution": 1.15,
        "elementwise": 4.0 if not unified_memory else 2.5,
        "transcendental": 12.0,
        "reduction": 3.0,
        "fft": 5.0,
        "data-movement": 0.0,
    }
    return {
        (operation_class, dtype): dense_fp32 * class_scale[operation_class] * dtype_scale[dtype]
        for operation_class in _operation_classes
        for dtype in _device_dtypes
    }


def _record_mapping(record: object, field: str) -> Mapping[str, object]:
    if not isinstance(record, Mapping):
        raise CostMetrologyError(f"{field} must be an object")
    return cast(Mapping[str, object], record)


def _record_sequence(record: object, field: str) -> Sequence[object]:
    if not isinstance(record, Sequence) or isinstance(record, (str, bytes)):
        raise CostMetrologyError(f"{field} must be a sequence")
    return cast(Sequence[object], record)


def _record_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CostMetrologyError(f"{field} must be a nonempty string")
    return value


def _record_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CostMetrologyError(f"{field} must be an integer")
    if value < 0:
        raise CostMetrologyError(f"{field} must be nonnegative")
    return value


def _record_signed_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CostMetrologyError(f"{field} must be an integer")
    return value


def _record_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CostMetrologyError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise CostMetrologyError(f"{field} must be finite and nonnegative")
    return numeric


def _record_bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise CostMetrologyError(f"{field} must be a boolean")
    return value


def _positive_item_count(value: int) -> int:
    if type(value) is not int or value < 1:
        raise CostMetrologyError("item_count must be a positive integer")
    return value


def _nonnegative_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CostMetrologyError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise CostMetrologyError(f"{field} must be finite and nonnegative")
    return numeric


def _record_execution_mode(value: object) -> _CostExecutionMode:
    if value == "measured" or value == "dry-run":
        return cast(_CostExecutionMode, value)
    raise CostMetrologyError("execution_mode must be measured or dry-run")


def _record_operation_class(value: object, field: str) -> OperationClass:
    if value in _operation_classes:
        return value
    raise CostMetrologyError(f"{field} must be a known operation class")


def _record_device_dtype(value: object, field: str) -> DeviceDType:
    if value in _device_dtypes:
        return value
    raise CostMetrologyError(f"{field} must be a known dtype")


def _require_nonempty_string(value: str, field: str) -> None:
    if type(value) is not str or not value:
        raise CostMetrologyError(f"{field} must be a nonempty string")


def _require_nonnegative_int(value: int, field: str) -> None:
    if type(value) is not int:
        raise CostMetrologyError(f"{field} must be an integer")
    if value < 0:
        raise CostMetrologyError(f"{field} must be nonnegative")


def _require_finite_nonnegative_float(value: float, field: str) -> None:
    if type(value) not in {int, float}:
        raise CostMetrologyError(f"{field} must be numeric")
    if not math.isfinite(float(value)) or value < 0.0:
        raise CostMetrologyError(f"{field} must be finite and nonnegative")


_device_cost_profiles: Mapping[str, DeviceCostProfile] = {
    profile.profile_id: profile
    for profile in (
        DeviceCostProfile(
            profile_id="cost-model.device.nvidia-a100@0.1.0",
            label="NVIDIA A100 reference",
            version="0.1.0",
            provenance=(
                "Estimated from NVIDIA A100 public peak throughput, memory "
                "bandwidth, and TDP specs.",
                "https://www.nvidia.com/en-us/data-center/a100/",
            ),
            compute_energy_joules=_compute_energy_table(dense_fp32=2.05e-11),
            bytes_moved_energy_joules=2.57e-10,
            bytes_resident_energy_joules=2.57e-13,
            notes=("Default reference profile for canonical score pricing.",),
        ),
        DeviceCostProfile(
            profile_id="cost-model.device.nvidia-rtx-3080@0.1.0",
            label="NVIDIA RTX 3080",
            version="0.1.0",
            provenance=(
                "Estimated from NVIDIA GeForce RTX 3080 public throughput, "
                "memory bandwidth, and board power specs.",
                "https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3080/",
            ),
            compute_energy_joules=_compute_energy_table(dense_fp32=1.07e-11),
            bytes_moved_energy_joules=4.21e-10,
            bytes_resident_energy_joules=4.21e-13,
        ),
        DeviceCostProfile(
            profile_id="cost-model.device.apple-m1@0.1.0",
            label="Apple M1 unified-memory estimate",
            version="0.1.0",
            provenance=(
                "Estimated from Apple M1 public GPU throughput and "
                "unified-memory bandwidth disclosures.",
                "https://www.apple.com/newsroom/2020/11/apple-unleashes-m1/",
            ),
            compute_energy_joules=_compute_energy_table(
                dense_fp32=7.7e-12,
                unified_memory=True,
            ),
            bytes_moved_energy_joules=2.93e-10,
            bytes_resident_energy_joules=2.93e-13,
            unified_memory=True,
            notes=("Unified-memory profile; movement coefficient is coarse.",),
        ),
        DeviceCostProfile(
            profile_id="cost-model.device.apple-m4@0.1.0",
            label="Apple M4 unified-memory estimate",
            version="0.1.0",
            provenance=(
                "Estimated from Apple M4 public neural/GPU performance and "
                "memory-system disclosures.",
                "https://www.apple.com/newsroom/2024/05/apple-introduces-m4-chip/",
            ),
            compute_energy_joules=_compute_energy_table(
                dense_fp32=4.8e-12,
                unified_memory=True,
            ),
            bytes_moved_energy_joules=1.83e-10,
            bytes_resident_energy_joules=1.83e-13,
            unified_memory=True,
            notes=("Unified-memory profile; movement coefficient is coarse.",),
        ),
    )
}
