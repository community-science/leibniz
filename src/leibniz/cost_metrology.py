"""Measured PyTorch program cost with a declared abstract-FLOP model."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Self, cast

from leibniz.tensor_runtime import TensorRuntime

__all__ = [
    "PYTORCH_COST_MODEL_ID",
    "CostMeasurement",
    "CostMetrologyError",
    "CostMeter",
    "CostOperationTraceRecord",
    "MovementCostRecord",
    "OperationCostRecord",
    "TensorValueSpec",
    "UnmodeledOperationRecord",
    "measure_program_cost",
]

PYTORCH_COST_MODEL_ID = "leibniz.cost-model.pytorch@0.1.0"


class CostMetrologyError(ValueError):
    """Raised when a cost measurement or record is invalid."""


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

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, "operation name")
        _require_nonnegative_int(self.calls, "operation calls")
        _require_nonnegative_int(self.abstract_flops, "operation abstract_flops")
        _require_nonnegative_int(self.output_elements, "operation output_elements")

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
        )

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "calls": self.calls,
            "abstract_flops": self.abstract_flops,
            "output_elements": self.output_elements,
        }


@dataclass(frozen=True, slots=True)
class MovementCostRecord:
    """Aggregated element movement for one movement-class op name."""

    name: str
    calls: int
    moved_elements: int

    def __post_init__(self) -> None:
        _require_nonempty_string(self.name, "movement operation name")
        _require_nonnegative_int(self.calls, "movement operation calls")
        _require_nonnegative_int(self.moved_elements, "movement operation moved_elements")

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
        )

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "calls": self.calls,
            "moved_elements": self.moved_elements,
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
    """Measured abstract cost for one program execution."""

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
    roofline: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _require_nonempty_string(self.cost_model_id, "cost_model_id")
        _require_nonnegative_int(self.abstract_flops, "abstract_flops")
        _require_nonnegative_int(self.moved_elements, "moved_elements")
        _require_nonnegative_int(self.operation_count, "operation_count")
        _require_finite_nonnegative_float(self.wall_seconds, "wall_seconds")
        _require_nonempty_string(self.tensor_device, "tensor_device")
        if sum(record.abstract_flops for record in self.per_op) != self.abstract_flops:
            raise CostMetrologyError("abstract_flops must equal summed per_op abstract_flops")
        if sum(record.moved_elements for record in self.movement) != self.moved_elements:
            raise CostMetrologyError("moved_elements must equal summed movement moved_elements")
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
        }
        if self.roofline is not None:
            record["roofline"] = dict(self.roofline)
        return record


class CostMeter:
    """Context manager that records a PyTorch aten op stream for a runtime."""

    def __init__(
        self,
        runtime: TensorRuntime,
        *,
        strict: bool = False,
        roofline: Mapping[str, object] | None = None,
    ) -> None:
        self._runtime = runtime
        self._strict = strict
        self._roofline = roofline
        self._mode: object | None = None
        self._started_at = 0.0
        self._wall_seconds = 0.0
        self._per_op: dict[str, _OperationAccumulator] = {}
        self._movement: dict[str, _MovementAccumulator] = {}
        self._unmodeled: dict[str, _UnmodeledAccumulator] = {}
        self._operation_trace: list[CostOperationTraceRecord] = []
        self.operation_count = 0

    def __enter__(self) -> Self:
        if self._mode is not None:
            raise CostMetrologyError("cost meter is already active")
        from torch.utils._python_dispatch import TorchDispatchMode

        parent = self

        class _DispatchCostMode(TorchDispatchMode):  # type: ignore[misc]
            def __torch_dispatch__(
                self,
                func: object,
                types: object,
                args: tuple[object, ...] = (),
                kwargs: Mapping[str, object] | None = None,
            ) -> object:
                del types
                keyword_arguments = dict(kwargs or {})
                name = _operation_name(func)
                input_specs = _tensor_specs_from_value(args)
                callable_func = cast(Callable[..., object], func)
                output = callable_func(*args, **keyword_arguments)
                output_specs = _tensor_specs_from_value(output)
                parent._record_operation(
                    name=name,
                    args=args,
                    input_specs=input_specs,
                    output_specs=output_specs,
                )
                return output

        self._started_at = time.perf_counter()
        self._mode = _DispatchCostMode()
        self._mode.__enter__()  # type: ignore[attr-defined]
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._mode is None:
            return
        self._wall_seconds += time.perf_counter() - self._started_at
        mode = self._mode
        self._mode = None
        mode.__exit__(exc_type, exc, traceback)  # type: ignore[attr-defined]

    def measurement(self) -> CostMeasurement:
        return CostMeasurement(
            cost_model_id=PYTORCH_COST_MODEL_ID,
            abstract_flops=sum(record.abstract_flops for record in self._per_op.values()),
            per_op=tuple(
                OperationCostRecord(
                    name=name,
                    calls=record.calls,
                    abstract_flops=record.abstract_flops,
                    output_elements=record.output_elements,
                )
                for name, record in sorted(self._per_op.items())
            ),
            moved_elements=sum(record.moved_elements for record in self._movement.values()),
            movement=tuple(
                MovementCostRecord(
                    name=name,
                    calls=record.calls,
                    moved_elements=record.moved_elements,
                )
                for name, record in sorted(self._movement.items())
            ),
            unmodeled_operations=tuple(
                UnmodeledOperationRecord(
                    name=name,
                    calls=record.calls,
                    output_elements=record.output_elements,
                )
                for name, record in sorted(self._unmodeled.items())
            ),
            operation_count=self.operation_count,
            operation_trace=tuple(self._operation_trace),
            wall_seconds=self._wall_seconds,
            tensor_device=self._runtime.device_kind,
            roofline=self._roofline,
        )

    def _record_operation(
        self,
        *,
        name: str,
        args: tuple[object, ...],
        input_specs: tuple[TensorValueSpec, ...],
        output_specs: tuple[TensorValueSpec, ...],
    ) -> None:
        self.operation_count += 1
        self._operation_trace.append(
            CostOperationTraceRecord(
                name=name,
                input_tensors=input_specs,
                output_tensors=output_specs,
            )
        )
        output_elements = _specs_numel(output_specs)
        moved_elements = _movement_elements(
            name=name,
            input_specs=input_specs,
            output_specs=output_specs,
        )
        if moved_elements is not None:
            record = self._movement.setdefault(name, _MovementAccumulator())
            record.calls += 1
            record.moved_elements += moved_elements
            return
        abstract_flops = _abstract_flops(
            name=name,
            args=args,
            input_specs=input_specs,
            output_specs=output_specs,
        )
        if abstract_flops is None:
            if self._strict:
                raise CostMetrologyError(
                    f"unmodeled operation in cost model {PYTORCH_COST_MODEL_ID}: {name}"
                )
            record = self._unmodeled.setdefault(name, _UnmodeledAccumulator())
            record.calls += 1
            record.output_elements += output_elements
            return
        record = self._per_op.setdefault(name, _OperationAccumulator())
        record.calls += 1
        record.abstract_flops += abstract_flops
        record.output_elements += output_elements


def measure_program_cost(
    runtime: TensorRuntime,
    program: Callable[..., object],
    inputs: object = (),
    *,
    strict: bool = False,
    roofline: Mapping[str, object] | None = None,
) -> CostMeasurement:
    """Measure one execution of a program under the declared PyTorch cost model."""

    args = cast(tuple[object, ...], inputs) if isinstance(inputs, tuple) else (inputs,)
    with CostMeter(runtime, strict=strict, roofline=roofline) as meter:
        program(*args)
    return meter.measurement()


@dataclass(slots=True)
class _OperationAccumulator:
    calls: int = 0
    abstract_flops: int = 0
    output_elements: int = 0


@dataclass(slots=True)
class _MovementAccumulator:
    calls: int = 0
    moved_elements: int = 0


@dataclass(slots=True)
class _UnmodeledAccumulator:
    calls: int = 0
    output_elements: int = 0


def _operation_name(func: object) -> str:
    raw = str(func)
    if raw.startswith("aten."):
        return raw
    name = getattr(func, "__name__", None)
    if isinstance(name, str) and name:
        return f"aten.{name}"
    return raw


def _tensor_specs_from_value(value: object) -> tuple[TensorValueSpec, ...]:
    specs: list[TensorValueSpec] = []
    _append_tensor_specs(specs, value)
    return tuple(specs)


def _append_tensor_specs(specs: list[TensorValueSpec], value: object) -> None:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        try:
            specs.append(
                TensorValueSpec(
                    shape=tuple(int(extent) for extent in shape),
                    dtype=str(dtype).removeprefix("torch."),
                )
            )
            return
        except TypeError:
            pass
    if isinstance(value, Mapping):
        for item in cast(Mapping[object, object], value).values():
            _append_tensor_specs(specs, item)
        return
    if isinstance(value, (tuple, list)):
        for item in cast(Sequence[object], value):
            _append_tensor_specs(specs, item)


def _abstract_flops(
    *,
    name: str,
    args: tuple[object, ...],
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
        return _fft_flops(name=name, args=args, input_specs=input_specs)
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
        return 2 * left[0] * right[1] * left[1]
    if len(left) >= 3 and len(right) >= 3 and left[:-2] == right[:-2]:
        return 2 * math.prod(left[:-2]) * left[-2] * right[-1] * left[-1]
    return None


def _addmm_flops(
    input_specs: tuple[TensorValueSpec, ...],
    output_specs: tuple[TensorValueSpec, ...],
) -> int | None:
    if len(input_specs) < 3:
        return None
    left, right = input_specs[1].shape, input_specs[2].shape
    if len(left) == 2 and len(right) == 2:
        return 2 * left[0] * right[1] * left[1]
    return _specs_numel(output_specs)


def _bmm_flops(input_specs: tuple[TensorValueSpec, ...]) -> int | None:
    if len(input_specs) < 2:
        return None
    left, right = input_specs[0].shape, input_specs[1].shape
    if len(left) != 3 or len(right) != 3:
        return None
    return 2 * left[0] * left[1] * right[2] * left[2]


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
    return 2 * _spec_numel(output_specs[0]) * kernel_elements_per_output


def _fft_flops(
    *,
    name: str,
    args: tuple[object, ...],
    input_specs: tuple[TensorValueSpec, ...],
) -> int | None:
    if not input_specs:
        return None
    shape = input_specs[0].shape
    if not shape:
        return 0
    dims = _fft_dims(args=args, rank=len(shape))
    real_factor = 0.5 if name in {"aten._fft_r2c.default", "aten._fft_c2r.default"} else 1.0
    total = 0.0
    for dim in dims:
        extent = shape[dim]
        if extent <= 1:
            continue
        transform_count = math.prod(shape) / extent
        total += real_factor * 5.0 * transform_count * extent * math.log2(extent)
    return int(round(total))


def _fft_dims(*, args: tuple[object, ...], rank: int) -> tuple[int, ...]:
    if len(args) < 2:
        return (rank - 1,)
    raw_dims = args[1]
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


def _is_indexing_op(name: str) -> bool:
    return (
        name.startswith("aten.index.")
        or name.startswith("aten.slice.")
        or name.startswith("aten.select.")
    )


def _spec_numel(spec: TensorValueSpec) -> int:
    return math.prod(spec.shape)


def _specs_numel(specs: Iterable[TensorValueSpec]) -> int:
    return sum(_spec_numel(spec) for spec in specs)


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

_movement_ops = frozenset(
    {
        "aten.clone.default",
        "aten.copy.default",
        "aten.copy_.default",
        "aten.detach.default",
        "aten.gather.default",
        "aten.index_select.default",
        "aten.lift_fresh.default",
        "aten.reshape.default",
        "aten.scatter.default",
        "aten.scatter_.src",
        "aten.t.default",
        "aten.to.device",
        "aten.view.default",
    }
)


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
