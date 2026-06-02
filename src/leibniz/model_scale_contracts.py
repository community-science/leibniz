"""Model input-shape envelopes for scale-general evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from leibniz.records import FieldSpec, RecordSpec
from leibniz.tensor_shapes import TensorShape, TensorShapeValidationError

__all__ = [
    "ModelScaleContract",
    "ModelScaleContractValidationError",
]

_fixed_axis_kind = "fixed"
_scaled_axis_kind = "scaled"

_scale_axis_record = RecordSpec(
    fields={
        "index": FieldSpec(kind="integer"),
        "kind": FieldSpec(kind="string"),
        "symbol": FieldSpec(kind="string", required=False),
        "size": FieldSpec(kind="integer", required=False),
    }
)
_scale_domain_record = RecordSpec(
    fields={
        "minimum": FieldSpec(kind="integer"),
        "maximum": FieldSpec(kind="integer", required=False),
    }
)
_model_scale_contract_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="literal", literal="positive-variable-shape-envelope"),
        "axis_symbol": FieldSpec(kind="string"),
        "anchor_shape": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="integer"),
        ),
        "axes": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record", record=_scale_axis_record),
        ),
        "scale_domain": FieldSpec(kind="record", record=_scale_domain_record),
    }
)


class ModelScaleContractValidationError(ValueError):
    """Raised when a model scale contract is invalid."""


@dataclass(frozen=True, slots=True)
class ModelScaleContract:
    """A variable-shape input envelope over positive integer tensor axes."""

    axis_symbol: str
    anchor_shape: tuple[int, ...]
    axes: tuple[Mapping[str, object], ...]
    minimum: int
    maximum: int | None = None

    def __post_init__(self) -> None:
        if not self.axis_symbol:
            raise ModelScaleContractValidationError("axis_symbol must be nonempty")
        _require_shape(self.anchor_shape, field="anchor_shape")
        if not self.axes:
            raise ModelScaleContractValidationError("axes must not be empty")
        if len(self.axes) != len(self.anchor_shape):
            raise ModelScaleContractValidationError("axes length must match anchor_shape rank")
        scaled_axes = 0
        for expected_index, axis in enumerate(self.axes):
            index = _as_int(axis.get("index"), "axes.index")
            if index != expected_index:
                raise ModelScaleContractValidationError("axis indexes must be contiguous")
            kind = _as_string(axis.get("kind"), "axes.kind")
            if kind == _scaled_axis_kind:
                scaled_axes += 1
                if _as_string(axis.get("symbol"), "axes.symbol") != self.axis_symbol:
                    raise ModelScaleContractValidationError(
                        "scaled axis symbol must match axis_symbol"
                    )
                if int(self.anchor_shape[index]) < self.minimum:
                    raise ModelScaleContractValidationError(
                        "scaled anchor axes must be at least minimum"
                    )
            elif kind == _fixed_axis_kind:
                if _as_int(axis.get("size"), "axes.size") != self.anchor_shape[index]:
                    raise ModelScaleContractValidationError(
                        "fixed axis size must match anchor_shape"
                    )
            else:
                raise ModelScaleContractValidationError(f"unsupported axis kind: {kind}")
        if scaled_axes < 1:
            raise ModelScaleContractValidationError(
                "contract must declare at least one scaled axis"
            )
        if type(self.minimum) is not int or self.minimum < 1:
            raise ModelScaleContractValidationError("minimum must be a positive integer")
        if self.maximum is not None and (
            type(self.maximum) is not int or self.maximum < self.minimum
        ):
            raise ModelScaleContractValidationError(
                "maximum must be at least minimum when present"
            )

    @classmethod
    def fixed_input_shape(
        cls,
        input_shape: tuple[int, ...],
        *,
        axis_symbol: str = "L",
        scale_axis_indices: Sequence[int] | None = None,
    ) -> ModelScaleContract:
        indices = _scale_axis_indices(input_shape, scale_axis_indices)
        minimum = input_shape[indices[0]]
        if any(input_shape[index] != minimum for index in indices):
            raise ModelScaleContractValidationError(
                "scaled input axes must share one anchor size"
            )
        return cls(
            axis_symbol=axis_symbol,
            anchor_shape=input_shape,
            axes=_axes_record(
                input_shape=input_shape,
                axis_symbol=axis_symbol,
                scale_axis_indices=indices,
            ),
            minimum=minimum,
            maximum=minimum,
        )

    @classmethod
    def variable_input_shape(
        cls,
        anchor_shape: tuple[int, ...],
        *,
        minimum: int,
        maximum: int | None = None,
        axis_symbol: str = "L",
        scale_axis_indices: Sequence[int] | None = None,
    ) -> ModelScaleContract:
        indices = _scale_axis_indices(anchor_shape, scale_axis_indices)
        return cls(
            axis_symbol=axis_symbol,
            anchor_shape=anchor_shape,
            axes=_axes_record(
                input_shape=anchor_shape,
                axis_symbol=axis_symbol,
                scale_axis_indices=indices,
            ),
            minimum=minimum,
            maximum=maximum,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelScaleContract:
        try:
            validated = _model_scale_contract_record.validate(record)
        except ValueError as error:
            raise ModelScaleContractValidationError(str(error)) from error
        domain = cast(Mapping[str, object], validated["scale_domain"])
        return cls(
            axis_symbol=_as_string(validated["axis_symbol"], "axis_symbol"),
            anchor_shape=_shape(validated["anchor_shape"], "anchor_shape"),
            axes=tuple(
                cast(Mapping[str, object], axis)
                for axis in _as_sequence(validated["axes"], "axes")
            ),
            minimum=_as_int(domain["minimum"], "scale_domain.minimum"),
            maximum=(
                None
                if "maximum" not in domain
                else _as_int(domain["maximum"], "scale_domain.maximum")
            ),
        )

    def accepts_scale(self, scale: int) -> bool:
        """Return whether the model claims support for this curriculum scale."""

        if type(scale) is not int or scale < self.minimum:
            return False
        return self.maximum is None or scale <= self.maximum

    def shape_for_scale(self, scale: int) -> tuple[int, ...]:
        """Return the declared input tensor shape at a curriculum scale."""

        if not self.accepts_scale(scale):
            raise ModelScaleContractValidationError("scale is outside contract domain")
        shape = list(self.anchor_shape)
        for axis in self.axes:
            if axis["kind"] == _scaled_axis_kind:
                shape[cast(int, axis["index"])] = scale
        return tuple(shape)

    def to_record(self) -> dict[str, object]:
        domain: dict[str, object] = {"minimum": self.minimum}
        if self.maximum is not None:
            domain["maximum"] = self.maximum
        return {
            "kind": "positive-variable-shape-envelope",
            "axis_symbol": self.axis_symbol,
            "anchor_shape": list(self.anchor_shape),
            "axes": [dict(axis) for axis in self.axes],
            "scale_domain": domain,
        }


def _scale_axis_indices(
    shape: tuple[int, ...],
    scale_axis_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    _require_shape(shape, field="input_shape")
    indices = tuple(scale_axis_indices) if scale_axis_indices is not None else (len(shape) - 1,)
    if not indices:
        raise ModelScaleContractValidationError("scale_axis_indices must not be empty")
    if len(set(indices)) != len(indices):
        raise ModelScaleContractValidationError("scale axis indexes must be unique")
    for index in indices:
        if type(index) is not int or index < 0 or index >= len(shape):
            raise ModelScaleContractValidationError("scale axis index is outside tensor rank")
    return tuple(sorted(indices))


def _axes_record(
    *,
    input_shape: tuple[int, ...],
    axis_symbol: str,
    scale_axis_indices: tuple[int, ...],
) -> tuple[Mapping[str, object], ...]:
    axes: list[Mapping[str, object]] = []
    for index, size in enumerate(input_shape):
        if index in scale_axis_indices:
            axes.append({"index": index, "kind": _scaled_axis_kind, "symbol": axis_symbol})
        else:
            axes.append({"index": index, "kind": _fixed_axis_kind, "size": size})
    return tuple(axes)


def _require_shape(shape: tuple[int, ...], *, field: str) -> None:
    try:
        TensorShape.from_axes(shape, field=field)
    except TensorShapeValidationError as error:
        raise ModelScaleContractValidationError(str(error)) from error


def _shape(value: object, field: str) -> tuple[int, ...]:
    try:
        return TensorShape.from_record(_as_sequence(value, field), field=field).axes
    except TensorShapeValidationError as error:
        raise ModelScaleContractValidationError(str(error)) from error


def _as_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise ModelScaleContractValidationError(f"{field}: expected integer")
    return value


def _as_sequence(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ModelScaleContractValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelScaleContractValidationError(f"{field}: expected nonempty string")
    return value
