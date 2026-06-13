"""Pure interpretation for declared model-operator semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from leibniz.operator_semantics import ModelOperatorSemantic
from leibniz.tensor_runtime import (
    tensor_runtime_shape_element_count,
)

__all__ = [
    "OperatorInterpretation",
    "OperatorInterpretationError",
    "interpret_operator_semantic",
    "spatial_axis_names",
]

_spatial_axis_names_by_dimension = {
    1: ("out_length",),
    2: ("out_height", "out_width"),
    3: ("out_depth", "out_height", "out_width"),
}


class OperatorInterpretationError(ValueError):
    """Raised when a declared operator semantic cannot be interpreted."""


@dataclass(frozen=True, slots=True)
class OperatorInterpretation:
    """Resolved shape and parameter laws for one declared operator."""

    output_shape: tuple[int, ...] | None
    parameter_count: int | None


def interpret_operator_semantic(
    semantic: ModelOperatorSemantic,
    *,
    parameters: Mapping[str, object],
    input_shape: tuple[int, ...] | None,
) -> OperatorInterpretation:
    """Interpret one declared semantic record against parameters and an input shape."""

    if input_shape is None:
        return OperatorInterpretation(
            output_shape=None,
            parameter_count=None,
        )
    if semantic.shape_law == "product-of-input-axes":
        return _interpret_product_of_input_axes(semantic, input_shape)
    if semantic.shape_law == "rank-1-output":
        return _interpret_rank_1_output(semantic, parameters, input_shape)
    if semantic.shape_law == "preserve-prefix-replace-trailing-axes":
        return _interpret_preserve_prefix_replace_trailing_axes(
            semantic,
            parameters,
            input_shape,
        )
    if semantic.shape_law == "preserve-prefix-local-window":
        return _interpret_preserve_prefix_local_window(
            semantic,
            parameters,
            input_shape,
        )
    if semantic.shape_law == "fixed-support-affine":
        return _interpret_fixed_support_affine(
            semantic,
            parameters,
            input_shape,
        )
    if semantic.shape_law == "preserve-input-shape":
        return _interpret_preserve_input_shape(semantic, input_shape)
    raise OperatorInterpretationError(f"unsupported shape_law: {semantic.shape_law}")


def spatial_axis_names(dimension: int) -> tuple[str, ...] | None:
    """Return fixed-support output-axis parameter names for a spatial dimension."""

    return _spatial_axis_names_by_dimension.get(dimension)


def _interpret_product_of_input_axes(
    semantic: ModelOperatorSemantic,
    input_shape: tuple[int, ...],
) -> OperatorInterpretation:
    _require_cost_law(semantic, "zero-arithmetic")
    return OperatorInterpretation(
        output_shape=(tensor_runtime_shape_element_count(input_shape),),
        parameter_count=0,
    )


def _interpret_rank_1_output(
    semantic: ModelOperatorSemantic,
    parameters: Mapping[str, object],
    input_shape: tuple[int, ...],
) -> OperatorInterpretation:
    _require_cost_law(semantic, "multiply-add-per-input-output-pair")
    if len(input_shape) != 1:
        return _unknown_interpretation()
    output_count = _optional_positive_int_parameter(parameters, "out")
    if output_count is None:
        return _unknown_interpretation()
    input_count = input_shape[0]
    return OperatorInterpretation(
        output_shape=(output_count,),
        parameter_count=(input_count + 1) * output_count,
    )


def _interpret_preserve_prefix_replace_trailing_axes(
    semantic: ModelOperatorSemantic,
    parameters: Mapping[str, object],
    input_shape: tuple[int, ...],
) -> OperatorInterpretation:
    _require_cost_law(semantic, "input-elements")
    if len(input_shape) < 2:
        return _unknown_interpretation()
    size = _optional_positive_int_parameter(parameters, "size")
    dimension = _optional_positive_int_parameter(parameters, "dimension")
    if dimension is None or dimension >= len(input_shape) + 1:
        return _unknown_interpretation()
    output_axes = _fixed_support_axes(parameters, dimension=dimension)
    preserved = input_shape[: len(input_shape) - dimension]
    if output_axes is None and size is not None:
        output_axes = tuple(size for _index in range(dimension))
    if output_axes is None:
        return _unknown_interpretation()
    return OperatorInterpretation(
        output_shape=(*preserved, *output_axes),
        parameter_count=0,
    )


def _interpret_preserve_prefix_local_window(
    semantic: ModelOperatorSemantic,
    parameters: Mapping[str, object],
    input_shape: tuple[int, ...],
) -> OperatorInterpretation:
    _require_cost_law(semantic, "local-window-multiply-add")
    dimension = _optional_positive_int_parameter(parameters, "dimension")
    size = _optional_positive_int_parameter(parameters, "size")
    out_channels = _optional_positive_int_parameter(parameters, "out_channels")
    stride = _optional_positive_int_parameter(parameters, "stride")
    padding = _optional_nonnegative_int_parameter(parameters, "padding")
    if (
        dimension is None
        or size is None
        or out_channels is None
        or stride is None
        or padding is None
        or len(input_shape) <= dimension
    ):
        return _unknown_interpretation()
    preserved = input_shape[: len(input_shape) - dimension - 1]
    input_channels = input_shape[len(input_shape) - dimension - 1]
    spatial_axes = input_shape[len(input_shape) - dimension :]
    output_spatial_axes = tuple(
        _local_window_output_axis(axis, size=size, stride=stride, padding=padding)
        for axis in spatial_axes
    )
    if any(axis is None for axis in output_spatial_axes):
        return _unknown_interpretation()
    output_spatial = tuple(axis for axis in output_spatial_axes if axis is not None)
    support_elements = size**dimension
    return OperatorInterpretation(
        output_shape=(*preserved, out_channels, *output_spatial),
        parameter_count=(input_channels * support_elements + 1) * out_channels,
    )


def _interpret_fixed_support_affine(
    semantic: ModelOperatorSemantic,
    parameters: Mapping[str, object],
    input_shape: tuple[int, ...],
) -> OperatorInterpretation:
    _require_cost_law(semantic, "adaptive-support-pointwise-affine")
    dimension = _optional_positive_int_parameter(parameters, "dimension")
    out_channels = _optional_positive_int_parameter(parameters, "out_channels")
    output_axes = (
        None if dimension is None else _fixed_support_axes(parameters, dimension=dimension)
    )
    if (
        dimension is None
        or out_channels is None
        or output_axes is None
        or len(input_shape) <= dimension
    ):
        return _unknown_interpretation()
    preserved = input_shape[: len(input_shape) - dimension - 1]
    input_channels = input_shape[len(input_shape) - dimension - 1]
    return OperatorInterpretation(
        output_shape=(*preserved, out_channels, *output_axes),
        parameter_count=(input_channels + 1) * out_channels,
    )


def _interpret_preserve_input_shape(
    semantic: ModelOperatorSemantic,
    input_shape: tuple[int, ...],
) -> OperatorInterpretation:
    _require_cost_law(semantic, "input-elements")
    return OperatorInterpretation(
        output_shape=input_shape,
        parameter_count=0,
    )


def _local_window_output_axis(
    axis: int,
    *,
    size: int,
    stride: int,
    padding: int,
) -> int | None:
    result = ((axis + 2 * padding - size) // stride) + 1
    if result < 1:
        return None
    return result


def _require_cost_law(semantic: ModelOperatorSemantic, expected: str) -> None:
    if semantic.cost_law != expected:
        raise OperatorInterpretationError(f"unsupported cost_law: {semantic.cost_law}")


def _optional_positive_int_parameter(
    parameters: Mapping[str, object],
    name: str,
) -> int | None:
    value = parameters.get(name)
    if type(value) is not int or value < 1:
        return None
    return value


def _optional_nonnegative_int_parameter(
    parameters: Mapping[str, object],
    name: str,
) -> int | None:
    value = parameters.get(name)
    if type(value) is not int or value < 0:
        return None
    return value


def _fixed_support_axes(
    parameters: Mapping[str, object],
    *,
    dimension: int,
) -> tuple[int, ...] | None:
    axis_names = spatial_axis_names(dimension)
    if axis_names is None:
        return None
    axes = tuple(_optional_positive_int_parameter(parameters, name) for name in axis_names)
    if any(axis is not None for axis in axes):
        if any(axis is None for axis in axes):
            return None
        return tuple(axis for axis in axes if axis is not None)
    size = _optional_positive_int_parameter(parameters, "size")
    if size is not None:
        return tuple(size for _index in range(dimension))
    return None


def _unknown_interpretation() -> OperatorInterpretation:
    return OperatorInterpretation(
        output_shape=None,
        parameter_count=None,
    )
