"""Pure interpretation for declared model-operator semantics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from leibniz.operator_semantics import ModelOperatorSemantic
from leibniz.tensor_shapes import TensorShape

__all__ = [
    "OperatorInterpretation",
    "OperatorInterpretationError",
    "interpret_operator_semantic",
]


class OperatorInterpretationError(ValueError):
    """Raised when a declared operator semantic cannot be interpreted."""


@dataclass(frozen=True, slots=True)
class OperatorInterpretation:
    """Resolved shape and resource laws for one declared operator."""

    output_shape: tuple[int, ...] | None
    parameter_count: int | None
    inference_compute: int | None
    training_compute_per_sample: int | None


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
            inference_compute=None,
            training_compute_per_sample=None,
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
    raise OperatorInterpretationError(f"unsupported shape_law: {semantic.shape_law}")


def _interpret_product_of_input_axes(
    semantic: ModelOperatorSemantic,
    input_shape: tuple[int, ...],
) -> OperatorInterpretation:
    _require_cost_law(semantic, "zero-arithmetic")
    return OperatorInterpretation(
        output_shape=(TensorShape.from_axes(input_shape).element_count,),
        parameter_count=0,
        inference_compute=0,
        training_compute_per_sample=0,
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
        inference_compute=(2 * input_count) * output_count,
        training_compute_per_sample=(6 * input_count) * output_count,
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
    out_height, out_width = _fixed_support_axes(parameters)
    dimension = _optional_positive_int_parameter(parameters, "dimension")
    if dimension is None or dimension >= len(input_shape) + 1:
        return _unknown_interpretation()
    preserved = input_shape[: len(input_shape) - dimension]
    if dimension == 2 and out_height is not None and out_width is not None:
        output_axes = (out_height, out_width)
    elif size is not None:
        output_axes = tuple(size for _index in range(dimension))
    else:
        return _unknown_interpretation()
    input_elements = TensorShape.from_axes(input_shape).element_count
    return OperatorInterpretation(
        output_shape=(*preserved, *output_axes),
        parameter_count=0,
        inference_compute=input_elements,
        training_compute_per_sample=2 * input_elements,
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
    output_positions = TensorShape.from_axes(output_spatial).element_count
    pair_compute = 2 * input_channels * support_elements * out_channels * output_positions
    return OperatorInterpretation(
        output_shape=(*preserved, out_channels, *output_spatial),
        parameter_count=(input_channels * support_elements + 1) * out_channels,
        inference_compute=pair_compute,
        training_compute_per_sample=3 * pair_compute,
    )


def _interpret_fixed_support_affine(
    semantic: ModelOperatorSemantic,
    parameters: Mapping[str, object],
    input_shape: tuple[int, ...],
) -> OperatorInterpretation:
    _require_cost_law(semantic, "adaptive-support-pointwise-affine")
    dimension = _optional_positive_int_parameter(parameters, "dimension")
    out_channels = _optional_positive_int_parameter(parameters, "out_channels")
    out_height = _optional_positive_int_parameter(parameters, "out_height")
    out_width = _optional_positive_int_parameter(parameters, "out_width")
    if (
        dimension != 2
        or out_channels is None
        or out_height is None
        or out_width is None
        or len(input_shape) <= dimension
    ):
        return _unknown_interpretation()
    preserved = input_shape[: len(input_shape) - dimension - 1]
    input_channels = input_shape[len(input_shape) - dimension - 1]
    input_elements = TensorShape.from_axes(input_shape).element_count
    output_positions = out_height * out_width
    affine_compute = 2 * input_channels * out_channels * output_positions
    inference_compute = input_elements + affine_compute
    return OperatorInterpretation(
        output_shape=(*preserved, out_channels, out_height, out_width),
        parameter_count=(input_channels + 1) * out_channels,
        inference_compute=inference_compute,
        training_compute_per_sample=(2 * input_elements) + (3 * affine_compute),
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


def _fixed_support_axes(parameters: Mapping[str, object]) -> tuple[int | None, int | None]:
    out_height = _optional_positive_int_parameter(parameters, "out_height")
    out_width = _optional_positive_int_parameter(parameters, "out_width")
    if out_height is not None or out_width is not None:
        return out_height, out_width
    size = _optional_positive_int_parameter(parameters, "size")
    return size, size


def _unknown_interpretation() -> OperatorInterpretation:
    return OperatorInterpretation(
        output_shape=None,
        parameter_count=None,
        inference_compute=None,
        training_compute_per_sample=None,
    )
