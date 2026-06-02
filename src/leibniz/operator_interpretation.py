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
    dimension = _optional_positive_int_parameter(parameters, "dimension")
    if size is None or dimension is None or dimension >= len(input_shape) + 1:
        return _unknown_interpretation()
    preserved = input_shape[: len(input_shape) - dimension]
    input_elements = TensorShape.from_axes(input_shape).element_count
    return OperatorInterpretation(
        output_shape=(*preserved, *(size for _index in range(dimension))),
        parameter_count=0,
        inference_compute=input_elements,
        training_compute_per_sample=2 * input_elements,
    )


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


def _unknown_interpretation() -> OperatorInterpretation:
    return OperatorInterpretation(
        output_shape=None,
        parameter_count=None,
        inference_compute=None,
        training_compute_per_sample=None,
    )
