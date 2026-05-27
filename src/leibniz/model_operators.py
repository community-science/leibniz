"""Formal model-operator semantics for architecture manifests."""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from leibniz.architectures import ArchitectureLayer, ArchitectureManifest

__all__ = [
    "ExecutableModelOperator",
    "ModelOperatorDescriptor",
    "ModelOperatorExecutionError",
    "ModelOperatorPlan",
    "ModelOperatorSummary",
    "formal_image_classifier_architecture",
    "summarize_architecture_operators",
]

_OperatorStateKind = Literal["learned", "fixed"]
_OperatorSupportKind = Literal["global", "local-window", "pointwise", "rank-collapsing"]
_TensorRelationKind = Literal["affine", "aggregation", "identity", "shape-transform"]
_operator_local_aggregation = "local-aggregation"
_operator_rank_collapse = "rank-collapse"
_operator_affine_readout = "affine-readout"


class ModelOperatorExecutionError(ValueError):
    """Raised when architecture operators cannot be interpreted."""


@dataclass(frozen=True, slots=True)
class ModelOperatorDescriptor:
    """Mathematical semantics of one executable operator specialization."""

    kind: str
    tensor_relation: _TensorRelationKind
    state: _OperatorStateKind
    support: _OperatorSupportKind
    projection_law: str
    aggregation_law: str
    parameter_sharing: str
    shape_law: str
    cost_law: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.kind:
            raise ModelOperatorExecutionError("operator kind must be nonempty")
        if not self.projection_law:
            raise ModelOperatorExecutionError("projection_law must be nonempty")
        if not self.aggregation_law:
            raise ModelOperatorExecutionError("aggregation_law must be nonempty")
        if not self.parameter_sharing:
            raise ModelOperatorExecutionError("parameter_sharing must be nonempty")
        if not self.shape_law:
            raise ModelOperatorExecutionError("shape_law must be nonempty")
        if not self.cost_law:
            raise ModelOperatorExecutionError("cost_law must be nonempty")
        if any(not alias for alias in self.aliases):
            raise ModelOperatorExecutionError("aliases must be nonempty")

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "tensor_relation": self.tensor_relation,
            "state": self.state,
            "support": self.support,
            "projection_law": self.projection_law,
            "aggregation_law": self.aggregation_law,
            "parameter_sharing": self.parameter_sharing,
            "shape_law": self.shape_law,
            "cost_law": self.cost_law,
            "aliases": list(self.aliases),
        }


@dataclass(frozen=True, slots=True)
class ModelOperatorSummary:
    """Shape and resource interpretation of one architecture layer."""

    index: int
    descriptor: ModelOperatorDescriptor
    input_shape: tuple[int, ...] | None
    output_shape: tuple[int, ...] | None
    parameter_count: int | None
    parameter_bytes: int | None
    inference_flops: int | None

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ModelOperatorExecutionError("index must be a nonnegative integer")
        _require_optional_shape(self.input_shape, field="input_shape")
        _require_optional_shape(self.output_shape, field="output_shape")
        _require_optional_count(self.parameter_count, field="parameter_count")
        _require_optional_count(self.parameter_bytes, field="parameter_bytes")
        _require_optional_count(self.inference_flops, field="inference_flops")

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "index": self.index,
            "operator": self.descriptor.to_record(),
        }
        if self.input_shape is not None:
            record["input_shape"] = list(self.input_shape)
        if self.output_shape is not None:
            record["output_shape"] = list(self.output_shape)
        if self.parameter_count is not None:
            record["parameter_count"] = self.parameter_count
        if self.parameter_bytes is not None:
            record["parameter_bytes"] = self.parameter_bytes
        if self.inference_flops is not None:
            record["inference_flops"] = self.inference_flops
        return record


@dataclass(frozen=True, slots=True)
class _LayerOperatorSpecialization:
    alias: str
    descriptor: ModelOperatorDescriptor


_layer_operator_specializations = (
    _LayerOperatorSpecialization(
        alias="adaptive-pooling",
        descriptor=ModelOperatorDescriptor(
            kind=_operator_local_aggregation,
            tensor_relation="aggregation",
            state="fixed",
            support="local-window",
            projection_law="equal-output-partition",
            aggregation_law="mean",
            parameter_sharing="none",
            shape_law="preserve-prefix-replace-trailing-axes",
            cost_law="input-elements",
        ),
    ),
    _LayerOperatorSpecialization(
        alias="flatten",
        descriptor=ModelOperatorDescriptor(
            kind=_operator_rank_collapse,
            tensor_relation="shape-transform",
            state="fixed",
            support="rank-collapsing",
            projection_law="row-major-axis-concatenation",
            aggregation_law="none",
            parameter_sharing="none",
            shape_law="product-of-input-axes",
            cost_law="zero-arithmetic",
        ),
    ),
    _LayerOperatorSpecialization(
        alias="dense",
        descriptor=ModelOperatorDescriptor(
            kind=_operator_affine_readout,
            tensor_relation="affine",
            state="learned",
            support="global",
            projection_law="full-input-support",
            aggregation_law="weighted-sum-plus-bias",
            parameter_sharing="none",
            shape_law="rank-1-output",
            cost_law="multiply-add-per-input-output-pair",
        ),
    ),
)
_layer_operator_descriptor_by_alias = {
    specialization.alias: specialization.descriptor
    for specialization in _layer_operator_specializations
}


@dataclass(frozen=True, slots=True)
class ModelOperatorPlan:
    """An executable formal-operator plan for one architecture manifest."""

    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    operators: tuple[ModelOperatorSummary, ...]

    @property
    def parameter_count(self) -> int | None:
        return _sum_known(operator.parameter_count for operator in self.operators)

    @property
    def parameter_bytes(self) -> int | None:
        return _sum_known(operator.parameter_bytes for operator in self.operators)

    @property
    def inference_flops(self) -> int | None:
        return _sum_known(operator.inference_flops for operator in self.operators)

    @property
    def unknown_parameter_layers(self) -> tuple[int, ...]:
        return tuple(
            operator.index
            for operator in self.operators
            if operator.parameter_count is None
        )

    @property
    def unknown_flop_layers(self) -> tuple[int, ...]:
        return tuple(
            operator.index
            for operator in self.operators
            if operator.inference_flops is None
        )


@dataclass(frozen=True, slots=True)
class ExecutableModelOperator:
    """Tiny PyTorch module wrapper for a manifest-backed operator plan."""

    architecture: ArchitectureManifest

    def torch_module(self) -> Any:
        """Instantiate a minimal PyTorch module for supported operator specializations."""

        try:
            torch = cast(Any, importlib.import_module("torch"))
        except ImportError as error:  # pragma: no cover - depends on local environment
            raise ModelOperatorExecutionError(
                "PyTorch is required to instantiate operators"
            ) from error

        modules: list[Any] = []
        shape = self.architecture.input_shape
        for layer in self.architecture.layers:
            descriptor = _descriptor_for_layer(layer)
            if descriptor.kind == _operator_local_aggregation:
                size = _positive_int_parameter(layer.parameters, "size")
                dimension = _positive_int_parameter(layer.parameters, "dimension")
                if dimension != 2:
                    raise ModelOperatorExecutionError(
                        "local aggregation currently supports dimension 2"
                    )
                modules.append(torch.nn.AdaptiveAvgPool2d(size))
                shape = (*shape[: len(shape) - dimension], size, size)
            elif descriptor.kind == _operator_rank_collapse:
                modules.append(torch.nn.Flatten())
                shape = (math.prod(shape),)
            elif descriptor.kind == _operator_affine_readout:
                if len(shape) != 1:
                    raise ModelOperatorExecutionError("affine readout requires rank-1 input")
                out = _positive_int_parameter(layer.parameters, "out")
                modules.append(torch.nn.Linear(shape[0], out))
                shape = (out,)
            else:
                raise ModelOperatorExecutionError(f"unsupported operator kind: {layer.kind}")
        return torch.nn.Sequential(*modules)


def summarize_architecture_operators(
    architecture: ArchitectureManifest,
    *,
    scalar_bytes: int = 4,
) -> ModelOperatorPlan:
    """Summarize shape, parameter, byte, and FLOP laws for an architecture."""

    if type(scalar_bytes) is not int or scalar_bytes < 1:
        raise ModelOperatorExecutionError("scalar_bytes must be a positive integer")
    shape: tuple[int, ...] | None = architecture.input_shape
    operators: list[ModelOperatorSummary] = []
    for index, layer in enumerate(architecture.layers):
        descriptor = _descriptor_for_layer(layer)
        input_shape = shape
        output_shape, parameter_count, inference_flops = _operator_shape_and_cost(
            layer,
            descriptor,
            input_shape,
        )
        operators.append(
            ModelOperatorSummary(
                index=index,
                descriptor=descriptor,
                input_shape=input_shape,
                output_shape=output_shape,
                parameter_count=parameter_count,
                parameter_bytes=(
                    None if parameter_count is None else parameter_count * scalar_bytes
                ),
                inference_flops=inference_flops,
            )
        )
        shape = output_shape
    if shape is not None and shape != architecture.output_shape:
        raise ModelOperatorExecutionError(
            "resolved operator output shape does not match architecture output_shape"
        )
    return ModelOperatorPlan(
        input_shape=architecture.input_shape,
        output_shape=architecture.output_shape,
        operators=tuple(operators),
    )


def formal_image_classifier_architecture(
    *,
    input_shape: tuple[int, ...],
    output_count: int,
    local_aggregation_size: int,
    local_aggregation_dimension: int = 2,
) -> ArchitectureManifest:
    """Build a manifest for the first formal image-classifier specialization."""

    _require_optional_shape(input_shape, field="input_shape")
    if type(output_count) is not int or output_count < 2:
        raise ModelOperatorExecutionError("output_count must be an integer at least 2")
    if type(local_aggregation_size) is not int or local_aggregation_size < 1:
        raise ModelOperatorExecutionError("local_aggregation_size must be positive")
    if type(local_aggregation_dimension) is not int or local_aggregation_dimension < 1:
        raise ModelOperatorExecutionError("local_aggregation_dimension must be positive")
    if len(input_shape) < 3:
        raise ModelOperatorExecutionError("image classifier input_shape must have rank at least 3")
    if local_aggregation_dimension >= len(input_shape) + 1:
        raise ModelOperatorExecutionError(
            "local_aggregation_dimension must not exceed input rank"
        )
    return ArchitectureManifest.from_record(
        {
            "input_shape": list(input_shape),
            "output_shape": [output_count],
            "layers": [
                {
                    "kind": _layer_operator_specializations[0].alias,
                    "parameters": {
                        "dimension": local_aggregation_dimension,
                        "size": local_aggregation_size,
                    },
                },
                {
                    "kind": _layer_operator_specializations[1].alias,
                },
                {
                    "kind": _layer_operator_specializations[2].alias,
                    "parameters": {
                        "out": output_count,
                    },
                },
            ],
        }
    )


def _descriptor_for_layer(layer: ArchitectureLayer) -> ModelOperatorDescriptor:
    descriptor = _layer_operator_descriptor_by_alias.get(layer.kind)
    if descriptor is None:
        raise ModelOperatorExecutionError(f"unsupported operator kind: {layer.kind}")
    return ModelOperatorDescriptor(
        kind=descriptor.kind,
        tensor_relation=descriptor.tensor_relation,
        state=descriptor.state,
        support=descriptor.support,
        projection_law=descriptor.projection_law,
        aggregation_law=descriptor.aggregation_law,
        parameter_sharing=descriptor.parameter_sharing,
        shape_law=descriptor.shape_law,
        cost_law=descriptor.cost_law,
        aliases=(layer.kind,),
    )


def _operator_shape_and_cost(
    layer: ArchitectureLayer,
    descriptor: ModelOperatorDescriptor,
    input_shape: tuple[int, ...] | None,
) -> tuple[tuple[int, ...] | None, int | None, int | None]:
    if input_shape is None:
        return None, None, None
    if descriptor.kind == _operator_rank_collapse:
        return (math.prod(input_shape),), 0, 0
    if descriptor.kind == _operator_affine_readout:
        if len(input_shape) != 1:
            return None, None, None
        out = _optional_positive_int_parameter(layer.parameters, "out")
        if out is None:
            return None, None, None
        return (out,), (input_shape[0] + 1) * out, (2 * input_shape[0]) * out
    if descriptor.kind == _operator_local_aggregation:
        if len(input_shape) < 2:
            return None, None, None
        size = _optional_positive_int_parameter(layer.parameters, "size")
        dimension = _optional_positive_int_parameter(layer.parameters, "dimension")
        if size is None or dimension is None or dimension >= len(input_shape) + 1:
            return None, None, None
        preserved = input_shape[: len(input_shape) - dimension]
        return (*preserved, *(size for _index in range(dimension))), 0, math.prod(input_shape)
    return None, None, None


def _positive_int_parameter(parameters: Mapping[str, object], key: str) -> int:
    value = _optional_positive_int_parameter(parameters, key)
    if value is None:
        raise ModelOperatorExecutionError(f"{key} must be a positive integer")
    return value


def _optional_positive_int_parameter(parameters: Mapping[str, object], key: str) -> int | None:
    value = parameters.get(key)
    if type(value) is not int or value < 1:
        return None
    return value


def _require_optional_shape(value: tuple[int, ...] | None, *, field: str) -> None:
    if value is None:
        return
    if not value or any(type(axis) is not int or axis < 1 for axis in value):
        raise ModelOperatorExecutionError(f"{field} must be a positive shape")


def _require_optional_count(value: int | None, *, field: str) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ModelOperatorExecutionError(f"{field} must be a nonnegative integer")


def _sum_known(values: Iterable[int | None]) -> int | None:
    sequence = tuple(values)
    if any(value is None for value in sequence):
        return None
    return sum(value for value in sequence if value is not None)
