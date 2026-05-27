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
    "ModelOperatorCoordinate",
    "ModelOperatorDescriptor",
    "ModelOperatorExecutionError",
    "ModelOperatorPlan",
    "ModelOperatorSearchPoint",
    "ModelOperatorSummary",
    "materialize_model_operator_search_point",
    "model_operator_semantic_coordinates",
    "model_operator_vocabulary",
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
class ModelOperatorCoordinate:
    """One stable semantic coordinate derived from an operator manifest."""

    name: str
    value: int | str

    def __post_init__(self) -> None:
        if not self.name:
            raise ModelOperatorExecutionError("coordinate name must be nonempty")
        if type(self.value) not in {int, str}:
            raise ModelOperatorExecutionError("coordinate value must be an integer or string")
        if self.value == "":
            raise ModelOperatorExecutionError("coordinate value must be nonempty")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class _LayerOperatorSpecialization:
    alias: str
    display_name: str
    descriptor: ModelOperatorDescriptor
    parameter_roles: tuple[tuple[str, str, str], ...] = ()


_layer_operator_specializations = (
    _LayerOperatorSpecialization(
        alias="adaptive-pooling",
        display_name="Local aggregation",
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
        parameter_roles=(
            ("dimension", "Support rank", "number of trailing axes aggregated"),
            ("size", "Output support size", "extent of each aggregated output axis"),
        ),
    ),
    _LayerOperatorSpecialization(
        alias="flatten",
        display_name="Rank collapse",
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
        display_name="Affine readout",
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
        parameter_roles=(("out", "Output coordinates", "rank-1 output extent"),),
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
class ModelOperatorSearchPoint:
    """Semantic coordinates for materializing an executable operator manifest."""

    local_support_dimension: int
    local_support_size: int

    def __post_init__(self) -> None:
        if type(self.local_support_dimension) is not int or self.local_support_dimension < 1:
            raise ModelOperatorExecutionError("local_support_dimension must be positive")
        if type(self.local_support_size) is not int or self.local_support_size < 1:
            raise ModelOperatorExecutionError("local_support_size must be positive")

    def to_parameters(self) -> tuple[tuple[str, int], ...]:
        return (
            ("local_support_dimension", self.local_support_dimension),
            ("local_support_size", self.local_support_size),
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


def model_operator_vocabulary() -> dict[str, object]:
    """Return the generated console-facing formal operator vocabulary."""

    return {
        "format": "leibniz.model-operator-vocabulary",
        "format_version": 1,
        "operators": [
            {
                "kind": specialization.descriptor.kind,
                "display_name": specialization.display_name,
                "descriptor": specialization.descriptor.to_record(),
                "syntax_aliases": [specialization.alias],
                "parameter_roles": [
                    {
                        "name": name,
                        "display_name": display_name,
                        "description": description,
                        "value_kind": "positive-integer",
                    }
                    for name, display_name, description in specialization.parameter_roles
                ],
            }
            for specialization in _layer_operator_specializations
        ],
        "descriptor_axes": _descriptor_axis_records(),
        "syntax_aliases": [
            {
                "alias": specialization.alias,
                "operator_kind": specialization.descriptor.kind,
                "display_name": specialization.display_name,
            }
            for specialization in _layer_operator_specializations
        ],
        "coordinate_descriptors": _coordinate_descriptor_records(),
    }


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


def model_operator_semantic_coordinates(
    architecture: ArchitectureManifest,
    *,
    plan: ModelOperatorPlan | None = None,
) -> tuple[ModelOperatorCoordinate, ...]:
    """Return architecture coordinates derived from operator semantics and resources."""

    resolved = summarize_architecture_operators(architecture) if plan is None else plan
    if resolved.input_shape != architecture.input_shape:
        raise ModelOperatorExecutionError("operator plan input_shape does not match architecture")
    if resolved.output_shape != architecture.output_shape:
        raise ModelOperatorExecutionError("operator plan output_shape does not match architecture")
    if len(resolved.operators) != len(architecture.layers):
        raise ModelOperatorExecutionError("operator plan length does not match architecture")

    coordinates: list[ModelOperatorCoordinate] = [
        ModelOperatorCoordinate("input.rank", len(architecture.input_shape)),
        ModelOperatorCoordinate("output.rank", len(architecture.output_shape)),
        ModelOperatorCoordinate("operator.count", len(resolved.operators)),
    ]
    for summary, layer in zip(resolved.operators, architecture.layers, strict=True):
        prefix = f"operator.{summary.index}"
        descriptor = summary.descriptor
        coordinates.extend(
            (
                ModelOperatorCoordinate(f"{prefix}.kind", descriptor.kind),
                ModelOperatorCoordinate(
                    f"{prefix}.tensor_relation",
                    descriptor.tensor_relation,
                ),
                ModelOperatorCoordinate(f"{prefix}.state", descriptor.state),
                ModelOperatorCoordinate(f"{prefix}.support", descriptor.support),
                ModelOperatorCoordinate(f"{prefix}.projection_law", descriptor.projection_law),
                ModelOperatorCoordinate(
                    f"{prefix}.aggregation_law",
                    descriptor.aggregation_law,
                ),
                ModelOperatorCoordinate(
                    f"{prefix}.parameter_sharing",
                    descriptor.parameter_sharing,
                ),
                ModelOperatorCoordinate(f"{prefix}.shape_law", descriptor.shape_law),
                ModelOperatorCoordinate(f"{prefix}.cost_law", descriptor.cost_law),
            )
        )
        _append_shape_coordinates(coordinates, prefix, summary)
        _append_salient_parameter_coordinates(coordinates, prefix, summary, layer)
    _append_optional_coordinate(coordinates, "resource.parameter_count", resolved.parameter_count)
    _append_optional_coordinate(coordinates, "resource.inference_flops", resolved.inference_flops)
    _reject_duplicate_coordinate_names(coordinates)
    return tuple(coordinates)


def materialize_model_operator_search_point(
    *,
    input_shape: tuple[int, ...],
    output_count: int,
    point: ModelOperatorSearchPoint,
) -> ArchitectureManifest:
    """Materialize semantic search coordinates through supported operator aliases."""

    _require_optional_shape(input_shape, field="input_shape")
    if type(output_count) is not int or output_count < 2:
        raise ModelOperatorExecutionError("output_count must be an integer at least 2")
    if len(input_shape) < 3:
        raise ModelOperatorExecutionError("input_shape must have rank at least 3")
    if point.local_support_dimension >= len(input_shape) + 1:
        raise ModelOperatorExecutionError(
            "local_support_dimension must not exceed input rank"
        )
    return ArchitectureManifest.from_record(
        {
            "input_shape": list(input_shape),
            "output_shape": [output_count],
            "layers": [
                {
                    "kind": _layer_operator_specializations[0].alias,
                    "parameters": {
                        "dimension": point.local_support_dimension,
                        "size": point.local_support_size,
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


def _append_shape_coordinates(
    coordinates: list[ModelOperatorCoordinate],
    prefix: str,
    summary: ModelOperatorSummary,
) -> None:
    if summary.input_shape is not None:
        coordinates.append(
            ModelOperatorCoordinate(f"{prefix}.input_rank", len(summary.input_shape))
        )
    if summary.output_shape is not None:
        coordinates.append(
            ModelOperatorCoordinate(f"{prefix}.output_rank", len(summary.output_shape))
        )


def _append_salient_parameter_coordinates(
    coordinates: list[ModelOperatorCoordinate],
    prefix: str,
    summary: ModelOperatorSummary,
    layer: ArchitectureLayer,
) -> None:
    if summary.descriptor.kind == _operator_local_aggregation:
        _append_required_parameter_coordinate(
            coordinates,
            f"{prefix}.local_support_dimension",
            layer,
            "dimension",
        )
        _append_required_parameter_coordinate(
            coordinates,
            f"{prefix}.local_support_size",
            layer,
            "size",
        )
    elif summary.descriptor.kind == _operator_affine_readout:
        _append_required_parameter_coordinate(
            coordinates,
            f"{prefix}.output_count",
            layer,
            "out",
        )


def _append_required_parameter_coordinate(
    coordinates: list[ModelOperatorCoordinate],
    coordinate_name: str,
    layer: ArchitectureLayer,
    parameter_name: str,
) -> None:
    coordinates.append(
        ModelOperatorCoordinate(
            coordinate_name,
            _positive_int_parameter(layer.parameters, parameter_name),
        )
    )


def _append_optional_coordinate(
    coordinates: list[ModelOperatorCoordinate],
    name: str,
    value: int | None,
) -> None:
    if value is not None:
        coordinates.append(ModelOperatorCoordinate(name, value))


def _reject_duplicate_coordinate_names(
    coordinates: list[ModelOperatorCoordinate],
) -> None:
    names = [coordinate.name for coordinate in coordinates]
    if len(set(names)) != len(names):
        raise ModelOperatorExecutionError("semantic coordinates must have unique names")


def _descriptor_axis_records() -> dict[str, list[dict[str, str]]]:
    return {
        "tensor_relation": _axis_value_records(
            {
                "affine": "Affine",
                "aggregation": "Aggregation",
                "identity": "Identity",
                "shape-transform": "Shape transform",
            }
        ),
        "state": _axis_value_records({"fixed": "Fixed", "learned": "Learned"}),
        "support": _axis_value_records(
            {
                "global": "Global",
                "local-window": "Local window",
                "pointwise": "Pointwise",
                "rank-collapsing": "Rank collapsing",
            }
        ),
        "projection_law": _axis_values_from_descriptors("projection_law"),
        "aggregation_law": _axis_values_from_descriptors("aggregation_law"),
        "parameter_sharing": _axis_values_from_descriptors("parameter_sharing"),
        "shape_law": _axis_values_from_descriptors("shape_law"),
        "cost_law": _axis_values_from_descriptors("cost_law"),
    }


def _axis_values_from_descriptors(field: str) -> list[dict[str, str]]:
    values = sorted(
        {
            str(getattr(specialization.descriptor, field))
            for specialization in _layer_operator_specializations
        }
    )
    return [{"value": value, "display_name": _title_from_token(value)} for value in values]


def _axis_value_records(values: Mapping[str, str]) -> list[dict[str, str]]:
    return [
        {"value": value, "display_name": display_name}
        for value, display_name in sorted(values.items())
    ]


def _coordinate_descriptor_records() -> list[dict[str, str]]:
    return [
        {
            "name": "input.rank",
            "display_name": "Input rank",
            "value_kind": "integer",
        },
        {
            "name": "output.rank",
            "display_name": "Output rank",
            "value_kind": "integer",
        },
        {
            "name": "operator.count",
            "display_name": "Operator count",
            "value_kind": "integer",
        },
        {
            "name": "operator.{index}.kind",
            "display_name": "Operator kind",
            "value_kind": "operator-kind",
        },
        {
            "name": "operator.{index}.tensor_relation",
            "display_name": "Tensor relation",
            "value_kind": "descriptor-axis",
        },
        {
            "name": "operator.{index}.support",
            "display_name": "Support",
            "value_kind": "descriptor-axis",
        },
        {
            "name": "operator.{index}.local_support_dimension",
            "display_name": "Local support dimension",
            "value_kind": "integer",
        },
        {
            "name": "operator.{index}.local_support_size",
            "display_name": "Local support size",
            "value_kind": "integer",
        },
        {
            "name": "operator.{index}.output_count",
            "display_name": "Output count",
            "value_kind": "integer",
        },
        {
            "name": "resource.parameter_count",
            "display_name": "Parameter count",
            "value_kind": "integer",
        },
        {
            "name": "resource.inference_flops",
            "display_name": "Inference FLOPs",
            "value_kind": "integer",
        },
    ]


def _title_from_token(value: str) -> str:
    return " ".join(part.capitalize() for part in value.replace("_", "-").split("-"))


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
