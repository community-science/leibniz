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
    "ModelProgramEffect",
    "ModelProgramEffectDescriptor",
    "ModelProgramEffectPlan",
    "ModelProgramEffectSummary",
    "materialize_model_operator_search_point",
    "model_operator_semantic_coordinates",
    "model_operator_vocabulary",
    "summarize_model_program_effects",
    "summarize_architecture_operators",
]

_OperatorStateKind = Literal["learned", "fixed"]
_OperatorSupportKind = Literal["global", "local-window", "pointwise", "rank-collapsing"]
_TensorRelationKind = Literal["affine", "aggregation", "identity", "shape-transform"]
_ProgramEffectKind = Literal[
    "branch",
    "identity-path",
    "merge",
    "parameter-sharing",
    "repeat",
    "route",
]
_operator_local_aggregation = "local-aggregation"
_operator_rank_collapse = "rank-collapse"
_operator_affine_readout = "affine-readout"
_program_effect_kinds = frozenset(
    ("branch", "identity-path", "merge", "parameter-sharing", "repeat", "route")
)


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
class ModelProgramEffect:
    """A higher-order program effect over operator paths."""

    kind: _ProgramEffectKind
    arity: int = 1
    repetitions: int = 1
    parameter_group: str | None = None
    nested_parameter_count: int = 0
    nested_inference_flops: int = 0

    def __post_init__(self) -> None:
        if self.kind not in _program_effect_kinds:
            raise ModelOperatorExecutionError(f"unsupported program effect kind: {self.kind}")
        if type(self.arity) is not int or self.arity < 1:
            raise ModelOperatorExecutionError("program effect arity must be positive")
        if self.kind in {"branch", "merge", "route", "parameter-sharing"} and self.arity < 2:
            raise ModelOperatorExecutionError(
                f"{self.kind} program effect arity must be at least 2"
            )
        if self.kind in {"identity-path", "repeat"} and self.arity != 1:
            raise ModelOperatorExecutionError(f"{self.kind} program effect arity must be 1")
        if type(self.repetitions) is not int or self.repetitions < 1:
            raise ModelOperatorExecutionError("program effect repetitions must be positive")
        if self.kind == "repeat" and self.repetitions < 2:
            raise ModelOperatorExecutionError(
                "repeat program effect repetitions must be at least 2"
            )
        if self.kind != "repeat" and self.repetitions != 1:
            raise ModelOperatorExecutionError(f"{self.kind} program effect repetitions must be 1")
        if self.parameter_group is not None and not self.parameter_group:
            raise ModelOperatorExecutionError("parameter_group must be nonempty")
        if self.kind == "parameter-sharing" and self.parameter_group is None:
            raise ModelOperatorExecutionError("parameter-sharing requires parameter_group")
        if self.kind != "parameter-sharing" and self.parameter_group is not None:
            raise ModelOperatorExecutionError(f"{self.kind} must not declare parameter_group")
        _require_count(self.nested_parameter_count, field="nested_parameter_count")
        _require_count(self.nested_inference_flops, field="nested_inference_flops")
        if self.kind != "repeat" and (
            self.nested_parameter_count != 0 or self.nested_inference_flops != 0
        ):
            raise ModelOperatorExecutionError(f"{self.kind} must not declare nested cost")

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": self.kind,
            "arity": self.arity,
        }
        if self.repetitions != 1:
            record["repetitions"] = self.repetitions
        if self.parameter_group is not None:
            record["parameter_group"] = self.parameter_group
        if self.nested_parameter_count != 0:
            record["nested_parameter_count"] = self.nested_parameter_count
        if self.nested_inference_flops != 0:
            record["nested_inference_flops"] = self.nested_inference_flops
        return record


@dataclass(frozen=True, slots=True)
class ModelProgramEffectDescriptor:
    """Shape, cost, and trace laws for one program effect kind."""

    kind: _ProgramEffectKind
    input_arity: int
    output_arity: int
    shape_law: str
    cost_law: str
    trace_law: str

    def __post_init__(self) -> None:
        if self.kind not in _program_effect_kinds:
            raise ModelOperatorExecutionError(f"unsupported program effect kind: {self.kind}")
        if type(self.input_arity) is not int or self.input_arity < 1:
            raise ModelOperatorExecutionError("input_arity must be positive")
        if type(self.output_arity) is not int or self.output_arity < 1:
            raise ModelOperatorExecutionError("output_arity must be positive")
        if not self.shape_law:
            raise ModelOperatorExecutionError("shape_law must be nonempty")
        if not self.cost_law:
            raise ModelOperatorExecutionError("cost_law must be nonempty")
        if not self.trace_law:
            raise ModelOperatorExecutionError("trace_law must be nonempty")

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "input_arity": self.input_arity,
            "output_arity": self.output_arity,
            "shape_law": self.shape_law,
            "cost_law": self.cost_law,
            "trace_law": self.trace_law,
        }


@dataclass(frozen=True, slots=True)
class ModelProgramEffectSummary:
    """Resolved shape, cost, and trace contribution of one program effect."""

    index: int
    effect: ModelProgramEffect
    descriptor: ModelProgramEffectDescriptor
    input_shapes: tuple[tuple[int, ...], ...]
    output_shapes: tuple[tuple[int, ...], ...]
    parameter_count: int
    inference_flops: int
    trace: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ModelOperatorExecutionError("index must be a nonnegative integer")
        if len(self.input_shapes) != self.descriptor.input_arity:
            raise ModelOperatorExecutionError("input_shapes length must match input_arity")
        if len(self.output_shapes) != self.descriptor.output_arity:
            raise ModelOperatorExecutionError("output_shapes length must match output_arity")
        for shape in (*self.input_shapes, *self.output_shapes):
            _require_optional_shape(shape, field="program effect shape")
        _require_count(self.parameter_count, field="parameter_count")
        _require_count(self.inference_flops, field="inference_flops")
        if not self.trace:
            raise ModelOperatorExecutionError("program effect trace must be nonempty")

    def to_record(self) -> dict[str, object]:
        return {
            "index": self.index,
            "effect": self.effect.to_record(),
            "descriptor": self.descriptor.to_record(),
            "input_shapes": [list(shape) for shape in self.input_shapes],
            "output_shapes": [list(shape) for shape in self.output_shapes],
            "parameter_count": self.parameter_count,
            "inference_flops": self.inference_flops,
            "trace": list(self.trace),
        }


@dataclass(frozen=True, slots=True)
class ModelProgramEffectPlan:
    """A resolved higher-order program-effect plan."""

    input_shape: tuple[int, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    effects: tuple[ModelProgramEffectSummary, ...]

    @property
    def output_shape(self) -> tuple[int, ...]:
        if len(self.output_shapes) != 1:
            raise ModelOperatorExecutionError("program effect plan has multiple outputs")
        return self.output_shapes[0]

    @property
    def parameter_count(self) -> int:
        return sum(effect.parameter_count for effect in self.effects)

    @property
    def inference_flops(self) -> int:
        return sum(effect.inference_flops for effect in self.effects)

    def to_record(self) -> dict[str, object]:
        return {
            "input_shape": list(self.input_shape),
            "output_shapes": [list(shape) for shape in self.output_shapes],
            "parameter_count": self.parameter_count,
            "inference_flops": self.inference_flops,
            "effects": [effect.to_record() for effect in self.effects],
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
        "descriptor_axis_descriptors": _descriptor_axis_descriptor_records(),
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


def summarize_model_program_effects(
    *,
    input_shape: tuple[int, ...],
    effects: tuple[ModelProgramEffect, ...],
) -> ModelProgramEffectPlan:
    """Resolve generic higher-order program effects into shape, cost, and trace summaries."""

    _require_optional_shape(input_shape, field="input_shape")
    if not effects:
        raise ModelOperatorExecutionError("program effects must not be empty")
    active_shapes = (input_shape,)
    summaries: list[ModelProgramEffectSummary] = []
    for index, effect in enumerate(effects):
        descriptor = _program_effect_descriptor(effect)
        if len(active_shapes) != descriptor.input_arity:
            raise ModelOperatorExecutionError(
                f"{effect.kind} expected {descriptor.input_arity} input path(s), "
                f"got {len(active_shapes)}"
            )
        output_shapes = _program_effect_output_shapes(effect, active_shapes)
        summary = ModelProgramEffectSummary(
            index=index,
            effect=effect,
            descriptor=descriptor,
            input_shapes=active_shapes,
            output_shapes=output_shapes,
            parameter_count=_program_effect_parameter_count(effect),
            inference_flops=_program_effect_inference_flops(effect),
            trace=_program_effect_trace(effect),
        )
        summaries.append(summary)
        active_shapes = output_shapes
    return ModelProgramEffectPlan(
        input_shape=input_shape,
        output_shapes=active_shapes,
        effects=tuple(summaries),
    )


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


def _descriptor_axis_descriptor_records() -> list[dict[str, str]]:
    return [
        {
            "name": "tensor_relation",
            "display_name": "Tensor relation",
        },
        {
            "name": "state",
            "display_name": "State",
        },
        {
            "name": "support",
            "display_name": "Support",
        },
        {
            "name": "projection_law",
            "display_name": "Projection law",
        },
        {
            "name": "aggregation_law",
            "display_name": "Aggregation law",
        },
        {
            "name": "parameter_sharing",
            "display_name": "Parameter sharing",
        },
        {
            "name": "shape_law",
            "display_name": "Shape law",
        },
        {
            "name": "cost_law",
            "display_name": "Cost law",
        },
    ]


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


def _program_effect_descriptor(effect: ModelProgramEffect) -> ModelProgramEffectDescriptor:
    if effect.kind == "branch":
        return ModelProgramEffectDescriptor(
            kind=effect.kind,
            input_arity=1,
            output_arity=effect.arity,
            shape_law="duplicate-input-shape",
            cost_law="zero-arithmetic",
            trace_law="fan-out",
        )
    if effect.kind == "merge":
        return ModelProgramEffectDescriptor(
            kind=effect.kind,
            input_arity=effect.arity,
            output_arity=1,
            shape_law="require-equal-input-shapes",
            cost_law="zero-arithmetic",
            trace_law="join-paths",
        )
    if effect.kind == "route":
        return ModelProgramEffectDescriptor(
            kind=effect.kind,
            input_arity=effect.arity,
            output_arity=1,
            shape_law="select-equal-input-shape",
            cost_law="control-flow-select",
            trace_law="select-path",
        )
    if effect.kind == "repeat":
        return ModelProgramEffectDescriptor(
            kind=effect.kind,
            input_arity=1,
            output_arity=1,
            shape_law="preserve-shape",
            cost_law="multiply-nested-cost",
            trace_law="repeat-nested-program",
        )
    if effect.kind == "identity-path":
        return ModelProgramEffectDescriptor(
            kind=effect.kind,
            input_arity=1,
            output_arity=1,
            shape_law="preserve-shape",
            cost_law="zero-arithmetic",
            trace_law="preserve-path",
        )
    return ModelProgramEffectDescriptor(
        kind=effect.kind,
        input_arity=effect.arity,
        output_arity=effect.arity,
        shape_law="preserve-shapes",
        cost_law="share-state",
        trace_law="share-parameter-group",
    )


def _program_effect_output_shapes(
    effect: ModelProgramEffect,
    input_shapes: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    if effect.kind == "branch":
        return tuple(input_shapes[0] for _index in range(effect.arity))
    if effect.kind in {"merge", "route"}:
        _require_equal_shapes(input_shapes, effect=effect.kind)
        return (input_shapes[0],)
    if effect.kind in {"identity-path", "repeat", "parameter-sharing"}:
        return input_shapes
    raise ModelOperatorExecutionError(f"unsupported program effect kind: {effect.kind}")


def _program_effect_parameter_count(effect: ModelProgramEffect) -> int:
    if effect.kind == "repeat":
        return effect.repetitions * effect.nested_parameter_count
    return 0


def _program_effect_inference_flops(effect: ModelProgramEffect) -> int:
    if effect.kind == "repeat":
        return effect.repetitions * effect.nested_inference_flops
    return 0


def _program_effect_trace(effect: ModelProgramEffect) -> tuple[str, ...]:
    if effect.kind == "branch":
        return (f"branch fan-out={effect.arity}",)
    if effect.kind == "merge":
        return (f"merge arity={effect.arity}",)
    if effect.kind == "route":
        return (f"route choices={effect.arity}",)
    if effect.kind == "repeat":
        return (
            f"repeat count={effect.repetitions}",
            f"nested_parameter_count={effect.nested_parameter_count}",
            f"nested_inference_flops={effect.nested_inference_flops}",
        )
    if effect.kind == "identity-path":
        return ("identity-path",)
    return (f"parameter-sharing group={effect.parameter_group} arity={effect.arity}",)


def _require_equal_shapes(
    shapes: tuple[tuple[int, ...], ...],
    *,
    effect: str,
) -> None:
    first = shapes[0]
    if any(shape != first for shape in shapes):
        raise ModelOperatorExecutionError(f"{effect} requires equal input shapes")


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


def _require_count(value: int, *, field: str) -> None:
    if type(value) is not int or value < 0:
        raise ModelOperatorExecutionError(f"{field} must be a nonnegative integer")


def _sum_known(values: Iterable[int | None]) -> int | None:
    sequence = tuple(values)
    if any(value is None for value in sequence):
        return None
    return sum(value for value in sequence if value is not None)
