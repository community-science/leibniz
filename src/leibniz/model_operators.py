"""Formal model-operator semantics for architecture manifests."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from leibniz.architectures import ArchitectureLayer, ArchitectureManifest
from leibniz.model_scale_contracts import ModelScaleContract
from leibniz.operator_interpretation import interpret_operator_semantic
from leibniz.operator_semantics import ModelOperatorSemantic, model_operator_semantic_registry
from leibniz.program_effect_semantics import program_effect_semantic_registry
from leibniz.tensor_shapes import TensorShape, TensorShapeValidationError

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
_operator_registry = model_operator_semantic_registry()
_operator_local_aggregation = _operator_registry.operators[0].kind
_operator_rank_collapse = _operator_registry.operators[1].kind
_operator_affine_readout = _operator_registry.operators[2].kind
_program_effect_registry = program_effect_semantic_registry()
_program_effect_kinds = frozenset(
    effect.kind for effect in _program_effect_registry.effects
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
                shape = (TensorShape.from_axes(shape).element_count,)
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
        "operators": _operator_registry.operator_records(),
        "descriptor_axis_descriptors": _operator_registry.descriptor_axis_descriptor_records(),
        "descriptor_axes": _operator_registry.descriptor_axis_records(),
        "syntax_aliases": _operator_registry.syntax_alias_records(),
        "coordinate_descriptors": _operator_registry.coordinate_descriptor_records(),
        "program_effects": _program_effect_registry.effect_records(),
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
        semantic = _semantic_for_layer(layer)
        descriptor = _descriptor_for_semantic(semantic, aliases=(layer.kind,))
        input_shape = shape
        interpretation = interpret_operator_semantic(
            semantic,
            parameters=layer.parameters,
            input_shape=input_shape,
        )
        operators.append(
            ModelOperatorSummary(
                index=index,
                descriptor=descriptor,
                input_shape=input_shape,
                output_shape=interpretation.output_shape,
                parameter_count=interpretation.parameter_count,
                parameter_bytes=(
                    None
                    if interpretation.parameter_count is None
                    else interpretation.parameter_count * scalar_bytes
                ),
                inference_flops=interpretation.inference_flops,
            )
        )
        shape = interpretation.output_shape
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
    scale_axis_indices = tuple(
        range(len(input_shape) - point.local_support_dimension, len(input_shape))
    )
    return ArchitectureManifest.from_record(
        {
            "input_shape": list(input_shape),
            "model_scale_contract": ModelScaleContract.variable_input_shape(
                input_shape,
                minimum=point.local_support_size,
                axis_symbol="S",
                scale_axis_indices=scale_axis_indices,
            ).to_record(),
            "output_shape": [output_count],
            "layers": [
                {
                    "kind": _operator_registry.operators[0].syntax_aliases[0],
                    "parameters": {
                        "dimension": point.local_support_dimension,
                        "size": point.local_support_size,
                    },
                },
                {
                    "kind": _operator_registry.operators[1].syntax_aliases[0],
                },
                {
                    "kind": _operator_registry.operators[2].syntax_aliases[0],
                    "parameters": {
                        "out": output_count,
                    },
                },
            ],
        }
    )


def _descriptor_for_layer(layer: ArchitectureLayer) -> ModelOperatorDescriptor:
    semantic = _semantic_for_layer(layer)
    return _descriptor_for_semantic(semantic, aliases=(layer.kind,))


def _semantic_for_layer(layer: ArchitectureLayer) -> ModelOperatorSemantic:
    semantic = _operator_registry.semantic_for_alias(layer.kind)
    if semantic is None:
        raise ModelOperatorExecutionError(f"unsupported operator kind: {layer.kind}")
    return semantic


def _descriptor_for_semantic(
    semantic: ModelOperatorSemantic,
    *,
    aliases: tuple[str, ...],
) -> ModelOperatorDescriptor:
    return ModelOperatorDescriptor(
        kind=semantic.kind,
        tensor_relation=cast(_TensorRelationKind, semantic.tensor_relation),
        state=cast(_OperatorStateKind, semantic.state),
        support=cast(_OperatorSupportKind, semantic.support),
        projection_law=semantic.projection_law,
        aggregation_law=semantic.aggregation_law,
        parameter_sharing=semantic.parameter_sharing,
        shape_law=semantic.shape_law,
        cost_law=semantic.cost_law,
        aliases=aliases,
    )


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


def _program_effect_descriptor(effect: ModelProgramEffect) -> ModelProgramEffectDescriptor:
    semantic = _program_effect_registry.semantic_for_kind(effect.kind)
    if semantic is None:
        raise ModelOperatorExecutionError(f"unsupported program effect kind: {effect.kind}")
    return ModelProgramEffectDescriptor(
        kind=effect.kind,
        input_arity=semantic.input_arity(effect.arity),
        output_arity=semantic.output_arity(effect.arity),
        shape_law=semantic.shape_law,
        cost_law=semantic.cost_law,
        trace_law=semantic.trace_law,
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
    try:
        TensorShape.from_axes(value, field=field)
    except TensorShapeValidationError as error:
        raise ModelOperatorExecutionError(f"{field} must be a positive shape") from error


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
