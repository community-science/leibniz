"""Semantic validation for architecture manifests."""

from __future__ import annotations

from collections.abc import Mapping

from leibniz.architectures import ArchitectureManifest
from leibniz.model_operators import (
    ModelOperatorExecutionError,
    ModelOperatorPlan,
    summarize_architecture_operators,
)
from leibniz.operator_interpretation import spatial_axis_names
from leibniz.operator_semantics import (
    ModelOperatorParameterRole,
    ModelOperatorSemantic,
    model_operator_semantic_registry,
)

__all__ = [
    "ArchitectureSemanticValidationError",
    "validate_architecture_semantics",
]


class ArchitectureSemanticValidationError(ValueError):
    """Raised when an architecture manifest fails semantic validation."""


def validate_architecture_semantics(
    architecture: ArchitectureManifest,
) -> ModelOperatorPlan:
    """Validate an architecture against declared operator semantics."""

    _validate_layer_parameters(architecture)
    try:
        plan = summarize_architecture_operators(architecture)
    except ModelOperatorExecutionError as error:
        raise ArchitectureSemanticValidationError(str(error)) from error
    for summary, layer in zip(plan.operators, architecture.layers, strict=True):
        if summary.output_shape is None:
            raise ArchitectureSemanticValidationError(
                f"layer {summary.index} ({layer.kind}): "
                "semantic interpretation could not resolve output_shape"
            )
        if summary.parameter_count is None:
            raise ArchitectureSemanticValidationError(
                f"layer {summary.index} ({layer.kind}): "
                "semantic interpretation could not resolve parameter_count"
            )
    return plan


def _validate_layer_parameters(architecture: ArchitectureManifest) -> None:
    registry = model_operator_semantic_registry()
    for index, layer in enumerate(architecture.layers):
        semantic = registry.semantic_for_alias(layer.kind)
        if semantic is None:
            raise ArchitectureSemanticValidationError(
                f"layer {index} ({layer.kind}): unsupported operator kind"
            )
        roles_by_name = {role.name: role for role in semantic.parameter_roles}
        required_roles = _required_parameter_roles(layer.kind, semantic, layer.parameters)
        for role in required_roles:
            if role.name not in layer.parameters:
                raise ArchitectureSemanticValidationError(
                    f"layer {index} ({layer.kind}): missing required parameter {role.name}"
                )
        for name, value in layer.parameters.items():
            role = roles_by_name.get(name)
            if role is None:
                continue
            _validate_parameter_value(
                value=value,
                role=role,
                layer_index=index,
                layer_kind=layer.kind,
            )


def _validate_parameter_value(
    *,
    value: object,
    role: ModelOperatorParameterRole,
    layer_index: int,
    layer_kind: str,
) -> None:
    if role.value_kind == "padding-mode":
        if value not in {"zeros", "periodic"}:
            raise ArchitectureSemanticValidationError(
                f"layer {layer_index} ({layer_kind}): parameter {role.name} must be "
                "one of: zeros, periodic"
            )
        return
    minimum = 0 if role.value_kind == "nonnegative-integer" else 1
    if role.value_kind not in {"positive-integer", "nonnegative-integer"}:
        raise ArchitectureSemanticValidationError(
            f"layer {layer_index} ({layer_kind}): "
            f"unsupported parameter value kind {role.value_kind}"
        )
    if type(value) is not int or value < minimum:
        requirement = (
            "a nonnegative integer"
            if role.value_kind == "nonnegative-integer"
            else "a positive integer"
        )
        raise ArchitectureSemanticValidationError(
            f"layer {layer_index} ({layer_kind}): parameter {role.name} must be "
            f"{requirement}"
        )


def _required_parameter_roles(
    layer_kind: str,
    semantic: ModelOperatorSemantic,
    parameters: Mapping[str, object],
) -> tuple[ModelOperatorParameterRole, ...]:
    roles = semantic.parameter_roles
    if semantic.kind == "local-aggregation" and layer_kind in semantic.syntax_aliases:
        required_names = {"dimension", "size"}
    elif layer_kind == "local-aggregation":
        required_names = _dimension_fixed_support_required_names(parameters)
    else:
        required_names = {role.name for role in roles if role.name != "padding_mode"}
        if semantic.kind == "fixed-support-affine":
            required_names -= {"out_length", "out_height", "out_width", "out_depth"}
            required_names |= _dimension_fixed_support_required_names(parameters)
    return tuple(role for role in roles if role.name in required_names)


def _dimension_fixed_support_required_names(
    parameters: Mapping[str, object],
) -> set[str]:
    dimension = parameters.get("dimension")
    if type(dimension) is not int:
        return {"dimension"}
    names = spatial_axis_names(dimension)
    if names is None:
        return {"dimension"}
    return {"dimension", *names}
