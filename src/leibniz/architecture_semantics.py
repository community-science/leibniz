"""Semantic validation for architecture manifests."""

from __future__ import annotations

from leibniz.architectures import ArchitectureManifest
from leibniz.model_operators import (
    ModelOperatorExecutionError,
    ModelOperatorPlan,
    summarize_architecture_operators,
)
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
        if summary.inference_compute is None:
            raise ArchitectureSemanticValidationError(
                f"layer {summary.index} ({layer.kind}): "
                "semantic interpretation could not resolve inference_compute"
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
        for role in _required_parameter_roles(layer.kind, semantic):
            if role.name not in layer.parameters:
                raise ArchitectureSemanticValidationError(
                    f"layer {index} ({layer.kind}): missing required parameter {role.name}"
                )
            value = layer.parameters[role.name]
            minimum = 0 if role.value_kind == "nonnegative-integer" else 1
            if role.value_kind not in {"positive-integer", "nonnegative-integer"}:
                raise ArchitectureSemanticValidationError(
                    f"layer {index} ({layer.kind}): "
                    f"unsupported parameter value kind {role.value_kind}"
                )
            if type(value) is not int or value < minimum:
                requirement = (
                    "a nonnegative integer"
                    if role.value_kind == "nonnegative-integer"
                    else "a positive integer"
                )
                raise ArchitectureSemanticValidationError(
                    f"layer {index} ({layer.kind}): parameter {role.name} must be "
                    f"{requirement}"
                )


def _required_parameter_roles(
    layer_kind: str,
    semantic: ModelOperatorSemantic,
) -> tuple[ModelOperatorParameterRole, ...]:
    roles = semantic.parameter_roles
    if semantic.kind == "local-aggregation" and layer_kind in semantic.syntax_aliases:
        required_names = {"dimension", "size"}
    elif layer_kind == "local-aggregation":
        required_names = {"dimension", "out_height", "out_width"}
    else:
        required_names = {role.name for role in roles}
    return tuple(role for role in roles if role.name in required_names)
    return roles
