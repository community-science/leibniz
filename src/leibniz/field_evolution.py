"""Shared field-valued timestepper protocol helpers."""

from __future__ import annotations

from typing import Any

from leibniz.tensor_runtime import (
    TensorRuntime,
    no_grad_context,
    tensor_runtime_backend,
    tensor_runtime_concat,
)

__all__ = [
    "FieldEvolutionError",
    "field_stepper_state",
    "field_stepper_trajectory",
    "validate_field_stepper_nondegenerate",
]


class FieldEvolutionError(ValueError):
    """Raised when a field-valued model violates the timestepper protocol."""


def field_stepper_trajectory(
    *,
    runtime: TensorRuntime,
    module: Any,
    fields: Any,
    horizon: float,
    time_count: int,
) -> Any:
    """Roll out ``state <- step(state, dt)`` over a fixed horizon."""

    if time_count < 2:
        raise FieldEvolutionError("field evolution trajectory requires at least two time samples")
    dt = horizon / float(time_count - 1)
    state = fields
    states = [fields]
    for _index in range(1, time_count):
        state = field_stepper_state(runtime=runtime, module=module, fields=state, dt=dt)
        states.append(state)
    return tensor_runtime_concat(runtime, states, dim=1)


def field_stepper_state(
    *,
    runtime: TensorRuntime,
    module: Any,
    fields: Any,
    dt: float,
) -> Any:
    """Evaluate one field timestep and enforce shape preservation."""

    _ = runtime
    try:
        state = module(fields, float(dt))
    except TypeError as error:
        raise FieldEvolutionError(
            "field-valued operator must accept an input state and dt"
        ) from error
    if tuple(state.shape) != tuple(fields.shape):
        raise FieldEvolutionError(
            "field-valued operator must return state shape "
            f"{tuple(fields.shape)}, got {tuple(state.shape)}"
        )
    return state


def validate_field_stepper_nondegenerate(
    *,
    runtime: TensorRuntime,
    module: Any,
    fields: Any,
    dt: float,
) -> None:
    """Reject identity and dt-insensitive field steppers before training."""

    backend = tensor_runtime_backend(runtime)
    module_training = getattr(module, "training", None)
    module_was_training = bool(module_training) if module_training is not None else False
    eval_module = getattr(module, "eval", None)
    if callable(eval_module):
        eval_module()
    try:
        with no_grad_context(runtime):
            first = field_stepper_state(
                runtime=runtime,
                module=module,
                fields=fields,
                dt=dt,
            )
            second = field_stepper_state(
                runtime=runtime,
                module=module,
                fields=fields,
                dt=dt * 2.0,
            )
        if bool(backend.allclose(first, fields)):
            raise FieldEvolutionError(
                "field-valued operator must not be identity at nonzero dt"
            )
        if bool(backend.allclose(first, second)):
            raise FieldEvolutionError("field-valued operator output must vary with dt")
    finally:
        if module_was_training:
            train_module = getattr(module, "train", None)
            if callable(train_module):
                train_module()
