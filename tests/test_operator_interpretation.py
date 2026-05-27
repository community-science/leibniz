from collections.abc import Callable

from leibniz.operator_interpretation import (
    OperatorInterpretationError,
    interpret_operator_semantic,
)
from leibniz.operator_semantics import ModelOperatorSemantic, model_operator_semantic_registry


def test_operator_interpreter_preserves_current_public_shape_and_cost_laws() -> None:
    registry = model_operator_semantic_registry()

    local = interpret_operator_semantic(
        registry.operators[0],
        parameters={"dimension": 2, "size": 2},
        input_shape=(1, 32, 32),
    )
    flattened = interpret_operator_semantic(
        registry.operators[1],
        parameters={},
        input_shape=local.output_shape,
    )
    readout = interpret_operator_semantic(
        registry.operators[2],
        parameters={"out": 10},
        input_shape=flattened.output_shape,
    )

    assert (local.output_shape, local.parameter_count, local.inference_flops) == (
        (1, 2, 2),
        0,
        1024,
    )
    assert (flattened.output_shape, flattened.parameter_count, flattened.inference_flops) == (
        (4,),
        0,
        0,
    )
    assert (readout.output_shape, readout.parameter_count, readout.inference_flops) == (
        (10,),
        50,
        80,
    )


def test_operator_interpreter_returns_unknown_values_for_unresolved_inputs() -> None:
    registry = model_operator_semantic_registry()

    assert interpret_operator_semantic(
        registry.operators[0],
        parameters={"dimension": 2, "size": 2},
        input_shape=None,
    ).output_shape is None
    assert interpret_operator_semantic(
        registry.operators[0],
        parameters={"dimension": 4, "size": 2},
        input_shape=(1, 32, 32),
    ).parameter_count is None
    assert interpret_operator_semantic(
        registry.operators[2],
        parameters={"out": 10},
        input_shape=(1, 2, 2),
    ).inference_flops is None


def test_operator_interpreter_rejects_undeclared_laws() -> None:
    assert str(
        capture_interpretation_error(
            lambda: interpret_operator_semantic(
                _semantic(shape_law="preserve-shape"),
                parameters={},
                input_shape=(4,),
            )
        )
    ) == "unsupported shape_law: preserve-shape"
    assert str(
        capture_interpretation_error(
            lambda: interpret_operator_semantic(
                _semantic(
                    shape_law="product-of-input-axes",
                    cost_law="input-elements",
                ),
                parameters={},
                input_shape=(4,),
            )
        )
    ) == "unsupported cost_law: input-elements"


def _semantic(
    *,
    shape_law: str,
    cost_law: str = "zero-arithmetic",
) -> ModelOperatorSemantic:
    return ModelOperatorSemantic(
        kind="test-operator",
        display_name="Test operator",
        tensor_relation="identity",
        state="fixed",
        support="pointwise",
        projection_law="none",
        aggregation_law="none",
        parameter_sharing="none",
        shape_law=shape_law,
        cost_law=cost_law,
        syntax_aliases=("test-alias",),
    )


def capture_interpretation_error(
    action: Callable[[], object],
) -> OperatorInterpretationError:
    try:
        action()
    except OperatorInterpretationError as error:
        return error
    raise AssertionError("expected OperatorInterpretationError")
