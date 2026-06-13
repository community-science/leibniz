from collections.abc import Callable

from leibniz.operator_interpretation import (
    OperatorInterpretationError,
    interpret_operator_semantic,
)
from leibniz.operator_semantics import ModelOperatorSemantic, model_operator_semantic_registry


def test_operator_interpreter_preserves_current_public_shape_laws() -> None:
    registry = model_operator_semantic_registry()
    local_aggregation = registry.semantic_for_alias("local-aggregation")
    local_affine = registry.semantic_for_alias("local-affine")
    fixed_support_affine = registry.semantic_for_alias("fixed-support-affine")
    rank_collapse = registry.semantic_for_alias("rank-collapse")
    affine_readout = registry.semantic_for_alias("affine-readout")
    assert local_aggregation is not None
    assert local_affine is not None
    assert fixed_support_affine is not None
    assert rank_collapse is not None
    assert affine_readout is not None

    local = interpret_operator_semantic(
        local_aggregation,
        parameters={"dimension": 2, "size": 2},
        input_shape=(1, 32, 32),
    )
    local_learned = interpret_operator_semantic(
        local_affine,
        parameters={
            "dimension": 2,
            "size": 3,
            "out_channels": 8,
            "stride": 1,
            "padding": 1,
        },
        input_shape=(1, 32, 32),
    )
    fixed_support_learned = interpret_operator_semantic(
        fixed_support_affine,
        parameters={
            "dimension": 2,
            "out_channels": 6,
            "out_height": 12,
            "out_width": 12,
        },
        input_shape=(1, 32, 48),
    )
    flattened = interpret_operator_semantic(
        rank_collapse,
        parameters={},
        input_shape=local.output_shape,
    )
    readout = interpret_operator_semantic(
        affine_readout,
        parameters={"out": 10},
        input_shape=flattened.output_shape,
    )

    assert (
        local.output_shape,
        local.parameter_count,
    ) == (
        (1, 2, 2),
        0,
    )
    assert (
        local_learned.output_shape,
        local_learned.parameter_count,
    ) == (
        (8, 32, 32),
        80,
    )
    assert (
        fixed_support_learned.output_shape,
        fixed_support_learned.parameter_count,
    ) == (
        (6, 12, 12),
        12,
    )
    assert (
        flattened.output_shape,
        flattened.parameter_count,
    ) == (
        (4,),
        0,
    )
    assert (
        readout.output_shape,
        readout.parameter_count,
    ) == (
        (10,),
        50,
    )


def test_operator_interpreter_uses_dimension_specific_fixed_support_axes() -> None:
    registry = model_operator_semantic_registry()
    local_aggregation = registry.semantic_for_alias("local-aggregation")
    fixed_support_affine = registry.semantic_for_alias("fixed-support-affine")
    assert local_aggregation is not None
    assert fixed_support_affine is not None

    local = interpret_operator_semantic(
        local_aggregation,
        parameters={
            "dimension": 3,
            "out_depth": 2,
            "out_height": 3,
            "out_width": 4,
        },
        input_shape=(1, 9, 10, 11),
    )
    learned = interpret_operator_semantic(
        fixed_support_affine,
        parameters={
            "dimension": 1,
            "out_channels": 5,
            "out_length": 7,
        },
        input_shape=(2, 13),
    )

    assert local.output_shape == (1, 2, 3, 4)
    assert learned.output_shape == (5, 7)
    assert learned.parameter_count == 15


def test_operator_interpreter_returns_unknown_values_for_unresolved_inputs() -> None:
    registry = model_operator_semantic_registry()
    local_aggregation = registry.semantic_for_alias("local-aggregation")
    local_affine = registry.semantic_for_alias("local-affine")
    affine_readout = registry.semantic_for_alias("affine-readout")
    assert local_aggregation is not None
    assert local_affine is not None
    assert affine_readout is not None

    assert interpret_operator_semantic(
        local_aggregation,
        parameters={"dimension": 2, "size": 2},
        input_shape=None,
    ).output_shape is None
    assert interpret_operator_semantic(
        local_aggregation,
        parameters={"dimension": 4, "size": 2},
        input_shape=(1, 32, 32),
    ).parameter_count is None
    assert interpret_operator_semantic(
        local_affine,
        parameters={
            "dimension": 2,
            "size": 3,
            "out_channels": 8,
            "stride": 1,
            "padding": 0,
        },
        input_shape=(1, 2, 2),
    ).output_shape is None
    assert interpret_operator_semantic(
        affine_readout,
        parameters={"out": 10},
        input_shape=(1, 2, 2),
    ).output_shape is None


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
