from collections.abc import Callable

from leibniz.operator_semantics import (
    DescriptorAxis,
    DescriptorAxisValue,
    ModelOperatorParameterRole,
    ModelOperatorSemantic,
    ModelOperatorSemanticRegistry,
    OperatorSemanticValidationError,
    SemanticCoordinateDescriptor,
    model_operator_semantic_registry,
)


def test_model_operator_semantic_registry_declares_current_public_vocabulary() -> None:
    registry = model_operator_semantic_registry()

    assert [operator.kind for operator in registry.operators] == [
        "local-aggregation",
        "local-affine",
        "fixed-support-affine",
        "rectified-linear-activation",
        "rank-collapse",
        "affine-readout",
    ]
    assert [record["alias"] for record in registry.syntax_alias_records()] == [
        "adaptive-pooling",
        "convolution",
        "relu",
        "flatten",
        "dense",
    ]
    assert registry.semantic_for_alias("local-aggregation") == registry.operators[0]
    assert registry.semantic_for_alias("adaptive-pooling") == registry.operators[0]
    assert registry.semantic_for_alias("local-affine") == registry.operators[1]
    assert registry.semantic_for_alias("convolution") == registry.operators[1]
    assert registry.semantic_for_alias("fixed-support-affine") == registry.operators[2]
    assert registry.semantic_for_alias("rectified-linear-activation") == registry.operators[3]
    assert registry.semantic_for_alias("relu") == registry.operators[3]
    assert registry.semantic_for_alias("rank-collapse") == registry.operators[4]
    assert registry.semantic_for_alias("flatten") == registry.operators[4]
    assert registry.semantic_for_alias("affine-readout") == registry.operators[5]
    assert registry.semantic_for_alias("dense") == registry.operators[5]
    assert registry.operators[0].descriptor_record(aliases=("adaptive-pooling",)) == {
        "kind": "local-aggregation",
        "tensor_relation": "aggregation",
        "state": "fixed",
        "support": "local-window",
        "projection_law": "equal-output-partition",
        "aggregation_law": "mean",
        "parameter_sharing": "none",
        "shape_law": "preserve-prefix-replace-trailing-axes",
        "cost_law": "input-elements",
        "aliases": ["adaptive-pooling"],
    }


def test_model_operator_semantic_registry_exports_console_metadata() -> None:
    registry = model_operator_semantic_registry()

    assert registry.operator_records()[0]["display_name"] == "Local aggregation"
    assert registry.operator_records()[0]["parameter_roles"] == [
        {
            "name": "dimension",
            "display_name": "Support rank",
            "description": "number of trailing axes aggregated",
            "value_kind": "positive-integer",
        },
        {
            "name": "out_length",
            "display_name": "Output length",
            "description": "fixed extent of a one-dimensional output support axis",
            "value_kind": "positive-integer",
        },
        {
            "name": "out_height",
            "display_name": "Output height",
            "description": "fixed extent of the first aggregated output support axis",
            "value_kind": "positive-integer",
        },
        {
            "name": "out_width",
            "display_name": "Output width",
            "description": "fixed extent of the second aggregated output support axis",
            "value_kind": "positive-integer",
        },
        {
            "name": "size",
            "display_name": "Output support size",
            "description": "square output support extent for adaptive-pooling syntax",
            "value_kind": "positive-integer",
        },
    ]
    assert registry.descriptor_axis_descriptor_records()[0] == {
        "name": "tensor_relation",
        "display_name": "Tensor relation",
    }
    assert registry.descriptor_axis_records()["support"][0] == {
        "value": "global",
        "display_name": "Global",
    }
    assert {
        descriptor["name"]: descriptor["display_name"]
        for descriptor in registry.coordinate_descriptor_records()
    }["operator.{index}.local_support_size"] == "Local support size"
    assert registry.operator_records()[1]["parameter_roles"] == [
        {
            "name": "dimension",
            "display_name": "Support rank",
            "description": "number of trailing axes in each local support window",
            "value_kind": "positive-integer",
        },
        {
            "name": "size",
            "display_name": "Support size",
            "description": "extent of each local support axis",
            "value_kind": "positive-integer",
        },
        {
            "name": "out_channels",
            "display_name": "Output channels",
            "description": "number of learned output coordinates per local window",
            "value_kind": "positive-integer",
        },
        {
            "name": "stride",
            "display_name": "Stride",
            "description": "step size between adjacent local windows",
            "value_kind": "positive-integer",
        },
        {
            "name": "padding",
            "display_name": "Padding",
            "description": "zero padding on each local support axis",
            "value_kind": "nonnegative-integer",
        },
        {
            "name": "padding_mode",
            "display_name": "Padding mode",
            "description": "boundary convention for padded local windows",
            "value_kind": "padding-mode",
        },
    ]
    assert registry.syntax_alias_records()[1]["specialization"] == {
        "kind": "local-affine",
        "tensor_relation": "affine",
        "state": "learned",
        "support": "local-window",
        "projection_law": "sliding-window",
        "aggregation_law": "weighted-sum-plus-bias",
        "parameter_sharing": "shared-local-window",
        "shape_law": "preserve-prefix-local-window",
        "cost_law": "local-window-multiply-add",
        "aliases": ["convolution"],
    }
    for alias_record in registry.syntax_alias_records():
        alias = str(alias_record["alias"])
        semantic = registry.semantic_for_alias(alias)
        assert semantic is not None
        assert alias_record["specialization"] == semantic.descriptor_record(
            aliases=(str(alias),)
        )


def test_model_operator_semantic_registry_rejects_duplicates() -> None:
    operator = _semantic()
    axis = DescriptorAxis(
        name="state",
        display_name="State",
        values=(DescriptorAxisValue(value="fixed", display_name="Fixed"),),
    )
    coordinate = SemanticCoordinateDescriptor(
        name="operator.count",
        display_name="Operator count",
        value_kind="integer",
    )

    assert str(
        capture_semantic_error(
            lambda: ModelOperatorSemanticRegistry(
                operators=(operator, operator),
                descriptor_axes=(axis,),
                coordinate_descriptors=(coordinate,),
            )
        )
    ) == "operator kinds must be unique"
    assert str(
        capture_semantic_error(
            lambda: ModelOperatorSemanticRegistry(
                operators=(
                    operator,
                    _semantic(kind="other-kind", syntax_aliases=("test-alias",)),
                ),
                descriptor_axes=(axis,),
                coordinate_descriptors=(coordinate,),
            )
        )
    ) == "operator public names must be unique"


def test_model_operator_semantic_records_reject_empty_fields() -> None:
    assert str(
        capture_semantic_error(
            lambda: ModelOperatorParameterRole(
                name="",
                display_name="Role",
                description="role description",
            )
        )
    ) == "parameter role name must be nonempty"
    assert str(
        capture_semantic_error(lambda: _semantic(kind=""))
    ) == "operator kind must be nonempty"
    assert str(
        capture_semantic_error(
            lambda: DescriptorAxis(
                name="state",
                display_name="State",
                values=(
                    DescriptorAxisValue(value="fixed", display_name="Fixed"),
                    DescriptorAxisValue(value="fixed", display_name="Fixed"),
                ),
            )
        )
    ) == "state values must be unique"


def _semantic(
    *,
    kind: str = "test-operator",
    syntax_aliases: tuple[str, ...] = ("test-alias",),
) -> ModelOperatorSemantic:
    return ModelOperatorSemantic(
        kind=kind,
        display_name="Test operator",
        tensor_relation="identity",
        state="fixed",
        support="pointwise",
        projection_law="none",
        aggregation_law="none",
        parameter_sharing="none",
        shape_law="preserve-shape",
        cost_law="zero-arithmetic",
        syntax_aliases=syntax_aliases,
    )


def capture_semantic_error(action: Callable[[], object]) -> OperatorSemanticValidationError:
    try:
        action()
    except OperatorSemanticValidationError as error:
        return error
    raise AssertionError("expected OperatorSemanticValidationError")
