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
        "rank-collapse",
        "affine-readout",
    ]
    assert [record["alias"] for record in registry.syntax_alias_records()] == [
        "adaptive-pooling",
        "flatten",
        "dense",
    ]
    assert registry.semantic_for_alias("adaptive-pooling") == registry.operators[0]
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
            "name": "size",
            "display_name": "Output support size",
            "description": "extent of each aggregated output axis",
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
    ) == "operator syntax aliases must be unique"


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
