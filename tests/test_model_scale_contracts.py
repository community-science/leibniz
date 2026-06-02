from collections.abc import Callable

from leibniz.model_scale_contracts import (
    ModelScaleContract,
    ModelScaleContractValidationError,
)


def test_model_scale_contract_declares_variable_shape_envelope() -> None:
    contract = ModelScaleContract.variable_input_shape(
        (1, 32, 32),
        minimum=32,
        axis_symbol="W",
        scale_axis_indices=(2,),
    )

    assert contract.accepts_scale(32)
    assert contract.accepts_scale(96)
    assert not contract.accepts_scale(31)
    assert contract.shape_for_scale(96) == (1, 32, 96)
    assert ModelScaleContract.from_record(contract.to_record()) == contract
    assert contract.to_record() == {
        "kind": "positive-variable-shape-envelope",
        "axis_symbol": "W",
        "anchor_shape": [1, 32, 32],
        "axes": [
            {"index": 0, "kind": "fixed", "size": 1},
            {"index": 1, "kind": "fixed", "size": 32},
            {"index": 2, "kind": "scaled", "symbol": "W"},
        ],
        "scale_domain": {"minimum": 32},
    }


def test_model_scale_contract_declares_fixed_shape_special_case() -> None:
    contract = ModelScaleContract.fixed_input_shape(
        (1, 32, 32),
        axis_symbol="W",
        scale_axis_indices=(2,),
    )

    assert contract.accepts_scale(32)
    assert not contract.accepts_scale(64)
    assert contract.to_record()["scale_domain"] == {"minimum": 32, "maximum": 32}


def test_model_scale_contract_rejects_invalid_axes() -> None:
    assert str(
        capture_scale_contract_error(
            lambda: ModelScaleContract.variable_input_shape(
                (1, 32, 32),
                minimum=64,
                axis_symbol="W",
                scale_axis_indices=(2,),
            )
        )
    ) == "scaled anchor axes must be at least minimum"

    assert str(
        capture_scale_contract_error(
            lambda: ModelScaleContract.variable_input_shape(
                (1, 32, 32),
                minimum=32,
                scale_axis_indices=(3,),
            )
        )
    ) == "scale axis index is outside tensor rank"


def capture_scale_contract_error(
    action: Callable[[], object],
) -> ModelScaleContractValidationError:
    try:
        action()
    except ModelScaleContractValidationError as error:
        return error
    raise AssertionError("expected ModelScaleContractValidationError")
