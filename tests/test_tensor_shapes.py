from collections.abc import Callable

from leibniz.tensor_shapes import TensorShape, TensorShapeValidationError


def test_tensor_shape_canonicalizes_existing_shape_list_records() -> None:
    shape = TensorShape.from_record([1, 32, 32], field="input_shape")

    assert shape.axes == (1, 32, 32)
    assert shape.rank == 3
    assert shape.element_count == 1024
    assert shape.to_record() == [1, 32, 32]


def test_tensor_shape_rejects_invalid_axes() -> None:
    assert str(capture_shape_error(lambda: TensorShape.from_record([], field="input_shape"))) == (
        "input_shape must contain at least one axis"
    )
    assert str(
        capture_shape_error(lambda: TensorShape.from_record([1, 0, 32], field="input_shape"))
    ) == "input_shape axes must be positive integers"
    assert str(
        capture_shape_error(lambda: TensorShape.from_record([1, True], field="input_shape"))
    ) == "input_shape axes must be positive integers"
    assert str(
        capture_shape_error(lambda: TensorShape.from_record({"axis": 1}, field="input_shape"))
    ) == "input_shape: expected shape sequence"


def capture_shape_error(action: Callable[[], object]) -> TensorShapeValidationError:
    try:
        action()
    except TensorShapeValidationError as error:
        return error
    raise AssertionError("expected TensorShapeValidationError")
