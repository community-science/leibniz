from collections.abc import Callable

from leibniz.answers import (
    AnswerElement,
    AnswerSpace,
    AnswerSpaceValidationError,
    answer_element,
    answer_space,
)
from leibniz.identifiers import ProtocolIdentifier


def test_answer_element_parses_record() -> None:
    element = answer_element({"id": "yes"})

    assert element == AnswerElement(id="yes")
    assert element.to_record() == {"id": "yes"}


def test_answer_element_rejects_invalid_id_and_unknown_fields() -> None:
    assert str(capture_answer_error(lambda: answer_element({"id": "Yes"}))) == (
        "invalid answer element id: 'Yes'"
    )
    assert str(capture_answer_error(lambda: answer_element({"id": "yes", "label": "Yes"}))) == (
        "label: unknown field"
    )


def test_answer_space_parses_nonempty_finite_space() -> None:
    space = answer_space(
        {
            "id": "core.boolean-answer@0.1.0",
            "elements": [{"id": "yes"}, {"id": "no"}],
        }
    )

    assert space == AnswerSpace(
        id=ProtocolIdentifier.parse("core.boolean-answer@0.1.0"),
        elements=(AnswerElement("yes"), AnswerElement("no")),
    )
    assert space.element_ids == frozenset({"yes", "no"})
    assert space.contains("yes")
    assert not space.contains("maybe")
    assert space.to_record() == {
        "id": "core.boolean-answer@0.1.0",
        "elements": [{"id": "yes"}, {"id": "no"}],
    }


def test_answer_space_rejects_empty_elements() -> None:
    error = capture_answer_error(
        lambda: answer_space({"id": "core.empty-answer@0.1.0", "elements": []})
    )

    assert str(error) == "answer space must contain at least one element"


def test_answer_space_rejects_duplicate_element_ids() -> None:
    error = capture_answer_error(
        lambda: answer_space(
            {
                "id": "core.duplicate-answer@0.1.0",
                "elements": [{"id": "yes"}, {"id": "yes"}],
            }
        )
    )

    assert str(error) == "answer element ids must be unique"


def test_answer_space_rejects_released_identifier() -> None:
    error = capture_answer_error(
        lambda: answer_space({"id": "core.boolean-answer@1.0.0", "elements": [{"id": "yes"}]})
    )

    assert str(error) == (
        "identifier must use a pre-1.0.0 version before release policy exists: "
        "core.boolean-answer@1.0.0"
    )


def test_answer_space_rejects_malformed_records() -> None:
    assert str(capture_answer_error(lambda: answer_space({"elements": [{"id": "yes"}]}))) == (
        "id: missing required field"
    )
    assert str(
        capture_answer_error(
            lambda: answer_space(
                {
                    "id": "core.boolean-answer@0.1.0",
                    "elements": [{"id": "yes", "text": "Yes"}],
                }
            )
        )
    ) == "elements.0.text: unknown field"


def capture_answer_error(call: Callable[[], object]) -> AnswerSpaceValidationError:
    try:
        call()
    except AnswerSpaceValidationError as error:
        return error
    raise AssertionError("expected AnswerSpaceValidationError")
