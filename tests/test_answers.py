from collections.abc import Callable

from leibniz.answers import (
    AcceptedEvent,
    AcceptedEventValidationError,
    AnswerElement,
    AnswerSpace,
    AnswerSpaceValidationError,
    accepted_event,
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


def test_accepted_event_parses_nonempty_subset() -> None:
    space = answer_space(
        {
            "id": "core.boolean-answer@0.1.0",
            "elements": [{"id": "yes"}, {"id": "no"}, {"id": "maybe"}],
        }
    )

    event = accepted_event(
        {
            "id": "core.boolean-accepted@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "elements": ["maybe", "yes"],
        },
        answer_space=space,
    )

    assert event == AcceptedEvent(
        id=ProtocolIdentifier.parse("core.boolean-accepted@0.1.0"),
        answer_space_id=ProtocolIdentifier.parse("core.boolean-answer@0.1.0"),
        elements=frozenset({"maybe", "yes"}),
    )
    assert event.accepts("yes")
    assert not event.accepts("no")
    assert event.to_record() == {
        "id": "core.boolean-accepted@0.1.0",
        "answer_space_id": "core.boolean-answer@0.1.0",
        "elements": ["maybe", "yes"],
    }


def test_accepted_event_rejects_empty_event() -> None:
    space = answer_space({"id": "core.boolean-answer@0.1.0", "elements": [{"id": "yes"}]})

    error = capture_event_error(
        lambda: accepted_event(
            {
                "id": "core.boolean-accepted@0.1.0",
                "answer_space_id": "core.boolean-answer@0.1.0",
                "elements": [],
            },
            answer_space=space,
        )
    )

    assert str(error) == "accepted event must contain at least one element"


def test_accepted_event_rejects_unknown_elements() -> None:
    space = answer_space({"id": "core.boolean-answer@0.1.0", "elements": [{"id": "yes"}]})

    error = capture_event_error(
        lambda: accepted_event(
            {
                "id": "core.boolean-accepted@0.1.0",
                "answer_space_id": "core.boolean-answer@0.1.0",
                "elements": ["maybe"],
            },
            answer_space=space,
        )
    )

    assert str(error) == "accepted elements are not in answer space: maybe"


def test_accepted_event_rejects_mismatched_answer_space() -> None:
    space = answer_space({"id": "core.boolean-answer@0.1.0", "elements": [{"id": "yes"}]})

    error = capture_event_error(
        lambda: accepted_event(
            {
                "id": "core.boolean-accepted@0.1.0",
                "answer_space_id": "core.other-answer@0.1.0",
                "elements": ["yes"],
            },
            answer_space=space,
        )
    )

    assert str(error) == (
        "answer_space_id core.other-answer@0.1.0 does not match core.boolean-answer@0.1.0"
    )


def test_accepted_event_rejects_invalid_element_ids_and_malformed_records() -> None:
    space = answer_space({"id": "core.boolean-answer@0.1.0", "elements": [{"id": "yes"}]})

    assert str(
        capture_event_error(
            lambda: accepted_event(
                {
                    "id": "core.boolean-accepted@0.1.0",
                    "answer_space_id": "core.boolean-answer@0.1.0",
                    "elements": ["Yes"],
                },
                answer_space=space,
            )
        )
    ) == "invalid accepted element id: 'Yes'"
    assert str(
        capture_event_error(
            lambda: accepted_event(
                {
                    "id": "core.boolean-accepted@0.1.0",
                    "answer_space_id": "core.boolean-answer@0.1.0",
                    "elements": ["yes"],
                    "extra": True,
                },
                answer_space=space,
            )
        )
    ) == "extra: unknown field"


def test_accepted_event_rejects_duplicate_elements() -> None:
    space = answer_space({"id": "core.boolean-answer@0.1.0", "elements": [{"id": "yes"}]})

    error = capture_event_error(
        lambda: accepted_event(
            {
                "id": "core.boolean-accepted@0.1.0",
                "answer_space_id": "core.boolean-answer@0.1.0",
                "elements": ["yes", "yes"],
            },
            answer_space=space,
        )
    )

    assert str(error) == "accepted element ids must be unique"


def test_accepted_event_rejects_released_identifier() -> None:
    space = answer_space({"id": "core.boolean-answer@0.1.0", "elements": [{"id": "yes"}]})

    error = capture_event_error(
        lambda: accepted_event(
            {
                "id": "core.boolean-accepted@1.0.0",
                "answer_space_id": "core.boolean-answer@0.1.0",
                "elements": ["yes"],
            },
            answer_space=space,
        )
    )

    assert str(error) == (
        "identifier must use a pre-1.0.0 version before release policy exists: "
        "core.boolean-accepted@1.0.0"
    )


def capture_answer_error(call: Callable[[], object]) -> AnswerSpaceValidationError:
    try:
        call()
    except AnswerSpaceValidationError as error:
        return error
    raise AssertionError("expected AnswerSpaceValidationError")


def capture_event_error(call: Callable[[], object]) -> AcceptedEventValidationError:
    try:
        call()
    except AcceptedEventValidationError as error:
        return error
    raise AssertionError("expected AcceptedEventValidationError")
