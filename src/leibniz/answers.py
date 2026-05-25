"""Finite answer-space declarations."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz.identifiers import ProtocolIdentifier, require_unreleased_identifier
from leibniz.records import RecordSpec, RecordValidationError, required, validate_record

_ELEMENT_ID = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")

_ANSWER_ELEMENT_RECORD = RecordSpec(
    fields={
        "id": required("string"),
    }
)
_ANSWER_SPACE_RECORD = RecordSpec(
    fields={
        "id": required("identifier"),
        "elements": required("sequence", item=required("record", record=_ANSWER_ELEMENT_RECORD)),
    }
)
_ACCEPTED_EVENT_RECORD = RecordSpec(
    fields={
        "id": required("identifier"),
        "answer_space_id": required("identifier"),
        "elements": required("sequence", item=required("string")),
    }
)


class AnswerSpaceValidationError(ValueError):
    """Raised when an answer element or answer space is invalid."""


class AcceptedEventValidationError(ValueError):
    """Raised when an accepted event is invalid."""


@dataclass(frozen=True, slots=True)
class AnswerElement:
    """One possible answer inside a finite answer space."""

    id: str

    def __post_init__(self) -> None:
        if _ELEMENT_ID.fullmatch(self.id) is None:
            raise AnswerSpaceValidationError(f"invalid answer element id: {self.id!r}")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> AnswerElement:
        try:
            validated = validate_record(record, _ANSWER_ELEMENT_RECORD)
        except RecordValidationError as error:
            raise AnswerSpaceValidationError(str(error)) from error
        return cls(id=str(validated["id"]))

    def to_record(self) -> dict[str, object]:
        return {"id": self.id}


@dataclass(frozen=True, slots=True)
class AnswerSpace:
    """A finite set of possible answers."""

    id: ProtocolIdentifier
    elements: tuple[AnswerElement, ...]

    def __post_init__(self) -> None:
        try:
            require_unreleased_identifier(self.id)
        except ValueError as error:
            raise AnswerSpaceValidationError(str(error)) from error
        if not self.elements:
            raise AnswerSpaceValidationError("answer space must contain at least one element")
        element_ids = tuple(element.id for element in self.elements)
        if len(set(element_ids)) != len(element_ids):
            raise AnswerSpaceValidationError("answer element ids must be unique")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> AnswerSpace:
        try:
            validated = validate_record(record, _ANSWER_SPACE_RECORD)
        except RecordValidationError as error:
            raise AnswerSpaceValidationError(str(error)) from error
        elements = tuple(
            AnswerElement.from_record(_as_mapping(element, field="elements"))
            for element in _as_tuple(validated["elements"], field="elements")
        )
        identifier = _as_identifier(validated["id"], field="id")
        return cls(id=identifier, elements=elements)

    @property
    def element_ids(self) -> frozenset[str]:
        return frozenset(element.id for element in self.elements)

    def contains(self, element_id: str) -> bool:
        return element_id in self.element_ids

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "elements": [element.to_record() for element in self.elements],
        }


def answer_element(record: Mapping[str, object]) -> AnswerElement:
    return AnswerElement.from_record(record)


def answer_space(record: Mapping[str, object]) -> AnswerSpace:
    return AnswerSpace.from_record(record)


@dataclass(frozen=True, slots=True)
class AcceptedEvent:
    """A nonempty subset of a finite answer space."""

    id: ProtocolIdentifier
    answer_space_id: ProtocolIdentifier
    elements: frozenset[str]

    def __post_init__(self) -> None:
        try:
            require_unreleased_identifier(self.id)
        except ValueError as error:
            raise AcceptedEventValidationError(str(error)) from error
        if not self.elements:
            raise AcceptedEventValidationError("accepted event must contain at least one element")
        for element_id in self.elements:
            if _ELEMENT_ID.fullmatch(element_id) is None:
                raise AcceptedEventValidationError(
                    f"invalid accepted element id: {element_id!r}"
                )

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], *, answer_space: AnswerSpace
    ) -> AcceptedEvent:
        try:
            validated = validate_record(record, _ACCEPTED_EVENT_RECORD)
        except RecordValidationError as error:
            raise AcceptedEventValidationError(str(error)) from error

        identifier = _as_identifier(validated["id"], field="id")
        answer_space_id = _as_identifier(validated["answer_space_id"], field="answer_space_id")
        if answer_space_id != answer_space.id:
            raise AcceptedEventValidationError(
                f"answer_space_id {answer_space_id} does not match {answer_space.id}"
            )

        element_ids = tuple(
            str(element) for element in _as_tuple(validated["elements"], field="elements")
        )
        if len(set(element_ids)) != len(element_ids):
            raise AcceptedEventValidationError("accepted element ids must be unique")

        event = cls(
            id=identifier,
            answer_space_id=answer_space_id,
            elements=frozenset(element_ids),
        )
        unknown = tuple(
            element_id
            for element_id in sorted(event.elements)
            if not answer_space.contains(element_id)
        )
        if unknown:
            raise AcceptedEventValidationError(
                f"accepted elements are not in answer space: {', '.join(unknown)}"
            )
        return event

    def accepts(self, element_id: str) -> bool:
        return element_id in self.elements

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "answer_space_id": str(self.answer_space_id),
            "elements": sorted(self.elements),
        }


def accepted_event(record: Mapping[str, object], *, answer_space: AnswerSpace) -> AcceptedEvent:
    return AcceptedEvent.from_record(record, answer_space=answer_space)


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AnswerSpaceValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_tuple(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise AnswerSpaceValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise AnswerSpaceValidationError(f"{field}: expected parsed identifier")
    return value
