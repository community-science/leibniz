"""Declared model prediction interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.outcomes import OutcomeSpace
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "ModelInterface",
    "ModelInterfaceDocument",
    "ModelInterfaceValidationError",
]

_PredictionSemantics: TypeAlias = Literal["finite-probability-measure"]
_OutputEncoding: TypeAlias = Literal["probability-mass-sequence"]

_model_interface_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "outcome_space_id": FieldSpec(kind="identifier"),
        "prediction_semantics": FieldSpec(
            kind="literal",
            literal="finite-probability-measure",
        ),
        "output_encoding": FieldSpec(
            kind="literal",
            literal="probability-mass-sequence",
        ),
    }
)


class ModelInterfaceValidationError(ValueError):
    """Raised when a model interface declaration is invalid."""


@dataclass(frozen=True, slots=True)
class ModelInterface:
    """A model-output contract over a public finite outcome space."""

    id: ProtocolIdentifier
    outcome_space_id: ProtocolIdentifier
    prediction_semantics: _PredictionSemantics = "finite-probability-measure"
    output_encoding: _OutputEncoding = "probability-mass-sequence"

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
            self.outcome_space_id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise ModelInterfaceValidationError(str(error)) from error
        if not str(self.id.name).startswith("model-interfaces."):
            raise ModelInterfaceValidationError("id must be a valid model interface id")
        if self.prediction_semantics != "finite-probability-measure":
            raise ModelInterfaceValidationError(
                f"unsupported prediction_semantics: {self.prediction_semantics}"
            )
        if self.output_encoding != "probability-mass-sequence":
            raise ModelInterfaceValidationError(
                f"unsupported output_encoding: {self.output_encoding}"
            )

    @classmethod
    def from_outcome_space(
        cls,
        *,
        id: ProtocolIdentifier,
        outcome_space: OutcomeSpace,
    ) -> ModelInterface:
        return cls(id=id, outcome_space_id=outcome_space.id)

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        outcome_space: OutcomeSpace,
    ) -> ModelInterface:
        try:
            validated = _model_interface_record.validate(record)
        except ValueError as error:
            raise ModelInterfaceValidationError(str(error)) from error
        interface = cls(
            id=_as_identifier(validated["id"], field="id"),
            outcome_space_id=_as_identifier(
                validated["outcome_space_id"],
                field="outcome_space_id",
            ),
            prediction_semantics=cast(
                _PredictionSemantics,
                validated["prediction_semantics"],
            ),
            output_encoding=cast(_OutputEncoding, validated["output_encoding"]),
        )
        interface.validate_outcome_space(outcome_space)
        return interface

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_outcome_space(self, outcome_space: OutcomeSpace) -> None:
        if self.outcome_space_id != outcome_space.id:
            raise ModelInterfaceValidationError(
                f"outcome_space_id {self.outcome_space_id} does not match {outcome_space.id}"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "outcome_space_id": str(self.outcome_space_id),
            "prediction_semantics": self.prediction_semantics,
            "output_encoding": self.output_encoding,
        }


@dataclass(frozen=True, slots=True)
class ModelInterfaceDocument:
    """A loaded model interface declaration and its canonical digest."""

    interface: ModelInterface
    digest: ContentDigest

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        outcome_space: OutcomeSpace,
    ) -> ModelInterfaceDocument:
        try:
            record = load_object_document(data, description="model interface document")
        except ContentEncodingError as error:
            raise ModelInterfaceValidationError(str(error)) from error
        interface = ModelInterface.from_record(record, outcome_space=outcome_space)
        return cls(interface=interface, digest=interface.digest)


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise ModelInterfaceValidationError(f"{field}: expected parsed identifier")
    return value
