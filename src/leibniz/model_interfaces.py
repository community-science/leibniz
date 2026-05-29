"""Declared model prediction interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.outcomes import OutcomeSpace
from leibniz.prediction_spaces import (
    FiniteOutcomeSpace,
    PredictionSpaceValidationError,
)
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "ModelInterface",
    "ModelInterfaceDocument",
    "ModelInterfaceValidationError",
]

_PredictionKind: TypeAlias = Literal["direct-finite-probability-measure"]
_OutputEncoding: TypeAlias = Literal["probability-mass-sequence"]

_model_interface_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "prediction_space": FieldSpec(kind="record"),
        "prediction_kind": FieldSpec(
            kind="literal",
            literal="direct-finite-probability-measure",
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
    """A model-output contract over a prediction target space."""

    id: ProtocolIdentifier
    prediction_space: FiniteOutcomeSpace
    prediction_kind: _PredictionKind = "direct-finite-probability-measure"
    output_encoding: _OutputEncoding = "probability-mass-sequence"

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise ModelInterfaceValidationError(str(error)) from error
        if not str(self.id.name).startswith("model-interfaces."):
            raise ModelInterfaceValidationError("id must be a valid model interface id")
        if self.prediction_kind != "direct-finite-probability-measure":
            raise ModelInterfaceValidationError(
                f"unsupported prediction_kind: {self.prediction_kind}"
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
        source_space: Mapping[str, object] | None = None,
    ) -> ModelInterface:
        return cls(
            id=id,
            prediction_space=FiniteOutcomeSpace.from_outcome_space(
                outcome_space,
                source_space=source_space,
            ),
        )

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
        try:
            prediction_space = FiniteOutcomeSpace.from_record(
                _as_mapping(validated["prediction_space"], field="prediction_space")
            )
        except PredictionSpaceValidationError as error:
            raise ModelInterfaceValidationError(str(error)) from error
        interface = cls(
            id=_as_identifier(validated["id"], field="id"),
            prediction_space=prediction_space,
            prediction_kind=cast(_PredictionKind, validated["prediction_kind"]),
            output_encoding=cast(_OutputEncoding, validated["output_encoding"]),
        )
        interface.validate_outcome_space(outcome_space)
        return interface

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_outcome_space(self, outcome_space: OutcomeSpace) -> None:
        try:
            self.prediction_space.validate_outcome_space(outcome_space)
        except PredictionSpaceValidationError as error:
            raise ModelInterfaceValidationError(str(error)) from error

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "prediction_space": self.prediction_space.to_record(),
            "prediction_kind": self.prediction_kind,
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


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ModelInterfaceValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)
