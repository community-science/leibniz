"""Declared model prediction interfaces."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import IdentifierSyntaxError, ProtocolIdentifier
from leibniz.outcomes import OutcomeSpace
from leibniz.prediction_results import (
    PredictionResultContract,
    PredictionResultValidationError,
)
from leibniz.prediction_spaces import (
    FiniteOutcomeSpace,
    FiniteTokenSequenceSpace,
    PredictionSpace,
    PredictionSpaceValidationError,
    RealVectorSpace,
    parse_prediction_space,
)
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec

__all__ = [
    "ModelInterface",
    "ModelInterfaceDocument",
    "ModelInterfaceValidationError",
]

_PredictionKind: TypeAlias = Literal[
    "direct-finite-probability-measure",
    "autoregressive-finite-token-sequence",
    "direct-real-vector",
]
_OutputEncoding: TypeAlias = Literal[
    "probability-mass-sequence",
    "sequence-probability",
    "coordinate-sequence",
]

_model_interface_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "prediction_space": FieldSpec(kind="record"),
        "prediction_kind": FieldSpec(kind="string"),
        "output_encoding": FieldSpec(kind="string"),
    }
)


class ModelInterfaceValidationError(ValueError):
    """Raised when a model interface declaration is invalid."""


_record = RecordExtractor(ModelInterfaceValidationError)


@dataclass(frozen=True, slots=True)
class ModelInterface:
    """A model-output contract over a prediction target space."""

    id: ProtocolIdentifier
    prediction_space: PredictionSpace
    prediction_kind: _PredictionKind = "direct-finite-probability-measure"
    output_encoding: _OutputEncoding = "probability-mass-sequence"

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except IdentifierSyntaxError as error:
            raise ModelInterfaceValidationError(str(error)) from error
        if not str(self.id.name).startswith("model-interfaces."):
            raise ModelInterfaceValidationError("id must be a valid model interface id")
        if (
            self.prediction_kind == "direct-finite-probability-measure"
            and self.output_encoding == "probability-mass-sequence"
            and isinstance(self.prediction_space, FiniteOutcomeSpace)
        ):
            return
        if (
            self.prediction_kind == "autoregressive-finite-token-sequence"
            and self.output_encoding == "sequence-probability"
            and isinstance(self.prediction_space, FiniteTokenSequenceSpace)
            and self.prediction_space.sequence_boundary == "eos-terminated"
        ):
            return
        if (
            self.prediction_kind == "direct-real-vector"
            and self.output_encoding == "coordinate-sequence"
            and isinstance(self.prediction_space, RealVectorSpace)
        ):
            return
        raise ModelInterfaceValidationError(
            "model interface must pair finite probability measures with finite outcome "
            "spaces, or autoregressive sequence probabilities with eos-terminated "
            "finite token sequence spaces, or direct real-vector outputs with real vector "
            "spaces"
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
    def from_real_vector_space(
        cls,
        *,
        id: ProtocolIdentifier,
        dimension: int,
        coordinate_name: str = "value",
    ) -> ModelInterface:
        return cls(
            id=id,
            prediction_space=RealVectorSpace(
                dimension=dimension,
                coordinate_name=coordinate_name,
            ),
            prediction_kind="direct-real-vector",
            output_encoding="coordinate-sequence",
        )

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        outcome_space: OutcomeSpace | None = None,
    ) -> ModelInterface:
        try:
            validated = _model_interface_record.validate(record)
        except ValueError as error:
            raise ModelInterfaceValidationError(str(error)) from error
        try:
            prediction_space = parse_prediction_space(
                _record.mapping(validated["prediction_space"], "prediction_space")
            )
        except PredictionSpaceValidationError as error:
            raise ModelInterfaceValidationError(str(error)) from error
        if (
            validated["prediction_kind"] == "direct-finite-probability-measure"
            and not isinstance(prediction_space, FiniteOutcomeSpace)
        ):
            raise ModelInterfaceValidationError(
                "direct-finite-probability-measure requires finite-outcome-space prediction_space"
            )
        if (
            validated["prediction_kind"] == "autoregressive-finite-token-sequence"
            and not isinstance(prediction_space, FiniteTokenSequenceSpace)
        ):
            raise ModelInterfaceValidationError(
                "autoregressive-finite-token-sequence requires finite-token-sequence "
                "prediction_space"
            )
        if validated["prediction_kind"] == "direct-real-vector" and not isinstance(
            prediction_space,
            RealVectorSpace,
        ):
            raise ModelInterfaceValidationError(
                "direct-real-vector requires real-vector prediction_space"
            )
        interface = cls(
            id=_record.identifier(validated["id"], "id"),
            prediction_space=prediction_space,
            prediction_kind=cast(_PredictionKind, validated["prediction_kind"]),
            output_encoding=cast(_OutputEncoding, validated["output_encoding"]),
        )
        if isinstance(interface.prediction_space, FiniteOutcomeSpace):
            if outcome_space is None:
                raise ModelInterfaceValidationError(
                    "finite outcome model interfaces require outcome_space"
                )
            interface.validate_outcome_space(outcome_space)
        return interface

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_outcome_space(self, outcome_space: OutcomeSpace) -> None:
        if not isinstance(self.prediction_space, FiniteOutcomeSpace):
            raise ModelInterfaceValidationError(
                "sequence model interfaces do not validate against finite outcome spaces"
            )
        try:
            self.prediction_space.validate_outcome_space(outcome_space)
        except PredictionSpaceValidationError as error:
            raise ModelInterfaceValidationError(str(error)) from error

    def validate_prediction_result(
        self,
        prediction: object,
    ) -> None:
        try:
            PredictionResultContract.from_prediction(prediction).require_matches(
                prediction_space=self.prediction_space,
                prediction_kind=self.prediction_kind,
                output_encoding=self.output_encoding,
            )
        except PredictionResultValidationError as error:
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
