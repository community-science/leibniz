"""Architecture manifests as declarative model-structure records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.model_scale_contracts import (
    ModelScaleContract,
    ModelScaleContractValidationError,
)
from leibniz.records import FieldSpec, RecordSpec
from leibniz.tensor_shapes import TensorShape, TensorShapeValidationError

__all__ = [
    "ArchitectureComponent",
    "ArchitectureLayer",
    "ArchitectureManifest",
    "ArchitectureManifestDocument",
    "ArchitectureManifestValidationError",
]

_architecture_manifest_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier", required=False),
        "input_shape": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="integer"),
        ),
        "output_shape": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="integer"),
        ),
        "layers": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "model_scale_contract": FieldSpec(kind="record", required=False),
    }
)
_architecture_layer_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="string"),
        "parameters": FieldSpec(kind="record", required=False),
    }
)


class ArchitectureManifestValidationError(ValueError):
    """Raised when an architecture manifest is invalid."""


@dataclass(frozen=True, slots=True)
class ArchitectureComponent:
    """One opaque model-structure component record."""

    kind: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.kind:
            raise ArchitectureManifestValidationError("component kind must be nonempty")
        try:
            ContentDigest.from_value(self.to_record())
        except ContentEncodingError as error:
            raise ArchitectureManifestValidationError(str(error)) from error

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ArchitectureComponent:
        try:
            validated = _architecture_layer_record.validate(record)
        except ValueError as error:
            raise ArchitectureManifestValidationError(str(error)) from error
        return cls(
            kind=str(validated["kind"]),
            parameters=_as_mapping(validated.get("parameters", {}), field="parameters"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "parameters": dict(self.parameters),
        }


ArchitectureLayer = ArchitectureComponent


@dataclass(frozen=True, slots=True)
class ArchitectureManifest:
    """A content-addressed declarative model-structure manifest."""

    id: ProtocolIdentifier
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    layers: tuple[ArchitectureLayer, ...]
    model_scale_contract: ModelScaleContract | None = None

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise ArchitectureManifestValidationError(str(error)) from error
        if self.id != self.derived_id():
            raise ArchitectureManifestValidationError(
                "id must be derived from architecture content"
            )
        _require_positive_shape(self.input_shape, field="input_shape")
        _require_positive_shape(self.output_shape, field="output_shape")
        if not self.layers:
            raise ArchitectureManifestValidationError("layers must contain at least one layer")
        if self.model_scale_contract is not None and (
            self.model_scale_contract.anchor_shape != self.input_shape
        ):
            raise ArchitectureManifestValidationError(
                "model_scale_contract anchor_shape must match input_shape"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ArchitectureManifest:
        try:
            validated = _architecture_manifest_record.validate(record)
            input_shape = _as_shape(validated["input_shape"], field="input_shape")
            output_shape = _as_shape(validated["output_shape"], field="output_shape")
            layers = tuple(
                ArchitectureLayer.from_record(_as_mapping(layer, field="layers"))
                for layer in _as_sequence(validated["layers"], field="layers")
            )
            scale_contract = _optional_scale_contract(
                validated.get("model_scale_contract")
            )
        except ValueError as error:
            raise ArchitectureManifestValidationError(str(error)) from error
        content_record = _architecture_content_record(
            input_shape=input_shape,
            output_shape=output_shape,
            layers=layers,
            model_scale_contract=scale_contract,
        )
        derived_id = _architecture_id(content_record)
        identifier = validated.get("id", derived_id)
        return cls(
            id=_as_identifier(identifier, field="id"),
            input_shape=input_shape,
            output_shape=output_shape,
            layers=layers,
            model_scale_contract=scale_contract,
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def derived_id(self) -> ProtocolIdentifier:
        return _architecture_id(self._content_record())

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            **self._content_record(),
        }

    @property
    def components(self) -> tuple[ArchitectureComponent, ...]:
        """Return the model-structure components in manifest order."""

        return self.layers

    def _content_record(self) -> dict[str, object]:
        return _architecture_content_record(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            layers=self.layers,
            model_scale_contract=self.model_scale_contract,
        )


@dataclass(frozen=True, slots=True)
class ArchitectureManifestDocument:
    """A loaded architecture manifest and the digest of its canonical record."""

    manifest: ArchitectureManifest
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> ArchitectureManifestDocument:
        try:
            record = load_object_document(data, description="architecture manifest document")
        except ContentEncodingError as error:
            raise ArchitectureManifestValidationError(str(error)) from error
        manifest = ArchitectureManifest.from_record(record)
        return cls(manifest=manifest, digest=manifest.digest)


def _architecture_id(content_record: Mapping[str, object]) -> ProtocolIdentifier:
    digest = ContentDigest.from_value(content_record)
    return ProtocolIdentifier.parse(f"architecture.sha-{digest.hex}@0.1.0")


def _architecture_content_record(
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    layers: tuple[ArchitectureLayer, ...],
    model_scale_contract: ModelScaleContract | None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "layers": [layer.to_record() for layer in layers],
    }
    if model_scale_contract is not None:
        record["model_scale_contract"] = model_scale_contract.to_record()
    return record


def _optional_scale_contract(value: object) -> ModelScaleContract | None:
    if value is None:
        return None
    try:
        return ModelScaleContract.from_record(_as_mapping(value, field="model_scale_contract"))
    except ModelScaleContractValidationError as error:
        raise ArchitectureManifestValidationError(str(error)) from error


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise ArchitectureManifestValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ArchitectureManifestValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ArchitectureManifestValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_shape(value: object, *, field: str) -> tuple[int, ...]:
    try:
        return TensorShape.from_record(_as_sequence(value, field=field), field=field).axes
    except TensorShapeValidationError as error:
        raise ArchitectureManifestValidationError(str(error)) from error


def _require_positive_shape(shape: tuple[int, ...], *, field: str) -> None:
    try:
        TensorShape.from_axes(shape, field=field)
    except TensorShapeValidationError as error:
        raise ArchitectureManifestValidationError(str(error)) from error
