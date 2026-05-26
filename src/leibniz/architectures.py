"""Architecture manifests as declarative model-structure records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz._documents import ContentEncodingError, load_object_document
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
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
class ArchitectureLayer:
    """One opaque model-structure layer record."""

    kind: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.kind:
            raise ArchitectureManifestValidationError("layer kind must be nonempty")
        try:
            ContentDigest.from_value(self.to_record())
        except ContentEncodingError as error:
            raise ArchitectureManifestValidationError(str(error)) from error

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ArchitectureLayer:
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


@dataclass(frozen=True, slots=True)
class ArchitectureManifest:
    """A content-addressed declarative model-structure manifest."""

    id: ProtocolIdentifier
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    layers: tuple[ArchitectureLayer, ...]

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
        except ValueError as error:
            raise ArchitectureManifestValidationError(str(error)) from error
        derived_id = _architecture_id(
            {
                "input_shape": list(input_shape),
                "output_shape": list(output_shape),
                "layers": [layer.to_record() for layer in layers],
            }
        )
        identifier = validated.get("id", derived_id)
        return cls(
            id=_as_identifier(identifier, field="id"),
            input_shape=input_shape,
            output_shape=output_shape,
            layers=layers,
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

    def _content_record(self) -> dict[str, object]:
        return {
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "layers": [layer.to_record() for layer in self.layers],
        }


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
    axes: list[int] = []
    for axis in _as_sequence(value, field=field):
        if not isinstance(axis, int) or isinstance(axis, bool):
            raise ArchitectureManifestValidationError(f"{field}: expected parsed integer")
        axes.append(axis)
    shape = tuple(axes)
    _require_positive_shape(shape, field=field)
    return shape


def _require_positive_shape(shape: tuple[int, ...], *, field: str) -> None:
    if not shape:
        raise ArchitectureManifestValidationError(f"{field} must contain at least one axis")
    if any(axis <= 0 for axis in shape):
        raise ArchitectureManifestValidationError(f"{field} axes must be positive integers")
