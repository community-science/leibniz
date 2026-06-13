"""Durable references to reproducible field-valued observations.

Large field values are referenced by a reproducible materialization record and
verified by a digest over the producing backend's flattened host values. The
digest is backend-local and bitwise: cross-backend agreement is a numerical
tolerance contract declared by the benchmark, never a digest comparison.
"""

from __future__ import annotations

import hashlib
import math
import sys
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from leibniz.content import ContentDigest

__all__ = [
    "FieldArtifactDType",
    "FieldArtifactError",
    "FieldArtifactReference",
    "field_content_digest",
    "verify_field_artifact",
]

FieldArtifactDType = Literal["float32", "float64"]


class FieldArtifactError(ValueError):
    """Raised when a field artifact reference is invalid."""


@dataclass(frozen=True, slots=True)
class FieldArtifactReference:
    """A reproducible-by-seed reference to a field-valued artifact."""

    shape: tuple[int, ...]
    dtype: FieldArtifactDType
    content_digest: ContentDigest
    materialization: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.shape or any(type(size) is not int or size < 1 for size in self.shape):
            raise FieldArtifactError("field artifact shape must be nonempty positive integers")
        _validate_dtype(self.dtype)
        materialization = _validate_materialization(self.materialization)
        object.__setattr__(self, "materialization", materialization)

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> FieldArtifactReference:
        shape_value = record.get("shape")
        if not isinstance(shape_value, Sequence) or isinstance(shape_value, str | bytes):
            raise FieldArtifactError("field artifact shape must be a sequence")
        shape = tuple(cast(Sequence[object], shape_value))
        if any(type(size) is not int for size in shape):
            raise FieldArtifactError("field artifact shape must contain integers")
        dtype = record.get("dtype")
        if type(dtype) is not str:
            raise FieldArtifactError("field artifact dtype must be a string")
        materialization = record.get("materialization")
        if not isinstance(materialization, Mapping):
            raise FieldArtifactError("field artifact materialization must be a mapping")
        return cls(
            shape=cast(tuple[int, ...], shape),
            dtype=cast(FieldArtifactDType, dtype),
            content_digest=ContentDigest.from_string(
                record.get("content_digest"),
                field="field artifact content_digest",
                error_type=FieldArtifactError,
            ),
            materialization=cast(Mapping[str, object], materialization),
        )

    def to_record(self) -> dict[str, object]:
        """Return a record for this field artifact reference."""

        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "content_digest": str(self.content_digest),
            "materialization": dict(self.materialization),
        }


def field_content_digest(
    values: Sequence[float],
    *,
    dtype: FieldArtifactDType,
) -> ContentDigest:
    """Return the raw little-endian digest for flattened field values."""

    _validate_dtype(dtype)
    typecode = "f" if dtype == "float32" else "d"
    buffer = array(typecode, (float(value) for value in values))
    if sys.byteorder == "big":
        buffer.byteswap()
    return ContentDigest(
        algorithm="sha256",
        hex=hashlib.sha256(buffer.tobytes()).hexdigest(),
    )


def verify_field_artifact(
    reference: FieldArtifactReference,
    values: Sequence[float],
) -> bool:
    """Return whether values exactly match a referenced field artifact digest."""

    if len(values) != math.prod(reference.shape):
        return False
    return field_content_digest(values, dtype=reference.dtype) == reference.content_digest


def _validate_dtype(dtype: str) -> None:
    if dtype not in {"float32", "float64"}:
        raise FieldArtifactError("field artifact dtype must be float32 or float64")


def _validate_materialization(materialization: Mapping[str, object]) -> dict[str, object]:
    for key in materialization:
        if type(key) is not str:
            raise FieldArtifactError("field artifact materialization keys must be strings")
    seed = materialization.get("seed")
    if type(seed) is not int:
        raise FieldArtifactError("field artifact materialization requires integer seed")
    generator = materialization.get("generator")
    if not isinstance(generator, Mapping) or not generator:
        raise FieldArtifactError(
            "field artifact materialization requires a nonempty generator mapping"
        )
    for key in cast(Mapping[object, object], generator):
        if type(key) is not str:
            raise FieldArtifactError("field artifact generator keys must be strings")
    return dict(materialization)
