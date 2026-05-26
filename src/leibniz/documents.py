"""Document encoding boundary for artifact loaders and content identity."""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "ContentEncodingError",
    "canonical_document_bytes",
    "load_object_document",
]


class ContentEncodingError(ValueError):
    """Raised when a value cannot be encoded for stable content identity."""


def canonical_document_bytes(value: object) -> bytes:
    from leibniz._formats._json import canonical_json_bytes

    return canonical_json_bytes(value)


def load_object_document(data: bytes, *, description: str) -> Mapping[str, object]:
    from leibniz._formats._json import load_json_object_file

    return load_json_object_file(data, description=description)
