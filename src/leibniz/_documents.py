"""Private compatibility shim for document helpers."""

from __future__ import annotations

from leibniz.documents import (
    ContentEncodingError,
    canonical_document_bytes,
    load_object_document,
)

__all__ = [
    "ContentEncodingError",
    "canonical_document_bytes",
    "load_object_document",
]
