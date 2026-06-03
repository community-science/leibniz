"""Content digests for protocol records."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from leibniz.documents import ContentEncodingError, canonical_document_bytes

__all__ = [
    "ContentEncodingError",
    "ContentDigest",
]


@dataclass(frozen=True, slots=True)
class ContentDigest:
    """Digest of canonical bytes."""

    algorithm: str
    hex: str

    def __post_init__(self) -> None:
        if self.algorithm != "sha256":
            raise ContentEncodingError(f"unsupported digest algorithm: {self.algorithm}")
        if len(self.hex) != 64 or any(
            character not in "0123456789abcdef" for character in self.hex
        ):
            raise ContentEncodingError("sha256 digest must be 64 lowercase hexadecimal characters")

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.hex}"

    @classmethod
    def from_value(cls, value: object) -> ContentDigest:
        digest = hashlib.sha256(canonical_document_bytes(value)).hexdigest()
        return cls(algorithm="sha256", hex=digest)

    @classmethod
    def from_string(
        cls,
        value: object,
        *,
        field: str,
        error_type: type[ValueError] = ContentEncodingError,
    ) -> ContentDigest:
        """Parse an algorithm-qualified digest string from a protocol record."""

        if not isinstance(value, str):
            raise error_type(f"{field}: expected digest string")
        algorithm, separator, digest_hex = value.partition(":")
        if separator == "":
            raise error_type(f"{field}: expected algorithm:digest")
        try:
            return cls(algorithm=algorithm, hex=digest_hex)
        except ContentEncodingError as error:
            raise error_type(str(error)) from error
