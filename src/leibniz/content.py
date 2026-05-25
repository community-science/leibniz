"""Canonical JSON bytes and content digests."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | Mapping[str, "JsonValue"] | Sequence["JsonValue"]

__all__ = [
    "CanonicalJson",
    "CanonicalJsonError",
    "ContentDigest",
    "JsonDocument",
    "JsonScalar",
    "JsonValue",
]


class CanonicalJsonError(ValueError):
    """Raised when a value cannot be represented as canonical JSON."""


@dataclass(frozen=True, slots=True)
class CanonicalJson:
    """Canonical UTF-8 JSON bytes."""

    data: bytes

    @classmethod
    def from_value(cls, value: object) -> CanonicalJson:
        normalized = _normalize_json(value, path=())
        data = json.dumps(
            normalized,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return cls(data=data)

    def __bytes__(self) -> bytes:
        return self.data

    def digest(self) -> ContentDigest:
        return ContentDigest.from_canonical_json(self)


@dataclass(frozen=True, slots=True)
class ContentDigest:
    """Digest of canonical bytes."""

    algorithm: str
    hex: str

    def __post_init__(self) -> None:
        if self.algorithm != "sha256":
            raise CanonicalJsonError(f"unsupported digest algorithm: {self.algorithm}")
        if len(self.hex) != 64 or any(
            character not in "0123456789abcdef" for character in self.hex
        ):
            raise CanonicalJsonError("sha256 digest must be 64 lowercase hexadecimal characters")

    def __str__(self) -> str:
        return f"{self.algorithm}:{self.hex}"

    @classmethod
    def from_value(cls, value: object) -> ContentDigest:
        return cls.from_canonical_json(CanonicalJson.from_value(value))

    @classmethod
    def from_canonical_json(cls, canonical_json: CanonicalJson) -> ContentDigest:
        digest = hashlib.sha256(canonical_json.data).hexdigest()
        return cls(algorithm="sha256", hex=digest)


@dataclass(frozen=True, slots=True)
class JsonDocument:
    """A loaded JSON object and the digest of its canonical value."""

    value: Mapping[str, JsonValue]
    digest: ContentDigest

    @classmethod
    def from_json_bytes(cls, data: bytes) -> JsonDocument:
        try:
            value = json.loads(data.decode("utf-8"))
        except UnicodeDecodeError as error:
            raise CanonicalJsonError("JSON document must be UTF-8") from error
        except json.JSONDecodeError as error:
            raise CanonicalJsonError(f"invalid JSON document: {error.msg}") from error
        if not isinstance(value, Mapping):
            raise CanonicalJsonError("JSON document must be an object")

        mapping = cast(Mapping[str, object], value)
        normalized = _normalize_json(mapping, path=())
        if not isinstance(normalized, Mapping):
            raise CanonicalJsonError("JSON document must be an object")
        document = cast(Mapping[str, JsonValue], normalized)
        return cls(value=document, digest=ContentDigest.from_value(document))


def _normalize_json(value: object, *, path: tuple[str, ...]) -> JsonValue:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJsonError(f"{_format_path(path)}: nonfinite number")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        mapping = cast(Mapping[object, object], value)
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise CanonicalJsonError(f"{_format_path(path)}: object key must be string")
            normalized[key] = _normalize_json(item, path=(*path, key))
        return normalized
    if isinstance(value, bytes | bytearray):
        raise CanonicalJsonError(f"{_format_path(path)}: unsupported JSON value")
    if isinstance(value, Sequence):
        return [
            _normalize_json(item, path=(*path, str(index)))
            for index, item in enumerate(cast(Sequence[JsonValue], value))
        ]
    raise CanonicalJsonError(f"{_format_path(path)}: unsupported JSON value")


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "<value>"
