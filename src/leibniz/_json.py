"""JSON encoding helpers used by file and content boundaries."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import TypeAlias, cast

_JsonScalar: TypeAlias = None | bool | int | float | str
_JsonValue: TypeAlias = _JsonScalar | Mapping[str, "_JsonValue"] | Sequence["_JsonValue"]
_JsonObject: TypeAlias = Mapping[str, _JsonValue]


class ContentEncodingError(ValueError):
    """Raised when a value cannot be encoded for stable content identity."""


def canonical_json_bytes(value: object) -> bytes:
    normalized = _normalize_json(value, path=())
    return json.dumps(
        normalized,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def load_json_object_file(data: bytes, *, description: str = "JSON file") -> _JsonObject:
    try:
        value = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ContentEncodingError(f"{description} must be UTF-8") from error
    except json.JSONDecodeError as error:
        raise ContentEncodingError(f"invalid {description}: {error.msg}") from error
    if not isinstance(value, Mapping):
        raise ContentEncodingError(f"{description} must contain an object")

    normalized = _normalize_json(cast(Mapping[str, object], value), path=())
    if not isinstance(normalized, Mapping):
        raise ContentEncodingError(f"{description} must contain an object")
    return cast(_JsonObject, normalized)


def _normalize_json(value: object, *, path: tuple[str, ...]) -> _JsonValue:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ContentEncodingError(f"{_format_path(path)}: nonfinite number")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, _JsonValue] = {}
        for key, item in cast(Mapping[object, object], value).items():
            if not isinstance(key, str):
                raise ContentEncodingError(f"{_format_path(path)}: object key must be string")
            normalized[key] = _normalize_json(item, path=(*path, key))
        return normalized
    if isinstance(value, bytes | bytearray):
        raise ContentEncodingError(f"{_format_path(path)}: unsupported JSON value")
    if isinstance(value, Sequence):
        return [
            _normalize_json(item, path=(*path, str(index)))
            for index, item in enumerate(cast(Sequence[_JsonValue], value))
        ]
    raise ContentEncodingError(f"{_format_path(path)}: unsupported JSON value")


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "<value>"
