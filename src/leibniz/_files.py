"""File-format helpers used by protocol artifact loaders."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TypeAlias, cast

from leibniz.content import CanonicalJsonError

_JsonScalar: TypeAlias = None | bool | int | float | str
_JsonValue: TypeAlias = _JsonScalar | Mapping[str, "_JsonValue"] | Sequence["_JsonValue"]
_JsonObject: TypeAlias = Mapping[str, _JsonValue]


def load_json_object_file(data: bytes) -> _JsonObject:
    try:
        value = json.loads(data.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise CanonicalJsonError("JSON file must be UTF-8") from error
    except json.JSONDecodeError as error:
        raise CanonicalJsonError(f"invalid JSON file: {error.msg}") from error
    if not isinstance(value, Mapping):
        raise CanonicalJsonError("JSON file must contain an object")

    mapping = cast(Mapping[str, object], value)
    normalized = _normalize_json(mapping, path=())
    if not isinstance(normalized, Mapping):
        raise CanonicalJsonError("JSON file must contain an object")
    return cast(_JsonObject, normalized)


def _normalize_json(value: object, *, path: tuple[str, ...]) -> _JsonValue:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        import math

        if not math.isfinite(value):
            raise CanonicalJsonError(f"{_format_path(path)}: nonfinite number")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, _JsonValue] = {}
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
            for index, item in enumerate(cast(Sequence[_JsonValue], value))
        ]
    raise CanonicalJsonError(f"{_format_path(path)}: unsupported JSON value")


def _format_path(path: tuple[str, ...]) -> str:
    return ".".join(path) if path else "<value>"
