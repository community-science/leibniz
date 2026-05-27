"""Tensor-shape descriptors for formal model semantics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

__all__ = [
    "TensorShape",
    "TensorShapeValidationError",
]


class TensorShapeValidationError(ValueError):
    """Raised when a tensor-shape descriptor is invalid."""


@dataclass(frozen=True, slots=True)
class TensorShape:
    """A finite positive tensor axis tuple.

    Tensor shapes are per-observation extents. They do not include backend
    device, dtype, batch-axis, image-layout, or framework-specific information.
    Existing public artifacts continue to represent shapes as lists of axes.
    """

    axes: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", self._validated_axes(self.axes, field="shape"))

    @classmethod
    def from_record(cls, value: object, *, field: str = "shape") -> TensorShape:
        """Parse a tensor shape from an existing public shape-list value."""

        if not isinstance(value, Sequence) or isinstance(value, str | bytes):
            raise TensorShapeValidationError(f"{field}: expected shape sequence")
        sequence = cast(Sequence[object], value)
        return cls(axes=cls._validated_axes(tuple(sequence), field=field))

    @classmethod
    def from_axes(cls, axes: tuple[int, ...], *, field: str = "shape") -> TensorShape:
        """Create a tensor shape from already-parsed axes."""

        return cls(axes=cls._validated_axes(axes, field=field))

    @property
    def rank(self) -> int:
        """Return the number of tensor axes."""

        return len(self.axes)

    @property
    def element_count(self) -> int:
        """Return the product of all axis extents."""

        return math.prod(self.axes)

    def to_record(self) -> list[int]:
        """Return the existing public shape-list representation."""

        return list(self.axes)

    @staticmethod
    def _validated_axes(axes: tuple[object, ...], *, field: str) -> tuple[int, ...]:
        if not axes:
            raise TensorShapeValidationError(f"{field} must contain at least one axis")
        if any(type(axis) is not int or axis < 1 for axis in axes):
            raise TensorShapeValidationError(f"{field} axes must be positive integers")
        return tuple(cast(tuple[int, ...], axes))
