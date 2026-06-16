"""Shared helpers for residual-certified precision estimates."""

from __future__ import annotations

import math
from typing import Any

__all__ = ["residual_certified_epsilon"]


def residual_certified_epsilon(
    residual_norm: Any,
    stability_factor: Any,
    *,
    floor: float = math.ulp(1.0),
) -> Any:
    """Return the certified precision from a residual and structural stability."""

    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("certified epsilon floor must be positive and finite")
    epsilon = residual_norm * stability_factor
    clamp_min = getattr(epsilon, "clamp_min", None)
    if callable(clamp_min):
        return clamp_min(floor)
    return max(float(epsilon), floor)
