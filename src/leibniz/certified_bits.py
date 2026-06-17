"""Shared certified-bits ledger for known-law correctness leaves.

The ledger owns the common residual-certified scoring skeleton:
residual ladder -> certified epsilon -> ambient entropy resolved above epsilon
-> stability/refinement refusal -> per-sample diagnostics. Structural benchmark
types provide those organs through ``CertificationEstimator``; they do not own
the skeleton.

This module is deliberately scoped to certified known-law (``g``) leaves. The
future frontier-relative (``h``) leaf attaches at the same leaf boundary by
replacing the stability organ with a frontier-relative reference while keeping
``ambient_entropy_above``. The partition-tree aggregator attaches above this
leaf result. Neither extension is implemented here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from leibniz.certified_precision import residual_certified_epsilon

__all__ = [
    "AmbientEntropy",
    "CertificationEstimator",
    "CertificationStability",
    "CertifiedBitsResult",
    "evaluate_certified_bits",
]


@dataclass(frozen=True, slots=True)
class CertificationStability:
    """Per-sample stability factor and refusal flag from a structural estimator."""

    factor: Any
    refused: Any
    diagnostics: Mapping[str, object] | Sequence[Mapping[str, object]] = ()
    refused_status: str = "refused-stability"


@dataclass(frozen=True, slots=True)
class AmbientEntropy:
    """Per-sample ambient entropy resolved above a requested precision."""

    bits: Any
    signal: Any
    diagnostics: Mapping[str, object] | Sequence[Mapping[str, object]] = ()


class CertificationEstimator(Protocol):
    """Structural organs consumed by the certified-bits ledger."""

    def residuals(self) -> Sequence[Any]:
        """Return per-sample residual norms across the refinement ladder."""
        ...

    def stability(self) -> CertificationStability:
        """Return the certified-epsilon factor and refinement refusal flag."""
        ...

    def ambient_entropy_above(self, precision: Any) -> AmbientEntropy:
        """Return resolved bits and signal scale above ``precision``."""
        ...


@dataclass(frozen=True, slots=True)
class CertifiedBitsResult:
    """Ledger values and unified per-sample diagnostic records."""

    values: Any
    diagnostics: tuple[Mapping[str, object], ...]


def evaluate_certified_bits(
    estimator: CertificationEstimator,
    *,
    sample_count: int | None = None,
    value_factory: Callable[[Sequence[float]], Any] | None = None,
) -> CertifiedBitsResult:
    """Evaluate certified bits for one estimator.

    ``value_factory`` converts host floats back into the caller's preferred value
    container. Benchmarks that score tensors should pass a factory such as
    ``finest.new_tensor``; tests and non-tensor callers may omit it and receive a
    tuple of floats.
    """

    residual_ladder = tuple(estimator.residuals())
    if not residual_ladder:
        raise ValueError("certified bits require at least one residual rung")
    inferred_count = _sample_count(residual_ladder[-1])
    if sample_count is None:
        sample_count = inferred_count
    if sample_count < 0:
        raise ValueError("sample count must be nonnegative")

    stability = estimator.stability()
    certified_epsilon = _certified_epsilon(
        residual_ladder[-1],
        stability.factor,
        sample_count=sample_count,
    )
    entropy = estimator.ambient_entropy_above(certified_epsilon)
    estimator_kind = _optional_string_attribute(estimator, "kind")
    structural_type = _optional_string_attribute(estimator, "structural_type")

    values: list[float] = []
    diagnostics: list[Mapping[str, object]] = []
    for sample_index in range(sample_count):
        residual_values = tuple(
            _sample_float(residuals, sample_index) for residuals in residual_ladder
        )
        epsilon = _sample_float(certified_epsilon, sample_index)
        signal = _sample_float(entropy.signal, sample_index)
        entropy_bits = _sample_float(entropy.bits, sample_index)
        refused = _sample_bool(stability.refused, sample_index)

        status = "certified"
        zero_credit_reason: str | None = None
        bits = entropy_bits
        if refused:
            status = stability.refused_status
            zero_credit_reason = "stability-refusal"
            bits = 0.0
        elif not math.isfinite(signal):
            status = "refused-nonfinite-signal"
            zero_credit_reason = "nonfinite-signal"
            bits = 0.0
        elif epsilon >= signal:
            zero_credit_reason = "precision-not-below-signal"
            bits = 0.0

        diagnostic: dict[str, object] = {
            "kind": "certified-bits-diagnostics",
            "sample_index": sample_index,
            "certification_status": status,
            "bits": bits,
            "residual_norm": residual_values[-1],
            "residual_norms": list(residual_values),
            "certified_epsilon": epsilon,
            "signal_scale": signal,
            "ambient_entropy_bits": entropy_bits,
            "zero_credit_reason": zero_credit_reason,
            "stability": _sample_diagnostic(stability.diagnostics, sample_index),
            "ambient_entropy": _sample_diagnostic(entropy.diagnostics, sample_index),
        }
        if estimator_kind is not None:
            diagnostic["estimator"] = estimator_kind
        if structural_type is not None:
            diagnostic["structural_type"] = structural_type
        values.append(bits)
        diagnostics.append(diagnostic)

    if value_factory is None:
        result_values: Any = tuple(values)
    else:
        result_values = value_factory(values)
    return CertifiedBitsResult(values=result_values, diagnostics=tuple(diagnostics))


def _sample_count(values: Any) -> int:
    shape = getattr(values, "shape", None)
    if shape is not None and len(tuple(shape)) > 0:
        return int(tuple(shape)[0])
    try:
        return len(values)
    except TypeError:
        return 1


def _certified_epsilon(residual_norm: Any, stability_factor: Any, *, sample_count: int) -> Any:
    try:
        return residual_certified_epsilon(residual_norm, stability_factor)
    except TypeError:
        return tuple(
            residual_certified_epsilon(
                _sample_float(residual_norm, sample_index),
                _sample_float(stability_factor, sample_index),
            )
            for sample_index in range(sample_count)
        )


def _sample_float(values: Any, sample_index: int) -> float:
    return float(_sample_value(values, sample_index))


def _sample_bool(values: Any, sample_index: int) -> bool:
    return bool(_sample_value(values, sample_index))


def _sample_value(values: Any, sample_index: int) -> Any:
    shape = getattr(values, "shape", None)
    if shape is not None and len(tuple(shape)) > 0:
        return values[sample_index]
    if isinstance(values, Sequence) and not isinstance(values, str | bytes):
        return cast(Any, values[sample_index])
    return values


def _sample_diagnostic(
    diagnostics: Mapping[str, object] | Sequence[Mapping[str, object]],
    sample_index: int,
) -> Mapping[str, object]:
    if isinstance(diagnostics, Mapping):
        return dict(diagnostics)
    if not diagnostics:
        return {}
    return dict(diagnostics[sample_index])


def _optional_string_attribute(owner: object, name: str) -> str | None:
    value = getattr(owner, name, None)
    return value if isinstance(value, str) else None
