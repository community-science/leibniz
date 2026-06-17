from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from leibniz.certified_bits import (
    AmbientEntropy,
    CertificationStability,
    evaluate_certified_bits,
)


@dataclass(frozen=True, slots=True)
class SyntheticEstimator:
    residual_ladder: tuple[tuple[float, ...], ...]
    factor: tuple[float, ...]
    refused: tuple[bool, ...]
    entropy_bits: tuple[float, ...]
    signal: tuple[float, ...]
    kind: str = "synthetic-estimator"
    structural_type: str = "synthetic"

    def residuals(self) -> tuple[tuple[float, ...], ...]:
        return self.residual_ladder

    def stability(self) -> CertificationStability:
        return CertificationStability(
            factor=self.factor,
            refused=self.refused,
            diagnostics=tuple(
                {"factor": factor, "refused": refused}
                for factor, refused in zip(self.factor, self.refused, strict=True)
            ),
            refused_status="refused-synthetic-stability",
        )

    def ambient_entropy_above(self, precision: Any) -> AmbientEntropy:
        return AmbientEntropy(
            bits=self.entropy_bits,
            signal=self.signal,
            diagnostics=tuple(
                {"precision": float(value)}
                for value in precision
            ),
        )


def test_certified_bits_core_credits_entropy_above_certified_precision() -> None:
    estimator = SyntheticEstimator(
        residual_ladder=((0.02, 0.01), (0.01, 0.005)),
        factor=(2.0, 2.0),
        refused=(False, False),
        entropy_bits=(4.5, 7.25),
        signal=(0.1, 0.1),
    )

    result = evaluate_certified_bits(estimator)

    assert result.values == (4.5, 7.25)
    first = result.diagnostics[0]
    assert first["kind"] == "certified-bits-diagnostics"
    assert first["estimator"] == "synthetic-estimator"
    assert first["structural_type"] == "synthetic"
    assert first["certification_status"] == "certified"
    assert first["zero_credit_reason"] is None
    assert first["residual_norms"] == [0.02, 0.01]
    certified_epsilon = first["certified_epsilon"]
    assert isinstance(certified_epsilon, float)
    assert math.isclose(certified_epsilon, 0.02)
    assert first["stability"] == {"factor": 2.0, "refused": False}
    assert first["ambient_entropy"] == {"precision": 0.02}


def test_certified_bits_core_zeroes_credit_when_precision_reaches_signal() -> None:
    estimator = SyntheticEstimator(
        residual_ladder=((0.02,),),
        factor=(5.0,),
        refused=(False,),
        entropy_bits=(12.0,),
        signal=(0.1,),
    )

    result = evaluate_certified_bits(estimator)

    assert result.values == (0.0,)
    assert result.diagnostics[0]["certification_status"] == "certified"
    assert result.diagnostics[0]["zero_credit_reason"] == "precision-not-below-signal"
    assert result.diagnostics[0]["ambient_entropy_bits"] == 12.0


def test_certified_bits_core_zeroes_credit_when_refinement_is_refused() -> None:
    estimator = SyntheticEstimator(
        residual_ladder=((0.001, 0.001),),
        factor=(1.0, 1.0),
        refused=(False, True),
        entropy_bits=(3.0, 9.0),
        signal=(1.0, 1.0),
    )

    result = evaluate_certified_bits(
        estimator,
        value_factory=lambda values: list(values),
    )

    assert result.values == [3.0, 0.0]
    assert result.diagnostics[1]["certification_status"] == "refused-synthetic-stability"
    assert result.diagnostics[1]["zero_credit_reason"] == "stability-refusal"
