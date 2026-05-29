"""Adaptive integer-scale evaluation records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "AdaptiveScaleEvaluation",
    "PerScaleScore",
    "ScaleAxis",
    "ScaleEvaluationLevel",
    "ScaleEvaluationTrace",
    "ScaleEvaluationValidationError",
]

_scale_axis_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="literal", literal="positive-integer-scale-axis"),
        "symbol": FieldSpec(kind="string"),
        "minimum": FieldSpec(kind="integer"),
        "maximum": FieldSpec(kind="integer", required=False),
    }
)
_scale_score_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="literal", literal="bounded-local-competence"),
        "name": FieldSpec(kind="string"),
        "lower_bound": FieldSpec(kind="number"),
        "upper_bound": FieldSpec(kind="number"),
        "direction": FieldSpec(kind="literal", literal="higher"),
    }
)
_adaptive_scale_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="literal", literal="start-at-minimum-until-zero-marginal-score"),
        "axis_symbol": FieldSpec(kind="string"),
        "step": FieldSpec(kind="integer"),
        "stopping_window": FieldSpec(kind="integer"),
        "marginal_score_epsilon": FieldSpec(kind="number"),
    }
)
_scale_level_record = RecordSpec(
    fields={
        "scale": FieldSpec(kind="integer"),
        "competence": FieldSpec(kind="number"),
        "resources": FieldSpec(kind="record", required=False),
        "boundary_reason": FieldSpec(kind="string", required=False),
    }
)
_scale_trace_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="literal", literal="adaptive-integer-scale-evaluation-trace"),
        "axis": FieldSpec(kind="record"),
        "per_scale_score": FieldSpec(kind="record"),
        "evaluation": FieldSpec(kind="record"),
        "levels": FieldSpec(kind="sequence", item=FieldSpec(kind="record")),
        "stop_reason": FieldSpec(kind="string"),
        "integrated_score": FieldSpec(kind="number"),
    }
)


class ScaleEvaluationValidationError(ValueError):
    """Raised when an adaptive scale-evaluation record is invalid."""


@dataclass(frozen=True, slots=True)
class ScaleAxis:
    """A public positive integer scale axis."""

    symbol: str
    minimum: int = 1
    maximum: int | None = None

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ScaleEvaluationValidationError("scale axis symbol must be nonempty")
        _require_positive_int(self.minimum, "minimum")
        if self.maximum is not None:
            _require_positive_int(self.maximum, "maximum")
            if self.maximum < self.minimum:
                raise ScaleEvaluationValidationError("maximum must be at least minimum")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ScaleAxis:
        try:
            validated = _scale_axis_record.validate(record)
        except ValueError as error:
            raise ScaleEvaluationValidationError(str(error)) from error
        return cls(
            symbol=str(validated["symbol"]),
            minimum=_as_int(validated["minimum"], "minimum"),
            maximum=(
                None
                if "maximum" not in validated
                else _as_int(validated["maximum"], "maximum")
            ),
        )

    def contains(self, scale: int) -> bool:
        if type(scale) is not int or scale < self.minimum:
            return False
        return self.maximum is None or scale <= self.maximum

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": "positive-integer-scale-axis",
            "symbol": self.symbol,
            "minimum": self.minimum,
        }
        if self.maximum is not None:
            record["maximum"] = self.maximum
        return record


@dataclass(frozen=True, slots=True)
class PerScaleScore:
    """A bounded local competence score at one scale."""

    name: str = "per-scale-competence"
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    direction: str = "higher"

    def __post_init__(self) -> None:
        if not self.name:
            raise ScaleEvaluationValidationError("per-scale score name must be nonempty")
        if self.direction != "higher":
            raise ScaleEvaluationValidationError("per-scale score direction must be higher")
        _require_finite(self.lower_bound, "lower_bound")
        _require_finite(self.upper_bound, "upper_bound")
        if self.lower_bound != 0.0 or self.upper_bound != 1.0:
            raise ScaleEvaluationValidationError("per-scale competence must use bounds [0, 1]")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> PerScaleScore:
        try:
            validated = _scale_score_record.validate(record)
        except ValueError as error:
            raise ScaleEvaluationValidationError(str(error)) from error
        return cls(
            name=str(validated["name"]),
            lower_bound=_as_float(validated["lower_bound"], "lower_bound"),
            upper_bound=_as_float(validated["upper_bound"], "upper_bound"),
            direction=str(validated["direction"]),
        )

    def validate_value(self, value: float) -> float:
        _require_finite(value, "competence")
        if value < self.lower_bound or value > self.upper_bound:
            raise ScaleEvaluationValidationError("competence must lie in [0, 1]")
        return float(value)

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "bounded-local-competence",
            "name": self.name,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "direction": self.direction,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveScaleEvaluation:
    """Evaluate consecutive scales until marginal competence is effectively zero."""

    axis_symbol: str
    step: int = 1
    stopping_window: int = 1
    marginal_score_epsilon: float = 0.0

    def __post_init__(self) -> None:
        if not self.axis_symbol:
            raise ScaleEvaluationValidationError("axis_symbol must be nonempty")
        _require_positive_int(self.step, "step")
        if self.step != 1:
            raise ScaleEvaluationValidationError("step must be one")
        _require_positive_int(self.stopping_window, "stopping_window")
        _require_finite(self.marginal_score_epsilon, "marginal_score_epsilon")
        if self.marginal_score_epsilon < 0:
            raise ScaleEvaluationValidationError("marginal_score_epsilon must be nonnegative")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> AdaptiveScaleEvaluation:
        try:
            validated = _adaptive_scale_record.validate(record)
        except ValueError as error:
            raise ScaleEvaluationValidationError(str(error)) from error
        return cls(
            axis_symbol=str(validated["axis_symbol"]),
            step=_as_int(validated["step"], "step"),
            stopping_window=_as_int(validated["stopping_window"], "stopping_window"),
            marginal_score_epsilon=_as_float(
                validated["marginal_score_epsilon"],
                "marginal_score_epsilon",
            ),
        )

    def should_stop(self, recent_competence: Sequence[float]) -> bool:
        if len(recent_competence) < self.stopping_window:
            return False
        window = tuple(float(value) for value in recent_competence[-self.stopping_window :])
        for value in window:
            _require_finite(value, "recent competence")
        return sum(window) <= self.marginal_score_epsilon

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "start-at-minimum-until-zero-marginal-score",
            "axis_symbol": self.axis_symbol,
            "step": self.step,
            "stopping_window": self.stopping_window,
            "marginal_score_epsilon": self.marginal_score_epsilon,
        }


@dataclass(frozen=True, slots=True)
class ScaleEvaluationLevel:
    """Evidence for one evaluated or boundary scale."""

    scale: int
    competence: float
    resources: Mapping[str, object] | None = None
    boundary_reason: str | None = None

    def validate(self, *, axis: ScaleAxis, score: PerScaleScore) -> None:
        if not axis.contains(self.scale):
            raise ScaleEvaluationValidationError("scale is outside axis domain")
        score.validate_value(self.competence)
        if self.resources is not None:
            _validate_scalar_mapping(self.resources, "resources")
        if self.boundary_reason is not None and not self.boundary_reason:
            raise ScaleEvaluationValidationError("boundary_reason must be nonempty")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ScaleEvaluationLevel:
        try:
            validated = _scale_level_record.validate(record)
        except ValueError as error:
            raise ScaleEvaluationValidationError(str(error)) from error
        return cls(
            scale=_as_int(validated["scale"], "scale"),
            competence=_as_float(validated["competence"], "competence"),
            resources=(
                None
                if "resources" not in validated
                else _as_mapping(validated["resources"], "resources")
            ),
            boundary_reason=(
                None
                if "boundary_reason" not in validated
                else str(validated["boundary_reason"])
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "scale": self.scale,
            "competence": self.competence,
        }
        if self.resources is not None:
            record["resources"] = dict(self.resources)
        if self.boundary_reason is not None:
            record["boundary_reason"] = self.boundary_reason
        return record


@dataclass(frozen=True, slots=True)
class ScaleEvaluationTrace:
    """Scale-integrated competence evidence for one model evaluation."""

    axis: ScaleAxis
    score: PerScaleScore
    evaluation: AdaptiveScaleEvaluation
    levels: tuple[ScaleEvaluationLevel, ...]
    stop_reason: str

    def __post_init__(self) -> None:
        self.validate()

    @property
    def integrated_score(self) -> float:
        return sum(level.competence for level in self.levels)

    def validate(self) -> None:
        if self.evaluation.axis_symbol != self.axis.symbol:
            raise ScaleEvaluationValidationError(
                "evaluation axis_symbol must match scale axis symbol"
            )
        if not self.levels:
            raise ScaleEvaluationValidationError("scale trace must contain at least one level")
        if not self.stop_reason:
            raise ScaleEvaluationValidationError("stop_reason must be nonempty")
        expected = self.axis.minimum
        for level in self.levels:
            level.validate(axis=self.axis, score=self.score)
            if level.scale != expected:
                raise ScaleEvaluationValidationError(
                    "scale trace levels must be consecutive from axis minimum"
                )
            expected += self.evaluation.step

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ScaleEvaluationTrace:
        try:
            validated = _scale_trace_record.validate(record)
        except ValueError as error:
            raise ScaleEvaluationValidationError(str(error)) from error
        trace = cls(
            axis=ScaleAxis.from_record(_as_mapping(validated["axis"], "axis")),
            score=PerScaleScore.from_record(
                _as_mapping(validated["per_scale_score"], "per_scale_score")
            ),
            evaluation=AdaptiveScaleEvaluation.from_record(
                _as_mapping(validated["evaluation"], "evaluation")
            ),
            levels=tuple(
                ScaleEvaluationLevel.from_record(_as_mapping(level, "levels"))
                for level in _as_sequence(validated["levels"], "levels")
            ),
            stop_reason=str(validated["stop_reason"]),
        )
        if not math.isclose(
            trace.integrated_score,
            _as_float(validated["integrated_score"], "integrated_score"),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ScaleEvaluationValidationError("integrated_score does not match levels")
        return trace

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "adaptive-integer-scale-evaluation-trace",
            "axis": self.axis.to_record(),
            "per_scale_score": self.score.to_record(),
            "evaluation": self.evaluation.to_record(),
            "levels": [level.to_record() for level in self.levels],
            "stop_reason": self.stop_reason,
            "integrated_score": self.integrated_score,
        }


def _require_positive_int(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ScaleEvaluationValidationError(f"{field} must be a positive integer")
    return value


def _require_finite(value: object, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ScaleEvaluationValidationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ScaleEvaluationValidationError(f"{field} must be finite")
    return result


def _as_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise ScaleEvaluationValidationError(f"{field}: expected integer")
    return value


def _as_float(value: object, field: str) -> float:
    return _require_finite(value, field)


def _as_sequence(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ScaleEvaluationValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ScaleEvaluationValidationError(f"{field}: expected parsed record")
    return cast(Mapping[str, object], value)


def _validate_scalar_mapping(value: Mapping[str, object], field: str) -> None:
    for key, item in value.items():
        if not key:
            raise ScaleEvaluationValidationError(f"{field} keys must be nonempty")
        if not isinstance(item, int | float | str) or isinstance(item, bool):
            raise ScaleEvaluationValidationError(f"{field} values must be scalar")
        if isinstance(item, float) and not math.isfinite(item):
            raise ScaleEvaluationValidationError(f"{field} values must be finite")
