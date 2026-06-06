"""Training protocol and validation-history records for local runs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from leibniz.records import FieldSpec, RecordExtractor, RecordSpec
from leibniz.tensor_runtime import tensor_runtime_default_device

__all__ = [
    "TrainingHistoryPoint",
    "TrainingProtocol",
    "TrainingRunRecord",
    "TrainingRunValidationError",
]

_optimizer_kind = Literal["sgd", "adam", "adamw"]
_schedule_kind = Literal["none", "cosine", "reduce-on-plateau"]
_training_status = Literal[
    "running",
    "completed",
    "converged",
    "budget-exhausted",
    "not-trainable",
    "failed",
]

_protocol_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="string"),
        "objective": FieldSpec(kind="string"),
        "optimizer": FieldSpec(kind="string"),
        "learning_rate": FieldSpec(kind="number"),
        "schedule": FieldSpec(kind="string"),
        "seed": FieldSpec(kind="integer"),
        "training_evidence_count": FieldSpec(kind="integer"),
        "max_steps": FieldSpec(kind="integer", required=False),
        "gate_check_interval": FieldSpec(kind="integer"),
        "gate_evidence_count": FieldSpec(kind="integer"),
        "gate_decision_rule": FieldSpec(kind="string"),
        "rung_competence_threshold": FieldSpec(kind="number", required=False),
        "min_delta": FieldSpec(kind="number"),
        "patience": FieldSpec(kind="integer"),
        "min_steps": FieldSpec(kind="integer", required=False),
        "tensor_runtime": FieldSpec(kind="string", required=False),
        "tensor_device": FieldSpec(kind="string", required=False),
        "runtime_memory_budget_fraction": FieldSpec(kind="number", required=False),
        "validation_source": FieldSpec(kind="string"),
    }
)
_history_point_record = RecordSpec(
    fields={
        "step": FieldSpec(kind="integer"),
        "validation_check": FieldSpec(kind="integer"),
        "validation_loss": FieldSpec(kind="number"),
        "stale_checks": FieldSpec(kind="integer"),
        "learning_rates": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="number"),
            required=False,
        ),
        "score_estimate": FieldSpec(kind="record", required=False),
    }
)
_training_run_record = RecordSpec(
    fields={
        "format": FieldSpec(kind="literal", literal="leibniz.training-run"),
        "format_version": FieldSpec(kind="literal", literal=1),
        "status": FieldSpec(kind="string"),
        "stop_reason": FieldSpec(kind="string"),
        "steps_run": FieldSpec(kind="integer"),
        "training_compute": FieldSpec(kind="number", required=False),
        "validation_checks": FieldSpec(kind="integer"),
        "protocol": FieldSpec(kind="record"),
        "validation_history": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
    }
)


class TrainingRunValidationError(ValueError):
    """Raised when a local training run record is invalid."""


_extract = RecordExtractor(error_type=TrainingRunValidationError)


@dataclass(frozen=True, slots=True)
class TrainingProtocol:
    """Declared local training protocol for a benchmark run."""

    kind: str
    objective: str
    optimizer: _optimizer_kind
    learning_rate: float
    schedule: _schedule_kind
    seed: int
    training_evidence_count: int
    max_steps: int | None
    gate_check_interval: int
    gate_evidence_count: int
    gate_decision_rule: str
    min_delta: float
    patience: int
    validation_source: str
    min_steps: int = 0
    rung_competence_threshold: float = 0.01
    tensor_runtime: str = "pytorch"
    tensor_device: str = tensor_runtime_default_device()
    runtime_memory_budget_fraction: float | None = None

    def __post_init__(self) -> None:
        if not self.kind:
            raise TrainingRunValidationError("kind must be nonempty")
        if not self.objective:
            raise TrainingRunValidationError("objective must be nonempty")
        if self.optimizer not in {"sgd", "adam", "adamw"}:
            raise TrainingRunValidationError(f"unsupported optimizer: {self.optimizer}")
        if not _is_positive_finite(self.learning_rate):
            raise TrainingRunValidationError("learning_rate must be positive and finite")
        if self.schedule not in {"none", "cosine", "reduce-on-plateau"}:
            raise TrainingRunValidationError(f"unsupported schedule: {self.schedule}")
        _require_nonnegative_int(self.seed, "seed")
        _require_positive_int(self.training_evidence_count, "training_evidence_count")
        if self.max_steps is not None:
            _require_nonnegative_int(self.max_steps, "max_steps")
        _require_positive_int(self.gate_check_interval, "gate_check_interval")
        _require_positive_int(self.gate_evidence_count, "gate_evidence_count")
        if not self.gate_decision_rule:
            raise TrainingRunValidationError("gate_decision_rule must be nonempty")
        if (
            not math.isfinite(float(self.rung_competence_threshold))
            or self.rung_competence_threshold < 0.0
            or self.rung_competence_threshold > 1.0
        ):
            raise TrainingRunValidationError(
                "rung_competence_threshold must be in [0, 1]"
            )
        _require_nonnegative_finite(self.min_delta, "min_delta")
        _require_nonnegative_int(self.patience, "patience")
        _require_nonnegative_int(self.min_steps, "min_steps")
        if not self.tensor_runtime:
            raise TrainingRunValidationError("tensor_runtime must be nonempty")
        if not self.tensor_device:
            raise TrainingRunValidationError("tensor_device must be nonempty")
        if self.runtime_memory_budget_fraction is not None and (
            not math.isfinite(float(self.runtime_memory_budget_fraction))
            or self.runtime_memory_budget_fraction <= 0.0
            or self.runtime_memory_budget_fraction > 1.0
        ):
            raise TrainingRunValidationError(
                "runtime_memory_budget_fraction must be in (0, 1]"
            )
        if not self.validation_source:
            raise TrainingRunValidationError("validation_source must be nonempty")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> TrainingProtocol:
        try:
            validated = _protocol_record.validate(record)
        except ValueError as error:
            raise TrainingRunValidationError(str(error)) from error
        return cls(
            kind=_extract.non_empty_string(validated["kind"], "kind"),
            objective=_extract.non_empty_string(validated["objective"], "objective"),
            optimizer=cast(
                _optimizer_kind,
                _extract.non_empty_string(validated["optimizer"], "optimizer"),
            ),
            learning_rate=_extract.finite_float(validated["learning_rate"], "learning_rate"),
            schedule=cast(
                _schedule_kind,
                _extract.non_empty_string(validated["schedule"], "schedule"),
            ),
            seed=_extract.integer(validated["seed"], "seed"),
            training_evidence_count=_extract.integer(
                validated["training_evidence_count"],
                "training_evidence_count",
            ),
            max_steps=(
                None
                if "max_steps" not in validated
                else _extract.integer(validated["max_steps"], "max_steps")
            ),
            gate_check_interval=_extract.integer(
                validated["gate_check_interval"],
                "gate_check_interval",
            ),
            gate_evidence_count=_extract.integer(
                validated["gate_evidence_count"],
                "gate_evidence_count",
            ),
            gate_decision_rule=_extract.non_empty_string(
                validated["gate_decision_rule"],
                "gate_decision_rule",
            ),
            rung_competence_threshold=_extract.finite_float(
                validated.get("rung_competence_threshold", 0.01),
                "rung_competence_threshold",
            ),
            min_delta=_extract.finite_float(validated["min_delta"], "min_delta"),
            patience=_extract.integer(validated["patience"], "patience"),
            validation_source=_extract.non_empty_string(
                validated["validation_source"],
                "validation_source",
            ),
            min_steps=_extract.integer(validated.get("min_steps", 0), "min_steps"),
            tensor_runtime=_extract.non_empty_string(
                validated.get("tensor_runtime", "pytorch"),
                "tensor_runtime",
            ),
            tensor_device=_extract.non_empty_string(
                validated.get("tensor_device", tensor_runtime_default_device()),
                "tensor_device",
            ),
            runtime_memory_budget_fraction=(
                None
                if "runtime_memory_budget_fraction" not in validated
                else _extract.finite_float(
                    validated["runtime_memory_budget_fraction"],
                    "runtime_memory_budget_fraction",
                )
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": self.kind,
            "objective": self.objective,
            "optimizer": self.optimizer,
            "learning_rate": self.learning_rate,
            "schedule": self.schedule,
            "seed": self.seed,
            "training_evidence_count": self.training_evidence_count,
            "gate_check_interval": self.gate_check_interval,
            "gate_evidence_count": self.gate_evidence_count,
            "gate_decision_rule": self.gate_decision_rule,
            "rung_competence_threshold": self.rung_competence_threshold,
            "min_delta": self.min_delta,
            "patience": self.patience,
            "tensor_runtime": self.tensor_runtime,
            "tensor_device": self.tensor_device,
            "validation_source": self.validation_source,
        }
        if self.runtime_memory_budget_fraction is not None:
            record["runtime_memory_budget_fraction"] = (
                self.runtime_memory_budget_fraction
            )
        if self.max_steps is not None:
            record["max_steps"] = self.max_steps
        if self.min_steps:
            record["min_steps"] = self.min_steps
        return record


@dataclass(frozen=True, slots=True)
class TrainingHistoryPoint:
    """One validation checkpoint from a local training run."""

    step: int
    validation_check: int
    validation_loss: float
    stale_checks: int
    learning_rates: tuple[float, ...] = ()
    score_estimate: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.step, "step")
        _require_nonnegative_int(self.validation_check, "validation_check")
        _require_nonnegative_finite(self.validation_loss, "validation_loss")
        _require_nonnegative_int(self.stale_checks, "stale_checks")
        for rate in self.learning_rates:
            if not math.isfinite(float(rate)) or rate < 0:
                raise TrainingRunValidationError("learning_rates must be nonnegative and finite")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> TrainingHistoryPoint:
        try:
            validated = _history_point_record.validate(record)
        except ValueError as error:
            raise TrainingRunValidationError(str(error)) from error
        return cls(
            step=_extract.integer(validated["step"], "step"),
            validation_check=_extract.integer(validated["validation_check"], "validation_check"),
            validation_loss=_extract.finite_float(validated["validation_loss"], "validation_loss"),
            stale_checks=_extract.integer(validated["stale_checks"], "stale_checks"),
            learning_rates=tuple(
                _extract.finite_float(rate, "learning_rates")
                for rate in _extract.sequence(validated.get("learning_rates", ()), "learning_rates")
            ),
            score_estimate=_extract.optional_mapping(
                validated.get("score_estimate"),
                "score_estimate",
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "step": self.step,
            "validation_check": self.validation_check,
            "validation_loss": self.validation_loss,
            "stale_checks": self.stale_checks,
        }
        if self.learning_rates:
            record["learning_rates"] = list(self.learning_rates)
        if self.score_estimate is not None:
            record["score_estimate"] = dict(self.score_estimate)
        return record


@dataclass(frozen=True, slots=True)
class TrainingRunRecord:
    """Validated local training run summary."""

    status: _training_status
    stop_reason: str
    steps_run: int
    training_compute: float | None
    validation_checks: int
    protocol: TrainingProtocol
    validation_history: tuple[TrainingHistoryPoint, ...]

    def __post_init__(self) -> None:
        if self.status not in {
            "running",
            "completed",
            "converged",
            "budget-exhausted",
            "not-trainable",
            "failed",
        }:
            raise TrainingRunValidationError(f"unsupported status: {self.status}")
        if not self.stop_reason:
            raise TrainingRunValidationError("stop_reason must be nonempty")
        _require_nonnegative_int(self.steps_run, "steps_run")
        if self.training_compute is not None:
            _require_nonnegative_finite(self.training_compute, "training_compute")
        _require_positive_int(self.validation_checks, "validation_checks")
        if not self.validation_history:
            raise TrainingRunValidationError("validation_history must contain at least one point")
        if self.validation_checks != len(self.validation_history):
            raise TrainingRunValidationError(
                "validation_checks must match validation_history length"
            )
        if self.validation_history != tuple(
            sorted(
                self.validation_history,
                key=lambda point: (point.validation_check, point.step),
            )
        ):
            raise TrainingRunValidationError("validation_history must be sorted")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> TrainingRunRecord:
        try:
            validated = _training_run_record.validate(record)
        except ValueError as error:
            raise TrainingRunValidationError(str(error)) from error
        return cls(
            status=cast(_training_status, _extract.non_empty_string(validated["status"], "status")),
            stop_reason=_extract.non_empty_string(validated["stop_reason"], "stop_reason"),
            steps_run=_extract.integer(validated["steps_run"], "steps_run"),
            training_compute=(
                None
                if "training_compute" not in validated
                else _extract.finite_float(validated["training_compute"], "training_compute")
            ),
            validation_checks=_extract.integer(validated["validation_checks"], "validation_checks"),
            protocol=TrainingProtocol.from_record(
                _extract.mapping(validated["protocol"], "protocol")
            ),
            validation_history=tuple(
                TrainingHistoryPoint.from_record(_extract.mapping(point, "validation_history"))
                for point in _extract.sequence(
                    validated["validation_history"],
                    "validation_history",
                )
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "format": "leibniz.training-run",
            "format_version": 1,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "steps_run": self.steps_run,
            "validation_checks": self.validation_checks,
            "protocol": self.protocol.to_record(),
            "validation_history": [
                point.to_record() for point in self.validation_history
            ],
        }
        if self.training_compute is not None:
            record["training_compute"] = self.training_compute
        return record
def _is_positive_finite(value: float) -> bool:
    return math.isfinite(float(value)) and value > 0


def _require_nonnegative_finite(value: float, field: str) -> None:
    if not math.isfinite(float(value)) or value < 0:
        raise TrainingRunValidationError(f"{field} must be nonnegative and finite")


def _require_nonnegative_int(value: int, field: str) -> None:
    if type(value) is not int or value < 0:
        raise TrainingRunValidationError(f"{field} must be a nonnegative integer")


def _require_positive_int(value: int, field: str) -> None:
    if type(value) is not int or value < 1:
        raise TrainingRunValidationError(f"{field} must be a positive integer")
