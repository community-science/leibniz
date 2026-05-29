"""Training protocol and validation-history records for local runs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from leibniz.records import FieldSpec, RecordSpec

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
        "batch_size": FieldSpec(kind="integer"),
        "max_steps": FieldSpec(kind="integer", required=False),
        "validation_interval": FieldSpec(kind="integer"),
        "validation_sample_count": FieldSpec(kind="integer"),
        "min_delta": FieldSpec(kind="number"),
        "patience": FieldSpec(kind="integer"),
        "min_steps": FieldSpec(kind="integer", required=False),
        "tensor_runtime": FieldSpec(kind="string", required=False),
        "tensor_device": FieldSpec(kind="string", required=False),
        "validation_source": FieldSpec(kind="string"),
    }
)
_history_point_record = RecordSpec(
    fields={
        "step": FieldSpec(kind="integer"),
        "validation_check": FieldSpec(kind="integer"),
        "validation_loss": FieldSpec(kind="number"),
        "best_validation_loss": FieldSpec(kind="number"),
        "best_validation_step": FieldSpec(kind="integer"),
        "best_validation_check": FieldSpec(kind="integer"),
        "stale_checks": FieldSpec(kind="integer"),
        "learning_rates": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="number"),
            required=False,
        ),
    }
)
_training_run_record = RecordSpec(
    fields={
        "format": FieldSpec(kind="literal", literal="leibniz.training-run"),
        "format_version": FieldSpec(kind="literal", literal=1),
        "status": FieldSpec(kind="string"),
        "stop_reason": FieldSpec(kind="string"),
        "steps_run": FieldSpec(kind="integer"),
        "validation_checks": FieldSpec(kind="integer"),
        "best_validation_loss": FieldSpec(kind="number"),
        "best_validation_step": FieldSpec(kind="integer"),
        "best_validation_check": FieldSpec(kind="integer"),
        "protocol": FieldSpec(kind="record"),
        "validation_history": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
    }
)


class TrainingRunValidationError(ValueError):
    """Raised when a local training run record is invalid."""


@dataclass(frozen=True, slots=True)
class TrainingProtocol:
    """Declared local training protocol for a benchmark run."""

    kind: str
    objective: str
    optimizer: _optimizer_kind
    learning_rate: float
    schedule: _schedule_kind
    seed: int
    batch_size: int
    max_steps: int | None
    validation_interval: int
    validation_sample_count: int
    min_delta: float
    patience: int
    validation_source: str
    min_steps: int = 0
    tensor_runtime: str = "pytorch"
    tensor_device: str = "cpu"

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
        _require_positive_int(self.batch_size, "batch_size")
        if self.max_steps is not None:
            _require_nonnegative_int(self.max_steps, "max_steps")
        _require_positive_int(self.validation_interval, "validation_interval")
        _require_positive_int(self.validation_sample_count, "validation_sample_count")
        _require_nonnegative_finite(self.min_delta, "min_delta")
        _require_nonnegative_int(self.patience, "patience")
        _require_nonnegative_int(self.min_steps, "min_steps")
        if not self.tensor_runtime:
            raise TrainingRunValidationError("tensor_runtime must be nonempty")
        if not self.tensor_device:
            raise TrainingRunValidationError("tensor_device must be nonempty")
        if not self.validation_source:
            raise TrainingRunValidationError("validation_source must be nonempty")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> TrainingProtocol:
        try:
            validated = _protocol_record.validate(record)
        except ValueError as error:
            raise TrainingRunValidationError(str(error)) from error
        return cls(
            kind=_as_string(validated["kind"], "kind"),
            objective=_as_string(validated["objective"], "objective"),
            optimizer=cast(_optimizer_kind, _as_string(validated["optimizer"], "optimizer")),
            learning_rate=_as_float(validated["learning_rate"], "learning_rate"),
            schedule=cast(_schedule_kind, _as_string(validated["schedule"], "schedule")),
            seed=_as_int(validated["seed"], "seed"),
            batch_size=_as_int(validated["batch_size"], "batch_size"),
            max_steps=(
                None
                if "max_steps" not in validated
                else _as_int(validated["max_steps"], "max_steps")
            ),
            validation_interval=_as_int(
                validated["validation_interval"],
                "validation_interval",
            ),
            validation_sample_count=_as_int(
                validated["validation_sample_count"],
                "validation_sample_count",
            ),
            min_delta=_as_float(validated["min_delta"], "min_delta"),
            patience=_as_int(validated["patience"], "patience"),
            validation_source=_as_string(
                validated["validation_source"],
                "validation_source",
            ),
            min_steps=_as_int(validated.get("min_steps", 0), "min_steps"),
            tensor_runtime=_as_string(
                validated.get("tensor_runtime", "pytorch"),
                "tensor_runtime",
            ),
            tensor_device=_as_string(
                validated.get("tensor_device", "cpu"),
                "tensor_device",
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
            "batch_size": self.batch_size,
            "validation_interval": self.validation_interval,
            "validation_sample_count": self.validation_sample_count,
            "min_delta": self.min_delta,
            "patience": self.patience,
            "tensor_runtime": self.tensor_runtime,
            "tensor_device": self.tensor_device,
            "validation_source": self.validation_source,
        }
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
    best_validation_loss: float
    best_validation_step: int
    best_validation_check: int
    stale_checks: int
    learning_rates: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        _require_nonnegative_int(self.step, "step")
        _require_nonnegative_int(self.validation_check, "validation_check")
        _require_nonnegative_finite(self.validation_loss, "validation_loss")
        _require_nonnegative_finite(self.best_validation_loss, "best_validation_loss")
        _require_nonnegative_int(self.best_validation_step, "best_validation_step")
        _require_nonnegative_int(self.best_validation_check, "best_validation_check")
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
            step=_as_int(validated["step"], "step"),
            validation_check=_as_int(validated["validation_check"], "validation_check"),
            validation_loss=_as_float(validated["validation_loss"], "validation_loss"),
            best_validation_loss=_as_float(
                validated["best_validation_loss"],
                "best_validation_loss",
            ),
            best_validation_step=_as_int(
                validated["best_validation_step"],
                "best_validation_step",
            ),
            best_validation_check=_as_int(
                validated["best_validation_check"],
                "best_validation_check",
            ),
            stale_checks=_as_int(validated["stale_checks"], "stale_checks"),
            learning_rates=tuple(
                _as_float(rate, "learning_rates")
                for rate in _as_sequence(validated.get("learning_rates", ()), "learning_rates")
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "step": self.step,
            "validation_check": self.validation_check,
            "validation_loss": self.validation_loss,
            "best_validation_loss": self.best_validation_loss,
            "best_validation_step": self.best_validation_step,
            "best_validation_check": self.best_validation_check,
            "stale_checks": self.stale_checks,
        }
        if self.learning_rates:
            record["learning_rates"] = list(self.learning_rates)
        return record


@dataclass(frozen=True, slots=True)
class TrainingRunRecord:
    """Validated local training run summary."""

    status: _training_status
    stop_reason: str
    steps_run: int
    validation_checks: int
    best_validation_loss: float
    best_validation_step: int
    best_validation_check: int
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
        _require_positive_int(self.validation_checks, "validation_checks")
        _require_nonnegative_finite(self.best_validation_loss, "best_validation_loss")
        _require_nonnegative_int(self.best_validation_step, "best_validation_step")
        _require_nonnegative_int(self.best_validation_check, "best_validation_check")
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
        best_point = min(
            self.validation_history,
            key=lambda point: (point.best_validation_loss, point.best_validation_check),
        )
        if not math.isclose(self.best_validation_loss, best_point.best_validation_loss):
            raise TrainingRunValidationError(
                "best_validation_loss must match validation_history"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> TrainingRunRecord:
        try:
            validated = _training_run_record.validate(record)
        except ValueError as error:
            raise TrainingRunValidationError(str(error)) from error
        return cls(
            status=cast(_training_status, _as_string(validated["status"], "status")),
            stop_reason=_as_string(validated["stop_reason"], "stop_reason"),
            steps_run=_as_int(validated["steps_run"], "steps_run"),
            validation_checks=_as_int(validated["validation_checks"], "validation_checks"),
            best_validation_loss=_as_float(
                validated["best_validation_loss"],
                "best_validation_loss",
            ),
            best_validation_step=_as_int(
                validated["best_validation_step"],
                "best_validation_step",
            ),
            best_validation_check=_as_int(
                validated["best_validation_check"],
                "best_validation_check",
            ),
            protocol=TrainingProtocol.from_record(
                _as_mapping(validated["protocol"], "protocol")
            ),
            validation_history=tuple(
                TrainingHistoryPoint.from_record(_as_mapping(point, "validation_history"))
                for point in _as_sequence(
                    validated["validation_history"],
                    "validation_history",
                )
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "format": "leibniz.training-run",
            "format_version": 1,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "steps_run": self.steps_run,
            "validation_checks": self.validation_checks,
            "best_validation_loss": self.best_validation_loss,
            "best_validation_step": self.best_validation_step,
            "best_validation_check": self.best_validation_check,
            "protocol": self.protocol.to_record(),
            "validation_history": [
                point.to_record() for point in self.validation_history
            ],
        }


def _as_float(value: object, field: str) -> float:
    if not isinstance(value, int | float):
        raise TrainingRunValidationError(f"{field}: expected number")
    result = float(value)
    if not math.isfinite(result):
        raise TrainingRunValidationError(f"{field}: expected finite number")
    return result


def _as_int(value: object, field: str) -> int:
    if type(value) is not int:
        raise TrainingRunValidationError(f"{field}: expected integer")
    return value


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TrainingRunValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple | list):
        raise TrainingRunValidationError(f"{field}: expected sequence")
    return tuple(cast(tuple[object, ...] | list[object], value))


def _as_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TrainingRunValidationError(f"{field}: expected nonempty string")
    return value


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
