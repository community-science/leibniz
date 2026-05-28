import pytest

from leibniz.training_runs import (
    TrainingHistoryPoint,
    TrainingProtocol,
    TrainingRunRecord,
    TrainingRunValidationError,
)


def test_training_run_record_round_trips_protocol_and_history() -> None:
    record = _training_run().to_record()

    parsed = TrainingRunRecord.from_record(record)

    assert parsed == _training_run()
    assert record["format"] == "leibniz.training-run"
    assert record["protocol"] == {
        "kind": "fixed-step-local-batch",
        "objective": "cross-entropy",
        "optimizer": "sgd",
        "learning_rate": 0.01,
        "schedule": "none",
        "seed": 101,
        "batch_size": 2,
        "max_steps": 1,
        "validation_interval": 1,
        "validation_sample_count": 2,
        "min_delta": 0.0,
        "patience": 0,
        "tensor_runtime": "pytorch",
        "tensor_device": "cpu",
        "validation_source": "training-batch",
    }
    assert record["validation_history"] == [
        {
            "step": 0,
            "validation_check": 0,
            "validation_loss": 2.0,
            "best_validation_loss": 2.0,
            "best_validation_step": 0,
            "best_validation_check": 0,
            "stale_checks": 0,
            "learning_rates": [0.01],
        },
        {
            "step": 1,
            "validation_check": 1,
            "validation_loss": 1.5,
            "best_validation_loss": 1.5,
            "best_validation_step": 1,
            "best_validation_check": 1,
            "stale_checks": 0,
            "learning_rates": [0.01],
        },
    ]


def test_training_run_record_rejects_inconsistent_validation_summary() -> None:
    with pytest.raises(
        TrainingRunValidationError,
        match="validation_checks must match validation_history length",
    ):
        TrainingRunRecord(
            status="completed",
            stop_reason="max-steps",
            steps_run=1,
            validation_checks=3,
            best_validation_loss=1.5,
            best_validation_step=1,
            best_validation_check=1,
            protocol=_protocol(),
            validation_history=_history(),
        )

    with pytest.raises(
        TrainingRunValidationError,
        match="best_validation_loss must match validation_history",
    ):
        TrainingRunRecord(
            status="completed",
            stop_reason="max-steps",
            steps_run=1,
            validation_checks=2,
            best_validation_loss=0.5,
            best_validation_step=1,
            best_validation_check=1,
            protocol=_protocol(),
            validation_history=_history(),
        )


def test_training_protocol_rejects_unsupported_optimizer() -> None:
    with pytest.raises(TrainingRunValidationError, match="unsupported optimizer"):
        TrainingProtocol(
            kind="fixed-step-local-batch",
            objective="cross-entropy",
            optimizer="rmsprop",  # type: ignore[arg-type]
            learning_rate=0.01,
            schedule="none",
            seed=101,
            batch_size=2,
            max_steps=1,
            validation_interval=1,
            validation_sample_count=2,
            min_delta=0.0,
            patience=0,
            validation_source="training-batch",
        )


def _training_run() -> TrainingRunRecord:
    return TrainingRunRecord(
        status="budget-exhausted",
        stop_reason="max-steps",
        steps_run=1,
        validation_checks=2,
        best_validation_loss=1.5,
        best_validation_step=1,
        best_validation_check=1,
        protocol=_protocol(),
        validation_history=_history(),
    )


def _protocol() -> TrainingProtocol:
    return TrainingProtocol(
        kind="fixed-step-local-batch",
        objective="cross-entropy",
        optimizer="sgd",
        learning_rate=0.01,
        schedule="none",
        seed=101,
        batch_size=2,
        max_steps=1,
        validation_interval=1,
        validation_sample_count=2,
        min_delta=0.0,
        patience=0,
        validation_source="training-batch",
    )


def _history() -> tuple[TrainingHistoryPoint, ...]:
    return (
        TrainingHistoryPoint(
            step=0,
            validation_check=0,
            validation_loss=2.0,
            best_validation_loss=2.0,
            best_validation_step=0,
            best_validation_check=0,
            stale_checks=0,
            learning_rates=(0.01,),
        ),
        TrainingHistoryPoint(
            step=1,
            validation_check=1,
            validation_loss=1.5,
            best_validation_loss=1.5,
            best_validation_step=1,
            best_validation_check=1,
            stale_checks=0,
            learning_rates=(0.01,),
        ),
    )
