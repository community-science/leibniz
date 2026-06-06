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
        "training_evidence_count": 2,
        "max_steps": 1,
        "gate_check_interval": 1,
        "gate_evidence_count": 2,
        "gate_decision_rule": "score-estimate-plateau",
        "rung_competence_threshold": 0.01,
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
            "stale_checks": 0,
            "learning_rates": [0.01],
        },
        {
            "step": 1,
            "validation_check": 1,
            "validation_loss": 1.5,
            "stale_checks": 0,
            "learning_rates": [0.01],
        },
    ]
    assert record["training_compute"] == 128.0


def test_training_history_point_round_trips_score_estimate() -> None:
    point = TrainingHistoryPoint(
        step=3,
        validation_check=2,
        validation_loss=1.25,
        stale_checks=0,
        score_estimate={
            "kind": "training-running-score-estimate",
            "score": 4.5,
        },
    )

    record = point.to_record()
    parsed = TrainingHistoryPoint.from_record(record)

    assert parsed == point
    assert record["score_estimate"] == {
        "kind": "training-running-score-estimate",
        "score": 4.5,
    }


def test_training_run_record_rejects_inconsistent_validation_summary() -> None:
    with pytest.raises(
        TrainingRunValidationError,
        match="validation_checks must match validation_history length",
    ):
        TrainingRunRecord(
            status="completed",
            stop_reason="max-steps",
            steps_run=1,
            training_compute=128.0,
            validation_checks=3,
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
            training_evidence_count=2,
            max_steps=1,
            gate_check_interval=1,
            gate_evidence_count=2,
            gate_decision_rule="score-estimate-plateau",
            min_delta=0.0,
            patience=0,
            validation_source="training-batch",
        )


def test_training_protocol_requires_positive_gate_check_interval() -> None:
    with pytest.raises(TrainingRunValidationError, match="gate_check_interval"):
        TrainingProtocol(
            kind="fixed-step-local-batch",
            objective="cross-entropy",
            optimizer="sgd",
            learning_rate=0.01,
            schedule="none",
            seed=101,
            training_evidence_count=2,
            max_steps=1,
            gate_check_interval=0,
            gate_evidence_count=2,
            gate_decision_rule="score-estimate-plateau",
            min_delta=0.0,
            patience=0,
            validation_source="training-batch",
        )


def test_training_protocol_rejects_invalid_rung_competence_threshold() -> None:
    with pytest.raises(TrainingRunValidationError, match="rung_competence_threshold"):
        TrainingProtocol(
            kind="fixed-step-local-batch",
            objective="cross-entropy",
            optimizer="sgd",
            learning_rate=0.01,
            schedule="none",
            seed=101,
            training_evidence_count=2,
            max_steps=1,
            gate_check_interval=1,
            gate_evidence_count=2,
            gate_decision_rule="score-estimate-plateau",
            rung_competence_threshold=1.1,
            min_delta=0.0,
            patience=0,
            validation_source="training-batch",
        )


def test_training_protocol_defaults_missing_rung_competence_threshold() -> None:
    record = _protocol().to_record()
    del record["rung_competence_threshold"]

    parsed = TrainingProtocol.from_record(record)

    assert parsed.rung_competence_threshold == 0.01


def _training_run() -> TrainingRunRecord:
    return TrainingRunRecord(
        status="budget-exhausted",
        stop_reason="max-steps",
        steps_run=1,
        training_compute=128.0,
        validation_checks=2,
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
        training_evidence_count=2,
        max_steps=1,
        gate_check_interval=1,
        gate_evidence_count=2,
        gate_decision_rule="score-estimate-plateau",
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
            stale_checks=0,
            learning_rates=(0.01,),
        ),
        TrainingHistoryPoint(
            step=1,
            validation_check=1,
            validation_loss=1.5,
            stale_checks=0,
            learning_rates=(0.01,),
        ),
    )
