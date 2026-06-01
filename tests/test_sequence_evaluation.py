import math

from leibniz.identifiers import ProtocolIdentifier
from leibniz.prediction_results import TokenSequencePrediction, TokenSequenceProbability
from leibniz.prediction_spaces import FiniteTokenSequenceSpace, FiniteTokenVocabulary
from leibniz.sequence_evaluation import (
    ExactSequenceScore,
    SequenceMeasurementDocument,
    SequenceMeasurementRecord,
)


def test_exact_sequence_score_credits_only_the_full_accepted_sequence() -> None:
    prediction = _prediction()

    score = ExactSequenceScore.from_prediction(
        prediction=prediction,
        accepted_sequence=(1, 2, 3),
    )
    wrong_length = ExactSequenceScore.from_prediction(
        prediction=prediction,
        accepted_sequence=(1, 2),
    )

    assert score.accepted_mass == 0.25
    assert math.isclose(score.negative_log_score, -math.log(0.25))
    assert wrong_length.accepted_mass == 0.0
    assert wrong_length.negative_log_score == math.inf


def test_sequence_measurement_record_round_trips_exact_sequence_evidence() -> None:
    measurement = SequenceMeasurementRecord.from_prediction(
        benchmark_id=ProtocolIdentifier.parse("benchmarks.digits@0.1.0"),
        observation_id="benchmarks.digits.observations.sample-1@0.1.0",
        prediction=_prediction(),
        accepted_sequence=(1, 2, 3),
    )

    assert measurement.scoring_rule == "exact-sequence-probability"
    assert measurement.accepted_mass == 0.25
    assert SequenceMeasurementRecord.from_record(measurement.to_record()) == measurement
    assert (
        SequenceMeasurementDocument.from_bytes(
            b"{"
            b'"format":"leibniz.sequence-measurement",'
            b'"format_version":1,'
            b'"benchmark_id":"benchmarks.digits@0.1.0",'
            b'"observation_id":"benchmarks.digits.observations.sample-1@0.1.0",'
            b'"prediction":'
            + _prediction_record_bytes()
            + b","
            b'"accepted_sequence":[1,2,3],'
            b'"accepted_mass":0.25,'
            b'"negative_log_score":1.3862943611198906,'
            b'"scoring_rule":"exact-sequence-probability"'
            b"}"
        ).measurement
        == measurement
    )


def _prediction() -> TokenSequencePrediction:
    return TokenSequencePrediction(
        id=ProtocolIdentifier.parse("benchmarks.digits.predictions.sample-1@0.1.0"),
        prediction_space=FiniteTokenSequenceSpace(
            vocabulary=FiniteTokenVocabulary(token_count=10, token_name="digit"),
            sequence_boundary="eos-terminated",
        ),
        sequence_probabilities=(
            TokenSequenceProbability(tokens=(1, 2, 3), probability=0.25),
        ),
    )


def _prediction_record_bytes() -> bytes:
    return (
        b"{"
        b'"id":"benchmarks.digits.predictions.sample-1@0.1.0",'
        b'"prediction_space":{'
        b'"kind":"finite-token-sequence",'
        b'"vocabulary":{"token_count":10,"token_name":"digit"},'
        b'"sequence_boundary":"eos-terminated",'
        b'"minimum_length":1'
        b"},"
        b'"prediction_kind":"autoregressive-finite-token-sequence",'
        b'"output_encoding":"sequence-probability",'
        b'"sequence_probabilities":[{"tokens":[1,2,3],"probability":0.25}]'
        b"}"
    )
