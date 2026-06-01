from collections.abc import Callable

import pytest

from leibniz.identifiers import ProtocolIdentifier
from leibniz.outcomes import FiniteProbabilityMeasure, Outcome, OutcomeSpace, ProbabilityMass
from leibniz.prediction_results import (
    DirectFiniteProbabilityPrediction,
    PredictionMass,
    PredictionResultContract,
    PredictionResultValidationError,
    TokenSequencePrediction,
    TokenSequenceProbability,
)
from leibniz.prediction_spaces import (
    FiniteOutcomeSpace,
    FiniteTokenSequenceSpace,
    FiniteTokenVocabulary,
)


def test_direct_finite_probability_prediction_records_indexed_mass_sequence() -> None:
    outcome_space = _outcome_space()
    prediction_space = FiniteOutcomeSpace.from_outcome_space(outcome_space)

    prediction = DirectFiniteProbabilityPrediction.from_probabilities(
        id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
        prediction_space=prediction_space,
        probabilities=(0.25, 0.75),
    )

    assert prediction.to_record() == {
        "id": "core.boolean-prediction@0.1.0",
        "prediction_space": {
            "kind": "finite-outcome-space",
            "outcome_space_id": "core.boolean-outcome@0.1.0",
            "outcome_count": 2,
        },
        "prediction_kind": "direct-finite-probability-measure",
        "output_encoding": "probability-mass-sequence",
        "probabilities": [
            {"outcome_index": 0, "probability": 0.25},
            {"outcome_index": 1, "probability": 0.75},
        ],
    }
    assert prediction.probability_at(0) == 0.25
    assert prediction.probability_at(1) == 0.75
    assert (
        DirectFiniteProbabilityPrediction.from_record(
            prediction.to_record(),
            outcome_space=outcome_space,
        )
        == prediction
    )
    assert prediction.contract == PredictionResultContract(
        prediction_space=prediction_space,
        prediction_kind="direct-finite-probability-measure",
        output_encoding="probability-mass-sequence",
    )


def test_prediction_result_contract_matches_interface_metadata() -> None:
    prediction_space = FiniteOutcomeSpace.from_outcome_space(_outcome_space())
    prediction = DirectFiniteProbabilityPrediction.from_probabilities(
        id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
        prediction_space=prediction_space,
        probabilities=(0.25, 0.75),
    )

    contract = PredictionResultContract.from_prediction(prediction)

    contract.require_matches(
        prediction_space=prediction_space,
        prediction_kind="direct-finite-probability-measure",
        output_encoding="probability-mass-sequence",
    )
    assert str(
        _capture_prediction_error(
            lambda: contract.require_matches(
                prediction_space=prediction_space,
                prediction_kind="other",
                output_encoding="probability-mass-sequence",
            )
        )
    ) == "prediction_kind does not match model interface"


def test_token_sequence_prediction_records_exact_sequence_probabilities() -> None:
    prediction_space = FiniteTokenSequenceSpace(
        vocabulary=FiniteTokenVocabulary(token_count=10, token_name="digit"),
        sequence_boundary="eos-terminated",
    )
    prediction = TokenSequencePrediction(
        id=ProtocolIdentifier.parse("benchmarks.digits.predictions.sample-1@0.1.0"),
        prediction_space=prediction_space,
        sequence_probabilities=(
            TokenSequenceProbability(tokens=(1, 2, 3), probability=0.7),
            TokenSequenceProbability(tokens=(1, 2), probability=0.2),
        ),
    )

    assert prediction.probability_of((1, 2, 3)) == 0.7
    assert prediction.probability_of((9,)) == 0.0
    assert prediction.contract == PredictionResultContract(
        prediction_space=prediction_space,
        prediction_kind="autoregressive-finite-token-sequence",
        output_encoding="sequence-probability",
    )
    assert TokenSequencePrediction.from_record(prediction.to_record()) == prediction


def test_direct_finite_probability_prediction_converts_to_probability_measure() -> None:
    outcome_space = _outcome_space()
    prediction_space = FiniteOutcomeSpace.from_outcome_space(outcome_space)
    prediction = DirectFiniteProbabilityPrediction.from_probabilities(
        id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
        prediction_space=prediction_space,
        probabilities=(0.0, 1.0),
    )

    measure = prediction.to_probability_measure(outcome_space=outcome_space)

    assert measure == FiniteProbabilityMeasure(
        id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
        outcome_space_id=outcome_space.id,
        probabilities=(ProbabilityMass("yes", 1.0),),
    )
    assert (
        DirectFiniteProbabilityPrediction.from_probability_measure(
            prediction_space=prediction_space,
            outcome_space=outcome_space,
            measure=measure,
        )
        == prediction
    )


def test_direct_finite_probability_prediction_rejects_invalid_probabilities() -> None:
    prediction_space = FiniteOutcomeSpace.from_outcome_space(_outcome_space())

    assert str(
        _capture_prediction_error(
            lambda: DirectFiniteProbabilityPrediction.from_probabilities(
                id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
                prediction_space=prediction_space,
                probabilities=(1.0,),
            )
        )
    ) == (
        "probability sequence length 1 does not match prediction space outcome_count 2"
    )
    assert str(
        _capture_prediction_error(
            lambda: DirectFiniteProbabilityPrediction(
                id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
                prediction_space=prediction_space,
                probabilities=(
                    PredictionMass(0, 0.25),
                    PredictionMass(0, 0.75),
                ),
            )
        )
    ) == "outcome indices must be unique"
    assert str(
        _capture_prediction_error(
            lambda: DirectFiniteProbabilityPrediction(
                id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
                prediction_space=prediction_space,
                probabilities=(PredictionMass(2, 1.0),),
            )
        )
    ) == "outcome_index 2 is outside prediction space"
    assert str(
        _capture_prediction_error(
            lambda: DirectFiniteProbabilityPrediction.from_probabilities(
                id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
                prediction_space=prediction_space,
                probabilities=(0.5, 0.4),
            )
        )
    ) == "probabilities must sum to 1 within tolerance 1e-12; got 0.9"


def test_direct_finite_probability_prediction_rejects_mismatched_outcome_space() -> None:
    record = DirectFiniteProbabilityPrediction.from_probabilities(
        id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
        prediction_space=FiniteOutcomeSpace.from_outcome_space(_outcome_space()),
        probabilities=(0.5, 0.5),
    ).to_record()

    assert str(
        _capture_prediction_error(
            lambda: DirectFiniteProbabilityPrediction.from_record(
                record,
                outcome_space=OutcomeSpace(
                    id=ProtocolIdentifier.parse("core.other-outcome@0.1.0"),
                    outcomes=(Outcome("no"), Outcome("yes")),
                ),
            )
        )
    ) == "outcome_space_id core.boolean-outcome@0.1.0 does not match core.other-outcome@0.1.0"


def _outcome_space() -> OutcomeSpace:
    return OutcomeSpace(
        id=ProtocolIdentifier.parse("core.boolean-outcome@0.1.0"),
        outcomes=(Outcome("no"), Outcome("yes")),
    )


def _capture_prediction_error(
    call: Callable[[], object],
) -> PredictionResultValidationError:
    with pytest.raises(PredictionResultValidationError) as exc_info:
        call()
    return exc_info.value
