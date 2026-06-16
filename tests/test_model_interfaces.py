from collections.abc import Callable
from typing import cast

from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.model_interfaces import (
    ModelInterface,
    ModelInterfaceDocument,
    ModelInterfaceValidationError,
)
from leibniz.outcomes import OutcomeSpace
from leibniz.prediction_results import (
    DirectFiniteProbabilityPrediction,
    TokenSequencePrediction,
    TokenSequenceProbability,
)
from leibniz.prediction_spaces import (
    FiniteOutcomeSpace,
    FiniteTokenSequenceSpace,
    FiniteTokenVocabulary,
    RealVectorSpace,
)


def test_model_interface_declares_finite_probability_measure_outputs() -> None:
    outcome_space = _outcome_space()

    interface = ModelInterface.from_record(_model_interface_record(), outcome_space=outcome_space)

    assert interface == ModelInterface(
        id=ProtocolIdentifier.parse("model-interfaces.boolean@0.1.0"),
        prediction_space=FiniteOutcomeSpace.from_outcome_space(outcome_space),
    )
    assert interface.to_record() == _model_interface_record()
    assert interface.digest == ContentDigest.from_value(interface.to_record())


def test_model_interface_from_outcome_space_canonicalizes() -> None:
    outcome_space = _outcome_space()

    interface = ModelInterface.from_outcome_space(
        id=ProtocolIdentifier.parse("model-interfaces.boolean@0.1.0"),
        outcome_space=outcome_space,
    )

    assert interface.to_record() == _model_interface_record()


def test_model_interface_can_record_finite_token_sequence_source_space() -> None:
    sequence_space = FiniteTokenSequenceSpace(
        vocabulary=FiniteTokenVocabulary(token_count=2, token_name="bit"),
        length=1,
    )
    outcome_space = sequence_space.outcome_space(
        id=ProtocolIdentifier.parse("core.boolean-outcome@0.1.0")
    )

    interface = ModelInterface.from_outcome_space(
        id=ProtocolIdentifier.parse("model-interfaces.boolean@0.1.0"),
        outcome_space=outcome_space,
        source_space=sequence_space.to_record(),
    )

    assert interface.to_record()["prediction_space"] == {
        "kind": "finite-outcome-space",
        "outcome_space_id": "core.boolean-outcome@0.1.0",
        "outcome_count": 2,
        "source_space": sequence_space.to_record(),
    }


def test_model_interface_declares_real_vector_outputs() -> None:
    interface = ModelInterface.from_real_vector_space(
        id=ProtocolIdentifier.parse("model-interfaces.inverse@0.1.0"),
        dimension=15,
        coordinate_name="target-coordinate",
    )

    assert interface == ModelInterface(
        id=ProtocolIdentifier.parse("model-interfaces.inverse@0.1.0"),
        prediction_space=RealVectorSpace(
            dimension=15,
            coordinate_name="target-coordinate",
        ),
        prediction_kind="direct-real-vector",
        output_encoding="coordinate-sequence",
    )
    assert ModelInterface.from_record(interface.to_record()) == interface


def test_model_interface_validates_direct_finite_prediction_results() -> None:
    outcome_space = _outcome_space()
    interface = ModelInterface.from_record(_model_interface_record(), outcome_space=outcome_space)
    assert isinstance(interface.prediction_space, FiniteOutcomeSpace)
    prediction = DirectFiniteProbabilityPrediction.from_probabilities(
        id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
        prediction_space=interface.prediction_space,
        probabilities=(0.25, 0.75),
    )

    interface.validate_prediction_result(prediction)

    other_space = OutcomeSpace.from_record(
        {
            "id": "core.other-outcome@0.1.0",
            "outcomes": [{"id": "yes"}, {"id": "no"}],
        }
    )
    mismatched_prediction = DirectFiniteProbabilityPrediction.from_probabilities(
        id=ProtocolIdentifier.parse("core.other-prediction@0.1.0"),
        prediction_space=FiniteOutcomeSpace.from_outcome_space(other_space),
        probabilities=(0.25, 0.75),
    )
    assert str(
        capture_model_interface_error(
            lambda: interface.validate_prediction_result(mismatched_prediction)
        )
    ) == "prediction_space does not match model interface"
    assert str(
        capture_model_interface_error(lambda: interface.validate_prediction_result(object()))
    ) == "prediction result does not expose prediction_space"


def test_model_interface_declares_autoregressive_token_sequence_outputs() -> None:
    prediction_space = FiniteTokenSequenceSpace(
        vocabulary=FiniteTokenVocabulary(token_count=10, token_name="digit"),
        sequence_boundary="eos-terminated",
    )
    interface = ModelInterface(
        id=ProtocolIdentifier.parse("model-interfaces.digits-sequence@0.1.0"),
        prediction_space=prediction_space,
        prediction_kind="autoregressive-finite-token-sequence",
        output_encoding="sequence-probability",
    )
    prediction = TokenSequencePrediction(
        id=ProtocolIdentifier.parse("benchmarks.digits.predictions.sample-1@0.1.0"),
        prediction_space=prediction_space,
        sequence_probabilities=(
            TokenSequenceProbability(tokens=(1, 2, 3), probability=0.75),
        ),
    )

    interface.validate_prediction_result(prediction)
    assert ModelInterface.from_record(
        interface.to_record(),
        outcome_space=_outcome_space(),
    ) == interface


def test_model_interface_document_loads_bytes_with_digest() -> None:
    outcome_space = _outcome_space()

    document = ModelInterfaceDocument.from_bytes(
        canonical_document_bytes(_model_interface_record()),
        outcome_space=outcome_space,
    )

    assert document.interface == ModelInterface.from_record(
        _model_interface_record(),
        outcome_space=outcome_space,
    )
    assert document.digest == ContentDigest.from_value(document.interface.to_record())


def test_model_interface_rejects_unknown_outcome_space() -> None:
    record = _model_interface_record()
    prediction_space = dict(cast(dict[str, object], record["prediction_space"]))
    prediction_space["outcome_space_id"] = "core.other-outcome@0.1.0"
    record["prediction_space"] = prediction_space

    assert str(
        capture_model_interface_error(
            lambda: ModelInterface.from_record(record, outcome_space=_outcome_space())
        )
    ) == "outcome_space_id core.other-outcome@0.1.0 does not match core.boolean-outcome@0.1.0"


def test_model_interface_rejects_unsupported_prediction_contracts() -> None:
    record = _model_interface_record()
    record["prediction_kind"] = "logits"
    assert str(
        capture_model_interface_error(
            lambda: ModelInterface.from_record(record, outcome_space=_outcome_space())
        )
    ) == (
        "model interface must pair finite probability measures with finite outcome spaces, "
        "or autoregressive sequence probabilities with eos-terminated finite token "
        "sequence spaces, or direct real-vector outputs with real vector spaces"
    )

    record = _model_interface_record()
    record["output_encoding"] = "tensor"
    assert str(
        capture_model_interface_error(
            lambda: ModelInterface.from_record(record, outcome_space=_outcome_space())
        )
    ) == (
        "model interface must pair finite probability measures with finite outcome spaces, "
        "or autoregressive sequence probabilities with eos-terminated finite token "
        "sequence spaces, or direct real-vector outputs with real vector spaces"
    )


def test_model_interface_rejects_scoring_benchmark_and_tensor_fields() -> None:
    record = _model_interface_record()
    record["score_function"] = "negative-log-accepted-mass"
    assert str(
        capture_model_interface_error(
            lambda: ModelInterface.from_record(record, outcome_space=_outcome_space())
        )
    ) == "score_function: unknown field"

    record = _model_interface_record()
    record["benchmark_id"] = "core.boolean-benchmark@0.1.0"
    assert str(
        capture_model_interface_error(
            lambda: ModelInterface.from_record(record, outcome_space=_outcome_space())
        )
    ) == "benchmark_id: unknown field"

    record = _model_interface_record()
    record["output_shape"] = [2]
    assert str(
        capture_model_interface_error(
            lambda: ModelInterface.from_record(record, outcome_space=_outcome_space())
        )
    ) == "output_shape: unknown field"


def test_model_interface_rejects_invalid_ids_and_documents() -> None:
    record = _model_interface_record()
    record["id"] = "core.boolean-interface@0.1.0"
    assert str(
        capture_model_interface_error(
            lambda: ModelInterface.from_record(record, outcome_space=_outcome_space())
        )
    ) == "id must be a valid model interface id"

    assert str(
        capture_model_interface_error(
            lambda: ModelInterfaceDocument.from_bytes(b"[]", outcome_space=_outcome_space())
        )
    ) == "model interface document must contain an object"


def _model_interface_record() -> dict[str, object]:
    return {
        "id": "model-interfaces.boolean@0.1.0",
        "prediction_space": {
            "kind": "finite-outcome-space",
            "outcome_space_id": "core.boolean-outcome@0.1.0",
            "outcome_count": 2,
        },
        "prediction_kind": "direct-finite-probability-measure",
        "output_encoding": "probability-mass-sequence",
    }


def _outcome_space() -> OutcomeSpace:
    return OutcomeSpace.from_record(
        {
            "id": "core.boolean-outcome@0.1.0",
            "outcomes": [{"id": "yes"}, {"id": "no"}],
        }
    )


def capture_model_interface_error(
    action: Callable[[], object],
) -> ModelInterfaceValidationError:
    try:
        action()
    except ModelInterfaceValidationError as error:
        return error
    raise AssertionError("expected ModelInterfaceValidationError")
