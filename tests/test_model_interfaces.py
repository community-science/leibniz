from collections.abc import Callable

from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.model_interfaces import (
    ModelInterface,
    ModelInterfaceDocument,
    ModelInterfaceValidationError,
)
from leibniz.outcomes import OutcomeSpace


def test_model_interface_declares_finite_probability_measure_outputs() -> None:
    outcome_space = _outcome_space()

    interface = ModelInterface.from_record(_model_interface_record(), outcome_space=outcome_space)

    assert interface == ModelInterface(
        id=ProtocolIdentifier.parse("model-interfaces.boolean@0.1.0"),
        outcome_space_id=outcome_space.id,
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
    record["outcome_space_id"] = "core.other-outcome@0.1.0"

    assert str(
        capture_model_interface_error(
            lambda: ModelInterface.from_record(record, outcome_space=_outcome_space())
        )
    ) == "outcome_space_id core.other-outcome@0.1.0 does not match core.boolean-outcome@0.1.0"


def test_model_interface_rejects_unsupported_prediction_contracts() -> None:
    record = _model_interface_record()
    record["prediction_semantics"] = "logits"
    assert str(
        capture_model_interface_error(
            lambda: ModelInterface.from_record(record, outcome_space=_outcome_space())
        )
    ) == "prediction_semantics: expected literal 'finite-probability-measure'"

    record = _model_interface_record()
    record["output_encoding"] = "tensor"
    assert str(
        capture_model_interface_error(
            lambda: ModelInterface.from_record(record, outcome_space=_outcome_space())
        )
    ) == "output_encoding: expected literal 'probability-mass-sequence'"


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
        "outcome_space_id": "core.boolean-outcome@0.1.0",
        "prediction_semantics": "finite-probability-measure",
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
