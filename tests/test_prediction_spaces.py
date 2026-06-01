import pytest

from leibniz.identifiers import ProtocolIdentifier
from leibniz.prediction_spaces import (
    FiniteOutcomeSpace,
    FiniteTokenSequenceSpace,
    FiniteTokenVocabulary,
    RealVectorSpace,
    parse_prediction_space,
)


def test_finite_token_sequence_space_indexes_sequences_lexicographically() -> None:
    space = FiniteTokenSequenceSpace(
        vocabulary=FiniteTokenVocabulary(token_count=10, token_name="digit"),
        length=3,
    )

    assert space.cardinality == 1000
    assert space.sequence_index((1, 2, 3)) == 123
    assert space.sequence_for_index(123) == (1, 2, 3)
    assert space.outcome_id((1, 2, 3)) == "digit-1-2-3"


def test_finite_token_sequence_space_resolves_finite_outcomes() -> None:
    space = FiniteTokenSequenceSpace(
        vocabulary=FiniteTokenVocabulary(token_count=2, token_name="bit"),
        length=2,
    )

    outcome_space = space.outcome_space(
        id=ProtocolIdentifier.parse("benchmarks.bits.outcomes.l2@0.1.0")
    )

    assert [outcome.id for outcome in outcome_space.outcomes] == [
        "bit-0-0",
        "bit-0-1",
        "bit-1-0",
        "bit-1-1",
    ]
    assert space.finite_outcome_space(id=outcome_space.id) == FiniteOutcomeSpace(
        outcome_space_id=outcome_space.id,
        outcome_count=4,
        source_space=space.to_record(),
    )


def test_finite_token_sequence_space_requires_exact_length() -> None:
    space = FiniteTokenSequenceSpace(
        vocabulary=FiniteTokenVocabulary(token_count=10, token_name="digit"),
        length=2,
    )

    with pytest.raises(ValueError, match="length must be 2"):
        space.sequence_index((1,))


def test_eos_terminated_token_sequence_space_accepts_variable_lengths() -> None:
    space = FiniteTokenSequenceSpace(
        vocabulary=FiniteTokenVocabulary(token_count=10, token_name="digit"),
        sequence_boundary="eos-terminated",
    )

    assert space.require_sequence((1,)) == (1,)
    assert space.require_sequence((1, 2, 3)) == (1, 2, 3)
    assert FiniteTokenSequenceSpace.from_record(space.to_record()) == space
    assert space.to_record() == {
        "kind": "finite-token-sequence",
        "vocabulary": {"token_count": 10, "token_name": "digit"},
        "sequence_boundary": "eos-terminated",
        "minimum_length": 1,
    }
    with pytest.raises(ValueError, match="do not have finite cardinality"):
        _ = space.cardinality
    with pytest.raises(ValueError, match="do not have finite outcome indices"):
        space.sequence_index((1,))


def test_real_vector_space_records_continuous_prediction_targets() -> None:
    space = RealVectorSpace(dimension=2, coordinate_name="position")

    assert RealVectorSpace.from_record(space.to_record()) == space
    assert space.to_record() == {
        "kind": "real-vector",
        "dimension": 2,
        "coordinate_name": "position",
        "measure": "lebesgue",
    }


def test_parse_prediction_space_dispatches_public_space_kinds() -> None:
    sequence_space = FiniteTokenSequenceSpace(
        vocabulary=FiniteTokenVocabulary(token_count=2, token_name="bit"),
        length=2,
    )
    outcome_space = sequence_space.outcome_space(
        id=ProtocolIdentifier.parse("benchmarks.bits.outcomes.l2@0.1.0")
    )
    finite_outcome_space = sequence_space.finite_outcome_space(id=outcome_space.id)
    real_vector_space = RealVectorSpace(dimension=2, coordinate_name="position")

    assert parse_prediction_space(sequence_space.to_record()) == sequence_space
    assert parse_prediction_space(finite_outcome_space.to_record()) == finite_outcome_space
    assert parse_prediction_space(real_vector_space.to_record()) == real_vector_space

    with pytest.raises(ValueError, match="unsupported prediction space kind: other"):
        parse_prediction_space({"kind": "other"})


def test_finite_outcome_space_validates_outcome_count() -> None:
    space = FiniteTokenSequenceSpace(
        vocabulary=FiniteTokenVocabulary(token_count=2, token_name="bit"),
        length=2,
    )
    outcome_space = space.outcome_space(
        id=ProtocolIdentifier.parse("benchmarks.bits.outcomes.l2@0.1.0")
    )
    prediction_space = FiniteOutcomeSpace.from_outcome_space(
        outcome_space,
        source_space=space.to_record(),
    )

    prediction_space.validate_outcome_space(outcome_space)
    assert FiniteOutcomeSpace.from_record(prediction_space.to_record()) == prediction_space
