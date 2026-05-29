import pytest

from leibniz.identifiers import ProtocolIdentifier
from leibniz.prediction_spaces import (
    FiniteTokenSequenceSpace,
    FiniteTokenVocabulary,
    RealVectorSpace,
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


def test_finite_token_sequence_space_requires_exact_length() -> None:
    space = FiniteTokenSequenceSpace(
        vocabulary=FiniteTokenVocabulary(token_count=10, token_name="digit"),
        length=2,
    )

    with pytest.raises(ValueError, match="length must be 2"):
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
