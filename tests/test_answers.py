import math
from collections.abc import Callable, Mapping
from typing import cast

from leibniz.answers import (
    AcceptedEvent,
    AcceptedEventValidationError,
    AcceptedMassScore,
    AcceptedMassScoreError,
    AnswerElement,
    AnswerSpace,
    AnswerSpaceValidationError,
    FiniteAnswerScoringBundle,
    FiniteAnswerScoringBundleValidationError,
    FiniteProbabilityMeasure,
    ProbabilityMass,
    ProbabilityMeasureValidationError,
    RawScoringEvidence,
    RawScoringEvidenceValidationError,
)
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier


def test_answer_element_parses_record() -> None:
    element = AnswerElement.from_record({"id": "yes"})

    assert element == AnswerElement(id="yes")
    assert element.to_record() == {"id": "yes"}


def test_answer_element_rejects_invalid_id_and_unknown_fields() -> None:
    assert str(capture_answer_error(lambda: AnswerElement.from_record({"id": "Yes"}))) == (
        "invalid answer element id: 'Yes'"
    )
    assert str(
        capture_answer_error(
            lambda: AnswerElement.from_record({"id": "yes", "label": "Yes"})
        )
    ) == "label: unknown field"


def test_answer_space_parses_nonempty_finite_space() -> None:
    space = AnswerSpace.from_record(
        {
            "id": "core.boolean-answer@0.1.0",
            "elements": [{"id": "yes"}, {"id": "no"}],
        }
    )

    assert space == AnswerSpace(
        id=ProtocolIdentifier.parse("core.boolean-answer@0.1.0"),
        elements=(AnswerElement("yes"), AnswerElement("no")),
    )
    assert space.element_ids == frozenset({"yes", "no"})
    assert space.contains("yes")
    assert not space.contains("maybe")
    assert space.to_record() == {
        "id": "core.boolean-answer@0.1.0",
        "elements": [{"id": "yes"}, {"id": "no"}],
    }


def test_answer_space_rejects_empty_elements() -> None:
    error = capture_answer_error(
        lambda: AnswerSpace.from_record({"id": "core.empty-answer@0.1.0", "elements": []})
    )

    assert str(error) == "answer space must contain at least one element"


def test_answer_space_rejects_duplicate_element_ids() -> None:
    error = capture_answer_error(
        lambda: AnswerSpace.from_record(
            {
                "id": "core.duplicate-answer@0.1.0",
                "elements": [{"id": "yes"}, {"id": "yes"}],
            }
        )
    )

    assert str(error) == "answer element ids must be unique"


def test_answer_space_rejects_released_identifier() -> None:
    error = capture_answer_error(
        lambda: AnswerSpace.from_record(
            {"id": "core.boolean-answer@1.0.0", "elements": [{"id": "yes"}]}
        )
    )

    assert str(error) == (
        "identifier must use a pre-1.0.0 version before release policy exists: "
        "core.boolean-answer@1.0.0"
    )


def test_answer_space_rejects_malformed_records() -> None:
    assert str(
        capture_answer_error(lambda: AnswerSpace.from_record({"elements": [{"id": "yes"}]}))
    ) == "id: missing required field"
    assert str(
        capture_answer_error(
            lambda: AnswerSpace.from_record(
                {
                    "id": "core.boolean-answer@0.1.0",
                    "elements": [{"id": "yes", "text": "Yes"}],
                }
            )
        )
    ) == "elements.0.text: unknown field"


def test_accepted_event_parses_nonempty_subset() -> None:
    space = AnswerSpace.from_record(
        {
            "id": "core.boolean-answer@0.1.0",
            "elements": [{"id": "yes"}, {"id": "no"}, {"id": "maybe"}],
        }
    )

    event = AcceptedEvent.from_record(
        {
            "id": "core.boolean-accepted@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "elements": ["maybe", "yes"],
        },
        answer_space=space,
    )

    assert event == AcceptedEvent(
        id=ProtocolIdentifier.parse("core.boolean-accepted@0.1.0"),
        answer_space_id=ProtocolIdentifier.parse("core.boolean-answer@0.1.0"),
        elements=frozenset({"maybe", "yes"}),
    )
    assert event.accepts("yes")
    assert not event.accepts("no")
    assert event.to_record() == {
        "id": "core.boolean-accepted@0.1.0",
        "answer_space_id": "core.boolean-answer@0.1.0",
        "elements": ["maybe", "yes"],
    }


def test_accepted_event_rejects_empty_event() -> None:
    space = _boolean_space()

    error = capture_event_error(
        lambda: AcceptedEvent.from_record(
            {
                "id": "core.boolean-accepted@0.1.0",
                "answer_space_id": "core.boolean-answer@0.1.0",
                "elements": [],
            },
            answer_space=space,
        )
    )

    assert str(error) == "accepted event must contain at least one element"


def test_accepted_event_rejects_unknown_elements() -> None:
    space = _boolean_space()

    error = capture_event_error(
        lambda: AcceptedEvent.from_record(
            {
                "id": "core.boolean-accepted@0.1.0",
                "answer_space_id": "core.boolean-answer@0.1.0",
                "elements": ["maybe"],
            },
            answer_space=space,
        )
    )

    assert str(error) == "accepted elements are not in answer space: maybe"


def test_accepted_event_rejects_mismatched_answer_space() -> None:
    space = _boolean_space()

    error = capture_event_error(
        lambda: AcceptedEvent.from_record(
            {
                "id": "core.boolean-accepted@0.1.0",
                "answer_space_id": "core.other-answer@0.1.0",
                "elements": ["yes"],
            },
            answer_space=space,
        )
    )

    assert str(error) == (
        "answer_space_id core.other-answer@0.1.0 does not match core.boolean-answer@0.1.0"
    )


def test_accepted_event_rejects_invalid_element_ids_and_malformed_records() -> None:
    space = _boolean_space()

    assert str(
        capture_event_error(
            lambda: AcceptedEvent.from_record(
                {
                    "id": "core.boolean-accepted@0.1.0",
                    "answer_space_id": "core.boolean-answer@0.1.0",
                    "elements": ["Yes"],
                },
                answer_space=space,
            )
        )
    ) == "invalid accepted element id: 'Yes'"
    assert str(
        capture_event_error(
            lambda: AcceptedEvent.from_record(
                {
                    "id": "core.boolean-accepted@0.1.0",
                    "answer_space_id": "core.boolean-answer@0.1.0",
                    "elements": ["yes"],
                    "extra": True,
                },
                answer_space=space,
            )
        )
    ) == "extra: unknown field"


def test_accepted_event_rejects_duplicate_elements() -> None:
    space = _boolean_space()

    error = capture_event_error(
        lambda: AcceptedEvent.from_record(
            {
                "id": "core.boolean-accepted@0.1.0",
                "answer_space_id": "core.boolean-answer@0.1.0",
                "elements": ["yes", "yes"],
            },
            answer_space=space,
        )
    )

    assert str(error) == "accepted element ids must be unique"


def test_accepted_event_rejects_released_identifier() -> None:
    space = _boolean_space()

    error = capture_event_error(
        lambda: AcceptedEvent.from_record(
            {
                "id": "core.boolean-accepted@1.0.0",
                "answer_space_id": "core.boolean-answer@0.1.0",
                "elements": ["yes"],
            },
            answer_space=space,
        )
    )

    assert str(error) == (
        "identifier must use a pre-1.0.0 version before release policy exists: "
        "core.boolean-accepted@1.0.0"
    )


def test_probability_mass_parses_record() -> None:
    mass = ProbabilityMass.from_record({"element_id": "yes", "probability": 0.25})

    assert mass == ProbabilityMass(element_id="yes", probability=0.25)
    assert mass.to_record() == {"element_id": "yes", "probability": 0.25}


def test_probability_measure_parses_sparse_normalized_measure() -> None:
    space = AnswerSpace.from_record(
        {
            "id": "core.boolean-answer@0.1.0",
            "elements": [{"id": "yes"}, {"id": "no"}, {"id": "maybe"}],
        }
    )

    measure = FiniteProbabilityMeasure.from_record(
        {
            "id": "core.boolean-prediction@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "probabilities": [
                {"element_id": "yes", "probability": 0.75},
                {"element_id": "no", "probability": 0.25},
            ],
        },
        answer_space=space,
    )

    assert measure == FiniteProbabilityMeasure(
        id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
        answer_space_id=ProtocolIdentifier.parse("core.boolean-answer@0.1.0"),
        probabilities=(
            ProbabilityMass("yes", 0.75),
            ProbabilityMass("no", 0.25),
        ),
    )
    assert measure.total_probability == 1.0
    assert measure.probability_of("yes") == 0.75
    assert measure.probability_of("maybe") == 0.0
    assert measure.to_record() == {
        "id": "core.boolean-prediction@0.1.0",
        "answer_space_id": "core.boolean-answer@0.1.0",
        "probabilities": [
            {"element_id": "no", "probability": 0.25},
            {"element_id": "yes", "probability": 0.75},
        ],
    }


def test_probability_measure_uses_explicit_normalization_tolerance() -> None:
    space = AnswerSpace.from_record(
        {"id": "core.boolean-answer@0.1.0", "elements": [{"id": "yes"}, {"id": "no"}]}
    )

    measure = FiniteProbabilityMeasure.from_record(
        {
            "id": "core.boolean-prediction@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "probabilities": [
                {"element_id": "yes", "probability": 0.5},
                {"element_id": "no", "probability": 0.500000001},
            ],
        },
        answer_space=space,
        normalization_tolerance=1e-8,
    )

    assert measure.total_probability == 1.000000001


def test_probability_measure_rejects_not_normalized_probability() -> None:
    space = _boolean_space()

    error = capture_measure_error(
        lambda: FiniteProbabilityMeasure.from_record(
            {
                "id": "core.boolean-prediction@0.1.0",
                "answer_space_id": "core.boolean-answer@0.1.0",
                "probabilities": [{"element_id": "yes", "probability": 0.9}],
            },
            answer_space=space,
        )
    )

    assert str(error) == "probabilities must sum to 1 within tolerance 1e-12; got 0.9"


def test_probability_measure_rejects_negative_nonfinite_and_invalid_probabilities() -> None:
    space = _boolean_space()

    assert str(
        capture_measure_error(
            lambda: FiniteProbabilityMeasure.from_record(
                {
                    "id": "core.boolean-prediction@0.1.0",
                    "answer_space_id": "core.boolean-answer@0.1.0",
                    "probabilities": [{"element_id": "yes", "probability": -1.0}],
                },
                answer_space=space,
            )
        )
    ) == "probability must be nonnegative"
    assert str(
        capture_measure_error(
            lambda: ProbabilityMass.from_record(
                {"element_id": "yes", "probability": float("inf")}
            )
        )
    ) == "probability: expected finite number"
    assert str(
        capture_measure_error(
            lambda: FiniteProbabilityMeasure.from_record(
                {
                    "id": "core.boolean-prediction@0.1.0",
                    "answer_space_id": "core.boolean-answer@0.1.0",
                    "probabilities": [{"element_id": "Yes", "probability": 1.0}],
                },
                answer_space=space,
            )
        )
    ) == "invalid probability element id: 'Yes'"


def test_probability_measure_rejects_unknown_and_duplicate_elements() -> None:
    space = _boolean_space()

    assert str(
        capture_measure_error(
            lambda: FiniteProbabilityMeasure.from_record(
                {
                    "id": "core.boolean-prediction@0.1.0",
                    "answer_space_id": "core.boolean-answer@0.1.0",
                    "probabilities": [{"element_id": "maybe", "probability": 1.0}],
                },
                answer_space=space,
            )
        )
    ) == "probability elements are not in answer space: maybe"
    assert str(
        capture_measure_error(
            lambda: FiniteProbabilityMeasure.from_record(
                {
                    "id": "core.boolean-prediction@0.1.0",
                    "answer_space_id": "core.boolean-answer@0.1.0",
                    "probabilities": [
                        {"element_id": "yes", "probability": 0.5},
                        {"element_id": "yes", "probability": 0.5},
                    ],
                },
                answer_space=space,
            )
        )
    ) == "probability element ids must be unique"


def test_probability_measure_rejects_mismatched_answer_space_and_released_identifier() -> None:
    space = _boolean_space()

    assert str(
        capture_measure_error(
            lambda: FiniteProbabilityMeasure.from_record(
                {
                    "id": "core.boolean-prediction@0.1.0",
                    "answer_space_id": "core.other-answer@0.1.0",
                    "probabilities": [{"element_id": "yes", "probability": 1.0}],
                },
                answer_space=space,
            )
        )
    ) == "answer_space_id core.other-answer@0.1.0 does not match core.boolean-answer@0.1.0"
    assert str(
        capture_measure_error(
            lambda: FiniteProbabilityMeasure.from_record(
                {
                    "id": "core.boolean-prediction@1.0.0",
                    "answer_space_id": "core.boolean-answer@0.1.0",
                    "probabilities": [{"element_id": "yes", "probability": 1.0}],
                },
                answer_space=space,
            )
        )
    ) == (
        "identifier must use a pre-1.0.0 version before release policy exists: "
        "core.boolean-prediction@1.0.0"
    )


def test_probability_measure_rejects_malformed_records_and_tolerance() -> None:
    space = _boolean_space()

    assert str(
        capture_measure_error(
            lambda: FiniteProbabilityMeasure.from_record(
                {
                    "id": "core.boolean-prediction@0.1.0",
                    "answer_space_id": "core.boolean-answer@0.1.0",
                    "probabilities": [{"element_id": "yes", "probability": 1.0, "extra": True}],
                },
                answer_space=space,
            )
        )
    ) == "probabilities.0.extra: unknown field"
    assert str(
        capture_measure_error(
            lambda: FiniteProbabilityMeasure.from_record(
                {
                    "id": "core.boolean-prediction@0.1.0",
                    "answer_space_id": "core.boolean-answer@0.1.0",
                    "probabilities": [{"element_id": "yes", "probability": 1.0}],
                },
                answer_space=space,
                normalization_tolerance=-1.0,
            )
        )
    ) == "normalization tolerance must be finite and nonnegative"


def test_accepted_mass_score_sums_event_probability_mass() -> None:
    space = AnswerSpace.from_record(
        {
            "id": "core.boolean-answer@0.1.0",
            "elements": [{"id": "yes"}, {"id": "no"}, {"id": "maybe"}],
        }
    )
    event = AcceptedEvent.from_record(
        {
            "id": "core.boolean-accepted@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "elements": ["yes", "maybe"],
        },
        answer_space=space,
    )
    measure = FiniteProbabilityMeasure.from_record(
        {
            "id": "core.boolean-prediction@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "probabilities": [
                {"element_id": "yes", "probability": 0.25},
                {"element_id": "no", "probability": 0.5},
                {"element_id": "maybe", "probability": 0.25},
            ],
        },
        answer_space=space,
    )

    score = AcceptedMassScore.from_event_and_measure(event=event, measure=measure)

    assert score == AcceptedMassScore(
        accepted_mass=0.5,
        negative_log_score=-math.log(0.5),
    )


def test_accepted_mass_score_handles_single_accepted_element() -> None:
    space = AnswerSpace.from_record(
        {
            "id": "core.boolean-answer@0.1.0",
            "elements": [{"id": "yes"}, {"id": "no"}],
        }
    )
    event = AcceptedEvent.from_record(
        {
            "id": "core.boolean-accepted@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "elements": ["yes"],
        },
        answer_space=space,
    )
    measure = FiniteProbabilityMeasure.from_record(
        {
            "id": "core.boolean-prediction@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "probabilities": [
                {"element_id": "yes", "probability": 0.8},
                {"element_id": "no", "probability": 0.2},
            ],
        },
        answer_space=space,
    )

    score = AcceptedMassScore.from_event_and_measure(event=event, measure=measure)

    assert score == AcceptedMassScore(
        accepted_mass=0.8,
        negative_log_score=-math.log(0.8),
    )


def test_accepted_mass_score_treats_omitted_accepted_elements_as_zero_mass() -> None:
    space = AnswerSpace.from_record(
        {
            "id": "core.boolean-answer@0.1.0",
            "elements": [{"id": "yes"}, {"id": "no"}],
        }
    )
    event = AcceptedEvent.from_record(
        {
            "id": "core.boolean-accepted@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "elements": ["yes"],
        },
        answer_space=space,
    )
    measure = FiniteProbabilityMeasure.from_record(
        {
            "id": "core.boolean-prediction@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "probabilities": [{"element_id": "no", "probability": 1.0}],
        },
        answer_space=space,
    )

    score = AcceptedMassScore.from_event_and_measure(event=event, measure=measure)

    assert score == AcceptedMassScore(accepted_mass=0.0, negative_log_score=math.inf)


def test_accepted_mass_score_rejects_mismatched_answer_space() -> None:
    event = AcceptedEvent(
        id=ProtocolIdentifier.parse("core.boolean-accepted@0.1.0"),
        answer_space_id=ProtocolIdentifier.parse("core.boolean-answer@0.1.0"),
        elements=frozenset({"yes"}),
    )
    measure = FiniteProbabilityMeasure(
        id=ProtocolIdentifier.parse("core.other-prediction@0.1.0"),
        answer_space_id=ProtocolIdentifier.parse("core.other-answer@0.1.0"),
        probabilities=(ProbabilityMass("yes", 1.0),),
    )

    error = capture_score_error(
        lambda: AcceptedMassScore.from_event_and_measure(event=event, measure=measure)
    )

    assert str(error) == (
        "accepted event answer_space_id core.boolean-answer@0.1.0 does not match "
        "probability measure core.other-answer@0.1.0"
    )


def test_accepted_mass_score_clamps_tolerance_sized_roundoff_above_one() -> None:
    event = AcceptedEvent(
        id=ProtocolIdentifier.parse("core.boolean-accepted@0.1.0"),
        answer_space_id=ProtocolIdentifier.parse("core.boolean-answer@0.1.0"),
        elements=frozenset({"yes", "no"}),
    )
    measure = FiniteProbabilityMeasure(
        id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
        answer_space_id=ProtocolIdentifier.parse("core.boolean-answer@0.1.0"),
        probabilities=(
            ProbabilityMass("yes", 0.5),
            ProbabilityMass("no", 0.5000000000001),
        ),
        normalization_tolerance=1e-12,
    )

    score = AcceptedMassScore.from_event_and_measure(event=event, measure=measure)

    assert score == AcceptedMassScore(accepted_mass=1.0, negative_log_score=-0.0)


def test_raw_scoring_evidence_recomputes_score_from_event_and_measure() -> None:
    space = AnswerSpace.from_record(
        {
            "id": "core.boolean-answer@0.1.0",
            "elements": [{"id": "yes"}, {"id": "no"}],
        }
    )
    event = AcceptedEvent.from_record(
        {
            "id": "core.boolean-accepted@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "elements": ["yes"],
        },
        answer_space=space,
    )
    measure = FiniteProbabilityMeasure.from_record(
        {
            "id": "core.boolean-prediction@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "probabilities": [
                {"element_id": "yes", "probability": 0.25},
                {"element_id": "no", "probability": 0.75},
            ],
        },
        answer_space=space,
    )

    evidence = RawScoringEvidence.from_event_and_measure(
        id=ProtocolIdentifier.parse("core.boolean-evidence@0.1.0"),
        observation_id="observation-1",
        event=event,
        measure=measure,
    )

    assert evidence == RawScoringEvidence(
        id=ProtocolIdentifier.parse("core.boolean-evidence@0.1.0"),
        observation_id="observation-1",
        answer_space_id=ProtocolIdentifier.parse("core.boolean-answer@0.1.0"),
        accepted_event_id=ProtocolIdentifier.parse("core.boolean-accepted@0.1.0"),
        probability_measure_id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
        accepted_mass=0.25,
        negative_log_score=-math.log(0.25),
    )
    assert evidence.to_record() == {
        "id": "core.boolean-evidence@0.1.0",
        "observation_id": "observation-1",
        "answer_space_id": "core.boolean-answer@0.1.0",
        "accepted_event_id": "core.boolean-accepted@0.1.0",
        "probability_measure_id": "core.boolean-prediction@0.1.0",
        "accepted_mass": 0.25,
        "negative_log_score": -math.log(0.25),
    }


def test_raw_scoring_evidence_parses_infinite_zero_mass_score() -> None:
    evidence = RawScoringEvidence.from_record(
        {
            "id": "core.boolean-evidence@0.1.0",
            "observation_id": "observation-1",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "accepted_event_id": "core.boolean-accepted@0.1.0",
            "probability_measure_id": "core.boolean-prediction@0.1.0",
            "accepted_mass": 0.0,
            "negative_log_score": "infinity",
        }
    )

    assert evidence.negative_log_score == math.inf
    assert evidence.to_record()["negative_log_score"] == "infinity"


def test_raw_scoring_evidence_rejects_aggregate_summary_fields() -> None:
    error = capture_evidence_error(
        lambda: RawScoringEvidence.from_record(
            {
                "id": "core.boolean-evidence@0.1.0",
                "observation_id": "observation-1",
                "answer_space_id": "core.boolean-answer@0.1.0",
                "accepted_event_id": "core.boolean-accepted@0.1.0",
                "probability_measure_id": "core.boolean-prediction@0.1.0",
                "accepted_mass": 0.25,
                "negative_log_score": -math.log(0.25),
                "mean_score": 0.25,
            }
        )
    )

    assert str(error) == "mean_score: unknown field"


def test_raw_scoring_evidence_rejects_inconsistent_zero_mass_score() -> None:
    error = capture_evidence_error(
        lambda: RawScoringEvidence.from_record(
            {
                "id": "core.boolean-evidence@0.1.0",
                "observation_id": "observation-1",
                "answer_space_id": "core.boolean-answer@0.1.0",
                "accepted_event_id": "core.boolean-accepted@0.1.0",
                "probability_measure_id": "core.boolean-prediction@0.1.0",
                "accepted_mass": 0.0,
                "negative_log_score": 1.0,
            }
        )
    )

    assert str(error) == "zero accepted_mass requires infinite negative_log_score"


def test_raw_scoring_evidence_rejects_inconsistent_finite_score() -> None:
    error = capture_evidence_error(
        lambda: RawScoringEvidence.from_record(
            {
                "id": "core.boolean-evidence@0.1.0",
                "observation_id": "observation-1",
                "answer_space_id": "core.boolean-answer@0.1.0",
                "accepted_event_id": "core.boolean-accepted@0.1.0",
                "probability_measure_id": "core.boolean-prediction@0.1.0",
                "accepted_mass": 0.25,
                "negative_log_score": 1.0,
            }
        )
    )

    assert str(error) == "negative_log_score must equal -log(accepted_mass)"


def test_raw_scoring_evidence_rejects_malformed_records() -> None:
    assert str(
        capture_evidence_error(
            lambda: RawScoringEvidence.from_record(
                {
                    "id": "core.boolean-evidence@1.0.0",
                    "observation_id": "observation-1",
                    "answer_space_id": "core.boolean-answer@0.1.0",
                    "accepted_event_id": "core.boolean-accepted@0.1.0",
                    "probability_measure_id": "core.boolean-prediction@0.1.0",
                    "accepted_mass": 0.25,
                    "negative_log_score": -math.log(0.25),
                }
            )
        )
    ) == (
        "identifier must use a pre-1.0.0 version before release policy exists: "
        "core.boolean-evidence@1.0.0"
    )
    assert str(
        capture_evidence_error(
            lambda: RawScoringEvidence.from_record(
                {
                    "id": "core.boolean-evidence@0.1.0",
                    "observation_id": "observation-1",
                    "answer_space_id": "core.boolean-answer@0.1.0",
                    "accepted_event_id": "core.boolean-accepted@0.1.0",
                    "probability_measure_id": "core.boolean-prediction@0.1.0",
                    "accepted_mass": 0.25,
                }
            )
        )
    ) == "negative_log_score: missing required field"
    assert str(
        capture_evidence_error(
            lambda: RawScoringEvidence.from_record(
                {
                    "id": "core.boolean-evidence@0.1.0",
                    "observation_id": "",
                    "answer_space_id": "core.boolean-answer@0.1.0",
                    "accepted_event_id": "core.boolean-accepted@0.1.0",
                    "probability_measure_id": "core.boolean-prediction@0.1.0",
                    "accepted_mass": 1.25,
                    "negative_log_score": -math.log(0.25),
                }
            )
        )
    ) == "observation_id must be nonempty"


def test_finite_answer_scoring_bundle_parses_complete_object_graph() -> None:
    bundle = FiniteAnswerScoringBundle.from_record(_boolean_bundle_record())

    assert bundle.answer_space.id == ProtocolIdentifier.parse("core.boolean-answer@0.1.0")
    assert bundle.accepted_event.id == ProtocolIdentifier.parse("core.boolean-accepted@0.1.0")
    assert bundle.probability_measure.id == ProtocolIdentifier.parse(
        "core.boolean-prediction@0.1.0"
    )
    assert bundle.raw_scoring_evidence.id == ProtocolIdentifier.parse(
        "core.boolean-evidence@0.1.0"
    )
    assert bundle.to_record() == _boolean_bundle_record()


def test_finite_answer_scoring_bundle_rejects_dangling_references() -> None:
    record = _boolean_bundle_record()
    raw_scoring_evidence = dict(
        cast(Mapping[str, object], record["raw_scoring_evidence"])
    )
    raw_scoring_evidence["accepted_event_id"] = "core.other-accepted@0.1.0"
    record["raw_scoring_evidence"] = raw_scoring_evidence

    error = capture_bundle_error(lambda: FiniteAnswerScoringBundle.from_record(record))

    assert str(error) == (
        "raw_scoring_evidence.accepted_event_id core.other-accepted@0.1.0 "
        "does not match core.boolean-accepted@0.1.0"
    )


def test_finite_answer_scoring_bundle_recomputes_score_from_event_and_measure() -> None:
    record = _boolean_bundle_record()
    raw_scoring_evidence = dict(
        cast(Mapping[str, object], record["raw_scoring_evidence"])
    )
    raw_scoring_evidence["accepted_mass"] = 0.25
    raw_scoring_evidence["negative_log_score"] = -math.log(0.25)
    record["raw_scoring_evidence"] = raw_scoring_evidence

    error = capture_bundle_error(lambda: FiniteAnswerScoringBundle.from_record(record))

    assert str(error) == (
        "raw_scoring_evidence.accepted_mass must equal recomputed accepted mass"
    )


def test_finite_answer_scoring_bundle_digest_is_stable() -> None:
    record = _boolean_bundle_record()
    reordered = {
        "raw_scoring_evidence": record["raw_scoring_evidence"],
        "probability_measure": record["probability_measure"],
        "accepted_event": record["accepted_event"],
        "answer_space": record["answer_space"],
    }

    assert ContentDigest.from_value(record) == ContentDigest.from_value(reordered)


def test_finite_answer_scoring_bundle_rejects_malformed_records() -> None:
    record = _boolean_bundle_record()
    record["summary"] = {"mean_score": 0.0}

    error = capture_bundle_error(lambda: FiniteAnswerScoringBundle.from_record(record))

    assert str(error) == "summary: unknown field"


def capture_answer_error(call: Callable[[], object]) -> AnswerSpaceValidationError:
    try:
        call()
    except AnswerSpaceValidationError as error:
        return error
    raise AssertionError("expected AnswerSpaceValidationError")


def capture_event_error(call: Callable[[], object]) -> AcceptedEventValidationError:
    try:
        call()
    except AcceptedEventValidationError as error:
        return error
    raise AssertionError("expected AcceptedEventValidationError")


def capture_measure_error(call: Callable[[], object]) -> ProbabilityMeasureValidationError:
    try:
        call()
    except ProbabilityMeasureValidationError as error:
        return error
    raise AssertionError("expected ProbabilityMeasureValidationError")


def capture_score_error(call: Callable[[], object]) -> AcceptedMassScoreError:
    try:
        call()
    except AcceptedMassScoreError as error:
        return error
    raise AssertionError("expected AcceptedMassScoreError")


def capture_evidence_error(call: Callable[[], object]) -> RawScoringEvidenceValidationError:
    try:
        call()
    except RawScoringEvidenceValidationError as error:
        return error
    raise AssertionError("expected RawScoringEvidenceValidationError")


def capture_bundle_error(
    call: Callable[[], object],
) -> FiniteAnswerScoringBundleValidationError:
    try:
        call()
    except FiniteAnswerScoringBundleValidationError as error:
        return error
    raise AssertionError("expected FiniteAnswerScoringBundleValidationError")


def _boolean_space() -> AnswerSpace:
    return AnswerSpace.from_record(
        {"id": "core.boolean-answer@0.1.0", "elements": [{"id": "yes"}]}
    )


def _boolean_bundle_record() -> dict[str, object]:
    return {
        "answer_space": {
            "id": "core.boolean-answer@0.1.0",
            "elements": [{"id": "yes"}, {"id": "no"}],
        },
        "accepted_event": {
            "id": "core.boolean-accepted@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "elements": ["yes"],
        },
        "probability_measure": {
            "id": "core.boolean-prediction@0.1.0",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "probabilities": [
                {"element_id": "no", "probability": 0.25},
                {"element_id": "yes", "probability": 0.75},
            ],
        },
        "raw_scoring_evidence": {
            "id": "core.boolean-evidence@0.1.0",
            "observation_id": "observation-1",
            "answer_space_id": "core.boolean-answer@0.1.0",
            "accepted_event_id": "core.boolean-accepted@0.1.0",
            "probability_measure_id": "core.boolean-prediction@0.1.0",
            "accepted_mass": 0.75,
            "negative_log_score": -math.log(0.75),
        },
    }
