import math
from collections.abc import Callable

from leibniz.answers import (
    AcceptedEvent,
    AnswerSpace,
    FiniteProbabilityMeasure,
    ProbabilityMass,
    RawScoringEvidence,
)
from leibniz.content import CanonicalJson, CanonicalJsonError, ContentDigest, JsonDocument
from leibniz.identifiers import ProtocolIdentifier


def test_canonical_json_is_stable_for_mapping_order() -> None:
    left = {"b": 2, "a": [{"z": "last", "m": "middle"}]}
    right = {"a": [{"m": "middle", "z": "last"}], "b": 2}

    assert bytes(CanonicalJson.from_value(left)) == b'{"a":[{"m":"middle","z":"last"}],"b":2}'
    assert CanonicalJson.from_value(left) == CanonicalJson.from_value(right)
    assert ContentDigest.from_value(left) == ContentDigest.from_value(right)


def test_content_digest_formats_sha256_digest() -> None:
    digest = ContentDigest.from_value({"id": "core.example@0.1.0"})

    assert digest == ContentDigest(algorithm="sha256", hex=digest.hex)
    assert str(digest) == f"sha256:{digest.hex}"
    assert CanonicalJson.from_value({"id": "core.example@0.1.0"}).digest() == digest


def test_canonical_json_rejects_values_that_json_cannot_represent() -> None:
    assert_error(lambda: CanonicalJson.from_value({"score": math.inf}), "score: nonfinite number")
    assert_error(
        lambda: CanonicalJson.from_value({1: "one"}), "<value>: object key must be string"
    )
    assert_error(
        lambda: CanonicalJson.from_value({"items": (object(),)}),
        "items.0: unsupported JSON value",
    )


def test_json_document_loads_object_with_canonical_digest() -> None:
    document = JsonDocument.from_json_bytes(b'{"b":2,"a":{"z":3}}')

    assert document.value == {"a": {"z": 3}, "b": 2}
    assert document.digest == ContentDigest.from_value({"a": {"z": 3}, "b": 2})


def test_json_document_digest_is_stable_for_mapping_order() -> None:
    left = JsonDocument.from_json_bytes(b'{"b":2,"a":1}')
    right = JsonDocument.from_json_bytes(b'{"a":1,"b":2}')

    assert left.digest == right.digest


def test_json_document_rejects_invalid_input() -> None:
    assert_error(lambda: JsonDocument.from_json_bytes(b"\xff"), "JSON document must be UTF-8")
    assert_error(
        lambda: JsonDocument.from_json_bytes(b"{"),
        "invalid JSON document: Expecting property name enclosed in double quotes",
    )
    assert_error(lambda: JsonDocument.from_json_bytes(b"[]"), "JSON document must be an object")
    assert_error(
        lambda: JsonDocument.from_json_bytes(b'{"score": Infinity}'),
        "score: nonfinite number",
    )


def test_canonical_json_and_digests_cover_finite_answer_records() -> None:
    space = AnswerSpace.from_record(
        {
            "elements": [{"id": "yes"}, {"id": "no"}],
            "id": "core.boolean-answer@0.1.0",
        }
    )
    event = AcceptedEvent.from_record(
        {
            "elements": ["yes"],
            "answer_space_id": "core.boolean-answer@0.1.0",
            "id": "core.boolean-accepted@0.1.0",
        },
        answer_space=space,
    )
    measure = FiniteProbabilityMeasure(
        id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
        answer_space_id=ProtocolIdentifier.parse("core.boolean-answer@0.1.0"),
        probabilities=(ProbabilityMass("no", 0.25), ProbabilityMass("yes", 0.75)),
    )
    evidence = RawScoringEvidence(
        id=ProtocolIdentifier.parse("core.boolean-evidence@0.1.0"),
        observation_id="observation-1",
        answer_space_id=ProtocolIdentifier.parse("core.boolean-answer@0.1.0"),
        accepted_event_id=ProtocolIdentifier.parse("core.boolean-accepted@0.1.0"),
        probability_measure_id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
        accepted_mass=0.75,
        negative_log_score=-math.log(0.75),
    )

    records = [
        space.to_record(),
        event.to_record(),
        measure.to_record(),
        evidence.to_record(),
    ]

    for record in records:
        assert CanonicalJson.from_value(record).data
        assert str(ContentDigest.from_value(record)).startswith("sha256:")

    reordered_space_record = {
        "elements": [{"id": "yes"}, {"id": "no"}],
        "id": "core.boolean-answer@0.1.0",
    }
    assert ContentDigest.from_value(space.to_record()) == ContentDigest.from_value(
        reordered_space_record
    )


def assert_error(call: Callable[[], object], message: str) -> None:
    try:
        call()
    except CanonicalJsonError as error:
        assert str(error) == message
        return
    raise AssertionError("expected CanonicalJsonError")
