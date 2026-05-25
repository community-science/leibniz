import math
import re
from collections.abc import Callable
from pathlib import Path

import leibniz.content as content
from leibniz._documents import canonical_document_bytes, load_object_document
from leibniz._formats._json import canonical_json_bytes, load_json_object_file
from leibniz.content import ContentDigest, ContentEncodingError
from leibniz.identifiers import ProtocolIdentifier
from leibniz.outcomes import (
    AcceptedEvent,
    FiniteProbabilityMeasure,
    OutcomeSpace,
    ProbabilityMass,
    RawScoringEvidence,
)

_source_root = Path(__file__).parents[1] / "src" / "leibniz"
_format_boundary_files = {
    _source_root / "_documents.py",
    _source_root / "_formats" / "_json.py",
}


def test_content_digest_is_stable_for_mapping_order() -> None:
    left = {"b": 2, "a": [{"z": "last", "m": "middle"}]}
    right = {"a": [{"m": "middle", "z": "last"}], "b": 2}

    assert canonical_json_bytes(left) == b'{"a":[{"m":"middle","z":"last"}],"b":2}'
    assert ContentDigest.from_value(left) == ContentDigest.from_value(right)


def test_content_digest_formats_sha256_digest() -> None:
    digest = ContentDigest.from_value({"id": "core.example@0.1.0"})

    assert digest == ContentDigest(algorithm="sha256", hex=digest.hex)
    assert str(digest) == f"sha256:{digest.hex}"


def test_content_digest_rejects_values_that_cannot_be_encoded() -> None:
    assert_error(lambda: ContentDigest.from_value({"score": math.inf}), "score: nonfinite number")
    assert_error(
        lambda: ContentDigest.from_value({1: "one"}), "<value>: object key must be string"
    )
    assert_error(
        lambda: ContentDigest.from_value({"items": (object(),)}),
        "items.0: unsupported JSON value",
    )


def test_json_object_file_loading_is_internal() -> None:
    assert "CanonicalJson" not in content.__all__
    assert "CanonicalJsonError" not in content.__all__
    assert "JsonDocument" not in content.__all__
    assert "JsonObject" not in content.__all__
    assert "JsonScalar" not in content.__all__
    assert "JsonValue" not in content.__all__
    assert "load_json_object_file" not in content.__all__
    assert "load_object_document" not in content.__all__


def test_json_object_file_loader_decodes_objects() -> None:
    assert load_json_object_file(b'{"b":2,"a":{"z":3}}') == {
        "a": {"z": 3},
        "b": 2,
    }


def test_json_object_file_loader_rejects_invalid_input() -> None:
    assert_error(lambda: load_json_object_file(b"\xff"), "JSON file must be UTF-8")
    assert_error(
        lambda: load_json_object_file(b"{"),
        "invalid JSON file: Expecting property name enclosed in double quotes",
    )
    assert_error(lambda: load_json_object_file(b"[]"), "JSON file must contain an object")
    assert_error(
        lambda: load_json_object_file(b'{"score": Infinity}'),
        "score: nonfinite number",
    )


def test_json_object_file_loader_uses_document_description() -> None:
    assert_error(
        lambda: load_json_object_file(b"\xff", description="manifest JSON file"),
        "manifest JSON file must be UTF-8",
    )
    assert_error(
        lambda: load_json_object_file(b"{", description="manifest JSON file"),
        "invalid manifest JSON file: Expecting property name enclosed in double quotes",
    )
    assert_error(
        lambda: load_json_object_file(b"[]", description="manifest JSON file"),
        "manifest JSON file must contain an object",
    )


def test_object_document_loader_delegates_to_current_format() -> None:
    assert load_object_document(
        b'{"b":2,"a":{"z":3}}',
        description="manifest document",
    ) == {
        "a": {"z": 3},
        "b": 2,
    }
    assert_error(
        lambda: load_object_document(b"[]", description="manifest document"),
        "manifest document must contain an object",
    )


def test_canonical_document_bytes_delegate_to_current_format() -> None:
    assert canonical_document_bytes({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_source_mentions_json_only_at_document_format_boundary() -> None:
    offenders = tuple(
        path.relative_to(_source_root.parents[1])
        for path in sorted(_source_root.rglob("*.py"))
        if path not in _format_boundary_files
        and re.search(r"\bjson\b|\bJSON\b|_json", path.read_text(encoding="utf-8"))
    )

    assert offenders == ()


def test_canonical_json_and_digests_cover_finite_outcome_records() -> None:
    space = OutcomeSpace.from_record(
        {
            "outcomes": [{"id": "yes"}, {"id": "no"}],
            "id": "core.boolean-outcome@0.1.0",
        }
    )
    event = AcceptedEvent.from_record(
        {
            "outcomes": ["yes"],
            "outcome_space_id": "core.boolean-outcome@0.1.0",
            "id": "core.boolean-accepted@0.1.0",
        },
        outcome_space=space,
    )
    measure = FiniteProbabilityMeasure(
        id=ProtocolIdentifier.parse("core.boolean-prediction@0.1.0"),
        outcome_space_id=ProtocolIdentifier.parse("core.boolean-outcome@0.1.0"),
        probabilities=(ProbabilityMass("no", 0.25), ProbabilityMass("yes", 0.75)),
    )
    evidence = RawScoringEvidence(
        id=ProtocolIdentifier.parse("core.boolean-evidence@0.1.0"),
        observation_id="observation-1",
        outcome_space_id=ProtocolIdentifier.parse("core.boolean-outcome@0.1.0"),
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
        assert str(ContentDigest.from_value(record)).startswith("sha256:")

    reordered_space_record = {
        "outcomes": [{"id": "yes"}, {"id": "no"}],
        "id": "core.boolean-outcome@0.1.0",
    }
    assert ContentDigest.from_value(space.to_record()) == ContentDigest.from_value(
        reordered_space_record
    )


def assert_error(call: Callable[[], object], message: str) -> None:
    try:
        call()
    except ContentEncodingError as error:
        assert str(error) == message
        return
    raise AssertionError("expected ContentEncodingError")
