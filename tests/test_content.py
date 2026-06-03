import ast
import math
import re
from collections.abc import Callable
from pathlib import Path

import leibniz.content as content
from leibniz._formats._json import canonical_json_bytes, load_json_object_file
from leibniz.content import ContentDigest, ContentEncodingError
from leibniz.documents import (
    canonical_document_bytes,
    document_filename_suffix,
    document_media_type,
    load_object_document,
)
from leibniz.identifiers import ProtocolIdentifier
from leibniz.outcomes import (
    AcceptedEvent,
    FiniteProbabilityMeasure,
    OutcomeSpace,
    ProbabilityMass,
    RawScoringEvidence,
)

_source_root = Path(__file__).parents[1] / "src" / "leibniz"
_tests_root = Path(__file__).parents[1] / "tests"
_format_boundary_files = {
    _source_root / "documents.py",
    _source_root / "_formats" / "_json.py",
}
_json_test_boundary_files = {
    _tests_root / "test_content.py",
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


def test_content_digest_parses_algorithm_qualified_digest_strings() -> None:
    digest = ContentDigest.from_value({"id": "core.example@0.1.0"})

    assert ContentDigest.from_string(str(digest), field="digest") == digest
    assert_error(
        lambda: ContentDigest.from_string("not-a-digest", field="digest"),
        "digest: expected algorithm:digest",
    )
    assert_error(
        lambda: ContentDigest.from_string("sha256:abcd", field="digest"),
        "sha256 digest must be 64 lowercase hexadecimal characters",
    )


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


def test_document_filename_suffix_is_defined_at_document_boundary() -> None:
    assert document_filename_suffix() == ".json"


def test_document_media_type_is_defined_at_document_boundary() -> None:
    assert document_media_type() == "application/json"


def test_source_mentions_json_only_at_document_format_boundary() -> None:
    offenders = tuple(
        path.relative_to(_source_root.parents[1])
        for path in sorted(_source_root.rglob("*.py"))
        if path not in _format_boundary_files
        and re.search(r"\bjson\b|\bJSON\b|_json", path.read_text(encoding="utf-8"))
    )

    assert offenders == ()


def test_generated_document_filename_suffix_is_defined_only_at_document_boundary() -> None:
    offenders = tuple(
        path.relative_to(_source_root.parents[1])
        for path in sorted(_source_root.rglob("*.py"))
        if path != _source_root / "documents.py"
        and _has_json_filename_suffix_literal(path.read_text(encoding="utf-8"))
    )

    assert offenders == ()


def test_source_does_not_hide_json_filename_suffixes_by_concatenating_string_literals() -> None:
    offenders = tuple(
        path.relative_to(_source_root.parents[1])
        for path in sorted(_source_root.rglob("*.py"))
        if _has_hidden_json_suffix(path.read_text(encoding="utf-8"))
    )

    assert offenders == ()


def test_json_suffix_detectors_reject_split_string_literal_terms() -> None:
    assert _has_hidden_json_suffix('suffix = "." + "".join(("j", "s", "o", "n"))')
    assert _has_hidden_json_suffix('suffix = "." + "js" + "on"')
    assert _has_hidden_json_suffix('load("name" + "." + "j" + "son")')
    assert _has_json_filename_suffix_literal('suffix = ".json"')
    assert not _has_hidden_json_suffix('suffix = ".json"')
    assert not _has_json_filename_suffix_literal('suffix = ".csv"')


def test_tests_import_json_only_at_document_format_boundary() -> None:
    offenders = tuple(
        path.relative_to(_tests_root)
        for path in sorted(_tests_root.rglob("*.py"))
        if path not in _json_test_boundary_files
        and re.search(r"^\s*import json\b|^\s*from json\b", path.read_text(encoding="utf-8"), re.M)
    )

    assert offenders == ()


def _has_hidden_json_suffix(source: str) -> bool:
    tree = ast.parse(source)
    return any(
        _hides_json_suffix(node)
        for node in ast.walk(tree)
    )


def _has_json_filename_suffix_literal(source: str) -> bool:
    tree = ast.parse(source)
    return any(
        isinstance(node, ast.Constant) and node.value == ".json"
        for node in ast.walk(tree)
    )


def _hides_json_suffix(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and node.value == ".json":
        return False
    return _string_literal_value(node) == ".json" or _contains_hidden_json_suffix(node)


def _contains_hidden_json_suffix(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return False
    literal = _string_literal_value(node)
    if literal is None or not literal.endswith(".json"):
        return False
    expression = ast.unparse(node)
    return ".json" not in expression


def _string_literal_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _string_literal_value(node.left)
        right = _string_literal_value(node.right)
        if left is not None and right is not None:
            return left + right
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and _string_literal_value(node.func.value) is not None
        and len(node.args) == 1
    ):
        values = _string_literal_sequence(node.args[0])
        if values is not None:
            return (_string_literal_value(node.func.value) or "").join(values)
    return None


def _string_literal_sequence(node: ast.AST) -> tuple[str, ...] | None:
    if not isinstance(node, ast.Tuple | ast.List):
        return None
    values: list[str] = []
    for item in node.elts:
        value = _string_literal_value(item)
        if value is None:
            return None
        values.append(value)
    return tuple(values)


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
