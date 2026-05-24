from typing import Literal

from leibniz.identifiers import (
    IdentifierSyntaxError,
    ProtocolIdentifier,
    ProtocolName,
    SemanticVersion,
    parse_protocol_identifier,
    parse_semantic_version,
    require_unreleased_identifier,
)


def assert_identifier_error(text: str, parser: Literal["identifier", "name", "version"]) -> None:
    try:
        if parser == "version":
            parse_semantic_version(text)
        elif parser == "name":
            ProtocolName.parse(text)
        elif parser == "identifier":
            parse_protocol_identifier(text)
        else:
            raise AssertionError(f"unknown parser: {parser!r}")
    except IdentifierSyntaxError:
        return
    raise AssertionError(f"expected IdentifierSyntaxError for {text!r}")


def test_semantic_version_parses_and_formats_release_core() -> None:
    version = parse_semantic_version("0.2.3")

    assert version == SemanticVersion(major=0, minor=2, patch=3)
    assert str(version) == "0.2.3"
    assert version.is_unreleased


def test_semantic_version_parses_prerelease_and_build_metadata() -> None:
    version = parse_semantic_version("0.2.3-alpha.1+build.7")

    assert version.prerelease == ("alpha", "1")
    assert version.build == ("build", "7")
    assert str(version) == "0.2.3-alpha.1+build.7"


def test_semantic_version_rejects_invalid_forms() -> None:
    invalid_versions = [
        "0.1",
        "0.1.2.3",
        "01.2.3",
        "1.02.3",
        "1.2.03",
        "1.2.3-01",
        "1.2.3-alpha..1",
        "1.2.3+",
    ]

    for text in invalid_versions:
        assert_identifier_error(text, "version")


def test_semantic_version_order_ignores_build_metadata() -> None:
    assert parse_semantic_version("1.0.0-alpha").precedes(parse_semantic_version("1.0.0"))
    assert parse_semantic_version("1.0.0-alpha.1").precedes(
        parse_semantic_version("1.0.0-alpha.beta")
    )
    assert parse_semantic_version("1.0.0+build.1").has_same_precedence_as(
        parse_semantic_version("1.0.0+build.2")
    )
    assert parse_semantic_version("1.0.0+build.1") != parse_semantic_version("1.0.0+build.2")


def test_protocol_name_accepts_lowercase_dotted_atoms() -> None:
    name = ProtocolName.parse("core.semantic-version")

    assert str(name) == "core.semantic-version"


def test_protocol_name_rejects_ambiguous_or_noncanonical_names() -> None:
    invalid_names = [
        "",
        "Core.semantic-version",
        "core..semantic-version",
        "core.semantic_version",
        "core.-semantic-version",
        "core.semantic-version-",
    ]

    for text in invalid_names:
        assert_identifier_error(text, "name")


def test_protocol_identifier_parses_name_and_version() -> None:
    identifier = parse_protocol_identifier("core.semantic-version@0.1.0")

    assert identifier == ProtocolIdentifier(
        name=ProtocolName("core.semantic-version"),
        version=SemanticVersion(major=0, minor=1, patch=0),
    )
    assert str(identifier) == "core.semantic-version@0.1.0"
    assert identifier.is_unreleased


def test_protocol_identifier_rejects_missing_or_ambiguous_separator() -> None:
    for text in ["core.semantic-version", "core.semantic-version@", "@0.1.0", "a@0.1.0@b"]:
        assert_identifier_error(text, "identifier")


def test_unreleased_policy_accepts_only_pre_1_0_versions() -> None:
    identifier = parse_protocol_identifier("core.semantic-version@0.1.0")

    assert require_unreleased_identifier(identifier) is identifier

    try:
        require_unreleased_identifier(parse_protocol_identifier("core.semantic-version@1.0.0"))
    except IdentifierSyntaxError:
        return
    raise AssertionError("expected IdentifierSyntaxError for released identifier")
