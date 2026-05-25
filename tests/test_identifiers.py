from typing import Literal

from leibniz.identifiers import (
    IdentifierSyntaxError,
    ProtocolIdentifier,
    ProtocolName,
    SemanticVersion,
)


def assert_identifier_error(text: str, parser: Literal["identifier", "name", "version"]) -> None:
    try:
        if parser == "version":
            SemanticVersion.parse(text)
        elif parser == "name":
            ProtocolName.parse(text)
        elif parser == "identifier":
            ProtocolIdentifier.parse(text)
        else:
            raise AssertionError(f"unknown parser: {parser!r}")
    except IdentifierSyntaxError:
        return
    raise AssertionError(f"expected IdentifierSyntaxError for {text!r}")


def test_semantic_version_parses_and_formats_release_core() -> None:
    version = SemanticVersion.parse("0.2.3")

    assert version == SemanticVersion(major=0, minor=2, patch=3)
    assert str(version) == "0.2.3"
    assert version.is_unreleased


def test_semantic_version_parses_prerelease_and_build_metadata() -> None:
    version = SemanticVersion.parse("0.2.3-alpha.1+build.7")

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
    assert SemanticVersion.parse("1.0.0-alpha").precedes(SemanticVersion.parse("1.0.0"))
    assert SemanticVersion.parse("1.0.0-alpha.1").precedes(
        SemanticVersion.parse("1.0.0-alpha.beta")
    )
    assert SemanticVersion.parse("1.0.0+build.1").has_same_precedence_as(
        SemanticVersion.parse("1.0.0+build.2")
    )
    assert SemanticVersion.parse("1.0.0+build.1") != SemanticVersion.parse(
        "1.0.0+build.2"
    )


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
    identifier = ProtocolIdentifier.parse("core.semantic-version@0.1.0")

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
    identifier = ProtocolIdentifier.parse("core.semantic-version@0.1.0")

    assert identifier.require_unreleased() is identifier

    try:
        ProtocolIdentifier.parse("core.semantic-version@1.0.0").require_unreleased()
    except IdentifierSyntaxError:
        return
    raise AssertionError("expected IdentifierSyntaxError for released identifier")
