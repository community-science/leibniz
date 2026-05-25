"""Semantic identifiers for durable protocol names."""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "IdentifierSyntaxError",
    "ProtocolIdentifier",
    "ProtocolName",
    "SemanticVersion",
]

_NUMERIC_IDENTIFIER = re.compile(r"0|[1-9][0-9]*")
_SEMVER = re.compile(
    r"^(?P<major>0|[1-9][0-9]*)"
    r"\.(?P<minor>0|[1-9][0-9]*)"
    r"\.(?P<patch>0|[1-9][0-9]*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+(?P<build>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
_NAME_ATOM = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
_IDENTIFIER_SEPARATOR = "@"


class IdentifierSyntaxError(ValueError):
    """Raised when text is not a valid Leibniz identifier."""


@dataclass(frozen=True, slots=True)
class SemanticVersion:
    """A SemVer 2.0.0 version."""

    major: int
    minor: int
    patch: int
    prerelease: tuple[str, ...] = ()
    build: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_nonnegative("major", self.major)
        _validate_nonnegative("minor", self.minor)
        _validate_nonnegative("patch", self.patch)
        _validate_identifiers("prerelease", self.prerelease, forbid_numeric_leading_zero=True)
        _validate_identifiers("build", self.build, forbid_numeric_leading_zero=False)

    @classmethod
    def parse(cls, text: str) -> SemanticVersion:
        match = _SEMVER.fullmatch(text)
        if match is None:
            raise IdentifierSyntaxError(f"invalid semantic version: {text!r}")

        prerelease = _split_identifiers(match.group("prerelease"))
        build = _split_identifiers(match.group("build"))
        return cls(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=prerelease,
            build=build,
        )

    @property
    def is_unreleased(self) -> bool:
        return self.major == 0

    def without_build(self) -> SemanticVersion:
        return SemanticVersion(
            major=self.major,
            minor=self.minor,
            patch=self.patch,
            prerelease=self.prerelease,
        )

    def __str__(self) -> str:
        value = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            value = f"{value}-{'.'.join(self.prerelease)}"
        if self.build:
            value = f"{value}+{'.'.join(self.build)}"
        return value

    def precedes(self, other: SemanticVersion) -> bool:
        return self._precedence_key() < other._precedence_key()

    def has_same_precedence_as(self, other: SemanticVersion) -> bool:
        return self._precedence_key() == other._precedence_key()

    def _precedence_key(self) -> tuple[int, int, int, tuple[tuple[int, int | str], ...]]:
        return (
            self.major,
            self.minor,
            self.patch,
            _prerelease_precedence(self.prerelease),
        )


@dataclass(frozen=True, slots=True)
class ProtocolName:
    """A lowercase dotted name for a durable protocol object."""

    value: str

    def __post_init__(self) -> None:
        atoms = self.value.split(".")
        if not atoms or any(not _NAME_ATOM.fullmatch(atom) for atom in atoms):
            raise IdentifierSyntaxError(f"invalid protocol name: {self.value!r}")

    @classmethod
    def parse(cls, text: str) -> ProtocolName:
        return cls(text)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ProtocolIdentifier:
    """A protocol name bound to a semantic version."""

    name: ProtocolName
    version: SemanticVersion

    @classmethod
    def parse(cls, text: str) -> ProtocolIdentifier:
        name_text, separator, version_text = text.partition(_IDENTIFIER_SEPARATOR)
        if separator == "" or _IDENTIFIER_SEPARATOR in version_text:
            raise IdentifierSyntaxError(f"invalid protocol identifier: {text!r}")
        return cls(name=ProtocolName.parse(name_text), version=SemanticVersion.parse(version_text))

    @property
    def is_unreleased(self) -> bool:
        return self.version.is_unreleased

    def require_unreleased(self) -> ProtocolIdentifier:
        if not self.is_unreleased:
            raise IdentifierSyntaxError(
                f"identifier must use a pre-1.0.0 version before release policy exists: {self}"
            )
        return self

    def __str__(self) -> str:
        return f"{self.name}{_IDENTIFIER_SEPARATOR}{self.version}"


def _split_identifiers(text: str | None) -> tuple[str, ...]:
    if text is None:
        return ()
    return tuple(text.split("."))


def _validate_nonnegative(name: str, value: int) -> None:
    if value < 0:
        raise IdentifierSyntaxError(f"{name} must be nonnegative")


def _validate_identifiers(
    field: str,
    identifiers: tuple[str, ...],
    *,
    forbid_numeric_leading_zero: bool,
) -> None:
    for identifier in identifiers:
        if identifier == "":
            raise IdentifierSyntaxError(f"{field} identifiers must not be empty")
        if not re.fullmatch(r"[0-9A-Za-z-]+", identifier):
            raise IdentifierSyntaxError(f"invalid {field} identifier: {identifier!r}")
        if (
            forbid_numeric_leading_zero
            and identifier.isdigit()
            and _NUMERIC_IDENTIFIER.fullmatch(identifier) is None
        ):
            raise IdentifierSyntaxError(
                f"numeric {field} identifiers must not contain leading zeroes: {identifier!r}"
            )


def _prerelease_precedence(prerelease: tuple[str, ...]) -> tuple[tuple[int, int | str], ...]:
    if not prerelease:
        return ((2, 0),)
    precedence: list[tuple[int, int | str]] = []
    for identifier in prerelease:
        if identifier.isdigit():
            precedence.append((0, int(identifier)))
        else:
            precedence.append((1, identifier))
    return tuple(precedence)
