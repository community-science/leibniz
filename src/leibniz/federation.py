"""Credential-free federation publication planning."""

from __future__ import annotations

from dataclasses import dataclass

from leibniz.content import ContentDigest
from leibniz.publications import SubmissionPublicationBundle

__all__ = [
    "FederationPublicationArtifact",
    "FederationPublicationPlan",
    "FederationTarget",
    "FederationValidationError",
    "plan_federation_publication",
]


class FederationValidationError(ValueError):
    """Raised when a federation publication plan is invalid."""


@dataclass(frozen=True, slots=True)
class FederationTarget:
    """A remote dataset-style destination named without credentials."""

    repository_id: str
    repository_type: str = "dataset"

    def __post_init__(self) -> None:
        if not self.repository_id:
            raise FederationValidationError("repository_id must be nonempty")
        if not self.repository_type:
            raise FederationValidationError("repository_type must be nonempty")
        if "://" in self.repository_id:
            raise FederationValidationError("repository_id must not include a URI scheme")
        if self.repository_id.startswith((".", "/")):
            raise FederationValidationError("repository_id must not be a local path")
        if any(part in {"", ".", ".."} for part in self.repository_id.split("/")):
            raise FederationValidationError("repository_id must be a stable repository path")

    def to_record(self) -> dict[str, object]:
        return {
            "repository_id": self.repository_id,
            "repository_type": self.repository_type,
        }


@dataclass(frozen=True, slots=True)
class FederationPublicationArtifact:
    """One bundle artifact that would be published to a federation target."""

    name: str
    digest: ContentDigest

    def __post_init__(self) -> None:
        if not self.name:
            raise FederationValidationError("artifact name must be nonempty")
        if self.name.startswith((".", "/")) or "/../" in f"/{self.name}/":
            raise FederationValidationError("artifact name must be a relative stable path")

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "digest": str(self.digest),
        }


@dataclass(frozen=True, slots=True)
class FederationPublicationPlan:
    """A credential-free plan for publishing an already-validated bundle."""

    target: FederationTarget
    bundle_digest: ContentDigest
    artifacts: tuple[FederationPublicationArtifact, ...]
    dry_run: bool = True

    def __post_init__(self) -> None:
        if not self.artifacts:
            raise FederationValidationError("artifacts must contain at least one artifact")
        names = tuple(artifact.name for artifact in self.artifacts)
        duplicate = _first_duplicate(names)
        if duplicate is not None:
            raise FederationValidationError(f"duplicate artifact name: {duplicate}")

    def to_record(self) -> dict[str, object]:
        return {
            "target": self.target.to_record(),
            "bundle_digest": str(self.bundle_digest),
            "dry_run": self.dry_run,
            "artifacts": [artifact.to_record() for artifact in self.artifacts],
        }


def plan_federation_publication(
    bundle: SubmissionPublicationBundle,
    *,
    target: FederationTarget,
    dry_run: bool = True,
) -> FederationPublicationPlan:
    """Plan publication operations for a validated bundle without network writes."""

    try:
        bundle.validate_sources()
    except ValueError as error:
        raise FederationValidationError(str(error)) from error
    return FederationPublicationPlan(
        target=target,
        bundle_digest=bundle.digest,
        dry_run=dry_run,
        artifacts=_publication_artifacts(bundle),
    )


def _publication_artifacts(
    bundle: SubmissionPublicationBundle,
) -> tuple[FederationPublicationArtifact, ...]:
    artifacts = [
        FederationPublicationArtifact(
            name="publication_bundle",
            digest=bundle.digest,
        ),
        FederationPublicationArtifact(
            name="submission_package",
            digest=bundle.submission_package.digest,
        ),
        FederationPublicationArtifact(
            name="measurement_dataset",
            digest=bundle.measurement_dataset.digest,
        ),
        FederationPublicationArtifact(
            name="measurement_score_view",
            digest=bundle.measurement_score_view.digest,
        ),
    ]
    if bundle.proposal_set is not None:
        artifacts.append(
            FederationPublicationArtifact(
                name="proposal_set",
                digest=bundle.proposal_set.digest,
            )
        )
    if bundle.architecture_surrogate is not None:
        artifacts.append(
            FederationPublicationArtifact(
                name="architecture_surrogate",
                digest=bundle.architecture_surrogate.digest,
            )
        )
    return tuple(artifacts)


def _first_duplicate(values: tuple[str, ...]) -> str | None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
