from collections.abc import Callable
from pathlib import Path
from typing import cast

from leibniz.architectures import ArchitectureManifestDocument
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.federation import (
    FederationPublicationArtifact,
    FederationPublicationPlan,
    FederationTarget,
    FederationValidationError,
    plan_federation_publication,
)
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset, MeasurementDatasetDocument, MeasurementDocument
from leibniz.proposals import ExperimentProposalSet
from leibniz.publications import SubmissionPublicationBundle
from leibniz.submissions import SubmissionPackageManifest
from leibniz.surrogates import ArchitectureSurrogateRecord
from leibniz.views import MeasurementScoreView

_fixtures_root = Path(__file__).parent / "fixtures"


def test_federation_publication_plan_reports_dry_run_artifacts() -> None:
    bundle = _publication_bundle()
    proposal_set = _required_proposal_set(bundle)
    architecture_surrogate = _required_architecture_surrogate(bundle)
    target = FederationTarget(repository_id="operator/submissions")

    plan = plan_federation_publication(bundle, target=target)

    assert plan == FederationPublicationPlan(
        target=target,
        bundle_digest=bundle.digest,
        dry_run=True,
        artifacts=(
            FederationPublicationArtifact("publication_bundle", bundle.digest),
            FederationPublicationArtifact("submission_package", bundle.submission_package.digest),
            FederationPublicationArtifact("measurement_dataset", bundle.measurement_dataset.digest),
            FederationPublicationArtifact(
                "measurement_score_view",
                bundle.measurement_score_view.digest,
            ),
            FederationPublicationArtifact("proposal_set", proposal_set.digest),
            FederationPublicationArtifact(
                "architecture_surrogate",
                architecture_surrogate.digest,
            ),
        ),
    )
    assert plan.to_record() == {
        "target": {
            "repository_id": "operator/submissions",
            "repository_type": "dataset",
        },
        "bundle_digest": str(bundle.digest),
        "dry_run": True,
        "artifacts": [artifact.to_record() for artifact in plan.artifacts],
    }


def test_federation_publication_plan_supports_validation_only_mode() -> None:
    bundle = _publication_bundle()

    plan = plan_federation_publication(
        bundle,
        target=FederationTarget(repository_id="operator/submissions"),
        dry_run=False,
    )

    assert plan.dry_run is False
    assert plan.bundle_digest == bundle.digest


def test_federation_target_rejects_credentials_and_local_paths() -> None:
    assert str(
        capture_federation_error(lambda: FederationTarget(repository_id="hf://owner/repo"))
    ) == "repository_id must not include a URI scheme"

    assert str(
        capture_federation_error(lambda: FederationTarget(repository_id="./results/submissions"))
    ) == "repository_id must not be a local path"

    assert str(
        capture_federation_error(lambda: FederationTarget(repository_id="owner//repo"))
    ) == "repository_id must be a stable repository path"


def test_federation_publication_plan_validates_bundle_before_planning() -> None:
    bundle = _publication_bundle()
    broken_bundle = _unchecked_publication_bundle(
        bundle=bundle,
        measurement_dataset=_expanded_dataset(),
    )

    assert str(
        capture_federation_error(
            lambda: plan_federation_publication(
                broken_bundle,
                target=FederationTarget(repository_id="operator/submissions"),
            )
        )
    ) == "submission_package measurement_dataset does not match bundle dataset"


def test_federation_publication_artifacts_reject_unsafe_names() -> None:
    assert str(
        capture_federation_error(
            lambda: FederationPublicationArtifact(
                name="../publication_bundle",
                digest=ContentDigest.from_value({"bundle": True}),
            )
        )
    ) == "artifact name must be a relative stable path"

    artifact = FederationPublicationArtifact(
        name="publication_bundle",
        digest=ContentDigest.from_value({"bundle": True}),
    )
    assert str(
        capture_federation_error(
            lambda: FederationPublicationPlan(
                target=FederationTarget(repository_id="operator/submissions"),
                bundle_digest=ContentDigest.from_value({"bundle": True}),
                artifacts=(artifact, artifact),
            )
        )
    ) == "duplicate artifact name: publication_bundle"


def _publication_bundle() -> SubmissionPublicationBundle:
    dataset = _dataset_document().dataset
    return SubmissionPublicationBundle.from_record(
        {
            "id": "publication-bundles.boolean@0.1.0",
            "submission_package": _submission_package_record(),
            "measurement_dataset": dataset.to_record(),
            "measurement_score_view": _measurement_score_view(dataset=dataset).to_record(),
            "proposal_set": _proposal_set_record(dataset=dataset),
            "architecture_surrogate": _surrogate_record(dataset=dataset),
        }
    )


def _submission_package_record() -> dict[str, object]:
    return {
        "id": "submissions.boolean-digits-pool@0.1.0",
        "benchmark_manifest": _benchmark_document().manifest.to_record(),
        "architecture_manifest": _architecture_document().manifest.to_record(),
        "measurement_dataset": _dataset_document().dataset.to_record(),
    }


def _proposal_set_record(*, dataset: MeasurementDataset) -> dict[str, object]:
    architecture = _architecture_document().manifest
    submission = SubmissionPackageManifest.from_record(_submission_package_record())
    return {
        "id": "experiment-proposal-sets.boolean@0.1.0",
        "source_dataset_digest": str(dataset.digest),
        "proposals": [
            {
                "id": "experiment-proposals.boolean.rank-1@0.1.0",
                "rank": 1,
                "candidate_kind": "architecture",
                "candidate_id": str(architecture.id),
                "rationale": "first unmeasured architecture declaration",
            },
            {
                "id": "experiment-proposals.boolean.rank-2@0.1.0",
                "rank": 2,
                "candidate_kind": "submission-package",
                "candidate_id": str(submission.id),
                "rationale": "complete submission package declaration",
            },
        ],
    }


def _surrogate_record(*, dataset: MeasurementDataset) -> dict[str, object]:
    return {
        "id": "architecture-surrogates.boolean@0.1.0",
        "source_dataset_digest": str(dataset.digest),
        "model_kind": "neural-empirical",
        "target_name": "negative_log_accepted_mass",
        "features": [
            {
                "name": "layer_count",
                "mean": 2.0,
                "scale": 1.0,
                "sensitivity": 0.1,
            }
        ],
        "training": {
            "status": "fit",
            "observation_count": 1,
        },
        "state": {
            "format": "dense-regressor-summary",
            "input_width": 1,
            "output_width": 1,
            "parameter_count": 3,
        },
    }


def _measurement_score_view(*, dataset: MeasurementDataset) -> MeasurementScoreView:
    return MeasurementScoreView.from_dataset(
        id=ProtocolIdentifier.parse("views.measurement-scores.boolean@0.1.0"),
        dataset=dataset,
    )


def _benchmark_document() -> BenchmarkManifestDocument:
    return BenchmarkManifestDocument.from_bytes(
        (_fixtures_root / "finite_outcome" / "manifest.json").read_bytes()
    )


def _architecture_document() -> ArchitectureManifestDocument:
    return ArchitectureManifestDocument.from_bytes(
        (_fixtures_root / "architecture" / "digits_pool.json").read_bytes()
    )


def _dataset_document() -> MeasurementDatasetDocument:
    measurement = MeasurementDocument.from_bytes(
        (_fixtures_root / "finite_outcome" / "measurement.json").read_bytes()
    ).measurement
    return MeasurementDatasetDocument.from_bytes(
        canonical_document_bytes({"measurements": [measurement.to_record()]})
    )


def _expanded_dataset() -> MeasurementDataset:
    measurement = _dataset_document().dataset.measurements[0].to_record()
    alternate = dict(measurement)
    raw_evidence: dict[str, object] = dict(_dict_record(alternate["raw_scoring_evidence"]))
    raw_evidence["id"] = "core.boolean-evidence-alt@0.1.0"
    alternate["raw_scoring_evidence"] = raw_evidence
    return MeasurementDataset.from_record({"measurements": [measurement, alternate]})


def _required_proposal_set(bundle: SubmissionPublicationBundle) -> ExperimentProposalSet:
    proposal_set = bundle.proposal_set
    assert proposal_set is not None
    return proposal_set


def _required_architecture_surrogate(
    bundle: SubmissionPublicationBundle,
) -> ArchitectureSurrogateRecord:
    architecture_surrogate = bundle.architecture_surrogate
    assert architecture_surrogate is not None
    return architecture_surrogate


def _unchecked_publication_bundle(
    *,
    bundle: SubmissionPublicationBundle,
    measurement_dataset: MeasurementDataset,
) -> SubmissionPublicationBundle:
    unchecked = object.__new__(SubmissionPublicationBundle)
    object.__setattr__(unchecked, "id", bundle.id)
    object.__setattr__(unchecked, "submission_package", bundle.submission_package)
    object.__setattr__(unchecked, "measurement_dataset", measurement_dataset)
    object.__setattr__(unchecked, "measurement_score_view", bundle.measurement_score_view)
    object.__setattr__(unchecked, "proposal_set", bundle.proposal_set)
    object.__setattr__(unchecked, "architecture_surrogate", bundle.architecture_surrogate)
    return unchecked


def _dict_record(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def capture_federation_error(
    action: Callable[[], object],
) -> FederationValidationError:
    try:
        action()
    except FederationValidationError as error:
        return error
    raise AssertionError("expected FederationValidationError")
