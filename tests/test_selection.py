from collections.abc import Callable
from pathlib import Path

from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.content import ContentDigest
from leibniz.measurements import MeasurementDataset, MeasurementDocument
from leibniz.proposals import ExperimentProposalSet
from leibniz.relationships import RelationshipFitRecord
from leibniz.selection import (
    ActiveSelectionResult,
    ActiveSelectionValidationError,
    select_experiments,
)
from leibniz.submissions import SubmissionPackageManifest
from leibniz.surrogates import ArchitectureSurrogateRecord

_fixtures_root = Path(__file__).parent / "fixtures"


def test_select_experiments_bootstraps_from_declared_proposals() -> None:
    dataset = _measurement_dataset()
    architecture = _architecture()
    relationship_fit = _relationship_fit(dataset=dataset, architecture=architecture)
    submission = _submission_package(dataset=dataset, architecture=architecture)
    proposal_set = _proposal_set(
        dataset=dataset,
        relationship_fit=relationship_fit,
        architecture=architecture,
        submission=submission,
    )

    selection = select_experiments(
        proposal_set,
        dataset=dataset,
        relationship_fit=relationship_fit,
        architectures=(architecture,),
        submission_packages=(submission,),
    )

    assert selection == ActiveSelectionResult(
        selected_proposals=proposal_set.proposals[:1],
        selection_rule="deterministic-bootstrap",
        source_dataset_digest=dataset.digest,
        source_proposal_set_digest=proposal_set.digest,
        relationship_fit_digest=relationship_fit.digest,
    )
    assert selection.to_record() == {
        "selection_rule": "deterministic-bootstrap",
        "source_dataset_digest": str(dataset.digest),
        "source_proposal_set_digest": str(proposal_set.digest),
        "selected_proposals": [proposal_set.proposals[0].to_record()],
        "relationship_fit_digest": str(relationship_fit.digest),
    }


def test_select_experiments_uses_surrogate_metadata_for_selection_mode() -> None:
    dataset = _measurement_dataset()
    architecture = _architecture()
    relationship_fit = _relationship_fit(dataset=dataset, architecture=architecture)
    submission = _submission_package(dataset=dataset, architecture=architecture)
    proposal_set = _proposal_set(
        dataset=dataset,
        relationship_fit=relationship_fit,
        architecture=architecture,
        submission=submission,
    )
    surrogate = _surrogate(dataset=dataset)

    selection = select_experiments(
        proposal_set,
        dataset=dataset,
        relationship_fit=relationship_fit,
        surrogate=surrogate,
        architectures=(architecture,),
        submission_packages=(submission,),
        limit=2,
    )

    assert selection.selection_rule == "surrogate-informed-declared-rank"
    assert selection.selected_proposals == proposal_set.proposals
    assert selection.surrogate_digest == surrogate.digest
    assert selection.to_record()["surrogate_digest"] == str(surrogate.digest)

    bootstrap_selection = select_experiments(
        proposal_set,
        dataset=dataset,
        relationship_fit=relationship_fit,
        surrogate=surrogate,
        architectures=(architecture,),
        submission_packages=(submission,),
        min_surrogate_observations=2,
    )

    assert bootstrap_selection.selection_rule == "deterministic-bootstrap"


def test_select_experiments_validates_sources_and_candidates() -> None:
    dataset = _measurement_dataset()
    architecture = _architecture()
    relationship_fit = _relationship_fit(dataset=dataset, architecture=architecture)
    submission = _submission_package(dataset=dataset, architecture=architecture)
    proposal_set = _proposal_set(
        dataset=dataset,
        relationship_fit=relationship_fit,
        architecture=architecture,
        submission=submission,
    )

    assert str(
        capture_selection_error(
            lambda: select_experiments(
                proposal_set,
                dataset=dataset,
                relationship_fit=relationship_fit,
                architectures=(),
                submission_packages=(submission,),
            )
        )
    ) == f"unknown architecture candidate: {architecture.id}"

    surrogate = ArchitectureSurrogateRecord.from_record(
        _surrogate_record(dataset=dataset),
        dataset=dataset,
    )
    mismatched_surrogate = ArchitectureSurrogateRecord(
        id=surrogate.id,
        source_dataset_digest=ContentDigest.from_value({"other": True}),
        model_kind=surrogate.model_kind,
        target_name=surrogate.target_name,
        features=surrogate.features,
        training=surrogate.training,
        state=surrogate.state,
    )
    assert str(
        capture_selection_error(
            lambda: select_experiments(
                proposal_set,
                dataset=dataset,
                relationship_fit=relationship_fit,
                surrogate=mismatched_surrogate,
                architectures=(architecture,),
                submission_packages=(submission,),
            )
        )
    ) == "source_dataset_digest does not match dataset"


def test_select_experiments_rejects_invalid_limits() -> None:
    dataset = _measurement_dataset()
    architecture = _architecture()
    relationship_fit = _relationship_fit(dataset=dataset, architecture=architecture)
    submission = _submission_package(dataset=dataset, architecture=architecture)
    proposal_set = _proposal_set(
        dataset=dataset,
        relationship_fit=relationship_fit,
        architecture=architecture,
        submission=submission,
    )

    assert str(
        capture_selection_error(
            lambda: select_experiments(
                proposal_set,
                dataset=dataset,
                relationship_fit=relationship_fit,
                architectures=(architecture,),
                submission_packages=(submission,),
                limit=0,
            )
        )
    ) == "limit must be positive"

    assert str(
        capture_selection_error(
            lambda: select_experiments(
                proposal_set,
                dataset=dataset,
                relationship_fit=relationship_fit,
                architectures=(architecture,),
                submission_packages=(submission,),
                min_surrogate_observations=-1,
            )
        )
    ) == "min_surrogate_observations must be nonnegative"


def _proposal_set(
    *,
    dataset: MeasurementDataset,
    relationship_fit: RelationshipFitRecord,
    architecture: ArchitectureManifest,
    submission: SubmissionPackageManifest,
) -> ExperimentProposalSet:
    return ExperimentProposalSet.from_record(
        {
            "id": "experiment-proposal-sets.boolean@0.1.0",
            "source_dataset_digest": str(dataset.digest),
            "relationship_fit_digest": str(relationship_fit.digest),
            "proposals": [
                {
                    "id": "experiment-proposals.boolean.rank-1@0.1.0",
                    "rank": 1,
                    "candidate_kind": "architecture",
                    "candidate_id": str(architecture.id),
                    "rationale": "first unmeasured architecture declaration",
                    "source_relationship_fit_id": str(relationship_fit.id),
                },
                {
                    "id": "experiment-proposals.boolean.rank-2@0.1.0",
                    "rank": 2,
                    "candidate_kind": "submission-package",
                    "candidate_id": str(submission.id),
                    "rationale": "complete submission package declaration",
                },
            ],
        },
        dataset=dataset,
        relationship_fit=relationship_fit,
        architectures=(architecture,),
        submission_packages=(submission,),
    )


def _measurement_dataset() -> MeasurementDataset:
    measurement = MeasurementDocument.from_bytes(
        (_fixtures_root / "finite_outcome" / "measurement.json").read_bytes()
    ).measurement
    return MeasurementDataset.from_record(
        {
            "measurements": [measurement.to_record()],
        }
    )


def _architecture() -> ArchitectureManifest:
    return ArchitectureManifestDocument.from_bytes(
        (_fixtures_root / "architecture" / "digits_pool" / "manifest.json").read_bytes()
    ).manifest


def _relationship_fit(
    *,
    dataset: MeasurementDataset,
    architecture: ArchitectureManifest,
) -> RelationshipFitRecord:
    return RelationshipFitRecord.from_record(
        {
            "id": "relationship-fits.boolean-linear@0.1.0",
            "source_dataset_digest": str(dataset.digest),
            "architecture_id": str(architecture.id),
            "hypothesis_family": "affine-score-vs-parameter-count",
            "parameters": [{"name": "intercept", "value": 1.0}],
            "residuals": {"rmse": 0.1, "max_abs": 0.2},
            "point_count": 1,
        },
        dataset=dataset,
        architecture=architecture,
    )


def _submission_package(
    *,
    dataset: MeasurementDataset,
    architecture: ArchitectureManifest,
) -> SubmissionPackageManifest:
    from leibniz.benchmarks import BenchmarkManifestDocument

    record: dict[str, object] = {
        "id": "submissions.boolean-digits-pool@0.1.0",
        "benchmark_manifest": BenchmarkManifestDocument.from_bytes(
            (_fixtures_root / "finite_outcome" / "manifest.json").read_bytes()
        ).manifest.to_record(),
        "architecture_manifest": architecture.to_record(),
        "measurement_dataset": dataset.to_record(),
    }
    return SubmissionPackageManifest.from_record(record)


def _surrogate(*, dataset: MeasurementDataset) -> ArchitectureSurrogateRecord:
    return ArchitectureSurrogateRecord.from_record(
        _surrogate_record(dataset=dataset),
        dataset=dataset,
    )


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
            },
            {
                "name": "parameter_count",
                "mean": 20.0,
                "scale": 5.0,
                "sensitivity": 0.3,
            },
        ],
        "training": {
            "status": "fit",
            "observation_count": 1,
        },
        "state": {
            "format": "dense-regressor-summary",
            "input_width": 2,
            "output_width": 1,
            "parameter_count": 7,
            "state_digest": str(ContentDigest.from_value({"weights": [0.2, -0.1]})),
        },
    }


def capture_selection_error(
    action: Callable[[], object],
) -> ActiveSelectionValidationError:
    try:
        action()
    except ActiveSelectionValidationError as error:
        return error
    raise AssertionError("expected ActiveSelectionValidationError")
