from collections.abc import Callable
from pathlib import Path
from typing import cast

from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset, MeasurementDocument
from leibniz.proposals import (
    ExperimentProposal,
    ExperimentProposalDocument,
    ExperimentProposalSet,
    ExperimentProposalValidationError,
)
from leibniz.relationships import RelationshipFitRecord
from leibniz.submissions import SubmissionPackageManifest

_fixtures_root = Path(__file__).parent / "fixtures"


def test_experiment_proposal_set_parses_and_canonicalizes() -> None:
    dataset = _measurement_dataset()
    architecture = _architecture()
    relationship_fit = _relationship_fit(dataset=dataset, architecture=architecture)
    submission = _submission_package(dataset=dataset, architecture=architecture)

    proposal_set = ExperimentProposalSet.from_record(
        _proposal_set_record(
            dataset=dataset,
            relationship_fit=relationship_fit,
            architecture=architecture,
            submission=submission,
        ),
        dataset=dataset,
        relationship_fit=relationship_fit,
        architectures=(architecture,),
        submission_packages=(submission,),
    )

    assert proposal_set == ExperimentProposalSet(
        id=ProtocolIdentifier.parse("experiment-proposal-sets.boolean@0.1.0"),
        source_dataset_digest=dataset.digest,
        relationship_fit_digest=relationship_fit.digest,
        proposals=(
            ExperimentProposal(
                id=ProtocolIdentifier.parse("experiment-proposals.boolean.rank-1@0.1.0"),
                rank=1,
                candidate_kind="architecture",
                candidate_id=architecture.id,
                rationale="lowest observed score leaves this architecture family untested",
                source_relationship_fit_id=relationship_fit.id,
            ),
            ExperimentProposal(
                id=ProtocolIdentifier.parse("experiment-proposals.boolean.rank-2@0.1.0"),
                rank=2,
                candidate_kind="submission-package",
                candidate_id=submission.id,
                rationale="submission package provides a complete evidence bundle",
                source_relationship_fit_id=None,
            ),
        ),
    )
    assert proposal_set.to_record() == _expanded_proposal_set_record(
        proposal_set,
        dataset=dataset,
        relationship_fit=relationship_fit,
    )
    assert proposal_set.digest == ContentDigest.from_value(proposal_set.to_record())


def test_experiment_proposal_document_loads_bytes_with_digest() -> None:
    dataset = _measurement_dataset()
    architecture = _architecture()
    relationship_fit = _relationship_fit(dataset=dataset, architecture=architecture)
    submission = _submission_package(dataset=dataset, architecture=architecture)

    document = ExperimentProposalDocument.from_bytes(
        canonical_document_bytes(
            _proposal_set_record(
                dataset=dataset,
                relationship_fit=relationship_fit,
                architecture=architecture,
                submission=submission,
            )
        ),
        dataset=dataset,
        relationship_fit=relationship_fit,
        architectures=(architecture,),
        submission_packages=(submission,),
    )

    assert document.proposal_set.source_dataset_digest == dataset.digest
    assert document.digest == ContentDigest.from_value(document.proposal_set.to_record())


def test_experiment_proposal_rejects_source_mismatches() -> None:
    dataset = _measurement_dataset()
    architecture = _architecture()
    relationship_fit = _relationship_fit(dataset=dataset, architecture=architecture)
    submission = _submission_package(dataset=dataset, architecture=architecture)

    record = _proposal_set_record(
        dataset=dataset,
        relationship_fit=relationship_fit,
        architecture=architecture,
        submission=submission,
    )
    record["source_dataset_digest"] = str(ContentDigest.from_value({"other": True}))
    assert str(
        capture_proposal_error(
            lambda: ExperimentProposalSet.from_record(
                record,
                dataset=dataset,
                relationship_fit=relationship_fit,
                architectures=(architecture,),
                submission_packages=(submission,),
            )
        )
    ) == "source_dataset_digest does not match dataset"

    record = _proposal_set_record(
        dataset=dataset,
        relationship_fit=relationship_fit,
        architecture=architecture,
        submission=submission,
    )
    record["relationship_fit_digest"] = str(ContentDigest.from_value({"other": True}))
    assert str(
        capture_proposal_error(
            lambda: ExperimentProposalSet.from_record(
                record,
                dataset=dataset,
                relationship_fit=relationship_fit,
                architectures=(architecture,),
                submission_packages=(submission,),
            )
        )
    ) == "relationship_fit_digest does not match relationship fit"


def test_experiment_proposal_rejects_candidate_reference_failures() -> None:
    dataset = _measurement_dataset()
    architecture = _architecture()
    relationship_fit = _relationship_fit(dataset=dataset, architecture=architecture)
    submission = _submission_package(dataset=dataset, architecture=architecture)

    record = _proposal_set_record(
        dataset=dataset,
        relationship_fit=relationship_fit,
        architecture=architecture,
        submission=submission,
    )
    proposals = _proposal_records(record)
    first = dict(proposals[0])
    first["candidate_id"] = "architecture.sha-0@0.1.0"
    proposals[0] = first
    record["proposals"] = proposals
    assert str(
        capture_proposal_error(
            lambda: ExperimentProposalSet.from_record(
                record,
                dataset=dataset,
                relationship_fit=relationship_fit,
                architectures=(architecture,),
                submission_packages=(submission,),
            )
        )
    ) == "unknown architecture candidate: architecture.sha-0@0.1.0"

    record = _proposal_set_record(
        dataset=dataset,
        relationship_fit=relationship_fit,
        architecture=architecture,
        submission=submission,
    )
    proposals = _proposal_records(record)
    first = dict(proposals[0])
    first["source_relationship_fit_id"] = "relationship-fits.other@0.1.0"
    proposals[0] = first
    record["proposals"] = proposals
    assert str(
        capture_proposal_error(
            lambda: ExperimentProposalSet.from_record(
                record,
                dataset=dataset,
                relationship_fit=relationship_fit,
                architectures=(architecture,),
                submission_packages=(submission,),
            )
        )
    ) == "source_relationship_fit_id does not match relationship fit"


def test_experiment_proposal_rejects_malformed_proposals() -> None:
    dataset = _measurement_dataset()
    architecture = _architecture()
    relationship_fit = _relationship_fit(dataset=dataset, architecture=architecture)
    submission = _submission_package(dataset=dataset, architecture=architecture)

    record = _proposal_set_record(
        dataset=dataset,
        relationship_fit=relationship_fit,
        architecture=architecture,
        submission=submission,
    )
    proposals = _proposal_records(record)
    first = dict(proposals[0])
    first["rank"] = 2
    proposals[0] = first
    record["proposals"] = proposals
    assert str(
        capture_proposal_error(
            lambda: ExperimentProposalSet.from_record(
                record,
                dataset=dataset,
                relationship_fit=relationship_fit,
                architectures=(architecture,),
                submission_packages=(submission,),
            )
        )
    ) == "duplicate proposal rank: 2"

    record = _proposal_set_record(
        dataset=dataset,
        relationship_fit=relationship_fit,
        architecture=architecture,
        submission=submission,
    )
    proposals = _proposal_records(record)
    first = dict(proposals[0])
    first["candidate_kind"] = "checkpoint"
    proposals[0] = first
    record["proposals"] = proposals
    assert str(
        capture_proposal_error(
            lambda: ExperimentProposalSet.from_record(
                record,
                dataset=dataset,
                relationship_fit=relationship_fit,
                architectures=(architecture,),
                submission_packages=(submission,),
            )
        )
    ) == "unsupported candidate_kind: checkpoint"

    record = _proposal_set_record(
        dataset=dataset,
        relationship_fit=relationship_fit,
        architecture=architecture,
        submission=submission,
    )
    proposals = _proposal_records(record)
    first = dict(proposals[0])
    first["rationale"] = ""
    proposals[0] = first
    record["proposals"] = proposals
    assert str(
        capture_proposal_error(
            lambda: ExperimentProposalSet.from_record(
                record,
                dataset=dataset,
                relationship_fit=relationship_fit,
                architectures=(architecture,),
                submission_packages=(submission,),
            )
        )
    ) == "rationale must be nonempty"


def test_experiment_proposal_document_rejects_invalid_bytes() -> None:
    assert str(
        capture_proposal_error(
            lambda: ExperimentProposalDocument.from_bytes(b"[]", dataset=_measurement_dataset())
        )
    ) == "experiment proposal document must contain an object"


def _proposal_set_record(
    *,
    dataset: MeasurementDataset,
    relationship_fit: RelationshipFitRecord,
    architecture: ArchitectureManifest,
    submission: SubmissionPackageManifest,
) -> dict[str, object]:
    return {
        "id": "experiment-proposal-sets.boolean@0.1.0",
        "source_dataset_digest": str(dataset.digest),
        "relationship_fit_digest": str(relationship_fit.digest),
        "proposals": [
            {
                "id": "experiment-proposals.boolean.rank-1@0.1.0",
                "rank": 1,
                "candidate_kind": "architecture",
                "candidate_id": str(architecture.id),
                "rationale": "lowest observed score leaves this architecture family untested",
                "source_relationship_fit_id": str(relationship_fit.id),
            },
            {
                "id": "experiment-proposals.boolean.rank-2@0.1.0",
                "rank": 2,
                "candidate_kind": "submission-package",
                "candidate_id": str(submission.id),
                "rationale": "submission package provides a complete evidence bundle",
            },
        ],
    }


def _expanded_proposal_set_record(
    proposal_set: ExperimentProposalSet,
    *,
    dataset: MeasurementDataset,
    relationship_fit: RelationshipFitRecord,
) -> dict[str, object]:
    return {
        "id": "experiment-proposal-sets.boolean@0.1.0",
        "source_dataset_digest": str(dataset.digest),
        "proposals": [proposal.to_record() for proposal in proposal_set.proposals],
        "relationship_fit_digest": str(relationship_fit.digest),
    }


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


def _proposal_records(record: dict[str, object]) -> list[dict[str, object]]:
    proposals = record["proposals"]
    assert isinstance(proposals, list)
    items = cast(list[object], proposals)
    return [dict(_proposal_record(proposal)) for proposal in items]


def _proposal_record(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def capture_proposal_error(
    action: Callable[[], object],
) -> ExperimentProposalValidationError:
    try:
        action()
    except ExperimentProposalValidationError as error:
        return error
    raise AssertionError("expected ExperimentProposalValidationError")
