"""Command line artifact validation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from leibniz.artifacts import ArtifactIndexDocument, ArtifactReferenceDocument
from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestDocument
from leibniz.documents import load_object_document
from leibniz.federation_ingest import FederationIngestPlanDocument
from leibniz.measurements import (
    MeasurementDataset,
    MeasurementDatasetDocument,
    MeasurementDocument,
    MeasurementRecord,
)
from leibniz.model_derivations import ModelDerivationCompatibilityReportDocument
from leibniz.model_interfaces import ModelInterfaceDocument
from leibniz.model_lineage import ModelLineageDocument
from leibniz.model_manifests import ModelArtifactManifestDocument
from leibniz.model_operations import ModelOperationDocument
from leibniz.outcomes import OutcomeSpace
from leibniz.projection_records import ProjectionRecordDocument
from leibniz.publications import SubmissionPublicationDocument
from leibniz.resources import ResourceReportDocument, ResourceReportSetDocument
from leibniz.submission_registries import SubmissionRegistry, SubmissionRegistryDocument
from leibniz.view_manifests import ViewManifestDocument

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Leibniz command line interface."""

    parser = _parser()
    args = parser.parse_args(argv)
    command = getattr(args, "command", None)
    if command == "validate":
        return _validate(args)
    parser.print_help(sys.stderr)
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leibniz")
    subcommands = parser.add_subparsers(dest="command")

    validate = subcommands.add_parser(
        "validate",
        description="validate artifact files",
        help="validate artifact files",
    )
    validate_subcommands = validate.add_subparsers(dest="artifact", required=True)

    manifest = validate_subcommands.add_parser(
        "manifest",
        help="validate a benchmark manifest document",
    )
    manifest.add_argument("path", type=Path)

    measurement = validate_subcommands.add_parser(
        "measurement",
        help="validate a measurement document",
    )
    measurement.add_argument("path", type=Path)
    measurement.add_argument("--manifest", type=Path)

    dataset = validate_subcommands.add_parser(
        "dataset",
        help="validate a measurement dataset document",
    )
    dataset.add_argument("path", type=Path)
    dataset.add_argument("--manifest", type=Path)

    artifact_reference = validate_subcommands.add_parser(
        "artifact-reference",
        help="validate an artifact reference document",
    )
    artifact_reference.add_argument("path", type=Path)

    artifact_index = validate_subcommands.add_parser(
        "artifact-index",
        help="validate an artifact index document",
    )
    artifact_index.add_argument("path", type=Path)

    resource_report = validate_subcommands.add_parser(
        "resource-report",
        help="validate a resource report document",
    )
    resource_report.add_argument("path", type=Path)

    resource_report_set = validate_subcommands.add_parser(
        "resource-report-set",
        help="validate a resource report set document",
    )
    resource_report_set.add_argument("path", type=Path)

    model_interface = validate_subcommands.add_parser(
        "model-interface",
        help="validate a model interface document",
    )
    model_interface.add_argument("path", type=Path)
    model_interface.add_argument("--outcome-space", type=Path, required=True)

    model_manifest = validate_subcommands.add_parser(
        "model-manifest",
        help="validate a model artifact manifest document",
    )
    model_manifest.add_argument("path", type=Path)

    model_operation = validate_subcommands.add_parser(
        "model-operation",
        help="validate a model operation document",
    )
    model_operation.add_argument("path", type=Path)

    model_lineage = validate_subcommands.add_parser(
        "model-lineage",
        help="validate a model lineage document",
    )
    model_lineage.add_argument("path", type=Path)

    view_manifest = validate_subcommands.add_parser(
        "view-manifest",
        help="validate a view manifest document",
    )
    view_manifest.add_argument("path", type=Path)

    projection_record = validate_subcommands.add_parser(
        "projection-record",
        help="validate a projection record document",
    )
    projection_record.add_argument("path", type=Path)

    model_derivation = validate_subcommands.add_parser(
        "model-derivation",
        help="validate a model derivation compatibility report document",
    )
    model_derivation.add_argument("path", type=Path)

    publication_bundle = validate_subcommands.add_parser(
        "publication-bundle",
        help="validate a submission publication bundle document",
    )
    publication_bundle.add_argument("path", type=Path)

    submission_registry = validate_subcommands.add_parser(
        "submission-registry",
        help="validate a submission registry document",
    )
    submission_registry.add_argument("path", type=Path)

    federation_ingest_plan = validate_subcommands.add_parser(
        "federation-ingest-plan",
        help="validate a federation ingest plan document",
    )
    federation_ingest_plan.add_argument("path", type=Path)
    federation_ingest_plan.add_argument("--registry", type=Path)

    return parser


def _validate(args: argparse.Namespace) -> int:
    try:
        artifact = str(args.artifact)
        if artifact == "manifest":
            manifest = _load_manifest(args.path)
            print(f"valid manifest {manifest.id}")
            return 0
        if artifact == "measurement":
            measurement = _load_measurement(args.path)
            manifest_path = getattr(args, "manifest", None)
            if manifest_path is not None:
                measurement.validate_manifest(_load_manifest(manifest_path))
            print(f"valid measurement {measurement.raw_scoring_evidence.id}")
            return 0
        if artifact == "dataset":
            dataset = _load_dataset(args.path)
            manifest_path = getattr(args, "manifest", None)
            if manifest_path is not None:
                dataset.validate_manifest(_load_manifest(manifest_path))
            print(f"valid measurement dataset ({len(dataset.measurements)} measurements)")
            return 0
        if artifact == "artifact-reference":
            document = ArtifactReferenceDocument.from_bytes(args.path.read_bytes())
            print(f"valid artifact reference {document.digest}")
            return 0
        if artifact == "artifact-index":
            document = ArtifactIndexDocument.from_bytes(args.path.read_bytes())
            print(f"valid artifact index {document.index.id}")
            return 0
        if artifact == "resource-report":
            document = ResourceReportDocument.from_bytes(args.path.read_bytes())
            print(f"valid resource report {document.report.id}")
            return 0
        if artifact == "resource-report-set":
            document = ResourceReportSetDocument.from_bytes(args.path.read_bytes())
            print(f"valid resource report set {document.report_set.id}")
            return 0
        if artifact == "model-interface":
            document = ModelInterfaceDocument.from_bytes(
                args.path.read_bytes(),
                outcome_space=_load_outcome_space(args.outcome_space),
            )
            print(f"valid model interface {document.interface.id}")
            return 0
        if artifact == "model-manifest":
            document = ModelArtifactManifestDocument.from_bytes(args.path.read_bytes())
            print(f"valid model manifest {document.manifest.id}")
            return 0
        if artifact == "model-operation":
            document = ModelOperationDocument.from_bytes(args.path.read_bytes())
            print(f"valid model operation {document.operation.id}")
            return 0
        if artifact == "model-lineage":
            document = ModelLineageDocument.from_bytes(args.path.read_bytes())
            print(f"valid model lineage {document.lineage.id}")
            return 0
        if artifact == "view-manifest":
            document = ViewManifestDocument.from_bytes(args.path.read_bytes())
            print(f"valid view manifest {document.manifest.id}")
            return 0
        if artifact == "projection-record":
            document = ProjectionRecordDocument.from_bytes(args.path.read_bytes())
            print(f"valid projection record {document.record.id}")
            return 0
        if artifact == "model-derivation":
            document = ModelDerivationCompatibilityReportDocument.from_bytes(args.path.read_bytes())
            print(f"valid model derivation compatibility report {document.report.id}")
            return 0
        if artifact == "publication-bundle":
            document = SubmissionPublicationDocument.from_bytes(args.path.read_bytes())
            print(f"valid publication bundle {document.bundle.id}")
            return 0
        if artifact == "submission-registry":
            document = SubmissionRegistryDocument.from_bytes(args.path.read_bytes())
            print(f"valid submission registry {document.registry.id}")
            return 0
        if artifact == "federation-ingest-plan":
            registry_path = getattr(args, "registry", None)
            registry = _load_submission_registry(registry_path) if registry_path else None
            document = FederationIngestPlanDocument.from_bytes(
                args.path.read_bytes(),
                registry=registry,
            )
            print(f"valid federation ingest plan {document.plan.id}")
            return 0
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"error: unsupported validation artifact {args.artifact!r}", file=sys.stderr)
    return 2


def _load_manifest(path: Path) -> BenchmarkManifest:
    return BenchmarkManifestDocument.from_bytes(path.read_bytes()).manifest


def _load_measurement(path: Path) -> MeasurementRecord:
    return MeasurementDocument.from_bytes(path.read_bytes()).measurement


def _load_dataset(path: Path) -> MeasurementDataset:
    return MeasurementDatasetDocument.from_bytes(path.read_bytes()).dataset


def _load_outcome_space(path: Path) -> OutcomeSpace:
    return OutcomeSpace.from_record(
        load_object_document(path.read_bytes(), description="outcome space document")
    )


def _load_submission_registry(path: Path) -> SubmissionRegistry:
    return SubmissionRegistryDocument.from_bytes(path.read_bytes()).registry
