"""Command line artifact validation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from leibniz.benchmarks import BenchmarkManifest, BenchmarkManifestDocument
from leibniz.measurements import (
    MeasurementDataset,
    MeasurementDatasetDocument,
    MeasurementDocument,
    MeasurementRecord,
)

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
