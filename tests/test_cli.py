from pathlib import Path

import pytest

from leibniz._documents import canonical_document_bytes
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.cli import main
from leibniz.measurements import MeasurementDocument

_fixtures_root = Path(__file__).parent / "fixtures"
_finite_fixture = _fixtures_root / "finite_outcome"


def test_cli_help_text(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["validate", "--help"])

    assert exit_info.value.code == 0
    assert "validate artifact files" in capsys.readouterr().out


def test_cli_validates_manifest_file(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["validate", "manifest", str(_finite_fixture / "manifest.json")])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "valid manifest core.boolean-benchmark@0.1.0\n"
    assert captured.err == ""


def test_cli_validates_measurement_file(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["validate", "measurement", str(_finite_fixture / "measurement.json")])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "valid measurement core.boolean-evidence@0.1.0\n"
    assert captured.err == ""


def test_cli_validates_measurement_file_against_manifest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "validate",
            "measurement",
            str(_finite_fixture / "measurement.json"),
            "--manifest",
            str(_finite_fixture / "manifest.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "valid measurement core.boolean-evidence@0.1.0\n"
    assert captured.err == ""


def test_cli_validates_dataset_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = _dataset_path(tmp_path)

    exit_code = main(["validate", "dataset", str(dataset_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "valid measurement dataset (1 measurements)\n"
    assert captured.err == ""


def test_cli_validates_dataset_file_against_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_path = _dataset_path(tmp_path)

    exit_code = main(
        [
            "validate",
            "dataset",
            str(dataset_path),
            "--manifest",
            str(_finite_fixture / "manifest.json"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "valid measurement dataset (1 measurements)\n"
    assert captured.err == ""


def test_cli_reports_malformed_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("[]", encoding="utf-8")

    exit_code = main(["validate", "manifest", str(manifest_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == "error: manifest document must contain an object\n"


def test_cli_reports_incompatible_manifest_pair(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest = BenchmarkManifestDocument.from_bytes(
        (_finite_fixture / "manifest.json").read_bytes()
    ).manifest.to_record()
    manifest["id"] = "core.other-benchmark@0.1.0"
    manifest["name"] = "core.other-benchmark"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_document_bytes(manifest))

    exit_code = main(
        [
            "validate",
            "measurement",
            str(_finite_fixture / "measurement.json"),
            "--manifest",
            str(manifest_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == (
        "error: benchmark_id core.boolean-benchmark@0.1.0 does not match manifest "
        "core.other-benchmark@0.1.0\n"
    )


def _dataset_path(tmp_path: Path) -> Path:
    measurement = MeasurementDocument.from_bytes(
        (_finite_fixture / "measurement.json").read_bytes()
    ).measurement.to_record()
    dataset_path = tmp_path / "measurements.json"
    dataset_path.write_bytes(canonical_document_bytes({"measurements": [measurement]}))
    return dataset_path
