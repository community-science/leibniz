from pathlib import Path

import pytest

from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.content import ContentDigest
from leibniz.measurements import MeasurementDocument

_fixtures_root = Path(__file__).parent / "fixtures"


def _paired_fixtures() -> tuple[Path, ...]:
    return tuple(
        sorted(
            path.parent
            for path in _fixtures_root.rglob("manifest.json")
            if (path.parent / "measurement.json").is_file()
        )
    )


_paired_fixture_paths = _paired_fixtures()


@pytest.mark.parametrize(
    "fixture",
    _paired_fixture_paths,
    ids=lambda path: path.relative_to(_fixtures_root).as_posix(),
)
def test_authoring_fixture_pair_loads_and_validates(fixture: Path) -> None:
    manifest_document = BenchmarkManifestDocument.from_bytes(
        (fixture / "manifest.json").read_bytes()
    )
    measurement_document = MeasurementDocument.from_bytes(
        (fixture / "measurement.json").read_bytes()
    )

    measurement_document.measurement.validate_manifest(manifest_document.manifest)
    assert manifest_document.digest == ContentDigest.from_value(
        manifest_document.manifest.to_record()
    )
    assert measurement_document.digest == ContentDigest.from_value(
        measurement_document.measurement.to_record()
    )
    assert measurement_document.measurement.raw_scoring_evidence.to_record()
