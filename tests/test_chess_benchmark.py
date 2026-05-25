from pathlib import Path

from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDocument

_repository_root = Path(__file__).parents[1]
_fixture_path = _repository_root / "tests" / "fixtures" / "chess" / "mate_in_one"
_manifest_path = _fixture_path / "manifest.json"
_measurement_path = _fixture_path / "measurement.json"


def test_chess_mate_in_one_manifest_loads() -> None:
    manifest_document = BenchmarkManifestDocument.from_bytes(_manifest_path.read_bytes())

    assert manifest_document.manifest.id == ProtocolIdentifier.parse("benchmarks.chess@0.1.0")
    assert manifest_document.manifest.observation_ids == frozenset(
        {"fen:7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"}
    )


def test_chess_mate_in_one_measurement_validates_against_manifest() -> None:
    manifest_document = BenchmarkManifestDocument.from_bytes(_manifest_path.read_bytes())
    measurement_document = MeasurementDocument.from_bytes(_measurement_path.read_bytes())

    measurement_document.measurement.validate_manifest(manifest_document.manifest)
    assert manifest_document.digest == ContentDigest.from_value(
        manifest_document.manifest.to_record()
    )
    assert measurement_document.digest == ContentDigest.from_value(
        measurement_document.measurement.to_record()
    )
    assert measurement_document.measurement.scoring_bundle.raw_scoring_evidence.to_record() == {
        "id": "benchmarks.chess.fixture.mate-in-one-evidence@0.1.0",
        "observation_id": "fen:7k/6Q1/6K1/8/8/8/8/8 w - - 0 1",
        "outcome_space_id": "benchmarks.chess.uci-moves@0.1.0",
        "accepted_event_id": "benchmarks.chess.fixture.mate-in-one-accepted@0.1.0",
        "probability_measure_id": "benchmarks.chess.fixture.mate-in-one-prediction@0.1.0",
        "accepted_mass": 0.7,
        "negative_log_score": 0.35667494393873245,
    }
