from pathlib import Path

from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.identifiers import ProtocolIdentifier

_repository_root = Path(__file__).parents[1]
_fixture_path = _repository_root / "tests" / "fixtures" / "chess" / "mate_in_one"
_manifest_path = _fixture_path / "manifest.json"


def test_chess_mate_in_one_manifest_loads() -> None:
    manifest_document = BenchmarkManifestDocument.from_bytes(_manifest_path.read_bytes())

    assert manifest_document.manifest.id == ProtocolIdentifier.parse("benchmarks.chess@0.1.0")
    assert manifest_document.manifest.observation_ids == frozenset(
        {"fen:7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"}
    )
