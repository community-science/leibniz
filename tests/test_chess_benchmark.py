from pathlib import Path

from leibniz.benchmark_implementations import load_benchmark
from leibniz.identifiers import ProtocolIdentifier

_repository_root = Path(__file__).parents[1]
_fixture_path = _repository_root / "tests" / "fixtures" / "chess" / "mate_in_one"


def test_chess_mate_in_one_manifest_is_python_owned() -> None:
    benchmark = load_benchmark(_fixture_path)

    assert benchmark.manifest.id == ProtocolIdentifier.parse("benchmarks.chess@0.1.0")
    assert benchmark.manifest.observation_ids == frozenset(
        {"fen:7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"}
    )
