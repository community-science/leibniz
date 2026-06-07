from pathlib import Path

from leibniz.benchmark_implementations import load_benchmark
from leibniz.identifiers import ProtocolIdentifier

_repository_root = Path(__file__).parents[1]
_benchmark_path = _repository_root / "src" / "leibniz" / "benchmarks" / "chess"
_minimum_chess_fen = "8/8/8/8/8/8/2Q5/krK5 w - - 0 1"


def test_chess_mate_in_one_manifest_is_python_owned() -> None:
    benchmark = load_benchmark(_benchmark_path)

    assert benchmark.manifest.id == ProtocolIdentifier.parse("benchmarks.chess@0.1.0")
    observation_ids = benchmark.manifest.observation_ids
    assert observation_ids is not None
    assert len(observation_ids) == 32
    assert f"fen:{_minimum_chess_fen}" in observation_ids
    outcome_ids = benchmark.manifest.outcome_space.outcome_ids
    assert "e2e4" in outcome_ids
    assert "e7e8q" in outcome_ids
