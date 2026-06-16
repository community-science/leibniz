from pathlib import Path

from leibniz.benchmark_implementations import load_benchmark
from leibniz.identifiers import ProtocolIdentifier

_repository_root = Path(__file__).parents[1]
_benchmark_path = _repository_root / "src" / "leibniz" / "benchmarks" / "chess"
def test_chess_mate_in_one_manifest_is_python_owned() -> None:
    benchmark = load_benchmark(_benchmark_path)

    assert benchmark.manifest.id == ProtocolIdentifier.parse("benchmarks.chess@0.1.0")
    assert benchmark.manifest.observation_ids is None
    outcome_space = benchmark.manifest.outcome_space
    assert outcome_space is not None
    outcome_ids = outcome_space.outcome_ids
    assert "e2e4" in outcome_ids
    assert "e7e8q" in outcome_ids
