from pathlib import Path

from leibniz.benchmark_implementations import (
    DeclarationBackedBenchmarkImplementation,
    load_benchmark_implementation,
)
from leibniz.observation_generation import ObservationGenerator, load_observation_generator

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def test_digits_benchmark_loads_python_implementation_entrypoint() -> None:
    implementation = load_benchmark_implementation(_digits_benchmark_root)

    assert implementation.root == _digits_benchmark_root
    assert str(implementation.benchmark_manifest.id) == "benchmarks.digits@0.1.0"
    assert isinstance(implementation.observation_generator(), ObservationGenerator)


def test_observation_generator_loads_through_benchmark_implementation() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    assert str(generator.benchmark_manifest.id) == "benchmarks.digits@0.1.0"
    assert generator.sample_batch(component_count=1, sample_count=1, seed=101).samples


def test_benchmark_loader_keeps_declaration_backed_transition_path(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "digits"
    benchmark_root.mkdir()
    for name in (
        "manifest",
        "latent_factors",
        "materialization",
        "observation_formation",
    ):
        source = _digits_benchmark_root / (name + ".json")
        target = benchmark_root / source.name
        target.write_bytes(source.read_bytes())

    implementation = load_benchmark_implementation(benchmark_root)

    assert isinstance(implementation, DeclarationBackedBenchmarkImplementation)
    assert str(implementation.benchmark_manifest.id) == "benchmarks.digits@0.1.0"
