from pathlib import Path

from leibniz.benchmark_implementations import (
    DeclarationBackedBenchmarkImplementation,
    discover_benchmark_roots,
    load_benchmark_implementation,
)
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.latent_factors import LatentFactorDeclarationDocument
from leibniz.materialization import MaterializationDeclarationDocument
from leibniz.observation_formation import ObservationFormationDeclarationDocument
from leibniz.observation_generation import ObservationGenerator, load_observation_generator

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def test_digits_benchmark_loads_python_implementation_entrypoint() -> None:
    implementation = load_benchmark_implementation(_digits_benchmark_root)

    assert implementation.root == _digits_benchmark_root
    assert not isinstance(implementation, DeclarationBackedBenchmarkImplementation)
    assert str(implementation.benchmark_manifest.id) == "benchmarks.digits@0.1.0"
    assert isinstance(implementation.observation_generator(), ObservationGenerator)


def test_digits_python_implementation_matches_exported_declarations() -> None:
    implementation = load_benchmark_implementation(_digits_benchmark_root)

    manifest = BenchmarkManifestDocument.from_bytes(
        (_digits_benchmark_root / "manifest.json").read_bytes()
    ).manifest
    latent_factors = LatentFactorDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "latent_factors.json").read_bytes()
    ).declaration
    materialization = MaterializationDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "materialization.json").read_bytes()
    ).declaration
    formation = ObservationFormationDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "observation_formation.json").read_bytes()
    ).declaration

    assert implementation.benchmark_manifest == manifest
    assert implementation.latent_factors == latent_factors
    assert implementation.materialization == materialization
    assert implementation.formation == formation


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


def test_discover_benchmark_roots_uses_entrypoint_or_transitional_manifest(
    tmp_path: Path,
) -> None:
    python_root = tmp_path / "python-benchmark"
    declaration_root = tmp_path / "declaration-benchmark"
    ignored_root = tmp_path / "__pycache__"
    python_root.mkdir()
    declaration_root.mkdir()
    ignored_root.mkdir()
    (python_root / "benchmark.py").write_text("", encoding="utf-8")
    (declaration_root / "manifest.json").write_text("{}", encoding="utf-8")

    roots = discover_benchmark_roots(tmp_path)

    assert roots == (declaration_root, python_root)
