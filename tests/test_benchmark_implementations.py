from pathlib import Path

import pytest
from benchmark_typing import load_digits_benchmark

from leibniz.benchmark_implementations import (
    BenchmarkError,
    discover_benchmark_roots,
    load_benchmark,
)
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.latent_factors import LatentFactorDeclarationDocument
from leibniz.materialization import MaterializationDeclarationDocument
from leibniz.observation_formation import ObservationFormationDeclarationDocument
from leibniz.observation_generation import (
    load_generator,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def test_digits_benchmark_loads_python_implementation_entrypoint() -> None:
    implementation = load_benchmark(_digits_benchmark_root)

    assert implementation.root == _digits_benchmark_root
    assert str(implementation.benchmark_manifest.id) == "benchmarks.digits@0.1.0"
    assert callable(implementation.generator)
    assert implementation.generator.__class__.__name__ == "Generator"


def test_digits_python_implementation_matches_exported_declarations() -> None:
    implementation = load_digits_benchmark(_digits_benchmark_root)

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


def test_generator_loads_through_benchmark_implementation() -> None:
    generator = load_generator(_digits_benchmark_root)

    assert str(generator.benchmark_manifest.id) == "benchmarks.digits@0.1.0"
    sample_set = generator(shape=1, seed=101)
    assert str(sample_set.generator_id) == "benchmarks.digits.generator@0.1.0"
    assert sample_set.shape == (1,)
    assert sample_set.samples
    assert not sample_set.includes_fields


def test_benchmark_loader_requires_python_entrypoint(tmp_path: Path) -> None:
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

    with pytest.raises(BenchmarkError, match="missing benchmark.py"):
        load_benchmark(benchmark_root)


def test_discover_benchmark_roots_uses_python_entrypoints(
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

    assert roots == (python_root,)
