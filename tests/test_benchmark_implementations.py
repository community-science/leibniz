from pathlib import Path

import pytest
from benchmark_typing import load_digits_benchmark

from leibniz.benchmark_implementations import (
    BenchmarkError,
    discover_benchmark_roots,
    load_benchmark,
)
from leibniz.observation_generation import (
    load_generator,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def test_digits_benchmark_loads_python_implementation_entrypoint() -> None:
    implementation = load_benchmark(_digits_benchmark_root)

    assert implementation.root == _digits_benchmark_root
    assert str(implementation.manifest.id) == "benchmarks.digits@0.1.0"
    assert callable(implementation.generator)
    assert implementation.generator.__class__.__name__ == "Generator"
    assert implementation.target_contract.kind == "finite-outcome"
    assert implementation.target_contract.loss_id == "cross-entropy"
    assert implementation.target_contract.expected_output_shape(None) == (10,)
    assert implementation.target_contract.chance_mass() == 0.1
    assert implementation.sampling_protocol.kind == "uniform-monte-carlo"
    assert implementation.sampling_protocol.confidence_method_id == "wilson"
    assert implementation.accessible_subspace.ladder_id == "digits-continuous-transform-covering"
    assert not implementation.accessible_subspace.exclusions


def test_digits_python_implementation_owns_declarations() -> None:
    implementation = load_digits_benchmark(_digits_benchmark_root)

    implementation.manifest.validate_latent_factor_declaration(
        implementation.latent_factors
    )
    assert implementation.materialization.benchmark_id == implementation.manifest.id
    assert implementation.formation.benchmark_id == implementation.manifest.id
    assert implementation.materialization.latent_factor_declaration is not None
    assert (
        implementation.materialization.latent_factor_declaration.protocol_id
        == implementation.latent_factors.id
    )
    assert implementation.showcase.benchmark_id == implementation.manifest.id
    assert implementation.sampling_protocol.to_record()["kind"] == "uniform-monte-carlo"
    sample_set = implementation.generator(shape=1, seed=101)
    assert sample_set.region is not None
    assert (
        implementation.accessible_subspace.per_configuration_capacity.ambient
        == sample_set.region.ambient
    )


def test_generator_loads_through_benchmark_implementation() -> None:
    generator = load_generator(_digits_benchmark_root)

    assert str(generator.manifest.id) == "benchmarks.digits@0.1.0"
    sample_set = generator(shape=1, seed=101)
    assert str(sample_set.generator_id) == "benchmarks.digits.generator@0.1.0"
    assert sample_set.shape == (1,)
    assert sample_set.samples
    assert not sample_set.includes_fields


def test_benchmark_loader_uses_implementation_target_contract(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "field"
    benchmark_root.mkdir()
    (benchmark_root / "benchmark.py").write_text(
        "from pathlib import Path\n"
        "from leibniz.benchmark_implementations import RawBenchmark\n"
        "from leibniz.benchmarks import BenchmarkManifest\n"
        "from leibniz.identifiers import ProtocolIdentifier, ProtocolName\n"
        "from leibniz.outcomes import Outcome, OutcomeSpace\n"
        "from leibniz.state_space import AccessibleSubspace, Distinguishability, "
        "DiscreteAxisRegion, IntegerRangeDomain, ProductRegion, SamplingProtocol, "
        "StateSpaceAmbient, StateSpaceAxis, StateSpaceRegion\n"
        "from leibniz.target_contracts import BaselinePredictor, "
        "CompetenceFunctional, TargetContract\n"
        "\n"
        "class Generator:\n"
        "    id = ProtocolIdentifier.parse('benchmarks.field.generator@0.1.0')\n"
        "    version = '0.1.0'\n"
        "    manifest = BenchmarkManifest(\n"
        "        id=ProtocolIdentifier.parse('benchmarks.field@0.1.0'),\n"
        "        name=ProtocolName.parse('benchmarks.field'),\n"
        "        outcome_space=OutcomeSpace(\n"
        "            id=ProtocolIdentifier.parse('benchmarks.field.placeholder@0.1.0'),\n"
        "            outcomes=(Outcome(id='placeholder'),),\n"
        "        ),\n"
        "    )\n"
        "    def __call__(self, **kwargs): raise NotImplementedError\n"
        "    def minimum_log2_volume(self): raise NotImplementedError\n"
        "\n"
        "def _region():\n"
        "    axis = StateSpaceAxis('index', IntegerRangeDomain(0, 0))\n"
        "    axis_region = DiscreteAxisRegion(axis, (0, 0), 1, 0.0)\n"
        "    component = ProductRegion((axis_region,), 'product-of-counts', 1, 0.0)\n"
        "    return StateSpaceRegion(\n"
        "        'region',\n"
        "        StateSpaceAmbient(\n"
        "            'box-1d', {'length_x': 1.0, 'boundary_id': 'periodic'},\n"
        "            'scalar-field', Distinguishability('exact'),\n"
        "        ),\n"
        "        (component,), 'disjoint-union', 1, 0.0,\n"
        "    )\n"
        "\n"
        "class Benchmark:\n"
        "    root = Path('.')\n"
        "    manifest = Generator.manifest\n"
        "    generator = Generator()\n"
        "    sampling_protocol = SamplingProtocol('census', census_budget=1)\n"
        "    accessible_subspace = AccessibleSubspace('field', _region(), (), 'open')\n"
        "    target_contract = TargetContract(\n"
        "        kind='field-valued', outcome_ids=None, loss_id='mse',\n"
        "        competence=CompetenceFunctional(\n"
        "            'mass-within-resolution',\n"
        "            {'residual_operator_id': 'operators.field@0.1.0'},\n"
        "        ),\n"
        "        baseline=BaselinePredictor('zero-field'),\n"
        "    )\n"
        "\n"
        "def benchmark(root: Path) -> RawBenchmark:\n"
        "    return Benchmark()\n",
        encoding="utf-8",
    )

    implementation = load_benchmark(benchmark_root)

    assert implementation.target_contract.kind == "field-valued"
    assert implementation.target_contract.loss_id == "mse"


def test_benchmark_loader_requires_python_entrypoint(tmp_path: Path) -> None:
    benchmark_root = tmp_path / "digits"
    benchmark_root.mkdir()
    (benchmark_root / "manifest.json").write_text("{}", encoding="utf-8")

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
