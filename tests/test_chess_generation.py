import math
from pathlib import Path

import pytest

from leibniz.benchmark_implementations import discover_benchmark_roots, load_benchmark
from leibniz.observation_generation import (
    ComplexityRequest,
    ObservationGenerationError,
    load_generator,
)

_repository_root = Path(__file__).parents[1]
_chess_parent = _repository_root / "tests" / "fixtures" / "chess"
_chess_benchmark_root = _chess_parent / "mate_in_one"


def test_chess_generator_loads_through_benchmark_entrypoint() -> None:
    roots = discover_benchmark_roots(_chess_parent)

    assert roots == (_chess_benchmark_root,)
    benchmark = load_benchmark(_chess_benchmark_root)
    generator = benchmark.generator

    assert str(benchmark.manifest.id) == "benchmarks.chess@0.1.0"
    assert str(generator.id) == "benchmarks.chess.generator@0.1.0"
    assert not hasattr(benchmark, "materialization")
    assert not hasattr(generator, "formation")


def test_chess_generator_exposes_python_manifest() -> None:
    generator = load_generator(_chess_benchmark_root)

    assert str(generator.manifest.id) == "benchmarks.chess@0.1.0"
    assert generator.manifest.observation_ids == frozenset(
        {"fen:7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"}
    )


def test_chess_generator_returns_complexity_valued_samples_without_fields() -> None:
    generator = load_generator(_chess_benchmark_root)
    sample_set = generator(seed=47, shape=3, include_fields=True)

    assert sample_set.shape == (3,)
    assert sample_set.sample_count == 3
    assert not sample_set.includes_fields
    assert sample_set.outcomes == ("g7f8", "g7f8", "g7f8")
    assert sample_set.complexities == (math.log2(3), math.log2(3), math.log2(3))
    sample = sample_set.samples[0]
    assert sample.complexity_value is not None
    assert sample.complexity_value.measure_id == ComplexityRequest(
        minimum=1.0,
        maximum=1.0,
    ).measure_id
    assert sample.complexity_value.value == math.log2(3)
    assert sample.latent_coordinates[0]["values"] == (
        "7k/6Q1/6K1/8/8/8/8/8 w - - 0 1"
    )
    assert sample.latent_coordinates[1]["values"] == 3
    with pytest.raises(ObservationGenerationError, match="does not include generated field"):
        sample.require_field()


def test_chess_sample_record_does_not_invent_image_surface_fields() -> None:
    generator = load_generator(_chess_benchmark_root)
    record = generator(seed=47, shape=()).samples[0].to_record(include_field=True)

    assert record["outcome_id"] == "g7f8"
    assert record["complexity_value"] == {
        "measure_id": ComplexityRequest(minimum=1.0, maximum=1.0).measure_id,
        "value": math.log2(3),
    }
    assert "materialization_plan" not in record
    assert "width" not in record
    assert "height" not in record
    assert "component_index" not in record
    assert "variation_coordinates" not in record
    assert "variation_values" not in record
    assert "field" not in record


def test_chess_complexity_request_returns_empty_set_for_unexpressible_interval() -> None:
    generator = load_generator(_chess_benchmark_root)
    request = ComplexityRequest(
        minimum=1.0,
        maximum=1.0,
    )
    sample_set = generator(seed=47, shape=5, complexity_request=request)

    assert sample_set.shape == (0,)
    assert sample_set.samples == ()
    assert sample_set.complexity_request == request
    assert sample_set.to_record()["sample_count"] == 0


def test_chess_complexity_request_accepts_matching_interval() -> None:
    generator = load_generator(_chess_benchmark_root)
    request = ComplexityRequest(
        minimum=math.log2(3),
        maximum=math.log2(3),
    )
    sample_set = generator(seed=47, shape=(2, 2), complexity_request=request)

    assert sample_set.shape == (2, 2)
    assert len(sample_set.samples) == 4
    assert all(sample.complexity_value is not None for sample in sample_set.samples)
