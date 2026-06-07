import math
from pathlib import Path
from typing import Any, cast

import chess
import pytest

from leibniz.benchmark_evaluation import finite_measurements_for_predictions
from leibniz.benchmark_implementations import discover_benchmark_roots, load_benchmark
from leibniz.observation_generation import (
    ComplexityRequest,
    ObservationGenerationError,
    load_generator,
)
from leibniz.tensor_runtime import resolve_tensor_runtime, tensor_value_to_host

_repository_root = Path(__file__).parents[1]
_benchmark_parent = _repository_root / "src" / "leibniz" / "benchmarks"
_chess_benchmark_root = _benchmark_parent / "chess"


def test_chess_generator_loads_through_benchmark_entrypoint() -> None:
    roots = discover_benchmark_roots(_benchmark_parent)

    assert _chess_benchmark_root in roots
    benchmark = load_benchmark(_chess_benchmark_root)
    generator = benchmark.generator

    assert str(benchmark.manifest.id) == "benchmarks.chess@0.1.0"
    assert str(generator.id) == "benchmarks.chess.generator@0.1.0"
    assert not hasattr(benchmark, "materialization")
    assert not hasattr(generator, "formation")


def test_chess_generator_exposes_python_manifest() -> None:
    generator = load_generator(_chess_benchmark_root)

    assert str(generator.manifest.id) == "benchmarks.chess@0.1.0"
    observation_ids = generator.manifest.observation_ids
    assert observation_ids is not None
    assert len(observation_ids) >= 6
    assert "fen:7k/6Q1/6K1/8/8/8/8/8 w - - 0 1" in observation_ids
    outcome_ids = tuple(outcome.id for outcome in generator.manifest.outcome_space.outcomes)
    assert outcome_ids == tuple(sorted(outcome_ids))
    assert "e2e4" in outcome_ids
    assert "g1f3" in outcome_ids
    assert "e1g1" in outcome_ids
    assert "e7e8q" in outcome_ids
    assert "e2e2" not in outcome_ids


def test_chess_generator_returns_complexity_valued_samples_without_fields() -> None:
    generator = load_generator(_chess_benchmark_root)
    complexity = generator.minimum_complexity().value
    request = ComplexityRequest(minimum=complexity, maximum=complexity)
    sample_set = generator(
        seed=47,
        shape=3,
        include_fields=True,
        complexity_request=request,
    )

    assert sample_set.shape == (3,)
    assert sample_set.sample_count == 3
    assert not sample_set.includes_fields
    sample = sample_set.samples[0]
    legal_move_count = _legal_move_count(sample)
    assert sample_set.outcomes == (sample.outcome_id,) * 3
    assert sample_set.complexities == (math.log2(legal_move_count),) * 3
    assert sample.complexity_value is not None
    assert sample.complexity_value.measure_id == ComplexityRequest(
        minimum=1.0,
        maximum=1.0,
    ).measure_id
    assert sample.complexity_value.value == math.log2(legal_move_count)
    assert sample.target_distribution is not None
    assert sample.outcome_id in sample.target_distribution
    assert sum(sample.target_distribution.values()) == 1.0
    assert len(sample.available_outcome_ids) == legal_move_count
    assert sample.outcome_id in sample.available_outcome_ids
    assert sample.latent_coordinates[0]["values"] == "8/8/8/8/8/8/k1K5/2Q5 w - - 0 1"
    assert sample.latent_coordinates[1]["values"] == legal_move_count
    assert len(cast(list[object], sample.latent_coordinates[2]["values"])) == legal_move_count
    with pytest.raises(ObservationGenerationError, match="does not include generated field"):
        sample.require_field()


def test_chess_sample_record_does_not_invent_image_surface_fields() -> None:
    generator = load_generator(_chess_benchmark_root)
    record = generator(seed=47, shape=()).samples[0].to_record(include_field=True)

    target_distribution = cast(list[dict[str, object]], record["target_distribution"])
    target_outcomes = {
        cast(str, entry["outcome_id"])
        for entry in target_distribution
    }
    assert record["outcome_id"] in target_outcomes
    available_outcomes = cast(list[str], record["available_outcome_ids"])
    assert record["outcome_id"] in available_outcomes
    assert len(available_outcomes) == _legal_move_count_complexity_cardinality(record)
    assert record["complexity_value"] == {
        "measure_id": ComplexityRequest(minimum=1.0, maximum=1.0).measure_id,
        "value": _legal_move_count_complexity(record),
    }
    assert "materialization_plan" not in record
    assert "width" not in record
    assert "height" not in record
    assert "component_index" not in record
    assert "variation_coordinates" not in record
    assert "variation_values" not in record
    assert "field" not in record


def test_chess_measurements_use_declared_fen_observation_ids() -> None:
    generator = load_generator(_chess_benchmark_root)
    outcome_space = generator.manifest.resolve_outcome_space()
    outcome_ids = tuple(outcome.id for outcome in outcome_space.outcomes)
    sample_set = generator(seed=47, shape=3)
    probabilities = tuple(
        tuple(1.0 if outcome_id == sample.outcome_id else 0.0 for outcome_id in outcome_ids)
        for sample in sample_set.samples
    )

    measurements = finite_measurements_for_predictions(
        batch=sample_set,
        outcome_space=outcome_space,
        probabilities=probabilities,
        run_slug="chess-fen-observation-test",
    )

    assert [
        measurement.raw_scoring_evidence.observation_id
        for measurement in measurements
    ] == [sample.observable_state_id for sample in sample_set.samples]


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
    complexity = generator.minimum_complexity().value
    request = ComplexityRequest(
        minimum=complexity,
        maximum=complexity,
    )
    sample_set = generator(seed=47, shape=(2, 2), complexity_request=request)

    assert sample_set.shape == (2, 2)
    assert len(sample_set.samples) == 4
    assert all(sample.complexity_value is not None for sample in sample_set.samples)


def test_chess_complexity_candidates_are_legal_move_cardinalities() -> None:
    generator = load_generator(_chess_benchmark_root)
    request = ComplexityRequest(minimum=1.0, maximum=6.0)

    candidates = tuple(generator.complexity_candidates_for_request(request=request))

    cardinalities = tuple(candidate.cardinality for candidate in candidates)
    assert cardinalities == (18, 23, 24, 28, 32, 33)
    for candidate in candidates:
        assert candidate.cardinality is not None
        assert candidate.complexity == math.log2(candidate.cardinality)
        assert candidate.metadata["kind"] == "chess-legal-move-cardinality"
        assert candidate.metadata["legal_move_count"] == candidate.cardinality


def test_chess_corpus_targets_are_rules_validated_mate_in_one_moves() -> None:
    generator = load_generator(_chess_benchmark_root)
    sample_set = generator(seed=47, shape=128)
    seen_fens = {
        cast(str, sample.latent_coordinates[0]["values"])
        for sample in sample_set.samples
    }
    observation_ids = generator.manifest.observation_ids
    assert observation_ids is not None

    assert seen_fens == {
        observation_id.removeprefix("fen:")
        for observation_id in observation_ids
    }
    for sample in sample_set.samples:
        fen = cast(str, sample.latent_coordinates[0]["values"])
        board = chess.Board(fen)
        legal_moves = {move.uci() for move in board.legal_moves}
        mate_moves = {
            move.uci()
            for move in board.legal_moves
            if _is_checkmate_after_move(board, move)
        }
        assert set(sample.available_outcome_ids) == legal_moves
        assert set(sample.target_distribution_or_one_hot()) == mate_moves


def test_chess_generator_returns_board_tensors_and_move_targets() -> None:
    generator = load_generator(_chess_benchmark_root)
    runtime = resolve_tensor_runtime("cpu")
    outcome_ids = tuple(outcome.id for outcome in generator.manifest.outcome_space.outcomes)
    complexity = generator.minimum_complexity().value
    request = ComplexityRequest(minimum=complexity, maximum=complexity)

    sample_set = generator(
        seed=47,
        shape=2,
        runtime=runtime,
        outcome_ids=outcome_ids,
        complexity_request=request,
    )
    fields, targets = sample_set.require_tensors()

    assert tuple(fields.shape) == (2, 18, 8, 8)
    assert tuple(targets.shape) == (2, len(outcome_ids))
    field_values = tensor_value_to_host(fields).tolist()
    target_values = tensor_value_to_host(targets).tolist()
    assert field_values[0][4][0][2] == 1.0
    assert field_values[0][5][1][2] == 1.0
    assert field_values[0][11][1][0] == 1.0
    assert field_values[0][12][0][0] == 1.0
    assert math.isclose(sum(target_values[0]), 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert sample_set.samples[0].outcome_id in outcome_ids
    assert target_values[0][outcome_ids.index(sample_set.samples[0].outcome_id)] > 0.0


def test_chess_generator_can_return_metadata_free_tensors() -> None:
    generator = load_generator(_chess_benchmark_root)
    runtime = resolve_tensor_runtime("cpu")
    outcome_ids = tuple(outcome.id for outcome in generator.manifest.outcome_space.outcomes)

    sample_set = generator(
        seed=47,
        shape=3,
        include_metadata=False,
        runtime=runtime,
        outcome_ids=outcome_ids,
    )

    fields, targets = sample_set.require_tensors()
    assert sample_set.samples == ()
    assert sample_set.includes_fields
    assert tuple(fields.shape) == (3, 18, 8, 8)
    assert tuple(targets.shape) == (3, len(outcome_ids))


def test_chess_console_preview_uses_text_samples() -> None:
    generator = load_generator(_chess_benchmark_root)
    atom_count = len(generator.manifest.outcome_space.outcomes)

    batches = tuple(cast(Any, generator).console_preview_batches(atom_count=atom_count))

    assert len(batches) == 6
    batch = batches[0]
    assert batch["mode"] == "complexity-window"
    assert batch["sample_count"] == 1
    sample = cast(list[dict[str, object]], batch["samples"])[0]
    assert sample["observable_state_id"] == (
        "fen:8/8/8/8/8/8/k1K5/2Q5 w - - 0 1"
    )
    assert "image_data_url" not in sample
    assert "target_distribution" in sample
    assert "available_outcome_ids" in sample


def _legal_move_count(sample: Any) -> int:
    legal_count = cast(tuple[dict[str, object], ...], sample.latent_coordinates)[1]["values"]
    assert isinstance(legal_count, int)
    return legal_count


def _legal_move_count_complexity(record: dict[str, object]) -> float:
    return math.log2(_legal_move_count_complexity_cardinality(record))


def _legal_move_count_complexity_cardinality(record: dict[str, object]) -> int:
    coordinates = cast(list[dict[str, object]], record["latent_coordinates"])
    legal_count = coordinates[1]["values"]
    assert isinstance(legal_count, int)
    return legal_count


def _is_checkmate_after_move(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    try:
        return board.is_checkmate()
    finally:
        board.pop()
