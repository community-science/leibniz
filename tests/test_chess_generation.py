import math
from base64 import b64decode
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
_expected_legal_move_cardinalities = tuple(range(2, 34))
_expected_sample_cardinalities = tuple(range(1, 33))
_minimum_chess_fen = "8/8/8/8/8/8/2Q5/krK5 w - - 0 1"


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
    assert len(observation_ids) == len(_expected_legal_move_cardinalities)
    assert f"fen:{_minimum_chess_fen}" in observation_ids
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
    legal_move_count = _sample_legal_move_count(sample)
    assert sample_set.outcomes == (sample.outcome_id,) * 3
    assert sample_set.complexities == (0.0,) * 3
    assert sample.complexity_value is not None
    assert sample.complexity_value.measure_id == ComplexityRequest(
        minimum=0.0,
        maximum=0.0,
    ).measure_id
    assert sample.complexity_value.value == 0.0
    assert sample.target_distribution is not None
    assert sample.outcome_id in sample.target_distribution
    assert sum(sample.target_distribution.values()) == 1.0
    assert len(sample.available_outcome_ids) == legal_move_count
    assert sample.outcome_id in sample.available_outcome_ids
    assert sample.latent_coordinates[0]["values"] == 1
    assert sample.latent_coordinates[1]["values"] == _minimum_chess_fen
    assert sample.latent_coordinates[2]["values"] == legal_move_count
    assert len(cast(list[object], sample.latent_coordinates[3]["values"])) == legal_move_count
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
    assert len(available_outcomes) == _legal_move_count(record)
    assert record["complexity_value"] == {
        "measure_id": ComplexityRequest(minimum=0.0, maximum=0.0).measure_id,
        "value": math.log2(_sample_space_cardinality(record)),
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
        minimum=1.5,
        maximum=1.5,
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


def test_chess_complexity_candidates_are_sample_space_cardinalities() -> None:
    generator = load_generator(_chess_benchmark_root)
    request = ComplexityRequest(minimum=0.0, maximum=5.0)

    candidates = tuple(generator.complexity_candidates_for_request(request=request))

    cardinalities = tuple(candidate.cardinality for candidate in candidates)
    assert cardinalities == _expected_sample_cardinalities
    for candidate in candidates:
        assert candidate.cardinality is not None
        assert candidate.complexity == math.log2(candidate.cardinality)
        assert candidate.metadata["kind"] == "chess-sample-space-cardinality"
        assert candidate.metadata["sample_cardinality"] == candidate.cardinality
        assert candidate.metadata["target_policy"] == "mate-in-one"
        assert len(cast(list[object], candidate.metadata["representatives"])) == (
            candidate.cardinality
        )
        assert len(cast(list[object], candidate.metadata["legal_move_counts"])) == (
            candidate.cardinality
        )


def test_chess_complexity_curriculum_uses_supported_sample_cardinalities() -> None:
    generator = load_generator(_chess_benchmark_root)

    candidates = tuple(
        generator.complexity_curriculum_candidates(start_index=0, count=5)
    )

    cardinalities = tuple(candidate.cardinality for candidate in candidates)
    assert cardinalities == _expected_sample_cardinalities[:5]
    for candidate in candidates:
        assert candidate.cardinality is not None
        assert math.isclose(candidate.complexity, math.log2(candidate.cardinality))
        assert candidate.request.minimum == candidate.request.maximum


def test_chess_corpus_targets_are_rules_validated_mate_in_one_moves() -> None:
    generator = load_generator(_chess_benchmark_root)
    observation_ids = generator.manifest.observation_ids
    assert observation_ids is not None

    legal_move_counts: list[int] = []
    for observation_id in observation_ids:
        fen = observation_id.removeprefix("fen:")
        board = chess.Board(fen)
        legal_moves = {move.uci() for move in board.legal_moves}
        mate_moves = {
            move.uci()
            for move in board.legal_moves
            if _is_checkmate_after_move(board, move)
        }
        assert legal_moves
        assert mate_moves
        legal_move_counts.append(len(legal_moves))
    assert tuple(sorted(legal_move_counts)) == _expected_legal_move_cardinalities


def test_chess_representative_ladder_declares_piece_coverage() -> None:
    generator = load_generator(_chess_benchmark_root)
    candidates = tuple(
        generator.complexity_curriculum_candidates(
            start_index=0,
            count=len(_expected_legal_move_cardinalities),
        )
    )

    assert tuple(candidate.cardinality for candidate in candidates) == (
        _expected_sample_cardinalities
    )
    legal_piece_symbols = {
        symbol
        for candidate in candidates
        for representative in cast(list[dict[str, object]], candidate.metadata["representatives"])
        for symbol in cast(
            list[str],
            representative["legal_move_piece_symbols"],
        )
    }
    mate_piece_symbols = {
        symbol
        for candidate in candidates
        for representative in cast(list[dict[str, object]], candidate.metadata["representatives"])
        for symbol in cast(
            list[str],
            representative["mate_move_piece_symbols"],
        )
    }

    assert {"B", "K", "N", "Q", "R"} <= legal_piece_symbols
    assert "Q" in mate_piece_symbols
    assert mate_piece_symbols != {"Q"}


def test_chess_representative_analysis_exposes_deterministic_quality_flags() -> None:
    generator = load_generator(_chess_benchmark_root)
    candidates = tuple(
        generator.complexity_curriculum_candidates(
            start_index=0,
            count=len(_expected_legal_move_cardinalities),
        )
    )

    flags_by_legal_move_count = {
        representative["legal_move_count"]: frozenset(
            flag
            for flag in cast(
                list[str],
                cast(dict[str, object], representative["analysis"])["quality_flags"],
            )
        )
        for candidate in candidates
        for representative in cast(
            list[dict[str, object]],
            candidate.metadata["representatives"],
        )
    }

    assert flags_by_legal_move_count[2] == {
        "minimal-material",
        "queen-only-mate",
        "single-mate-move",
    }
    assert "single-legal-piece-type" in flags_by_legal_move_count[15]
    assert any(
        "queen-only-mate" not in flags
        for flags in flags_by_legal_move_count.values()
    )
    assert (
        sum("minimal-material" in flags for flags in flags_by_legal_move_count.values())
        >= 16
    )
    assert (
        sum("queen-only-mate" in flags for flags in flags_by_legal_move_count.values())
        >= 24
    )


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
    assert field_values[0][4][1][2] == 1.0
    assert field_values[0][5][0][2] == 1.0
    assert field_values[0][9][0][1] == 1.0
    assert field_values[0][11][0][0] == 1.0
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


def test_chess_console_preview_uses_board_images_and_text_metadata() -> None:
    generator = load_generator(_chess_benchmark_root)
    atom_count = len(generator.manifest.outcome_space.outcomes)

    batches = tuple(cast(Any, generator).console_preview_batches(atom_count=atom_count))

    assert len(batches) == len(_expected_legal_move_cardinalities)
    batch = batches[0]
    assert batch["mode"] == "complexity-window"
    assert batch["sample_count"] == 1
    sample = cast(list[dict[str, object]], batch["samples"])[0]
    assert sample["observable_state_id"] == f"fen:{_minimum_chess_fen}"
    coverage_coordinate = cast(list[dict[str, object]], sample["latent_coordinates"])[5]
    assert coverage_coordinate["name"] == (
        "benchmarks.chess.position.representative-piece-coverage"
    )
    coverage_values = cast(dict[str, object], coverage_coordinate["values"])
    assert coverage_values["legal_move_piece_symbols"] == ["K", "Q"]
    assert coverage_values["mate_move_piece_symbols"] == ["Q"]
    coverage_analysis = cast(dict[str, object], coverage_values["analysis"])
    assert coverage_analysis["quality_flags"] == [
        "minimal-material",
        "single-mate-move",
        "queen-only-mate",
    ]
    image_data_url = cast(str, sample["image_data_url"])
    assert image_data_url.startswith("data:image/svg+xml;base64,")
    svg = _decode_svg_data_url(image_data_url)
    assert 'viewBox="0 0 512 512"' in svg
    assert 'aria-label="Q"' in svg
    assert 'aria-label="K"' in svg
    assert 'aria-label="k"' in svg
    overlay = cast(dict[str, object], sample["image_overlay"])
    assert overlay["kind"] == "grid-move-highlights"
    assert overlay["columns"] == 8
    assert overlay["rows"] == 8
    moves = cast(list[dict[str, object]], overlay["moves"])
    assert any(move["from"] == [2, 6] and move["to"] == [1, 7] for move in moves)
    target_moves = [
        move for move in moves if cast(float, move["target_probability"]) > 0.0
    ]
    assert len(target_moves) == len(cast(list[object], sample["target_distribution"]))
    assert cast(float, target_moves[0]["target_probability"]) == 1.0
    assert "target_distribution" in sample
    assert "available_outcome_ids" in sample


def _legal_move_count(record: dict[str, object]) -> int:
    coordinates = cast(list[dict[str, object]], record["latent_coordinates"])
    legal_count = coordinates[2]["values"]
    assert isinstance(legal_count, int)
    return legal_count


def _sample_legal_move_count(sample: Any) -> int:
    legal_count = cast(tuple[dict[str, object], ...], sample.latent_coordinates)[2][
        "values"
    ]
    assert isinstance(legal_count, int)
    return legal_count


def _sample_space_cardinality(record: dict[str, object]) -> int:
    coordinates = cast(list[dict[str, object]], record["latent_coordinates"])
    cardinality = coordinates[0]["values"]
    assert isinstance(cardinality, int)
    return cardinality


def _is_checkmate_after_move(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    try:
        return board.is_checkmate()
    finally:
        board.pop()


def _decode_svg_data_url(data_url: str) -> str:
    prefix = "data:image/svg+xml;base64,"
    assert data_url.startswith(prefix)
    return b64decode(data_url.removeprefix(prefix)).decode("utf-8")
