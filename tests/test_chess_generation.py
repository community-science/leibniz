import importlib.util
import math
import sys
from base64 import b64decode
from collections.abc import Callable
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
    sample_indices_for_even_state_coverage,
)
from leibniz.state_space import state_space_region_from_record
from leibniz.tensor_runtime import resolve_tensor_runtime, tensor_value_to_host

_repository_root = Path(__file__).parents[1]
_benchmark_parent = _repository_root / "src" / "leibniz" / "benchmarks"
_chess_benchmark_root = _benchmark_parent / "chess"
_expected_transform_count = 8
_expected_preview_limit = 4
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
    assert generator.manifest.observation_ids is None
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
    assert sample.target_distribution is None
    assert sample_set.region is not None
    assert sample_set.request_outcome is not None
    assert sample_set.request_outcome.kind == "realized"
    assert sample_set.request_outcome.region == sample_set.region
    assert sample_set.region.volume == 1
    assert len(sample_set.region.components) == 1
    component = sample_set.region.components[0]
    assert component.measure_rule == "benchmark-computed-finite-count"
    assert component.volume == 1
    spectator_regions = [
        axis_region
        for axis_region in component.axis_regions
        if axis_region.axis_id.endswith(".spectator-occupancy")
    ]
    assert len(spectator_regions) == 1
    assert spectator_regions[0].axis.coordinate_kind == "binary-vector"
    assert spectator_regions[0].coordinate_region == ()
    assert sample.region_component_index is not None
    assert sample.axis_coordinates is not None
    assert sample_set.region.contains(sample.region_component_index, sample.axis_coordinates)
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

    available_outcomes = cast(list[str], record["available_outcome_ids"])
    assert record["outcome_id"] in available_outcomes
    assert "target_distribution" not in record
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
    assert sample_set.request_outcome is not None
    assert sample_set.request_outcome.kind == "unrepresentable-below-minimum"
    assert sample_set.to_record()["sample_count"] == 0
    assert sample_set.to_record()["request_outcome"] == {
        "kind": "unrepresentable-below-minimum"
    }


def test_chess_complexity_request_reports_exhausted_capacity() -> None:
    generator = load_generator(_chess_benchmark_root)
    request = ComplexityRequest(minimum=1000.0, maximum=1001.0)

    sample_set = generator(seed=47, shape=5, complexity_request=request)

    assert sample_set.shape == (0,)
    assert sample_set.samples == ()
    assert sample_set.request_outcome is not None
    assert sample_set.request_outcome.kind == "exhausted-capacity"
    assert sample_set.request_outcome.capacity_region is not None
    expected_capacity = _chess_family_capacity()
    assert sample_set.request_outcome.capacity_region.volume == expected_capacity
    assert len(sample_set.request_outcome.capacity_region.components) == 48
    assert {
        component.volume
        for component in sample_set.request_outcome.capacity_region.components
    } == {expected_capacity // 48}
    outcome_record = cast(dict[str, object], sample_set.to_record()["request_outcome"])
    assert outcome_record["kind"] == "exhausted-capacity"
    capacity_region = state_space_region_from_record(outcome_record["capacity_region"])
    assert capacity_region.volume == expected_capacity


def test_chess_realized_region_decomposes_exactly_per_stratum() -> None:
    generator = load_generator(_chess_benchmark_root)
    cardinality = 96
    complexity = math.log2(cardinality)
    sample_set = generator(
        seed=47,
        shape=8,
        complexity_request=ComplexityRequest(minimum=complexity, maximum=complexity),
    )

    assert sample_set.region is not None
    region = sample_set.region
    assert region.volume == cardinality
    assert math.isclose(region.log2_volume, complexity)
    assert region.ambient.field_domain_kind == "lattice-2d"
    assert region.ambient.field_domain == {"width": 8, "height": 8}
    assert region.ambient.field_codomain_id == "piece-occupancy"
    assert region.ambient.distinguishability.kind == "exact"
    assert len(region.components) == 48
    assert sum(component.volume for component in region.components) == cardinality
    assert {component.volume for component in region.components} == {2}
    assert {
        component.measure_rule for component in region.components
    } == {"benchmark-computed-finite-count"}
    assert {
        cast(dict[str, object], component.stratum_target)["base_index"]
        for component in region.components
    } == set(range(48))
    for component in region.components:
        stratum_target = cast(dict[str, object], component.stratum_target)
        assert stratum_target["spectator_rank_count"] == component.volume
        lower = cast(int, stratum_target["spectator_rank_lower"])
        upper = cast(int, stratum_target["spectator_rank_upper"])
        assert upper - lower + 1 == component.volume
    for sample in sample_set.samples:
        assert sample.region_component_index is not None
        assert sample.axis_coordinates is not None
        assert region.contains(sample.region_component_index, sample.axis_coordinates)


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


def test_chess_generator_does_not_expose_complexity_candidates() -> None:
    generator = load_generator(_chess_benchmark_root)

    assert not hasattr(generator, "complexity_candidate_for_request")
    assert not hasattr(generator, "complexity_curriculum_candidates")
    assert not hasattr(generator, "complexity_candidates_for_request")


def test_chess_integer_shell_requests_use_power_of_two_cardinalities() -> None:
    generator = load_generator(_chess_benchmark_root)

    for shell in range(5):
        sample_set = generator(
            seed=47 + shell,
            shape=2,
            complexity_request=ComplexityRequest(
                minimum=float(shell),
                maximum=float(shell + 1),
            ),
        )
        assert sample_set.samples
        assert {sample.complexity for sample in sample_set.samples} == {float(shell)}
        assert {
            _sample_space_cardinality(sample.to_record())
            for sample in sample_set.samples
        } == {2**shell}


def test_chess_power_of_two_shells_do_not_reuse_global_positions() -> None:
    global_sample_index = _chess_global_sample_index()
    seen: set[int] = set()

    for shell in range(8):
        cardinality = 2**shell
        shell_indices = {
            global_sample_index(cardinality=cardinality, local_index=local_index)
            for local_index in range(cardinality)
        }
        assert not seen & shell_indices
        seen.update(shell_indices)


def test_chess_indexed_family_samples_are_rules_validated_mate_in_one_moves() -> None:
    generator = load_generator(_chess_benchmark_root)
    requests = (
        ComplexityRequest(minimum=0.0, maximum=0.0),
        ComplexityRequest(minimum=3.0, maximum=3.0),
        ComplexityRequest(minimum=10.0, maximum=10.0),
        ComplexityRequest(minimum=32.0, maximum=32.0),
    )
    samples = tuple(
        sample
        for request in requests
        for sample in generator(seed=47, shape=8, complexity_request=request).samples
    )

    assert len(samples) == 32
    for sample in samples:
        assert sample.observable_state_id is not None
        fen = sample.observable_state_id.removeprefix("fen:")
        board = chess.Board(fen)
        legal_moves = {move.uci() for move in board.legal_moves}
        mate_moves = {
            move.uci()
            for move in board.legal_moves
            if _is_checkmate_after_move(board, move)
        }
        assert legal_moves
        assert mate_moves
        assert sample.outcome_id in mate_moves
        assert set(sample.available_outcome_ids) == legal_moves


def test_chess_indexed_family_expands_by_sample_cardinality() -> None:
    representatives_for_cardinality = _chess_representatives_for_cardinality()

    first_representative_set = representatives_for_cardinality(1)
    full_representative_set = representatives_for_cardinality(16)
    assert len(first_representative_set) == 1
    assert len(full_representative_set) == _expected_preview_limit

    family_indexes = [
        cast(int, representative["family_index"])
        for representative in full_representative_set
    ]
    assert len(family_indexes) == len(frozenset(family_indexes))
    assert min(family_indexes) > 0
    transforms = {
        cast(str, representative["transform"])
        for representative in full_representative_set
    }
    assert transforms <= {
        "identity",
        "mirror-file",
        "mirror-rank",
        "rotate-180",
        "transpose",
        "anti-transpose",
        "rotate-90",
        "rotate-270",
    }
    assert any(
        cast(int, representative["spectator_count"]) > 0
        for representative in full_representative_set
    )

    legal_piece_symbols = {
        symbol
        for cardinality in range(1, 17)
        for representative in representatives_for_cardinality(cardinality)
        for symbol in cast(
            list[str],
            representative["legal_move_piece_symbols"],
        )
    }
    mate_piece_symbols = {
        symbol
        for cardinality in range(1, 17)
        for representative in representatives_for_cardinality(cardinality)
        for symbol in cast(
            list[str],
            representative["mate_move_piece_symbols"],
        )
    }

    assert {"K", "Q"} <= legal_piece_symbols
    assert "Q" in mate_piece_symbols


def test_chess_cardinality_two_keeps_canonical_family_simple() -> None:
    generator = load_generator(_chess_benchmark_root)
    request = ComplexityRequest(minimum=1.0, maximum=1.0)

    sample_set = generator(seed=47, shape=2, complexity_request=request)

    spectator_counts = {
        _sample_analysis(sample)["spectator_count"] for sample in sample_set.samples
    }
    assert spectator_counts == {0}


def test_chess_sampling_does_not_repeat_before_exhausting_cardinality() -> None:
    generator = load_generator(_chess_benchmark_root)
    request = ComplexityRequest(minimum=4.0, maximum=4.0)

    sample_set = generator(seed=47, shape=16, complexity_request=request)

    observable_state_ids = tuple(sample.observable_state_id for sample in sample_set.samples)
    assert len(observable_state_ids) == len(frozenset(observable_state_ids))


def test_chess_larger_rungs_do_not_repeat_low_cardinality_boards() -> None:
    generator = load_generator(_chess_benchmark_root)
    low_request = ComplexityRequest(minimum=2.0, maximum=2.0)
    spectator_request = ComplexityRequest(minimum=4.0, maximum=4.0)

    low_sample_set = generator(seed=404, shape=4, complexity_request=low_request)
    spectator_sample_set = generator(
        seed=416,
        shape=4,
        complexity_request=spectator_request,
    )

    low_observable_ids = {
        sample.observable_state_id for sample in low_sample_set.samples
    }
    spectator_observable_ids = {
        sample.observable_state_id for sample in spectator_sample_set.samples
    }
    assert not low_observable_ids & spectator_observable_ids


def test_chess_varies_mate_mechanism_before_adding_spectators() -> None:
    generator = load_generator(_chess_benchmark_root)
    sparse_request = ComplexityRequest(minimum=2.0, maximum=2.0)
    supported_request = ComplexityRequest(minimum=3.0, maximum=3.0)

    sparse_sample_set = generator(seed=47, shape=4, complexity_request=sparse_request)
    supported_sample_set = generator(
        seed=47,
        shape=8,
        complexity_request=supported_request,
    )
    sparse_analyses = [_sample_analysis(sample) for sample in sparse_sample_set.samples]
    supported_analyses = [
        _sample_analysis(sample) for sample in supported_sample_set.samples
    ]

    assert {analysis["spectator_count"] for analysis in sparse_analyses} == {0}
    assert {analysis["mechanism_piece_count"] for analysis in sparse_analyses} == {4}
    assert len({analysis["mechanism"] for analysis in sparse_analyses}) > 1
    assert {analysis["spectator_count"] for analysis in supported_analyses} == {0}
    assert {analysis["mechanism_piece_count"] for analysis in supported_analyses} == {5}


def test_chess_adds_spectator_material_as_cardinality_grows() -> None:
    generator = load_generator(_chess_benchmark_root)
    request = ComplexityRequest(minimum=5.0, maximum=5.0)

    sample_set = generator(seed=47, shape=32, complexity_request=request)
    analyses = [_sample_analysis(sample) for sample in sample_set.samples]

    assert {analysis["spectator_count"] for analysis in analyses} == {2}


def test_chess_complete_small_cardinality_rungs_do_not_overlap() -> None:
    global_sample_index = _chess_global_sample_index()
    family_index_sets = {
        cardinality: {
            global_sample_index(
                cardinality=cardinality,
                local_index=local_index,
            )
            for local_index in range(cardinality)
        }
        for cardinality in range(1, 17)
    }

    for lower_cardinality, lower_family_indices in family_index_sets.items():
        for higher_cardinality, higher_family_indices in family_index_sets.items():
            if lower_cardinality >= higher_cardinality:
                continue
            assert not lower_family_indices & higher_family_indices


@pytest.mark.parametrize(
    ("lower_cardinality", "higher_cardinality"),
    [
        (16, 17),
        (31, 32),
        (32, 33),
        (63, 64),
        (64, 65),
        (127, 128),
        (128, 129),
    ],
)
def test_chess_complete_boundary_cardinality_rungs_do_not_overlap(
    lower_cardinality: int,
    higher_cardinality: int,
) -> None:
    global_sample_index = _chess_global_sample_index()
    lower_family_indices = {
        global_sample_index(
            cardinality=lower_cardinality,
            local_index=local_index,
        )
        for local_index in range(lower_cardinality)
    }
    higher_family_indices = {
        global_sample_index(
            cardinality=higher_cardinality,
            local_index=local_index,
        )
        for local_index in range(higher_cardinality)
    }

    assert not lower_family_indices & higher_family_indices


@pytest.mark.parametrize("cardinality", [1, 2, 4, 8, 9, 16, 32, 64, 257, 1024])
def test_chess_sample_mapping_has_no_repetition_within_cardinality(
    cardinality: int,
) -> None:
    global_sample_index = _chess_global_sample_index()
    family_indices = [
        global_sample_index(
            cardinality=cardinality,
            local_index=local_index,
        )
        for local_index in range(cardinality)
    ]

    assert len(family_indices) == len(frozenset(family_indices))


def test_chess_adjacent_cardinality_previews_mostly_avoid_repetition() -> None:
    representatives_for_cardinality = _chess_representatives_for_cardinality()
    preview_sets = [
        {
            cast(int, representative["family_index"])
            for representative in representatives_for_cardinality(cardinality)
        }
        for cardinality in (32, 33)
    ]

    assert len(preview_sets[0]) == _expected_preview_limit
    assert len(preview_sets[1]) == _expected_preview_limit
    assert not preview_sets[0] & preview_sets[1]


def test_chess_representative_analysis_exposes_indexed_family_metadata() -> None:
    representatives_for_cardinality = _chess_representatives_for_cardinality()

    flags_by_family_index = {
        representative["family_index"]: frozenset(
            flag
            for flag in cast(
                list[str],
                cast(dict[str, object], representative["analysis"])["quality_flags"],
            )
        )
        for cardinality in range(1, 17)
        for representative in representatives_for_cardinality(cardinality)
    }
    analyses = [
        cast(dict[str, object], representative["analysis"])
        for cardinality in range(1, 17)
        for representative in representatives_for_cardinality(cardinality)
    ]

    assert all(analysis["family"] == "corner-net-indexed-family" for analysis in analyses)
    assert {analysis["transform"] for analysis in analyses} >= {
        "identity",
        "mirror-file",
    }
    assert flags_by_family_index[0] == {
        "minimal-material",
        "queen-only-mate",
        "single-mate-move",
    }
    assert any(
        cast(int, analysis["spectator_count"]) > 0
        for analysis in analyses
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


def test_chess_metadata_and_tensor_construction_share_sample_addresses() -> None:
    generator = load_generator(_chess_benchmark_root)
    runtime = resolve_tensor_runtime("cpu")
    outcome_ids = tuple(outcome.id for outcome in generator.manifest.outcome_space.outcomes)

    metadata_sample_set = generator(
        seed=47,
        shape=3,
        complexity_request=ComplexityRequest(minimum=5.0, maximum=5.0),
    )
    tensor_sample_set = generator(
        seed=47,
        shape=3,
        include_metadata=False,
        runtime=runtime,
        outcome_ids=outcome_ids,
        complexity_request=ComplexityRequest(minimum=5.0, maximum=5.0),
    )
    fields, targets = tensor_sample_set.require_tensors()
    field_values = tensor_value_to_host(fields).tolist()
    target_values = tensor_value_to_host(targets).tolist()
    piece_plane = cast(Callable[[chess.Piece], int], _chess_benchmark_module()["_piece_plane"])

    for sample_index, sample in enumerate(metadata_sample_set.samples):
        assert sample.observable_state_id is not None
        board = chess.Board(sample.observable_state_id.removeprefix("fen:"))
        for square, piece in board.piece_map().items():
            plane_index = piece_plane(piece)
            rank_index = chess.square_rank(square)
            file_index = chess.square_file(square)
            assert field_values[sample_index][plane_index][rank_index][file_index] == 1.0
        assert target_values[sample_index][outcome_ids.index(sample.outcome_id)] == 1.0


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
    sample_indices = sample_indices_for_even_state_coverage(
        state_count=1,
        seed=401,
        sample_limit=_expected_preview_limit,
    )

    batch = generator(
        seed=401,
        shape=len(sample_indices),
        include_artifacts=True,
        complexity_request=ComplexityRequest(minimum=0.0, maximum=1.0),
        sample_indices=sample_indices,
    )

    assert batch.sample_count == 1
    sample = batch.samples[0].to_record()
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
    assert len(target_moves) == 1
    assert cast(float, target_moves[0]["target_probability"]) == 1.0
    assert "target_distribution" not in sample
    assert "available_outcome_ids" in sample
    cardinality_16_indices = sample_indices_for_even_state_coverage(
        state_count=16,
        seed=405,
        sample_limit=_expected_preview_limit,
    )
    cardinality_16_batch = generator(
        seed=405,
        shape=len(cardinality_16_indices),
        include_artifacts=True,
        complexity_request=ComplexityRequest(minimum=4.0, maximum=5.0),
        sample_indices=cardinality_16_indices,
    )
    cardinality_16_samples = [
        sample.to_record() for sample in cardinality_16_batch.samples
    ]
    cardinality_16_observable_ids = [
        cast(str, sample["observable_state_id"])
        for sample in cardinality_16_samples
    ]
    assert len(cardinality_16_observable_ids) == _expected_preview_limit
    assert len(cardinality_16_observable_ids) == len(
        frozenset(cardinality_16_observable_ids)
    )


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


def _sample_analysis(sample: Any) -> dict[str, object]:
    coordinates = cast(tuple[dict[str, object], ...], sample.latent_coordinates)
    analysis_coordinate = cast(dict[str, object], coordinates[5]["values"])
    return cast(dict[str, object], analysis_coordinate["analysis"])


def _chess_global_sample_index() -> Callable[..., int]:
    return cast(Callable[..., int], _chess_benchmark_module()["_global_sample_index"])


def _chess_family_capacity() -> int:
    return cast(Callable[[], int], _chess_benchmark_module()["_family_capacity"])()


def _chess_representatives_for_cardinality() -> Callable[[int], list[dict[str, object]]]:
    module = _chess_benchmark_module()
    global_sample_index = cast(Callable[..., int], module["_global_sample_index"])
    position_for_sample_index = cast(
        Callable[[int], Any],
        module["_position_for_sample_index"],
    )

    def representatives(cardinality: int) -> list[dict[str, object]]:
        return [
            dict(
                position_for_sample_index(
                    global_sample_index(
                        cardinality=cardinality,
                        local_index=index,
                    )
                ).representative_metadata()
            )
            for index in sample_indices_for_even_state_coverage(
                state_count=cardinality,
                seed=0,
                sample_limit=_expected_preview_limit,
            )
        ]

    return representatives


def _chess_benchmark_module() -> dict[str, object]:
    module_name = "test_chess_benchmark"
    entrypoint = _chess_benchmark_root / "benchmark.py"
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return vars(module)


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
