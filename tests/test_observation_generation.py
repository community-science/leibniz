import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest

from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import MaterializationPlan
from leibniz.observation_generation import (
    ObservationGenerationError,
    field_to_png_bytes,
    field_to_png_data_url,
    load_observation_generator,
    sample_variation_transform_coordinates,
)
from leibniz.timing import TimingCollector

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def test_digits_observation_generator_is_deterministic() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    left = generator.sample_batch(component_count=1, sample_count=2, seed=101)
    right = generator.sample_batch(component_count=1, sample_count=2, seed=101)

    assert left == right
    assert left.component_count == 1
    assert left.samples[0].field.shape == (1, 42, 118)
    assert math.isclose(left.samples[0].complexity, math.log2(49286227620))
    assert len(left.samples[0].observation.component_sequence) == 1
    assert left.samples[0].outcome_id == "digit-8"
    assert _coordinate(left.samples[0].latent_coordinates, role="content")["values"] == list(
        left.samples[0].observation.component_sequence
    )
    variation = _coordinate(left.samples[0].latent_coordinates, role="variation")
    assert variation["multiplicity"] == 1
    assert variation["name"] == "benchmarks.digits.sample.field-variation-transform"
    assert variation["degree_measure"] == {"kind": "vector-dimension", "count": 6.0}
    variation_values = cast(dict[str, object], variation["values"])
    assert variation_values["kind"] == "field-variation-transform-samples"
    assert cast(dict[str, object], variation_values["bounds"]) == (
        generator.formation.variation_transform.to_record()
    )
    coordinates = cast(list[dict[str, object]], variation_values["coordinates"])
    assert [coordinate["sequence_index"] for coordinate in coordinates] == [0]
    assert len(coordinates) == 1
    assert _within_transform_bounds(
        coordinates[0],
        bounds=generator.formation.variation_transform.to_record(),
    )
    for coordinate in coordinates:
        sequence_index = cast(int, coordinate["sequence_index"])
        report = generator.formation.component_discriminability_report(
            width=left.samples[0].materialization_plan.resolution_assignment.require_axis("W"),
            height=left.samples[0].materialization_plan.resolution_assignment.require_axis("H"),
            sequence_length=len(left.samples[0].observation.component_sequence),
            sequence_index=sequence_index,
            variation_coordinates=(coordinate,),
            minimum_pairwise_l1=generator.benchmark_manifest.resolution_discriminability_margin(),
        )
        assert report.passed


def test_digits_observation_generator_samples_formation_batch_without_fields() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    observation_batch = generator.sample_batch(component_count=1, sample_count=2, seed=101)
    formation_batch = generator.sample_formation_batch(component_count=1, sample_count=2, seed=101)

    assert formation_batch.benchmark_id == observation_batch.benchmark_id
    assert formation_batch.component_count == observation_batch.component_count
    assert formation_batch.seed == observation_batch.seed
    assert [(sample.width, sample.height) for sample in formation_batch.samples] == [
        (118, 42),
        (118, 42),
    ]
    assert [sample.component_sequence for sample in formation_batch.samples] == [
        sample.observation.component_sequence for sample in observation_batch.samples
    ]
    assert [sample.outcome_id for sample in formation_batch.samples] == [
        sample.outcome_id for sample in observation_batch.samples
    ]
    assert [sample.variation_coordinates for sample in formation_batch.samples] == [
        tuple(
            cast(
                list[Mapping[str, object]],
                cast(
                    Mapping[str, object],
                    _coordinate(sample.latent_coordinates, role="variation")["values"],
                )["coordinates"],
            )
        )
        for sample in observation_batch.samples
    ]
    minimum_plan = MaterializationPlan.resolve(
        id=formation_batch.samples[0].materialization_plan.id,
        declaration=generator.materialization,
        seed=101,
    )
    generated_plan = formation_batch.samples[0].materialization_plan
    assert minimum_plan.resolution_assignment.values == {"W": 1, "H": 1}
    assert generated_plan.resolution_assignment.values == {"W": 118, "H": 42}


def test_digits_observation_generator_records_optional_timing() -> None:
    generator = load_observation_generator(_digits_benchmark_root)
    timing = TimingCollector()

    generator.sample_batch(
        component_count=1,
        sample_count=2,
        seed=303,
        timing=timing,
        timing_prefix="digits.",
    )

    record = timing.to_record(kind="test-timing")
    phases = cast(dict[str, object], record["phases"])
    materialization = cast(
        dict[str, object],
        phases["digits.formation_batch.materialization_plan"],
    )
    variation = cast(
        dict[str, object],
        phases["digits.formation_batch.variation_coordinates"],
    )
    observation = cast(dict[str, object], phases["digits.materialized_observation"])
    assert record["kind"] == "test-timing"
    assert materialization["calls"] == 1
    assert materialization["sample_count"] == 2
    assert variation["sample_count"] == 2
    variation_counters = cast(dict[str, float], variation["counters"])
    assert variation_counters["candidate_count"] >= variation_counters["accepted_count"]
    assert variation_counters["accepted_count"] == 2.0
    assert variation_counters["candidate_count"] == variation_counters["accepted_count"]
    assert "fast_reject_count" not in variation_counters
    assert observation["sample_count"] == 2
    assert cast(float, observation["seconds"]) > 0


def test_digits_observation_generator_samples_resolution_from_memory_bound() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    batch = generator.sample_batch(
        component_count=1,
        sample_count=1,
        seed=202,
        component_sequences=((1,),),
    )
    sample = batch.samples[0]

    assert sample.field.shape == (1, 82, 42)
    assert sample.materialization_plan.resolution_assignment.values == {"W": 42, "H": 82}
    assert math.isclose(sample.complexity, math.log2(17659483380))
    assert sample.outcome_id == "digit-1"


def test_digits_observation_generator_counts_discretized_nuisance_state_space() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    scale_one = generator.sample_formation_batch(component_count=1, sample_count=3, seed=101)
    scale_one_other_seed = generator.sample_formation_batch(
        component_count=1,
        sample_count=3,
        seed=102,
    )

    assert {round(sample.complexity, 12) for sample in scale_one.samples} == {
        round(math.log2(15103350800), 12)
    }
    assert {round(sample.complexity, 12) for sample in scale_one_other_seed.samples} == {
        round(math.log2(68520684960), 12)
    }
    assert [(sample.width, sample.height) for sample in scale_one.samples] == [
        (115, 28),
        (115, 28),
        (115, 28),
    ]
    assert [(sample.width, sample.height) for sample in scale_one_other_seed.samples] == [
        (75, 72),
        (75, 72),
        (75, 72),
    ]
    assert 20 <= scale_one.samples[0].width <= 130
    assert 20 <= scale_one.samples[0].height <= 130
    assert (scale_one.samples[0].width, scale_one.samples[0].height) != (20, 20)


def test_digits_observation_generator_uses_runtime_memory_limit_as_canvas_cap() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    small = generator.sample_formation_batch(
        component_count=1,
        sample_count=3,
        seed=101,
        memory_limit_bytes=1_024_000,
    )
    large = generator.sample_formation_batch(
        component_count=1,
        sample_count=3,
        seed=101,
        memory_limit_bytes=100_000_000,
    )

    assert [(sample.width, sample.height) for sample in small.samples] == [
        (21, 22),
        (21, 22),
        (21, 22),
    ]
    assert [(sample.width, sample.height) for sample in large.samples] == [
        (193, 214),
        (193, 214),
        (193, 214),
    ]
    assert large.samples[0].complexity > small.samples[0].complexity


def test_digits_observation_generator_keeps_minimum_canvas_when_memory_cap_is_tiny() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    batch = generator.sample_formation_batch(
        component_count=1,
        sample_count=8,
        seed=101,
        memory_limit_bytes=1,
    )

    assert [(sample.width, sample.height) for sample in batch.samples] == [(20, 20)] * 8


def test_digits_observation_generator_applies_recorded_variation_coordinates() -> None:
    generator = load_observation_generator(_digits_benchmark_root)
    sample = generator.sample_batch(
        component_count=1,
        sample_count=1,
        seed=909,
        component_sequences=((1,),),
    ).samples[0]
    variation = _coordinate(sample.latent_coordinates, role="variation")
    variation_values = cast(dict[str, object], variation["values"])
    direct = generator.formation.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.direct@0.1.0"),
        plan=sample.materialization_plan,
        component_sequence=sample.observation.component_sequence,
        variation_coordinates=cast(
            list[Mapping[str, object]],
            variation_values["coordinates"],
        ),
    )
    untransformed = generator.formation.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.untransformed@0.1.0"),
        plan=sample.materialization_plan,
        component_sequence=sample.observation.component_sequence,
    )

    assert sample.observation.field == direct.field
    assert sample.observation.field != untransformed.field
    assert all(0.0 <= value <= 1.0 for value in sample.field.values)


def test_generated_observation_records_can_include_fields() -> None:
    generator = load_observation_generator(_digits_benchmark_root)
    batch = generator.sample_batch(component_count=1, sample_count=1, seed=303)

    compact = batch.to_record()
    expanded = batch.to_record(include_fields=True)
    compact_sample = cast(dict[str, object], cast(list[object], compact["samples"])[0])
    expanded_sample = cast(dict[str, object], cast(list[object], expanded["samples"])[0])

    assert "field" not in compact_sample
    assert "field" in expanded_sample


def test_variation_transform_sampling_is_deterministic_and_declaration_driven() -> None:
    generator = load_observation_generator(_digits_benchmark_root)
    transform = generator.formation.variation_transform

    left = sample_variation_transform_coordinates(
        transform=transform,
        seed=707,
        sample_index=2,
        sequence_index=1,
    )
    right = sample_variation_transform_coordinates(
        transform=transform,
        seed=707,
        sample_index=2,
        sequence_index=1,
    )
    other = sample_variation_transform_coordinates(
        transform=transform,
        seed=707,
        sample_index=3,
        sequence_index=1,
    )
    other_sequence_element = sample_variation_transform_coordinates(
        transform=transform,
        seed=707,
        sample_index=2,
        sequence_index=2,
    )

    assert left == right
    assert left != other
    assert left != other_sequence_element
    assert _within_transform_bounds(left, bounds=transform.to_record())


def test_field_png_encoding_is_deterministic() -> None:
    generator = load_observation_generator(_digits_benchmark_root)
    field = generator.sample_batch(component_count=1, sample_count=1, seed=404).samples[0].field

    left = field_to_png_bytes(field)
    right = field_to_png_bytes(field)
    data_url = field_to_png_data_url(field)

    assert left == right
    assert left.startswith(b"\x89PNG\r\n\x1a\n")
    assert data_url.startswith("data:image/png;base64,")


def test_observation_generator_rejects_invalid_requests() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    assert (
        str(
            capture_generation_error(
                lambda: generator.sample_batch(component_count=0, sample_count=1, seed=1)
            )
        )
        == "component_count must be a positive integer"
    )
    assert (
        str(
            capture_generation_error(
                lambda: generator.sample_batch(component_count=1, sample_count=0, seed=1)
            )
        )
        == "sample_count must be a positive integer"
    )
    assert (
        str(
            capture_generation_error(
                lambda: generator.sample_batch(
                    component_count=1,
                    sample_count=2,
                    seed=1,
                    component_sequences=((1,),),
                )
            )
        )
        == "component_sequences length must match sample_count"
    )


def _coordinate(
    coordinates: tuple[Mapping[str, object], ...],
    *,
    role: str,
) -> dict[str, object]:
    for coordinate in coordinates:
        assert isinstance(coordinate, dict)
        if coordinate["role"] == role:
            return coordinate
    raise AssertionError(f"missing coordinate role {role}")


def _within_transform_bounds(
    coordinate: Mapping[str, object],
    *,
    bounds: Mapping[str, object],
) -> bool:
    spatial = cast(Mapping[str, object], coordinate["spatial_affine"])
    spatial_bounds = cast(Mapping[str, object], bounds["spatial_affine"])
    matrix = cast(list[list[float]], spatial["matrix"])
    matrix_bounds = cast(list[list[list[float]]], spatial_bounds["matrix"])
    return all(
        lower <= value <= upper
        for row, bound_row in zip(matrix, matrix_bounds, strict=True)
        for value, (lower, upper) in zip(row, bound_row, strict=True)
    )


def capture_generation_error(call: Callable[[], object]) -> ObservationGenerationError:
    with pytest.raises(ObservationGenerationError) as error:
        call()
    return error.value
