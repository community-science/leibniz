from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest

from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import AxisAssignment, MaterializationPlan
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

    left = generator.sample_batch(scale=3, sample_count=2, seed=101)
    right = generator.sample_batch(scale=3, sample_count=2, seed=101)

    assert left == right
    assert left.scale == 3
    assert left.samples[0].field.shape == (1, 96, 96)
    assert left.samples[0].complexity == 3.0
    assert len(left.samples[0].observation.component_sequence) == 3
    assert left.samples[0].outcome_id.startswith("digit-")
    assert _coordinate(left.samples[0].latent_coordinates, role="content")["values"] == list(
        left.samples[0].observation.component_sequence
    )
    variation = _coordinate(left.samples[0].latent_coordinates, role="variation")
    assert variation["multiplicity"] == 3
    assert variation["name"] == "benchmarks.digits.sample.field-variation-transform"
    assert variation["degree_measure"] == {"kind": "vector-dimension", "count": 7.0}
    variation_values = cast(dict[str, object], variation["values"])
    assert variation_values["kind"] == "field-variation-transform-samples"
    assert cast(dict[str, object], variation_values["bounds"]) == (
        generator.formation.variation_transform.to_record()
    )
    coordinates = cast(list[dict[str, object]], variation_values["coordinates"])
    assert [coordinate["slot_index"] for coordinate in coordinates] == [0, 1, 2]
    assert len(coordinates) == 3
    assert _within_transform_bounds(
        coordinates[0],
        bounds=generator.formation.variation_transform.to_record(),
    )


def test_digits_observation_generator_samples_formation_batch_without_fields() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    observation_batch = generator.sample_batch(scale=3, sample_count=2, seed=101)
    formation_batch = generator.sample_formation_batch(scale=3, sample_count=2, seed=101)

    assert formation_batch.benchmark_id == observation_batch.benchmark_id
    assert formation_batch.scale == observation_batch.scale
    assert formation_batch.seed == observation_batch.seed
    assert [sample.resolution for sample in formation_batch.samples] == [96, 96]
    assert [sample.component_sequence for sample in formation_batch.samples] == [
        sample.observation.component_sequence
        for sample in observation_batch.samples
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
    direct_plan = MaterializationPlan.resolve(
        id=formation_batch.samples[0].materialization_plan.id,
        declaration=generator.materialization,
        scale_assignment=AxisAssignment(values={"L": 3}),
        complexity_assignment=AxisAssignment(values={"C": 3}),
        seed=101,
    )
    assert formation_batch.samples[0].materialization_plan == direct_plan
    assert formation_batch.samples[0].variation_coordinates[0] == (
        sample_variation_transform_coordinates(
            transform=generator.formation.variation_transform,
            seed=101,
            sample_index=0,
            slot_index=0,
        )
    )


def test_digits_observation_generator_records_optional_timing() -> None:
    generator = load_observation_generator(_digits_benchmark_root)
    timing = TimingCollector()

    generator.sample_batch(
        scale=2,
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
    assert observation["sample_count"] == 2
    assert cast(float, observation["seconds"]) > 0


def test_digits_observation_generator_scales_resolution_and_complexity() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    batch = generator.sample_batch(
        scale=4,
        sample_count=1,
        seed=202,
        component_sequences=((1, 2, 3, 4),),
    )
    sample = batch.samples[0]

    assert sample.field.shape == (1, 128, 128)
    assert sample.materialization_plan.scale_assignment.require_axis("L") == 4
    assert sample.materialization_plan.resolution_assignment.require_axis("N") == 128
    assert sample.materialization_plan.complexity_assignment.require_axis("C") == 4
    assert sample.complexity == 4.0
    assert sample.outcome_id == "digit-1-2-3-4"


def test_digits_observation_generator_applies_recorded_variation_coordinates() -> None:
    generator = load_observation_generator(_digits_benchmark_root)
    sample = generator.sample_batch(
        scale=2,
        sample_count=1,
        seed=909,
        component_sequences=((1, 2),),
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
    batch = generator.sample_batch(scale=1, sample_count=1, seed=303)

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
        slot_index=1,
    )
    right = sample_variation_transform_coordinates(
        transform=transform,
        seed=707,
        sample_index=2,
        slot_index=1,
    )
    other = sample_variation_transform_coordinates(
        transform=transform,
        seed=707,
        sample_index=3,
        slot_index=1,
    )
    other_slot = sample_variation_transform_coordinates(
        transform=transform,
        seed=707,
        sample_index=2,
        slot_index=2,
    )

    assert left == right
    assert left != other
    assert left != other_slot
    assert _within_transform_bounds(left, bounds=transform.to_record())


def test_field_png_encoding_is_deterministic() -> None:
    generator = load_observation_generator(_digits_benchmark_root)
    field = generator.sample_batch(scale=1, sample_count=1, seed=404).samples[0].field

    left = field_to_png_bytes(field)
    right = field_to_png_bytes(field)
    data_url = field_to_png_data_url(field)

    assert left == right
    assert left.startswith(b"\x89PNG\r\n\x1a\n")
    assert data_url.startswith("data:image/png;base64,")


def test_observation_generator_rejects_invalid_requests() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    assert str(
        capture_generation_error(lambda: generator.sample_batch(scale=0, sample_count=1, seed=1))
    ) == "scale must be a positive integer"
    assert str(
        capture_generation_error(lambda: generator.sample_batch(scale=1, sample_count=0, seed=1))
    ) == "sample_count must be a positive integer"
    assert str(
        capture_generation_error(
            lambda: generator.sample_batch(
                scale=1,
                sample_count=2,
                seed=1,
                component_sequences=((1,),),
            )
        )
    ) == "component_sequences length must match sample_count"


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
    translation = cast(list[float], spatial["translation"])
    translation_bounds = cast(list[list[float]], spatial_bounds["translation"])
    scale = cast(list[float], spatial["scale"])
    scale_bounds = cast(list[list[float]], spatial_bounds["scale"])
    rotations = cast(list[float], spatial["rotation_degrees"])
    rotation_bounds = cast(list[float], spatial_bounds["rotation_degrees"])
    shears = cast(list[float], spatial["shear_degrees"])
    shear_bounds = cast(list[float], spatial_bounds["shear_degrees"])
    value_scale = cast(Mapping[str, object], coordinate["value_scale"])
    value_scale_bounds = cast(Mapping[str, object], bounds["value_scale"])
    return (
        all(
            lower <= value <= upper
            for value, (lower, upper) in zip(translation, translation_bounds, strict=True)
        )
        and all(
            lower <= value <= upper
            for value, (lower, upper) in zip(scale, scale_bounds, strict=True)
        )
        and all(
            -bound <= value <= bound
            for value, bound in zip(rotations, rotation_bounds, strict=True)
        )
        and all(
            -bound <= value <= bound
            for value, bound in zip(shears, shear_bounds, strict=True)
        )
        and cast(list[float], value_scale_bounds["scale"])[0]
        <= cast(float, value_scale["scale"])
        <= cast(list[float], value_scale_bounds["scale"])[1]
    )


def capture_generation_error(call: Callable[[], object]) -> ObservationGenerationError:
    with pytest.raises(ObservationGenerationError) as error:
        call()
    return error.value
