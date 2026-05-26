from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest

from leibniz.observation_generation import (
    ObservationGenerationError,
    field_to_png_bytes,
    field_to_png_data_url,
    load_observation_generator,
)

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
    nuisance = _coordinate(left.samples[0].latent_coordinates, role="nuisance")
    assert nuisance["multiplicity"] == 3
    assert len(cast(list[float], nuisance["values"])) == 6


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


def test_generated_observation_records_can_include_fields() -> None:
    generator = load_observation_generator(_digits_benchmark_root)
    batch = generator.sample_batch(scale=1, sample_count=1, seed=303)

    compact = batch.to_record()
    expanded = batch.to_record(include_fields=True)
    compact_sample = cast(dict[str, object], cast(list[object], compact["samples"])[0])
    expanded_sample = cast(dict[str, object], cast(list[object], expanded["samples"])[0])

    assert "field" not in compact_sample
    assert "field" in expanded_sample


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


def capture_generation_error(call: Callable[[], object]) -> ObservationGenerationError:
    with pytest.raises(ObservationGenerationError) as error:
        call()
    return error.value
