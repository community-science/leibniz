import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from benchmark_typing import (
    DigitsGenerator,
    load_digits_generator,
    sample_component_index,
    sample_height,
    sample_materialization_plan,
    sample_width,
)

from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import AxisAssignment, MaterializationPlan
from leibniz.observation_formation import FieldObservation
from leibniz.observation_generation import (
    GeneratedSampleSet,
    ObservationGenerationError,
    StateSpaceCandidate,
    StateSpaceMeasureRequest,
    StateSpaceMeasureValue,
)
from leibniz.tensor_runtime import TensorRuntimeError, resolve_tensor_runtime
from leibniz.timing import TimingCollector

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def _observation_payload(
    generator: DigitsGenerator,
    **kwargs: object,
) -> GeneratedSampleSet:
    sample_count = cast(int, kwargs.pop("sample_count"))
    sample_set = generator(
        shape=sample_count,
        include_fields=True,
        **cast(Any, kwargs),
    )
    return sample_set


def _formation_payload(
    generator: DigitsGenerator,
    **kwargs: object,
) -> GeneratedSampleSet:
    sample_count = cast(int, kwargs.pop("sample_count"))
    sample_set = generator(
        shape=sample_count,
        **cast(Any, kwargs),
    )
    return sample_set


def test_digits_generator_is_deterministic() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    left = _observation_payload(generator, sample_count=2, seed=101)
    right = _observation_payload(generator, sample_count=2, seed=101)

    assert left == right
    assert "component_count" not in left.to_record()
    first_sample = left.samples[0]
    first_width = sample_width(first_sample)
    first_height = sample_height(first_sample)
    first_plan = sample_materialization_plan(first_sample)
    assert first_sample.require_field().shape == (
        1,
        first_height,
        first_width,
    )
    assert math.isclose(
        first_sample.complexity,
        generator.distinguishable_state_complexity(
            width=first_width,
            height=first_height,
        ),
    )
    field_record = left.samples[0].field_record()
    assert sample_component_index(first_sample) == field_record.component_index
    assert first_sample.outcome_id == "digit-8"
    assert _coordinate(first_sample.latent_coordinates, role="content")["values"] == {
        "digit_index": field_record.component_index,
        "digit_variant_index": 0,
        "outcome_id": "digit-8",
    }
    assert first_sample.observable_state_id is None
    assert first_sample.target_distribution is None
    variation = _coordinate(first_sample.latent_coordinates, role="variation")
    assert variation["multiplicity"] == 1
    assert variation["name"] == "benchmarks.digits.sample.field-variation-transform"
    assert variation["degree_measure"] == {"kind": "vector-dimension", "count": 6.0}
    variation_values = cast(dict[str, object], variation["values"])
    assert variation_values["kind"] == "constructed-field-variation-transform-samples"
    assert variation_values["transform_count"] == 2
    assert cast(dict[str, object], variation_values["bounds"]) == (
        generator.formation.variation_transform.to_record()
    )
    coordinates = cast(list[dict[str, object]], variation_values["coordinates"])
    assert [coordinate["component_index"] for coordinate in coordinates] == [
        field_record.component_index
    ]
    assert len(coordinates) == 1
    constructed_parameters = cast(
        dict[str, float],
        coordinates[0]["constructed_affine_parameters"],
    )
    constructed_indices = cast(
        dict[str, int],
        coordinates[0]["constructed_affine_indices"],
    )
    assert set(constructed_parameters) == {
        "x_translation",
        "y_translation",
        "scale",
        "rotation",
        "x_shear",
    }
    assert set(constructed_indices) == {"preset"}
    assert _within_transform_bounds(
        coordinates[0],
        bounds=generator.formation.variation_transform.to_record(),
    )
    for coordinate in coordinates:
        report = generator.formation.component_discriminability_report(
            width=first_plan.resolution_assignment.require_axis("W"),
            height=first_plan.resolution_assignment.require_axis("H"),
            variation_coordinates=(coordinate,),
            minimum_pairwise_l1=generator.manifest.resolution_discriminability_margin(),
        )
        assert report.passed


def test_digits_generator_samples_formation_batch_without_fields() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    observation_batch = _observation_payload(generator, sample_count=2, seed=101)
    formation_batch = _formation_payload(generator, sample_count=2, seed=101)

    assert formation_batch.benchmark_id == observation_batch.benchmark_id
    assert "component_count" not in formation_batch.to_record()
    assert formation_batch.seed == observation_batch.seed
    assert [(sample.width, sample.height) for sample in formation_batch.samples] == [
        (sample.width, sample.height) for sample in observation_batch.samples
    ]
    assert [sample.component_index for sample in formation_batch.samples] == [
        sample.field_record().component_index for sample in observation_batch.samples
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
    generated_plan = sample_materialization_plan(formation_batch.samples[0])
    minimum_plan = MaterializationPlan.resolve(
        id=generated_plan.id,
        declaration=generator.materialization,
        seed=101,
    )
    assert minimum_plan.resolution_assignment.values == {"W": 1, "H": 1}
    assert generated_plan.resolution_assignment.values == (
        sample_materialization_plan(
            formation_batch.samples[0]
        ).resolution_assignment.values
    )
    assert generated_plan.resolution_assignment.require_axis("W") >= 1
    assert generated_plan.resolution_assignment.require_axis("H") >= 1


def test_digits_generator_accepts_lattice_resolution_assignment() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    batch = _formation_payload(generator,
        sample_count=2,
        seed=101,
        resolution_assignment=AxisAssignment(values={"W": 48, "H": 24}),
    )

    assert [
        sample_materialization_plan(sample).resolution_assignment.values
        for sample in batch.samples
    ] == [{"W": 48, "H": 24}, {"W": 48, "H": 24}]


def test_digits_generator_accepts_arbitrary_positive_resolution_assignment() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    batch = _formation_payload(generator,
        sample_count=2,
        seed=101,
        resolution_assignment=AxisAssignment(values={"W": 32, "H": 32}),
    )

    assert [
        sample_materialization_plan(sample).resolution_assignment.values
        for sample in batch.samples
    ] == [{"W": 32, "H": 32}, {"W": 32, "H": 32}]


def test_digits_generator_records_optional_timing() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    timing = TimingCollector()

    _observation_payload(generator,
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
    field_generation = cast(dict[str, object], phases["digits.field_generation"])
    assert record["kind"] == "test-timing"
    assert materialization["calls"] == 1
    assert materialization["sample_count"] == 2
    assert variation["sample_count"] == 2
    assert "counters" not in variation
    assert field_generation["sample_count"] == 2
    assert cast(float, field_generation["seconds"]) > 0


def test_digits_generator_samples_resolution_from_memory_bound() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    batch = _observation_payload(generator,
        sample_count=1,
        seed=202,
        component_indices=(1,),
    )
    sample = batch.samples[0]
    width = sample_width(sample)
    height = sample_height(sample)

    assert sample.require_field().shape == (1, height, width)
    assert width >= 1
    assert height >= 1
    assert math.isclose(
        sample.complexity,
        generator.distinguishable_state_complexity(width=width, height=height),
    )
    assert sample.outcome_id == "digit-1"


def test_digits_generator_counts_constructed_finite_state_space() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    scale_one = _formation_payload(generator, sample_count=3, seed=101)
    scale_one_other_seed = _formation_payload(generator,
        sample_count=3,
        seed=102,
    )

    assert {
        round(sample.complexity, 12)
        for sample in scale_one.samples
    } == {
        round(
            generator.distinguishable_state_complexity(
                width=sample_width(scale_one.samples[0]),
                height=sample_height(scale_one.samples[0]),
            ),
            12,
        )
    }
    assert {
        round(sample.complexity, 12)
        for sample in scale_one_other_seed.samples
    } == {
        round(
            generator.distinguishable_state_complexity(
                width=sample_width(scale_one_other_seed.samples[0]),
                height=sample_height(scale_one_other_seed.samples[0]),
            ),
            12,
        )
    }
    assert sample_width(scale_one.samples[0]) >= 1
    assert sample_height(scale_one.samples[0]) >= 1
    minimum = generator.constructed_state_space_complexity(
        affine_transform_count=1,
    )
    larger = generator.constructed_state_space_complexity(
        affine_transform_count=8,
    )
    assert math.isclose(minimum, math.log2(10))
    assert math.isclose(larger, math.log2(80))


def test_digits_generator_uses_runtime_memory_limit_as_canvas_cap() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    small = _formation_payload(generator,
        sample_count=3,
        seed=101,
        memory_limit_bytes=1_024_000,
    )
    large = _formation_payload(generator,
        sample_count=3,
        seed=101,
        memory_limit_bytes=100_000_000,
    )

    assert all((sample_width(sample), sample_height(sample)) == (13, 21)
        for sample in small.samples
    )
    assert all(sample_width(sample) >= 1 and sample_height(sample) >= 1
        for sample in large.samples
    )
    assert large.samples[0].complexity == small.samples[0].complexity


def test_digits_generator_accepts_state_space_measure_requests() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    requested_complexity = generator.minimum_state_space_measure().value

    state_space_request = StateSpaceMeasureRequest(
        minimum=requested_complexity,
        maximum=requested_complexity + 1.0,
    )
    batch = _observation_payload(
        generator,
        sample_count=2,
        seed=101,
        state_space_request=state_space_request,
    )

    assert batch.state_space_request is not None
    assert "component_count" not in batch.to_record()
    assert [sample.require_field().shape for sample in batch.samples] == [
        (1, 16, 16),
        (1, 16, 16),
    ]
    assert [sample.outcome_id for sample in batch.samples] == ["digit-8", "digit-7"]
    assert [sample.component_index for sample in batch.samples] == [8, 7]
    assert {
        sample.state_space_measure
        for sample in batch.samples
    } == {
        batch.samples[0].state_space_measure,
    }
    assert batch.samples[0].state_space_measure is not None
    assert batch.samples[0].state_space_measure.measure_id == state_space_request.measure_id
    assert math.isclose(batch.samples[0].state_space_measure.value, requested_complexity)


def test_digits_generator_materializes_target_state_space_band() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    target = generator.minimum_state_space_measure().value + 3.0

    state_space = generator.state_space_for_request(
        request=StateSpaceMeasureRequest(
            minimum=target,
            maximum=target + 1.0,
        )
    )

    assert state_space is not None
    assert state_space.cardinality == 80
    assert math.isclose(state_space.complexity, math.log2(80))
    assert state_space.resolution_assignment is not None
    assert state_space.metadata["affine_transform_count"] == 8
    assert state_space.metadata["digit_count"] == 10
    assert state_space.metadata["requested_state_count"] == 80
    assert state_space.metadata["realized_state_count"] == 80
    assert state_space.metadata["construction"] == (
        "symmetric-digits-over-finite-affine-product-grid"
    )
    assert state_space.metadata["affine_parameters"] == [
        "x_translation",
        "y_translation",
        "scale",
        "rotation",
        "x_shear",
    ]


def test_digits_generator_materializes_large_target_state_space_directly() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    state_space = generator.state_space_for_request(
        request=StateSpaceMeasureRequest(minimum=20.0, maximum=21.0)
    )

    assert state_space is not None
    assert state_space.cardinality == 1_048_580
    assert math.isclose(state_space.complexity, math.log2(1_048_580))
    assert state_space.resolution_assignment is not None
    assert state_space.resolution_assignment.values == {"W": 1452, "H": 1452}
    assert state_space.metadata["affine_grid"] == {
        "x_translation": 109,
        "y_translation": 74,
        "scale": 13,
        "rotation": 1,
        "x_shear": 1,
    }


def test_state_space_candidates_can_declare_exact_integer_cardinality() -> None:
    candidate = StateSpaceCandidate(
        request=StateSpaceMeasureRequest(
            minimum=math.log2(17),
            maximum=math.log2(17),
        ),
        cardinality=17,
    )

    assert candidate.complexity == math.log2(17)
    assert candidate.to_record()["cardinality"] == 17


def test_digits_generator_returns_empty_set_below_canonical_digit_space() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    state_space_request = StateSpaceMeasureRequest(
        minimum=1.0,
        maximum=1.0,
    )
    batch = generator(
        shape=3,
        seed=101,
        state_space_request=state_space_request,
    )

    assert batch.shape == (0,)
    assert len(batch.samples) == 0
    assert batch.state_space_request is not None
    assert batch.state_space_request.measure_id == state_space_request.measure_id


def test_state_space_measure_ids_are_core_contract() -> None:
    assert (
        str(
            capture_generation_error(
                lambda: StateSpaceMeasureRequest(
                    minimum=0.0,
                    maximum=0.0,
                )
            )
        )
        == "state-space measure minimum must be at least 1"
    )
    assert (
        str(
            capture_generation_error(
                lambda: StateSpaceMeasureValue(
                    value=0.0,
                )
            )
        )
        == "state-space measure value must be at least 1"
    )
    assert (
        str(
            capture_generation_error(
                lambda: StateSpaceMeasureRequest(
                    measure_id="benchmarks.chess.valid-move-count",
                    minimum=1.0,
                    maximum=1.0,
                )
            )
        )
        == "state-space measure id is not a core measure"
    )
    assert (
        str(
            capture_generation_error(
                lambda: StateSpaceMeasureValue(
                    measure_id="benchmarks.chess.valid-move-count",
                    value=1.0,
                )
            )
        )
        == "state-space measure id is not a core measure"
    )


def test_digits_generator_keeps_minimum_canvas_when_memory_cap_is_tiny() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    batch = _formation_payload(generator,
        sample_count=8,
        seed=101,
        memory_limit_bytes=1,
    )

    assert [(sample.width, sample.height) for sample in batch.samples] == [(1, 1)] * 8


def test_digits_generator_applies_recorded_variation_coordinates() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    sample = _observation_payload(generator,
        sample_count=1,
        seed=909,
        component_indices=(1,),
    ).samples[0]
    variation = _coordinate(sample.latent_coordinates, role="variation")
    variation_values = cast(dict[str, object], variation["values"])
    plan = sample_materialization_plan(sample)
    component_index = sample_component_index(sample)
    direct = generator.formation.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.direct@0.1.0"),
        plan=plan,
        component_index=component_index,
        variation_coordinates=cast(
            list[Mapping[str, object]],
            variation_values["coordinates"],
        ),
    )
    untransformed = generator.formation.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.untransformed@0.1.0"),
        plan=plan,
        component_index=component_index,
    )

    field_record = sample.field_record()
    assert field_record.field == direct.field
    assert field_record.field != untransformed.field
    assert all(0.0 <= value <= 1.0 for value in sample.require_field().values)


def test_digits_tensor_fields_match_recorded_field_samples() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    runtime = resolve_tensor_runtime("cpu")
    outcome_space = generator.manifest.resolve_outcome_space()
    batch = generator(
        shape=4,
        seed=909,
        include_fields=True,
        runtime=runtime,
        outcome_ids=tuple(outcome.id for outcome in outcome_space.outcomes),
    )

    fields = batch.require_tensors()[0].detach().cpu()
    for index, sample in enumerate(batch.samples):
        assert tuple(fields[index].shape) == sample.require_field().shape
        assert fields[index].flatten().tolist() == list(sample.require_field().values)


def test_digits_cuda_tensor_fields_match_cpu_reference() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    cpu_runtime = resolve_tensor_runtime("cpu")
    try:
        cuda_runtime = resolve_tensor_runtime("cuda")
    except TensorRuntimeError as error:
        pytest.skip(str(error))
    outcome_space = generator.manifest.resolve_outcome_space()
    outcome_ids = tuple(outcome.id for outcome in outcome_space.outcomes)
    request = StateSpaceMeasureRequest(
        minimum=generator.minimum_state_space_measure().value + 6.0,
        maximum=generator.minimum_state_space_measure().value + 7.0,
    )

    cpu_fields = generator(
        shape=16,
        seed=444,
        include_fields=False,
        runtime=cpu_runtime,
        outcome_ids=outcome_ids,
        state_space_request=request,
    ).require_tensors()[0]
    cuda_fields = generator(
        shape=16,
        seed=444,
        include_fields=False,
        runtime=cuda_runtime,
        outcome_ids=outcome_ids,
        state_space_request=request,
    ).require_tensors()[0]

    assert cuda_runtime.torch.equal(cpu_fields, cuda_fields.detach().cpu())


def test_digits_tensor_generation_rejects_unmatched_state_space_requests() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    runtime = resolve_tensor_runtime("cpu")
    outcome_space = generator.manifest.resolve_outcome_space()
    request = StateSpaceMeasureRequest(minimum=1.0, maximum=1.0)

    with pytest.raises(
        ObservationGenerationError,
        match="tensor generation state-space request matched no candidate",
    ):
        generator(
            shape=1,
            seed=910,
            runtime=runtime,
            outcome_ids=tuple(outcome.id for outcome in outcome_space.outcomes),
            state_space_request=request,
        )


def test_digits_generator_rejects_edge_clipped_affine_samples() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    batch = _observation_payload(generator, sample_count=3, seed=1)

    for sample in batch.samples:
        assert not _field_has_positive_edge(sample.require_field())


def test_generated_observation_records_can_include_fields() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    batch = _observation_payload(generator, sample_count=1, seed=303)

    compact = batch.to_record()
    expanded = batch.to_record(include_fields=True)
    compact_sample = cast(dict[str, object], cast(list[object], compact["samples"])[0])
    expanded_sample = cast(dict[str, object], cast(list[object], expanded["samples"])[0])

    assert "field" not in compact_sample
    assert "field" in expanded_sample


def test_digits_benchmark_uses_single_tensor_render_path() -> None:
    source = (_digits_benchmark_root / "benchmark.py").read_text(encoding="utf-8")

    assert "sample_variation_transform_coordinates" not in source
    assert ".formation.form_observation(" not in source
    assert "state_tensor_cache" not in source
    assert "full_state_tensor" not in source
    assert "sampled_state_tensor" not in source
    assert "field_tensor_gather" not in source
    assert "_FormationTensorCache" not in source


def test_digits_console_preview_png_encoding_is_deterministic() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    left = generator.console_preview_batches(atom_count=10)
    right = generator.console_preview_batches(atom_count=10)
    sample = cast(dict[str, object], cast(list[object], left[2]["samples"])[0])
    data_url = cast(str, sample["image_data_url"])

    assert left == right
    assert [
        (batch["label"], batch["sample_count"])
        for batch in left
    ] == [
        ("[3, 4]", 10),
        ("[5, 6]", 50),
        ("[8, 9]", 50),
    ]
    assert data_url.startswith("data:image/png;base64,")


def test_generator_rejects_invalid_requests() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    assert (
        str(
            capture_generation_error(
                lambda: _observation_payload(generator, sample_count=0, seed=1)
            )
        )
        == "sample shape axes must be positive integers"
    )
    assert (
        str(
            capture_generation_error(
                lambda: _observation_payload(generator,
                    sample_count=2,
                    seed=1,
                    component_indices=(1,),
                )
            )
        )
        == "component_indices length must match sample_count"
    )
    assert (
        str(
            capture_generation_error(
                lambda: _observation_payload(generator,
                    sample_count=1,
                    seed=1,
                    variation_extent=1.1,
                )
            )
        )
        == "variation_extent must be between 0 and 1"
    )


def test_digits_variation_extent_zero_samples_canonical_affine() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    canonical = _observation_payload(generator,
        sample_count=1,
        seed=101,
        variation_extent=0.0,
    )
    full = _observation_payload(generator,
        sample_count=1,
        seed=101,
        variation_extent=1.0,
    )

    assert math.isclose(canonical.samples[0].complexity, math.log2(10))
    assert full.samples[0].complexity > canonical.samples[0].complexity
    variation = _coordinate(canonical.samples[0].latent_coordinates, role="variation")
    values = cast(dict[str, object], variation["values"])
    coordinates = cast(list[dict[str, object]], values["coordinates"])
    assert _spatial_affine_matrix(coordinates[0]) == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


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


def _spatial_affine_matrix(coordinate: Mapping[str, object]) -> list[list[float]]:
    spatial = cast(Mapping[str, object], coordinate["spatial_affine"])
    matrix = cast(list[list[int | float]], spatial["matrix"])
    return [
        [float(value) for value in row]
        for row in matrix
    ]


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


def _field_has_positive_edge(field: FieldObservation) -> bool:
    channels, height, width = field.shape
    for channel in range(channels):
        channel_offset = channel * width * height
        for x_index in range(width):
            if field.values[channel_offset + x_index] > 0.0:
                return True
            if field.values[channel_offset + (height - 1) * width + x_index] > 0.0:
                return True
        for y_index in range(height):
            if field.values[channel_offset + y_index * width] > 0.0:
                return True
            if field.values[channel_offset + y_index * width + width - 1] > 0.0:
                return True
    return False


def capture_generation_error(call: Callable[[], object]) -> ObservationGenerationError:
    with pytest.raises(ObservationGenerationError) as error:
        call()
    return error.value
