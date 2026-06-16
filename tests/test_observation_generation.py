import importlib.util
import math
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import replace
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

from leibniz.cost_metrology import CostMeasurement
from leibniz.documents import canonical_document_bytes, load_object_document
from leibniz.materialization import AxisAssignment, MaterializationPlan
from leibniz.observation_formation import FieldObservation
from leibniz.observation_generation import (
    GeneratedSample,
    GeneratedSampleSet,
    ObservationGenerationError,
    StateSpaceVolumeRequest,
    StateSpaceVolumeValue,
    sample_indices_for_even_state_coverage,
)
from leibniz.state_space import state_space_region_from_record
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


def test_generated_sample_records_available_outcome_ids() -> None:
    sample = GeneratedSample(
        index=0,
        outcome_id="yes",
        available_outcome_ids=("no", "yes"),
    )

    assert sample.to_record()["available_outcome_ids"] == ["no", "yes"]


def test_generated_sample_rejects_invalid_available_outcome_ids() -> None:
    with pytest.raises(ObservationGenerationError, match="available_outcome_ids must be unique"):
        GeneratedSample(
            index=0,
            outcome_id="yes",
            available_outcome_ids=("yes", "yes"),
        )

    with pytest.raises(ObservationGenerationError, match="available_outcome_ids must be nonempty"):
        GeneratedSample(
            index=0,
            outcome_id="yes",
            available_outcome_ids=("",),
        )


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
        left.log2_volume,
        generator.distinguishable_state_log2_volume(
            width=first_width,
            height=first_height,
        ),
    )
    assert sample_component_index(first_sample) == first_sample.component_index
    assert first_sample.outcome_id == "inverse-observation"
    assert _coordinate(first_sample.latent_coordinates, role="content")["values"] == {
            "digit_index": first_sample.component_index,
            "digit_variant_index": 0,
            "outcome_id": first_sample.outcome_id,
        }
    assert first_sample.observable_state_id is None
    assert first_sample.available_outcome_ids == ()
    assert first_sample.target_distribution is None
    variation = _coordinate(first_sample.latent_coordinates, role="variation")
    assert variation["multiplicity"] == 1
    assert variation["name"] == "benchmarks.digits.sample.field-variation-transform"
    assert variation["degree_measure"] == {"kind": "vector-dimension", "count": 3.0}
    variation_values = cast(dict[str, object], variation["values"])
    assert variation_values["kind"] == "constructed-field-variation-transform-samples"
    assert isinstance(variation_values["transform_ordinal"], int)
    assert variation_values["transform_ordinal"] >= 0
    volume_class = cast(dict[str, object], variation_values["volume_class"])
    assert volume_class["kind"] == "digits-realized-setup-window"
    assert volume_class["transform_axes"] == [
        "x_translation",
        "y_translation",
        "scale",
    ]
    assert volume_class["canvas_side"] == first_width
    assert cast(dict[str, object], variation_values["bounds"]) == (
        generator.formation.variation_transform.to_record()
    )
    coordinates = cast(list[dict[str, object]], variation_values["coordinates"])
    assert [coordinate["component_index"] for coordinate in coordinates] == [
        first_sample.component_index
    ]
    assert len(coordinates) == 1
    assert coordinates[0]["transform_ordinal"] == variation_values["transform_ordinal"]
    transform_cell = cast(dict[str, int], coordinates[0]["transform_cell"])
    assert set(transform_cell) == {
        "x_translation_step",
        "y_translation_step",
        "scale_level",
    }
    assert set(cast(dict[str, float], coordinates[0]["normalized_transform"])) == {
        "x_translation",
        "y_translation",
        "scale",
    }
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
        sample.component_index for sample in observation_batch.samples
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
    )
    sample = batch.samples[0]
    width = sample_width(sample)
    height = sample_height(sample)

    assert sample.require_field().shape == (1, height, width)
    assert width >= 1
    assert height >= 1
    assert math.isclose(
        batch.log2_volume,
        generator.distinguishable_state_log2_volume(width=width, height=height),
    )
    assert sample.outcome_id == "inverse-observation"


def test_digits_generator_counts_setup_window_volume_class() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    generator_impl = cast(Any, generator)

    preview = _formation_payload(generator, sample_count=3, seed=101)
    preview_other_seed = _formation_payload(generator,
        sample_count=3,
        seed=102,
    )

    assert round(preview.log2_volume, 12) == round(
        generator.distinguishable_state_log2_volume(
            width=sample_width(preview.samples[0]),
            height=sample_height(preview.samples[0]),
        ),
        12,
    )
    assert round(preview_other_seed.log2_volume, 12) == round(
        generator.distinguishable_state_log2_volume(
            width=sample_width(preview_other_seed.samples[0]),
            height=sample_height(preview_other_seed.samples[0]),
        ),
        12,
    )
    assert sample_width(preview.samples[0]) >= 28
    assert sample_height(preview.samples[0]) >= 28
    assert generator.minimum_log2_volume().value == 0.0
    window = generator_impl._volume_class_for_request(
        request=StateSpaceVolumeRequest(minimum=5.0, maximum=6.0)
    )
    assert window is not None
    assert window.minimum_address == 31
    assert window.cardinality == 32
    assert math.isclose(window.log2_volume, 5.0)


def test_digits_generator_accepts_memory_limit_without_changing_canvas() -> None:
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

    assert [(sample_width(sample), sample_height(sample)) for sample in small.samples] == (
        [(36, 36)] * 3
    )
    assert [(sample_width(sample), sample_height(sample)) for sample in large.samples] == (
        [(36, 36)] * 3
    )
    assert large.log2_volume == small.log2_volume


def test_digits_generator_accepts_volume_value_requests() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    requested_log2_volume = generator.minimum_log2_volume().value

    volume_request = StateSpaceVolumeRequest(
        minimum=requested_log2_volume,
        maximum=requested_log2_volume + 1.0,
    )
    batch = _observation_payload(
        generator,
        sample_count=2,
        seed=101,
        volume_request=volume_request,
    )

    assert batch.volume_request is not None
    assert "component_count" not in batch.to_record()
    assert [sample.require_field().shape for sample in batch.samples] == [
        (1, 28, 28),
        (1, 28, 28),
    ]
    assert [sample.outcome_id for sample in batch.samples] == [
        "inverse-observation",
        "inverse-observation",
    ]
    assert [sample.component_index for sample in batch.samples] == [0, 0]
    assert math.isclose(batch.log2_volume, requested_log2_volume)
    assert volume_request.contains(batch.log2_volume)


def test_digits_generated_sample_set_records_region_document_boundary() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    batch = _formation_payload(generator, sample_count=4, seed=101)

    assert batch.region is not None
    assert batch.request_outcome is not None
    assert batch.request_outcome.kind == "realized"
    assert batch.request_outcome.region == batch.region
    assert batch.region.volume == 20
    assert math.isclose(batch.region.log2_volume, batch.log2_volume)
    record = batch.to_record()
    loaded = load_object_document(
        canonical_document_bytes(record),
        description="generated sample set record",
    )
    region = state_space_region_from_record(loaded["region"])
    request_outcome = cast(dict[str, object], loaded["request_outcome"])

    assert region == batch.region
    assert request_outcome["kind"] == "realized"
    assert state_space_region_from_record(request_outcome["region"]) == batch.region
    for sample in batch.samples:
        assert sample.region_component_index is not None
        assert sample.axis_coordinates is not None
        assert batch.region.contains(sample.region_component_index, sample.axis_coordinates)
    sample_record = cast(list[dict[str, object]], loaded["samples"])[0]
    assert sample_record["axis_coordinates"] == batch.samples[0].axis_coordinates
    assert sample_record["region_component_index"] == batch.samples[0].region_component_index


def test_digits_truncated_address_window_region_has_unequal_strata() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    module = _digits_benchmark_module()
    volume_class_type = cast(Callable[..., object], module["_DigitsVolumeClass"])
    volume_class = volume_class_type(
        minimum_address=0,
        cardinality=13,
        canvas_side=28,
    )
    region_for_class = cast(
        Callable[..., object],
        module["_digits_state_space_region"],
    )
    region = state_space_region_from_record(
        cast(
            Any,
            region_for_class(
                volume_class=volume_class,
                margin=generator.manifest.resolution_discriminability_margin(),
            ),
        ).to_record()
    )

    assert region.volume == 13
    assert math.isclose(region.log2_volume, math.log2(13))
    assert all(component.volume == 1 for component in region.components)
    strata = Counter(component.stratum_id for component in region.components)
    assert tuple(strata[f"digit-{index}"] for index in range(3)) == (2, 2, 2)
    assert tuple(strata[f"digit-{index}"] for index in range(3, 10)) == (1,) * 7


def test_digits_regions_cover_continuous_transform_increment_coordinates() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    preset = _formation_payload(
        generator,
        sample_count=3,
        seed=101,
        volume_request=StateSpaceVolumeRequest(minimum=3.0, maximum=4.0),
    )
    grid = _formation_payload(
        generator,
        sample_count=3,
        seed=101,
        volume_request=StateSpaceVolumeRequest(minimum=8.0, maximum=9.0),
    )

    assert preset.region is not None
    assert grid.region is not None
    assert preset.region.volume == 8
    assert grid.region.volume == 256
    assert all(
        axis_region.axis.coordinate_kind == "real-interval"
        for component in grid.region.components
        for axis_region in component.axis_regions
    )
    for batch in (preset, grid):
        assert batch.region is not None
        assert math.isclose(batch.log2_volume, batch.region.log2_volume)
        for sample in batch.samples:
            assert sample.region_component_index is not None
            assert sample.axis_coordinates is not None
            assert batch.region.contains(sample.region_component_index, sample.axis_coordinates)


def test_digits_samples_requested_window_by_seeded_monte_carlo() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    request = StateSpaceVolumeRequest(minimum=2.0, maximum=3.0)

    first = _formation_payload(
        generator,
        sample_count=400,
        seed=101,
        volume_request=request,
    )
    repeated = _formation_payload(
        generator,
        sample_count=400,
        seed=101,
        volume_request=request,
    )
    fresh_seed = _formation_payload(
        generator,
        sample_count=400,
        seed=202,
        volume_request=request,
    )

    assert first == repeated
    assert first.region is not None
    assert first.region.volume == 4
    first_signature = tuple(sample.region_component_index for sample in first.samples)
    fresh_signature = tuple(sample.region_component_index for sample in fresh_seed.samples)
    assert first_signature != fresh_signature
    counts = Counter(first_signature)
    assert set(counts) == set(range(first.region.volume))
    assert all(60 <= count <= 140 for count in counts.values())


def test_digits_fresh_seed_samples_have_no_exact_continuous_coordinate_collision() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    request = StateSpaceVolumeRequest(minimum=8.0, maximum=9.0)

    first = _formation_payload(
        generator,
        sample_count=32,
        seed=101,
        volume_request=request,
    )
    fresh_seed = _formation_payload(
        generator,
        sample_count=32,
        seed=202,
        volume_request=request,
    )

    def coordinate_tuples(batch: GeneratedSampleSet) -> set[tuple[tuple[str, object], ...]]:
        return {
            tuple(sorted(cast(Mapping[str, object], sample.axis_coordinates).items()))
            for sample in batch.samples
        }

    assert coordinate_tuples(first).isdisjoint(coordinate_tuples(fresh_seed))


def test_generated_sample_set_rejects_region_coordinates_outside_region() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    batch = _formation_payload(generator, sample_count=1, seed=101)
    assert batch.region is not None
    sample = batch.samples[0]
    assert sample.axis_coordinates is not None

    bad_coordinates = dict(sample.axis_coordinates)
    bad_coordinates["transform-ordinal"] = 2**40
    bad_sample = replace(sample, axis_coordinates=bad_coordinates)

    with pytest.raises(
        ObservationGenerationError,
        match="sample axis coordinates are outside the generated region",
    ):
        GeneratedSampleSet(
            benchmark_id=batch.benchmark_id,
            generator_id=batch.generator_id,
            generator_version=batch.generator_version,
            seed=batch.seed,
            shape=batch.shape,
            samples=(bad_sample,),
            region=batch.region,
        )


def test_digits_generator_materializes_target_volume_class_band() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    generator_impl = cast(Any, generator)
    target = generator.minimum_log2_volume().value + 3.0

    volume_class = generator_impl._volume_class_for_request(
        request=StateSpaceVolumeRequest(
            minimum=target,
            maximum=target + 1.0,
        )
    )

    assert volume_class is not None
    assert volume_class.cardinality == 8
    assert math.isclose(volume_class.log2_volume, math.log2(8))
    metadata = volume_class.metadata()
    assert metadata["kind"] == "digits-realized-setup-window"
    assert metadata["digit_count"] == 10
    assert metadata["output_digit_count"] == 10
    assert metadata["minimum_address"] == 7
    assert metadata["maximum_address"] == 14
    assert metadata["cardinality"] == 8
    assert metadata["realized_cardinality"] == 8
    assert metadata["construction"] == (
        "digit-setups-over-shell-ordered-transform-lattice"
    )
    oracle_reference = CostMeasurement.from_record(metadata["oracle_cost_measurement"])
    assert oracle_reference.abstract_flops == 36 * 36
    assert oracle_reference.execution_mode == "dry-run"
    assert not oracle_reference.operations_executed
    assert metadata["oracle_cost_components"] == {
        "height": 36,
        "width": 36,
        "pixel_count": 36 * 36,
    }
    assert metadata["transform_axes"] == [
        "x_translation",
        "y_translation",
        "scale",
    ]


def test_digits_integer_shells_decode_unique_latent_addresses() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    generator_impl = cast(Any, generator)

    coordinates: set[tuple[int, int]] = set()
    for shell in range(4):
        volume_class = generator_impl._volume_class_for_request(
            request=StateSpaceVolumeRequest(
                minimum=float(shell),
                maximum=float(shell + 1),
            )
        )
        assert volume_class is not None
        expected_cardinality = 2**shell
        expected_minimum_address = 2**shell - 1
        assert volume_class.cardinality == expected_cardinality
        assert volume_class.minimum_address == expected_minimum_address
        for state_index in range(volume_class.cardinality):
            sample_address = volume_class.minimum_address + state_index
            coordinate = (sample_address % 10, sample_address // 10)
            assert coordinate not in coordinates
            coordinates.add(coordinate)


def test_digits_oracle_cost_reference_spans_requested_cost() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    points = cast(Any, generator).oracle_cost_reference_points(
        maximum_cost=10_000_000_000
    )

    assert len(points) > 10
    costs = [
        CostMeasurement.from_record(point["cost_measurement"]).abstract_flops
        for point in points
    ]
    scores = [cast(int | float, point["score"]) for point in points]
    assert costs == sorted(costs)
    assert scores == sorted(scores)
    assert costs[0] == 28 * 28
    assert scores[0] == math.log2(10)
    first_metadata = cast(dict[str, object], points[0]["metadata"])
    first_components = cast(dict[str, object], first_metadata["components"])
    assert first_components["sample_cardinality"] == 10
    assert costs[-1] >= 10_000_000_000
    metadata = cast(dict[str, object], points[-1]["metadata"])
    components = cast(dict[str, object], metadata["components"])
    assert components["height"] == components["width"]
    assert components["pixel_count"] == costs[-1]


def test_digits_generator_high_cardinality_request_has_direct_representative() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    generator_impl = cast(Any, generator)

    volume_class = generator_impl._volume_class_for_request(
        request=StateSpaceVolumeRequest(
            minimum=21.0,
            maximum=22.0,
        )
    )

    assert volume_class is not None
    assert 21.0 <= math.log2(volume_class.cardinality) <= 22.0
    assert not hasattr(generator, "complexity_candidate_for_request")
    assert not hasattr(generator, "complexity_curriculum_candidates")


def test_digits_generator_materializes_large_target_volume_class_directly() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    generator_impl = cast(Any, generator)

    volume_class = generator_impl._volume_class_for_request(
        request=StateSpaceVolumeRequest(minimum=20.0, maximum=21.0)
    )

    assert volume_class is not None
    assert volume_class.cardinality == 1_048_576
    assert 20.0 <= volume_class.log2_volume <= 21.0
    assert cast(int, volume_class.metadata()["maximum_transform_ordinal"]) >= 1


def test_digits_generator_returns_null_set_for_empty_volume_requests() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    volume_request = StateSpaceVolumeRequest(
        minimum=1.0,
        maximum=1.0,
    )
    batch = generator(
        shape=3,
        seed=101,
        volume_request=volume_request,
    )

    assert batch.shape == (0,)
    assert len(batch.samples) == 0
    assert batch.volume_request is not None
    assert batch.volume_request.measure_id == volume_request.measure_id


def test_volume_value_ids_are_core_contract() -> None:
    assert StateSpaceVolumeRequest(minimum=0.0, maximum=0.0).minimum == 0.0
    assert StateSpaceVolumeValue(value=0.0).value == 0.0
    assert (
        str(
            capture_generation_error(
                lambda: StateSpaceVolumeRequest(
                    measure_id="benchmarks.chess.valid-move-count",
                    minimum=1.0,
                    maximum=1.0,
                )
            )
        )
        == "volume measure id is not a core measure"
    )
    assert (
        str(
            capture_generation_error(
                lambda: StateSpaceVolumeValue(
                    measure_id="benchmarks.chess.valid-move-count",
                    value=1.0,
                )
            )
        )
        == "volume measure id is not a core measure"
    )


def test_digits_generator_keeps_chart_canvas_when_memory_cap_is_tiny() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    batch = _formation_payload(generator,
        sample_count=8,
        seed=101,
        memory_limit_bytes=1,
    )

    assert [(sample.width, sample.height) for sample in batch.samples] == [(36, 36)] * 8


def test_digits_generator_records_variation_coordinates_and_renders_inverse_field() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    sample = _observation_payload(generator,
        sample_count=1,
        seed=909,
    ).samples[0]
    variation = _coordinate(sample.latent_coordinates, role="variation")
    variation_values = cast(dict[str, object], variation["values"])
    plan = sample_materialization_plan(sample)
    component_index = sample_component_index(sample)
    width = plan.resolution_assignment.require_axis("W")
    height = plan.resolution_assignment.require_axis("H")
    untransformed = generator.formation.component_field(
        width=width,
        height=height,
        component_index=component_index,
    )

    assert sample.require_field().shape == (1, height, width)
    coordinate = cast(list[Mapping[str, object]], variation_values["coordinates"])[0]
    assert "transform_ordinal" in coordinate
    assert sample.require_field() != untransformed
    assert all(0.0 <= value <= 1.0 for value in sample.require_field().values)


def test_digits_tensor_fields_match_recorded_field_samples() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    runtime = resolve_tensor_runtime("cpu")
    batch = generator(
        shape=4,
        seed=909,
        include_fields=True,
        runtime=runtime,
    )
    host_batch = generator(
        shape=4,
        seed=909,
        include_fields=True,
    )

    fields = batch.require_tensors()[0].detach().cpu()
    for index, sample in enumerate(host_batch.samples):
        assert tuple(fields[index].shape) == sample.require_field().shape
        assert fields[index].flatten().tolist() == list(sample.require_field().values)


def test_digits_tensor_renderer_uses_per_sample_deformation_parameters() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    generator_impl = cast(Any, generator)
    runtime = resolve_tensor_runtime("cpu")
    volume_class = generator_impl._default_volume_class()
    resolution_assignment = volume_class.resolution_assignment(
        width_axis=generator.formation.width_axis,
        height_axis=generator.formation.height_axis,
    )
    width = resolution_assignment.require_axis(generator.formation.width_axis)
    height = resolution_assignment.require_axis(generator.formation.height_axis)
    render_kwargs: dict[str, Any] = {
        "sample_count": 4,
        "width": width,
        "height": height,
        "digit_count": volume_class.digit_count,
        "seed": 101,
        "sample_indices": tuple(range(4)),
        "cardinality": volume_class.cardinality,
        "minimum_address": volume_class.minimum_address,
        "runtime": runtime,
        "timing": None,
        "timing_prefix": "",
    }

    deformed = generator_impl._build_batch_tensor(**render_kwargs)
    module_globals = generator_impl._build_batch_tensor.__globals__
    zero_deformation_latents = tuple(
        module_globals["_inverse_digits_latent_from_nuisance_vector"](
            latent.identity,
            latent.to_nuisance_tuple()[:5]
            + (0.0,) * module_globals["_inverse_deformation_dimension"],
        )
        for latent in (
            module_globals["_inverse_digits_latent_from_address"](
                seed=cast(int, render_kwargs["seed"]),
                sample_index=sample_index,
                sample_address=sample_address,
            )
            for sample_index, sample_address in zip(
                cast(tuple[int, ...], render_kwargs["sample_indices"]),
                module_globals["_digits_sample_addresses"](
                    seed=cast(int, render_kwargs["seed"]),
                    sample_indices=cast(tuple[int, ...], render_kwargs["sample_indices"]),
                    cardinality=cast(int, render_kwargs["cardinality"]),
                    minimum_address=cast(int, render_kwargs["minimum_address"]),
                ),
                strict=True,
            )
        )
    )
    undeformed = module_globals["render_inverse_digits"](
        runtime=runtime,
        latents=zero_deformation_latents,
        canvas_side=width,
    )

    assert not runtime.torch.allclose(deformed, undeformed)


def test_digits_cuda_tensor_fields_match_cpu_reference() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    cpu_runtime = resolve_tensor_runtime("cpu")
    try:
        cuda_runtime = resolve_tensor_runtime("cuda")
    except TensorRuntimeError as error:
        pytest.skip(str(error))
    request = StateSpaceVolumeRequest(
        minimum=generator.minimum_log2_volume().value + 6.0,
        maximum=generator.minimum_log2_volume().value + 7.0,
    )

    cpu_fields = generator(
        shape=16,
        seed=444,
        include_fields=False,
        runtime=cpu_runtime,
        volume_request=request,
    ).require_tensors()[0]
    cuda_fields = generator(
        shape=16,
        seed=444,
        include_fields=False,
        runtime=cuda_runtime,
        volume_request=request,
    ).require_tensors()[0]

    assert cuda_runtime.torch.allclose(cpu_fields, cuda_fields.detach().cpu(), atol=1e-6)


def test_digits_mps_tensor_fields_match_cpu_reference() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    generator_impl = cast(Any, generator)
    cpu_runtime = resolve_tensor_runtime("cpu")
    try:
        mps_runtime = resolve_tensor_runtime("mps")
    except TensorRuntimeError as error:
        pytest.skip(str(error))
    volume_class = generator_impl._volume_class_for_request(
        request=StateSpaceVolumeRequest(
            minimum=generator.minimum_log2_volume().value + 5.0,
            maximum=generator.minimum_log2_volume().value + 6.0,
        )
    )
    assert volume_class is not None
    resolution_assignment = volume_class.resolution_assignment(
        width_axis=generator.formation.width_axis,
        height_axis=generator.formation.height_axis,
    )
    width = resolution_assignment.require_axis(generator.formation.width_axis)
    height = resolution_assignment.require_axis(generator.formation.height_axis)
    render_kwargs: dict[str, Any] = {
        "sample_count": 64,
        "width": width,
        "height": height,
        "digit_count": volume_class.digit_count,
        "seed": 101,
        "sample_indices": tuple(range(64)),
        "cardinality": volume_class.cardinality,
        "minimum_address": volume_class.minimum_address,
        "timing": None,
        "timing_prefix": "",
    }

    cpu_fields = generator_impl._build_batch_tensor(runtime=cpu_runtime, **render_kwargs)
    mps_fields = generator_impl._build_batch_tensor(runtime=mps_runtime, **render_kwargs)
    mps_runtime.torch.mps.synchronize()

    assert cpu_runtime.torch.equal(cpu_fields, mps_fields.detach().cpu())


def test_digits_tensor_generation_returns_null_set_for_unmatched_volume_requests() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    runtime = resolve_tensor_runtime("cpu")
    request = StateSpaceVolumeRequest(minimum=0.5, maximum=0.5)

    sample_set = generator(
        shape=1,
        seed=910,
        runtime=runtime,
        volume_request=request,
    )

    assert sample_set.sample_count == 0
    assert sample_set.samples == ()
    assert sample_set.fields is None
    assert sample_set.targets is None


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
    assert "_constructed_affine_count_products" not in source
    assert "_build_batch_tensor_triton" not in source
    assert "tensor_runtime_prefers_compiled_renderer" not in source
    assert "tl.load" not in source
    assert "def kernel(" not in source


def test_digits_console_preview_png_encoding_is_deterministic() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    seed = 403
    sample_indices = sample_indices_for_even_state_coverage(
        state_count=256,
        seed=seed,
        sample_limit=50,
    )

    left = generator(
        seed=seed,
        shape=len(sample_indices),
        include_artifacts=True,
        volume_request=StateSpaceVolumeRequest(minimum=8.0, maximum=9.0),
        sample_indices=sample_indices,
    )
    right = generator(
        seed=seed,
        shape=len(sample_indices),
        include_artifacts=True,
        volume_request=StateSpaceVolumeRequest(minimum=8.0, maximum=9.0),
        sample_indices=sample_indices,
    )
    sample = left.samples[0].to_record()
    data_url = cast(str, sample["image_data_url"])

    assert left.to_record() == right.to_record()
    assert len(left.samples) == 50
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

    assert math.isclose(canonical.log2_volume, math.log2(10))
    assert full.log2_volume > canonical.log2_volume
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


def _digits_benchmark_module() -> dict[str, object]:
    module_name = "test_digits_benchmark"
    entrypoint = _digits_benchmark_root / "benchmark.py"
    spec = importlib.util.spec_from_file_location(module_name, entrypoint)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return vars(module)


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
    edge_threshold = 5.0e-2
    channels, height, width = field.shape
    for channel in range(channels):
        channel_offset = channel * width * height
        for x_index in range(width):
            if field.values[channel_offset + x_index] > edge_threshold:
                return True
            if (
                field.values[channel_offset + (height - 1) * width + x_index]
                > edge_threshold
            ):
                return True
        for y_index in range(height):
            if field.values[channel_offset + y_index * width] > edge_threshold:
                return True
            if (
                field.values[channel_offset + y_index * width + width - 1]
                > edge_threshold
            ):
                return True
    return False


def capture_generation_error(call: Callable[[], object]) -> ObservationGenerationError:
    with pytest.raises(ObservationGenerationError) as error:
        call()
    return error.value
