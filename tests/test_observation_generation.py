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

from leibniz.materialization import AxisAssignment, MaterializationPlan
from leibniz.observation_formation import FieldObservation
from leibniz.observation_generation import (
    ComplexityRequest,
    ComplexityValue,
    GeneratedSample,
    GeneratedSampleSet,
    ObservationGenerationError,
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


def test_generated_sample_records_available_outcome_ids() -> None:
    sample = GeneratedSample(
        index=0,
        outcome_id="yes",
        complexity=1.0,
        available_outcome_ids=("no", "yes"),
    )

    assert sample.to_record()["available_outcome_ids"] == ["no", "yes"]


def test_generated_sample_rejects_invalid_available_outcome_ids() -> None:
    with pytest.raises(ObservationGenerationError, match="available_outcome_ids must be unique"):
        GeneratedSample(
            index=0,
            outcome_id="yes",
            complexity=1.0,
            available_outcome_ids=("yes", "yes"),
        )

    with pytest.raises(ObservationGenerationError, match="available_outcome_ids must be nonempty"):
        GeneratedSample(
            index=0,
            outcome_id="yes",
            complexity=1.0,
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
        first_sample.complexity,
        generator.distinguishable_state_complexity(
            width=first_width,
            height=first_height,
        ),
    )
    assert sample_component_index(first_sample) == first_sample.component_index
    assert first_sample.outcome_id == f"digit-{first_sample.component_index}"
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
    assert variation["degree_measure"] == {"kind": "vector-dimension", "count": 6.0}
    variation_values = cast(dict[str, object], variation["values"])
    assert variation_values["kind"] == "constructed-field-variation-transform-samples"
    assert variation_values["transform_count"] == 2
    assert cast(dict[str, object], variation_values["bounds"]) == (
        generator.formation.variation_transform.to_record()
    )
    coordinates = cast(list[dict[str, object]], variation_values["coordinates"])
    assert [coordinate["component_index"] for coordinate in coordinates] == [
        first_sample.component_index
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
        sample.complexity,
        generator.distinguishable_state_complexity(width=width, height=height),
    )
    assert sample.outcome_id == f"digit-{sample.component_index}"


def test_digits_generator_counts_constructed_finite_complexity_class() -> None:
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
    minimum = generator.constructed_complexity_class_complexity(
        affine_transform_count=1,
    )
    larger = generator.constructed_complexity_class_complexity(
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


def test_digits_generator_accepts_complexity_value_requests() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    requested_complexity = generator.minimum_complexity().value

    complexity_request = ComplexityRequest(
        minimum=requested_complexity,
        maximum=requested_complexity + 1.0,
    )
    batch = _observation_payload(
        generator,
        sample_count=2,
        seed=101,
        complexity_request=complexity_request,
    )

    assert batch.complexity_request is not None
    assert "component_count" not in batch.to_record()
    assert [sample.require_field().shape for sample in batch.samples] == [
        (1, 16, 16),
        (1, 16, 16),
    ]
    assert [sample.outcome_id for sample in batch.samples] == ["digit-0", "digit-0"]
    assert [sample.component_index for sample in batch.samples] == [0, 0]
    assert {
        sample.complexity_value
        for sample in batch.samples
    } == {
        batch.samples[0].complexity_value,
    }
    assert batch.samples[0].complexity_value is not None
    assert batch.samples[0].complexity_value.measure_id == complexity_request.measure_id
    assert math.isclose(batch.samples[0].complexity_value.value, requested_complexity)


def test_digits_generator_materializes_target_complexity_class_band() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    generator_impl = cast(Any, generator)
    target = generator.minimum_complexity().value + 3.0

    complexity_class = generator_impl._complexity_class_for_request(
        request=ComplexityRequest(
            minimum=target,
            maximum=target + 1.0,
        )
    )

    assert complexity_class is not None
    assert complexity_class.cardinality == 8
    assert math.isclose(complexity_class.complexity, math.log2(8))
    assert complexity_class.resolution_assignment is not None
    metadata = complexity_class.metadata()
    assert metadata["affine_transform_count"] == 2
    assert metadata["digit_count"] == 10
    assert metadata["output_digit_count"] == 10
    assert metadata["minimum_address"] == 7
    assert metadata["maximum_address"] == 14
    assert metadata["requested_cardinality"] == 8
    assert metadata["realized_cardinality"] == 8
    assert metadata["construction"] == (
        "symmetric-digits-over-finite-affine-product-grid"
    )
    oracle_reference = cast(
        dict[str, object],
        metadata["oracle_inference_compute"],
    )
    assert oracle_reference["kind"] == "oracle-inference-compute-reference-v1"
    assert oracle_reference["unit"] == "abstract-ops"
    assert oracle_reference["value"] == 16 * 16
    assert oracle_reference["components"] == {
        "height": 16,
        "width": 16,
        "pixel_count": 16 * 16,
    }
    assert metadata["affine_parameters"] == [
        "x_translation",
        "y_translation",
        "scale",
        "rotation",
        "x_shear",
    ]


def test_digits_integer_shells_decode_unique_latent_addresses() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    generator_impl = cast(Any, generator)

    coordinates: set[tuple[int, int]] = set()
    for shell in range(4):
        complexity_class = generator_impl._complexity_class_for_request(
            request=ComplexityRequest(
                minimum=float(shell),
                maximum=float(shell + 1),
            )
        )
        assert complexity_class is not None
        assert complexity_class.cardinality == 2**shell
        assert complexity_class.minimum_address == 2**shell - 1
        for state_index in range(complexity_class.cardinality):
            sample_address = complexity_class.minimum_address + state_index
            coordinate = (sample_address % 10, sample_address // 10)
            assert coordinate not in coordinates
            coordinates.add(coordinate)


def test_digits_oracle_inference_reference_spans_requested_cost() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    points = cast(Any, generator).oracle_inference_reference_points(
        maximum_cost=10_000_000_000
    )

    assert len(points) > 10
    costs = [cast(int | float, point["cost"]) for point in points]
    scores = [cast(int | float, point["score"]) for point in points]
    assert costs == sorted(costs)
    assert scores == sorted(scores)
    assert costs[0] == 16 * 16
    assert scores[0] == 0
    first_metadata = cast(dict[str, object], points[0]["metadata"])
    first_components = cast(dict[str, object], first_metadata["components"])
    assert first_components["sample_cardinality"] == 1
    assert costs[-1] >= 10_000_000_000
    metadata = cast(dict[str, object], points[-1]["metadata"])
    components = cast(dict[str, object], metadata["components"])
    assert components["height"] == components["width"]
    assert components["pixel_count"] == costs[-1]


def test_digits_generator_high_cardinality_request_has_direct_representative() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    generator_impl = cast(Any, generator)

    complexity_class = generator_impl._complexity_class_for_request(
        request=ComplexityRequest(
            minimum=21.0,
            maximum=22.0,
        )
    )

    assert complexity_class is not None
    assert 21.0 <= math.log2(complexity_class.cardinality) <= 22.0
    assert not hasattr(generator, "complexity_candidate_for_request")
    assert not hasattr(generator, "complexity_curriculum_candidates")


def test_digits_generator_materializes_large_target_complexity_class_directly() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    generator_impl = cast(Any, generator)

    complexity_class = generator_impl._complexity_class_for_request(
        request=ComplexityRequest(minimum=20.0, maximum=21.0)
    )

    assert complexity_class is not None
    assert complexity_class.cardinality == 1_048_576
    assert 20.0 <= complexity_class.complexity <= 21.0
    assert complexity_class.resolution_assignment is not None
    assert cast(int, complexity_class.metadata()["affine_product_cardinality"]) >= (
        complexity_class.maximum_address + 1
    )


def test_digits_generator_accepts_low_sample_cardinality_requests() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    complexity_request = ComplexityRequest(
        minimum=1.0,
        maximum=1.0,
    )
    batch = generator(
        shape=3,
        seed=101,
        complexity_request=complexity_request,
    )

    assert batch.shape == (3,)
    assert len(batch.samples) == 3
    assert batch.complexity_request is not None
    assert batch.complexity_request.measure_id == complexity_request.measure_id


def test_complexity_value_ids_are_core_contract() -> None:
    assert ComplexityRequest(minimum=0.0, maximum=0.0).minimum == 0.0
    assert ComplexityValue(value=0.0).value == 0.0
    assert (
        str(
            capture_generation_error(
                lambda: ComplexityRequest(
                    measure_id="benchmarks.chess.valid-move-count",
                    minimum=1.0,
                    maximum=1.0,
                )
            )
        )
        == "complexity id is not a core measure"
    )
    assert (
        str(
            capture_generation_error(
                lambda: ComplexityValue(
                    measure_id="benchmarks.chess.valid-move-count",
                    value=1.0,
                )
            )
        )
        == "complexity id is not a core measure"
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
    ).samples[0]
    variation = _coordinate(sample.latent_coordinates, role="variation")
    variation_values = cast(dict[str, object], variation["values"])
    plan = sample_materialization_plan(sample)
    component_index = sample_component_index(sample)
    direct = generator.formation.component_field(
        width=plan.resolution_assignment.require_axis("W"),
        height=plan.resolution_assignment.require_axis("H"),
        component_index=component_index,
        variation_coordinate=cast(list[Mapping[str, object]], variation_values["coordinates"])[0],
    )
    untransformed = generator.formation.component_field(
        width=plan.resolution_assignment.require_axis("W"),
        height=plan.resolution_assignment.require_axis("H"),
        component_index=component_index,
    )

    assert sample.require_field() == direct
    assert sample.require_field() != untransformed
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
    host_batch = generator(
        shape=4,
        seed=909,
        include_fields=True,
    )

    fields = batch.require_tensors()[0].detach().cpu()
    for index, sample in enumerate(host_batch.samples):
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
    request = ComplexityRequest(
        minimum=generator.minimum_complexity().value + 6.0,
        maximum=generator.minimum_complexity().value + 7.0,
    )

    cpu_fields = generator(
        shape=16,
        seed=444,
        include_fields=False,
        runtime=cpu_runtime,
        outcome_ids=outcome_ids,
        complexity_request=request,
    ).require_tensors()[0]
    cuda_fields = generator(
        shape=16,
        seed=444,
        include_fields=False,
        runtime=cuda_runtime,
        outcome_ids=outcome_ids,
        complexity_request=request,
    ).require_tensors()[0]

    assert cuda_runtime.torch.equal(cpu_fields, cuda_fields.detach().cpu())


def test_digits_mps_tensor_fields_match_cpu_reference() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    generator_impl = cast(Any, generator)
    cpu_runtime = resolve_tensor_runtime("cpu")
    try:
        mps_runtime = resolve_tensor_runtime("mps")
    except TensorRuntimeError as error:
        pytest.skip(str(error))
    complexity_class = generator_impl._complexity_class_for_request(
        request=ComplexityRequest(
            minimum=generator.minimum_complexity().value + 5.0,
            maximum=generator.minimum_complexity().value + 6.0,
        )
    )
    assert complexity_class is not None
    resolution_assignment = complexity_class.resolution_assignment
    assert resolution_assignment is not None
    width = resolution_assignment.require_axis(generator.formation.width_axis)
    height = resolution_assignment.require_axis(generator.formation.height_axis)
    render_kwargs: dict[str, Any] = {
        "sample_count": 64,
        "width": width,
        "height": height,
        "digit_count": complexity_class.digit_count,
        "transform": generator.formation.variation_transform,
        "grid": complexity_class.affine_grid,
        "seed": 101,
        "cardinality": complexity_class.cardinality,
        "minimum_address": complexity_class.minimum_address,
        "timing": None,
        "timing_prefix": "",
    }

    cpu_fields = generator_impl._build_batch_tensor(runtime=cpu_runtime, **render_kwargs)
    mps_fields = generator_impl._build_batch_tensor(runtime=mps_runtime, **render_kwargs)
    mps_runtime.torch.mps.synchronize()

    assert cpu_runtime.torch.equal(cpu_fields, mps_fields.detach().cpu())


def test_digits_tensor_generation_rejects_unmatched_complexity_requests() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    runtime = resolve_tensor_runtime("cpu")
    outcome_space = generator.manifest.resolve_outcome_space()
    request = ComplexityRequest(minimum=0.5, maximum=0.5)

    with pytest.raises(
        ObservationGenerationError,
        match="tensor generation complexity request matched no shell",
    ):
        generator(
            shape=1,
            seed=910,
            runtime=runtime,
            outcome_ids=tuple(outcome.id for outcome in outcome_space.outcomes),
            complexity_request=request,
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
    assert "_constructed_affine_count_products" not in source
    assert "_build_batch_tensor_triton" not in source
    assert "tensor_runtime_prefers_compiled_renderer" not in source
    assert "tl.load" not in source
    assert "def kernel(" not in source


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
                ("[3, 4]", 8),
                ("[5, 6]", 32),
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
