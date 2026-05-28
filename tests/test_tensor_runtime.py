from pathlib import Path

import pytest

from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import MaterializationPlanDocument
from leibniz.observation_formation import ObservationFormationDeclarationDocument
from leibniz.observation_generation import (
    load_observation_generator,
    sample_variation_transform_coordinates,
)
from leibniz.tensor_runtime import (
    FormationTensorCache,
    TensorRuntimeError,
    resolve_tensor_runtime,
    validate_tensor_runtime_device,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_digits_fixture_root = _repository_root / "tests" / "fixtures" / "digits"


def test_validate_tensor_runtime_device_accepts_supported_names() -> None:
    assert validate_tensor_runtime_device("auto") == "auto"
    assert validate_tensor_runtime_device("cpu") == "cpu"
    assert validate_tensor_runtime_device("cuda") == "cuda"
    assert validate_tensor_runtime_device("mps") == "mps"


def test_validate_tensor_runtime_device_rejects_unknown_name() -> None:
    with pytest.raises(TensorRuntimeError, match="auto, cpu, cuda, mps"):
        validate_tensor_runtime_device("gpu")


def test_resolve_tensor_runtime_cpu_uses_cpu_device() -> None:
    runtime = resolve_tensor_runtime("cpu")

    assert runtime.device_kind == "cpu"
    assert str(runtime.device) == "cpu"


def test_resolve_tensor_runtime_auto_uses_available_device() -> None:
    runtime = resolve_tensor_runtime("auto")

    assert runtime.device_kind in {"cpu", "cuda", "mps"}


def test_resolve_tensor_runtime_rejects_unavailable_explicit_device() -> None:
    runtime = resolve_tensor_runtime("auto")
    if runtime.torch.cuda.is_available():
        assert resolve_tensor_runtime("cuda").device_kind == "cuda"
    else:
        with pytest.raises(TensorRuntimeError, match="cuda is not available"):
            resolve_tensor_runtime("cuda")


def test_formation_tensor_cache_matches_unvaried_pure_digits_formation() -> None:
    runtime = resolve_tensor_runtime("cpu")
    declaration = ObservationFormationDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "observation_formation.json").read_bytes()
    ).declaration
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l3.json").read_bytes()
    ).plan
    sequence = (1, 2, 3)
    resolution = plan.resolution_assignment.require_axis(declaration.resolution_axis)
    cache = FormationTensorCache(runtime=runtime, formation=declaration)

    pure = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.tensor-cache@0.1.0"),
        plan=plan,
        component_sequence=sequence,
    )
    tensor = cache.component_sequence_tensor(
        resolution=resolution,
        component_sequence=sequence,
    )

    assert tensor.shape == pure.field.shape
    assert tuple(tensor.reshape(-1).cpu().tolist()) == pure.field.values


def test_formation_tensor_cache_matches_varied_pure_digits_formation() -> None:
    runtime = resolve_tensor_runtime("cpu")
    declaration = ObservationFormationDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "observation_formation.json").read_bytes()
    ).declaration
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l3.json").read_bytes()
    ).plan
    sequence = (1, 2, 3)
    resolution = plan.resolution_assignment.require_axis(declaration.resolution_axis)
    coordinates = tuple(
        sample_variation_transform_coordinates(
            transform=declaration.variation_transform,
            seed=plan.seed,
            sample_index=7,
            slot_index=slot_index,
        )
        for slot_index in range(len(sequence))
    )
    cache = FormationTensorCache(runtime=runtime, formation=declaration)

    pure = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.tensor-varied@0.1.0"),
        plan=plan,
        component_sequence=sequence,
        variation_coordinates=coordinates,
    )
    tensor = cache.varied_component_sequence_tensor(
        resolution=resolution,
        component_sequence=sequence,
        variation_coordinates=coordinates,
    )

    pure_tensor = runtime.torch.tensor(
        pure.field.values,
        dtype=runtime.torch.float32,
        device=runtime.device,
    ).reshape(pure.field.shape)
    assert tensor.shape == pure.field.shape
    assert runtime.torch.allclose(tensor, pure_tensor, atol=2e-5)


def test_formation_tensor_cache_batch_tensors_match_pure_observation_batch() -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = load_observation_generator(_digits_benchmark_root)
    observation_batch = generator.sample_batch(scale=2, sample_count=3, seed=515)
    formation_batch = generator.sample_formation_batch(scale=2, sample_count=3, seed=515)
    outcome_ids = tuple(
        outcome.id
        for outcome in generator.benchmark_manifest.resolve_outcome_space(scale=2).outcomes
    )
    cache = FormationTensorCache(runtime=runtime, formation=generator.formation)

    fields, labels = cache.batch_tensors(batch=formation_batch, outcome_ids=outcome_ids)

    pure_fields = runtime.torch.tensor(
        [list(sample.field.values) for sample in observation_batch.samples],
        dtype=runtime.torch.float32,
        device=runtime.device,
    ).reshape((len(observation_batch.samples), *observation_batch.samples[0].field.shape))
    assert fields.shape == pure_fields.shape
    assert runtime.torch.allclose(fields, pure_fields, atol=2e-5)
    assert labels.cpu().tolist() == [
        outcome_ids.index(sample.outcome_id)
        for sample in observation_batch.samples
    ]


def test_formation_tensor_cache_reuses_component_tensors() -> None:
    runtime = resolve_tensor_runtime("cpu")
    declaration = ObservationFormationDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "observation_formation.json").read_bytes()
    ).declaration
    cache = FormationTensorCache(runtime=runtime, formation=declaration)

    left = cache.component_tensor(
        resolution=96,
        slot_count=3,
        slot_index=1,
        component_index=4,
    )
    right = cache.component_tensor(
        resolution=96,
        slot_count=3,
        slot_index=1,
        component_index=4,
    )

    assert left is right


def test_formation_tensor_cache_rejects_invalid_component_requests() -> None:
    runtime = resolve_tensor_runtime("cpu")
    declaration = ObservationFormationDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "observation_formation.json").read_bytes()
    ).declaration
    cache = FormationTensorCache(runtime=runtime, formation=declaration)

    with pytest.raises(TensorRuntimeError, match="component_sequence must not be empty"):
        cache.component_sequence_tensor(resolution=96, component_sequence=())
    with pytest.raises(TensorRuntimeError, match="slot_index must be within slot_count"):
        cache.component_tensor(
            resolution=96,
            slot_count=3,
            slot_index=3,
            component_index=0,
        )
    with pytest.raises(TensorRuntimeError, match="component_index is outside"):
        cache.component_tensor(
            resolution=96,
            slot_count=3,
            slot_index=0,
            component_index=len(declaration.components),
        )


def test_formation_tensor_cache_rejects_invalid_variation_coordinates() -> None:
    runtime = resolve_tensor_runtime("cpu")
    declaration = ObservationFormationDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "observation_formation.json").read_bytes()
    ).declaration
    cache = FormationTensorCache(runtime=runtime, formation=declaration)

    with pytest.raises(TensorRuntimeError, match="length must match slot count"):
        cache.varied_component_sequence_tensor(
            resolution=96,
            component_sequence=(1, 2),
            variation_coordinates=(),
        )
    coordinate = sample_variation_transform_coordinates(
        transform=declaration.variation_transform,
        seed=101,
        sample_index=0,
        slot_index=1,
    )
    with pytest.raises(TensorRuntimeError, match="slot_index must match"):
        cache.varied_component_sequence_tensor(
            resolution=96,
            component_sequence=(1,),
            variation_coordinates=(coordinate,),
        )
