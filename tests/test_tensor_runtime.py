from pathlib import Path

import pytest

from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import MaterializationPlanDocument
from leibniz.observation_formation import ObservationFormationDeclarationDocument
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
