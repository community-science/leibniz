from pathlib import Path
from typing import Any, cast

import pytest
from benchmark_typing import DigitsGenerator, load_digits_benchmark, load_digits_generator

from leibniz.architectures import ArchitectureManifest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import MaterializationPlanDocument
from leibniz.tensor_runtime import (
    FormationTensorCache,
    TensorRuntimeError,
    architecture_supported_by_tensor_runtime,
    architecture_tensor_runtime_issue,
    resolve_tensor_runtime,
    runtime_roofline_record,
    tensor_runtime_device_kinds,
    validate_tensor_runtime_device,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_digits_fixture_root = _repository_root / "tests" / "fixtures" / "digits"


def _formation_payload(generator: DigitsGenerator, *, sample_count: int, seed: int):
    sample_set = generator(
        shape=sample_count,
        seed=seed,
    )
    return sample_set


def _observation_payload(generator: DigitsGenerator, *, sample_count: int, seed: int):
    sample_set = generator(
        shape=sample_count,
        seed=seed,
        include_fields=True,
    )
    return sample_set


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
    assert tensor_runtime_device_kinds("auto")[-1] == "cpu"


def test_resolve_tensor_runtime_rejects_unavailable_explicit_device() -> None:
    runtime = resolve_tensor_runtime("auto")
    if runtime.torch.cuda.is_available():
        assert resolve_tensor_runtime("cuda").device_kind == "cuda"
    else:
        with pytest.raises(TensorRuntimeError, match="cuda is not available"):
            resolve_tensor_runtime("cuda")


def test_mps_architecture_support_allows_operation_level_fallback() -> None:
    supported = ArchitectureManifest.from_record(
        {
            "input_shape": [1, 32, 32],
            "output_shape": [10],
            "layers": [
                {"kind": "adaptive-pooling", "parameters": {"dimension": 2, "size": 16}},
                {"kind": "flatten"},
                {"kind": "dense", "parameters": {"out": 10}},
            ],
        }
    )
    unsupported = ArchitectureManifest.from_record(
        {
            "input_shape": [1, 32, 32],
            "output_shape": [10],
            "layers": [
                {"kind": "adaptive-pooling", "parameters": {"dimension": 2, "size": 31}},
                {"kind": "flatten"},
                {"kind": "dense", "parameters": {"out": 10}},
            ],
        }
    )

    assert architecture_supported_by_tensor_runtime(supported, device_kind="mps")
    assert architecture_supported_by_tensor_runtime(unsupported, device_kind="cpu")
    assert architecture_supported_by_tensor_runtime(unsupported, device_kind="mps")
    assert architecture_tensor_runtime_issue(unsupported, device_kind="mps") is None


def test_runtime_roofline_record_calibrates_cpu_ceiling() -> None:
    runtime = resolve_tensor_runtime("cpu")

    record = runtime_roofline_record(runtime)

    assert record["kind"] == "system-roofline"
    assert record["status"] == "calibrated"
    assert record["tensor_runtime"] == "pytorch"
    assert record["tensor_device"] == "cpu"
    assert cast(float, record["peak_compute_per_second"]) > 0
    assert cast(float, record["peak_bytes_per_second"]) > 0
    assert record["method"] == "dense-matmul-and-copy-calibration"


def test_formation_tensor_cache_matches_unvaried_pure_digits_formation() -> None:
    runtime = resolve_tensor_runtime("cpu")
    declaration = load_digits_benchmark(_digits_benchmark_root).formation
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l3.json").read_bytes()
    ).plan
    component_index = 1
    width = plan.resolution_assignment.require_axis(declaration.width_axis)
    height = plan.resolution_assignment.require_axis(declaration.height_axis)
    cache = FormationTensorCache(runtime=runtime, formation=declaration)

    pure = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.tensor-cache@0.1.0"),
        plan=plan,
        component_index=component_index,
    )
    tensor = cache.component_tensor(
        width=width,
        height=height,
        component_index=component_index,
    )

    assert tensor.shape == pure.field.shape
    assert tuple(tensor.reshape(-1).cpu().tolist()) == pure.field.values


def test_formation_tensor_cache_batch_tensors_match_pure_observation_batch() -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = load_digits_generator(_digits_benchmark_root)
    observation_batch = _observation_payload(generator, sample_count=3, seed=515)
    formation_batch = _formation_payload(generator, sample_count=3, seed=515)
    outcome_ids = tuple(
        outcome.id
        for outcome in generator.manifest.resolve_outcome_space().outcomes
    )
    cache = FormationTensorCache(runtime=runtime, formation=generator.formation)

    fields, labels = cache.batch_tensors(batch=formation_batch, outcome_ids=outcome_ids)

    pure_fields = runtime.torch.tensor(
        [list(sample.require_field().values) for sample in observation_batch.samples],
        dtype=runtime.torch.float32,
        device=runtime.device,
    ).reshape((len(observation_batch.samples), *observation_batch.samples[0].require_field().shape))
    assert fields.shape == pure_fields.shape
    assert runtime.torch.allclose(fields, pure_fields, atol=2e-5)
    assert labels.cpu().tolist() == [
        outcome_ids.index(sample.outcome_id) for sample in observation_batch.samples
    ]


def test_formation_tensor_cache_batches_grid_sampling_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = load_digits_generator(_digits_benchmark_root)
    formation_batch = _formation_payload(generator, sample_count=3, seed=515)
    outcome_ids = tuple(
        outcome.id
        for outcome in generator.manifest.resolve_outcome_space().outcomes
    )
    cache = FormationTensorCache(runtime=runtime, formation=generator.formation)
    calls = {"affine_grid": 0, "grid_sample": 0}
    original_affine_grid = runtime.torch.nn.functional.affine_grid
    original_grid_sample = runtime.torch.nn.functional.grid_sample

    def count_affine_grid(*args: Any, **kwargs: Any) -> Any:
        calls["affine_grid"] += 1
        return original_affine_grid(*args, **kwargs)

    def count_grid_sample(*args: Any, **kwargs: Any) -> Any:
        calls["grid_sample"] += 1
        return original_grid_sample(*args, **kwargs)

    monkeypatch.setattr(
        runtime.torch.nn.functional,
        "affine_grid",
        cast(Any, count_affine_grid),
    )
    monkeypatch.setattr(
        runtime.torch.nn.functional,
        "grid_sample",
        cast(Any, count_grid_sample),
    )

    cache.batch_tensors(batch=formation_batch, outcome_ids=outcome_ids)

    assert calls == {"affine_grid": 1, "grid_sample": 1}


def test_formation_tensor_cache_batch_tensors_use_generated_coordinate_values() -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = load_digits_generator(_digits_benchmark_root)
    formation_batch = _formation_payload(generator, sample_count=3, seed=515)
    outcome_ids = tuple(
        outcome.id
        for outcome in generator.manifest.resolve_outcome_space().outcomes
    )
    cache = FormationTensorCache(runtime=runtime, formation=generator.formation)

    fields, labels = cache.batch_tensors(batch=formation_batch, outcome_ids=outcome_ids)

    assert fields.shape[0] == len(formation_batch.samples)
    assert labels.cpu().tolist() == [
        outcome_ids.index(sample.outcome_id) for sample in formation_batch.samples
    ]


def test_formation_tensor_cache_reuses_component_tensors() -> None:
    runtime = resolve_tensor_runtime("cpu")
    declaration = load_digits_benchmark(_digits_benchmark_root).formation
    cache = FormationTensorCache(runtime=runtime, formation=declaration)

    left = cache.component_tensor(
        width=96,
        height=32,
        component_index=4,
    )
    right = cache.component_tensor(
        width=96,
        height=32,
        component_index=4,
    )

    assert left is right


def test_formation_tensor_cache_rejects_invalid_component_requests() -> None:
    runtime = resolve_tensor_runtime("cpu")
    declaration = load_digits_benchmark(_digits_benchmark_root).formation
    cache = FormationTensorCache(runtime=runtime, formation=declaration)

    with pytest.raises(TensorRuntimeError, match="component_index is outside"):
        cache.component_tensor(
            width=96,
            height=32,
            component_index=len(declaration.components),
        )
