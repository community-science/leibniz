import math
from pathlib import Path
from typing import Any, cast

import pytest
from benchmark_typing import load_digits_generator

from leibniz import tensor_runtime as tensor_runtime_module
from leibniz.architectures import ArchitectureManifest
from leibniz.observation_generation import ObservationGenerationError
from leibniz.tensor_runtime import (
    TensorElementParameter,
    TensorElementProgram,
    TensorElementRecipe,
    TensorRuntime,
    TensorRuntimeError,
    architecture_supported_by_tensor_runtime,
    architecture_tensor_runtime_issue,
    build_optimizer,
    optimizer_step,
    resolve_tensor_runtime,
    runtime_roofline_record,
    softmax_target_masses,
    tensor_runtime_construct_tensor,
    tensor_runtime_device_kinds,
    validate_tensor_runtime_device,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


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


def test_softmax_target_masses_accepts_labels_and_distributions() -> None:
    runtime = resolve_tensor_runtime("cpu")
    logits = runtime.torch.tensor([[0.0, 2.0, 1.0], [3.0, 1.0, 0.0]])
    labels = runtime.torch.tensor([1, 0])
    distributions = runtime.torch.tensor([[0.0, 1.0, 0.0], [0.5, 0.5, 0.0]])

    label_masses = softmax_target_masses(runtime, logits, labels)
    distribution_masses = softmax_target_masses(runtime, logits, distributions)
    probabilities = runtime.torch.softmax(logits, dim=1)

    expected_label_masses = probabilities.gather(
        1, labels.reshape((-1, 1))
    ).reshape((-1,)).tolist()
    expected_distribution_masses = (probabilities * distributions).sum(dim=1).tolist()

    assert all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-7)
        for actual, expected in zip(label_masses, expected_label_masses, strict=True)
    )
    assert all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-7)
        for actual, expected in zip(
            distribution_masses,
            expected_distribution_masses,
            strict=True,
        )
    )


def test_loss_search_optimizer_decreases_loss_without_learning_rate() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    parameter = torch.nn.Parameter(torch.tensor([2.0], device=runtime.device))
    optimizer = build_optimizer(
        runtime,
        name="loss-search",
        parameters=[parameter],
        learning_rate=None,
    )

    def closure() -> Any:
        optimizer.zero_grad(set_to_none=True)
        loss = (parameter - 0.25).pow(2).sum()
        loss.backward()
        return loss

    baseline_loss = float(closure().detach())
    optimizer_step(runtime, optimizer, closure)
    stepped_loss = float(closure().detach())

    assert "lr" not in optimizer.param_groups[0]
    assert optimizer.state[parameter]["step"] == 1
    assert "exp_avg" in optimizer.state[parameter]
    assert "exp_avg_sq" in optimizer.state[parameter]
    assert stepped_loss < baseline_loss


def test_loss_search_optimizer_rejects_missing_closure() -> None:
    runtime = resolve_tensor_runtime("cpu")
    parameter = runtime.torch.nn.Parameter(runtime.torch.tensor([1.0]))
    optimizer = build_optimizer(
        runtime,
        name="loss-search",
        parameters=[parameter],
        learning_rate=None,
    )

    with pytest.raises(TensorRuntimeError, match="requires a loss closure"):
        optimizer.step()


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


def test_digits_generator_call_tensors_are_deterministic_and_tensor_native() -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = load_digits_generator(_digits_benchmark_root)
    outcome_ids = tuple(
        outcome.id
        for outcome in generator.manifest.resolve_outcome_space().outcomes
    )

    left = generator(
        shape=3,
        seed=515,
        include_metadata=False,
        runtime=runtime,
        outcome_ids=outcome_ids,
    )
    right = generator(
        shape=3,
        seed=515,
        include_metadata=False,
        runtime=runtime,
        outcome_ids=outcome_ids,
    )
    other = generator(
        shape=3,
        seed=516,
        include_metadata=False,
        runtime=runtime,
        outcome_ids=outcome_ids,
    )
    fields, labels = left.require_tensors()
    right_fields, right_labels = right.require_tensors()
    other_fields, other_labels = other.require_tensors()

    assert left.samples == ()
    assert tuple(fields.shape[:2]) == (3, 1)
    assert fields.ndim == 4
    assert labels.shape == (3, len(outcome_ids))
    assert runtime.torch.allclose(fields, right_fields)
    assert runtime.torch.equal(labels, right_labels)
    assert (
        fields.shape != other_fields.shape
        or not runtime.torch.allclose(fields, other_fields)
        or not runtime.torch.equal(labels, other_labels)
    )
    assert labels.sum(dim=1).cpu().tolist() == [1.0, 1.0, 1.0]
    assert set(labels.cpu().flatten().tolist()) <= {0.0, 1.0}


def test_digits_generator_call_tensors_do_not_post_warp_rasters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = load_digits_generator(_digits_benchmark_root)
    outcome_ids = tuple(
        outcome.id
        for outcome in generator.manifest.resolve_outcome_space().outcomes
    )
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

    generator(
        shape=3,
        seed=515,
        include_metadata=False,
        runtime=runtime,
        outcome_ids=outcome_ids,
    ).require_tensors()

    assert calls == {"affine_grid": 0, "grid_sample": 0}


def test_tensor_element_compile_cache_is_not_extent_dependent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_runtime = resolve_tensor_runtime("cpu")
    runtime = TensorRuntime(
        torch=host_runtime.torch,
        device=host_runtime.device,
        device_kind="cuda",
    )
    compile_calls: list[bool | None] = []

    monkeypatch.setattr(tensor_runtime_module, "_tensor_runtime_compile_available", lambda: True)

    def compile_kernel(kernel: Any, **kwargs: Any) -> Any:
        compile_calls.append(cast(bool | None, kwargs.get("dynamic")))
        return kernel

    monkeypatch.setattr(runtime.torch, "compile", compile_kernel)

    def element_function(coordinates: tuple[Any, ...], flat_indices: Any) -> Any:
        _ = flat_indices
        return coordinates[0]

    program = TensorElementProgram(
        kernel=element_function,
        parameters={},
        cache_key=("extent-independent-test", id(element_function)),
    )

    first = tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(shape=(2,), dtype="int64", program=program),
    )
    second = tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(shape=(3,), dtype="int64", program=program),
    )

    assert compile_calls == [None]
    assert first.tolist() == [0, 1]
    assert second.tolist() == [0, 1, 2]


def test_tensor_element_parameter_cache_reuses_constant_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = resolve_tensor_runtime("cpu")
    tensor_calls = 0
    original_tensor = runtime.torch.tensor

    def tensor_with_count(*args: Any, **kwargs: Any) -> Any:
        nonlocal tensor_calls
        tensor_calls += 1
        return original_tensor(*args, **kwargs)

    monkeypatch.setattr(runtime.torch, "tensor", tensor_with_count)

    def element_function(
        coordinates: tuple[Any, ...],
        flat_indices: Any,
        *,
        constant: Any,
        dynamic: Any,
    ) -> Any:
        _ = flat_indices
        return coordinates[0] + constant[0] + dynamic[coordinates[0]]

    program = TensorElementProgram(
        kernel=element_function,
        parameters={
            "constant": TensorElementParameter(
                dtype="int64",
                shape=(1,),
                values=(7,),
            ),
            "dynamic": TensorElementParameter(
                dtype="int64",
                shape=(2,),
                values=(1, 2),
                dynamic_axes=(0,),
            ),
        },
        cache_key=("parameter-cache-test",),
    )

    first = tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(shape=(2,), dtype="int64", program=program),
    )
    second = tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(shape=(2,), dtype="int64", program=program),
    )

    assert first.tolist() == [8, 10]
    assert second.tolist() == [8, 10]
    assert tensor_calls == 3


def test_digits_tensor_generation_compiles_two_extent_independent_programs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_runtime = resolve_tensor_runtime("cpu")
    runtime = TensorRuntime(
        torch=host_runtime.torch,
        device=host_runtime.device,
        device_kind="cuda",
    )
    compile_calls: list[bool | None] = []

    monkeypatch.setattr(tensor_runtime_module, "_tensor_runtime_compile_available", lambda: True)

    def compile_kernel(kernel: Any, **kwargs: Any) -> Any:
        compile_calls.append(cast(bool | None, kwargs.get("dynamic")))
        return kernel

    monkeypatch.setattr(runtime.torch, "compile", compile_kernel)

    generator = load_digits_generator(_digits_benchmark_root)
    outcome_ids = tuple(
        outcome.id
        for outcome in generator.manifest.resolve_outcome_space().outcomes
    )

    for sample_count in (3, 5):
        generated = generator(
            shape=sample_count,
            seed=515,
            include_metadata=False,
            runtime=runtime,
            outcome_ids=outcome_ids,
        )
        fields, labels = generated.require_tensors()
        assert fields.shape[0] == sample_count
        assert labels.shape == (sample_count, len(outcome_ids))

    assert compile_calls == [None, None]


def test_digits_generator_call_tensors_match_pinned_canonical_metadata_batch() -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = load_digits_generator(_digits_benchmark_root)
    outcome_ids = tuple(
        outcome.id
        for outcome in generator.manifest.resolve_outcome_space().outcomes
    )
    component_indices = (2, 5, 7)
    observation_batch = generator(
        shape=3,
        seed=515,
        include_fields=True,
        component_indices=component_indices,
        variation_extent=0.0,
    )

    generated = generator(
        shape=3,
        seed=515,
        include_metadata=False,
        component_indices=component_indices,
        variation_extent=0.0,
        runtime=runtime,
        outcome_ids=outcome_ids,
    )
    fields, labels = generated.require_tensors()
    pure_fields = runtime.torch.tensor(
        [list(sample.require_field().values) for sample in observation_batch.samples],
        dtype=runtime.torch.float32,
        device=runtime.device,
    ).reshape((len(observation_batch.samples), *observation_batch.samples[0].require_field().shape))

    assert fields.shape == pure_fields.shape
    assert runtime.torch.allclose(fields, pure_fields, atol=2e-5)
    assert labels.cpu().tolist() == [
        [
            1.0 if outcome_id == sample.outcome_id else 0.0
            for outcome_id in outcome_ids
        ]
        for sample in observation_batch.samples
    ]


def test_digits_generator_call_tensors_reject_invalid_component_requests() -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = load_digits_generator(_digits_benchmark_root)
    outcome_ids = tuple(
        outcome.id
        for outcome in generator.manifest.resolve_outcome_space().outcomes
    )

    with pytest.raises(ObservationGenerationError, match="component index is outside"):
        generator(
            shape=1,
            seed=515,
            include_metadata=False,
            component_indices=(len(generator.formation.components),),
            runtime=runtime,
            outcome_ids=outcome_ids,
        )
