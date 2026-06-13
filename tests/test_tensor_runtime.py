import importlib
import math
import sys
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from benchmark_typing import load_digits_generator

from leibniz import tensor_runtime as tensor_runtime_module
from leibniz.architectures import ArchitectureManifest
from leibniz.target_contracts import (
    BaselinePredictor,
    CompetenceFunctional,
    TargetContract,
)
from leibniz.tensor_runtime import (
    OperationFallbackSequential,
    TensorBatchProgram,
    TensorElementParameter,
    TensorElementRecipe,
    TensorRuntime,
    TensorRuntimeError,
    architecture_supported_by_tensor_runtime,
    architecture_tensor_runtime_issue,
    build_loss,
    build_mse_loss,
    build_optimizer,
    build_relative_l2_loss,
    optimizer_step,
    resolve_tensor_runtime,
    runtime_roofline_record,
    softmax_target_masses,
    tensor_element_compile_fallback_records,
    tensor_runtime_construct_tensor,
    tensor_runtime_device_kinds,
    tensor_runtime_shape_element_count,
    tensor_value_to_host_values,
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


def test_tensor_runtime_shape_element_count_declares_shape_convention() -> None:
    assert tensor_runtime_shape_element_count((2, 3, 4)) == 24


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


def test_build_loss_dispatches_target_contract_losses() -> None:
    runtime = resolve_tensor_runtime("cpu")
    finite = TargetContract.finite_outcome(("a", "b"))
    field = TargetContract(
        kind="field-valued",
        outcome_ids=None,
        loss_id="mse",
        competence=CompetenceFunctional(
            kind="mass-within-resolution",
            parameters={"residual_operator_id": "op"},
        ),
        baseline=BaselinePredictor(kind="zero-field"),
    )

    assert build_loss(runtime, finite).__class__.__name__ == "CrossEntropyLoss"
    assert build_loss(runtime, field).__class__.__name__ == "MSELoss"


def test_mse_loss_matches_closed_form() -> None:
    runtime = resolve_tensor_runtime("cpu")
    loss = build_mse_loss(runtime)
    predictions = runtime.torch.tensor([1.0, 3.0])
    targets = runtime.torch.tensor([2.0, 1.0])

    assert float(loss(predictions, targets)) == 2.5


def test_relative_l2_loss_matches_closed_form_and_handles_zero_target() -> None:
    runtime = resolve_tensor_runtime("cpu")
    loss = build_relative_l2_loss(runtime)
    predictions = runtime.torch.tensor([3.0, 4.0])
    targets = runtime.torch.tensor([0.0, 0.0])
    nonzero_targets = runtime.torch.tensor([0.0, 4.0])

    assert math.isclose(float(loss(predictions, nonzero_targets)), 0.75)
    assert math.isclose(float(loss(predictions, targets)), 5e12, rel_tol=1e-6)


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


@pytest.mark.parametrize(
    ("initial", "steps", "loss_kind", "target", "maximum_loss"),
    [
        ((2.0,), 48, "shifted-quadratic", (0.25,), 1e-6),
        ((-3.0, 2.0), 96, "ill-conditioned-quadratic", (1.0, -0.5), 1e-4),
        ((-1.5,), 96, "quartic-bowl", (0.75,), 1e-5),
    ],
)
def test_loss_search_optimizer_reaches_known_simple_optima_and_matches_adam(
    initial: tuple[float, ...],
    steps: int,
    loss_kind: str,
    target: tuple[float, ...],
    maximum_loss: float,
) -> None:
    runtime = resolve_tensor_runtime("cpu")
    loss_search = _optimize_simple_objective(
        runtime=runtime,
        optimizer_name="loss-search",
        learning_rate=None,
        initial=initial,
        steps=steps,
        loss_kind=loss_kind,
    )
    adam = _optimize_simple_objective(
        runtime=runtime,
        optimizer_name="adam",
        learning_rate=1e-3,
        initial=initial,
        steps=steps,
        loss_kind=loss_kind,
    )

    assert loss_search.loss <= maximum_loss
    assert all(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-2)
        for actual, expected in zip(loss_search.parameters, target, strict=True)
    )
    assert loss_search.loss <= adam.loss + 1e-8


def test_loss_search_optimizer_matches_adam_on_known_logistic_fixture() -> None:
    runtime = resolve_tensor_runtime("cpu")
    loss_search = _optimize_logistic_fixture(
        runtime=runtime,
        optimizer_name="loss-search",
        learning_rate=None,
        steps=128,
    )
    adam = _optimize_logistic_fixture(
        runtime=runtime,
        optimizer_name="adam",
        learning_rate=1e-3,
        steps=128,
    )

    assert loss_search.loss <= 0.05
    assert loss_search.accuracy == 1.0
    assert loss_search.loss <= adam.loss + 1e-8


def test_loss_search_optimizer_matches_adam_on_alternating_batch_fixture() -> None:
    runtime = resolve_tensor_runtime("cpu")
    loss_search = _optimize_alternating_batch_fixture(
        runtime=runtime,
        optimizer_name="loss-search",
        learning_rate=None,
        steps=96,
    )
    adam = _optimize_alternating_batch_fixture(
        runtime=runtime,
        optimizer_name="adam",
        learning_rate=1e-3,
        steps=96,
    )

    assert loss_search.loss <= 0.05
    assert loss_search.loss <= adam.loss + 1e-8


def test_loss_search_optimizer_updates_moments_on_rejected_steps() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    parameter = torch.nn.Parameter(torch.tensor([1.0], device=runtime.device))
    optimizer = build_optimizer(
        runtime,
        name="loss-search",
        parameters=[parameter],
        learning_rate=None,
    )
    optimizer._armijo_sufficient_decrease = 1e12

    def closure() -> Any:
        optimizer.zero_grad(set_to_none=True)
        loss = parameter.pow(2).sum()
        loss.backward()
        return loss

    baseline_parameter = float(parameter.detach()[0])
    baseline_loss = float(closure().detach())
    optimizer_step(runtime, optimizer, closure)

    assert float(parameter.detach()[0]) == baseline_parameter
    assert float(closure().detach()) == baseline_loss
    assert optimizer.state[parameter]["step"] == 1
    assert optimizer._accepted_step_size >= optimizer._minimum_step_size


def test_loss_search_optimizer_uses_raw_gradient_when_momentum_is_not_descent() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    parameter = torch.nn.Parameter(torch.tensor([1.0], device=runtime.device))
    optimizer = build_optimizer(
        runtime,
        name="loss-search",
        parameters=[parameter],
        learning_rate=None,
    )
    optimizer.state[parameter] = {
        "step": 1,
        "exp_avg": torch.tensor([-10.0], device=runtime.device),
        "exp_avg_sq": torch.tensor([1.0], device=runtime.device),
    }

    def closure() -> Any:
        optimizer.zero_grad(set_to_none=True)
        loss = parameter.pow(2).sum()
        loss.backward()
        return loss

    baseline_loss = float(closure().detach())
    optimizer_step(runtime, optimizer, closure)

    assert float(closure().detach()) < baseline_loss


@dataclass(frozen=True, slots=True)
class _OptimizationResult:
    loss: float
    parameters: tuple[float, ...]
    accuracy: float | None = None


def _optimize_simple_objective(
    *,
    runtime: TensorRuntime,
    optimizer_name: str,
    learning_rate: float | None,
    initial: tuple[float, ...],
    steps: int,
    loss_kind: str,
) -> _OptimizationResult:
    torch = runtime.torch
    parameter = torch.nn.Parameter(
        torch.tensor(initial, dtype=torch.float32, device=runtime.device)
    )
    optimizer = build_optimizer(
        runtime,
        name=optimizer_name,
        parameters=[parameter],
        learning_rate=learning_rate,
    )

    def closure() -> Any:
        optimizer.zero_grad(set_to_none=True)
        if loss_kind == "shifted-quadratic":
            loss = (parameter[0] - 0.25).pow(2)
        elif loss_kind == "ill-conditioned-quadratic":
            loss = (parameter[0] - 1.0).pow(2) + 25.0 * (parameter[1] + 0.5).pow(2)
        elif loss_kind == "quartic-bowl":
            loss = (parameter[0] - 0.75).pow(4) + 0.1 * (parameter[0] - 0.75).pow(2)
        else:  # pragma: no cover - parametrization guard
            raise AssertionError(f"unknown loss fixture: {loss_kind}")
        loss.backward()
        return loss

    for _step in range(steps):
        optimizer_step(runtime, optimizer, closure)
    return _OptimizationResult(
        loss=float(closure().detach()),
        parameters=tuple(float(value) for value in parameter.detach().tolist()),
    )


def _optimize_logistic_fixture(
    *,
    runtime: TensorRuntime,
    optimizer_name: str,
    learning_rate: float | None,
    steps: int,
) -> _OptimizationResult:
    torch = runtime.torch
    features = torch.tensor(
        [
            [-2.0, -1.0],
            [-1.5, -1.0],
            [-1.0, -2.0],
            [1.0, 2.0],
            [1.5, 1.0],
            [2.0, 1.0],
        ],
        dtype=torch.float32,
        device=runtime.device,
    )
    labels = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=runtime.device)
    parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.float32, device=runtime.device))
    optimizer = build_optimizer(
        runtime,
        name=optimizer_name,
        parameters=[parameter],
        learning_rate=learning_rate,
    )

    def closure() -> Any:
        optimizer.zero_grad(set_to_none=True)
        logits = features @ parameter[:2] + parameter[2]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        return loss

    for _step in range(steps):
        optimizer_step(runtime, optimizer, closure)
    with torch.no_grad():
        logits = features @ parameter[:2] + parameter[2]
        predictions = (logits >= 0).to(labels.dtype)
        accuracy = float((predictions == labels).to(torch.float32).mean())
    return _OptimizationResult(
        loss=float(closure().detach()),
        parameters=tuple(float(value) for value in parameter.detach().tolist()),
        accuracy=accuracy,
    )


def _optimize_alternating_batch_fixture(
    *,
    runtime: TensorRuntime,
    optimizer_name: str,
    learning_rate: float | None,
    steps: int,
) -> _OptimizationResult:
    torch = runtime.torch
    parameter = torch.nn.Parameter(
        torch.tensor([-2.0, 2.0], dtype=torch.float32, device=runtime.device)
    )
    centers = (
        torch.tensor([1.0, -1.0], dtype=torch.float32, device=runtime.device),
        torch.tensor([1.2, -0.8], dtype=torch.float32, device=runtime.device),
    )
    optimizer = build_optimizer(
        runtime,
        name=optimizer_name,
        parameters=[parameter],
        learning_rate=learning_rate,
    )
    current_center = centers[0]

    def closure() -> Any:
        optimizer.zero_grad(set_to_none=True)
        loss = (parameter - current_center).pow(2).mean()
        loss.backward()
        return loss

    for step in range(steps):
        current_center = centers[step % len(centers)]
        optimizer_step(runtime, optimizer, closure)
    with torch.no_grad():
        average_center = sum(centers) / len(centers)
        evaluation_loss = (parameter - average_center).pow(2).mean()
    return _OptimizationResult(
        loss=float(evaluation_loss.detach()),
        parameters=tuple(float(value) for value in parameter.detach().tolist()),
    )


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
    assert record["compute_calibration_chosen_matrix_size"] in {512, 1024, 2048, 4096}
    points = cast(tuple[object, ...] | list[object], record["compute_calibration_points"])
    assert len(points) >= 1
    assert record["compute_calibration_matrix_size"] == record[
        "compute_calibration_chosen_matrix_size"
    ]
    first_point = cast(dict[str, object], points[0])
    assert cast(float, record["peak_compute_per_second"]) >= cast(
        float,
        first_point["peak_compute_per_second"],
    )


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


@pytest.mark.parametrize("device_kind", ["cuda", "mps"])
def test_tensor_element_compile_cache_is_not_extent_dependent(
    monkeypatch: pytest.MonkeyPatch,
    device_kind: str,
) -> None:
    runtime, compile_calls = _compile_counting_runtime(
        monkeypatch,
        device_kind=cast(Any, device_kind),
    )

    def element_function(coordinates: tuple[Any, ...]) -> Any:
        return coordinates[0]

    program = TensorBatchProgram(
        kernel=element_function,
        parameters={},
        cache_key=("extent-independent-test", device_kind, id(element_function)),
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


def test_tensor_element_constructs_float64_on_cpu_eager_path() -> None:
    runtime = resolve_tensor_runtime("cpu")

    def element_function(coordinates: tuple[Any, ...]) -> Any:
        return coordinates[0].to(dtype=runtime.torch.float64) + 0.25

    program = TensorBatchProgram(
        kernel=element_function,
        parameters={},
        compile=False,
        cache_key=("float64-eager-test", id(element_function)),
    )

    values = tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(shape=(3,), dtype="float64", program=program),
    )

    assert values.dtype == runtime.torch.float64
    assert values.tolist() == [0.25, 1.25, 2.25]


def test_tensor_element_constructs_float64_with_compiled_scalar_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, compile_calls = _compile_counting_runtime(monkeypatch, device_kind="cuda")

    def element_function(coordinates: tuple[Any, ...], *, offset: Any) -> Any:
        return coordinates[0].to(dtype=offset.dtype) + offset

    program = TensorBatchProgram(
        kernel=element_function,
        parameters={
            "offset": TensorElementParameter(
                dtype="float64",
                shape=(),
                values=(0.5,),
            ),
        },
        cache_key=("float64-compiled-parameter-test", id(element_function)),
    )

    values = tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(shape=(3,), dtype="float64", program=program),
    )

    assert compile_calls == [None]
    assert values.dtype == runtime.torch.float64
    assert values.tolist() == [0.5, 1.5, 2.5]


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
        *,
        constant: Any,
        dynamic: Any,
    ) -> Any:
        return coordinates[0] + constant[0] + dynamic[coordinates[0]]

    program = TensorBatchProgram(
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
    assert tensor_calls == 2
    cache = cast(Any, tensor_runtime_module)._tensor_element_parameter_cache
    assert any(key[3] == "dynamic" for key in cache)


def test_tensor_element_parameter_cache_reuses_dynamic_axis_parameters() -> None:
    runtime = resolve_tensor_runtime("cpu")
    cache = cast(Any, tensor_runtime_module)._tensor_element_parameter_cache
    cache.clear()

    def element_function(coordinates: tuple[Any, ...], *, dynamic: Any) -> Any:
        return dynamic[coordinates[0]]

    parameter = TensorElementParameter(
        dtype="int64",
        shape=(2,),
        values=(1, 2),
        dynamic_axes=(0,),
    )
    program = TensorBatchProgram(
        kernel=element_function,
        parameters={"dynamic": parameter},
        cache_key=("dynamic-parameter-cache-test",),
    )

    first = tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(shape=(2,), dtype="int64", program=program),
    )
    second = tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(shape=(2,), dtype="int64", program=program),
    )

    assert first.tolist() == [1, 2]
    assert second.tolist() == [1, 2]
    assert len(cache) == 1


def test_digits_tensor_generation_compiles_two_extent_independent_programs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, compile_calls = _compile_counting_runtime(monkeypatch, device_kind="cuda")
    generator = load_digits_generator(_digits_benchmark_root)
    digits_benchmark = cast(Any, sys.modules[generator.__class__.__module__])
    outcome_ids = tuple(
        outcome.id
        for outcome in generator.manifest.resolve_outcome_space().outcomes
    )
    transform_value_lengths: list[int] = []
    original_sample_transform_values = digits_benchmark._digits_sample_transform_values

    def sample_transform_values(
        *,
        canvas_side: int,
        sample_addresses: tuple[int, ...],
        digit_count: int,
    ) -> tuple[float, ...]:
        values = original_sample_transform_values(
            canvas_side=canvas_side,
            sample_addresses=sample_addresses,
            digit_count=digit_count,
        )
        transform_value_lengths.append(len(values))
        return values

    monkeypatch.setattr(
        digits_benchmark,
        "_digits_sample_transform_values",
        sample_transform_values,
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
    assert transform_value_lengths == [9, 15]


def test_digits_generator_call_tensors_match_metadata_batch() -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = load_digits_generator(_digits_benchmark_root)
    outcome_ids = tuple(
        outcome.id
        for outcome in generator.manifest.resolve_outcome_space().outcomes
    )
    observation_batch = generator(
        shape=3,
        seed=515,
        include_fields=True,
        variation_extent=0.0,
    )

    generated = generator(
        shape=3,
        seed=515,
        include_metadata=False,
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


def test_digits_tensor_generation_accepts_scalar_sample_shape() -> None:
    runtime = resolve_tensor_runtime("cpu")
    generator = load_digits_generator(_digits_benchmark_root)
    outcome_ids = tuple(
        outcome.id
        for outcome in generator.manifest.resolve_outcome_space().outcomes
    )

    generated = generator(
        seed=515,
        include_metadata=False,
        runtime=runtime,
        outcome_ids=outcome_ids,
    )
    fields, labels = generated.require_tensors()

    assert tuple(fields.shape[:1]) == (1,)
    assert labels.shape == (len(outcome_ids),)
    assert labels.sum().item() == 1.0


def test_tensor_batch_program_chunks_only_leading_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = resolve_tensor_runtime("cpu")
    monkeypatch.setattr(tensor_runtime_module, "_tensor_element_tile_size", 6)
    observed_shapes: list[tuple[tuple[int, ...], ...]] = []

    def element_function(coordinates: tuple[Any, ...]) -> Any:
        observed_shapes.append(tuple(tuple(coordinate.shape) for coordinate in coordinates))
        sample, row, column = coordinates
        return (
            sample.reshape((-1, 1, 1)) * 100
            + row.reshape((1, -1, 1)) * 10
            + column.reshape((1, 1, -1))
        )

    program = TensorBatchProgram(
        kernel=element_function,
        parameters={},
        compile=False,
        cache_key=("chunk-leading-axis-test",),
    )

    values = tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(shape=(5, 2, 3), dtype="int64", program=program),
    )

    assert observed_shapes == [
        ((1,), (2,), (3,)),
        ((1,), (2,), (3,)),
        ((1,), (2,), (3,)),
        ((1,), (2,), (3,)),
        ((1,), (2,), (3,)),
    ]
    assert values.tolist() == [
        [[0, 1, 2], [10, 11, 12]],
        [[100, 101, 102], [110, 111, 112]],
        [[200, 201, 202], [210, 211, 212]],
        [[300, 301, 302], [310, 311, 312]],
        [[400, 401, 402], [410, 411, 412]],
    ]


def test_tensor_batch_program_chunk_boundary_matches_unchunked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = resolve_tensor_runtime("cpu")

    def element_function(coordinates: tuple[Any, ...]) -> Any:
        sample, row, column = coordinates
        return (
            sample.reshape((-1, 1, 1)) * 100
            + row.reshape((1, -1, 1)) * 10
            + column.reshape((1, 1, -1))
        )

    program = TensorBatchProgram(
        kernel=element_function,
        parameters={},
        compile=False,
        cache_key=("chunk-boundary-test",),
    )
    recipe = TensorElementRecipe(shape=(5, 2, 3), dtype="int64", program=program)

    monkeypatch.setattr(tensor_runtime_module, "_tensor_element_tile_size", 6)
    chunked = tensor_runtime_construct_tensor(runtime, recipe=recipe)
    monkeypatch.setattr(tensor_runtime_module, "_tensor_element_tile_size", 1024)
    unchunked = tensor_runtime_construct_tensor(runtime, recipe=recipe)

    assert runtime.torch.equal(chunked, unchunked)


def test_tensor_element_compile_failure_records_loud_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _compile_failing_runtime(monkeypatch, reason="inductor exploded")

    def element_function(coordinates: tuple[Any, ...]) -> Any:
        return coordinates[0]

    program = TensorBatchProgram(
        kernel=element_function,
        parameters={},
        cache_key=("loud-fallback-test", id(element_function)),
    )

    values = tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(shape=(2,), dtype="int64", program=program),
    )
    tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(shape=(2,), dtype="int64", program=program),
    )

    assert values.tolist() == [0, 1]
    matching = [
        record
        for record in tensor_element_compile_fallback_records()
        if record["program"] == str(program.cache_key)
    ]
    assert len(matching) == 1
    assert matching[0]["kind"] == "tensor-element-compile-fallback"
    assert matching[0]["tensor_device"] == "cuda"
    assert "inductor exploded" in str(matching[0]["reason"])
    assert matching[0]["constructions"] == 2


def test_operation_fallback_raises_when_device_residency_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_runtime = resolve_tensor_runtime("cpu")
    runtime = TensorRuntime(
        torch=host_runtime.torch,
        device=host_runtime.device,
        device_kind="cuda",
    )

    class FailingOperation(host_runtime.torch.nn.Module):
        def forward(self, value: Any) -> Any:
            _ = value
            raise RuntimeError("backend op unavailable")

    module = OperationFallbackSequential(
        runtime=runtime,
        operations=[FailingOperation()],
    )
    monkeypatch.setenv("LEIBNIZ_REQUIRE_DEVICE_RESIDENCY", "1")

    with pytest.raises(
        TensorRuntimeError,
        match="LEIBNIZ_REQUIRE_DEVICE_RESIDENCY blocked CPU fallback for operation 0",
    ):
        module(host_runtime.torch.ones((1,), dtype=host_runtime.torch.float32))

    assert module.operation_fallback_records() == ()


def test_tensor_element_compile_fallback_raises_in_strict_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _compile_failing_runtime(monkeypatch, reason="inductor exploded")
    monkeypatch.setenv("LEIBNIZ_REQUIRE_TENSOR_COMPILE", "1")

    def element_function(coordinates: tuple[Any, ...]) -> Any:
        return coordinates[0]

    program = TensorBatchProgram(
        kernel=element_function,
        parameters={},
        cache_key=("strict-fallback-test", id(element_function)),
    )

    with pytest.raises(TensorRuntimeError, match="LEIBNIZ_REQUIRE_TENSOR_COMPILE"):
        tensor_runtime_construct_tensor(
            runtime,
            recipe=TensorElementRecipe(shape=(2,), dtype="int64", program=program),
        )


def test_strict_mode_allows_declared_eager_programs_and_cpu_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEIBNIZ_REQUIRE_TENSOR_COMPILE", "1")

    def element_function(coordinates: tuple[Any, ...]) -> Any:
        return coordinates[0]

    declared_eager = TensorBatchProgram(
        kernel=element_function,
        parameters={},
        compile=False,
        cache_key=("strict-declared-eager-test", id(element_function)),
    )
    cuda_like_runtime = _compile_failing_runtime(monkeypatch, reason="unused")
    eager_values = tensor_runtime_construct_tensor(
        cuda_like_runtime,
        recipe=TensorElementRecipe(shape=(2,), dtype="int64", program=declared_eager),
    )

    compiled_program = TensorBatchProgram(
        kernel=element_function,
        parameters={},
        cache_key=("strict-cpu-test", id(element_function)),
    )
    cpu_values = tensor_runtime_construct_tensor(
        resolve_tensor_runtime("cpu"),
        recipe=TensorElementRecipe(shape=(2,), dtype="int64", program=compiled_program),
    )

    assert eager_values.tolist() == [0, 1]
    assert cpu_values.tolist() == [0, 1]
    fallback_programs = {
        record["program"] for record in tensor_element_compile_fallback_records()
    }
    assert str(declared_eager.cache_key) not in fallback_programs
    assert str(compiled_program.cache_key) not in fallback_programs


def _compile_failing_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reason: str,
) -> TensorRuntime:
    host_runtime = resolve_tensor_runtime("cpu")
    runtime = TensorRuntime(
        torch=host_runtime.torch,
        device=host_runtime.device,
        device_kind="cuda",
    )

    def compile_available(_runtime: TensorRuntime) -> bool:
        return True

    monkeypatch.setattr(
        tensor_runtime_module,
        "_tensor_runtime_compile_available",
        compile_available,
    )

    def failing_compile(kernel: Any, **kwargs: Any) -> Any:
        _ = kernel
        _ = kwargs
        raise RuntimeError(reason)

    monkeypatch.setattr(runtime.torch, "compile", failing_compile)
    return runtime


def _compile_counting_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    device_kind: Any,
) -> tuple[TensorRuntime, list[bool | None]]:
    host_runtime = resolve_tensor_runtime("cpu")
    runtime = TensorRuntime(
        torch=host_runtime.torch,
        device=host_runtime.device,
        device_kind=device_kind,
    )
    compile_calls: list[bool | None] = []
    def compile_available(_runtime: TensorRuntime) -> bool:
        return True

    monkeypatch.setattr(
        tensor_runtime_module,
        "_tensor_runtime_compile_available",
        compile_available,
    )

    def compile_kernel(kernel: Any, **kwargs: Any) -> Any:
        compile_calls.append(cast(bool | None, kwargs.get("dynamic")))
        return kernel

    monkeypatch.setattr(runtime.torch, "compile", compile_kernel)
    return runtime, compile_calls


def test_compile_availability_follows_torch_triton_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host_runtime = resolve_tensor_runtime("cpu")
    cuda_like_runtime = TensorRuntime(
        torch=host_runtime.torch,
        device=host_runtime.device,
        device_kind="cuda",
    )
    triton_support = importlib.import_module("torch.utils._triton")
    compile_available = cast(Any, tensor_runtime_module)._tensor_runtime_compile_available

    monkeypatch.setattr(triton_support, "has_triton", lambda: True)
    assert compile_available(cuda_like_runtime) is True

    monkeypatch.setattr(triton_support, "has_triton", lambda: False)
    assert compile_available(cuda_like_runtime) is False


def test_tensor_value_to_host_values_flattens_to_floats() -> None:
    runtime = resolve_tensor_runtime("cpu")
    tensor = runtime.torch.arange(6, dtype=runtime.torch.float32).reshape((2, 3))

    values = tensor_value_to_host_values(tensor)

    assert values == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert all(type(value) is float for value in values)


def test_parameter_values_key_cache_skips_per_batch_scalars() -> None:
    cache = cast(Any, tensor_runtime_module)._tensor_element_parameter_values_key_cache
    values_key = cast(Any, tensor_runtime_module)._tensor_element_parameter_values_key
    minimum_size = cast(
        int,
        cast(Any, tensor_runtime_module)._tensor_element_parameter_values_key_cache_minimum_size,
    )
    cache.clear()

    small = TensorElementParameter(dtype="int64", shape=(1,), values=(7,))
    large_values = tuple(range(minimum_size))
    large = TensorElementParameter(
        dtype="int64",
        shape=(minimum_size,),
        values=large_values,
    )

    small_key = values_key(small)
    large_key = values_key(large)

    assert small_key
    assert large_key
    assert len(cache) == 1
    assert next(iter(cache.values()))[0] is large_values


def test_float64_parameter_values_key_uses_double_array() -> None:
    values_key = cast(Any, tensor_runtime_module)._tensor_element_parameter_values_key
    parameter = TensorElementParameter(dtype="float64", shape=(2,), values=(0.25, 1.5))

    assert values_key(parameter) == array("d", (0.25, 1.5)).tobytes()


def test_parameter_tensor_cache_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = resolve_tensor_runtime("cpu")
    cache = cast(Any, tensor_runtime_module)._tensor_element_parameter_cache
    cache.clear()
    monkeypatch.setattr(
        tensor_runtime_module,
        "_tensor_element_parameter_cache_capacity",
        4,
    )

    def element_function(coordinates: tuple[Any, ...], *, offset: Any) -> Any:
        return coordinates[0] + offset

    for seed in range(10):
        program = TensorBatchProgram(
            kernel=element_function,
            parameters={
                "offset": TensorElementParameter(
                    dtype="int64",
                    shape=(),
                    values=(seed,),
                ),
            },
            compile=False,
            cache_key=("parameter-cache-bound-test",),
        )
        values = tensor_runtime_construct_tensor(
            runtime,
            recipe=TensorElementRecipe(shape=(2,), dtype="int64", program=program),
        )
        assert values.tolist() == [seed, seed + 1]

    assert len(cache) <= 4
