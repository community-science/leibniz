import os
from pathlib import Path
from typing import Any, cast

import pytest
from benchmark_typing import load_digits_generator

import leibniz.tensor_runtime as tensor_runtime_module
from leibniz.architectures import ArchitectureManifestDocument
from leibniz.benchmark_runner import BenchmarkRunPlan, run_benchmark
from leibniz.documents import load_object_document
from leibniz.model_operators import ExecutableModelOperator, architecture_with_input_shape
from leibniz.tensor_runtime import (
    OperationFallbackSequential,
    TensorRuntime,
    TensorRuntimeError,
    build_cross_entropy_loss,
    resolve_tensor_runtime,
    seed_runtime,
    tensor_element_compile_fallback_records,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_digits_architecture = (
    _repository_root / "tests" / "fixtures" / "architecture" / "digits_pool.json"
)
_tiny_digits_training_op_budget_per_step = 90_000


def test_tiny_digits_training_op_count_stays_within_budget() -> None:
    runtime = resolve_tensor_runtime("cpu")
    op_count = _profile_tiny_digits_training_op_count(runtime)

    assert op_count / 3 <= _tiny_digits_training_op_budget_per_step


@pytest.mark.perf_gate
@pytest.mark.parametrize("device_kind", ["cuda", "mps"])
def test_device_digits_generation_avoids_host_device_memcpy_events(
    device_kind: str,
) -> None:
    runtime = _available_runtime_or_skip(device_kind)
    profiler = runtime.torch.profiler
    activities = [profiler.ProfilerActivity.CPU]
    if runtime.device_kind == "cuda":
        activities.append(profiler.ProfilerActivity.CUDA)
    generator = load_digits_generator(_digits_benchmark_root)
    outcome_ids = tuple(
        outcome.id for outcome in generator.manifest.resolve_outcome_space().outcomes
    )

    with profiler.profile(activities=activities, acc_events=True) as profile:
        generator(
            shape=4,
            seed=515,
            include_metadata=False,
            runtime=runtime,
            outcome_ids=outcome_ids,
        ).require_tensors()

    event_names = {str(event.key).lower() for event in profile.key_averages()}
    assert not any("memcpy" in name for name in event_names)


@pytest.mark.perf_gate
@pytest.mark.parametrize("device_kind", ["cuda", "mps"])
def test_device_digits_generation_uses_compiled_tensor_programs(
    monkeypatch: pytest.MonkeyPatch,
    device_kind: str,
) -> None:
    runtime = _available_runtime_or_skip(device_kind)
    if not cast(Any, tensor_runtime_module)._tensor_runtime_compile_available(runtime):
        pytest.skip(f"torch.compile is not available for {device_kind}")
    generator = load_digits_generator(_digits_benchmark_root)
    outcome_ids = tuple(
        outcome.id for outcome in generator.manifest.resolve_outcome_space().outcomes
    )
    before = len(tensor_element_compile_fallback_records())

    monkeypatch.setenv("LEIBNIZ_REQUIRE_TENSOR_COMPILE", "1")
    generator(
        shape=4,
        seed=515,
        include_metadata=False,
        runtime=runtime,
        outcome_ids=outcome_ids,
    ).require_tensors()

    assert len(tensor_element_compile_fallback_records()) == before


@pytest.mark.perf_roofline
@pytest.mark.parametrize("device_kind", ["cuda", "mps"])
def test_manual_device_tiny_digits_training_reaches_roofline_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_kind: str,
) -> None:
    if os.environ.get("LEIBNIZ_PERF_GATES", "") != "1":
        pytest.skip("set LEIBNIZ_PERF_GATES=1 to run manual roofline gates")
    _available_runtime_or_skip(device_kind)

    def runtime_memory_budget_bytes(_runtime: TensorRuntime) -> int:
        return 2 * (1 * 28 * 28 + 10) * 4 * 8

    def runtime_used_memory_bytes(_runtime: TensorRuntime) -> int:
        return 0

    monkeypatch.setattr(
        "leibniz.benchmark_runner._runtime_memory_budget_bytes",
        runtime_memory_budget_bytes,
    )
    monkeypatch.setattr(
        "leibniz.benchmark_runner._runtime_used_memory_bytes",
        runtime_used_memory_bytes,
    )

    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            seed=101,
            train_steps=3,
            tensor_device=cast(Any, device_kind),
        )
    )
    record = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )
    throughput = cast(dict[str, object], record["throughput"])
    comparison = cast(dict[str, object], throughput["roofline_comparison"])

    assert cast(float, comparison["training_fraction_of_roofline"]) >= 0.01


def _profile_tiny_digits_training_op_count(runtime: TensorRuntime) -> int:
    seed_runtime(runtime, seed=101)
    generator = load_digits_generator(_digits_benchmark_root)
    outcome_ids = tuple(
        outcome.id for outcome in generator.manifest.resolve_outcome_space().outcomes
    )
    architecture = ArchitectureManifestDocument.from_bytes(
        _digits_architecture.read_bytes()
    ).manifest
    first_batch = generator(
        shape=2,
        seed=101,
        include_metadata=False,
        runtime=runtime,
        outcome_ids=outcome_ids,
    )
    fields, _labels = first_batch.require_tensors()
    executable = ExecutableModelOperator(
        architecture_with_input_shape(architecture, tuple(fields.shape[1:]))
    )
    module = OperationFallbackSequential(
        runtime=runtime,
        operations=executable.operation_modules(),
    )
    loss_function = build_cross_entropy_loss(runtime)
    profiler = runtime.torch.profiler

    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        acc_events=True,
    ) as profile:
        for step in range(3):
            batch = generator(
                shape=2,
                seed=101 + step,
                include_metadata=False,
                runtime=runtime,
                outcome_ids=outcome_ids,
            )
            fields, labels = batch.require_tensors()
            module.zero_grad(set_to_none=True)
            loss = loss_function(module(fields), labels)
            loss.backward()

    return sum(int(event.count) for event in profile.key_averages())


def _available_runtime_or_skip(device_kind: str) -> TensorRuntime:
    try:
        return resolve_tensor_runtime(cast(Any, device_kind))
    except TensorRuntimeError as error:
        pytest.skip(str(error))
