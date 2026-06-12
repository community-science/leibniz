import os
from pathlib import Path
from typing import Any, cast

import pytest
from benchmark_typing import load_digits_generator

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
_digits_perf_conv_architecture = (
    _repository_root / "tests" / "fixtures" / "architecture" / "digits_perf_conv.json"
)
# Measured 2,548 dispatched ops per step on CPU (2026-06-11); budget is
# measured + ~50% headroom so a graph-size regression fails loudly.
_tiny_digits_training_op_budget_per_step = 4_000
# Per-batch parameter uploads (seed scalar and sample indices per program) are
# small constant-count async HtoD copies, not hot-path data movement. The gate
# bounds them per step and forbids DtoH readbacks entirely.
_training_step_htod_copy_budget = 8


def test_tiny_digits_training_op_count_stays_within_budget() -> None:
    runtime = resolve_tensor_runtime("cpu")
    op_count = _profile_tiny_digits_training_op_count(runtime)

    assert op_count / 3 <= _tiny_digits_training_op_budget_per_step


@pytest.mark.perf_gate
@pytest.mark.parametrize("device_kind", ["cuda", "mps"])
def test_device_training_steps_avoid_host_readback_and_bound_uploads(
    device_kind: str,
) -> None:
    runtime = _available_runtime_or_skip(device_kind)
    profiler = runtime.torch.profiler
    activities = [profiler.ProfilerActivity.CPU]
    if runtime.device_kind == "cuda":
        activities.append(profiler.ProfilerActivity.CUDA)
    step = _tiny_digits_training_step_callback(runtime)
    step(0)

    measured_steps = 3
    with profiler.profile(activities=activities, acc_events=True) as profile:
        for offset in range(1, measured_steps + 1):
            step(offset)

    device_to_host_count = 0
    host_to_device_count = 0
    for event in profile.key_averages():
        name = str(event.key).lower()
        if "memcpy dtoh" in name:
            device_to_host_count += int(event.count)
        if "memcpy htod" in name:
            host_to_device_count += int(event.count)
    assert device_to_host_count == 0
    assert host_to_device_count <= _training_step_htod_copy_budget * measured_steps


@pytest.mark.perf_gate
@pytest.mark.parametrize("device_kind", ["cuda", "mps"])
def test_device_digits_generation_uses_compiled_tensor_programs(
    monkeypatch: pytest.MonkeyPatch,
    device_kind: str,
) -> None:
    runtime = _available_runtime_or_skip(device_kind)
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
@pytest.mark.parametrize("device_kind", ["cpu", "cuda", "mps"])
def test_manual_device_training_reaches_roofline_floor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    device_kind: str,
) -> None:
    """Require >= 1% of the device's calibrated roofline for steady-state training.

    The measured run must be compute-dense (convolutional architecture, large
    batch) and warm: an identical warmup run first populates the process-wide
    kernel compile and parameter caches, so the measured run reflects
    steady-state training rather than one-time compilation.
    """

    if os.environ.get("LEIBNIZ_PERF_GATES", "") != "1":
        pytest.skip("set LEIBNIZ_PERF_GATES=1 to run manual roofline gates")
    _available_runtime_or_skip(device_kind)

    def runtime_memory_budget_bytes(_runtime: TensorRuntime) -> int:
        return 512 * (1 * 28 * 28 + 10) * 4 * 8

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

    def measured_training_fraction(results_root: Path) -> float:
        summary = run_benchmark(
            BenchmarkRunPlan(
                architecture_path=_digits_perf_conv_architecture,
                benchmark_root=_digits_benchmark_root,
                results_root=results_root,
                seed=101,
                train_steps=8,
                tensor_device=cast(Any, device_kind),
            )
        )
        record = load_object_document(
            summary.training_summary_path.read_bytes(),
            description="training summary",
        )
        throughput = cast(dict[str, object], record["throughput"])
        comparison = cast(dict[str, object], throughput["roofline_comparison"])
        return cast(float, comparison["training_fraction_of_roofline"])

    measured_training_fraction(tmp_path / "warmup-results")
    fraction = measured_training_fraction(tmp_path / "results")

    assert fraction >= 0.01


def _tiny_digits_training_step_callback(runtime: TensorRuntime) -> Any:
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

    def training_step(offset: int) -> None:
        batch = generator(
            shape=2,
            seed=101 + offset,
            include_metadata=False,
            runtime=runtime,
            outcome_ids=outcome_ids,
        )
        fields, labels = batch.require_tensors()
        module.zero_grad(set_to_none=True)
        loss = loss_function(module(fields), labels)
        loss.backward()

    return training_step


def _profile_tiny_digits_training_op_count(runtime: TensorRuntime) -> int:
    step = _tiny_digits_training_step_callback(runtime)
    profiler = runtime.torch.profiler

    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU],
        acc_events=True,
    ) as profile:
        for offset in range(3):
            step(offset)

    return sum(int(event.count) for event in profile.key_averages())


def _available_runtime_or_skip(device_kind: str) -> TensorRuntime:
    try:
        return resolve_tensor_runtime(cast(Any, device_kind))
    except TensorRuntimeError as error:
        pytest.skip(str(error))
