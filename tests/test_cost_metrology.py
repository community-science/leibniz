from typing import Any

import pytest

from leibniz.cost_metrology import (
    TENSOR_RUNTIME_COST_MODEL_ID,
    CostMeasurement,
    CostMeter,
    CostMetrologyError,
    estimate_operation_stream_cost,
    estimate_program_cost,
    measure_program_cost,
)
from leibniz.tensor_runtime import (
    TensorRuntimeError,
    TensorRuntimeOperationRecord,
    TensorRuntimeTensorSpec,
    resolve_tensor_runtime,
)


def test_measure_program_cost_counts_matmul_formula() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    left = torch.randn(2, 3, device=runtime.device)
    right = torch.randn(3, 4, device=runtime.device)

    def program(a: Any, b: Any) -> Any:
        return a @ b

    measurement = measure_program_cost(runtime, program, (left, right), strict=True)

    assert measurement.cost_model_id == TENSOR_RUNTIME_COST_MODEL_ID
    assert measurement.abstract_flops == 2 * 2 * 4 * 3
    assert measurement.operation_count == len(measurement.operation_trace)
    assert measurement.execution_mode == "measured"
    assert measurement.operation_stream_source == "runtime-executed"
    assert measurement.operations_executed is True
    assert measurement.operation_trace[0].name == "aten.mm.default"
    assert measurement.operation_trace[0].input_tensors[0].shape == (2, 3)
    assert measurement.operation_trace[0].output_tensors[0].shape == (2, 4)
    assert measurement.per_op == (
        measurement.per_op[0].__class__(
            name="aten.mm.default",
            calls=1,
            abstract_flops=48,
            output_elements=8,
        ),
    )
    assert measurement.unmodeled_operations == ()


def test_estimate_operation_stream_cost_uses_unified_formula_table() -> None:
    runtime = resolve_tensor_runtime("cpu")
    operation = TensorRuntimeOperationRecord(
        name="aten.mm.default",
        arguments=(),
        keyword_arguments=(),
        input_tensors=(
            TensorRuntimeTensorSpec(shape=(2, 3), dtype="torch.float32"),
            TensorRuntimeTensorSpec(shape=(3, 4), dtype="torch.float32"),
        ),
        output_tensors=(TensorRuntimeTensorSpec(shape=(2, 4), dtype="torch.float32"),),
    )

    measurement = estimate_operation_stream_cost(runtime, (operation,), strict=True)

    assert measurement.abstract_flops == 2 * 2 * 4 * 3
    assert measurement.execution_mode == "dry-run"
    assert measurement.operation_stream_source == "runtime-dry-run"
    assert measurement.operations_executed is False
    assert measurement.per_op[0].name == "aten.mm.default"


def test_estimate_program_cost_projects_without_executing_ops() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    left = torch.randn(2, 3, device=runtime.device)
    right = torch.randn(3, 4, device=runtime.device)

    def program(a: Any, b: Any) -> Any:
        return (a @ b).relu()

    measurement = estimate_program_cost(runtime, program, (left, right), strict=True)

    assert measurement.abstract_flops == 48 + 8
    assert measurement.execution_mode == "dry-run"
    assert measurement.operation_stream_source == "runtime-dry-run"
    assert measurement.operations_executed is False
    assert measurement.wall_seconds == 0.0
    assert [record.name for record in measurement.per_op] == [
        "aten.mm.default",
        "aten.relu.default",
    ]


def test_measure_program_cost_counts_fft_formula() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    values = torch.randn(8, dtype=torch.complex64, device=runtime.device)

    measurement = measure_program_cost(runtime, torch.fft.fft, values, strict=True)

    assert measurement.abstract_flops == 5 * 8 * 3
    assert measurement.per_op[0].name == "aten._fft_c2c.default"


def test_measure_program_cost_counts_conv2d_formula() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    values = torch.randn(1, 2, 5, 5, device=runtime.device)
    weights = torch.randn(3, 2, 3, 3, device=runtime.device)

    def program(tensor: Any, kernel: Any) -> Any:
        return torch.nn.functional.conv2d(tensor, kernel)

    measurement = measure_program_cost(
        runtime,
        program,
        (values, weights),
        strict=True,
    )

    assert measurement.abstract_flops == 2 * 1 * 3 * 3 * 3 * 2 * 3 * 3
    assert measurement.per_op[0].name == "aten.convolution.default"


def test_measure_program_cost_counts_pointwise_chain() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    left = torch.randn(2, 3, device=runtime.device)
    right = torch.randn(2, 3, device=runtime.device)

    def program(a: Any, b: Any) -> Any:
        return torch.relu(a + b) * b

    measurement = measure_program_cost(runtime, program, (left, right), strict=True)

    assert measurement.abstract_flops == 3 * 6
    assert [record.name for record in measurement.per_op] == [
        "aten.add.Tensor",
        "aten.mul.Tensor",
        "aten.relu.default",
    ]


def test_measure_program_cost_sums_mixed_program() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    left = torch.randn(2, 3, device=runtime.device)
    right = torch.randn(3, 4, device=runtime.device)
    signal = torch.randn(8, dtype=torch.complex64, device=runtime.device)

    def program(a: Any, b: Any, x: Any) -> Any:
        return (a @ b).sin(), torch.fft.fft(x)

    measurement = measure_program_cost(runtime, program, (left, right, signal), strict=True)

    assert measurement.abstract_flops == 48 + 8 + 120
    assert measurement.operation_count == 3


def test_measure_program_cost_is_deterministic_for_repeated_runs() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    left = torch.randn(2, 3, device=runtime.device)
    right = torch.randn(3, 4, device=runtime.device)

    def program(a: Any, b: Any) -> Any:
        return (a @ b).relu()

    first = measure_program_cost(runtime, program, (left, right), strict=True)
    second = measure_program_cost(runtime, program, (left, right), strict=True)

    assert first.abstract_flops == second.abstract_flops
    assert first.per_op == second.per_op
    assert first.operation_trace == second.operation_trace
    assert first.movement == second.movement
    assert first.unmodeled_operations == second.unmodeled_operations


def test_measure_program_cost_records_unmodeled_ops_and_strict_mode_raises() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    matrix = torch.eye(2, device=runtime.device)

    measurement = measure_program_cost(runtime, torch.linalg.inv, matrix)

    assert measurement.abstract_flops == 0
    assert measurement.unmodeled_operations
    assert "aten.linalg_inv_ex.default" in {
        record.name for record in measurement.unmodeled_operations
    }
    with pytest.raises(CostMetrologyError, match="unmodeled operation"):
        measure_program_cost(runtime, torch.linalg.inv, matrix, strict=True)


def test_measure_program_cost_records_movement_outside_abstract_flops() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    values = torch.randn(2, 4, device=runtime.device)
    indices = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], device=runtime.device)

    def program(source: Any, gather_index: Any) -> Any:
        return torch.gather(source, 1, gather_index)

    measurement = measure_program_cost(runtime, program, (values, indices), strict=True)

    assert measurement.abstract_flops == 0
    assert measurement.moved_elements == 8
    assert measurement.movement[0].name == "aten.gather.default"


def test_cost_measurement_round_trips_through_record() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    values = torch.randn(2, 3, device=runtime.device)
    measurement = measure_program_cost(
        runtime,
        torch.relu,
        values,
        strict=True,
        roofline={"source": "test", "peak_flops_per_second": 1.0},
    )

    assert CostMeasurement.from_record(measurement.to_record()) == measurement


def test_cost_meter_context_manager_measures_existing_flow() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    values = torch.randn(2, 3, device=runtime.device)

    with CostMeter(runtime, strict=True) as meter:
        torch.relu(values)

    assert meter.measurement().abstract_flops == 6


def test_measure_program_cost_is_device_independent_for_cuda() -> None:
    cpu_runtime = resolve_tensor_runtime("cpu")
    try:
        cuda_runtime = resolve_tensor_runtime("cuda")
    except TensorRuntimeError as error:
        pytest.skip(str(error))
    cpu_torch = cpu_runtime.torch
    cuda_torch = cuda_runtime.torch
    cpu_left = cpu_torch.randn(2, 3, device=cpu_runtime.device)
    cpu_right = cpu_torch.randn(3, 4, device=cpu_runtime.device)
    cuda_left = cuda_torch.randn(2, 3, device=cuda_runtime.device)
    cuda_right = cuda_torch.randn(3, 4, device=cuda_runtime.device)

    def program(a: Any, b: Any) -> Any:
        return (a @ b).relu()

    cpu_measurement = measure_program_cost(cpu_runtime, program, (cpu_left, cpu_right), strict=True)
    cuda_measurement = measure_program_cost(
        cuda_runtime,
        program,
        (cuda_left, cuda_right),
        strict=True,
    )

    assert cpu_measurement.abstract_flops == cuda_measurement.abstract_flops
