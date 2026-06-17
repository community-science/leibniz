from typing import Any, cast

import pytest

from leibniz.cost_metrology import (
    CostMeasurement,
    CostMeter,
    CostMetrologyError,
    DeviceCostProfile,
    OperationCostRecord,
    device_cost_profile,
    device_cost_profiles,
    estimate_operation_stream_cost,
    estimate_program_cost,
    measure_program_cost,
    normalize_tensor_dtype,
    operation_class_for_name,
    price_cost_measurement_energy,
)
from leibniz.tensor_runtime import (
    TensorRuntimeError,
    TensorRuntimeOperationRecord,
    TensorRuntimeTensorSpec,
    resolve_tensor_runtime,
)


def test_device_cost_profile_round_trips_declared_energy_schema() -> None:
    profile = DeviceCostProfile(
        profile_id="cost-model.device.test@0.1.0",
        label="Test Device",
        version="0.1.0",
        provenance=("unit-test estimate",),
        compute_energy_joules={
            ("dense-matmul", "fp32"): 1.0e-12,
            ("elementwise", "fp32"): 4.0e-12,
            ("transcendental", "fp32"): 1.2e-11,
            ("reduction", "fp32"): 6.0e-12,
            ("fft", "fp32"): 8.0e-12,
        },
        bytes_moved_energy_joules=2.0e-12,
        bytes_resident_energy_joules=1.0e-15,
        unified_memory=True,
        notes=("per-evaluation residency footprint",),
    )

    assert DeviceCostProfile.from_record(profile.to_record()) == profile


def test_device_cost_profile_rejects_unknown_operation_class_and_dtype() -> None:
    record = {
        "profile_id": "cost-model.device.bad@0.1.0",
        "label": "Bad Device",
        "version": "0.1.0",
        "provenance": ["unit-test"],
        "compute_energy_joules": [
            {"operation_class": "scalar", "dtype": "fp32", "joules": 1.0}
        ],
        "bytes_moved_energy_joules": 1.0,
        "bytes_resident_energy_joules": 1.0,
    }

    with pytest.raises(CostMetrologyError, match="operation class"):
        DeviceCostProfile.from_record(record)

    record["compute_energy_joules"] = [
        {"operation_class": "elementwise", "dtype": "fp8", "joules": 1.0}
    ]
    with pytest.raises(CostMetrologyError, match="dtype"):
        DeviceCostProfile.from_record(record)


def test_runtime_operation_names_map_to_device_cost_classes() -> None:
    assert operation_class_for_name("aten.mm.default") == "dense-matmul"
    assert operation_class_for_name("aten.convolution.default") == "convolution"
    assert operation_class_for_name("aten.add.Tensor") == "elementwise"
    assert operation_class_for_name("aten.sin.default") == "transcendental"
    assert operation_class_for_name("aten.sum.default") == "reduction"
    assert operation_class_for_name("aten._fft_c2c.default") == "fft"
    assert operation_class_for_name("aten.gather.default") == "data-movement"
    assert operation_class_for_name("aten.unknown.default") is None


def test_tensor_dtype_strings_map_to_device_profile_dtypes() -> None:
    assert normalize_tensor_dtype("torch.float64") == "fp64"
    assert normalize_tensor_dtype("torch.complex64") == "fp32"
    assert normalize_tensor_dtype("torch.float16") == "fp16"
    assert normalize_tensor_dtype("torch.bfloat16") == "bf16"
    assert normalize_tensor_dtype("torch.int8") == "int8"
    assert normalize_tensor_dtype("torch.bool") is None


def test_device_profile_registry_exposes_default_and_declared_devices() -> None:
    profiles = device_cost_profiles()

    assert device_cost_profile().profile_id == "cost-model.device.nvidia-a100@0.1.0"
    assert {
        "cost-model.device.nvidia-a100@0.1.0",
        "cost-model.device.nvidia-rtx-3080@0.1.0",
        "cost-model.device.apple-m1@0.1.0",
        "cost-model.device.apple-m4@0.1.0",
    }.issubset(profiles)
    assert profiles["cost-model.device.apple-m1@0.1.0"].unified_memory is True


def test_energy_pricing_depends_on_profile_not_counts() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    left = torch.randn(2, 3, device=runtime.device)
    right = torch.randn(3, 4, device=runtime.device)

    def matmul(a: Any, b: Any) -> Any:
        return a @ b

    measurement = measure_program_cost(
        runtime,
        matmul,
        (left, right),
        strict=True,
    )

    a100 = price_cost_measurement_energy(
        measurement,
        profile="cost-model.device.nvidia-a100@0.1.0",
    )
    rtx = price_cost_measurement_energy(
        measurement,
        profile="cost-model.device.nvidia-rtx-3080@0.1.0",
    )

    assert measurement.abstract_flops == 48
    assert a100.total_joules != rtx.total_joules
    assert measurement.abstract_flops == 48


def test_energy_pricing_keeps_matmul_cheaper_than_vector_for_equal_flops() -> None:
    matmul = _synthetic_measurement(
        OperationCostRecord(
            name="aten.mm.default",
            calls=1,
            abstract_flops=1024,
            output_elements=16,
            operation_class="dense-matmul",
            dtype="fp32",
        )
    )
    vector = _synthetic_measurement(
        OperationCostRecord(
            name="aten.add.Tensor",
            calls=1,
            abstract_flops=1024,
            output_elements=1024,
            operation_class="elementwise",
            dtype="fp32",
        )
    )

    assert (
        price_cost_measurement_energy(matmul).compute_joules
        < price_cost_measurement_energy(vector).compute_joules
    )


def test_measure_program_cost_populates_roofline_energy_when_profile_selected() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    values = torch.randn(2, 3, device=runtime.device)

    measurement = measure_program_cost(
        runtime,
        torch.relu,
        values,
        strict=True,
        roofline={"source": "test"},
        device_profile="cost-model.device.apple-m4@0.1.0",
    )

    assert measurement.roofline is not None
    assert measurement.roofline["source"] == "test"
    energy = measurement.roofline["energy"]
    assert isinstance(energy, dict)
    assert energy["profile_id"] == "cost-model.device.apple-m4@0.1.0"
    assert float(cast(float, cast(dict[str, object], energy)["total_joules"])) > 0.0


def test_measure_program_cost_counts_matmul_formula() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    left = torch.randn(2, 3, device=runtime.device)
    right = torch.randn(3, 4, device=runtime.device)

    def program(a: Any, b: Any) -> Any:
        return a @ b

    measurement = measure_program_cost(runtime, program, (left, right), strict=True)

    assert measurement.cost_model_id == CostMeasurement.tensor_runtime_cost_model_id()
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
            operation_class="dense-matmul",
            dtype="fp32",
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
    assert first.bytes_resident == second.bytes_resident
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
    assert measurement.bytes_resident == 128
    assert measurement.movement[0].name == "aten.gather.default"


def test_measure_program_cost_records_resident_table_bytes() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    table = torch.randn(128, 16, dtype=torch.float32, device=runtime.device)
    indices = torch.tensor([0, 3, 5, 7], device=runtime.device)

    def program(held_table: Any, rows: Any) -> Any:
        return torch.index_select(held_table, 0, rows)

    measurement = measure_program_cost(runtime, program, (table, indices), strict=True)

    table_bytes = 128 * 16 * 4
    index_bytes = 4 * 8
    output_bytes = 4 * 16 * 4
    assert measurement.bytes_resident >= table_bytes
    assert measurement.bytes_resident == table_bytes + index_bytes + output_bytes
    assert measurement.moved_elements == 4 * 16


def test_measure_program_cost_treats_unsqueeze_as_shape_movement() -> None:
    runtime = resolve_tensor_runtime("cpu")
    values = runtime.torch.randn(2, 4, device=runtime.device)

    def unsqueeze(tensor: Any) -> Any:
        return tensor.unsqueeze(1)

    measurement = measure_program_cost(
        runtime,
        unsqueeze,
        (values,),
        strict=True,
    )

    assert measurement.abstract_flops == 0
    assert measurement.moved_elements == 8
    assert measurement.movement[0].name == "aten.unsqueeze.default"


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


def test_cost_measurement_exposes_public_cost_normalization_api() -> None:
    measurement = CostMeasurement(
        cost_model_id=CostMeasurement.tensor_runtime_cost_model_id(),
        abstract_flops=120,
        per_op=(
            OperationCostRecord(
                name="test.op",
                calls=1,
                abstract_flops=120,
                output_elements=1,
            ),
        ),
        moved_elements=0,
        movement=(),
        unmodeled_operations=(),
        operation_count=0,
        operation_trace=(),
        wall_seconds=0.0,
        tensor_device="cpu",
    )

    assert measurement.abstract_flops_per_item(5) == 24.0
    assert measurement.abstract_flops_per_byte(12) == 10.0
    assert measurement.abstract_flops_rate(2.5, item_count=5) == 60.0
    assert measurement.bit_density(item_count=5) == 768.0
    assert CostMeasurement.abstract_flops_bit_density(2.5) == 80.0
    assert CostMeasurement.abstract_flops_per_item_value(120, 5) == 24.0
    assert CostMeasurement.abstract_flops_per_byte_value(48, 12) == 4.0
    assert CostMeasurement.abstract_flops_rate_value(12, 2.5) == 30.0


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


def test_without_operation_trace_drops_trace_and_keeps_totals() -> None:
    runtime = resolve_tensor_runtime("cpu")
    left = runtime.torch.ones((4, 8))
    right = runtime.torch.ones((8, 2))

    measurement = measure_program_cost(runtime, lambda: left @ right)
    stripped = measurement.without_operation_trace()

    assert measurement.operation_trace
    assert stripped.operation_trace == ()
    assert stripped.abstract_flops == measurement.abstract_flops
    assert stripped.operation_count == measurement.operation_count
    assert stripped.per_op == measurement.per_op
    assert stripped.bytes_resident == measurement.bytes_resident
    assert "operation_trace" in stripped.to_record()
    assert stripped.to_record()["operation_trace"] == []
    assert stripped.without_operation_trace() is stripped


def _synthetic_measurement(record: OperationCostRecord) -> CostMeasurement:
    return CostMeasurement(
        cost_model_id=CostMeasurement.tensor_runtime_cost_model_id(),
        abstract_flops=record.abstract_flops,
        per_op=(record,),
        moved_elements=0,
        movement=(),
        unmodeled_operations=(),
        operation_count=0,
        operation_trace=(),
        wall_seconds=0.0,
        tensor_device="cpu",
    )
