import inspect
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

import leibniz.benchmark_runner as benchmark_runner
from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.benchmark_evaluation import ValidationCompetencePoint
from leibniz.benchmark_runner import (
    BenchmarkRunnerError,
    BenchmarkRunPlan,
    BenchmarkRunSummary,
    run_benchmark,
)
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.cli import main
from leibniz.documents import canonical_document_bytes, load_object_document
from leibniz.local_results import load_console_result_view, materialize_benchmark_result_views
from leibniz.measurements import MeasurementDatasetDocument
from leibniz.model_inspection import ModelInspectionDocument
from leibniz.observation_generation import load_observation_generator
from leibniz.tensor_runtime import (
    TensorRuntime,
    TensorRuntimeDeviceKind,
    resolve_tensor_runtime,
)
from leibniz.training_runs import TrainingHistoryPoint, TrainingRunRecord

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_digits_architecture = (
    _repository_root / "tests" / "fixtures" / "architecture" / "digits_pool.json"
)
_digits_convnet_architecture = (
    _repository_root / "tests" / "fixtures" / "architecture" / "digits_convnet.json"
)


def test_digits_benchmark_runner_dry_run_does_not_write_state(tmp_path: Path) -> None:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            sample_count=2,
            train_steps=1,
            dry_run=True,
        )
    )

    assert summary.dry_run is True
    assert summary.measurement_count == 2
    assert summary.run_slug.startswith(
        "digits-arch-4a2277aa9fd5-c1-seed101-samples2-steps1-train-"
    )
    assert summary.measurement_dataset_path == (
        tmp_path
        / "results"
        / "measurements"
        / "digits"
        / f"{summary.run_slug}.json"
    )
    assert not summary.measurement_dataset_path.exists()
    assert not (tmp_path / "results").exists()


def test_digits_benchmark_runner_dry_run_does_not_discover_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_runtime_discovery(_requested: object) -> TensorRuntime:
        raise AssertionError("dry-run should not resolve a tensor runtime")

    monkeypatch.setattr(benchmark_runner, "resolve_tensor_runtime", fail_runtime_discovery)

    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            sample_count=2,
            train_steps=1,
            dry_run=True,
        )
    )

    assert summary.dry_run is True


def test_benchmark_run_plan_requires_positive_model_checkpoint_gate_interval(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        BenchmarkRunnerError,
        match="model_checkpoint_gate_interval must be a positive integer",
    ):
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            model_checkpoint_gate_interval=0,
        )


def test_digits_benchmark_runner_rejects_fixed_shape_architecture(
    tmp_path: Path,
) -> None:
    sample = load_observation_generator(_digits_benchmark_root).sample_batch(
        component_count=1,
        sample_count=1,
        seed=101,
    ).samples[0]
    architecture = ArchitectureManifest.from_record(
        {
            "input_shape": list(sample.field.shape),
            "output_shape": [10],
            "layers": [
                {"kind": "flatten"},
                {"kind": "dense", "parameters": {"out": 10}},
            ],
        }
    )
    architecture_path = tmp_path / "fixed-shape-architecture.json"
    architecture_path.write_bytes(canonical_document_bytes(architecture.to_record()))

    with pytest.raises(BenchmarkRunnerError, match="variable-shape scale contract"):
        run_benchmark(
            BenchmarkRunPlan(
                architecture_path=architecture_path,
                benchmark_root=_digits_benchmark_root,
                results_root=tmp_path / "results",
                sample_count=2,
                train_steps=1,
                dry_run=True,
            )
        )


def test_digits_benchmark_runner_rejects_adaptive_pool_without_scale_contract(
    tmp_path: Path,
) -> None:
    architecture = ArchitectureManifest.from_record(
        {
            "input_shape": [1, 32, 32],
            "output_shape": [10],
            "layers": [
                {"kind": "adaptive-pooling", "parameters": {"dimension": 2, "size": 2}},
                {"kind": "flatten"},
                {"kind": "dense", "parameters": {"out": 10}},
            ],
        }
    )
    architecture_path = tmp_path / "adaptive-pool-no-contract.json"
    architecture_path.write_bytes(canonical_document_bytes(architecture.to_record()))

    with pytest.raises(BenchmarkRunnerError, match="variable-shape scale contract"):
        run_benchmark(
            BenchmarkRunPlan(
                architecture_path=architecture_path,
                benchmark_root=_digits_benchmark_root,
                results_root=tmp_path / "results",
                sample_count=2,
                train_steps=1,
                dry_run=True,
            )
        )


def test_digits_scale_contract_accepts_rectangular_generated_shapes() -> None:
    architecture = ArchitectureManifestDocument.from_bytes(
        _digits_architecture.read_bytes()
    ).manifest

    assert cast(Any, benchmark_runner)._input_shape_boundary_reason(
        architecture=architecture,
        sample_shape=(1, 27, 24),
    ) is None


def test_digits_benchmark_runner_writes_valid_tiny_cpu_outputs(tmp_path: Path) -> None:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            sample_count=2,
            evaluation_sample_count=3,
            seed=101,
            train_steps=1,
            tensor_device="cpu",
        )
    )

    dataset_document = MeasurementDatasetDocument.from_bytes(
        summary.measurement_dataset_path.read_bytes()
    )
    inspection_document = ModelInspectionDocument.from_bytes(
        summary.model_inspection_path.read_bytes()
    )
    manifest = BenchmarkManifestDocument.from_bytes(
        (_digits_benchmark_root / "manifest.json").read_bytes()
    ).manifest

    dataset_document.dataset.validate_manifest(manifest)
    assert summary.measurement_count == 3
    assert len(dataset_document.dataset.measurements) == 3
    assert inspection_document.inspection.cost_summary.parameter_count == 50
    assert inspection_document.inspection.cost_summary.inference_compute == 656
    assert summary.training_summary_path.exists()
    training_summary = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )
    training_run = TrainingRunRecord.from_record(
        cast(Mapping[str, object], training_summary["training_run"])
    )
    assert training_run.protocol.optimizer == "adam"
    assert training_run.protocol.objective == "cross-entropy"
    assert training_run.protocol.tensor_runtime == "pytorch"
    assert training_run.protocol.tensor_device == "cpu"
    assert training_run.protocol.max_steps == 1
    assert training_run.protocol.gate_sample_count == 2
    assert training_summary["tensor_runtime"] == "pytorch"
    assert training_summary["tensor_device"] == "cpu"
    throughput = cast(dict[str, object], training_summary["throughput"])
    training_throughput = cast(dict[str, object], throughput["training"])
    evaluation_throughput = cast(dict[str, object], throughput["evaluation"])
    checkpoint_evaluation_throughput = cast(
        dict[str, object],
        throughput["checkpoint_evaluation"],
    )
    phase_timing = cast(dict[str, object], throughput["phase_timing"])
    timing_phases = cast(dict[str, object], phase_timing["phases"])
    tensor_batch_timing = cast(dict[str, object], timing_phases["training_tensor_batch"])
    forward_timing = cast(dict[str, object], timing_phases["training_forward_loss"])
    materialization_timing = cast(
        dict[str, object],
        timing_phases["training_formation_generation.materialization_plan"],
    )
    variation_timing = cast(
        dict[str, object],
        timing_phases["training_formation_generation.variation_coordinates"],
    )
    roofline = cast(dict[str, object], throughput["roofline"])
    roofline_comparison = cast(dict[str, object], throughput["roofline_comparison"])
    phases = cast(dict[str, object], roofline_comparison["phases"])
    training_phase = cast(dict[str, object], phases["training"])
    assert throughput["tensor_runtime"] == "pytorch"
    assert throughput["tensor_device"] == "cpu"
    assert phase_timing["kind"] == "benchmark-phase-timing"
    assert tensor_batch_timing["sample_count"] == 2
    assert cast(float, tensor_batch_timing["seconds"]) > 0
    assert materialization_timing["sample_count"] == 2
    assert variation_timing["sample_count"] == 2
    assert forward_timing["sample_count"] == 2
    assert cast(float, forward_timing["seconds"]) > 0
    assert cast(float, roofline["peak_bytes_per_second"]) > 0
    assert training_throughput["sample_count"] == 2
    assert cast(float, training_throughput["samples_per_second"]) > 0
    assert evaluation_throughput["sample_count"] == 3
    assert checkpoint_evaluation_throughput["sample_count"] == 3
    assert cast(float, evaluation_throughput["samples_per_second"]) > 0
    assert roofline_comparison["status"] == "available"
    assert roofline_comparison["model"] == "operational-intensity"
    assert cast(float, roofline_comparison["training_fraction_of_roofline"]) > 0
    assert training_phase["limiting_resource"] in {"compute", "memory-bandwidth"}
    assert cast(float, training_phase["arithmetic_intensity_compute_per_byte"]) > 0
    assert cast(float, training_phase["expected_roofline_compute_per_second"]) > 0
    assert training_run.steps_run == 1
    assert training_run.validation_checks == 2
    assert training_run.validation_history[0].step == 0
    assert training_run.validation_history[-1].step == 1
    evaluation_curriculum = cast(dict[str, object], training_summary["evaluation_curriculum"])
    training_curriculum = cast(dict[str, object], training_summary["training_curriculum"])
    curriculum_rungs = cast(list[dict[str, object]], evaluation_curriculum["rungs"])
    assert evaluation_curriculum["kind"] == "competence-gated-evaluation-curriculum"
    assert (
        evaluation_curriculum["curriculum_variable"]
        == "internal-distinguishable-state-complexity"
    )
    assert evaluation_curriculum["sampling_levers"] == ["canvas-size"]
    assert cast(dict[str, object], evaluation_curriculum["canvas_growth"])["kind"] == (
        "logarithmic"
    )
    assert evaluation_curriculum["rung_policy"] == "unbounded-competence-frontier"
    assert (
        evaluation_curriculum["gating_metric"]
        == "monotone-frontier-validation-competence"
    )
    assert evaluation_curriculum["frontier_index"] == 0
    assert [rung["index"] for rung in curriculum_rungs] == [0]
    assert {
        cast(str, rung["complexity_axis"])
        for rung in curriculum_rungs
    } == {"internal-distinguishable-state-complexity"}
    assert all("generation_memory_limit_bytes" not in rung for rung in curriculum_rungs)
    assert all("resolution_assignment" in rung for rung in curriculum_rungs)
    expected_rung_keys = {
        "complexity",
        "complexity_axis",
        "index",
        "resolution_assignment",
        "sample_count",
        "seed",
        "status",
    }
    assert all(set(rung) == expected_rung_keys for rung in curriculum_rungs)
    assert [rung["sample_count"] for rung in curriculum_rungs] == [3]
    assert [cast(float, rung["complexity"]) for rung in curriculum_rungs] == sorted(
        cast(float, rung["complexity"]) for rung in curriculum_rungs
    )
    assert training_curriculum["kind"] == "competence-gated-training-curriculum"
    assert training_curriculum["source"] == "structured-training-curriculum"
    assert training_curriculum["frontier_sampling_weight"] == 0.7
    assert training_curriculum["replay_sampling_weight"] == 0.3
    assert (
        training_curriculum["gating_metric"]
        == "monotone-frontier-validation-competence"
    )
    assert training_curriculum["sampling_levers"] == ["canvas-size"]
    assert training_curriculum["frontier_index"] == 0
    training_rungs = cast(list[dict[str, object]], training_curriculum["rungs"])
    assert [rung["index"] for rung in training_rungs] == [0]
    assert all(set(rung) == expected_rung_keys for rung in training_rungs)
    sampled_competence = cast(dict[str, object], training_summary["sampled_competence"])
    assert sampled_competence["kind"] == "sampled-competence-curriculum"
    assert sampled_competence["sampling_rule"] == "generator-uniform-component-sequence-v1"
    assert (
        sampled_competence["difficulty_assumption"]
        == "approximately-uniform-within-complexity-class"
    )
    assert sampled_competence["complexity_axis"] is None
    expected_complexity = load_observation_generator(
        _digits_benchmark_root
    ).distinguishable_state_complexity(
        component_count=1,
        width=24,
        height=24,
        variation_extent=1.0,
    )
    assert math.isclose(cast(float, sampled_competence["complexity"]), expected_complexity)
    assert sampled_competence["sample_count"] == 3
    assert 0.0 <= cast(float, sampled_competence["mean_accepted_mass"]) <= 1.0
    points = cast(list[dict[str, object]], sampled_competence["points"])
    assert len(points) == 1
    assert [point["sample_count"] for point in points] == [3]
    assert math.isclose(cast(float, points[0]["complexity"]), expected_complexity)
    assert [cast(float, point["complexity"]) for point in points] == sorted(
        cast(float, point["complexity"]) for point in points
    )


def test_digits_benchmark_runner_accepts_convnet_architecture(
    tmp_path: Path,
) -> None:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_convnet_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            sample_count=2,
            evaluation_sample_count=2,
            seed=101,
            train_steps=1,
            tensor_device="cpu",
        )
    )

    inspection = ModelInspectionDocument.from_bytes(
        summary.model_inspection_path.read_bytes()
    ).inspection

    assert summary.measurement_count == 2
    assert [stage.operator_kind for stage in inspection.architecture_trace.stages] == [
        "fixed-support-affine",
        "local-affine",
        "rank-collapse",
        "affine-readout",
    ]
    assert [stage.output_shape for stage in inspection.architecture_trace.stages] == [
        (4, 12, 12),
        (4, 12, 12),
        (576,),
        (10,),
    ]
    assert inspection.cost_summary.parameter_count == 5926


def test_benchmark_runner_reports_only_final_evaluation_rung(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = load_observation_generator(_digits_benchmark_root)
    architecture = ArchitectureManifestDocument.from_bytes(
        _digits_architecture.read_bytes()
    ).manifest
    final_rung = cast(Any, benchmark_runner)._evaluation_curriculum_rung(
        architecture=architecture,
        generator=generator,
        component_count=1,
        sample_count=2,
        seed=101,
        index=1,
    )
    competition_rung = cast(Any, benchmark_runner)._competition_curriculum_rung(
        generator=generator,
        component_count=1,
        sample_count=2,
        seed=101,
        index=1,
        resolution_assignment=final_rung.resolution_assignment,
    )

    fake_evaluation_results: list[tuple[object, tuple[tuple[float, ...], ...]]] = []

    def fake_train_and_predict(**kwargs: object) -> object:
        initial_rung = cast(Any, kwargs["initial_evaluation_rung"])
        progress_callback = cast(Any, kwargs["progress_callback"])
        probabilities = tuple((0.1,) * 10 for _sample in initial_rung.batch.samples)
        final_probabilities = tuple((0.1,) * 10 for _sample in final_rung.batch.samples)
        fake_evaluation_results[:] = [
            (initial_rung, probabilities),
            (final_rung, final_probabilities),
        ]
        training_run = cast(Any, benchmark_runner)._training_run_record(
            seed=101,
            batch_size=2,
            max_steps=1,
            learning_rate=0.01,
            optimizer_name="adam",
            schedule_name="reduce-on-plateau",
            gate_check_interval=32,
            gate_sample_count=2,
            gate_decision_rule="validation-loss-plateau",
            convergence_patience=6,
            convergence_min_delta=0.001,
            convergence_min_steps=500,
            tensor_device="cpu",
            validation_history=(
                TrainingHistoryPoint(
                    step=0,
                    validation_check=0,
                    validation_loss=math.log(10),
                    stale_checks=0,
                    learning_rates=(0.01,),
                ),
            ),
            stop_reason="max-steps",
            training_compute=0.0,
        )
        runtime = resolve_tensor_runtime("cpu")
        executable = benchmark_runner.ExecutableModelOperator(cast(Any, kwargs["architecture"]))
        module = benchmark_runner.OperationFallbackSequential(
            runtime=runtime,
            operations=executable.operation_modules(),
        )
        progress_callback(training_run, {}, {}, module)
        return cast(Any, benchmark_runner)._TrainingResult(
            evaluation_results=(),
            training_rungs=(initial_rung, final_rung),
            training_frontier_index=1,
            training_run=training_run,
            throughput={},
        )

    def fake_evaluate_model_checkpoint_artifact(**_kwargs: object) -> object:
        return (
            tuple(fake_evaluation_results),
            {"kind": "checkpoint-evaluation-throughput", "sample_count": 2, "seconds": 0.1},
        )

    def fake_generate_model_checkpoint_competition_profile(**kwargs: object) -> object:
        competition_probabilities = tuple(
            (0.1,) * 10 for _sample in competition_rung.batch.samples
        )
        return (
            cast(Any, benchmark_runner)._prediction_competition_profile_record(
                batch=competition_rung.batch,
                probabilities=competition_probabilities,
                outcome_space=cast(Any, kwargs["outcome_space"]),
                run_slug=cast(str, kwargs["run_slug"]),
                benchmark_id=cast(Any, kwargs["benchmark_id"]),
                architecture_digest=cast(Any, kwargs["architecture_digest"]),
            ),
            {
                "kind": "checkpoint-competition-throughput",
                "sample_count": 2,
                "seconds": 0.1,
            },
        )

    monkeypatch.setattr(benchmark_runner, "_train_and_predict", fake_train_and_predict)
    monkeypatch.setattr(
        benchmark_runner,
        "evaluate_model_checkpoint_artifact",
        fake_evaluate_model_checkpoint_artifact,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "generate_model_checkpoint_competition_profile",
        fake_generate_model_checkpoint_competition_profile,
    )

    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            evaluation_sample_count=2,
            sample_count=2,
            train_steps=1,
            tensor_device="cpu",
        )
    )
    dataset = MeasurementDatasetDocument.from_bytes(
        summary.measurement_dataset_path.read_bytes()
    ).dataset
    training_summary = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )
    competition_profile = load_object_document(
        summary.competition_profile_path.read_bytes(),
        description="competition profile",
    )
    sampled_competence = cast(dict[str, object], training_summary["sampled_competence"])
    points = cast(list[dict[str, object]], sampled_competence["points"])

    assert summary.measurement_count == 2
    assert len(dataset.measurements) == 2
    assert training_summary["evaluation_curriculum_rung_count"] == 2
    evaluation_curriculum = cast(
        dict[str, object],
        training_summary["evaluation_curriculum"],
    )
    assert len(cast(list[object], evaluation_curriculum["rungs"])) == 2
    assert len(points) == 1
    assert math.isclose(cast(float, points[0]["complexity"]), final_rung.complexity)
    measurement_ids = [
        str(measurement.raw_scoring_evidence.id)
        for measurement in dataset.measurements
    ]
    assert all(".final." in measurement_id for measurement_id in measurement_ids)
    assert all(".rung0." not in measurement_id for measurement_id in measurement_ids)
    competition_entries = cast(list[dict[str, object]], competition_profile["entries"])
    assert competition_profile["mechanic"] == "paired-prediction-accepted-mass"
    assert summary.competition_profile_path.parent.name == "digits"
    assert summary.competition_profile_path.parent.parent.name == "competitions"
    assert all(".competition." in cast(str, entry["id"]) for entry in competition_entries)
    assert not {
        cast(str, entry["observation_id"])
        for entry in competition_entries
    } & {
        str(measurement.raw_scoring_evidence.observation_id)
        for measurement in dataset.measurements
    }


def test_digits_benchmark_runner_records_convergence_protocol_controls(
    tmp_path: Path,
) -> None:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            sample_count=2,
            seed=101,
            train_steps=3,
            learning_rate=0.005,
            optimizer="adam",
            schedule="cosine",
            gate_check_interval=1,
            model_checkpoint_gate_interval=2,
            gate_sample_count=3,
            convergence_patience=2,
            convergence_min_delta=0.001,
            tensor_device="cpu",
        )
    )

    training_summary = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )
    training_run = TrainingRunRecord.from_record(
        cast(Mapping[str, object], training_summary["training_run"])
    )

    assert training_run.protocol.optimizer == "adam"
    assert training_run.protocol.schedule == "cosine"
    assert training_run.protocol.learning_rate == 0.005
    assert training_run.protocol.gate_check_interval == 1
    assert training_run.protocol.gate_sample_count == 3
    assert training_run.protocol.gate_decision_rule == "validation-loss-plateau"
    assert training_run.protocol.patience == 2
    assert training_run.protocol.min_delta == 0.001
    assert training_run.protocol.validation_source == "generator-resample"
    assert [point.step for point in training_run.validation_history] == [0, 1, 2, 3]
    assert training_run.validation_history[-1].learning_rates
    assert training_summary["model_checkpoint_gate_interval"] == 2
    checkpoints = cast(list[dict[str, object]], training_summary["model_checkpoints"])
    checkpoint_checks = [checkpoint["validation_check"] for checkpoint in checkpoints]
    assert checkpoint_checks == [0, 2]
    assert all(Path(cast(str, checkpoint["path"])).is_file() for checkpoint in checkpoints)


def test_windowed_plateau_ignores_tiny_recent_best_loss_resets() -> None:
    history = (
        _history_point(check=0, step=0, loss=1.0),
        _history_point(check=1, step=250, loss=1.01, stale_checks=1),
        _history_point(check=2, step=500, loss=0.9995),
    )

    assert benchmark_runner.has_windowed_validation_plateau(
        history,
        window_checks=2,
        min_delta=0.001,
    )


def test_windowed_plateau_continues_after_material_best_loss_improvement() -> None:
    history = (
        _history_point(check=0, step=0, loss=1.0),
        _history_point(check=1, step=250, loss=1.01, stale_checks=1),
        _history_point(check=2, step=500, loss=0.998),
    )

    assert not benchmark_runner.has_windowed_validation_plateau(
        history,
        window_checks=2,
        min_delta=0.001,
    )


def test_training_curriculum_advances_only_after_actual_stage_convergence() -> None:
    assert benchmark_runner.training_stage_converged("validation-plateau")
    assert not benchmark_runner.training_stage_converged("max-steps")
    assert not benchmark_runner.training_stage_converged("training-stopped")
    assert not benchmark_runner.training_stage_converged("no-training-steps")


def test_plateau_scheduler_requires_progressive_learning_rate_reductions() -> None:
    class FakeOptimizer:
        param_groups: list[dict[str, float]]

        def __init__(self) -> None:
            self.param_groups = [{"lr": 1.0}]

    class FakeScheduler:
        def __init__(self, optimizer: FakeOptimizer) -> None:
            self.optimizer = optimizer

        def step(self, _validation_loss: float) -> None:
            for group in self.optimizer.param_groups:
                group["lr"] *= 0.1

    optimizer = FakeOptimizer()
    schedule_class = cast(Any, benchmark_runner)._LearningRateSchedule
    schedule = schedule_class(
        scheduler=FakeScheduler(optimizer),
        optimizer=optimizer,
        update_on="validation-loss",
    )

    for _index in range(2):
        schedule.step_after_validation(1.0)
        assert not schedule.has_exhausted_plateau_response()
    schedule.step_after_validation(1.0)

    assert schedule.has_exhausted_plateau_response()
    assert schedule.learning_rates() == (0.0010000000000000002,)


def test_plateau_scheduler_exhausts_at_effective_learning_rate_floor() -> None:
    class FakeScheduler:
        def step(self, _validation_loss: float) -> None:
            return None

    class FakeOptimizer:
        param_groups: list[dict[str, float]]

        def __init__(self) -> None:
            self.param_groups = [{"lr": 1e-8}]

    schedule_class = cast(Any, benchmark_runner)._LearningRateSchedule
    schedule = schedule_class(
        scheduler=FakeScheduler(),
        optimizer=FakeOptimizer(),
        update_on="validation-loss",
        minimum_effective_learning_rate=1e-8,
    )

    assert schedule.has_exhausted_plateau_response()


def test_plateau_scheduler_resets_learning_rate_for_curriculum_expansion() -> None:
    class FakeScheduler:
        reset_count: int

        def __init__(self) -> None:
            self.reset_count = 0

        def _reset(self) -> None:
            self.reset_count += 1

    class FakeOptimizer:
        param_groups: list[dict[str, float]]

        def __init__(self) -> None:
            self.param_groups = [{"lr": 1e-8}]

    scheduler = FakeScheduler()
    optimizer = FakeOptimizer()
    schedule_class = cast(Any, benchmark_runner)._LearningRateSchedule
    schedule = schedule_class(
        scheduler=scheduler,
        optimizer=optimizer,
        update_on="validation-loss",
        lr_reduction_count=3,
        minimum_effective_learning_rate=1e-8,
        base_learning_rates=(0.01,),
    )

    schedule.reset_for_curriculum_expansion()

    assert schedule.lr_reduction_count == 0
    assert schedule.learning_rates() == (0.01,)
    assert scheduler.reset_count == 1


def test_reduce_on_plateau_scheduler_uses_convergence_patience() -> None:
    import importlib

    runtime = resolve_tensor_runtime("cpu")
    torch = importlib.import_module("torch")
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=0.01)

    schedule = cast(Any, benchmark_runner)._make_scheduler(
        runtime=runtime,
        optimizer=optimizer,
        name="reduce-on-plateau",
        max_steps=None,
        min_delta=1e-3,
        patience=6,
    )

    assert schedule is not None
    assert schedule.scheduler.patience == 6


def test_training_stage_records_current_validation_loss_without_global_best(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def constant_validation_loss(**_kwargs: object) -> float:
        return 2.0

    def fake_batch(_index: int) -> tuple[object, object]:
        return object(), object()

    monkeypatch.setattr(benchmark_runner, "_validation_loss", constant_validation_loss)
    stage_result = cast(Any, benchmark_runner)._train_until_convergence(
        runtime=resolve_tensor_runtime("cpu"),
        module=object(),
        optimizer=type("FakeOptimizer", (), {"param_groups": [{"lr": 0.01}]})(),
        scheduler=None,
        loss_function=object(),
        train_batch=fake_batch,
        validation_batch=fake_batch,
        max_steps=100,
        gate_check_interval=1,
        patience=1,
        min_delta=0.0,
        min_steps=0,
        batch_size=1,
        gate_sample_count=1,
        training_compute_per_sample=10.0,
        training_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        training_compute_counter=cast(Any, benchmark_runner)._ComputeCounter(),
        validation_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        phase_timings=benchmark_runner.TimingCollector(),
        start_step=100,
        start_check=9,
    )

    point = stage_result.validation_history[0]
    assert point.validation_loss == 2.0
    assert "best_validation_loss" not in point.to_record()


def test_training_curriculum_is_not_step_indexed() -> None:
    source = Path(benchmark_runner.__file__).read_text(encoding="utf-8")

    assert "curriculum_step" not in source
    assert "_generation_curriculum_growth_interval" not in source
    assert "step // _generation_curriculum" not in source
    assert "on_plateau=advance_memory_limit" not in source
    assert "current_memory_limit()" not in source
    assert "_curriculum_memory_limits" not in source
    assert "generation_memory_limit_bytes" not in source
    assert "on_plateau=advance_frontier" in source
    assert "initial_evaluation_rung" in source


def test_training_curriculum_only_advances_on_improved_frontier_competence() -> None:
    advances = cast(Any, benchmark_runner)._frontier_plateau_advances
    chance_mass = 0.1

    assert advances(
        frontier_point=ValidationCompetencePoint(complexity=10.0, accepted_mass=1.0),
        previous_frontier_points=(),
        chance_mass=chance_mass,
    )
    assert advances(
        frontier_point=ValidationCompetencePoint(complexity=20.0, accepted_mass=1.0),
        previous_frontier_points=(
            ValidationCompetencePoint(complexity=10.0, accepted_mass=1.0),
        ),
        chance_mass=chance_mass,
    )
    assert not advances(
        frontier_point=ValidationCompetencePoint(complexity=30.0, accepted_mass=chance_mass),
        previous_frontier_points=(
            ValidationCompetencePoint(complexity=10.0, accepted_mass=0.5),
            ValidationCompetencePoint(complexity=20.0, accepted_mass=chance_mass),
        ),
        chance_mass=chance_mass,
    )
    assert not advances(
        frontier_point=ValidationCompetencePoint(complexity=10.0, accepted_mass=chance_mass),
        previous_frontier_points=(),
        chance_mass=chance_mass,
    )


def test_training_curriculum_gate_delegates_frontier_scoring_to_benchmark_api() -> None:
    source = inspect.getsource(cast(Any, benchmark_runner)._frontier_plateau_advances)

    assert "validation_competence_frontier_score" in source
    for leaked_scoring_detail in (
        "accepted_mass",
        "complexity",
        "local_competence",
        "validation_loss",
        "trapezoid",
        "_above_chance",
    ):
        assert leaked_scoring_detail not in source


def test_frontier_plateau_competence_point_uses_validation_loss_scoring_api() -> None:
    competence_point = cast(Any, benchmark_runner)._frontier_plateau_competence_point

    assert competence_point(
        validation_loss=0.0,
        outcome_count=10,
        complexity=math.log2(10),
    ) == ValidationCompetencePoint(
        complexity=math.log2(10),
        accepted_mass=1.0,
    )
    assert math.isclose(
        competence_point(
            validation_loss=0.8 * math.log(10),
            outcome_count=10,
            complexity=10.0,
        ).accepted_mass,
        0.28,
    )


def test_training_curriculum_can_advance_after_worse_loss_on_larger_rung() -> None:
    competence_point = cast(Any, benchmark_runner)._frontier_plateau_competence_point
    advances = cast(Any, benchmark_runner)._frontier_plateau_advances
    first_rung_point = competence_point(
        validation_loss=0.0,
        outcome_count=10,
        complexity=math.log2(10),
    )
    larger_rung_point = competence_point(
        validation_loss=0.8 * math.log(10),
        outcome_count=10,
        complexity=40.0,
    )

    assert advances(
        frontier_point=larger_rung_point,
        previous_frontier_points=(first_rung_point,),
        chance_mass=0.1,
    )


def test_evaluation_curriculum_candidates_are_complexity_sorted() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    candidates = cast(Any, benchmark_runner)._logarithmic_curriculum_candidates(
        generator=generator,
        component_count=1,
        start_index=0,
    )

    assert [candidate.complexity for candidate in candidates] == sorted(
        candidate.complexity for candidate in candidates
    )
    assert candidates[0].resolution_assignment.values == {"W": 24, "H": 24}


def test_training_curriculum_candidates_increment_canvas_size_only() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    candidates = cast(Any, benchmark_runner)._structured_training_curriculum_candidates(
        generator=generator,
        component_count=1,
        start_index=0,
    )
    first_assignments = [
        candidate.resolution_assignment.values
        for candidate in candidates[:12]
    ]

    assert first_assignments == [
        {"W": 24, "H": 24},
        {"W": 24, "H": 48},
        {"W": 48, "H": 24},
        {"W": 48, "H": 48},
        {"W": 24, "H": 72},
        {"W": 48, "H": 72},
        {"W": 72, "H": 24},
        {"W": 72, "H": 48},
        {"W": 72, "H": 72},
        {"W": 24, "H": 120},
        {"W": 48, "H": 120},
        {"W": 72, "H": 120},
    ]
    assert all(
        candidate.resolution_assignment.values["W"] % 24 == 0
        and candidate.resolution_assignment.values["H"] % 24 == 0
        for candidate in candidates[:12]
    )


def test_training_curriculum_does_not_restart_step_counter() -> None:
    source = Path(benchmark_runner.__file__).read_text(encoding="utf-8")

    assert "start_step = validation_history[-1].step" not in source
    assert "start_check = validation_history[-1].validation_check + 1" not in source


def test_loss_threshold_is_not_a_training_option() -> None:
    snake_name = "target_" + "validation_loss"
    flag_name = "--target-" + "validation-loss"
    for path in (
        Path(benchmark_runner.__file__),
        _repository_root / "src" / "leibniz" / "cli.py",
    ):
        source = path.read_text(encoding="utf-8")
        assert snake_name not in source
        assert flag_name not in source


def test_digits_benchmark_runner_auto_falls_back_after_runtime_compile_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_runtime = resolve_tensor_runtime("cpu")
    calls: list[str] = []
    module_calls = 0

    def fake_device_kinds(_requested: object) -> tuple[TensorRuntimeDeviceKind, ...]:
        return ("mps", "cpu")

    def fake_resolve_tensor_runtime(requested: object) -> TensorRuntime:
        device_kind = cast(str, requested)
        calls.append(device_kind)
        return TensorRuntime(
            torch=cpu_runtime.torch,
            device=cpu_runtime.device,
            device_kind=cast(Any, device_kind),
        )

    original_operation_modules = (
        benchmark_runner.ExecutableModelOperator.operation_modules
    )

    def flaky_operation_modules(self: object) -> object:
        nonlocal module_calls
        module_calls += 1
        if module_calls == 1:
            raise RuntimeError("MPS backend failed to compile adaptive pooling")
        return original_operation_modules(cast(Any, self))

    monkeypatch.setattr(benchmark_runner, "tensor_runtime_device_kinds", fake_device_kinds)
    monkeypatch.setattr(benchmark_runner, "resolve_tensor_runtime", fake_resolve_tensor_runtime)
    monkeypatch.setattr(
        benchmark_runner.ExecutableModelOperator,
        "operation_modules",
        flaky_operation_modules,
    )

    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            sample_count=2,
            train_steps=1,
            tensor_device="auto",
        )
    )

    training_summary = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )
    throughput = cast(dict[str, object], training_summary["throughput"])
    fallbacks = cast(list[dict[str, object]], throughput["runtime_fallbacks"])

    assert calls == ["mps", "cpu", "cpu", "cpu"]
    assert training_summary["tensor_device"] == "cpu"
    assert throughput["tensor_device"] == "cpu"
    assert cast(dict[str, object], throughput["evaluation"])["kind"] == (
        "checkpoint-evaluation-throughput"
    )
    assert cast(dict[str, object], throughput["competition"])["kind"] == (
        "checkpoint-competition-throughput"
    )
    assert fallbacks == [
        {
            "from_device": "mps",
            "to_device": "cpu",
            "reason": "MPS backend failed to compile adaptive pooling",
        }
    ]


def test_digits_benchmark_runner_falls_back_per_operation_without_restarting_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_runtime = resolve_tensor_runtime("cpu")
    torch = cpu_runtime.torch
    calls: list[str] = []
    forward_calls = 0

    class FlakyOperation(torch.nn.Module):  # type: ignore[misc]
        def forward(self, value: object) -> object:
            nonlocal forward_calls
            forward_calls += 1
            if forward_calls == 1:
                raise RuntimeError("preferred operation failed to compile")
            return value

    def fake_device_kinds(_requested: object) -> tuple[TensorRuntimeDeviceKind, ...]:
        return ("mps",)

    def fake_resolve_tensor_runtime(requested: object) -> TensorRuntime:
        device_kind = cast(str, requested)
        calls.append(device_kind)
        return TensorRuntime(
            torch=torch,
            device=cpu_runtime.device,
            device_kind=cast(Any, device_kind),
        )

    def fake_roofline_record(_runtime: object) -> dict[str, object]:
        return {
            "kind": "system-roofline",
            "status": "calibrated",
            "tensor_runtime": "pytorch",
            "tensor_device": "mps",
            "method": "test",
            "peak_compute_per_second": 1_000_000.0,
            "peak_bytes_per_second": 1_000_000.0,
        }

    original_operation_modules = (
        benchmark_runner.ExecutableModelOperator.operation_modules
    )

    def flaky_first_operation(self: object) -> object:
        import importlib
        torch = cast(Any, importlib.import_module("torch"))
        modules = list(original_operation_modules(cast(Any, self)))
        modules[0] = torch.nn.Sequential(FlakyOperation(), modules[0])
        return tuple(modules)

    monkeypatch.setattr(benchmark_runner, "tensor_runtime_device_kinds", fake_device_kinds)
    monkeypatch.setattr(benchmark_runner, "resolve_tensor_runtime", fake_resolve_tensor_runtime)
    monkeypatch.setattr(benchmark_runner, "runtime_roofline_record", fake_roofline_record)
    monkeypatch.setattr(
        benchmark_runner.ExecutableModelOperator,
        "operation_modules",
        flaky_first_operation,
    )

    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            sample_count=2,
            train_steps=1,
            tensor_device="auto",
        )
    )

    training_summary = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )
    throughput = cast(dict[str, object], training_summary["throughput"])
    operation_fallbacks = cast(
        list[dict[str, object]],
        throughput["operation_runtime_fallbacks"],
    )

    assert calls == ["mps", "mps", "mps"]
    assert training_summary["tensor_device"] == "mps"
    assert "runtime_fallbacks" not in throughput
    assert operation_fallbacks == [
        {
            "operation_index": 0,
            "from_device": "mps",
            "to_device": "cpu",
            "reason": "preferred operation failed to compile",
        }
    ]


def test_digits_benchmark_runner_run_slug_includes_training_controls() -> None:
    base_plan = BenchmarkRunPlan(
        architecture_path=_digits_architecture,
        benchmark_root=_digits_benchmark_root,
        sample_count=4,
        seed=401,
        train_steps=10,
        optimizer="sgd",
    )
    alternate_plan = BenchmarkRunPlan(
        architecture_path=_digits_architecture,
        benchmark_root=_digits_benchmark_root,
        sample_count=4,
        seed=401,
        train_steps=10,
        optimizer="adam",
    )

    assert base_plan.run_slug.startswith("c1-seed401-samples4-steps10-train-")
    assert alternate_plan.run_slug.startswith("c1-seed401-samples4-steps10-train-")
    assert base_plan.run_slug != alternate_plan.run_slug


def test_digits_benchmark_runner_outputs_feed_benchmark_result_views(tmp_path: Path) -> None:
    run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            sample_count=2,
            seed=101,
            train_steps=1,
            tensor_device="cpu",
        )
    )

    summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )

    view = load_console_result_view(summary.view_file.read_bytes())
    results = cast(list[dict[str, object]], view["benchmark_results"])
    result = results[0]
    history = cast(list[dict[str, object]], result["training_history"])
    assert history[0]["source_kind"] == "local-run"
    assert "model_inspection_digest" in history[0]
    assert "model_inspection_path" in history[0]
    assert "sampled_competence" in history[0]
    diagnostics = cast(dict[str, object], history[0]["training_diagnostics"])
    protocol = cast(dict[str, object], diagnostics["protocol"])
    throughput = cast(dict[str, object], diagnostics["throughput"])
    cost_summary = cast(dict[str, object], history[0]["cost_summary"])
    phase_timing = cast(dict[str, object], throughput["phase_timing"])
    roofline_comparison = cast(dict[str, object], throughput["roofline_comparison"])
    assert protocol["optimizer"] == "adam"
    assert protocol["schedule"] == "reduce-on-plateau"
    assert protocol["patience"] == 6
    assert diagnostics["stop_reason"] == "max-steps"
    assert diagnostics["steps_run"] == 1
    assert diagnostics["training_compute"] == 2784.0
    assert cost_summary["training_compute"] == 2784.0
    assert cost_summary["training_compute_per_sample"] == 1392
    assert diagnostics["validation_checks"] == 2
    assert "final_validation_loss" in diagnostics
    assert "training_tensor_batch" in cast(dict[str, object], phase_timing["phases"])
    assert roofline_comparison["status"] == "available"
    assert roofline_comparison["model"] == "operational-intensity"
    assert len(cast(list[dict[str, object]], diagnostics["validation_history"])) == 2
    console_view_model = cast(dict[str, object], history[0]["console_view_model"])
    detail_sections = cast(list[dict[str, object]], console_view_model["detail_sections"])
    assert [section["title"] for section in detail_sections] == [
        "Sampled Competence",
        "Training Protocol",
        "Training Outcome",
        "Throughput",
        "Validation History",
    ]
    validation_table = cast(dict[str, object], detail_sections[-1]["table"])
    assert validation_table["columns"] == ["Step", "Loss", "Stale"]
    artifact_kinds = {
        artifact["kind"]
        for artifact in cast(list[dict[str, object]], diagnostics["artifacts"])
    }
    assert artifact_kinds == {
        "measurement-dataset",
        "model-checkpoint",
        "model-inspection",
        "model-manifest",
        "training-summary",
    }
    leaderboard = cast(list[dict[str, object]], result["leaderboard"])
    score_basis = cast(dict[str, object], leaderboard[0]["score_basis"])
    assert score_basis["kind"] == "competence-integral-over-complexity-v1"
    assert score_basis["score_unit"] == "bits"
    assert score_basis["complexity_axis"] == "log2-distinguishable-states"
    assert math.isclose(
        cast(float, score_basis["reference_baseline_complexity"]),
        math.log2(10),
    )
    assert math.isclose(cast(float, score_basis["chance_mass"]), 0.1)
    observed_complexities = cast(list[float], leaderboard[0]["observed_complexities"])
    expected_complexity = load_observation_generator(
        _digits_benchmark_root
    ).distinguishable_state_complexity(
        component_count=1,
        width=24,
        height=24,
        variation_extent=1.0,
    )
    assert math.isclose(observed_complexities[0], expected_complexity)
    assert observed_complexities == sorted(observed_complexities)
    points = cast(list[dict[str, object]], leaderboard[0]["points"])
    assert points[0]["sample_count"] == 2
    inspections = cast(list[dict[str, object]], result["model_inspections"])
    assert len(inspections) == 1
    assert inspections[0]["source_path"] == history[0]["model_inspection_path"]
    assert "measurement_dataset" in inspections[0]
    assert "training_provenance" in inspections[0]


def test_digits_benchmark_runner_materializes_running_training_history(
    tmp_path: Path,
) -> None:
    progress_views: list[Mapping[str, object]] = []

    def refresh_progress(_summary: BenchmarkRunSummary) -> None:
        view_summary = materialize_benchmark_result_views(
            repository_root=_repository_root,
            results_root=tmp_path / "results",
        )
        progress_views.append(load_console_result_view(view_summary.view_file.read_bytes()))

    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            sample_count=2,
            evaluation_sample_count=2,
            seed=101,
            train_steps=2,
            gate_check_interval=1,
            model_checkpoint_gate_interval=1,
            convergence_patience=0,
            convergence_min_delta=0.0,
            convergence_min_steps=0,
            tensor_device="cpu",
        ),
        progress_callback=refresh_progress,
    )

    assert progress_views
    progress_view = progress_views[0]
    result = cast(list[dict[str, object]], progress_view["benchmark_results"])[0]
    history = cast(list[dict[str, object]], result["training_history"])
    running_run = history[0]
    diagnostics = cast(dict[str, object], running_run["training_diagnostics"])
    throughput = cast(dict[str, object], diagnostics["throughput"])
    training_throughput = cast(dict[str, object], throughput["training"])

    assert running_run["source_kind"] == "local-progress"
    assert running_run["measurement_count"] == 0
    assert diagnostics["status"] == "running"
    assert diagnostics["stop_reason"] == "validation-checkpoint"
    assert diagnostics["validation_checks"] == 1
    assert training_throughput["sample_count"] == 0
    assert cast(dict[str, object], throughput["roofline_comparison"])["status"] == "available"
    assert cast(list[dict[str, object]], result["leaderboard"])[0]["source_kinds"] == [
        "local-progress"
    ]
    points = cast(
        list[dict[str, object]],
        cast(list[dict[str, object]], result["leaderboard"])[0]["points"],
    )
    sampled_competence = cast(dict[str, object], running_run["sampled_competence"])
    evaluation_curriculum = cast(dict[str, object], diagnostics["evaluation_curriculum"])
    curriculum_rungs = cast(list[dict[str, object]], evaluation_curriculum["rungs"])
    assert points
    assert points[0]["sample_count"] == 2
    assert evaluation_curriculum["kind"] == "competence-gated-evaluation-curriculum"
    assert (
        evaluation_curriculum["curriculum_variable"]
        == "internal-distinguishable-state-complexity"
    )
    assert evaluation_curriculum["frontier_index"] == 0
    assert curriculum_rungs[0]["status"] == "frontier"
    assert sampled_competence["validation_competence"] == running_run["score"]
    assert math.isclose(
        (cast(float, points[0]["score"]) - 0.1) / 0.9,
        cast(float, running_run["score"]),
    )

    final_view_summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )
    final_view = load_console_result_view(final_view_summary.view_file.read_bytes())
    final_result = cast(list[dict[str, object]], final_view["benchmark_results"])[0]
    final_history = cast(list[dict[str, object]], final_result["training_history"])
    progress_path = (
        summary.training_summary_path.parent.parent.parent
        / "training-progress"
        / summary.training_summary_path.parent.name
        / summary.training_summary_path.name
    )
    assert final_history[0]["source_kind"] == "local-run"
    assert not progress_path.exists()


def test_digits_benchmark_runner_records_fixed_component_count(
    tmp_path: Path,
) -> None:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            sample_count=2,
            train_steps=0,
            tensor_device="cpu",
        )
    )

    training_summary = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )
    assert training_summary["component_count"] == 1


def test_cli_runs_digits_benchmark_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "benchmark",
            "train",
            "--architecture",
            str(_digits_architecture),
            "--benchmark-root",
            str(_digits_benchmark_root),
            "--results-root",
            str(tmp_path / "results"),
            "--sample-count",
            "2",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.startswith(
        "planned benchmark training run "
        "digits-arch-4a2277aa9fd5-c1-seed101-samples2-stepsconverge-train-"
    )
    assert not (tmp_path / "results").exists()


def _history_point(
    *,
    check: int,
    step: int,
    loss: float,
    stale_checks: int = 0,
) -> TrainingHistoryPoint:
    return TrainingHistoryPoint(
        step=step,
        validation_check=check,
        validation_loss=loss,
        stale_checks=stale_checks,
    )
