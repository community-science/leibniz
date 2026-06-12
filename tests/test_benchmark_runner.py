import math
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from benchmark_typing import load_digits_benchmark, load_digits_generator

import leibniz.benchmark_runner as benchmark_runner
from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.benchmark_evaluation import (
    CompetencePoint,
    ValidationCompetencePoint,
    finite_measurements_for_predictions,
    sampled_competence_frontier_integral,
    sampled_competence_record,
    validation_competence_frontier_advances,
)
from leibniz.benchmark_implementations import Generator as BenchmarkGenerator
from leibniz.benchmark_runner import (
    BenchmarkEvaluationPlan,
    BenchmarkRunnerError,
    BenchmarkRunPlan,
    BenchmarkRunSummary,
    evaluate_benchmark_checkpoint,
    run_benchmark,
)
from leibniz.cli import main
from leibniz.documents import canonical_document_bytes, load_object_document
from leibniz.evaluation_bundles import BenchmarkEvaluationBundleDocument
from leibniz.identifiers import ProtocolIdentifier
from leibniz.local_results import load_console_result_view, materialize_benchmark_result_views
from leibniz.materialization import AxisAssignment
from leibniz.observation_generation import (
    GeneratedSample,
    GeneratedSampleSet,
    GenerationRequestOutcome,
    StateSpaceVolumeRequest,
    load_generator,
)
from leibniz.state_space import state_space_region_from_record
from leibniz.tensor_runtime import (
    TensorRuntime,
    TensorRuntimeDeviceKind,
    resolve_tensor_runtime,
)
from leibniz.training_runs import TrainingHistoryPoint, TrainingRunRecord

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_chess_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "chess"
_digits_architecture = (
    _repository_root / "tests" / "fixtures" / "architecture" / "digits_pool.json"
)
_digits_convnet_architecture = (
    _repository_root / "tests" / "fixtures" / "architecture" / "digits_convnet.json"
)
_chess_linear_architecture = (
    _repository_root
    / "tests"
    / "fixtures"
    / "architecture"
    / "chess_board_linear.json"
)


def _observation_payload(
    generator: BenchmarkGenerator,
    **kwargs: object,
) -> GeneratedSampleSet:
    sample_set = generator(include_fields=True, **cast(Any, kwargs))
    return sample_set


def test_fieldless_tensor_samples_can_score_predictions() -> None:
    generator = load_generator(_digits_benchmark_root)
    runtime = resolve_tensor_runtime("cpu")
    outcome_space = generator.manifest.resolve_outcome_space()
    outcome_ids = tuple(outcome.id for outcome in outcome_space.outcomes)
    batch = generator(
        seed=101,
        shape=2,
        include_fields=False,
        runtime=runtime,
        outcome_ids=outcome_ids,
    )

    assert batch.fields is not None
    assert batch.targets is not None
    assert batch.samples
    assert all(sample.field is None for sample in batch.samples)
    target_indices = batch.targets.argmax(dim=1).detach().cpu().tolist()
    assert [
        outcome_ids[int(target_index)] for target_index in target_indices
    ] == [sample.outcome_id for sample in batch.samples]

    probabilities = tuple(
        tuple(1.0 if outcome_id == sample.outcome_id else 0.0 for outcome_id in outcome_ids)
        for sample in batch.samples
    )
    measurements = finite_measurements_for_predictions(
        batch=batch,
        outcome_space=outcome_space,
        probabilities=probabilities,
        run_slug="fieldless-tensor-test",
    )

    assert len(measurements) == batch.sample_count
    assert all(not measurement.evidence_artifacts for measurement in measurements)
    assert all(
        measurement.raw_scoring_evidence.observation_id.startswith(
            f"{batch.generator_id}.seed-{batch.seed}.sample-"
        )
        for measurement in measurements
    )


def test_runtime_phase_timings_do_not_sync_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = resolve_tensor_runtime("cpu")
    sync_calls = 0

    def count_sync(_runtime: TensorRuntime) -> None:
        nonlocal sync_calls
        sync_calls += 1

    monkeypatch.delenv("LEIBNIZ_SYNC_TIMING", raising=False)
    monkeypatch.setattr(benchmark_runner, "synchronize_runtime", count_sync)

    timings = cast(Any, benchmark_runner)._runtime_phase_timings(runtime)
    with timings.span("phase"):
        pass

    assert sync_calls == 0


def test_runtime_phase_timings_sync_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = resolve_tensor_runtime("cpu")
    sync_calls = 0

    def count_sync(_runtime: TensorRuntime) -> None:
        nonlocal sync_calls
        sync_calls += 1

    monkeypatch.setenv("LEIBNIZ_SYNC_TIMING", "1")
    monkeypatch.setattr(benchmark_runner, "synchronize_runtime", count_sync)

    timings = cast(Any, benchmark_runner)._runtime_phase_timings(runtime)
    with timings.span("phase"):
        pass

    assert sync_calls == 2


def test_training_sampled_competence_matches_measurement_scoring() -> None:
    generator = load_generator(_digits_benchmark_root)
    runtime = resolve_tensor_runtime("cpu")
    outcome_space = generator.manifest.resolve_outcome_space()
    outcome_ids = tuple(outcome.id for outcome in outcome_space.outcomes)
    request = StateSpaceVolumeRequest(
        minimum=generator.minimum_log2_volume().value,
        maximum=generator.minimum_log2_volume().value + 1.0,
    )
    batch = generator(
        seed=101,
        shape=8,
        include_fields=True,
        volume_request=request,
        runtime=runtime,
        outcome_ids=outcome_ids,
    )
    probabilities = tuple(
        tuple(
            0.7 if outcome_id == sample.outcome_id else 0.3 / (len(outcome_ids) - 1)
            for outcome_id in outcome_ids
        )
        for sample in batch.samples
    )
    measurements = finite_measurements_for_predictions(
        batch=batch,
        outcome_space=outcome_space,
        probabilities=probabilities,
        run_slug="training-direct-score-test",
    )
    measurement_record = sampled_competence_record(
        batch=batch,
        measurements=measurements,
        volume_axis=None,
    )
    direct_record = cast(Any, benchmark_runner)._sampled_competence_record_from_accepted_mass(
        batch=batch,
        accepted_mass=tuple(0.7 for _sample in batch.samples),
        volume_axis=None,
    )

    comparable_keys = {
        "benchmark_id",
        "log2_volume",
        "volume_axis",
        "log2_volume_maximum",
        "log2_volume_minimum",
        "difficulty_assumption",
        "kind",
        "input_shape",
        "mean_accepted_mass",
        "mean_negative_log_score",
        "sample_count",
        "sampling_rule",
        "seed",
    }
    assert {key: measurement_record[key] for key in comparable_keys} == {
        key: direct_record[key] for key in comparable_keys
    }
    assert "measurement_ids" not in direct_record
    assert "observation_ids" not in direct_record


def test_digits_benchmark_runner_dry_run_does_not_write_state(tmp_path: Path) -> None:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            train_steps=1,
            dry_run=True,
        )
    )

    assert summary.dry_run is True
    assert summary.measurement_count == 0
    assert summary.run_slug.startswith(
        "digits-arch-186021388794-seed101-steps1-train-"
    )
    assert "-samples" not in summary.run_slug
    assert not (tmp_path / "results").exists()


def test_dynamic_cuda_batch_sizing_uses_canvas_area_and_memory_budget() -> None:
    class _FakeCuda:
        @staticmethod
        def mem_get_info(device: object) -> tuple[int, int]:
            del device
            total = 10 * 1024 * 1024 * 1024
            free = 9 * 1024 * 1024 * 1024
            return free, total

    runtime = cast(
        TensorRuntime,
        SimpleNamespace(
            device_kind="cuda",
            torch=SimpleNamespace(cuda=_FakeCuda()),
            device="cuda:0",
        ),
    )
    generator = cast(
        Any,
        SimpleNamespace(
            formation=SimpleNamespace(
                channel_count=1,
                height_axis="height",
                width_axis="width",
            )
        ),
    )
    request = StateSpaceVolumeRequest(minimum=20.0, maximum=20.0)
    rung = cast(Any, benchmark_runner)._CurriculumRung(
        index=0,
        resolution_assignment=AxisAssignment(values={"height": 1024, "width": 1024}),
        seed=101,
        batch=GeneratedSampleSet(
            benchmark_id=ProtocolIdentifier.parse("benchmarks.fake@0.1.0"),
            generator_id=ProtocolIdentifier.parse("generators.fake@0.1.0"),
            generator_version="0.1.0",
                seed=101,
                shape=(0,),
                volume_request=request,
                request_outcome=GenerationRequestOutcome(kind="exhausted-capacity"),
            ),
        )
    timings = benchmark_runner.TimingCollector()
    architecture = ArchitectureManifest.from_record(
        {
            "input_shape": [1, 1024, 1024],
            "output_shape": [10],
            "layers": [
                {"kind": "flatten"},
                {"kind": "dense", "parameters": {"out": 10}},
            ],
        }
    )

    physical_count = cast(Any, benchmark_runner)._physical_execution_sample_count(
        runtime=runtime,
        architecture=architecture,
        generator=generator,
        rung=rung,
        requested_sample_count=512,
        outcome_count=10,
        phase_timings=timings,
        phase="training_formation_generation",
    )

    assert 25 <= physical_count <= 35
    phases = cast(Mapping[str, object], timings.to_record()["phases"])
    dynamic_record = cast(
        Mapping[str, object],
        phases["training_formation_generation.dynamic_batch"],
    )
    counters = cast(Mapping[str, object], dynamic_record["counters"])
    assert counters["requested_sample_count"] == 512.0
    assert counters["physical_sample_count"] == float(physical_count)


def test_capacity_limited_training_run_is_budget_exhausted() -> None:
    training_run = cast(Any, benchmark_runner)._training_run_record(
        seed=101,
        max_steps=None,
        learning_rate=0.01,
        optimizer_name="adam",
        schedule_name="reduce-on-plateau",
        gate_check_interval=32,
        gate_decision_rule="score-estimate-plateau",
        rung_competence_threshold=0.01,
        convergence_patience=6,
        convergence_min_delta=0.001,
        tensor_device="cuda",
        runtime_memory_budget_fraction=0.1,
        validation_history=(
            TrainingHistoryPoint(
                step=320,
                validation_check=10,
                validation_loss=math.log(10),
                stale_checks=0,
                learning_rates=(0.01,),
                score_estimate=_score_estimate(
                    check=10,
                    step=320,
                    score=12.0,
                    log2_volume=12.0,
                    accepted_mass=1.0,
                ),
            ),
        ),
        stop_reason="capacity-limited",
        training_compute=100.0,
    )

    assert training_run.status == "budget-exhausted"
    assert training_run.protocol.runtime_memory_budget_fraction == 0.1
    assert cast(Any, benchmark_runner)._training_stage_finished_legally(
        "capacity-limited"
    )
    assert not benchmark_runner.training_stage_converged("capacity-limited")


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
    generator = load_generator(_digits_benchmark_root)
    sample = _observation_payload(
        generator,
        shape=1,
        seed=101,
    ).samples[0]
    architecture = ArchitectureManifest.from_record(
        {
            "input_shape": list(sample.require_field().shape),
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


def test_runner_accepts_exact_fixed_input_shape() -> None:
    architecture = ArchitectureManifest.from_record(
        {
            "input_shape": [18, 8, 8],
            "output_shape": [16],
            "layers": [
                {"kind": "flatten"},
                {"kind": "dense", "parameters": {"out": 16}},
            ],
        }
    )

    assert cast(Any, benchmark_runner)._input_shape_boundary_reason(
        architecture=architecture,
        sample_shape=(18, 8, 8),
    ) is None
    assert cast(Any, benchmark_runner)._input_shape_boundary_reason(
        architecture=architecture,
        sample_shape=(18, 9, 8),
    ) == (
        "architecture input_shape must match generated tensor shape or declare "
        "a variable-shape scale contract for generated observation shape (18, 9, 8)"
    )


def test_chess_benchmark_runner_accepts_fixed_board_tensor(tmp_path: Path) -> None:
    generator = load_generator(_chess_benchmark_root)
    results_root = tmp_path / "results"

    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_chess_linear_architecture,
            benchmark_root=_chess_benchmark_root,
            results_root=results_root,
            seed=101,
            train_steps=1,
            tensor_device="cpu",
        )
    )

    assert summary.benchmark_id == generator.manifest.id
    assert summary.dry_run is False
    assert summary.training_summary_path.exists()
    training_summary = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )
    checkpoint_artifact_path = results_root / "chess-evaluation-model-artifact.json"
    checkpoint_artifact_path.write_bytes(
        canonical_document_bytes(training_summary["evaluation_model_artifact"]) + b"\n"
    )

    evaluation_summary = evaluate_benchmark_checkpoint(
        BenchmarkEvaluationPlan(
            checkpoint_artifact_path=checkpoint_artifact_path,
            benchmark_root=_chess_benchmark_root,
            results_root=results_root,
            tensor_device="cpu",
        )
    )
    bundle = BenchmarkEvaluationBundleDocument.from_bytes(
        evaluation_summary.evaluation_bundle_path.read_bytes()
    ).bundle
    evaluation_protocol = cast(dict[str, object], bundle.evaluation_protocol)
    evaluation_curriculum = cast(dict[str, object], bundle.evaluation_curriculum)

    assert evaluation_protocol["score_status"] in {"accepted", "provisional"}
    assert cast(int, evaluation_protocol["evaluation_curriculum_rung_count"]) >= 1
    assert cast(int, evaluation_curriculum["unlocked_rung_count"]) >= 1


def test_checkpoint_evaluation_treats_empty_later_rung_as_curriculum_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    architecture = ArchitectureManifestDocument.from_bytes(
        _chess_linear_architecture.read_bytes()
    ).manifest
    generator = load_generator(_chess_benchmark_root)
    outcome_space = generator.manifest.resolve_outcome_space()
    runtime = resolve_tensor_runtime("cpu")
    outcome_id = outcome_space.outcomes[0].id
    minimum_log2_volume = generator.minimum_log2_volume().value
    region_batch = generator(
        seed=101,
        shape=1,
        volume_request=StateSpaceVolumeRequest(
            minimum=minimum_log2_volume,
            maximum=minimum_log2_volume,
        ),
    )
    region_sample = region_batch.samples[0]
    sample = GeneratedSample(
        index=0,
        outcome_id=outcome_id,
        available_outcome_ids=(outcome_id,),
        region_component_index=region_sample.region_component_index,
        axis_coordinates=region_sample.axis_coordinates,
    )
    batch = GeneratedSampleSet(
        benchmark_id=generator.manifest.id,
        generator_id=cast(ProtocolIdentifier, generator.id),
        generator_version=generator.version,
        seed=101,
        shape=(1,),
        samples=(sample,),
        fields=runtime.torch.zeros((1, 18, 8, 8), dtype=runtime.torch.float32),
        targets=runtime.torch.zeros((1, len(outcome_space.outcomes)), dtype=runtime.torch.float32),
        region=region_batch.region,
    )
    rung = cast(Any, benchmark_runner)._CurriculumRung(
        index=0,
        resolution_assignment=None,
        seed=101,
        batch=batch,
        sample_count=1,
        log2_volume_minimum=0.0,
        log2_volume_maximum=1.0,
    )
    evidence = cast(Any, benchmark_runner)._CheckpointEvaluationRungEvidence(
        rung=rung,
        mean_accepted_mass=1.0,
        sample_count=1,
        confidence_half_width=0.0,
        input_shape=(18, 8, 8),
    )

    def fake_load_predictor(**_kwargs: object) -> object:
        return SimpleNamespace(runtime=runtime)

    def fake_curriculum_rung(**kwargs: object) -> object:
        if cast(int, kwargs["index"]) == 0:
            return rung
        raise cast(Any, benchmark_runner)._EmptyCurriculumWindow()

    def fake_evaluate_rung(**_kwargs: object) -> tuple[object, int]:
        return evidence, 1

    def fake_final_measurements(
        **_kwargs: object,
    ) -> tuple[GeneratedSampleSet, tuple[tuple[float, ...], ...], int]:
        return batch, ((1.0,),), 1

    monkeypatch.setattr(benchmark_runner, "load_model_checkpoint_predictor", fake_load_predictor)
    monkeypatch.setattr(benchmark_runner, "_evaluation_curriculum_rung", fake_curriculum_rung)
    monkeypatch.setattr(benchmark_runner, "_evaluate_checkpoint_rung", fake_evaluate_rung)
    monkeypatch.setattr(
        benchmark_runner,
        "_evaluate_checkpoint_rung_measurements",
        fake_final_measurements,
    )

    results, final_batch, probabilities, throughput = (
        benchmark_runner.evaluate_model_checkpoint_artifact(
            architecture=architecture,
            generator=cast(Any, generator),
            outcome_space=outcome_space,
            seed=101,
            tensor_device="cpu",
            checkpoint=cast(Any, object()),
        )
    )

    assert results == (evidence,)
    assert final_batch is batch
    assert probabilities == ((1.0,),)
    assert throughput["curriculum_exhausted"] is True
    assert throughput["capacity_limited"] is False


def test_digits_benchmark_runner_writes_valid_tiny_cpu_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def runtime_memory_budget_bytes(_runtime: TensorRuntime) -> int:
        return (
            2
            * (1 * 28 * 28 + 10)
            * cast(Any, benchmark_runner)._float32_bytes
            * cast(Any, benchmark_runner)._runtime_batch_memory_safety_factor
        )

    def runtime_used_memory_bytes(_runtime: TensorRuntime) -> int:
        return 0

    monkeypatch.setattr(
        benchmark_runner,
        "_runtime_memory_budget_bytes",
        runtime_memory_budget_bytes,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_runtime_used_memory_bytes",
        runtime_used_memory_bytes,
    )
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            seed=101,
            train_steps=1,
            tensor_device="cpu",
        )
    )
    evaluation_summary = evaluate_benchmark_checkpoint(
        BenchmarkEvaluationPlan(
            checkpoint_artifact_path=_selected_checkpoint_artifact_path(
                summary.training_summary_path
            ),
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            tensor_device="cpu",
        )
    )

    evaluation_bundle = BenchmarkEvaluationBundleDocument.from_bytes(
        evaluation_summary.evaluation_bundle_path.read_bytes()
    ).bundle
    manifest = load_digits_benchmark(_digits_benchmark_root).manifest

    evaluation_bundle.measurement_dataset.validate_manifest(manifest)
    assert evaluation_summary.measurement_count > 0
    assert len(evaluation_bundle.measurement_dataset.measurements) == (
        evaluation_summary.measurement_count
    )
    assert evaluation_bundle.measurement_score_view.source_dataset_digest == (
        evaluation_bundle.measurement_dataset.digest
    )
    assert evaluation_bundle.model_inspection.cost_summary.parameter_count == 50
    assert evaluation_bundle.model_inspection.cost_summary.inference_compute == 656
    checkpoint_evaluation_throughput = cast(
        dict[str, object],
        evaluation_bundle.throughput["checkpoint_evaluation"],
    )
    phase_timing = cast(dict[str, object], checkpoint_evaluation_throughput["phase_timing"])
    phases = cast(dict[str, object], phase_timing["phases"])
    score_generation = cast(
        dict[str, object],
        phases["checkpoint_evaluation_score_generation"],
    )
    score_prediction = cast(
        dict[str, object],
        phases["checkpoint_evaluation_score_prediction"],
    )
    dynamic_batch = cast(
        dict[str, object],
        phases["checkpoint_evaluation_score.dynamic_batch"],
    )
    dynamic_counters = cast(dict[str, object], dynamic_batch["counters"])
    assert cast(int, score_generation["calls"]) > 1
    assert cast(int, score_prediction["calls"]) > 1
    assert cast(float, dynamic_counters["requested_sample_count"]) > cast(
        float,
        dynamic_counters["physical_sample_count"],
    )
    assert isinstance(checkpoint_evaluation_throughput["max_inference_compute"], int)
    assert checkpoint_evaluation_throughput["max_inference_compute"] >= 0
    assert (
        evaluation_bundle.model_checkpoint["manifest_digest"]
        == str(evaluation_bundle.model_manifest.digest)
    )
    assert evaluation_bundle.evaluation_protocol["tensor_device"] == "cpu"
    assert evaluation_bundle.evaluation_protocol["requested_tensor_device"] == "cpu"
    assert evaluation_summary.evaluation_bundle_path.parent == (
        tmp_path / "results" / "evaluations" / "digits"
    )
    assert not (tmp_path / "results" / "measurements").exists()
    assert not (tmp_path / "results" / "model-inspections").exists()
    assert summary.training_summary_path.exists()
    training_summary = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )
    training_run = TrainingRunRecord.from_record(
        cast(Mapping[str, object], training_summary["training_run"])
    )
    assert training_run.protocol.optimizer == "loss-search"
    assert training_run.protocol.learning_rate is None
    assert "learning_rate" not in training_run.protocol.to_record()
    assert "learning_rate" not in training_summary
    assert training_run.protocol.objective == "cross-entropy"
    assert training_run.protocol.tensor_runtime == "pytorch"
    assert training_run.protocol.tensor_device == "cpu"
    assert training_run.protocol.max_steps == 1
    assert training_run.protocol.rung_competence_threshold == 0.5
    assert training_summary["tensor_runtime"] == "pytorch"
    assert training_summary["tensor_device"] == "cpu"
    throughput = cast(dict[str, object], training_summary["throughput"])
    training_throughput = cast(dict[str, object], throughput["training"])
    evaluation_throughput = cast(dict[str, object], throughput["evaluation"])
    phase_timing = cast(dict[str, object], throughput["phase_timing"])
    timing_phases = cast(dict[str, object], phase_timing["phases"])
    tensor_batch_timing = cast(dict[str, object], timing_phases["training_tensor_batch"])
    forward_timing = cast(dict[str, object], timing_phases["training_forward_loss"])
    render_timing = cast(
        dict[str, object],
        timing_phases["training_formation_generation.batch_tensor_render"],
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
    assert cast(float, render_timing["seconds"]) > 0
    assert cast(int, forward_timing["sample_count"]) >= 2
    assert cast(float, forward_timing["seconds"]) > 0
    assert cast(float, roofline["peak_bytes_per_second"]) > 0
    assert training_throughput["sample_count"] == 2
    assert cast(float, training_throughput["samples_per_second"]) > 0
    assert evaluation_throughput["sample_count"] == 0
    assert evaluation_throughput["samples_per_second"] == 0.0
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
    evaluation_record = evaluation_bundle.to_record()
    evaluation_curriculum = cast(dict[str, object], evaluation_record["evaluation_curriculum"])
    evaluation_protocol = cast(dict[str, object], evaluation_record["evaluation_protocol"])
    training_curriculum = cast(dict[str, object], training_summary["training_curriculum"])
    curriculum_rungs = cast(list[dict[str, object]], evaluation_curriculum["rungs"])
    assert evaluation_curriculum["kind"] == "checkpoint-benchmark-evaluation-curriculum"
    assert evaluation_curriculum["curriculum_variable"] == "log2-state-space-volume"
    assert evaluation_curriculum["sampling_levers"] == ["log2-state-space-volume"]
    volume_value = cast(dict[str, object], evaluation_curriculum["volume_value"])
    assert volume_value["scale"] == "log2"
    assert evaluation_curriculum["volume_axis"] == volume_value["measure_id"]
    window_policy = cast(dict[str, object], evaluation_curriculum["window_policy"])
    assert window_policy == {"kind": "integer-bit-shells"}
    assert evaluation_curriculum["rung_policy"] == "unbounded-competence-frontier"
    assert (
        evaluation_curriculum["gating_metric"]
        == "monotone-frontier-validation-competence"
    )
    evaluation_frontier_index = cast(int, evaluation_curriculum["frontier_index"])
    assert evaluation_frontier_index >= 0
    assert evaluation_frontier_index + 1 <= len(curriculum_rungs)
    integration_convergence = cast(
        dict[str, object],
        evaluation_protocol["integration_convergence"],
    )
    assert integration_convergence["kind"] == "adaptive-score-integral-confidence"
    assert cast(float, integration_convergence["score_integral"]) >= 0.0
    assert cast(float, integration_convergence["score_integral_half_width"]) >= 0.0
    assert (
        integration_convergence["score_integral_relative_half_width_threshold"]
        == cast(Any, benchmark_runner)._default_evaluation_integral_relative_half_width
    )
    assert (
        integration_convergence["score_integral_minimum_half_width_threshold"]
        == cast(Any, benchmark_runner)._default_evaluation_integral_minimum_half_width
    )
    assert cast(float, integration_convergence["score_integral_half_width_threshold"]) > 0.0
    assert (
        integration_convergence["terminal_failure_threshold"]
        == cast(Any, benchmark_runner)._default_evaluation_terminal_failure_rungs
    )
    if checkpoint_evaluation_throughput["capacity_limited"] is not True:
        assert (
            cast(int, integration_convergence["terminal_failure_count"])
            >= cast(Any, benchmark_runner)._default_evaluation_terminal_failure_rungs
        )
    assert [rung["index"] for rung in curriculum_rungs] == list(range(len(curriculum_rungs)))
    assert {
        cast(str, rung["volume_axis"])
        for rung in curriculum_rungs
    } == {volume_value["measure_id"]}
    assert all("generation_memory_limit_bytes" not in rung for rung in curriculum_rungs)
    expected_evaluation_rung_keys = {
        "log2_volume",
        "volume_axis",
        "confidence_half_width",
        "index",
        "mean_accepted_mass",
        "sample_count",
        "seed",
        "volume_value",
        "volume_request",
            "score_interval",
            "status",
            "request_outcome",
        }
    assert all(set(rung) == expected_evaluation_rung_keys for rung in curriculum_rungs)
    assert all(
        isinstance(rung["mean_accepted_mass"], float)
        for rung in curriculum_rungs
    )
    assert all(
        isinstance(rung["confidence_half_width"], float)
        for rung in curriculum_rungs
    )
    for rung in curriculum_rungs:
        rung_volume_value = cast(dict[str, object], rung["volume_value"])
        volume_request = cast(dict[str, object], rung["volume_request"])
        assert rung_volume_value["measure_id"] == volume_value["measure_id"]
        assert volume_request["measure_id"] == volume_value["measure_id"]
        assert math.isclose(
            cast(float, rung_volume_value["value"]),
            cast(float, rung["log2_volume"]),
        )
        request_minimum = cast(float, volume_request["minimum"])
        request_maximum = cast(float, volume_request["maximum"])
        rung_log2_volume = cast(float, rung["log2_volume"])
        assert request_minimum <= rung_log2_volume
        assert rung_log2_volume <= request_maximum
        assert math.isclose(request_maximum - request_minimum, 1.0)
        score_interval = cast(dict[str, object], rung["score_interval"])
        assert cast(float, score_interval["log2_volume_minimum"]) <= rung_log2_volume
        assert rung_log2_volume <= cast(float, score_interval["log2_volume_maximum"])
    assert all(cast(int, rung["sample_count"]) > 0 for rung in curriculum_rungs)
    assert [cast(float, rung["log2_volume"]) for rung in curriculum_rungs] == sorted(
        cast(float, rung["log2_volume"]) for rung in curriculum_rungs
    )
    assert training_curriculum["kind"] == "competence-gated-training-curriculum"
    assert training_curriculum["source"] == "structured-training-curriculum"
    assert training_curriculum["frontier_sampling_weight"] == 0.5
    assert training_curriculum["replay_sampling_weight"] == 0.5
    assert training_curriculum["rung_competence_threshold"] == 0.5
    assert (
        training_curriculum["gating_metric"]
        == "monotone-frontier-validation-competence"
    )
    assert training_curriculum["sampling_levers"] == ["log2-state-space-volume"]
    assert training_curriculum["frontier_index"] == 0
    training_rungs = cast(list[dict[str, object]], training_curriculum["rungs"])
    assert [rung["index"] for rung in training_rungs] == [0]
    expected_training_rung_keys = {
        key
        for key in expected_evaluation_rung_keys
        if key not in {"confidence_half_width", "mean_accepted_mass"}
    }
    expected_training_rung_keys_with_resolution = (
        expected_training_rung_keys | {"resolution_assignment"}
    )
    assert all(
        set(rung) == expected_training_rung_keys
        or set(rung) == expected_training_rung_keys_with_resolution
        for rung in training_rungs
    )
    sampled_competence = cast(dict[str, object], evaluation_record["sampled_competence"])
    assert sampled_competence["kind"] == "sampled-competence-curriculum"
    assert sampled_competence["sampling_rule"] == "generator-uniform-component-index-v1"
    assert (
        sampled_competence["difficulty_assumption"]
        == "approximately-uniform-within-volume-window"
    )
    assert sampled_competence["volume_axis"] is None
    assert math.isclose(
        cast(float, sampled_competence["log2_volume"]),
        cast(float, curriculum_rungs[0]["log2_volume"]),
    )
    assert sampled_competence["sample_count"] == sum(
        cast(int, rung["sample_count"]) for rung in curriculum_rungs
    )
    assert 0.0 <= cast(float, sampled_competence["mean_accepted_mass"]) <= 1.0
    points = cast(list[dict[str, object]], sampled_competence["points"])
    assert len(points) == len(curriculum_rungs)
    assert [point["sample_count"] for point in points] == [
        rung["sample_count"] for rung in curriculum_rungs
    ]
    assert all(isinstance(point["input_shape"], list) for point in points)
    assert [point["log2_volume"] for point in points] == [
        rung["log2_volume"] for rung in curriculum_rungs
    ]
    assert [cast(float, point["log2_volume"]) for point in points] == sorted(
        cast(float, point["log2_volume"]) for point in points
    )
    assert [
        (point["log2_volume_minimum"], point["log2_volume_maximum"])
        for point in points
    ] == [
        (
            cast(dict[str, object], rung["score_interval"])["log2_volume_minimum"],
            cast(dict[str, object], rung["score_interval"])["log2_volume_maximum"],
        )
        for rung in curriculum_rungs
    ]
    assert all("region" in point for point in points)
    assert all(
        state_space_region_from_record(cast(dict[str, object], point["region"]))
        == state_space_region_from_record(
            cast(
                dict[str, object],
                cast(dict[str, object], rung["request_outcome"])["region"],
            )
        )
        for point, rung in zip(points, curriculum_rungs, strict=True)
    )


def test_benchmark_evaluation_rejects_checkpoint_artifact_for_wrong_benchmark(
    tmp_path: Path,
) -> None:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            seed=101,
            train_steps=0,
            tensor_device="cpu",
        )
    )
    checkpoint_artifact_path = _selected_checkpoint_artifact_path(summary.training_summary_path)
    checkpoint_record = dict(
        load_object_document(
            checkpoint_artifact_path.read_bytes(),
            description="checkpoint artifact",
        )
    )
    checkpoint_record["benchmark_id"] = "benchmarks.synthetic-bars@0.1.0"
    checkpoint_artifact_path.write_bytes(canonical_document_bytes(checkpoint_record) + b"\n")

    with pytest.raises(BenchmarkRunnerError, match="does not match benchmark root"):
        evaluate_benchmark_checkpoint(
            BenchmarkEvaluationPlan(
                checkpoint_artifact_path=checkpoint_artifact_path,
                benchmark_root=_digits_benchmark_root,
                results_root=tmp_path / "results",
                tensor_device="cpu",
            )
        )


def test_digits_benchmark_runner_accepts_convnet_architecture(
    tmp_path: Path,
) -> None:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_convnet_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            seed=101,
            train_steps=1,
            tensor_device="cpu",
        )
    )
    evaluation_summary = evaluate_benchmark_checkpoint(
        BenchmarkEvaluationPlan(
            checkpoint_artifact_path=_selected_checkpoint_artifact_path(
                summary.training_summary_path
            ),
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            tensor_device="cpu",
        )
    )

    inspection = BenchmarkEvaluationBundleDocument.from_bytes(
        evaluation_summary.evaluation_bundle_path.read_bytes()
    ).bundle.model_inspection

    assert evaluation_summary.measurement_count == 64
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


def test_checkpoint_evaluation_stops_at_runtime_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            seed=101,
            train_steps=1,
            tensor_device="cpu",
        )
    )
    checkpoint_artifact_path = _selected_checkpoint_artifact_path(
        summary.training_summary_path
    )
    checkpoint_record = dict(
        load_object_document(
            checkpoint_artifact_path.read_bytes(),
            description="checkpoint artifact",
        )
    )
    checkpoint_artifact_path.write_bytes(canonical_document_bytes(checkpoint_record) + b"\n")

    original_physical_count = cast(Any, benchmark_runner)._physical_execution_sample_count

    def capacity_after_first_rung(**kwargs: object) -> int:
        rung = cast(Any, kwargs["rung"])
        if rung.index > 0:
            raise cast(Any, benchmark_runner)._RuntimeCapacityReached()
        return cast(int, original_physical_count(**kwargs))

    monkeypatch.setattr(
        benchmark_runner,
        "_physical_execution_sample_count",
        capacity_after_first_rung,
    )

    evaluation_summary = evaluate_benchmark_checkpoint(
        BenchmarkEvaluationPlan(
            checkpoint_artifact_path=checkpoint_artifact_path,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            tensor_device="cpu",
        )
    )

    bundle = BenchmarkEvaluationBundleDocument.from_bytes(
        evaluation_summary.evaluation_bundle_path.read_bytes()
    ).bundle
    evaluation_curriculum = cast(dict[str, object], bundle.evaluation_curriculum)
    evaluation_protocol = cast(dict[str, object], bundle.evaluation_protocol)
    throughput = cast(dict[str, object], bundle.throughput["checkpoint_evaluation"])

    assert evaluation_protocol["score_status"] == "provisional"
    assert throughput["capacity_limited"] is True
    assert evaluation_curriculum["frontier_index"] == 0
    assert len(cast(list[object], evaluation_curriculum["rungs"])) == 1
    view_summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )
    view = load_console_result_view(view_summary.view_file.read_bytes())
    result = cast(list[dict[str, object]], view["benchmark_results"])[0]
    assert cast(list[dict[str, object]], result["leaderboard"]) == []
    assert cast(dict[str, object], result["frontiers"])["cost"] == []
    plot_runs = cast(list[dict[str, object]], result["plot_runs"])
    assert [run["result_status"] for run in plot_runs] == ["provisional"]
    assert cast(list[dict[str, object]], result["model_candidates"])[0][
        "result_status"
    ] == "provisional"


def test_evaluation_frontier_requires_contiguous_confidence_above_chance() -> None:
    rung_evidence = cast(Any, benchmark_runner)._CheckpointEvaluationRungEvidence
    results = (
        rung_evidence(
            rung=SimpleNamespace(index=0),
            mean_accepted_mass=0.20,
            sample_count=100,
            confidence_half_width=0.01,
            input_shape=(1, 16, 16),
        ),
        rung_evidence(
            rung=SimpleNamespace(index=1),
            mean_accepted_mass=0.16,
            sample_count=100,
            confidence_half_width=0.08,
            input_shape=(1, 16, 16),
        ),
        rung_evidence(
            rung=SimpleNamespace(index=2),
            mean_accepted_mass=0.18,
            sample_count=100,
            confidence_half_width=0.03,
            input_shape=(1, 16, 16),
        ),
    )

    assert (
        cast(Any, benchmark_runner)._evaluation_result_frontier_index(
            evaluation_results=results,
            outcome_ids=tuple(f"digit-{index}" for index in range(10)),
        )
        == 0
    )


def test_evaluation_integration_converges_after_confident_terminal_failures() -> None:
    rung_evidence = cast(Any, benchmark_runner)._CheckpointEvaluationRungEvidence
    integration_evidence = cast(Any, benchmark_runner)._evaluation_integration_evidence
    outcome_ids = tuple(f"digit-{index}" for index in range(10))

    def rung(index: int) -> SimpleNamespace:
        return SimpleNamespace(
            index=index,
            log2_volume=float(index + 1),
            seed=101 + index,
            log2_volume_minimum=float(index),
            log2_volume_maximum=float(index + 1),
        )

    results = (
        rung_evidence(
            rung=rung(0),
            mean_accepted_mass=0.20,
            sample_count=100,
            confidence_half_width=0.01,
            input_shape=(1, 16, 16),
        ),
    )

    evidence = integration_evidence(evaluation_results=results, outcome_ids=outcome_ids)
    assert evidence.frontier_index == 0
    assert not evidence.converged

    results = tuple(
        rung_evidence(
            rung=rung(index),
            mean_accepted_mass=0.20 if index == 0 else 0.10,
            sample_count=100,
            confidence_half_width=0.01,
            input_shape=(1, 16, 16),
        )
        for index in range(4)
    )

    evidence = integration_evidence(evaluation_results=results, outcome_ids=outcome_ids)
    assert evidence.frontier_index == 0
    assert evidence.terminal_failure_count == 3
    assert math.isclose(evidence.score_integral_half_width, 0.01 / 0.9)
    assert evidence.converged


def test_evaluation_integration_does_not_reset_after_failed_ladder_gap() -> None:
    rung_evidence = cast(Any, benchmark_runner)._CheckpointEvaluationRungEvidence
    integration_evidence = cast(Any, benchmark_runner)._evaluation_integration_evidence
    outcome_ids = tuple(f"digit-{index}" for index in range(10))

    def rung(index: int) -> SimpleNamespace:
        return SimpleNamespace(
            index=index,
            log2_volume=float(index + 1),
            seed=101 + index,
            log2_volume_minimum=float(index),
            log2_volume_maximum=float(index + 1),
        )

    results = tuple(
        rung_evidence(
            rung=rung(index),
            mean_accepted_mass=mean,
            sample_count=100,
            confidence_half_width=0.01,
            input_shape=(1, 16, 16),
        )
        for index, mean in enumerate((0.90, 0.80, 0.05, 0.04, 0.70))
    )

    evidence = integration_evidence(evaluation_results=results, outcome_ids=outcome_ids)

    assert evidence.frontier_index == 1
    assert evidence.terminal_failure_count == 3
    assert evidence.converged


def test_benchmark_runner_reports_only_final_evaluation_rung(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = load_generator(_digits_benchmark_root)
    architecture = ArchitectureManifestDocument.from_bytes(
        _digits_architecture.read_bytes()
    ).manifest
    final_rung = cast(Any, benchmark_runner)._evaluation_curriculum_rung(
        architecture=architecture,
        generator=generator,
        seed=101,
        index=1,
    )
    def fake_train_and_predict(**kwargs: object) -> object:
        initial_rung = cast(Any, kwargs["initial_evaluation_rung"])
        progress_callback = cast(Any, kwargs["progress_callback"])
        training_run = cast(Any, benchmark_runner)._training_run_record(
            seed=101,
            max_steps=1,
            learning_rate=0.01,
            optimizer_name="adam",
            schedule_name="reduce-on-plateau",
            gate_check_interval=32,
            gate_decision_rule="score-estimate-plateau",
            rung_competence_threshold=0.01,
            convergence_patience=6,
            convergence_min_delta=0.001,
            tensor_device="cpu",
            validation_history=(
                    TrainingHistoryPoint(
                        step=0,
                        validation_check=0,
                        validation_loss=math.log(10),
                        stale_checks=0,
                        learning_rates=(0.01,),
                        score_estimate=_score_estimate(
                            check=0,
                            step=0,
                            score=0.0,
                            log2_volume=initial_rung.log2_volume,
                            accepted_mass=0.1,
                        ),
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

    monkeypatch.setattr(benchmark_runner, "_train_and_predict", fake_train_and_predict)

    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            train_steps=1,
            tensor_device="cpu",
        )
    )
    training_summary = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )

    assert summary.measurement_count == 0
    assert "sampled_competence" not in training_summary
    assert "evaluation_curriculum" not in training_summary
    assert "measurement_dataset_digest" not in training_summary
    assert "model_inspection_digest" not in training_summary
    assert "selected_model_checkpoint" in training_summary


def test_digits_benchmark_runner_records_convergence_protocol_controls(
    tmp_path: Path,
) -> None:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            seed=101,
            train_steps=3,
            learning_rate=0.005,
            optimizer="adam",
            schedule="cosine",
            gate_check_interval=1,
            model_checkpoint_gate_interval=2,
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
    assert training_run.protocol.gate_decision_rule == "score-estimate-plateau"
    assert training_run.protocol.patience == 2
    assert training_run.protocol.min_delta == 0.001
    assert training_run.protocol.validation_source == "generator-resample"
    assert [point.step for point in training_run.validation_history] == [0, 1, 2]
    assert training_run.validation_history[-1].learning_rates
    assert training_summary["model_checkpoint_gate_interval"] == 2
    checkpoints = cast(list[dict[str, object]], training_summary["model_checkpoints"])
    checkpoint_checks = [checkpoint["validation_check"] for checkpoint in checkpoints]
    assert checkpoint_checks == [0]
    assert all("evaluation_rung_count" not in checkpoint for checkpoint in checkpoints)
    assert all("score_estimate" not in checkpoint for checkpoint in checkpoints)
    assert all("score" in checkpoint for checkpoint in checkpoints)
    assert all("architecture_manifest" not in checkpoint for checkpoint in checkpoints)
    assert all("model_manifest" not in checkpoint for checkpoint in checkpoints)
    selected_checkpoint = cast(dict[str, object], training_summary["selected_model_checkpoint"])
    assert "evaluation_rung_count" not in selected_checkpoint
    assert "score_estimate" not in selected_checkpoint
    assert "selected_model_checkpoint_score_estimate" in training_summary
    assert all(
        (tmp_path / cast(str, checkpoint["path"])).is_file()
        for checkpoint in checkpoints
    )
    checkpoint_record = load_object_document(
        (tmp_path / cast(str, checkpoints[0]["record_path"])).read_bytes(),
        description="checkpoint record",
    )
    assert "score_estimate" not in checkpoint_record


def test_windowed_plateau_uses_current_rung_competence() -> None:
    history = (
        _history_point(check=0, step=0, loss=1.0, score=1.0, accepted_mass=0.20),
        _history_point(
            check=1,
            step=250,
            loss=1.01,
            score=1.0,
            accepted_mass=0.2005,
            stale_checks=1,
        ),
        _history_point(check=2, step=500, loss=0.9995, score=1.0, accepted_mass=0.2009),
    )

    assert benchmark_runner.has_windowed_validation_plateau(
        history,
        window_checks=2,
        min_delta=0.001,
        chance_mass=0.1,
    )


def test_windowed_plateau_continues_after_current_rung_competence_improvement() -> None:
    history = (
        _history_point(check=0, step=0, loss=1.0, score=1.0, accepted_mass=0.20),
        _history_point(
            check=1,
            step=250,
            loss=1.01,
            score=1.0,
            accepted_mass=0.2005,
            stale_checks=1,
        ),
        _history_point(check=2, step=500, loss=0.998, score=1.0, accepted_mass=0.205),
    )

    assert not benchmark_runner.has_windowed_validation_plateau(
        history,
        window_checks=2,
        min_delta=0.001,
        chance_mass=0.1,
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
        update_on="score-estimate",
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
        update_on="score-estimate",
        minimum_effective_learning_rate=1e-8,
    )

    assert schedule.has_exhausted_plateau_response()


def test_plateau_scheduler_uses_capped_restart_for_curriculum_expansion() -> None:
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
        update_on="score-estimate",
        lr_reduction_count=3,
        minimum_effective_learning_rate=1e-8,
        curriculum_expansion_learning_rates=(0.001,),
    )

    schedule.reset_for_curriculum_expansion()

    assert schedule.lr_reduction_count == 0
    assert schedule.learning_rates() == (0.001,)
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
    def fake_batch(_index: int) -> tuple[object, object]:
        return object(), object()

    class FakeLossValue:
        def item(self) -> float:
            return 2.0

        def backward(self) -> None:
            pass

    class FakeLossFunction:
        def __call__(self, _logits: object, _labels: object) -> FakeLossValue:
            return FakeLossValue()

    class FakeModule:
        training = True
        train_called = False

        def __call__(self, _fields: object) -> object:
            return object()

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True
            self.train_called = True

    class FakeValidationBatch:
        sample_count = 1

    def fake_validation_batch(_index: int) -> FakeValidationBatch:
        return FakeValidationBatch()

    def fake_batch_tensors(**_kwargs: object) -> tuple[object, object]:
        return object(), object()

    def fake_target_masses(
        _runtime: object,
        _logits: object,
        _labels: object,
    ) -> list[float]:
        return [1.0]

    def fake_training_gate_score_estimate(**kwargs: object) -> dict[str, object]:
        return _score_estimate(
            check=cast(int, kwargs["validation_check"]),
            step=cast(int, kwargs["step"]),
            score=1.0,
            log2_volume=1.0,
            accepted_mass=1.0,
        )

    def fake_batch_max_compute(**_kwargs: object) -> int:
        return 10

    monkeypatch.setattr(benchmark_runner, "_batch_tensors", fake_batch_tensors)
    monkeypatch.setattr(benchmark_runner, "softmax_target_masses", fake_target_masses)
    monkeypatch.setattr(
        benchmark_runner,
        "_batch_max_inference_compute",
        fake_batch_max_compute,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_batch_max_training_compute_per_sample",
        fake_batch_max_compute,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_training_gate_score_estimate",
        fake_training_gate_score_estimate,
    )
    module = FakeModule()
    stage_result = cast(Any, benchmark_runner)._train_until_convergence(
        runtime=resolve_tensor_runtime("cpu"),
        module=module,
        optimizer=type("FakeOptimizer", (), {"param_groups": [{"lr": 0.01}]})(),
        scheduler=None,
        loss_function=FakeLossFunction(),
        train_batch=fake_batch,
        validation_batch=cast(Any, fake_validation_batch),
        outcome_space=(
            load_digits_benchmark(_digits_benchmark_root)
            .manifest.resolve_outcome_space()
        ),
        outcome_ids=("digit-0",),
        max_steps=100,
        gate_check_interval=1,
        patience=1,
        min_delta=0.0,
        rung_competence_threshold=0.9,
        architecture=ArchitectureManifestDocument.from_bytes(
            _digits_architecture.read_bytes()
        ).manifest,
        training_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        training_compute_counter=cast(Any, benchmark_runner)._ComputeCounter(),
        validation_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        phase_timings=benchmark_runner.TimingCollector(),
        start_step=100,
        start_check=9,
    )

    point = stage_result.validation_history[0]
    assert point.validation_loss == 2.0
    assert module.train_called
    assert "best_validation_loss" not in point.to_record()


def test_training_run_artifact_record_omits_historical_score_estimates() -> None:
    training_run = cast(Any, benchmark_runner)._training_run_record(
        seed=101,
        max_steps=None,
        learning_rate=None,
        optimizer_name="loss-search",
        schedule_name="none",
        gate_check_interval=32,
        gate_decision_rule="score-estimate-plateau",
        rung_competence_threshold=0.5,
        convergence_patience=6,
        convergence_min_delta=0.001,
        tensor_device="cpu",
        validation_history=(
            TrainingHistoryPoint(
                step=32,
                validation_check=1,
                validation_loss=1.0,
                stale_checks=0,
                score_estimate=_score_estimate(
                    check=1,
                    step=32,
                    score=1.0,
                    log2_volume=1.0,
                    accepted_mass=1.0,
                ),
            ),
        ),
        stop_reason="validation-plateau",
        training_compute=10.0,
    )

    record = cast(Any, benchmark_runner)._training_run_artifact_record(training_run)

    assert "score_estimate" not in record["validation_history"][0]
    assert record["validation_history"][0]["validation_loss"] == 1.0


def test_throughput_record_surfaces_tensor_compile_fallbacks() -> None:
    fallback = {
        "kind": "tensor-element-compile-fallback",
        "tensor_device": "cuda",
        "program": "('digits-field', 10, 6, 25, 2)",
        "reason": "tensor element compile failed: inductor exploded",
        "constructions": 3,
    }

    record = cast(Any, benchmark_runner)._throughput_record(
        runtime_device="cuda",
        training_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        validation_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        evaluation_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        roofline={"kind": "system-roofline", "status": "unavailable"},
        work_estimates=None,
        phase_timings=benchmark_runner.TimingCollector(),
        tensor_compile_fallbacks=(fallback,),
    )
    silent_record = cast(Any, benchmark_runner)._throughput_record(
        runtime_device="cuda",
        training_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        validation_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        evaluation_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        roofline={"kind": "system-roofline", "status": "unavailable"},
        work_estimates=None,
        phase_timings=benchmark_runner.TimingCollector(),
    )

    assert record["tensor_compile_fallbacks"] == [fallback]
    assert "tensor_compile_fallbacks" not in silent_record


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


def test_training_curriculum_advances_on_current_rung_competence() -> None:
    chance_mass = 0.1

    assert validation_competence_frontier_advances(
        frontier_point=ValidationCompetencePoint(
            log2_volume=0.0,
            accepted_mass=1.0,
            log2_volume_minimum=0.0,
            log2_volume_maximum=0.0,
        ),
        previous_frontier_points=(),
        chance_mass=chance_mass,
    )
    assert validation_competence_frontier_advances(
        frontier_point=ValidationCompetencePoint(log2_volume=10.0, accepted_mass=1.0),
        previous_frontier_points=(
            ValidationCompetencePoint(log2_volume=10.0, accepted_mass=1.0),
        ),
        chance_mass=chance_mass,
    )
    assert not validation_competence_frontier_advances(
        frontier_point=ValidationCompetencePoint(log2_volume=30.0, accepted_mass=chance_mass),
        previous_frontier_points=(
            ValidationCompetencePoint(log2_volume=10.0, accepted_mass=0.5),
            ValidationCompetencePoint(log2_volume=20.0, accepted_mass=chance_mass),
        ),
        chance_mass=chance_mass,
    )
    assert not validation_competence_frontier_advances(
        frontier_point=ValidationCompetencePoint(log2_volume=10.0, accepted_mass=chance_mass),
        previous_frontier_points=(),
        chance_mass=chance_mass,
    )


def test_training_gate_score_estimate_records_prior_frontier_points() -> None:
    generator = load_generator(_digits_benchmark_root)
    outcome_space = generator.manifest.resolve_outcome_space()
    batch = generator(
        shape=2,
        seed=101,
        include_fields=True,
        volume_request=StateSpaceVolumeRequest(
            minimum=4.321928094887362,
            maximum=5.321928094887362,
        ),
    )
    accepted_mass = tuple(1.0 for _sample in batch.samples)

    estimate = cast(Any, benchmark_runner)._training_gate_score_estimate(
        batch=batch,
        outcome_space=outcome_space,
        accepted_mass=accepted_mass,
        previous_frontier_points=(
            ValidationCompetencePoint(
                log2_volume=math.log2(10),
                accepted_mass=1.0,
                sample_count=64,
                seed=202,
                input_shape=(1, 16, 16),
            ),
        ),
        validation_check=1,
        step=32,
        max_inference_compute=10,
        running_max_inference_compute=10,
        training_compute_per_sample=20,
    )

    sampled_competence = cast(dict[str, object], estimate["sampled_competence"])
    points = cast(list[dict[str, object]], sampled_competence["points"])
    assert [point["log2_volume"] for point in points] == [
        math.log2(10),
        batch.log2_volume,
    ]
    assert points[0]["sample_count"] == 64
    assert points[0]["seed"] == 202
    assert points[0]["mean_accepted_mass"] == 1.0
    assert points[0]["input_shape"] == [1, 16, 16]
    assert points[1]["input_shape"] == list(batch.samples[0].require_field().shape)
    assert batch.region is not None
    assert state_space_region_from_record(points[1]["region"]) == batch.region
    score_terms = cast(
        list[dict[str, object]],
        cast(dict[str, object], estimate["score_integral"])["terms"],
    )
    assert state_space_region_from_record(score_terms[-1]["region"]) == batch.region
    assert math.isclose(
        cast(float, estimate["score"]),
        sampled_competence_frontier_integral(
            tuple(
                CompetencePoint(
                    log2_volume=cast(float, point["log2_volume"]),
                    accepted_mass=cast(float, point["mean_accepted_mass"]),
                    sample_count=cast(int, point["sample_count"]),
                    log2_volume_minimum=(
                        None
                        if point.get("log2_volume_minimum") is None
                        else cast(float, point["log2_volume_minimum"])
                    ),
                    log2_volume_maximum=(
                        None
                        if point.get("log2_volume_maximum") is None
                        else cast(float, point["log2_volume_maximum"])
                    ),
                )
                for point in points
            ),
            chance_mass=0.1,
        ).value,
    )


def test_training_replay_batches_refresh_prior_frontier_score_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    runtime = resolve_tensor_runtime("cpu")
    torch = importlib.import_module("torch")
    generator = load_generator(_digits_benchmark_root)
    outcome_space = generator.manifest.resolve_outcome_space()
    outcome_ids = tuple(outcome.id for outcome in outcome_space.outcomes)
    replay_request = StateSpaceVolumeRequest(
        minimum=math.log2(10),
        maximum=math.log2(20),
    )
    replay_batch = generator(
        shape=4,
        seed=101,
        include_fields=True,
        include_metadata=True,
        runtime=runtime,
        outcome_ids=outcome_ids,
        volume_request=replay_request,
    )
    fields, labels = replay_batch.require_tensors()
    prior_point = ValidationCompetencePoint(
        log2_volume=math.log2(10),
        log2_volume_minimum=math.log2(10),
        log2_volume_maximum=math.log2(20),
        accepted_mass=1.0,
        sample_count=64,
        seed=202,
        input_shape=(1, 16, 16),
    )
    gate_prior_points: list[tuple[ValidationCompetencePoint, ...]] = []

    class FakeLossValue:
        def item(self) -> float:
            return 2.0

        def backward(self) -> None:
            pass

    class FakeLossFunction:
        def __call__(self, _logits: object, _labels: object) -> FakeLossValue:
            return FakeLossValue()

    class FakeModule:
        training = True

        def __call__(self, batch_fields: object) -> object:
            sample_count = cast(Any, batch_fields).shape[0]
            logits = torch.full((sample_count, len(outcome_ids)), -1000.0)
            logits[:, 0] = 1000.0
            return logits

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

    class FakeOptimizer:
        param_groups = [{"lr": 0.01}]

        def zero_grad(self, *, set_to_none: bool) -> None:
            _ = set_to_none

        def step(self) -> None:
            pass

    def fake_training_batch(_step: int) -> object:
        return cast(Any, benchmark_runner)._TrainingStepBatch(
            fields=fields,
            labels=labels,
            sample_set=replay_batch,
        )

    def fake_validation_batch(_index: int) -> GeneratedSampleSet:
        return replay_batch

    def fake_batch_tensors(**_kwargs: object) -> tuple[object, object]:
        return fields, labels

    def fake_training_gate_score_estimate(**kwargs: object) -> dict[str, object]:
        gate_prior_points.append(
            cast(tuple[ValidationCompetencePoint, ...], kwargs["previous_frontier_points"])
        )
        return _score_estimate(
            check=cast(int, kwargs["validation_check"]),
            step=cast(int, kwargs["step"]),
            score=1.0,
            log2_volume=20.0,
            accepted_mass=1.0,
        )

    def fake_batch_max_compute(**_kwargs: object) -> int:
        return 10

    monkeypatch.setattr(benchmark_runner, "_batch_tensors", fake_batch_tensors)
    monkeypatch.setattr(
        benchmark_runner,
        "_batch_max_inference_compute",
        fake_batch_max_compute,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_batch_max_training_compute_per_sample",
        fake_batch_max_compute,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_training_gate_score_estimate",
        fake_training_gate_score_estimate,
    )

    stage_result = cast(Any, benchmark_runner)._train_until_convergence(
        runtime=runtime,
        module=FakeModule(),
        optimizer=FakeOptimizer(),
        scheduler=None,
        loss_function=FakeLossFunction(),
        train_batch=cast(Any, fake_training_batch),
        validation_batch=fake_validation_batch,
        outcome_space=outcome_space,
        outcome_ids=outcome_ids,
        max_steps=1,
        gate_check_interval=1,
        patience=10,
        min_delta=0.001,
        rung_competence_threshold=0.9,
        architecture=ArchitectureManifestDocument.from_bytes(
            _digits_architecture.read_bytes()
        ).manifest,
        training_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        training_compute_counter=cast(Any, benchmark_runner)._ComputeCounter(),
        validation_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        phase_timings=benchmark_runner.TimingCollector(),
        frontier_points=lambda: (prior_point,),
    )

    assert stage_result.stop_reason == "max-steps"
    assert len(gate_prior_points) == 2
    assert gate_prior_points[0] == (prior_point,)
    replay_refreshed_points = gate_prior_points[1]
    assert len(replay_refreshed_points) == 1
    assert replay_refreshed_points[0].sample_count == replay_batch.sample_count
    assert replay_refreshed_points[0].seed == replay_batch.seed
    assert replay_refreshed_points[0].input_shape == tuple(fields.shape[1:])
    assert replay_refreshed_points[0].log2_volume_minimum is not None
    assert replay_refreshed_points[0].log2_volume_maximum is not None
    assert math.isclose(
        replay_refreshed_points[0].log2_volume_minimum,
        replay_request.minimum,
    )
    assert math.isclose(
        replay_refreshed_points[0].log2_volume_maximum,
        replay_request.maximum,
    )
    assert math.isclose(replay_refreshed_points[0].accepted_mass, 0.25)


def test_training_replay_frontier_points_are_sample_weighted_by_interval() -> None:
    replay_points: dict[tuple[float, float], object] = {}

    cast(Any, benchmark_runner)._accumulate_replay_frontier_point(
        replay_points,
        ValidationCompetencePoint(
            log2_volume=4.0,
            accepted_mass=0.25,
            sample_count=4,
            seed=101,
            log2_volume_minimum=4.0,
            log2_volume_maximum=5.0,
            input_shape=(1, 16, 16),
        ),
    )
    cast(Any, benchmark_runner)._accumulate_replay_frontier_point(
        replay_points,
        ValidationCompetencePoint(
            log2_volume=4.0,
            accepted_mass=0.75,
            sample_count=12,
            seed=202,
            log2_volume_minimum=4.0,
            log2_volume_maximum=5.0,
            input_shape=(1, 20, 20),
        ),
    )

    rolling_point = cast(Any, replay_points[(4.0, 5.0)]).point
    assert rolling_point.sample_count == 16
    assert rolling_point.seed == 202
    assert rolling_point.log2_volume_minimum == 4.0
    assert rolling_point.log2_volume_maximum == 5.0
    assert rolling_point.input_shape == (1, 20, 20)
    assert math.isclose(rolling_point.accepted_mass, 0.625)


def test_training_replay_frontier_points_keep_recent_window() -> None:
    replay_points: dict[tuple[float, float], object] = {}

    for index in range(9):
        cast(Any, benchmark_runner)._accumulate_replay_frontier_point(
            replay_points,
            ValidationCompetencePoint(
                log2_volume=4.0,
                accepted_mass=float(index),
                sample_count=1,
                seed=100 + index,
                log2_volume_minimum=4.0,
                log2_volume_maximum=5.0,
            ),
        )

    rolling = cast(Any, replay_points[(4.0, 5.0)])
    rolling_point = rolling.point
    assert len(rolling.points) == 8
    assert rolling_point.sample_count == 8
    assert rolling_point.seed == 108
    assert math.isclose(rolling_point.accepted_mass, 4.5)


def test_training_steps_materialize_replay_masses_only_at_gate_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_log: list[str] = []

    class RecordingMass:
        def tolist(self) -> list[float]:
            event_log.append("tolist")
            return [0.5, 0.5]

    class FakeSampleSet:
        sample_count = 2

    def fake_batch(_index: int) -> object:
        event_log.append("step")
        return cast(Any, benchmark_runner)._TrainingStepBatch(
            fields=object(),
            labels=object(),
            sample_set=cast(Any, FakeSampleSet()),
        )

    class FakeLossValue:
        def item(self) -> float:
            return 2.0

        def backward(self) -> None:
            pass

    class FakeLossFunction:
        def __call__(self, _logits: object, _labels: object) -> FakeLossValue:
            return FakeLossValue()

    class FakeModule:
        training = True

        def __call__(self, _fields: object) -> object:
            return object()

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

    class FakeValidationBatch:
        sample_count = 2

    class FakeOptimizer:
        param_groups: list[dict[str, object]] = [{}]

        def zero_grad(self, *, set_to_none: bool) -> None:
            _ = set_to_none

        def step(self) -> None:
            pass

    def fake_validation_batch(_index: int) -> FakeValidationBatch:
        event_log.append("gate")
        return FakeValidationBatch()

    def fake_batch_tensors(**_kwargs: object) -> tuple[object, object]:
        return object(), object()

    def fake_target_mass_tensor(
        _runtime: object,
        _logits: object,
        _labels: object,
    ) -> RecordingMass:
        return RecordingMass()

    def fake_target_masses(
        _runtime: object,
        _logits: object,
        _labels: object,
    ) -> list[float]:
        return [1.0]

    def fake_sampled_record(**kwargs: object) -> dict[str, object]:
        assert kwargs["accepted_mass"] == (0.5, 0.5)
        return {}

    def fake_training_gate_score_estimate(**kwargs: object) -> dict[str, object]:
        return _score_estimate(
            check=cast(int, kwargs["validation_check"]),
            step=cast(int, kwargs["step"]),
            score=1.0,
            log2_volume=math.log2(10),
            accepted_mass=0.5,
        )

    def fake_batch_max_compute(**_kwargs: object) -> int:
        return 10

    benchmark = load_digits_benchmark(_digits_benchmark_root)
    outcome_ids = tuple(
        outcome.id for outcome in benchmark.manifest.resolve_outcome_space().outcomes
    )
    monkeypatch.setattr(benchmark_runner, "_batch_tensors", fake_batch_tensors)
    monkeypatch.setattr(
        benchmark_runner,
        "softmax_target_mass_tensor",
        fake_target_mass_tensor,
    )
    monkeypatch.setattr(benchmark_runner, "softmax_target_masses", fake_target_masses)
    monkeypatch.setattr(
        benchmark_runner,
        "_sampled_competence_record_from_accepted_mass",
        fake_sampled_record,
    )
    monkeypatch.setattr(
        benchmark_runner.ValidationCompetencePoint,
        "from_sampled_record",
        staticmethod(lambda *_args, **_kwargs: None),
    )
    def fake_accumulate(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        benchmark_runner,
        "_accumulate_replay_frontier_point",
        fake_accumulate,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_batch_max_inference_compute",
        fake_batch_max_compute,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_batch_max_training_compute_per_sample",
        fake_batch_max_compute,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_training_gate_score_estimate",
        fake_training_gate_score_estimate,
    )

    cast(Any, benchmark_runner)._train_until_convergence(
        runtime=resolve_tensor_runtime("cpu"),
        module=FakeModule(),
        optimizer=FakeOptimizer(),
        scheduler=None,
        loss_function=FakeLossFunction(),
        train_batch=fake_batch,
        validation_batch=cast(Any, fake_validation_batch),
        outcome_space=benchmark.manifest.resolve_outcome_space(),
        outcome_ids=outcome_ids,
        max_steps=4,
        gate_check_interval=2,
        patience=5,
        min_delta=0.001,
        rung_competence_threshold=0.9,
        architecture=ArchitectureManifestDocument.from_bytes(
            _digits_architecture.read_bytes()
        ).manifest,
        training_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        training_compute_counter=cast(Any, benchmark_runner)._ComputeCounter(),
        validation_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        phase_timings=benchmark_runner.TimingCollector(),
    )

    assert event_log == [
        "gate",
        "step",
        "step",
        "tolist",
        "tolist",
        "gate",
        "step",
        "step",
        "tolist",
        "tolist",
        "gate",
    ]


def test_training_compute_per_sample_is_cached_by_architecture_and_input_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    architecture = ArchitectureManifestDocument.from_bytes(
        _digits_architecture.read_bytes()
    ).manifest
    fields = SimpleNamespace(shape=(4, 1, 16, 16))
    calls = 0

    def fake_summary(_architecture: ArchitectureManifest) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        return SimpleNamespace(training_compute_per_sample=123)

    cache = cast(
        dict[tuple[str, tuple[int, ...]], int | None],
        benchmark_runner.__dict__["_training_compute_per_sample_cache"],
    )
    cache.clear()
    monkeypatch.setattr(
        benchmark_runner,
        "summarize_architecture_operators",
        fake_summary,
    )

    first = cast(Any, benchmark_runner)._batch_max_training_compute_per_sample(
        architecture=architecture,
        fields=fields,
    )
    second = cast(Any, benchmark_runner)._batch_max_training_compute_per_sample(
        architecture=architecture,
        fields=fields,
    )

    assert first == 123
    assert second == 123
    assert calls == 1


def test_training_rung_schedule_alternates_frontier_and_replay() -> None:
    rung_index = cast(Any, benchmark_runner)._training_rung_index_for_step

    assert [rung_index(step=step, frontier_index=0) for step in range(4)] == [
        0,
        0,
        0,
        0,
    ]
    assert [rung_index(step=step, frontier_index=3) for step in range(10)] == [
        3,
        0,
        3,
        1,
        3,
        2,
        3,
        0,
        3,
        1,
    ]


def test_checkpoint_selection_uses_global_training_score_estimate() -> None:
    higher_global_estimate = _score_estimate(
        check=1,
        step=32,
        score=100.0,
        log2_volume=20.0,
        accepted_mass=0.2,
    )
    higher_current_rung_estimate = _score_estimate(
        check=2,
        step=64,
        score=1.0,
        log2_volume=20.0,
        accepted_mass=0.8,
    )
    selected = cast(Any, benchmark_runner)._selected_model_checkpoint(
        (
            SimpleNamespace(
                score_estimate=higher_global_estimate,
                validation_loss=0.1,
                validation_check=1,
                step=32,
            ),
            SimpleNamespace(
                score_estimate=higher_current_rung_estimate,
                validation_loss=1.0,
                validation_check=2,
                step=64,
            ),
        )
    )

    assert selected.step == 32


def test_checkpoint_write_gate_uses_global_training_score_estimate() -> None:
    higher_saved_global_estimate = _score_estimate(
        check=1,
        step=32,
        score=100.0,
        log2_volume=20.0,
        accepted_mass=0.2,
    )
    lower_global_higher_current_rung_estimate = _score_estimate(
        check=2,
        step=64,
        score=1.0,
        log2_volume=20.0,
        accepted_mass=0.8,
    )
    training_run = SimpleNamespace(
        validation_history=(
            TrainingHistoryPoint(
                step=64,
                validation_check=2,
                validation_loss=1.0,
                stale_checks=0,
                score_estimate=lower_global_higher_current_rung_estimate,
            ),
        )
    )

    assert not cast(Any, benchmark_runner)._should_write_model_checkpoint(
        training_run=training_run,
        gate_interval=1,
        checkpoint_artifacts=(
            SimpleNamespace(score_estimate=higher_saved_global_estimate),
        ),
    )


def test_training_plateau_signal_uses_current_rung_competence() -> None:
    score_estimate = _score_estimate(
        check=1,
        step=32,
        score=20.0,
        log2_volume=20.0,
        accepted_mass=0.55,
    )
    sampled_competence = cast(dict[str, object], score_estimate["sampled_competence"])
    points = cast(list[dict[str, object]], sampled_competence["points"])
    points.insert(
        0,
        {
            "kind": "sampled-state-space-volume-window",
            "sampling_rule": "generator-uniform-component-index-v1",
            "difficulty_assumption": "approximately-uniform-within-volume-window",
            "benchmark_id": "benchmarks.digits@0.1.0",
            "volume_axis": None,
            "log2_volume": math.log2(10),
            "seed": 101,
            "sample_count": 64,
            "mean_accepted_mass": 1.0,
        },
    )

    assert math.isclose(
        cast(Any, benchmark_runner)._training_score_estimate_frontier_competence(
            score_estimate,
            chance_mass=0.1,
        ),
        0.5,
    )


def test_training_rung_threshold_uses_current_rung_competence() -> None:
    score_estimate = _score_estimate(
        check=1,
        step=32,
        score=20.0,
        log2_volume=20.0,
        accepted_mass=0.19,
    )
    sampled_competence = cast(dict[str, object], score_estimate["sampled_competence"])
    points = cast(list[dict[str, object]], sampled_competence["points"])
    points.insert(
        0,
        {
            "kind": "sampled-state-space-volume-window",
            "sampling_rule": "generator-uniform-component-index-v1",
            "difficulty_assumption": "approximately-uniform-within-volume-window",
            "benchmark_id": "benchmarks.digits@0.1.0",
            "volume_axis": None,
            "log2_volume": math.log2(10),
            "seed": 101,
            "sample_count": 64,
            "mean_accepted_mass": 1.0,
        },
    )
    history = (
        TrainingHistoryPoint(
            step=32,
            validation_check=1,
            validation_loss=1.0,
            stale_checks=0,
            score_estimate=score_estimate,
        ),
    )

    assert math.isclose(
        cast(Any, benchmark_runner)._training_history_best_competence_fraction(
            history,
            chance_mass=0.1,
        ),
        0.1,
    )


def test_training_curriculum_can_advance_after_worse_loss_on_larger_rung() -> None:
    first_rung_point = ValidationCompetencePoint(
        log2_volume=math.log2(10),
        accepted_mass=1.0,
    )
    larger_rung_point = ValidationCompetencePoint(
        log2_volume=40.0,
        accepted_mass=0.28,
    )

    assert validation_competence_frontier_advances(
        frontier_point=larger_rung_point,
        previous_frontier_points=(first_rung_point,),
        chance_mass=0.1,
    )


def test_training_plateau_below_rung_competence_threshold_converges_without_advancing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_batch(_index: int) -> tuple[object, object]:
        return object(), object()

    class FakeLossValue:
        def item(self) -> float:
            return 2.0

        def backward(self) -> None:
            pass

    class FakeLossFunction:
        def __call__(self, _logits: object, _labels: object) -> FakeLossValue:
            return FakeLossValue()

    class FakeModule:
        training = True

        def __call__(self, _fields: object) -> object:
            return object()

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

    class FakeValidationBatch:
        sample_count = 2

    class FakeOptimizer:
        param_groups = [{"lr": 0.01}]

        def zero_grad(self, *, set_to_none: bool) -> None:
            _ = set_to_none

        def step(self) -> None:
            pass

    def fake_validation_batch(_index: int) -> FakeValidationBatch:
        return FakeValidationBatch()

    def fake_batch_tensors(**_kwargs: object) -> tuple[object, object]:
        return object(), object()

    def fake_target_masses(
        _runtime: object,
        _logits: object,
        _labels: object,
    ) -> list[float]:
        return [1.0]

    def fake_training_gate_score_estimate(**kwargs: object) -> dict[str, object]:
        return _score_estimate(
            check=cast(int, kwargs["validation_check"]),
            step=cast(int, kwargs["step"]),
            score=1.0,
            log2_volume=math.log2(10),
            accepted_mass=0.5,
        )

    def fake_batch_max_compute(**_kwargs: object) -> int:
        return 10

    def fail_if_advancing(_history: object) -> bool:
        raise AssertionError("frontier should not advance below competence threshold")

    benchmark = load_digits_benchmark(_digits_benchmark_root)
    outcome_ids = tuple(
        outcome.id for outcome in benchmark.manifest.resolve_outcome_space().outcomes
    )
    monkeypatch.setattr(benchmark_runner, "_batch_tensors", fake_batch_tensors)
    monkeypatch.setattr(benchmark_runner, "softmax_target_masses", fake_target_masses)
    monkeypatch.setattr(
        benchmark_runner,
        "_batch_max_inference_compute",
        fake_batch_max_compute,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_batch_max_training_compute_per_sample",
        fake_batch_max_compute,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_training_gate_score_estimate",
        fake_training_gate_score_estimate,
    )

    stage_result = cast(Any, benchmark_runner)._train_until_convergence(
        runtime=resolve_tensor_runtime("cpu"),
        module=FakeModule(),
        optimizer=FakeOptimizer(),
        scheduler=None,
        loss_function=FakeLossFunction(),
        train_batch=fake_batch,
        validation_batch=cast(Any, fake_validation_batch),
        outcome_space=benchmark.manifest.resolve_outcome_space(),
        outcome_ids=outcome_ids,
        max_steps=10,
        gate_check_interval=1,
        patience=1,
        min_delta=0.001,
        rung_competence_threshold=0.9,
        architecture=ArchitectureManifestDocument.from_bytes(
            _digits_architecture.read_bytes()
        ).manifest,
        training_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        training_compute_counter=cast(Any, benchmark_runner)._ComputeCounter(),
        validation_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        phase_timings=benchmark_runner.TimingCollector(),
        on_plateau=fail_if_advancing,
    )

    assert stage_result.stop_reason == "validation-plateau"
    assert len(stage_result.validation_history) == 2


def test_training_plateau_above_rung_competence_threshold_advances_frontier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_batch(_index: int) -> tuple[object, object]:
        return object(), object()

    class FakeLossValue:
        def item(self) -> float:
            return 2.0

        def backward(self) -> None:
            pass

    class FakeLossFunction:
        def __call__(self, _logits: object, _labels: object) -> FakeLossValue:
            return FakeLossValue()

    class FakeModule:
        training = True

        def __call__(self, _fields: object) -> object:
            return object()

        def eval(self) -> None:
            self.training = False

        def train(self) -> None:
            self.training = True

    class FakeValidationBatch:
        sample_count = 2

    class FakeOptimizer:
        param_groups: list[dict[str, object]] = [{}]

        def zero_grad(self, *, set_to_none: bool) -> None:
            _ = set_to_none

        def step(self) -> None:
            pass

    def fake_validation_batch(_index: int) -> FakeValidationBatch:
        return FakeValidationBatch()

    def fake_batch_tensors(**_kwargs: object) -> tuple[object, object]:
        return object(), object()

    def fake_target_masses(
        _runtime: object,
        _logits: object,
        _labels: object,
    ) -> list[float]:
        return [1.0]

    def fake_training_gate_score_estimate(**kwargs: object) -> dict[str, object]:
        return _score_estimate(
            check=cast(int, kwargs["validation_check"]),
            step=cast(int, kwargs["step"]),
            score=1.0,
            log2_volume=math.log2(10),
            accepted_mass=0.6,
        )

    def fake_batch_max_compute(**_kwargs: object) -> int:
        return 10

    plateau_history_lengths: list[int] = []

    def advance_frontier(history: object) -> bool:
        plateau_history_lengths.append(len(cast(tuple[object, ...], history)))
        return True

    benchmark = load_digits_benchmark(_digits_benchmark_root)
    outcome_ids = tuple(
        outcome.id for outcome in benchmark.manifest.resolve_outcome_space().outcomes
    )
    monkeypatch.setattr(benchmark_runner, "_batch_tensors", fake_batch_tensors)
    monkeypatch.setattr(benchmark_runner, "softmax_target_masses", fake_target_masses)
    monkeypatch.setattr(
        benchmark_runner,
        "_batch_max_inference_compute",
        fake_batch_max_compute,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_batch_max_training_compute_per_sample",
        fake_batch_max_compute,
    )
    monkeypatch.setattr(
        benchmark_runner,
        "_training_gate_score_estimate",
        fake_training_gate_score_estimate,
    )

    stage_result = cast(Any, benchmark_runner)._train_until_convergence(
        runtime=resolve_tensor_runtime("cpu"),
        module=FakeModule(),
        optimizer=FakeOptimizer(),
        scheduler=None,
        loss_function=FakeLossFunction(),
        train_batch=fake_batch,
        validation_batch=cast(Any, fake_validation_batch),
        outcome_space=benchmark.manifest.resolve_outcome_space(),
        outcome_ids=outcome_ids,
        max_steps=2,
        gate_check_interval=1,
        patience=1,
        min_delta=0.001,
        rung_competence_threshold=0.5,
        architecture=ArchitectureManifestDocument.from_bytes(
            _digits_architecture.read_bytes()
        ).manifest,
        training_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        training_compute_counter=cast(Any, benchmark_runner)._ComputeCounter(),
        validation_counter=cast(Any, benchmark_runner)._ThroughputCounter(),
        phase_timings=benchmark_runner.TimingCollector(),
        on_plateau=advance_frontier,
    )

    assert stage_result.stop_reason == "max-steps"
    assert len(stage_result.validation_history) == 3
    assert plateau_history_lengths == [2]


def test_volume_curriculum_planner_emits_integer_windows() -> None:
    planner = cast(Any, benchmark_runner)._VolumeCurriculumPlanner()

    windows = [planner.next() for _ in range(4)]

    assert [(window.minimum, window.maximum, window.log2_volume) for window in windows] == [
        (0.0, 1.0, 1.0),
        (1.0, 2.0, 2.0),
        (2.0, 3.0, 3.0),
        (3.0, 4.0, 4.0),
    ]
    assert [window.request.to_record() for window in windows] == [
        StateSpaceVolumeRequest(minimum=0.0, maximum=1.0).to_record(),
        StateSpaceVolumeRequest(minimum=1.0, maximum=2.0).to_record(),
        StateSpaceVolumeRequest(minimum=2.0, maximum=3.0).to_record(),
        StateSpaceVolumeRequest(minimum=3.0, maximum=4.0).to_record(),
    ]


def test_training_curriculum_integer_window_rematerializes() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    window_request = StateSpaceVolumeRequest(minimum=8.0, maximum=9.0)
    window_sample = generator(
        shape=1,
        seed=123,
        include_fields=True,
        volume_request=window_request,
    )

    assert window_sample.sample_count == 1
    assert window_sample.volume_request == window_request
    assert window_sample.log2_volume == 8.0
    assert window_sample.samples[0].require_field().shape[1:] == (48, 48)


def test_digits_integer_window_materializes_setup_increment() -> None:
    generator = load_digits_generator(_digits_benchmark_root)
    generator_impl = cast(Any, generator)

    volume_class = generator_impl._volume_class_for_request(
        request=StateSpaceVolumeRequest(
            minimum=8.0,
            maximum=9.0,
        )
    )

    assert volume_class is not None
    assert volume_class.cardinality == 256
    assert math.isclose(volume_class.log2_volume, 8.0)
    metadata = volume_class.metadata()
    assert metadata["kind"] == "digits-realized-setup-window"
    assert metadata["minimum_address"] == 255
    assert metadata["maximum_address"] == 510
    assert metadata["cardinality"] == 256
    assert metadata["realized_cardinality"] == 256
    assert metadata["maximum_transform_ordinal"] == 51
    assert metadata["canvas_side"] == 48
    assert metadata["transform_axes"] == ["x_translation", "y_translation", "scale"]
    assert metadata["construction"] == (
        "digit-setups-over-shell-ordered-transform-lattice"
    )
    resolution_assignment = volume_class.resolution_assignment(
        width_axis=generator.formation.width_axis,
        height_axis=generator.formation.height_axis,
    )
    assert resolution_assignment.values == {
        "W": 48,
        "H": 48,
    }


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

    assert calls == ["mps", "cpu"]
    assert training_summary["tensor_device"] == "cpu"
    assert throughput["tensor_device"] == "cpu"
    assert cast(dict[str, object], throughput["evaluation"])["kind"] == "evaluation-throughput"
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

    assert calls == ["mps"]
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
        seed=401,
        train_steps=10,
    )
    alternate_plan = BenchmarkRunPlan(
        architecture_path=_digits_architecture,
        benchmark_root=_digits_benchmark_root,
        seed=401,
        train_steps=10,
        optimizer="adam",
        learning_rate=0.01,
        schedule="reduce-on-plateau",
    )

    assert base_plan.run_slug.startswith("seed401-steps10-train-")
    assert alternate_plan.run_slug.startswith("seed401-steps10-train-")
    assert "samples" not in base_plan.run_slug
    assert "samples" not in alternate_plan.run_slug
    assert base_plan.run_slug != alternate_plan.run_slug


def test_digits_benchmark_runner_outputs_feed_benchmark_result_views(tmp_path: Path) -> None:
    training_summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            seed=101,
            train_steps=1,
            tensor_device="cpu",
        )
    )
    evaluate_benchmark_checkpoint(
        BenchmarkEvaluationPlan(
            checkpoint_artifact_path=_selected_checkpoint_artifact_path(
                training_summary.training_summary_path
            ),
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
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
    plot_runs = cast(list[dict[str, object]], result["plot_runs"])
    assert history[0]["source_kind"] == "local-run"
    assert [run["result_status"] for run in plot_runs] == ["accepted"]
    assert "model_inspection_digest" in history[0]
    assert "model_inspection_path" in history[0]
    assert "sampled_competence" in history[0]
    diagnostics = cast(dict[str, object], history[0]["training_diagnostics"])
    assert isinstance(diagnostics["status"], str)
    validation_history = cast(list[dict[str, object]], diagnostics["validation_history"])
    assert validation_history
    cost_summary = cast(dict[str, object], history[0]["cost_summary"])
    assert cost_summary["training_compute"] == 925696.0
    assert "training_compute_per_sample" not in cost_summary
    console_view_model = cast(dict[str, object], history[0]["console_view_model"])
    detail_sections = cast(list[dict[str, object]], console_view_model["detail_sections"])
    assert [section["title"] for section in detail_sections] == [
        "Sampled Competence",
        "Training Protocol",
        "Training Outcome",
        "Throughput",
        "Validation History",
    ]
    inspections = cast(list[dict[str, object]], result["model_inspections"])
    artifact_kinds = {
        artifact["kind"]
        for artifact in cast(list[dict[str, object]], inspections[0]["artifacts"])
    }
    assert artifact_kinds == {
        "measurement-dataset",
        "model-checkpoint",
        "model-inspection",
        "model-manifest",
    }
    leaderboard = cast(list[dict[str, object]], result["leaderboard"])
    score_integral = cast(dict[str, object], leaderboard[0]["score_integral"])
    assert score_integral["kind"] == "sampled-competence-integral"
    assert math.isclose(
        cast(float, score_integral["value"]),
        cast(float, leaderboard[0]["score"]),
    )
    score_terms = cast(list[dict[str, object]], score_integral["terms"])
    assert score_terms
    assert all("region" in term for term in score_terms)
    points = cast(list[dict[str, object]], leaderboard[0]["points"])
    assert all("region" in point for point in points)
    log2_volumes = [cast(float, point["log2_volume"]) for point in points]
    assert math.isclose(log2_volumes[0], 0.0)
    assert log2_volumes == sorted(log2_volumes)
    assert points[0]["sample_count"] == 64
    assert len(inspections) == 1
    assert inspections[0]["source_path"] == history[0]["model_inspection_path"]
    assert "measurement_dataset" in inspections[0]


def _selected_checkpoint_artifact_path(training_summary_path: Path) -> Path:
    training_summary = load_object_document(
        training_summary_path.read_bytes(),
        description="training summary",
    )
    checkpoint = cast(dict[str, object], training_summary["selected_model_checkpoint"])
    path = Path(cast(str, checkpoint["record_path"]))
    if path.parts[:1] == ("results",):
        return training_summary_path.parents[2] / path.relative_to("results")
    return path


def test_digits_benchmark_runner_keeps_running_training_out_of_result_views(
    tmp_path: Path,
) -> None:
    progress_records: list[Mapping[str, object]] = []

    def refresh_progress(_summary: BenchmarkRunSummary) -> None:
        progress_records.append(
            load_object_document(
                _summary.training_summary_path.read_bytes(),
                description="training progress",
            )
        )

    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            seed=101,
            train_steps=2,
            gate_check_interval=1,
            model_checkpoint_gate_interval=1,
            convergence_patience=0,
            convergence_min_delta=0.0,
            tensor_device="cpu",
        ),
        progress_callback=refresh_progress,
    )

    assert progress_records
    assert progress_records[0]["format"] == "leibniz.benchmark-training-progress"
    assert progress_records[0]["run_status"] == "running"
    progress_view_summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )
    progress_view = load_console_result_view(progress_view_summary.view_file.read_bytes())
    progress_result = cast(list[dict[str, object]], progress_view["benchmark_results"])[0]
    assert cast(list[dict[str, object]], progress_result["leaderboard"]) == []
    assert cast(dict[str, object], progress_result["frontiers"])["cost"] == []
    progress_plot_runs = cast(list[dict[str, object]], progress_result["plot_runs"])
    assert len(progress_plot_runs) == 1
    assert progress_plot_runs[0]["result_status"] == "provisional"
    assert progress_plot_runs[0]["source_kind"] == "local-training-estimate"
    progress_cost = cast(dict[str, object], progress_plot_runs[0]["cost_summary"])
    assert isinstance(progress_cost["cost"], int | float)
    training_estimate = cast(dict[str, object], progress_records[0]["training_estimate"])
    training_cost_integral = cast(dict[str, object], training_estimate["cost_integral"])
    assert training_cost_integral["kind"] == "compute-cost-integral"
    assert math.isclose(
        cast(float, training_estimate["cost"]),
        cast(float, training_cost_integral["value"]),
    )
    assert math.isclose(
        cast(float, progress_cost["cost"]),
        cast(float, training_estimate["cost"]),
    )
    materialized_progress_record = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training progress",
    )
    materialized_training_estimate = cast(
        dict[str, object],
        materialized_progress_record["training_estimate"],
    )
    assert isinstance(training_estimate["max_inference_compute"], int)
    sampled_competence = cast(dict[str, object], training_estimate["sampled_competence"])
    sampled_points = cast(list[dict[str, object]], sampled_competence["points"])
    assert training_estimate["seed"] == sampled_points[0]["seed"]
    assert "measurement_ids" not in sampled_points[0]
    assert "observation_ids" not in sampled_points[0]
    assert math.isclose(
        cast(float, progress_plot_runs[0]["score"]),
        cast(float, materialized_training_estimate["score"]),
    )
    progress_candidates = cast(list[dict[str, object]], progress_result["model_candidates"])
    assert len(progress_candidates) == 1
    assert len(cast(list[dict[str, object]], progress_candidates[0]["points"])) == len(
        sampled_points
    )
    final_record = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )
    assert final_record["format"] == "leibniz.benchmark-run"
    assert final_record["run_status"] == "completed"
    final_view_summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=tmp_path / "results",
    )
    final_view = load_console_result_view(final_view_summary.view_file.read_bytes())
    final_result = cast(list[dict[str, object]], final_view["benchmark_results"])[0]
    assert cast(list[dict[str, object]], final_result["leaderboard"]) == []
    final_plot_runs = cast(list[dict[str, object]], final_result["plot_runs"])
    assert len(final_plot_runs) == 1
    assert final_plot_runs[0]["result_status"] == "provisional"
    final_estimate = cast(dict[str, object], final_record["training_estimate"])
    final_cost_summary = cast(dict[str, object], final_record["cost_summary"])
    assert math.isclose(
        cast(float, final_plot_runs[0]["score"]),
        cast(float, final_estimate["score"]),
    )
    assert math.isclose(
        cast(float, final_cost_summary["cost"]),
        cast(float, final_estimate["cost"]),
    )


def test_digits_benchmark_runner_omits_legacy_component_count(
    tmp_path: Path,
) -> None:
    summary = run_benchmark(
        BenchmarkRunPlan(
            architecture_path=_digits_architecture,
            benchmark_root=_digits_benchmark_root,
            results_root=tmp_path / "results",
            train_steps=0,
            tensor_device="cpu",
        )
    )

    training_summary = load_object_document(
        summary.training_summary_path.read_bytes(),
        description="training summary",
    )
    assert "component_count" not in training_summary


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
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert captured.out.startswith(
        "planned benchmark training run "
        "digits-arch-186021388794-seed101-stepsconverge-train-"
    )
    assert "-samples" not in captured.out
    assert not (tmp_path / "results").exists()


def _history_point(
    *,
    check: int,
    step: int,
    loss: float,
    score: float = 0.0,
    accepted_mass: float = 0.5,
    stale_checks: int = 0,
) -> TrainingHistoryPoint:
    return TrainingHistoryPoint(
        step=step,
        validation_check=check,
        validation_loss=loss,
        stale_checks=stale_checks,
        score_estimate=_score_estimate(
            check=check,
            step=step,
            score=score,
            log2_volume=1.0,
            accepted_mass=accepted_mass,
        ),
    )


def _score_estimate(
    *,
    check: int,
    step: int,
    score: float,
    log2_volume: float,
    accepted_mass: float,
) -> dict[str, object]:
    sampled_competence = {
        "kind": "sampled-competence-curriculum",
        "sampling_rule": "generator-uniform-component-index-v1",
        "difficulty_assumption": "approximately-uniform-within-volume-window",
        "benchmark_id": "benchmarks.digits@0.1.0",
        "volume_axis": None,
        "log2_volume": log2_volume,
        "sample_count": 2,
        "mean_accepted_mass": accepted_mass,
        "points": [
            {
                "kind": "sampled-state-space-volume-window",
                "sampling_rule": "generator-uniform-component-index-v1",
                "difficulty_assumption": "approximately-uniform-within-volume-window",
                "benchmark_id": "benchmarks.digits@0.1.0",
                "volume_axis": None,
                "log2_volume": log2_volume,
                "seed": 101,
                "sample_count": 2,
                "mean_accepted_mass": accepted_mass,
                "input_shape": [1, 16, 16],
            }
        ],
    }
    return {
        "kind": "training-running-score-estimate",
        "status": "provisional",
        "evidence_status": "not-accepted",
        "score_frame": "none",
        "scoring_recipe": "sampled-competence-v1",
        "score": score,
        "validation_check": check,
        "step": step,
        "max_inference_compute": 10,
        "running_max_inference_compute": 10,
        "training_compute_per_sample": 10,
        "chance_mass": 0.1,
        "score_integral": {
            "kind": "sampled-competence-integral",
            "value": score,
            "terms": [
                {
                    "kind": "measured-state-space-competence",
                    "log2_volume_minimum": 0.0,
                    "log2_volume_maximum": log2_volume,
                    "width_in_bits": log2_volume,
                    "competence_density": 0.0 if log2_volume == 0.0 else score / log2_volume,
                    "contribution": score,
                    "representative_log2_volume": log2_volume,
                    "sample_count": 2,
                }
            ],
        },
        "sampled_competence": sampled_competence,
    }
