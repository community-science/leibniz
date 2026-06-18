from __future__ import annotations

import math
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from ks_oracle import ks_reference_step

from leibniz.benchmark_implementations import load_benchmark
from leibniz.benchmark_runner import (
    BenchmarkEvaluationPlan,
    BenchmarkRunnerError,
    BenchmarkRunPlan,
    evaluate_benchmark_checkpoint,
    run_benchmark,
)
from leibniz.cli import main
from leibniz.documents import load_object_document
from leibniz.field_evolution import (
    FieldEvolutionError,
    field_stepper_trajectory,
    validate_field_stepper_nondegenerate,
)
from leibniz.local_results import load_console_result_view, materialize_benchmark_result_views
from leibniz.result_schema import EvaluationDocument
from leibniz.tensor_runtime import resolve_tensor_runtime

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src/leibniz/benchmarks/digits"
_ks_benchmark_root = _repository_root / "src/leibniz/benchmarks/ks"
_ks_program = _repository_root / "tests/fixtures/programs/ks_variable_conv.py"


def test_field_stepper_rollout_is_autoregressive_and_time_dependent() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    fields = torch.zeros((2, 1, 3), dtype=torch.float32, device=runtime.device)

    class AddDtStep:
        def __call__(self, state: Any, dt: float) -> Any:
            return state + float(dt)

    trajectory = field_stepper_trajectory(
        runtime=runtime,
        module=AddDtStep(),
        fields=fields,
        horizon=1.0,
        time_count=5,
    )

    assert trajectory.shape == (2, 5, 3)
    assert torch.allclose(trajectory[:, 0, :], torch.zeros((2, 3), device=runtime.device))
    assert torch.allclose(trajectory[:, 1, :], torch.full((2, 3), 0.25, device=runtime.device))
    assert torch.allclose(trajectory[:, -1, :], torch.ones((2, 3), device=runtime.device))


def test_field_stepper_nondegeneracy_rejects_identity_and_dt_insensitive_steps() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    fields = torch.zeros((1, 1, 4), dtype=torch.float32, device=runtime.device)

    class IdentityStep:
        def __call__(self, state: Any, dt: float) -> Any:
            _ = dt
            return state

    class DtInsensitiveStep:
        def __call__(self, state: Any, dt: float) -> Any:
            _ = dt
            return state + 1.0

    with pytest.raises(FieldEvolutionError, match="must not be identity"):
        validate_field_stepper_nondegenerate(
            runtime=runtime,
            module=IdentityStep(),
            fields=fields,
            dt=0.125,
        )
    with pytest.raises(FieldEvolutionError, match="must vary with dt"):
        validate_field_stepper_nondegenerate(
            runtime=runtime,
            module=DtInsensitiveStep(),
            fields=fields,
            dt=0.125,
        )


def test_ks_oracle_stepper_scores_positive_through_real_residual() -> None:
    runtime = resolve_tensor_runtime("cpu")
    loaded = cast(Any, load_benchmark(_ks_benchmark_root))
    batch = loaded.generator(
        seed=101,
        shape=1,
        sample_indices=(0,),
        runtime=runtime,
        spatial_points=32,
    )
    fields, targets = batch.require_tensors()

    class ReferenceStep:
        def __call__(self, state: Any, dt: float) -> Any:
            return ks_reference_step(runtime=runtime, fields=state, dt=float(dt)).to(
                dtype=state.dtype
            )

    predictions = field_stepper_trajectory(
        runtime=runtime,
        module=ReferenceStep(),
        fields=fields,
        horizon=1.0,
        time_count=9,
    )
    competence = loaded.implementation.build_training_competence(
        runtime,
        loaded.target_contract,
    )
    bits = competence(
        SimpleNamespace(
            runtime=runtime,
            module=ReferenceStep(),
            generator=loaded.generator,
            batch=batch,
            sample_keys=tuple(sample.to_record() for sample in batch.samples),
            predictions=predictions,
            targets=targets,
            horizons=tuple(index / 8 for index in range(1, 9)),
        )
    )
    diagnostics = bits.leibniz_competence_diagnostics
    stability = cast(Mapping[str, object], diagnostics[0]["stability"])

    assert cast(float, diagnostics[0]["residual_norm"]) >= 0.0
    assert cast(float, stability["law_amplification"]) >= 1.0
    assert cast(float, diagnostics[0]["certified_epsilon"]) < cast(
        float,
        diagnostics[0]["signal_scale"],
    )
    assert cast(float, diagnostics[0]["ambient_entropy_bits"]) > 0.0
    assert diagnostics[0]["predictability_boundary"] > 0.0
    assert float(bits[0]) > 0.0


def test_ks_certified_epsilon_bounds_imperfect_stepper_error() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    loaded = cast(Any, load_benchmark(_ks_benchmark_root))
    module = sys.modules[type(loaded.implementation).__module__]
    horizon = 0.25
    drift_scale = 1.0e-3
    imperfect_ladder: list[Any] = []
    oracle_ladder: list[Any] = []

    class DriftedReferenceStep:
        def __call__(self, state: Any, dt: float) -> Any:
            spatial_points = int(state.shape[-1])
            phase = (
                torch.arange(spatial_points, dtype=state.dtype, device=state.device).reshape(
                    1,
                    1,
                    spatial_points,
                )
                * (2.0 * math.pi / float(spatial_points))
            )
            drift = drift_scale * float(dt) * torch.sin(3.0 * phase)
            return (
                ks_reference_step(runtime=runtime, fields=state, dt=float(dt)).to(
                    dtype=state.dtype
                )
                + drift
            )

    class ReferenceStep:
        def __call__(self, state: Any, dt: float) -> Any:
            return ks_reference_step(runtime=runtime, fields=state, dt=float(dt)).to(
                dtype=state.dtype
            )

    for factor in (1, 2, 4):
        batch = loaded.generator(
            seed=101,
            shape=1,
            sample_indices=(0,),
            runtime=runtime,
            spatial_points=32 * factor,
            include_metadata=False,
        )
        fields, _targets = batch.require_tensors()
        time_count = 1 + (2 * factor)
        imperfect_ladder.append(
            field_stepper_trajectory(
                runtime=runtime,
                module=DriftedReferenceStep(),
                fields=fields,
                horizon=horizon,
                time_count=time_count,
            ).double()
        )
        oracle_ladder.append(
            field_stepper_trajectory(
                runtime=runtime,
                module=ReferenceStep(),
                fields=fields,
                horizon=horizon,
                time_count=time_count,
            ).double()
        )

    values, diagnostics = module._ks_ladder_prefix_certified_bits(
        runtime=runtime,
        ladder=tuple(imperfect_ladder),
        horizon=horizon,
    )
    error_tensor = imperfect_ladder[-1] - oracle_ladder[-1]
    actual_error = float(error_tensor.pow(2).mean().sqrt())
    certified_epsilon = cast(float, diagnostics[0]["certified_epsilon"])

    assert actual_error > 0.0
    assert certified_epsilon >= actual_error
    assert certified_epsilon / actual_error < 3.0
    assert float(values[0]) > 0.0


def test_ks_real_path_bits_rise_as_stepper_residual_falls() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    loaded = cast(Any, load_benchmark(_ks_benchmark_root))
    module = sys.modules[type(loaded.implementation).__module__]

    def score_for_drift(drift_scale: float) -> float:
        class DriftedReferenceStep:
            def __call__(self, state: Any, dt: float) -> Any:
                spatial_points = int(state.shape[-1])
                phase = (
                    torch.arange(
                        spatial_points,
                        dtype=state.dtype,
                        device=state.device,
                    ).reshape(1, 1, spatial_points)
                    * (2.0 * math.pi / float(spatial_points))
                )
                drift = drift_scale * float(dt) * torch.sin(3.0 * phase)
                return (
                    ks_reference_step(runtime=runtime, fields=state, dt=float(dt)).to(
                        dtype=state.dtype
                    )
                    + drift
                )

        ladder: list[Any] = []
        for factor in (1, 2, 4):
            batch = loaded.generator(
                seed=101,
                shape=1,
                sample_indices=(0,),
                runtime=runtime,
                spatial_points=32 * factor,
                include_metadata=False,
            )
            fields, _targets = batch.require_tensors()
            ladder.append(
                field_stepper_trajectory(
                    runtime=runtime,
                    module=DriftedReferenceStep(),
                    fields=fields,
                    horizon=0.25,
                    time_count=1 + (2 * factor),
                ).double()
            )
        values, _diagnostics = module._ks_ladder_prefix_certified_bits(
            runtime=runtime,
            ladder=tuple(ladder),
            horizon=0.25,
        )
        return float(values[0])

    assert score_for_drift(1.0e-3) > score_for_drift(1.0e-2) > 0.0


def test_program_checkpoint_evaluation_materializes_ks_result_view(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    training_summary = run_benchmark(
        BenchmarkRunPlan(
            program_path=_ks_program,
            benchmark_root=_ks_benchmark_root,
            results_root=results_root,
            seed=101,
            train_steps=0,
            gate_check_interval=1,
            model_checkpoint_gate_interval=1,
            tensor_device="cpu",
            optimizer="adam",
            learning_rate=1e-3,
        )
    )
    submission_record = load_object_document(
        training_summary.submission_record_path.read_bytes(),
        description="submission record",
    )
    assert submission_record["format"] == "leibniz.submission"
    training_provenance = cast(dict[str, object], submission_record["training_provenance"])
    training_estimate = cast(dict[str, object], training_provenance["training_estimate"])
    training_sampled_competence = cast(
        dict[str, object],
        training_estimate["sampled_competence"],
    )
    training_partition_score = cast(
        dict[str, object],
        training_sampled_competence["partition_score"],
    )
    assert training_estimate["score"] == training_partition_score["value"]
    evaluation_summary = evaluate_benchmark_checkpoint(
        BenchmarkEvaluationPlan(
            checkpoint_artifact_path=_selected_checkpoint_artifact_path(
                training_summary.submission_record_path,
                results_root=results_root,
            ),
            benchmark_root=_ks_benchmark_root,
            results_root=results_root,
            tensor_device="cpu",
        )
    )

    assert evaluation_summary.measurement_count == 0
    evaluation_record = EvaluationDocument.from_bytes(
        evaluation_summary.evaluation_record_path.read_bytes()
    ).evaluation
    assert evaluation_record.converged
    assert not evaluation_record.evidence_budget_limited
    assert evaluation_record.diagnostics
    first_diagnostic = dict(evaluation_record.diagnostics[0])
    assert first_diagnostic.get("kind") == "certified-bits-diagnostics"
    assert "certified_epsilon" in first_diagnostic
    assert cast(list[dict[str, object]], first_diagnostic["time_points"])
    view_summary = materialize_benchmark_result_views(
        repository_root=_repository_root,
        results_root=results_root,
    )
    view = load_console_result_view(view_summary.view_file.read_bytes())
    benchmark_results = cast(list[dict[str, object]], view["benchmark_results"])
    result = benchmark_results[0]
    leaderboard = cast(list[dict[str, object]], result["leaderboard"])
    plot_runs = cast(list[dict[str, object]], result["plot_runs"])

    assert result["benchmark_id"] == "benchmarks.ks@0.1.0"
    assert leaderboard
    assert [run["result_status"] for run in plot_runs] == ["accepted"]
    assert math.isfinite(cast(float, leaderboard[0]["score"]))
    assert "program_digest" in leaderboard[0]
    assert "program_graph" in plot_runs[0]
    capability_map = cast(dict[str, object], leaderboard[0]["capability_map"])
    assert leaderboard[0]["score"] == capability_map["value"]
    assert "sampled_competence" not in plot_runs[0]
    score_integral = cast(dict[str, object], leaderboard[0]["score_integral"])
    score_terms = cast(list[dict[str, object]], score_integral["terms"])
    assert "competence_density" not in score_terms[0]
    assert cast(float, capability_map["score_width_bits"]) > 0.0
    assert cast(float, capability_map["mean_competence"]) >= 0.0
    assert cast(int, capability_map["leaf_count"]) >= 1
    assert cast(list[dict[str, object]], capability_map["refinement_ladder"])


def test_cli_benchmark_train_accepts_program_flag(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    program_path = _repository_root / "tests/fixtures/programs/digits_inverse_conv_encoder.py"
    exit_code = main(
        [
            "benchmark",
            "train",
            "--program",
            str(program_path),
            "--benchmark-root",
            str(_digits_benchmark_root),
            "--results-root",
            str(tmp_path / "results"),
            "--device",
            "cpu",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "planned benchmark training run digits-program-" in output
    assert "submission record:" in output


def test_field_program_scale_violation_is_rejected_before_training(tmp_path: Path) -> None:
    program_path = tmp_path / "fixed_width_stepper.py"
    program_path.write_text(
        """\
from leibniz.program_graphs import (
    ProgramGraph,
    ProgramGraphEdge,
    ProgramGraphNode,
    ProgramTensorContract,
)


def build_program_graph(runtime):
    torch = runtime.torch
    return ProgramGraph(
        contract_kind="prediction",
        inputs=(
            ProgramTensorContract("field", (1, "S")),
            ProgramTensorContract("dt", ()),
        ),
        outputs=(ProgramTensorContract("future_field", (1, "S")),),
        nodes=(
            ProgramGraphNode(
                "pool",
                torch.nn.AdaptiveAvgPool1d(4),
                "fixed-pool",
            ),
        ),
        edges=(
            ProgramGraphEdge("field", "pool"),
            ProgramGraphEdge("pool", "future_field"),
        ),
    )
""",
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkRunnerError, match="does not match symbolic axis"):
        run_benchmark(
            BenchmarkRunPlan(
                program_path=program_path,
                benchmark_root=_ks_benchmark_root,
                results_root=tmp_path / "results",
                train_steps=0,
                tensor_device="cpu",
                dry_run=True,
                optimizer="adam",
                learning_rate=1e-3,
            )
        )


def _selected_checkpoint_artifact_path(
    submission_record_path: Path,
    *,
    results_root: Path,
) -> Path:
    submission = load_object_document(
        submission_record_path.read_bytes(),
        description="submission record",
    )
    provenance = cast(dict[str, object], submission["training_provenance"])
    checkpoint = cast(dict[str, object], provenance["selected_model_checkpoint"])
    return results_root / cast(str, checkpoint["record_path"]).removeprefix("results/")
