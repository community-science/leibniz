from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

from leibniz.benchmark_implementations import load_benchmark
from leibniz.benchmark_runner import (
    _field_valued_model_trajectory,  # pyright: ignore[reportPrivateUsage]
)
from leibniz.observation_generation import StateSpaceVolumeRequest
from leibniz.program_graphs import load_program_graph
from leibniz.tensor_runtime import resolve_tensor_runtime

_repository_root = Path(__file__).parents[1]
_ks_benchmark_root = _repository_root / "src/leibniz/benchmarks/ks"
_program_root = _repository_root / "tests/fixtures/programs"
_ks_horizon = 1.0
_ks_time_count = 9


def test_ks_submission_ladder_program_graphs_load_and_score() -> None:
    runtime = resolve_tensor_runtime("cpu")
    benchmark = load_benchmark(_ks_benchmark_root)
    batch = cast(Any, benchmark.generator)(
        seed=17,
        shape=2,
        sample_indices=(0, 1),
        volume_request=StateSpaceVolumeRequest(0.0, 1.0),
        runtime=runtime,
    )
    fields, targets = batch.require_tensors()
    competence = cast(Any, benchmark).build_training_competence(
        runtime,
        benchmark.target_contract,
    )
    scores = {
        name: _score_program(
            runtime=runtime,
            benchmark=benchmark,
            competence=competence,
            batch=batch,
            fields=fields,
            targets=targets,
            program_path=_program_root / path,
        )
        for name, path in {
            "persistence": "ks_persistence.py",
            "partial": "ks_partial_dynamics.py",
            "solver": "ks_spectral_solver.py",
        }.items()
    }

    assert scores["persistence"][0] == 0.0
    assert scores["partial"][0] > scores["persistence"][0]
    assert scores["solver"][0] > scores["partial"][0]
    assert scores["solver"][1] > scores["persistence"][1]


def test_ks_learned_predictor_trains_against_label_free_residual() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    torch.manual_seed(353)
    benchmark = load_benchmark(_ks_benchmark_root)
    batch = cast(Any, benchmark.generator)(
        seed=353,
        shape=4,
        sample_indices=(0, 1, 2, 3),
        volume_request=StateSpaceVolumeRequest(0.0, 1.0),
        runtime=runtime,
    )
    fields, targets = batch.require_tensors()
    module = _load_program_module(
        runtime=runtime,
        program_path=_program_root / "ks_learned_residual.py",
    )
    loss_function = cast(Any, benchmark).build_training_loss(
        runtime,
        benchmark.target_contract,
    )

    initial_loss = _residual_training_loss(
        runtime=runtime,
        module=module,
        fields=fields,
        targets=targets,
        loss_function=loss_function,
    )
    _train_learned_module(
        runtime=runtime,
        module=module,
        fields=fields,
        targets=targets,
        loss_function=loss_function,
    )
    final_loss = _residual_training_loss(
        runtime=runtime,
        module=module,
        fields=fields,
        targets=targets,
        loss_function=loss_function,
    )

    assert float(final_loss.detach()) < float(initial_loss.detach())


def test_ks_certified_bits_climb_with_predictor_capability() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    torch.manual_seed(353)
    benchmark = load_benchmark(_ks_benchmark_root)
    batch = cast(Any, benchmark.generator)(
        seed=17,
        shape=2,
        sample_indices=(0, 1),
        volume_request=StateSpaceVolumeRequest(0.0, 1.0),
        runtime=runtime,
    )
    fields, targets = batch.require_tensors()
    competence = cast(Any, benchmark).build_training_competence(
        runtime,
        benchmark.target_contract,
    )
    learned = _load_program_module(
        runtime=runtime,
        program_path=_program_root / "ks_learned_residual.py",
    )
    loss_function = cast(Any, benchmark).build_training_loss(
        runtime,
        benchmark.target_contract,
    )
    _train_learned_module(
        runtime=runtime,
        module=learned,
        fields=fields,
        targets=targets,
        loss_function=loss_function,
    )
    scores = {
        "persistence": _score_program(
            runtime=runtime,
            benchmark=benchmark,
            competence=competence,
            batch=batch,
            fields=fields,
            targets=targets,
            program_path=_program_root / "ks_persistence.py",
        ),
        "partial": _score_program(
            runtime=runtime,
            benchmark=benchmark,
            competence=competence,
            batch=batch,
            fields=fields,
            targets=targets,
            program_path=_program_root / "ks_partial_dynamics.py",
        ),
        "learned": _score_module(
            runtime=runtime,
            benchmark=benchmark,
            competence=competence,
            batch=batch,
            fields=fields,
            targets=targets,
            module=learned,
        ),
        "solver": _score_program(
            runtime=runtime,
            benchmark=benchmark,
            competence=competence,
            batch=batch,
            fields=fields,
            targets=targets,
            program_path=_program_root / "ks_spectral_solver.py",
        ),
    }

    assert scores["persistence"][0] == 0.0
    assert scores["persistence"][1] == 0.0
    assert scores["persistence"][0] < scores["partial"][0] < scores["learned"][0]
    assert scores["learned"][0] <= scores["solver"][0]
    assert scores["learned"][1] == _ks_horizon
    assert scores["solver"][1] == _ks_horizon


def _score_program(
    *,
    runtime: Any,
    benchmark: Any,
    competence: Any,
    batch: Any,
    fields: Any,
    targets: Any,
    program_path: Path,
) -> tuple[float, float]:
    module = _load_program_module(runtime=runtime, program_path=program_path)
    return _score_module(
        runtime=runtime,
        benchmark=benchmark,
        competence=competence,
        batch=batch,
        fields=fields,
        targets=targets,
        module=module,
    )


def _load_program_module(*, runtime: Any, program_path: Path) -> Any:
    program = load_program_graph(program_path, runtime)
    return program.graph.build_module(runtime)


def _score_module(
    *,
    runtime: Any,
    benchmark: Any,
    competence: Any,
    batch: Any,
    fields: Any,
    targets: Any,
    module: Any,
) -> tuple[float, float]:
    horizons = tuple(
        _ks_horizon * index / float(_ks_time_count - 1)
        for index in range(1, _ks_time_count)
    )
    with runtime.torch.no_grad():
        trajectory = _field_valued_model_trajectory(
            runtime=runtime,
            module=module,
            fields=fields,
            labels=targets,
            horizons=horizons,
        )
        bits = competence(
            type(
                "Request",
                (),
                {
                    "runtime": runtime,
                    "module": module,
                    "generator": benchmark.generator,
                    "batch": batch,
                    "sample_keys": tuple(sample.to_record() for sample in batch.samples),
                    "predictions": trajectory,
                    "targets": targets,
                    "horizons": horizons,
                },
            )()
    )
    diagnostics = bits.leibniz_competence_diagnostics
    boundaries = [
        float(
            cast(
                float,
                cast(dict[str, object], diagnostic).get("predictability_boundary", 0.0),
            )
        )
        for diagnostic in diagnostics
    ]
    mean_bits = math.fsum(float(value) for value in bits) / int(bits.shape[0])
    mean_boundary = math.fsum(boundaries) / len(boundaries)
    return (mean_bits, mean_boundary)


def _residual_training_loss(
    *,
    runtime: Any,
    module: Any,
    fields: Any,
    targets: Any,
    loss_function: Any,
) -> Any:
    horizons = tuple(
        _ks_horizon * index / float(_ks_time_count - 1)
        for index in range(1, _ks_time_count)
    )
    trajectory = _field_valued_model_trajectory(
        runtime=runtime,
        module=module,
        fields=fields,
        labels=targets,
        horizons=horizons,
    )
    return loss_function(trajectory, targets)


def _train_learned_module(
    *,
    runtime: Any,
    module: Any,
    fields: Any,
    targets: Any,
    loss_function: Any,
) -> None:
    optimizer = runtime.torch.optim.Adam(module.parameters(), lr=0.4)
    for _step in range(12):
        optimizer.zero_grad()
        loss = _residual_training_loss(
            runtime=runtime,
            module=module,
            fields=fields,
            targets=targets,
            loss_function=loss_function,
        )
        loss.backward()
        optimizer.step()
