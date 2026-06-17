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
    assert scores["solver"][0] > scores["persistence"][0]
    assert scores["solver"][1] > scores["persistence"][1]
    assert scores["partial"][0] >= scores["persistence"][0]


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
    program = load_program_graph(program_path, runtime)
    module = program.graph.build_module(runtime)
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
        float(cast(dict[str, object], diagnostic).get("predictability_boundary", 0.0))
        for diagnostic in diagnostics
    ]
    mean_bits = math.fsum(float(value) for value in bits) / int(bits.shape[0])
    mean_boundary = math.fsum(boundaries) / len(boundaries)
    return (mean_bits, mean_boundary)
