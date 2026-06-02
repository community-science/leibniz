from pathlib import Path
from typing import cast

import pytest

from leibniz.formation_timing import (
    FormationTimingError,
    FormationTimingPlan,
    time_formation_paths,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def test_time_formation_paths_reports_samples_per_second() -> None:
    summary = time_formation_paths(
        FormationTimingPlan(
            benchmark_root=_digits_benchmark_root,
            component_count=1,
            sample_count=1,
            repeats=1,
            warmup_repeats=0,
            tensor_device="cpu",
        )
    )

    record = summary.to_record()
    assert record["format"] == "leibniz.formation-timing"
    assert record["benchmark_id"] == "benchmarks.digits@0.1.0"
    assert record["component_count"] == 1
    assert record["sample_count"] == 1
    assert record["repeats"] == 1
    assert record["tensor_runtime"] == "pytorch"
    assert record["tensor_device"] == "cpu"
    assert record["roofline"] != {}
    assert "pure_phase_timing" in record
    assert "tensor_phase_timing" in record
    pure_timing = cast(dict[str, object], record["pure_phase_timing"])
    pure_phases = cast(dict[str, object], pure_timing["phases"])
    assert "pure.formation_batch.variation_coordinates" in pure_phases
    assert summary.pure_observation_seconds > 0
    assert summary.tensor_batch_seconds > 0
    assert summary.pure_observation_samples_per_second > 0
    assert summary.tensor_batch_samples_per_second > 0


def test_time_formation_plan_rejects_invalid_counts() -> None:
    with pytest.raises(FormationTimingError, match="sample_count must be"):
        FormationTimingPlan(
            benchmark_root=_digits_benchmark_root,
            sample_count=0,
        )
    with pytest.raises(FormationTimingError, match="repeats must be"):
        FormationTimingPlan(
            benchmark_root=_digits_benchmark_root,
            repeats=0,
        )
