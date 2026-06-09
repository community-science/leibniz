from pathlib import Path
from typing import cast

import pytest

from leibniz.formation_timing import (
    FormationOperatorProfilePlan,
    FormationTimingError,
    profile_formation_operators,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def test_time_formation_plan_rejects_invalid_counts() -> None:
    with pytest.raises(FormationTimingError, match="sample_count must be"):
        FormationOperatorProfilePlan(
            benchmark_root=_digits_benchmark_root,
            sample_count=0,
        )
    with pytest.raises(FormationTimingError, match="repeats must be"):
        FormationOperatorProfilePlan(
            benchmark_root=_digits_benchmark_root,
            repeats=0,
        )


def test_profile_formation_operators_reports_bounded_rows() -> None:
    record = profile_formation_operators(
        FormationOperatorProfilePlan(
            benchmark_root=_digits_benchmark_root,
            sample_count=1,
            repeats=1,
            warmup_repeats=0,
            tensor_device="cpu",
            row_limit=5,
        )
    )

    assert record["format"] == "leibniz.formation-operator-profile"
    assert record["benchmark_id"] == "benchmarks.digits@0.1.0"
    assert record["sample_count"] == 1
    assert record["repeats"] == 1
    assert record["tensor_device"] == "cpu"
    rows = cast(list[dict[str, object]], record["rows"])
    assert 0 < len(rows) <= 5
    assert {"name", "calls", "cpu_time_total_us"} <= rows[0].keys()
