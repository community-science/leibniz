from typing import cast

from leibniz.timing import TimingCollector


def test_timing_collector_accumulates_named_spans() -> None:
    timing = TimingCollector()

    timing.add("phase_b", seconds=0.25, samples=5)
    timing.add("phase_b", seconds=0.25, samples=5, counters={"items": 2})
    timing.increment("phase_b", "items", 3)
    with timing.span("phase_a", samples=1):
        pass

    record = timing.to_record(kind="diagnostic-timing")
    phases = cast(dict[str, object], record["phases"])
    phase_b = cast(dict[str, object], phases["phase_b"])
    assert record["kind"] == "diagnostic-timing"
    assert list(phases) == ["phase_a", "phase_b"]
    assert phase_b["calls"] == 2
    assert phase_b["sample_count"] == 10
    assert phase_b["seconds"] == 0.5
    assert phase_b["seconds_per_call"] == 0.25
    assert phase_b["samples_per_second"] == 20.0
    assert phase_b["counters"] == {"items": 5.0}
