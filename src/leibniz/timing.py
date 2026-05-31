"""Low-overhead timing records for diagnostic instrumentation."""

from __future__ import annotations

import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field

__all__ = [
    "TimingCollector",
    "TimingCounter",
]


@dataclass(slots=True)
class TimingCounter:
    """Accumulate wall-time for one named diagnostic phase."""

    seconds: float = 0.0
    calls: int = 0
    samples: int = 0
    counters: dict[str, float] = field(default_factory=lambda: {})

    def add(
        self,
        *,
        seconds: float,
        samples: int = 0,
        counters: Mapping[str, float] | None = None,
    ) -> None:
        """Add one measured interval."""

        self.seconds += max(0.0, float(seconds))
        self.calls += 1
        self.samples += samples
        if counters is not None:
            for name, value in counters.items():
                self.counters[name] = self.counters.get(name, 0.0) + float(value)

    def increment(self, name: str, value: float = 1.0) -> None:
        """Increment one numeric diagnostic counter."""

        self.counters[name] = self.counters.get(name, 0.0) + float(value)

    def add_counters(self, counters: Mapping[str, float]) -> None:
        """Add numeric diagnostic counters without recording a timed call."""

        for name, value in counters.items():
            self.increment(name, value)

    def to_record(self, *, name: str) -> dict[str, object]:
        """Return a document-friendly timing record."""

        record: dict[str, object] = {
            "kind": "phase-timing",
            "phase": name,
            "calls": self.calls,
            "sample_count": self.samples,
            "seconds": self.seconds,
            "seconds_per_call": self.seconds / self.calls if self.calls else 0.0,
            "samples_per_second": (
                self.samples / self.seconds if self.samples > 0 and self.seconds > 0 else 0.0
            ),
        }
        if self.counters:
            record["counters"] = dict(sorted(self.counters.items()))
        return record


@dataclass(slots=True)
class TimingCollector:
    """Collect named wall-time spans and serialize them as phase timing records."""

    counters: dict[str, TimingCounter] = field(default_factory=lambda: {})

    def add(
        self,
        phase: str,
        *,
        seconds: float,
        samples: int = 0,
        counters: Mapping[str, float] | None = None,
    ) -> None:
        """Add one measured interval to a named phase."""

        counter = self.counters.setdefault(phase, TimingCounter())
        counter.add(seconds=seconds, samples=samples, counters=counters)

    def increment(self, phase: str, counter: str, value: float = 1.0) -> None:
        """Increment one numeric diagnostic counter for a named phase."""

        self.counters.setdefault(phase, TimingCounter()).increment(counter, value)

    def add_counters(self, phase: str, counters: Mapping[str, float]) -> None:
        """Add numeric diagnostic counters for a named phase."""

        self.counters.setdefault(phase, TimingCounter()).add_counters(counters)

    @contextmanager
    def span(self, phase: str, *, samples: int = 0) -> Generator[None]:
        """Measure a block with ``time.perf_counter``."""

        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(phase, seconds=time.perf_counter() - started, samples=samples)

    def to_record(self, *, kind: str = "timing-breakdown") -> dict[str, object]:
        """Return a sorted document-friendly timing breakdown."""

        return {
            "kind": kind,
            "phases": {
                phase: counter.to_record(name=phase)
                for phase, counter in sorted(self.counters.items())
            },
        }
