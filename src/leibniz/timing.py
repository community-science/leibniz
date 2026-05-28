"""Low-overhead timing records for diagnostic instrumentation."""

from __future__ import annotations

import time
from collections.abc import Generator
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

    def add(self, *, seconds: float, samples: int = 0) -> None:
        """Add one measured interval."""

        self.seconds += max(0.0, float(seconds))
        self.calls += 1
        self.samples += samples

    def to_record(self, *, name: str) -> dict[str, object]:
        """Return a document-friendly timing record."""

        return {
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


@dataclass(slots=True)
class TimingCollector:
    """Collect named wall-time spans and serialize them as phase timing records."""

    counters: dict[str, TimingCounter] = field(default_factory=lambda: {})

    def add(self, phase: str, *, seconds: float, samples: int = 0) -> None:
        """Add one measured interval to a named phase."""

        counter = self.counters.setdefault(phase, TimingCounter())
        counter.add(seconds=seconds, samples=samples)

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
