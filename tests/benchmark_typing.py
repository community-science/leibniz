"""Test-only protocols for benchmark-specific implementation surfaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, cast

from leibniz.benchmark_implementations import (
    Benchmark as BenchmarkProtocol,
)
from leibniz.benchmark_implementations import (
    Generator as BenchmarkGenerator,
)
from leibniz.benchmark_implementations import (
    load_benchmark,
)
from leibniz.latent_factors import LatentFactorDeclaration
from leibniz.materialization import (
    AxisAssignment,
    MaterializationDeclaration,
    MaterializationPlan,
)
from leibniz.observation_formation import ObservationFormationDeclaration
from leibniz.observation_generation import (
    GeneratedSample,
    GeneratedSampleSet,
    ObservationGenerationError,
    StateSpaceMeasureRequest,
    load_generator,
)
from leibniz.observation_showcases import ObservationShowcaseManifest
from leibniz.timing import TimingCollector


class DigitsGenerator(BenchmarkGenerator, Protocol):
    @property
    def materialization(self) -> MaterializationDeclaration: ...

    @property
    def formation(self) -> ObservationFormationDeclaration: ...

    def minimum_discriminatable_resolution_assignment(
        self,
        *,
        minimum_assignment: AxisAssignment,
    ) -> AxisAssignment: ...

    def distinguishable_state_complexity(
        self,
        *,
        width: int,
        height: int,
        variation_extent: float = 1.0,
    ) -> float: ...

    def __call__(
        self,
        *,
        seed: int,
        shape: int | Sequence[int] | None = None,
        include_fields: bool = False,
        state_space_request: StateSpaceMeasureRequest | None = None,
        component_indices: Iterable[int] | None = None,
        memory_limit_bytes: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
        variation_extent: float = 1.0,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedSampleSet: ...


class DigitsBenchmark(BenchmarkProtocol, Protocol):
    @property
    def latent_factors(self) -> LatentFactorDeclaration: ...

    @property
    def materialization(self) -> MaterializationDeclaration: ...

    @property
    def formation(self) -> ObservationFormationDeclaration: ...

    @property
    def showcase(self) -> ObservationShowcaseManifest: ...

    @property
    def generator(self) -> DigitsGenerator: ...


def load_digits_benchmark(root: Path) -> DigitsBenchmark:
    return cast(DigitsBenchmark, load_benchmark(root))


def load_digits_generator(root: Path) -> DigitsGenerator:
    return cast(DigitsGenerator, load_generator(root))


def sample_materialization_plan(sample: GeneratedSample) -> MaterializationPlan:
    if sample.materialization_plan is None:
        raise ObservationGenerationError("sample is missing materialization plan")
    return sample.materialization_plan


def sample_width(sample: GeneratedSample) -> int:
    if sample.width is None:
        raise ObservationGenerationError("sample is missing width")
    return sample.width


def sample_height(sample: GeneratedSample) -> int:
    if sample.height is None:
        raise ObservationGenerationError("sample is missing height")
    return sample.height


def sample_component_index(sample: GeneratedSample) -> int:
    if sample.component_index is None:
        raise ObservationGenerationError("sample is missing component index")
    return sample.component_index


def sample_variation_values(sample: GeneratedSample) -> Mapping[str, object]:
    if sample.variation_values is None:
        raise ObservationGenerationError("sample is missing variation values")
    return sample.variation_values
