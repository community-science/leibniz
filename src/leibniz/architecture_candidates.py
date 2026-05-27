"""Generic architecture candidate distributions for local proposal workflows."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from leibniz.architectures import ArchitectureManifest
from leibniz.model_operators import (
    ModelOperatorPlan,
    ModelOperatorSearchPoint,
    materialize_model_operator_search_point,
    summarize_architecture_operators,
)

__all__ = [
    "ArchitectureCandidate",
    "ArchitectureSearchDistribution",
    "ArchitectureSearchDistributionValidationError",
    "default_architecture_search_distribution",
    "generate_architecture_candidates",
    "sample_architecture_candidates",
]

_SizeMaximumKind = Literal["minimum-trailing-input-axis"]
_minimum_trailing_input_axis: _SizeMaximumKind = "minimum-trailing-input-axis"


class ArchitectureSearchDistributionValidationError(ValueError):
    """Raised when an architecture search distribution is invalid."""


@dataclass(frozen=True, slots=True)
class ArchitectureSearchDistribution:
    """Source-defined semantic search coordinates and resource bounds."""

    local_support_dimension: int
    local_support_size_minimum: int
    local_support_size_maximum: int | None = None
    local_support_size_maximum_from: _SizeMaximumKind | None = None
    parameter_count_minimum: int | None = None
    parameter_count_maximum: int | None = None

    def __post_init__(self) -> None:
        if type(self.local_support_dimension) is not int or self.local_support_dimension < 1:
            raise ArchitectureSearchDistributionValidationError(
                "local_support_dimension must be positive"
            )
        if (
            type(self.local_support_size_minimum) is not int
            or self.local_support_size_minimum < 1
        ):
            raise ArchitectureSearchDistributionValidationError(
                "local_support_size.minimum must be positive"
            )
        if self.local_support_size_maximum is None:
            if self.local_support_size_maximum_from != _minimum_trailing_input_axis:
                raise ArchitectureSearchDistributionValidationError(
                    "local_support_size must declare maximum or maximum_from"
                )
        elif self.local_support_size_maximum < self.local_support_size_minimum:
            raise ArchitectureSearchDistributionValidationError(
                "local_support_size.maximum must be at least minimum"
            )
        if self.local_support_size_maximum_from is not None and (
            self.local_support_size_maximum_from != _minimum_trailing_input_axis
        ):
            raise ArchitectureSearchDistributionValidationError(
                "unsupported local_support_size.maximum_from"
            )
        _require_optional_count(
            self.parameter_count_minimum,
            field="parameter_count.minimum",
        )
        _require_optional_count(
            self.parameter_count_maximum,
            field="parameter_count.maximum",
        )
        if (
            self.parameter_count_minimum is not None
            and self.parameter_count_maximum is not None
            and self.parameter_count_maximum < self.parameter_count_minimum
        ):
            raise ArchitectureSearchDistributionValidationError(
                "parameter_count.maximum must be at least minimum"
            )

    def search_points(self, input_shape: tuple[int, ...]) -> tuple[ModelOperatorSearchPoint, ...]:
        """Resolve the bounded semantic search coordinates for an input shape."""

        minimum, maximum = self.local_support_size_bounds(input_shape)
        if maximum < minimum:
            return ()
        return tuple(
            ModelOperatorSearchPoint(
                local_support_dimension=self.local_support_dimension,
                local_support_size=size,
            )
            for size in range(minimum, maximum + 1)
        )

    def local_support_size_bounds(self, input_shape: tuple[int, ...]) -> tuple[int, int]:
        """Resolve inclusive local support size bounds without enumerating them."""

        if self.local_support_size_maximum is None:
            maximum = self._resolved_size_maximum(input_shape)
        else:
            maximum = self.local_support_size_maximum
        return self.local_support_size_minimum, maximum

    def includes_plan(self, plan: ModelOperatorPlan) -> bool:
        """Return whether a summarized architecture satisfies resource bounds."""

        parameter_count = plan.parameter_count
        if parameter_count is None:
            return False
        if (
            self.parameter_count_minimum is not None
            and parameter_count < self.parameter_count_minimum
        ):
            return False
        return not (
            self.parameter_count_maximum is not None
            and parameter_count > self.parameter_count_maximum
        )

    def _resolved_size_maximum(self, input_shape: tuple[int, ...]) -> int:
        if len(input_shape) < self.local_support_dimension:
            raise ArchitectureSearchDistributionValidationError(
                "input_shape rank is smaller than local support dimension"
            )
        return min(input_shape[-self.local_support_dimension :])


@dataclass(frozen=True, slots=True)
class ArchitectureCandidate:
    """One generated candidate architecture and its resolved formal metadata."""

    architecture: ArchitectureManifest
    operator_plan: ModelOperatorPlan
    parameters: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _value in self.parameters)
        if len(set(names)) != len(names):
            raise ArchitectureSearchDistributionValidationError(
                "candidate parameters must not repeat names"
            )

    @property
    def parameter_count(self) -> int:
        parameter_count = self.operator_plan.parameter_count
        if parameter_count is None:
            raise ArchitectureSearchDistributionValidationError(
                "candidate operator plan must have known parameter_count"
            )
        return parameter_count

    def parameter(self, name: str) -> int:
        for key, value in self.parameters:
            if key == name:
                return value
        raise ArchitectureSearchDistributionValidationError(
            f"candidate does not include parameter: {name}"
        )


def default_architecture_search_distribution() -> ArchitectureSearchDistribution:
    """Return the generic formal-operator search distribution for local proposals."""

    return ArchitectureSearchDistribution(
        local_support_dimension=2,
        local_support_size_minimum=1,
        local_support_size_maximum_from=_minimum_trailing_input_axis,
    )


def generate_architecture_candidates(
    distribution: ArchitectureSearchDistribution,
    *,
    input_shape: tuple[int, ...],
    output_count: int,
) -> tuple[ArchitectureCandidate, ...]:
    """Expand a search distribution into concrete architecture manifests."""

    return _deduplicate_and_sort(
        _candidate_from_point(
            point,
            distribution=distribution,
            input_shape=input_shape,
            output_count=output_count,
        )
        for point in distribution.search_points(input_shape)
    )


def sample_architecture_candidates(
    distribution: ArchitectureSearchDistribution,
    *,
    input_shape: tuple[int, ...],
    output_count: int,
    sample_count: int,
    seed: int = 0,
) -> tuple[ArchitectureCandidate, ...]:
    """Draw a deterministic bounded sample from a search distribution."""

    if type(sample_count) is not int or sample_count < 1:
        raise ArchitectureSearchDistributionValidationError("sample_count must be positive")
    if type(seed) is not int or seed < 0:
        raise ArchitectureSearchDistributionValidationError("seed must be nonnegative")
    minimum, maximum = distribution.local_support_size_bounds(input_shape)
    if maximum < minimum:
        return ()
    span = maximum - minimum + 1
    if sample_count >= span:
        sizes = range(minimum, maximum + 1)
    else:
        sizes = sorted(random.Random(seed).sample(range(minimum, maximum + 1), sample_count))
    return _deduplicate_and_sort(
        _candidate_from_point(
            ModelOperatorSearchPoint(
                local_support_dimension=distribution.local_support_dimension,
                local_support_size=size,
            ),
            distribution=distribution,
            input_shape=input_shape,
            output_count=output_count,
        )
        for size in sizes
    )[:sample_count]


def _candidate_from_point(
    point: ModelOperatorSearchPoint,
    *,
    distribution: ArchitectureSearchDistribution,
    input_shape: tuple[int, ...],
    output_count: int,
) -> ArchitectureCandidate | None:
    architecture = materialize_model_operator_search_point(
        input_shape=input_shape,
        output_count=output_count,
        point=point,
    )
    plan = summarize_architecture_operators(architecture)
    if not distribution.includes_plan(plan):
        return None
    return ArchitectureCandidate(
        architecture=architecture,
        operator_plan=plan,
        parameters=point.to_parameters(),
    )


def _deduplicate_and_sort(
    candidates: Iterable[ArchitectureCandidate | None],
) -> tuple[ArchitectureCandidate, ...]:
    deduplicated: list[ArchitectureCandidate] = []
    seen: set[object] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        if candidate.architecture.digest in seen:
            continue
        seen.add(candidate.architecture.digest)
        deduplicated.append(candidate)
    return tuple(sorted(deduplicated, key=_candidate_sort_key))


def _candidate_sort_key(candidate: ArchitectureCandidate) -> tuple[int, int, str]:
    return (
        candidate.parameter_count,
        candidate.parameter("local_support_size"),
        str(candidate.architecture.id),
    )


def _require_optional_count(value: int | None, *, field: str) -> None:
    if value is None:
        return
    if type(value) is not int or value < 0:
        raise ArchitectureSearchDistributionValidationError(f"{field} must be nonnegative")
