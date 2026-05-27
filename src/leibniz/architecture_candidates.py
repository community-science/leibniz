"""Generic architecture candidate distributions for local proposal workflows."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from leibniz.architectures import ArchitectureManifest
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.model_operators import (
    ModelOperatorCoordinate,
    ModelOperatorPlan,
    ModelOperatorSearchPoint,
    materialize_model_operator_search_point,
    model_operator_semantic_coordinates,
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

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    @property
    def id(self) -> ProtocolIdentifier:
        return ProtocolIdentifier.parse(
            f"architecture-search-distributions.sha-{self.digest.hex}@0.1.0"
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "local_support_dimension": self.local_support_dimension,
            "local_support_size_minimum": self.local_support_size_minimum,
        }
        if self.local_support_size_maximum is not None:
            record["local_support_size_maximum"] = self.local_support_size_maximum
        if self.local_support_size_maximum_from is not None:
            record["local_support_size_maximum_from"] = self.local_support_size_maximum_from
        if self.parameter_count_minimum is not None:
            record["parameter_count_minimum"] = self.parameter_count_minimum
        if self.parameter_count_maximum is not None:
            record["parameter_count_maximum"] = self.parameter_count_maximum
        return record

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
    semantic_coordinates: tuple[ModelOperatorCoordinate, ...]
    parameters: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _value in self.parameters)
        if len(set(names)) != len(names):
            raise ArchitectureSearchDistributionValidationError(
                "candidate parameters must not repeat names"
            )
        coordinate_names = tuple(coordinate.name for coordinate in self.semantic_coordinates)
        if not coordinate_names:
            raise ArchitectureSearchDistributionValidationError(
                "candidate semantic_coordinates must be nonempty"
            )
        if len(set(coordinate_names)) != len(coordinate_names):
            raise ArchitectureSearchDistributionValidationError(
                "candidate semantic_coordinates must not repeat names"
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

    def coordinate(self, name: str) -> int | str:
        for coordinate in self.semantic_coordinates:
            if coordinate.name == name:
                return coordinate.value
        raise ArchitectureSearchDistributionValidationError(
            f"candidate does not include semantic coordinate: {name}"
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
    minimum, maximum = _sample_support_size_bounds(
        distribution,
        input_shape=input_shape,
        output_count=output_count,
    )
    if maximum < minimum:
        return ()
    span = maximum - minimum + 1
    if sample_count >= span:
        sizes = range(minimum, maximum + 1)
    else:
        sizes = _resource_stratified_support_sizes(
            distribution,
            input_shape=input_shape,
            output_count=output_count,
            sample_count=sample_count,
            seed=seed,
            minimum=minimum,
            maximum=maximum,
        )
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
        semantic_coordinates=model_operator_semantic_coordinates(architecture, plan=plan),
        parameters=point.to_parameters(),
    )


def _resource_stratified_support_sizes(
    distribution: ArchitectureSearchDistribution,
    *,
    input_shape: tuple[int, ...],
    output_count: int,
    sample_count: int,
    seed: int,
    minimum: int,
    maximum: int,
) -> tuple[int, ...]:
    lower_resource = _resource_coordinate_for_support_size(
        input_shape=input_shape,
        output_count=output_count,
        dimension=distribution.local_support_dimension,
        size=minimum,
    )
    upper_resource = _resource_coordinate_for_support_size(
        input_shape=input_shape,
        output_count=output_count,
        dimension=distribution.local_support_dimension,
        size=maximum,
    )
    if upper_resource < lower_resource:
        lower_resource, upper_resource = upper_resource, lower_resource
    if sample_count == 1 or lower_resource == upper_resource:
        return (minimum,)

    rng = random.Random(seed)
    selected: list[int] = [minimum, maximum]
    selected_set: set[int] = {minimum, maximum}
    interior_count = sample_count - 2
    if interior_count < 1:
        return tuple(sorted(selected))
    width = (upper_resource - lower_resource) / (interior_count + 1)
    for stratum_index in range(interior_count):
        target = lower_resource + width * (stratum_index + 1 + rng.random() - 0.5)
        nearest = _nearest_support_size_for_resource(
            distribution,
            input_shape=input_shape,
            output_count=output_count,
            target=target,
            minimum=minimum,
            maximum=maximum,
        )
        available = _nearest_available_size(
            nearest,
            selected=selected_set,
            minimum=minimum,
            maximum=maximum,
        )
        if available is None:
            break
        selected.append(available)
        selected_set.add(available)
    return tuple(sorted(selected))


def _nearest_support_size_for_resource(
    distribution: ArchitectureSearchDistribution,
    *,
    input_shape: tuple[int, ...],
    output_count: int,
    target: float,
    minimum: int,
    maximum: int,
) -> int:
    low = minimum
    high = maximum
    while low < high:
        midpoint = (low + high) // 2
        coordinate = _resource_coordinate_for_support_size(
            input_shape=input_shape,
            output_count=output_count,
            dimension=distribution.local_support_dimension,
            size=midpoint,
        )
        if coordinate < target:
            low = midpoint + 1
        else:
            high = midpoint
    candidates = [low]
    if low > minimum:
        candidates.append(low - 1)
    if low < maximum:
        candidates.append(low + 1)
    return min(
        candidates,
        key=lambda size: (
            abs(
                _resource_coordinate_for_support_size(
                    input_shape=input_shape,
                    output_count=output_count,
                    dimension=distribution.local_support_dimension,
                    size=size,
                )
                - target
            ),
            size,
        ),
    )


def _nearest_available_size(
    preferred: int,
    *,
    selected: set[int],
    minimum: int,
    maximum: int,
) -> int | None:
    if preferred not in selected:
        return preferred
    for offset in range(1, maximum - minimum + 1):
        lower = preferred - offset
        if lower >= minimum and lower not in selected:
            return lower
        upper = preferred + offset
        if upper <= maximum and upper not in selected:
            return upper
    return None


def _sample_support_size_bounds(
    distribution: ArchitectureSearchDistribution,
    *,
    input_shape: tuple[int, ...],
    output_count: int,
) -> tuple[int, int]:
    minimum, maximum = distribution.local_support_size_bounds(input_shape)
    if maximum < minimum:
        return minimum, maximum
    if distribution.parameter_count_minimum is not None:
        minimum = _first_size_with_parameter_count_at_least(
            distribution,
            input_shape=input_shape,
            output_count=output_count,
            minimum=minimum,
            maximum=maximum,
            parameter_count=distribution.parameter_count_minimum,
        )
    if distribution.parameter_count_maximum is not None:
        maximum = _last_size_with_parameter_count_at_most(
            distribution,
            input_shape=input_shape,
            output_count=output_count,
            minimum=minimum,
            maximum=maximum,
            parameter_count=distribution.parameter_count_maximum,
        )
    return minimum, maximum


def _first_size_with_parameter_count_at_least(
    distribution: ArchitectureSearchDistribution,
    *,
    input_shape: tuple[int, ...],
    output_count: int,
    minimum: int,
    maximum: int,
    parameter_count: int,
) -> int:
    low = minimum
    high = maximum
    while low < high:
        midpoint = (low + high) // 2
        if (
            _parameter_count_for_support_size(
                input_shape=input_shape,
                output_count=output_count,
                dimension=distribution.local_support_dimension,
                size=midpoint,
            )
            < parameter_count
        ):
            low = midpoint + 1
        else:
            high = midpoint
    return low


def _last_size_with_parameter_count_at_most(
    distribution: ArchitectureSearchDistribution,
    *,
    input_shape: tuple[int, ...],
    output_count: int,
    minimum: int,
    maximum: int,
    parameter_count: int,
) -> int:
    low = minimum
    high = maximum
    while low < high:
        midpoint = (low + high + 1) // 2
        if (
            _parameter_count_for_support_size(
                input_shape=input_shape,
                output_count=output_count,
                dimension=distribution.local_support_dimension,
                size=midpoint,
            )
            > parameter_count
        ):
            high = midpoint - 1
        else:
            low = midpoint
    return low


def _resource_coordinate_for_support_size(
    *,
    input_shape: tuple[int, ...],
    output_count: int,
    dimension: int,
    size: int,
) -> float:
    return math.log1p(
        _parameter_count_for_support_size(
            input_shape=input_shape,
            output_count=output_count,
            dimension=dimension,
            size=size,
        )
    )


def _parameter_count_for_support_size(
    *,
    input_shape: tuple[int, ...],
    output_count: int,
    dimension: int,
    size: int,
) -> int:
    architecture = materialize_model_operator_search_point(
        input_shape=input_shape,
        output_count=output_count,
        point=ModelOperatorSearchPoint(
            local_support_dimension=dimension,
            local_support_size=size,
        ),
    )
    plan = summarize_architecture_operators(architecture)
    if plan.parameter_count is None:
        raise ArchitectureSearchDistributionValidationError(
            "sampled support size must have known parameter_count"
        )
    return plan.parameter_count


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
