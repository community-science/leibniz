"""Generic architecture candidate spaces for local proposal workflows."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from leibniz.architectures import ArchitectureManifest
from leibniz.model_operators import (
    ModelOperatorPlan,
    formal_image_classifier_architecture,
    summarize_architecture_operators,
)

__all__ = [
    "ArchitectureCandidate",
    "ArchitectureCandidateRecipe",
    "ArchitectureCandidateSpace",
    "ArchitectureCandidateSpaceValidationError",
    "default_architecture_candidate_space",
    "generate_architecture_candidates",
    "sample_architecture_candidates",
]

_RecipeKind = Literal["local-aggregation-readout"]
_SizeMaximumKind = Literal["minimum-trailing-input-axis"]
_local_aggregation_readout: _RecipeKind = "local-aggregation-readout"
_minimum_trailing_input_axis: _SizeMaximumKind = "minimum-trailing-input-axis"


class ArchitectureCandidateSpaceValidationError(ValueError):
    """Raised when an architecture candidate space is invalid."""


@dataclass(frozen=True, slots=True)
class ArchitectureCandidateRecipe:
    """One formal construction recipe within an architecture candidate space."""

    kind: _RecipeKind
    local_aggregation_dimension: int
    local_aggregation_size_minimum: int
    local_aggregation_size_maximum: int | None = None
    local_aggregation_size_maximum_from: _SizeMaximumKind | None = None
    parameter_count_minimum: int | None = None
    parameter_count_maximum: int | None = None

    def __post_init__(self) -> None:
        if self.kind != _local_aggregation_readout:
            raise ArchitectureCandidateSpaceValidationError(
                f"unsupported candidate recipe: {self.kind}"
            )
        if type(self.local_aggregation_dimension) is not int or (
            self.local_aggregation_dimension < 1
        ):
            raise ArchitectureCandidateSpaceValidationError(
                "local_aggregation_dimension must be positive"
            )
        if type(self.local_aggregation_size_minimum) is not int or (
            self.local_aggregation_size_minimum < 1
        ):
            raise ArchitectureCandidateSpaceValidationError(
                "local_aggregation_size.minimum must be positive"
            )
        if self.local_aggregation_size_maximum is None:
            if self.local_aggregation_size_maximum_from != _minimum_trailing_input_axis:
                raise ArchitectureCandidateSpaceValidationError(
                    "local_aggregation_size must declare maximum or maximum_from"
                )
        elif self.local_aggregation_size_maximum < self.local_aggregation_size_minimum:
            raise ArchitectureCandidateSpaceValidationError(
                "local_aggregation_size.maximum must be at least minimum"
            )
        if self.local_aggregation_size_maximum_from is not None and (
            self.local_aggregation_size_maximum_from != _minimum_trailing_input_axis
        ):
            raise ArchitectureCandidateSpaceValidationError(
                "unsupported local_aggregation_size.maximum_from"
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
            raise ArchitectureCandidateSpaceValidationError(
                "parameter_count.maximum must be at least minimum"
            )

    def local_aggregation_sizes(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Resolve the declared local aggregation size range for an input shape."""

        minimum, maximum = self.local_aggregation_size_bounds(input_shape)
        if maximum < minimum:
            return ()
        return tuple(range(minimum, maximum + 1))

    def local_aggregation_size_bounds(self, input_shape: tuple[int, ...]) -> tuple[int, int]:
        """Resolve inclusive local aggregation size bounds without enumerating them."""

        if self.local_aggregation_size_maximum is None:
            maximum = self._resolved_size_maximum(input_shape)
        else:
            maximum = self.local_aggregation_size_maximum
        return self.local_aggregation_size_minimum, maximum

    def includes_plan(self, plan: ModelOperatorPlan) -> bool:
        """Return whether a summarized architecture satisfies this recipe's cost bounds."""

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
        if len(input_shape) < self.local_aggregation_dimension:
            raise ArchitectureCandidateSpaceValidationError(
                "input_shape rank is smaller than local aggregation dimension"
            )
        return min(input_shape[-self.local_aggregation_dimension :])


@dataclass(frozen=True, slots=True)
class ArchitectureCandidateSpace:
    """A source-defined architecture candidate-space declaration."""

    recipes: tuple[ArchitectureCandidateRecipe, ...]

    def __post_init__(self) -> None:
        if not self.recipes:
            raise ArchitectureCandidateSpaceValidationError(
                "candidate space must contain at least one recipe"
            )


@dataclass(frozen=True, slots=True)
class ArchitectureCandidate:
    """One generated candidate architecture and its resolved formal metadata."""

    architecture: ArchitectureManifest
    operator_plan: ModelOperatorPlan
    parameters: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        names = tuple(name for name, _value in self.parameters)
        if len(set(names)) != len(names):
            raise ArchitectureCandidateSpaceValidationError(
                "candidate parameters must not repeat names"
            )

    @property
    def parameter_count(self) -> int:
        parameter_count = self.operator_plan.parameter_count
        if parameter_count is None:
            raise ArchitectureCandidateSpaceValidationError(
                "candidate operator plan must have known parameter_count"
            )
        return parameter_count

    def parameter(self, name: str) -> int:
        for key, value in self.parameters:
            if key == name:
                return value
        raise ArchitectureCandidateSpaceValidationError(
            f"candidate does not include parameter: {name}"
        )


def default_architecture_candidate_space() -> ArchitectureCandidateSpace:
    """Return the generic formal-operator candidate space used by local proposals."""

    return ArchitectureCandidateSpace(
        recipes=(
            ArchitectureCandidateRecipe(
                kind=_local_aggregation_readout,
                local_aggregation_dimension=2,
                local_aggregation_size_minimum=1,
                local_aggregation_size_maximum_from=_minimum_trailing_input_axis,
            ),
        )
    )


def generate_architecture_candidates(
    space: ArchitectureCandidateSpace,
    *,
    input_shape: tuple[int, ...],
    output_count: int,
) -> tuple[ArchitectureCandidate, ...]:
    """Expand a declared candidate space into concrete architecture manifests."""

    candidates: list[ArchitectureCandidate] = []
    seen: set[object] = set()
    for recipe in space.recipes:
        candidates.extend(
            _recipe_candidates(
                recipe,
                input_shape=input_shape,
                output_count=output_count,
            )
        )
    deduplicated: list[ArchitectureCandidate] = []
    for candidate in candidates:
        if candidate.architecture.digest in seen:
            continue
        seen.add(candidate.architecture.digest)
        deduplicated.append(candidate)
    return tuple(sorted(deduplicated, key=_candidate_sort_key))


def sample_architecture_candidates(
    space: ArchitectureCandidateSpace,
    *,
    input_shape: tuple[int, ...],
    output_count: int,
    sample_count: int,
    seed: int = 0,
) -> tuple[ArchitectureCandidate, ...]:
    """Draw a deterministic bounded sample from a candidate space."""

    if type(sample_count) is not int or sample_count < 1:
        raise ArchitectureCandidateSpaceValidationError("sample_count must be positive")
    if type(seed) is not int or seed < 0:
        raise ArchitectureCandidateSpaceValidationError("seed must be nonnegative")
    candidates: list[ArchitectureCandidate] = []
    per_recipe = max(1, -(-sample_count // len(space.recipes)))
    for recipe_index, recipe in enumerate(space.recipes):
        candidates.extend(
            _sample_recipe_candidates(
                recipe,
                input_shape=input_shape,
                output_count=output_count,
                sample_count=per_recipe,
                seed=seed + recipe_index,
            )
        )
    deduplicated: list[ArchitectureCandidate] = []
    seen: set[object] = set()
    for candidate in sorted(candidates, key=_candidate_sort_key):
        if candidate.architecture.digest in seen:
            continue
        seen.add(candidate.architecture.digest)
        deduplicated.append(candidate)
        if len(deduplicated) == sample_count:
            break
    return tuple(deduplicated)


def _recipe_candidates(
    recipe: ArchitectureCandidateRecipe,
    *,
    input_shape: tuple[int, ...],
    output_count: int,
) -> tuple[ArchitectureCandidate, ...]:
    if recipe.kind != _local_aggregation_readout:
        raise ArchitectureCandidateSpaceValidationError(
            f"unsupported candidate recipe: {recipe.kind}"
        )
    candidates: list[ArchitectureCandidate] = []
    for size in recipe.local_aggregation_sizes(input_shape):
        architecture = formal_image_classifier_architecture(
            input_shape=input_shape,
            output_count=output_count,
            local_aggregation_size=size,
            local_aggregation_dimension=recipe.local_aggregation_dimension,
        )
        plan = summarize_architecture_operators(architecture)
        if not recipe.includes_plan(plan):
            continue
        candidates.append(
            ArchitectureCandidate(
                architecture=architecture,
                operator_plan=plan,
                parameters=(("local_aggregation_size", size),),
            )
        )
    return tuple(candidates)


def _sample_recipe_candidates(
    recipe: ArchitectureCandidateRecipe,
    *,
    input_shape: tuple[int, ...],
    output_count: int,
    sample_count: int,
    seed: int,
) -> tuple[ArchitectureCandidate, ...]:
    if recipe.kind != _local_aggregation_readout:
        raise ArchitectureCandidateSpaceValidationError(
            f"unsupported candidate recipe: {recipe.kind}"
        )
    minimum, maximum = recipe.local_aggregation_size_bounds(input_shape)
    if maximum < minimum:
        return ()
    span = maximum - minimum + 1
    if sample_count >= span:
        sizes = range(minimum, maximum + 1)
    else:
        sizes = sorted(random.Random(seed).sample(range(minimum, maximum + 1), sample_count))
    candidates: list[ArchitectureCandidate] = []
    for size in sizes:
        architecture = formal_image_classifier_architecture(
            input_shape=input_shape,
            output_count=output_count,
            local_aggregation_size=size,
            local_aggregation_dimension=recipe.local_aggregation_dimension,
        )
        plan = summarize_architecture_operators(architecture)
        if not recipe.includes_plan(plan):
            continue
        candidates.append(
            ArchitectureCandidate(
                architecture=architecture,
                operator_plan=plan,
                parameters=(("local_aggregation_size", size),),
            )
        )
    return tuple(candidates)


def _candidate_sort_key(candidate: ArchitectureCandidate) -> tuple[int, int, str]:
    return (
        candidate.parameter_count,
        candidate.parameter("local_aggregation_size"),
        str(candidate.architecture.id),
    )


def _require_optional_count(value: int | None, *, field: str) -> None:
    if value is None:
        return
    if type(value) is not int or value < 0:
        raise ArchitectureCandidateSpaceValidationError(f"{field} must be nonnegative")
