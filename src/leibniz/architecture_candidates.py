"""Generic architecture candidate spaces for local proposal workflows."""

from __future__ import annotations

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
    "ArchitectureCandidateFamily",
    "ArchitectureCandidateSpace",
    "ArchitectureCandidateSpaceValidationError",
    "default_architecture_candidate_space",
    "generate_architecture_candidates",
]

_FamilyKind = Literal["local-aggregation-readout"]
_SizeMaximumKind = Literal["minimum-trailing-input-axis"]
_local_aggregation_readout: _FamilyKind = "local-aggregation-readout"
_minimum_trailing_input_axis: _SizeMaximumKind = "minimum-trailing-input-axis"


class ArchitectureCandidateSpaceValidationError(ValueError):
    """Raised when an architecture candidate space is invalid."""


@dataclass(frozen=True, slots=True)
class ArchitectureCandidateFamily:
    """One formal family within an architecture candidate space."""

    kind: _FamilyKind
    local_aggregation_dimension: int
    local_aggregation_size_minimum: int
    local_aggregation_size_maximum: int | None = None
    local_aggregation_size_maximum_from: _SizeMaximumKind | None = None
    parameter_count_minimum: int | None = None
    parameter_count_maximum: int | None = None

    def __post_init__(self) -> None:
        if self.kind != _local_aggregation_readout:
            raise ArchitectureCandidateSpaceValidationError(
                f"unsupported candidate family: {self.kind}"
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

        if self.local_aggregation_size_maximum is None:
            maximum = self._resolved_size_maximum(input_shape)
        else:
            maximum = self.local_aggregation_size_maximum
        if maximum < self.local_aggregation_size_minimum:
            return ()
        return tuple(range(self.local_aggregation_size_minimum, maximum + 1))

    def includes_plan(self, plan: ModelOperatorPlan) -> bool:
        """Return whether a summarized architecture satisfies this family's cost bounds."""

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

    families: tuple[ArchitectureCandidateFamily, ...]

    def __post_init__(self) -> None:
        if not self.families:
            raise ArchitectureCandidateSpaceValidationError(
                "candidate space must contain at least one family"
            )


@dataclass(frozen=True, slots=True)
class ArchitectureCandidate:
    """One generated candidate architecture and its resolved formal metadata."""

    architecture: ArchitectureManifest
    operator_plan: ModelOperatorPlan
    family_kind: str
    parameters: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not self.family_kind:
            raise ArchitectureCandidateSpaceValidationError("family_kind must be nonempty")
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
        families=(
            ArchitectureCandidateFamily(
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
    for family in space.families:
        candidates.extend(
            _family_candidates(
                family,
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


def _family_candidates(
    family: ArchitectureCandidateFamily,
    *,
    input_shape: tuple[int, ...],
    output_count: int,
) -> tuple[ArchitectureCandidate, ...]:
    if family.kind != _local_aggregation_readout:
        raise ArchitectureCandidateSpaceValidationError(
            f"unsupported candidate family: {family.kind}"
        )
    candidates: list[ArchitectureCandidate] = []
    for size in family.local_aggregation_sizes(input_shape):
        architecture = formal_image_classifier_architecture(
            input_shape=input_shape,
            output_count=output_count,
            local_aggregation_size=size,
            local_aggregation_dimension=family.local_aggregation_dimension,
        )
        plan = summarize_architecture_operators(architecture)
        if not family.includes_plan(plan):
            continue
        candidates.append(
            ArchitectureCandidate(
                architecture=architecture,
                operator_plan=plan,
                family_kind=family.kind,
                parameters=(("local_aggregation_size", size),),
            )
        )
    return tuple(candidates)


def _candidate_sort_key(candidate: ArchitectureCandidate) -> tuple[int, str, int, str]:
    return (
        candidate.parameter_count,
        candidate.family_kind,
        candidate.parameter("local_aggregation_size"),
        str(candidate.architecture.id),
    )


def _require_optional_count(value: int | None, *, field: str) -> None:
    if value is None:
        return
    if type(value) is not int or value < 0:
        raise ArchitectureCandidateSpaceValidationError(f"{field} must be nonnegative")
