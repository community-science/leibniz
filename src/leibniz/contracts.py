"""Generic contract object and projection boundaries."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field

__all__ = [
    "ConformanceCase",
    "ContractObject",
    "ContractRuntimeSupport",
    "ContractSurface",
    "RuntimeProjection",
]


def _empty_metadata() -> Mapping[str, object]:
    return {}


class ContractRuntimeSupport(ABC):
    """Marker base for handwritten contract runtime support."""

    @property
    @abstractmethod
    def contract_runtime_role(self) -> str:
        """Return the generic contract-runtime role this object serves."""


class ContractObject(ContractRuntimeSupport):
    """Base for hand-authored objects that own contract projections."""

    @property
    @abstractmethod
    def contract_name(self) -> str:
        """Return the durable contract name owned by this object."""

    def runtime_projections(self) -> tuple[RuntimeProjection, ...]:
        """Return runtime projections that do not need caller-specific options."""

        return ()

    def conformance_cases(self) -> tuple[ConformanceCase, ...]:
        """Return executable conformance cases owned by this contract object."""

        return ()

    def source_graph_facts(self) -> tuple[Mapping[str, object], ...]:
        """Return source-graph facts emitted by this contract object."""

        return ()


@dataclass(frozen=True, slots=True)
class ContractSurface:
    """A named public surface projected from a contract object."""

    name: str
    scope: str
    owner: str
    metadata: Mapping[str, object] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class RuntimeProjection:
    """A materializable runtime artifact projected from a contract object."""

    contract_name: str
    surface: str
    target: str
    content: object
    metadata: Mapping[str, object] = field(default_factory=_empty_metadata)


@dataclass(frozen=True, slots=True)
class ConformanceCase:
    """One input case a contract object claims should pass or fail."""

    contract_name: str
    name: str
    record: Mapping[str, object]
    expected_valid: bool
    reason: str
