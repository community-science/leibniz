"""Declared semantics for model program effects."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

__all__ = [
    "ProgramEffectSemantic",
    "ProgramEffectSemanticRegistry",
    "ProgramEffectSemanticValidationError",
    "program_effect_semantic_registry",
]


class ProgramEffectSemanticValidationError(ValueError):
    """Raised when program-effect semantic records are invalid."""


@dataclass(frozen=True, slots=True)
class ProgramEffectSemantic:
    """Declared shape, cost, and trace semantics for one program effect kind."""

    kind: str
    input_arity_law: str
    output_arity_law: str
    shape_law: str
    cost_law: str
    trace_law: str

    def __post_init__(self) -> None:
        _require_nonempty(self.kind, field="program effect kind")
        _require_nonempty(self.input_arity_law, field="input_arity_law")
        _require_nonempty(self.output_arity_law, field="output_arity_law")
        _require_nonempty(self.shape_law, field="shape_law")
        _require_nonempty(self.cost_law, field="cost_law")
        _require_nonempty(self.trace_law, field="trace_law")

    def input_arity(self, effect_arity: int) -> int:
        """Return the descriptor input arity for an effect instance."""

        return _resolve_arity_law(
            self.input_arity_law,
            effect_arity=effect_arity,
            field=f"{self.kind} input_arity_law",
        )

    def output_arity(self, effect_arity: int) -> int:
        """Return the descriptor output arity for an effect instance."""

        return _resolve_arity_law(
            self.output_arity_law,
            effect_arity=effect_arity,
            field=f"{self.kind} output_arity_law",
        )

    def descriptor_record(self, *, effect_arity: int) -> dict[str, object]:
        """Return the descriptor record for an effect instance."""

        return {
            "kind": self.kind,
            "input_arity": self.input_arity(effect_arity),
            "output_arity": self.output_arity(effect_arity),
            "shape_law": self.shape_law,
            "cost_law": self.cost_law,
            "trace_law": self.trace_law,
        }


@dataclass(frozen=True, slots=True)
class ProgramEffectSemanticRegistry:
    """Declared semantic records for supported model program effects."""

    effects: tuple[ProgramEffectSemantic, ...]

    def __post_init__(self) -> None:
        _require_unique((effect.kind for effect in self.effects), field="program effect kinds")

    def semantic_for_kind(self, kind: str) -> ProgramEffectSemantic | None:
        """Return the semantic declaration for a program effect kind."""

        return self._semantic_by_kind().get(kind)

    def effect_records(self) -> list[dict[str, object]]:
        """Return canonical program-effect semantic records."""

        return [
            {
                "kind": effect.kind,
                "input_arity_law": effect.input_arity_law,
                "output_arity_law": effect.output_arity_law,
                "shape_law": effect.shape_law,
                "cost_law": effect.cost_law,
                "trace_law": effect.trace_law,
            }
            for effect in self.effects
        ]

    def _semantic_by_kind(self) -> dict[str, ProgramEffectSemantic]:
        return {effect.kind: effect for effect in self.effects}


def program_effect_semantic_registry() -> ProgramEffectSemanticRegistry:
    """Return the declared model program-effect semantic registry."""

    return _program_effect_semantic_registry


def _resolve_arity_law(law: str, *, effect_arity: int, field: str) -> int:
    if law == "one":
        return 1
    if law == "effect-arity":
        return effect_arity
    raise ProgramEffectSemanticValidationError(f"{field} is unsupported")


def _require_nonempty(value: str, *, field: str) -> None:
    if not value:
        raise ProgramEffectSemanticValidationError(f"{field} must be nonempty")


def _require_unique(values: Iterable[str], *, field: str) -> None:
    sequence = tuple(values)
    if any(not value for value in sequence):
        raise ProgramEffectSemanticValidationError(f"{field} must contain nonempty values")
    if len(set(sequence)) != len(sequence):
        raise ProgramEffectSemanticValidationError(f"{field} must be unique")


_program_effect_semantics = (
    ProgramEffectSemantic(
        kind="branch",
        input_arity_law="one",
        output_arity_law="effect-arity",
        shape_law="duplicate-input-shape",
        cost_law="zero-arithmetic",
        trace_law="fan-out",
    ),
    ProgramEffectSemantic(
        kind="merge",
        input_arity_law="effect-arity",
        output_arity_law="one",
        shape_law="require-equal-input-shapes",
        cost_law="zero-arithmetic",
        trace_law="join-paths",
    ),
    ProgramEffectSemantic(
        kind="route",
        input_arity_law="effect-arity",
        output_arity_law="one",
        shape_law="select-equal-input-shape",
        cost_law="control-flow-select",
        trace_law="select-path",
    ),
    ProgramEffectSemantic(
        kind="repeat",
        input_arity_law="one",
        output_arity_law="one",
        shape_law="preserve-shape",
        cost_law="multiply-nested-cost",
        trace_law="repeat-nested-program",
    ),
    ProgramEffectSemantic(
        kind="identity-path",
        input_arity_law="one",
        output_arity_law="one",
        shape_law="preserve-shape",
        cost_law="zero-arithmetic",
        trace_law="preserve-path",
    ),
    ProgramEffectSemantic(
        kind="parameter-sharing",
        input_arity_law="effect-arity",
        output_arity_law="effect-arity",
        shape_law="preserve-shapes",
        cost_law="share-state",
        trace_law="share-parameter-group",
    ),
)

_program_effect_semantic_registry = ProgramEffectSemanticRegistry(
    effects=_program_effect_semantics,
)
