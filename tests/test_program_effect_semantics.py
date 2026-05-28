from collections.abc import Callable

from leibniz.program_effect_semantics import (
    ProgramEffectSemantic,
    ProgramEffectSemanticRegistry,
    ProgramEffectSemanticValidationError,
    program_effect_semantic_registry,
)


def test_program_effect_semantic_registry_declares_current_public_effects() -> None:
    registry = program_effect_semantic_registry()

    assert [effect.kind for effect in registry.effects] == [
        "branch",
        "merge",
        "route",
        "repeat",
        "identity-path",
        "parameter-sharing",
    ]
    assert registry.semantic_for_kind("branch") == registry.effects[0]
    assert registry.effects[0].descriptor_record(effect_arity=3) == {
        "kind": "branch",
        "input_arity": 1,
        "output_arity": 3,
        "shape_law": "duplicate-input-shape",
        "cost_law": "zero-arithmetic",
        "trace_law": "fan-out",
    }
    parameter_sharing = registry.semantic_for_kind("parameter-sharing")

    assert parameter_sharing is not None
    assert parameter_sharing.descriptor_record(effect_arity=2) == {
        "kind": "parameter-sharing",
        "input_arity": 2,
        "output_arity": 2,
        "shape_law": "preserve-shapes",
        "cost_law": "share-state",
        "trace_law": "share-parameter-group",
    }


def test_program_effect_semantic_registry_exports_records() -> None:
    records = program_effect_semantic_registry().effect_records()

    assert records[3] == {
        "kind": "repeat",
        "input_arity_law": "one",
        "output_arity_law": "one",
        "shape_law": "preserve-shape",
        "cost_law": "multiply-nested-cost",
        "trace_law": "repeat-nested-program",
    }


def test_program_effect_semantic_records_reject_invalid_declarations() -> None:
    assert str(
        capture_program_effect_error(lambda: _semantic(kind=""))
    ) == "program effect kind must be nonempty"
    assert str(
        capture_program_effect_error(
            lambda: ProgramEffectSemanticRegistry(
                effects=(_semantic(kind="test-effect"), _semantic(kind="test-effect")),
            )
        )
    ) == "program effect kinds must be unique"
    assert str(
        capture_program_effect_error(
            lambda: _semantic(input_arity_law="unsupported").input_arity(1)
        )
    ) == "test-effect input_arity_law is unsupported"


def _semantic(
    *,
    kind: str = "test-effect",
    input_arity_law: str = "one",
) -> ProgramEffectSemantic:
    return ProgramEffectSemantic(
        kind=kind,
        input_arity_law=input_arity_law,
        output_arity_law="one",
        shape_law="preserve-shape",
        cost_law="zero-arithmetic",
        trace_law="preserve-path",
    )


def capture_program_effect_error(
    action: Callable[[], object],
) -> ProgramEffectSemanticValidationError:
    try:
        action()
    except ProgramEffectSemanticValidationError as error:
        return error
    raise AssertionError("expected ProgramEffectSemanticValidationError")
