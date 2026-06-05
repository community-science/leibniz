from collections.abc import Callable
from pathlib import Path

import pytest
from benchmark_typing import load_digits_benchmark

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.materialization import (
    AxisAssignment,
    LinearResolutionRequirement,
    MaterializationDeclaration,
    MaterializationDeclarationDocument,
    MaterializationPlan,
    MaterializationPlanDocument,
    MaterializationValidationError,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_digits_fixture_root = _repository_root / "tests" / "fixtures" / "digits"


def test_axis_assignment_round_trips_in_canonical_order() -> None:
    assignment = AxisAssignment(values={"N": 96, "L": 3})

    assert assignment.require_axis("L") == 3
    assert assignment.to_record() == {
        "values": [
            {"axis": "L", "value": 3},
            {"axis": "N", "value": 96},
        ]
    }
    assert AxisAssignment.from_record(assignment.to_record()) == assignment


def test_axis_assignment_rejects_invalid_values() -> None:
    assert str(capture_materialization_error(lambda: AxisAssignment(values={}))) == (
        "axis assignment must not be empty"
    )
    assert str(capture_materialization_error(lambda: AxisAssignment(values={"L": -1}))) == (
        "L: axis value must be nonnegative"
    )
    duplicate_record = {
        "values": [
            {"axis": "L", "value": 1},
            {"axis": "L", "value": 2},
        ]
    }

    assert str(
        capture_materialization_error(lambda: AxisAssignment.from_record(duplicate_record))
    ) == "duplicate axis assignment: L"


def test_linear_resolution_requirement_derives_minimum_resolution() -> None:
    requirement = LinearResolutionRequirement(
        name=ProtocolName.parse("benchmarks.digits.resolution.canvas-side"),
        source_axis="L",
        resolution_axis="N",
        coefficient=32,
        minimum=32,
        basis="analytic-bound",
    )

    assert requirement.minimum_resolution(AxisAssignment(values={"L": 1})) == 32
    assert requirement.minimum_resolution(AxisAssignment(values={"L": 3})) == 96

    requirement.require_resolution(
        source_assignment=AxisAssignment(values={"L": 3}),
        resolution_assignment=AxisAssignment(values={"N": 96}),
    )
    assert str(
        capture_materialization_error(
            lambda: requirement.require_resolution(
                source_assignment=AxisAssignment(values={"L": 3}),
                resolution_assignment=AxisAssignment(values={"N": 95}),
            )
        )
    ) == "N=95 is below minimum 96 for L=3"


def test_digits_materialization_declaration_is_python_owned() -> None:
    declaration = _digits_materialization()

    assert declaration.id == ProtocolIdentifier.parse("benchmarks.digits.materialization@0.1.0")
    assert declaration.benchmark_id == ProtocolIdentifier.parse("benchmarks.digits@0.1.0")
    assert declaration.latent_factor_declaration == ArtifactReference(
        kind="latent-factor-declaration",
        protocol_id=ProtocolIdentifier.parse("benchmarks.digits.latent-factors@0.1.0"),
    )
    assert declaration.requirements == ()
    assert declaration.minimum_resolution() == AxisAssignment(values={"W": 1, "H": 1})
    assert declaration.resolution_lattice_steps() == {}
    assert declaration.digest == ContentDigest.from_value(declaration.to_record())


def test_materialization_declaration_uses_strongest_requirement_per_resolution_axis() -> None:
    declaration = _digits_materialization()
    stronger = LinearResolutionRequirement(
        name=ProtocolName.parse("benchmarks.digits.resolution.extra-margin"),
        source_axis="L",
        resolution_axis="W",
        coefficient=40,
        basis="analytic-bound",
    )
    declaration = MaterializationDeclaration(
        id=declaration.id,
        benchmark_id=declaration.benchmark_id,
        requirements=(*declaration.requirements, stronger),
        latent_factor_declaration=declaration.latent_factor_declaration,
        layout=declaration.layout,
    )

    assert declaration.minimum_resolution(AxisAssignment(values={"L": 3})) == AxisAssignment(
        values={"W": 120, "H": 1}
    )


def test_materialization_plan_resolves_from_declaration_deterministically() -> None:
    declaration = _digits_materialization()

    left = MaterializationPlan.resolve(
        id=ProtocolIdentifier.parse("benchmarks.digits.materialization-plan.l3.seed101@0.1.0"),
        declaration=declaration,
        seed=101,
    )
    right = MaterializationPlan.resolve(
        id=ProtocolIdentifier.parse("benchmarks.digits.materialization-plan.l3.seed101@0.1.0"),
        declaration=declaration,
        seed=101,
    )

    assert left == right
    assert left.resolution_assignment == AxisAssignment(values={"W": 1, "H": 1})
    left.validate_declaration(declaration)


def test_materialization_plan_preserves_source_assignment_for_requirement_validation() -> None:
    declaration = MaterializationDeclaration(
        id=ProtocolIdentifier.parse("benchmarks.example.materialization@0.1.0"),
        benchmark_id=ProtocolIdentifier.parse("benchmarks.example@0.1.0"),
        requirements=(
            LinearResolutionRequirement(
                name=ProtocolName.parse("benchmarks.example.resolution.width"),
                source_axis="L",
                resolution_axis="W",
                coefficient=10,
                basis="analytic-bound",
            ),
        ),
    )

    plan = MaterializationPlan.resolve(
        id=ProtocolIdentifier.parse("benchmarks.example.materialization-plan.seed101@0.1.0"),
        declaration=declaration,
        seed=101,
        source_assignment=AxisAssignment(values={"L": 3}),
    )

    assert plan.resolution_assignment == AxisAssignment(values={"W": 30})
    plan.validate_declaration(declaration)
    assert MaterializationPlan.from_record(plan.to_record()) == plan


def test_materialization_plan_documents_validate_digits_fixtures() -> None:
    declaration = _digits_materialization()
    l1 = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l1.json").read_bytes()
    )
    l3 = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l3.json").read_bytes()
    )

    l1.plan.validate_declaration(declaration)
    l3.plan.validate_declaration(declaration)

    assert l1.plan.resolution_assignment.values == {"W": 24, "H": 24}
    assert l3.plan.resolution_assignment.values == {"W": 72, "H": 24}
    assert l3.digest == ContentDigest.from_value(l3.plan.to_record())


def test_materialization_plan_accepts_arbitrary_positive_resolution() -> None:
    declaration = _digits_materialization()
    plan = MaterializationPlan(
        id=ProtocolIdentifier.parse("benchmarks.digits.materialization-plan.bad@0.1.0"),
        benchmark_id=ProtocolIdentifier.parse("benchmarks.digits@0.1.0"),
        materialization_declaration=ArtifactReference(
            kind="materialization-declaration",
            protocol_id=ProtocolIdentifier.parse("benchmarks.digits.materialization@0.1.0"),
            record_digest=declaration.digest,
        ),
        latent_factor_declaration=ArtifactReference(
            kind="latent-factor-declaration",
            protocol_id=ProtocolIdentifier.parse("benchmarks.digits.latent-factors@0.1.0"),
        ),
        resolution_assignment=AxisAssignment(values={"W": 48, "H": 28}),
        seed=101,
    )

    plan.validate_declaration(declaration)


def test_materialization_plan_rejects_under_resolved_request() -> None:
    declaration = _digits_materialization()
    plan = MaterializationPlan(
        id=ProtocolIdentifier.parse("benchmarks.digits.materialization-plan.bad@0.1.0"),
        benchmark_id=ProtocolIdentifier.parse("benchmarks.digits@0.1.0"),
        materialization_declaration=ArtifactReference(
            kind="materialization-declaration",
            protocol_id=ProtocolIdentifier.parse("benchmarks.digits.materialization@0.1.0"),
        ),
        latent_factor_declaration=ArtifactReference(
            kind="latent-factor-declaration",
            protocol_id=ProtocolIdentifier.parse("benchmarks.digits.latent-factors@0.1.0"),
        ),
        resolution_assignment=AxisAssignment(values={"W": 0, "H": 1}),
        seed=101,
    )

    assert str(capture_materialization_error(lambda: plan.validate_declaration(declaration))) == (
        "W=0 is below layout minimum 1"
    )


def test_materialization_documents_reject_invalid_bytes() -> None:
    assert str(
        capture_materialization_error(
            lambda: MaterializationDeclarationDocument.from_bytes(b"\xff")
        )
    ) == "materialization declaration must be UTF-8"
    assert str(
        capture_materialization_error(lambda: MaterializationPlanDocument.from_bytes(b"[]"))
    ) == "materialization plan must contain an object"


def _digits_materialization() -> MaterializationDeclaration:
    return load_digits_benchmark(_digits_benchmark_root).materialization


def capture_materialization_error(
    call: Callable[[], object],
) -> MaterializationValidationError:
    with pytest.raises(MaterializationValidationError) as error:
        call()
    return error.value
