from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.latent_factors import (
    ComplexityProjection,
    DegreeMeasure,
    GeneratorConstructionFactor,
    LatentFactorDeclaration,
    LatentFactorDeclarationDocument,
    LatentFactorValidationError,
    ResolutionRequirement,
    SampleLatentFactor,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def test_degree_measure_validates_supported_kinds() -> None:
    assert DegreeMeasure.discrete_choice(10).to_record() == {
        "kind": "discrete-choice",
        "count": 1.0,
        "domain_size": 10,
    }
    assert DegreeMeasure.vector_dimension(2).to_record() == {
        "kind": "vector-dimension",
        "count": 2.0,
    }

    assert str(capture_latent_error(lambda: DegreeMeasure.discrete_choice(1))) == (
        "domain_size must be at least 2"
    )
    assert str(
        capture_latent_error(
            lambda: DegreeMeasure.from_record(
                {"kind": "discrete-choice", "count": 1.0}
            )
        )
    ) == "discrete-choice degree measure requires domain_size"


def test_complexity_projection_counts_content_and_excludes_variation() -> None:
    declaration = _digits_declaration(sequence_length=3)

    assert declaration.evaluate_complexity("C") == 3.0
    assert [
        factor.name
        for factor in declaration.sample_factors
        if factor.role == "variation"
    ] == [ProtocolName.parse("benchmarks.digits.sample.field-variation-transform")]
    assert declaration.construction_factors == (
        GeneratorConstructionFactor(
            name=ProtocolName.parse("benchmarks.digits.construction.stroke-basis"),
            degree_measure=DegreeMeasure.constant_count(7),
        ),
    )


def test_digits_content_complexity_is_linear_in_sequence_length() -> None:
    values = [
        _digits_declaration(sequence_length=length).evaluate_complexity("C")
        for length in (1, 2, 5)
    ]

    assert values == [1.0, 2.0, 5.0]


def test_complexity_projection_rejects_resolution_axis() -> None:
    assert str(
        capture_latent_error(
            lambda: ComplexityProjection(
                name=ProtocolName.parse("benchmarks.digits.complexity.bad"),
                coordinate="N",
                included_roles=frozenset({"content"}),
            )
        )
    ) == "N is a resolution axis, not a complexity axis"


def test_resolution_requirement_rejects_infeasible_canvas_size() -> None:
    requirement = ResolutionRequirement.from_record(
        {
            "name": "benchmarks.digits.resolution.minimum-c1",
            "resolution_axis": "N",
            "content_coordinate": "C",
            "content_complexity": 1,
            "minimum_resolution": 32,
            "basis": "declared-minimum",
        }
    )

    requirement.require_resolution(32)

    assert str(capture_latent_error(lambda: requirement.require_resolution(31))) == (
        "N=31 is below minimum 32 for C=1"
    )


def test_latent_factor_declaration_document_loads_digits_source_artifact() -> None:
    document = LatentFactorDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "latent_factors.json").read_bytes()
    )

    assert document.declaration.id == ProtocolIdentifier.parse(
        "benchmarks.digits.latent-factors@0.1.0"
    )
    assert document.declaration.complexity_projections == ()
    assert document.digest == ContentDigest.from_value(document.declaration.to_record())


def test_latent_factor_declaration_digest_is_stable() -> None:
    declaration = _digits_declaration(sequence_length=2)
    record = declaration.to_record()
    reordered = {
        "sample_factors": record["sample_factors"],
        "construction_factors": record["construction_factors"],
        "complexity_projections": record["complexity_projections"],
        "id": record["id"],
    }

    assert ContentDigest.from_value(record) == ContentDigest.from_value(reordered)


def test_latent_factor_declaration_rejects_duplicate_factor_names() -> None:
    record = _digits_declaration(sequence_length=1).to_record()
    sample_factors = list(cast(list[dict[str, object]], record["sample_factors"]))
    sample_factors.append(sample_factors[0])
    record["sample_factors"] = sample_factors

    assert str(
        capture_latent_error(lambda: LatentFactorDeclaration.from_record(record))
    ) == "duplicate sample factor: benchmarks.digits.sample.digit-identity"


def test_latent_factor_declaration_document_rejects_invalid_bytes() -> None:
    assert str(
        capture_latent_error(lambda: LatentFactorDeclarationDocument.from_bytes(b"\xff"))
    ) == "latent factor declaration must be UTF-8"
    assert str(
        capture_latent_error(lambda: LatentFactorDeclarationDocument.from_bytes(b"[]"))
    ) == "latent factor declaration must contain an object"


def _digits_declaration(sequence_length: int) -> LatentFactorDeclaration:
    return LatentFactorDeclaration(
        id=ProtocolIdentifier.parse("benchmarks.digits.latent-factors@0.1.0"),
        construction_factors=(
            GeneratorConstructionFactor(
                name=ProtocolName.parse("benchmarks.digits.construction.stroke-basis"),
                degree_measure=DegreeMeasure.constant_count(7),
            ),
        ),
        sample_factors=(
            SampleLatentFactor(
                name=ProtocolName.parse("benchmarks.digits.sample.digit-identity"),
                role="content",
                degree_measure=DegreeMeasure.discrete_choice(10),
                multiplicity=sequence_length,
            ),
            SampleLatentFactor(
                name=ProtocolName.parse("benchmarks.digits.sample.field-variation-transform"),
                role="variation",
                degree_measure=DegreeMeasure.vector_dimension(6),
                multiplicity=sequence_length,
            ),
            SampleLatentFactor(
                name=ProtocolName.parse("benchmarks.digits.materialization.canvas-side"),
                role="materialization",
                degree_measure=DegreeMeasure.scalar(),
            ),
        ),
        complexity_projections=(
            ComplexityProjection(
                name=ProtocolName.parse("benchmarks.digits.complexity.content"),
                coordinate="C",
                included_roles=frozenset({"content"}),
            ),
        ),
    )


def capture_latent_error(call: Callable[[], object]) -> LatentFactorValidationError:
    with pytest.raises(LatentFactorValidationError) as error:
        call()
    return error.value
