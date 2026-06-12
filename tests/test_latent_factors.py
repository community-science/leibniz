from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest
from benchmark_typing import load_digits_benchmark

from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.latent_factors import (
    DegreeMeasure,
    GeneratorConstructionFactor,
    LatentFactorDeclaration,
    LatentFactorDeclarationDocument,
    LatentFactorValidationError,
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


def test_digits_latent_factor_declaration_is_python_owned() -> None:
    declaration = load_digits_benchmark(_digits_benchmark_root).latent_factors

    assert declaration.id == ProtocolIdentifier.parse(
        "benchmarks.digits.latent-factors@0.2.0"
    )
    assert ContentDigest.from_value(declaration.to_record()) == declaration.digest


def test_latent_factor_declaration_digest_is_stable() -> None:
    declaration = _digits_declaration(sample_multiplicity=2)
    record = declaration.to_record()
    reordered = {
        "sample_factors": record["sample_factors"],
        "construction_factors": record["construction_factors"],
        "id": record["id"],
    }

    assert ContentDigest.from_value(record) == ContentDigest.from_value(reordered)


def test_latent_factor_declaration_rejects_duplicate_factor_names() -> None:
    record = _digits_declaration(sample_multiplicity=1).to_record()
    sample_factors = list(cast(list[dict[str, object]], record["sample_factors"]))
    sample_factors.append(sample_factors[0])
    record["sample_factors"] = sample_factors

    assert str(
        capture_latent_error(lambda: LatentFactorDeclaration.from_record(record))
    ) == "duplicate sample factor: benchmarks.digits.sample.digit-identity"


def test_latent_factor_declaration_rejects_removed_complexity_projection_key() -> None:
    record = _digits_declaration(sample_multiplicity=1).to_record()
    record["complexity_projections"] = []

    assert str(
        capture_latent_error(lambda: LatentFactorDeclaration.from_record(record))
    ) == "complexity_projections: unknown field"


def test_latent_factor_declaration_rejects_removed_resolution_requirements_key() -> None:
    record = _digits_declaration(sample_multiplicity=1).to_record()
    record["resolution_requirements"] = []

    assert str(
        capture_latent_error(lambda: LatentFactorDeclaration.from_record(record))
    ) == "resolution_requirements: unknown field"


def test_latent_factor_declaration_document_rejects_invalid_bytes() -> None:
    assert str(
        capture_latent_error(lambda: LatentFactorDeclarationDocument.from_bytes(b"\xff"))
    ) == "latent factor declaration must be UTF-8"
    assert str(
        capture_latent_error(lambda: LatentFactorDeclarationDocument.from_bytes(b"[]"))
    ) == "latent factor declaration must contain an object"


def _digits_declaration(sample_multiplicity: int) -> LatentFactorDeclaration:
    return LatentFactorDeclaration(
        id=ProtocolIdentifier.parse("benchmarks.digits.latent-factors@0.2.0"),
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
                multiplicity=sample_multiplicity,
            ),
            SampleLatentFactor(
                name=ProtocolName.parse("benchmarks.digits.sample.field-variation-transform"),
                role="variation",
                degree_measure=DegreeMeasure.vector_dimension(6),
                multiplicity=sample_multiplicity,
            ),
            SampleLatentFactor(
                name=ProtocolName.parse("benchmarks.digits.materialization.canvas-side"),
                role="materialization",
                degree_measure=DegreeMeasure.scalar(),
            ),
        ),
    )


def capture_latent_error(call: Callable[[], object]) -> LatentFactorValidationError:
    with pytest.raises(LatentFactorValidationError) as error:
        call()
    return error.value
