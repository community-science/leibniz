from pathlib import Path
from typing import Any, cast

from benchmark_typing import (
    DigitsGenerator,
    load_digits_generator,
    sample_height,
    sample_width,
)

from leibniz.observation_generation import (
    GeneratedSampleSet,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def _formation_payload(
    generator: DigitsGenerator,
    **kwargs: object,
) -> GeneratedSampleSet:
    sample_set = generator(**cast(Any, kwargs))
    return sample_set


def test_digits_resolution_analysis_preserves_digit_discriminability() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    report = generator.formation.component_discriminability_report(
        width=20,
        height=20,
        minimum_pairwise_l1=generator.manifest.resolution_discriminability_margin(),
    )

    assert report.passed
    assert report.component_vocabulary_size == 10
    assert report.variation_case_count == 1
    assert report.minimum_pairwise_l1 >= 20.0


def test_digits_resolution_analysis_finds_minimum_live_resolution() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    assert generator.formation.minimum_discriminatable_resolution(
        minimum_width=1,
        minimum_height=1,
        minimum_pairwise_l1=generator.manifest.resolution_discriminability_margin(),
    ) == (13, 24)
    assert generator.formation.minimum_discriminatable_resolution(
        minimum_width=3,
        minimum_height=1,
        minimum_pairwise_l1=generator.manifest.resolution_discriminability_margin(),
    ) == (13, 24)


# The native render footprint of a digit at the fixed pitch. Setups place this
# footprint-sized digit at centre-relative offsets on a larger canvas; the
# digit pixels are identical to the native rendering, so discriminability is
# certified once at the native footprint rather than at each placement.
_native_footprint = 28


def test_digits_resolution_analysis_keeps_reported_console_sample_readable() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    batch = _formation_payload(
        generator,
        shape=1,
        seed=4703,
    )
    sample = batch.samples[0]

    # The materialized canvas frames a native-footprint digit (canvas >= footprint).
    assert sample_width(sample) >= _native_footprint
    assert sample_height(sample) >= _native_footprint
    report = generator.formation.component_discriminability_report(
        width=_native_footprint,
        height=_native_footprint,
        minimum_pairwise_l1=generator.manifest.resolution_discriminability_margin(),
    )
    assert report.passed


def test_digits_resolution_analysis_certifies_native_footprint_digits() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    report = generator.formation.component_discriminability_report(
        width=_native_footprint,
        height=_native_footprint,
        minimum_pairwise_l1=generator.manifest.resolution_discriminability_margin(),
    )

    assert report.passed
    assert report.component_vocabulary_size == 10
    assert report.minimum_pairwise_l1 >= generator.manifest.resolution_discriminability_margin()
    assert generator.formation.component_discriminability_passes(
        width=_native_footprint,
        height=_native_footprint,
        minimum_pairwise_l1=generator.manifest.resolution_discriminability_margin(),
    )


def test_digits_resolution_analysis_detects_destroyed_discriminability() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    report = generator.formation.component_discriminability_report(
        width=4,
        height=4,
        minimum_pairwise_l1=generator.manifest.resolution_discriminability_margin(),
    )

    assert not report.passed
    assert report.minimum_pairwise_l1 < 20.0
    assert report.nearest_pair is not None
