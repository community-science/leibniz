from pathlib import Path
from typing import Any, cast

from benchmark_typing import (
    DigitsGenerator,
    load_digits_generator,
    sample_height,
    sample_width,
)

from leibniz.materialization import AxisAssignment
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


def test_digits_resolution_analysis_keeps_reported_console_sample_readable() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    batch = _formation_payload(
        generator,
        shape=1,
        seed=4703,
        component_indices=(1,),
        resolution_assignment=AxisAssignment(values={"W": 24, "H": 24}),
    )
    sample = batch.samples[0]

    width = sample_width(sample)
    height = sample_height(sample)
    assert width % 24 == 0
    assert height % 24 == 0
    for coordinate in sample.variation_coordinates:
        report = generator.formation.component_discriminability_report(
            width=width,
            height=height,
            variation_coordinates=(coordinate,),
            minimum_pairwise_l1=generator.manifest.resolution_discriminability_margin(),
        )
        assert report.passed
        assert generator.formation.component_discriminability_passes(
            width=width,
            height=height,
            variation_coordinates=(coordinate,),
            minimum_pairwise_l1=generator.manifest.resolution_discriminability_margin(),
        )


def test_digits_resolution_analysis_certifies_sampled_training_affines() -> None:
    generator = load_digits_generator(_digits_benchmark_root)

    batch = _formation_payload(
        generator,
        shape=64,
        seed=123,
        memory_limit_bytes=100_000_000,
        resolution_assignment=AxisAssignment(values={"W": 24, "H": 24}),
    )

    for sample in batch.samples:
        width = sample_width(sample)
        height = sample_height(sample)
        assert width % 24 == 0
        assert height % 24 == 0
        for coordinate in sample.variation_coordinates:
            assert generator.formation.component_discriminability_passes(
                width=width,
                height=height,
                variation_coordinates=(coordinate,),
                minimum_pairwise_l1=(
                    generator.manifest.resolution_discriminability_margin()
                ),
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
