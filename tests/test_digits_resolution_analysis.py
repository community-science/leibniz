from pathlib import Path

from leibniz.observation_generation import load_observation_generator

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def test_digits_resolution_analysis_preserves_digit_discriminability() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    report = generator.formation.component_discriminability_report(
        width=9,
        height=4,
        variation_coordinates=generator.formation.boundary_variation_coordinates(
            sequence_index=0
        ),
    )

    assert report.passed
    assert report.component_count == 10
    assert report.variation_case_count == 64
    assert report.minimum_pairwise_l1 > 0.0


def test_digits_resolution_analysis_finds_minimum_live_resolution() -> None:
    generator = load_observation_generator(_digits_benchmark_root)
    boundary_coordinates = generator.formation.boundary_variation_coordinates(sequence_index=0)

    assert generator.formation.minimum_discriminatable_resolution(
        minimum_width=1,
        minimum_height=1,
        sequence_length=1,
        variation_coordinates=boundary_coordinates,
        minimum_pairwise_l1=generator.benchmark_manifest.resolution_discriminability_margin(),
    ) == (30, 30)
    assert generator.formation.minimum_discriminatable_resolution(
        minimum_width=3,
        minimum_height=1,
        sequence_length=3,
        variation_coordinates=boundary_coordinates,
        minimum_pairwise_l1=generator.benchmark_manifest.resolution_discriminability_margin(),
    ) == (90, 30)


def test_digits_resolution_analysis_keeps_reported_console_sequence_readable() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    batch = generator.sample_formation_batch(
        scale=7,
        sample_count=1,
        seed=4703,
        component_sequences=((1, 0, 6, 1, 0, 0, 4),),
    )
    sample = batch.samples[0]

    assert (sample.width, sample.height) == (363, 41)
    for sequence_index, coordinate in enumerate(sample.variation_coordinates):
        report = generator.formation.component_discriminability_report(
            width=sample.width,
            height=sample.height,
            sequence_length=len(sample.component_sequence),
            sequence_index=sequence_index,
            variation_coordinates=(coordinate,),
            minimum_pairwise_l1=generator.benchmark_manifest.resolution_discriminability_margin(),
        )
        assert report.passed


def test_digits_resolution_analysis_detects_destroyed_discriminability() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    report = generator.formation.component_discriminability_report(
        width=4,
        height=4,
        variation_coordinates=generator.formation.boundary_variation_coordinates(
            sequence_index=0
        )[:1],
    )

    assert not report.passed
    assert report.minimum_pairwise_l1 == 0.0
    assert report.nearest_pair is not None
