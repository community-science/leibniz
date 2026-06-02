from pathlib import Path

from leibniz.observation_generation import load_observation_generator

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"


def test_digits_resolution_analysis_preserves_digit_discriminability() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    report = generator.formation.component_discriminability_report(
        width=20,
        height=20,
        minimum_pairwise_l1=generator.benchmark_manifest.resolution_discriminability_margin(),
    )

    assert report.passed
    assert report.component_count == 10
    assert report.variation_case_count == 1
    assert report.minimum_pairwise_l1 >= 20.0


def test_digits_resolution_analysis_finds_minimum_live_resolution() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    assert generator.formation.minimum_discriminatable_resolution(
        minimum_width=1,
        minimum_height=1,
        sequence_length=1,
        minimum_pairwise_l1=generator.benchmark_manifest.resolution_discriminability_margin(),
    ) == (20, 20)
    assert generator.formation.minimum_discriminatable_resolution(
        minimum_width=3,
        minimum_height=1,
        sequence_length=3,
        minimum_pairwise_l1=generator.benchmark_manifest.resolution_discriminability_margin(),
    ) == (60, 20)


def test_digits_resolution_analysis_keeps_reported_console_sample_readable() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    batch = generator.sample_formation_batch(
        component_count=1,
        sample_count=1,
        seed=4703,
        component_sequences=((1,),),
    )
    sample = batch.samples[0]

    assert (sample.width, sample.height) == (89, 144)
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
        assert generator.formation.component_discriminability_passes(
            width=sample.width,
            height=sample.height,
            sequence_length=len(sample.component_sequence),
            sequence_index=sequence_index,
            variation_coordinates=(coordinate,),
            minimum_pairwise_l1=generator.benchmark_manifest.resolution_discriminability_margin(),
        )


def test_digits_resolution_analysis_certifies_sampled_training_affines() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    batch = generator.sample_formation_batch(
        component_count=1,
        sample_count=64,
        seed=123,
        memory_limit_bytes=100_000_000,
    )

    assert (batch.samples[0].width, batch.samples[0].height) == (32, 42)
    for sample in batch.samples:
        for coordinate in sample.variation_coordinates:
            assert generator.formation.component_discriminability_passes(
                width=sample.width,
                height=sample.height,
                sequence_length=len(sample.component_sequence),
                sequence_index=0,
                variation_coordinates=(coordinate,),
                minimum_pairwise_l1=(
                    generator.benchmark_manifest.resolution_discriminability_margin()
                ),
            )


def test_digits_resolution_analysis_detects_destroyed_discriminability() -> None:
    generator = load_observation_generator(_digits_benchmark_root)

    report = generator.formation.component_discriminability_report(
        width=4,
        height=4,
        minimum_pairwise_l1=generator.benchmark_manifest.resolution_discriminability_margin(),
    )

    assert not report.passed
    assert report.minimum_pairwise_l1 < 20.0
    assert report.nearest_pair is not None
