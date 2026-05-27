from pathlib import Path

from leibniz.architecture_candidates import (
    ArchitectureSearchDistribution,
    default_architecture_search_distribution,
    generate_architecture_candidates,
    sample_architecture_candidates,
)


def test_default_search_distribution_expands_formal_operator_architectures() -> None:
    candidates = generate_architecture_candidates(
        default_architecture_search_distribution(),
        input_shape=(1, 32, 32),
        output_count=10,
    )

    assert len(candidates) == 32
    assert [candidate.parameter("local_support_size") for candidate in candidates[:3]] == [
        1,
        2,
        3,
    ]
    assert [candidate.parameter_count for candidate in candidates[:3]] == [20, 50, 100]
    assert all(candidate.architecture.input_shape == (1, 32, 32) for candidate in candidates)
    assert all(candidate.architecture.output_shape == (10,) for candidate in candidates)
    assert candidates[1].coordinate("operator.0.local_support_size") == 2
    assert candidates[1].coordinate("operator.0.support") == "local-window"
    assert candidates[1].coordinate("resource.parameter_count") == 50
    assert {
        operator.descriptor.kind
        for candidate in candidates
        for operator in candidate.operator_plan.operators
    } == {"affine-readout", "local-aggregation", "rank-collapse"}


def test_search_distribution_applies_generic_cost_bounds_and_deduplicates() -> None:
    distribution = ArchitectureSearchDistribution(
        local_support_dimension=2,
        local_support_size_minimum=1,
        local_support_size_maximum=4,
        parameter_count_minimum=50,
        parameter_count_maximum=170,
    )
    candidates = generate_architecture_candidates(
        distribution,
        input_shape=(1, 8, 8),
        output_count=10,
    )

    assert [candidate.parameter("local_support_size") for candidate in candidates] == [2, 3, 4]
    assert [candidate.parameter_count for candidate in candidates] == [50, 100, 170]
    assert len({candidate.architecture.digest for candidate in candidates}) == len(candidates)


def test_search_distribution_is_independent_of_benchmark_identity() -> None:
    small_output = generate_architecture_candidates(
        default_architecture_search_distribution(),
        input_shape=(1, 6, 6),
        output_count=3,
    )
    larger_output = generate_architecture_candidates(
        default_architecture_search_distribution(),
        input_shape=(1, 6, 6),
        output_count=7,
    )

    assert len(small_output) == len(larger_output) == 6
    assert small_output[0].architecture.output_shape == (3,)
    assert larger_output[0].architecture.output_shape == (7,)


def test_candidate_sampler_draws_bounded_deterministic_subsets_without_enumerating() -> None:
    sampled = sample_architecture_candidates(
        default_architecture_search_distribution(),
        input_shape=(1, 512, 512),
        output_count=10,
        sample_count=8,
        seed=17,
    )
    repeated = sample_architecture_candidates(
        default_architecture_search_distribution(),
        input_shape=(1, 512, 512),
        output_count=10,
        sample_count=8,
        seed=17,
    )

    assert len(sampled) == 8
    assert sampled == repeated
    assert len({candidate.architecture.digest for candidate in sampled}) == len(sampled)
    assert [candidate.parameter("local_support_size") for candidate in sampled] == [
        1,
        3,
        10,
        26,
        35,
        120,
        263,
        512,
    ]
    assert [candidate.parameter_count for candidate in sampled] == [
        20,
        100,
        1010,
        6770,
        12260,
        144010,
        691700,
        2621450,
    ]


def test_candidate_sampler_respects_resource_bounds_before_stratifying() -> None:
    sampled = sample_architecture_candidates(
        ArchitectureSearchDistribution(
            local_support_dimension=2,
            local_support_size_minimum=1,
            local_support_size_maximum=4,
            parameter_count_minimum=50,
            parameter_count_maximum=170,
        ),
        input_shape=(1, 8, 8),
        output_count=10,
        sample_count=3,
        seed=17,
    )

    assert [candidate.parameter("local_support_size") for candidate in sampled] == [2, 3, 4]
    assert [candidate.parameter_count for candidate in sampled] == [50, 100, 170]


def test_candidate_generation_does_not_name_layer_aliases_or_fixed_graphs() -> None:
    path = Path(__file__).parents[1] / "src" / "leibniz" / "architecture_candidates.py"
    source = path.read_text()

    assert "adaptive-pooling" not in source
    assert "flatten" not in source
    assert "dense" not in source
    assert "local-aggregation-readout" not in source
    assert "formal_image_classifier_architecture" not in source
