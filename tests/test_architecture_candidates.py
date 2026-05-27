from leibniz.architecture_candidates import (
    ArchitectureCandidateRecipe,
    ArchitectureCandidateSpace,
    default_architecture_candidate_space,
    generate_architecture_candidates,
    sample_architecture_candidates,
)


def test_default_candidate_space_expands_formal_operator_architectures() -> None:
    candidates = generate_architecture_candidates(
        default_architecture_candidate_space(),
        input_shape=(1, 32, 32),
        output_count=10,
    )

    assert len(candidates) == 32
    assert [candidate.parameter("local_aggregation_size") for candidate in candidates[:3]] == [
        1,
        2,
        3,
    ]
    assert [candidate.parameter_count for candidate in candidates[:3]] == [20, 50, 100]
    assert all(candidate.architecture.input_shape == (1, 32, 32) for candidate in candidates)
    assert all(candidate.architecture.output_shape == (10,) for candidate in candidates)
    assert {
        operator.descriptor.kind
        for candidate in candidates
        for operator in candidate.operator_plan.operators
    } == {"affine-readout", "local-aggregation", "rank-collapse"}


def test_candidate_space_applies_generic_cost_bounds_and_deduplicates() -> None:
    bounded_recipe = ArchitectureCandidateRecipe(
        kind="local-aggregation-readout",
        local_aggregation_dimension=2,
        local_aggregation_size_minimum=1,
        local_aggregation_size_maximum=4,
        parameter_count_minimum=50,
        parameter_count_maximum=170,
    )
    candidates = generate_architecture_candidates(
        ArchitectureCandidateSpace(recipes=(bounded_recipe, bounded_recipe)),
        input_shape=(1, 8, 8),
        output_count=10,
    )

    assert [candidate.parameter("local_aggregation_size") for candidate in candidates] == [2, 3, 4]
    assert [candidate.parameter_count for candidate in candidates] == [50, 100, 170]
    assert len({candidate.architecture.digest for candidate in candidates}) == len(candidates)


def test_candidate_space_is_independent_of_benchmark_identity() -> None:
    small_output = generate_architecture_candidates(
        default_architecture_candidate_space(),
        input_shape=(1, 6, 6),
        output_count=3,
    )
    larger_output = generate_architecture_candidates(
        default_architecture_candidate_space(),
        input_shape=(1, 6, 6),
        output_count=7,
    )

    assert len(small_output) == len(larger_output) == 6
    assert small_output[0].architecture.output_shape == (3,)
    assert larger_output[0].architecture.output_shape == (7,)


def test_candidate_sampler_draws_bounded_deterministic_subsets_without_enumerating() -> None:
    sampled = sample_architecture_candidates(
        default_architecture_candidate_space(),
        input_shape=(1, 512, 512),
        output_count=10,
        sample_count=8,
        seed=17,
    )
    repeated = sample_architecture_candidates(
        default_architecture_candidate_space(),
        input_shape=(1, 512, 512),
        output_count=10,
        sample_count=8,
        seed=17,
    )

    assert len(sampled) == 8
    assert sampled == repeated
    assert len({candidate.architecture.digest for candidate in sampled}) == len(sampled)
    assert all(
        1 <= candidate.parameter("local_aggregation_size") <= 512
        for candidate in sampled
    )
