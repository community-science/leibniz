import math
import sys
from pathlib import Path
from typing import Any, cast

from benchmark_typing import load_digits_benchmark

from leibniz.artifacts import ArtifactReference
from leibniz.benchmarks import BenchmarkManifest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import MaterializationPlanDocument
from leibniz.measurements import MeasurementRecord
from leibniz.observation_generation import StateSpaceVolumeRequest
from leibniz.outcomes import (
    AcceptedEvent,
    FiniteProbabilityMeasure,
    ProbabilityMass,
    RawScoringEvidence,
)
from leibniz.state_space import ContinuousAxisRegion, RealIntervalDomain
from leibniz.tensor_runtime import resolve_tensor_runtime

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_digits_fixture_root = _repository_root / "tests" / "fixtures" / "digits"


def test_digits_length_one_perfect_measurement_scores_full_accepted_mass() -> None:
    measurement = _measurement_for_sequence(
        sequence=(7,),
        plan_name="materialization_plan_l1.json",
        measure_kind="perfect",
    )

    measurement.validate_manifest(_digits_manifest())

    assert measurement.raw_scoring_evidence.observation_id == (
        "benchmarks.digits.observations.l1.digit-7@0.1.0"
    )
    assert measurement.raw_scoring_evidence.accepted_mass == 1.0
    assert measurement.raw_scoring_evidence.negative_log_score == 0.0
    assert [artifact.kind for artifact in measurement.evidence_artifacts] == [
        "observation-formation-declaration",
        "materialization-plan",
    ]


def test_digits_length_one_uniform_and_wrong_measurements_score_expected_mass() -> None:
    uniform = _measurement_for_sequence(
        sequence=(7,),
        plan_name="materialization_plan_l1.json",
        measure_kind="uniform",
    )
    wrong = _measurement_for_sequence(
        sequence=(7,),
        plan_name="materialization_plan_l1.json",
        measure_kind="wrong",
    )

    assert math.isclose(uniform.raw_scoring_evidence.accepted_mass, 0.1)
    assert math.isclose(uniform.raw_scoring_evidence.negative_log_score, math.log(10))
    assert wrong.raw_scoring_evidence.accepted_mass == 0.0
    assert wrong.raw_scoring_evidence.negative_log_score == math.inf


def test_digits_manifest_declares_single_digit_outcomes() -> None:
    manifest = _digits_manifest()

    assert manifest.outcome_space is not None
    assert [outcome.id for outcome in manifest.outcome_space.outcomes] == [
        f"digit-{index}" for index in range(10)
    ]


def test_digits_realized_regions_claim_continuous_transform_cells() -> None:
    benchmark = load_digits_benchmark(_digits_benchmark_root)

    batch = benchmark.generator(
        seed=407,
        shape=4,
        volume_request=StateSpaceVolumeRequest(2.0, 3.0),
    )

    assert batch.region is not None
    assert batch.region.measure_estimate is not None
    assert batch.region.measure_estimate.kind == "estimated"
    assert batch.region.components
    for component in batch.region.components:
        assert component.measure_estimate is not None
        assert component.measure_estimate.kind == "estimated"
        assert {axis_region.axis_id for axis_region in component.axis_regions} == {
            "x_translation",
            "y_translation",
            "scale",
        }
        for axis_region in component.axis_regions:
            assert isinstance(axis_region, ContinuousAxisRegion)
            assert isinstance(axis_region.axis.domain, RealIntervalDomain)
    for sample in batch.samples:
        assert sample.axis_coordinates is not None
        assert sample.region_component_index is not None
        assert set(sample.axis_coordinates) == {
            "x_translation",
            "y_translation",
            "scale",
        }
        assert "transform-ordinal" not in sample.axis_coordinates
        assert batch.region.contains(sample.region_component_index, sample.axis_coordinates)


def test_inverse_digits_secret_seed_samples_deterministic_private_latents() -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _digits_module()
    secret = b"deterministic inverse digits seed"

    left = module.sample_inverse_digits_observations(
        runtime=runtime,
        secret=secret,
        sample_count=3,
        canvas_side=32,
    )
    right = module.sample_inverse_digits_observations(
        runtime=runtime,
        secret=secret,
        sample_count=3,
        canvas_side=32,
    )
    other = module.sample_inverse_digits_observations(
        runtime=runtime,
        secret=b"different inverse digits seed!!",
        sample_count=3,
        canvas_side=32,
    )

    assert left.latents == right.latents
    assert left.observations.allclose(right.observations)
    assert not left.observations.allclose(other.observations)
    assert left.secret_digest == right.secret_digest
    assert left.secret_digest != other.secret_digest


def test_inverse_digits_submission_view_exposes_law_but_not_secret_or_latent() -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _digits_module()
    batch = module.sample_inverse_digits_observations(
        runtime=runtime,
        secret=b"submission view private seed",
        sample_count=2,
    )

    view = batch.submission_view()

    assert view["law_id"] == "benchmarks.digits.inverse-renderer@0.1.0"
    assert view["observation_shape"] == [2, 1, 28, 28]
    assert "latents" not in view
    assert "secret_digest" not in view
    assert "seed" not in view


def test_inverse_digits_renderer_changes_continuously_with_pose() -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _digits_module()
    base = module.InverseDigitsLatent(
        identity=8,
        x_translation=0.0,
        y_translation=0.0,
        scale=0.95,
        shear=0.02,
        stroke_width=1.0,
    )
    nearby = module.InverseDigitsLatent(
        identity=8,
        x_translation=0.01,
        y_translation=0.0,
        scale=0.95,
        shear=0.02,
        stroke_width=1.0,
    )
    farther = module.InverseDigitsLatent(
        identity=8,
        x_translation=0.08,
        y_translation=0.0,
        scale=0.95,
        shear=0.02,
        stroke_width=1.0,
    )

    rendered = module.render_inverse_digits(
        runtime=runtime,
        latents=(base, nearby, farther),
        canvas_side=32,
    )
    near_delta = float((rendered[0] - rendered[1]).pow(2).mean().sqrt())
    far_delta = float((rendered[0] - rendered[2]).pow(2).mean().sqrt())

    assert near_delta > 0.0
    assert far_delta > near_delta


def test_inverse_digits_canonical_template_baseline_does_not_saturate_headroom() -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _digits_module()
    observations = module.render_inverse_digits(
        runtime=runtime,
        latents=(
            module.InverseDigitsLatent(0, 0.12, -0.08, 0.82, 0.16, 1.2),
            module.InverseDigitsLatent(1, -0.14, 0.07, 1.16, -0.15, 0.8),
            module.InverseDigitsLatent(8, 0.13, 0.11, 0.78, 0.18, 1.3),
        ),
        canvas_side=32,
    )
    templates = module.render_inverse_digits(
        runtime=runtime,
        latents=tuple(
            module.InverseDigitsLatent(index, 0.0, 0.0, 1.0, 0.0, 1.0)
            for index in range(10)
        ),
        canvas_side=32,
    )
    squared_errors = (observations.reshape(3, 1, -1) - templates.reshape(1, 10, -1)).pow(2)
    template_predictions = squared_errors.mean(dim=2).argmin(dim=1)

    assert tuple(int(value) for value in template_predictions) != (0, 1, 8)


def test_inverse_digits_static_certification_bounds_perturbed_latent_error() -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _digits_module()
    true_latent = module.InverseDigitsLatent(8, 0.0, 0.0, 1.0, 0.02, 1.0)
    recovered_latent = module.InverseDigitsLatent(8, 0.01, 0.0, 1.0, 0.02, 1.0)
    observation = module.render_inverse_digits(
        runtime=runtime,
        latents=(true_latent,),
        canvas_side=32,
    )

    certificate = module.inverse_digits_static_certification(
        runtime=runtime,
        recovered_latent=recovered_latent,
        observation=observation,
        canvas_side=32,
        refinement_sides=(32, 64),
    )
    actual_error = math.sqrt(
        sum(
            (actual - recovered) ** 2
            for actual, recovered in zip(
                true_latent.to_nuisance_tuple(),
                recovered_latent.to_nuisance_tuple(),
                strict=True,
            )
        )
    )

    assert certificate.certification_status == "certified"
    assert certificate.certified_epsilon >= actual_error
    assert certificate.certified_epsilon / actual_error < 20.0
    assert certificate.residual_norm > 0.0
    assert certificate.sigma_min > 0.0


def test_inverse_digits_static_conditioning_is_stable_for_well_posed_glyph() -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _digits_module()
    latent = module.InverseDigitsLatent(8, 0.0, 0.0, 1.0, 0.02, 1.0)
    observation = module.render_inverse_digits(
        runtime=runtime,
        latents=(latent,),
        canvas_side=32,
    )

    certificate = module.inverse_digits_static_certification(
        runtime=runtime,
        recovered_latent=latent,
        observation=observation,
        canvas_side=32,
        refinement_sides=(32, 64),
    )
    record = certificate.to_record()

    assert certificate.certification_status == "certified"
    assert certificate.conditioning_stability < 1.1
    assert record["estimator"] == "renderer-jvp-gram-sigma-min"
    assert record["sigma_min_ladder"] == list(certificate.sigma_min_ladder)


def test_inverse_digits_product_entropy_is_resolution_independent() -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _digits_module()
    latent = module.InverseDigitsLatent(8, 0.0, 0.0, 1.0, 0.02, 1.0)

    coarse = module.inverse_digits_validated_bits(
        runtime=runtime,
        recovered_latent=latent,
        certified_epsilon=1.0e-2,
        canvas_side=32,
    )
    fine = module.inverse_digits_validated_bits(
        runtime=runtime,
        recovered_latent=latent,
        certified_epsilon=1.0e-2,
        canvas_side=64,
    )

    assert coarse.distinguishable_identity_count == fine.distinguishable_identity_count
    assert math.isclose(coarse.bits, fine.bits, rel_tol=0.0, abs_tol=0.05)


def test_inverse_digits_identity_bits_drop_at_certified_precision_boundary() -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _digits_module()
    latent = module.InverseDigitsLatent(8, 0.0, 0.0, 1.0, 0.02, 1.0)

    resolved = module.inverse_digits_validated_bits(
        runtime=runtime,
        recovered_latent=latent,
        certified_epsilon=1.0e-2,
        canvas_side=32,
    )
    unresolved = module.inverse_digits_validated_bits(
        runtime=runtime,
        recovered_latent=latent,
        certified_epsilon=10.0,
        canvas_side=32,
    )

    assert resolved.identity_bits > 0.0
    assert unresolved.identity_bits == 0.0
    assert unresolved.distinguishable_identity_count == 1


def test_inverse_digits_bits_rise_as_reconstruction_residual_falls() -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _digits_module()
    true_latent = module.InverseDigitsLatent(8, 0.0, 0.0, 1.0, 0.02, 1.0)
    observation = module.render_inverse_digits(
        runtime=runtime,
        latents=(true_latent,),
        canvas_side=32,
    )

    def bits_for_offset(offset: float) -> float:
        recovered_latent = module.InverseDigitsLatent(8, offset, 0.0, 1.0, 0.02, 1.0)
        certificate = module.inverse_digits_static_certification(
            runtime=runtime,
            recovered_latent=recovered_latent,
            observation=observation,
            canvas_side=32,
            refinement_sides=(32, 64),
        )
        return module.inverse_digits_validated_bits(
            runtime=runtime,
            recovered_latent=recovered_latent,
            certified_epsilon=certificate.certified_epsilon,
            canvas_side=32,
        ).bits

    assert bits_for_offset(0.005) > bits_for_offset(0.02) > 0.0


def test_inverse_digits_label_free_probe_trains_by_reconstruction_only() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    module = _digits_module()
    true_latent = module.InverseDigitsLatent(8, 0.02, -0.01, 1.0, 0.02, 1.0)
    observations = module.render_inverse_digits(
        runtime=runtime,
        latents=(true_latent,),
        canvas_side=28,
    )

    identity_logits = torch.zeros((1, 10), requires_grad=True)
    nuisance = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0]], requires_grad=True)
    optimizer = torch.optim.Adam([identity_logits, nuisance], lr=0.1)

    initial_loss = float(
        module.inverse_digits_reconstruction_loss(
            runtime=runtime,
            identity_logits=identity_logits,
            nuisance=nuisance,
            observations=observations,
            canvas_side=28,
        ).detach()
    )
    for _step in range(80):
        optimizer.zero_grad()
        loss = module.inverse_digits_reconstruction_loss(
            runtime=runtime,
            identity_logits=identity_logits,
            nuisance=nuisance,
            observations=observations,
            canvas_side=28,
        )
        loss.backward()
        optimizer.step()

    final_loss = float(
        module.inverse_digits_reconstruction_loss(
            runtime=runtime,
            identity_logits=identity_logits,
            nuisance=nuisance,
            observations=observations,
            canvas_side=28,
        ).detach()
    )

    assert final_loss < initial_loss * 0.01
    assert int(identity_logits.detach().argmax(dim=1)[0]) == true_latent.identity


def _measurement_for_sequence(
    *,
    sequence: tuple[int, ...],
    plan_name: str,
    measure_kind: str,
) -> MeasurementRecord:
    manifest = _digits_manifest()
    declaration = load_digits_benchmark(_digits_benchmark_root).formation
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / plan_name).read_bytes()
    ).plan
    scale = 1
    outcome_space = manifest.resolve_outcome_space()
    assert manifest.outcome_space is not None
    assert len(sequence) == 1
    accepted_outcome = f"digit-{sequence[0]}"
    sequence_label = accepted_outcome.removeprefix("digit-")
    observation_id = ProtocolIdentifier.parse(
        f"benchmarks.digits.observations.l{scale}.digit-{sequence_label}@0.1.0"
    )
    accepted_event = AcceptedEvent.from_record(
        {
            "id": f"benchmarks.digits.events.l{scale}.digit-{sequence_label}@0.1.0",
            "outcome_space_id": str(outcome_space.id),
            "outcomes": [accepted_outcome],
        },
        outcome_space=outcome_space,
    )
    probability_measure = FiniteProbabilityMeasure(
        id=ProtocolIdentifier.parse(
            f"benchmarks.digits.measures.l{scale}.digit-{sequence_label}.{measure_kind}@0.1.0"
        ),
        outcome_space_id=outcome_space.id,
        probabilities=_probabilities(
            outcome_ids=tuple(outcome.id for outcome in outcome_space.outcomes),
            accepted_outcome=accepted_outcome,
            measure_kind=measure_kind,
        ),
    )
    return MeasurementRecord(
        benchmark_id=manifest.id,
        outcome_space=outcome_space,
        accepted_event=accepted_event,
        probability_measure=probability_measure,
        raw_scoring_evidence=RawScoringEvidence.from_event_and_measure(
            id=ProtocolIdentifier.parse(
                f"benchmarks.digits.evidence.l{scale}.digit-{sequence_label}.{measure_kind}@0.1.0"
            ),
            observation_id=str(observation_id),
            event=accepted_event,
            measure=probability_measure,
        ),
        evidence_artifacts=(
            ArtifactReference(
                kind="observation-formation-declaration",
                protocol_id=declaration.id,
                record_digest=declaration.digest,
            ),
            ArtifactReference(
                kind="materialization-plan",
                protocol_id=plan.id,
                record_digest=plan.digest,
            ),
        ),
    )


def _probabilities(
    *,
    outcome_ids: tuple[str, ...],
    accepted_outcome: str,
    measure_kind: str,
) -> tuple[ProbabilityMass, ...]:
    if measure_kind == "perfect":
        return (ProbabilityMass(accepted_outcome, 1.0),)
    if measure_kind == "wrong":
        wrong_outcome = next(
            outcome_id for outcome_id in outcome_ids if outcome_id != accepted_outcome
        )
        return (ProbabilityMass(wrong_outcome, 1.0),)
    if measure_kind == "uniform":
        probability = 1.0 / len(outcome_ids)
        return tuple(ProbabilityMass(outcome_id, probability) for outcome_id in outcome_ids)
    raise AssertionError(f"unknown measure kind: {measure_kind}")


def _digits_manifest() -> BenchmarkManifest:
    return load_digits_benchmark(_digits_benchmark_root).manifest


def _digits_module() -> Any:
    loaded = cast(Any, load_digits_benchmark(_digits_benchmark_root))
    return sys.modules[type(loaded.implementation).__module__]
