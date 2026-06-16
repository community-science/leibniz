import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from benchmark_typing import load_digits_benchmark

from leibniz.benchmarks import BenchmarkManifest
from leibniz.observation_generation import StateSpaceVolumeRequest
from leibniz.state_space import ContinuousAxisRegion, RealIntervalDomain
from leibniz.tensor_runtime import resolve_tensor_runtime

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"

def test_digits_manifest_declares_inverse_contract_without_finite_outcomes() -> None:
    manifest = _digits_manifest()
    benchmark = load_digits_benchmark(_digits_benchmark_root)

    assert manifest.outcome_space is None
    assert benchmark.target_contract.kind == "inverse"
    assert benchmark.target_contract.expected_output_shape(None) == (15,)


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


def test_inverse_digits_canonical_template_baseline_leaves_headroom_on_prior() -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _digits_module()
    batch = module.sample_inverse_digits_observations(
        runtime=runtime,
        secret=b"deterministic inverse headroom seed",
        sample_count=64,
        canvas_side=32,
    )
    observations = module.render_inverse_digits(
        runtime=runtime,
        latents=batch.latents,
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
    true_residual = (observations - batch.observations).pow(2).mean().sqrt()
    squared_errors = (
        observations.reshape(len(batch.latents), 1, -1) - templates.reshape(1, 10, -1)
    ).pow(2)
    template_predictions = squared_errors.mean(dim=2).argmin(dim=1)
    template_residual = squared_errors.mean(dim=2).min(dim=1).values.sqrt().mean()
    matched = sum(
        int(template_predictions[index]) == latent.identity
        for index, latent in enumerate(batch.latents)
    )

    assert float(true_residual) == 0.0
    assert matched <= len(batch.latents) // 2
    assert float(template_residual) > 0.2


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
    # This uses the worst nuisance direction from sigma_min(J), not the
    # perturbation's directional derivative, so it is conservative by design.
    assert certificate.certified_epsilon / actual_error < 12.5
    assert certificate.residual_norm > 0.0
    assert certificate.sigma_min > 0.0


def test_inverse_digits_static_conditioning_refuses_degenerate_submitted_latent() -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _digits_module()
    true_latent = module.InverseDigitsLatent(8, 0.0, 0.0, 1.0, 0.02, 1.0)
    degenerate_recovered = module.InverseDigitsLatent(8, 0.0, 0.0, 0.0, 0.0, 1.0)
    observation = module.render_inverse_digits(
        runtime=runtime,
        latents=(true_latent,),
        canvas_side=28,
    )

    certificate = module.inverse_digits_static_certification(
        runtime=runtime,
        recovered_latent=degenerate_recovered,
        observation=observation,
        canvas_side=28,
        refinement_sides=(28, 56),
    )
    bits = module.inverse_digits_validated_bits(
        runtime=runtime,
        recovered_latent=degenerate_recovered,
        certified_epsilon=certificate.certified_epsilon,
        image_epsilon=certificate.residual_norm,
        canvas_side=28,
    )

    assert certificate.certification_status == "refused-conditioning-unstable"
    assert certificate.sigma_min == 0.0
    assert certificate.conditioning_stability == math.inf
    assert bits.bits == 0.0
    assert bits.distinguishable_identity_count == 1


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
        image_epsilon=1.0e-2,
        canvas_side=32,
    )
    fine = module.inverse_digits_validated_bits(
        runtime=runtime,
        recovered_latent=latent,
        certified_epsilon=1.0e-2,
        image_epsilon=1.0e-2,
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
        image_epsilon=1.0e-2,
        canvas_side=32,
    )
    unresolved = module.inverse_digits_validated_bits(
        runtime=runtime,
        recovered_latent=latent,
        certified_epsilon=10.0,
        image_epsilon=10.0,
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

    def bits_for_offset(offset: float) -> Any:
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
            image_epsilon=certificate.residual_norm,
            canvas_side=32,
        )

    near = bits_for_offset(0.005)
    far = bits_for_offset(0.02)

    assert near.bits > far.bits > 0.0
    assert near.identity_bits == math.log2(10)
    assert near.distinguishable_identity_count == 10


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


def test_inverse_digits_mnist_held_out_check_if_local_idx_available() -> None:
    image_path = os.environ.get("LEIBNIZ_MNIST_IMAGES_IDX")
    label_path = os.environ.get("LEIBNIZ_MNIST_LABELS_IDX")
    if image_path is None or label_path is None:
        pytest.skip("set LEIBNIZ_MNIST_IMAGES_IDX and LEIBNIZ_MNIST_LABELS_IDX")
    images = _load_mnist_idx_images(Path(image_path), limit=32)
    labels = _load_mnist_idx_labels(Path(label_path), limit=32)
    if len(images) != len(labels) or len(images) < 16:
        pytest.skip("local MNIST IDX sample must contain at least 16 paired examples")

    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    module = _digits_module()
    observations = torch.tensor(images, dtype=torch.float32).reshape((len(labels), 1, 28, 28))
    identity_logits = torch.zeros((len(labels), 10), requires_grad=True)
    nuisance = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 1.0] for _label in labels],
        dtype=torch.float32,
        requires_grad=True,
    )
    optimizer = torch.optim.Adam([identity_logits, nuisance], lr=0.08)
    for _step in range(120):
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

    recovered = tuple(int(value) for value in identity_logits.detach().argmax(dim=1).tolist())
    matches = sum(
        1
        for recovered_digit, label in zip(recovered, labels, strict=True)
        if recovered_digit == label
    )
    assert _binomial_upper_tail(matches, len(labels), 0.1) < 0.05


def _digits_manifest() -> BenchmarkManifest:
    return load_digits_benchmark(_digits_benchmark_root).manifest


def _digits_module() -> Any:
    loaded = cast(Any, load_digits_benchmark(_digits_benchmark_root))
    return sys.modules[type(loaded.implementation).__module__]


def _load_mnist_idx_images(path: Path, *, limit: int) -> list[float]:
    data = path.read_bytes()
    if len(data) < 16:
        raise ValueError("MNIST image IDX file is truncated")
    magic, count, rows, columns = struct.unpack(">IIII", data[:16])
    if magic != 2051 or rows != 28 or columns != 28:
        raise ValueError("expected MNIST image IDX with 28x28 images")
    image_count = min(count, limit)
    expected = 16 + image_count * rows * columns
    if len(data) < expected:
        raise ValueError("MNIST image IDX file is truncated")
    return [value / 255.0 for value in data[16:expected]]


def _load_mnist_idx_labels(path: Path, *, limit: int) -> list[int]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError("MNIST label IDX file is truncated")
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError("expected MNIST label IDX")
    label_count = min(count, limit)
    expected = 8 + label_count
    if len(data) < expected:
        raise ValueError("MNIST label IDX file is truncated")
    return [int(value) for value in data[8:expected]]


def _binomial_upper_tail(successes: int, trials: int, probability: float) -> float:
    return math.fsum(
        math.comb(trials, count)
        * probability**count
        * (1.0 - probability) ** (trials - count)
        for count in range(successes, trials + 1)
    )
