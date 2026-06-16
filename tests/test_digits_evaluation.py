import math
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
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
    assert benchmark.target_contract.expected_output_shape(None) == (85,)


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


def test_inverse_digits_deformation_renderer_has_pose_aware_headroom() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    module = _digits_module()
    true_latent = module._inverse_digits_latent_from_secret(
        b"headroom deformation seed",
        sample_index=0,
    )
    observation = module.render_inverse_digits(
        runtime=runtime,
        latents=(true_latent,),
        canvas_side=28,
    )

    observation = observation.detach()
    deform_dim = module._inverse_deformation_dimension
    freq = torch.tensor(
        [fx + fy for fx, fy in module._inverse_deformation_modes],
        dtype=torch.float32,
    )
    freq_full = torch.cat([freq, freq])
    torch.manual_seed(0)

    def solve(stages: tuple[object, ...], steps: int) -> Any:
        identity_logits = torch.zeros((1, 10), requires_grad=True)
        affine = torch.tensor([[0.0, 0.0, 1.0, 0.0, 1.0]], requires_grad=True)
        deform = torch.zeros((1, deform_dim), requires_grad=True)
        for cutoff in stages:
            active = (
                torch.ones(deform_dim, dtype=torch.bool)
                if cutoff == "all"
                else freq_full <= float(cast(float, cutoff))
            )
            optimizer = torch.optim.Adam([identity_logits, affine, deform], lr=0.05)
            for _step in range(steps):
                optimizer.zero_grad()
                nuisance = torch.cat([affine, deform], dim=1)
                module.inverse_digits_reconstruction_loss(
                    runtime=runtime,
                    identity_logits=identity_logits,
                    nuisance=nuisance,
                    observations=observation,
                    canvas_side=28,
                ).backward()
                if deform.grad is not None:
                    deform.grad[0, ~active] = 0.0
                optimizer.step()
                with torch.no_grad():
                    deform[0, ~active] = 0.0
        identity = int(identity_logits.detach().argmax(dim=1)[0])
        nuisance_vec = tuple(
            float(value) for value in torch.cat([affine, deform], dim=1).detach()[0]
        )
        return module._inverse_digits_latent_from_nuisance_vector(identity, nuisance_vec)

    # Cheap: the obvious pose-aware solver -- full-latent gradient descent from a generic
    # init. Strong: coarse-to-fine over deformation frequency. Both are real (residual>0),
    # so this is the genuine open-frontier gap, not the oracle and not a no-pose strawman.
    cheap_latent = solve(("all",), 300)
    strong_latent = solve((1.0, 2.0, "all"), 150)
    cheap_bits = _certified_bits_for_latent(
        module=module,
        runtime=runtime,
        recovered_latent=cheap_latent,
        observation=observation,
        canvas_side=28,
    )
    strong_bits = _certified_bits_for_latent(
        module=module,
        runtime=runtime,
        recovered_latent=strong_latent,
        observation=observation,
        canvas_side=28,
    )

    assert cheap_bits.bits > 0.0
    assert strong_bits.bits - cheap_bits.bits > 8.0


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
    # The graded renderer includes null and near-null high-frequency deformation
    # directions; the certificate uses the smallest resolvable singular value, so
    # an affine perturbation is intentionally much tighter than the global bound.
    assert certificate.certified_epsilon / actual_error > 100.0
    assert certificate.residual_norm > 0.0
    assert certificate.sigma_min > 0.0


def test_inverse_digits_static_conditioning_refuses_degenerate_submitted_latent() -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _digits_module()
    true_latent = module.InverseDigitsLatent(8, 0.0, 0.0, 1.0, 0.02, 1.0)
    degenerate_recovered = module.InverseDigitsLatent(8, 0.0, 0.0, 0.0, 0.0, 0.0)
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
        certification=certificate,
        canvas_side=28,
    )

    assert certificate.certification_status == "refused-conditioning-unstable"
    assert certificate.sigma_min == 0.0
    assert certificate.conditioning_stability > 8.0
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
    assert certificate.conditioning_stability < 2.0
    assert record["estimator"] == "renderer-jvp-gram-per-mode-spectrum"
    assert record["sigma_min_ladder"] == list(certificate.sigma_min_ladder)


def test_inverse_digits_product_entropy_is_resolution_independent() -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _digits_module()
    true_latent = module._inverse_digits_latent_from_secret(
        b"resolution independence seed",
        sample_index=0,
    )
    recovered = module.InverseDigitsLatent(
        true_latent.identity,
        true_latent.x_translation + 0.02,
        true_latent.y_translation,
        true_latent.scale,
        true_latent.shear,
        true_latent.stroke_width,
        deformation=true_latent.deformation,
    )

    def certify(side: int) -> Any:
        observation = module.render_inverse_digits(
            runtime=runtime, latents=(true_latent,), canvas_side=side
        )
        return module.inverse_digits_static_certification(
            runtime=runtime,
            recovered_latent=recovered,
            observation=observation,
            canvas_side=side,
            refinement_sides=(side, side * 2),
        )

    coarse = certify(32)
    fine = certify(64)

    # Refining the grid must not inflate the observable latent modes (ambient, not chart),
    # and the certified nuisance bits stay stable rather than growing with resolution.
    assert coarse.certification_status == "certified"
    assert abs(coarse.observable_mode_count - fine.observable_mode_count) <= 3
    assert math.isclose(coarse.nuisance_bits, fine.nuisance_bits, rel_tol=0.35)


def test_inverse_digits_identity_bits_drop_at_certified_precision_boundary() -> None:
    runtime = resolve_tensor_runtime("cpu")
    module = _digits_module()
    latent = module.InverseDigitsLatent(8, 0.0, 0.0, 1.0, 0.02, 1.0)

    # Identity distinguishability is an image-space epsilon: a fine precision separates
    # the digit identities, a coarse one (above the inter-render distances) cannot.
    resolved = module._distinguishable_identity_count(
        runtime=runtime, recovered_latent=latent, image_epsilon=1.0e-3, canvas_side=32
    )
    unresolved = module._distinguishable_identity_count(
        runtime=runtime, recovered_latent=latent, image_epsilon=10.0, canvas_side=32
    )

    assert resolved > 1
    assert unresolved == 1


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
            certification=certificate,
            canvas_side=32,
        )

    near = bits_for_offset(0.0)
    far = bits_for_offset(0.02)

    assert near.bits > far.bits
    assert near.bits > 0.0
    assert near.identity_bits == math.log2(10)
    assert near.distinguishable_identity_count == 10


def test_inverse_digits_submitted_encoder_trains_label_free_and_earns_bits() -> None:
    runtime = resolve_tensor_runtime("cpu")
    torch = runtime.torch
    from leibniz.program_graphs import load_program_graph

    benchmark = load_digits_benchmark(_digits_benchmark_root)
    program = load_program_graph(
        _repository_root / "tests/fixtures/programs/digits_inverse_conv_encoder.py",
        runtime,
    )
    encoder = program.graph.nodes[0].operation
    batch = benchmark.generator(seed=23, shape=4, runtime=runtime)
    fields, targets = batch.require_tensors()
    training_benchmark = cast(Any, benchmark)
    loss_fn = training_benchmark.build_training_loss(runtime, benchmark.target_contract)
    competence = training_benchmark.build_training_competence(
        runtime,
        benchmark.target_contract,
    )
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.003)

    initial_loss = float(loss_fn(encoder(fields), targets).detach())
    for _step in range(100):
        optimizer.zero_grad()
        predictions = encoder(fields)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

    predictions = encoder(fields)
    final_loss = float(loss_fn(predictions, targets).detach())

    request = SimpleNamespace(
        runtime=runtime,
        predictions=predictions,
        targets=targets,
    )
    bits = competence(request)

    assert final_loss < initial_loss * 0.1
    assert float(bits.mean()) > 1.0
    assert not any(
        "label" in diagnostic
        for diagnostic in getattr(bits, "leibniz_competence_diagnostics", ())
    )


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
        [_default_inverse_nuisance(module) for _label in labels],
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


def _default_inverse_nuisance(module: Any) -> list[float]:
    return [0.0, 0.0, 1.0, 0.0, 1.0] + [0.0] * cast(
        int,
        module._inverse_deformation_dimension,
    )


def _certified_bits_for_latent(
    *,
    module: Any,
    runtime: Any,
    recovered_latent: Any,
    observation: Any,
    canvas_side: int,
) -> Any:
    certificate = module.inverse_digits_static_certification(
        runtime=runtime,
        recovered_latent=recovered_latent,
        observation=observation,
        canvas_side=canvas_side,
        refinement_sides=(canvas_side, canvas_side * 2),
    )
    return module.inverse_digits_validated_bits(
        runtime=runtime,
        recovered_latent=recovered_latent,
        certification=certificate,
        canvas_side=canvas_side,
    )


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
