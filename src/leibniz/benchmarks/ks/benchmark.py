"""Kuramoto-Sivashinsky field benchmark implementation entry point."""

from __future__ import annotations

import cmath
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from leibniz.benchmark_implementations import RawBenchmark as BenchmarkProtocol
from leibniz.benchmarks import BenchmarkManifest
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.observation_generation import (
    GeneratedSample,
    GeneratedSampleSet,
    GenerationRequestOutcome,
    ObservationGenerationError,
    StateSpaceVolumeRequest,
    StateSpaceVolumeValue,
)
from leibniz.outcomes import Outcome, OutcomeSpace
from leibniz.state_space import (
    AccessibleSubspace,
    ContinuousAxisRegion,
    Distinguishability,
    MeasureEstimate,
    ProductRegion,
    RealIntervalDomain,
    SamplingProtocol,
    StateSpaceAmbient,
    StateSpaceAxis,
    StateSpaceRegion,
)
from leibniz.target_contracts import (
    BaselinePredictor,
    CompetenceFunctional,
    TargetContract,
)
from leibniz.tensor_runtime import (
    TensorBatchProgram,
    TensorElementParameter,
    TensorElementRecipe,
    TensorRuntime,
    TensorSolverProgram,
    resolve_host_tensor_runtime,
    tensor_runtime_concat,
    tensor_runtime_construct_tensor,
    tensor_runtime_solve_tensor_trajectory,
)
from leibniz.timing import TimingCollector

__all__ = ["benchmark"]

_benchmark_id = ProtocolIdentifier.parse("benchmarks.ks@0.1.0")
_generator_id = ProtocolIdentifier.parse("benchmarks.ks.generator@0.1.0")
_placeholder_outcome_space_id = ProtocolIdentifier.parse(
    "benchmarks.ks.placeholder-outcomes@0.1.0"
)
_residual_operator_id = "benchmarks.ks.residual-operator@0.1.0"
_default_space_count = 32
_time_count = 9
_box_length = 22.0
_horizon = 1.0
_state_discriminability_resolution = 0.05
_convergence_refinement_factor = 2
_convergence_rung_count = 3
_convergence_gate_uncertainty_scale = 1.0
_convergence_expected_observed_order = 2.0
_convergence_observed_order_tolerance = 0.5
_initial_condition_mode_count = 4
_maximum_window = 8
_window_axis = StateSpaceAxis(
    id="ks-space-time-log2-window",
    domain=RealIntervalDomain(lower=0.0, upper=float(_maximum_window + 1)),
)


@dataclass(frozen=True, slots=True)
class RichardsonEstimate:
    """Scalar Richardson extrapolation estimate from a refinement sequence."""

    observed_order: float
    limit: float
    uncertainty: float


@dataclass(frozen=True, slots=True)
class RichardsonFieldEstimate:
    """Field Richardson extrapolation estimate on the common restricted grid."""

    observed_order: float
    extrapolated_field: Any
    error: float


def benchmark(root: Path) -> BenchmarkProtocol:
    """Return the Kuramoto-Sivashinsky benchmark implementation."""

    return Benchmark(root=root)


class Benchmark:
    """Executable Kuramoto-Sivashinsky benchmark declaration."""

    def __init__(self, *, root: Path) -> None:
        self._root = root
        self._manifest = _manifest()
        self._target_contract = _target_contract()
        self._generator = Generator(manifest=self._manifest)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest(self) -> BenchmarkManifest:
        return self._manifest

    @property
    def generator(self) -> Generator:
        return self._generator

    @property
    def sampling_protocol(self) -> SamplingProtocol:
        return SamplingProtocol(
            kind="uniform-monte-carlo",
            estimator_id="sample-mean",
            confidence_method_id="wilson",
        )

    @property
    def accessible_subspace(self) -> AccessibleSubspace:
        return AccessibleSubspace(
            ladder_id="ks-space-time-covering",
            per_configuration_capacity=_ks_capacity_region(),
            frontier_rationale=(
                "Chaotic Kuramoto-Sivashinsky trajectories over the declared periodic "
                "space-time box are in scope; score grows by extending the declared "
                "space-time covering ladder within this bounded conformance capacity."
            ),
        )

    @property
    def target_contract(self) -> TargetContract:
        return self._target_contract

    def build_training_loss(
        self,
        runtime: TensorRuntime,
        target_contract: TargetContract,
    ) -> Any:
        del runtime
        if (
            target_contract.kind != "field-valued"
            or target_contract.loss_id != "equation-residual"
            or target_contract.competence.parameters.get("residual_operator_id")
            != _residual_operator_id
        ):
            raise ValueError("KS benchmark only builds its declared residual loss")
        return _ks_residual_loss

    def build_training_competence(
        self,
        runtime: TensorRuntime,
        target_contract: TargetContract,
    ) -> Callable[[Any], Any]:
        del runtime
        if (
            target_contract.kind != "field-valued"
            or target_contract.competence.kind != "convergence-resolved-bits"
            or target_contract.competence.parameters.get("residual_operator_id")
            != _residual_operator_id
        ):
            raise ValueError("KS benchmark only builds its declared convergence competence")

        def competence(request: Any) -> Any:
            return _ks_convergence_resolved_bits(request)

        return competence

@dataclass(frozen=True, slots=True)
class Generator:
    """Generate KS initial-condition fields and reference trajectory targets."""

    manifest: BenchmarkManifest

    @property
    def id(self) -> ProtocolIdentifier:
        return _generator_id

    @property
    def version(self) -> str:
        return "0.1.0"

    def minimum_log2_volume(self) -> StateSpaceVolumeValue:
        return StateSpaceVolumeValue(value=0.0)

    def __call__(
        self,
        *,
        seed: int,
        shape: int | Sequence[int] | None = None,
        include_fields: bool = False,
        include_metadata: bool = True,
        include_artifacts: bool = False,
        volume_request: StateSpaceVolumeRequest | None = None,
        sample_indices: Sequence[int] | None = None,
        spatial_points: int | None = None,
        memory_limit_bytes: int | None = None,
        runtime: TensorRuntime | None = None,
        outcome_ids: tuple[str, ...] | None = None,
        variation_extent: float = 1.0,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedSampleSet:
        del (
            include_fields,
            include_artifacts,
            memory_limit_bytes,
            outcome_ids,
            variation_extent,
            timing,
            timing_prefix,
        )
        sample_shape = _sample_shape(shape)
        sample_count = _sample_count(sample_shape)
        resolved_sample_indices = _sample_indices(
            sample_count=sample_count,
            sample_indices=sample_indices,
        )
        window = _requested_window(volume_request)
        resolved_spatial_points = (
            _spatial_points(spatial_points)
            if spatial_points is not None
            else _spatial_points_for_window(window)
        )
        if window is None:
            return GeneratedSampleSet(
                benchmark_id=self.manifest.id,
                generator_id=self.id,
                generator_version=self.version,
                seed=seed,
                shape=(0,),
                volume_request=volume_request,
                request_outcome=GenerationRequestOutcome(
                    kind="unrepresentable-below-minimum",
                    minimum_region=_ks_region(
                        window=0,
                        spatial_points=resolved_spatial_points,
                    ),
                ),
            )
        region = _ks_region(window=window, spatial_points=resolved_spatial_points)
        tensor_runtime = runtime if runtime is not None else resolve_host_tensor_runtime()
        fields, targets = _ks_tensors(
            runtime=tensor_runtime,
            sample_count=sample_count,
            seed=seed,
            sample_indices=resolved_sample_indices,
            window=window,
            spatial_points=resolved_spatial_points,
        )
        samples = (
            _ks_samples(
                seed=seed,
                sample_indices=resolved_sample_indices,
                window=window,
                spatial_points=resolved_spatial_points,
            )
            if include_metadata
            else ()
        )
        return GeneratedSampleSet(
            benchmark_id=self.manifest.id,
            generator_id=self.id,
            generator_version=self.version,
            seed=seed,
            shape=sample_shape,
            samples=samples,
            fields=fields,
            targets=targets,
            volume_request=volume_request,
            region=region,
        )


def _manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        id=_benchmark_id,
        name=ProtocolName.parse("benchmarks.ks"),
        outcome_space=OutcomeSpace(
            id=_placeholder_outcome_space_id,
            outcomes=(Outcome(id="field"),),
        ),
        resolution_analysis={
            "kind": "component-discriminability-margin",
            "display_name": "Kuramoto-Sivashinsky",
            "discriminability_margin": _state_discriminability_resolution,
            "equation": "u_t = -u*u_x - u_xx - u_xxxx",
            "field_domain_kind": "box-2d",
            "space_boundary": "periodic",
            "time_boundary": "initial-value",
            "volume_value": {
                "kind": "estimated-space-time-covering-window",
                "measure_id": "log2-state-space-volume",
                "method_id": "ks-space-time-entropy-bracket-v1",
            },
        },
    )


def _target_contract() -> TargetContract:
    return TargetContract(
        kind="field-valued",
        outcome_ids=None,
        loss_id="equation-residual",
        competence=CompetenceFunctional(
            kind="convergence-resolved-bits",
            parameters={"residual_operator_id": _residual_operator_id},
        ),
        baseline=BaselinePredictor(kind="persistence"),
    )


def _ks_ambient() -> StateSpaceAmbient:
    return StateSpaceAmbient(
        field_domain_kind="box-2d",
        field_domain={
            "length_x": _box_length,
            "length_y": _horizon,
            "boundary_id": "periodic-space-initial-time",
            "space_axis": "x",
            "time_axis": "t",
            "time_resolution": _horizon / (_time_count - 1),
        },
        field_codomain_id="scalar-field",
        distinguishability=Distinguishability(
            kind="metric-resolution",
            metric_id="anisotropic-space-time-l2",
            resolution=_state_discriminability_resolution,
            certificate_id=_residual_operator_id,
        ),
    )


def _ks_region(
    *,
    window: int,
    spatial_points: int = _default_space_count,
) -> StateSpaceRegion:
    volume = 2**window
    estimate = _ks_measure_estimate(window=window)
    component = ProductRegion(
        axis_regions=(
            ContinuousAxisRegion(
                axis=_window_axis,
                coordinate_region=(float(window), float(window + 1)),
                measure_estimate=estimate,
            ),
        ),
        measure_rule="benchmark-computed-finite-count",
        volume=volume,
        log2_volume=float(window),
        measure_estimate=estimate,
    )
    return StateSpaceRegion(
        id=f"benchmarks.ks.realized-window-{window}",
        ambient=_ks_ambient(),
        components=(component,),
        union_rule="disjoint-union",
        volume=volume,
        log2_volume=float(window),
        measure_estimate=estimate,
    )


def _ks_capacity_region() -> StateSpaceRegion:
    volume = (2 ** (_maximum_window + 1)) - 1
    log2_volume = math.log2(volume)
    estimate = MeasureEstimate(
        kind="estimated",
        method_id="ks-space-time-entropy-bracket-v1",
        log2_lower=0.0,
        log2_upper=float(_maximum_window + 1),
    )
    component = ProductRegion(
        axis_regions=(
            ContinuousAxisRegion(
                axis=_window_axis,
                coordinate_region=(0.0, float(_maximum_window + 1)),
                measure_estimate=estimate,
            ),
        ),
        measure_rule="benchmark-computed-finite-count",
        volume=volume,
        log2_volume=log2_volume,
        measure_estimate=estimate,
    )
    return StateSpaceRegion(
        id="benchmarks.ks.accessible-capacity",
        ambient=_ks_ambient(),
        components=(component,),
        union_rule="disjoint-union",
        volume=volume,
        log2_volume=log2_volume,
        measure_estimate=estimate,
    )


def _ks_measure_estimate(*, window: int) -> MeasureEstimate:
    return MeasureEstimate(
        kind="estimated",
        method_id="ks-space-time-entropy-bracket-v1",
        log2_lower=float(window),
        log2_upper=float(window),
    )


def _ks_tensors(
    *,
    runtime: TensorRuntime,
    sample_count: int,
    seed: int,
    sample_indices: tuple[int, ...],
    window: int,
    spatial_points: int = _default_space_count,
) -> tuple[Any, Any]:
    initial = _ks_initial_fields(
        runtime=runtime,
        sample_count=sample_count,
        seed=seed,
        sample_indices=sample_indices,
        window=window,
        spatial_points=spatial_points,
    )
    fields = initial.float()
    targets = initial[:, 0:1, :].float()
    return fields, targets


def _ks_reference_trajectory(  # pyright: ignore[reportUnusedFunction]
    *,
    runtime: TensorRuntime,
    sample_count: int,
    seed: int,
    sample_indices: tuple[int, ...],
    window: int,
    spatial_points: int = _default_space_count,
    horizon: float = _horizon,
    time_count: int = _time_count,
) -> Any:
    if time_count < 2:
        raise ValueError("KS reference trajectory requires at least two time samples")
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("KS reference trajectory horizon must be positive and finite")
    time_step = horizon / float(time_count - 1)
    return tensor_runtime_solve_tensor_trajectory(
        runtime,
        program=TensorSolverProgram(
            initial_state=_ks_initial_program(
                seed=seed,
                sample_indices=sample_indices,
                window=window,
                spatial_points=spatial_points,
            ),
            step_kernel=_ks_step_kernel,
            step_count=time_count - 1,
            parameters=_ks_solver_parameters(
                spatial_points=spatial_points,
                time_step=time_step,
            ),
            dtype="float64",
            cache_key=(
                "ks-reference-etdrk4-step",
                spatial_points,
                time_count,
                time_step,
            ),
        ),
        shape=(sample_count, 1, spatial_points),
    )


def _ks_initial_fields(
    *,
    runtime: TensorRuntime,
    sample_count: int,
    seed: int,
    sample_indices: tuple[int, ...],
    window: int,
    spatial_points: int = _default_space_count,
) -> Any:
    initial_program = _ks_initial_program(
        seed=seed,
        sample_indices=sample_indices,
        window=window,
        spatial_points=spatial_points,
    )
    return tensor_runtime_construct_tensor(
        runtime,
        recipe=TensorElementRecipe(
            shape=(sample_count, 1, spatial_points),
            dtype="float64",
            program=initial_program,
        ),
    )


def _ks_initial_program(
    *,
    seed: int,
    sample_indices: tuple[int, ...],
    window: int,
    spatial_points: int = _default_space_count,
) -> TensorBatchProgram:
    return TensorBatchProgram(
        kernel=_ks_initial_condition_kernel,
        parameters=_ks_initial_parameters(
            seed=seed,
            sample_indices=sample_indices,
            window=window,
            spatial_points=spatial_points,
        ),
        cache_key=("ks-initial-condition", spatial_points),
    )


def _ks_initial_parameters(
    *,
    seed: int,
    sample_indices: tuple[int, ...],
    window: int,
    spatial_points: int,
) -> dict[str, TensorElementParameter]:
    mode_numbers = tuple(float(mode) for mode in range(1, _initial_condition_mode_count + 1))
    mode_decay = tuple(1.0 / (mode * mode) for mode in mode_numbers)
    return {
        "sample_indices": TensorElementParameter(
            dtype="int64",
            shape=(len(sample_indices),),
            values=sample_indices,
            dynamic_axes=(0,),
        ),
        "seed_value": TensorElementParameter(dtype="float64", shape=(), values=(float(seed),)),
        "amplitude": TensorElementParameter(
            dtype="float64",
            shape=(),
            values=(0.15,),
        ),
        "spatial_points": TensorElementParameter(
            dtype="float64",
            shape=(),
            values=(float(spatial_points),),
        ),
        "mode_numbers": TensorElementParameter(
            dtype="float64",
            shape=(_initial_condition_mode_count,),
            values=mode_numbers,
        ),
        "mode_decay": TensorElementParameter(
            dtype="float64",
            shape=(_initial_condition_mode_count,),
            values=mode_decay,
        ),
    }


def _ks_solver_parameters(
    *,
    spatial_points: int = _default_space_count,
    time_step: float | None = None,
) -> dict[str, TensorElementParameter]:
    frequencies = tuple(
        index if index <= spatial_points // 2 else index - spatial_points
        for index in range(spatial_points)
    )
    wave_numbers = tuple(2.0 * math.pi * frequency / _box_length for frequency in frequencies)
    dt = _horizon / (_time_count - 1) if time_step is None else time_step
    linear_values = tuple(
        (wave_number * wave_number) - (wave_number**4)
        for wave_number in wave_numbers
    )
    linear_factors = tuple(complex(math.exp(dt * value), 0.0) for value in linear_values)
    half_linear_factors = tuple(
        complex(math.exp(0.5 * dt * value), 0.0) for value in linear_values
    )
    coefficient_rows = tuple(
        _etdrk4_coefficients(value, dt=dt)
        for value in linear_values
    )
    q_coefficients = tuple(row[0] for row in coefficient_rows)
    f1_coefficients = tuple(row[1] for row in coefficient_rows)
    f2_coefficients = tuple(row[2] for row in coefficient_rows)
    f3_coefficients = tuple(row[3] for row in coefficient_rows)
    derivative_coefficients = tuple(complex(0.0, wave_number) for wave_number in wave_numbers)
    dealias_mask = tuple(
        1.0 if abs(frequency) <= spatial_points // 3 else 0.0
        for frequency in frequencies
    )
    return {
        "linear_factors": TensorElementParameter(
            dtype="complex128",
            shape=(spatial_points,),
            values=linear_factors,
        ),
        "half_linear_factors": TensorElementParameter(
            dtype="complex128",
            shape=(spatial_points,),
            values=half_linear_factors,
        ),
        "q_coefficients": TensorElementParameter(
            dtype="complex128",
            shape=(spatial_points,),
            values=q_coefficients,
        ),
        "f1_coefficients": TensorElementParameter(
            dtype="complex128",
            shape=(spatial_points,),
            values=f1_coefficients,
        ),
        "f2_coefficients": TensorElementParameter(
            dtype="complex128",
            shape=(spatial_points,),
            values=f2_coefficients,
        ),
        "f3_coefficients": TensorElementParameter(
            dtype="complex128",
            shape=(spatial_points,),
            values=f3_coefficients,
        ),
        "derivative_coefficients": TensorElementParameter(
            dtype="complex128",
            shape=(spatial_points,),
            values=derivative_coefficients,
        ),
        "dealias_mask": TensorElementParameter(
            dtype="float64",
            shape=(spatial_points,),
            values=dealias_mask,
        ),
    }


def _etdrk4_coefficients(
    linear_value: float,
    *,
    dt: float,
) -> tuple[complex, complex, complex, complex]:
    roots = tuple(
        complex(
            math.cos(math.pi * ((index + 0.5) / 16.0)),
            math.sin(math.pi * ((index + 0.5) / 16.0)),
        )
        for index in range(16)
    )
    lr_values = tuple(dt * linear_value + root for root in roots)
    q = dt * sum((cmath.exp(lr / 2.0) - 1.0) / lr for lr in lr_values) / len(lr_values)
    f1 = (
        dt
        * sum(
            (-4.0 - lr + cmath.exp(lr) * (4.0 - (3.0 * lr) + (lr * lr)))
            / (lr * lr * lr)
            for lr in lr_values
        )
        / len(lr_values)
    )
    f2 = (
        dt
        * sum(
            (2.0 + lr + cmath.exp(lr) * (-2.0 + lr)) / (lr * lr * lr)
            for lr in lr_values
        )
        / len(lr_values)
    )
    f3 = (
        dt
        * sum(
            (-4.0 - (3.0 * lr) - (lr * lr) + cmath.exp(lr) * (4.0 - lr))
            / (lr * lr * lr)
            for lr in lr_values
        )
        / len(lr_values)
    )
    return q, f1, f2, f3


def _ks_initial_condition_kernel(
    coordinates: tuple[Any, ...],
    *,
    sample_indices: Any,
    seed_value: Any,
    amplitude: Any,
    spatial_points: Any,
    mode_numbers: Any,
    mode_decay: Any,
) -> Any:
    sample, channel, x = coordinates
    _ = channel
    sample_key = sample_indices[sample].reshape((-1, 1, 1))
    modes = mode_numbers.reshape((1, -1, 1))
    decay = mode_decay.reshape((1, -1, 1))
    spatial = x.reshape((1, 1, -1)) * (2.0 * math.pi / spatial_points)
    random_key = seed_value.reshape((1, 1, 1)) + (0.173 * sample_key)
    sine_coefficients = (
        (random_key * 12.9898 + modes * 78.233 + 0.37).sin()
        * decay
    )
    cosine_coefficients = (
        (random_key * 4.1414 + modes * 31.416 + 1.91).sin()
        * decay
    )
    energy = (
        (sine_coefficients * sine_coefficients)
        + (cosine_coefficients * cosine_coefficients)
    ).sum(dim=1, keepdim=True).sqrt().clamp_min(math.ulp(1.0))
    field = (
        (sine_coefficients * (modes * spatial).sin())
        + (cosine_coefficients * (modes * spatial).cos())
    ).sum(dim=1, keepdim=True)
    return amplitude * field / energy


def _ks_step_kernel(
    state: Any,
    ops: Any,
    *,
    linear_factors: Any,
    half_linear_factors: Any,
    q_coefficients: Any,
    f1_coefficients: Any,
    f2_coefficients: Any,
    f3_coefficients: Any,
    derivative_coefficients: Any,
    dealias_mask: Any,
) -> Any:
    spectrum = ops.fft(state, axis=-1)
    nonlinear_spectrum = _ks_nonlinear_spectrum(
        spectrum,
        ops,
        derivative_coefficients=derivative_coefficients,
        dealias_mask=dealias_mask,
    )
    half_linear = half_linear_factors.reshape((1, 1, -1))
    q = q_coefficients.reshape((1, 1, -1))
    a_spectrum = (half_linear * spectrum) + (q * nonlinear_spectrum)
    a_nonlinear = _ks_nonlinear_spectrum(
        a_spectrum,
        ops,
        derivative_coefficients=derivative_coefficients,
        dealias_mask=dealias_mask,
    )
    b_spectrum = (half_linear * spectrum) + (q * a_nonlinear)
    b_nonlinear = _ks_nonlinear_spectrum(
        b_spectrum,
        ops,
        derivative_coefficients=derivative_coefficients,
        dealias_mask=dealias_mask,
    )
    c_spectrum = (half_linear * a_spectrum) + (
        q * ((2.0 * b_nonlinear) - nonlinear_spectrum)
    )
    c_nonlinear = _ks_nonlinear_spectrum(
        c_spectrum,
        ops,
        derivative_coefficients=derivative_coefficients,
        dealias_mask=dealias_mask,
    )
    next_spectrum = (
        linear_factors.reshape((1, 1, -1)) * spectrum
        + f1_coefficients.reshape((1, 1, -1)) * nonlinear_spectrum
        + 2.0
        * f2_coefficients.reshape((1, 1, -1))
        * (a_nonlinear + b_nonlinear)
        + f3_coefficients.reshape((1, 1, -1)) * c_nonlinear
    )
    return ops.real(ops.ifft(next_spectrum, axis=-1))


def _ks_nonlinear_spectrum(
    spectrum: Any,
    ops: Any,
    *,
    derivative_coefficients: Any,
    dealias_mask: Any,
) -> Any:
    state = ops.real(ops.ifft(spectrum, axis=-1))
    gradient = ops.real(
        ops.ifft(spectrum * derivative_coefficients.reshape((1, 1, -1)), axis=-1)
    )
    nonlinear_spectrum = ops.fft(-state * gradient, axis=-1)
    return nonlinear_spectrum * dealias_mask.reshape((1, 1, -1))


def _ks_residual_loss(predictions: Any, targets: Any) -> Any:
    _validate_initial_condition_target(
        predictions=predictions,
        targets=targets,
        context="KS residual loss",
    )
    if int(predictions.shape[1]) < 2:
        raise ValueError("KS residual loss requires a predicted trajectory")
    residual = _ks_discrete_residual(predictions)
    residual_loss = (residual * residual).mean()
    initial_error = predictions[:, 0:1, :] - targets[:, 0:1, :]
    initial_loss = (initial_error * initial_error).mean()
    return residual_loss + initial_loss


def _ks_convergence_resolved_bits(request: Any) -> Any:
    predictions = request.predictions
    targets = request.targets
    _validate_initial_condition_target(
        predictions=predictions,
        targets=targets,
        context="KS convergence competence",
    )
    try:
        ladder = _ks_prediction_ladder(request)
    except ValueError:
        return _zero_bits(runtime=request.runtime, predictions=predictions)
    return _ks_ladder_convergence_bits(
        runtime=request.runtime,
        ladder=ladder,
        horizon=_request_horizon(request),
    )


def _validate_initial_condition_target(
    *,
    predictions: Any,
    targets: Any,
    context: str,
) -> None:
    prediction_shape = tuple(predictions.shape)
    target_shape = tuple(targets.shape)
    if len(prediction_shape) != 3 or len(target_shape) != 3:
        raise ValueError(f"{context} requires rank-3 prediction and target tensors")
    if int(target_shape[1]) < 1:
        raise ValueError(f"{context} requires an initial-condition target")
    if prediction_shape[0] != target_shape[0] or prediction_shape[-1] != target_shape[-1]:
        raise ValueError(f"{context} requires matching batch and spatial dimensions")


def _ks_prediction_ladder(request: Any) -> tuple[Any, ...]:
    if request.module is None or request.generator is None or request.batch is None:
        return (request.predictions,)
    sample_indices, volume_request = _request_sample_identity(request)
    base_space_count = int(request.predictions.shape[-1])
    base_time_count = int(request.predictions.shape[1])
    ladder: list[Any] = []
    for rung in range(_convergence_rung_count):
        factor = _convergence_refinement_factor**rung
        spatial_points = base_space_count * factor
        time_count = 1 + ((base_time_count - 1) * factor)
        generated = request.generator(
            seed=request.batch.seed,
            shape=len(sample_indices),
            sample_indices=sample_indices,
            spatial_points=spatial_points,
            volume_request=volume_request,
            runtime=request.runtime,
            include_metadata=False,
        )
        fields, _targets = generated.require_tensors()
        ladder.append(
            _query_operator_trajectory(
                runtime=request.runtime,
                module=request.module,
                fields=fields,
                horizon=_request_horizon(request),
                time_count=time_count,
            )
        )
    return tuple(ladder)


def _request_sample_identity(request: Any) -> tuple[tuple[int, ...], StateSpaceVolumeRequest]:
    if request.batch is None:
        raise ValueError("KS convergence competence requires a generated sample batch")
    sample_indices: list[int] = []
    windows: set[int] = set()
    for sample_key in request.sample_keys:
        latent_coordinates = sample_key.get("latent_coordinates")
        if not isinstance(latent_coordinates, Sequence) or not latent_coordinates:
            raise ValueError("KS convergence competence requires sample latent coordinates")
        coordinate_value = cast(Sequence[object], latent_coordinates)[0]
        if not isinstance(coordinate_value, Mapping):
            raise ValueError("KS convergence competence requires sample coordinate records")
        coordinate = cast(Mapping[str, object], coordinate_value)
        sample_index = coordinate.get("sample_index")
        window = coordinate.get("window")
        if not isinstance(sample_index, int) or not isinstance(window, int):
            raise ValueError("KS convergence competence requires sample index and window")
        sample_indices.append(sample_index)
        windows.add(window)
    if len(windows) != 1:
        raise ValueError("KS convergence competence requires one volume window per batch")
    window = next(iter(windows))
    return (
        tuple(sample_indices),
        StateSpaceVolumeRequest(minimum=float(window), maximum=float(window + 1)),
    )


def _query_operator_trajectory(
    *,
    runtime: TensorRuntime,
    module: Any,
    fields: Any,
    horizon: float,
    time_count: int,
) -> Any:
    if time_count < 2:
        raise ValueError("KS convergence trajectory requires at least two time samples")
    states = [fields]
    step = horizon / float(time_count - 1)
    for index in range(1, time_count):
        state = module(fields, step * index)
        if tuple(state.shape) != tuple(fields.shape):
            raise ValueError("KS convergence operator changed state shape")
        states.append(state)
    return tensor_runtime_concat(runtime, states, dim=1)


def _request_horizon(request: Any) -> float:
    horizon = float(request.horizons[-1]) if request.horizons else _horizon
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("KS convergence horizon must be positive and finite")
    return horizon


def _ks_ladder_convergence_bits(
    *,
    runtime: TensorRuntime,
    ladder: tuple[Any, ...],
    horizon: float,
) -> Any:
    if len(ladder) < _convergence_rung_count:
        return _zero_bits(runtime=runtime, predictions=ladder[-1])
    measured_ladder = tuple(_float64_tensor(trajectory) for trajectory in ladder)
    base_time_count = int(measured_ladder[0].shape[1])
    if base_time_count < 2:
        return _zero_bits(runtime=runtime, predictions=measured_ladder[-1])
    factor = _convergence_refinement_factor
    boundary_bits = [0.0 for _sample in range(int(ladder[-1].shape[0]))]
    prefix_open = [True for _sample in boundary_bits]
    boundaries = [0.0 for _sample in boundary_bits]
    time_points: list[list[Mapping[str, object]]] = [[] for _sample in boundary_bits]
    latest_records: list[Mapping[str, object] | None] = [None for _sample in boundary_bits]
    boundary_records: list[Mapping[str, object] | None] = [None for _sample in boundary_bits]
    for base_time_index in range(1, base_time_count):
        time_value = horizon * float(base_time_index) / float(base_time_count - 1)
        prefix_ladder = tuple(
            trajectory[:, : (base_time_index * (factor**rung_index)) + 1, :]
            for rung_index, trajectory in enumerate(measured_ladder)
        )
        point_values, point_records = _ks_ladder_prefix_convergence_bits(
            runtime=runtime,
            ladder=prefix_ladder,
            horizon=time_value,
        )
        for sample_index, point_record in enumerate(point_records):
            latest_records[sample_index] = point_record
            point_bits = float(point_values[sample_index])
            gated = point_record.get("gate_decision") == "passed" and point_bits > 0.0
            time_points[sample_index].append(
                {
                    "time": time_value,
                    "bits": point_bits if prefix_open[sample_index] and gated else 0.0,
                    "gate_decision": point_record.get("gate_decision"),
                }
            )
            if not prefix_open[sample_index]:
                continue
            if gated:
                boundary_bits[sample_index] = point_bits
                boundaries[sample_index] = time_value
                boundary_records[sample_index] = point_record
            else:
                prefix_open[sample_index] = False
    diagnostics: list[Mapping[str, object]] = []
    for sample_index, total in enumerate(boundary_bits):
        latest = dict(boundary_records[sample_index] or latest_records[sample_index] or {})
        latest["kind"] = "ks-convergence-diagnostics"
        latest["sample_index"] = sample_index
        latest["predictability_boundary"] = boundaries[sample_index]
        latest["time_points"] = [dict(point) for point in time_points[sample_index]]
        latest["gate_decision"] = "passed" if total > 0.0 else "failed"
        latest["bits"] = total
        diagnostics.append(latest)
    result = measured_ladder[-1].new_tensor(boundary_bits)
    result.leibniz_competence_diagnostics = tuple(diagnostics)
    return result


def _float64_tensor(tensor: Any) -> Any:
    convert = getattr(tensor, "double", None)
    return convert() if callable(convert) else tensor


def _ks_ladder_prefix_convergence_bits(
    *,
    runtime: TensorRuntime,
    ladder: tuple[Any, ...],
    horizon: float,
) -> tuple[Any, tuple[Mapping[str, object], ...]]:
    factor = _convergence_refinement_factor
    residual_norms = tuple(
        _per_sample_residual_norm(trajectory, horizon=horizon) for trajectory in ladder
    )
    restricted = tuple(
        restrict_to_common_grid(trajectory, rung_index=index, factor=factor)
        for index, trajectory in enumerate(ladder)
    )
    field_estimate = _per_sample_richardson_field(
        runtime=runtime,
        restricted=restricted,
        factor=factor,
    )
    finest = ladder[-1]
    evolution = finest - finest[:, 0:1, :]
    signal = evolution.std(dim=(1, 2), unbiased=False)
    finest_nodes = int(finest.shape[1]) * int(finest.shape[2])
    values: list[float] = []
    diagnostics: list[Mapping[str, object]] = []
    for sample_index in range(int(finest.shape[0])):
        residual_values = tuple(float(norm[sample_index]) for norm in residual_norms)
        try:
            residual_estimate = richardson(
                residual_values,
                factor=float(factor),
            )
        except ValueError:
            values.append(0.0)
            diagnostics.append(
                {
                    "kind": "ks-convergence-diagnostics",
                    "sample_index": sample_index,
                    "residual_norms": list(residual_values),
                    "gate_decision": "failed-richardson",
                    "bits": 0.0,
                }
            )
            continue
        sigma = float(signal[sample_index])
        error = float(field_estimate.error[sample_index])
        sample_bits = _resolved_bits(signal=sigma, error=error, node_count=finest_nodes)
        converged = _ks_convergence_gate(residual_estimate)
        if not converged:
            values.append(0.0)
        else:
            values.append(sample_bits)
        diagnostics.append(
            _ks_convergence_diagnostic_record(
                sample_index=sample_index,
                residual_values=residual_values,
                residual_estimate=residual_estimate,
                field_error=error,
                signal=sigma,
                node_count=finest_nodes,
                bits=values[-1],
            )
        )
    return finest.new_tensor(values), tuple(diagnostics)


def _ks_convergence_diagnostic_record(
    *,
    sample_index: int,
    residual_values: tuple[float, ...],
    residual_estimate: RichardsonEstimate,
    field_error: float,
    signal: float,
    node_count: int,
    bits: float,
) -> Mapping[str, object]:
    return {
        "kind": "ks-convergence-diagnostics",
        "sample_index": sample_index,
        "residual_norms": list(residual_values),
        "residual_observed_order": residual_estimate.observed_order,
        "residual_extrapolated_limit": residual_estimate.limit,
        "residual_extrapolation_uncertainty": residual_estimate.uncertainty,
        "expected_observed_order": _convergence_expected_observed_order,
        "observed_order_tolerance": _convergence_observed_order_tolerance,
        "field_error": field_error,
        "evolution_scale": signal,
        "node_count": node_count,
        "rung_count": _convergence_rung_count,
        "gate_decision": "passed" if bits > 0.0 else "failed",
        "bits": bits,
        "k_sensitivity": [
            _ks_k_sensitivity_record(
                residual_estimate=residual_estimate,
                k_value=k_value,
                signal=signal,
                field_error=field_error,
                node_count=node_count,
            )
            for k_value in (0.5, 1.0, 2.0)
        ],
    }


def _ks_k_sensitivity_record(
    *,
    residual_estimate: RichardsonEstimate,
    k_value: float,
    signal: float,
    field_error: float,
    node_count: int,
) -> Mapping[str, object]:
    gated = _ks_convergence_gate(residual_estimate, k_value=k_value)
    bits = (
        _resolved_bits(signal=signal, error=field_error, node_count=node_count)
        if gated
        else 0.0
    )
    return {"k": k_value, "gated": gated, "bits": bits}


def _ks_convergence_gate(
    residual_estimate: RichardsonEstimate,
    *,
    k_value: float = _convergence_gate_uncertainty_scale,
) -> bool:
    lower_order = _convergence_expected_observed_order - _convergence_observed_order_tolerance
    upper_order = _convergence_expected_observed_order + _convergence_observed_order_tolerance
    return (
        lower_order <= residual_estimate.observed_order <= upper_order
        and abs(residual_estimate.limit) <= k_value * residual_estimate.uncertainty
    )


def _zero_bits(*, runtime: TensorRuntime, predictions: Any) -> Any:
    _ = runtime
    return predictions.new_zeros((int(predictions.shape[0]),))


def _per_sample_residual_norm(trajectory: Any, *, horizon: float) -> Any:
    spatial_points = int(trajectory.shape[-1])
    time_count = int(trajectory.shape[1])
    residual = ks_space_time_residual(
        trajectory,
        dx=_box_length / float(spatial_points),
        dt=horizon / float(time_count - 1),
    )
    return residual.pow(2).mean(dim=(1, 2)).sqrt()


@dataclass(frozen=True, slots=True)
class _PerSampleFieldEstimate:
    observed_order: Any
    error: Any


def _per_sample_richardson_field(
    *,
    runtime: TensorRuntime,
    restricted: tuple[Any, ...],
    factor: int,
) -> _PerSampleFieldEstimate:
    previous, current, finest = restricted[-3:]
    first_gap = (current - previous).pow(2).mean(dim=(1, 2)).sqrt()
    second_gap = (finest - current).pow(2).mean(dim=(1, 2)).sqrt()
    _ = runtime
    numeric_floor = math.ulp(1.0)
    safe_first = first_gap.clamp_min(numeric_floor)
    safe_second = second_gap.clamp_min(numeric_floor)
    observed_order = (safe_first / safe_second).log() / math.log(float(factor))
    denominator = observed_order.mul(math.log(float(factor))).exp().sub(1.0).clamp_min(
        numeric_floor
    )
    correction = (finest - current) / denominator.reshape((-1, 1, 1))
    error = correction.pow(2).mean(dim=(1, 2)).sqrt()
    return _PerSampleFieldEstimate(observed_order=observed_order, error=error)


def _resolved_bits(*, signal: float, error: float, node_count: int) -> float:
    if not math.isfinite(signal) or signal <= 0.0:
        return 0.0
    if not math.isfinite(error):
        return 0.0
    floor = math.ulp(signal) if signal > 0.0 else math.ulp(1.0)
    effective_error = max(error, floor)
    if effective_error >= signal:
        return 0.0
    return float(node_count) * math.log2(signal / effective_error)


def _ks_discrete_residual(predictions: Any) -> Any:
    spatial_points = predictions.shape[-1]
    dx = _box_length / spatial_points
    dt = _horizon / (_time_count - 1)
    u = predictions[:, :-1, :]
    u_t = (predictions[:, 1:, :] - u) / dt
    u_x = (u.roll(shifts=-1, dims=-1) - u.roll(shifts=1, dims=-1)) / (2.0 * dx)
    u_xx = (
        u.roll(shifts=-1, dims=-1)
        - 2.0 * u
        + u.roll(shifts=1, dims=-1)
    ) / (dx * dx)
    u_xxxx = (
        u.roll(shifts=-2, dims=-1)
        - 4.0 * u.roll(shifts=-1, dims=-1)
        + 6.0 * u
        - 4.0 * u.roll(shifts=1, dims=-1)
        + u.roll(shifts=2, dims=-1)
    ) / (dx**4)
    return u_t + (u * u_x) + u_xx + u_xxxx


def ks_space_time_residual(
    trajectory: Any,
    *,
    dx: float,
    dt: float,
) -> Any:
    """Return the central space-time KS residual field for a trajectory."""

    _validate_positive_spacing(dx, field="dx")
    _validate_positive_spacing(dt, field="dt")
    shape = tuple(trajectory.shape)
    if len(shape) != 3:
        raise ValueError("KS residual trajectory must have shape (batch, time, space)")
    if shape[1] < 2:
        raise ValueError("KS residual trajectory must contain at least two time samples")
    if shape[2] < 5:
        raise ValueError("KS residual trajectory must contain at least five space samples")
    u_t = central_time_derivative(trajectory, dt=dt)
    u_x = (trajectory.roll(shifts=-1, dims=-1) - trajectory.roll(shifts=1, dims=-1)) / (
        2.0 * dx
    )
    u_xx = (
        trajectory.roll(shifts=-1, dims=-1)
        - 2.0 * trajectory
        + trajectory.roll(shifts=1, dims=-1)
    ) / (dx * dx)
    u_xxxx = (
        trajectory.roll(shifts=-2, dims=-1)
        - 4.0 * trajectory.roll(shifts=-1, dims=-1)
        + 6.0 * trajectory
        - 4.0 * trajectory.roll(shifts=1, dims=-1)
        + trajectory.roll(shifts=2, dims=-1)
    ) / (dx**4)
    return u_t + (trajectory * u_x) + u_xx + u_xxxx


def grid_l2_norm(field: Any) -> float:
    """Return the root-mean-square grid L2 norm as a host float."""

    return float((field * field).mean().sqrt())


def richardson(sequence: tuple[float, ...], *, factor: float) -> RichardsonEstimate:
    """Extrapolate the final three entries of a scalar refinement sequence."""

    _validate_refinement_factor(factor)
    if len(sequence) < 3:
        raise ValueError("Richardson extrapolation requires at least three values")
    previous, current, finest = (
        float(sequence[-3]),
        float(sequence[-2]),
        float(sequence[-1]),
    )
    first_gap = abs(current - previous)
    second_gap = abs(finest - current)
    if not math.isfinite(first_gap) or not math.isfinite(second_gap):
        raise ValueError("Richardson sequence values must be finite")
    if first_gap <= 0.0 or second_gap <= 0.0:
        raise ValueError("Richardson extrapolation requires two nonzero final gaps")
    observed_order = math.log(first_gap / second_gap, factor)
    denominator = (factor**observed_order) - 1.0
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("Richardson extrapolation observed order must be positive")
    correction = (finest - current) / denominator
    return RichardsonEstimate(
        observed_order=observed_order,
        limit=finest + correction,
        uncertainty=abs(correction),
    )


def richardson_field(
    ladder: tuple[Any, ...],
    *,
    factor: int,
) -> RichardsonFieldEstimate:
    """Extrapolate a nested space-time field ladder onto its common coarse grid."""

    _validate_refinement_factor(float(factor))
    if factor < 2:
        raise ValueError("field Richardson factor must be at least 2")
    if len(ladder) < 3:
        raise ValueError("field Richardson extrapolation requires at least three rungs")
    restricted = tuple(
        restrict_to_common_grid(field, rung_index=index, factor=factor)
        for index, field in enumerate(ladder)
    )
    common_shape = tuple(restricted[0].shape)
    if any(tuple(field.shape) != common_shape for field in restricted):
        raise ValueError("field Richardson ladder rungs do not share a nested grid")
    previous, current, finest = restricted[-3:]
    first_gap = grid_l2_norm(current - previous)
    second_gap = grid_l2_norm(finest - current)
    if first_gap <= 0.0 or second_gap <= 0.0:
        raise ValueError("field Richardson extrapolation requires nonzero final gaps")
    observed_order = math.log(first_gap / second_gap, float(factor))
    denominator = (float(factor) ** observed_order) - 1.0
    if denominator <= 0.0 or not math.isfinite(denominator):
        raise ValueError("field Richardson observed order must be positive")
    correction = (finest - current) / denominator
    return RichardsonFieldEstimate(
        observed_order=observed_order,
        extrapolated_field=finest + correction,
        error=grid_l2_norm(correction),
    )


def central_time_derivative(trajectory: Any, *, dt: float) -> Any:
    derivative = trajectory.clone()
    derivative[:, 0, :] = (trajectory[:, 1, :] - trajectory[:, 0, :]) / dt
    derivative[:, -1, :] = (trajectory[:, -1, :] - trajectory[:, -2, :]) / dt
    if int(trajectory.shape[1]) > 2:
        derivative[:, 1:-1, :] = (trajectory[:, 2:, :] - trajectory[:, :-2, :]) / (
            2.0 * dt
        )
    return derivative


def restrict_to_common_grid(field: Any, *, rung_index: int, factor: int) -> Any:
    stride = factor**rung_index
    return field[..., ::stride, ::stride]


def _validate_positive_spacing(value: float, *, field: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be positive and finite")


def _validate_refinement_factor(value: float) -> None:
    if not math.isfinite(value) or value <= 1.0:
        raise ValueError("Richardson refinement factor must be greater than one")

def _ks_samples(
    *,
    seed: int,
    sample_indices: tuple[int, ...],
    window: int,
    spatial_points: int = _default_space_count,
) -> tuple[GeneratedSample, ...]:
    return tuple(
        GeneratedSample(
            index=index,
            outcome_id="field",
            region_component_index=0,
            axis_coordinates={
                _window_axis.id: float(window) + _unit_interval_coordinate(seed, sample_index)
            },
            latent_coordinates=(
                {
                    "chart": "cartesian-fourier",
                    "sample_index": sample_index,
                    "spatial_points": spatial_points,
                    "window": window,
                },
            ),
        )
        for index, sample_index in enumerate(sample_indices)
    )


def _unit_interval_coordinate(seed: int, sample_index: int) -> float:
    value = ((seed + 1) * 1_103_515_245 + (sample_index + 1) * 12_345) % 1_000_003
    return (value + 0.5) / 1_000_004.0


def _requested_window(request: StateSpaceVolumeRequest | None) -> int | None:
    if request is None:
        return 0
    lower = math.ceil(request.minimum)
    if lower > request.maximum:
        return None
    if lower < 0 or lower > _maximum_window:
        return None
    return lower


def _sample_shape(shape: int | Sequence[int] | None) -> tuple[int, ...]:
    if shape is None:
        return ()
    if isinstance(shape, int):
        if shape < 1:
            raise ObservationGenerationError("sample shape axes must be positive integers")
        return (shape,)
    normalized = tuple(shape)
    if any(type(axis) is not int or axis < 1 for axis in normalized):
        raise ObservationGenerationError("sample shape axes must be positive integers")
    return normalized


def _sample_count(shape: Sequence[int]) -> int:
    if not shape:
        return 1
    count = 1
    for axis in shape:
        count *= axis
    return count


def _spatial_points(value: int | None) -> int:
    if value is None:
        return _default_space_count
    if type(value) is not int or value < _default_space_count:
        raise ObservationGenerationError(
            f"spatial_points must be a power-of-two multiple of {_default_space_count}"
        )
    multiple = value // _default_space_count
    if value % _default_space_count != 0 or multiple & (multiple - 1):
        raise ObservationGenerationError(
            f"spatial_points must be a power-of-two multiple of {_default_space_count}"
        )
    return value


def _spatial_points_for_window(window: int | None) -> int:
    if window is None:
        return _default_space_count
    return _default_space_count * (2**window)


def _sample_indices(
    *,
    sample_count: int,
    sample_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if sample_indices is None:
        return tuple(range(sample_count))
    normalized = tuple(sample_indices)
    if len(normalized) != sample_count:
        raise ObservationGenerationError("sample_indices length must match sample shape")
    if any(type(index) is not int or index < 0 for index in normalized):
        raise ObservationGenerationError("sample_indices must be nonnegative integers")
    return normalized
