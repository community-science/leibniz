"""Kuramoto-Sivashinsky field benchmark implementation entry point."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from leibniz.benchmark_implementations import RawBenchmark as BenchmarkProtocol
from leibniz.benchmarks import BenchmarkManifest
from leibniz.certified_bits import (
    AmbientEntropy,
    CertificationStability,
    evaluate_certified_bits,
)
from leibniz.field_evolution import field_stepper_trajectory
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
    resolve_host_tensor_runtime,
    tensor_runtime_concat,
    tensor_runtime_construct_tensor,
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
_certification_refinement_factors = (1, 2, 4)
_law_amplification_refusal_ratio = 2.0
_law_amplification_log_cap = 16.0
_initial_condition_mode_count = 4
_maximum_window = 8
_window_axis = StateSpaceAxis(
    id="ks-space-time-log2-window",
    domain=RealIntervalDomain(lower=0.0, upper=float(_maximum_window + 1)),
)


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
            or target_contract.competence.kind != "ambient-certified-bits"
            or target_contract.competence.parameters.get("residual_operator_id")
            != _residual_operator_id
        ):
            raise ValueError("KS benchmark only builds its declared certified competence")

        def competence(request: Any) -> Any:
            return _ks_ambient_certified_bits(request)

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
            kind="ambient-certified-bits",
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


def _ks_ambient_certified_bits(request: Any) -> Any:
    predictions = request.predictions
    targets = request.targets
    _validate_initial_condition_target(
        predictions=predictions,
        targets=targets,
        context="KS ambient certified competence",
    )
    try:
        ladder = _ks_prediction_ladder(request)
    except ValueError as error:
        result = _zero_bits(runtime=request.runtime, predictions=predictions)
        result.leibniz_competence_diagnostics = tuple(
            _ks_missing_ladder_diagnostic(sample_index=sample_index, reason=str(error))
            for sample_index in range(int(predictions.shape[0]))
        )
        return result
    return _ks_ladder_ambient_certified_bits(
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


def _ks_missing_ladder_diagnostic(
    *,
    sample_index: int,
    reason: str,
) -> Mapping[str, object]:
    return {
        "kind": "certified-bits-diagnostics",
        "sample_index": sample_index,
        "structural_type": "dynamical-amplification",
        "certification_status": "refused-missing-refinement-ladder",
        "reason": reason,
        "bits": 0.0,
        "predictability_boundary": 0.0,
        "stability": {},
        "ambient_entropy": {},
    }


def _ks_prediction_ladder(request: Any) -> tuple[Any, ...]:
    if request.module is None or request.generator is None or request.batch is None:
        raise ValueError("KS ambient certification requires a refined prediction ladder")
    sample_indices, volume_request = _request_sample_identity(request)
    base_space_count = int(request.predictions.shape[-1])
    base_time_count = int(request.predictions.shape[1])
    ladder: list[Any] = []
    for factor in _certification_refinement_factors:
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
            field_stepper_trajectory(
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
        raise ValueError("KS ambient certification requires a generated sample batch")
    sample_indices: list[int] = []
    windows: set[int] = set()
    for sample_key in request.sample_keys:
        latent_coordinates = sample_key.get("latent_coordinates")
        if not isinstance(latent_coordinates, Sequence) or not latent_coordinates:
            raise ValueError("KS ambient certification requires sample latent coordinates")
        coordinate_value = cast(Sequence[object], latent_coordinates)[0]
        if not isinstance(coordinate_value, Mapping):
            raise ValueError("KS ambient certification requires sample coordinate records")
        coordinate = cast(Mapping[str, object], coordinate_value)
        sample_index = coordinate.get("sample_index")
        window = coordinate.get("window")
        if not isinstance(sample_index, int) or not isinstance(window, int):
            raise ValueError("KS ambient certification requires sample index and window")
        sample_indices.append(sample_index)
        windows.add(window)
    if len(windows) != 1:
        raise ValueError("KS ambient certification requires one volume window per batch")
    window = next(iter(windows))
    return (
        tuple(sample_indices),
        StateSpaceVolumeRequest(minimum=float(window), maximum=float(window + 1)),
    )


def _request_horizon(request: Any) -> float:
    horizon = float(request.horizons[-1]) if request.horizons else _horizon
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("KS ambient certification horizon must be positive and finite")
    return horizon


def _ks_ladder_ambient_certified_bits(
    *,
    runtime: TensorRuntime,
    ladder: tuple[Any, ...],
    horizon: float,
) -> Any:
    measured_ladder = tuple(_float64_tensor(trajectory) for trajectory in ladder)
    if not measured_ladder:
        raise ValueError("KS certified bits require at least one trajectory")
    base_time_count = int(measured_ladder[0].shape[1])
    if base_time_count < 2:
        return _zero_bits(runtime=runtime, predictions=measured_ladder[-1])
    boundary_bits = [0.0 for _sample in range(int(ladder[-1].shape[0]))]
    prefix_open = [True for _sample in boundary_bits]
    boundaries = [0.0 for _sample in boundary_bits]
    time_points: list[list[Mapping[str, object]]] = [[] for _sample in boundary_bits]
    latest_records: list[Mapping[str, object] | None] = [None for _sample in boundary_bits]
    boundary_records: list[Mapping[str, object] | None] = [None for _sample in boundary_bits]
    for base_time_index in range(1, base_time_count):
        time_value = horizon * float(base_time_index) / float(base_time_count - 1)
        prefix_ladder = tuple(
            trajectory[:, : (base_time_index * factor) + 1, :]
            for factor, trajectory in zip(
                _certification_refinement_factors,
                measured_ladder,
                strict=True,
            )
        )
        point_values, point_records = _ks_ladder_prefix_certified_bits(
            runtime=runtime,
            ladder=prefix_ladder,
            horizon=time_value,
        )
        for sample_index, point_record in enumerate(point_records):
            latest_records[sample_index] = point_record
            point_bits = float(point_values[sample_index])
            certified = point_bits > 0.0
            time_points[sample_index].append(
                {
                    "time": time_value,
                    "bits": point_bits if prefix_open[sample_index] and certified else 0.0,
                    "certified_epsilon": point_record.get("certified_epsilon"),
                    "evolution_scale": point_record.get("signal_scale"),
                }
            )
            if not prefix_open[sample_index]:
                continue
            if certified:
                boundary_bits[sample_index] = point_bits
                boundaries[sample_index] = time_value
                boundary_records[sample_index] = point_record
            else:
                prefix_open[sample_index] = False
    diagnostics: list[Mapping[str, object]] = []
    for sample_index, total in enumerate(boundary_bits):
        latest = dict(boundary_records[sample_index] or latest_records[sample_index] or {})
        latest["kind"] = "certified-bits-diagnostics"
        latest["sample_index"] = sample_index
        latest["predictability_boundary"] = boundaries[sample_index]
        latest["time_points"] = [dict(point) for point in time_points[sample_index]]
        latest["bits"] = total
        diagnostics.append(latest)
    result = measured_ladder[-1].new_tensor(boundary_bits)
    result.leibniz_competence_diagnostics = tuple(diagnostics)
    return result


def _float64_tensor(tensor: Any) -> Any:
    convert = getattr(tensor, "double", None)
    return convert() if callable(convert) else tensor


def _ks_ladder_prefix_certified_bits(
    *,
    runtime: TensorRuntime,
    ladder: tuple[Any, ...],
    horizon: float,
) -> tuple[Any, tuple[Mapping[str, object], ...]]:
    finest = ladder[-1]
    evaluation = evaluate_certified_bits(
        _KSDynamicalAmplificationEstimator(
            runtime=runtime,
            ladder=ladder,
            horizon=horizon,
        ),
        sample_count=int(finest.shape[0]),
        value_factory=finest.new_tensor,
    )
    return evaluation.values, evaluation.diagnostics


@dataclass(frozen=True, slots=True)
class _KSDynamicalAmplificationEstimator:
    runtime: TensorRuntime
    ladder: tuple[Any, ...]
    horizon: float
    kind: str = "ks-log-norm-upper-bound"
    structural_type: str = "dynamical-amplification"

    def residuals(self) -> tuple[Any, ...]:
        return tuple(
            _per_sample_residual_norm(trajectory, horizon=self.horizon)
            for trajectory in self.ladder
        )

    def stability(self) -> CertificationStability:
        amplifications = self._amplifications()
        amplification_stability = _amplification_stability_ratio(
            runtime=self.runtime,
            amplifications=amplifications,
        )
        amplification_growing = _amplification_grows_under_refinement(
            runtime=self.runtime,
            amplifications=amplifications,
        )
        return CertificationStability(
            factor=amplifications[-1] * float(self.horizon),
            refused=amplification_growing,
            diagnostics=tuple(
                {
                    "law_amplification": float(amplifications[-1][sample_index]),
                    "law_amplification_ladder": [
                        float(amplification[sample_index])
                        for amplification in amplifications
                    ],
                    "law_amplification_stability": float(
                        amplification_stability[sample_index]
                    ),
                    "law_amplification_estimator": self.kind,
                    "law_amplification_log_cap": _law_amplification_log_cap,
                    "law_amplification_refusal_ratio": _law_amplification_refusal_ratio,
                    "certification_refinement_factors": list(
                        _certification_refinement_factors
                    ),
                }
                for sample_index in range(int(self.ladder[-1].shape[0]))
            ),
            refused_status="refused-amplification-growing",
        )

    def ambient_entropy_above(self, precision: Any) -> AmbientEntropy:
        entropy = _ambient_evolution_entropy(
            runtime=self.runtime,
            trajectory=self.ladder[-1],
            precision=precision,
        )
        return AmbientEntropy(
            bits=entropy.bits,
            signal=entropy.signal,
            diagnostics=tuple(
                {
                    "evolution_scale": float(entropy.signal[sample_index]),
                    "resolved_mode_count": int(
                        entropy.resolved_mode_count[sample_index]
                    ),
                    "ambient_evolution_entropy_bits": float(
                        entropy.bits[sample_index]
                    ),
                }
                for sample_index in range(int(self.ladder[-1].shape[0]))
            ),
        )

    def _amplifications(self) -> tuple[Any, ...]:
        return tuple(
            _per_sample_law_amplification(trajectory, horizon=self.horizon)
            for trajectory in self.ladder
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
class _AmbientEvolutionEntropy:
    bits: Any
    signal: Any
    resolved_mode_count: Any


def _ambient_evolution_entropy(
    *,
    runtime: TensorRuntime,
    trajectory: Any,
    precision: Any,
) -> _AmbientEvolutionEntropy:
    backend = runtime.backend
    evolution = trajectory - trajectory[:, 0:1, :]
    signal = evolution.pow(2).mean(dim=(1, 2)).sqrt()
    spatial_points = float(trajectory.shape[-1])
    spectrum = backend.fft.rfft(evolution.double(), dim=-1).abs() / spatial_points
    precision_tensor = precision.reshape((-1, 1, 1)).to(
        dtype=spectrum.dtype,
        device=spectrum.device,
    )
    precision_tensor = precision_tensor.clamp_min(math.ulp(1.0))
    resolved = spectrum > precision_tensor
    ratios = (spectrum / precision_tensor).clamp_min(1.0)
    bits_by_mode = backend.where(
        resolved,
        backend.log2(ratios),
        backend.zeros_like(ratios),
    )
    return _AmbientEvolutionEntropy(
        bits=bits_by_mode.sum(dim=(1, 2)),
        signal=signal,
        resolved_mode_count=resolved.sum(dim=(1, 2)),
    )


def _per_sample_law_amplification(trajectory: Any, *, horizon: float) -> Any:
    spatial_points = int(trajectory.shape[-1])
    dx = _box_length / float(spatial_points)
    detached = trajectory.detach().double()
    u_x = (
        detached.roll(shifts=-1, dims=-1) - detached.roll(shifts=1, dims=-1)
    ) / (2.0 * dx)
    shear = u_x.abs().amax(dim=(1, 2))
    frequency = trajectory.new_tensor(
        [2.0 * math.pi * index / _box_length for index in range((spatial_points // 2) + 1)]
    ).double()
    linear_growth = (frequency * frequency) - (frequency**4)
    leading_growth = linear_growth.max().clamp_min(0.0)
    exponent = (leading_growth + shear).clamp_max(_law_amplification_log_cap) * float(
        horizon
    )
    return exponent.exp()


def _amplification_stability_ratio(
    *,
    runtime: TensorRuntime,
    amplifications: tuple[Any, ...],
) -> Any:
    values = tensor_runtime_concat(
        runtime,
        tuple(value.reshape((1, -1)) for value in amplifications),
        dim=0,
    )
    numerator = values.max(dim=0).values
    denominator = values.min(dim=0).values.clamp_min(math.ulp(1.0))
    return numerator / denominator


def _amplification_grows_under_refinement(
    *,
    runtime: TensorRuntime,
    amplifications: tuple[Any, ...],
) -> Any:
    if not amplifications:
        raise ValueError("amplification refinement check requires at least one rung")
    finest = amplifications[len(amplifications) - 1]
    batch_count = int(finest.shape[0])
    if len(amplifications) < 3:
        return finest.new_zeros((batch_count,), dtype=bool)
    growth = finest.new_ones((batch_count,), dtype=bool)
    for previous, current in zip(amplifications[:-1], amplifications[1:], strict=True):
        growth = growth & (current > previous)
    ratio = _amplification_stability_ratio(runtime=runtime, amplifications=amplifications)
    return growth & (ratio > _law_amplification_refusal_ratio)


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


def central_time_derivative(trajectory: Any, *, dt: float) -> Any:
    derivative = trajectory.clone()
    derivative[:, 0, :] = (trajectory[:, 1, :] - trajectory[:, 0, :]) / dt
    derivative[:, -1, :] = (trajectory[:, -1, :] - trajectory[:, -2, :]) / dt
    if int(trajectory.shape[1]) > 2:
        derivative[:, 1:-1, :] = (trajectory[:, 2:, :] - trajectory[:, :-2, :]) / (
            2.0 * dt
        )
    return derivative


def _validate_positive_spacing(value: float, *, field: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be positive and finite")


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
