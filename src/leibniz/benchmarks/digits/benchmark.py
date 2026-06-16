"""Digits benchmark implementation entry point."""

from __future__ import annotations

import base64
import hashlib
import math
import os
import random
import struct
import zlib
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from operator import attrgetter
from pathlib import Path
from typing import Any, TypeAlias, cast

from leibniz.artifacts import ArtifactReference
from leibniz.benchmark_implementations import RawBenchmark as BenchmarkProtocol
from leibniz.benchmarks import BenchmarkManifest
from leibniz.cost_metrology import (
    CostMeasurement,
    OperationCostRecord,
)
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.latent_factors import (
    DegreeMeasure,
    GeneratorConstructionFactor,
    LatentFactorDeclaration,
    SampleLatentFactor,
)
from leibniz.materialization import (
    AxisAssignment,
    MaterializationDeclaration,
    MaterializationPlan,
    MaterializationValidationError,
)
from leibniz.observation_formation import (
    ComponentMark,
    FieldObservation,
    ObservationComponent,
    ObservationFormationDeclaration,
    SequenceLayout,
    SpatialAffineVariation,
    VariationTransformDeclaration,
)
from leibniz.observation_generation import (
    GeneratedSample,
    GeneratedSampleSet,
    GenerationRequestOutcome,
    ObservationGenerationError,
    StateSpaceVolumeRequest,
    StateSpaceVolumeValue,
)
from leibniz.observation_showcases import (
    ObservationShowcaseManifest,
    ObservationShowcaseSample,
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
from leibniz.target_contracts import TargetContract
from leibniz.tensor_runtime import (
    TensorBatchProgram,
    TensorElementParameter,
    TensorElementRecipe,
    TensorRuntime,
    TensorRuntimeError,
    resolve_host_tensor_runtime,
    tensor_runtime_construct_tensor,
    tensor_runtime_shape_element_count,
    tensor_value_to_host,
    tensor_value_to_host_values,
)
from leibniz.timing import TimingCollector

__all__ = [
    "InverseDigitsLatent",
    "InverseDigitsObservationBatch",
    "InverseDigitsValidatedBits",
    "StaticMapCertification",
    "benchmark",
    "inverse_digits_static_certification",
    "inverse_digits_latent_vector_reconstruction_loss",
    "inverse_digits_reconstruction",
    "inverse_digits_reconstruction_loss",
    "inverse_digits_validated_bits",
    "new_inverse_digits_secret",
    "render_inverse_digits",
    "render_inverse_digits_tensor",
    "sample_inverse_digits_observations",
]

_benchmark_id = ProtocolIdentifier.parse("benchmarks.digits@0.1.0")
_latent_factor_id = ProtocolIdentifier.parse("benchmarks.digits.latent-factors@0.2.0")
_materialization_id = ProtocolIdentifier.parse("benchmarks.digits.materialization@0.1.0")
_formation_id = ProtocolIdentifier.parse("benchmarks.digits.observation-formation@0.1.0")
_generator_id = ProtocolIdentifier.parse("benchmarks.digits.generator@0.1.0")
_outcome_space_id = ProtocolIdentifier.parse("benchmarks.digits.outcomes@0.1.0")
_field_scalar_construction_bytes = 64
_default_memory_budget_fraction = 0.10
_default_generation_memory_limit_bytes = 32_768_000
_volume_class_digit_count = 10
_volume_class_canvas_minimum_side = 16
_volume_class_canvas_side_step = 4
_canonical_digits_cardinality = _volume_class_digit_count
_batch_render_curve_sample_count = 25
_static_conditioning_refusal_ratio = 2.0
_static_sigma_floor = 1.0e-8
_inverse_residual_operator_id = "benchmarks.digits.inverse-renderer@0.1.0"
_runtime_device = attrgetter("device")
_inverse_nuisance_axis_ranges = {
    "x_translation": 0.32,
    "y_translation": 0.32,
    "scale": 0.46,
    "shear": 0.36,
    "stroke_width": 0.60,
}

# --- Domain-growth chart geometry (fixed render resolution, growing canvas) ---
#
# A problem setup is a digit identity placed at a centre-relative offset and a
# scale. Offsets and scales live on an integer lattice of "transform cells"
# centred on identity; the canvas is a per-sample materialization detail (the
# smallest pixel grid that frames the realized cells at the fixed render
# resolution), not a score-bearing axis. State-space volume counts distinct
# setups, so the score filtration stays over one fixed-resolution ambient of
# unbounded extent. Cells are walked in concentric shells growing outward from
# identity, with within-shell order fixed by a benchmark permutation so the
# walk is deterministic and global but not a trivially reversible spiral.
_render_unit_side = 28  # pixels for a scale-1 digit box. The lowest rung frames a
# single setup, so its canvas equals this footprint; 28px is the MNIST native
# resolution (an upper bound for future MNIST validation) and clears the digit
# discriminability margin at the fixed render pitch with headroom.
_translation_step_pixels = 4  # distinguishable centre-relative shift between cells
_scale_shell_weight = 2  # one scale level costs this many shells of radius
_max_scale_level = 3  # scale levels clamp to +/- this
_scale_ratio_per_level = 0.12  # fractional digit-footprint change per scale level
_component_discriminability_margin = 20.0
_chart_translation_axis_ids = ("x_translation", "y_translation")
_chart_scale_axis_id = "scale"
_chart_axis_ids = (*_chart_translation_axis_ids, _chart_scale_axis_id)
_CurvePoints: TypeAlias = tuple[tuple[float, float], ...]


def _digits_oracle_cost_measurement(
    *,
    pixel_count: int,
    operation_stream_source: str,
) -> CostMeasurement:
    return CostMeasurement(
        cost_model_id=CostMeasurement.tensor_runtime_cost_model_id(),
        abstract_flops=pixel_count,
        per_op=(
            OperationCostRecord(
                name="digits.oracle.pixel-classification",
                calls=1,
                abstract_flops=pixel_count,
                output_elements=pixel_count,
            ),
        ),
        moved_elements=0,
        movement=(),
        unmodeled_operations=(),
        operation_count=1,
        operation_trace=(),
        wall_seconds=0.0,
        tensor_device="oracle",
        execution_mode="dry-run",
        operation_stream_source=operation_stream_source,
        operations_executed=False,
    )


def _cell_radius(tx: int, ty: int, sl: int) -> int:
    """Return the concentric-shell radius of a transform cell."""

    return max(abs(tx), abs(ty), abs(sl) * _scale_shell_weight)


def _shell_cells(radius: int) -> tuple[tuple[int, int, int], ...]:
    """Return every transform cell whose shell radius equals ``radius``."""

    if radius < 0:
        raise ObservationGenerationError("shell radius must be nonnegative")
    if radius == 0:
        return ((0, 0, 0),)
    cached = _shell_cells_cache.get(radius)
    if cached is not None:
        return cached
    cells: list[tuple[int, int, int]] = []
    span = range(-radius, radius + 1)
    for sl in range(-_max_scale_level, _max_scale_level + 1):
        if abs(sl) * _scale_shell_weight > radius:
            continue
        for tx in span:
            for ty in span:
                if _cell_radius(tx, ty, sl) == radius:
                    cells.append((tx, ty, sl))
    ordered = tuple(
        cells[index]
        for index in sorted(
            range(len(cells)), key=lambda i: _shell_permutation_key(radius, i)
        )
    )
    _shell_cells_cache[radius] = ordered
    return ordered


def _shell_permutation_key(radius: int, index: int) -> int:
    digest = hashlib.sha256(
        f"benchmarks.digits.shell-permutation.v1:{radius}:{index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _transform_cell_for_ordinal(ordinal: int) -> tuple[int, int, int]:
    """Return the (x, y, scale) lattice cell at a global transform ordinal."""

    if type(ordinal) is not int or ordinal < 0:
        raise ObservationGenerationError("transform ordinal must be a nonnegative integer")
    while len(_chart_cell_prefix) <= ordinal:
        radius = _chart_prefix_radius[0]
        _chart_cell_prefix.extend(_shell_cells(radius))
        _chart_prefix_radius[0] = radius + 1
    return _chart_cell_prefix[ordinal]


_shell_cells_cache: dict[int, tuple[tuple[int, int, int], ...]] = {}
_chart_cell_prefix: list[tuple[int, int, int]] = []
_chart_prefix_radius: list[int] = [0]


@dataclass(frozen=True, slots=True)
class InverseDigitsLatent:
    """A label-free inverse Digits latent sampled from the renderer prior."""

    identity: int
    x_translation: float
    y_translation: float
    scale: float
    shear: float
    stroke_width: float

    def __post_init__(self) -> None:
        if type(self.identity) is not int or not 0 <= self.identity < _volume_class_digit_count:
            raise ObservationGenerationError("inverse digit identity is outside the prior")
        for name, value in self.to_nuisance_record().items():
            if not isinstance(value, float) or not math.isfinite(value):
                raise ObservationGenerationError(f"inverse digit {name} must be finite")

    def to_nuisance_tuple(self) -> tuple[float, float, float, float, float]:
        return (
            self.x_translation,
            self.y_translation,
            self.scale,
            self.shear,
            self.stroke_width,
        )

    def to_nuisance_record(self) -> dict[str, float]:
        return {
            "x_translation": self.x_translation,
            "y_translation": self.y_translation,
            "scale": self.scale,
            "shear": self.shear,
            "stroke_width": self.stroke_width,
        }

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "digits-inverse-latent",
            "identity": self.identity,
            "nuisance": self.to_nuisance_record(),
        }


@dataclass(frozen=True, slots=True)
class InverseDigitsObservationBatch:
    """Private inverse-Digits evaluation batch plus its public submission view."""

    observations: Any
    latents: tuple[InverseDigitsLatent, ...]
    canvas_side: int
    secret_digest: str

    def submission_view(self) -> Mapping[str, object]:
        return {
            "kind": "digits-inverse-submission-view",
            "observation_count": len(self.latents),
            "canvas_side": self.canvas_side,
            "observation_shape": [int(axis) for axis in self.observations.shape],
            "law_id": "benchmarks.digits.inverse-renderer@0.1.0",
        }


@dataclass(frozen=True, slots=True)
class StaticMapCertification:
    """A static-map residual-to-latent-precision certificate."""

    certified_epsilon: float
    residual_norm: float
    sigma_min: float
    sigma_min_ladder: tuple[float, ...]
    conditioning_stability: float
    certification_status: str

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "digits-static-map-certification",
            "estimator": "renderer-jvp-gram-sigma-min",
            "certified_epsilon": self.certified_epsilon,
            "residual_norm": self.residual_norm,
            "sigma_min": self.sigma_min,
            "sigma_min_ladder": list(self.sigma_min_ladder),
            "conditioning_stability": self.conditioning_stability,
            "conditioning_refusal_ratio": _static_conditioning_refusal_ratio,
            "certification_status": self.certification_status,
        }


@dataclass(frozen=True, slots=True)
class InverseDigitsValidatedBits:
    """Product-latent ambient entropy resolved to certified precision."""

    bits: float
    identity_bits: float
    nuisance_bits: float
    distinguishable_identity_count: int
    certified_epsilon: float

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "digits-inverse-product-entropy",
            "bits": self.bits,
            "identity_bits": self.identity_bits,
            "nuisance_bits": self.nuisance_bits,
            "distinguishable_identity_count": self.distinguishable_identity_count,
            "certified_epsilon": self.certified_epsilon,
        }


@dataclass(frozen=True, slots=True)
class _DigitsChart:
    """The fixed-resolution chart sizing for one realized window.

    The transform enumeration is global; ``canvas_side`` is the only per-window
    materialization parameter, derived from the deepest realized cell.
    """

    canvas_side: int

    def normalized_transform(self, ordinal: int) -> tuple[float, float, float]:
        """Return centre-relative (x_translation, y_translation, scale) in [0, 1]."""

        tx, ty, sl = _transform_cell_for_ordinal(ordinal)
        scale = _scale_footprint(sl) / self.canvas_side
        x_translation = (tx * _translation_step_pixels) / self.canvas_side
        y_translation = (ty * _translation_step_pixels) / self.canvas_side
        return (x_translation, y_translation, scale)


def _scale_footprint(scale_level: int) -> float:
    """Return the pixel footprint of a scale-1 digit box at a scale level."""

    return _render_unit_side * (1.0 + _scale_ratio_per_level * scale_level)


def _shell_size(radius: int) -> int:
    """Return the number of transform cells at a shell radius (closed form)."""

    if radius <= 0:
        return 1
    inner_scale_levels = min(_max_scale_level, (radius - 1) // _scale_shell_weight)
    size = (2 * inner_scale_levels + 1) * 8 * radius
    if radius % _scale_shell_weight == 0 and radius // _scale_shell_weight <= _max_scale_level:
        size += 2 * (2 * radius + 1) ** 2
    return size


def _radius_for_ordinal(ordinal: int) -> int:
    """Return the shell radius that contains a global transform ordinal."""

    cumulative = 0
    radius = 0
    while True:
        cumulative += _shell_size(radius)
        if ordinal < cumulative:
            return radius
        radius += 1


def _chart_canvas_side_for_max_ordinal(max_ordinal: int) -> int:
    """Return the smallest stepped square canvas framing cells up to an ordinal.

    Every cell up to ``max_ordinal`` has shell radius at most the radius of the
    shell containing it, so a canvas sized for that radius's reach frames them
    all. This is closed form in the radius, so it stays fast at any depth.
    """

    radius = _radius_for_ordinal(max_ordinal)
    max_translation_steps = radius
    max_scale_level = min(_max_scale_level, radius // _scale_shell_weight)
    footprint = _scale_footprint(max_scale_level)
    side = footprint + 2.0 * max_translation_steps * _translation_step_pixels
    side = max(float(_render_unit_side), side)
    stepped = math.ceil(side / _volume_class_canvas_side_step) * _volume_class_canvas_side_step
    return int(stepped)


@dataclass(frozen=True, slots=True)
class _DigitsVolumeClass:
    """A realized window: a contiguous global-address increment plus its canvas.

    The window covers global addresses ``[minimum_address, minimum_address +
    cardinality)``; each address ``a`` is the setup ``(digit a % 10, transform
    a // 10)``. Volume is the number of distinct setups in the increment.
    """

    minimum_address: int
    cardinality: int
    canvas_side: int

    def __post_init__(self) -> None:
        if type(self.minimum_address) is not int or self.minimum_address < 0:
            raise ObservationGenerationError("minimum_address must be a nonnegative integer")
        if type(self.cardinality) is not int or self.cardinality < 1:
            raise ObservationGenerationError("cardinality must be a positive integer")
        if type(self.canvas_side) is not int or self.canvas_side < _render_unit_side:
            raise ObservationGenerationError("canvas_side must be at least the unit render side")

    @property
    def digit_count(self) -> int:
        return _volume_class_digit_count

    @property
    def maximum_address(self) -> int:
        return self.minimum_address + self.cardinality - 1

    @property
    def maximum_transform_ordinal(self) -> int:
        return self.maximum_address // _volume_class_digit_count

    @property
    def chart(self) -> _DigitsChart:
        return _DigitsChart(canvas_side=self.canvas_side)

    @property
    def log2_volume(self) -> float:
        return math.log2(self.cardinality)

    def measure(self) -> StateSpaceVolumeValue:
        return _volume_value(self.log2_volume)

    def resolution_assignment(self, *, width_axis: str, height_axis: str) -> AxisAssignment:
        return AxisAssignment(values={width_axis: self.canvas_side, height_axis: self.canvas_side})

    def metadata(self) -> dict[str, object]:
        pixel_count = tensor_runtime_shape_element_count(
            (self.canvas_side, self.canvas_side)
        )
        return {
            "kind": "digits-realized-setup-window",
            "digit_count": self.digit_count,
            "output_digit_count": _volume_class_digit_count,
            "minimum_address": self.minimum_address,
            "maximum_address": self.maximum_address,
            "cardinality": self.cardinality,
            "realized_cardinality": self.cardinality,
            "maximum_transform_ordinal": self.maximum_transform_ordinal,
            "canvas_side": self.canvas_side,
            "transform_axes": list(_chart_axis_ids),
            "construction": "digit-setups-over-shell-ordered-transform-lattice",
            "oracle_cost_measurement": _digits_oracle_cost_measurement(
                pixel_count=pixel_count,
                operation_stream_source="digits-volume-class-oracle",
            ).to_record(),
            "oracle_cost_components": {
                "height": self.canvas_side,
                "width": self.canvas_side,
                "pixel_count": pixel_count,
            },
        }


_digit_strokes: tuple[tuple[_CurvePoints, ...], ...] = (
    (
        ((0.5, 0.2768), (0.3056, 0.2912), (0.2984, 0.5)),
        ((0.2984, 0.5), (0.3056, 0.7088), (0.5, 0.7232)),
        ((0.5, 0.7232), (0.6944, 0.7088), (0.7016, 0.5)),
        ((0.7016, 0.5), (0.6944, 0.2912), (0.5, 0.2768)),
    ),
    (
        ((0.4208, 0.3704), (0.5072, 0.2912), (0.5648, 0.2768)),
        ((0.5648, 0.2768), (0.5576, 0.7016)),
        ((0.4424, 0.7088), (0.6512, 0.7088)),
    ),
    (
        ((0.3272, 0.3488), (0.428, 0.2552), (0.5792, 0.2912)),
        ((0.5792, 0.2912), (0.7448, 0.3416), (0.6224, 0.4712)),
        ((0.6224, 0.4712), (0.5144, 0.572), (0.3488, 0.6872)),
        ((0.3488, 0.6872), (0.68, 0.6944)),
    ),
    (
        ((0.3344, 0.32), (0.5432, 0.2408), (0.6584, 0.3704)),
        ((0.6584, 0.3704), (0.716, 0.4784), (0.5144, 0.4928)),
        ((0.5144, 0.4928), (0.7304, 0.5432), (0.6512, 0.6584)),
        ((0.6512, 0.6584), (0.5072, 0.7808), (0.32, 0.6728)),
    ),
    (
        ((0.6224, 0.284), (0.3488, 0.5288)),
        ((0.3488, 0.5288), (0.6656, 0.5288)),
        ((0.6224, 0.284), (0.6224, 0.7088)),
    ),
    (
        ((0.6584, 0.2912), (0.3632, 0.2912)),
        ((0.3632, 0.2912), (0.32, 0.4352), (0.4064, 0.4928)),
        ((0.4064, 0.4928), (0.6584, 0.4352), (0.6728, 0.6152)),
        ((0.6728, 0.6152), (0.5792, 0.7592), (0.3416, 0.68)),
    ),
    (
        ((0.6368, 0.3056), (0.3776, 0.32), (0.3272, 0.5648)),
        ((0.3272, 0.5648), (0.356, 0.752), (0.5288, 0.7232)),
        ((0.5288, 0.7232), (0.7088, 0.68), (0.6512, 0.536)),
        ((0.6512, 0.536), (0.5288, 0.428), (0.356, 0.5216)),
    ),
    (
        ((0.3344, 0.2912), (0.68, 0.2912)),
        ((0.68, 0.2912), (0.5432, 0.4928), (0.4712, 0.7088)),
    ),
    (
        ((0.5, 0.4928), (0.3416, 0.4352), (0.3848, 0.3272)),
        ((0.3848, 0.3272), (0.5072, 0.2264), (0.6296, 0.3272)),
        ((0.6296, 0.3272), (0.6728, 0.4424), (0.5, 0.4928)),
        ((0.5, 0.4928), (0.3056, 0.5576), (0.3632, 0.6728)),
        ((0.3632, 0.6728), (0.5, 0.7808), (0.644, 0.6728)),
        ((0.644, 0.6728), (0.7016, 0.5576), (0.5, 0.4928)),
    ),
    (
        ((0.644, 0.4712), (0.5288, 0.572), (0.3632, 0.4928)),
        ((0.3632, 0.4928), (0.3128, 0.3272), (0.4856, 0.2768)),
        ((0.4856, 0.2768), (0.6728, 0.2912), (0.68, 0.4784)),
        ((0.68, 0.4784), (0.6512, 0.6584), (0.4208, 0.7088)),
    ),
)


@dataclass(frozen=True, slots=True)
class _ResolutionSampling:
    width_axis: str
    height_axis: str


def benchmark(root: Path) -> BenchmarkProtocol:
    """Return the Digits benchmark implementation."""

    return Benchmark(root=root)


class Benchmark:
    """Executable Digits benchmark declaration."""

    def __init__(self, *, root: Path) -> None:
        self._root = root
        self._latent_factors = _latent_factors()
        self._manifest = _manifest()
        self._materialization = _materialization()
        self._formation = _formation()
        self._showcase = _showcase()
        self._target_contract = TargetContract.inverse_latent(
            identity_count=10,
            nuisance_dimension=5,
            residual_operator_id=_inverse_residual_operator_id,
        )
        self._generator = Generator(
            manifest=self._manifest,
            latent_factors=self._latent_factors,
            materialization=self._materialization,
            formation=self._formation,
        )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def manifest(self) -> BenchmarkManifest:
        return self._manifest

    @property
    def latent_factors(self) -> LatentFactorDeclaration:
        return self._latent_factors

    @property
    def materialization(self) -> MaterializationDeclaration:
        return self._materialization

    @property
    def formation(self) -> ObservationFormationDeclaration:
        return self._formation

    @property
    def showcase(self) -> ObservationShowcaseManifest:
        return self._showcase

    @property
    def target_contract(self) -> TargetContract:
        return self._target_contract

    def build_training_loss(
        self,
        runtime: TensorRuntime,
        target_contract: TargetContract,
    ) -> Any:
        if (
            target_contract.kind != "inverse"
            or target_contract.loss_id != "reconstruction"
            or target_contract.competence.parameters.get("residual_operator_id")
            != _inverse_residual_operator_id
        ):
            raise ValueError("Digits benchmark only builds its declared inverse loss")

        def loss(predictions: Any, targets: Any) -> Any:
            if len(tuple(targets.shape)) != 4 or int(targets.shape[1]) != 1:
                raise ValueError(
                    "inverse Digits targets must have shape (batch, 1, side, side)"
                )
            side = int(targets.shape[-1])
            if int(targets.shape[-2]) != side:
                raise ValueError("inverse Digits targets must be square images")
            return inverse_digits_latent_vector_reconstruction_loss(
                runtime=runtime,
                latent_vector=predictions,
                observations=targets,
                canvas_side=side,
            )

        return loss

    def build_training_competence(
        self,
        runtime: TensorRuntime,
        target_contract: TargetContract,
    ) -> Any:
        del runtime
        if (
            target_contract.kind != "inverse"
            or target_contract.competence.kind != "ambient-certified-bits"
            or target_contract.competence.parameters.get("residual_operator_id")
            != _inverse_residual_operator_id
        ):
            raise ValueError(
                "Digits benchmark only builds its declared inverse certified competence"
            )

        def competence(request: Any) -> Any:
            return _inverse_digits_ambient_certified_bits(request)

        return competence

    @property
    def generator(self) -> Generator:
        return self._generator

    @property
    def sampling_protocol(self) -> SamplingProtocol:
        # Digits is a metric-resolution space, so its protocol never saturates to
        # a census (that path is gated to exact-distinguishability regions); the
        # saturation threshold is left unset rather than carrying a count.
        return SamplingProtocol(
            kind="uniform-monte-carlo",
            estimator_id="sample-mean",
            confidence_method_id="wilson",
            census_budget=None,
        )

    @property
    def accessible_subspace(self) -> AccessibleSubspace:
        return AccessibleSubspace(
            ladder_id="digits-continuous-transform-covering",
            per_configuration_capacity=_digits_capacity_region(),
            frontier_rationale=(
                "All digit identities over the benchmark continuous transform chart are in scope; "
                "the finite capacity is the practical shared per-configuration bound."
            ),
        )


@dataclass(frozen=True, slots=True)
class Generator:
    """Generate samples from the Digits scientific model."""

    manifest: BenchmarkManifest
    latent_factors: LatentFactorDeclaration
    materialization: MaterializationDeclaration
    formation: ObservationFormationDeclaration

    def __post_init__(self) -> None:
        if self.materialization.benchmark_id != self.manifest.id:
            raise ObservationGenerationError(
                "materialization benchmark_id does not match benchmark manifest"
            )
        if self.formation.benchmark_id != self.manifest.id:
            raise ObservationGenerationError(
                "formation benchmark_id does not match benchmark manifest"
            )
        if self.manifest.latent_factor_declaration is None:
            raise ObservationGenerationError("benchmark manifest must declare latent factors")
        try:
            self.manifest.validate_latent_factor_declaration(self.latent_factors)
        except ValueError as error:
            raise ObservationGenerationError(str(error)) from error

    def _generate_samples(
        self,
        *,
        sample_count: int,
        seed: int,
        sample_indices: tuple[int, ...],
        include_fields: bool,
        include_artifacts: bool,
        memory_limit_bytes: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
        volume_class: _DigitsVolumeClass,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
        output_timing_prefix: str = "",
    ) -> tuple[GeneratedSample, ...]:
        """Generate Digits samples by choosing digit, canvas, and affine variation."""

        if type(sample_count) is not int or sample_count < 1:
            raise ObservationGenerationError("sample_count must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ObservationGenerationError("seed must be a nonnegative integer")
        component_index_samples, transform_index_samples = (
            self._sample_state_coordinates(
                sample_count=sample_count,
                seed=seed,
                sample_indices=sample_indices,
                volume_class=volume_class,
                timing=timing,
                timing_prefix=timing_prefix,
            )
        )
        resolved_resolution_assignment = self._generation_resolution_assignment(
            sample_count=sample_count,
            seed=seed,
            requested_assignment=resolution_assignment,
            memory_limit_bytes=memory_limit_bytes,
        )
        transform = self.formation.variation_transform
        transform_record = transform.to_record()
        materialization_declaration = ArtifactReference(
            kind="materialization-declaration",
            protocol_id=self.materialization.id,
            record_digest=self.materialization.digest,
        )
        with _timing_span(
            timing,
            f"{timing_prefix}materialization_plan",
            samples=sample_count,
        ):
            plans = tuple(
                self._materialization_plan(
                    seed=seed,
                    index=sample_indices[index],
                    resolution_assignment=resolved_resolution_assignment,
                    materialization_declaration=materialization_declaration,
                )
                for index in range(sample_count)
            )
        variation_timing_phase = f"{timing_prefix}variation_coordinates"
        variation_samples = self._sample_variation_coordinates(
            seed=seed,
            sample_indices=sample_indices,
            plans=plans,
            transform=transform,
            transform_record=transform_record,
            component_indices=component_index_samples,
            transform_indices=transform_index_samples,
            volume_class=volume_class,
            timing=timing,
            timing_phase=variation_timing_phase,
        )

        with _timing_span(timing, f"{output_timing_prefix}scaled_factors"):
            scaled_factors = tuple(self.latent_factors.sample_factors)

        field_records: tuple[FieldObservation, ...]
        if include_fields:
            with _timing_span(
                timing,
                f"{output_timing_prefix}field_generation",
                samples=sample_count,
            ):
                fields = self._generate_tensor_fields(
                    sample_shape=(sample_count,),
                    seed=seed,
                    sample_indices=sample_indices,
                    volume_class=volume_class,
                    resolution_assignment=resolved_resolution_assignment,
                    memory_limit_bytes=memory_limit_bytes,
                    runtime=resolve_host_tensor_runtime(),
                    timing=timing,
                    timing_prefix=output_timing_prefix,
                )
                field_records = self._field_observations_from_tensor_fields(
                    plans=plans,
                    fields=fields,
                )
        else:
            field_records = ()

        samples: list[GeneratedSample] = []
        with _timing_span(
            timing,
            f"{output_timing_prefix}sample_assembly",
            samples=sample_count,
        ):
            for row_index, (
                index,
                plan,
                component_index,
                transform_index,
                variation_sample,
            ) in enumerate(
                zip(
                    sample_indices,
                    plans,
                    component_index_samples,
                    transform_index_samples,
                    variation_samples,
                    strict=True,
                )
            ):
                variation_values, variation_coordinates = variation_sample
                field_record = field_records[row_index] if include_fields else None
                artifacts = (
                    {
                        "field_shape": list(field_record.shape),
                        "image_data_url": _field_to_png_data_url(field_record),
                    }
                    if include_artifacts and field_record is not None
                    else None
                )
                digit_variant_index = 0
                samples.append(
                    GeneratedSample(
                        index=index,
                        materialization_plan=plan,
                        width=plan.resolution_assignment.require_axis(self.formation.width_axis),
                        height=plan.resolution_assignment.require_axis(self.formation.height_axis),
                        component_index=component_index,
                        region_component_index=_digits_region_component_index(
                            component_index=component_index,
                            transform_ordinal=transform_index,
                            volume_class=volume_class,
                        ),
                        axis_coordinates=_digits_region_axis_coordinates(
                            seed=seed,
                            sample_index=index,
                            transform_ordinal=transform_index,
                            canvas_side=volume_class.canvas_side,
                            canonical_transform=_digits_volume_class_is_canonical(
                                volume_class
                            ),
                        ),
                        variation_coordinates=variation_coordinates,
                        variation_values=variation_values,
                        outcome_id=self._outcome_id(component_index),
                        latent_coordinates=self._latent_coordinates(
                            component_index=component_index,
                            digit_variant_index=digit_variant_index,
                            scaled_factors=scaled_factors,
                            plan=plan,
                            variation_values=variation_values,
                        ),
                        field=field_record,
                        artifacts=artifacts,
                    )
                )
        return tuple(samples)

    def _generate_tensor_metadata_samples(
        self,
        *,
        sample_count: int,
        seed: int,
        sample_indices: tuple[int, ...],
        volume_class: _DigitsVolumeClass,
        resolution_assignment: AxisAssignment,
    ) -> tuple[GeneratedSample, ...]:
        component_indices, transform_indices = self._sample_state_coordinates(
            sample_count=sample_count,
            seed=seed,
            sample_indices=sample_indices,
            volume_class=volume_class,
            timing=None,
            timing_prefix="",
        )
        return tuple(
            GeneratedSample(
                index=sample_indices[index],
                outcome_id=self._outcome_id(component_index),
                component_index=component_index,
                region_component_index=_digits_region_component_index(
                    component_index=component_index,
                    transform_ordinal=transform_indices[index],
                    volume_class=volume_class,
                ),
                axis_coordinates=_digits_region_axis_coordinates(
                    seed=seed,
                    sample_index=sample_indices[index],
                    transform_ordinal=transform_indices[index],
                    canvas_side=volume_class.canvas_side,
                    canonical_transform=_digits_volume_class_is_canonical(volume_class),
                ),
            )
            for index, component_index in enumerate(component_indices)
        )

    def _generation_resolution_assignment(
        self,
        *,
        sample_count: int,
        seed: int,
        requested_assignment: AxisAssignment | None,
        memory_limit_bytes: int | None,
    ) -> AxisAssignment:
        minimum_assignment = self.materialization.minimum_resolution()
        if requested_assignment is None:
            resolution_assignment = self._sample_resolution_assignment(
                sample_count=sample_count,
                seed=seed,
                minimum_assignment=minimum_assignment,
                memory_limit_bytes=memory_limit_bytes,
            )
        else:
            resolution_assignment = self._requested_resolution_assignment(
                minimum_assignment=minimum_assignment,
                requested_assignment=requested_assignment,
            )
        try:
            self.materialization.require_resolution(
                resolution_assignment=resolution_assignment,
            )
        except MaterializationValidationError as error:
            raise ObservationGenerationError(str(error)) from error
        return resolution_assignment

    def _sample_variation_coordinates(
        self,
        *,
        seed: int,
        sample_indices: tuple[int, ...],
        plans: tuple[MaterializationPlan, ...],
        transform: VariationTransformDeclaration,
        transform_record: Mapping[str, object],
        component_indices: tuple[int, ...],
        transform_indices: tuple[int, ...],
        volume_class: _DigitsVolumeClass,
        timing: TimingCollector | None,
        timing_phase: str,
    ) -> tuple[
        tuple[
            Mapping[str, object],
            tuple[Mapping[str, object], ...],
        ],
        ...,
    ]:
        samples: list[
            tuple[
                Mapping[str, object],
                tuple[Mapping[str, object], ...],
            ]
        ] = []
        if len(component_indices) != len(plans):
            raise ObservationGenerationError("component_indices length must match sample_count")
        if len(transform_indices) != len(plans):
            raise ObservationGenerationError("transform_indices length must match sample_count")
        chart = volume_class.chart
        with _timing_span(timing, timing_phase, samples=len(plans)):
            for index, _plan in enumerate(plans):
                component_index = component_indices[index]
                transform_ordinal = transform_indices[index]
                if transform_ordinal < 0:
                    raise ObservationGenerationError("transform ordinal must be nonnegative")
                coordinate = _constructed_variation_coordinate_record(
                    transform=transform,
                    component_index=component_index,
                    transform_ordinal=transform_ordinal,
                    transform_coordinates=_digits_sample_transform_coordinates(
                        seed=seed,
                        sample_index=sample_indices[index],
                        transform_ordinal=transform_ordinal,
                        canvas_side=volume_class.canvas_side,
                        canonical_transform=_digits_volume_class_is_canonical(
                            volume_class
                        ),
                    ),
                    chart=chart,
                )
                coordinates = (coordinate,)
                samples.append(
                    (
                        {
                            "kind": "constructed-field-variation-transform-samples",
                            "bounds": transform_record,
                            "volume_class": volume_class.metadata(),
                            "transform_ordinal": transform_ordinal,
                            "coordinates": [dict(item) for item in coordinates],
                        },
                        coordinates,
                    )
                )
        return tuple(samples)

    def _sample_state_coordinates(
        self,
        *,
        sample_count: int,
        seed: int,
        sample_indices: tuple[int, ...],
        volume_class: _DigitsVolumeClass,
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        component_indices: list[int] = []
        transform_indices: list[int] = []
        with _timing_span(timing, f"{timing_prefix}sample_state", samples=sample_count):
            for index in range(sample_count):
                sample_address = _digits_sample_address(
                    seed=seed,
                    sample_index=sample_indices[index],
                    cardinality=volume_class.cardinality,
                    minimum_address=volume_class.minimum_address,
                )
                component_index, transform_index = _digits_state_coordinate(
                    sample_address=sample_address,
                )
                component_indices.append(component_index)
                transform_indices.append(transform_index)
        return (tuple(component_indices), tuple(transform_indices))

    def _generate_tensors(
        self,
        *,
        sample_shape: tuple[int, ...],
        seed: int,
        sample_indices: tuple[int, ...],
        volume_class: _DigitsVolumeClass,
        resolution_assignment: AxisAssignment | None,
        memory_limit_bytes: int | None,
        runtime: TensorRuntime,
        outcome_ids: tuple[str, ...],
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> tuple[Any, Any]:
        """Generate tensor fields and targets directly from the Digits volume shell."""

        sample_count = _sample_count(sample_shape)
        fields = self._generate_tensor_fields(
            sample_shape=sample_shape,
            seed=seed,
            sample_indices=sample_indices,
            volume_class=volume_class,
            resolution_assignment=resolution_assignment,
            memory_limit_bytes=memory_limit_bytes,
            runtime=runtime,
            timing=timing,
            timing_prefix=timing_prefix,
        )
        if not outcome_ids:
            return fields, fields
        with _timing_span(timing, f"{timing_prefix}target_tensor", samples=sample_count):
            component_outcome_ids = tuple(
                outcome.id for outcome in self.manifest.outcome_space.outcomes
            )
            unknown = tuple(
                outcome_id
                for outcome_id in component_outcome_ids[: volume_class.digit_count]
                if outcome_id not in outcome_ids
            )
            if unknown:
                raise TensorRuntimeError(f"unknown target outcome id: {unknown[0]}")
            component_to_outcome = tuple(
                outcome_ids.index(outcome_id)
                for outcome_id in component_outcome_ids[: volume_class.digit_count]
            )
            labels = tensor_runtime_construct_tensor(
                runtime,
                recipe=TensorElementRecipe(
                    shape=(*sample_shape, len(outcome_ids))
                    if sample_shape
                    else (len(outcome_ids),),
                    dtype="float32",
                    program=_target_tensor_program(
                        seed=seed,
                        sample_indices=sample_indices,
                        cardinality=volume_class.cardinality,
                        minimum_address=volume_class.minimum_address,
                        component_to_outcome=component_to_outcome,
                        outcome_count=len(outcome_ids),
                    ),
                ),
            )
        return fields, labels

    def _generate_tensor_fields(
        self,
        *,
        sample_shape: tuple[int, ...],
        seed: int,
        sample_indices: tuple[int, ...],
        volume_class: _DigitsVolumeClass,
        resolution_assignment: AxisAssignment | None,
        memory_limit_bytes: int | None,
        runtime: TensorRuntime,
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> Any:
        sample_count = _sample_count(sample_shape)
        resolved_resolution_assignment = self._generation_resolution_assignment(
            sample_count=sample_count,
            seed=seed,
            requested_assignment=resolution_assignment,
            memory_limit_bytes=memory_limit_bytes,
        )
        width = resolved_resolution_assignment.require_axis(self.formation.width_axis)
        height = resolved_resolution_assignment.require_axis(self.formation.height_axis)
        with _timing_span(timing, f"{timing_prefix}batch_tensor_render", samples=sample_count):
            fields = self._build_batch_tensor(
                sample_count=sample_count,
                width=width,
                height=height,
                digit_count=volume_class.digit_count,
                seed=seed,
                sample_indices=sample_indices,
                cardinality=volume_class.cardinality,
                minimum_address=volume_class.minimum_address,
                runtime=runtime,
                timing=timing,
                timing_prefix=timing_prefix,
            )
        if sample_shape:
            return fields.reshape((*sample_shape, *tuple(fields.shape[1:])))
        return fields.reshape(tuple(fields.shape[1:]))

    def _field_observations_from_tensor_fields(
        self,
        *,
        plans: tuple[MaterializationPlan, ...],
        fields: Any,
    ) -> tuple[FieldObservation, ...]:
        flat_fields = fields.reshape(
            (
                len(plans),
                self.formation.channel_count,
                plans[0].resolution_assignment.require_axis(self.formation.height_axis),
                plans[0].resolution_assignment.require_axis(self.formation.width_axis),
            )
        )
        host_fields = tensor_value_to_host(flat_fields)
        records: list[FieldObservation] = []
        for index, _plan in enumerate(plans):
            field_shape = tuple(int(size) for size in host_fields[index].shape)
            if len(field_shape) != 3:
                raise ObservationGenerationError("rendered field shape must have rank 3")
            records.append(
                FieldObservation(
                    shape=(field_shape[0], field_shape[1], field_shape[2]),
                    values=tuple(tensor_value_to_host_values(host_fields[index])),
                )
            )
        return tuple(records)

    def _build_batch_tensor(
        self,
        *,
        sample_count: int,
        width: int,
        height: int,
        digit_count: int,
        seed: int,
        sample_indices: tuple[int, ...],
        cardinality: int,
        minimum_address: int,
        runtime: TensorRuntime,
        timing: TimingCollector | None,
        timing_prefix: str,
    ) -> Any:
        _require_positive_integer(width, "width")
        _require_positive_integer(height, "height")
        _require_positive_integer(digit_count, "digit_count")
        if digit_count > len(self.formation.components):
            raise TensorRuntimeError("digit_count exceeds component vocabulary")
        if self.formation.channel_count != 1:
            raise TensorRuntimeError("Digits tensor renderer requires one channel")
        max_mark_count = max(
            len(self.formation.components[component_index].marks)
            for component_index in range(digit_count)
        )
        component_mark_counts: list[int] = []
        component_mark_parameters: list[list[list[float]]] = []
        component_control_points: list[list[list[list[float]]]] = []
        for component_index in range(digit_count):
            mark_parameters: list[list[float]] = []
            control_points: list[list[list[float]]] = []
            for mark in self.formation.components[component_index].marks:
                if mark.channel != 0:
                    raise TensorRuntimeError("Digits tensor renderer requires single-channel marks")
                controls = _quadratic_control_points(mark)
                mark_parameters.append([float(mark.value), float(mark.width)])
                control_points.append([[point[0], point[1]] for point in controls])
            component_mark_counts.append(len(mark_parameters))
            while len(mark_parameters) < max_mark_count:
                mark_parameters.append([0.0, 0.0])
                control_points.append([[0.0, 0.0] for _ in range(3)])
            component_mark_parameters.append(mark_parameters)
            component_control_points.append(control_points)
        if not max_mark_count:
            return tensor_runtime_construct_tensor(
                runtime,
                recipe=TensorElementRecipe(
                    shape=(sample_count, 1, height, width),
                    dtype="float32",
                    program=_constant_tensor_program(0.0),
                ),
            )
        return tensor_runtime_construct_tensor(
            runtime,
            recipe=TensorElementRecipe(
                shape=(sample_count, 1, height, width),
                dtype="float32",
                program=_digits_tensor_program(
                    seed=seed,
                    sample_indices=sample_indices,
                    cardinality=cardinality,
                    minimum_address=minimum_address,
                    digit_count=digit_count,
                    transform_values=_digits_sample_transform_values(
                        seed=seed,
                        canvas_side=width,
                        sample_indices=sample_indices,
                        sample_addresses=_digits_sample_addresses(
                            seed=seed,
                            sample_indices=sample_indices,
                            cardinality=cardinality,
                            minimum_address=minimum_address,
                        ),
                        digit_count=digit_count,
                        canonical_transform=(
                            minimum_address == 0 and cardinality == digit_count
                        ),
                    ),
                    component_mark_counts=tuple(component_mark_counts),
                    component_mark_parameters=tuple(
                        tuple(tuple(parameters) for parameters in component)
                        for component in component_mark_parameters
                    ),
                    component_control_points=tuple(
                        tuple(tuple(tuple(point) for point in mark) for mark in component)
                        for component in component_control_points
                    ),
                    height=height,
                    width=width,
                ),
            ),
        )

    def distinguishable_state_log2_volume(
        self,
        *,
        width: int,
        height: int,
        variation_extent: float = 1.0,
    ) -> float:
        _require_generation_positive_integer(width, "width")
        _require_generation_positive_integer(height, "height")
        try:
            variation_extent_value = float(variation_extent)
        except (TypeError, ValueError) as error:
            raise ObservationGenerationError("variation_extent must be finite") from error
        if not math.isfinite(variation_extent_value):
            raise ObservationGenerationError("variation_extent must be finite")
        if variation_extent_value < 0.0 or variation_extent_value > 1.0:
            raise ObservationGenerationError("variation_extent must be between 0 and 1")
        return self._default_volume_class(
            canonical_variation=variation_extent_value == 0.0,
        ).log2_volume

    def minimum_log2_volume(self) -> StateSpaceVolumeValue:
        """Return the smallest score-bearing Digits log2 volume.

        The task is always 10-way digit classification; a window with one setup
        is the floor, so the minimum is ~0 bits, not the 10-way base.
        """

        return _volume_value(0.0)

    def _volume_class_for_request(
        self,
        *,
        request: StateSpaceVolumeRequest,
    ) -> _DigitsVolumeClass | None:
        """Return the realized setup-window increment for a volume request.

        The window covers zero-based setup addresses ``[N(min)-1, N(max)-1)``
        where ``N(level) = round(2 ** level)`` is the cumulative setup count,
        so consecutive curriculum windows realize disjoint new setups while
        the origin setup remains address 0.
        """

        if request.maximum < self.minimum_log2_volume().value:
            return None
        lower_count = _cumulative_setup_count(
            max(request.minimum, self.minimum_log2_volume().value)
        )
        upper_count = _cumulative_setup_count(request.maximum)
        lower = max(0, lower_count - 1)
        upper = max(0, upper_count - 1)
        if upper <= lower:
            return None
        cardinality = upper - lower
        max_ordinal = (upper - 1) // _volume_class_digit_count
        volume_class = _DigitsVolumeClass(
            minimum_address=lower,
            cardinality=cardinality,
            canvas_side=_chart_canvas_side_for_max_ordinal(max_ordinal),
        )
        if not request.contains(volume_class.measure()):
            return None
        return volume_class

    def oracle_cost_reference_points(
        self,
        *,
        maximum_cost: float,
    ) -> tuple[dict[str, object], ...]:
        """Return deterministic Digits oracle-reference points across inference cost."""

        if not math.isfinite(maximum_cost) or maximum_cost <= 0.0:
            raise ObservationGenerationError("maximum_cost must be positive and finite")
        points: list[dict[str, object]] = []
        max_ordinal = 0
        while True:
            canvas_side = _chart_canvas_side_for_max_ordinal(max_ordinal)
            cost = tensor_runtime_shape_element_count((canvas_side, canvas_side))
            cardinality = _volume_class_digit_count * (max_ordinal + 1)
            log2_volume = math.log2(cardinality)
            points.append(
                {
                    "log2_volume": log2_volume,
                    "score": log2_volume,
                    "cost_measurement": _digits_oracle_cost_measurement(
                        pixel_count=cost,
                        operation_stream_source="digits-oracle-reference-curve",
                    ).to_record(),
                    "metadata": {
                        "kind": "oracle-cost-measurement-reference-v1",
                        "components": {
                            "height": canvas_side,
                            "width": canvas_side,
                            "pixel_count": cost,
                            "sample_cardinality": cardinality,
                            "maximum_transform_ordinal": max_ordinal,
                        },
                    },
                }
            )
            if cost >= maximum_cost and len(points) >= 2:
                break
            max_ordinal = max_ordinal * 2 + 1
        return tuple(points)

    def _default_volume_class(
        self,
        *,
        canonical_variation: bool = False,
    ) -> _DigitsVolumeClass:
        """Return the default preview window: identity-only, or the first shell."""

        cardinality = (
            _volume_class_digit_count
            if canonical_variation
            else 2 * _volume_class_digit_count
        )
        max_ordinal = (cardinality - 1) // _volume_class_digit_count
        return _DigitsVolumeClass(
            minimum_address=0,
            cardinality=cardinality,
            canvas_side=_chart_canvas_side_for_max_ordinal(max_ordinal),
        )

    def _materialization_plan(
        self,
        *,
        seed: int,
        index: int,
        resolution_assignment: AxisAssignment,
        materialization_declaration: ArtifactReference,
    ) -> MaterializationPlan:
        return MaterializationPlan(
            id=self._plan_id(seed=seed, index=index),
            benchmark_id=self.materialization.benchmark_id,
            materialization_declaration=materialization_declaration,
            resolution_assignment=resolution_assignment,
            seed=seed,
            latent_factor_declaration=self.materialization.latent_factor_declaration,
        )

    def _sample_resolution_assignment(
        self,
        *,
        sample_count: int,
        seed: int,
        minimum_assignment: AxisAssignment,
        memory_limit_bytes: int | None,
    ) -> AxisAssignment:
        sampling = _resolution_sampling(self.materialization.layout)
        if sampling is None:
            return minimum_assignment
        minimum_width = minimum_assignment.require_axis(sampling.width_axis)
        minimum_height = minimum_assignment.require_axis(sampling.height_axis)
        lattice_steps = self.materialization.resolution_lattice_steps()
        width_step = lattice_steps.get(sampling.width_axis, 1)
        height_step = lattice_steps.get(sampling.height_axis, 1)
        maximum_pixel_count = _batch_sample_pixel_limit(
            memory_limit_bytes=(
                memory_limit_bytes
                if memory_limit_bytes is not None
                else _default_generation_memory_limit_bytes
            ),
            memory_budget_fraction=_default_memory_budget_fraction,
            sample_count=sample_count,
            channel_count=self.formation.channel_count,
        )
        maximum_width, maximum_height = _sampled_resolution_maximum(
            minimum_width=minimum_width,
            minimum_height=minimum_height,
            maximum_pixel_count=maximum_pixel_count,
        )
        if maximum_width < minimum_width:
            raise ObservationGenerationError(
                f"{sampling.width_axis} maximum {maximum_width} is below "
                f"required minimum {minimum_width}"
            )
        if maximum_height < minimum_height:
            raise ObservationGenerationError(
                f"{sampling.height_axis} maximum {maximum_height} is below "
                f"required minimum {minimum_height}"
            )
        minimum_width_multiplier = _minimum_axis_multiplier(
            minimum_width,
            step=width_step,
        )
        minimum_height_multiplier = _minimum_axis_multiplier(
            minimum_height,
            step=height_step,
        )
        maximum_width_multiplier = maximum_width // width_step
        maximum_height_multiplier = maximum_height // height_step
        if maximum_width_multiplier < minimum_width_multiplier:
            maximum_width_multiplier = minimum_width_multiplier
        if maximum_height_multiplier < minimum_height_multiplier:
            maximum_height_multiplier = minimum_height_multiplier
        generator = random.Random(
            f"{seed}:resolution:{sampling.width_axis}:{minimum_width}:"
            f"{maximum_width}:{width_step}:{sampling.height_axis}:{minimum_height}:"
            f"{maximum_height}:{height_step}:"
            f"{maximum_pixel_count}"
        )
        values = dict(minimum_assignment.values)
        values[sampling.width_axis] = (
            generator.randint(minimum_width_multiplier, maximum_width_multiplier)
            * width_step
        )
        values[sampling.height_axis] = (
            generator.randint(minimum_height_multiplier, maximum_height_multiplier)
            * height_step
        )
        return AxisAssignment(values=values)

    def _requested_resolution_assignment(
        self,
        *,
        minimum_assignment: AxisAssignment,
        requested_assignment: AxisAssignment,
    ) -> AxisAssignment:
        values = dict(minimum_assignment.values)
        for axis, requested_value in requested_assignment.values.items():
            minimum_value = minimum_assignment.require_axis(axis)
            if requested_value < minimum_value:
                raise ObservationGenerationError(
                    f"{axis} requested resolution {requested_value} is below "
                    f"required minimum {minimum_value}"
                )
            values[axis] = requested_value
        return AxisAssignment(values=values)

    def _outcome_id(self, component_index: int) -> str:
        if component_index >= len(self.manifest.outcome_space.outcomes):
            raise ObservationGenerationError("component index is outside outcome space")
        return self.manifest.outcome_space.outcomes[component_index].id

    def _latent_coordinates(
        self,
        *,
        component_index: int,
        digit_variant_index: int,
        scaled_factors: tuple[SampleLatentFactor, ...],
        plan: MaterializationPlan,
        variation_values: Mapping[str, object],
    ) -> tuple[Mapping[str, object], ...]:
        records: list[Mapping[str, object]] = []
        for factor in scaled_factors:
            if factor.role == "content":
                values: object = {
                    "digit_index": component_index,
                    "digit_variant_index": digit_variant_index,
                    "outcome_id": self._outcome_id(component_index),
                }
            elif factor.role == "materialization":
                values = dict(plan.resolution_assignment.values)
            else:
                values = dict(variation_values)
            records.append(
                {
                    "name": str(factor.name),
                    "role": factor.role,
                    "degree_measure": factor.degree_measure.to_record(),
                    "multiplicity": factor.multiplicity,
                    "values": values,
                }
            )
        return tuple(records)

    def _plan_id(
        self,
        *,
        seed: int,
        index: int,
    ) -> ProtocolIdentifier:
        return _child_identifier(
            self.manifest.id,
            f"materialization-plans.seed{seed}.sample-{index}",
        )

    def _observation_id(
        self,
        *,
        seed: int,
        index: int,
    ) -> ProtocolIdentifier:
        return _child_identifier(
            self.manifest.id,
            f"observations.seed{seed}.sample-{index}",
        )

    @property
    def id(self) -> ProtocolIdentifier:
        return _generator_id

    @property
    def version(self) -> str:
        return "0.1.0"

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
        memory_limit_bytes: int | None = None,
        resolution_assignment: AxisAssignment | None = None,
        variation_extent: float = 1.0,
        runtime: TensorRuntime | None = None,
        outcome_ids: tuple[str, ...] | None = None,
        timing: TimingCollector | None = None,
        timing_prefix: str = "",
    ) -> GeneratedSampleSet:
        """Generate a shape-aware Digits sample set."""

        if include_artifacts:
            include_fields = True
        sample_shape = _sample_shape(shape)
        sample_count = _sample_count(sample_shape)
        resolved_sample_indices = _sample_indices(
            sample_count=sample_count,
            sample_indices=sample_indices,
        )
        variation_extent_value = _variation_extent_value(variation_extent)
        volume_class = self._default_volume_class(
            canonical_variation=variation_extent_value == 0.0,
        )
        if volume_request is not None:
            if resolution_assignment is not None:
                raise ObservationGenerationError(
                    "volume request cannot be combined with resolution_assignment"
            )
            requested_volume_class = self._volume_class_for_request(
                request=volume_request
            )
            if requested_volume_class is None:
                return GeneratedSampleSet(
                    benchmark_id=self.manifest.id,
                    generator_id=self.id,
                    generator_version=self.version,
                    seed=seed,
                    shape=(0,),
                    variation_extent=variation_extent,
                    volume_request=volume_request,
                    samples=(),
                    request_outcome=_digits_unrealized_request_outcome(
                        request=volume_request,
                        minimum_log2_volume=self.minimum_log2_volume().value,
                    ),
                )
            resolution_assignment = requested_volume_class.resolution_assignment(
                width_axis=self.formation.width_axis,
                height_axis=self.formation.height_axis,
            )
            volume_class = requested_volume_class
        if resolution_assignment is None:
            resolution_assignment = volume_class.resolution_assignment(
                width_axis=self.formation.width_axis,
                height_axis=self.formation.height_axis,
            )
        resolved_resolution_assignment = self._generation_resolution_assignment(
            sample_count=sample_count,
            seed=seed,
            requested_assignment=resolution_assignment,
            memory_limit_bytes=memory_limit_bytes,
        )
        region = _digits_state_space_region(
            volume_class=volume_class,
            margin=self.manifest.resolution_discriminability_margin(),
        )
        fields = None
        targets = None
        if runtime is not None:
            fields, targets = self._generate_tensors(
                sample_shape=sample_shape,
                seed=seed,
                sample_indices=resolved_sample_indices,
                memory_limit_bytes=memory_limit_bytes,
                resolution_assignment=resolved_resolution_assignment,
                volume_class=volume_class,
                runtime=runtime,
                outcome_ids=tuple(outcome_ids or ()),
                timing=timing,
                timing_prefix=timing_prefix,
            )
        samples: tuple[GeneratedSample, ...] = ()
        if include_metadata:
            if runtime is not None:
                samples = self._generate_tensor_metadata_samples(
                    sample_count=sample_count,
                    seed=seed,
                    sample_indices=resolved_sample_indices,
                    volume_class=volume_class,
                    resolution_assignment=resolved_resolution_assignment,
                )
            else:
                samples = self._generate_samples(
                    sample_count=sample_count,
                    seed=seed,
                    sample_indices=resolved_sample_indices,
                    include_fields=include_fields,
                    include_artifacts=include_artifacts,
                    memory_limit_bytes=memory_limit_bytes,
                    resolution_assignment=resolved_resolution_assignment,
                    volume_class=volume_class,
                    timing=timing,
                    timing_prefix=(
                        f"{timing_prefix}formation_batch." if include_fields else timing_prefix
                    ),
                    output_timing_prefix=timing_prefix,
                )
        return GeneratedSampleSet(
            benchmark_id=self.manifest.id,
            generator_id=self.id,
            generator_version=self.version,
            seed=seed,
            shape=sample_shape,
            variation_extent=variation_extent,
            volume_request=volume_request,
            samples=samples,
            fields=fields,
            targets=targets,
            region=region,
        )


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


def _volume_value(log2_volume: float) -> StateSpaceVolumeValue:
    return StateSpaceVolumeValue(
        value=log2_volume,
    )


def _cumulative_setup_count(level: float) -> int:
    """Return the cumulative distinguishable-setup count at a log2-volume level.

    ``N(level) = round(2 ** level)`` from the origin upward,
    so the curriculum window ``[i, i+1]`` realizes the address increment
    ``[N(i), N(i+1))`` and consecutive windows are disjoint.
    """

    if not math.isfinite(level):
        return 0
    if level <= 0.0:
        return 1
    return round(2.0**level)


# Practical representable bound on transform ordinals; the global chart is
# conceptually unbounded, but the region grammar needs a finite, window-stable
# axis domain so increments share an identical axis declaration. Requests beyond
# this resolve as exhausted capacity. ~10^12 setups is effectively unbounded.
_chart_ordinal_domain_extent = 2**40
_chart_continuous_measure_method_id = "digits-continuous-covering-bracket-v1"


def _digits_state_coordinate(*, sample_address: int) -> tuple[int, int]:
    if type(sample_address) is not int or sample_address < 0:
        raise ObservationGenerationError("sample_address must be a nonnegative integer")
    component_index = sample_address % _volume_class_digit_count
    transform_ordinal = sample_address // _volume_class_digit_count
    return (component_index, transform_ordinal)


def _digits_ambient(*, margin: float) -> StateSpaceAmbient:
    """Return the fixed-resolution, unbounded-extent Digits ambient.

    The field domain declares the render pitch, not a concrete canvas: the
    canvas is a per-sample materialization detail, so every realized window
    shares this one ambient and the score filtration stays coherent.
    """

    return StateSpaceAmbient(
        field_domain_kind="lattice-2d",
        field_domain={
            "render_unit_side": _render_unit_side,
            "translation_step_pixels": _translation_step_pixels,
        },
        field_codomain_id="unit-intensity",
        distinguishability=Distinguishability(
            kind="metric-resolution",
            metric_id="l1-field-distance",
            resolution=margin,
            certificate_id="component-discriminability-margin",
        ),
    )


def _digits_state_space_region(
    *,
    volume_class: _DigitsVolumeClass,
    margin: float,
) -> StateSpaceRegion:
    components = tuple(
        _digits_product_region(digit_index=digit_index, transform_ordinal=transform_ordinal)
        for address in range(volume_class.minimum_address, volume_class.maximum_address + 1)
        for digit_index, transform_ordinal in (divmod(address, _volume_class_digit_count)[::-1],)
    )
    measure_estimate = _digits_measure_estimate(volume_class.log2_volume)
    return StateSpaceRegion(
        id=(
            "benchmarks.digits.realized-region."
            f"addresses-{volume_class.minimum_address}-{volume_class.maximum_address}"
        ),
        ambient=_digits_ambient(margin=margin),
        components=components,
        union_rule="disjoint-union",
        volume=volume_class.cardinality,
        log2_volume=volume_class.log2_volume,
        measure_estimate=measure_estimate,
    )


def _digits_capacity_region() -> StateSpaceRegion:
    volume = _volume_class_digit_count * (_chart_ordinal_domain_extent + 1)
    components = tuple(
        _digits_capacity_product_region(digit_index=digit_index)
        for digit_index in range(_volume_class_digit_count)
    )
    log2_volume = math.log2(volume)
    return StateSpaceRegion(
        id="benchmarks.digits.accessible-capacity",
        ambient=_digits_ambient(margin=_component_discriminability_margin),
        components=components,
        union_rule="disjoint-union",
        volume=volume,
        log2_volume=log2_volume,
        measure_estimate=_digits_measure_estimate(log2_volume),
    )


def _digits_product_region(
    *,
    digit_index: int,
    transform_ordinal: int,
) -> ProductRegion:
    axis_regions = _digits_transform_cell_axis_regions(transform_ordinal)
    measure_estimate = _digits_measure_estimate(0.0)
    return ProductRegion(
        axis_regions=axis_regions,
        measure_rule="benchmark-computed-finite-count",
        volume=1,
        log2_volume=0.0,
        stratum_id=f"digit-{digit_index}",
        stratum_target={
            "digit_index": digit_index,
            "outcome_id": f"digit-{digit_index}",
        },
        measure_estimate=measure_estimate,
    )


def _digits_capacity_product_region(*, digit_index: int) -> ProductRegion:
    volume = _chart_ordinal_domain_extent + 1
    log2_volume = math.log2(volume)
    return ProductRegion(
        axis_regions=tuple(
            ContinuousAxisRegion(
                axis=axis,
                coordinate_region=(
                    float(cast(RealIntervalDomain, axis.domain).lower),
                    float(cast(RealIntervalDomain, axis.domain).upper),
                ),
                measure_estimate=_digits_measure_estimate(log2_volume),
            )
            for axis in _digits_transform_axes()
        ),
        measure_rule="benchmark-computed-finite-count",
        volume=volume,
        log2_volume=log2_volume,
        stratum_id=f"digit-{digit_index}",
        stratum_target={
            "digit_index": digit_index,
            "outcome_id": f"digit-{digit_index}",
        },
        measure_estimate=_digits_measure_estimate(log2_volume),
    )


def _digits_transform_cell_axis_regions(
    transform_ordinal: int,
) -> tuple[ContinuousAxisRegion, ...]:
    intervals = _digits_transform_cell_intervals(transform_ordinal)
    return tuple(
        ContinuousAxisRegion(
            axis=axis,
            coordinate_region=intervals[axis.id],
            measure_estimate=_digits_measure_estimate(0.0),
        )
        for axis in _digits_transform_axes()
    )


def _digits_transform_cell_intervals(
    transform_ordinal: int,
) -> dict[str, tuple[float, float]]:
    tx_step, ty_step, scale_level = _transform_cell_for_ordinal(transform_ordinal)
    scale_step = _render_unit_side * _scale_ratio_per_level
    return {
        "x_translation": _half_open_cell_interval(
            center=tx_step * _translation_step_pixels,
            width=float(_translation_step_pixels),
        ),
        "y_translation": _half_open_cell_interval(
            center=ty_step * _translation_step_pixels,
            width=float(_translation_step_pixels),
        ),
        "scale": _half_open_cell_interval(
            center=_scale_footprint(scale_level),
            width=scale_step,
        ),
    }


def _digits_transform_axes() -> tuple[StateSpaceAxis, ...]:
    translation_reach = float((_chart_ordinal_domain_extent + 1) * _translation_step_pixels)
    scale_step = _render_unit_side * _scale_ratio_per_level
    minimum_scale = _scale_footprint(-_max_scale_level) - 0.5 * scale_step
    maximum_scale = _scale_footprint(_max_scale_level) + 0.5 * scale_step
    return (
        StateSpaceAxis(
            id="x_translation",
            domain=RealIntervalDomain(
                lower=-translation_reach,
                upper=translation_reach,
            ),
        ),
        StateSpaceAxis(
            id="y_translation",
            domain=RealIntervalDomain(
                lower=-translation_reach,
                upper=translation_reach,
            ),
        ),
        StateSpaceAxis(
            id="scale",
            domain=RealIntervalDomain(
                lower=minimum_scale,
                upper=maximum_scale,
            ),
        ),
    )


def _half_open_cell_interval(*, center: float, width: float) -> tuple[float, float]:
    half_width = 0.5 * width
    return (center - half_width, center + half_width)


def _digits_measure_estimate(log2_volume: float) -> MeasureEstimate:
    return MeasureEstimate(
        kind="estimated",
        method_id=_chart_continuous_measure_method_id,
        log2_lower=log2_volume,
        log2_upper=log2_volume,
    )


def _digits_region_component_index(
    *,
    component_index: int,
    transform_ordinal: int,
    volume_class: _DigitsVolumeClass,
) -> int:
    address = transform_ordinal * _volume_class_digit_count + component_index
    if not volume_class.minimum_address <= address <= volume_class.maximum_address:
        raise ObservationGenerationError(
            "digit component is outside the realized region"
        )
    return address - volume_class.minimum_address


def _digits_region_axis_coordinates(
    *,
    seed: int,
    sample_index: int,
    transform_ordinal: int,
    canvas_side: int,
    canonical_transform: bool,
) -> Mapping[str, object]:
    x_translation, y_translation, scale = _digits_sample_transform_coordinates(
        seed=seed,
        sample_index=sample_index,
        transform_ordinal=transform_ordinal,
        canvas_side=canvas_side,
        canonical_transform=canonical_transform,
    )
    return {"x_translation": x_translation, "y_translation": y_translation, "scale": scale}


def _digits_unrealized_request_outcome(
    *,
    request: StateSpaceVolumeRequest,
    minimum_log2_volume: float,
) -> GenerationRequestOutcome:
    _ = minimum_log2_volume
    return GenerationRequestOutcome(kind="unrepresentable-below-minimum")


def _digits_sample_address(
    *,
    seed: int,
    sample_index: int,
    cardinality: int,
    minimum_address: int,
) -> int:
    if type(sample_index) is not int or sample_index < 0:
        raise ObservationGenerationError("sample_index must be a nonnegative integer")
    if type(cardinality) is not int or cardinality < 1:
        raise ObservationGenerationError("cardinality must be positive")
    if type(minimum_address) is not int or minimum_address < 0:
        raise ObservationGenerationError("minimum_address must be a nonnegative integer")
    generator = random.Random(f"digits-state-sample-v1:{seed}:{sample_index}:{cardinality}")
    return minimum_address + generator.randrange(cardinality)


def _digits_sample_addresses(
    *,
    seed: int,
    sample_indices: tuple[int, ...],
    cardinality: int,
    minimum_address: int,
) -> tuple[int, ...]:
    return tuple(
        _digits_sample_address(
            seed=seed,
            sample_index=sample_index,
            cardinality=cardinality,
            minimum_address=minimum_address,
        )
        for sample_index in sample_indices
    )


def _digits_sample_transform_coordinates(
    *,
    seed: int,
    sample_index: int,
    transform_ordinal: int,
    canvas_side: int,
    canonical_transform: bool = False,
) -> tuple[float, float, float]:
    if canonical_transform:
        tx_step, ty_step, scale_level = _transform_cell_for_ordinal(transform_ordinal)
        return (
            tx_step * _translation_step_pixels,
            ty_step * _translation_step_pixels,
            _scale_footprint(scale_level),
        )
    intervals = _digits_transform_cell_intervals(transform_ordinal)
    bounded_intervals = _digits_identity_preserving_intervals(
        intervals=intervals,
        canvas_side=canvas_side,
    )
    generator = random.Random(
        f"digits-transform-coordinate-v1:{seed}:{sample_index}:{transform_ordinal}"
    )
    # Each transform cell is one indistinguishable epsilon-cell, so the
    # within-cell representative is sampled uniformly across the cell's linear
    # geometry on every axis. Scale is linear in footprint (``_scale_footprint``
    # is affine in the level), so it is drawn the same way as translation;
    # cross-cell selection over equal-count cells (``_digits_sample_address``)
    # is what carries the covering measure.
    return (
        _sample_linear_interval(generator, bounded_intervals["x_translation"]),
        _sample_linear_interval(generator, bounded_intervals["y_translation"]),
        _sample_linear_interval(generator, bounded_intervals["scale"]),
    )


def _digits_identity_preserving_intervals(
    *,
    intervals: Mapping[str, tuple[float, float]],
    canvas_side: int,
) -> dict[str, tuple[float, float]]:
    bounds = {
        "x_translation": (-0.15 * canvas_side, 0.15 * canvas_side),
        "y_translation": (-0.15 * canvas_side, 0.15 * canvas_side),
        "scale": (0.76 * canvas_side, 1.14 * canvas_side),
    }
    constrained: dict[str, tuple[float, float]] = {}
    for axis_id, interval in intervals.items():
        lower = max(interval[0], bounds[axis_id][0])
        upper = min(interval[1], bounds[axis_id][1])
        if upper <= lower:
            center = (interval[0] + interval[1]) / 2.0
            constrained[axis_id] = (center, math.nextafter(center, math.inf))
        else:
            constrained[axis_id] = (lower, upper)
    return constrained


def _digits_volume_class_is_canonical(volume_class: _DigitsVolumeClass) -> bool:
    return (
        volume_class.minimum_address == 0
        and volume_class.cardinality == _volume_class_digit_count
    )


def _sample_linear_interval(
    generator: random.Random,
    interval: tuple[float, float],
) -> float:
    lower, upper = interval
    return lower + generator.random() * (upper - lower)


def _variation_extent_value(variation_extent: float) -> float:
    try:
        value = float(variation_extent)
    except (TypeError, ValueError) as error:
        raise ObservationGenerationError("variation_extent must be finite") from error
    if not math.isfinite(value):
        raise ObservationGenerationError("variation_extent must be finite")
    if value < 0.0 or value > 1.0:
        raise ObservationGenerationError("variation_extent must be between 0 and 1")
    return value


def _constructed_variation_coordinate_record(
    *,
    transform: VariationTransformDeclaration,
    component_index: int,
    transform_ordinal: int,
    transform_coordinates: tuple[float, float, float],
    chart: _DigitsChart,
) -> Mapping[str, object]:
    if type(transform_ordinal) is not int or transform_ordinal < 0:
        raise ObservationGenerationError("transform_ordinal must be a nonnegative integer")
    spatial = transform.spatial_affine
    tx_step, ty_step, scale_level = _transform_cell_for_ordinal(transform_ordinal)
    x_coordinate, y_coordinate, scale_coordinate = transform_coordinates
    x_translation = x_coordinate / chart.canvas_side
    y_translation = y_coordinate / chart.canvas_side
    scale = scale_coordinate / chart.canvas_side
    matrix = [
        [scale, 0.0, x_translation],
        [0.0, scale, y_translation],
        [0.0, 0.0, 1.0],
    ]
    return {
        "kind": "field-variation-transform-coordinate",
        "component_index": component_index,
        "spatial_affine": {
            "kind": "spatial-affine-coordinate",
            "coordinate_system": spatial.coordinate_system,
            "matrix": matrix,
        },
        "native_footprint_side": _render_unit_side,
        "transform_ordinal": transform_ordinal,
        "transform_cell": {
            "x_translation_step": tx_step,
            "y_translation_step": ty_step,
            "scale_level": scale_level,
        },
        "normalized_transform": {
            "x_translation": x_translation,
            "y_translation": y_translation,
            "scale": scale,
        },
    }


def _digits_sample_transform_values(
    *,
    seed: int,
    canvas_side: int,
    sample_indices: tuple[int, ...],
    sample_addresses: tuple[int, ...],
    digit_count: int,
    canonical_transform: bool = False,
) -> tuple[float, ...]:
    return tuple(
        coordinate / canvas_side
        for sample_index, sample_address in zip(sample_indices, sample_addresses, strict=True)
        for ordinal in (sample_address // digit_count,)
        for coordinate in _digits_sample_transform_coordinates(
            seed=seed,
            sample_index=sample_index,
            transform_ordinal=ordinal,
            canvas_side=canvas_side,
            canonical_transform=canonical_transform,
        )
    )


def _quadratic_control_points(
    mark: ComponentMark,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    points = mark.control_points
    if len(points) == 3:
        return (points[0], points[1], points[2])
    if len(points) == 2:
        midpoint = (
            (points[0][0] + points[1][0]) / 2.0,
            (points[0][1] + points[1][1]) / 2.0,
        )
        return (points[0], midpoint, points[1])
    raise TensorRuntimeError("Digits marks must be linear or quadratic curves")


def _quadratic_tensor_point(
    controls: Any,
    component_index: Any,
    mark_slot: int,
    coordinate_index: int,
    t: Any,
) -> Any:
    t_by_segment = t.reshape((1, -1))
    one_minus_t = 1.0 - t_by_segment
    start = controls[component_index, mark_slot, 0, coordinate_index].reshape((-1, 1))
    control = controls[component_index, mark_slot, 1, coordinate_index].reshape((-1, 1))
    end = controls[component_index, mark_slot, 2, coordinate_index].reshape((-1, 1))
    return (
        one_minus_t * one_minus_t * start
        + 2.0 * one_minus_t * t_by_segment * control
        + t_by_segment * t_by_segment * end
    )


def _constant_tensor_program(value: float) -> TensorBatchProgram:
    def element_function(coordinates: tuple[Any, ...], *, ops: Any) -> Any:
        return ops.broadcast_zeros(coordinates) + value

    return TensorBatchProgram(
        kernel=element_function,
        parameters={},
        cache_key=("constant", value),
    )


def _target_tensor_program(
    *,
    seed: int,
    sample_indices: tuple[int, ...],
    cardinality: int,
    minimum_address: int,
    component_to_outcome: tuple[int, ...],
    outcome_count: int,
) -> TensorBatchProgram:
    sample_addresses = _digits_sample_addresses(
        seed=seed,
        sample_indices=sample_indices,
        cardinality=cardinality,
        minimum_address=minimum_address,
    )

    def element_function(
        coordinates: tuple[Any, ...],
        *,
        sample_address_values: Any,
        component_to_outcome: Any,
    ) -> Any:
        if len(coordinates) == 1:
            outcome_index = coordinates[0]
            sample_axis_index = outcome_index[0] * 0
        else:
            sample_axis_index, outcome_index = coordinates
        sample_address = sample_address_values[sample_axis_index]
        component_index = sample_address.remainder(len(component_to_outcome))
        target_index = component_to_outcome[component_index]
        return (target_index.reshape((-1, 1)) == outcome_index.reshape((1, -1))).to(
            dtype=target_index.dtype
        )

    return TensorBatchProgram(
        kernel=element_function,
        parameters={
            "sample_address_values": TensorElementParameter(
                dtype="int64",
                shape=(len(sample_addresses),),
                values=sample_addresses,
                dynamic_axes=(0,),
            ),
            "component_to_outcome": TensorElementParameter(
                dtype="int64",
                shape=(len(component_to_outcome),),
                values=component_to_outcome,
            ),
        },
        cache_key=("target", outcome_count),
    )


def _digits_tensor_program(
    *,
    seed: int,
    sample_indices: tuple[int, ...],
    cardinality: int,
    minimum_address: int,
    digit_count: int,
    transform_values: tuple[float, ...],
    component_mark_counts: tuple[int, ...],
    component_mark_parameters: tuple[tuple[tuple[float, ...], ...], ...],
    component_control_points: tuple[tuple[tuple[tuple[float, ...], ...], ...], ...],
    height: int,
    width: int,
) -> TensorBatchProgram:
    component_count = len(component_mark_counts)
    max_mark_count = len(component_mark_parameters[0]) if component_count else 0
    mark_parameter_count = 2
    control_point_count = 3
    control_coordinate_count = 2
    sample_addresses = _digits_sample_addresses(
        seed=seed,
        sample_indices=sample_indices,
        cardinality=cardinality,
        minimum_address=minimum_address,
    )

    def element_function(
        coordinates: tuple[Any, ...],
        *,
        ops: Any,
        sample_address_values: Any,
        transform_values: Any,
        component_mark_counts: Any,
        component_mark_parameters: Any,
        component_control_points: Any,
        segment_start_t: Any,
        image_height: Any,
        image_width: Any,
    ) -> Any:
        sample_axis_index, channel_index, y_index, x_index = coordinates
        sample_address = sample_address_values[sample_axis_index]
        component_index = sample_address.remainder(digit_count)
        x_center = x_index.reshape((1, 1, 1, -1)) + 0.5
        y_center = y_index.reshape((1, 1, -1, 1)) + 0.5
        # Diagonal affine: a digit placed at a centre-relative offset and scale.
        # Rotation and shear are removed, so the off-diagonal terms are zero.
        scale = transform_values[sample_axis_index, 2].reshape((-1, 1))
        x_translation = transform_values[sample_axis_index, 0].reshape((-1, 1))
        y_translation = transform_values[sample_axis_index, 1].reshape((-1, 1))
        m00 = scale
        m02 = x_translation
        m11 = scale
        m12 = y_translation
        # Stroke width tracks the physical digit footprint (canvas-independent),
        # not the canvas-normalized scale, so a digit renders with identical
        # strokes whatever size canvas frames it.
        width_scale = scale * image_width / _render_unit_side
        value = ops.broadcast_zeros(coordinates)
        active_channel = channel_index.reshape((1, -1, 1, 1)) == 0
        mark_count = component_mark_counts[component_index]
        for mark_slot in range(max_mark_count):
            active_mark = active_channel & (
                mark_slot < mark_count.reshape((-1, 1, 1, 1))
            )
            segment_end_t = segment_start_t + (
                1.0 / float(_batch_render_curve_sample_count - 1)
            )
            raw_sx = _quadratic_tensor_point(
                component_control_points,
                component_index,
                mark_slot,
                0,
                segment_start_t,
            )
            raw_sy = _quadratic_tensor_point(
                component_control_points,
                component_index,
                mark_slot,
                1,
                segment_start_t,
            )
            raw_ex = _quadratic_tensor_point(
                component_control_points,
                component_index,
                mark_slot,
                0,
                segment_end_t,
            )
            raw_ey = _quadratic_tensor_point(
                component_control_points,
                component_index,
                mark_slot,
                1,
                segment_end_t,
            )
            sx = (0.5 + m00 * (raw_sx - 0.5) + m02) * image_width
            sy = (0.5 + m11 * (raw_sy - 0.5) + m12) * image_height
            ex = (0.5 + m00 * (raw_ex - 0.5) + m02) * image_width
            ey = (0.5 + m11 * (raw_ey - 0.5) + m12) * image_height
            sx = sx.reshape((sx.shape[0], sx.shape[1], 1, 1))
            sy = sy.reshape((sy.shape[0], sy.shape[1], 1, 1))
            ex = ex.reshape((ex.shape[0], ex.shape[1], 1, 1))
            ey = ey.reshape((ey.shape[0], ey.shape[1], 1, 1))
            dx = ex - sx
            dy = ey - sy
            length_squared = dx * dx + dy * dy
            safe_length_squared = length_squared.where(
                length_squared != 0.0,
                length_squared * 0.0 + 1.0,
            )
            segment_t = ((x_center - sx) * dx + (y_center - sy) * dy) / safe_length_squared
            segment_t = segment_t.clamp(0.0, 1.0)
            closest_x = sx + segment_t * dx
            closest_y = sy + segment_t * dy
            segment_distance_squared = (
                (x_center - closest_x) * (x_center - closest_x)
                + (y_center - closest_y) * (y_center - closest_y)
            )
            point_distance_squared = (
                (x_center - sx) * (x_center - sx)
                + (y_center - sy) * (y_center - sy)
            )
            segment_distance_squared = segment_distance_squared.where(
                length_squared != 0.0,
                point_distance_squared,
            )
            distance_squared = segment_distance_squared.min(dim=1).values.reshape(
                (-1, 1, y_index.shape[0], x_index.shape[0])
            )
            mark_width = component_mark_parameters[component_index, mark_slot, 1].reshape(
                (-1, 1, 1, 1)
            )
            width_scale_field = width_scale.reshape((-1, 1, 1, 1))
            threshold = (width_scale_field * mark_width / 2.0) * (
                width_scale_field * mark_width / 2.0
            )
            contribution = (
                active_mark & (distance_squared <= threshold)
            ) * component_mark_parameters[component_index, mark_slot, 0].reshape(
                (-1, 1, 1, 1)
            )
            value = value.maximum(contribution)
        return value

    return TensorBatchProgram(
        kernel=element_function,
        parameters={
            "sample_address_values": TensorElementParameter(
                dtype="int64",
                shape=(len(sample_addresses),),
                values=sample_addresses,
                dynamic_axes=(0,),
            ),
            "transform_values": TensorElementParameter(
                dtype="float32",
                shape=(len(sample_addresses), 3),
                values=transform_values,
                dynamic_axes=(0,),
            ),
            "component_mark_counts": TensorElementParameter(
                dtype="int64",
                shape=(component_count,),
                values=component_mark_counts,
            ),
            "component_mark_parameters": TensorElementParameter(
                dtype="float32",
                shape=(component_count, max_mark_count, mark_parameter_count),
                values=tuple(
                    value
                    for component in component_mark_parameters
                    for mark in component
                    for value in mark
                ),
            ),
            "component_control_points": TensorElementParameter(
                dtype="float32",
                shape=(
                    component_count,
                    max_mark_count,
                    control_point_count,
                    control_coordinate_count,
                ),
                values=tuple(
                    value
                    for component in component_control_points
                    for mark in component
                    for point in mark
                    for value in point
                ),
            ),
            "segment_start_t": TensorElementParameter(
                dtype="float32",
                shape=(_batch_render_curve_sample_count - 1,),
                values=tuple(
                    index / float(_batch_render_curve_sample_count - 1)
                    for index in range(_batch_render_curve_sample_count - 1)
                ),
            ),
            "image_height": TensorElementParameter(
                dtype="float32",
                shape=(),
                values=(float(height),),
            ),
            "image_width": TensorElementParameter(
                dtype="float32",
                shape=(),
                values=(float(width),),
            ),
        },
        cache_key=(
            "digits-field",
            component_count,
            max_mark_count,
            _batch_render_curve_sample_count,
        ),
    )


def new_inverse_digits_secret(*, byte_count: int = 32) -> bytes:
    """Return a physical-entropy secret for inverse-Digits evaluation batches."""

    if type(byte_count) is not int or byte_count < 16:
        raise ObservationGenerationError("inverse digits secret must have at least 128 bits")
    return os.urandom(byte_count)


def sample_inverse_digits_observations(
    *,
    runtime: TensorRuntime,
    secret: bytes,
    sample_count: int,
    canvas_side: int = _render_unit_side,
    sample_indices: Sequence[int] | None = None,
) -> InverseDigitsObservationBatch:
    """Sample secret-seeded inverse-Digits observations without exposing latents."""

    if len(secret) < 16:
        raise ObservationGenerationError("inverse digits secret must have at least 128 bits")
    if type(sample_count) is not int or sample_count < 1:
        raise ObservationGenerationError("sample_count must be a positive integer")
    if type(canvas_side) is not int or canvas_side < _render_unit_side:
        raise ObservationGenerationError("canvas_side must be at least the render unit side")
    indices = _sample_indices(sample_count=sample_count, sample_indices=sample_indices)
    latents = tuple(
        _inverse_digits_latent_from_secret(secret, sample_index=index)
        for index in indices
    )
    observations = render_inverse_digits(
        runtime=runtime,
        latents=latents,
        canvas_side=canvas_side,
    )
    return InverseDigitsObservationBatch(
        observations=observations,
        latents=latents,
        canvas_side=canvas_side,
        secret_digest=hashlib.sha256(secret).hexdigest(),
    )


def render_inverse_digits(
    *,
    runtime: TensorRuntime,
    latents: Sequence[InverseDigitsLatent],
    canvas_side: int = _render_unit_side,
) -> Any:
    """Render inverse-Digits latents through the differentiable forward law."""

    normalized_latents = tuple(latents)
    if not normalized_latents:
        raise ObservationGenerationError("inverse digits rendering requires latents")
    if type(canvas_side) is not int or canvas_side < _render_unit_side:
        raise ObservationGenerationError("canvas_side must be at least the render unit side")
    backend = runtime.backend
    device = _runtime_device(runtime)
    identities = backend.tensor(
        tuple(latent.identity for latent in normalized_latents),
        dtype=backend.long,
        device=device,
    )
    nuisance = backend.tensor(
        tuple(latent.to_nuisance_tuple() for latent in normalized_latents),
        dtype=backend.float32,
        device=device,
    )
    return render_inverse_digits_tensor(
        runtime,
        identities=identities,
        nuisance=nuisance,
        canvas_side=canvas_side,
    )


def render_inverse_digits_tensor(
    runtime: TensorRuntime,
    *,
    identities: Any,
    nuisance: Any,
    canvas_side: int,
) -> Any:
    """Render inverse-Digits latent tensors through the differentiable law."""

    backend = runtime.backend
    device = _runtime_device(runtime)
    if type(canvas_side) is not int or canvas_side < _render_unit_side:
        raise ObservationGenerationError("canvas_side must be at least the render unit side")
    if len(tuple(identities.shape)) != 1:
        raise ObservationGenerationError("identity tensor must have shape (batch,)")
    if tuple(nuisance.shape) != (int(identities.shape[0]), 5):
        raise ObservationGenerationError("nuisance tensor must have shape (batch, 5)")
    mark_counts, control_points = _inverse_digit_component_tensors(runtime)
    identities = identities.to(device=device, dtype=backend.long)
    nuisance = nuisance.to(device=device, dtype=backend.float32)
    batch_size = int(identities.shape[0])
    x_center = (
        backend.arange(canvas_side, dtype=nuisance.dtype, device=device).reshape(
            1,
            1,
            1,
            canvas_side,
        )
        + 0.5
    ) / float(canvas_side)
    y_center = (
        backend.arange(canvas_side, dtype=nuisance.dtype, device=device).reshape(
            1,
            1,
            canvas_side,
            1,
        )
        + 0.5
    ) / float(canvas_side)
    segment_start_t = backend.linspace(
        0.0,
        1.0 - (1.0 / float(_batch_render_curve_sample_count - 1)),
        _batch_render_curve_sample_count - 1,
        dtype=nuisance.dtype,
        device=device,
    )
    value = backend.zeros(
        (batch_size, 1, canvas_side, canvas_side),
        dtype=nuisance.dtype,
        device=device,
    )
    max_mark_count = int(control_points.shape[1])
    selected_mark_counts = mark_counts[identities]
    x_translation = nuisance[:, 0].reshape((-1, 1))
    y_translation = nuisance[:, 1].reshape((-1, 1))
    scale = nuisance[:, 2].reshape((-1, 1))
    shear = nuisance[:, 3].reshape((-1, 1))
    stroke_width = nuisance[:, 4].reshape((-1, 1, 1, 1))
    base_width = (3.0 / float(canvas_side)) * stroke_width
    for mark_slot in range(max_mark_count):
        active_mark = mark_slot < selected_mark_counts.reshape((-1, 1, 1, 1))
        selected_controls = control_points[identities, mark_slot, :, :].to(
            dtype=nuisance.dtype
        )
        raw_start = _tensor_quadratic_points(selected_controls, segment_start_t)
        segment_end_t = segment_start_t + (1.0 / float(_batch_render_curve_sample_count - 1))
        raw_end = _tensor_quadratic_points(selected_controls, segment_end_t)
        raw_sx = raw_start[:, :, 0]
        raw_sy = raw_start[:, :, 1]
        raw_ex = raw_end[:, :, 0]
        raw_ey = raw_end[:, :, 1]
        sx = 0.5 + scale * (raw_sx - 0.5) + shear * (raw_sy - 0.5) + x_translation
        sy = 0.5 + scale * (raw_sy - 0.5) + y_translation
        ex = 0.5 + scale * (raw_ex - 0.5) + shear * (raw_ey - 0.5) + x_translation
        ey = 0.5 + scale * (raw_ey - 0.5) + y_translation
        sx = sx.reshape((batch_size, -1, 1, 1))
        sy = sy.reshape((batch_size, -1, 1, 1))
        ex = ex.reshape((batch_size, -1, 1, 1))
        ey = ey.reshape((batch_size, -1, 1, 1))
        dx = ex - sx
        dy = ey - sy
        length_squared = dx * dx + dy * dy
        safe_length_squared = length_squared.where(
            length_squared != 0.0,
            length_squared * 0.0 + 1.0,
        )
        segment_t = ((x_center - sx) * dx + (y_center - sy) * dy) / safe_length_squared
        segment_t = segment_t.clamp(0.0, 1.0)
        closest_x = sx + segment_t * dx
        closest_y = sy + segment_t * dy
        distance_squared = (
            (x_center - closest_x) * (x_center - closest_x)
            + (y_center - closest_y) * (y_center - closest_y)
        )
        distance_squared = distance_squared.min(dim=1).values.reshape(
            (batch_size, 1, canvas_side, canvas_side)
        )
        contribution = (-distance_squared / (base_width * base_width + 1.0e-8)).exp()
        value = value.maximum(active_mark * contribution)
    return value.clamp(0.0, 1.0)


def inverse_digits_static_certification(
    *,
    runtime: TensorRuntime,
    recovered_latent: InverseDigitsLatent,
    observation: Any,
    canvas_side: int,
    refinement_sides: Sequence[int] = (_render_unit_side, 2 * _render_unit_side),
) -> StaticMapCertification:
    """Certify latent precision for a recovered inverse-Digits latent."""

    backend = runtime.backend
    device = _runtime_device(runtime)
    identity = backend.tensor([recovered_latent.identity], dtype=backend.long, device=device)
    nuisance = backend.tensor(
        [recovered_latent.to_nuisance_tuple()],
        dtype=backend.float32,
        device=device,
    )
    prediction = render_inverse_digits_tensor(
        runtime,
        identities=identity,
        nuisance=nuisance,
        canvas_side=canvas_side,
    )
    residual_norm = float(
        tensor_value_to_host_values(
            (prediction - observation.reshape(prediction.shape)).pow(2).mean().sqrt()
        )[0]
    )
    ladder_sides = tuple(int(side) for side in refinement_sides)
    sigma_ladder = tuple(
        _inverse_digits_sigma_min(
            runtime=runtime,
            identity=recovered_latent.identity,
            nuisance=recovered_latent.to_nuisance_tuple(),
            canvas_side=side,
        )
        for side in ladder_sides
    )
    sigma_min = sigma_ladder[-1]
    positive_sigmas = tuple(value for value in sigma_ladder if value > _static_sigma_floor)
    stability = (
        math.inf if not positive_sigmas else max(positive_sigmas) / min(positive_sigmas)
    )
    refused = (
        sigma_min <= _static_sigma_floor
        or not math.isfinite(stability)
        or stability > _static_conditioning_refusal_ratio
    )
    certified_epsilon = (
        math.inf if refused else residual_norm / max(sigma_min, _static_sigma_floor)
    )
    return StaticMapCertification(
        certified_epsilon=certified_epsilon,
        residual_norm=residual_norm,
        sigma_min=sigma_min,
        sigma_min_ladder=sigma_ladder,
        conditioning_stability=stability,
        certification_status="refused-conditioning-unstable" if refused else "certified",
    )


def inverse_digits_validated_bits(
    *,
    runtime: TensorRuntime,
    recovered_latent: InverseDigitsLatent,
    certified_epsilon: float,
    canvas_side: int,
) -> InverseDigitsValidatedBits:
    """Return product-latent ambient entropy resolved to certified precision."""

    if not math.isfinite(certified_epsilon) or certified_epsilon <= 0.0:
        return InverseDigitsValidatedBits(
            bits=0.0,
            identity_bits=0.0,
            nuisance_bits=0.0,
            distinguishable_identity_count=1,
            certified_epsilon=float(certified_epsilon),
        )
    identity_count = _distinguishable_identity_count(
        runtime=runtime,
        recovered_latent=recovered_latent,
        certified_epsilon=certified_epsilon,
        canvas_side=canvas_side,
    )
    identity_bits = math.log2(identity_count)
    nuisance_bits = sum(
        max(0.0, math.log2(axis_range / certified_epsilon))
        for axis_range in _inverse_nuisance_axis_ranges.values()
    )
    bits = identity_bits + nuisance_bits
    return InverseDigitsValidatedBits(
        bits=bits,
        identity_bits=identity_bits,
        nuisance_bits=nuisance_bits,
        distinguishable_identity_count=identity_count,
        certified_epsilon=certified_epsilon,
    )


def inverse_digits_reconstruction(
    *,
    runtime: TensorRuntime,
    identity_logits: Any,
    nuisance: Any,
    canvas_side: int,
) -> Any:
    """Render a differentiable identity-mixture reconstruction from inverse outputs."""

    backend = runtime.backend
    device = _runtime_device(runtime)
    if len(tuple(identity_logits.shape)) != 2 or int(identity_logits.shape[1]) != 10:
        raise ObservationGenerationError("identity logits must have shape (batch, 10)")
    if tuple(nuisance.shape) != (int(identity_logits.shape[0]), 5):
        raise ObservationGenerationError("nuisance tensor must have shape (batch, 5)")
    batch_size = int(identity_logits.shape[0])
    identities = backend.arange(10, dtype=backend.long, device=device).repeat(batch_size)
    repeated_nuisance = nuisance.reshape(batch_size, 1, 5).repeat(1, 10, 1).reshape(
        batch_size * 10,
        5,
    )
    renders = render_inverse_digits_tensor(
        runtime,
        identities=identities,
        nuisance=repeated_nuisance,
        canvas_side=canvas_side,
    ).reshape(batch_size, 10, 1, canvas_side, canvas_side)
    weights = backend.softmax(identity_logits, dim=1).reshape(batch_size, 10, 1, 1, 1)
    return (weights * renders).sum(dim=1)


def inverse_digits_reconstruction_loss(
    *,
    runtime: TensorRuntime,
    identity_logits: Any,
    nuisance: Any,
    observations: Any,
    canvas_side: int,
) -> Any:
    """Return the label-free inverse-Digits reconstruction residual."""

    reconstruction = inverse_digits_reconstruction(
        runtime=runtime,
        identity_logits=identity_logits,
        nuisance=nuisance,
        canvas_side=canvas_side,
    )
    return (reconstruction - observations).pow(2).mean()


def inverse_digits_latent_vector_reconstruction_loss(
    *,
    runtime: TensorRuntime,
    latent_vector: Any,
    observations: Any,
    canvas_side: int,
) -> Any:
    """Return reconstruction loss for a submitted inverse latent-vector output."""

    if len(tuple(latent_vector.shape)) != 2 or int(latent_vector.shape[1]) != 15:
        raise ObservationGenerationError("inverse latent vector must have shape (batch, 15)")
    return inverse_digits_reconstruction_loss(
        runtime=runtime,
        identity_logits=latent_vector[:, :10],
        nuisance=latent_vector[:, 10:],
        observations=observations,
        canvas_side=canvas_side,
    )


def _inverse_digits_ambient_certified_bits(request: Any) -> Any:
    runtime = request.runtime
    predictions = request.predictions
    targets = request.targets
    if len(tuple(predictions.shape)) != 2 or int(predictions.shape[1]) != 15:
        raise ValueError("inverse Digits predictions must have shape (batch, 15)")
    if len(tuple(targets.shape)) != 4 or int(targets.shape[1]) != 1:
        raise ValueError("inverse Digits targets must have shape (batch, 1, side, side)")
    side = int(targets.shape[-1])
    if int(targets.shape[-2]) != side:
        raise ValueError("inverse Digits targets must be square images")

    values: list[float] = []
    diagnostics: list[Mapping[str, object]] = []
    detached_predictions = predictions.detach()
    for sample_index in range(int(predictions.shape[0])):
        row = detached_predictions[sample_index]
        identity = int(tensor_value_to_host_values(row[:10].argmax())[0])
        nuisance_values = tuple(float(value) for value in tensor_value_to_host_values(row[10:]))
        try:
            recovered_latent = InverseDigitsLatent(identity, *nuisance_values)
            certification = inverse_digits_static_certification(
                runtime=runtime,
                recovered_latent=recovered_latent,
                observation=targets[sample_index : sample_index + 1],
                canvas_side=side,
                refinement_sides=(side, max(side + 1, side * 2)),
            )
            if certification.certification_status != "certified":
                values.append(0.0)
                diagnostics.append(
                    {
                        "kind": "digits-inverse-certified-diagnostics",
                        "sample_index": sample_index,
                        "certification_status": certification.certification_status,
                        "bits": 0.0,
                        "residual_norm": certification.residual_norm,
                        "sigma_min": certification.sigma_min,
                    }
                )
                continue
            bits = inverse_digits_validated_bits(
                runtime=runtime,
                recovered_latent=recovered_latent,
                certified_epsilon=certification.certified_epsilon,
                canvas_side=side,
            )
            values.append(bits.bits)
            diagnostics.append(
                {
                    "kind": "digits-inverse-certified-diagnostics",
                    "sample_index": sample_index,
                    "certification_status": certification.certification_status,
                    "bits": bits.bits,
                    "identity_bits": bits.identity_bits,
                    "nuisance_bits": bits.nuisance_bits,
                    "distinguishable_identity_count": bits.distinguishable_identity_count,
                    "certified_epsilon": certification.certified_epsilon,
                    "residual_norm": certification.residual_norm,
                    "sigma_min": certification.sigma_min,
                    "conditioning_stability": certification.conditioning_stability,
                }
            )
        except (ObservationGenerationError, ValueError):
            values.append(0.0)
            diagnostics.append(
                {
                    "kind": "digits-inverse-certified-diagnostics",
                    "sample_index": sample_index,
                    "certification_status": "refused-invalid-latent",
                    "bits": 0.0,
                }
            )
    result = predictions.new_tensor(values)
    result.leibniz_competence_diagnostics = tuple(diagnostics)
    return result


def _distinguishable_identity_count(
    *,
    runtime: TensorRuntime,
    recovered_latent: InverseDigitsLatent,
    certified_epsilon: float,
    canvas_side: int,
) -> int:
    backend = runtime.backend
    device = _runtime_device(runtime)
    identities = backend.arange(_volume_class_digit_count, dtype=backend.long, device=device)
    nuisance = backend.tensor(
        [recovered_latent.to_nuisance_tuple()] * _volume_class_digit_count,
        dtype=backend.float32,
        device=device,
    )
    renders = render_inverse_digits_tensor(
        runtime,
        identities=identities,
        nuisance=nuisance,
        canvas_side=canvas_side,
    )
    recovered = renders[recovered_latent.identity].reshape(1, -1)
    distances = (renders.reshape(_volume_class_digit_count, -1) - recovered).pow(2).mean(
        dim=1
    ).sqrt()
    separated = int(tensor_value_to_host_values((distances > float(certified_epsilon)).sum())[0])
    return max(1, separated + 1)


def _inverse_digits_sigma_min(
    *,
    runtime: TensorRuntime,
    identity: int,
    nuisance: tuple[float, float, float, float, float],
    canvas_side: int,
) -> float:
    backend = runtime.backend
    device = _runtime_device(runtime)
    identity_tensor = backend.tensor([identity], dtype=backend.long, device=device)
    nuisance_tensor = backend.tensor(
        nuisance,
        dtype=backend.float32,
        device=device,
        requires_grad=True,
    )

    def render_flat(nuisance_value: Any) -> Any:
        rendered = render_inverse_digits_tensor(
            runtime,
            identities=identity_tensor,
            nuisance=nuisance_value.reshape(1, 5),
            canvas_side=canvas_side,
        )
        return rendered.reshape(-1) / math.sqrt(float(rendered.numel()))

    basis = backend.eye(5, dtype=nuisance_tensor.dtype, device=device)
    columns: list[Any] = []
    for axis in range(5):
        _value, jvp = backend.autograd.functional.jvp(
            render_flat,
            nuisance_tensor,
            basis[axis],
            create_graph=False,
            strict=True,
        )
        columns.append(jvp)
    jacobian_columns = backend.stack(columns, dim=1)
    gram = jacobian_columns.transpose(0, 1).matmul(jacobian_columns)
    eigenvalues = backend.linalg.eigvalsh(gram)
    sigma_min = eigenvalues.clamp_min(0.0).sqrt().min()
    return float(tensor_value_to_host_values(sigma_min)[0])


def _inverse_digits_latent_from_secret(
    secret: bytes,
    *,
    sample_index: int,
) -> InverseDigitsLatent:
    digest = hashlib.sha256(
        b"leibniz-digits-inverse-latent-v1:" + secret + sample_index.to_bytes(8, "big")
    ).digest()
    identity = digest[0] % _volume_class_digit_count
    uniforms = tuple(_digest_unit_interval(digest, offset) for offset in range(1, 6))
    return InverseDigitsLatent(
        identity=identity,
        x_translation=_affine_interval(uniforms[0], -0.16, 0.16),
        y_translation=_affine_interval(uniforms[1], -0.16, 0.16),
        scale=_affine_interval(uniforms[2], 0.72, 1.18),
        shear=_affine_interval(uniforms[3], -0.18, 0.18),
        stroke_width=_affine_interval(uniforms[4], 0.75, 1.35),
    )


def _digest_unit_interval(digest: bytes, offset: int) -> float:
    start = 1 + offset * 4
    return int.from_bytes(digest[start : start + 4], "big") / float(2**32)


def _affine_interval(value: float, lower: float, upper: float) -> float:
    return lower + value * (upper - lower)


def _inverse_digit_component_tensors(runtime: TensorRuntime) -> tuple[Any, Any]:
    backend = runtime.backend
    device = _runtime_device(runtime)
    max_mark_count = max(len(strokes) for strokes in _digit_strokes)
    mark_counts: list[int] = []
    component_control_points: list[list[list[list[float]]]] = []
    for strokes in _digit_strokes:
        control_points: list[list[list[float]]] = []
        for points in strokes:
            controls = _curve_control_points(points)
            control_points.append([[point[0], point[1]] for point in controls])
        mark_counts.append(len(control_points))
        while len(control_points) < max_mark_count:
            control_points.append([[0.0, 0.0] for _ in range(3)])
        component_control_points.append(control_points)
    return (
        backend.tensor(mark_counts, dtype=backend.long, device=device),
        backend.tensor(component_control_points, dtype=backend.float32, device=device),
    )


def _tensor_quadratic_points(controls: Any, t: Any) -> Any:
    t_values = t.reshape((1, -1, 1))
    one_minus_t = 1.0 - t_values
    return (
        one_minus_t * one_minus_t * controls[:, 0:1, :]
        + 2.0 * one_minus_t * t_values * controls[:, 1:2, :]
        + t_values * t_values * controls[:, 2:3, :]
    )


def _curve_control_points(
    points: _CurvePoints,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    if len(points) == 3:
        return (points[0], points[1], points[2])
    if len(points) == 2:
        midpoint = (
            (points[0][0] + points[1][0]) / 2.0,
            (points[0][1] + points[1][1]) / 2.0,
        )
        return (points[0], midpoint, points[1])
    raise TensorRuntimeError("Digits inverse renderer requires linear or quadratic curves")


def _resolution_sampling(layout: Mapping[str, object] | None) -> _ResolutionSampling | None:
    if layout is None or "resolution_sampling" not in layout:
        return None
    value = layout["resolution_sampling"]
    if not isinstance(value, Mapping):
        raise ObservationGenerationError("resolution_sampling must be a record")
    value = cast(Mapping[str, object], value)
    kind = value.get("kind")
    if kind != "uniform-integer-rectangle":
        raise ObservationGenerationError(f"unsupported resolution_sampling kind: {kind}")
    width_axis = value.get("width_axis")
    if not isinstance(width_axis, str) or not width_axis:
        raise ObservationGenerationError("resolution_sampling width_axis must be nonempty")
    height_axis = value.get("height_axis")
    if not isinstance(height_axis, str) or not height_axis:
        raise ObservationGenerationError("resolution_sampling height_axis must be nonempty")
    return _ResolutionSampling(
        width_axis=width_axis,
        height_axis=height_axis,
    )


def _require_generation_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ObservationGenerationError(f"{name} must be a positive integer")


def _sampled_resolution_maximum(
    *,
    minimum_width: int,
    minimum_height: int,
    maximum_pixel_count: int,
) -> tuple[int, int]:
    minimum_pixel_count = minimum_width * minimum_height
    if maximum_pixel_count < minimum_pixel_count:
        return (minimum_width, minimum_height)
    side_multiplier = math.sqrt(maximum_pixel_count / minimum_pixel_count)
    return (
        math.floor(minimum_width * side_multiplier),
        math.floor(minimum_height * side_multiplier),
    )


def _minimum_axis_multiplier(value: int, *, step: int) -> int:
    return max(1, math.ceil(value / step))


def _batch_sample_pixel_limit(
    *,
    memory_limit_bytes: int,
    memory_budget_fraction: float,
    sample_count: int,
    channel_count: int,
) -> int:
    if type(memory_limit_bytes) is not int or memory_limit_bytes < 1:
        raise ObservationGenerationError("memory_limit_bytes must be a positive integer")
    if not math.isfinite(memory_budget_fraction) or memory_budget_fraction <= 0.0:
        raise ObservationGenerationError("memory_budget_fraction must be positive")
    if memory_budget_fraction > 1.0:
        raise ObservationGenerationError("memory_budget_fraction must not exceed 1")
    if sample_count < 1:
        raise ObservationGenerationError("sample_count must be positive")
    if channel_count < 1:
        raise ObservationGenerationError("channel_count must be positive")
    budget = math.floor(memory_limit_bytes * memory_budget_fraction)
    per_sample_denominator = sample_count * channel_count * _field_scalar_construction_bytes
    return max(1, budget // per_sample_denominator)


def _child_identifier(parent: ProtocolIdentifier, suffix: str) -> ProtocolIdentifier:
    return ProtocolIdentifier(
        name=ProtocolName.parse(f"{parent.name}.{suffix}"),
        version=parent.version,
    )


def _timing_span(
    timing: TimingCollector | None,
    phase: str,
    *,
    samples: int = 0,
) -> AbstractContextManager[None]:
    if timing is None:
        return nullcontext()
    return timing.span(phase, samples=samples)


def _field_to_png_bytes(field: FieldObservation) -> bytes:
    channels, height, width = field.shape
    if channels != 1:
        raise ObservationGenerationError("PNG encoding currently requires one channel")
    rows: list[bytes] = []
    for y_index in range(height):
        offset = y_index * width
        row = bytes(_uint8(field.values[offset + x_index]) for x_index in range(width))
        rows.append(b"\x00" + row)
    payload = b"".join(rows)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(payload))
        + _png_chunk(b"IEND", b"")
    )


def _field_to_png_data_url(field: FieldObservation) -> str:
    encoded = base64.b64encode(_field_to_png_bytes(field)).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum)
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _uint8(value: float) -> int:
    return max(0, min(255, round(float(value) * 255)))


def _require_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise TensorRuntimeError(f"{name} must be a positive integer")


def _manifest() -> BenchmarkManifest:
    return BenchmarkManifest(
        id=_benchmark_id,
        name=ProtocolName.parse("benchmarks.digits"),
        outcome_space=OutcomeSpace(
            id=_outcome_space_id,
            outcomes=tuple(Outcome(id=f"digit-{digit}") for digit in range(10)),
        ),
        latent_factor_declaration=_latent_factor_reference(),
        resolution_analysis={
            "kind": "component-discriminability-margin",
            "discriminability_margin": _component_discriminability_margin,
            "render_unit_side": _render_unit_side,
            "translation_step_pixels": _translation_step_pixels,
            "scale_ratio_per_level": _scale_ratio_per_level,
            "volume_value": {
                "kind": "domain-growth-setup-window",
                "measure_id": "log2-state-space-volume",
                "formula": "log2(realized_cardinality)",
                "digit_count": _volume_class_digit_count,
                "transform_axes": list(_chart_axis_ids),
                "target_policy": "contiguous-global-address-increment",
                "description": (
                    "Score-bearing Digits volume windows are finite increments "
                    "of one global address walk over digit identity and a "
                    "shell-ordered translation/scale transform lattice. The "
                    "output task remains 10-way classification; volume counts "
                    "problem setups. Canvas size is materialization metadata "
                    "derived from the deepest realized transform ordinal."
                ),
            },
            "description": (
                "Native-footprint component separation at the fixed render pitch."
            ),
        },
    )


def _latent_factors() -> LatentFactorDeclaration:
    return LatentFactorDeclaration(
        id=_latent_factor_id,
        construction_factors=(
            GeneratorConstructionFactor(
                name=ProtocolName.parse("benchmarks.digits.construction.stroke-basis"),
                degree_measure=DegreeMeasure.constant_count(7),
                description="Fixed digit construction basis used by this declaration.",
            ),
        ),
        sample_factors=(
            SampleLatentFactor(
                name=ProtocolName.parse("benchmarks.digits.sample.digit-identity"),
                role="content",
                degree_measure=DegreeMeasure.discrete_choice(10),
            ),
            SampleLatentFactor(
                name=ProtocolName.parse(
                    "benchmarks.digits.sample.field-variation-transform"
                ),
                role="variation",
                degree_measure=DegreeMeasure.vector_dimension(3),
            ),
            SampleLatentFactor(
                name=ProtocolName.parse("benchmarks.digits.materialization.canvas-shape"),
                role="materialization",
                degree_measure=DegreeMeasure.vector_dimension(2),
            ),
        ),
    )


def _materialization() -> MaterializationDeclaration:
    return MaterializationDeclaration(
        id=_materialization_id,
        benchmark_id=_benchmark_id,
        latent_factor_declaration=_latent_factor_reference(),
        requirements=(),
        layout={
            "kind": "sequence-layout",
            "sequence_axis": "L",
            "width_axis": "W",
            "height_axis": "H",
            "sequence_spacing": "left-to-right",
            "placement_axis": "x",
            "resolution_sampling": {
                "kind": "uniform-integer-rectangle",
                "width_axis": "W",
                "height_axis": "H",
                "description": (
                    "Batch canvas area is sampled up to the active runtime memory budget."
                ),
            },
        },
    )


def _formation() -> ObservationFormationDeclaration:
    return ObservationFormationDeclaration(
        id=_formation_id,
        benchmark_id=_benchmark_id,
        interpreter="field-mark-composition@0.1.0",
        channel_count=1,
        width_axis="W",
        height_axis="H",
        sequence_layout=SequenceLayout(
            sequence_axis="L",
            width_axis="W",
            height_axis="H",
            placement_axis="x",
        ),
        variation_transform=VariationTransformDeclaration(
            kind="field-variation-transform",
            spatial_affine=SpatialAffineVariation(
                kind="spatial-affine",
                coordinate_system="normalized-sequence-element",
                spatial_rank=2,
                matrix=(
                    ((0.76, 1.14), (-0.07, 0.07), (-0.15, 0.15)),
                    ((-0.07, 0.07), (0.76, 1.14), (-0.15, 0.15)),
                    ((0.0, 0.0), (0.0, 0.0), (1.0, 1.0)),
                ),
            ),
        ),
        components=_digit_components(),
    )


def _showcase() -> ObservationShowcaseManifest:
    return ObservationShowcaseManifest(
        id=ProtocolIdentifier.parse("benchmarks.digits.inspection-showcase@0.1.0"),
        benchmark_id=_benchmark_id,
        formation_declaration=ArtifactReference(
            kind="observation-formation-declaration",
            protocol_id=_formation_id,
        ),
        materialization_declaration=ArtifactReference(
            kind="materialization-declaration",
            protocol_id=_materialization_id,
        ),
        samples=(
            ObservationShowcaseSample(
                id=ProtocolIdentifier.parse(
                    "benchmarks.digits.inspection-samples.digit-7@0.1.0"
                ),
                label="Single digit 7",
                sample_index=0,
                seed=101,
                component_index=7,
                outcome_id="digit-7",
            ),
            ObservationShowcaseSample(
                id=ProtocolIdentifier.parse(
                    "benchmarks.digits.inspection-samples.digit-3@0.1.0"
                ),
                label="Single digit 3",
                sample_index=1,
                seed=101,
                component_index=3,
                outcome_id="digit-3",
            ),
        ),
    )


def _latent_factor_reference() -> ArtifactReference:
    return ArtifactReference(
        kind="latent-factor-declaration",
        protocol_id=_latent_factor_id,
    )


def _digit_components() -> tuple[ObservationComponent, ...]:
    return tuple(
        ObservationComponent(
            id=f"digit-{digit}",
            marks=tuple(_curve(points) for points in curves),
        )
        for digit, curves in enumerate(_digit_strokes)
    )


def _curve(points: _CurvePoints) -> ComponentMark:
    return ComponentMark(
        kind="bezier-curve",
        channel=0,
        degree=len(points) - 1,
        control_points=points,
        width=3.0,
    )
