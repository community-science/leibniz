"""Digits benchmark implementation entry point."""

from __future__ import annotations

from pathlib import Path

from leibniz.artifacts import ArtifactReference
from leibniz.benchmark_implementations import BenchmarkImplementation
from leibniz.benchmarks import BenchmarkManifest
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.latent_factors import (
    DegreeMeasure,
    GeneratorConstructionFactor,
    LatentFactorDeclaration,
    SampleLatentFactor,
)
from leibniz.materialization import MaterializationDeclaration
from leibniz.observation_formation import (
    ComponentMark,
    ObservationComponent,
    ObservationFormationDeclaration,
    SequenceLayout,
    SpatialAffineVariation,
    VariationTransformDeclaration,
)
from leibniz.outcomes import Outcome, OutcomeSpace

__all__ = ["benchmark"]

_benchmark_id = ProtocolIdentifier.parse("benchmarks.digits@0.1.0")
_latent_factor_id = ProtocolIdentifier.parse("benchmarks.digits.latent-factors@0.1.0")
_materialization_id = ProtocolIdentifier.parse("benchmarks.digits.materialization@0.1.0")
_formation_id = ProtocolIdentifier.parse("benchmarks.digits.observation-formation@0.1.0")
_outcome_space_id = ProtocolIdentifier.parse("benchmarks.digits.outcomes@0.1.0")


def benchmark(root: Path) -> BenchmarkImplementation:
    """Return the Digits benchmark implementation."""

    return DigitsBenchmarkImplementation(root=root)


class DigitsBenchmarkImplementation:
    """Executable Digits benchmark declaration."""

    def __init__(self, *, root: Path) -> None:
        self._root = root
        self._latent_factors = _latent_factors()
        self._benchmark_manifest = _benchmark_manifest()
        self._materialization = _materialization()
        self._formation = _formation()

    @property
    def root(self) -> Path:
        return self._root

    @property
    def benchmark_manifest(self) -> BenchmarkManifest:
        return self._benchmark_manifest

    @property
    def latent_factors(self) -> LatentFactorDeclaration:
        return self._latent_factors

    @property
    def materialization(self) -> MaterializationDeclaration:
        return self._materialization

    @property
    def formation(self) -> ObservationFormationDeclaration:
        return self._formation

    def observation_generator(self) -> object:
        from leibniz.observation_generation import ObservationGenerator

        return ObservationGenerator(
            benchmark_manifest=self.benchmark_manifest,
            latent_factors=self.latent_factors,
            materialization=self.materialization,
            formation=self.formation,
        )


def _benchmark_manifest() -> BenchmarkManifest:
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
            "discriminability_margin": 20.0,
            "affine_minimum_absolute_determinant": 0.25,
            "affine_minimum_axis_alignment": 0.95,
            "affine_minimum_cell_overlap_ratio": 0.55,
            "affine_minimum_singular_value": 0.72,
            "affine_maximum_singular_value": 1.28,
            "affine_maximum_condition_number": 1.6,
            "affine_minimum_projected_extent": 0.65,
            "affine_maximum_projected_extent": 1.35,
            "description": (
                "Minimum rendered component separation required when choosing live "
                "observation resolution."
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
                degree_measure=DegreeMeasure.vector_dimension(6),
            ),
            SampleLatentFactor(
                name=ProtocolName.parse("benchmarks.digits.materialization.canvas-shape"),
                role="materialization",
                degree_measure=DegreeMeasure.vector_dimension(2),
            ),
        ),
        complexity_projections=(),
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
            "resolution_floor": {"W": 24, "H": 24},
            "resolution_lattice": {
                "kind": "axis-multiple",
                "steps": {"W": 24, "H": 24},
                "description": (
                    "Score-bearing sampled canvases lie on independent "
                    "base-resolution multiples for each spatial axis."
                ),
            },
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
        components=(
            _component(
                "digit-0",
                (
                    _curve(((0.5, 0.2768), (0.3056, 0.2912), (0.2984, 0.5))),
                    _curve(((0.2984, 0.5), (0.3056, 0.7088), (0.5, 0.7232))),
                    _curve(((0.5, 0.7232), (0.6944, 0.7088), (0.7016, 0.5))),
                    _curve(((0.7016, 0.5), (0.6944, 0.2912), (0.5, 0.2768))),
                ),
            ),
            _component(
                "digit-1",
                (
                    _curve(((0.4208, 0.3704), (0.5072, 0.2912), (0.5648, 0.2768))),
                    _curve(((0.5648, 0.2768), (0.5576, 0.7016))),
                    _curve(((0.4424, 0.7088), (0.6512, 0.7088))),
                ),
            ),
            _component(
                "digit-2",
                (
                    _curve(((0.3272, 0.3488), (0.428, 0.2552), (0.5792, 0.2912))),
                    _curve(((0.5792, 0.2912), (0.7448, 0.3416), (0.6224, 0.4712))),
                    _curve(((0.6224, 0.4712), (0.5144, 0.572), (0.3488, 0.6872))),
                    _curve(((0.3488, 0.6872), (0.68, 0.6944))),
                ),
            ),
            _component(
                "digit-3",
                (
                    _curve(((0.3344, 0.32), (0.5432, 0.2408), (0.6584, 0.3704))),
                    _curve(((0.6584, 0.3704), (0.716, 0.4784), (0.5144, 0.4928))),
                    _curve(((0.5144, 0.4928), (0.7304, 0.5432), (0.6512, 0.6584))),
                    _curve(((0.6512, 0.6584), (0.5072, 0.7808), (0.32, 0.6728))),
                ),
            ),
            _component(
                "digit-4",
                (
                    _curve(((0.6224, 0.284), (0.3488, 0.5288))),
                    _curve(((0.3488, 0.5288), (0.6656, 0.5288))),
                    _curve(((0.6224, 0.284), (0.6224, 0.7088))),
                ),
            ),
            _component(
                "digit-5",
                (
                    _curve(((0.6584, 0.2912), (0.3632, 0.2912))),
                    _curve(((0.3632, 0.2912), (0.32, 0.4352), (0.4064, 0.4928))),
                    _curve(((0.4064, 0.4928), (0.6584, 0.4352), (0.6728, 0.6152))),
                    _curve(((0.6728, 0.6152), (0.5792, 0.7592), (0.3416, 0.68))),
                ),
            ),
            _component(
                "digit-6",
                (
                    _curve(((0.6368, 0.3056), (0.3776, 0.32), (0.3272, 0.5648))),
                    _curve(((0.3272, 0.5648), (0.356, 0.752), (0.5288, 0.7232))),
                    _curve(((0.5288, 0.7232), (0.7088, 0.68), (0.6512, 0.536))),
                    _curve(((0.6512, 0.536), (0.5288, 0.428), (0.356, 0.5216))),
                ),
            ),
            _component(
                "digit-7",
                (
                    _curve(((0.3344, 0.2912), (0.68, 0.2912))),
                    _curve(((0.68, 0.2912), (0.5432, 0.4928), (0.4712, 0.7088))),
                ),
            ),
            _component(
                "digit-8",
                (
                    _curve(((0.5, 0.4928), (0.3416, 0.4352), (0.3848, 0.3272))),
                    _curve(((0.3848, 0.3272), (0.5072, 0.2264), (0.6296, 0.3272))),
                    _curve(((0.6296, 0.3272), (0.6728, 0.4424), (0.5, 0.4928))),
                    _curve(((0.5, 0.4928), (0.3056, 0.5576), (0.3632, 0.6728))),
                    _curve(((0.3632, 0.6728), (0.5, 0.7808), (0.644, 0.6728))),
                    _curve(((0.644, 0.6728), (0.7016, 0.5576), (0.5, 0.4928))),
                ),
            ),
            _component(
                "digit-9",
                (
                    _curve(((0.644, 0.4712), (0.5288, 0.572), (0.3632, 0.4928))),
                    _curve(((0.3632, 0.4928), (0.3128, 0.3272), (0.4856, 0.2768))),
                    _curve(((0.4856, 0.2768), (0.6728, 0.2912), (0.68, 0.4784))),
                    _curve(((0.68, 0.4784), (0.6512, 0.6584), (0.4208, 0.7088))),
                ),
            ),
        ),
    )


def _latent_factor_reference() -> ArtifactReference:
    return ArtifactReference(
        kind="latent-factor-declaration",
        protocol_id=_latent_factor_id,
    )


def _component(id: str, marks: tuple[ComponentMark, ...]) -> ObservationComponent:
    return ObservationComponent(id=id, marks=marks)


def _curve(points: tuple[tuple[float, float], ...]) -> ComponentMark:
    return ComponentMark(
        kind="bezier-curve",
        channel=0,
        degree=len(points) - 1,
        control_points=points,
        width=3.0,
    )
