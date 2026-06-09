from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import cast

import pytest
from benchmark_typing import load_digits_benchmark

from leibniz.artifacts import ArtifactReference
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import AxisAssignment, MaterializationPlan, MaterializationPlanDocument
from leibniz.observation_formation import (
    FieldObservation,
    ObservationFormationDeclaration,
    ObservationFormationDeclarationDocument,
    ObservationFormationValidationError,
    SpatialAffineVariation,
    VariationTransformDeclaration,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_digits_fixture_root = _repository_root / "tests" / "fixtures" / "digits"
_AffineMatrix2D = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def test_digits_observation_formation_declaration_loads_source_artifact() -> None:
    declaration = load_digits_benchmark(_digits_benchmark_root).formation

    assert declaration.id == ProtocolIdentifier.parse(
        "benchmarks.digits.observation-formation@0.1.0"
    )
    assert declaration.benchmark_id == ProtocolIdentifier.parse("benchmarks.digits@0.1.0")
    assert declaration.channel_count == 1
    assert declaration.width_axis == "W"
    assert declaration.height_axis == "H"
    assert declaration.sequence_layout.sequence_axis == "L"
    assert (
        declaration.variation_transform.spatial_affine.coordinate_system
        == "normalized-sequence-element"
    )
    assert [component.id for component in declaration.components] == [
        f"digit-{digit}" for digit in range(10)
    ]


def test_digits_spatial_variation_bounds_leave_canvas_margin() -> None:
    declaration = _digits_declaration()
    plan = MaterializationPlan(
        id=ProtocolIdentifier.parse("benchmarks.digits.materialization-plan.l1-test@0.1.0"),
        benchmark_id=ProtocolIdentifier.parse("benchmarks.digits@0.1.0"),
        materialization_declaration=ArtifactReference(
            kind="materialization-declaration",
            protocol_id=ProtocolIdentifier.parse("benchmarks.digits.materialization@0.1.0"),
        ),
        resolution_assignment=AxisAssignment(values={"W": 32, "H": 32}),
        seed=101,
    )
    matrix_extremes = (
        ((1.3, 0.0, 0.0), (0.0, 1.3, 0.0), (0.0, 0.0, 1.0)),
        ((1.2, 0.2, 0.0), (-0.2, 1.2, 0.0), (0.0, 0.0, 1.0)),
        ((0.8, -0.2, 0.0), (0.2, 0.8, 0.0), (0.0, 0.0, 1.0)),
    )

    for component_index in range(len(declaration.components)):
        for translation_x, translation_y, matrix in product(
            (-0.08, 0.08),
            (-0.08, 0.08),
            matrix_extremes,
        ):
            affine_matrix = (
                (matrix[0][0], matrix[0][1], translation_x),
                (matrix[1][0], matrix[1][1], translation_y),
                matrix[2],
            )
            observation = declaration.component_field(
                width=plan.resolution_assignment.require_axis("W"),
                height=plan.resolution_assignment.require_axis("H"),
                component_index=component_index,
                variation_coordinate=_variation_coordinate(matrix=affine_matrix),
            )
            min_x, max_x, min_y, max_y = _nonzero_bounds(observation)
            assert min_x > 0
            assert max_x < 31
            assert min_y > 0
            assert max_y < 31


def test_digits_observation_formation_is_deterministic_for_materialization_plan() -> None:
    declaration = _digits_declaration()
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l3.json").read_bytes()
    ).plan
    component_index = declaration.sample_component_index(seed=plan.seed, sample_index=0)

    left = declaration.component_field(
        width=plan.resolution_assignment.require_axis("W"),
        height=plan.resolution_assignment.require_axis("H"),
        component_index=component_index,
    )
    right = declaration.component_field(
        width=plan.resolution_assignment.require_axis("W"),
        height=plan.resolution_assignment.require_axis("H"),
        component_index=component_index,
    )

    assert component_index == declaration.sample_component_index(seed=plan.seed, sample_index=0)
    assert left == right
    assert left.shape == (1, 24, 72)
    assert max(left.values) == 1.0
    assert sum(1 for value in left.values if value > 0) > 0
    assert left.to_record()["shape"] == [1, 24, 72]
    assert str(left.digest)


def test_digits_observation_formation_uses_sampled_canvas_extent() -> None:
    declaration = _digits_declaration()
    plan = MaterializationPlan(
        id=ProtocolIdentifier.parse("benchmarks.digits.materialization-plan.large-canvas@0.1.0"),
        benchmark_id=ProtocolIdentifier.parse("benchmarks.digits@0.1.0"),
        materialization_declaration=ArtifactReference(
            kind="materialization-declaration",
            protocol_id=ProtocolIdentifier.parse("benchmarks.digits.materialization@0.1.0"),
        ),
        resolution_assignment=AxisAssignment(values={"W": 128, "H": 64}),
        seed=101,
    )

    observation = declaration.component_field(
        width=plan.resolution_assignment.require_axis("W"),
        height=plan.resolution_assignment.require_axis("H"),
        component_index=1,
    )
    min_x, max_x, min_y, max_y = _nonzero_bounds(observation)

    assert min_x > 0
    assert max_x < 127
    assert min_y > 0
    assert max_y < 63


def test_observation_formation_rejects_invalid_component_index() -> None:
    declaration = _digits_declaration()
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l3.json").read_bytes()
    ).plan

    assert (
        str(
            capture_observation_error(
                lambda: declaration.component_field(
                    width=plan.resolution_assignment.require_axis("W"),
                    height=plan.resolution_assignment.require_axis("H"),
                    component_index=len(declaration.components),
                )
            )
        )
        == "component_index is outside component vocabulary"
    )


def test_observation_formation_rejects_unknown_fields() -> None:
    record = _minimal_declaration_record()
    record["unsupported_field"] = record["sequence_layout"]

    assert (
        str(capture_observation_error(lambda: ObservationFormationDeclaration.from_record(record)))
        == "unsupported_field: unknown field"
    )


def test_observation_formation_preserves_explicit_zero_mark_values() -> None:
    record = _minimal_declaration_record()
    record["id"] = "benchmarks.synthetic-masks.observation-formation@0.1.0"
    record["benchmark_id"] = "benchmarks.synthetic-masks@0.1.0"
    components = cast(list[dict[str, object]], record["components"])
    marks = cast(list[dict[str, object]], components[0]["marks"])
    marks[0]["value"] = 0
    declaration = ObservationFormationDeclaration.from_record(record)

    assert declaration.components[0].marks[0].value == 0.0
    assert declaration.components[0].marks[0].to_record()["value"] == 0.0
    assert declaration.variation_transform == VariationTransformDeclaration.identity()


def test_variation_identity_coordinates_preserve_observation_field() -> None:
    declaration = _digits_declaration()
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l3.json").read_bytes()
    ).plan
    component_index = declaration.sample_component_index(seed=plan.seed, sample_index=0)

    untransformed = declaration.component_field(
        width=plan.resolution_assignment.require_axis("W"),
        height=plan.resolution_assignment.require_axis("H"),
        component_index=component_index,
    )
    transformed = declaration.component_field(
        width=plan.resolution_assignment.require_axis("W"),
        height=plan.resolution_assignment.require_axis("H"),
        component_index=component_index,
        variation_coordinate=_variation_coordinate(),
    )

    assert transformed == untransformed


def test_variation_coordinates_apply_spatial_translation() -> None:
    declaration = _synthetic_mark_declaration()
    plan = _synthetic_plan()
    identity = declaration.component_field(
        width=plan.resolution_assignment.require_axis("W"),
        height=plan.resolution_assignment.require_axis("H"),
        component_index=0,
        variation_coordinate=_variation_coordinate(),
    )
    shifted = declaration.component_field(
        width=plan.resolution_assignment.require_axis("W"),
        height=plan.resolution_assignment.require_axis("H"),
        component_index=0,
        variation_coordinate=_variation_coordinate(matrix=_affine_matrix(tx=0.25)),
    )

    assert _weighted_x_mean(shifted) > _weighted_x_mean(identity) + 0.2
    assert all(0.0 <= value <= 1.0 for value in shifted.values)


def test_observation_formation_rejects_invalid_variation_coordinate() -> None:
    declaration = _synthetic_mark_declaration()
    plan = _synthetic_plan()

    assert (
        str(
            capture_observation_error(
                lambda: declaration.component_field(
                    width=plan.resolution_assignment.require_axis("W"),
                    height=plan.resolution_assignment.require_axis("H"),
                    component_index=0,
                    variation_coordinate={},
                )
            )
        )
        == "variation_coordinates: expected field-variation-transform-coordinate"
    )


def test_variation_transform_declaration_round_trips_canonically() -> None:
    transform = VariationTransformDeclaration.from_record(_variation_transform_record())

    assert transform.to_record() == _canonical_variation_transform_record()
    assert SpatialAffineVariation.identity(spatial_rank=2).to_record() == {
        "kind": "spatial-affine",
        "coordinate_system": "normalized-sequence-element",
        "spatial_rank": 2,
        "matrix": [
            [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
        ],
    }


def test_variation_transform_declaration_rejects_invalid_bounds() -> None:
    record = _variation_transform_record()
    spatial = dict(cast(dict[str, object], record["spatial_affine"]))
    spatial["spatial_rank"] = 3
    record["spatial_affine"] = spatial
    assert (
        str(capture_observation_error(lambda: VariationTransformDeclaration.from_record(record)))
        == "matrix row count must equal spatial_rank plus one"
    )

    record = _variation_transform_record()
    spatial = dict(cast(dict[str, object], record["spatial_affine"]))
    spatial["matrix"] = [[[0.0, 1.0]]]
    record["spatial_affine"] = spatial
    assert (
        str(capture_observation_error(lambda: VariationTransformDeclaration.from_record(record)))
        == "matrix row count must equal spatial_rank plus one"
    )

    record = _variation_transform_record()
    spatial = dict(cast(dict[str, object], record["spatial_affine"]))
    spatial["matrix"] = [
        [[0.9, 1.1], [-0.2, 0.2], [-0.1, 0.1]],
        [[-0.1, 0.1], [0.8, 1.2], [-0.2, 0.2]],
        [[0.0, 0.0], [0.0, 0.0], [0.9, 1.1]],
    ]
    record["spatial_affine"] = spatial
    assert (
        str(capture_observation_error(lambda: VariationTransformDeclaration.from_record(record)))
        == "matrix final row must be fixed affine coordinates"
    )


def test_variation_transform_declaration_rejects_unsupported_kinds() -> None:
    record = _variation_transform_record()
    record["kind"] = "other-transform"

    assert (
        str(capture_observation_error(lambda: VariationTransformDeclaration.from_record(record)))
        == "unsupported variation transform kind: other-transform"
    )


def test_non_digits_declaration_uses_same_interpreter_path() -> None:
    declaration = ObservationFormationDeclaration.from_record(
        {
            "id": "benchmarks.synthetic-bars.observation-formation@0.1.0",
            "benchmark_id": "benchmarks.synthetic-bars@0.1.0",
            "interpreter": "field-mark-composition@0.1.0",
            "output_field": {"channel_count": 1, "width_axis": "W", "height_axis": "H"},
            "sequence_layout": {
                "sequence_axis": "S",
                "width_axis": "W",
                "height_axis": "H",
                "placement_axis": "y",
            },
            "components": [
                {
                    "id": "bar",
                    "marks": [
                        {
                            "kind": "bezier-curve",
                            "channel": 0,
                            "degree": 2,
                            "control_points": [[0.2, 0.5], [0.5, 0.2], [0.8, 0.5]],
                            "width": 2,
                        }
                    ],
                }
            ],
        }
    )
    plan = MaterializationPlan(
        id=ProtocolIdentifier.parse("benchmarks.synthetic-bars.materialization-plan@0.1.0"),
        benchmark_id=ProtocolIdentifier.parse("benchmarks.synthetic-bars@0.1.0"),
        materialization_declaration=ArtifactReference(
            kind="materialization-declaration",
            protocol_id=ProtocolIdentifier.parse("benchmarks.synthetic-bars.materialization@0.1.0"),
        ),
        resolution_assignment=AxisAssignment(values={"W": 96, "H": 96}),
        seed=101,
    )

    observation = declaration.component_field(
        width=plan.resolution_assignment.require_axis("W"),
        height=plan.resolution_assignment.require_axis("H"),
        component_index=0,
    )

    assert observation.shape == (1, 96, 96)
    assert sum(1 for value in observation.values if value > 0) > 0


def test_observation_formation_documents_reject_invalid_bytes() -> None:
    assert (
        str(
            capture_observation_error(
                lambda: ObservationFormationDeclarationDocument.from_bytes(b"\xff")
            )
        )
        == "observation formation declaration must be UTF-8"
    )
    assert (
        str(
            capture_observation_error(
                lambda: ObservationFormationDeclarationDocument.from_bytes(b"[]")
            )
        )
        == "observation formation declaration must contain an object"
    )


def _digits_declaration() -> ObservationFormationDeclaration:
    return load_digits_benchmark(_digits_benchmark_root).formation


def _variation_transform_record() -> dict[str, object]:
    return {
        "kind": "field-variation-transform",
        "spatial_affine": {
            "kind": "spatial-affine",
            "coordinate_system": "normalized-sequence-element",
            "spatial_rank": 2,
            "matrix": [
                [[0.9, 1.1], [-0.2, 0.2], [-0.1, 0.1]],
                [[-0.1, 0.1], [0.8, 1.2], [-0.2, 0.2]],
                [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
            ],
        },
    }


def _canonical_variation_transform_record() -> dict[str, object]:
    return {
        "kind": "field-variation-transform",
        "spatial_affine": {
            "kind": "spatial-affine",
            "coordinate_system": "normalized-sequence-element",
            "spatial_rank": 2,
            "matrix": [
                [[0.9, 1.1], [-0.2, 0.2], [-0.1, 0.1]],
                [[-0.1, 0.1], [0.8, 1.2], [-0.2, 0.2]],
                [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
            ],
        },
    }


def _synthetic_mark_declaration() -> ObservationFormationDeclaration:
    return ObservationFormationDeclaration.from_record(_minimal_declaration_record())


def _minimal_declaration_record() -> dict[str, object]:
    return {
        "id": "benchmarks.synthetic-marks.observation-formation@0.1.0",
        "benchmark_id": "benchmarks.synthetic-marks@0.1.0",
        "interpreter": "field-mark-composition@0.1.0",
        "output_field": {"channel_count": 1, "width_axis": "W", "height_axis": "H"},
        "sequence_layout": {
            "sequence_axis": "S",
            "width_axis": "W",
            "height_axis": "H",
            "placement_axis": "x",
        },
        "components": [
            {
                "id": "mark",
                "marks": [
                    {
                        "kind": "bezier-curve",
                        "channel": 0,
                        "degree": 1,
                        "control_points": [[0.35, 0.5], [0.65, 0.5]],
                        "width": 4,
                    }
                ],
            }
        ],
    }


def _synthetic_plan() -> MaterializationPlan:
    return MaterializationPlan(
        id=ProtocolIdentifier.parse("benchmarks.synthetic-marks.materialization-plan@0.1.0"),
        benchmark_id=ProtocolIdentifier.parse("benchmarks.synthetic-marks@0.1.0"),
        materialization_declaration=ArtifactReference(
            kind="materialization-declaration",
            protocol_id=ProtocolIdentifier.parse(
                "benchmarks.synthetic-marks.materialization@0.1.0"
            ),
        ),
        resolution_assignment=AxisAssignment(values={"W": 32, "H": 32}),
        seed=101,
    )


def _variation_coordinate(
    *,
    matrix: _AffineMatrix2D = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
) -> dict[str, object]:
    return {
        "kind": "field-variation-transform-coordinate",
        "component_index": 0,
        "spatial_affine": {
            "kind": "spatial-affine-coordinate",
            "coordinate_system": "normalized-sequence-element",
            "matrix": [list(row) for row in matrix],
        },
    }


def _affine_matrix(*, tx: float = 0.0) -> _AffineMatrix2D:
    return ((1.0, 0.0, tx), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _nonzero_bounds(field: FieldObservation) -> tuple[int, int, int, int]:
    _channels, height, width = field.shape
    coordinates = [
        (x, y) for y in range(height) for x in range(width) if field.values[y * width + x] > 0
    ]
    if not coordinates:
        raise AssertionError("expected nonzero field")
    return (
        min(x for x, _y in coordinates),
        max(x for x, _y in coordinates),
        min(y for _x, y in coordinates),
        max(y for _x, y in coordinates),
    )


def _weighted_x_mean(field: FieldObservation) -> float:
    _channels, _height, width = field.shape
    total = sum(field.values)
    if total <= 0:
        raise AssertionError("expected nonzero field")
    return sum((index % width) / width * value for index, value in enumerate(field.values)) / total


def capture_observation_error(
    call: Callable[[], object],
) -> ObservationFormationValidationError:
    with pytest.raises(ObservationFormationValidationError) as error:
        call()
    return error.value
