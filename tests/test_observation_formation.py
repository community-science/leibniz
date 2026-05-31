from collections.abc import Callable
from itertools import product
from pathlib import Path
from typing import cast

import pytest

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
    document = ObservationFormationDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "observation_formation.json").read_bytes()
    )
    declaration = document.declaration

    assert declaration.id == ProtocolIdentifier.parse(
        "benchmarks.digits.observation-formation@0.1.0"
    )
    assert declaration.benchmark_id == ProtocolIdentifier.parse("benchmarks.digits@0.1.0")
    assert declaration.channel_count == 1
    assert declaration.width_axis == "W"
    assert declaration.height_axis == "H"
    assert declaration.sequence_layout.sequence_axis == "L"
    assert declaration.to_record()["sequence_layout"] == {
        "sequence_axis": "L",
        "placement_axis": "x",
        "width_axis": "W",
        "height_axis": "H",
    }
    assert "slot_composition" not in declaration.to_record()
    assert (
        declaration.variation_transform.spatial_affine.coordinate_system
        == "normalized-sequence-element"
    )
    assert declaration.variation_transform.spatial_affine.matrix == (
        ((0.8, 1.2), (-0.08, 0.08), (-0.05, 0.05)),
        ((-0.08, 0.08), (0.8, 1.2), (-0.05, 0.05)),
        ((0.0, 0.0), (0.0, 0.0), (1.0, 1.0)),
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
        scale_assignment=AxisAssignment(values={"L": 1}),
        complexity_assignment=AxisAssignment(values={"C": 1}),
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
            observation = declaration.form_observation(
                id=ProtocolIdentifier.parse("benchmarks.digits.observations.margin-test@0.1.0"),
                plan=plan,
                component_sequence=(component_index,),
                variation_coordinates=(_variation_coordinate(matrix=affine_matrix),),
            )
            min_x, max_x, min_y, max_y = _nonzero_bounds(observation.field)
            assert min_x > 0
            assert max_x < 31
            assert min_y > 0
            assert max_y < 31


def test_digits_observation_formation_is_deterministic_for_materialization_plan() -> None:
    declaration = _digits_declaration()
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l3.json").read_bytes()
    ).plan
    sequence = declaration.sample_component_sequence(plan=plan, sample_index=0)

    left = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.l3-sample-zero@0.1.0"),
        plan=plan,
        component_sequence=sequence,
    )
    right = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.l3-sample-zero@0.1.0"),
        plan=plan,
        component_sequence=sequence,
    )

    assert sequence == declaration.sample_component_sequence(plan=plan, sample_index=0)
    assert left == right
    assert left.field.shape == (1, 16, 48)
    assert max(left.field.values) == 1.0
    assert sum(1 for value in left.field.values if value > 0) > 0
    assert left.to_record()["component_sequence"] == list(sequence)
    assert left.to_record()["field_digest"] == str(left.field.digest)


def test_digits_observation_formation_separates_sequence_elements() -> None:
    declaration = _digits_declaration()
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l3.json").read_bytes()
    ).plan
    observation = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.l3.manual@0.1.0"),
        plan=plan,
        component_sequence=(1, 2, 3),
    )

    assert observation.component_sequence == (1, 2, 3)
    assert _nonzero_count(observation.field, x_start=0, x_stop=16) > 0
    assert _nonzero_count(observation.field, x_start=16, x_stop=32) > 0
    assert _nonzero_count(observation.field, x_start=32, x_stop=48) > 0


def test_digits_observation_formation_uses_sampled_canvas_extent() -> None:
    declaration = _digits_declaration()
    plan = MaterializationPlan(
        id=ProtocolIdentifier.parse("benchmarks.digits.materialization-plan.large-canvas@0.1.0"),
        benchmark_id=ProtocolIdentifier.parse("benchmarks.digits@0.1.0"),
        materialization_declaration=ArtifactReference(
            kind="materialization-declaration",
            protocol_id=ProtocolIdentifier.parse("benchmarks.digits.materialization@0.1.0"),
        ),
        scale_assignment=AxisAssignment(values={"L": 3}),
        complexity_assignment=AxisAssignment(values={"C": 3}),
        resolution_assignment=AxisAssignment(values={"W": 128, "H": 64}),
        seed=101,
    )

    observation = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.large-canvas@0.1.0"),
        plan=plan,
        component_sequence=(1, 2, 3),
    )
    min_x, max_x, min_y, max_y = _nonzero_bounds(observation.field)

    assert min_x > 0
    assert max_x < 127
    assert min_y > 0
    assert max_y < 63


def test_digits_observation_formation_keeps_stroke_width_in_pixel_space() -> None:
    declaration = _digits_declaration()
    plan = MaterializationPlan(
        id=ProtocolIdentifier.parse("benchmarks.digits.materialization-plan.wide-canvas@0.1.0"),
        benchmark_id=ProtocolIdentifier.parse("benchmarks.digits@0.1.0"),
        materialization_declaration=ArtifactReference(
            kind="materialization-declaration",
            protocol_id=ProtocolIdentifier.parse("benchmarks.digits.materialization@0.1.0"),
        ),
        scale_assignment=AxisAssignment(values={"L": 7}),
        complexity_assignment=AxisAssignment(values={"C": 7}),
        resolution_assignment=AxisAssignment(values={"W": 339, "H": 41}),
        seed=407,
    )

    observation = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.wide-canvas@0.1.0"),
        plan=plan,
        component_sequence=(9, 0, 1, 2, 3, 4, 5),
    )
    _channels, height, width = observation.field.shape

    for sequence_index in range(7):
        x_start = round(sequence_index * width / 7)
        x_stop = round((sequence_index + 1) * width / 7)
        cell_area = (x_stop - x_start) * height
        nonzero_fraction = _nonzero_count(
            observation.field,
            x_start=x_start,
            x_stop=x_stop,
        ) / cell_area
        assert 0.04 < nonzero_fraction < 0.15


def test_observation_formation_rejects_component_sequence_mismatch() -> None:
    declaration = _digits_declaration()
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l3.json").read_bytes()
    ).plan

    assert (
        str(
            capture_observation_error(
                lambda: declaration.form_observation(
                    id=ProtocolIdentifier.parse("benchmarks.digits.observations.bad@0.1.0"),
                    plan=plan,
                    component_sequence=(1, 2),
                )
            )
        )
        == "component_sequence length 2 does not match sequence length 3"
    )


def test_observation_formation_rejects_retired_slot_contract() -> None:
    record = _minimal_declaration_record()
    record["slot_composition"] = record.pop("sequence_layout")

    assert (
        str(capture_observation_error(lambda: ObservationFormationDeclaration.from_record(record)))
        == "sequence_layout: missing required field; slot_composition: unknown field"
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
    sequence = declaration.sample_component_sequence(plan=plan, sample_index=0)

    untransformed = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.identity-left@0.1.0"),
        plan=plan,
        component_sequence=sequence,
    )
    transformed = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.identity-right@0.1.0"),
        plan=plan,
        component_sequence=sequence,
        variation_coordinates=tuple(
            _variation_coordinate(sequence_index=sequence_index)
            for sequence_index in range(len(sequence))
        ),
    )

    assert transformed.field == untransformed.field


def test_variation_coordinates_apply_spatial_translation() -> None:
    declaration = _synthetic_mark_declaration()
    plan = _synthetic_plan()
    identity = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.synthetic-marks.observations.base@0.1.0"),
        plan=plan,
        component_sequence=(0,),
        variation_coordinates=(_variation_coordinate(),),
    )
    shifted = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.synthetic-marks.observations.shifted@0.1.0"),
        plan=plan,
        component_sequence=(0,),
        variation_coordinates=(_variation_coordinate(matrix=_affine_matrix(tx=0.25)),),
    )

    assert _weighted_x_mean(shifted.field) > _weighted_x_mean(identity.field) + 0.2
    assert all(0.0 <= value <= 1.0 for value in shifted.field.values)


def test_variation_translation_is_position_relative_for_longer_sequences() -> None:
    declaration = _synthetic_mark_declaration()
    plan = _synthetic_plan_with(sequence_length=4, resolution=128)
    identity = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.synthetic-marks.observations.base@0.1.0"),
        plan=plan,
        component_sequence=(0, 0, 0, 0),
        variation_coordinates=tuple(
            _variation_coordinate(sequence_index=sequence_index) for sequence_index in range(4)
        ),
    )
    shifted = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.synthetic-marks.observations.shifted@0.1.0"),
        plan=plan,
        component_sequence=(0, 0, 0, 0),
        variation_coordinates=tuple(
            _variation_coordinate(sequence_index=sequence_index, matrix=_affine_matrix(tx=0.25))
            for sequence_index in range(4)
        ),
    )

    expected_shifted_mean = _weighted_x_mean(identity.field) + 0.25 / 4
    assert abs(_weighted_x_mean(shifted.field) - expected_shifted_mean) <= 0.02
    assert _nonzero_count(shifted.field, x_start=0, x_stop=32) > 0
    assert _nonzero_count(shifted.field, x_start=32, x_stop=64) > 0
    assert _nonzero_count(shifted.field, x_start=64, x_stop=96) > 0
    assert _nonzero_count(shifted.field, x_start=96, x_stop=128) > 0


def test_observation_formation_rejects_variation_coordinate_mismatch() -> None:
    declaration = _synthetic_mark_declaration()
    plan = _synthetic_plan()

    assert (
        str(
            capture_observation_error(
                lambda: declaration.form_observation(
                    id=ProtocolIdentifier.parse(
                        "benchmarks.synthetic-marks.observations.bad@0.1.0"
                    ),
                    plan=plan,
                    component_sequence=(0,),
                    variation_coordinates=(),
                )
            )
        )
        == "variation_coordinates length must match sequence length"
    )

    assert (
        str(
            capture_observation_error(
                lambda: declaration.form_observation(
                    id=ProtocolIdentifier.parse(
                        "benchmarks.synthetic-marks.observations.bad@0.1.0"
                    ),
                    plan=plan,
                    component_sequence=(0,),
                    variation_coordinates=(_variation_coordinate(sequence_index=1),),
                )
            )
        )
        == "variation coordinate sequence_index must match coordinate position"
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
        scale_assignment=AxisAssignment(values={"S": 3}),
        complexity_assignment=AxisAssignment(values={"C": 3}),
        resolution_assignment=AxisAssignment(values={"W": 96, "H": 96}),
        seed=101,
    )

    observation = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.synthetic-bars.observations.sample-zero@0.1.0"),
        plan=plan,
        component_sequence=(0, 0, 0),
    )

    assert observation.field.shape == (1, 96, 96)
    assert sum(1 for value in observation.field.values if value > 0) > 0


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
    return ObservationFormationDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "observation_formation.json").read_bytes()
    ).declaration


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
    return _synthetic_plan_with(sequence_length=1, resolution=32)


def _synthetic_plan_with(*, sequence_length: int, resolution: int) -> MaterializationPlan:
    return MaterializationPlan(
        id=ProtocolIdentifier.parse("benchmarks.synthetic-marks.materialization-plan@0.1.0"),
        benchmark_id=ProtocolIdentifier.parse("benchmarks.synthetic-marks@0.1.0"),
        materialization_declaration=ArtifactReference(
            kind="materialization-declaration",
            protocol_id=ProtocolIdentifier.parse(
                "benchmarks.synthetic-marks.materialization@0.1.0"
            ),
        ),
        scale_assignment=AxisAssignment(values={"S": sequence_length}),
        complexity_assignment=AxisAssignment(values={"C": sequence_length}),
        resolution_assignment=AxisAssignment(values={"W": resolution, "H": resolution}),
        seed=101,
    )


def _variation_coordinate(
    *,
    sequence_index: int = 0,
    matrix: _AffineMatrix2D = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
) -> dict[str, object]:
    return {
        "kind": "field-variation-transform-coordinate",
        "sequence_index": sequence_index,
        "spatial_affine": {
            "kind": "spatial-affine-coordinate",
            "coordinate_system": "normalized-sequence-element",
            "matrix": [list(row) for row in matrix],
        },
    }


def _affine_matrix(*, tx: float = 0.0, ty: float = 0.0) -> _AffineMatrix2D:
    return ((1.0, 0.0, tx), (0.0, 1.0, ty), (0.0, 0.0, 1.0))


def _nonzero_count(field: FieldObservation, *, x_start: int, x_stop: int) -> int:
    _channels, height, width = field.shape
    count = 0
    for y in range(height):
        for x in range(x_start, x_stop):
            if field.values[y * width + x] > 0:
                count += 1
    return count


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
    _channels, height, width = field.shape
    weighted_sum = 0.0
    total = 0.0
    for y in range(height):
        for x in range(width):
            value = field.values[y * width + x]
            weighted_sum += ((x + 0.5) / width) * value
            total += value
    if total == 0.0:
        raise AssertionError("expected nonzero field")
    return weighted_sum / total


def capture_observation_error(
    call: Callable[[], object],
) -> ObservationFormationValidationError:
    with pytest.raises(ObservationFormationValidationError) as error:
        call()
    return error.value
