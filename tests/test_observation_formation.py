from collections.abc import Callable
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
    ValueScaleVariation,
    VariationTransformDeclaration,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_digits_fixture_root = _repository_root / "tests" / "fixtures" / "digits"


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
    assert declaration.resolution_axis == "N"
    assert declaration.slot_composition.count_axis == "L"
    assert declaration.variation_transform.spatial_affine.coordinate_system == "normalized-slot"
    assert declaration.variation_transform.spatial_affine.translation == (
        (-0.08, 0.08),
        (-0.08, 0.08),
    )
    assert declaration.variation_transform.spatial_affine.scale == (
        (0.9, 1.1),
        (0.9, 1.1),
    )
    assert declaration.variation_transform.spatial_affine.rotation_degrees == (8.0,)
    assert declaration.variation_transform.spatial_affine.shear_degrees == (5.0,)
    assert declaration.variation_transform.value_scale.scale == (0.85, 1.15)
    assert [component.id for component in declaration.components] == [
        f"digit-{digit}" for digit in range(10)
    ]


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
    assert left.field.shape == (1, 96, 96)
    assert max(left.field.values) == 1.0
    assert sum(1 for value in left.field.values if value > 0) > 0
    assert left.to_record()["component_sequence"] == list(sequence)
    assert left.to_record()["field_digest"] == str(left.field.digest)


def test_digits_observation_formation_separates_slots() -> None:
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
    assert _nonzero_count(observation.field, x_start=0, x_stop=32) > 0
    assert _nonzero_count(observation.field, x_start=32, x_stop=64) > 0
    assert _nonzero_count(observation.field, x_start=64, x_stop=96) > 0


def test_observation_formation_rejects_component_sequence_mismatch() -> None:
    declaration = _digits_declaration()
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l3.json").read_bytes()
    ).plan

    assert str(
        capture_observation_error(
            lambda: declaration.form_observation(
                id=ProtocolIdentifier.parse("benchmarks.digits.observations.bad@0.1.0"),
                plan=plan,
                component_sequence=(1, 2),
            )
        )
    ) == "component_sequence length 2 does not match slot count 3"


def test_observation_formation_preserves_explicit_zero_mark_values() -> None:
    declaration = ObservationFormationDeclaration.from_record(
        {
            "id": "benchmarks.synthetic-masks.observation-formation@0.1.0",
            "benchmark_id": "benchmarks.synthetic-masks@0.1.0",
            "interpreter": "field-mark-composition@0.1.0",
            "output_field": {"channel_count": 1, "resolution_axis": "N"},
            "slot_composition": {
                "count_axis": "S",
                "resolution_axis": "N",
                "slot_axis": "x",
            },
            "components": [
                {
                    "id": "mask",
                    "marks": [
                        {
                            "kind": "bezier-curve",
                            "channel": 0,
                            "degree": 1,
                            "control_points": [[0.2, 0.5], [0.8, 0.5]],
                            "width": 2,
                            "value": 0,
                        }
                    ],
                }
            ],
        }
    )

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
            _variation_coordinate(slot_index=slot_index)
            for slot_index in range(len(sequence))
        ),
    )

    assert transformed.field == untransformed.field


def test_variation_coordinates_apply_value_scale() -> None:
    declaration = _synthetic_mark_declaration()
    plan = _synthetic_plan()

    observation = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.synthetic-marks.observations.scaled@0.1.0"),
        plan=plan,
        component_sequence=(0,),
        variation_coordinates=(_variation_coordinate(value_scale=0.5),),
    )

    assert max(observation.field.values) == 0.5


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
        variation_coordinates=(_variation_coordinate(translation=(0.25, 0.0)),),
    )

    assert _weighted_x_mean(shifted.field) > _weighted_x_mean(identity.field) + 0.2
    assert all(0.0 <= value <= 1.0 for value in shifted.field.values)


def test_variation_translation_is_slot_relative_for_multi_slot_observations() -> None:
    declaration = _synthetic_mark_declaration()
    plan = _synthetic_plan_with(slot_count=4, resolution=128)
    identity = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.synthetic-marks.observations.base@0.1.0"),
        plan=plan,
        component_sequence=(0, 0, 0, 0),
        variation_coordinates=tuple(
            _variation_coordinate(slot_index=slot_index) for slot_index in range(4)
        ),
    )
    shifted = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.synthetic-marks.observations.shifted@0.1.0"),
        plan=plan,
        component_sequence=(0, 0, 0, 0),
        variation_coordinates=tuple(
            _variation_coordinate(slot_index=slot_index, translation=(0.25, 0.0))
            for slot_index in range(4)
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

    assert str(
        capture_observation_error(
            lambda: declaration.form_observation(
                id=ProtocolIdentifier.parse("benchmarks.synthetic-marks.observations.bad@0.1.0"),
                plan=plan,
                component_sequence=(0,),
                variation_coordinates=(),
            )
        )
    ) == "variation_coordinates length must match slot count"

    assert str(
        capture_observation_error(
            lambda: declaration.form_observation(
                id=ProtocolIdentifier.parse("benchmarks.synthetic-marks.observations.bad@0.1.0"),
                plan=plan,
                component_sequence=(0,),
                variation_coordinates=(_variation_coordinate(slot_index=1),),
            )
        )
    ) == "variation coordinate slot_index must match coordinate position"


def test_variation_transform_declaration_round_trips_canonically() -> None:
    transform = VariationTransformDeclaration.from_record(_variation_transform_record())

    assert transform.to_record() == _canonical_variation_transform_record()
    assert SpatialAffineVariation.identity(spatial_rank=2).to_record() == {
        "kind": "spatial-affine",
        "coordinate_system": "normalized-slot",
        "spatial_rank": 2,
        "translation": [[0.0, 0.0], [0.0, 0.0]],
        "scale": [[1.0, 1.0], [1.0, 1.0]],
        "rotation_degrees": [0.0],
        "shear_degrees": [0.0],
    }
    assert ValueScaleVariation.identity().to_record() == {
        "kind": "value-scale",
        "scale": [1.0, 1.0],
    }


def test_variation_transform_declaration_rejects_invalid_bounds() -> None:
    record = _variation_transform_record()
    spatial = dict(cast(dict[str, object], record["spatial_affine"]))
    spatial["spatial_rank"] = 3
    record["spatial_affine"] = spatial
    assert str(
        capture_observation_error(lambda: VariationTransformDeclaration.from_record(record))
    ) == "translation bounds length must equal spatial_rank"

    record = _variation_transform_record()
    spatial = dict(cast(dict[str, object], record["spatial_affine"]))
    spatial["scale"] = [[0.0, 1.0], [1.0, 1.0]]
    record["spatial_affine"] = spatial
    assert str(
        capture_observation_error(lambda: VariationTransformDeclaration.from_record(record))
    ) == "scale.0 bounds must be positive"

    record = _variation_transform_record()
    value_scale = dict(cast(dict[str, object], record["value_scale"]))
    value_scale["scale"] = [1.2, 0.8]
    record["value_scale"] = value_scale
    assert str(
        capture_observation_error(lambda: VariationTransformDeclaration.from_record(record))
    ) == "value_scale.scale lower bound must not exceed upper"


def test_variation_transform_declaration_rejects_unsupported_kinds() -> None:
    record = _variation_transform_record()
    record["kind"] = "other-transform"

    assert str(
        capture_observation_error(lambda: VariationTransformDeclaration.from_record(record))
    ) == "unsupported variation transform kind: other-transform"


def test_non_digits_declaration_uses_same_interpreter_path() -> None:
    declaration = ObservationFormationDeclaration.from_record(
        {
            "id": "benchmarks.synthetic-bars.observation-formation@0.1.0",
            "benchmark_id": "benchmarks.synthetic-bars@0.1.0",
            "interpreter": "field-mark-composition@0.1.0",
            "output_field": {"channel_count": 1, "resolution_axis": "N"},
            "slot_composition": {
                "count_axis": "S",
                "resolution_axis": "N",
                "slot_axis": "y",
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
        resolution_assignment=AxisAssignment(values={"N": 96}),
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
    assert str(
        capture_observation_error(
            lambda: ObservationFormationDeclarationDocument.from_bytes(b"\xff")
        )
    ) == "observation formation declaration must be UTF-8"
    assert str(
        capture_observation_error(
            lambda: ObservationFormationDeclarationDocument.from_bytes(b"[]")
        )
    ) == "observation formation declaration must contain an object"


def _digits_declaration() -> ObservationFormationDeclaration:
    return ObservationFormationDeclarationDocument.from_bytes(
        (_digits_benchmark_root / "observation_formation.json").read_bytes()
    ).declaration


def _variation_transform_record() -> dict[str, object]:
    return {
        "kind": "field-variation-transform",
        "spatial_affine": {
            "kind": "spatial-affine",
            "coordinate_system": "normalized-slot",
            "spatial_rank": 2,
            "translation": [[-0.1, 0.1], [-0.2, 0.2]],
            "scale": [[0.9, 1.1], [0.8, 1.2]],
            "rotation_degrees": [12],
            "shear_degrees": [5],
        },
        "value_scale": {
            "kind": "value-scale",
            "scale": [0.75, 1.25],
        },
    }


def _canonical_variation_transform_record() -> dict[str, object]:
    return {
        "kind": "field-variation-transform",
        "spatial_affine": {
            "kind": "spatial-affine",
            "coordinate_system": "normalized-slot",
            "spatial_rank": 2,
            "translation": [[-0.1, 0.1], [-0.2, 0.2]],
            "scale": [[0.9, 1.1], [0.8, 1.2]],
            "rotation_degrees": [12.0],
            "shear_degrees": [5.0],
        },
        "value_scale": {
            "kind": "value-scale",
            "scale": [0.75, 1.25],
        },
    }


def _synthetic_mark_declaration() -> ObservationFormationDeclaration:
    return ObservationFormationDeclaration.from_record(
        {
            "id": "benchmarks.synthetic-marks.observation-formation@0.1.0",
            "benchmark_id": "benchmarks.synthetic-marks@0.1.0",
            "interpreter": "field-mark-composition@0.1.0",
            "output_field": {"channel_count": 1, "resolution_axis": "N"},
            "slot_composition": {
                "count_axis": "S",
                "resolution_axis": "N",
                "slot_axis": "x",
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
    )


def _synthetic_plan() -> MaterializationPlan:
    return _synthetic_plan_with(slot_count=1, resolution=32)


def _synthetic_plan_with(*, slot_count: int, resolution: int) -> MaterializationPlan:
    return MaterializationPlan(
        id=ProtocolIdentifier.parse("benchmarks.synthetic-marks.materialization-plan@0.1.0"),
        benchmark_id=ProtocolIdentifier.parse("benchmarks.synthetic-marks@0.1.0"),
        materialization_declaration=ArtifactReference(
            kind="materialization-declaration",
            protocol_id=ProtocolIdentifier.parse("benchmarks.synthetic-marks.materialization@0.1.0"),
        ),
        scale_assignment=AxisAssignment(values={"S": slot_count}),
        complexity_assignment=AxisAssignment(values={"C": slot_count}),
        resolution_assignment=AxisAssignment(values={"N": resolution}),
        seed=101,
    )


def _variation_coordinate(
    *,
    slot_index: int = 0,
    translation: tuple[float, float] = (0.0, 0.0),
    scale: tuple[float, float] = (1.0, 1.0),
    rotation_degrees: float = 0.0,
    shear_degrees: float = 0.0,
    value_scale: float = 1.0,
) -> dict[str, object]:
    return {
        "kind": "field-variation-transform-coordinate",
        "slot_index": slot_index,
        "spatial_affine": {
            "kind": "spatial-affine-coordinate",
            "coordinate_system": "normalized-slot",
            "translation": list(translation),
            "scale": list(scale),
            "rotation_degrees": [rotation_degrees],
            "shear_degrees": [shear_degrees],
        },
        "value_scale": {
            "kind": "value-scale-coordinate",
            "scale": value_scale,
        },
    }


def _nonzero_count(field: FieldObservation, *, x_start: int, x_stop: int) -> int:
    _channels, height, width = field.shape
    count = 0
    for y in range(height):
        for x in range(x_start, x_stop):
            if field.values[y * width + x] > 0:
                count += 1
    return count


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
