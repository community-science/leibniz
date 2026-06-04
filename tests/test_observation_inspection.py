from collections.abc import Callable
from pathlib import Path

import pytest
from benchmark_typing import load_digits_benchmark

from leibniz.artifacts import ArtifactReference
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import AxisAssignment, MaterializationPlan, MaterializationPlanDocument
from leibniz.observation_formation import (
    ObservationFormationDeclaration,
)
from leibniz.observation_inspection import (
    FieldPreview,
    ObservationInspectionDocument,
    ObservationInspectionRecord,
    ObservationInspectionValidationError,
)

_repository_root = Path(__file__).parents[1]
_digits_benchmark_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
_digits_fixture_root = _repository_root / "tests" / "fixtures" / "digits"


def test_digits_observation_inspection_records_sample_provenance() -> None:
    declaration = _digits_declaration()
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l3.json").read_bytes()
    ).plan
    component_index = declaration.sample_component_index(seed=plan.seed, sample_index=2)
    observation = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.l3.sample-2@0.1.0"),
        plan=plan,
        component_index=component_index,
    )

    inspection = ObservationInspectionRecord.from_formed_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.inspections.l3.sample-2@0.1.0"),
        observation=observation,
        materialization_plan=plan,
        sample_index=2,
        outcome_id=f"digit-{component_index}",
    )
    record = inspection.to_record()

    assert inspection.benchmark_id == ProtocolIdentifier.parse("benchmarks.digits@0.1.0")
    assert inspection.component_index == component_index
    assert inspection.resolution_assignment.values == {"W": 72, "H": 24}
    assert inspection.field_shape == (1, 24, 72)
    assert inspection.field_preview is not None
    assert sum(run.count for run in inspection.field_preview.runs) == 72 * 24
    assert record["component_index"] == component_index
    assert record["field_digest"] == str(observation.field.digest)
    assert record["formed_observation"] == {
        "kind": "formed-observation",
        "protocol_id": "benchmarks.digits.observations.l3.sample-2@0.1.0",
        "record_digest": str(observation.digest),
    }


def test_observation_inspection_document_round_trips_canonically() -> None:
    inspection = _digits_inspection()

    document = ObservationInspectionDocument.from_bytes(inspection.to_bytes())

    assert document.inspection == inspection
    assert document.digest == inspection.digest
    assert document.inspection.to_bytes() == inspection.to_bytes()


def test_observation_inspection_preview_is_deterministic_rle() -> None:
    field = _digits_observation().field

    left = FieldPreview.from_field(field)
    right = FieldPreview.from_field(field)

    assert left == right
    assert left.shape == field.shape
    assert {run.value for run in left.runs}.issubset(set(range(256)))
    assert any(run.value == 255 for run in left.runs)
    assert any(run.value == 0 for run in left.runs)


def test_non_digits_observation_uses_same_inspection_record_path() -> None:
    declaration = ObservationFormationDeclaration.from_record(
        {
            "id": "benchmarks.synthetic-bars.observation-formation@0.1.0",
            "benchmark_id": "benchmarks.synthetic-bars@0.1.0",
            "interpreter": "field-mark-composition@0.1.0",
            "output_field": {
                "channel_count": 1,
                "width_axis": "W",
                "height_axis": "H",
            },
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
                            "degree": 1,
                            "control_points": [[0.2, 0.5], [0.8, 0.5]],
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
        resolution_assignment=AxisAssignment(values={"W": 16, "H": 20}),
        seed=11,
    )
    observation = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.synthetic-bars.observations.sample-0@0.1.0"),
        plan=plan,
        component_index=0,
    )

    inspection = ObservationInspectionRecord.from_formed_observation(
        id=ProtocolIdentifier.parse("benchmarks.synthetic-bars.inspections.sample-0@0.1.0"),
        observation=observation,
        materialization_plan=plan,
        sample_index=0,
    )

    assert inspection.resolution_assignment.values == {"W": 16, "H": 20}
    assert inspection.field_shape == (1, 20, 16)
    assert inspection.outcome_id is None


def test_observation_inspection_rejects_mismatched_plan_reference() -> None:
    observation = _digits_observation()
    wrong_plan = MaterializationPlan(
        id=ProtocolIdentifier.parse("benchmarks.digits.materialization-plan.wrong@0.1.0"),
        benchmark_id=ProtocolIdentifier.parse("benchmarks.digits@0.1.0"),
        materialization_declaration=ArtifactReference(
            kind="materialization-declaration",
            protocol_id=ProtocolIdentifier.parse("benchmarks.digits.materialization@0.1.0"),
        ),
        resolution_assignment=AxisAssignment(values={"W": 48, "H": 16}),
        seed=101,
    )

    assert (
        str(
            capture_inspection_error(
                lambda: ObservationInspectionRecord.from_formed_observation(
                    id=ProtocolIdentifier.parse("benchmarks.digits.inspections.bad@0.1.0"),
                    observation=observation,
                    materialization_plan=wrong_plan,
                    sample_index=0,
                )
            )
        )
        == "observation materialization_plan reference does not match plan"
    )


def test_observation_inspection_rejects_invalid_preview_run_length() -> None:
    record = _digits_inspection().to_record()
    preview = record["field_preview"]
    assert isinstance(preview, dict)
    preview["runs"] = [{"value": 0, "count": 1}]

    assert (
        str(capture_inspection_error(lambda: ObservationInspectionRecord.from_record(record)))
        == "preview run length 1 does not match shape size 1728"
    )


def _digits_declaration() -> ObservationFormationDeclaration:
    return load_digits_benchmark(_digits_benchmark_root).formation


def _digits_observation():
    declaration = _digits_declaration()
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l3.json").read_bytes()
    ).plan
    return declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.l3.sample-0@0.1.0"),
        plan=plan,
        component_index=declaration.sample_component_index(seed=plan.seed, sample_index=0),
    )


def _digits_inspection() -> ObservationInspectionRecord:
    declaration = _digits_declaration()
    plan = MaterializationPlanDocument.from_bytes(
        (_digits_fixture_root / "materialization_plan_l3.json").read_bytes()
    ).plan
    observation = declaration.form_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.observations.l3.sample-0@0.1.0"),
        plan=plan,
        component_index=declaration.sample_component_index(seed=plan.seed, sample_index=0),
    )
    return ObservationInspectionRecord.from_formed_observation(
        id=ProtocolIdentifier.parse("benchmarks.digits.inspections.l3.sample-0@0.1.0"),
        observation=observation,
        materialization_plan=plan,
        sample_index=0,
    )


def capture_inspection_error(
    call: Callable[[], object],
) -> ObservationInspectionValidationError:
    with pytest.raises(ObservationInspectionValidationError) as error:
        call()
    return error.value
