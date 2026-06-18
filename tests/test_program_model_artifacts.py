from __future__ import annotations

from pathlib import Path

from leibniz.artifacts import ArtifactReference, reference_for_record
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.model_inspection import ModelInspectionRecord
from leibniz.model_interfaces import ModelInterface
from leibniz.model_manifests import ModelArtifactManifest, ModelExecutionFamily
from leibniz.program_graphs import load_program_graph
from leibniz.tensor_runtime import resolve_host_tensor_runtime


def test_model_manifest_references_program_graph() -> None:
    program_graph = _digits_program_graph()
    manifest = _model_manifest(program_graph)

    manifest.validate_program(program_graph)
    record = manifest.to_record()

    assert record["program"] == reference_for_record(
        kind="program-graph",
        record=program_graph,
    ).to_record()
    assert record["execution_family"] == {
        "kind": "submitted-program-graph",
        "runtime": "pytorch",
        "program_family": "open-node-program-graph",
    }


def test_model_inspection_summarizes_program_nodes() -> None:
    program_graph = _digits_program_graph()
    inspection = ModelInspectionRecord.from_program_graph(
        id=ProtocolIdentifier.parse("model-inspections.tests.digits-program@0.1.0"),
        program_graph=program_graph,
        input_shape=(1, 24, 24),
        output_shape=(85,),
    )

    assert inspection.program.matches_record(program_graph)
    assert tuple(component.kind for component in inspection.components) == (
        "encoder",
    )
    assert inspection.to_record()["program_graph"] == program_graph


def _model_manifest(program_graph: dict[str, object]) -> ModelArtifactManifest:
    return ModelArtifactManifest(
        id=ProtocolIdentifier.parse("model-manifests.tests.program@0.1.0"),
        program=reference_for_record(kind="program-graph", record=program_graph),
        interface=ArtifactReference(
            kind="model-interface",
            record_digest=ContentDigest.from_value(
                ModelInterface.from_real_vector_space(
                    id=ProtocolIdentifier.parse(
                        "model-interfaces.tests.program@0.1.0"
                    ),
                    dimension=85,
                    coordinate_name="target-coordinate",
                ).to_record()
            ),
        ),
        execution_family=ModelExecutionFamily.submitted_program_graph(),
        model_artifacts=(
            ArtifactReference(
                kind="model-checkpoint",
                content_digest=ContentDigest.from_value({"weights": "checkpoint"}),
            ),
        ),
    )


def _digits_program_graph() -> dict[str, object]:
    return load_program_graph(
        _fixture("fixtures/programs/digits_inverse_conv_encoder.py"),
        resolve_host_tensor_runtime(),
    ).graph.to_record()


def _fixture(relative: str) -> Path:
    return Path(__file__).parent / relative
