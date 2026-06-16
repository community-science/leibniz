from __future__ import annotations

from pathlib import Path

from leibniz.artifacts import ArtifactReference, reference_for_record
from leibniz.benchmark_implementations import load_benchmark
from leibniz.content import ContentDigest
from leibniz.evaluation_bundles import BenchmarkEvaluationBundle
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset
from leibniz.model_inspection import ModelInspectionRecord
from leibniz.model_interfaces import ModelInterface
from leibniz.model_manifests import ModelArtifactManifest, ModelExecutionFamily
from leibniz.program_graphs import load_program_graph
from leibniz.tensor_runtime import resolve_host_tensor_runtime
from leibniz.views import MeasurementScoreView


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


def test_evaluation_bundle_validates_program_sources() -> None:
    benchmark = _digits_benchmark_manifest()
    program_graph = _digits_program_graph()
    model_manifest = _model_manifest(program_graph)
    inspection = ModelInspectionRecord.from_model_manifest(
        id=ProtocolIdentifier.parse("model-inspections.tests.bundle@0.1.0"),
        model_manifest=model_manifest,
        program_graph=program_graph,
        input_shape=(1, 24, 24),
        output_shape=(85,),
    )
    dataset = MeasurementDataset(measurements=())
    score_view = MeasurementScoreView.from_dataset(
        id=ProtocolIdentifier.parse("views.measurement-scores.tests.bundle@0.1.0"),
        dataset=dataset,
    )
    checkpoint = {
        "kind": "model-checkpoint",
        "path": "results/models/digits/checkpoint.pt",
        "digest": str(ContentDigest.from_value({"weights": "checkpoint"})),
        "manifest_path": "results/models/digits/checkpoint.model.json",
        "manifest_digest": str(model_manifest.digest),
        "step": 0,
        "validation_check": 0,
        "validation_loss": 1.0,
    }

    bundle = BenchmarkEvaluationBundle(
        id=ProtocolIdentifier.parse("benchmark-evaluations.tests.bundle@0.1.0"),
        run_slug="tests-bundle",
        benchmark_manifest=benchmark,
        program_graph=program_graph,
        model_manifest=model_manifest,
        model_checkpoint=checkpoint,
        model_inspection=inspection,
        measurement_dataset=dataset,
        measurement_score_view=score_view,
        sampled_competence={
            "competence_value_kind": "validated-bits",
            "benchmark_id": str(benchmark.id),
        },
        evaluation_protocol={"kind": "test"},
        evaluation_seed=101,
        evaluation_curriculum={"kind": "test", "rungs": []},
        throughput={
            "checkpoint_evaluation": {
                "inference_cost_measurement": _cost_measurement_record(),
                "inference_cost_sample_count": 1,
            }
        },
    )

    assert bundle.to_record()["program_graph"] == program_graph


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


def _digits_benchmark_manifest():
    return load_benchmark(_repository_root() / "src/leibniz/benchmarks/digits").manifest


def _cost_measurement_record() -> dict[str, object]:
    return {
        "cost_model_id": "test-cost-model",
        "abstract_flops": 0,
        "per_op": [],
        "moved_elements": 0,
        "movement": [],
        "unmodeled_operations": [],
        "operation_count": 0,
        "operation_trace": [],
        "wall_seconds": 0.0,
        "tensor_device": "cpu",
        "execution_mode": "measured",
        "operation_stream_source": "test",
        "operations_executed": True,
    }


def _fixture(relative: str) -> Path:
    return Path(__file__).parent / relative


def _repository_root() -> Path:
    return Path(__file__).parents[1]
