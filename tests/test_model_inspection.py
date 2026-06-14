from collections.abc import Callable
from pathlib import Path
from typing import cast

from leibniz.architectures import ArchitectureManifest, ArchitectureManifestDocument
from leibniz.artifacts import ArtifactReference, reference_for_record
from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDatasetDocument, MeasurementDocument
from leibniz.model_inspection import (
    ModelInspectionDocument,
    ModelInspectionRecord,
    ModelInspectionValidationError,
)
from leibniz.model_interfaces import ModelInterface
from leibniz.model_manifests import ModelArtifactManifest, ModelExecutionFamily
from leibniz.outcomes import OutcomeSpace
from leibniz.submissions import SubmissionPackageManifest

_fixtures_root = Path(__file__).parent / "fixtures"


def test_model_inspection_derives_architecture_components_and_costs() -> None:
    inspection = ModelInspectionRecord.from_architecture(
        id=ProtocolIdentifier.parse("model-inspections.digits-pool@0.1.0"),
        architecture_manifest=_architecture_manifest(),
    )
    architecture_reference = reference_for_record(
        kind="architecture-manifest",
        record=_architecture_manifest().to_record(),
    )

    assert inspection.input_shape == (1, 24, 24)
    assert inspection.model_source == architecture_reference
    assert inspection.output_shape == (10,)
    assert tuple(component.kind for component in inspection.components) == (
        "adaptive-pooling",
        "flatten",
        "dense",
    )
    assert inspection.components[0].input_shape == (1, 24, 24)
    assert inspection.components[0].output_shape == (1, 2, 2)
    assert inspection.components[0].operator is not None
    assert inspection.components[0].operator["kind"] == "local-aggregation"
    assert inspection.components[0].operator["aliases"] == ["adaptive-pooling"]
    assert inspection.components[0].parameter_count == 0
    assert inspection.components[1].input_shape == (1, 2, 2)
    assert inspection.components[1].output_shape == (4,)
    assert inspection.components[1].operator is not None
    assert inspection.components[1].operator["kind"] == "rank-collapse"
    assert inspection.components[1].parameter_count == 0
    assert inspection.components[2].input_shape == (4,)
    assert inspection.components[2].output_shape == (10,)
    assert inspection.components[2].operator is not None
    assert inspection.components[2].operator["kind"] == "affine-readout"
    assert inspection.components[2].parameter_count == 50
    assert inspection.components[2].storage_bytes == 200
    assert inspection.cost_summary.component_count == 3
    assert inspection.cost_summary.parameter_count == 50
    assert inspection.cost_summary.storage_bytes == 200
    assert inspection.cost_summary.inference_cost_measurement is not None
    assert inspection.cost_summary.inference_cost_measurement.abstract_flops == 656
    assert inspection.cost_summary.inference_cost_measurement.execution_mode == "dry-run"
    assert not inspection.cost_summary.inference_cost_measurement.operations_executed
    assert inspection.cost_summary.inference_cost_sample_count == 1
    assert inspection.cost_summary.unknown_parameter_components == ()
    assert inspection.cost_summary.unknown_cost_components == ()
    assert inspection.architecture_summary.component_count == 3
    assert inspection.architecture_summary.edge_count == 2
    assert inspection.architecture_summary.input_node_ids == ("component-0",)
    assert inspection.architecture_summary.output_node_ids == ("component-2",)
    assert inspection.architecture_summary.component_kinds == (
        "adaptive-pooling",
        "flatten",
        "dense",
    )
    assert inspection.architecture_summary.unsupported_parameter_components == ()
    assert inspection.architecture_summary.unsupported_cost_components == ()
    assert inspection.architecture_trace.input_shape == (1, 24, 24)
    assert inspection.architecture_trace.output_shape == (10,)
    assert [stage.operator_kind for stage in inspection.architecture_trace.stages] == [
        "local-aggregation",
        "rank-collapse",
        "affine-readout",
    ]
    assert inspection.architecture_trace.stages[0].descriptor_axes == {
        "aggregation_law": "mean",
        "parameter_sharing": "none",
        "projection_law": "equal-output-partition",
        "state": "fixed",
        "support": "local-window",
        "tensor_relation": "aggregation",
    }
    assert inspection.architecture_trace.stages[0].shape_law == (
        "preserve-prefix-replace-trailing-axes"
    )
    assert inspection.architecture_trace.stages[2].parameter_count == 50
    assert [node.id for node in inspection.architecture_graph.nodes] == [
        "component-0",
        "component-1",
        "component-2",
    ]
    assert [
        (edge.source_node_id, edge.target_node_id, edge.kind)
        for edge in inspection.architecture_graph.edges
    ] == [
        ("component-0", "component-1", "data-flow"),
        ("component-1", "component-2", "data-flow"),
    ]
    assert tuple(node.component.kind for node in inspection.architecture_graph.nodes) == (
        "adaptive-pooling",
        "flatten",
        "dense",
    )
    assert [evidence.node_path for evidence in inspection.node_evidence] == [
        ("component-0",),
        ("component-1",),
        ("component-2",),
    ]
    assert inspection.node_evidence[0].claim_kinds == (
        "architecture-structure",
        "operator-semantics",
        "resource-accounting",
    )
    assert inspection.digest == ContentDigest.from_value(inspection.to_record())


def test_model_inspection_includes_model_manifest_sources() -> None:
    record = _model_manifest_record()
    record["model_source"] = _program_graph_reference().to_record()
    model_manifest = ModelArtifactManifest.from_record(
        record,
        architecture_manifest=_architecture_manifest(),
        model_interface=_model_interface(),
    )

    inspection = ModelInspectionRecord.from_model_manifest(
        id=ProtocolIdentifier.parse("model-inspections.boolean-digits-pool@0.1.0"),
        model_manifest=model_manifest,
        architecture_manifest=_architecture_manifest(),
    )

    assert inspection.model_manifest == reference_for_record(
        kind="model-manifest",
        record=model_manifest.to_record(),
    )
    assert inspection.model_source == _program_graph_reference()
    assert inspection.model_artifacts == (_checkpoint_reference(),)
    assert inspection.training_provenance == (_training_reference(),)
    assert {
        reference.kind
        for evidence in inspection.node_evidence
        for reference in evidence.evidence_artifacts
    } == {
        "architecture-manifest",
        "model-checkpoint",
        "model-manifest",
        "training-provenance",
    }


def test_model_inspection_rejects_model_manifest_architecture_mismatch() -> None:
    model_manifest = ModelArtifactManifest.from_record(_model_manifest_record())
    altered_architecture = ArchitectureManifest.from_record(
        {
            "input_shape": [1, 16, 16],
            "output_shape": [10],
            "layers": [{"kind": "dense", "parameters": {"out": 10}}],
        }
    )

    error = capture_model_inspection_error(
        lambda: ModelInspectionRecord.from_model_manifest(
            id=ProtocolIdentifier.parse("model-inspections.boolean-digits-pool@0.1.0"),
            model_manifest=model_manifest,
            architecture_manifest=altered_architecture,
        )
    )

    assert str(error) == "architecture reference does not match architecture manifest"


def test_model_inspection_includes_submission_package_sources() -> None:
    submission_package = SubmissionPackageManifest.from_record(_submission_package_record())

    inspection = ModelInspectionRecord.from_submission_package(
        id=ProtocolIdentifier.parse("model-inspections.boolean-submission@0.1.0"),
        submission_package=submission_package,
    )

    assert inspection.submission_package == reference_for_record(
        kind="submission-package",
        record=submission_package.to_record(),
    )
    assert inspection.benchmark_manifest == reference_for_record(
        kind="benchmark-manifest",
        record=submission_package.benchmark_manifest.to_record(),
    )
    assert inspection.measurement_dataset == ArtifactReference(
        kind="measurement-dataset",
        content_digest=submission_package.measurement_dataset.digest,
    )
    assert inspection.model_artifacts == (
        ArtifactReference(
            kind="submission-artifact",
            protocol_id=submission_package.artifacts[0].id,
            content_digest=submission_package.artifacts[0].digest,
        ),
    )


def test_model_inspection_round_trips_canonically() -> None:
    inspection = ModelInspectionRecord.from_submission_package(
        id=ProtocolIdentifier.parse("model-inspections.boolean-submission@0.1.0"),
        submission_package=SubmissionPackageManifest.from_record(_submission_package_record()),
    )

    parsed = ModelInspectionRecord.from_record(inspection.to_record())
    document = ModelInspectionDocument.from_bytes(canonical_document_bytes(inspection.to_record()))

    assert parsed == inspection
    assert document.inspection == inspection
    assert document.digest == inspection.digest


def test_model_inspection_rejects_malformed_records() -> None:
    inspection = ModelInspectionRecord.from_architecture(
        id=ProtocolIdentifier.parse("model-inspections.digits-pool@0.1.0"),
        architecture_manifest=_architecture_manifest(),
    )

    record = inspection.to_record()
    record["architecture"] = {"kind": "model-manifest", "protocol_id": "models.other@0.1.0"}
    error = capture_model_inspection_error(lambda: ModelInspectionRecord.from_record(record))
    assert str(error) == (
        "architecture reference must have kind architecture-manifest"
    )

    record = inspection.to_record()
    components = list(record["components"])  # type: ignore[arg-type]
    components[1] = {**components[1], "index": 7}
    record["components"] = components
    error = capture_model_inspection_error(lambda: ModelInspectionRecord.from_record(record))
    assert str(error) == "component indexes must be contiguous"

    record = inspection.to_record()
    trace = dict(cast(dict[str, object], record["architecture_trace"]))
    trace["output_shape"] = [11]
    record["architecture_trace"] = trace
    error = capture_model_inspection_error(lambda: ModelInspectionRecord.from_record(record))
    assert str(error) == "trace final output_shape does not match"

    record = inspection.to_record()
    graph = dict(cast(dict[str, object], record["architecture_graph"]))
    nodes = list(graph["nodes"])  # type: ignore[arg-type]
    nodes[1] = {
        **cast(dict[str, object], nodes[1]),
        "component": {"kind": "other", "parameters": {}},
    }
    graph["nodes"] = nodes
    record["architecture_graph"] = graph
    error = capture_model_inspection_error(lambda: ModelInspectionRecord.from_record(record))
    assert str(error) == "architecture_graph node does not match component"

    record = inspection.to_record()
    summary = dict(cast(dict[str, object], record["architecture_summary"]))
    summary["edge_count"] = 7
    record["architecture_summary"] = summary
    error = capture_model_inspection_error(lambda: ModelInspectionRecord.from_record(record))
    assert str(error) == "architecture_summary does not match graph"

    record = inspection.to_record()
    node_evidence = list(cast(list[object], record["node_evidence"]))
    node_evidence.append(
        {
            **cast(dict[str, object], node_evidence[0]),
            "node_path": ["unknown-node"],
        }
    )
    record["node_evidence"] = node_evidence
    error = capture_model_inspection_error(lambda: ModelInspectionRecord.from_record(record))
    assert str(error) == "node_evidence node_path must start with an architecture graph node"


def _model_manifest_record() -> dict[str, object]:
    return {
        "id": "model-manifests.boolean-digits-pool@0.1.0",
        "architecture": reference_for_record(
            kind="architecture-manifest",
            record=_architecture_manifest().to_record(),
        ).to_record(),
        "interface": reference_for_record(
            kind="model-interface",
            record=_model_interface().to_record(),
        ).to_record(),
        "execution_family": ModelExecutionFamily.reference_runner_pytorch_sequential().to_record(),
        "model_artifacts": [_checkpoint_reference().to_record()],
        "training_provenance": [_training_reference().to_record()],
    }


def _submission_package_record() -> dict[str, object]:
    return {
        "id": "submissions.boolean-digits-pool@0.1.0",
        "benchmark_manifest": _benchmark_document().manifest.to_record(),
        "architecture_manifest": _architecture_manifest().to_record(),
        "measurement_dataset": _dataset_document().dataset.to_record(),
        "artifacts": [
            {
                "id": "artifacts.model-weights@0.1.0",
                "digest": str(ContentDigest.from_value({"weights": [1, 2, 3]})),
                "description": "checkpoint metadata only",
            }
        ],
    }


def _checkpoint_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "model-checkpoint",
            "content_digest": str(ContentDigest.from_value({"weights": [1, 2, 3]})),
        }
    )


def _program_graph_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "program-graph",
            "record_digest": str(ContentDigest.from_value({"nodes": []})),
        }
    )


def _training_reference() -> ArtifactReference:
    return ArtifactReference.from_record(
        {
            "kind": "training-provenance",
            "record_digest": str(ContentDigest.from_value({"optimizer": "declared"})),
        }
    )


def _architecture_manifest() -> ArchitectureManifest:
    return ArchitectureManifestDocument.from_bytes(
        (_fixtures_root / "architecture" / "digits_pool.json").read_bytes()
    ).manifest


def _model_interface() -> ModelInterface:
    return ModelInterface.from_outcome_space(
        id=ProtocolIdentifier.parse("model-interfaces.boolean@0.1.0"),
        outcome_space=OutcomeSpace.from_record(
            {
                "id": "core.boolean-outcome@0.1.0",
                "outcomes": [{"id": "yes"}, {"id": "no"}],
            }
        ),
    )


def _benchmark_document() -> BenchmarkManifestDocument:
    return BenchmarkManifestDocument.from_bytes(
        (_fixtures_root / "finite_outcome" / "manifest.json").read_bytes()
    )


def _dataset_document() -> MeasurementDatasetDocument:
    measurement = MeasurementDocument.from_bytes(
        (_fixtures_root / "finite_outcome" / "measurement.json").read_bytes()
    ).measurement
    return MeasurementDatasetDocument.from_bytes(
        canonical_document_bytes({"measurements": [measurement.to_record()]})
    )


def capture_model_inspection_error(
    action: Callable[[], object],
) -> ModelInspectionValidationError:
    try:
        action()
    except ModelInspectionValidationError as error:
        return error
    raise AssertionError("expected ModelInspectionValidationError")
