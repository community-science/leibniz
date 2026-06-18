"""Round-trip and validation tests for the program-neutral result schema."""

from __future__ import annotations

import pytest

from leibniz.content import ContentDigest
from leibniz.cost_metrology import CostMeasurement, OperationCostRecord
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.result_schema import (
    ArtifactReference,
    BenchmarkMetadataDocument,
    BenchmarkMetadataRecord,
    EvaluationDocument,
    EvaluationLineage,
    EvaluationRecord,
    ResultSchemaError,
    SubmissionDocument,
    SubmissionRecord,
)


def _program_graph() -> dict[str, object]:
    return {
        "kind": "program-graph",
        "nodes": [{"id": "predict", "kind": "spectral-solver"}],
        "edges": [],
        "inputs": ["state"],
        "outputs": ["prediction"],
    }


def _cost_measurement() -> CostMeasurement:
    return CostMeasurement(
        cost_model_id=CostMeasurement.tensor_runtime_cost_model_id(),
        abstract_flops=120,
        per_op=(
            OperationCostRecord(
                name="test.op",
                calls=1,
                abstract_flops=120,
                output_elements=1,
                operation_class="elementwise",
                dtype="fp32",
            ),
        ),
        moved_elements=0,
        movement=(),
        unmodeled_operations=(),
        operation_count=0,
        operation_trace=(),
        wall_seconds=0.0,
        tensor_device="cpu",
    )


def _capability_map(value: float) -> dict[str, object]:
    return {
        "kind": "measure-weighted-partition-competence-integral-v1",
        "value": value,
        "root": {
            "region": {"kind": "whole-space"},
            "mean_competence": 0.5,
            "measure": 1.0,
        },
        "refinement_ladder": [{"depth": 0, "value": value}],
    }


def _deterministic_submission() -> SubmissionRecord:
    return SubmissionRecord(
        id=ProtocolIdentifier.parse("submissions.ks-spectral-solver@0.1.0"),
        program_graph=_program_graph(),
    )


def _learned_submission() -> SubmissionRecord:
    weights_digest = ContentDigest.from_value({"weights": "stub"})
    return SubmissionRecord(
        id=ProtocolIdentifier.parse("submissions.ks-learned-residual@0.1.0"),
        program_graph=_program_graph(),
        model_inspection={"kind": "model-inspection", "components": []},
        fitted_parameters=(
            ArtifactReference(
                kind="fitted-parameters",
                digest=weights_digest,
                path="submissions/ks-learned-residual/weights.pt",
                description="trained residual head",
            ),
        ),
        training_provenance={
            "kind": "training-provenance",
            "protocol": {"objective": "validated-bits", "optimizer": "adam"},
            "validation_checks": 4,
        },
    )


def _evaluation(submission: SubmissionRecord) -> EvaluationRecord:
    benchmark = _benchmark_metadata()
    value = 3.5
    return EvaluationRecord(
        id=ProtocolIdentifier.parse("evaluations.ks-spectral-solver.ks@0.1.0"),
        submission_id=submission.id,
        benchmark_id=benchmark.id,
        validated_bits=value,
        capability_map=_capability_map(value),
        cost=_cost_measurement(),
        lineage=EvaluationLineage(
            submission_digest=submission.digest,
            benchmark_digest=benchmark.digest,
            measurement_dataset_digest=ContentDigest.from_value({"dataset": "stub"}),
        ),
        evaluation_seed=7,
        converged=True,
        evidence_budget_limited=False,
    )


def _benchmark_metadata() -> BenchmarkMetadataRecord:
    return BenchmarkMetadataRecord(
        id=ProtocolIdentifier.parse("benchmarks.ks.generator@0.1.0"),
        name="benchmarks.ks.generator",
        representation="field-valued",
        competence={"kind": "convergence-resolved-bits"},
        baseline={"kind": "persistence"},
        structural_type="dynamical-amplification",
    )


def test_deterministic_submission_round_trips() -> None:
    submission = _deterministic_submission()
    assert not submission.is_learned
    restored = SubmissionDocument.from_bytes(
        canonical_document_bytes(submission.to_record())
    )
    assert restored.submission == submission
    assert restored.digest == submission.digest


def test_learned_submission_round_trips_with_provenance() -> None:
    submission = _learned_submission()
    assert submission.is_learned
    assert submission.fitted_parameters
    assert submission.training_provenance is not None
    restored = SubmissionDocument.from_bytes(
        canonical_document_bytes(submission.to_record())
    )
    assert restored.submission == submission


def test_benchmark_metadata_round_trips() -> None:
    benchmark = _benchmark_metadata()
    restored = BenchmarkMetadataDocument.from_bytes(
        canonical_document_bytes(benchmark.to_record())
    )
    assert restored.benchmark == benchmark


def test_evaluation_round_trips_and_pivots_by_id() -> None:
    submission = _deterministic_submission()
    evaluation = _evaluation(submission)
    restored = EvaluationDocument.from_bytes(
        canonical_document_bytes(evaluation.to_record())
    )
    assert restored.evaluation == evaluation
    # The console pivots the matrix on these ids without opening the bundle.
    assert str(restored.evaluation.submission_id) == "submissions.ks-spectral-solver@0.1.0"
    assert str(restored.evaluation.benchmark_id) == "benchmarks.ks.generator@0.1.0"
    assert restored.evaluation.cost == _cost_measurement()


def test_evaluation_rejects_capability_map_value_mismatch() -> None:
    submission = _deterministic_submission()
    benchmark = _benchmark_metadata()
    with pytest.raises(ResultSchemaError, match="capability_map.value must match"):
        EvaluationRecord(
            id=ProtocolIdentifier.parse("evaluations.mismatch.ks@0.1.0"),
            submission_id=submission.id,
            benchmark_id=benchmark.id,
            validated_bits=3.5,
            capability_map=_capability_map(2.0),
            cost=_cost_measurement(),
            lineage=EvaluationLineage(
                submission_digest=submission.digest,
                benchmark_digest=benchmark.digest,
            ),
            evaluation_seed=0,
            converged=True,
            evidence_budget_limited=False,
        )


def test_evaluation_rejects_budget_limited_converged_status() -> None:
    submission = _deterministic_submission()
    benchmark = _benchmark_metadata()
    with pytest.raises(ResultSchemaError, match="budget-limited"):
        EvaluationRecord(
            id=ProtocolIdentifier.parse("evaluations.status.ks@0.1.0"),
            submission_id=submission.id,
            benchmark_id=benchmark.id,
            validated_bits=3.5,
            capability_map=_capability_map(3.5),
            cost=_cost_measurement(),
            lineage=EvaluationLineage(
                submission_digest=submission.digest,
                benchmark_digest=benchmark.digest,
            ),
            evaluation_seed=0,
            converged=True,
            evidence_budget_limited=True,
        )


def test_submission_rejects_wrong_namespace() -> None:
    with pytest.raises(ResultSchemaError, match="submissions namespace"):
        SubmissionRecord(
            id=ProtocolIdentifier.parse("models.bad@0.1.0"),
            program_graph=_program_graph(),
        )
