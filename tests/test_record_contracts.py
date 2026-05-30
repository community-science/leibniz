from leibniz.contracts import ContractObject, RuntimeProjection
from leibniz.record_contracts import record_contract_set_from_package
from leibniz.records import record_specs_from_package_contract


def test_record_contract_exposes_generic_projection_boundaries() -> None:
    contract = record_contract_set_from_package(
        "leibniz.contract_artifacts",
        "work_queue_items",
        description="work queue item record contracts",
    ).require_record("work_queue_item")

    assert isinstance(contract, ContractObject)
    assert contract.contract_name == "work_queue_item"
    assert contract.runtime_projections() == (
        RuntimeProjection(
            contract_name="work_queue_item",
            surface="python-record-validation",
            target="leibniz.records.RecordSpec",
            content={
                "fields": (
                    "format",
                    "format_version",
                    "id",
                    "benchmark_id",
                    "proposal_id",
                    "candidate_id",
                    "proposal_set_path",
                    "command",
                    "status",
                    "sequence",
                    "run_id",
                    "measurement_dataset_path",
                    "error",
                ),
                "allow_unknown": False,
            },
        ),
    )

    typescript_projection = contract.typescript_runtime_projection(
        exported_type="WorkQueueItemRecord",
        parser_name="parseWorkQueueItem",
        error_name="WorkQueueTransportError",
    )

    assert typescript_projection.surface == "typescript-record-parser"
    assert typescript_projection.target == "WorkQueueItemRecord"
    assert isinstance(typescript_projection.content, str)
    assert "export function parseWorkQueueItem" in typescript_projection.content


def test_record_contract_owns_conformance_cases_and_source_graph_facts() -> None:
    contract = record_contract_set_from_package(
        "leibniz.contract_artifacts",
        "work_queue_items",
        description="work queue item record contracts",
    ).require_record("work_queue_item")
    spec = record_specs_from_package_contract(
        "leibniz.contract_artifacts",
        "work_queue_items",
        description="work queue item record contracts",
    )["work_queue_item"]

    conformance_cases = contract.conformance_cases()

    assert conformance_cases[0].expected_valid
    assert spec.collect_violations(conformance_cases[0].record) == ()
    assert not conformance_cases[1].expected_valid
    assert spec.collect_violations(conformance_cases[1].record)
    assert contract.source_graph_facts() == (
        {
            "kind": "record-contract",
            "name": "work_queue_item",
            "fields": (
                "format",
                "format_version",
                "id",
                "benchmark_id",
                "proposal_id",
                "candidate_id",
                "proposal_set_path",
                "command",
                "status",
                "sequence",
                "run_id",
                "measurement_dataset_path",
                "error",
            ),
            "allow_unknown": False,
        },
    )
