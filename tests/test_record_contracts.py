from leibniz.architectures import ArchitectureComponent
from leibniz.contracts import ContractObject, RuntimeProjection


def test_record_contract_exposes_generic_projection_boundaries() -> None:
    contract = ArchitectureComponent.record_contract()

    assert isinstance(contract, ContractObject)
    assert contract.contract_name == "architecture_component"
    assert contract.runtime_projections() == (
        RuntimeProjection(
            contract_name="architecture_component",
            surface="python-record-validation",
            target="leibniz.records.RecordSpec",
            content={
                "fields": (
                    "kind",
                    "parameters",
                ),
                "allow_unknown": False,
            },
        ),
    )

    typescript_projection = contract.typescript_runtime_projection(
        exported_type="ArchitectureComponentRecord",
        parser_name="parseArchitectureComponent",
        error_name="ArchitectureComponentTransportError",
    )

    assert typescript_projection.surface == "typescript-record-parser"
    assert typescript_projection.target == "ArchitectureComponentRecord"
    assert isinstance(typescript_projection.content, str)
    assert "export function parseArchitectureComponent" in typescript_projection.content


def test_record_contract_owns_conformance_cases_and_source_graph_facts() -> None:
    contract = ArchitectureComponent.record_contract()
    spec = ArchitectureComponent.record_spec()

    conformance_cases = contract.conformance_cases()

    assert conformance_cases[0].expected_valid
    assert spec.collect_violations(conformance_cases[0].record) == ()
    assert not conformance_cases[1].expected_valid
    assert spec.collect_violations(conformance_cases[1].record)
    assert contract.source_graph_facts() == (
        {
            "kind": "record-contract",
            "name": "architecture_component",
            "fields": (
                "kind",
                "parameters",
            ),
            "allow_unknown": False,
        },
    )
