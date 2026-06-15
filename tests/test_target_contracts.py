import math

import pytest

from leibniz.documents import canonical_document_bytes, load_object_document
from leibniz.target_contracts import (
    BaselinePredictor,
    CompetenceFunctional,
    TargetContract,
    TargetContractError,
)


def test_finite_outcome_target_contract_round_trips_through_document_boundary() -> None:
    contract = TargetContract.finite_outcome(("zero", "one"))

    record = load_object_document(
        canonical_document_bytes(contract.to_record()),
        description="target contract",
    )
    parsed = TargetContract.from_record(record)

    assert parsed == contract
    assert parsed.expected_output_shape(None) == (2,)
    assert parsed.chance_mass() == 0.5


def test_field_valued_target_contract_round_trips_and_declares_shape() -> None:
    contract = TargetContract(
        kind="field-valued",
        outcome_ids=None,
        loss_id="relative-l2",
        competence=CompetenceFunctional(
            kind="ambient-certified-bits",
            parameters={"residual_operator_id": "operators.example@0.1.0"},
        ),
        baseline=BaselinePredictor(kind="zero-field"),
    )

    parsed = TargetContract.from_record(contract.to_record())

    assert parsed == contract
    assert parsed.expected_output_shape((2, 3)) == (2, 3)
    assert parsed.chance_mass() is None


@pytest.mark.parametrize(
    "record",
    [
        {
            "kind": "finite-outcome",
            "loss_id": "cross-entropy",
            "competence": {"kind": "above-chance-accepted-mass"},
            "baseline": {"kind": "uniform-outcome-mass"},
        },
        {
            "kind": "finite-outcome",
            "outcome_ids": ["a", "a"],
            "loss_id": "cross-entropy",
            "competence": {"kind": "above-chance-accepted-mass"},
            "baseline": {"kind": "uniform-outcome-mass"},
        },
        {
            "kind": "field-valued",
            "outcome_ids": ["a"],
            "loss_id": "mse",
            "competence": {
                "kind": "mass-within-resolution",
                "parameters": {"residual_operator_id": "op"},
            },
            "baseline": {"kind": "zero-field"},
        },
        {
            "kind": "field-valued",
            "loss_id": "cross-entropy",
            "competence": {
                "kind": "mass-within-resolution",
                "parameters": {"residual_operator_id": "op"},
            },
            "baseline": {"kind": "zero-field"},
        },
        {
            "kind": "finite-outcome",
            "outcome_ids": ["a"],
            "loss_id": "cross-entropy",
            "competence": {
                "kind": "above-chance-accepted-mass",
                "parameters": {"threshold": 0.5},
            },
            "baseline": {"kind": "uniform-outcome-mass"},
        },
    ],
)
def test_target_contract_rejects_invalid_invariants(record: dict[str, object]) -> None:
    with pytest.raises(TargetContractError):
        TargetContract.from_record(record)


def test_field_valued_target_contract_requires_field_shape() -> None:
    contract = TargetContract(
        kind="field-valued",
        outcome_ids=None,
        loss_id="mse",
        competence=CompetenceFunctional(
            kind="mass-within-resolution",
            parameters={"residual_operator_id": "op"},
        ),
        baseline=BaselinePredictor(kind="persistence"),
    )

    with pytest.raises(TargetContractError, match="field_shape"):
        contract.expected_output_shape(None)


def test_target_contract_rejects_nonfinite_parameters() -> None:
    with pytest.raises(TargetContractError, match="finite"):
        CompetenceFunctional(
            kind="mass-within-resolution",
            parameters={
                "residual_operator_id": "op",
                "scale": math.inf,
            },
        )
