"""Benchmark-declared target contracts for training and evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, cast

from leibniz.records import FieldSpec, RecordExtractor, RecordSpec

__all__ = [
    "BaselinePredictor",
    "CompetenceFunctional",
    "TargetContract",
    "TargetContractError",
]

_target_contract_kind = Literal["finite-outcome", "field-valued", "inverse"]
_competence_parameter_value = int | float | str
_baseline_parameter_value = int | float | str

_parameter_record = FieldSpec(kind="record", required=False)
_functional_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="string"),
        "parameters": _parameter_record,
    }
)
_target_contract_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="string"),
        "outcome_ids": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="string"),
            required=False,
        ),
        "loss_id": FieldSpec(kind="string"),
        "competence": FieldSpec(kind="record"),
        "baseline": FieldSpec(kind="record"),
    }
)


class TargetContractError(ValueError):
    """Raised when a target contract record is invalid."""


_extract = RecordExtractor(error_type=TargetContractError)


@dataclass(frozen=True, slots=True)
class CompetenceFunctional:
    """Benchmark-declared competence functional."""

    kind: str
    parameters: Mapping[str, _competence_parameter_value] = field(
        default_factory=lambda: {}
    )

    def __post_init__(self) -> None:
        if not self.kind:
            raise TargetContractError("competence kind must be nonempty")
        parameters: dict[str, _competence_parameter_value] = dict(self.parameters)
        _validate_parameter_mapping(parameters, field="competence.parameters")
        if "threshold" in parameters:
            raise TargetContractError("competence parameters must not include threshold")
        if self.kind == "above-chance-accepted-mass":
            if parameters:
                raise TargetContractError(
                    "above-chance-accepted-mass competence does not accept parameters"
                )
        elif self.kind in {
            "ambient-certified-bits",
            "convergence-resolved-bits",
            "mass-within-resolution",
        }:
            residual_operator_id = parameters.get("residual_operator_id")
            if not isinstance(residual_operator_id, str) or not residual_operator_id:
                raise TargetContractError(
                    f"{self.kind} competence requires residual_operator_id"
                )
        else:
            raise TargetContractError(f"unsupported competence kind: {self.kind}")
        object.__setattr__(self, "parameters", parameters)

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> CompetenceFunctional:
        try:
            validated = _functional_record.validate(record)
        except ValueError as error:
            raise TargetContractError(str(error)) from error
        return cls(
            kind=_extract.non_empty_string(validated["kind"], "kind"),
            parameters=cast(
                Mapping[str, _competence_parameter_value],
                validated.get("parameters", {}),
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {"kind": self.kind}
        if self.parameters:
            record["parameters"] = dict(self.parameters)
        return record


@dataclass(frozen=True, slots=True)
class BaselinePredictor:
    """Benchmark-declared baseline predictor."""

    kind: str
    parameters: Mapping[str, _baseline_parameter_value] = field(
        default_factory=lambda: {}
    )

    def __post_init__(self) -> None:
        if not self.kind:
            raise TargetContractError("baseline kind must be nonempty")
        parameters: dict[str, _baseline_parameter_value] = dict(self.parameters)
        _validate_parameter_mapping(parameters, field="baseline.parameters")
        if self.kind in {
            "uniform-outcome-mass",
            "zero-field",
            "persistence",
            "uninformed-latent-prior",
        }:
            if parameters:
                raise TargetContractError(f"{self.kind} baseline does not accept parameters")
        else:
            raise TargetContractError(f"unsupported baseline kind: {self.kind}")
        object.__setattr__(self, "parameters", parameters)

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BaselinePredictor:
        try:
            validated = _functional_record.validate(record)
        except ValueError as error:
            raise TargetContractError(str(error)) from error
        return cls(
            kind=_extract.non_empty_string(validated["kind"], "kind"),
            parameters=cast(
                Mapping[str, _baseline_parameter_value],
                validated.get("parameters", {}),
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {"kind": self.kind}
        if self.parameters:
            record["parameters"] = dict(self.parameters)
        return record


@dataclass(frozen=True, slots=True)
class TargetContract:
    """Benchmark target contract consumed by generic runners."""

    kind: _target_contract_kind
    outcome_ids: tuple[str, ...] | None
    loss_id: str
    competence: CompetenceFunctional
    baseline: BaselinePredictor

    def __post_init__(self) -> None:
        if self.kind not in {"finite-outcome", "field-valued", "inverse"}:
            raise TargetContractError(f"unsupported target contract kind: {self.kind}")
        if not self.loss_id:
            raise TargetContractError("loss_id must be nonempty")
        if self.kind == "finite-outcome":
            if self.outcome_ids is None or not self.outcome_ids:
                raise TargetContractError("finite-outcome contract requires outcome_ids")
            if len(set(self.outcome_ids)) != len(self.outcome_ids):
                raise TargetContractError("outcome_ids must be unique")
            if any(not outcome_id for outcome_id in self.outcome_ids):
                raise TargetContractError("outcome_ids must be nonempty strings")
            if self.loss_id != "cross-entropy":
                raise TargetContractError("finite-outcome contract requires cross-entropy")
            if self.competence.kind != "above-chance-accepted-mass":
                raise TargetContractError(
                    "finite-outcome contract requires above-chance-accepted-mass"
                )
            if self.baseline.kind != "uniform-outcome-mass":
                raise TargetContractError(
                    "finite-outcome contract requires uniform-outcome-mass baseline"
                )
        elif self.kind == "field-valued":
            if self.outcome_ids is not None:
                raise TargetContractError("field-valued contract does not accept outcome_ids")
            if self.loss_id not in {"mse", "relative-l2", "equation-residual"}:
                raise TargetContractError(f"unsupported field-valued loss_id: {self.loss_id}")
            if self.competence.kind not in {
                "ambient-certified-bits",
                "convergence-resolved-bits",
                "mass-within-resolution",
            }:
                raise TargetContractError(
                    "field-valued contract requires ambient-certified-bits, "
                    "convergence-resolved-bits, or mass-within-resolution"
                )
            if self.baseline.kind not in {"zero-field", "persistence"}:
                raise TargetContractError(
                    "field-valued contract requires zero-field or persistence baseline"
                )
        else:
            if self.outcome_ids is not None:
                raise TargetContractError("inverse contract does not accept outcome_ids")
            if self.loss_id != "reconstruction":
                raise TargetContractError("inverse contract requires reconstruction loss")
            if self.competence.kind != "ambient-certified-bits":
                raise TargetContractError(
                    "inverse contract requires ambient-certified-bits competence"
                )
            if self.baseline.kind != "uninformed-latent-prior":
                raise TargetContractError(
                    "inverse contract requires uninformed-latent-prior baseline"
                )
            identity_count = self.competence.parameters.get("identity_count")
            nuisance_dimension = self.competence.parameters.get("nuisance_dimension")
            if (
                type(identity_count) is not int
                or identity_count < 2
                or type(nuisance_dimension) is not int
                or nuisance_dimension < 1
            ):
                raise TargetContractError(
                    "inverse contract requires integer identity_count >= 2 and "
                    "nuisance_dimension >= 1"
                )

    @classmethod
    def finite_outcome(cls, outcome_ids: tuple[str, ...]) -> TargetContract:
        return cls(
            kind="finite-outcome",
            outcome_ids=outcome_ids,
            loss_id="cross-entropy",
            competence=CompetenceFunctional(kind="above-chance-accepted-mass"),
            baseline=BaselinePredictor(kind="uniform-outcome-mass"),
        )

    @classmethod
    def inverse_latent(
        cls,
        *,
        identity_count: int,
        nuisance_dimension: int,
        residual_operator_id: str,
    ) -> TargetContract:
        return cls(
            kind="inverse",
            outcome_ids=None,
            loss_id="reconstruction",
            competence=CompetenceFunctional(
                kind="ambient-certified-bits",
                parameters={
                    "residual_operator_id": residual_operator_id,
                    "identity_count": identity_count,
                    "nuisance_dimension": nuisance_dimension,
                },
            ),
            baseline=BaselinePredictor(kind="uninformed-latent-prior"),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> TargetContract:
        try:
            validated = _target_contract_record.validate(record)
        except ValueError as error:
            raise TargetContractError(str(error)) from error
        outcome_ids = (
            None
            if "outcome_ids" not in validated
            else tuple(
                _extract.non_empty_string(outcome_id, "outcome_ids")
                for outcome_id in cast(tuple[object, ...], validated["outcome_ids"])
            )
        )
        return cls(
            kind=cast(
                _target_contract_kind,
                _extract.non_empty_string(validated["kind"], "kind"),
            ),
            outcome_ids=outcome_ids,
            loss_id=_extract.non_empty_string(validated["loss_id"], "loss_id"),
            competence=CompetenceFunctional.from_record(
                cast(Mapping[str, object], validated["competence"])
            ),
            baseline=BaselinePredictor.from_record(
                cast(Mapping[str, object], validated["baseline"])
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "kind": self.kind,
            "loss_id": self.loss_id,
            "competence": self.competence.to_record(),
            "baseline": self.baseline.to_record(),
        }
        if self.outcome_ids is not None:
            record["outcome_ids"] = list(self.outcome_ids)
        return record

    def expected_output_shape(
        self,
        field_shape: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if self.kind == "finite-outcome":
            if self.outcome_ids is None:
                raise TargetContractError("finite-outcome contract requires outcome_ids")
            return (len(self.outcome_ids),)
        if self.kind == "inverse":
            identity_count = self.competence.parameters.get("identity_count")
            nuisance_dimension = self.competence.parameters.get("nuisance_dimension")
            if type(identity_count) is not int or type(nuisance_dimension) is not int:
                raise TargetContractError("inverse contract requires latent dimensions")
            return (identity_count + nuisance_dimension,)
        if field_shape is None:
            raise TargetContractError("field-valued contract requires field_shape")
        return field_shape

    def chance_mass(self) -> float | None:
        if self.kind in {"field-valued", "inverse"}:
            return None
        if self.outcome_ids is None or not self.outcome_ids:
            raise TargetContractError("finite-outcome contract requires outcome_ids")
        return 1.0 / len(self.outcome_ids)


def _validate_parameter_mapping(
    parameters: Mapping[str, object],
    *,
    field: str,
) -> None:
    for key, value in parameters.items():
        if not key:
            raise TargetContractError(f"{field} keys must be nonempty strings")
        if not isinstance(value, int | float | str) or isinstance(value, bool):
            raise TargetContractError(
                f"{field}.{key} must be an int, float, or string"
            )
        if isinstance(value, float) and not math.isfinite(value):
            raise TargetContractError(f"{field}.{key} must be finite")
