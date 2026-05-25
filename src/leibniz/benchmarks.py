"""Benchmark declarations for finite-answer scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from leibniz.answers import FiniteAnswerScoringBundle
from leibniz.identifiers import ProtocolIdentifier, require_unreleased_identifier
from leibniz.records import RecordSpec, RecordValidationError, required, validate_record

__all__ = [
    "BenchmarkDeclaration",
    "BenchmarkDeclarationValidationError",
]

_oracle_acceptance_id = ProtocolIdentifier.parse("core.finite-answer-accepted-event@0.1.0")
_prediction_interface_id = ProtocolIdentifier.parse(
    "core.finite-probability-measure-prediction@0.1.0"
)
_score_functional_id = ProtocolIdentifier.parse("core.negative-log-accepted-mass@0.1.0")
_evidence_bundle_id = ProtocolIdentifier.parse("core.finite-answer-scoring-bundle@0.1.0")

_benchmark_declaration_record = RecordSpec(
    fields={
        "id": required("identifier"),
        "answer_space_id": required("identifier"),
        "oracle_acceptance_id": required("identifier"),
        "prediction_interface_id": required("identifier"),
        "score_functional_id": required("identifier"),
        "evidence_bundle_id": required("identifier"),
    }
)


class BenchmarkDeclarationValidationError(ValueError):
    """Raised when a benchmark declaration is invalid."""


@dataclass(frozen=True, slots=True)
class BenchmarkDeclaration:
    """A finite-answer benchmark scoring declaration."""

    id: ProtocolIdentifier
    answer_space_id: ProtocolIdentifier
    oracle_acceptance_id: ProtocolIdentifier = _oracle_acceptance_id
    prediction_interface_id: ProtocolIdentifier = _prediction_interface_id
    score_functional_id: ProtocolIdentifier = _score_functional_id
    evidence_bundle_id: ProtocolIdentifier = _evidence_bundle_id

    def __post_init__(self) -> None:
        try:
            require_unreleased_identifier(self.id)
        except ValueError as error:
            raise BenchmarkDeclarationValidationError(str(error)) from error
        _require_identifier(
            field="oracle_acceptance_id",
            actual=self.oracle_acceptance_id,
            expected=_oracle_acceptance_id,
        )
        _require_identifier(
            field="prediction_interface_id",
            actual=self.prediction_interface_id,
            expected=_prediction_interface_id,
        )
        _require_identifier(
            field="score_functional_id",
            actual=self.score_functional_id,
            expected=_score_functional_id,
        )
        _require_identifier(
            field="evidence_bundle_id",
            actual=self.evidence_bundle_id,
            expected=_evidence_bundle_id,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BenchmarkDeclaration:
        try:
            validated = validate_record(record, _benchmark_declaration_record)
        except RecordValidationError as error:
            raise BenchmarkDeclarationValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            answer_space_id=_as_identifier(
                validated["answer_space_id"],
                field="answer_space_id",
            ),
            oracle_acceptance_id=_as_identifier(
                validated["oracle_acceptance_id"],
                field="oracle_acceptance_id",
            ),
            prediction_interface_id=_as_identifier(
                validated["prediction_interface_id"],
                field="prediction_interface_id",
            ),
            score_functional_id=_as_identifier(
                validated["score_functional_id"],
                field="score_functional_id",
            ),
            evidence_bundle_id=_as_identifier(
                validated["evidence_bundle_id"],
                field="evidence_bundle_id",
            ),
        )

    def validate_bundle(self, bundle: FiniteAnswerScoringBundle) -> None:
        if bundle.answer_space.id != self.answer_space_id:
            raise BenchmarkDeclarationValidationError(
                f"bundle answer_space_id {bundle.answer_space.id} does not match "
                f"{self.answer_space_id}"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "answer_space_id": str(self.answer_space_id),
            "oracle_acceptance_id": str(self.oracle_acceptance_id),
            "prediction_interface_id": str(self.prediction_interface_id),
            "score_functional_id": str(self.score_functional_id),
            "evidence_bundle_id": str(self.evidence_bundle_id),
        }


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise BenchmarkDeclarationValidationError(f"{field}: expected parsed identifier")
    return value


def _require_identifier(
    *,
    field: str,
    actual: ProtocolIdentifier,
    expected: ProtocolIdentifier,
) -> None:
    if actual != expected:
        raise BenchmarkDeclarationValidationError(f"{field} {actual} does not match {expected}")
