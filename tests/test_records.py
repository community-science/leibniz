from collections.abc import Callable

from leibniz.identifiers import ProtocolIdentifier, ProtocolName, SemanticVersion
from leibniz.records import (
    FieldSpec,
    RecordSpec,
    RecordValidationError,
    RecordViolation,
)


def test_record_validation_parses_valid_record() -> None:
    spec = RecordSpec(
        fields={
            "id": FieldSpec(kind="identifier"),
            "kind": FieldSpec(kind="literal", literal="example"),
            "version": FieldSpec(kind="version"),
            "name": FieldSpec(kind="name"),
            "count": FieldSpec(kind="integer"),
            "weight": FieldSpec(kind="number"),
            "active": FieldSpec(kind="boolean"),
            "tags": FieldSpec(kind="sequence", item=FieldSpec(kind="string")),
        }
    )

    record = spec.validate(
        {
            "id": "core.example@0.1.0",
            "kind": "example",
            "version": "0.1.0",
            "name": "core.example",
            "count": 3,
            "weight": 2.5,
            "active": True,
            "tags": ["a", "b"],
        }
    )

    assert record == {
        "id": ProtocolIdentifier.parse("core.example@0.1.0"),
        "kind": "example",
        "version": SemanticVersion.parse("0.1.0"),
        "name": ProtocolName.parse("core.example"),
        "count": 3,
        "weight": 2.5,
        "active": True,
        "tags": ("a", "b"),
    }


def test_record_validation_rejects_missing_unknown_and_wrong_type_fields() -> None:
    spec = RecordSpec(
        fields={
            "id": FieldSpec(kind="identifier"),
            "count": FieldSpec(kind="integer"),
            "description": FieldSpec(kind="string", required=False),
        }
    )

    violations = spec.collect_violations(
        {
            "id": "not an identifier",
            "count": True,
            "extra": "ignored nowhere",
        }
    )

    assert violations == (
        RecordViolation(path=("id",), message="invalid protocol identifier: 'not an identifier'"),
        RecordViolation(path=("count",), message="expected integer"),
        RecordViolation(path=("extra",), message="unknown field"),
    )


def test_record_validation_can_allow_unknown_fields() -> None:
    spec = RecordSpec(fields={"id": FieldSpec(kind="identifier")}, allow_unknown=True)

    record = spec.validate({"id": "core.example@0.1.0", "extra": "not parsed"})

    assert record == {"id": ProtocolIdentifier.parse("core.example@0.1.0")}


def test_record_validation_reports_missing_required_fields() -> None:
    spec = RecordSpec(fields={"id": FieldSpec(kind="identifier")})

    violations = spec.collect_violations({})

    assert violations == (RecordViolation(path=("id",), message="missing required field"),)


def test_record_validation_rejects_nonfinite_numbers() -> None:
    spec = RecordSpec(fields={"weight": FieldSpec(kind="number")})

    assert spec.collect_violations({"weight": float("nan")}) == (
        RecordViolation(path=("weight",), message="expected finite number"),
    )
    assert spec.collect_violations({"weight": float("inf")}) == (
        RecordViolation(path=("weight",), message="expected finite number"),
    )


def test_record_validation_parses_nested_records_and_sequences() -> None:
    spec = RecordSpec(
        fields={
            "metadata": FieldSpec(
                kind="record",
                record=RecordSpec(
                    fields={
                        "owner": FieldSpec(kind="name"),
                        "labels": FieldSpec(kind="sequence", item=FieldSpec(kind="string")),
                    }
                ),
            )
        }
    )

    record = spec.validate(
        {
            "metadata": {
                "owner": "core.owner",
                "labels": ["alpha", "beta"],
            }
        }
    )

    assert record == {
        "metadata": {
            "owner": ProtocolName.parse("core.owner"),
            "labels": ("alpha", "beta"),
        }
    }


def test_record_validation_accepts_open_record_fields() -> None:
    spec = RecordSpec(fields={"payload": FieldSpec(kind="record")})

    record = spec.validate({"payload": {"nested": {"value": 1}}})

    assert record == {"payload": {"nested": {"value": 1}}}
    assert spec.collect_violations({"payload": "not a record"}) == (
        RecordViolation(path=("payload",), message="expected record"),
    )


def test_record_validation_supports_none_literal() -> None:
    spec = RecordSpec(fields={"empty": FieldSpec(kind="literal", literal=None)})

    record = spec.validate({"empty": None})

    assert record == {"empty": None}


def test_record_validation_reports_nested_paths() -> None:
    spec = RecordSpec(
        fields={
            "metadata": FieldSpec(
                kind="record",
                record=RecordSpec(
                    fields={
                        "labels": FieldSpec(
                            kind="sequence",
                            item=FieldSpec(kind="string"),
                        )
                    }
                ),
            )
        }
    )

    violations = spec.collect_violations({"metadata": {"labels": ["ok", 1]}})

    assert violations == (
        RecordViolation(path=("metadata", "labels", "1"), message="expected string"),
    )


def test_record_validation_raises_structured_error() -> None:
    spec = RecordSpec(fields={"id": FieldSpec(kind="identifier")})

    error = capture_validation_error(lambda: spec.validate({"id": "bad"}))

    assert error.violations == (
        RecordViolation(path=("id",), message="invalid protocol identifier: 'bad'"),
    )
    assert str(error) == "id: invalid protocol identifier: 'bad'"


def test_field_specs_detect_missing_sequence_configuration() -> None:
    spec = RecordSpec(fields={"items": FieldSpec(kind="sequence")})

    violations = spec.collect_violations({"items": ["a"]})

    assert violations == (
        RecordViolation(path=("items",), message="missing sequence item specification"),
    )


def test_field_specs_record_required_and_optional_fields() -> None:
    assert FieldSpec(kind="string") == FieldSpec(kind="string", required=True)
    assert FieldSpec(kind="string", required=False) == FieldSpec(
        kind="string",
        required=False,
    )


def capture_validation_error(call: Callable[[], object]) -> RecordValidationError:
    try:
        call()
    except RecordValidationError as error:
        return error
    raise AssertionError("expected RecordValidationError")
