from collections.abc import Callable

from leibniz.identifiers import ProtocolIdentifier, ProtocolName, SemanticVersion
from leibniz.records import (
    FieldSpec,
    RecordSpec,
    RecordValidationError,
    RecordViolation,
    collect_record_violations,
    optional,
    required,
    validate_record,
)


def test_record_validation_parses_valid_record() -> None:
    spec = RecordSpec(
        fields={
            "id": required("identifier"),
            "kind": required("literal", literal="example"),
            "version": required("version"),
            "name": required("name"),
            "count": required("integer"),
            "weight": required("number"),
            "active": required("boolean"),
            "tags": optional("sequence", item=required("string")),
        }
    )

    record = validate_record(
        {
            "id": "core.example@0.1.0",
            "kind": "example",
            "version": "0.1.0",
            "name": "core.example",
            "count": 3,
            "weight": 2.5,
            "active": True,
            "tags": ["a", "b"],
        },
        spec,
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
            "id": required("identifier"),
            "count": required("integer"),
            "description": optional("string"),
        }
    )

    violations = collect_record_violations(
        {
            "id": "not an identifier",
            "count": True,
            "extra": "ignored nowhere",
        },
        spec,
    )

    assert violations == (
        RecordViolation(path=("id",), message="invalid protocol identifier: 'not an identifier'"),
        RecordViolation(path=("count",), message="expected integer"),
        RecordViolation(path=("extra",), message="unknown field"),
    )


def test_record_validation_can_allow_unknown_fields() -> None:
    spec = RecordSpec(fields={"id": required("identifier")}, allow_unknown=True)

    record = validate_record({"id": "core.example@0.1.0", "extra": "not parsed"}, spec)

    assert record == {"id": ProtocolIdentifier.parse("core.example@0.1.0")}


def test_record_validation_reports_missing_required_fields() -> None:
    violations = collect_record_violations({}, RecordSpec(fields={"id": required("identifier")}))

    assert violations == (RecordViolation(path=("id",), message="missing required field"),)


def test_record_validation_rejects_nonfinite_numbers() -> None:
    spec = RecordSpec(fields={"weight": required("number")})

    assert collect_record_violations({"weight": float("nan")}, spec) == (
        RecordViolation(path=("weight",), message="expected finite number"),
    )
    assert collect_record_violations({"weight": float("inf")}, spec) == (
        RecordViolation(path=("weight",), message="expected finite number"),
    )


def test_record_validation_parses_nested_records_and_sequences() -> None:
    spec = RecordSpec(
        fields={
            "metadata": required(
                "record",
                record=RecordSpec(
                    fields={
                        "owner": required("name"),
                        "labels": required("sequence", item=required("string")),
                    }
                ),
            )
        }
    )

    record = validate_record(
        {
            "metadata": {
                "owner": "core.owner",
                "labels": ["alpha", "beta"],
            }
        },
        spec,
    )

    assert record == {
        "metadata": {
            "owner": ProtocolName.parse("core.owner"),
            "labels": ("alpha", "beta"),
        }
    }


def test_record_validation_supports_none_literal() -> None:
    spec = RecordSpec(fields={"empty": required("literal", literal=None)})

    record = validate_record({"empty": None}, spec)

    assert record == {"empty": None}


def test_record_validation_reports_nested_paths() -> None:
    spec = RecordSpec(
        fields={
            "metadata": required(
                "record",
                record=RecordSpec(fields={"labels": required("sequence", item=required("string"))}),
            )
        }
    )

    violations = collect_record_violations({"metadata": {"labels": ["ok", 1]}}, spec)

    assert violations == (
        RecordViolation(path=("metadata", "labels", "1"), message="expected string"),
    )


def test_record_validation_raises_structured_error() -> None:
    spec = RecordSpec(fields={"id": required("identifier")})

    error = capture_validation_error(lambda: validate_record({"id": "bad"}, spec))

    assert error.violations == (
        RecordViolation(path=("id",), message="invalid protocol identifier: 'bad'"),
    )
    assert str(error) == "id: invalid protocol identifier: 'bad'"


def test_field_specs_detect_missing_nested_configuration() -> None:
    violations = collect_record_violations(
        {"items": ["a"]},
        RecordSpec(fields={"items": required("sequence")}),
    )

    assert violations == (
        RecordViolation(path=("items",), message="missing sequence item specification"),
    )

    violations = collect_record_violations(
        {"nested": {}},
        RecordSpec(fields={"nested": required("record")}),
    )

    assert violations == (
        RecordViolation(path=("nested",), message="missing nested record specification"),
    )


def test_required_and_optional_helpers_build_field_specs() -> None:
    assert required("string") == FieldSpec(kind="string", required=True)
    assert optional("string") == FieldSpec(kind="string", required=False)


def capture_validation_error(call: Callable[[], object]) -> RecordValidationError:
    try:
        call()
    except RecordValidationError as error:
        return error
    raise AssertionError("expected RecordValidationError")
