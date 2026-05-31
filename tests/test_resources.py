from collections.abc import Callable
from pathlib import Path

from leibniz.architectures import ArchitectureManifestDocument
from leibniz.artifacts import reference_for_record
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.resources import (
    ResourceAxis,
    ResourcePayload,
    ResourceReport,
    ResourceReportDocument,
    ResourceReportSet,
    ResourceReportSetDocument,
    ResourceValidationError,
)

_fixtures_root = Path(__file__).parent / "fixtures"


def test_resource_payloads_canonicalize_and_sum_deterministically() -> None:
    tensor = ResourcePayload.tensor(name="weights", shape=(3, 5), element_bits=16)
    table = ResourcePayload.table(name="lookup-table", entries=7, entry_bits=9)

    assert tensor.to_record() == {
        "name": "weights",
        "kind": "tensor",
        "total_bits": 240,
        "total_bytes": 30,
        "shape": [3, 5],
        "element_bits": 16,
    }
    assert table.to_record() == {
        "name": "lookup-table",
        "kind": "table",
        "total_bits": 63,
        "total_bytes": 8,
        "entries": 7,
        "entry_bits": 9,
    }

    report = ResourceReport.from_record(
        {
            "id": "resource-reports.digits-pool@0.1.0",
            "artifact": _architecture_reference_record(),
            "parameter_count": 15,
            "parameter_bits": 120,
            "parameter_bytes": 15,
            "payloads": [table.to_record(), tensor.to_record()],
            "inference_axes": [
                {"name": "multiply-adds", "value": 32, "unit": "operation"},
                {"name": "latency", "value": 0.5, "unit": "millisecond"},
            ],
        }
    )

    assert report.payloads == (table, tensor)
    assert report.inference_axes == (
        ResourceAxis(name="latency", value=0.5, unit="millisecond"),
        ResourceAxis(name="multiply-adds", value=32.0, unit="operation"),
    )
    assert report.total_bits == 423
    assert report.total_bytes == 53
    assert report.digest == ContentDigest.from_value(report.to_record())


def test_resource_report_document_loads_bytes_with_digest() -> None:
    record = _resource_report_record()

    document = ResourceReportDocument.from_bytes(canonical_document_bytes(record))

    assert document.report == ResourceReport.from_record(record)
    assert document.digest == ContentDigest.from_value(document.report.to_record())


def test_resource_report_set_canonicalizes_reports() -> None:
    first = ResourceReport.from_record(_resource_report_record(id_suffix="a"))
    second = ResourceReport.from_record(_resource_report_record(id_suffix="b"))

    report_set = ResourceReportSet.from_record(
        {
            "id": "resource-report-sets.boolean@0.1.0",
            "reports": [second.to_record(), first.to_record()],
        }
    )

    assert report_set == ResourceReportSet(
        id=ProtocolIdentifier.parse("resource-report-sets.boolean@0.1.0"),
        reports=(first, second),
    )
    assert report_set.total_bits == first.total_bits + second.total_bits
    assert report_set.total_bytes == (report_set.total_bits + 7) // 8
    assert report_set.to_record() == {
        "id": "resource-report-sets.boolean@0.1.0",
        "reports": [first.to_record(), second.to_record()],
    }


def test_resource_report_set_document_loads_bytes_with_digest() -> None:
    report = ResourceReport.from_record(_resource_report_record())
    record = {
        "id": "resource-report-sets.boolean@0.1.0",
        "reports": [report.to_record()],
    }

    document = ResourceReportSetDocument.from_bytes(canonical_document_bytes(record))

    assert document.report_set == ResourceReportSet.from_record(record)
    assert document.digest == ContentDigest.from_value(document.report_set.to_record())


def test_resource_report_cites_artifact_without_model_imports() -> None:
    report = ResourceReport.from_record(_resource_report_record())

    assert report.artifact.matches_record(_architecture_record())
    assert report.to_record()["artifact"] == _architecture_reference_record()


def test_resource_records_reject_malformed_payloads() -> None:
    assert str(
        capture_resource_error(
            lambda: ResourcePayload.from_record(
                {
                    "name": "weights",
                    "kind": "tensor",
                    "shape": [3, 0],
                    "element_bits": 16,
                    "total_bits": 0,
                    "total_bytes": 0,
                }
            )
        )
    ) == "shape axes must be positive integers"

    assert str(
        capture_resource_error(
            lambda: ResourcePayload.from_record(
                {
                    "name": "weights",
                    "kind": "tensor",
                    "shape": [3, 5],
                    "element_bits": 16,
                    "total_bits": 239,
                    "total_bytes": 30,
                }
            )
        )
    ) == "tensor total_bits must equal shape product times element_bits"

    assert str(
        capture_resource_error(
            lambda: ResourcePayload.from_record(
                {
                    "name": "lookup-table",
                    "kind": "table",
                    "entries": 7,
                    "entry_bits": 9,
                    "total_bits": 63,
                    "total_bytes": 7,
                }
            )
        )
    ) == "total_bytes must derive from total_bits"

    assert str(
        capture_resource_error(
            lambda: ResourcePayload.from_record(
                {
                    "name": "weights",
                    "kind": "tensor",
                    "shape": [3, 5],
                    "entries": 15,
                    "element_bits": 16,
                    "total_bits": 240,
                    "total_bytes": 30,
                }
            )
        )
    ) == "tensor payloads must not declare table fields"


def test_resource_records_reject_negative_or_inconsistent_report_fields() -> None:
    record = _resource_report_record()
    record["parameter_count"] = -1
    assert str(capture_resource_error(lambda: ResourceReport.from_record(record))) == (
        "parameter_count must be nonnegative"
    )

    record = _resource_report_record()
    record["parameter_bits"] = 121
    assert str(capture_resource_error(lambda: ResourceReport.from_record(record))) == (
        "parameter_bytes must derive from parameter_bits"
    )

    record = _resource_report_record()
    record["parameter_bytes"] = 15
    del record["parameter_bits"]
    assert str(capture_resource_error(lambda: ResourceReport.from_record(record))) == (
        "parameter_bits is required with parameter_bytes"
    )

    record = _resource_report_record()
    record["checkpoint_path"] = "results/checkpoints/model.pt"
    assert str(capture_resource_error(lambda: ResourceReport.from_record(record))) == (
        "checkpoint_path: unknown field"
    )

    record = _resource_report_record()
    record["inference_axes"] = [{"name": "latency", "value": float("inf"), "unit": "ms"}]
    assert str(capture_resource_error(lambda: ResourceReport.from_record(record))) == (
        "value: expected finite number"
    )


def test_resource_report_set_rejects_duplicates_and_invalid_documents() -> None:
    report = ResourceReport.from_record(_resource_report_record())
    assert str(
        capture_resource_error(
            lambda: ResourceReportSet.from_record(
                {
                    "id": "resource-report-sets.boolean@0.1.0",
                    "reports": [report.to_record(), report.to_record()],
                }
            )
        )
    ) == f"duplicate resource report id: {report.id}"

    assert str(
        capture_resource_error(lambda: ResourceReportDocument.from_bytes(b"[]"))
    ) == "resource report document must contain an object"
    assert str(
        capture_resource_error(lambda: ResourceReportSetDocument.from_bytes(b"[]"))
    ) == "resource report set document must contain an object"


def _resource_report_record(*, id_suffix: str = "digits-pool") -> dict[str, object]:
    return {
        "id": f"resource-reports.{id_suffix}@0.1.0",
        "artifact": _architecture_reference_record(),
        "parameter_count": 15,
        "parameter_bits": 120,
        "parameter_bytes": 15,
        "payloads": [
            ResourcePayload.tensor(name="weights", shape=(3, 5), element_bits=8).to_record()
        ],
    }


def _architecture_reference_record() -> dict[str, object]:
    return reference_for_record(
        kind="architecture-manifest",
        record=_architecture_record(),
    ).to_record()


def _architecture_record() -> dict[str, object]:
    return ArchitectureManifestDocument.from_bytes(
        (_fixtures_root / "architecture" / "digits_pool" / "manifest.json").read_bytes()
    ).manifest.to_record()


def capture_resource_error(action: Callable[[], object]) -> ResourceValidationError:
    try:
        action()
    except ResourceValidationError as error:
        return error
    raise AssertionError("expected ResourceValidationError")
