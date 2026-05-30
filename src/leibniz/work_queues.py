"""Ignored local work-queue records for active benchmark runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from leibniz.console.protocol import (
    console_protocol_format_versions,
    console_protocol_formats,
)
from leibniz.documents import (
    canonical_document_bytes,
    document_filename_suffix,
    load_object_document,
)
from leibniz.identifiers import ProtocolIdentifier
from leibniz.records import RecordExtractor, record_specs_from_package_contract

__all__ = [
    "WorkQueueError",
    "WorkQueueItem",
    "load_work_queue_items",
    "materialize_work_queue_view",
    "write_work_queue_item",
]

_protocol_formats = console_protocol_formats()
_protocol_format_versions = console_protocol_format_versions()
_item_format = _protocol_formats.work_queue_item
_view_format = _protocol_formats.work_queue_view
_format_version = _protocol_format_versions.work_queue_item
_document_suffix = document_filename_suffix()
_Status = Literal["pending", "reserved", "completed", "failed"]
_record_contracts = record_specs_from_package_contract(
    "leibniz.contract_artifacts",
    "work_queue_items",
    description="work queue item record contracts",
)
_work_queue_item_record = _record_contracts["work_queue_item"]


class WorkQueueError(ValueError):
    """Raised when local work-queue records are invalid."""


@dataclass(frozen=True, slots=True)
class WorkQueueItem:
    """One local work item projected from an active-loop proposal."""

    id: str
    benchmark_id: ProtocolIdentifier
    proposal_id: str
    proposal_set_path: Path
    command: tuple[str, ...]
    status: _Status
    sequence: int
    candidate_id: str | None = None
    run_id: str | None = None
    measurement_dataset_path: Path | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise WorkQueueError("id must be nonempty")
        if not self.proposal_id:
            raise WorkQueueError("proposal_id must be nonempty")
        if self.candidate_id is not None and not self.candidate_id:
            raise WorkQueueError("candidate_id must be nonempty")
        if not self.command or any(not argument for argument in self.command):
            raise WorkQueueError("command must contain nonempty strings")
        if self.status not in {"pending", "reserved", "completed", "failed"}:
            raise WorkQueueError(f"unsupported status: {self.status}")
        if type(self.sequence) is not int or self.sequence < 0:
            raise WorkQueueError("sequence must be nonnegative")
        if self.run_id is not None and not self.run_id:
            raise WorkQueueError("run_id must be nonempty")
        if self.error is not None and not self.error:
            raise WorkQueueError("error must be nonempty")

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "format": _item_format,
            "format_version": _format_version,
            "id": self.id,
            "benchmark_id": str(self.benchmark_id),
            "proposal_id": self.proposal_id,
            "proposal_set_path": self.proposal_set_path.as_posix(),
            "command": list(self.command),
            "status": self.status,
            "sequence": self.sequence,
        }
        if self.candidate_id is not None:
            record["candidate_id"] = self.candidate_id
        if self.run_id is not None:
            record["run_id"] = self.run_id
        if self.measurement_dataset_path is not None:
            record["measurement_dataset_path"] = self.measurement_dataset_path.as_posix()
        if self.error is not None:
            record["error"] = self.error
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> WorkQueueItem:
        if record.get("format") != _item_format:
            raise WorkQueueError("work queue item has unsupported format")
        if record.get("format_version") != _format_version:
            raise WorkQueueError("work queue item has unsupported format_version")
        try:
            validated = _work_queue_item_record.validate(record)
        except ValueError as error:
            raise WorkQueueError(str(error)) from error
        extractor = RecordExtractor(WorkQueueError)
        return cls(
            id=extractor.string(validated["id"], "id"),
            benchmark_id=extractor.identifier(validated["benchmark_id"], "benchmark_id"),
            proposal_id=extractor.string(validated["proposal_id"], "proposal_id"),
            candidate_id=(
                extractor.string(validated["candidate_id"], "candidate_id")
                if "candidate_id" in validated
                else None
            ),
            proposal_set_path=Path(
                extractor.string(validated["proposal_set_path"], "proposal_set_path")
            ),
            command=tuple(
                extractor.string(item, "command")
                for item in extractor.sequence(validated["command"], "command")
            ),
            status=cast(_Status, extractor.string(validated["status"], "status")),
            sequence=extractor.integer(validated["sequence"], "sequence"),
            run_id=(
                extractor.string(validated["run_id"], "run_id")
                if "run_id" in validated
                else None
            ),
            measurement_dataset_path=(
                Path(
                    extractor.string(
                        validated["measurement_dataset_path"],
                        "measurement_dataset_path",
                    )
                )
                if "measurement_dataset_path" in validated
                else None
            ),
            error=(
                extractor.string(validated["error"], "error")
                if "error" in validated
                else None
            ),
        )


def write_work_queue_item(runs_root: Path, item: WorkQueueItem) -> Path:
    """Write or replace one ignored local work-queue item."""

    path = _item_path(runs_root, item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_document_bytes(item.to_record()) + b"\n")
    return path


def load_work_queue_items(runs_root: Path) -> tuple[WorkQueueItem, ...]:
    """Load ignored local work-queue items from a runs root."""

    root = runs_root / "work-queues"
    if not root.is_dir():
        return ()
    items: list[WorkQueueItem] = []
    for path in sorted(root.rglob("*" + _document_suffix)):
        record = load_object_document(path.read_bytes(), description="work queue item")
        items.append(WorkQueueItem.from_record(record))
    return tuple(sorted(items, key=lambda item: (item.sequence, item.id)))


def materialize_work_queue_view(runs_root: Path) -> Path:
    """Project ignored local work-queue items into a read-only console view."""

    items = load_work_queue_items(runs_root)
    view_root = runs_root / "views"
    view_root.mkdir(parents=True, exist_ok=True)
    view_file = view_root / ("work_queue" + _document_suffix)
    view_file.write_bytes(
        canonical_document_bytes(
            {
                "format": _view_format,
                "format_version": _format_version,
                "queue_items": [item.to_record() for item in items],
            }
        )
        + b"\n"
    )
    return view_file


def _item_path(runs_root: Path, item: WorkQueueItem) -> Path:
    benchmark_atom = str(item.benchmark_id.name).replace(".", "_")
    return runs_root / "work-queues" / benchmark_atom / f"{item.id}{_document_suffix}"
