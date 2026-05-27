from pathlib import Path
from typing import cast

from leibniz.identifiers import ProtocolIdentifier
from leibniz.local_results import load_console_result_view
from leibniz.work_queues import (
    WorkQueueItem,
    load_work_queue_items,
    materialize_work_queue_view,
    write_work_queue_item,
)


def test_work_queue_items_round_trip_and_materialize_console_view(tmp_path: Path) -> None:
    runs_root = tmp_path / ".runs"
    item = WorkQueueItem(
        id="iteration-1-rank-1",
        benchmark_id=ProtocolIdentifier.parse("benchmarks.example@0.1.0"),
        proposal_id="proposal-1",
        proposal_set_path=runs_root / "proposals" / "proposal-set.json",
        command=("leibniz", "benchmark", "run"),
        status="pending",
        sequence=0,
    )

    item_path = write_work_queue_item(runs_root, item)
    loaded = load_work_queue_items(runs_root)
    view_path = materialize_work_queue_view(runs_root)
    view = load_console_result_view(view_path.read_bytes())

    assert item_path == (
        runs_root / "work-queues" / "benchmarks_example" / "iteration-1-rank-1.json"
    )
    assert loaded == (item,)
    assert view_path == runs_root / "views" / "work_queue.json"
    assert view["format"] == "leibniz.console.work-queue"
    queue_items = cast(list[dict[str, object]], view["queue_items"])
    assert queue_items == [item.to_record()]
