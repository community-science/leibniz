from __future__ import annotations

import math
from pathlib import Path
from typing import cast

from leibniz.benchmark_implementations import load_benchmark
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.local_results import (
    _BenchmarkRunRecord,  # pyright: ignore[reportPrivateUsage]
    _model_result_records,  # pyright: ignore[reportPrivateUsage]
)
from leibniz.measurements import MeasurementDataset
from leibniz.observation_generation import StateSpaceVolumeRequest, load_generator
from leibniz.partition_score import (
    adversarial_partition_competence_integral,
    partition_samples_from_generated,
)


def test_model_result_records_use_partition_score_and_capability_map() -> None:
    benchmark_root = Path("src/leibniz/benchmarks/digits")
    benchmark = load_benchmark(benchmark_root)
    generator = load_generator(benchmark_root)
    batch = generator(
        seed=31,
        shape=32,
        include_metadata=True,
        volume_request=StateSpaceVolumeRequest(minimum=3.0, maximum=4.0),
        sample_indices=tuple(range(32)),
    )
    assert batch.region is not None
    samples = partition_samples_from_generated(
        batch.samples,
        {
            sample.index: (
                0.0
                if sample.region_component_index is not None
                and sample.region_component_index < 2
                else 1.0
            )
            for sample in batch.samples
        },
    )
    partition_score = adversarial_partition_competence_integral(
        root_region=batch.region,
        samples=samples,
        score_width_bits=1.0,
    )
    program_graph = _program_graph_record()
    empty_dataset = MeasurementDataset(measurements=())
    run = _BenchmarkRunRecord(
        source_kind="local-run",
        result_status="accepted",
        source_path=Path("results/evaluations/digits/run.json"),
        run_id="run-a",
        run_slug="run-a",
        benchmark_id=ProtocolIdentifier.parse("benchmarks.digits@0.1.0"),
        program_digest=ContentDigest.from_value(program_graph),
        model_key="model-a",
        log2_volume=4.0,
        measurement_count=len(samples),
        score=0.0,
        cost_summary={"component_count": 1, "storage_bytes": 1},
        program={"kind": "program-graph-reference"},
        program_graph=program_graph,
        model_inspection={},
        model_inspection_digest=ContentDigest.from_value({"inspection": "a"}),
        model_inspection_path=None,
        measurement_dataset=empty_dataset,
        measurement_dataset_digest=empty_dataset.digest,
        sampled_competence={
            "kind": "sampled-competence-curriculum",
            "benchmark_id": "benchmarks.digits@0.1.0",
            "log2_volume": 4.0,
            "sample_count": len(samples),
            "mean_accepted_mass": partition_score.mean_competence,
            "partition_score": partition_score.to_record(),
            "points": [
                {
                    "kind": "sampled-state-space-volume-window",
                    "benchmark_id": "benchmarks.digits@0.1.0",
                    "log2_volume": 4.0,
                    "log2_volume_minimum": 3.0,
                    "log2_volume_maximum": 4.0,
                    "sample_count": len(samples),
                    "score": partition_score.mean_competence,
                    "mean_accepted_mass": partition_score.mean_competence,
                    "region": batch.region.to_record(),
                }
            ],
        },
    )

    records = _model_result_records(
        (run,),
        manifest=benchmark.manifest,
        repository_root=Path.cwd(),
        include_console_view_model=False,
    )

    record = records[0]
    capability_map = cast(dict[str, object], record["capability_map"])
    assert math.isclose(cast(float, record["score"]), partition_score.value)
    assert math.isclose(cast(float, capability_map["value"]), partition_score.value)
    assert cast(int, capability_map["leaf_count"]) > 1
    assert len(cast(list[dict[str, object]], capability_map["refinement_ladder"])) > 1


def _program_graph_record() -> dict[str, object]:
    return {
        "contract_kind": "inverse",
        "inputs": [{"name": "image", "axes": [1, "N", "N"]}],
        "outputs": [{"name": "latent", "axes": [85]}],
        "nodes": [{"id": "model", "kind": "test-model", "parameters": {}}],
        "edges": [
            {"source_id": "image", "target_id": "model", "target_input_index": 0},
            {"source_id": "model", "target_id": "latent", "target_input_index": 0},
        ],
    }
