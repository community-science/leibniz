import base64
import struct
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import cast

import pytest

from leibniz.console.data import ConsoleDataBuilder, ConsoleDataValidationError
from leibniz.documents import load_object_document

_repository_root = Path(__file__).parents[1]


def test_console_data_discovery_is_deterministic() -> None:
    builder = ConsoleDataBuilder(_repository_root)

    first = builder.discover(
        (PurePosixPath("tests/fixtures"), PurePosixPath("src/leibniz/benchmarks"))
    )
    second = builder.discover(
        (
            PurePosixPath("tests/fixtures/finite_outcome"),
            PurePosixPath("tests/fixtures/chess"),
            PurePosixPath("tests/fixtures/architecture"),
            PurePosixPath("tests/fixtures/digits"),
            PurePosixPath("src/leibniz/benchmarks"),
        )
    )

    assert first.to_bytes() == second.to_bytes()


def test_console_data_discovers_supported_public_fixture_documents() -> None:
    data = ConsoleDataBuilder(_repository_root).discover(
        (PurePosixPath("tests/fixtures"), PurePosixPath("src/leibniz/benchmarks"))
    )
    record = data.to_record()

    assert record["format"] == "leibniz.console-data"
    assert record["format_version"] == 1
    operator_vocabulary = cast(dict[str, object], record["operator_vocabulary"])
    assert operator_vocabulary["format"] == "leibniz.model-operator-vocabulary"
    assert {
        alias["alias"]
        for alias in cast(list[dict[str, object]], operator_vocabulary["syntax_aliases"])
    } == {"adaptive-pooling", "flatten", "dense"}

    artifact_index = cast(dict[str, object], record["artifact_index"])
    artifacts = cast(list[dict[str, object]], artifact_index["artifacts"])
    details = cast(list[dict[str, object]], record["artifact_details"])

    assert [(artifact["kind"], artifact["source_path"]) for artifact in artifacts] == [
        ("architecture-manifest", "tests/fixtures/architecture/digits_pool/manifest.json"),
        ("benchmark-manifest", "src/leibniz/benchmarks/digits/manifest.json"),
        ("benchmark-manifest", "tests/fixtures/chess/mate_in_one/manifest.json"),
        ("benchmark-manifest", "tests/fixtures/finite_outcome/manifest.json"),
        ("latent-factor-declaration", "src/leibniz/benchmarks/digits/latent_factors.json"),
        ("materialization-declaration", "src/leibniz/benchmarks/digits/materialization.json"),
        ("materialization-plan", "tests/fixtures/digits/materialization_plan_l1.json"),
        ("materialization-plan", "tests/fixtures/digits/materialization_plan_l3.json"),
        ("measurement", "tests/fixtures/chess/mate_in_one/measurement.json"),
        ("measurement", "tests/fixtures/finite_outcome/measurement.json"),
        (
            "observation-formation-declaration",
            "src/leibniz/benchmarks/digits/observation_formation.json",
        ),
        ("observation-showcase", "src/leibniz/benchmarks/digits/inspection_showcase.json"),
    ]
    assert [(detail["kind"], detail["source_path"]) for detail in details] == [
        (artifact["kind"], artifact["source_path"]) for artifact in artifacts
    ]
    architecture_detail = details[0]
    architecture_graph = cast(dict[str, object], architecture_detail["architecture_graph"])
    assert [node["id"] for node in cast(list[dict[str, object]], architecture_graph["nodes"])] == [
        "component-0",
        "component-1",
        "component-2",
    ]
    assert [
        (edge["source_node_id"], edge["target_node_id"])
        for edge in cast(list[dict[str, object]], architecture_graph["edges"])
    ] == [
        ("component-0", "component-1"),
        ("component-1", "component-2"),
    ]
    assert {artifact["validation_status"] for artifact in artifacts} == {"valid"}

    assert "observation_inspections" not in record

    assert "performance_views" not in record

    model_inspections = cast(list[dict[str, object]], record["model_inspections"])
    assert len(model_inspections) == 1
    model_inspection = model_inspections[0]
    assert model_inspection["source_path"] == (
        "tests/fixtures/architecture/digits_pool/manifest.json"
    )
    assert model_inspection["input_shape"] == [1, 24, 24]
    assert model_inspection["output_shape"] == [10]
    assert model_inspection["cost_summary"] == {
        "component_count": 3,
        "parameter_count": 50,
        "parameter_bytes": 200,
        "inference_flops": 656,
        "unknown_parameter_components": [],
    }
    model_components = cast(list[dict[str, object]], model_inspection["components"])
    assert [
        (component["kind"], component.get("output_shape")) for component in model_components
    ] == [
        ("adaptive-pooling", [1, 2, 2]),
        ("flatten", [4]),
        ("dense", [10]),
    ]
    inspection_graph = cast(dict[str, object], model_inspection["architecture_graph"])
    assert [
        node["id"] for node in cast(list[dict[str, object]], inspection_graph["nodes"])
    ] == [
        "component-0",
        "component-1",
        "component-2",
    ]
    assert len(cast(list[dict[str, object]], inspection_graph["edges"])) == 2
    assert model_inspection["architecture_summary"] == {
        "component_count": 3,
        "edge_count": 2,
        "input_count": 1,
        "output_count": 1,
        "input_node_ids": ["component-0"],
        "output_node_ids": ["component-2"],
        "component_kinds": ["adaptive-pooling", "flatten", "dense"],
        "unsupported_parameter_components": [],
        "unsupported_flop_components": [],
    }
    assert [
        evidence["node_path"]
        for evidence in cast(list[dict[str, object]], model_inspection["node_evidence"])
    ] == [["component-0"], ["component-1"], ["component-2"]]
    trace = cast(dict[str, object], model_inspection["architecture_trace"])
    trace_stages = cast(list[dict[str, object]], trace["stages"])
    assert [(stage["operator_kind"], stage["syntax_alias"]) for stage in trace_stages] == [
        ("local-aggregation", "adaptive-pooling"),
        ("rank-collapse", "flatten"),
        ("affine-readout", "dense"),
    ]
    assert cast(dict[str, object], trace_stages[0]["descriptor_axes"])["support"] == (
        "local-window"
    )
    assert [component["operator"] for component in model_components] == [
        {
            "kind": "local-aggregation",
            "tensor_relation": "aggregation",
            "state": "fixed",
            "support": "local-window",
            "projection_law": "equal-output-partition",
            "aggregation_law": "mean",
            "parameter_sharing": "none",
            "shape_law": "preserve-prefix-replace-trailing-axes",
            "cost_law": "input-elements",
            "aliases": ["adaptive-pooling"],
        },
        {
            "kind": "rank-collapse",
            "tensor_relation": "shape-transform",
            "state": "fixed",
            "support": "rank-collapsing",
            "projection_law": "row-major-axis-concatenation",
            "aggregation_law": "none",
            "parameter_sharing": "none",
            "shape_law": "product-of-input-axes",
            "cost_law": "zero-arithmetic",
            "aliases": ["flatten"],
        },
        {
            "kind": "affine-readout",
            "tensor_relation": "affine",
            "state": "learned",
            "support": "global",
            "projection_law": "full-input-support",
            "aggregation_law": "weighted-sum-plus-bias",
            "parameter_sharing": "none",
            "shape_law": "rank-1-output",
            "cost_law": "multiply-add-per-input-output-pair",
            "aliases": ["dense"],
        },
    ]

    benchmark_tasks = cast(list[dict[str, object]], record["benchmark_tasks"])
    assert len(benchmark_tasks) == 1
    task = benchmark_tasks[0]
    assert task["kind"] == "generated-observations"
    assert task["benchmark_id"] == "benchmarks.digits@0.1.0"
    assert task["complexity_axis"] is None
    assert task["outcome_atom_count"] == 10
    batches = cast(list[dict[str, object]], task["batches"])
    assert [
        (batch["mode"], batch["component_count"], batch["sample_count"])
        for batch in batches
    ] == [
        ("balanced", 1, 40),
    ]
    batch = batches[0]
    presentation = cast(dict[str, object], batch["presentation"])
    assert presentation == {
        "sample_card_density": "standard",
        "aggregate_mode": False,
    }
    samples = cast(list[dict[str, object]], batch["samples"])
    component_sequences = [
        cast(list[int], sample["component_sequence"]) for sample in samples
    ]
    assert Counter(len(sequence) for sequence in component_sequences) == {1: 40}
    digit_counts = Counter(digit for sequence in component_sequences for digit in sequence)
    assert digit_counts == dict.fromkeys(range(10), 4)
    field_shapes = [tuple(cast(list[int], sample["field_shape"])) for sample in samples]
    assert len(set(field_shapes)) == len(field_shapes)
    materialization_plans = [
        cast(dict[str, object], sample["materialization_plan"]) for sample in samples
    ]
    assert all(".sample-0@" in str(plan["id"]) for plan in materialization_plans)
    assert len({plan["seed"] for plan in materialization_plans}) == len(materialization_plans)
    assert str(samples[0]["image_data_url"]).startswith("data:image/png;base64,")
    assert samples[0]["field_shape"] == [1, 128, 139]
    assert _png_dimensions(str(samples[0]["image_data_url"])) == (139, 128)
    assert _png_dimensions(str(samples[1]["image_data_url"])) == (43, 171)
    assert "preview_crop" not in samples[0]
    assert "preview_crop" not in samples[1]
    latent_coordinates = cast(list[dict[str, object]], samples[0]["latent_coordinates"])
    variation = next(
        coordinate for coordinate in latent_coordinates if coordinate["role"] == "variation"
    )
    variation_values = cast(dict[str, object], variation["values"])
    assert variation_values["kind"] == "field-variation-transform-samples"
    variation_bounds = cast(dict[str, object], variation_values["bounds"])
    assert variation_bounds["kind"] == "field-variation-transform"
    variation_coordinates = cast(list[dict[str, object]], variation_values["coordinates"])
    assert len(variation_coordinates) == 1
    assert variation_coordinates[0]["kind"] == "field-variation-transform-coordinate"


def test_console_data_payload_is_a_canonical_object_document() -> None:
    data = ConsoleDataBuilder(_repository_root).discover(
        (PurePosixPath("tests/fixtures"), PurePosixPath("src/leibniz/benchmarks"))
    )

    record = load_object_document(data.to_bytes(), description="console data")

    assert record["format"] == "leibniz.console-data"


def _png_dimensions(data_url: str) -> tuple[int, int]:
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise AssertionError("expected PNG data URL")
    data = base64.b64decode(data_url[len(prefix) :])
    return struct.unpack(">II", data[16:24])


def test_console_data_discovers_explicit_result_views(tmp_path: Path) -> None:
    result_root = tmp_path / "views"
    result_root.mkdir()
    (result_root / "imported_results.json").write_text(
        """
{
  "format": "leibniz.console.imported-results",
  "format_version": 1,
  "publication_bundles": [
    {
      "id": "publication-bundles.digits@0.1.0",
      "digest": "sha256:abc",
      "source_path": "/tmp/submissions/digits.json",
      "submission_package_id": "submissions.digits@0.1.0",
      "benchmark_ids": ["benchmarks.digits@0.1.0"],
      "measurement_count": 1,
      "measurement_dataset": {"measurements": []},
      "measurement_score_view": {"entries": []}
    }
  ]
}
""",
        encoding="utf-8",
    )
    (result_root / "benchmark_results.json").write_text(
        """
{
  "format": "leibniz.console.benchmark-results",
  "format_version": 1,
  "benchmark_results": [
    {
      "benchmark_id": "benchmarks.digits@0.1.0",
      "complexity_axis": "C",
      "cost_axes": [{"key": "parameter_count", "label": "Parameters"}],
      "leaderboard": [
        {
          "model_key": "sha256:model",
          "architecture_digest": "sha256:model",
          "benchmark_id": "benchmarks.digits@0.1.0",
          "score": 1.0,
          "observed_complexities": [1.0],
          "points": [{"complexity": 1.0, "score": 1.0, "run_ids": ["run-1"]}],
          "cost_summary": {
            "component_count": 1,
            "parameter_count": 10,
            "parameter_bytes": 40,
            "inference_flops": 20,
            "unknown_parameter_components": []
          },
          "run_ids": ["run-1"],
          "measurement_count": 1,
          "source_kinds": ["local-run"]
        }
      ],
      "frontiers": {
        "parameter_count": [],
        "inference_flops": [],
        "parameter_bytes": []
      },
      "training_history": [
        {
          "source_kind": "local-run",
          "source_path": "results/training/example.json",
          "run_id": "run-1",
          "run_slug": "run-1",
          "benchmark_id": "benchmarks.digits@0.1.0",
          "architecture_digest": "sha256:model",
          "model_key": "sha256:model",
          "complexity": 10,
          "measurement_count": 1,
          "score": 1.0,
          "cost_summary": {
            "component_count": 1,
            "parameter_count": 10,
            "parameter_bytes": 40,
            "inference_flops": 20,
            "unknown_parameter_components": []
          },
          "architecture": {"kind": "architecture-manifest"},
          "measurement_dataset_digest": "sha256:dataset"
        }
      ]
    }
  ]
}
""",
        encoding="utf-8",
    )

    data = ConsoleDataBuilder(_repository_root).discover(
        (PurePosixPath("tests/fixtures"),),
        result_roots=(result_root,),
    )
    record = data.to_record()
    result_views = cast(list[dict[str, object]], record["result_views"])

    assert len(result_views) == 2
    imported = next(
        view for view in result_views if view["format"] == "leibniz.console.imported-results"
    )
    benchmark = next(
        view for view in result_views if view["format"] == "leibniz.console.benchmark-results"
    )
    assert imported["source_path"] == (result_root / "imported_results.json").as_posix()
    assert isinstance(imported["source_mtime_ms"], int)
    assert isinstance(imported["source_size_bytes"], int)
    bundles = cast(list[dict[str, object]], imported["publication_bundles"])
    assert bundles[0]["measurement_count"] == 1
    results = cast(list[dict[str, object]], benchmark["benchmark_results"])
    assert results[0]["benchmark_id"] == "benchmarks.digits@0.1.0"


def test_console_data_discovers_materialized_result_root_views(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    view_root = result_root / "views"
    view_root.mkdir(parents=True)
    (view_root / "benchmark_results.json").write_text(
        """
{
  "format": "leibniz.console.benchmark-results",
  "format_version": 1,
  "benchmark_results": [
    {
      "benchmark_id": "benchmarks.digits@0.1.0",
      "cost_axes": [{"key": "parameter_count", "label": "Parameters"}],
      "leaderboard": [],
      "frontiers": {
        "parameter_count": [],
        "inference_flops": [],
        "parameter_bytes": []
      },
      "training_history": []
    }
  ]
}
""",
        encoding="utf-8",
    )

    data = ConsoleDataBuilder(_repository_root).discover(
        (PurePosixPath("tests/fixtures"),),
        result_roots=(result_root,),
    )
    record = data.to_record()
    result_views = cast(list[dict[str, object]], record["result_views"])

    assert len(result_views) == 1
    assert result_views[0]["source_path"] == (
        view_root / "benchmark_results.json"
    ).as_posix()


def test_console_data_rejects_local_state_roots() -> None:
    with pytest.raises(ConsoleDataValidationError, match="local state"):
        ConsoleDataBuilder(_repository_root).discover((PurePosixPath("results"),))


def test_console_data_rejects_nested_local_result_roots() -> None:
    with pytest.raises(ConsoleDataValidationError, match="results or results/views"):
        ConsoleDataBuilder(_repository_root).discover(
            (PurePosixPath("tests/fixtures"),),
            result_roots=(Path("results/training-progress"),),
        )


def test_console_data_rejects_missing_public_roots() -> None:
    with pytest.raises(ConsoleDataValidationError, match="does not name a directory"):
        ConsoleDataBuilder(_repository_root).discover((PurePosixPath("tests/missing"),))


def test_console_data_rejects_roots_without_supported_documents(tmp_path: Path) -> None:
    (tmp_path / "README.txt").write_text("not a supported document", encoding="utf-8")

    with pytest.raises(ConsoleDataValidationError, match="did not contain supported documents"):
        ConsoleDataBuilder(tmp_path).discover((PurePosixPath("."),))
