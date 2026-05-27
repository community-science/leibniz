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
    assert {artifact["validation_status"] for artifact in artifacts} == {"valid"}

    assert "observation_inspections" not in record

    assert "performance_views" not in record

    model_inspections = cast(list[dict[str, object]], record["model_inspections"])
    assert len(model_inspections) == 1
    model_inspection = model_inspections[0]
    assert model_inspection["source_path"] == (
        "tests/fixtures/architecture/digits_pool/manifest.json"
    )
    assert model_inspection["input_shape"] == [1, 32, 32]
    assert model_inspection["output_shape"] == [10]
    assert model_inspection["cost_summary"] == {
        "layer_count": 3,
        "parameter_count": 50,
        "parameter_bytes": 200,
        "inference_flops": 1104,
        "unknown_parameter_layers": [],
    }
    model_layers = cast(list[dict[str, object]], model_inspection["layers"])
    assert [(layer["kind"], layer.get("output_shape")) for layer in model_layers] == [
        ("adaptive-pooling", [1, 2, 2]),
        ("flatten", [4]),
        ("dense", [10]),
    ]
    assert [layer["operator"] for layer in model_layers] == [
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
    assert task["scale_axis"] == "L"
    assert task["complexity_axis"] == "C"
    assert task["outcome_atom_count"] == 10
    batches = cast(list[dict[str, object]], task["batches"])
    assert [(batch["mode"], batch["scale"], batch["sample_count"]) for batch in batches] == [
        ("canonical", 1, 4),
        ("canonical", 2, 4),
        ("canonical", 3, 4),
        ("canonical", 4, 4),
        ("symbol-probe", 1, 10),
        ("complexity-sweep", 1, 1),
        ("complexity-sweep", 2, 1),
        ("complexity-sweep", 3, 1),
        ("complexity-sweep", 4, 1),
    ]
    symbol_probe = batches[4]
    symbol_presentation = cast(dict[str, object], symbol_probe["presentation"])
    assert symbol_presentation == {
        "sample_card_density": "compact",
        "aggregate_mode": False,
    }
    symbol_samples = cast(list[dict[str, object]], symbol_probe["samples"])
    assert [sample["component_sequence"] for sample in symbol_samples] == [
        [digit] for digit in range(10)
    ]
    assert str(symbol_samples[0]["image_data_url"]).startswith("data:image/png;base64,")
    assert symbol_samples[0]["field_shape"] == [1, 32, 32]
    sweep_presentation = cast(dict[str, object], batches[5]["presentation"])
    assert sweep_presentation == {
        "sample_card_density": "standard",
        "aggregate_mode": True,
    }


def test_console_data_payload_is_a_canonical_object_document() -> None:
    data = ConsoleDataBuilder(_repository_root).discover(
        (PurePosixPath("tests/fixtures"), PurePosixPath("src/leibniz/benchmarks"))
    )

    record = load_object_document(data.to_bytes(), description="console data")

    assert record["format"] == "leibniz.console-data"


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
      "scale_axis": "L",
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
            "layer_count": 1,
            "parameter_count": 10,
            "parameter_bytes": 40,
            "inference_flops": 20,
            "unknown_parameter_layers": []
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
          "source_path": ".runs/training/example.json",
          "run_id": "run-1",
          "run_slug": "run-1",
          "benchmark_id": "benchmarks.digits@0.1.0",
          "architecture_digest": "sha256:model",
          "model_key": "sha256:model",
          "scale": 1,
          "measurement_count": 1,
          "score": 1.0,
          "cost_summary": {
            "layer_count": 1,
            "parameter_count": 10,
            "parameter_bytes": 40,
            "inference_flops": 20,
            "unknown_parameter_layers": []
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


def test_console_data_rejects_local_state_roots() -> None:
    with pytest.raises(ConsoleDataValidationError, match="local state"):
        ConsoleDataBuilder(_repository_root).discover((PurePosixPath(".leibniz"),))

    with pytest.raises(ConsoleDataValidationError, match="local state"):
        ConsoleDataBuilder(_repository_root).discover((PurePosixPath(".runs"),))


def test_console_data_rejects_raw_runs_result_roots() -> None:
    with pytest.raises(ConsoleDataValidationError, match=".runs/views"):
        ConsoleDataBuilder(_repository_root).discover(
            (PurePosixPath("tests/fixtures"),),
            result_roots=(Path(".runs"),),
        )


def test_console_data_rejects_missing_public_roots() -> None:
    with pytest.raises(ConsoleDataValidationError, match="does not name a directory"):
        ConsoleDataBuilder(_repository_root).discover((PurePosixPath("tests/missing"),))


def test_console_data_rejects_roots_without_supported_documents(tmp_path: Path) -> None:
    (tmp_path / "README.txt").write_text("not a supported document", encoding="utf-8")

    with pytest.raises(ConsoleDataValidationError, match="did not contain supported documents"):
        ConsoleDataBuilder(tmp_path).discover((PurePosixPath("."),))
