import base64
import struct
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, cast

import pytest

import leibniz.console.data as console_data
from leibniz.console.data import ConsoleDataBuilder, ConsoleDataValidationError
from leibniz.documents import canonical_document_bytes, load_object_document
from leibniz.identifiers import ProtocolIdentifier

_repository_root = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def _use_test_console_sample_cache(  # pyright: ignore[reportUnusedFunction]
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "leibniz.console.data._generated_batch_cache_path",
        tmp_path / "generatedSampleSets.leibniz.json",
    )
    cast(
        dict[tuple[str, str, str], tuple[Mapping[str, object], ...]],
        console_data._generated_batch_cache,  # type: ignore[reportPrivateUsage]
    ).clear()


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


def test_console_data_reuses_persistent_generated_observation_batch_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_path = tmp_path / "generatedSampleSets.leibniz.json"
    monkeypatch.setattr("leibniz.console.data._generated_batch_cache_path", cache_path)
    cast(
        dict[tuple[str, str, str], tuple[Mapping[str, object], ...]],
        console_data._generated_batch_cache,  # type: ignore[reportPrivateUsage]
    ).clear()
    cache_key = (
        "benchmarks.fake@0.1.0",
        "volume-window-samples",
        "source-fingerprint",
    )
    cached_batches: tuple[Mapping[str, object], ...] = ({
        "mode": "volume-window",
        "label": "Cached samples",
        "seed": 401,
        "sample_count": 0,
        "volume_window": {
            "measure_id": "log2-state-space-volume",
            "minimum": 1,
            "maximum": 2,
        },
        "presentation": {"sample_card_density": "standard", "aggregate_mode": False},
        "samples": [],
    },)
    cache_path.write_bytes(
        canonical_document_bytes(
            {
                "format": "leibniz.console.generated-sample-set-cache",
                "format_version": 2,
                "batches": {
                    "\0".join(cache_key): list(cached_batches),
                },
            }
        )
        + b"\n"
    )
    fake_generator = SimpleNamespace(
        manifest=SimpleNamespace(
            id=ProtocolIdentifier.parse("benchmarks.fake@0.1.0"),
        )
    )
    batches = ConsoleDataBuilder(_repository_root)._sample_sets(  # type: ignore[reportPrivateUsage]
        generator=cast(Any, fake_generator),
        atom_count=10,
        source_fingerprint="source-fingerprint",
    )

    assert batches == cached_batches


def test_console_data_discovery_does_not_swallow_loader_bugs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_loader(_kind: str, _data: bytes) -> object:
        raise RuntimeError("loader bug")

    monkeypatch.setattr(
        "leibniz.console.data.ConsoleArtifactIndexBuilder.load_supported_artifact",
        broken_loader,
    )

    with pytest.raises(RuntimeError, match="loader bug"):
        ConsoleDataBuilder(_repository_root).discover((PurePosixPath("tests/fixtures"),))


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
    } == {
        "adaptive-pooling",
        "convolution",
        "relu",
        "flatten",
        "dense",
    }

    artifact_index = cast(dict[str, object], record["artifact_index"])
    artifacts = cast(list[dict[str, object]], artifact_index["artifacts"])
    details = cast(list[dict[str, object]], record["artifact_details"])

    assert [(artifact["kind"], artifact["source_path"]) for artifact in artifacts] == [
        (
            "architecture-manifest",
            "tests/fixtures/architecture/chess_board_linear.json",
        ),
        (
            "architecture-manifest",
            "tests/fixtures/architecture/digits_convnet.json",
        ),
        (
            "architecture-manifest",
            "tests/fixtures/architecture/digits_perf_conv.json",
        ),
        ("architecture-manifest", "tests/fixtures/architecture/digits_pool.json"),
        ("benchmark-manifest", "tests/fixtures/finite_outcome/manifest.json"),
        ("materialization-plan", "tests/fixtures/digits/materialization_plan_l1.json"),
        ("materialization-plan", "tests/fixtures/digits/materialization_plan_l3.json"),
        ("measurement", "tests/fixtures/chess/mate_in_one/measurement.json"),
        ("measurement", "tests/fixtures/finite_outcome/measurement.json"),
    ]
    assert [(detail["kind"], detail["source_path"]) for detail in details] == [
        (artifact["kind"], artifact["source_path"]) for artifact in artifacts
    ]
    architecture_detail = next(
        detail
        for detail in details
        if detail["source_path"] == "tests/fixtures/architecture/digits_pool.json"
    )
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

    benchmark_tasks = cast(list[dict[str, object]], record["benchmark_tasks"])
    digits_task = next(
        task
        for task in benchmark_tasks
        if task["benchmark_id"] == "benchmarks.digits@0.1.0"
    )
    code_surfaces = cast(list[dict[str, object]], digits_task["code_surfaces"])
    assert [
        (surface["label"], surface["symbol"])
        for surface in code_surfaces
    ] == [
        ("Generator", "Generator.__call__"),
    ]
    for surface in code_surfaces:
        source_path = _repository_root / cast(str, surface["source_path"])
        start_line = cast(int, surface["start_line"])
        end_line = cast(int, surface["end_line"])
        expected_code = "\n".join(source_path.read_text(encoding="utf-8").splitlines()[
            start_line - 1 : end_line
        ])
        assert surface["code"] == expected_code

    model_inspections = cast(list[dict[str, object]], record["model_inspections"])
    assert [
        inspection["source_path"] for inspection in model_inspections
    ] == [
        "tests/fixtures/architecture/chess_board_linear.json",
        "tests/fixtures/architecture/digits_convnet.json",
        "tests/fixtures/architecture/digits_perf_conv.json",
        "tests/fixtures/architecture/digits_pool.json",
    ]
    model_inspection = next(
        inspection
        for inspection in model_inspections
        if inspection["source_path"] == "tests/fixtures/architecture/digits_pool.json"
    )
    assert model_inspection["source_path"] == (
        "tests/fixtures/architecture/digits_pool.json"
    )
    assert model_inspection["input_shape"] == [1, 24, 24]
    assert model_inspection["output_shape"] == [10]
    cost_summary = cast(dict[str, object], model_inspection["cost_summary"])
    assert cost_summary["component_count"] == 3
    assert cost_summary["parameter_count"] == 50
    assert cost_summary["storage_bytes"] == 200
    assert cost_summary["inference_cost_sample_count"] == 1
    assert cost_summary["unknown_parameter_components"] == []
    inference_cost = cast(dict[str, object], cost_summary["inference_cost_measurement"])
    assert inference_cost["abstract_flops"] == 656
    assert inference_cost["execution_mode"] == "dry-run"
    assert inference_cost["operations_executed"] is False
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
        "unsupported_cost_components": [],
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
    assert {
        task["benchmark_id"]
        for task in benchmark_tasks
    } == {
        "benchmarks.chess@0.1.0",
        "benchmarks.digits@0.1.0",
    }
    chess_task = next(
        task
        for task in benchmark_tasks
        if task["benchmark_id"] == "benchmarks.chess@0.1.0"
    )
    assert cast(int, chess_task["outcome_atom_count"]) > 1000
    chess_batches = cast(list[dict[str, object]], chess_task["batches"])
    assert [
        batch["volumes"] for batch in chess_batches
    ] == [[cardinality] for cardinality in (1, 2, 4, 8, 16, 32, 64, 128, 256)]
    chess_sample = cast(list[dict[str, object]], chess_batches[0]["samples"])[0]
    assert str(chess_sample["image_data_url"]).startswith(
        "data:image/svg+xml;base64,"
    )

    task = next(
        task
        for task in benchmark_tasks
        if task["benchmark_id"] == "benchmarks.digits@0.1.0"
    )
    assert task["kind"] == "generated-observations"
    assert task["benchmark_id"] == "benchmarks.digits@0.1.0"
    assert task["volume_axis"] == "log2-state-space-volume"
    assert task["outcome_atom_count"] == 10
    batches = cast(list[dict[str, object]], task["batches"])
    assert [
        (batch["mode"], batch["sample_count"])
        for batch in batches
    ] == [
        ("volume-window", 1),
        ("volume-window", 2),
        ("volume-window", 4),
        ("volume-window", 8),
        ("volume-window", 16),
        ("volume-window", 32),
        ("volume-window", 50),
        ("volume-window", 50),
        ("volume-window", 50),
    ]
    assert [batch["label"] for batch in batches] == [
        "[0, 1]",
        "[1, 2]",
        "[2, 3]",
        "[3, 4]",
        "[4, 5]",
        "[5, 6]",
        "[6, 7]",
        "[7, 8]",
        "[8, 9]",
    ]
    presentation = cast(dict[str, object], batches[0]["presentation"])
    assert presentation == {
        "sample_card_density": "standard",
        "aggregate_mode": False,
    }
    assert cast(dict[str, object], batches[-1]["presentation"])["sample_card_density"] == "standard"
    samples = cast(list[dict[str, object]], batches[3]["samples"])
    component_indices = {cast(int, sample["component_index"]) for sample in samples}
    assert component_indices <= set(range(10))
    assert len(component_indices) == len(samples)
    field_shapes = [tuple(cast(list[int], sample["field_shape"])) for sample in samples]
    assert set(field_shapes) == {(1, 36, 36)}
    materialization_plans = [
        cast(dict[str, object], sample["materialization_plan"]) for sample in samples
    ]
    assert all(".sample-" in str(plan["id"]) for plan in materialization_plans)
    assert {plan["seed"] for plan in materialization_plans} == {404}
    assert str(samples[0]["image_data_url"]).startswith("data:image/png;base64,")
    assert samples[0]["field_shape"] == [1, 36, 36]
    assert _png_dimensions(str(samples[0]["image_data_url"])) == (36, 36)
    assert _png_dimensions(str(samples[1]["image_data_url"])) == (36, 36)
    assert "preview_crop" not in samples[0]
    assert "preview_crop" not in samples[1]
    latent_coordinates = cast(list[dict[str, object]], samples[0]["latent_coordinates"])
    variation = next(
        coordinate for coordinate in latent_coordinates if coordinate["role"] == "variation"
    )
    variation_values = cast(dict[str, object], variation["values"])
    assert variation_values["kind"] == "constructed-field-variation-transform-samples"
    assert variation_values["transform_ordinal"] == 0
    volume_class = cast(dict[str, object], variation_values["volume_class"])
    assert volume_class["kind"] == "digits-realized-setup-window"
    assert volume_class["canvas_side"] == 36
    assert volume_class["transform_axes"] == ["x_translation", "y_translation", "scale"]
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


def test_console_benchmark_tasks_load_python_implementation_without_exported_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_parent = tmp_path / "src" / "leibniz" / "benchmarks"
    benchmark_root = benchmark_parent / "digits"
    benchmark_root.mkdir(parents=True)
    source_root = _repository_root / "src" / "leibniz" / "benchmarks" / "digits"
    (benchmark_root / "benchmark.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "from leibniz.benchmark_implementations import load_benchmark",
                f"_source = Path({str(source_root)!r})",
                "def benchmark(root: Path):",
                "    return load_benchmark(_source)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    architecture_root = tmp_path / "architectures"
    architecture_root.mkdir()
    architecture_source = (
        _repository_root / "tests" / "fixtures" / "architecture" / "digits_pool.json"
    )
    (architecture_root / "digits_pool.json").write_bytes(architecture_source.read_bytes())

    def fake_batches(
        self: ConsoleDataBuilder,
        *,
        generator: object,
        atom_count: int,
        source_fingerprint: str,
    ) -> tuple[Mapping[str, object], ...]:
        assert atom_count == 10
        assert source_fingerprint
        return ({
            "mode": "volume-window",
            "label": "Fake samples",
            "seed": 401,
            "sample_count": 0,
            "presentation": {"sample_card_density": "standard", "aggregate_mode": False},
            "samples": [],
        },)

    monkeypatch.setattr(ConsoleDataBuilder, "_sample_sets", fake_batches)

    data = ConsoleDataBuilder(tmp_path).discover(
        (PurePosixPath("architectures"), PurePosixPath("src/leibniz/benchmarks"))
    )
    record = data.to_record()

    artifact_index = cast(dict[str, object], record["artifact_index"])
    artifacts = cast(list[dict[str, object]], artifact_index["artifacts"])
    assert [artifact["source_path"] for artifact in artifacts] == [
        "architectures/digits_pool.json"
    ]
    benchmark_tasks = cast(list[dict[str, object]], record["benchmark_tasks"])
    assert len(benchmark_tasks) == 1
    assert benchmark_tasks[0]["benchmark_id"] == "benchmarks.digits@0.1.0"
    assert benchmark_tasks[0]["source_path"] == "src/leibniz/benchmarks/digits"


def _png_dimensions(data_url: str) -> tuple[int, int]:
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise AssertionError("expected PNG data URL")
    data = base64.b64decode(data_url[len(prefix) :])
    return struct.unpack(">II", data[16:24])


def test_console_data_discovers_explicit_result_views(tmp_path: Path) -> None:
    result_root = tmp_path / "views"
    benchmark_view_root = result_root / "digits"
    benchmark_view_root.mkdir(parents=True)
    (benchmark_view_root / "benchmark_results.json").write_text(
        """
{
  "format": "leibniz.console.benchmark-results",
  "format_version": 1,
  "benchmark_results": [
    {
      "benchmark_id": "benchmarks.digits@0.1.0",
      "volume_axis": "C",
          "leaderboard": [
            {
              "model_key": "sha256:model",
              "result_status": "accepted",
              "architecture_digest": "sha256:model",
          "benchmark_id": "benchmarks.digits@0.1.0",
          "score": 1.0,
          "score_integral": {
            "kind": "sampled-competence-integral",
            "value": 1.0,
            "terms": [
              {
                "kind": "measured-state-space-competence",
                "log2_volume_minimum": 0.0,
                "log2_volume_maximum": 1.0,
                "width_in_bits": 1.0,
                "competence_density": 1.0,
                "contribution": 1.0,
                "representative_log2_volume": 1.0
              }
            ]
          },
          "points": [{"log2_volume": 1.0, "score": 1.0, "run_ids": ["run-1"]}],
          "cost_summary": {
            "component_count": 1,
            "cost": 640,
            "storage_bytes": 40,
            "inference_cost_measurement": {"abstract_flops": 20},
            "inference_cost_sample_count": 1,
            "unknown_parameter_components": []
          },
          "run_ids": ["run-1"],
          "measurement_count": 1,
          "source_kinds": ["local-run"]
        }
      ],
      "frontiers": {
        "cost": []
      },
          "training_history": [
            {
              "source_kind": "local-run",
              "result_status": "accepted",
              "source_path": "results/training/example.json",
          "run_id": "run-1",
          "run_slug": "run-1",
          "benchmark_id": "benchmarks.digits@0.1.0",
          "architecture_digest": "sha256:model",
          "model_key": "sha256:model",
          "log2_volume": 10,
          "measurement_count": 1,
          "score": 1.0,
          "cost_summary": {
            "component_count": 1,
            "cost": 640,
            "storage_bytes": 40,
            "inference_cost_measurement": {"abstract_flops": 20},
            "inference_cost_sample_count": 1,
            "unknown_parameter_components": []
          },
          "architecture": {"kind": "architecture-manifest"},
              "measurement_dataset_digest": "sha256:dataset"
            }
          ],
          "model_candidates": [
            {
              "model_key": "sha256:model",
              "result_status": "accepted",
              "architecture_digest": "sha256:model",
              "benchmark_id": "benchmarks.digits@0.1.0",
              "score": 1.0,
              "score_integral": {
                "kind": "sampled-competence-integral",
                "value": 1.0,
                "terms": [
                  {
                    "kind": "measured-state-space-competence",
                    "log2_volume_minimum": 0.0,
                    "log2_volume_maximum": 1.0,
                    "width_in_bits": 1.0,
                    "competence_density": 1.0,
                    "contribution": 1.0,
                    "representative_log2_volume": 1.0
                  }
                ]
              },
              "points": [{"log2_volume": 1.0, "score": 1.0, "run_ids": ["run-1"]}],
              "cost_summary": {
                "component_count": 1,
                "cost": 640,
                "storage_bytes": 40,
                "inference_cost_measurement": {"abstract_flops": 20},
            "inference_cost_sample_count": 1,
                "unknown_parameter_components": []
              },
              "run_ids": ["run-1"],
              "measurement_count": 1,
              "source_kinds": ["local-run"]
            }
          ],
          "plot_runs": [
            {
              "source_kind": "local-run",
              "result_status": "accepted",
              "source_path": "results/training/example.json",
              "run_id": "run-1",
              "run_slug": "run-1",
              "benchmark_id": "benchmarks.digits@0.1.0",
              "architecture_digest": "sha256:model",
              "model_key": "sha256:model",
              "log2_volume": 10,
              "measurement_count": 1,
              "score": 1.0,
              "cost_summary": {
                "component_count": 1,
                "cost": 640,
                "storage_bytes": 40,
                "inference_cost_measurement": {"abstract_flops": 20},
            "inference_cost_sample_count": 1,
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

    assert len(result_views) == 1
    benchmark = next(
        view for view in result_views if view["format"] == "leibniz.console.benchmark-results"
    )
    results = cast(list[dict[str, object]], benchmark["benchmark_results"])
    assert results[0]["benchmark_id"] == "benchmarks.digits@0.1.0"


def test_console_data_discovers_materialized_result_root_views(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    view_root = result_root / "views"
    (view_root / "digits").mkdir(parents=True)
    (view_root / "digits" / "benchmark_results.json").write_text(
        """
{
  "format": "leibniz.console.benchmark-results",
  "format_version": 1,
  "benchmark_results": [
    {
      "benchmark_id": "benchmarks.digits@0.1.0",
      "leaderboard": [],
      "model_candidates": [],
      "frontiers": {
        "cost": []
      },
      "training_history": [],
      "plot_runs": []
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

    assert {
        cast(str, view["source_path"])
        for view in result_views
    } == {
        (view_root / "chess" / "benchmark_results.json").as_posix(),
        (view_root / "digits" / "benchmark_results.json").as_posix(),
    }


def test_console_data_rejects_local_state_roots() -> None:
    with pytest.raises(ConsoleDataValidationError, match="local state"):
        ConsoleDataBuilder(_repository_root).discover((PurePosixPath("results"),))


def test_console_data_rejects_nested_local_result_roots() -> None:
    with pytest.raises(ConsoleDataValidationError, match="results or results/views"):
        ConsoleDataBuilder(_repository_root).discover(
            (PurePosixPath("tests/fixtures"),),
            result_roots=(Path("results/training"),),
        )


def test_console_data_rejects_missing_public_roots() -> None:
    with pytest.raises(ConsoleDataValidationError, match="does not name a directory"):
        ConsoleDataBuilder(_repository_root).discover((PurePosixPath("tests/missing"),))


def test_console_data_rejects_roots_without_supported_documents(tmp_path: Path) -> None:
    (tmp_path / "README.txt").write_text("not a supported document", encoding="utf-8")

    with pytest.raises(ConsoleDataValidationError, match="did not contain supported documents"):
        ConsoleDataBuilder(tmp_path).discover((PurePosixPath("."),))
