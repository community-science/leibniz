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
        (
            "performance-view-bundle",
            "src/leibniz/benchmarks/digits/performance_view_bundle.json",
        ),
    ]
    assert [(detail["kind"], detail["source_path"]) for detail in details] == [
        (artifact["kind"], artifact["source_path"]) for artifact in artifacts
    ]
    assert {artifact["validation_status"] for artifact in artifacts} == {"valid"}

    inspections = cast(list[dict[str, object]], record["observation_inspections"])
    assert [inspection["label"] for inspection in inspections] == [
        "Single digit 7",
        "Three digit sequence 123",
    ]
    assert inspections[0]["scale_assignment"] == {"values": [{"axis": "L", "value": 1}]}
    assert inspections[1]["complexity_assignment"] == {"values": [{"axis": "C", "value": 3}]}
    assert inspections[1]["component_sequence"] == [1, 2, 3]
    assert inspections[1]["outcome_id"] == "digit-1-2-3"

    performance_detail = next(
        detail for detail in details if detail["kind"] == "performance-view-bundle"
    )
    assert performance_detail["measurement_count"] == 2
    assert performance_detail["view_id"] == "views.competence-integrals.digits.performance@0.1.0"
    assert performance_detail["expected_complexities"] == [1.0, 2.0, 3.0]
    cases = cast(list[dict[str, object]], performance_detail["measurement_cases"])
    assert cases[1]["component_sequence"] == [1, 2]
    assert cases[1]["accepted_outcome_sequence"] == [1, 2]

    performance_views = cast(list[dict[str, object]], record["performance_views"])
    assert len(performance_views) == 1
    performance_view = performance_views[0]
    assert performance_view["source_path"] == (
        "src/leibniz/benchmarks/digits/performance_view_bundle.json"
    )
    competence_view = cast(dict[str, object], performance_view["competence_integral_view"])
    entries = cast(list[dict[str, object]], competence_view["entries"])
    assert competence_view["expected_complexities"] == [1.0, 2.0, 3.0]
    assert entries[0]["integral"] == 0.25
    assert entries[0]["coverage"] == 2 / 3
    assert entries[0]["missing_complexities"] == [3.0]

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
        "unknown_parameter_layers": [],
    }
    model_layers = cast(list[dict[str, object]], model_inspection["layers"])
    assert [(layer["kind"], layer.get("output_shape")) for layer in model_layers] == [
        ("adaptive-pooling", [1, 2, 2]),
        ("flatten", [4]),
        ("dense", [10]),
    ]


def test_console_data_includes_public_source_module_inventory() -> None:
    data = ConsoleDataBuilder(_repository_root).discover((PurePosixPath("tests/fixtures"),))
    record = data.to_record()
    modules = {
        source_module["module_name"]: source_module
        for source_module in cast(list[dict[str, object]], record["source_modules"])
    }

    assert "leibniz.console.data" in modules
    console_data = modules["leibniz.console.data"]

    assert console_data["source_path"] == "src/leibniz/console/data.py"
    assert console_data["public_exports"] == [
        "ConsoleData",
        "ConsoleDataBuilder",
        "ConsoleDataValidationError",
    ]
    validation_commands = cast(list[str], console_data["validation_commands"])
    assert "python -m pytest tests/test_console_data.py" in validation_commands
    assert "python -m pytest tests/test_public_surface.py" in validation_commands


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

    data = ConsoleDataBuilder(_repository_root).discover(
        (PurePosixPath("tests/fixtures"),),
        result_roots=(result_root,),
    )
    record = data.to_record()
    result_views = cast(list[dict[str, object]], record["result_views"])

    assert len(result_views) == 1
    assert result_views[0]["source_path"] == (result_root / "imported_results.json").as_posix()
    bundles = cast(list[dict[str, object]], result_views[0]["publication_bundles"])
    assert bundles[0]["measurement_count"] == 1


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
