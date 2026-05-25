from pathlib import Path

from leibniz.benchmarks import BenchmarkManifestDocument
from leibniz.content import ContentDigest
from leibniz.measurements import MeasurementDocument

_FIXTURES = Path(__file__).parent / "fixtures" / "finite_answer"


def test_minimal_authoring_fixtures_load_and_validate() -> None:
    manifest_document = BenchmarkManifestDocument.from_bytes(
        (_FIXTURES / "minimal_benchmark_manifest.json").read_bytes()
    )
    measurement_document = MeasurementDocument.from_bytes(
        (_FIXTURES / "minimal_measurement.json").read_bytes()
    )

    measurement_document.measurement.validate_manifest(manifest_document.manifest)
    assert manifest_document.digest == ContentDigest.from_value(
        manifest_document.manifest.to_record()
    )
    assert measurement_document.digest == ContentDigest.from_value(
        measurement_document.measurement.to_record()
    )
    assert measurement_document.measurement.scoring_bundle.raw_scoring_evidence.to_record() == {
        "id": "core.boolean-evidence@0.1.0",
        "observation_id": "observation-1",
        "answer_space_id": "core.boolean-answer@0.1.0",
        "accepted_event_id": "core.boolean-accepted@0.1.0",
        "probability_measure_id": "core.boolean-prediction@0.1.0",
        "accepted_mass": 0.25,
        "negative_log_score": 1.3862943611198906,
    }
