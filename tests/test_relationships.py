from collections.abc import Callable
from pathlib import Path

from leibniz.architectures import ArchitectureManifestDocument
from leibniz.content import ContentDigest
from leibniz.documents import canonical_document_bytes
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset, MeasurementDocument
from leibniz.relationships import (
    RelationshipFitDocument,
    RelationshipFitParameter,
    RelationshipFitRecord,
    RelationshipFitResiduals,
    RelationshipFitValidationError,
)

_fixtures_root = Path(__file__).parent / "fixtures"


def test_relationship_fit_record_parses_and_canonicalizes() -> None:
    dataset = _measurement_dataset()
    architecture = _architecture_document().manifest
    fit = RelationshipFitRecord.from_record(
        _relationship_fit_record(),
        dataset=dataset,
        architecture=architecture,
    )

    assert fit == RelationshipFitRecord(
        id=ProtocolIdentifier.parse("relationship-fits.boolean-linear@0.1.0"),
        source_dataset_digest=dataset.digest,
        architecture_id=architecture.id,
        hypothesis_family="affine-score-vs-parameter-count",
        parameters=(
            RelationshipFitParameter(name="intercept", value=1.0),
            RelationshipFitParameter(name="slope", value=-0.25),
        ),
        residuals=RelationshipFitResiduals(rmse=0.1, max_abs=0.2, r_squared=0.9),
        point_count=1,
    )
    assert fit.to_record() == _expanded_relationship_fit_record(fit)
    assert fit.digest == ContentDigest.from_value(fit.to_record())


def test_relationship_fit_document_loads_bytes_with_digest() -> None:
    dataset = _measurement_dataset()
    architecture = _architecture_document().manifest

    document = RelationshipFitDocument.from_bytes(
        canonical_document_bytes(_relationship_fit_record()),
        dataset=dataset,
        architecture=architecture,
    )

    assert document.fit.source_dataset_digest == dataset.digest
    assert document.digest == ContentDigest.from_value(document.fit.to_record())


def test_relationship_fit_rejects_corrupt_source_references() -> None:
    dataset = _measurement_dataset()
    architecture = _architecture_document().manifest

    record = _relationship_fit_record()
    record["source_dataset_digest"] = str(ContentDigest.from_value({"other": True}))
    assert str(
        capture_relationship_error(
            lambda: RelationshipFitRecord.from_record(
                record,
                dataset=dataset,
                architecture=architecture,
            )
        )
    ) == "source_dataset_digest does not match dataset"

    record = _relationship_fit_record()
    record["architecture_id"] = "architecture.sha-0@0.1.0"
    assert str(
        capture_relationship_error(
            lambda: RelationshipFitRecord.from_record(
                record,
                dataset=dataset,
                architecture=architecture,
            )
        )
    ) == "architecture_id does not match architecture"

    assert str(
        capture_relationship_error(
            lambda: RelationshipFitRecord.from_record(record, dataset=dataset)
        )
    ) == "architecture source is required"


def test_relationship_fit_rejects_malformed_parameters_and_residuals() -> None:
    dataset = _measurement_dataset()
    architecture = _architecture_document().manifest

    record = _relationship_fit_record()
    record["hypothesis_family"] = ""
    assert str(
        capture_relationship_error(
            lambda: RelationshipFitRecord.from_record(
                record,
                dataset=dataset,
                architecture=architecture,
            )
        )
    ) == "hypothesis_family must be nonempty"

    record = _relationship_fit_record()
    record["parameters"] = []
    assert str(
        capture_relationship_error(
            lambda: RelationshipFitRecord.from_record(
                record,
                dataset=dataset,
                architecture=architecture,
            )
        )
    ) == "parameters must contain at least one parameter"

    record = _relationship_fit_record()
    record["parameters"] = [
        {"name": "slope", "value": 1.0},
        {"name": "slope", "value": 2.0},
    ]
    assert str(
        capture_relationship_error(
            lambda: RelationshipFitRecord.from_record(
                record,
                dataset=dataset,
                architecture=architecture,
            )
        )
    ) == "duplicate parameter name: slope"

    record = _relationship_fit_record()
    record["residuals"] = {"rmse": -0.1, "max_abs": 0.2}
    assert str(
        capture_relationship_error(
            lambda: RelationshipFitRecord.from_record(
                record,
                dataset=dataset,
                architecture=architecture,
            )
        )
    ) == "residuals.rmse must be nonnegative"


def test_relationship_fit_rejects_point_count_and_local_state_fields() -> None:
    dataset = _measurement_dataset()
    architecture = _architecture_document().manifest

    record = _relationship_fit_record()
    record["point_count"] = 2
    assert str(
        capture_relationship_error(
            lambda: RelationshipFitRecord.from_record(
                record,
                dataset=dataset,
                architecture=architecture,
            )
        )
    ) == "point_count exceeds source dataset size"

    record = _relationship_fit_record()
    record["local_path"] = "results/measurement_records/digits/relationship_fits.json"
    assert str(
        capture_relationship_error(
            lambda: RelationshipFitRecord.from_record(
                record,
                dataset=dataset,
                architecture=architecture,
            )
        )
    ) == "local_path: unknown field"


def test_relationship_fit_document_rejects_invalid_bytes() -> None:
    assert str(
        capture_relationship_error(
            lambda: RelationshipFitDocument.from_bytes(b"[]", dataset=_measurement_dataset())
        )
    ) == "relationship fit document must contain an object"


def _relationship_fit_record() -> dict[str, object]:
    return {
        "id": "relationship-fits.boolean-linear@0.1.0",
        "source_dataset_digest": str(_measurement_dataset().digest),
        "architecture_id": str(_architecture_document().manifest.id),
        "hypothesis_family": "affine-score-vs-parameter-count",
        "parameters": [
            {"name": "slope", "value": -0.25},
            {"name": "intercept", "value": 1.0},
        ],
        "residuals": {
            "rmse": 0.1,
            "max_abs": 0.2,
            "r_squared": 0.9,
        },
        "point_count": 1,
    }


def _expanded_relationship_fit_record(fit: RelationshipFitRecord) -> dict[str, object]:
    return {
        "id": "relationship-fits.boolean-linear@0.1.0",
        "source_dataset_digest": str(_measurement_dataset().digest),
        "hypothesis_family": "affine-score-vs-parameter-count",
        "parameters": [
            {"name": "intercept", "value": 1.0},
            {"name": "slope", "value": -0.25},
        ],
        "residuals": {
            "rmse": 0.1,
            "max_abs": 0.2,
            "r_squared": 0.9,
        },
        "point_count": 1,
        "architecture_id": str(fit.architecture_id),
    }


def _measurement_dataset() -> MeasurementDataset:
    measurement = MeasurementDocument.from_bytes(
        (_fixtures_root / "finite_outcome" / "measurement.json").read_bytes()
    ).measurement
    return MeasurementDataset.from_record(
        {
            "measurements": [
                measurement.to_record(),
            ]
        }
    )


def _architecture_document() -> ArchitectureManifestDocument:
    return ArchitectureManifestDocument.from_bytes(
        (_fixtures_root / "architecture" / "digits_pool" / "manifest.json").read_bytes()
    )


def capture_relationship_error(
    action: Callable[[], object],
) -> RelationshipFitValidationError:
    try:
        action()
    except RelationshipFitValidationError as error:
        return error
    raise AssertionError("expected RelationshipFitValidationError")
