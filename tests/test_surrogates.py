from collections.abc import Callable
from pathlib import Path
from typing import cast

from leibniz._documents import canonical_document_bytes
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset, MeasurementDocument
from leibniz.surrogates import (
    ArchitectureSurrogateDocument,
    ArchitectureSurrogateFeature,
    ArchitectureSurrogateRecord,
    ArchitectureSurrogateState,
    ArchitectureSurrogateTrainingSummary,
    ArchitectureSurrogateValidationError,
)

_fixtures_root = Path(__file__).parent / "fixtures"


def test_architecture_surrogate_record_parses_and_canonicalizes() -> None:
    dataset = _measurement_dataset()

    surrogate = ArchitectureSurrogateRecord.from_record(
        _surrogate_record(dataset=dataset),
        dataset=dataset,
    )

    assert surrogate == ArchitectureSurrogateRecord(
        id=ProtocolIdentifier.parse("architecture-surrogates.boolean@0.1.0"),
        source_dataset_digest=dataset.digest,
        model_kind="neural-empirical",
        target_name="negative_log_accepted_mass",
        features=(
            ArchitectureSurrogateFeature(
                name="layer_count",
                mean=2.0,
                scale=1.0,
                sensitivity=0.1,
            ),
            ArchitectureSurrogateFeature(
                name="parameter_count",
                mean=20.0,
                scale=5.0,
                sensitivity=0.3,
            ),
        ),
        training=ArchitectureSurrogateTrainingSummary(
            status="fit",
            observation_count=1,
            selector="architecture-surrogate-fit.linear-shadow",
            training_step_count=12,
        ),
        state=ArchitectureSurrogateState(
            format="dense-regressor-summary",
            input_width=2,
            output_width=1,
            parameter_count=7,
            state_digest=ContentDigest.from_value({"weights": [0.2, -0.1]}),
        ),
    )
    assert surrogate.to_record() == _surrogate_record(dataset=dataset)
    assert surrogate.digest == ContentDigest.from_value(surrogate.to_record())


def test_architecture_surrogate_document_loads_bytes_with_digest() -> None:
    dataset = _measurement_dataset()

    document = ArchitectureSurrogateDocument.from_bytes(
        canonical_document_bytes(_surrogate_record(dataset=dataset)),
        dataset=dataset,
    )

    assert document.surrogate.source_dataset_digest == dataset.digest
    assert document.digest == ContentDigest.from_value(document.surrogate.to_record())


def test_architecture_surrogate_rejects_source_mismatches() -> None:
    dataset = _measurement_dataset()

    record = _surrogate_record(dataset=dataset)
    record["source_dataset_digest"] = str(ContentDigest.from_value({"other": True}))
    assert str(
        capture_surrogate_error(
            lambda: ArchitectureSurrogateRecord.from_record(record, dataset=dataset)
        )
    ) == "source_dataset_digest does not match dataset"

    record = _surrogate_record(dataset=dataset)
    training = dict(_training_record(record))
    training["observation_count"] = 2
    record["training"] = training
    assert str(
        capture_surrogate_error(
            lambda: ArchitectureSurrogateRecord.from_record(record, dataset=dataset)
        )
    ) == "observation_count exceeds source dataset size"


def test_architecture_surrogate_rejects_feature_catalog_failures() -> None:
    dataset = _measurement_dataset()

    record = _surrogate_record(dataset=dataset)
    features = _feature_records(record)
    first = dict(features[0])
    first["name"] = features[1]["name"]
    features[0] = first
    record["features"] = features
    assert str(
        capture_surrogate_error(
            lambda: ArchitectureSurrogateRecord.from_record(record, dataset=dataset)
        )
    ) == "duplicate feature name: parameter_count"

    record = _surrogate_record(dataset=dataset)
    features = _feature_records(record)
    first = dict(features[0])
    first["scale"] = 0.0
    features[0] = first
    record["features"] = features
    assert str(
        capture_surrogate_error(
            lambda: ArchitectureSurrogateRecord.from_record(record, dataset=dataset)
        )
    ) == "features.layer_count.scale must be positive"


def test_architecture_surrogate_rejects_prediction_state_failures() -> None:
    dataset = _measurement_dataset()

    record = _surrogate_record(dataset=dataset)
    state = dict(_state_record(record))
    state["input_width"] = 3
    record["state"] = state
    assert str(
        capture_surrogate_error(
            lambda: ArchitectureSurrogateRecord.from_record(record, dataset=dataset)
        )
    ) == "state input_width must match feature count"

    record = _surrogate_record(dataset=dataset)
    training = dict(_training_record(record))
    training["status"] = "insufficient-observations"
    training["observation_count"] = 0
    record["training"] = training
    assert str(
        capture_surrogate_error(
            lambda: ArchitectureSurrogateRecord.from_record(record, dataset=dataset)
        )
    ) == "insufficient-observations surrogate must not declare parameters"


def test_architecture_surrogate_document_rejects_invalid_bytes() -> None:
    assert str(
        capture_surrogate_error(
            lambda: ArchitectureSurrogateDocument.from_bytes(
                b"[]",
                dataset=_measurement_dataset(),
            )
        )
    ) == "architecture surrogate document must contain an object"


def _surrogate_record(*, dataset: MeasurementDataset) -> dict[str, object]:
    return {
        "id": "architecture-surrogates.boolean@0.1.0",
        "source_dataset_digest": str(dataset.digest),
        "model_kind": "neural-empirical",
        "target_name": "negative_log_accepted_mass",
        "features": [
            {
                "name": "layer_count",
                "mean": 2.0,
                "scale": 1.0,
                "sensitivity": 0.1,
            },
            {
                "name": "parameter_count",
                "mean": 20.0,
                "scale": 5.0,
                "sensitivity": 0.3,
            },
        ],
        "training": {
            "status": "fit",
            "observation_count": 1,
            "selector": "architecture-surrogate-fit.linear-shadow",
            "training_step_count": 12,
        },
        "state": {
            "format": "dense-regressor-summary",
            "input_width": 2,
            "output_width": 1,
            "parameter_count": 7,
            "state_digest": str(ContentDigest.from_value({"weights": [0.2, -0.1]})),
        },
    }


def _measurement_dataset() -> MeasurementDataset:
    measurement = MeasurementDocument.from_bytes(
        (_fixtures_root / "finite_outcome" / "measurement.json").read_bytes()
    ).measurement
    return MeasurementDataset.from_record(
        {
            "measurements": [measurement.to_record()],
        }
    )


def _feature_records(record: dict[str, object]) -> list[dict[str, object]]:
    features = record["features"]
    assert isinstance(features, list)
    items = cast(list[object], features)
    return [dict(_feature_record(feature)) for feature in items]


def _feature_record(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _training_record(record: dict[str, object]) -> dict[str, object]:
    training = record["training"]
    assert isinstance(training, dict)
    return cast(dict[str, object], training)


def _state_record(record: dict[str, object]) -> dict[str, object]:
    state = record["state"]
    assert isinstance(state, dict)
    return cast(dict[str, object], state)


def capture_surrogate_error(
    action: Callable[[], object],
) -> ArchitectureSurrogateValidationError:
    try:
        action()
    except ArchitectureSurrogateValidationError as error:
        return error
    raise AssertionError("expected ArchitectureSurrogateValidationError")
