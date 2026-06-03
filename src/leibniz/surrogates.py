"""Architecture-surrogate export records."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec

__all__ = [
    "ArchitectureSurrogateDocument",
    "ArchitectureSurrogateFeature",
    "ArchitectureSurrogateRecord",
    "ArchitectureSurrogateState",
    "ArchitectureSurrogateTrainingSummary",
    "ArchitectureSurrogateValidationError",
]

_TrainingStatus = Literal["fit", "insufficient-observations"]

_surrogate_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "source_dataset_digest": FieldSpec(kind="string"),
        "model_kind": FieldSpec(kind="string"),
        "target_name": FieldSpec(kind="string"),
        "features": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "training": FieldSpec(kind="record"),
        "state": FieldSpec(kind="record"),
    }
)
_feature_record = RecordSpec(
    fields={
        "name": FieldSpec(kind="string"),
        "mean": FieldSpec(kind="number"),
        "scale": FieldSpec(kind="number"),
        "sensitivity": FieldSpec(kind="number"),
    }
)
_training_record = RecordSpec(
    fields={
        "status": FieldSpec(kind="string"),
        "observation_count": FieldSpec(kind="integer"),
        "selector": FieldSpec(kind="string", required=False),
        "training_step_count": FieldSpec(kind="integer", required=False),
    }
)
_state_record = RecordSpec(
    fields={
        "format": FieldSpec(kind="string"),
        "input_width": FieldSpec(kind="integer"),
        "output_width": FieldSpec(kind="integer"),
        "parameter_count": FieldSpec(kind="integer"),
        "state_digest": FieldSpec(kind="string", required=False),
    }
)


class ArchitectureSurrogateValidationError(ValueError):
    """Raised when an architecture-surrogate export record is invalid."""


_extract = RecordExtractor(error_type=ArchitectureSurrogateValidationError)


@dataclass(frozen=True, slots=True)
class ArchitectureSurrogateFeature:
    """One numeric feature used by an exported architecture surrogate."""

    name: str
    mean: float
    scale: float
    sensitivity: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ArchitectureSurrogateValidationError("feature name must be nonempty")
        _require_finite(self.mean, field=f"features.{self.name}.mean")
        _require_positive_finite(self.scale, field=f"features.{self.name}.scale")
        _require_nonnegative_finite(
            self.sensitivity,
            field=f"features.{self.name}.sensitivity",
        )

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
    ) -> ArchitectureSurrogateFeature:
        try:
            validated = _feature_record.validate(record)
        except ValueError as error:
            raise ArchitectureSurrogateValidationError(str(error)) from error
        return cls(
            name=str(validated["name"]),
            mean=_extract.float(validated["mean"], "features.mean"),
            scale=_extract.float(validated["scale"], "features.scale"),
            sensitivity=_extract.float(validated["sensitivity"], "features.sensitivity"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mean": self.mean,
            "scale": self.scale,
            "sensitivity": self.sensitivity,
        }


@dataclass(frozen=True, slots=True)
class ArchitectureSurrogateTrainingSummary:
    """Training metadata for a static surrogate export."""

    status: _TrainingStatus
    observation_count: int
    selector: str | None = None
    training_step_count: int | None = None

    def __post_init__(self) -> None:
        if self.status not in {"fit", "insufficient-observations"}:
            raise ArchitectureSurrogateValidationError(
                f"unsupported training status: {self.status}"
            )
        if self.observation_count < 0:
            raise ArchitectureSurrogateValidationError("observation_count must be nonnegative")
        if self.status == "fit" and self.observation_count == 0:
            raise ArchitectureSurrogateValidationError(
                "fit surrogate must have at least one observation"
            )
        if self.selector is not None and not self.selector:
            raise ArchitectureSurrogateValidationError("selector must be nonempty")
        if self.training_step_count is not None and self.training_step_count <= 0:
            raise ArchitectureSurrogateValidationError("training_step_count must be positive")

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
    ) -> ArchitectureSurrogateTrainingSummary:
        try:
            validated = _training_record.validate(record)
        except ValueError as error:
            raise ArchitectureSurrogateValidationError(str(error)) from error
        return cls(
            status=cast(_TrainingStatus, str(validated["status"])),
            observation_count=_extract.integer(validated["observation_count"], "observation_count"),
            selector=str(validated["selector"]) if "selector" in validated else None,
            training_step_count=(
                _extract.integer(validated["training_step_count"], "training_step_count")
                if "training_step_count" in validated
                else None
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "status": self.status,
            "observation_count": self.observation_count,
        }
        if self.selector is not None:
            record["selector"] = self.selector
        if self.training_step_count is not None:
            record["training_step_count"] = self.training_step_count
        return record


@dataclass(frozen=True, slots=True)
class ArchitectureSurrogateState:
    """Compact descriptor for an exported surrogate prediction state."""

    format: str
    input_width: int
    output_width: int
    parameter_count: int
    state_digest: ContentDigest | None = None

    def __post_init__(self) -> None:
        if not self.format:
            raise ArchitectureSurrogateValidationError("state format must be nonempty")
        if self.input_width <= 0:
            raise ArchitectureSurrogateValidationError("state input_width must be positive")
        if self.output_width <= 0:
            raise ArchitectureSurrogateValidationError("state output_width must be positive")
        if self.parameter_count < 0:
            raise ArchitectureSurrogateValidationError("state parameter_count must be nonnegative")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ArchitectureSurrogateState:
        try:
            validated = _state_record.validate(record)
        except ValueError as error:
            raise ArchitectureSurrogateValidationError(str(error)) from error
        return cls(
            format=str(validated["format"]),
            input_width=_extract.integer(validated["input_width"], "state.input_width"),
            output_width=_extract.integer(validated["output_width"], "state.output_width"),
            parameter_count=_extract.integer(validated["parameter_count"], "state.parameter_count"),
            state_digest=(
                _as_digest(validated["state_digest"], field="state.state_digest")
                if "state_digest" in validated
                else None
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "format": self.format,
            "input_width": self.input_width,
            "output_width": self.output_width,
            "parameter_count": self.parameter_count,
        }
        if self.state_digest is not None:
            record["state_digest"] = str(self.state_digest)
        return record


@dataclass(frozen=True, slots=True)
class ArchitectureSurrogateRecord:
    """A static architecture-surrogate export over measurement evidence."""

    id: ProtocolIdentifier
    source_dataset_digest: ContentDigest
    model_kind: str
    target_name: str
    features: tuple[ArchitectureSurrogateFeature, ...]
    training: ArchitectureSurrogateTrainingSummary
    state: ArchitectureSurrogateState

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise ArchitectureSurrogateValidationError(str(error)) from error
        if not str(self.id.name).startswith("architecture-surrogates."):
            raise ArchitectureSurrogateValidationError(
                "id must be a valid architecture surrogate id"
            )
        if not self.model_kind:
            raise ArchitectureSurrogateValidationError("model_kind must be nonempty")
        if not self.target_name:
            raise ArchitectureSurrogateValidationError("target_name must be nonempty")
        if not self.features:
            raise ArchitectureSurrogateValidationError(
                "features must contain at least one feature"
            )
        _reject_duplicate_feature_names(self.features)
        if self.features != tuple(sorted(self.features, key=lambda feature: feature.name)):
            raise ArchitectureSurrogateValidationError("features must be sorted by name")
        if self.state.input_width != len(self.features):
            raise ArchitectureSurrogateValidationError(
                "state input_width must match feature count"
            )
        if self.training.status == "insufficient-observations" and self.state.parameter_count != 0:
            raise ArchitectureSurrogateValidationError(
                "insufficient-observations surrogate must not declare parameters"
            )

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        dataset: MeasurementDataset,
    ) -> ArchitectureSurrogateRecord:
        try:
            validated = _surrogate_record.validate(record)
            features = tuple(
                ArchitectureSurrogateFeature.from_record(_extract.mapping(item, "features"))
                for item in _extract.sequence(validated["features"], "features")
            )
            training = ArchitectureSurrogateTrainingSummary.from_record(
                _extract.mapping(validated["training"], "training")
            )
            state = ArchitectureSurrogateState.from_record(
                _extract.mapping(validated["state"], "state")
            )
        except ValueError as error:
            raise ArchitectureSurrogateValidationError(str(error)) from error
        surrogate = cls(
            id=_extract.identifier(validated["id"], "id"),
            source_dataset_digest=_as_digest(
                validated["source_dataset_digest"],
                field="source_dataset_digest",
            ),
            model_kind=str(validated["model_kind"]),
            target_name=str(validated["target_name"]),
            features=features,
            training=training,
            state=state,
        )
        surrogate.validate_sources(dataset=dataset)
        return surrogate

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_sources(self, *, dataset: MeasurementDataset) -> None:
        if self.source_dataset_digest != dataset.digest:
            raise ArchitectureSurrogateValidationError(
                "source_dataset_digest does not match dataset"
            )
        if self.training.observation_count > len(dataset.measurements):
            raise ArchitectureSurrogateValidationError(
                "observation_count exceeds source dataset size"
            )

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "source_dataset_digest": str(self.source_dataset_digest),
            "model_kind": self.model_kind,
            "target_name": self.target_name,
            "features": [feature.to_record() for feature in self.features],
            "training": self.training.to_record(),
            "state": self.state.to_record(),
        }


@dataclass(frozen=True, slots=True)
class ArchitectureSurrogateDocument:
    """A loaded architecture-surrogate record and its canonical digest."""

    surrogate: ArchitectureSurrogateRecord
    digest: ContentDigest

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        dataset: MeasurementDataset,
    ) -> ArchitectureSurrogateDocument:
        try:
            record = load_object_document(data, description="architecture surrogate document")
        except ContentEncodingError as error:
            raise ArchitectureSurrogateValidationError(str(error)) from error
        surrogate = ArchitectureSurrogateRecord.from_record(record, dataset=dataset)
        return cls(surrogate=surrogate, digest=surrogate.digest)
def _as_digest(value: object, *, field: str) -> ContentDigest:
    if not isinstance(value, str):
        raise ArchitectureSurrogateValidationError(f"{field}: expected digest string")
    algorithm, separator, digest_hex = value.partition(":")
    if separator == "":
        raise ArchitectureSurrogateValidationError(f"{field}: expected algorithm:digest")
    try:
        return ContentDigest(algorithm=algorithm, hex=digest_hex)
    except ContentEncodingError as error:
        raise ArchitectureSurrogateValidationError(str(error)) from error
def _require_finite(value: float, *, field: str) -> None:
    if not math.isfinite(value):
        raise ArchitectureSurrogateValidationError(f"{field} must be finite")


def _require_positive_finite(value: float, *, field: str) -> None:
    _require_finite(value, field=field)
    if value <= 0:
        raise ArchitectureSurrogateValidationError(f"{field} must be positive")


def _require_nonnegative_finite(value: float, *, field: str) -> None:
    _require_finite(value, field=field)
    if value < 0:
        raise ArchitectureSurrogateValidationError(f"{field} must be nonnegative")


def _reject_duplicate_feature_names(
    features: tuple[ArchitectureSurrogateFeature, ...],
) -> None:
    seen: set[str] = set()
    for feature in features:
        if feature.name in seen:
            raise ArchitectureSurrogateValidationError(
                f"duplicate feature name: {feature.name}"
            )
        seen.add(feature.name)
