"""Relationship-fit records derived from measurement datasets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz._documents import ContentEncodingError, load_object_document
from leibniz.architectures import ArchitectureManifest
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "RelationshipFitDocument",
    "RelationshipFitParameter",
    "RelationshipFitRecord",
    "RelationshipFitResiduals",
    "RelationshipFitValidationError",
]

_relationship_fit_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "source_dataset_digest": FieldSpec(kind="string"),
        "architecture_id": FieldSpec(kind="identifier", required=False),
        "hypothesis_family": FieldSpec(kind="string"),
        "parameters": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
        "residuals": FieldSpec(kind="record"),
        "point_count": FieldSpec(kind="integer"),
    }
)
_relationship_fit_parameter_record = RecordSpec(
    fields={
        "name": FieldSpec(kind="string"),
        "value": FieldSpec(kind="number"),
    }
)
_relationship_fit_residuals_record = RecordSpec(
    fields={
        "rmse": FieldSpec(kind="number"),
        "max_abs": FieldSpec(kind="number"),
        "r_squared": FieldSpec(kind="number", required=False),
    }
)


class RelationshipFitValidationError(ValueError):
    """Raised when a relationship-fit record is invalid."""


@dataclass(frozen=True, slots=True)
class RelationshipFitParameter:
    """One numeric fit parameter."""

    name: str
    value: float

    def __post_init__(self) -> None:
        if not self.name:
            raise RelationshipFitValidationError("parameter name must be nonempty")
        _require_finite(self.value, field=f"parameters.{self.name}")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> RelationshipFitParameter:
        try:
            validated = _relationship_fit_parameter_record.validate(record)
        except ValueError as error:
            raise RelationshipFitValidationError(str(error)) from error
        return cls(
            name=str(validated["name"]),
            value=_as_float(validated["value"], field="parameters.value"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value": self.value,
        }


@dataclass(frozen=True, slots=True)
class RelationshipFitResiduals:
    """Residual summary for a fitted relationship."""

    rmse: float
    max_abs: float
    r_squared: float | None = None

    def __post_init__(self) -> None:
        _require_nonnegative_finite(self.rmse, field="residuals.rmse")
        _require_nonnegative_finite(self.max_abs, field="residuals.max_abs")
        if self.r_squared is not None:
            _require_finite(self.r_squared, field="residuals.r_squared")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> RelationshipFitResiduals:
        try:
            validated = _relationship_fit_residuals_record.validate(record)
        except ValueError as error:
            raise RelationshipFitValidationError(str(error)) from error
        r_squared = validated.get("r_squared")
        return cls(
            rmse=_as_float(validated["rmse"], field="residuals.rmse"),
            max_abs=_as_float(validated["max_abs"], field="residuals.max_abs"),
            r_squared=(
                _as_float(r_squared, field="residuals.r_squared")
                if r_squared is not None
                else None
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "rmse": self.rmse,
            "max_abs": self.max_abs,
        }
        if self.r_squared is not None:
            record["r_squared"] = self.r_squared
        return record


@dataclass(frozen=True, slots=True)
class RelationshipFitRecord:
    """A derived scientific relationship fit over measurement evidence."""

    id: ProtocolIdentifier
    source_dataset_digest: ContentDigest
    hypothesis_family: str
    parameters: tuple[RelationshipFitParameter, ...]
    residuals: RelationshipFitResiduals
    point_count: int
    architecture_id: ProtocolIdentifier | None = None

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise RelationshipFitValidationError(str(error)) from error
        if not str(self.id.name).startswith("relationship-fits."):
            raise RelationshipFitValidationError("id must be a valid relationship fit id")
        if not self.hypothesis_family:
            raise RelationshipFitValidationError("hypothesis_family must be nonempty")
        if not self.parameters:
            raise RelationshipFitValidationError("parameters must contain at least one parameter")
        _reject_duplicate_parameter_names(self.parameters)
        object.__setattr__(
            self,
            "parameters",
            tuple(sorted(self.parameters, key=lambda parameter: parameter.name)),
        )
        if self.point_count <= 0:
            raise RelationshipFitValidationError("point_count must be positive")

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        dataset: MeasurementDataset,
        architecture: ArchitectureManifest | None = None,
    ) -> RelationshipFitRecord:
        try:
            validated = _relationship_fit_record.validate(record)
            parameters = tuple(
                RelationshipFitParameter.from_record(_as_mapping(item, field="parameters"))
                for item in _as_sequence(validated["parameters"], field="parameters")
            )
            residuals = RelationshipFitResiduals.from_record(
                _as_mapping(validated["residuals"], field="residuals")
            )
        except ValueError as error:
            raise RelationshipFitValidationError(str(error)) from error
        fit = cls(
            id=_as_identifier(validated["id"], field="id"),
            source_dataset_digest=_as_digest(
                validated["source_dataset_digest"],
                field="source_dataset_digest",
            ),
            architecture_id=(
                _as_identifier(validated["architecture_id"], field="architecture_id")
                if "architecture_id" in validated
                else None
            ),
            hypothesis_family=str(validated["hypothesis_family"]),
            parameters=parameters,
            residuals=residuals,
            point_count=_as_int(validated["point_count"], field="point_count"),
        )
        fit.validate_sources(dataset=dataset, architecture=architecture)
        return fit

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_sources(
        self,
        *,
        dataset: MeasurementDataset,
        architecture: ArchitectureManifest | None = None,
    ) -> None:
        if self.source_dataset_digest != dataset.digest:
            raise RelationshipFitValidationError("source_dataset_digest does not match dataset")
        if self.point_count > len(dataset.measurements):
            raise RelationshipFitValidationError("point_count exceeds source dataset size")
        if self.architecture_id is not None:
            if architecture is None:
                raise RelationshipFitValidationError("architecture source is required")
            if self.architecture_id != architecture.id:
                raise RelationshipFitValidationError("architecture_id does not match architecture")

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "source_dataset_digest": str(self.source_dataset_digest),
            "hypothesis_family": self.hypothesis_family,
            "parameters": [
                parameter.to_record()
                for parameter in sorted(self.parameters, key=lambda parameter: parameter.name)
            ],
            "residuals": self.residuals.to_record(),
            "point_count": self.point_count,
        }
        if self.architecture_id is not None:
            record["architecture_id"] = str(self.architecture_id)
        return record


@dataclass(frozen=True, slots=True)
class RelationshipFitDocument:
    """A loaded relationship-fit record and its canonical digest."""

    fit: RelationshipFitRecord
    digest: ContentDigest

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        dataset: MeasurementDataset,
        architecture: ArchitectureManifest | None = None,
    ) -> RelationshipFitDocument:
        try:
            record = load_object_document(data, description="relationship fit document")
        except ContentEncodingError as error:
            raise RelationshipFitValidationError(str(error)) from error
        fit = RelationshipFitRecord.from_record(
            record,
            dataset=dataset,
            architecture=architecture,
        )
        return cls(fit=fit, digest=fit.digest)


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise RelationshipFitValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RelationshipFitValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise RelationshipFitValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_digest(value: object, *, field: str) -> ContentDigest:
    if not isinstance(value, str):
        raise RelationshipFitValidationError(f"{field}: expected digest string")
    algorithm, separator, digest_hex = value.partition(":")
    if separator == "":
        raise RelationshipFitValidationError(f"{field}: expected algorithm:digest")
    try:
        return ContentDigest(algorithm=algorithm, hex=digest_hex)
    except ContentEncodingError as error:
        raise RelationshipFitValidationError(str(error)) from error


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, float):
        return value
    raise RelationshipFitValidationError(f"{field}: expected parsed number")


def _as_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise RelationshipFitValidationError(f"{field}: expected parsed integer")
    return value


def _require_finite(value: float, *, field: str) -> None:
    if not math.isfinite(value):
        raise RelationshipFitValidationError(f"{field} must be finite")


def _require_nonnegative_finite(value: float, *, field: str) -> None:
    _require_finite(value, field=field)
    if value < 0:
        raise RelationshipFitValidationError(f"{field} must be nonnegative")


def _reject_duplicate_parameter_names(parameters: tuple[RelationshipFitParameter, ...]) -> None:
    seen: set[str] = set()
    for parameter in parameters:
        if parameter.name in seen:
            raise RelationshipFitValidationError(
                f"duplicate parameter name: {parameter.name}"
            )
        seen.add(parameter.name)
