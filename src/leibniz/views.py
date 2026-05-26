"""Derived views over measurement datasets."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast

from leibniz._documents import ContentEncodingError, load_object_document
from leibniz.content import ContentDigest
from leibniz.identifiers import ProtocolIdentifier
from leibniz.measurements import MeasurementDataset, MeasurementRecord
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "MeasurementScoreEntry",
    "MeasurementScoreView",
    "MeasurementScoreViewDocument",
    "MeasurementScoreViewValidationError",
]

_ProjectionRule = Literal["measurement-score-ascending"]

_measurement_score_view_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "source_dataset_digest": FieldSpec(kind="string"),
        "projection_rule": FieldSpec(kind="literal", literal="measurement-score-ascending"),
        "entries": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
    }
)
_measurement_score_entry_record = RecordSpec(
    fields={
        "measurement_id": FieldSpec(kind="identifier"),
        "benchmark_id": FieldSpec(kind="identifier"),
        "observation_id": FieldSpec(kind="string"),
        "accepted_mass": FieldSpec(kind="number"),
    },
    allow_unknown=True,
)


class MeasurementScoreViewValidationError(ValueError):
    """Raised when a derived measurement score view is invalid."""


@dataclass(frozen=True, slots=True)
class MeasurementScoreEntry:
    """One projected measurement score in a derived score view."""

    measurement_id: ProtocolIdentifier
    benchmark_id: ProtocolIdentifier
    observation_id: str
    accepted_mass: float
    negative_log_score: float

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise MeasurementScoreViewValidationError("observation_id must be nonempty")
        _require_finite_nonnegative(self.accepted_mass, field="accepted_mass")
        _require_finite_nonnegative(self.negative_log_score, field="negative_log_score")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> MeasurementScoreEntry:
        try:
            validated = _measurement_score_entry_record.validate(record)
        except ValueError as error:
            raise MeasurementScoreViewValidationError(str(error)) from error
        return cls(
            measurement_id=_as_identifier(validated["measurement_id"], field="measurement_id"),
            benchmark_id=_as_identifier(validated["benchmark_id"], field="benchmark_id"),
            observation_id=str(validated["observation_id"]),
            accepted_mass=_as_float(validated["accepted_mass"], field="accepted_mass"),
            negative_log_score=_as_score(record),
        )

    def to_record(self) -> dict[str, object]:
        negative_log_score: float | str
        if self.negative_log_score == math.inf:
            negative_log_score = "infinity"
        else:
            negative_log_score = self.negative_log_score
        return {
            "measurement_id": str(self.measurement_id),
            "benchmark_id": str(self.benchmark_id),
            "observation_id": self.observation_id,
            "accepted_mass": self.accepted_mass,
            "negative_log_score": negative_log_score,
        }


@dataclass(frozen=True, slots=True)
class MeasurementScoreView:
    """A deterministic score projection derived from a measurement dataset."""

    id: ProtocolIdentifier
    source_dataset_digest: ContentDigest
    projection_rule: _ProjectionRule
    entries: tuple[MeasurementScoreEntry, ...]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise MeasurementScoreViewValidationError(str(error)) from error
        if self.projection_rule != "measurement-score-ascending":
            raise MeasurementScoreViewValidationError(
                f"unsupported projection_rule: {self.projection_rule}"
            )
        if not str(self.id.name).startswith("views.measurement-scores."):
            raise MeasurementScoreViewValidationError(
                "id must be a valid measurement score view id"
            )
        expected_entries = tuple(sorted(self.entries, key=_entry_sort_key))
        if self.entries != expected_entries:
            raise MeasurementScoreViewValidationError("entries must be sorted by score")
        duplicate = _first_duplicate(tuple(entry.measurement_id for entry in self.entries))
        if duplicate is not None:
            raise MeasurementScoreViewValidationError(f"duplicate measurement id: {duplicate}")

    @classmethod
    def from_dataset(
        cls,
        *,
        id: ProtocolIdentifier,
        dataset: MeasurementDataset,
    ) -> MeasurementScoreView:
        return cls(
            id=id,
            source_dataset_digest=dataset.digest,
            projection_rule="measurement-score-ascending",
            entries=tuple(
                sorted(
                    (_entry_from_measurement(measurement) for measurement in dataset.measurements),
                    key=_entry_sort_key,
                )
            ),
        )

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        dataset: MeasurementDataset,
    ) -> MeasurementScoreView:
        try:
            validated = _measurement_score_view_record.validate(record)
            entries = tuple(
                MeasurementScoreEntry.from_record(_as_mapping(entry, field="entries"))
                for entry in _as_sequence(validated["entries"], field="entries")
            )
        except ValueError as error:
            raise MeasurementScoreViewValidationError(str(error)) from error
        view = cls(
            id=_as_identifier(validated["id"], field="id"),
            source_dataset_digest=_as_digest(
                validated["source_dataset_digest"],
                field="source_dataset_digest",
            ),
            projection_rule=cast(_ProjectionRule, validated["projection_rule"]),
            entries=entries,
        )
        expected = cls.from_dataset(id=view.id, dataset=dataset)
        if view.source_dataset_digest != expected.source_dataset_digest:
            raise MeasurementScoreViewValidationError(
                "source_dataset_digest does not match dataset"
            )
        if view.entries != expected.entries:
            raise MeasurementScoreViewValidationError(
                "entries do not match derived measurement score view"
            )
        return view

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "source_dataset_digest": str(self.source_dataset_digest),
            "projection_rule": self.projection_rule,
            "entries": [entry.to_record() for entry in self.entries],
        }


@dataclass(frozen=True, slots=True)
class MeasurementScoreViewDocument:
    """A loaded measurement score view and its canonical digest."""

    view: MeasurementScoreView
    digest: ContentDigest

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        dataset: MeasurementDataset,
    ) -> MeasurementScoreViewDocument:
        try:
            record = load_object_document(data, description="measurement score view document")
        except ContentEncodingError as error:
            raise MeasurementScoreViewValidationError(str(error)) from error
        view = MeasurementScoreView.from_record(record, dataset=dataset)
        return cls(view=view, digest=view.digest)


def _entry_from_measurement(measurement: MeasurementRecord) -> MeasurementScoreEntry:
    evidence = measurement.raw_scoring_evidence
    return MeasurementScoreEntry(
        measurement_id=evidence.id,
        benchmark_id=measurement.benchmark_id,
        observation_id=evidence.observation_id,
        accepted_mass=evidence.accepted_mass,
        negative_log_score=evidence.negative_log_score,
    )


def _entry_sort_key(entry: MeasurementScoreEntry) -> tuple[float, str]:
    return (entry.negative_log_score, str(entry.measurement_id))


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise MeasurementScoreViewValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MeasurementScoreViewValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise MeasurementScoreViewValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_digest(value: object, *, field: str) -> ContentDigest:
    if not isinstance(value, str):
        raise MeasurementScoreViewValidationError(f"{field}: expected digest string")
    algorithm, separator, digest_hex = value.partition(":")
    if separator == "":
        raise MeasurementScoreViewValidationError(f"{field}: expected algorithm:digest")
    try:
        return ContentDigest(algorithm=algorithm, hex=digest_hex)
    except ContentEncodingError as error:
        raise MeasurementScoreViewValidationError(str(error)) from error


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, float):
        return value
    raise MeasurementScoreViewValidationError(f"{field}: expected parsed number")


def _as_score(record: Mapping[str, object]) -> float:
    unknown_fields = tuple(
        sorted(
            field
            for field in record
            if field not in {*_measurement_score_entry_record.fields, "negative_log_score"}
        )
    )
    if unknown_fields:
        raise MeasurementScoreViewValidationError(f"{unknown_fields[0]}: unknown field")
    if "negative_log_score" not in record:
        raise MeasurementScoreViewValidationError(
            "negative_log_score: missing required field"
        )
    value = record["negative_log_score"]
    if value == "infinity":
        return math.inf
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise MeasurementScoreViewValidationError(
        "negative_log_score: expected finite number or 'infinity'"
    )


def _require_finite_nonnegative(value: float, *, field: str) -> None:
    if value < 0:
        raise MeasurementScoreViewValidationError(f"{field} must be nonnegative")
    if field != "negative_log_score" and not math.isfinite(value):
        raise MeasurementScoreViewValidationError(f"{field} must be finite")


def _first_duplicate(values: tuple[ProtocolIdentifier, ...]) -> ProtocolIdentifier | None:
    seen: set[ProtocolIdentifier] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
