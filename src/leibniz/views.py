"""Derived views over measurement datasets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from leibniz.artifacts import ArtifactReference
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.materialization import MaterializationPlan
from leibniz.measurements import MeasurementDataset, MeasurementRecord
from leibniz.records import FieldSpec, RecordSpec

__all__ = [
    "CompetenceIntegralEntry",
    "CompetenceIntegralPoint",
    "CompetenceIntegralSource",
    "CompetenceIntegralView",
    "CompetenceIntegralViewDocument",
    "MeasurementScoreEntry",
    "MeasurementScoreView",
    "MeasurementScoreViewDocument",
    "MeasurementScoreViewValidationError",
]

_CompetenceIntegralRule = Literal["normalized-trapezoid-accepted-mass-v1"]
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
_competence_integral_view_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "source_dataset_digest": FieldSpec(kind="string"),
        "projection_rule": FieldSpec(
            kind="literal",
            literal="normalized-trapezoid-accepted-mass-v1",
        ),
        "complexity_axis": FieldSpec(kind="string"),
        "expected_complexities": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="number"),
        ),
        "entries": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
    }
)
_competence_integral_entry_record = RecordSpec(
    fields={
        "benchmark_id": FieldSpec(kind="identifier"),
        "integral": FieldSpec(kind="number"),
        "coverage": FieldSpec(kind="number"),
        "observed_complexities": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="number"),
        ),
        "missing_complexities": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="number"),
        ),
        "points": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
    }
)
_competence_integral_point_record = RecordSpec(
    fields={
        "measurement_id": FieldSpec(kind="identifier"),
        "materialization_plan": FieldSpec(kind="record"),
        "complexity": FieldSpec(kind="number"),
        "competence": FieldSpec(kind="number"),
    }
)


class MeasurementScoreViewValidationError(ValueError):
    """Raised when a derived measurement score view is invalid."""


@dataclass(frozen=True, slots=True)
class CompetenceIntegralSource:
    """A measurement paired with the materialization plan that declares its complexity."""

    measurement: MeasurementRecord
    materialization_plan: MaterializationPlan

    def __post_init__(self) -> None:
        if self.measurement.benchmark_id != self.materialization_plan.benchmark_id:
            raise MeasurementScoreViewValidationError(
                "measurement benchmark_id does not match materialization plan"
            )
        expected = ArtifactReference(
            kind="materialization-plan",
            protocol_id=self.materialization_plan.id,
            record_digest=self.materialization_plan.digest,
        )
        if expected not in self.measurement.evidence_artifacts:
            raise MeasurementScoreViewValidationError(
                "measurement does not cite materialization plan"
            )


@dataclass(frozen=True, slots=True)
class CompetenceIntegralPoint:
    """One competence sample at a declared complexity coordinate."""

    measurement_id: ProtocolIdentifier
    materialization_plan: ArtifactReference
    complexity: float
    competence: float

    def __post_init__(self) -> None:
        _require_finite_nonnegative(self.complexity, field="complexity")
        _require_finite_probability(self.competence, field="competence")
        if self.materialization_plan.kind != "materialization-plan":
            raise MeasurementScoreViewValidationError(
                "materialization_plan reference must have kind materialization-plan"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> CompetenceIntegralPoint:
        try:
            validated = _competence_integral_point_record.validate(record)
        except ValueError as error:
            raise MeasurementScoreViewValidationError(str(error)) from error
        return cls(
            measurement_id=_as_identifier(validated["measurement_id"], field="measurement_id"),
            materialization_plan=ArtifactReference.from_record(
                _as_mapping(validated["materialization_plan"], field="materialization_plan")
            ),
            complexity=_as_float(validated["complexity"], field="complexity"),
            competence=_as_float(validated["competence"], field="competence"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "measurement_id": str(self.measurement_id),
            "materialization_plan": self.materialization_plan.to_record(),
            "complexity": self.complexity,
            "competence": self.competence,
        }


@dataclass(frozen=True, slots=True)
class CompetenceIntegralEntry:
    """One derived competence integral for a benchmark."""

    benchmark_id: ProtocolIdentifier
    integral: float
    coverage: float
    observed_complexities: tuple[float, ...]
    missing_complexities: tuple[float, ...]
    points: tuple[CompetenceIntegralPoint, ...]

    def __post_init__(self) -> None:
        _require_finite_probability(self.integral, field="integral")
        _require_finite_probability(self.coverage, field="coverage")
        _require_ordered_values(self.observed_complexities, field="observed_complexities")
        _require_ordered_values(
            self.missing_complexities,
            field="missing_complexities",
            allow_empty=True,
        )
        expected_points = tuple(sorted(self.points, key=_competence_point_sort_key))
        if self.points != expected_points:
            raise MeasurementScoreViewValidationError("points must be sorted by complexity")
        duplicate = _first_duplicate(tuple(point.measurement_id for point in self.points))
        if duplicate is not None:
            raise MeasurementScoreViewValidationError(f"duplicate measurement id: {duplicate}")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> CompetenceIntegralEntry:
        try:
            validated = _competence_integral_entry_record.validate(record)
        except ValueError as error:
            raise MeasurementScoreViewValidationError(str(error)) from error
        return cls(
            benchmark_id=_as_identifier(validated["benchmark_id"], field="benchmark_id"),
            integral=_as_float(validated["integral"], field="integral"),
            coverage=_as_float(validated["coverage"], field="coverage"),
            observed_complexities=_float_sequence(
                validated["observed_complexities"],
                field="observed_complexities",
            ),
            missing_complexities=_float_sequence(
                validated["missing_complexities"],
                field="missing_complexities",
                allow_empty=True,
            ),
            points=tuple(
                CompetenceIntegralPoint.from_record(_as_mapping(point, field="points"))
                for point in _as_sequence(validated["points"], field="points")
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "benchmark_id": str(self.benchmark_id),
            "integral": self.integral,
            "coverage": self.coverage,
            "observed_complexities": list(self.observed_complexities),
            "missing_complexities": list(self.missing_complexities),
            "points": [point.to_record() for point in self.points],
        }


@dataclass(frozen=True, slots=True)
class CompetenceIntegralView:
    """A normalized competence integral derived from complexity-indexed measurements."""

    id: ProtocolIdentifier
    source_dataset_digest: ContentDigest
    projection_rule: _CompetenceIntegralRule
    complexity_axis: str
    expected_complexities: tuple[float, ...]
    entries: tuple[CompetenceIntegralEntry, ...]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise MeasurementScoreViewValidationError(str(error)) from error
        if not str(self.id.name).startswith("views.competence-integrals."):
            raise MeasurementScoreViewValidationError(
                "id must be a valid competence integral view id"
            )
        if self.projection_rule != "normalized-trapezoid-accepted-mass-v1":
            raise MeasurementScoreViewValidationError(
                f"unsupported projection_rule: {self.projection_rule}"
            )
        if not self.complexity_axis:
            raise MeasurementScoreViewValidationError("complexity_axis must be nonempty")
        _require_ordered_values(self.expected_complexities, field="expected_complexities")
        if not self.entries:
            raise MeasurementScoreViewValidationError("entries must not be empty")
        expected_entries = tuple(sorted(self.entries, key=_competence_entry_sort_key))
        if self.entries != expected_entries:
            raise MeasurementScoreViewValidationError("entries must be sorted by integral")
        duplicate = _first_duplicate(tuple(entry.benchmark_id for entry in self.entries))
        if duplicate is not None:
            raise MeasurementScoreViewValidationError(f"duplicate benchmark id: {duplicate}")

    @classmethod
    def from_sources(
        cls,
        *,
        id: ProtocolIdentifier,
        dataset: MeasurementDataset,
        sources: Sequence[CompetenceIntegralSource],
        complexity_axis: str,
        expected_complexities: Sequence[float],
    ) -> CompetenceIntegralView:
        normalized_expected = _ordered_unique_values(
            tuple(
                _as_float(value, field="expected_complexities")
                for value in expected_complexities
            ),
            field="expected_complexities",
        )
        _validate_sources_belong_to_dataset(sources=tuple(sources), dataset=dataset)
        return cls(
            id=id,
            source_dataset_digest=dataset.digest,
            projection_rule="normalized-trapezoid-accepted-mass-v1",
            complexity_axis=complexity_axis,
            expected_complexities=normalized_expected,
            entries=tuple(
                sorted(
                    _entries_from_sources(
                        sources=tuple(sources),
                        complexity_axis=complexity_axis,
                        expected_complexities=normalized_expected,
                    ),
                    key=_competence_entry_sort_key,
                )
            ),
        )

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
        *,
        dataset: MeasurementDataset,
        sources: Sequence[CompetenceIntegralSource],
    ) -> CompetenceIntegralView:
        try:
            validated = _competence_integral_view_record.validate(record)
            entries = tuple(
                CompetenceIntegralEntry.from_record(_as_mapping(entry, field="entries"))
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
            projection_rule=cast(_CompetenceIntegralRule, validated["projection_rule"]),
            complexity_axis=str(validated["complexity_axis"]),
            expected_complexities=_float_sequence(
                validated["expected_complexities"],
                field="expected_complexities",
            ),
            entries=entries,
        )
        expected = cls.from_sources(
            id=view.id,
            dataset=dataset,
            sources=sources,
            complexity_axis=view.complexity_axis,
            expected_complexities=view.expected_complexities,
        )
        if view.source_dataset_digest != expected.source_dataset_digest:
            raise MeasurementScoreViewValidationError(
                "source_dataset_digest does not match dataset"
            )
        if view.entries != expected.entries:
            raise MeasurementScoreViewValidationError(
                "entries do not match derived competence integral view"
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
            "complexity_axis": self.complexity_axis,
            "expected_complexities": list(self.expected_complexities),
            "entries": [entry.to_record() for entry in self.entries],
        }


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


@dataclass(frozen=True, slots=True)
class CompetenceIntegralViewDocument:
    """A loaded competence integral view and its canonical digest."""

    view: CompetenceIntegralView
    digest: ContentDigest

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        dataset: MeasurementDataset,
        sources: Sequence[CompetenceIntegralSource],
    ) -> CompetenceIntegralViewDocument:
        try:
            record = load_object_document(data, description="competence integral view document")
        except ContentEncodingError as error:
            raise MeasurementScoreViewValidationError(str(error)) from error
        view = CompetenceIntegralView.from_record(
            record,
            dataset=dataset,
            sources=sources,
        )
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


def _entries_from_sources(
    *,
    sources: tuple[CompetenceIntegralSource, ...],
    complexity_axis: str,
    expected_complexities: tuple[float, ...],
) -> tuple[CompetenceIntegralEntry, ...]:
    grouped: dict[ProtocolIdentifier, list[CompetenceIntegralPoint]] = {}
    for source in sources:
        evidence = source.measurement.raw_scoring_evidence
        complexity = float(
            source.materialization_plan.complexity_assignment.require_axis(complexity_axis)
        )
        point = CompetenceIntegralPoint(
            measurement_id=evidence.id,
            materialization_plan=ArtifactReference(
                kind="materialization-plan",
                protocol_id=source.materialization_plan.id,
                record_digest=source.materialization_plan.digest,
            ),
            complexity=complexity,
            competence=evidence.accepted_mass,
        )
        grouped.setdefault(source.measurement.benchmark_id, []).append(point)
    return tuple(
        _competence_entry(
            benchmark_id=benchmark_id,
            points=tuple(points),
            expected_complexities=expected_complexities,
        )
        for benchmark_id, points in grouped.items()
    )


def _validate_sources_belong_to_dataset(
    *,
    sources: tuple[CompetenceIntegralSource, ...],
    dataset: MeasurementDataset,
) -> None:
    if not sources:
        raise MeasurementScoreViewValidationError("sources must not be empty")
    dataset_ids = {
        measurement.raw_scoring_evidence.id
        for measurement in dataset.measurements
    }
    source_ids = tuple(source.measurement.raw_scoring_evidence.id for source in sources)
    duplicate = _first_duplicate(source_ids)
    if duplicate is not None:
        raise MeasurementScoreViewValidationError(f"duplicate source measurement id: {duplicate}")
    missing = tuple(source_id for source_id in source_ids if source_id not in dataset_ids)
    if missing:
        raise MeasurementScoreViewValidationError(
            f"source measurement {missing[0]} is not in dataset"
        )


def _competence_entry(
    *,
    benchmark_id: ProtocolIdentifier,
    points: tuple[CompetenceIntegralPoint, ...],
    expected_complexities: tuple[float, ...],
) -> CompetenceIntegralEntry:
    ordered_points = tuple(sorted(points, key=_competence_point_sort_key))
    observed = _ordered_unique_values(
        tuple(point.complexity for point in ordered_points),
        field="observed_complexities",
    )
    missing = tuple(value for value in expected_complexities if value not in observed)
    coverage = len(observed) / len(expected_complexities)
    return CompetenceIntegralEntry(
        benchmark_id=benchmark_id,
        integral=_normalized_trapezoid_integral(ordered_points, expected_complexities),
        coverage=coverage,
        observed_complexities=observed,
        missing_complexities=missing,
        points=ordered_points,
    )


def _normalized_trapezoid_integral(
    points: tuple[CompetenceIntegralPoint, ...],
    expected_complexities: tuple[float, ...],
) -> float:
    if len(expected_complexities) == 1:
        for point in points:
            if point.complexity == expected_complexities[0]:
                return point.competence
        return 0.0
    expected_index = {value: index for index, value in enumerate(expected_complexities)}
    competence_by_complexity = {
        point.complexity: point.competence
        for point in points
        if point.complexity in expected_index
    }
    total_width = expected_complexities[-1] - expected_complexities[0]
    if total_width <= 0:
        raise MeasurementScoreViewValidationError(
            "expected_complexities must span a positive interval"
        )
    area = 0.0
    for left, right in zip(
        expected_complexities,
        expected_complexities[1:],
        strict=False,
    ):
        left_competence = competence_by_complexity.get(left, 0.0)
        right_competence = competence_by_complexity.get(right, 0.0)
        area += (right - left) * (left_competence + right_competence) / 2.0
    return area / total_width


def _competence_point_sort_key(point: CompetenceIntegralPoint) -> tuple[float, str]:
    return (point.complexity, str(point.measurement_id))


def _competence_entry_sort_key(entry: CompetenceIntegralEntry) -> tuple[float, str]:
    return (-entry.integral, str(entry.benchmark_id))


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


def _float_sequence(
    value: object,
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[float, ...]:
    return _ordered_unique_values(
        tuple(
            _as_float(item, field=field)
            for item in _as_sequence(value, field=field)
        ),
        field=field,
        allow_empty=allow_empty,
    )


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


def _ordered_unique_values(
    values: tuple[float, ...],
    *,
    field: str,
    allow_empty: bool = False,
) -> tuple[float, ...]:
    _require_ordered_values(values, field=field, allow_empty=allow_empty)
    if len(set(values)) != len(values):
        raise MeasurementScoreViewValidationError(f"{field} must not contain duplicates")
    return values


def _require_ordered_values(
    values: tuple[float, ...],
    *,
    field: str,
    allow_empty: bool = False,
) -> None:
    if not values and not allow_empty:
        raise MeasurementScoreViewValidationError(f"{field} must not be empty")
    for value in values:
        _require_finite_nonnegative(value, field=field)
    if values != tuple(sorted(values)):
        raise MeasurementScoreViewValidationError(f"{field} must be sorted")


def _require_finite_probability(value: float, *, field: str) -> None:
    _require_finite_nonnegative(value, field=field)
    if value > 1.0:
        raise MeasurementScoreViewValidationError(f"{field} must not exceed 1")


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
