"""Source-validated bundles for benchmark performance views."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz.artifacts import ArtifactReference
from leibniz.benchmarks import BenchmarkManifest
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier, ProtocolName
from leibniz.materialization import AxisAssignment, MaterializationDeclaration, MaterializationPlan
from leibniz.measurements import MeasurementDataset, MeasurementRecord
from leibniz.observation_formation import ObservationFormationDeclaration
from leibniz.outcomes import (
    AcceptedEvent,
    FiniteProbabilityMeasure,
    ProbabilityMass,
    RawScoringEvidence,
)
from leibniz.records import FieldSpec, RecordSpec
from leibniz.views import CompetenceIntegralSource, CompetenceIntegralView

__all__ = [
    "PerformanceMeasurementCase",
    "PerformanceProbabilityMass",
    "PerformanceViewBundle",
    "PerformanceViewBundleDocument",
    "PerformanceViewBundleManifest",
    "PerformanceViewBundleValidationError",
]

_probability_mass_record = RecordSpec(
    fields={
        "outcome_id": FieldSpec(kind="string", required=False),
        "outcome_sequence": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="integer"),
            required=False,
        ),
        "probability": FieldSpec(kind="number"),
    }
)
_measurement_case_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "component_sequence": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "accepted_outcome_sequence": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="integer"),
        ),
        "scale_assignment": FieldSpec(kind="record"),
        "complexity_assignment": FieldSpec(kind="record"),
        "seed": FieldSpec(kind="integer"),
        "probabilities": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
    }
)
_bundle_manifest_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "benchmark_manifest": FieldSpec(kind="record"),
        "materialization_declaration": FieldSpec(kind="record"),
        "observation_formation_declaration": FieldSpec(kind="record"),
        "view_id": FieldSpec(kind="identifier"),
        "complexity_axis": FieldSpec(kind="string"),
        "expected_complexities": FieldSpec(kind="sequence", item=FieldSpec(kind="number")),
        "measurement_cases": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
        ),
    }
)


class PerformanceViewBundleValidationError(ValueError):
    """Raised when a performance view bundle is invalid."""


@dataclass(frozen=True, slots=True)
class PerformanceProbabilityMass:
    """A compact probability assignment for one generated measurement case."""

    probability: float
    outcome_id: str | None = None
    outcome_sequence: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (self.outcome_id is None) == (self.outcome_sequence is None):
            raise PerformanceViewBundleValidationError(
                "probability mass must declare exactly one outcome identity"
            )
        _require_probability(self.probability, field="probability")
        if self.outcome_id is not None and self.outcome_id == "":
            raise PerformanceViewBundleValidationError("outcome_id must be nonempty")
        if self.outcome_sequence is not None:
            _require_nonnegative_sequence(
                self.outcome_sequence,
                field="outcome_sequence",
            )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> PerformanceProbabilityMass:
        try:
            validated = _probability_mass_record.validate(record)
        except ValueError as error:
            raise PerformanceViewBundleValidationError(str(error)) from error
        outcome_sequence_value = validated.get("outcome_sequence")
        outcome_sequence = (
            None
            if outcome_sequence_value is None
            else tuple(
                _as_int(item, field="outcome_sequence")
                for item in _as_sequence(outcome_sequence_value, field="outcome_sequence")
            )
        )
        return cls(
            probability=_as_float(validated["probability"], field="probability"),
            outcome_id=_optional_string(validated.get("outcome_id"), field="outcome_id"),
            outcome_sequence=outcome_sequence,
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {"probability": self.probability}
        if self.outcome_id is not None:
            record["outcome_id"] = self.outcome_id
        if self.outcome_sequence is not None:
            record["outcome_sequence"] = list(self.outcome_sequence)
        return record


@dataclass(frozen=True, slots=True)
class PerformanceMeasurementCase:
    """A compact declaration of one benchmark performance measurement."""

    id: ProtocolIdentifier
    component_sequence: tuple[int, ...]
    accepted_outcome_sequence: tuple[int, ...]
    scale_assignment: AxisAssignment
    complexity_assignment: AxisAssignment
    seed: int
    probabilities: tuple[PerformanceProbabilityMass, ...]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise PerformanceViewBundleValidationError(str(error)) from error
        _require_nonnegative_sequence(self.component_sequence, field="component_sequence")
        _require_nonnegative_sequence(
            self.accepted_outcome_sequence,
            field="accepted_outcome_sequence",
        )
        if type(self.seed) is not int:
            raise PerformanceViewBundleValidationError("seed must be an integer")
        if self.seed < 0:
            raise PerformanceViewBundleValidationError("seed must be nonnegative")
        if not self.probabilities:
            raise PerformanceViewBundleValidationError("probabilities must not be empty")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> PerformanceMeasurementCase:
        try:
            validated = _measurement_case_record.validate(record)
        except ValueError as error:
            raise PerformanceViewBundleValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            component_sequence=tuple(
                _as_int(item, field="component_sequence")
                for item in _as_sequence(
                    validated["component_sequence"],
                    field="component_sequence",
                )
            ),
            accepted_outcome_sequence=tuple(
                _as_int(item, field="accepted_outcome_sequence")
                for item in _as_sequence(
                    validated["accepted_outcome_sequence"],
                    field="accepted_outcome_sequence",
                )
            ),
            scale_assignment=AxisAssignment.from_record(
                _as_mapping(validated["scale_assignment"], field="scale_assignment")
            ),
            complexity_assignment=AxisAssignment.from_record(
                _as_mapping(validated["complexity_assignment"], field="complexity_assignment")
            ),
            seed=_as_int(validated["seed"], field="seed"),
            probabilities=tuple(
                PerformanceProbabilityMass.from_record(_as_mapping(item, field="probabilities"))
                for item in _as_sequence(validated["probabilities"], field="probabilities")
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "component_sequence": list(self.component_sequence),
            "accepted_outcome_sequence": list(self.accepted_outcome_sequence),
            "scale_assignment": self.scale_assignment.to_record(),
            "complexity_assignment": self.complexity_assignment.to_record(),
            "seed": self.seed,
            "probabilities": [probability.to_record() for probability in self.probabilities],
        }


@dataclass(frozen=True, slots=True)
class PerformanceViewBundleManifest:
    """A compact benchmark-owned declaration for a derived performance view."""

    id: ProtocolIdentifier
    benchmark_manifest: ArtifactReference
    materialization_declaration: ArtifactReference
    observation_formation_declaration: ArtifactReference
    view_id: ProtocolIdentifier
    complexity_axis: str
    expected_complexities: tuple[float, ...]
    measurement_cases: tuple[PerformanceMeasurementCase, ...]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
            self.view_id.require_unreleased()
        except ValueError as error:
            raise PerformanceViewBundleValidationError(str(error)) from error
        if not str(self.id.name).startswith("performance-view-bundles."):
            raise PerformanceViewBundleValidationError(
                "id must be a valid performance view bundle id"
            )
        if not str(self.view_id.name).startswith("views.competence-integrals."):
            raise PerformanceViewBundleValidationError(
                "view_id must be a valid competence integral view id"
            )
        if self.benchmark_manifest.kind != "benchmark-manifest":
            raise PerformanceViewBundleValidationError(
                "benchmark_manifest reference must have kind benchmark-manifest"
            )
        if self.materialization_declaration.kind != "materialization-declaration":
            raise PerformanceViewBundleValidationError(
                "materialization_declaration reference must have kind materialization-declaration"
            )
        if self.observation_formation_declaration.kind != "observation-formation-declaration":
            raise PerformanceViewBundleValidationError(
                "observation_formation_declaration reference must have kind "
                "observation-formation-declaration"
            )
        if not self.complexity_axis:
            raise PerformanceViewBundleValidationError("complexity_axis must be nonempty")
        _require_ordered_values(self.expected_complexities, field="expected_complexities")
        if not self.measurement_cases:
            raise PerformanceViewBundleValidationError("measurement_cases must not be empty")
        duplicate = _first_duplicate(tuple(item.id for item in self.measurement_cases))
        if duplicate is not None:
            raise PerformanceViewBundleValidationError(
                f"duplicate measurement case id: {duplicate}"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> PerformanceViewBundleManifest:
        try:
            validated = _bundle_manifest_record.validate(record)
        except ValueError as error:
            raise PerformanceViewBundleValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            benchmark_manifest=ArtifactReference.from_record(
                _as_mapping(validated["benchmark_manifest"], field="benchmark_manifest")
            ),
            materialization_declaration=ArtifactReference.from_record(
                _as_mapping(
                    validated["materialization_declaration"],
                    field="materialization_declaration",
                )
            ),
            observation_formation_declaration=ArtifactReference.from_record(
                _as_mapping(
                    validated["observation_formation_declaration"],
                    field="observation_formation_declaration",
                )
            ),
            view_id=_as_identifier(validated["view_id"], field="view_id"),
            complexity_axis=_as_string(validated["complexity_axis"], field="complexity_axis"),
            expected_complexities=tuple(
                _as_float(item, field="expected_complexities")
                for item in _as_sequence(
                    validated["expected_complexities"],
                    field="expected_complexities",
                )
            ),
            measurement_cases=tuple(
                PerformanceMeasurementCase.from_record(
                    _as_mapping(item, field="measurement_cases")
                )
                for item in _as_sequence(validated["measurement_cases"], field="measurement_cases")
            ),
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "benchmark_manifest": self.benchmark_manifest.to_record(),
            "materialization_declaration": self.materialization_declaration.to_record(),
            "observation_formation_declaration": (
                self.observation_formation_declaration.to_record()
            ),
            "view_id": str(self.view_id),
            "complexity_axis": self.complexity_axis,
            "expected_complexities": list(self.expected_complexities),
            "measurement_cases": [item.to_record() for item in self.measurement_cases],
        }


@dataclass(frozen=True, slots=True)
class PerformanceViewBundle:
    """A validated measurement dataset and derived competence-integral view."""

    manifest: PerformanceViewBundleManifest
    measurement_dataset: MeasurementDataset
    materialization_plans: tuple[MaterializationPlan, ...]
    competence_integral_view: CompetenceIntegralView

    @classmethod
    def from_manifest(
        cls,
        manifest: PerformanceViewBundleManifest,
        *,
        benchmark_manifest: BenchmarkManifest,
        materialization_declaration: MaterializationDeclaration,
        observation_formation_declaration: ObservationFormationDeclaration,
    ) -> PerformanceViewBundle:
        try:
            _validate_sources(
                manifest=manifest,
                benchmark_manifest=benchmark_manifest,
                materialization_declaration=materialization_declaration,
                observation_formation_declaration=observation_formation_declaration,
            )
            materialization_plans = tuple(
                _materialization_plan(case, declaration=materialization_declaration)
                for case in manifest.measurement_cases
            )
            measurements = tuple(
                _measurement_record(
                    case=case,
                    benchmark_manifest=benchmark_manifest,
                    materialization_plan=plan,
                    observation_formation_declaration=observation_formation_declaration,
                )
                for case, plan in zip(
                    manifest.measurement_cases,
                    materialization_plans,
                    strict=True,
                )
            )
            dataset = MeasurementDataset(measurements=measurements)
            measurements_by_id = {
                measurement.raw_scoring_evidence.id: measurement
                for measurement in dataset.measurements
            }
            sources = tuple(
                CompetenceIntegralSource(
                    measurement=measurements_by_id[_evidence_id(case)],
                    materialization_plan=plan,
                )
                for case, plan in zip(
                    manifest.measurement_cases,
                    materialization_plans,
                    strict=True,
                )
            )
            competence_integral_view = CompetenceIntegralView.from_sources(
                id=manifest.view_id,
                dataset=dataset,
                sources=sources,
                complexity_axis=manifest.complexity_axis,
                expected_complexities=manifest.expected_complexities,
            )
        except ValueError as error:
            raise PerformanceViewBundleValidationError(str(error)) from error
        return cls(
            manifest=manifest,
            measurement_dataset=dataset,
            materialization_plans=materialization_plans,
            competence_integral_view=competence_integral_view,
        )

    @property
    def id(self) -> ProtocolIdentifier:
        return self.manifest.id

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "manifest": self.manifest.to_record(),
            "measurement_dataset": self.measurement_dataset.to_record(),
            "materialization_plans": [plan.to_record() for plan in self.materialization_plans],
            "competence_integral_view": self.competence_integral_view.to_record(),
        }


@dataclass(frozen=True, slots=True)
class PerformanceViewBundleDocument:
    """A loaded compact performance view bundle manifest and its canonical digest."""

    manifest: PerformanceViewBundleManifest
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> PerformanceViewBundleDocument:
        try:
            record = load_object_document(data, description="performance view bundle")
        except ContentEncodingError as error:
            raise PerformanceViewBundleValidationError(str(error)) from error
        manifest = PerformanceViewBundleManifest.from_record(record)
        return cls(manifest=manifest, digest=manifest.digest)


def _validate_sources(
    *,
    manifest: PerformanceViewBundleManifest,
    benchmark_manifest: BenchmarkManifest,
    materialization_declaration: MaterializationDeclaration,
    observation_formation_declaration: ObservationFormationDeclaration,
) -> None:
    if not manifest.benchmark_manifest.matches_record(benchmark_manifest.to_record()):
        raise PerformanceViewBundleValidationError(
            "benchmark_manifest reference does not match source manifest"
        )
    if not manifest.materialization_declaration.matches_record(
        materialization_declaration.to_record()
    ):
        raise PerformanceViewBundleValidationError(
            "materialization_declaration reference does not match source declaration"
        )
    if not manifest.observation_formation_declaration.matches_record(
        observation_formation_declaration.to_record()
    ):
        raise PerformanceViewBundleValidationError(
            "observation_formation_declaration reference does not match source declaration"
        )
    if materialization_declaration.benchmark_id != benchmark_manifest.id:
        raise PerformanceViewBundleValidationError(
            "materialization_declaration benchmark_id does not match benchmark manifest"
        )
    if observation_formation_declaration.benchmark_id != benchmark_manifest.id:
        raise PerformanceViewBundleValidationError(
            "observation_formation_declaration benchmark_id does not match benchmark manifest"
        )


def _materialization_plan(
    case: PerformanceMeasurementCase,
    *,
    declaration: MaterializationDeclaration,
) -> MaterializationPlan:
    return MaterializationPlan.resolve(
        id=_child_identifier(case.id, "materialization-plan"),
        declaration=declaration,
        scale_assignment=case.scale_assignment,
        complexity_assignment=case.complexity_assignment,
        seed=case.seed,
    )


def _measurement_record(
    *,
    case: PerformanceMeasurementCase,
    benchmark_manifest: BenchmarkManifest,
    materialization_plan: MaterializationPlan,
    observation_formation_declaration: ObservationFormationDeclaration,
) -> MeasurementRecord:
    scale_axis = _scale_axis(benchmark_manifest)
    scale = case.scale_assignment.require_axis(scale_axis)
    outcome_space = benchmark_manifest.resolve_outcome_space(scale=scale)
    accepted_outcome = _outcome_id(
        case.accepted_outcome_sequence,
        benchmark_manifest=benchmark_manifest,
    )
    observation = observation_formation_declaration.form_observation(
        id=_child_identifier(case.id, "formed-observation"),
        plan=materialization_plan,
        component_sequence=case.component_sequence,
    )
    accepted_event = AcceptedEvent.from_record(
        {
            "id": str(_child_identifier(case.id, "accepted-event")),
            "outcome_space_id": str(outcome_space.id),
            "outcomes": [accepted_outcome],
        },
        outcome_space=outcome_space,
    )
    probability_measure = FiniteProbabilityMeasure(
        id=_child_identifier(case.id, "probability-measure"),
        outcome_space_id=outcome_space.id,
        probabilities=tuple(
            ProbabilityMass(
                outcome_id=_probability_outcome_id(
                    mass,
                    benchmark_manifest=benchmark_manifest,
                ),
                probability=mass.probability,
            )
            for mass in case.probabilities
        ),
    )
    raw_scoring_evidence = RawScoringEvidence.from_event_and_measure(
        id=_evidence_id(case),
        observation_id=str(observation.id),
        event=accepted_event,
        measure=probability_measure,
    )
    return MeasurementRecord(
        benchmark_id=benchmark_manifest.id,
        outcome_space=outcome_space,
        accepted_event=accepted_event,
        probability_measure=probability_measure,
        raw_scoring_evidence=raw_scoring_evidence,
        evidence_artifacts=(
            observation.formation_declaration,
            observation.materialization_plan,
            ArtifactReference(
                kind="formed-observation",
                protocol_id=observation.id,
                record_digest=observation.digest,
            ),
        ),
    )


def _probability_outcome_id(
    mass: PerformanceProbabilityMass,
    *,
    benchmark_manifest: BenchmarkManifest,
) -> str:
    if mass.outcome_id is not None:
        return mass.outcome_id
    if mass.outcome_sequence is None:
        raise PerformanceViewBundleValidationError("probability mass has no outcome identity")
    return _outcome_id(mass.outcome_sequence, benchmark_manifest=benchmark_manifest)


def _outcome_id(sequence: tuple[int, ...], *, benchmark_manifest: BenchmarkManifest) -> str:
    if benchmark_manifest.outcome_sequence is None:
        if len(sequence) != 1:
            raise PerformanceViewBundleValidationError(
                "fixed outcome spaces require single-item outcome sequences"
            )
        return str(sequence[0])
    return benchmark_manifest.outcome_sequence.outcome_id(sequence)


def _scale_axis(benchmark_manifest: BenchmarkManifest) -> str:
    if benchmark_manifest.scale_parameter is None:
        raise PerformanceViewBundleValidationError(
            "performance bundles require a scale-indexed benchmark"
        )
    return benchmark_manifest.scale_parameter.symbol


def _evidence_id(case: PerformanceMeasurementCase) -> ProtocolIdentifier:
    return _child_identifier(case.id, "evidence")


def _child_identifier(parent: ProtocolIdentifier, suffix: str) -> ProtocolIdentifier:
    return ProtocolIdentifier(
        name=ProtocolName.parse(f"{parent.name}.{suffix}"),
        version=parent.version,
    )


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise PerformanceViewBundleValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PerformanceViewBundleValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise PerformanceViewBundleValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise PerformanceViewBundleValidationError(f"{field}: expected string")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _as_string(value, field=field)


def _as_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise PerformanceViewBundleValidationError(f"{field}: expected integer")
    return value


def _as_float(value: object, *, field: str) -> float:
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, float):
        return value
    raise PerformanceViewBundleValidationError(f"{field}: expected parsed number")


def _require_nonnegative_sequence(values: tuple[int, ...], *, field: str) -> None:
    if not values:
        raise PerformanceViewBundleValidationError(f"{field} must not be empty")
    if any(type(value) is not int or value < 0 for value in values):
        raise PerformanceViewBundleValidationError(
            f"{field} values must be nonnegative integers"
        )


def _require_ordered_values(values: tuple[float, ...], *, field: str) -> None:
    if not values:
        raise PerformanceViewBundleValidationError(f"{field} must not be empty")
    if len(set(values)) != len(values):
        raise PerformanceViewBundleValidationError(f"{field} must not contain duplicates")
    if values != tuple(sorted(values)):
        raise PerformanceViewBundleValidationError(f"{field} must be sorted")
    for value in values:
        if value < 0:
            raise PerformanceViewBundleValidationError(f"{field} values must be nonnegative")


def _require_probability(value: float, *, field: str) -> None:
    if value < 0 or value > 1:
        raise PerformanceViewBundleValidationError(f"{field} must be between 0 and 1")


def _first_duplicate(values: tuple[object, ...]) -> object | None:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
