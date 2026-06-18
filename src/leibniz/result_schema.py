"""Program-neutral result-tier records: submissions, benchmarks, evaluations.

The published ``results/`` dataset has three source-of-truth tiers, and this
module owns the record for each:

* ``SubmissionRecord`` — a runnable program evaluated against benchmarks. A
  deterministic program and a learned model are the *same kind of thing*: the
  program graph is the submission, and fitted parameters plus a training
  provenance record are *optional in-bundle provenance* of a learned
  submission. A deterministic submission carries neither. Submissions are
  benchmark-agnostic; the same submission is evaluated against many benchmarks.

* ``BenchmarkMetadataRecord`` — descriptive universe metadata (representation,
  competence functional, baseline, structural type). Descriptive, never
  prescriptive of which programs may be submitted.

* ``EvaluationRecord`` — the result of evaluating one submission against one
  benchmark: the validated-bit Score, the capability-map data backing it, the
  measured energy cost, and lineage references back to the submission, the
  benchmark, and the measurement evidence.

These records reference one another by id and content digest; they do not embed
each other. The console derives every presentation (leaderboards, class
generality, the capability map) from the evaluation matrix — nothing here is a
stored, renderer-shaped view.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from leibniz.content import ContentDigest
from leibniz.cost_metrology import CostMeasurement
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec

__all__ = [
    "ArtifactReference",
    "BenchmarkMetadataDocument",
    "BenchmarkMetadataRecord",
    "EvaluationDocument",
    "EvaluationLineage",
    "EvaluationRecord",
    "ResultSchemaError",
    "SubmissionDocument",
    "SubmissionRecord",
]

_submission_format = "leibniz.submission"
_benchmark_format = "leibniz.benchmark"
_evaluation_format = "leibniz.evaluation"
_format_version = 1

# The capability-map data carried by an evaluation is the measure-weighted
# partition competence integral produced by ``partition_score.py``; that module
# owns its deep structure. Here we check only the load-bearing surface.
_capability_map_kind = "measure-weighted-partition-competence-integral-v1"


class ResultSchemaError(ValueError):
    """Raised when a result-tier record is invalid."""


_extract = RecordExtractor(error_type=ResultSchemaError)

_artifact_record = RecordSpec(
    fields={
        "kind": FieldSpec(kind="string"),
        "digest": FieldSpec(kind="string"),
        "path": FieldSpec(kind="string", required=False),
        "description": FieldSpec(kind="string", required=False),
    }
)
_submission_record = RecordSpec(
    fields={
        "format": FieldSpec(kind="literal", literal=_submission_format),
        "format_version": FieldSpec(kind="integer"),
        "id": FieldSpec(kind="identifier"),
        "program_graph": FieldSpec(kind="record"),
        "model_inspection": FieldSpec(kind="record", required=False),
        "fitted_parameters": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
            required=False,
        ),
        "training_provenance": FieldSpec(kind="record", required=False),
    }
)
_benchmark_record = RecordSpec(
    fields={
        "format": FieldSpec(kind="literal", literal=_benchmark_format),
        "format_version": FieldSpec(kind="integer"),
        "id": FieldSpec(kind="identifier"),
        "name": FieldSpec(kind="string"),
        "representation": FieldSpec(kind="string"),
        "competence": FieldSpec(kind="record"),
        "baseline": FieldSpec(kind="record"),
        "structural_type": FieldSpec(kind="string", required=False),
    }
)
_lineage_record = RecordSpec(
    fields={
        "submission_digest": FieldSpec(kind="string"),
        "benchmark_digest": FieldSpec(kind="string"),
        "measurement_dataset_digest": FieldSpec(kind="string", required=False),
    }
)
_evaluation_record = RecordSpec(
    fields={
        "format": FieldSpec(kind="literal", literal=_evaluation_format),
        "format_version": FieldSpec(kind="integer"),
        "id": FieldSpec(kind="identifier"),
        "submission_id": FieldSpec(kind="identifier"),
        "benchmark_id": FieldSpec(kind="identifier"),
        "validated_bits": FieldSpec(kind="record"),
        "cost": FieldSpec(kind="record"),
        "lineage": FieldSpec(kind="record"),
        "evaluation_seed": FieldSpec(kind="integer"),
        "converged": FieldSpec(kind="boolean"),
        "evidence_budget_limited": FieldSpec(kind="boolean"),
        "diagnostics": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
            required=False,
        ),
    }
)
_representation_kinds = frozenset({"finite-outcome", "field-valued", "inverse"})


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """A reference to a content-addressed artifact bundled with a submission."""

    kind: str
    digest: ContentDigest
    path: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.kind:
            raise ResultSchemaError("artifact kind must be nonempty")
        if self.path is not None and not self.path:
            raise ResultSchemaError("artifact path must be nonempty when present")
        if self.description is not None and not self.description:
            raise ResultSchemaError("artifact description must be nonempty when present")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ArtifactReference:
        try:
            validated = _artifact_record.validate(record)
        except ValueError as error:
            raise ResultSchemaError(str(error)) from error
        return cls(
            kind=_extract.non_empty_string(validated["kind"], "kind"),
            digest=ContentDigest.from_string(
                validated["digest"],
                field="digest",
                error_type=ResultSchemaError,
            ),
            path=_extract.optional_string(validated.get("path"), "path"),
            description=_extract.optional_string(validated.get("description"), "description"),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {"kind": self.kind, "digest": str(self.digest)}
        if self.path is not None:
            record["path"] = self.path
        if self.description is not None:
            record["description"] = self.description
        return record


@dataclass(frozen=True, slots=True)
class SubmissionRecord:
    """A runnable program submitted for evaluation.

    Program-neutral: ``program_graph`` is the submission. ``fitted_parameters``
    and ``training_provenance`` are optional in-bundle provenance present only
    for learned submissions; a deterministic program leaves both empty.
    """

    id: ProtocolIdentifier
    program_graph: Mapping[str, object]
    model_inspection: Mapping[str, object] | None = None
    fitted_parameters: tuple[ArtifactReference, ...] = ()
    training_provenance: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise ResultSchemaError(str(error)) from error
        if not str(self.id.name).startswith("submissions."):
            raise ResultSchemaError("id must be in the submissions namespace")
        if not self.program_graph:
            raise ResultSchemaError("program_graph must be nonempty")
        _reject_duplicate_artifact_digests(self.fitted_parameters)

    @property
    def is_learned(self) -> bool:
        """A learned submission carries fitted parameters or training provenance.

        This is a derived convenience, not a stored discriminator: the Score
        does not branch on it. Deterministic and learned submissions are scored
        identically.
        """

        return bool(self.fitted_parameters) or self.training_provenance is not None

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> SubmissionRecord:
        try:
            validated = _submission_record.validate(record)
        except ValueError as error:
            raise ResultSchemaError(str(error)) from error
        _require_format_version(validated, "submission")
        fitted_parameters = tuple(
            ArtifactReference.from_record(_extract.mapping(item, "fitted_parameters"))
            for item in _extract.sequence(
                validated.get("fitted_parameters", ()),
                "fitted_parameters",
            )
        )
        return cls(
            id=_extract.identifier(validated["id"], "id"),
            program_graph=_extract.mapping(validated["program_graph"], "program_graph"),
            model_inspection=_extract.optional_mapping(
                validated.get("model_inspection"),
                "model_inspection",
            ),
            fitted_parameters=fitted_parameters,
            training_provenance=_extract.optional_mapping(
                validated.get("training_provenance"),
                "training_provenance",
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "format": _submission_format,
            "format_version": _format_version,
            "id": str(self.id),
            "program_graph": dict(self.program_graph),
        }
        if self.model_inspection is not None:
            record["model_inspection"] = dict(self.model_inspection)
        if self.fitted_parameters:
            record["fitted_parameters"] = [
                artifact.to_record()
                for artifact in sorted(
                    self.fitted_parameters,
                    key=lambda artifact: str(artifact.digest),
                )
            ]
        if self.training_provenance is not None:
            record["training_provenance"] = dict(self.training_provenance)
        return record


@dataclass(frozen=True, slots=True)
class BenchmarkMetadataRecord:
    """Descriptive metadata for a benchmark universe."""

    id: ProtocolIdentifier
    name: str
    representation: str
    competence: Mapping[str, object]
    baseline: Mapping[str, object]
    structural_type: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ResultSchemaError("name must be nonempty")
        if self.representation not in _representation_kinds:
            raise ResultSchemaError(
                f"representation must be one of {sorted(_representation_kinds)}"
            )
        if not self.competence:
            raise ResultSchemaError("competence must be nonempty")
        if not self.baseline:
            raise ResultSchemaError("baseline must be nonempty")
        if self.structural_type is not None and not self.structural_type:
            raise ResultSchemaError("structural_type must be nonempty when present")

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BenchmarkMetadataRecord:
        try:
            validated = _benchmark_record.validate(record)
        except ValueError as error:
            raise ResultSchemaError(str(error)) from error
        _require_format_version(validated, "benchmark")
        return cls(
            id=_extract.identifier(validated["id"], "id"),
            name=_extract.non_empty_string(validated["name"], "name"),
            representation=_extract.non_empty_string(
                validated["representation"],
                "representation",
            ),
            competence=_extract.mapping(validated["competence"], "competence"),
            baseline=_extract.mapping(validated["baseline"], "baseline"),
            structural_type=_extract.optional_string(
                validated.get("structural_type"),
                "structural_type",
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "format": _benchmark_format,
            "format_version": _format_version,
            "id": str(self.id),
            "name": self.name,
            "representation": self.representation,
            "competence": dict(self.competence),
            "baseline": dict(self.baseline),
        }
        if self.structural_type is not None:
            record["structural_type"] = self.structural_type
        return record


@dataclass(frozen=True, slots=True)
class EvaluationLineage:
    """Content-addressed references resolving an evaluation to its inputs."""

    submission_digest: ContentDigest
    benchmark_digest: ContentDigest
    measurement_dataset_digest: ContentDigest | None = None

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> EvaluationLineage:
        try:
            validated = _lineage_record.validate(record)
        except ValueError as error:
            raise ResultSchemaError(str(error)) from error
        dataset_digest = validated.get("measurement_dataset_digest")
        return cls(
            submission_digest=ContentDigest.from_string(
                validated["submission_digest"],
                field="lineage.submission_digest",
                error_type=ResultSchemaError,
            ),
            benchmark_digest=ContentDigest.from_string(
                validated["benchmark_digest"],
                field="lineage.benchmark_digest",
                error_type=ResultSchemaError,
            ),
            measurement_dataset_digest=(
                None
                if dataset_digest is None
                else ContentDigest.from_string(
                    dataset_digest,
                    field="lineage.measurement_dataset_digest",
                    error_type=ResultSchemaError,
                )
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "submission_digest": str(self.submission_digest),
            "benchmark_digest": str(self.benchmark_digest),
        }
        if self.measurement_dataset_digest is not None:
            record["measurement_dataset_digest"] = str(self.measurement_dataset_digest)
        return record


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """The result of evaluating one submission against one benchmark.

    The Score lives on two measured axes: ``validated_bits`` (the extensive
    measure-weighted partition Score, with ``capability_map`` carrying the
    partition tree, per-region competence, and refinement ladder that back it)
    and ``cost`` (the measured energy of producing the predictions).
    """

    id: ProtocolIdentifier
    submission_id: ProtocolIdentifier
    benchmark_id: ProtocolIdentifier
    validated_bits: float
    capability_map: Mapping[str, object]
    cost: CostMeasurement
    lineage: EvaluationLineage
    evaluation_seed: int
    converged: bool
    evidence_budget_limited: bool
    diagnostics: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise ResultSchemaError(str(error)) from error
        if not str(self.id.name).startswith("evaluations."):
            raise ResultSchemaError("id must be in the evaluations namespace")
        if not str(self.submission_id.name).startswith("submissions."):
            raise ResultSchemaError("submission_id must be in the submissions namespace")
        _nonnegative_number(self.validated_bits, field="validated_bits")
        if type(self.evaluation_seed) is not int or self.evaluation_seed < 0:
            raise ResultSchemaError("evaluation_seed must be a nonnegative integer")
        if type(self.converged) is not bool:
            raise ResultSchemaError("converged must be a boolean")
        if type(self.evidence_budget_limited) is not bool:
            raise ResultSchemaError("evidence_budget_limited must be a boolean")
        if self.evidence_budget_limited and self.converged:
            raise ResultSchemaError("budget-limited evaluations cannot be marked converged")
        _validate_diagnostics(self.diagnostics)
        _validate_capability_map(self.capability_map, validated_bits=self.validated_bits)

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> EvaluationRecord:
        try:
            validated = _evaluation_record.validate(record)
            cost = CostMeasurement.from_record(validated["cost"])
        except ValueError as error:
            raise ResultSchemaError(str(error)) from error
        _require_format_version(validated, "evaluation")
        validated_bits_record = _extract.mapping(
            validated["validated_bits"],
            "validated_bits",
        )
        return cls(
            id=_extract.identifier(validated["id"], "id"),
            submission_id=_extract.identifier(validated["submission_id"], "submission_id"),
            benchmark_id=_extract.identifier(validated["benchmark_id"], "benchmark_id"),
            validated_bits=_nonnegative_number(
                validated_bits_record.get("value"),
                field="validated_bits.value",
            ),
            capability_map=_extract.mapping(
                validated_bits_record.get("capability_map"),
                "validated_bits.capability_map",
            ),
            cost=cost,
            lineage=EvaluationLineage.from_record(
                _extract.mapping(validated["lineage"], "lineage")
            ),
            evaluation_seed=_extract.integer(validated["evaluation_seed"], "evaluation_seed"),
            converged=_extract.boolean(validated["converged"], "converged"),
            evidence_budget_limited=_extract.boolean(
                validated["evidence_budget_limited"],
                "evidence_budget_limited",
            ),
            diagnostics=_diagnostics_from_record(validated.get("diagnostics")),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "format": _evaluation_format,
            "format_version": _format_version,
            "id": str(self.id),
            "submission_id": str(self.submission_id),
            "benchmark_id": str(self.benchmark_id),
            "validated_bits": {
                "value": self.validated_bits,
                "capability_map": dict(self.capability_map),
            },
            "cost": self.cost.to_record(),
            "lineage": self.lineage.to_record(),
            "evaluation_seed": self.evaluation_seed,
            "converged": self.converged,
            "evidence_budget_limited": self.evidence_budget_limited,
        }
        if self.diagnostics:
            record["diagnostics"] = [dict(item) for item in self.diagnostics]
        return record


@dataclass(frozen=True, slots=True)
class SubmissionDocument:
    """A loaded submission record and its canonical digest."""

    submission: SubmissionRecord
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> SubmissionDocument:
        record = _load_record(data, description="submission record")
        submission = SubmissionRecord.from_record(record)
        return cls(submission=submission, digest=submission.digest)


@dataclass(frozen=True, slots=True)
class BenchmarkMetadataDocument:
    """A loaded benchmark metadata record and its canonical digest."""

    benchmark: BenchmarkMetadataRecord
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> BenchmarkMetadataDocument:
        record = _load_record(data, description="benchmark metadata record")
        benchmark = BenchmarkMetadataRecord.from_record(record)
        return cls(benchmark=benchmark, digest=benchmark.digest)


@dataclass(frozen=True, slots=True)
class EvaluationDocument:
    """A loaded evaluation record and its canonical digest."""

    evaluation: EvaluationRecord
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> EvaluationDocument:
        record = _load_record(data, description="evaluation record")
        evaluation = EvaluationRecord.from_record(record)
        return cls(evaluation=evaluation, digest=evaluation.digest)


def _load_record(data: bytes, *, description: str) -> Mapping[str, object]:
    try:
        return load_object_document(data, description=description)
    except ContentEncodingError as error:
        raise ResultSchemaError(str(error)) from error


def _require_format_version(record: Mapping[str, object], label: str) -> None:
    if record.get("format_version") != _format_version:
        raise ResultSchemaError(f"{label} record has unsupported format_version")


def _validate_capability_map(
    capability_map: Mapping[str, object],
    *,
    validated_bits: float,
) -> None:
    if capability_map.get("kind") != _capability_map_kind:
        raise ResultSchemaError(
            f"capability_map.kind must be {_capability_map_kind}"
        )
    value = _nonnegative_number(capability_map.get("value"), field="capability_map.value")
    if not math.isclose(value, validated_bits, rel_tol=1e-9, abs_tol=1e-9):
        raise ResultSchemaError("capability_map.value must match validated_bits.value")
    if not isinstance(capability_map.get("root"), Mapping):
        raise ResultSchemaError("capability_map.root must be a record")


def _diagnostics_from_record(value: object) -> tuple[Mapping[str, object], ...]:
    if value is None:
        return ()
    return tuple(
        _extract.mapping(item, "diagnostics[]")
        for item in _extract.sequence(value, "diagnostics")
    )


def _validate_diagnostics(diagnostics: Sequence[Mapping[str, object]]) -> None:
    for index, item in enumerate(diagnostics):
        kind = item.get("kind")
        if kind is not None and (not isinstance(kind, str) or not kind):
            raise ResultSchemaError(f"diagnostics[{index}].kind must be a nonempty string")


def _nonnegative_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ResultSchemaError(f"{field}: expected nonnegative number")
    number = float(value)
    if number < 0.0 or not math.isfinite(number):
        raise ResultSchemaError(f"{field}: expected nonnegative number")
    return number


def _reject_duplicate_artifact_digests(artifacts: Sequence[ArtifactReference]) -> None:
    seen: set[str] = set()
    for artifact in artifacts:
        digest = str(artifact.digest)
        if digest in seen:
            raise ResultSchemaError(f"duplicate fitted-parameter artifact digest: {digest}")
        seen.add(digest)
