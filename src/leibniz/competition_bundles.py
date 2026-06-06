"""Canonical benchmark competition result bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from leibniz.benchmarks import BenchmarkManifest
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.evaluation_bundles import BenchmarkEvaluationBundle
from leibniz.identifiers import ProtocolIdentifier
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec

__all__ = [
    "BenchmarkCompetitionBundle",
    "BenchmarkCompetitionBundleDocument",
    "BenchmarkCompetitionBundleSummary",
    "BenchmarkCompetitionBundleValidationError",
]

_competition_bundle_record = RecordSpec(
    fields={
        "format": FieldSpec(kind="literal", literal="leibniz.benchmark-competition"),
        "format_version": FieldSpec(kind="integer"),
        "id": FieldSpec(kind="identifier"),
        "benchmark_manifest": FieldSpec(kind="record"),
        "left_evaluation_bundle": FieldSpec(kind="record"),
        "right_evaluation_bundle": FieldSpec(kind="record"),
        "competition_result": FieldSpec(kind="record"),
        "competition_protocol": FieldSpec(kind="record"),
        "competition_seed": FieldSpec(kind="integer"),
        "throughput": FieldSpec(kind="record"),
    }
)


class BenchmarkCompetitionBundleValidationError(ValueError):
    """Raised when an accepted benchmark competition bundle is invalid."""


_record = RecordExtractor(BenchmarkCompetitionBundleValidationError)


@dataclass(frozen=True, slots=True)
class BenchmarkCompetitionBundleSummary:
    """Compact competition evidence fields used by schedulers and derived views."""

    benchmark_id: str
    competition_id: str
    left_model_key: str
    right_model_key: str
    left_score: float
    right_score: float
    sample_count: int
    throughput: Mapping[str, object]

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
    ) -> BenchmarkCompetitionBundleSummary:
        if record.get("format") != "leibniz.benchmark-competition":
            raise BenchmarkCompetitionBundleValidationError(
                "competition bundle has unsupported format"
            )
        if record.get("format_version") != 1:
            raise BenchmarkCompetitionBundleValidationError(
                "competition bundle has unsupported format_version"
            )
        result = _record.mapping(
            record.get("competition_result"),
            "competition_result",
        )
        _validate_competition_result_summary(result)
        benchmark_id = _record.non_empty_string(
            result.get("benchmark_id"),
            "competition_result.benchmark_id",
        )
        raw_manifest = record.get("benchmark_manifest")
        if isinstance(raw_manifest, Mapping):
            manifest = cast(Mapping[str, object], raw_manifest)
            manifest_id = manifest.get("id")
            if isinstance(manifest_id, str) and manifest_id and manifest_id != benchmark_id:
                raise BenchmarkCompetitionBundleValidationError(
                    "competition_result benchmark_id does not match benchmark_manifest"
                )
        throughput = _record.mapping(record.get("throughput"), "throughput")
        _validate_measured_max_inference_compute(
            throughput,
            "throughput.left_max_inference_compute",
            field="left_max_inference_compute",
        )
        _validate_measured_max_inference_compute(
            throughput,
            "throughput.right_max_inference_compute",
            field="right_max_inference_compute",
        )
        return cls(
            benchmark_id=benchmark_id,
            competition_id=_record.non_empty_string(
                result.get("competition_id"),
                "competition_result.competition_id",
            ),
            left_model_key=_record.non_empty_string(
                result.get("left_model_key"),
                "competition_result.left_model_key",
            ),
            right_model_key=_record.non_empty_string(
                result.get("right_model_key"),
                "competition_result.right_model_key",
            ),
            left_score=_probability(
                result.get("left_score"),
                "competition_result.left_score",
            ),
            right_score=_probability(
                result.get("right_score"),
                "competition_result.right_score",
            ),
            sample_count=_record.positive_integer(
                result.get("sample_count"),
                "competition_result.sample_count",
            ),
            throughput=throughput,
        )

    def competition_result_record(self) -> dict[str, object]:
        return {
            "format": "leibniz.model-competition",
            "format_version": 1,
            "benchmark_id": self.benchmark_id,
            "competition_id": self.competition_id,
            "left_model_key": self.left_model_key,
            "right_model_key": self.right_model_key,
            "left_score": self.left_score,
            "right_score": self.right_score,
            "sample_count": self.sample_count,
            "throughput": dict(self.throughput),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkCompetitionBundle:
    """Accepted pairwise benchmark evidence generated by the evaluator."""

    id: ProtocolIdentifier
    benchmark_manifest: BenchmarkManifest
    left_evaluation_bundle: BenchmarkEvaluationBundle
    right_evaluation_bundle: BenchmarkEvaluationBundle
    competition_result: Mapping[str, object]
    competition_protocol: Mapping[str, object]
    competition_seed: int
    throughput: Mapping[str, object]

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise BenchmarkCompetitionBundleValidationError(str(error)) from error
        if not str(self.id.name).startswith("benchmark-competitions."):
            raise BenchmarkCompetitionBundleValidationError(
                "id must be a valid benchmark competition id"
            )
        if type(self.competition_seed) is not int or self.competition_seed < 0:
            raise BenchmarkCompetitionBundleValidationError(
                "competition_seed must be a nonnegative integer"
            )
        self.validate_sources()

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> BenchmarkCompetitionBundle:
        try:
            validated = _competition_bundle_record.validate(record)
            benchmark_manifest = BenchmarkManifest.from_record(
                _record.mapping(validated["benchmark_manifest"], "benchmark_manifest")
            )
            left_evaluation_bundle = BenchmarkEvaluationBundle.from_record(
                _record.mapping(
                    validated["left_evaluation_bundle"],
                    "left_evaluation_bundle",
                )
            )
            right_evaluation_bundle = BenchmarkEvaluationBundle.from_record(
                _record.mapping(
                    validated["right_evaluation_bundle"],
                    "right_evaluation_bundle",
                )
            )
        except ValueError as error:
            raise BenchmarkCompetitionBundleValidationError(str(error)) from error
        return cls(
            id=_record.identifier(validated["id"], "id"),
            benchmark_manifest=benchmark_manifest,
            left_evaluation_bundle=left_evaluation_bundle,
            right_evaluation_bundle=right_evaluation_bundle,
            competition_result=_record.mapping(
                validated["competition_result"],
                "competition_result",
            ),
            competition_protocol=_record.mapping(
                validated["competition_protocol"],
                "competition_protocol",
            ),
            competition_seed=_record.integer(
                validated["competition_seed"],
                "competition_seed",
            ),
            throughput=_record.mapping(validated["throughput"], "throughput"),
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def validate_sources(self) -> None:
        benchmark_id = self.benchmark_manifest.id
        if self.left_evaluation_bundle.benchmark_manifest.id != benchmark_id:
            raise BenchmarkCompetitionBundleValidationError(
                "left_evaluation_bundle benchmark does not match benchmark_manifest"
            )
        if self.right_evaluation_bundle.benchmark_manifest.id != benchmark_id:
            raise BenchmarkCompetitionBundleValidationError(
                "right_evaluation_bundle benchmark does not match benchmark_manifest"
            )
        _validate_competition_result(
            self.competition_result,
            benchmark_id=benchmark_id,
            competition_seed=self.competition_seed,
        )
        left_model_key = _record.non_empty_string(
            self.competition_result.get("left_model_key"),
            "competition_result.left_model_key",
        )
        right_model_key = _record.non_empty_string(
            self.competition_result.get("right_model_key"),
            "competition_result.right_model_key",
        )
        if left_model_key != _checkpoint_model_key(self.left_evaluation_bundle.model_checkpoint):
            raise BenchmarkCompetitionBundleValidationError(
                "competition_result left_model_key does not match left_evaluation_bundle"
            )
        if right_model_key != _checkpoint_model_key(self.right_evaluation_bundle.model_checkpoint):
            raise BenchmarkCompetitionBundleValidationError(
                "competition_result right_model_key does not match right_evaluation_bundle"
            )
        sample_count = _record.integer(
            self.competition_result.get("sample_count"),
            "competition_result.sample_count",
        )
        protocol_evidence_count = _record.integer(
            self.competition_protocol.get("evidence_count"),
            "competition_protocol.evidence_count",
        )
        if sample_count != protocol_evidence_count:
            raise BenchmarkCompetitionBundleValidationError(
                "competition_protocol evidence_count does not match competition_result"
            )
        _validate_measured_max_inference_compute(
            self.throughput,
            "throughput.left_max_inference_compute",
            field="left_max_inference_compute",
        )
        _validate_measured_max_inference_compute(
            self.throughput,
            "throughput.right_max_inference_compute",
            field="right_max_inference_compute",
        )

    def to_record(self) -> dict[str, object]:
        return {
            "format": "leibniz.benchmark-competition",
            "format_version": 1,
            "id": str(self.id),
            "benchmark_manifest": self.benchmark_manifest.to_record(),
            "left_evaluation_bundle": self.left_evaluation_bundle.to_record(),
            "right_evaluation_bundle": self.right_evaluation_bundle.to_record(),
            "competition_result": dict(self.competition_result),
            "competition_protocol": dict(self.competition_protocol),
            "competition_seed": self.competition_seed,
            "throughput": dict(self.throughput),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkCompetitionBundleDocument:
    """A loaded accepted benchmark competition bundle and its digest."""

    bundle: BenchmarkCompetitionBundle
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> BenchmarkCompetitionBundleDocument:
        try:
            record = load_object_document(data, description="benchmark competition bundle")
        except ContentEncodingError as error:
            raise BenchmarkCompetitionBundleValidationError(str(error)) from error
        bundle = BenchmarkCompetitionBundle.from_record(record)
        return cls(bundle=bundle, digest=ContentDigest.from_value(bundle.to_record()))


def _validate_competition_result(
    record: Mapping[str, object],
    *,
    benchmark_id: ProtocolIdentifier,
    competition_seed: int,
) -> None:
    if record.get("format") != "leibniz.model-competition":
        raise BenchmarkCompetitionBundleValidationError(
            "competition_result has unsupported format"
        )
    if record.get("format_version") != 1:
        raise BenchmarkCompetitionBundleValidationError(
            "competition_result has unsupported format_version"
        )
    record_benchmark_id = _record.non_empty_string(
        record.get("benchmark_id"),
        "competition_result.benchmark_id",
    )
    if record_benchmark_id != str(benchmark_id):
        raise BenchmarkCompetitionBundleValidationError(
            "competition_result benchmark_id does not match benchmark_manifest"
        )
    if _record.integer(record.get("seed"), "competition_result.seed") != competition_seed:
        raise BenchmarkCompetitionBundleValidationError(
            "competition_result seed does not match competition_seed"
        )
    _record.non_empty_string(record.get("competition_id"), "competition_result.competition_id")
    _record.non_empty_string(record.get("mechanic"), "competition_result.mechanic")
    _record.non_empty_string(record.get("outcome_space_id"), "competition_result.outcome_space_id")
    _record.non_empty_string(record.get("left_model_key"), "competition_result.left_model_key")
    _record.non_empty_string(record.get("right_model_key"), "competition_result.right_model_key")
    sample_count = _record.integer(record.get("sample_count"), "competition_result.sample_count")
    if sample_count < 1:
        raise BenchmarkCompetitionBundleValidationError(
            "competition_result sample_count must be positive"
        )
    _probability(record.get("left_score"), "competition_result.left_score")
    _probability(record.get("right_score"), "competition_result.right_score")
    entries = _sequence(record.get("entries"), "competition_result.entries")
    if len(entries) != sample_count:
        raise BenchmarkCompetitionBundleValidationError(
            "competition_result entries length does not match sample_count"
        )
    for index, entry in enumerate(entries):
        entry_record = _record.mapping(entry, f"competition_result.entries.{index}")
        _record.non_empty_string(entry_record.get("id"), "competition_result.entries.id")
        _record.non_empty_string(
            entry_record.get("observation_id"),
            "competition_result.entries.observation_id",
        )
        _record.non_empty_string(
            entry_record.get("accepted_outcome_id"),
            "competition_result.entries.accepted_outcome_id",
        )
        winner = _record.non_empty_string(
            entry_record.get("winner"),
            "competition_result.entries.winner",
        )
        if winner not in {"left", "right", "tie"}:
            raise BenchmarkCompetitionBundleValidationError(
                "competition_result entry winner is invalid"
            )
        _probability(entry_record.get("left_score"), "competition_result.entries.left_score")
        _probability(entry_record.get("right_score"), "competition_result.entries.right_score")


def _validate_competition_result_summary(record: Mapping[str, object]) -> None:
    if record.get("format") != "leibniz.model-competition":
        raise BenchmarkCompetitionBundleValidationError(
            "competition_result has unsupported format"
        )
    if record.get("format_version") != 1:
        raise BenchmarkCompetitionBundleValidationError(
            "competition_result has unsupported format_version"
        )
    _record.non_empty_string(record.get("benchmark_id"), "competition_result.benchmark_id")
    _record.non_empty_string(record.get("competition_id"), "competition_result.competition_id")
    _record.non_empty_string(record.get("left_model_key"), "competition_result.left_model_key")
    _record.non_empty_string(record.get("right_model_key"), "competition_result.right_model_key")
    _record.positive_integer(record.get("sample_count"), "competition_result.sample_count")
    _probability(record.get("left_score"), "competition_result.left_score")
    _probability(record.get("right_score"), "competition_result.right_score")


def _probability(value: object, field: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise BenchmarkCompetitionBundleValidationError(f"{field} must be a probability")
    probability = float(value)
    if probability < 0.0 or probability > 1.0:
        raise BenchmarkCompetitionBundleValidationError(f"{field} must be a probability")
    return probability


def _validate_measured_max_inference_compute(
    throughput: Mapping[str, object],
    field_path: str,
    *,
    field: str,
) -> int:
    value = _record.integer(throughput.get(field), field_path)
    if value < 0:
        raise BenchmarkCompetitionBundleValidationError(f"{field_path} must be nonnegative")
    return value


def _checkpoint_model_key(record: Mapping[str, object]) -> str:
    return str(
        ContentDigest.from_string(
            record.get("digest"),
            field="model_checkpoint.digest",
            error_type=BenchmarkCompetitionBundleValidationError,
        )
    )


def _sequence(value: object, field: str) -> tuple[object, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise BenchmarkCompetitionBundleValidationError(f"{field} must be a sequence")
    return tuple(cast(Sequence[object], value))
