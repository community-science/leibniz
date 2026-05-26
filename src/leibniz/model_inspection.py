"""Read-only model inspection records derived from public model artifacts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz.architectures import ArchitectureLayer, ArchitectureManifest
from leibniz.artifacts import ArtifactReference, reference_for_record
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.model_manifests import ModelArtifactManifest
from leibniz.records import FieldSpec, RecordSpec
from leibniz.submissions import SubmissionPackageManifest

__all__ = [
    "ModelInspectionCostSummary",
    "ModelInspectionDocument",
    "ModelInspectionLayer",
    "ModelInspectionRecord",
    "ModelInspectionValidationError",
]

_layer_record = RecordSpec(
    fields={
        "index": FieldSpec(kind="integer"),
        "kind": FieldSpec(kind="string"),
        "parameters": FieldSpec(kind="record"),
        "input_shape": FieldSpec(kind="sequence", item=FieldSpec(kind="integer"), required=False),
        "output_shape": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="integer"),
            required=False,
        ),
        "parameter_count": FieldSpec(kind="integer", required=False),
    }
)
_cost_summary_record = RecordSpec(
    fields={
        "layer_count": FieldSpec(kind="integer"),
        "parameter_count": FieldSpec(kind="integer", required=False),
        "unknown_parameter_layers": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="integer"),
        ),
    }
)
_inspection_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "architecture": FieldSpec(kind="record"),
        "input_shape": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "output_shape": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "layers": FieldSpec(kind="sequence", item=FieldSpec(kind="record")),
        "cost_summary": FieldSpec(kind="record"),
        "model_manifest": FieldSpec(kind="record", required=False),
        "submission_package": FieldSpec(kind="record", required=False),
        "benchmark_manifest": FieldSpec(kind="record", required=False),
        "measurement_dataset": FieldSpec(kind="record", required=False),
        "model_artifacts": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
            required=False,
        ),
        "training_provenance": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
            required=False,
        ),
    }
)


class ModelInspectionValidationError(ValueError):
    """Raised when a model inspection record is invalid."""


@dataclass(frozen=True, slots=True)
class ModelInspectionLayer:
    """One layer summary for read-only model inspection."""

    index: int
    kind: str
    parameters: Mapping[str, object]
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    parameter_count: int | None = None

    def __post_init__(self) -> None:
        if type(self.index) is not int:
            raise ModelInspectionValidationError("index must be an integer")
        if self.index < 0:
            raise ModelInspectionValidationError("index must be nonnegative")
        if not self.kind:
            raise ModelInspectionValidationError("kind must be nonempty")
        _require_shape(self.input_shape, field="input_shape", allow_none=True)
        _require_shape(self.output_shape, field="output_shape", allow_none=True)
        if self.parameter_count is not None and self.parameter_count < 0:
            raise ModelInspectionValidationError("parameter_count must be nonnegative")
        try:
            ContentDigest.from_value(self.to_record())
        except ContentEncodingError as error:
            raise ModelInspectionValidationError(str(error)) from error

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelInspectionLayer:
        try:
            validated = _layer_record.validate(record)
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        return cls(
            index=_as_int(validated["index"], field="index"),
            kind=_as_string(validated["kind"], field="kind"),
            parameters=_as_mapping(validated["parameters"], field="parameters"),
            input_shape=_optional_shape(validated.get("input_shape"), field="input_shape"),
            output_shape=_optional_shape(validated.get("output_shape"), field="output_shape"),
            parameter_count=_optional_int(
                validated.get("parameter_count"),
                field="parameter_count",
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "index": self.index,
            "kind": self.kind,
            "parameters": dict(self.parameters),
        }
        if self.input_shape is not None:
            record["input_shape"] = list(self.input_shape)
        if self.output_shape is not None:
            record["output_shape"] = list(self.output_shape)
        if self.parameter_count is not None:
            record["parameter_count"] = self.parameter_count
        return record


@dataclass(frozen=True, slots=True)
class ModelInspectionCostSummary:
    """Conservative model cost summary derived from public architecture structure."""

    layer_count: int
    parameter_count: int | None
    unknown_parameter_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.layer_count) is not int or self.layer_count < 0:
            raise ModelInspectionValidationError("layer_count must be a nonnegative integer")
        if self.parameter_count is not None and self.parameter_count < 0:
            raise ModelInspectionValidationError("parameter_count must be nonnegative")
        if any(type(index) is not int or index < 0 for index in self.unknown_parameter_layers):
            raise ModelInspectionValidationError(
                "unknown_parameter_layers must contain nonnegative integers"
            )
        if self.unknown_parameter_layers != tuple(sorted(set(self.unknown_parameter_layers))):
            raise ModelInspectionValidationError("unknown_parameter_layers must be sorted unique")
        if self.parameter_count is None and not self.unknown_parameter_layers:
            raise ModelInspectionValidationError(
                "unknown parameter_count requires unknown_parameter_layers"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelInspectionCostSummary:
        try:
            validated = _cost_summary_record.validate(record)
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        return cls(
            layer_count=_as_int(validated["layer_count"], field="layer_count"),
            parameter_count=_optional_int(
                validated.get("parameter_count"),
                field="parameter_count",
            ),
            unknown_parameter_layers=tuple(
                _as_int(index, field="unknown_parameter_layers")
                for index in _as_sequence(
                    validated["unknown_parameter_layers"],
                    field="unknown_parameter_layers",
                )
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "layer_count": self.layer_count,
            "unknown_parameter_layers": list(self.unknown_parameter_layers),
        }
        if self.parameter_count is not None:
            record["parameter_count"] = self.parameter_count
        return record


@dataclass(frozen=True, slots=True)
class ModelInspectionRecord:
    """A normalized read-only inspection record for public model artifacts."""

    id: ProtocolIdentifier
    architecture: ArtifactReference
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    layers: tuple[ModelInspectionLayer, ...]
    cost_summary: ModelInspectionCostSummary
    model_manifest: ArtifactReference | None = None
    submission_package: ArtifactReference | None = None
    benchmark_manifest: ArtifactReference | None = None
    measurement_dataset: ArtifactReference | None = None
    model_artifacts: tuple[ArtifactReference, ...] = ()
    training_provenance: tuple[ArtifactReference, ...] = ()

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        if not str(self.id.name).startswith("model-inspections."):
            raise ModelInspectionValidationError("id must be a valid model inspection id")
        if self.architecture.kind != "architecture-manifest":
            raise ModelInspectionValidationError(
                "architecture reference must have kind architecture-manifest"
            )
        _require_shape(self.input_shape, field="input_shape")
        _require_shape(self.output_shape, field="output_shape")
        if not self.layers:
            raise ModelInspectionValidationError("layers must not be empty")
        expected_indexes = tuple(range(len(self.layers)))
        actual_indexes = tuple(layer.index for layer in self.layers)
        if actual_indexes != expected_indexes:
            raise ModelInspectionValidationError("layer indexes must be contiguous")
        if self.cost_summary.layer_count != len(self.layers):
            raise ModelInspectionValidationError("cost_summary layer_count does not match layers")
        _require_reference_kind(
            self.model_manifest,
            kind="model-manifest",
            field="model_manifest",
        )
        _require_reference_kind(
            self.submission_package,
            kind="submission-package",
            field="submission_package",
        )
        _require_reference_kind(
            self.benchmark_manifest,
            kind="benchmark-manifest",
            field="benchmark_manifest",
        )
        _require_reference_kind(
            self.measurement_dataset,
            kind="measurement-dataset",
            field="measurement_dataset",
        )
        object.__setattr__(
            self,
            "model_artifacts",
            tuple(sorted(self.model_artifacts, key=_reference_sort_key)),
        )
        object.__setattr__(
            self,
            "training_provenance",
            tuple(sorted(self.training_provenance, key=_reference_sort_key)),
        )

    @classmethod
    def from_architecture(
        cls,
        *,
        id: ProtocolIdentifier,
        architecture_manifest: ArchitectureManifest,
    ) -> ModelInspectionRecord:
        layers, cost_summary = _architecture_layers(architecture_manifest)
        return cls(
            id=id,
            architecture=reference_for_record(
                kind="architecture-manifest",
                record=architecture_manifest.to_record(),
            ),
            input_shape=architecture_manifest.input_shape,
            output_shape=architecture_manifest.output_shape,
            layers=layers,
            cost_summary=cost_summary,
        )

    @classmethod
    def from_model_manifest(
        cls,
        *,
        id: ProtocolIdentifier,
        model_manifest: ModelArtifactManifest,
        architecture_manifest: ArchitectureManifest,
    ) -> ModelInspectionRecord:
        try:
            model_manifest.validate_architecture(architecture_manifest)
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        record = cls.from_architecture(id=id, architecture_manifest=architecture_manifest)
        return cls(
            id=record.id,
            architecture=record.architecture,
            input_shape=record.input_shape,
            output_shape=record.output_shape,
            layers=record.layers,
            cost_summary=record.cost_summary,
            model_manifest=reference_for_record(
                kind="model-manifest",
                record=model_manifest.to_record(),
            ),
            model_artifacts=model_manifest.model_artifacts,
            training_provenance=model_manifest.training_provenance,
        )

    @classmethod
    def from_submission_package(
        cls,
        *,
        id: ProtocolIdentifier,
        submission_package: SubmissionPackageManifest,
    ) -> ModelInspectionRecord:
        record = cls.from_architecture(
            id=id,
            architecture_manifest=submission_package.architecture_manifest,
        )
        return cls(
            id=record.id,
            architecture=record.architecture,
            input_shape=record.input_shape,
            output_shape=record.output_shape,
            layers=record.layers,
            cost_summary=record.cost_summary,
            submission_package=reference_for_record(
                kind="submission-package",
                record=submission_package.to_record(),
            ),
            benchmark_manifest=reference_for_record(
                kind="benchmark-manifest",
                record=submission_package.benchmark_manifest.to_record(),
            ),
            measurement_dataset=ArtifactReference(
                kind="measurement-dataset",
                content_digest=submission_package.measurement_dataset.digest,
            ),
            model_artifacts=tuple(
                ArtifactReference(
                    kind="submission-artifact",
                    protocol_id=artifact.id,
                    content_digest=artifact.digest,
                )
                for artifact in submission_package.artifacts
            ),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelInspectionRecord:
        try:
            validated = _inspection_record.validate(record)
            layers = tuple(
                ModelInspectionLayer.from_record(_as_mapping(layer, field="layers"))
                for layer in _as_sequence(validated["layers"], field="layers")
            )
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        return cls(
            id=_as_identifier(validated["id"], field="id"),
            architecture=ArtifactReference.from_record(
                _as_mapping(validated["architecture"], field="architecture")
            ),
            input_shape=_as_shape(validated["input_shape"], field="input_shape"),
            output_shape=_as_shape(validated["output_shape"], field="output_shape"),
            layers=layers,
            cost_summary=ModelInspectionCostSummary.from_record(
                _as_mapping(validated["cost_summary"], field="cost_summary")
            ),
            model_manifest=_optional_reference(validated.get("model_manifest"), "model_manifest"),
            submission_package=_optional_reference(
                validated.get("submission_package"),
                "submission_package",
            ),
            benchmark_manifest=_optional_reference(
                validated.get("benchmark_manifest"),
                "benchmark_manifest",
            ),
            measurement_dataset=_optional_reference(
                validated.get("measurement_dataset"),
                "measurement_dataset",
            ),
            model_artifacts=tuple(
                ArtifactReference.from_record(_as_mapping(item, field="model_artifacts"))
                for item in _as_sequence(
                    validated.get("model_artifacts", ()),
                    field="model_artifacts",
                )
            ),
            training_provenance=tuple(
                ArtifactReference.from_record(_as_mapping(item, field="training_provenance"))
                for item in _as_sequence(
                    validated.get("training_provenance", ()),
                    field="training_provenance",
                )
            ),
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "architecture": self.architecture.to_record(),
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "layers": [layer.to_record() for layer in self.layers],
            "cost_summary": self.cost_summary.to_record(),
        }
        if self.model_manifest is not None:
            record["model_manifest"] = self.model_manifest.to_record()
        if self.submission_package is not None:
            record["submission_package"] = self.submission_package.to_record()
        if self.benchmark_manifest is not None:
            record["benchmark_manifest"] = self.benchmark_manifest.to_record()
        if self.measurement_dataset is not None:
            record["measurement_dataset"] = self.measurement_dataset.to_record()
        if self.model_artifacts:
            record["model_artifacts"] = [artifact.to_record() for artifact in self.model_artifacts]
        if self.training_provenance:
            record["training_provenance"] = [
                artifact.to_record() for artifact in self.training_provenance
            ]
        return record


@dataclass(frozen=True, slots=True)
class ModelInspectionDocument:
    """A loaded model inspection record and its canonical digest."""

    inspection: ModelInspectionRecord
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> ModelInspectionDocument:
        try:
            record = load_object_document(data, description="model inspection document")
        except ContentEncodingError as error:
            raise ModelInspectionValidationError(str(error)) from error
        inspection = ModelInspectionRecord.from_record(record)
        return cls(inspection=inspection, digest=inspection.digest)


def _architecture_layers(
    architecture_manifest: ArchitectureManifest,
) -> tuple[tuple[ModelInspectionLayer, ...], ModelInspectionCostSummary]:
    layers: list[ModelInspectionLayer] = []
    shape: tuple[int, ...] | None = architecture_manifest.input_shape
    parameter_counts: list[int] = []
    unknown_parameter_layers: list[int] = []
    for index, layer in enumerate(architecture_manifest.layers):
        input_shape = shape
        output_shape, parameter_count = _layer_output_and_cost(layer, input_shape)
        if parameter_count is None:
            unknown_parameter_layers.append(index)
        else:
            parameter_counts.append(parameter_count)
        layers.append(
            ModelInspectionLayer(
                index=index,
                kind=layer.kind,
                parameters=layer.parameters,
                input_shape=input_shape,
                output_shape=output_shape,
                parameter_count=parameter_count,
            )
        )
        shape = output_shape
    total_parameters = None if unknown_parameter_layers else sum(parameter_counts)
    return (
        tuple(layers),
        ModelInspectionCostSummary(
            layer_count=len(layers),
            parameter_count=total_parameters,
            unknown_parameter_layers=tuple(unknown_parameter_layers),
        ),
    )


def _layer_output_and_cost(
    layer: ArchitectureLayer,
    input_shape: tuple[int, ...] | None,
) -> tuple[tuple[int, ...] | None, int | None]:
    if layer.kind == "flatten":
        if input_shape is None:
            return None, 0
        return (math.prod(input_shape),), 0
    if layer.kind == "dense":
        if input_shape is None or len(input_shape) != 1:
            return None, None
        out = _optional_positive_int_parameter(layer.parameters, "out")
        if out is None:
            return None, None
        return (out,), (input_shape[0] + 1) * out
    if layer.kind == "adaptive-pooling":
        if input_shape is None or len(input_shape) < 2:
            return None, None
        size = _optional_positive_int_parameter(layer.parameters, "size")
        dimension = _optional_positive_int_parameter(layer.parameters, "dimension")
        if size is None or dimension is None or dimension >= len(input_shape) + 1:
            return None, None
        preserved = input_shape[: len(input_shape) - dimension]
        return (*preserved, *(size for _index in range(dimension))), 0
    return None, None


def _optional_positive_int_parameter(parameters: Mapping[str, object], key: str) -> int | None:
    value = parameters.get(key)
    if type(value) is not int or value < 1:
        return None
    return value


def _require_reference_kind(
    reference: ArtifactReference | None,
    *,
    kind: str,
    field: str,
) -> None:
    if reference is not None and reference.kind != kind:
        raise ModelInspectionValidationError(f"{field} reference must have kind {kind}")


def _optional_reference(value: object, field: str) -> ArtifactReference | None:
    if value is None:
        return None
    return ArtifactReference.from_record(_as_mapping(value, field=field))


def _reference_sort_key(reference: ArtifactReference) -> tuple[str, str, str, str, str]:
    return (
        reference.kind,
        str(reference.protocol_id) if reference.protocol_id is not None else "",
        str(reference.content_digest) if reference.content_digest is not None else "",
        str(reference.record_digest) if reference.record_digest is not None else "",
        reference.external_uri if reference.external_uri is not None else "",
    )


def _as_identifier(value: object, *, field: str) -> ProtocolIdentifier:
    if not isinstance(value, ProtocolIdentifier):
        raise ModelInspectionValidationError(f"{field}: expected parsed identifier")
    return value


def _as_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ModelInspectionValidationError(f"{field}: expected record")
    return cast(Mapping[str, object], value)


def _as_sequence(value: object, *, field: str) -> tuple[object, ...]:
    if not isinstance(value, tuple):
        raise ModelInspectionValidationError(f"{field}: expected parsed sequence")
    return cast(tuple[object, ...], value)


def _as_string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ModelInspectionValidationError(f"{field}: expected string")
    return value


def _as_int(value: object, *, field: str) -> int:
    if type(value) is not int:
        raise ModelInspectionValidationError(f"{field}: expected integer")
    return value


def _optional_int(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    return _as_int(value, field=field)


def _as_shape(value: object, *, field: str) -> tuple[int, ...]:
    shape = tuple(_as_int(axis, field=field) for axis in _as_sequence(value, field=field))
    _require_shape(shape, field=field)
    return shape


def _optional_shape(value: object, *, field: str) -> tuple[int, ...] | None:
    if value is None:
        return None
    return _as_shape(value, field=field)


def _require_shape(
    value: tuple[int, ...] | None,
    *,
    field: str,
    allow_none: bool = False,
) -> None:
    if value is None:
        if allow_none:
            return
        raise ModelInspectionValidationError(f"{field} must not be None")
    if not value:
        raise ModelInspectionValidationError(f"{field} must contain at least one axis")
    if any(type(axis) is not int or axis < 1 for axis in value):
        raise ModelInspectionValidationError(f"{field} axes must be positive integers")
