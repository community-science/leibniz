"""Read-only model inspection records derived from public model artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from leibniz.artifacts import (
    ArtifactReference,
    first_duplicate_reference,
    reference_for_record,
    reference_sort_key,
)
from leibniz.content import ContentDigest
from leibniz.cost_metrology import CostMeasurement
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.model_manifests import ModelArtifactManifest
from leibniz.program_graphs import ProgramGraphSpec
from leibniz.records import FieldSpec, RecordExtractor, RecordSpec
from leibniz.tensor_shapes import TensorShape, TensorShapeValidationError

__all__ = [
    "ModelGraphNodeEvidence",
    "ModelInspectionComponent",
    "ModelInspectionCostSummary",
    "ModelInspectionDocument",
    "ModelInspectionGraphSummary",
    "ModelInspectionRecord",
    "ModelInspectionTrace",
    "ModelInspectionTraceStage",
    "ModelInspectionValidationError",
]

_component_record = RecordSpec(
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
        "operator": FieldSpec(kind="record", required=False),
        "parameter_count": FieldSpec(kind="integer", required=False),
        "storage_bytes": FieldSpec(kind="integer", required=False),
    }
)
_cost_summary_record = RecordSpec(
    fields={
        "component_count": FieldSpec(kind="integer"),
        "parameter_count": FieldSpec(kind="integer", required=False),
        "storage_bytes": FieldSpec(kind="integer", required=False),
        "inference_cost_measurement": FieldSpec(kind="record", required=False),
        "inference_cost_sample_count": FieldSpec(kind="integer", required=False),
        "training_cost_measurement": FieldSpec(kind="record", required=False),
        "training_cost_sample_count": FieldSpec(kind="integer", required=False),
        "unknown_parameter_components": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="integer"),
        ),
        "unknown_cost_components": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="integer"),
            required=False,
        ),
    }
)
_trace_stage_record = RecordSpec(
    fields={
        "index": FieldSpec(kind="integer"),
        "kind": FieldSpec(kind="string"),
        "syntax_alias": FieldSpec(kind="string"),
        "operator_kind": FieldSpec(kind="string"),
        "input_shape": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "output_shape": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "descriptor_axes": FieldSpec(kind="record"),
        "shape_law": FieldSpec(kind="string"),
        "cost_law": FieldSpec(kind="string"),
        "parameter_count": FieldSpec(kind="integer", required=False),
    }
)
_trace_record = RecordSpec(
    fields={
        "input_shape": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "output_shape": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "stages": FieldSpec(kind="sequence", item=FieldSpec(kind="record")),
        "program_effects": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="record"),
            required=False,
        ),
    }
)
_graph_summary_record = RecordSpec(
    fields={
        "component_count": FieldSpec(kind="integer"),
        "edge_count": FieldSpec(kind="integer"),
        "input_count": FieldSpec(kind="integer"),
        "output_count": FieldSpec(kind="integer"),
        "input_node_ids": FieldSpec(kind="sequence", item=FieldSpec(kind="string")),
        "output_node_ids": FieldSpec(kind="sequence", item=FieldSpec(kind="string")),
        "component_kinds": FieldSpec(kind="sequence", item=FieldSpec(kind="string")),
        "unsupported_parameter_components": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="integer"),
        ),
        "unsupported_cost_components": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="integer"),
        ),
    }
)
_node_evidence_record = RecordSpec(
    fields={
        "node_path": FieldSpec(kind="sequence", item=FieldSpec(kind="string")),
        "claim_kinds": FieldSpec(kind="sequence", item=FieldSpec(kind="string")),
        "evidence_artifacts": FieldSpec(kind="sequence", item=FieldSpec(kind="record")),
    }
)
_inspection_record = RecordSpec(
    fields={
        "id": FieldSpec(kind="identifier"),
        "program": FieldSpec(kind="record"),
        "input_shape": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "output_shape": FieldSpec(kind="sequence", item=FieldSpec(kind="integer")),
        "components": FieldSpec(kind="sequence", item=FieldSpec(kind="record")),
        "cost_summary": FieldSpec(kind="record"),
        "program_graph": FieldSpec(kind="record"),
        "node_evidence": FieldSpec(kind="sequence", item=FieldSpec(kind="record")),
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


_extract = RecordExtractor(error_type=ModelInspectionValidationError)


@dataclass(frozen=True, slots=True)
class ModelGraphNodeEvidence:
    """Evidence references supporting claims about one architecture graph node path."""

    node_path: tuple[str, ...]
    claim_kinds: tuple[str, ...]
    evidence_artifacts: tuple[ArtifactReference, ...]

    def __post_init__(self) -> None:
        if not self.node_path:
            raise ModelInspectionValidationError("node_path must not be empty")
        if any(not node_id for node_id in self.node_path):
            raise ModelInspectionValidationError("node_path values must be nonempty")
        if not self.claim_kinds:
            raise ModelInspectionValidationError("claim_kinds must not be empty")
        if any(not claim for claim in self.claim_kinds):
            raise ModelInspectionValidationError("claim_kinds must be nonempty")
        if self.claim_kinds != tuple(sorted(set(self.claim_kinds))):
            raise ModelInspectionValidationError("claim_kinds must be sorted unique")
        if not self.evidence_artifacts:
            raise ModelInspectionValidationError("evidence_artifacts must not be empty")
        duplicate = first_duplicate_reference(self.evidence_artifacts)
        if duplicate is not None:
            raise ModelInspectionValidationError(
                f"duplicate node evidence artifact reference: {duplicate}"
            )
        object.__setattr__(
            self,
            "evidence_artifacts",
            tuple(sorted(self.evidence_artifacts, key=reference_sort_key)),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelGraphNodeEvidence:
        try:
            validated = _node_evidence_record.validate(record)
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        return cls(
            node_path=tuple(
                _extract.string(node_id, "node_path")
                for node_id in _extract.sequence(validated["node_path"], "node_path")
            ),
            claim_kinds=tuple(
                _extract.string(claim, "claim_kinds")
                for claim in _extract.sequence(validated["claim_kinds"], "claim_kinds")
            ),
            evidence_artifacts=tuple(
                ArtifactReference.from_record(_extract.mapping(item, "evidence_artifacts"))
                for item in _extract.sequence(
                    validated["evidence_artifacts"],
                    "evidence_artifacts",
                )
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "node_path": list(self.node_path),
            "claim_kinds": list(self.claim_kinds),
            "evidence_artifacts": [
                artifact.to_record() for artifact in self.evidence_artifacts
            ],
        }


@dataclass(frozen=True, slots=True)
class ModelInspectionComponent:
    """One component summary for read-only model inspection."""

    index: int
    kind: str
    parameters: Mapping[str, object]
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    operator: Mapping[str, object] | None = None
    parameter_count: int | None = None
    storage_bytes: int | None = None

    def __post_init__(self) -> None:
        if type(self.index) is not int:
            raise ModelInspectionValidationError("index must be an integer")
        if self.index < 0:
            raise ModelInspectionValidationError("index must be nonnegative")
        if not self.kind:
            raise ModelInspectionValidationError("kind must be nonempty")
        _require_shape(self.input_shape, field="input_shape", allow_none=True)
        _require_shape(self.output_shape, field="output_shape", allow_none=True)
        if self.operator is not None:
            try:
                ContentDigest.from_value(self.operator)
            except ContentEncodingError as error:
                raise ModelInspectionValidationError(str(error)) from error
        if self.parameter_count is not None and self.parameter_count < 0:
            raise ModelInspectionValidationError("parameter_count must be nonnegative")
        if self.storage_bytes is not None and self.storage_bytes < 0:
            raise ModelInspectionValidationError("storage_bytes must be nonnegative")
        try:
            ContentDigest.from_value(self.to_record())
        except ContentEncodingError as error:
            raise ModelInspectionValidationError(str(error)) from error

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelInspectionComponent:
        try:
            validated = _component_record.validate(record)
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        return cls(
            index=_extract.integer(validated["index"], "index"),
            kind=_extract.string(validated["kind"], "kind"),
            parameters=_extract.mapping(validated["parameters"], "parameters"),
            input_shape=_optional_shape(validated.get("input_shape"), field="input_shape"),
            output_shape=_optional_shape(validated.get("output_shape"), field="output_shape"),
            operator=_extract.optional_mapping(validated.get("operator"), "operator"),
            parameter_count=_extract.optional_integer(
                validated.get("parameter_count"),
                "parameter_count",
            ),
            storage_bytes=_extract.optional_integer(
                validated.get("storage_bytes"),
                "storage_bytes",
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
        if self.operator is not None:
            record["operator"] = dict(self.operator)
        if self.parameter_count is not None:
            record["parameter_count"] = self.parameter_count
        if self.storage_bytes is not None:
            record["storage_bytes"] = self.storage_bytes
        return record

@dataclass(frozen=True, slots=True)
class ModelInspectionCostSummary:
    """Conservative model cost summary derived from public architecture structure."""

    component_count: int
    parameter_count: int | None
    storage_bytes: int | None = None
    inference_cost_measurement: CostMeasurement | None = None
    inference_cost_sample_count: int | None = None
    training_cost_measurement: CostMeasurement | None = None
    training_cost_sample_count: int | None = None
    unknown_parameter_components: tuple[int, ...] = ()
    unknown_cost_components: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.component_count) is not int or self.component_count < 0:
            raise ModelInspectionValidationError("component_count must be a nonnegative integer")
        if self.parameter_count is not None and self.parameter_count < 0:
            raise ModelInspectionValidationError("parameter_count must be nonnegative")
        if self.storage_bytes is not None and self.storage_bytes < 0:
            raise ModelInspectionValidationError("storage_bytes must be nonnegative")
        if (
            self.inference_cost_sample_count is not None
            and self.inference_cost_sample_count < 1
        ):
            raise ModelInspectionValidationError(
                "inference_cost_sample_count must be positive"
            )
        if (
            self.training_cost_sample_count is not None
            and self.training_cost_sample_count < 1
        ):
            raise ModelInspectionValidationError(
                "training_cost_sample_count must be positive"
            )
        if any(
            type(index) is not int or index < 0 for index in self.unknown_parameter_components
        ):
            raise ModelInspectionValidationError(
                "unknown_parameter_components must contain nonnegative integers"
            )
        if self.unknown_parameter_components != tuple(
            sorted(set(self.unknown_parameter_components))
        ):
            raise ModelInspectionValidationError(
                "unknown_parameter_components must be sorted unique"
            )
        if any(type(index) is not int or index < 0 for index in self.unknown_cost_components):
            raise ModelInspectionValidationError(
                "unknown_cost_components must contain nonnegative integers"
            )
        if self.unknown_cost_components != tuple(sorted(set(self.unknown_cost_components))):
            raise ModelInspectionValidationError("unknown_cost_components must be sorted unique")
        if self.parameter_count is None and not self.unknown_parameter_components:
            raise ModelInspectionValidationError(
                "unknown parameter_count requires unknown_parameter_components"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelInspectionCostSummary:
        try:
            validated = _cost_summary_record.validate(record)
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        return cls(
            component_count=_extract.integer(validated["component_count"], "component_count"),
            parameter_count=_extract.optional_integer(
                validated.get("parameter_count"),
                "parameter_count",
            ),
            storage_bytes=_extract.optional_integer(
                validated.get("storage_bytes"),
                "storage_bytes",
            ),
            inference_cost_measurement=(
                None
                if validated.get("inference_cost_measurement") is None
                else CostMeasurement.from_record(
                    _extract.mapping(
                        validated.get("inference_cost_measurement"),
                        "inference_cost_measurement",
                    )
                )
            ),
            inference_cost_sample_count=_extract.optional_integer(
                validated.get("inference_cost_sample_count"),
                "inference_cost_sample_count",
            ),
            training_cost_measurement=(
                None
                if validated.get("training_cost_measurement") is None
                else CostMeasurement.from_record(
                    _extract.mapping(
                        validated.get("training_cost_measurement"),
                        "training_cost_measurement",
                    )
                )
            ),
            training_cost_sample_count=_extract.optional_integer(
                validated.get("training_cost_sample_count"),
                "training_cost_sample_count",
            ),
            unknown_parameter_components=tuple(
                _extract.integer(index, "unknown_parameter_components")
                for index in _extract.sequence(
                    validated["unknown_parameter_components"],
                    "unknown_parameter_components",
                )
            ),
            unknown_cost_components=tuple(
                _extract.integer(index, "unknown_cost_components")
                for index in _extract.sequence(
                    validated.get("unknown_cost_components", ()),
                    "unknown_cost_components",
                )
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "component_count": self.component_count,
            "unknown_parameter_components": list(self.unknown_parameter_components),
        }
        if self.parameter_count is not None:
            record["parameter_count"] = self.parameter_count
        if self.storage_bytes is not None:
            record["storage_bytes"] = self.storage_bytes
        if self.inference_cost_measurement is not None:
            record["inference_cost_measurement"] = self.inference_cost_measurement.to_record()
        if self.inference_cost_sample_count is not None:
            record["inference_cost_sample_count"] = self.inference_cost_sample_count
        if self.training_cost_measurement is not None:
            record["training_cost_measurement"] = self.training_cost_measurement.to_record()
        if self.training_cost_sample_count is not None:
            record["training_cost_sample_count"] = self.training_cost_sample_count
        if self.unknown_cost_components:
            record["unknown_cost_components"] = list(self.unknown_cost_components)
        return record


@dataclass(frozen=True, slots=True)
class ModelInspectionTraceStage:
    """One data-driven architecture trace stage for console presentation."""

    index: int
    kind: str
    syntax_alias: str
    operator_kind: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    descriptor_axes: Mapping[str, str]
    shape_law: str
    cost_law: str
    parameter_count: int | None = None

    def __post_init__(self) -> None:
        if type(self.index) is not int or self.index < 0:
            raise ModelInspectionValidationError("trace stage index must be nonnegative")
        if self.kind != "operator":
            raise ModelInspectionValidationError("trace stage kind must be operator")
        if not self.syntax_alias:
            raise ModelInspectionValidationError("trace stage syntax_alias must be nonempty")
        if not self.operator_kind:
            raise ModelInspectionValidationError("trace stage operator_kind must be nonempty")
        _require_shape(self.input_shape, field="trace stage input_shape")
        _require_shape(self.output_shape, field="trace stage output_shape")
        if not self.descriptor_axes:
            raise ModelInspectionValidationError("trace stage descriptor_axes must not be empty")
        for axis, value in self.descriptor_axes.items():
            if not axis or not value:
                raise ModelInspectionValidationError(
                    "trace stage descriptor axes must be nonempty"
                )
        if not self.shape_law:
            raise ModelInspectionValidationError("trace stage shape_law must be nonempty")
        if not self.cost_law:
            raise ModelInspectionValidationError("trace stage cost_law must be nonempty")
        if self.parameter_count is not None and self.parameter_count < 0:
            raise ModelInspectionValidationError("trace stage parameter_count must be nonnegative")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelInspectionTraceStage:
        try:
            validated = _trace_stage_record.validate(record)
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        return cls(
            index=_extract.integer(validated["index"], "index"),
            kind=_extract.string(validated["kind"], "kind"),
            syntax_alias=_extract.string(validated["syntax_alias"], "syntax_alias"),
            operator_kind=_extract.string(validated["operator_kind"], "operator_kind"),
            input_shape=_as_shape(validated["input_shape"], field="input_shape"),
            output_shape=_as_shape(validated["output_shape"], field="output_shape"),
            descriptor_axes=_string_mapping(
                validated["descriptor_axes"],
                field="descriptor_axes",
            ),
            shape_law=_extract.string(validated["shape_law"], "shape_law"),
            cost_law=_extract.string(validated["cost_law"], "cost_law"),
            parameter_count=_extract.optional_integer(
                validated.get("parameter_count"),
                "parameter_count",
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "index": self.index,
            "kind": self.kind,
            "syntax_alias": self.syntax_alias,
            "operator_kind": self.operator_kind,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "descriptor_axes": dict(sorted(self.descriptor_axes.items())),
            "shape_law": self.shape_law,
            "cost_law": self.cost_law,
        }
        if self.parameter_count is not None:
            record["parameter_count"] = self.parameter_count
        return record


@dataclass(frozen=True, slots=True)
class ModelInspectionTrace:
    """A data-driven architecture trace for model inspection."""

    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    stages: tuple[ModelInspectionTraceStage, ...]
    program_effects: tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        _require_shape(self.input_shape, field="trace input_shape")
        _require_shape(self.output_shape, field="trace output_shape")
        if not self.stages:
            raise ModelInspectionValidationError("trace stages must not be empty")
        expected_indexes = tuple(range(len(self.stages)))
        actual_indexes = tuple(stage.index for stage in self.stages)
        if actual_indexes != expected_indexes:
            raise ModelInspectionValidationError("trace stage indexes must be contiguous")
        if self.stages[0].input_shape != self.input_shape:
            raise ModelInspectionValidationError("trace first input_shape does not match")
        if self.stages[-1].output_shape != self.output_shape:
            raise ModelInspectionValidationError("trace final output_shape does not match")
        for previous, current in zip(self.stages, self.stages[1:], strict=False):
            if previous.output_shape != current.input_shape:
                raise ModelInspectionValidationError("trace stage shapes must compose")
        for effect in self.program_effects:
            try:
                ContentDigest.from_value(effect)
            except ContentEncodingError as error:
                raise ModelInspectionValidationError(str(error)) from error

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelInspectionTrace:
        try:
            validated = _trace_record.validate(record)
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        return cls(
            input_shape=_as_shape(validated["input_shape"], field="input_shape"),
            output_shape=_as_shape(validated["output_shape"], field="output_shape"),
            stages=tuple(
                ModelInspectionTraceStage.from_record(_extract.mapping(stage, "stages"))
                for stage in _extract.sequence(validated["stages"], "stages")
            ),
            program_effects=tuple(
                _extract.mapping(effect, "program_effects")
                for effect in _extract.sequence(
                    validated.get("program_effects", ()),
                    "program_effects",
                )
            ),
        )

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "stages": [stage.to_record() for stage in self.stages],
        }
        if self.program_effects:
            record["program_effects"] = [dict(effect) for effect in self.program_effects]
        return record


@dataclass(frozen=True, slots=True)
class ModelInspectionGraphSummary:
    """Graph-derived model architecture summary independent of execution traces."""

    component_count: int
    edge_count: int
    input_count: int
    output_count: int
    input_node_ids: tuple[str, ...]
    output_node_ids: tuple[str, ...]
    component_kinds: tuple[str, ...]
    unsupported_parameter_components: tuple[int, ...] = ()
    unsupported_cost_components: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.component_count) is not int or self.component_count <= 0:
            raise ModelInspectionValidationError("component_count must be a positive integer")
        if type(self.edge_count) is not int or self.edge_count < 0:
            raise ModelInspectionValidationError("edge_count must be a nonnegative integer")
        if type(self.input_count) is not int or self.input_count <= 0:
            raise ModelInspectionValidationError("input_count must be a positive integer")
        if type(self.output_count) is not int or self.output_count <= 0:
            raise ModelInspectionValidationError("output_count must be a positive integer")
        if len(self.input_node_ids) != self.input_count:
            raise ModelInspectionValidationError("input_count does not match input_node_ids")
        if len(self.output_node_ids) != self.output_count:
            raise ModelInspectionValidationError("output_count does not match output_node_ids")
        if len(self.component_kinds) != self.component_count:
            raise ModelInspectionValidationError("component_count does not match component_kinds")
        if any(not node_id for node_id in self.input_node_ids + self.output_node_ids):
            raise ModelInspectionValidationError("graph summary node ids must be nonempty")
        if any(not kind for kind in self.component_kinds):
            raise ModelInspectionValidationError("component_kinds must be nonempty")
        _require_index_set(
            self.unsupported_parameter_components,
            component_count=self.component_count,
            field="unsupported_parameter_components",
        )
        _require_index_set(
            self.unsupported_cost_components,
            component_count=self.component_count,
            field="unsupported_cost_components",
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelInspectionGraphSummary:
        try:
            validated = _graph_summary_record.validate(record)
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        return cls(
            component_count=_extract.integer(validated["component_count"], "component_count"),
            edge_count=_extract.integer(validated["edge_count"], "edge_count"),
            input_count=_extract.integer(validated["input_count"], "input_count"),
            output_count=_extract.integer(validated["output_count"], "output_count"),
            input_node_ids=tuple(
                _extract.string(node_id, "input_node_ids")
                for node_id in _extract.sequence(
                    validated["input_node_ids"],
                    "input_node_ids",
                )
            ),
            output_node_ids=tuple(
                _extract.string(node_id, "output_node_ids")
                for node_id in _extract.sequence(
                    validated["output_node_ids"],
                    "output_node_ids",
                )
            ),
            component_kinds=tuple(
                _extract.string(kind, "component_kinds")
                for kind in _extract.sequence(
                    validated["component_kinds"],
                    "component_kinds",
                )
            ),
            unsupported_parameter_components=tuple(
                _extract.integer(index, "unsupported_parameter_components")
                for index in _extract.sequence(
                    validated["unsupported_parameter_components"],
                    "unsupported_parameter_components",
                )
            ),
            unsupported_cost_components=tuple(
                _extract.integer(index, "unsupported_cost_components")
                for index in _extract.sequence(
                    validated["unsupported_cost_components"],
                    "unsupported_cost_components",
                )
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "component_count": self.component_count,
            "edge_count": self.edge_count,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "input_node_ids": list(self.input_node_ids),
            "output_node_ids": list(self.output_node_ids),
            "component_kinds": list(self.component_kinds),
            "unsupported_parameter_components": list(self.unsupported_parameter_components),
            "unsupported_cost_components": list(self.unsupported_cost_components),
        }


@dataclass(frozen=True, slots=True)
class ModelInspectionRecord:
    """A normalized read-only inspection record for public model artifacts."""

    id: ProtocolIdentifier
    program: ArtifactReference
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    components: tuple[ModelInspectionComponent, ...]
    cost_summary: ModelInspectionCostSummary
    program_graph: Mapping[str, object]
    node_evidence: tuple[ModelGraphNodeEvidence, ...]
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
        if self.program.kind != "program-graph":
            raise ModelInspectionValidationError("program reference must have kind program-graph")
        try:
            ContentDigest.from_value(self.program_graph)
        except ContentEncodingError as error:
            raise ModelInspectionValidationError(str(error)) from error
        _require_shape(self.input_shape, field="input_shape")
        _require_shape(self.output_shape, field="output_shape")
        if not self.components:
            raise ModelInspectionValidationError("components must not be empty")
        expected_indexes = tuple(range(len(self.components)))
        actual_indexes = tuple(component.index for component in self.components)
        if actual_indexes != expected_indexes:
            raise ModelInspectionValidationError("component indexes must be contiguous")
        if self.cost_summary.component_count != len(self.components):
            raise ModelInspectionValidationError(
                "cost_summary component_count does not match components"
            )
        if not self.node_evidence:
            raise ModelInspectionValidationError(
                "node_evidence must not be empty"
            )
        actual_node_paths = tuple(evidence.node_path for evidence in self.node_evidence)
        if actual_node_paths != tuple(sorted(set(actual_node_paths))):
            raise ModelInspectionValidationError("node_evidence node_paths must be sorted unique")
        graph_node_ids = frozenset(component.kind for component in self.components)
        for node_path in actual_node_paths:
            if node_path[0] not in graph_node_ids:
                raise ModelInspectionValidationError(
                    "node_evidence node_path must start with a program component"
                )
        expected_node_paths = frozenset((component.kind,) for component in self.components)
        if not expected_node_paths.issubset(actual_node_paths):
            raise ModelInspectionValidationError(
                "node_evidence must describe each program component"
            )
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
            tuple(sorted(self.model_artifacts, key=reference_sort_key)),
        )
        object.__setattr__(
            self,
            "training_provenance",
            tuple(sorted(self.training_provenance, key=reference_sort_key)),
        )

    @classmethod
    def from_program_graph(
        cls,
        *,
        id: ProtocolIdentifier,
        program_graph: Mapping[str, object],
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ) -> ModelInspectionRecord:
        spec = ProgramGraphSpec.from_record(program_graph)
        components = tuple(
            ModelInspectionComponent(
                index=index,
                kind=node.id,
                parameters={
                    "program_node_kind": node.kind,
                    **dict(node.parameters or {}),
                },
            )
            for index, node in enumerate(spec.nodes)
        )
        program_reference = reference_for_record(kind="program-graph", record=program_graph)
        return cls(
            id=id,
            program=program_reference,
            input_shape=input_shape,
            output_shape=output_shape,
            components=components,
            cost_summary=ModelInspectionCostSummary(
                component_count=len(components),
                parameter_count=None,
                unknown_parameter_components=tuple(range(len(components))),
                unknown_cost_components=tuple(range(len(components))),
            ),
            program_graph=dict(program_graph),
            node_evidence=_program_node_evidence_records(
                components=components,
                evidence_artifacts=(program_reference,),
            ),
        )

    @classmethod
    def from_model_manifest(
        cls,
        *,
        id: ProtocolIdentifier,
        model_manifest: ModelArtifactManifest,
        program_graph: Mapping[str, object],
        input_shape: tuple[int, ...],
        output_shape: tuple[int, ...],
    ) -> ModelInspectionRecord:
        try:
            model_manifest.validate_program(program_graph)
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        record = cls.from_program_graph(
            id=id,
            program_graph=program_graph,
            input_shape=input_shape,
            output_shape=output_shape,
        )
        model_manifest_reference = reference_for_record(
            kind="model-manifest",
            record=model_manifest.to_record(),
        )
        return cls(
            id=record.id,
            program=record.program,
            input_shape=record.input_shape,
            output_shape=record.output_shape,
            components=record.components,
            cost_summary=record.cost_summary,
            program_graph=record.program_graph,
            node_evidence=_program_node_evidence_records(
                components=record.components,
                evidence_artifacts=(
                    record.program,
                    model_manifest_reference,
                    *model_manifest.model_artifacts,
                    *model_manifest.training_provenance,
                ),
            ),
            model_manifest=model_manifest_reference,
            model_artifacts=model_manifest.model_artifacts,
            training_provenance=model_manifest.training_provenance,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelInspectionRecord:
        try:
            validated = _inspection_record.validate(record)
            components = tuple(
                ModelInspectionComponent.from_record(
                    _extract.mapping(component, "components")
                )
                for component in _extract.sequence(validated["components"], "components")
            )
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        return cls(
            id=_extract.identifier(validated["id"], "id"),
            program=ArtifactReference.from_record(
                _extract.mapping(validated["program"], "program")
            ),
            input_shape=_as_shape(validated["input_shape"], field="input_shape"),
            output_shape=_as_shape(validated["output_shape"], field="output_shape"),
            components=components,
            cost_summary=ModelInspectionCostSummary.from_record(
                _extract.mapping(validated["cost_summary"], "cost_summary")
            ),
            program_graph=_extract.mapping(
                validated["program_graph"],
                "program_graph",
            ),
            node_evidence=tuple(
                ModelGraphNodeEvidence.from_record(
                    _extract.mapping(item, "node_evidence")
                )
                for item in _extract.sequence(
                    validated["node_evidence"],
                    "node_evidence",
                )
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
                ArtifactReference.from_record(_extract.mapping(item, "model_artifacts"))
                for item in _extract.sequence(
                    validated.get("model_artifacts", ()),
                    "model_artifacts",
                )
            ),
            training_provenance=tuple(
                ArtifactReference.from_record(_extract.mapping(item, "training_provenance"))
                for item in _extract.sequence(
                    validated.get("training_provenance", ()),
                    "training_provenance",
                )
            ),
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "id": str(self.id),
            "program": self.program.to_record(),
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "components": [component.to_record() for component in self.components],
            "cost_summary": self.cost_summary.to_record(),
            "program_graph": dict(self.program_graph),
            "node_evidence": [
                evidence.to_record() for evidence in self.node_evidence
            ],
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


def _program_node_evidence_records(
    *,
    components: tuple[ModelInspectionComponent, ...],
    evidence_artifacts: tuple[ArtifactReference, ...],
) -> tuple[ModelGraphNodeEvidence, ...]:
    return tuple(
        ModelGraphNodeEvidence(
            node_path=(component.kind,),
            claim_kinds=(
                "program-structure",
                "resource-accounting",
            ),
            evidence_artifacts=evidence_artifacts,
        )
        for component in components
    )


def _require_reference_kind(
    reference: ArtifactReference | None,
    *,
    kind: str,
    field: str,
) -> None:
    if reference is not None and reference.kind != kind:
        raise ModelInspectionValidationError(f"{field} reference must have kind {kind}")


def _require_index_set(
    indexes: tuple[int, ...],
    *,
    component_count: int,
    field: str,
) -> None:
    if any(type(index) is not int or index < 0 for index in indexes):
        raise ModelInspectionValidationError(f"{field} must contain nonnegative integers")
    if indexes != tuple(sorted(set(indexes))):
        raise ModelInspectionValidationError(f"{field} must be sorted unique")
    if any(index >= component_count for index in indexes):
        raise ModelInspectionValidationError(f"{field} indexes must reference components")


def _optional_reference(value: object, field: str) -> ArtifactReference | None:
    if value is None:
        return None
    return ArtifactReference.from_record(_extract.mapping(value, field))


def _string_mapping(value: object, *, field: str) -> Mapping[str, str]:
    mapping = _extract.mapping(value, field)
    result: dict[str, str] = {}
    for key, item in mapping.items():
        if not isinstance(item, str):
            raise ModelInspectionValidationError(f"{field}: expected string mapping")
        result[key] = item
    return result


def _as_shape(value: object, *, field: str) -> tuple[int, ...]:
    try:
        return TensorShape.from_record(
            tuple(_extract.integer(axis, field) for axis in _extract.sequence(value, field)),
            field=field,
        ).axes
    except TensorShapeValidationError as error:
        raise ModelInspectionValidationError(str(error)) from error


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
    try:
        TensorShape.from_axes(value, field=field)
    except TensorShapeValidationError as error:
        raise ModelInspectionValidationError(str(error)) from error
