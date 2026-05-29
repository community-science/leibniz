"""Read-only model inspection records derived from public model artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from leibniz.architectures import ArchitectureGraph, ArchitectureManifest
from leibniz.artifacts import ArtifactReference, reference_for_record
from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.model_manifests import ModelArtifactManifest
from leibniz.model_operators import summarize_architecture_operators
from leibniz.records import FieldSpec, RecordSpec
from leibniz.submissions import SubmissionPackageManifest
from leibniz.tensor_shapes import TensorShape, TensorShapeValidationError

__all__ = [
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
        "parameter_bytes": FieldSpec(kind="integer", required=False),
        "inference_flops": FieldSpec(kind="integer", required=False),
    }
)
_cost_summary_record = RecordSpec(
    fields={
        "component_count": FieldSpec(kind="integer"),
        "parameter_count": FieldSpec(kind="integer", required=False),
        "parameter_bytes": FieldSpec(kind="integer", required=False),
        "inference_flops": FieldSpec(kind="integer", required=False),
        "unknown_parameter_components": FieldSpec(
            kind="sequence",
            item=FieldSpec(kind="integer"),
        ),
        "unknown_flop_components": FieldSpec(
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
        "inference_flops": FieldSpec(kind="integer", required=False),
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
        "unsupported_flop_components": FieldSpec(
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
        "components": FieldSpec(kind="sequence", item=FieldSpec(kind="record")),
        "cost_summary": FieldSpec(kind="record"),
        "architecture_trace": FieldSpec(kind="record"),
        "architecture_graph": FieldSpec(kind="record"),
        "architecture_summary": FieldSpec(kind="record"),
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
class ModelInspectionComponent:
    """One component summary for read-only model inspection."""

    index: int
    kind: str
    parameters: Mapping[str, object]
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None
    operator: Mapping[str, object] | None = None
    parameter_count: int | None = None
    parameter_bytes: int | None = None
    inference_flops: int | None = None

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
        if self.parameter_bytes is not None and self.parameter_bytes < 0:
            raise ModelInspectionValidationError("parameter_bytes must be nonnegative")
        if self.inference_flops is not None and self.inference_flops < 0:
            raise ModelInspectionValidationError("inference_flops must be nonnegative")
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
            index=_as_int(validated["index"], field="index"),
            kind=_as_string(validated["kind"], field="kind"),
            parameters=_as_mapping(validated["parameters"], field="parameters"),
            input_shape=_optional_shape(validated.get("input_shape"), field="input_shape"),
            output_shape=_optional_shape(validated.get("output_shape"), field="output_shape"),
            operator=_optional_mapping(validated.get("operator"), field="operator"),
            parameter_count=_optional_int(
                validated.get("parameter_count"),
                field="parameter_count",
            ),
            parameter_bytes=_optional_int(
                validated.get("parameter_bytes"),
                field="parameter_bytes",
            ),
            inference_flops=_optional_int(
                validated.get("inference_flops"),
                field="inference_flops",
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
        if self.parameter_bytes is not None:
            record["parameter_bytes"] = self.parameter_bytes
        if self.inference_flops is not None:
            record["inference_flops"] = self.inference_flops
        return record

@dataclass(frozen=True, slots=True)
class ModelInspectionCostSummary:
    """Conservative model cost summary derived from public architecture structure."""

    component_count: int
    parameter_count: int | None
    parameter_bytes: int | None = None
    inference_flops: int | None = None
    unknown_parameter_components: tuple[int, ...] = ()
    unknown_flop_components: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.component_count) is not int or self.component_count < 0:
            raise ModelInspectionValidationError("component_count must be a nonnegative integer")
        if self.parameter_count is not None and self.parameter_count < 0:
            raise ModelInspectionValidationError("parameter_count must be nonnegative")
        if self.parameter_bytes is not None and self.parameter_bytes < 0:
            raise ModelInspectionValidationError("parameter_bytes must be nonnegative")
        if self.inference_flops is not None and self.inference_flops < 0:
            raise ModelInspectionValidationError("inference_flops must be nonnegative")
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
        if any(type(index) is not int or index < 0 for index in self.unknown_flop_components):
            raise ModelInspectionValidationError(
                "unknown_flop_components must contain nonnegative integers"
            )
        if self.unknown_flop_components != tuple(sorted(set(self.unknown_flop_components))):
            raise ModelInspectionValidationError("unknown_flop_components must be sorted unique")
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
            component_count=_as_int(validated["component_count"], field="component_count"),
            parameter_count=_optional_int(
                validated.get("parameter_count"),
                field="parameter_count",
            ),
            parameter_bytes=_optional_int(
                validated.get("parameter_bytes"),
                field="parameter_bytes",
            ),
            inference_flops=_optional_int(
                validated.get("inference_flops"),
                field="inference_flops",
            ),
            unknown_parameter_components=tuple(
                _as_int(index, field="unknown_parameter_components")
                for index in _as_sequence(
                    validated["unknown_parameter_components"],
                    field="unknown_parameter_components",
                )
            ),
            unknown_flop_components=tuple(
                _as_int(index, field="unknown_flop_components")
                for index in _as_sequence(
                    validated.get("unknown_flop_components", ()),
                    field="unknown_flop_components",
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
        if self.parameter_bytes is not None:
            record["parameter_bytes"] = self.parameter_bytes
        if self.inference_flops is not None:
            record["inference_flops"] = self.inference_flops
        if self.unknown_flop_components:
            record["unknown_flop_components"] = list(self.unknown_flop_components)
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
    inference_flops: int | None = None

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
        if self.inference_flops is not None and self.inference_flops < 0:
            raise ModelInspectionValidationError("trace stage inference_flops must be nonnegative")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelInspectionTraceStage:
        try:
            validated = _trace_stage_record.validate(record)
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        return cls(
            index=_as_int(validated["index"], field="index"),
            kind=_as_string(validated["kind"], field="kind"),
            syntax_alias=_as_string(validated["syntax_alias"], field="syntax_alias"),
            operator_kind=_as_string(validated["operator_kind"], field="operator_kind"),
            input_shape=_as_shape(validated["input_shape"], field="input_shape"),
            output_shape=_as_shape(validated["output_shape"], field="output_shape"),
            descriptor_axes=_string_mapping(
                validated["descriptor_axes"],
                field="descriptor_axes",
            ),
            shape_law=_as_string(validated["shape_law"], field="shape_law"),
            cost_law=_as_string(validated["cost_law"], field="cost_law"),
            parameter_count=_optional_int(
                validated.get("parameter_count"),
                field="parameter_count",
            ),
            inference_flops=_optional_int(
                validated.get("inference_flops"),
                field="inference_flops",
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
        if self.inference_flops is not None:
            record["inference_flops"] = self.inference_flops
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
                ModelInspectionTraceStage.from_record(_as_mapping(stage, field="stages"))
                for stage in _as_sequence(validated["stages"], field="stages")
            ),
            program_effects=tuple(
                _as_mapping(effect, field="program_effects")
                for effect in _as_sequence(
                    validated.get("program_effects", ()),
                    field="program_effects",
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
    unsupported_flop_components: tuple[int, ...] = ()

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
            self.unsupported_flop_components,
            component_count=self.component_count,
            field="unsupported_flop_components",
        )

    @classmethod
    def from_graph(
        cls,
        *,
        graph: ArchitectureGraph,
        cost_summary: ModelInspectionCostSummary,
    ) -> ModelInspectionGraphSummary:
        return cls(
            component_count=len(graph.nodes),
            edge_count=len(graph.edges),
            input_count=len(graph.input_node_ids),
            output_count=len(graph.output_node_ids),
            input_node_ids=graph.input_node_ids,
            output_node_ids=graph.output_node_ids,
            component_kinds=tuple(node.component.kind for node in graph.nodes),
            unsupported_parameter_components=cost_summary.unknown_parameter_components,
            unsupported_flop_components=cost_summary.unknown_flop_components,
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ModelInspectionGraphSummary:
        try:
            validated = _graph_summary_record.validate(record)
        except ValueError as error:
            raise ModelInspectionValidationError(str(error)) from error
        return cls(
            component_count=_as_int(validated["component_count"], field="component_count"),
            edge_count=_as_int(validated["edge_count"], field="edge_count"),
            input_count=_as_int(validated["input_count"], field="input_count"),
            output_count=_as_int(validated["output_count"], field="output_count"),
            input_node_ids=tuple(
                _as_string(node_id, field="input_node_ids")
                for node_id in _as_sequence(
                    validated["input_node_ids"],
                    field="input_node_ids",
                )
            ),
            output_node_ids=tuple(
                _as_string(node_id, field="output_node_ids")
                for node_id in _as_sequence(
                    validated["output_node_ids"],
                    field="output_node_ids",
                )
            ),
            component_kinds=tuple(
                _as_string(kind, field="component_kinds")
                for kind in _as_sequence(
                    validated["component_kinds"],
                    field="component_kinds",
                )
            ),
            unsupported_parameter_components=tuple(
                _as_int(index, field="unsupported_parameter_components")
                for index in _as_sequence(
                    validated["unsupported_parameter_components"],
                    field="unsupported_parameter_components",
                )
            ),
            unsupported_flop_components=tuple(
                _as_int(index, field="unsupported_flop_components")
                for index in _as_sequence(
                    validated["unsupported_flop_components"],
                    field="unsupported_flop_components",
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
            "unsupported_flop_components": list(self.unsupported_flop_components),
        }


@dataclass(frozen=True, slots=True)
class ModelInspectionRecord:
    """A normalized read-only inspection record for public model artifacts."""

    id: ProtocolIdentifier
    architecture: ArtifactReference
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    components: tuple[ModelInspectionComponent, ...]
    cost_summary: ModelInspectionCostSummary
    architecture_trace: ModelInspectionTrace
    architecture_graph: ArchitectureGraph
    architecture_summary: ModelInspectionGraphSummary
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
        if self.architecture_trace.input_shape != self.input_shape:
            raise ModelInspectionValidationError("architecture_trace input_shape does not match")
        if self.architecture_trace.output_shape != self.output_shape:
            raise ModelInspectionValidationError("architecture_trace output_shape does not match")
        if len(self.architecture_trace.stages) != len(self.components):
            raise ModelInspectionValidationError(
                "architecture_trace stages do not match components"
            )
        for component, stage in zip(
            self.components,
            self.architecture_trace.stages,
            strict=True,
        ):
            if component.index != stage.index or component.kind != stage.syntax_alias:
                raise ModelInspectionValidationError(
                    "architecture_trace stage does not match component"
                )
        if len(self.architecture_graph.nodes) != len(self.components):
            raise ModelInspectionValidationError(
                "architecture_graph nodes do not match components"
            )
        for component, node in zip(self.components, self.architecture_graph.nodes, strict=True):
            if component.kind != node.component.kind or dict(component.parameters) != dict(
                node.component.parameters
            ):
                raise ModelInspectionValidationError(
                    "architecture_graph node does not match component"
                )
        expected_summary = ModelInspectionGraphSummary.from_graph(
            graph=self.architecture_graph,
            cost_summary=self.cost_summary,
        )
        if self.architecture_summary != expected_summary:
            raise ModelInspectionValidationError("architecture_summary does not match graph")
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
        components, cost_summary, architecture_trace = _architecture_components(
            architecture_manifest
        )
        return cls(
            id=id,
            architecture=reference_for_record(
                kind="architecture-manifest",
                record=architecture_manifest.to_record(),
            ),
            input_shape=architecture_manifest.input_shape,
            output_shape=architecture_manifest.output_shape,
            components=components,
            cost_summary=cost_summary,
            architecture_trace=architecture_trace,
            architecture_graph=architecture_manifest.graph,
            architecture_summary=ModelInspectionGraphSummary.from_graph(
                graph=architecture_manifest.graph,
                cost_summary=cost_summary,
            ),
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
            components=record.components,
            cost_summary=record.cost_summary,
            architecture_trace=record.architecture_trace,
            architecture_graph=record.architecture_graph,
            architecture_summary=record.architecture_summary,
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
            components=record.components,
            cost_summary=record.cost_summary,
            architecture_trace=record.architecture_trace,
            architecture_graph=record.architecture_graph,
            architecture_summary=record.architecture_summary,
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
            components = tuple(
                ModelInspectionComponent.from_record(
                    _as_mapping(component, field="components")
                )
                for component in _as_sequence(validated["components"], field="components")
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
            components=components,
            cost_summary=ModelInspectionCostSummary.from_record(
                _as_mapping(validated["cost_summary"], field="cost_summary")
            ),
            architecture_trace=ModelInspectionTrace.from_record(
                _as_mapping(validated["architecture_trace"], field="architecture_trace")
            ),
            architecture_graph=ArchitectureGraph.from_record(
                _as_mapping(validated["architecture_graph"], field="architecture_graph")
            ),
            architecture_summary=ModelInspectionGraphSummary.from_record(
                _as_mapping(validated["architecture_summary"], field="architecture_summary")
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
            "components": [component.to_record() for component in self.components],
            "cost_summary": self.cost_summary.to_record(),
            "architecture_trace": self.architecture_trace.to_record(),
            "architecture_graph": self.architecture_graph.to_record(),
            "architecture_summary": self.architecture_summary.to_record(),
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


def _architecture_components(
    architecture_manifest: ArchitectureManifest,
) -> tuple[tuple[ModelInspectionComponent, ...], ModelInspectionCostSummary, ModelInspectionTrace]:
    components: list[ModelInspectionComponent] = []
    stages: list[ModelInspectionTraceStage] = []
    plan = summarize_architecture_operators(architecture_manifest)
    for component, operator in zip(architecture_manifest.components, plan.operators, strict=True):
        descriptor = operator.descriptor
        components.append(
            ModelInspectionComponent(
                index=operator.index,
                kind=component.kind,
                parameters=component.parameters,
                input_shape=operator.input_shape,
                output_shape=operator.output_shape,
                operator=descriptor.to_record(),
                parameter_count=operator.parameter_count,
                parameter_bytes=operator.parameter_bytes,
                inference_flops=operator.inference_flops,
            )
        )
        if operator.input_shape is not None and operator.output_shape is not None:
            stages.append(
                ModelInspectionTraceStage(
                    index=operator.index,
                    kind="operator",
                    syntax_alias=component.kind,
                    operator_kind=descriptor.kind,
                    input_shape=operator.input_shape,
                    output_shape=operator.output_shape,
                    descriptor_axes={
                        "tensor_relation": descriptor.tensor_relation,
                        "state": descriptor.state,
                        "support": descriptor.support,
                        "projection_law": descriptor.projection_law,
                        "aggregation_law": descriptor.aggregation_law,
                        "parameter_sharing": descriptor.parameter_sharing,
                    },
                    shape_law=descriptor.shape_law,
                    cost_law=descriptor.cost_law,
                    parameter_count=operator.parameter_count,
                    inference_flops=operator.inference_flops,
                )
            )
    return (
        tuple(components),
        ModelInspectionCostSummary(
            component_count=len(components),
            parameter_count=plan.parameter_count,
            parameter_bytes=plan.parameter_bytes,
            inference_flops=plan.inference_flops,
            unknown_parameter_components=plan.unknown_parameter_layers,
            unknown_flop_components=plan.unknown_flop_layers,
        ),
        ModelInspectionTrace(
            input_shape=architecture_manifest.input_shape,
            output_shape=architecture_manifest.output_shape,
            stages=tuple(stages),
        ),
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


def _optional_mapping(value: object, *, field: str) -> Mapping[str, object] | None:
    if value is None:
        return None
    return _as_mapping(value, field=field)


def _string_mapping(value: object, *, field: str) -> Mapping[str, str]:
    mapping = _as_mapping(value, field=field)
    result: dict[str, str] = {}
    for key, item in mapping.items():
        if not isinstance(item, str):
            raise ModelInspectionValidationError(f"{field}: expected string mapping")
        result[key] = item
    return result


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
    try:
        return TensorShape.from_record(
            tuple(_as_int(axis, field=field) for axis in _as_sequence(value, field=field)),
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
