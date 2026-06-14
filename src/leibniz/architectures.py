"""Architecture manifests as declarative model-structure records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError, load_object_document
from leibniz.identifiers import ProtocolIdentifier
from leibniz.model_scale_contracts import (
    ModelScaleContract,
    ModelScaleContractValidationError,
)
from leibniz.record_contracts import FieldContract, RecordContract
from leibniz.records import RecordExtractor, RecordSpec, record_specs_from_contract
from leibniz.tensor_shapes import TensorShape, TensorShapeValidationError

__all__ = [
    "ArchitectureComponent",
    "ArchitectureGraph",
    "ArchitectureGraphEdge",
    "ArchitectureGraphNode",
    "ArchitectureLayer",
    "ArchitectureManifest",
    "ArchitectureManifestDocument",
    "ArchitectureManifestValidationError",
]

class ArchitectureManifestValidationError(ValueError):
    """Raised when an architecture manifest is invalid."""


_extract = RecordExtractor(error_type=ArchitectureManifestValidationError)


@dataclass(frozen=True, slots=True)
class ArchitectureComponent:
    """One opaque model-structure component record."""

    kind: str
    parameters: Mapping[str, object]

    @classmethod
    def record_contract(cls) -> RecordContract:
        """Return the component record contract owned by this class."""

        return RecordContract(
            name="architecture_component",
            fields=(
                FieldContract(name="kind", kind="string"),
                FieldContract(name="parameters", kind="record", required=False),
            ),
        )

    @classmethod
    def record_spec(cls) -> RecordSpec:
        """Generate the Python validation runtime for component records."""

        return _record_spec_from_contract(cls.record_contract())

    def __post_init__(self) -> None:
        if not self.kind:
            raise ArchitectureManifestValidationError("component kind must be nonempty")
        try:
            ContentDigest.from_value(self.to_record())
        except ContentEncodingError as error:
            raise ArchitectureManifestValidationError(str(error)) from error

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ArchitectureComponent:
        try:
            validated = cls.record_spec().validate(record)
        except ValueError as error:
            raise ArchitectureManifestValidationError(str(error)) from error
        return cls(
            kind=str(validated["kind"]),
            parameters=_extract.mapping(validated.get("parameters", {}), "parameters"),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "parameters": dict(self.parameters),
        }


ArchitectureLayer = ArchitectureComponent


@dataclass(frozen=True, slots=True)
class ArchitectureGraphNode:
    """One node in a declarative model-architecture graph."""

    id: str
    component: ArchitectureComponent

    @classmethod
    def record_contract(cls) -> RecordContract:
        """Return the graph-node record contract owned by this class."""

        return RecordContract(
            name="architecture_graph_node",
            fields=(
                FieldContract(name="id", kind="string"),
                FieldContract(name="component", kind="record"),
            ),
        )

    @classmethod
    def record_spec(cls) -> RecordSpec:
        """Generate the Python validation runtime for graph-node records."""

        return _record_spec_from_contract(cls.record_contract())

    def __post_init__(self) -> None:
        if not self.id:
            raise ArchitectureManifestValidationError("graph node id must be nonempty")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ArchitectureGraphNode:
        try:
            validated = cls.record_spec().validate(record)
        except ValueError as error:
            raise ArchitectureManifestValidationError(str(error)) from error
        return cls(
            id=str(validated["id"]),
            component=ArchitectureComponent.from_record(
                _extract.mapping(validated["component"], "component")
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "component": self.component.to_record(),
        }


@dataclass(frozen=True, slots=True)
class ArchitectureGraphEdge:
    """One dependency edge in a declarative model-architecture graph."""

    source_node_id: str
    target_node_id: str
    kind: str = "data-flow"

    @classmethod
    def record_contract(cls) -> RecordContract:
        """Return the graph-edge record contract owned by this class."""

        return RecordContract(
            name="architecture_graph_edge",
            fields=(
                FieldContract(name="source_node_id", kind="string"),
                FieldContract(name="target_node_id", kind="string"),
                FieldContract(name="kind", kind="string"),
            ),
        )

    @classmethod
    def record_spec(cls) -> RecordSpec:
        """Generate the Python validation runtime for graph-edge records."""

        return _record_spec_from_contract(cls.record_contract())

    def __post_init__(self) -> None:
        if not self.source_node_id:
            raise ArchitectureManifestValidationError("edge source_node_id must be nonempty")
        if not self.target_node_id:
            raise ArchitectureManifestValidationError("edge target_node_id must be nonempty")
        if not self.kind:
            raise ArchitectureManifestValidationError("edge kind must be nonempty")
        if self.source_node_id == self.target_node_id:
            raise ArchitectureManifestValidationError("edge must not be a self-loop")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ArchitectureGraphEdge:
        try:
            validated = cls.record_spec().validate(record)
        except ValueError as error:
            raise ArchitectureManifestValidationError(str(error)) from error
        return cls(
            source_node_id=str(validated["source_node_id"]),
            target_node_id=str(validated["target_node_id"]),
            kind=str(validated["kind"]),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class ArchitectureGraph:
    """A graph-shaped model architecture over general components."""

    nodes: tuple[ArchitectureGraphNode, ...]
    edges: tuple[ArchitectureGraphEdge, ...]
    input_node_ids: tuple[str, ...]
    output_node_ids: tuple[str, ...]

    @classmethod
    def record_contract(cls) -> RecordContract:
        """Return the architecture graph record contract owned by this class."""

        return RecordContract(
            name="architecture_graph",
            fields=(
                FieldContract(
                    name="nodes",
                    kind="sequence",
                    item=FieldContract(kind="record"),
                ),
                FieldContract(
                    name="edges",
                    kind="sequence",
                    item=FieldContract(kind="record"),
                ),
                FieldContract(
                    name="input_node_ids",
                    kind="sequence",
                    item=FieldContract(kind="string"),
                ),
                FieldContract(
                    name="output_node_ids",
                    kind="sequence",
                    item=FieldContract(kind="string"),
                ),
            ),
        )

    @classmethod
    def record_spec(cls) -> RecordSpec:
        """Generate the Python validation runtime for graph records."""

        return _record_spec_from_contract(cls.record_contract())

    @classmethod
    def source_graph_facts(cls) -> tuple[Mapping[str, object], ...]:
        """Return source-graph facts for the graph contract hierarchy."""

        contracts = (
            ArchitectureComponent.record_contract(),
            ArchitectureGraphNode.record_contract(),
            ArchitectureGraphEdge.record_contract(),
            cls.record_contract(),
        )
        return (
            *tuple(fact for contract in contracts for fact in contract.source_graph_facts()),
            {
                "kind": "record-contract-edge",
                "source": cls.record_contract().name,
                "target": ArchitectureGraphNode.record_contract().name,
                "relationship": "contains-nodes",
            },
            {
                "kind": "record-contract-edge",
                "source": cls.record_contract().name,
                "target": ArchitectureGraphEdge.record_contract().name,
                "relationship": "contains-edges",
            },
            {
                "kind": "record-contract-edge",
                "source": ArchitectureGraphNode.record_contract().name,
                "target": ArchitectureComponent.record_contract().name,
                "relationship": "contains-component",
            },
        )

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ArchitectureManifestValidationError("graph nodes must not be empty")
        node_ids = tuple(node.id for node in self.nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ArchitectureManifestValidationError("graph node ids must be unique")
        known_node_ids = frozenset(node_ids)
        _require_known_node_ids(
            self.input_node_ids,
            known_node_ids=known_node_ids,
            field="input_node_ids",
        )
        _require_known_node_ids(
            self.output_node_ids,
            known_node_ids=known_node_ids,
            field="output_node_ids",
        )
        for edge in self.edges:
            if edge.source_node_id not in known_node_ids:
                raise ArchitectureManifestValidationError(
                    f"edge source_node_id {edge.source_node_id!r} is not a graph node"
                )
            if edge.target_node_id not in known_node_ids:
                raise ArchitectureManifestValidationError(
                    f"edge target_node_id {edge.target_node_id!r} is not a graph node"
                )
        _require_acyclic_graph(nodes=node_ids, edges=self.edges)

    @classmethod
    def sequential(cls, components: tuple[ArchitectureComponent, ...]) -> ArchitectureGraph:
        nodes = tuple(
            ArchitectureGraphNode(id=f"component-{index}", component=component)
            for index, component in enumerate(components)
        )
        if not nodes:
            raise ArchitectureManifestValidationError("graph nodes must not be empty")
        return cls(
            nodes=nodes,
            edges=tuple(
                ArchitectureGraphEdge(
                    source_node_id=nodes[index].id,
                    target_node_id=nodes[index + 1].id,
                )
                for index in range(len(nodes) - 1)
            ),
            input_node_ids=(nodes[0].id,),
            output_node_ids=(nodes[-1].id,),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ArchitectureGraph:
        try:
            validated = cls.record_spec().validate(record)
        except ValueError as error:
            raise ArchitectureManifestValidationError(str(error)) from error
        return cls(
            nodes=tuple(
                ArchitectureGraphNode.from_record(_extract.mapping(node, "nodes"))
                for node in _extract.sequence(validated["nodes"], "nodes")
            ),
            edges=tuple(
                ArchitectureGraphEdge.from_record(_extract.mapping(edge, "edges"))
                for edge in _extract.sequence(validated["edges"], "edges")
            ),
            input_node_ids=tuple(
                str(node_id)
                for node_id in _extract.sequence(
                    validated["input_node_ids"],
                    "input_node_ids",
                )
            ),
            output_node_ids=tuple(
                str(node_id)
                for node_id in _extract.sequence(
                    validated["output_node_ids"],
                    "output_node_ids",
                )
            ),
        )

    def to_record(self) -> dict[str, object]:
        return {
            "nodes": [node.to_record() for node in self.nodes],
            "edges": [edge.to_record() for edge in self.edges],
            "input_node_ids": list(self.input_node_ids),
            "output_node_ids": list(self.output_node_ids),
        }


@dataclass(frozen=True, slots=True)
class ArchitectureManifest:
    """A content-addressed declarative model-structure manifest."""

    id: ProtocolIdentifier
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    layers: tuple[ArchitectureLayer, ...]
    model_scale_contract: ModelScaleContract | None = None
    input_conditioning: Mapping[str, object] | None = None

    @classmethod
    def record_contract(cls) -> RecordContract:
        """Return the architecture manifest record contract owned by this class."""

        return RecordContract(
            name="architecture_manifest",
            fields=(
                FieldContract(name="id", kind="identifier", required=False),
                FieldContract(
                    name="input_shape",
                    kind="sequence",
                    item=FieldContract(kind="integer"),
                ),
                FieldContract(
                    name="output_shape",
                    kind="sequence",
                    item=FieldContract(kind="integer"),
                ),
                FieldContract(
                    name="layers",
                    kind="sequence",
                    item=FieldContract(kind="record"),
                ),
                FieldContract(
                    name="model_scale_contract",
                    kind="record",
                    required=False,
                ),
                FieldContract(
                    name="input_conditioning",
                    kind="record",
                    required=False,
                ),
            ),
        )

    @classmethod
    def record_spec(cls) -> RecordSpec:
        """Generate the Python validation runtime for architecture manifests."""

        return _record_spec_from_contract(cls.record_contract())

    @classmethod
    def source_graph_facts(cls) -> tuple[Mapping[str, object], ...]:
        """Return source-graph facts for manifest and graph contracts."""

        return (
            *cls.record_contract().source_graph_facts(),
            *ArchitectureGraph.source_graph_facts(),
            {
                "kind": "record-contract-edge",
                "source": cls.record_contract().name,
                "target": ArchitectureComponent.record_contract().name,
                "relationship": "contains-layers",
            },
            {
                "kind": "record-contract-edge",
                "source": cls.record_contract().name,
                "target": ArchitectureGraph.record_contract().name,
                "relationship": "projects-to-graph",
            },
        )

    def __post_init__(self) -> None:
        try:
            self.id.require_unreleased()
        except ValueError as error:
            raise ArchitectureManifestValidationError(str(error)) from error
        if self.id != self.derived_id():
            raise ArchitectureManifestValidationError(
                "id must be derived from architecture content"
            )
        _require_positive_shape(self.input_shape, field="input_shape")
        _require_positive_shape(self.output_shape, field="output_shape")
        if not self.layers:
            raise ArchitectureManifestValidationError("layers must contain at least one layer")
        if self.model_scale_contract is not None and (
            self.model_scale_contract.anchor_shape != self.input_shape
        ):
            raise ArchitectureManifestValidationError(
                "model_scale_contract anchor_shape must match input_shape"
            )

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> ArchitectureManifest:
        try:
            validated = cls.record_spec().validate(record)
            input_shape = _as_shape(validated["input_shape"], field="input_shape")
            output_shape = _as_shape(validated["output_shape"], field="output_shape")
            layers = tuple(
                ArchitectureLayer.from_record(_extract.mapping(layer, "layers"))
                for layer in _extract.sequence(validated["layers"], "layers")
            )
            scale_contract = _optional_scale_contract(
                validated.get("model_scale_contract")
            )
            input_conditioning = _optional_input_conditioning(
                validated.get("input_conditioning")
            )
        except ValueError as error:
            raise ArchitectureManifestValidationError(str(error)) from error
        content_record = _architecture_content_record(
            input_shape=input_shape,
            output_shape=output_shape,
            layers=layers,
            model_scale_contract=scale_contract,
            input_conditioning=input_conditioning,
        )
        derived_id = _architecture_id(content_record)
        identifier = validated.get("id", derived_id)
        return cls(
            id=_extract.identifier(identifier, "id"),
            input_shape=input_shape,
            output_shape=output_shape,
            layers=layers,
            model_scale_contract=scale_contract,
            input_conditioning=input_conditioning,
        )

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def derived_id(self) -> ProtocolIdentifier:
        return _architecture_id(self._content_record())

    def to_record(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            **self._content_record(),
        }

    @property
    def components(self) -> tuple[ArchitectureComponent, ...]:
        """Return the model-structure components in manifest order."""

        return self.layers

    @property
    def graph(self) -> ArchitectureGraph:
        """Return this sequential manifest as a single-path component graph."""

        return ArchitectureGraph.sequential(self.components)

    def _content_record(self) -> dict[str, object]:
        return _architecture_content_record(
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            layers=self.layers,
            model_scale_contract=self.model_scale_contract,
            input_conditioning=self.input_conditioning,
        )


@dataclass(frozen=True, slots=True)
class ArchitectureManifestDocument:
    """A loaded architecture manifest and the digest of its canonical record."""

    manifest: ArchitectureManifest
    digest: ContentDigest

    @classmethod
    def from_bytes(cls, data: bytes) -> ArchitectureManifestDocument:
        try:
            record = load_object_document(data, description="architecture manifest document")
        except ContentEncodingError as error:
            raise ArchitectureManifestValidationError(str(error)) from error
        manifest = ArchitectureManifest.from_record(record)
        return cls(manifest=manifest, digest=manifest.digest)


def _architecture_id(content_record: Mapping[str, object]) -> ProtocolIdentifier:
    digest = ContentDigest.from_value(content_record)
    return ProtocolIdentifier.parse(f"architecture.sha-{digest.hex}@0.1.0")


def _record_spec_from_contract(contract: RecordContract) -> RecordSpec:
    return record_specs_from_contract(_record_contract_set_record(contract))[contract.name]


def _record_contract_set_record(contract: RecordContract) -> Mapping[str, object]:
    return {
        "format": "leibniz.record-contract-set",
        "format_version": 1,
        "records": [
            {
                "name": contract.name,
                "allow_unknown": contract.allow_unknown,
                "fields": tuple(_field_contract_record(field) for field in contract.fields),
            }
        ],
    }


def _field_contract_record(field: FieldContract) -> dict[str, object]:
    record: dict[str, object] = {"kind": field.kind}
    if field.name is not None:
        record["name"] = field.name
    if not field.required:
        record["required"] = False
    if field.kind == "literal":
        record["literal"] = field.literal_or(None)
    if field.item is not None:
        record["item"] = _field_contract_record(field.item)
    if field.values is not None:
        record["values"] = list(field.values)
    return record


def _architecture_content_record(
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    layers: tuple[ArchitectureLayer, ...],
    model_scale_contract: ModelScaleContract | None,
    input_conditioning: Mapping[str, object] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "layers": [layer.to_record() for layer in layers],
    }
    if model_scale_contract is not None:
        record["model_scale_contract"] = model_scale_contract.to_record()
    if input_conditioning is not None:
        record["input_conditioning"] = dict(input_conditioning)
    return record


def _optional_scale_contract(value: object) -> ModelScaleContract | None:
    if value is None:
        return None
    try:
        return ModelScaleContract.from_record(_extract.mapping(value, "model_scale_contract"))
    except ModelScaleContractValidationError as error:
        raise ArchitectureManifestValidationError(str(error)) from error


def _optional_input_conditioning(value: object) -> Mapping[str, object] | None:
    if value is None:
        return None
    record = _extract.mapping(value, "input_conditioning")
    kind = record.get("kind")
    if kind != "horizon-channel":
        raise ArchitectureManifestValidationError(
            "input_conditioning kind must be horizon-channel"
        )
    if set(record) != {"kind"}:
        raise ArchitectureManifestValidationError(
            "input_conditioning horizon-channel does not accept extra fields"
        )
    return {"kind": "horizon-channel"}


def _as_shape(value: object, *, field: str) -> tuple[int, ...]:
    try:
        return TensorShape.from_record(_extract.sequence(value, field), field=field).axes
    except TensorShapeValidationError as error:
        raise ArchitectureManifestValidationError(str(error)) from error


def _require_positive_shape(shape: tuple[int, ...], *, field: str) -> None:
    try:
        TensorShape.from_axes(shape, field=field)
    except TensorShapeValidationError as error:
        raise ArchitectureManifestValidationError(str(error)) from error


def _require_known_node_ids(
    node_ids: tuple[str, ...],
    *,
    known_node_ids: frozenset[str],
    field: str,
) -> None:
    if not node_ids:
        raise ArchitectureManifestValidationError(f"{field} must not be empty")
    if len(set(node_ids)) != len(node_ids):
        raise ArchitectureManifestValidationError(f"{field} must be unique")
    for node_id in node_ids:
        if not node_id:
            raise ArchitectureManifestValidationError(f"{field} must not contain empty ids")
        if node_id not in known_node_ids:
            raise ArchitectureManifestValidationError(
                f"{field} contains unknown node id {node_id!r}"
            )


def _require_acyclic_graph(
    *,
    nodes: tuple[str, ...],
    edges: tuple[ArchitectureGraphEdge, ...],
) -> None:
    outgoing: dict[str, list[str]] = {node: [] for node in nodes}
    for edge in edges:
        outgoing[edge.source_node_id].append(edge.target_node_id)
    temporary: set[str] = set()
    permanent: set[str] = set()

    def visit(node: str) -> None:
        if node in permanent:
            return
        if node in temporary:
            raise ArchitectureManifestValidationError("architecture graph must be acyclic")
        temporary.add(node)
        for target in outgoing[node]:
            visit(target)
        temporary.remove(node)
        permanent.add(node)

    for node in nodes:
        visit(node)
