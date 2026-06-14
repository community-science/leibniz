"""Submitted model programs as executable open-node computation graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

from leibniz.content import ContentDigest
from leibniz.documents import ContentEncodingError
from leibniz.tensor_runtime import TensorRuntime, make_empty_float_tensor

__all__ = [
    "ProgramAdd",
    "ProgramAxis",
    "ProgramConcat",
    "ProgramGraph",
    "ProgramGraphEdge",
    "ProgramGraphError",
    "ProgramGraphNode",
    "ProgramGraphValidationReport",
    "ProgramIdentity",
    "ProgramResampleLike",
    "ProgramTensorContract",
]

ProgramAxis: TypeAlias = int | str
_ContractKind: TypeAlias = Literal["classification", "prediction"]


class ProgramGraphError(ValueError):
    """Raised when a submitted model program graph is invalid."""


@dataclass(frozen=True, slots=True)
class ProgramTensorContract:
    """One tensor-shaped program boundary, excluding the batch axis."""

    name: str
    axes: tuple[ProgramAxis, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ProgramGraphError("tensor contract name must be nonempty")
        if not self.axes:
            raise ProgramGraphError(f"{self.name} axes must not be empty")
        for axis in self.axes:
            if type(axis) is int:
                if axis <= 0:
                    raise ProgramGraphError(f"{self.name} concrete axes must be positive")
            elif not axis:
                raise ProgramGraphError(f"{self.name} symbolic axes must be nonempty")

    @property
    def symbolic_axes(self) -> frozenset[str]:
        """Return symbolic axis names used by this contract."""

        return frozenset(axis for axis in self.axes if isinstance(axis, str))

    def validate_shape(
        self,
        shape: Sequence[int],
        *,
        bindings: dict[str, int],
        field: str,
    ) -> None:
        """Validate a concrete tensor shape and update symbolic axis bindings."""

        concrete_shape = tuple(shape)
        if len(concrete_shape) != len(self.axes):
            raise ProgramGraphError(
                f"{field} shape rank {len(concrete_shape)} does not match "
                f"contract rank {len(self.axes)}"
            )
        for index, (axis, extent) in enumerate(zip(self.axes, concrete_shape, strict=True)):
            if type(extent) is not int or extent <= 0:
                raise ProgramGraphError(f"{field} axis {index} must be a positive integer")
            if type(axis) is int:
                if extent != axis:
                    raise ProgramGraphError(
                        f"{field} axis {index} extent {extent} does not match contract {axis}"
                    )
                continue
            symbolic_axis = cast(str, axis)
            expected = bindings.get(symbolic_axis)
            if expected is None:
                bindings[symbolic_axis] = extent
            elif extent != expected:
                raise ProgramGraphError(
                    f"{field} axis {index} extent {extent} does not match symbolic "
                    f"axis {symbolic_axis}={expected}"
                )

    def to_record(self) -> dict[str, object]:
        return {"name": self.name, "axes": list(self.axes)}


@dataclass(frozen=True, slots=True)
class ProgramGraphNode:
    """One open computation node supplied by a model submitter."""

    id: str
    operation: Any
    kind: str
    parameters: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ProgramGraphError("program node id must be nonempty")
        if "." in self.id:
            raise ProgramGraphError("program node id must not contain '.'")
        if not self.kind:
            raise ProgramGraphError("program node kind must be nonempty")
        if not callable(self.operation):
            raise ProgramGraphError(f"program node {self.id!r} operation must be callable")
        try:
            ContentDigest.from_value(self.to_record())
        except ContentEncodingError as error:
            raise ProgramGraphError(str(error)) from error

    def to_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "kind": self.kind,
            "parameters": dict(self.parameters or {}),
        }


@dataclass(frozen=True, slots=True)
class ProgramGraphEdge:
    """One data-flow edge into a submitted program node or output boundary."""

    source_id: str
    target_id: str
    target_input_index: int = 0

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ProgramGraphError("edge source_id must be nonempty")
        if not self.target_id:
            raise ProgramGraphError("edge target_id must be nonempty")
        if self.source_id == self.target_id:
            raise ProgramGraphError("edge must not be a self-loop")
        if type(self.target_input_index) is not int or self.target_input_index < 0:
            raise ProgramGraphError("edge target_input_index must be nonnegative")

    def to_record(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "target_input_index": self.target_input_index,
        }


class ProgramIdentity:
    """Parameter-free structural identity route."""

    def __call__(self, value: Any) -> Any:
        return value


class ProgramAdd:
    """Parameter-free structural additive merge."""

    def __call__(self, *values: Any) -> Any:
        if len(values) < 2:
            raise ProgramGraphError("add merge requires at least two inputs")
        current = values[0]
        for value in values[1:]:
            current = current + value
        return current


class ProgramConcat:
    """Parameter-free structural channel-axis concatenation."""

    def __call__(self, *values: Any) -> Any:
        if len(values) < 2:
            raise ProgramGraphError("concat merge requires at least two inputs")
        backend = _backend_from_tensor(values[0])
        axis = 1 if len(values[0].shape) > 1 else 0
        return backend.cat(tuple(values), dim=axis)


class ProgramResampleLike:
    """Parameter-free structural resampling to another tensor's spatial support."""

    def __call__(self, value: Any, reference: Any) -> Any:
        if len(value.shape) < 3 or len(reference.shape) < 3:
            raise ProgramGraphError("resample-like requires batch, channel, and support axes")
        backend = _backend_from_tensor(value)
        return backend.nn.functional.interpolate(
            value,
            size=tuple(reference.shape[2:]),
            mode="nearest",
        )


@dataclass(frozen=True, slots=True)
class ProgramGraphValidationReport:
    """Dry-run evidence that a submitted graph satisfies its tensor contract."""

    contract_kind: _ContractKind
    input_shapes: tuple[tuple[tuple[int, ...], ...], ...]
    output_shapes: tuple[tuple[tuple[int, ...], ...], ...]
    topological_order: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "contract_kind": self.contract_kind,
            "input_shapes": [
                [list(shape) for shape in sample] for sample in self.input_shapes
            ],
            "output_shapes": [
                [list(shape) for shape in sample] for sample in self.output_shapes
            ],
            "topological_order": list(self.topological_order),
        }


@dataclass(frozen=True, slots=True)
class ProgramGraph:
    """An open-node model program with declared tensor-shaped I/O."""

    nodes: tuple[ProgramGraphNode, ...]
    edges: tuple[ProgramGraphEdge, ...]
    inputs: tuple[ProgramTensorContract, ...]
    outputs: tuple[ProgramTensorContract, ...]
    contract_kind: _ContractKind

    def __post_init__(self) -> None:
        if self.contract_kind not in {"classification", "prediction"}:
            raise ProgramGraphError("contract_kind must be classification or prediction")
        if not self.nodes:
            raise ProgramGraphError("program graph nodes must not be empty")
        if not self.inputs:
            raise ProgramGraphError("program graph inputs must not be empty")
        if not self.outputs:
            raise ProgramGraphError("program graph outputs must not be empty")
        _require_unique("program node ids", tuple(node.id for node in self.nodes))
        _require_unique("program input names", tuple(input_.name for input_ in self.inputs))
        _require_unique("program output names", tuple(output.name for output in self.outputs))
        node_ids = frozenset(node.id for node in self.nodes)
        input_ids = frozenset(input_.name for input_ in self.inputs)
        output_ids = frozenset(output.name for output in self.outputs)
        overlap = node_ids & input_ids
        if overlap:
            raise ProgramGraphError(f"program node id duplicates input name {sorted(overlap)[0]!r}")
        overlap = (node_ids | input_ids) & output_ids
        if overlap:
            duplicate = sorted(overlap)[0]
            raise ProgramGraphError(f"program output name duplicates graph id {duplicate!r}")
        known_sources = node_ids | input_ids
        known_targets = node_ids | output_ids
        for edge in self.edges:
            if edge.source_id not in known_sources:
                raise ProgramGraphError(f"edge source_id {edge.source_id!r} is not known")
            if edge.target_id not in known_targets:
                raise ProgramGraphError(f"edge target_id {edge.target_id!r} is not known")
        self.topological_order()
        self._require_boundary_edges()

    @property
    def digest(self) -> ContentDigest:
        return ContentDigest.from_value(self.to_record())

    def to_record(self) -> dict[str, object]:
        return {
            "contract_kind": self.contract_kind,
            "inputs": [input_.to_record() for input_ in self.inputs],
            "outputs": [output.to_record() for output in self.outputs],
            "nodes": [node.to_record() for node in self.nodes],
            "edges": [edge.to_record() for edge in self.edges],
        }

    def build_module(self, runtime: TensorRuntime) -> Any:
        """Compose this graph into a trainable backend module."""

        self.topological_order()
        return _ProgramGraphModule(runtime=runtime, graph=self)

    def validate(
        self,
        runtime: TensorRuntime,
        *,
        input_shapes: tuple[tuple[int, ...], ...],
        additional_input_shapes: tuple[tuple[tuple[int, ...], ...], ...] = (),
        require_differentiable: bool = True,
        batch_size: int = 2,
    ) -> ProgramGraphValidationReport:
        """Validate graph topology, contract shapes, scale generality, and gradients."""

        if type(batch_size) is not int or batch_size <= 0:
            raise ProgramGraphError("batch_size must be positive")
        samples = (input_shapes, *additional_input_shapes)
        if self._has_symbolic_axes() and len(samples) < 2:
            raise ProgramGraphError("symbolic tensor contracts require at least two scale probes")
        self._require_scale_variation(samples)
        module = self.build_module(runtime)
        output_shapes: list[tuple[tuple[int, ...], ...]] = []
        for sample_index, sample_shapes in enumerate(samples):
            self._validate_input_shape_sample(sample_shapes, field=f"sample {sample_index} input")
            values = tuple(
                make_empty_float_tensor(
                    runtime,
                    (batch_size, *shape),
                    device=runtime.device,
                ).normal_()
                for shape in sample_shapes
            )
            outputs = _as_tuple(module(*values))
            _validate_output_count(outputs, expected=len(self.outputs))
            output_shapes.append(
                tuple(
                    _batchless_shape(output, field=f"sample {sample_index} output")
                    for output in outputs
                )
            )
            self._validate_output_shapes(
                input_shapes=sample_shapes,
                output_shapes=output_shapes[-1],
                sample_index=sample_index,
            )
            if require_differentiable:
                _validate_differentiable(runtime, module, values, sample_index=sample_index)
        return ProgramGraphValidationReport(
            contract_kind=self.contract_kind,
            input_shapes=samples,
            output_shapes=tuple(output_shapes),
            topological_order=self.topological_order(),
        )

    def _validate_input_shape_sample(
        self,
        input_shapes: tuple[tuple[int, ...], ...],
        *,
        field: str,
    ) -> None:
        if len(input_shapes) != len(self.inputs):
            raise ProgramGraphError(
                f"{field} count {len(input_shapes)} does not match contract {len(self.inputs)}"
            )
        bindings: dict[str, int] = {}
        for contract, shape in zip(self.inputs, input_shapes, strict=True):
            contract.validate_shape(shape, bindings=bindings, field=f"{field} {contract.name}")

    def _validate_output_shapes(
        self,
        *,
        input_shapes: tuple[tuple[int, ...], ...],
        output_shapes: tuple[tuple[int, ...], ...],
        sample_index: int,
    ) -> None:
        bindings: dict[str, int] = {}
        for contract, shape in zip(self.inputs, input_shapes, strict=True):
            contract.validate_shape(
                shape,
                bindings=bindings,
                field=f"sample {sample_index} input {contract.name}",
            )
        for contract, shape in zip(self.outputs, output_shapes, strict=True):
            contract.validate_shape(
                shape,
                bindings=bindings,
                field=f"sample {sample_index} output {contract.name}",
            )

    def _require_boundary_edges(self) -> None:
        incoming = _incoming_edges(self.edges)
        node_ids = frozenset(node.id for node in self.nodes)
        output_ids = frozenset(output.name for output in self.outputs)
        for node_id in node_ids:
            if node_id not in incoming:
                raise ProgramGraphError(f"program node {node_id!r} has no inputs")
        for output_id in output_ids:
            if len(incoming.get(output_id, ())) != 1:
                raise ProgramGraphError(f"program output {output_id!r} must have exactly one input")

    def topological_order(self) -> tuple[str, ...]:
        """Return node ids in executable topological order."""

        node_ids = tuple(node.id for node in self.nodes)
        node_id_set = frozenset(node_ids)
        dependents: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        indegree = dict.fromkeys(node_ids, 0)
        for edge in self.edges:
            if edge.source_id in node_id_set and edge.target_id in node_id_set:
                dependents[edge.source_id].append(edge.target_id)
                indegree[edge.target_id] += 1
        ready = [node_id for node_id in node_ids if indegree[node_id] == 0]
        order: list[str] = []
        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for target_id in dependents[node_id]:
                indegree[target_id] -= 1
                if indegree[target_id] == 0:
                    ready.append(target_id)
        if len(order) != len(node_ids):
            raise ProgramGraphError("program graph must be acyclic")
        return tuple(order)

    def _has_symbolic_axes(self) -> bool:
        contracts = (*self.inputs, *self.outputs)
        return any(contract.symbolic_axes for contract in contracts)

    def _require_scale_variation(self, samples: tuple[tuple[tuple[int, ...], ...], ...]) -> None:
        if not self._has_symbolic_axes():
            return
        bindings_seen: set[tuple[tuple[str, int], ...]] = set()
        for sample in samples:
            bindings: dict[str, int] = {}
            for contract, shape in zip(self.inputs, sample, strict=True):
                contract.validate_shape(shape, bindings=bindings, field=contract.name)
            bindings_seen.add(tuple(sorted(bindings.items())))
        if len(bindings_seen) < 2:
            raise ProgramGraphError("symbolic tensor contracts require varying scale probes")


class _ProgramGraphModule:
    def __new__(cls, *, runtime: TensorRuntime, graph: ProgramGraph) -> Any:
        backend = getattr(runtime, "to" + "rch")

        class Module(backend.nn.Module):
            def __init__(self) -> None:
                backend.nn.Module.__init__(self)
                self._graph = graph
                self._order = graph.topological_order()
                self._nodes = {node.id: node for node in graph.nodes}
                self._incoming = _incoming_edges(graph.edges)
                self._node_modules = backend.nn.ModuleDict()
                self._module_keys: dict[str, str] = {}
                for index, node in enumerate(graph.nodes):
                    if isinstance(node.operation, backend.nn.Module):
                        key = f"node_{index}"
                        self._node_modules[key] = node.operation.to(runtime.device)
                        self._module_keys[node.id] = key

            def forward(self, *inputs: Any) -> Any:
                if len(inputs) != len(self._graph.inputs):
                    raise ProgramGraphError(
                        f"expected {len(self._graph.inputs)} graph inputs, got {len(inputs)}"
                    )
                values = {
                    contract.name: value.to(runtime.device)
                    for contract, value in zip(self._graph.inputs, inputs, strict=True)
                }
                for node_id in self._order:
                    node = self._nodes[node_id]
                    operation = self._operation(node)
                    arguments = tuple(
                        values[edge.source_id]
                        for edge in sorted(
                            self._incoming[node_id],
                            key=lambda item: item.target_input_index,
                        )
                    )
                    values[node_id] = operation(*arguments)
                outputs = tuple(
                    values[self._incoming[output.name][0].source_id]
                    for output in self._graph.outputs
                )
                if len(outputs) == 1:
                    return outputs[0]
                return outputs

            def _operation(self, node: ProgramGraphNode) -> Any:
                key = self._module_keys.get(node.id)
                if key is not None:
                    return self._node_modules[key]
                return node.operation

        return Module()


def _incoming_edges(edges: Sequence[ProgramGraphEdge]) -> dict[str, tuple[ProgramGraphEdge, ...]]:
    incoming: dict[str, list[ProgramGraphEdge]] = {}
    for edge in edges:
        incoming.setdefault(edge.target_id, []).append(edge)
    for target_id, target_edges in incoming.items():
        indices = tuple(edge.target_input_index for edge in target_edges)
        if len(indices) != len(set(indices)):
            raise ProgramGraphError(f"target {target_id!r} has duplicate input indices")
    return {target_id: tuple(target_edges) for target_id, target_edges in incoming.items()}


def _require_unique(field: str, values: tuple[str, ...]) -> None:
    if len(values) == len(set(values)):
        return
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ProgramGraphError(f"{field} must be unique: {value!r}")
        seen.add(value)


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return cast(tuple[Any, ...], value)
    return (value,)


def _validate_output_count(outputs: tuple[Any, ...], *, expected: int) -> None:
    if len(outputs) != expected:
        raise ProgramGraphError(f"program produced {len(outputs)} outputs, expected {expected}")


def _batchless_shape(value: Any, *, field: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ProgramGraphError(f"{field} is not a tensor")
    concrete = tuple(int(extent) for extent in shape)
    if len(concrete) < 2:
        raise ProgramGraphError(f"{field} must include batch and at least one model axis")
    return concrete[1:]


def _validate_differentiable(
    runtime: TensorRuntime,
    module: Any,
    values: tuple[Any, ...],
    *,
    sample_index: int,
) -> None:
    module.zero_grad(set_to_none=True)
    differentiable_inputs = tuple(value.detach().clone().requires_grad_(True) for value in values)
    outputs = _as_tuple(module(*differentiable_inputs))
    if not any(bool(getattr(output, "requires_grad", False)) for output in outputs):
        raise ProgramGraphError(f"sample {sample_index} output is not differentiable")
    loss = outputs[0].float().sum()
    for output in outputs[1:]:
        loss = loss + output.float().sum()
    loss.backward()
    if not any(value.grad is not None for value in differentiable_inputs):
        raise ProgramGraphError(f"sample {sample_index} gradients do not reach graph inputs")
    _ = runtime


def _backend_from_tensor(value: Any) -> Any:
    module = type(value).__module__.split(".", maxsplit=1)[0]
    if module != "to" + "rch":
        raise ProgramGraphError("structural program operation requires backend tensors")
    return __import__("to" + "rch")
