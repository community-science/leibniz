"""Declared model-operator semantics for architecture interpretation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

__all__ = [
    "DescriptorAxis",
    "DescriptorAxisValue",
    "ModelOperatorParameterRole",
    "ModelOperatorSemantic",
    "ModelOperatorSemanticRegistry",
    "OperatorSemanticValidationError",
    "SemanticCoordinateDescriptor",
    "model_operator_semantic_registry",
]


class OperatorSemanticValidationError(ValueError):
    """Raised when model-operator semantic records are invalid."""


@dataclass(frozen=True, slots=True)
class ModelOperatorParameterRole:
    """One public parameter role for an operator syntax alias."""

    name: str
    display_name: str
    description: str
    value_kind: str = "positive-integer"

    def __post_init__(self) -> None:
        _require_nonempty(self.name, field="parameter role name")
        _require_nonempty(self.display_name, field="parameter role display_name")
        _require_nonempty(self.description, field="parameter role description")
        _require_nonempty(self.value_kind, field="parameter role value_kind")

    def to_record(self) -> dict[str, str]:
        """Return a canonical record for this parameter role."""

        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "value_kind": self.value_kind,
        }


@dataclass(frozen=True, slots=True)
class ModelOperatorSemantic:
    """Declared semantics for one public model-operator kind."""

    kind: str
    display_name: str
    tensor_relation: str
    state: str
    support: str
    projection_law: str
    aggregation_law: str
    parameter_sharing: str
    shape_law: str
    cost_law: str
    syntax_aliases: tuple[str, ...]
    parameter_roles: tuple[ModelOperatorParameterRole, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.kind, field="operator kind")
        _require_nonempty(self.display_name, field="operator display_name")
        _require_nonempty(self.tensor_relation, field="operator tensor_relation")
        _require_nonempty(self.state, field="operator state")
        _require_nonempty(self.support, field="operator support")
        _require_nonempty(self.projection_law, field="operator projection_law")
        _require_nonempty(self.aggregation_law, field="operator aggregation_law")
        _require_nonempty(self.parameter_sharing, field="operator parameter_sharing")
        _require_nonempty(self.shape_law, field="operator shape_law")
        _require_nonempty(self.cost_law, field="operator cost_law")
        _require_unique(self.syntax_aliases, field="operator syntax_aliases")
        _require_unique(
            (role.name for role in self.parameter_roles),
            field=f"{self.kind} parameter roles",
        )

    def descriptor_record(self, *, aliases: tuple[str, ...] | None = None) -> dict[str, object]:
        """Return the descriptor record consumed by model inspections and console data."""

        return {
            "kind": self.kind,
            "tensor_relation": self.tensor_relation,
            "state": self.state,
            "support": self.support,
            "projection_law": self.projection_law,
            "aggregation_law": self.aggregation_law,
            "parameter_sharing": self.parameter_sharing,
            "shape_law": self.shape_law,
            "cost_law": self.cost_law,
            "aliases": list(self.syntax_aliases if aliases is None else aliases),
        }

    def operator_record(self) -> dict[str, object]:
        """Return the console-facing operator vocabulary record."""

        return {
            "kind": self.kind,
            "display_name": self.display_name,
            "descriptor": self.descriptor_record(),
            "syntax_aliases": list(self.syntax_aliases),
            "parameter_roles": [role.to_record() for role in self.parameter_roles],
        }


@dataclass(frozen=True, slots=True)
class DescriptorAxisValue:
    """One declared value of an operator descriptor axis."""

    value: str
    display_name: str

    def __post_init__(self) -> None:
        _require_nonempty(self.value, field="descriptor axis value")
        _require_nonempty(self.display_name, field="descriptor axis value display_name")

    def to_record(self) -> dict[str, str]:
        """Return a canonical record for this descriptor-axis value."""

        return {
            "value": self.value,
            "display_name": self.display_name,
        }


@dataclass(frozen=True, slots=True)
class DescriptorAxis:
    """Declared display metadata for one semantic descriptor axis."""

    name: str
    display_name: str
    values: tuple[DescriptorAxisValue, ...]

    def __post_init__(self) -> None:
        _require_nonempty(self.name, field="descriptor axis name")
        _require_nonempty(self.display_name, field="descriptor axis display_name")
        _require_unique((value.value for value in self.values), field=f"{self.name} values")

    def descriptor_record(self) -> dict[str, str]:
        """Return the console-facing axis descriptor record."""

        return {
            "name": self.name,
            "display_name": self.display_name,
        }

    def value_records(self) -> list[dict[str, str]]:
        """Return ordered console-facing records for declared axis values."""

        return [value.to_record() for value in self.values]


@dataclass(frozen=True, slots=True)
class SemanticCoordinateDescriptor:
    """Declared metadata for one derived model-operator coordinate."""

    name: str
    display_name: str
    value_kind: str

    def __post_init__(self) -> None:
        _require_nonempty(self.name, field="coordinate name")
        _require_nonempty(self.display_name, field="coordinate display_name")
        _require_nonempty(self.value_kind, field="coordinate value_kind")

    def to_record(self) -> dict[str, str]:
        """Return a canonical coordinate descriptor record."""

        return {
            "name": self.name,
            "display_name": self.display_name,
            "value_kind": self.value_kind,
        }


@dataclass(frozen=True, slots=True)
class ModelOperatorSemanticRegistry:
    """Declared operator semantics and their console vocabulary metadata."""

    operators: tuple[ModelOperatorSemantic, ...]
    descriptor_axes: tuple[DescriptorAxis, ...]
    coordinate_descriptors: tuple[SemanticCoordinateDescriptor, ...]

    def __post_init__(self) -> None:
        _require_unique((operator.kind for operator in self.operators), field="operator kinds")
        _require_unique(
            (
                alias
                for operator in self.operators
                for alias in (operator.kind, *operator.syntax_aliases)
            ),
            field="operator public names",
        )
        _require_unique(
            (alias for operator in self.operators for alias in operator.syntax_aliases),
            field="operator syntax aliases",
        )
        _require_unique((axis.name for axis in self.descriptor_axes), field="descriptor axes")
        _require_unique(
            (coordinate.name for coordinate in self.coordinate_descriptors),
            field="coordinate descriptors",
        )

    def semantic_for_alias(self, alias: str) -> ModelOperatorSemantic | None:
        """Return the semantic declaration for a public architecture name or syntax alias."""

        return self._semantic_by_alias().get(alias)

    def operator_records(self) -> list[dict[str, object]]:
        """Return ordered console-facing operator records."""

        return [operator.operator_record() for operator in self.operators]

    def syntax_alias_records(self) -> list[dict[str, object]]:
        """Return ordered public syntax alias records."""

        return [
            {
                "alias": alias,
                "operator_kind": operator.kind,
                "display_name": operator.display_name,
                "specialization": operator.descriptor_record(aliases=(alias,)),
            }
            for operator in self.operators
            for alias in operator.syntax_aliases
        ]

    def descriptor_axis_descriptor_records(self) -> list[dict[str, str]]:
        """Return ordered descriptor-axis metadata records."""

        return [axis.descriptor_record() for axis in self.descriptor_axes]

    def descriptor_axis_records(self) -> dict[str, list[dict[str, str]]]:
        """Return ordered descriptor-axis value records keyed by axis name."""

        return {axis.name: axis.value_records() for axis in self.descriptor_axes}

    def coordinate_descriptor_records(self) -> list[dict[str, str]]:
        """Return ordered semantic-coordinate metadata records."""

        return [coordinate.to_record() for coordinate in self.coordinate_descriptors]

    def _semantic_by_alias(self) -> dict[str, ModelOperatorSemantic]:
        return {
            name: operator
            for operator in self.operators
            for name in (operator.kind, *operator.syntax_aliases)
        }


def model_operator_semantic_registry() -> ModelOperatorSemanticRegistry:
    """Return the declared public model-operator semantic registry."""

    return _model_operator_semantic_registry


def _axis_values(values: Mapping[str, str]) -> tuple[DescriptorAxisValue, ...]:
    return tuple(
        DescriptorAxisValue(value=value, display_name=display_name)
        for value, display_name in sorted(values.items())
    )


def _values_from_operator_field(field: str) -> tuple[DescriptorAxisValue, ...]:
    return _axis_values(
        {
            str(getattr(operator, field)): _title_from_token(str(getattr(operator, field)))
            for operator in _operator_semantics
        }
    )


def _title_from_token(value: str) -> str:
    return " ".join(part.capitalize() for part in value.replace("_", "-").split("-"))


def _require_nonempty(value: str, *, field: str) -> None:
    if not value:
        raise OperatorSemanticValidationError(f"{field} must be nonempty")


def _require_unique(values: Iterable[str], *, field: str) -> None:
    sequence = tuple(values)
    if any(not value for value in sequence):
        raise OperatorSemanticValidationError(f"{field} must contain nonempty values")
    if len(set(sequence)) != len(sequence):
        raise OperatorSemanticValidationError(f"{field} must be unique")


_operator_semantics = (
    ModelOperatorSemantic(
        kind="local-aggregation",
        display_name="Local aggregation",
        tensor_relation="aggregation",
        state="fixed",
        support="local-window",
        projection_law="equal-output-partition",
        aggregation_law="mean",
        parameter_sharing="none",
        shape_law="preserve-prefix-replace-trailing-axes",
        cost_law="input-elements",
        syntax_aliases=("adaptive-pooling",),
        parameter_roles=(
            ModelOperatorParameterRole(
                name="dimension",
                display_name="Support rank",
                description="number of trailing axes aggregated",
            ),
            ModelOperatorParameterRole(
                name="out_length",
                display_name="Output length",
                description="fixed extent of a one-dimensional output support axis",
            ),
            ModelOperatorParameterRole(
                name="out_height",
                display_name="Output height",
                description="fixed extent of the first aggregated output support axis",
            ),
            ModelOperatorParameterRole(
                name="out_width",
                display_name="Output width",
                description="fixed extent of the second aggregated output support axis",
            ),
            ModelOperatorParameterRole(
                name="size",
                display_name="Output support size",
                description="square output support extent for adaptive-pooling syntax",
            ),
        ),
    ),
    ModelOperatorSemantic(
        kind="local-affine",
        display_name="Local affine",
        tensor_relation="affine",
        state="learned",
        support="local-window",
        projection_law="sliding-window",
        aggregation_law="weighted-sum-plus-bias",
        parameter_sharing="shared-local-window",
        shape_law="preserve-prefix-local-window",
        cost_law="local-window-multiply-add",
        syntax_aliases=("convolution",),
        parameter_roles=(
            ModelOperatorParameterRole(
                name="dimension",
                display_name="Support rank",
                description="number of trailing axes in each local support window",
            ),
            ModelOperatorParameterRole(
                name="size",
                display_name="Support size",
                description="extent of each local support axis",
            ),
            ModelOperatorParameterRole(
                name="out_channels",
                display_name="Output channels",
                description="number of learned output coordinates per local window",
            ),
            ModelOperatorParameterRole(
                name="stride",
                display_name="Stride",
                description="step size between adjacent local windows",
            ),
            ModelOperatorParameterRole(
                name="padding",
                display_name="Padding",
                description="zero padding on each local support axis",
                value_kind="nonnegative-integer",
            ),
            ModelOperatorParameterRole(
                name="padding_mode",
                display_name="Padding mode",
                description="boundary convention for padded local windows",
                value_kind="padding-mode",
            ),
        ),
    ),
    ModelOperatorSemantic(
        kind="fixed-support-affine",
        display_name="Fixed support affine",
        tensor_relation="affine",
        state="learned",
        support="global",
        projection_law="adaptive-support-partition",
        aggregation_law="mean-then-weighted-sum-plus-bias",
        parameter_sharing="shared-support-position",
        shape_law="fixed-support-affine",
        cost_law="adaptive-support-pointwise-affine",
        syntax_aliases=(),
        parameter_roles=(
            ModelOperatorParameterRole(
                name="dimension",
                display_name="Support rank",
                description="number of trailing support axes projected to fixed extent",
            ),
            ModelOperatorParameterRole(
                name="out_length",
                display_name="Output length",
                description="fixed extent of a one-dimensional fixed support axis",
            ),
            ModelOperatorParameterRole(
                name="out_height",
                display_name="Output height",
                description="fixed extent of the first output support axis",
            ),
            ModelOperatorParameterRole(
                name="out_width",
                display_name="Output width",
                description="fixed extent of the second output support axis",
            ),
            ModelOperatorParameterRole(
                name="out_channels",
                display_name="Output channels",
                description="number of learned output coordinates per fixed support position",
            ),
        ),
    ),
    ModelOperatorSemantic(
        kind="rectified-linear-activation",
        display_name="Rectified linear activation",
        tensor_relation="nonlinear",
        state="fixed",
        support="pointwise",
        projection_law="pointwise-coordinate-map",
        aggregation_law="thresholded-identity",
        parameter_sharing="none",
        shape_law="preserve-input-shape",
        cost_law="input-elements",
        syntax_aliases=("relu",),
    ),
    ModelOperatorSemantic(
        kind="rank-collapse",
        display_name="Rank collapse",
        tensor_relation="shape-transform",
        state="fixed",
        support="rank-collapsing",
        projection_law="row-major-axis-concatenation",
        aggregation_law="none",
        parameter_sharing="none",
        shape_law="product-of-input-axes",
        cost_law="zero-arithmetic",
        syntax_aliases=("flatten",),
    ),
    ModelOperatorSemantic(
        kind="affine-readout",
        display_name="Affine readout",
        tensor_relation="affine",
        state="learned",
        support="global",
        projection_law="full-input-support",
        aggregation_law="weighted-sum-plus-bias",
        parameter_sharing="none",
        shape_law="rank-1-output",
        cost_law="multiply-add-per-input-output-pair",
        syntax_aliases=("dense",),
        parameter_roles=(
            ModelOperatorParameterRole(
                name="out",
                display_name="Output coordinates",
                description="rank-1 output extent",
            ),
        ),
    ),
)

_descriptor_axes = (
    DescriptorAxis(
        name="tensor_relation",
        display_name="Tensor relation",
        values=_axis_values(
            {
                "affine": "Affine",
                "aggregation": "Aggregation",
                "identity": "Identity",
                "nonlinear": "Nonlinear",
                "shape-transform": "Shape transform",
            }
        ),
    ),
    DescriptorAxis(
        name="state",
        display_name="State",
        values=_axis_values({"fixed": "Fixed", "learned": "Learned"}),
    ),
    DescriptorAxis(
        name="support",
        display_name="Support",
        values=_axis_values(
            {
                "global": "Global",
                "local-window": "Local window",
                "pointwise": "Pointwise",
                "rank-collapsing": "Rank collapsing",
            }
        ),
    ),
    DescriptorAxis(
        name="projection_law",
        display_name="Projection law",
        values=_values_from_operator_field("projection_law"),
    ),
    DescriptorAxis(
        name="aggregation_law",
        display_name="Aggregation law",
        values=_values_from_operator_field("aggregation_law"),
    ),
    DescriptorAxis(
        name="parameter_sharing",
        display_name="Parameter sharing",
        values=_values_from_operator_field("parameter_sharing"),
    ),
    DescriptorAxis(
        name="shape_law",
        display_name="Shape law",
        values=_values_from_operator_field("shape_law"),
    ),
    DescriptorAxis(
        name="cost_law",
        display_name="Cost law",
        values=_values_from_operator_field("cost_law"),
    ),
)

_coordinate_descriptors = (
    SemanticCoordinateDescriptor(
        name="input.rank",
        display_name="Input rank",
        value_kind="integer",
    ),
    SemanticCoordinateDescriptor(
        name="output.rank",
        display_name="Output rank",
        value_kind="integer",
    ),
    SemanticCoordinateDescriptor(
        name="operator.count",
        display_name="Operator count",
        value_kind="integer",
    ),
    SemanticCoordinateDescriptor(
        name="operator.{index}.kind",
        display_name="Operator kind",
        value_kind="operator-kind",
    ),
    SemanticCoordinateDescriptor(
        name="operator.{index}.tensor_relation",
        display_name="Tensor relation",
        value_kind="descriptor-axis",
    ),
    SemanticCoordinateDescriptor(
        name="operator.{index}.support",
        display_name="Support",
        value_kind="descriptor-axis",
    ),
    SemanticCoordinateDescriptor(
        name="operator.{index}.local_support_dimension",
        display_name="Local support dimension",
        value_kind="integer",
    ),
    SemanticCoordinateDescriptor(
        name="operator.{index}.local_support_size",
        display_name="Local support size",
        value_kind="integer",
    ),
    SemanticCoordinateDescriptor(
        name="operator.{index}.local_stride",
        display_name="Local stride",
        value_kind="integer",
    ),
    SemanticCoordinateDescriptor(
        name="operator.{index}.local_padding",
        display_name="Local padding",
        value_kind="integer",
    ),
    SemanticCoordinateDescriptor(
        name="operator.{index}.output_channels",
        display_name="Output channels",
        value_kind="integer",
    ),
    SemanticCoordinateDescriptor(
        name="operator.{index}.output_height",
        display_name="Output height",
        value_kind="integer",
    ),
    SemanticCoordinateDescriptor(
        name="operator.{index}.output_width",
        display_name="Output width",
        value_kind="integer",
    ),
    SemanticCoordinateDescriptor(
        name="operator.{index}.output_count",
        display_name="Output count",
        value_kind="integer",
    ),
    SemanticCoordinateDescriptor(
        name="resource.parameter_count",
        display_name="Parameter count",
        value_kind="integer",
    ),
)

_model_operator_semantic_registry = ModelOperatorSemanticRegistry(
    operators=_operator_semantics,
    descriptor_axes=_descriptor_axes,
    coordinate_descriptors=_coordinate_descriptors,
)
