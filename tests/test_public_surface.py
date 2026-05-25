import ast
import importlib
import pkgutil
import re
from pathlib import Path
from types import ModuleType

import leibniz

CAMEL_CASE = re.compile(r"^[A-Z][A-Za-z0-9]*$")


def test_leibniz_modules_export_only_camel_case_public_names() -> None:
    for module_name in _leibniz_module_names():
        module = importlib.import_module(module_name)
        public_names = _public_names(module)
        non_camel_names = tuple(
            name for name in public_names if CAMEL_CASE.fullmatch(name) is None
        )

        assert public_names, f"{module_name} must declare at least one public name"
        assert non_camel_names == (), (
            f"{module_name} exports non-CamelCase public names: {non_camel_names}"
        )


def test_leibniz_public_module_definitions_are_explicitly_exported() -> None:
    for module_name in _leibniz_module_names():
        module = importlib.import_module(module_name)
        exported = set(_public_names(module))
        missing = tuple(name for name in _defined_public_names(module) if name not in exported)

        assert missing == (), (
            f"{module_name} defines public names not listed in __all__: {missing}"
        )


def _leibniz_module_names() -> tuple[str, ...]:
    return tuple(
        module_info.name
        for module_info in pkgutil.walk_packages(leibniz.__path__, prefix=f"{leibniz.__name__}.")
        if not module_info.ispkg and not module_info.name.rsplit(".", 1)[-1].startswith("_")
    )


def _public_names(module: ModuleType) -> tuple[str, ...]:
    exported = getattr(module, "__all__", None)
    assert exported is not None, f"{module.__name__} must declare __all__"
    return tuple(str(name) for name in exported)


def _defined_public_names(module: ModuleType) -> tuple[str, ...]:
    source_path = getattr(module, "__file__", None)
    assert source_path is not None, f"{module.__name__} must have a source file"
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))

    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                names.append(node.name)
        elif isinstance(node, ast.AnnAssign):
            names.extend(_public_assignment_names(node.target))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.extend(_public_assignment_names(target))
    return tuple(names)


def _public_assignment_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name) and not target.id.startswith("_"):
        return (target.id,)
    if isinstance(target, ast.Tuple | ast.List):
        names: list[str] = []
        for item in target.elts:
            names.extend(_public_assignment_names(item))
        return tuple(names)
    return ()
