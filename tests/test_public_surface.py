import ast
import importlib
import inspect
import pkgutil
import re
from pathlib import Path
from types import ModuleType

import leibniz

_pascal_case = re.compile(r"^[A-Z][A-Za-z0-9]*$")
_snake_case = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
_answer_terminology = re.compile(r"answer|Answer")
_module_scope_upper_snake = re.compile(r"_?[A-Z][A-Z0-9_]*$")


def test_leibniz_modules_export_pythonic_public_names() -> None:
    for module_name in _leibniz_module_names():
        module = importlib.import_module(module_name)
        exported_names = _public_names(module)
        offenders = tuple(_non_pythonic_exported_names(module, exported_names))

        assert exported_names, f"{module_name} must declare at least one public name"
        assert offenders == (), (
            f"{module_name} exports non-pythonic public names: {offenders}"
        )


def test_leibniz_public_class_methods_use_snake_case() -> None:
    for module_name in _leibniz_module_names():
        module = importlib.import_module(module_name)
        offenders = _non_snake_case_public_methods(module)

        assert offenders == (), (
            f"{module_name} defines non-snake-case public methods: {offenders}"
        )


def test_leibniz_public_module_definitions_are_explicitly_exported() -> None:
    for module_name in _leibniz_module_names():
        module = importlib.import_module(module_name)
        exported = set(_public_names(module))
        missing = tuple(name for name in _defined_public_names(module) if name not in exported)

        assert missing == (), (
            f"{module_name} defines public names not listed in __all__: {missing}"
        )


def test_leibniz_source_uses_outcome_terminology() -> None:
    source_root = Path(leibniz.__file__).parent
    offenders = tuple(
        path.relative_to(source_root)
        for path in sorted(source_root.rglob("*.py"))
        if _answer_terminology.search(path.read_text(encoding="utf-8"))
    )

    assert offenders == ()


def test_python_source_avoids_module_scope_upper_snake_names() -> None:
    source_roots = (Path(leibniz.__file__).parent, Path(__file__).parent)
    offenders = tuple(
        f"{path.relative_to(root)}:{name}"
        for root in source_roots
        for path in sorted(root.rglob("*.py"))
        for name in _module_scope_assignment_names(path)
        if _module_scope_upper_snake.fullmatch(name)
    )

    assert offenders == ()


def test_public_modules_do_not_import_private_leibniz_modules() -> None:
    offenders = tuple(
        f"{module_name}:{imported_name}"
        for module_name in _leibniz_module_names()
        for imported_name in _private_leibniz_imports(module_name)
    )

    assert offenders == ()


def _leibniz_module_names() -> tuple[str, ...]:
    return tuple(
        module_info.name
        for module_info in pkgutil.walk_packages(leibniz.__path__, prefix=f"{leibniz.__name__}.")
        if not module_info.ispkg and not module_info.name.rsplit(".", 1)[-1].startswith("_")
    )


def _private_leibniz_imports(module_name: str) -> tuple[str, ...]:
    module = importlib.import_module(module_name)
    source_path = getattr(module, "__file__", None)
    assert source_path is not None, f"{module.__name__} must have a source file"
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if _is_private_leibniz_module(node.module) and not _allowed_private_import(
                module_name,
                node.module,
            ):
                imports.append(node.module)
        elif isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
                if _is_private_leibniz_module(alias.name)
                and not _allowed_private_import(module_name, alias.name)
            )
    return tuple(sorted(imports))


def _allowed_private_import(module_name: str, imported_name: str) -> bool:
    return module_name == "leibniz.documents" and imported_name == "leibniz._formats._json"


def _is_private_leibniz_module(module_name: str) -> bool:
    if not module_name.startswith("leibniz."):
        return False
    return any(part.startswith("_") for part in module_name.split(".")[1:])


def _public_names(module: ModuleType) -> tuple[str, ...]:
    exported = getattr(module, "__all__", None)
    assert exported is not None, f"{module.__name__} must declare __all__"
    return tuple(str(name) for name in exported)


def _non_pythonic_exported_names(
    module: ModuleType,
    exported_names: tuple[str, ...],
) -> tuple[str, ...]:
    offenders: list[str] = []
    for name in exported_names:
        value = getattr(module, name)
        if inspect.isfunction(value):
            if _snake_case.fullmatch(name) is None:
                offenders.append(name)
            continue
        if _pascal_case.fullmatch(name) is None:
            offenders.append(name)
    return tuple(offenders)


def _non_snake_case_public_methods(module: ModuleType) -> tuple[str, ...]:
    source_path = getattr(module, "__file__", None)
    assert source_path is not None, f"{module.__name__} must have a source file"
    tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))

    offenders: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name.startswith("_"):
            continue
        for item in node.body:
            if (
                isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                and not item.name.startswith("_")
                and _snake_case.fullmatch(item.name) is None
            ):
                offenders.append(f"{node.name}.{item.name}")
    return tuple(offenders)


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


def _module_scope_assignment_names(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            names.extend(_assignment_names(node.target))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.extend(_assignment_names(target))
    return tuple(names)


def _public_assignment_names(target: ast.expr) -> tuple[str, ...]:
    return tuple(name for name in _assignment_names(target) if not name.startswith("_"))


def _assignment_names(target: ast.expr) -> tuple[str, ...]:
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, ast.Tuple | ast.List):
        names: list[str] = []
        for item in target.elts:
            names.extend(_assignment_names(item))
        return tuple(names)
    return ()
