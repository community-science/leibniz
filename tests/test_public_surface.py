import importlib
import pkgutil
import re
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


def _leibniz_module_names() -> tuple[str, ...]:
    return tuple(
        module_info.name
        for module_info in pkgutil.walk_packages(leibniz.__path__, prefix=f"{leibniz.__name__}.")
        if not module_info.ispkg
    )


def _public_names(module: ModuleType) -> tuple[str, ...]:
    exported = getattr(module, "__all__", None)
    assert exported is not None, f"{module.__name__} must declare __all__"
    return tuple(str(name) for name in exported)
