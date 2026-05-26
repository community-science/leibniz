import importlib


def test_console_package_imports() -> None:
    module = importlib.import_module("leibniz.console")

    assert module.__name__ == "leibniz.console"


def test_console_package_does_not_expose_runtime_static_helpers() -> None:
    module = importlib.import_module("leibniz.console")

    assert "console_static_root" not in vars(module)
    assert "console_static_exists" not in vars(module)
