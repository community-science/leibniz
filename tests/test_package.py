from importlib.metadata import version

import leibniz


def test_package_imports() -> None:
    assert leibniz.__version__ == version("leibniz")
