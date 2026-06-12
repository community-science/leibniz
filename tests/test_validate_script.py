import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_repository_root = Path(__file__).parents[1]
_script_path = _repository_root / "scripts" / "validate.py"


def _load_validate_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_script", _script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validation_commands_match_routine_ci_gates() -> None:
    module = _load_validate_module()

    checks = module.validation_checks(
        repository_root=_repository_root,
        include_console=True,
    )
    commands = module.validation_commands(
        repository_root=_repository_root,
        include_console=True,
    )

    assert [check.name for check in checks] == [
        "tests",
        "lint",
        "type",
        "repository-policy",
        "package",
        "console",
    ]
    assert [command.label for command in commands] == [
        "Python tests",
        "Ruff lint",
        "Pyright type check",
        "Repository policy",
        "Build wheel and source distribution",
        "Install built wheel",
        "Import installed package",
        "Console build and browser tests",
    ]
    assert [command.argv[1:] for command in commands[:4]] == [
        ("-m", "pytest"),
        ("-m", "ruff", "check", "."),
        ("-m", "pyright"),
        ("-m", "leibniz._repository_policy", "."),
    ]
    assert commands[4].argv[1:] == ("-m", "build", "--no-isolation")
    assert commands[5].argv[1:] == (
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "__wheel__",
    )
    assert commands[6].argv[1:] == ("-c", "import leibniz; print(leibniz.__version__)")
    assert commands[-1].argv == ("npm", "test")
    assert commands[-1].cwd == _repository_root / "src" / "leibniz" / "console" / "_web_src"


def test_validation_commands_can_omit_console_gate() -> None:
    module = _load_validate_module()

    commands = module.validation_commands(
        repository_root=_repository_root,
        include_console=False,
    )

    assert [command.label for command in commands] == [
        "Python tests",
        "Ruff lint",
        "Pyright type check",
        "Repository policy",
        "Build wheel and source distribution",
        "Install built wheel",
        "Import installed package",
    ]
