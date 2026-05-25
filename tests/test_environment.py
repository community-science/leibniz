import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_miniforge_environment_declares_development_toolchain() -> None:
    lines = (ROOT / "environment.yml").read_text(encoding="utf-8").splitlines()

    assert "name: leibniz-dev" in lines
    assert "  - conda-forge" in lines
    assert "  - python=3.12" in lines
    assert "  - pip>=24" in lines
    assert '      - "-e .[dev]"' in lines


def test_environment_scripts_use_repo_local_miniforge() -> None:
    setup_script = (ROOT / "scripts" / "setup_environment.sh").read_text(encoding="utf-8")
    activate_script = (ROOT / "scripts" / "activate_environment.sh").read_text(
        encoding="utf-8"
    )

    assert "MINIFORGE_ROOT=\"${LEIBNIZ_MINIFORGE_ROOT:-$LEIBNIZ_ROOT/.miniforge}\"" in setup_script
    assert "ENV_NAME=\"${LEIBNIZ_ENV_NAME:-leibniz-dev}\"" in setup_script
    assert "conda-forge/miniforge/releases/latest/download" in setup_script
    assert "env create -n \"$ENV_NAME\" -f environment.yml" in setup_script
    assert "This script must be sourced" in activate_script
    assert "source \"$CONDA_SH\"" in activate_script
    assert "conda activate \"$ENV_NAME\"" in activate_script


def test_ci_uses_miniforge_environment_file() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "conda-incubator/setup-miniconda@v3" in workflow
    assert "miniforge-version: latest" in workflow
    assert "activate-environment: leibniz-dev" in workflow
    assert "environment-file: environment.yml" in workflow
    assert "name: Tests" in workflow
    assert "name: Lint" in workflow
    assert "name: Type check" in workflow
    assert "name: Repository policy" in workflow
    assert "name: Build and import package" in workflow
    assert "python -m build --no-isolation" in workflow
    assert "actions/setup-python" not in workflow


def test_development_extra_contains_no_isolation_build_backend() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]

    assert "build>=1.2" in dev_dependencies
    assert "hatchling>=1.25" in dev_dependencies
