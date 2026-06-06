import tomllib
from pathlib import Path

_root = Path(__file__).resolve().parents[1]


def test_miniforge_environment_declares_development_toolchain() -> None:
    lines = (_root / "environment.yml").read_text(encoding="utf-8").splitlines()

    assert "name: leibniz-dev" in lines
    assert "  - pytorch" in lines
    assert "  - nvidia" in lines
    assert "  - conda-forge" in lines
    assert "  - python=3.12" in lines
    assert "  - nodejs=24" in lines
    assert "  - pip>=24" in lines
    assert "  - numpy" in lines
    assert '      - "-e .[dev]"' in lines


def test_environment_scripts_use_repo_local_miniforge() -> None:
    setup_script = (_root / "scripts" / "setup_environment.sh").read_text(encoding="utf-8")
    activate_script = (_root / "scripts" / "activate_environment.sh").read_text(
        encoding="utf-8"
    )

    assert "MINIFORGE_ROOT=\"${LEIBNIZ_MINIFORGE_ROOT:-$LEIBNIZ_ROOT/.miniforge}\"" in setup_script
    assert "ENV_NAME=\"${LEIBNIZ_ENV_NAME:-leibniz-dev}\"" in setup_script
    assert "conda-forge/miniforge/releases/latest/download" in setup_script
    assert "--environment-specifier cep-24" in setup_script
    assert "-n \"$ENV_NAME\"" in setup_script
    assert "-f environment.yml" in setup_script
    assert "--prune" in setup_script
    assert "CONSOLE_WEB_ROOT=\"$LEIBNIZ_ROOT/src/leibniz/console/_web_src\"" in setup_script
    assert "Syncing console npm dependencies" in setup_script
    assert "\"$CONDA_BIN\" run -n \"$ENV_NAME\" npm ci" in setup_script
    assert "This script must be sourced" in activate_script
    assert "source \"$CONDA_SH\"" in activate_script
    assert "conda activate \"$ENV_NAME\"" in activate_script


def test_ci_uses_miniforge_environment_file() -> None:
    workflow = (_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "conda-incubator/setup-miniconda@v3" in workflow
    assert "miniforge-version: latest" in workflow
    assert "activate-environment: leibniz-dev" in workflow
    assert "environment-file: environment.yml" in workflow
    assert "name: Tests" in workflow
    assert "name: Lint" in workflow
    assert "name: Type check" in workflow
    assert "name: Repository policy" in workflow
    assert "name: Console build and browser tests" in workflow
    assert "name: Build and import package" in workflow
    assert "actions/setup-node@v4" in workflow
    assert "npm ci" in workflow
    assert "npx playwright install --with-deps chromium" in workflow
    assert "npm test" in workflow
    assert "python -m build --no-isolation" in workflow
    assert "actions/setup-python" not in workflow


def test_miniforge_environment_declares_linux_cuda_pytorch_selector() -> None:
    lines = (_root / "environment.yml").read_text(encoding="utf-8").splitlines()

    assert "  - sel(linux): pytorch-cuda=12.4" in lines
    assert "      - \"triton; platform_system == 'Linux'\"" in lines


def test_main_branch_triggers_pages_rebuild() -> None:
    workflow = (_root / ".github" / "workflows" / "pages-dispatch.yml").read_text(
        encoding="utf-8"
    )

    assert "name: Trigger Pages Rebuild" in workflow
    assert "branches: [main]" in workflow
    assert "LEIBNIZ_PAGES_DISPATCH_TOKEN" in workflow
    assert "repos/community-science/community-science.github.io/dispatches" in workflow
    assert "event_type=leibniz-updated" in workflow
    assert "client_payload[sha]" in workflow


def test_development_extra_contains_no_isolation_build_backend() -> None:
    pyproject = tomllib.loads((_root / "pyproject.toml").read_text(encoding="utf-8"))

    dev_dependencies = pyproject["project"]["optional-dependencies"]["dev"]

    assert "build>=1.2" in dev_dependencies
    assert "hatchling>=1.25" in dev_dependencies
