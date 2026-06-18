#!/usr/bin/env bash
# Bootstrap the repository-local Leibniz development environment.
set -euo pipefail

LEIBNIZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINIFORGE_ROOT="${LEIBNIZ_MINIFORGE_ROOT:-$LEIBNIZ_ROOT/.miniforge}"
ENV_NAME="${LEIBNIZ_ENV_NAME:-leibniz-dev}"
CONDA_BIN="$MINIFORGE_ROOT/bin/conda"
CONSOLE_WEB_ROOT="$LEIBNIZ_ROOT/src/leibniz/console/web"

OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS/$ARCH" in
    Darwin/arm64) MINIFORGE_PLATFORM="MacOSX-arm64" ;;
    Darwin/x86_64) MINIFORGE_PLATFORM="MacOSX-x86_64" ;;
    Linux/aarch64) MINIFORGE_PLATFORM="Linux-aarch64" ;;
    Linux/x86_64) MINIFORGE_PLATFORM="Linux-x86_64" ;;
    *)
        echo "Unsupported platform for Miniforge: $OS/$ARCH" >&2
        exit 1
        ;;
esac

if [[ ! -x "$CONDA_BIN" ]]; then
    INSTALLER="$(mktemp).sh"
    INSTALLER_URL="https://github.com/conda-forge/miniforge/releases/latest/download"
    echo "Downloading Miniforge3 for $MINIFORGE_PLATFORM"
    curl -fsSL "$INSTALLER_URL/Miniforge3-${MINIFORGE_PLATFORM}.sh" -o "$INSTALLER"
    bash "$INSTALLER" -b -p "$MINIFORGE_ROOT"
    rm -f "$INSTALLER"
else
    echo "Using Miniforge at $MINIFORGE_ROOT"
fi

if "$CONDA_BIN" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Updating conda environment $ENV_NAME from environment.yml"
    (
        cd "$LEIBNIZ_ROOT"
        "$CONDA_BIN" env update \
            --environment-specifier cep-24 \
            -n "$ENV_NAME" \
            -f environment.yml \
            --prune
    )
else
    echo "Creating conda environment $ENV_NAME from environment.yml"
    (
        cd "$LEIBNIZ_ROOT"
        "$CONDA_BIN" env create \
            --environment-specifier cep-24 \
            -n "$ENV_NAME" \
            -f environment.yml
    )
fi

echo "Syncing console npm dependencies"
(
    cd "$CONSOLE_WEB_ROOT"
    "$CONDA_BIN" run -n "$ENV_NAME" npm ci
)

echo "Environment ready. Run: source scripts/activate_environment.sh"
