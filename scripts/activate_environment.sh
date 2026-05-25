#!/usr/bin/env bash
# Source this file to activate the repository-local Leibniz conda environment.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced so it can activate conda in the current shell."
    echo "Run: source scripts/activate_environment.sh"
    exit 1
fi

LEIBNIZ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MINIFORGE_ROOT="${LEIBNIZ_MINIFORGE_ROOT:-$LEIBNIZ_ROOT/.miniforge}"
ENV_NAME="${LEIBNIZ_ENV_NAME:-leibniz-dev}"
CONDA_BIN="$MINIFORGE_ROOT/bin/conda"
CONDA_SH="$MINIFORGE_ROOT/etc/profile.d/conda.sh"

if [[ ! -x "$CONDA_BIN" ]] \
        || ! "$CONDA_BIN" env list 2>/dev/null | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    "$LEIBNIZ_ROOT/scripts/setup_environment.sh" || return
fi

if [[ ! -f "$CONDA_SH" ]]; then
    echo "Expected conda activation hook at $CONDA_SH" >&2
    return 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH" || return
conda activate "$ENV_NAME" || return
cd "$LEIBNIZ_ROOT" || return

export LEIBNIZ_ROOT
export LEIBNIZ_PYTHON
LEIBNIZ_PYTHON="$(command -v python)"

echo "Activated $ENV_NAME at $CONDA_PREFIX"
