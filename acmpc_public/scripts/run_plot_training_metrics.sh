#!/usr/bin/env bash
set -euo pipefail

# Plot AC-MPC Gym training metrics from a saved run directory.
#
# Default command, plots newest run under runs/acmpc_gym:
#   bash scripts/run_plot_training_metrics.sh
#
# Useful overrides:
#   RUN_DIR=runs/acmpc_gym/my_run bash scripts/run_plot_training_metrics.sh
#   WINDOW=100 bash scripts/run_plot_training_metrics.sh
#   OUTPUT_DIR=tmp/plots bash scripts/run_plot_training_metrics.sh
#   SHOW=1 bash scripts/run_plot_training_metrics.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV_NAME="${CONDA_ENV_NAME:-acmpc}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
else
  echo "conda was not found in PATH. Activate '${CONDA_ENV_NAME}' before running this script." >&2
  exit 1
fi

ARGS=(
  --window "${WINDOW:-50}"
  --dpi "${DPI:-160}"
)

if [[ -n "${RUN_DIR:-}" ]]; then
  ARGS+=(--run-dir "${RUN_DIR}")
else
  ARGS+=(--latest)
fi

if [[ -n "${OUTPUT_DIR:-}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi

if [[ "${SHOW:-0}" == "1" ]]; then
  ARGS+=(--show)
fi

echo "Running from: ${REPO_ROOT}"
echo "Conda env: ${CONDA_ENV_NAME}"
echo "Command: python scripts/plot_training_metrics.py ${ARGS[*]}"

python scripts/plot_training_metrics.py "${ARGS[@]}"
