#!/bin/bash -l
set -euo pipefail

if command -v module >/dev/null 2>&1; then
  module load cuda/11.8
fi
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME:-pyronn_diffusion}"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || command -v python3)}"
if [ -z "${PYTHON_BIN}" ]; then
  echo "[Error] Neither python nor python3 found in PATH." >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results}"
OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/eval_models}"

# Optional: space-separated model directory names under OUT_ROOT.
# If empty, aggregate all first-level subdirectories that contain seed_* folders.
MODEL_NAMES="${MODEL_NAMES:-}"

aggregate_one() {
  local name="$1"
  local input_root="${OUT_ROOT}/${name}"
  local output_json="${input_root}/seed_aggregate.json"

  if [ ! -d "${input_root}" ]; then
    echo "[Skip] Missing model directory: ${input_root}"
    return 0
  fi

  if ! find "${input_root}" -maxdepth 1 -type d -name 'seed_*' | grep -q .; then
    echo "[Skip] No seed_* directories under: ${input_root}"
    return 0
  fi

  echo "[Aggregate] ${input_root}"
  "${PYTHON_BIN}" -u scripts/aggregate_eval_seeds.py \
    --input_root "${input_root}" \
    --output "${output_json}"
}

if [ -n "${MODEL_NAMES}" ]; then
  for name in ${MODEL_NAMES}; do
    aggregate_one "${name}"
  done
else
  mapfile -t model_dirs < <(find "${OUT_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
  if [ "${#model_dirs[@]}" -eq 0 ]; then
    echo "[Error] No model directories found under ${OUT_ROOT}" >&2
    exit 1
  fi
  for name in "${model_dirs[@]}"; do
    aggregate_one "${name}"
  done
fi
