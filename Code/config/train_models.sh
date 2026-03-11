#!/bin/bash -l
#SBATCH --job-name=train_models
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --signal=USR1@120
#SBATCH --output=logs/train_models_%A_%a.out
#SBATCH --error=logs/train_models_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --array=0-99

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
mkdir -p logs

TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-${ROOT_DIR}/data/train_val_sinograms}"
BASE_CONFIG="${BASE_CONFIG:-config/base.yaml}"

# CONFIG_LIST format: comma-separated config paths (relative to repo root).
# Example: config/baseline_cold_diffusion_k320.yaml,config/baseline_ddpm_k320.yaml
if [ -z "${CONFIG_LIST:-}" ]; then
  CONFIG_LIST="config/exp3_timesteps_50.yaml,config/exp3_timesteps_150.yaml"
fi

# Optional seed sweep. Space-separated list; empty means no --seed override.
TRAIN_SEEDS="${TRAIN_SEEDS:-}"

# Optional overrides passed to scripts/train.py
OVERRIDE_RESULTS_DIR="${OVERRIDE_RESULTS_DIR:-}"
OVERRIDE_BATCH_SIZE="${OVERRIDE_BATCH_SIZE:-}"
OVERRIDE_TOTAL_STEPS="${OVERRIDE_TOTAL_STEPS:-}"
OVERRIDE_LR="${OVERRIDE_LR:-}"

RESUME_LATEST="${RESUME_LATEST:-1}"

IFS=',' read -r -a CONFIG_ARR <<< "$CONFIG_LIST"
if [ -z "$TRAIN_SEEDS" ]; then
  SEED_ARR=("")
else
  read -r -a SEED_ARR <<< "$TRAIN_SEEDS"
fi

NUM_CONFIGS=${#CONFIG_ARR[@]}
NUM_SEEDS=${#SEED_ARR[@]}
TOTAL_TASKS=$((NUM_CONFIGS * NUM_SEEDS))

latest_ckpt() {
  local ckpt_dir="$1"
  ls -1t "${ckpt_dir}"/ckpt_step*.pt 2>/dev/null | head -n 1
}

config_results_dir() {
  local config_path="$1"
  grep -m1 "^results_dir:" "$config_path" | awk '{print $2}'
}

run_train() {
  local config="$1"
  local seed="$2"

  if [ ! -f "$config" ]; then
    echo "[Error] Config not found: $config" >&2
    return 1
  fi

  local resume_arg=()
  if [ "$RESUME_LATEST" = "1" ]; then
    local results_dir
    results_dir="$(config_results_dir "$config")"
    if [ -n "$results_dir" ]; then
      local ckpt_dir="${results_dir}/checkpoints"
      local ckpt
      ckpt="$(latest_ckpt "$ckpt_dir")"
      if [ -n "$ckpt" ]; then
        echo "[Resume] Using checkpoint ${ckpt}"
        resume_arg=(--resume "$ckpt")
      fi
    fi
  fi

  local cmd=(
    "${PYTHON_BIN}" -u scripts/train.py
    --config "$config"
    --base_config "$BASE_CONFIG"
    --data_dir "$TRAIN_DATA_DIR"
  )

  if [ -n "$seed" ]; then
    cmd+=(--seed "$seed")
  fi
  if [ -n "$OVERRIDE_RESULTS_DIR" ]; then
    cmd+=(--results_dir "$OVERRIDE_RESULTS_DIR")
  fi
  if [ -n "$OVERRIDE_BATCH_SIZE" ]; then
    cmd+=(--batch_size "$OVERRIDE_BATCH_SIZE")
  fi
  if [ -n "$OVERRIDE_TOTAL_STEPS" ]; then
    cmd+=(--total_steps "$OVERRIDE_TOTAL_STEPS")
  fi
  if [ -n "$OVERRIDE_LR" ]; then
    cmd+=(--lr "$OVERRIDE_LR")
  fi
  cmd+=("${resume_arg[@]}")

  echo "[Train] config=${config} seed=${seed:-default}"
  if [ -n "${SLURM_JOB_ID:-}" ] && command -v srun >/dev/null 2>&1; then
    srun --ntasks=1 --gpus-per-task=1 --gpu-bind=single:1 "${cmd[@]}"
  else
    "${cmd[@]}"
  fi
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL_TASKS}" ]; then
    echo "[Skip] task_id ${SLURM_ARRAY_TASK_ID} >= total tasks ${TOTAL_TASKS}"
    exit 0
  fi

  CONFIG_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
  SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

  config="${CONFIG_ARR[$CONFIG_IDX]}"
  seed="${SEED_ARR[$SEED_IDX]}"
  run_train "$config" "$seed"
else
  for config in "${CONFIG_ARR[@]}"; do
    for seed in "${SEED_ARR[@]}"; do
      run_train "$config" "$seed"
    done
  done
fi
