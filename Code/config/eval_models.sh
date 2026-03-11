#!/bin/bash -l
#SBATCH --job-name=eval_models
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/eval_models_%A_%a.out
#SBATCH --error=logs/eval_models_%A_%a.err
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

TEST_SINO_DIR="${TEST_SINO_DIR:-${ROOT_DIR}/data/test_sinograms_3d}"
RESULTS_ROOT="${RESULTS_ROOT:-${ROOT_DIR}/results}"
OUT_ROOT="${OUT_ROOT:-${RESULTS_ROOT}/eval_models}"
RESOLVED_ROOT="${RESOLVED_ROOT:-${OUT_ROOT}/resolved_ckpt_dirs}"

MAX_CASES="${MAX_CASES:-6}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.1}"
SAMPLE_SEEDS="${SAMPLE_SEEDS:-42 52 62 72}"

# Evaluation condition: prefer keep-center if provided; otherwise use truncation ratio.
KEEP_CENTER_EVAL="${KEEP_CENTER_EVAL:-}"
TRUNC_RATIO_EVAL="${TRUNC_RATIO_EVAL:-0.25}"

# DDPM-specific sampling options.
DDPM_MAX_STEP="${DDPM_MAX_STEP:-100000}"
DDPM_SAMPLING="${DDPM_SAMPLING:-ddim}"
DDPM_STEPS="${DDPM_STEPS:-75}"

VIZ_PERCENTILES_LOW="${VIZ_PERCENTILES_LOW:-1}"
VIZ_PERCENTILES_HIGH="${VIZ_PERCENTILES_HIGH:-99}"
FOV_CENTER_CROP_FRAC="${FOV_CENTER_CROP_FRAC:-0.4}"

# MODEL_SPECS format (comma-separated): name:type:relative_model_dir
# type in {cold,ddpm,unet}
if [ -z "${MODEL_SPECS:-}" ]; then
  MODEL_SPECS="t50:cold:exp3_timesteps/t50,t150:cold:exp3_timesteps/t150"
fi

IFS=',' read -r -a MODEL_SPEC_ARR <<< "$MODEL_SPECS"
read -r -a SEED_ARR <<< "$SAMPLE_SEEDS"

NUM_MODELS=${#MODEL_SPEC_ARR[@]}
NUM_SEEDS=${#SEED_ARR[@]}
TOTAL_TASKS=$((NUM_MODELS * NUM_SEEDS))

latest_ckpt() {
  local ckpt_dir="$1"
  ls -1 "${ckpt_dir}"/ckpt_step*.pt 2>/dev/null | sort -V | tail -n 1
}

prepare_latest_only_dir() {
  local src_dir="$1"
  local dst_dir="$2"
  local src_ckpt_dir="${src_dir}/checkpoints"
  local latest

  latest=$(latest_ckpt "${src_ckpt_dir}")
  if [ -z "${latest}" ]; then
    echo "[Error] No ckpt_step*.pt found under ${src_ckpt_dir}" >&2
    return 1
  fi

  mkdir -p "${dst_dir}/checkpoints"
  rm -f "${dst_dir}/checkpoints"/ckpt_step*.pt
  cp -f "${src_dir}/config.yaml" "${dst_dir}/config.yaml"
  cp -f "${latest}" "${dst_dir}/checkpoints/$(basename "${latest}")"
  echo "[Checkpoint] ${src_dir} -> ${latest}"
}

run_eval() {
  local model_name="$1"
  local model_type="$2"
  local model_rel_dir="$3"
  local seed="$4"

  local model_dir="${RESULTS_ROOT}/${model_rel_dir}"
  local resolved_dir="${RESOLVED_ROOT}/${model_name}"
  local seed_out="${OUT_ROOT}/${model_name}/seed_${seed}"

  mkdir -p "${seed_out}" "${RESOLVED_ROOT}"
  prepare_latest_only_dir "${model_dir}" "${resolved_dir}"

  local cmd=(
    "${PYTHON_BIN}" -u scripts/eval_baseline_check.py
    --test_sino3d_dir "${TEST_SINO_DIR}"
    --output "${seed_out}"
    --fraction "${SAMPLE_FRACTION}"
    --max_cases "${MAX_CASES}"
    --batch_size "${EVAL_BATCH_SIZE}"
    --viz_percentiles "${VIZ_PERCENTILES_LOW}" "${VIZ_PERCENTILES_HIGH}"
    --viz_exclude_zeros
    --fov_center_crop_frac "${FOV_CENTER_CROP_FRAC}"
    --sample_seed "${seed}"
  )

  if [ -n "${KEEP_CENTER_EVAL}" ]; then
    cmd+=(--keep_center_eval "${KEEP_CENTER_EVAL}")
  else
    cmd+=(--truncation_ratio_eval "${TRUNC_RATIO_EVAL}")
  fi

  case "${model_type}" in
    cold)
      cmd+=(--cold_dir "${resolved_dir}")
      ;;
    ddpm)
      cmd+=(--ddpm_dir "${resolved_dir}" --ddpm_max_step "${DDPM_MAX_STEP}" --ddpm_sampling "${DDPM_SAMPLING}" --ddpm_steps "${DDPM_STEPS}")
      ;;
    unet)
      cmd+=(--unet_dir "${resolved_dir}")
      ;;
    *)
      echo "[Error] Unknown model type '${model_type}' in spec '${model_name}:${model_type}:${model_rel_dir}'" >&2
      return 1
      ;;
  esac

  echo "[Eval] model=${model_name} type=${model_type} seed=${seed} out=${seed_out}"
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

  MODEL_IDX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
  SEED_IDX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))

  spec="${MODEL_SPEC_ARR[$MODEL_IDX]}"
  IFS=':' read -r model_name model_type model_rel_dir <<< "$spec"
  seed="${SEED_ARR[$SEED_IDX]}"

  run_eval "$model_name" "$model_type" "$model_rel_dir" "$seed"
else
  # local/non-slurm mode: evaluate all combinations
  for spec in "${MODEL_SPEC_ARR[@]}"; do
    IFS=':' read -r model_name model_type model_rel_dir <<< "$spec"
    for seed in "${SEED_ARR[@]}"; do
      run_eval "$model_name" "$model_type" "$model_rel_dir" "$seed"
    done
  done
fi
