# CT Field-of-View Extension

This repository contains thesis code for CT field-of-view extension from truncated sinograms.
It includes three model families:

- Cold Diffusion
- DDPM
- UNet inpainting

The codebase keeps the practical pipeline used for preprocessing, training, and evaluation.

## Structure

- `scripts/`: entry scripts
- `config/`: YAML configs and batch shell scripts
- `src/models/`: model definitions
- `src/data/`: preprocessing and dataset code
- `src/training/`: trainer and checkpoint logic
- `src/evaluation/`: metrics and visualization helpers

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

Run commands from the repository root.

## Main scripts

### 1. Data preprocessing

Build the unified dataset, generate splits, and export sinograms:

```bash
python3 scripts/batch_preprocess.py --help
```

This script is the preprocessing entry point for CT-ORG and LIDC-IDRI style data.

### 2. Training

Train a single model:

```bash
python3 scripts/train.py --config config/baseline_cold_diffusion_k320.yaml
```

Batch training through the provided shell script:

```bash
bash config/train_models.sh
```

Common environment overrides:

```bash
TRAIN_DATA_DIR=/path/to/train_val_sinograms \
CONFIG_LIST="config/baseline_cold_diffusion_k320.yaml,config/baseline_ddpm_k320.yaml" \
bash config/train_models.sh
```

### 3. Evaluation

Evaluate trained models with the unified evaluation script:

```bash
bash config/eval_models.sh
```

Example:

```bash
TEST_SINO_DIR=/path/to/test_sinograms_3d \
RESULTS_ROOT=./results \
MODEL_SPECS="cold_k320:cold:baselines/cold_diffusion_k320,ddpm_k320:ddpm:baselines/ddpm_k320" \
bash config/eval_models.sh
```

For direct image-domain evaluation of one checkpoint:

```bash
python3 scripts/evaluate_fbp.py --help
```

### 4. Aggregate multi-seed results

```bash
bash config/aggregate_eval_models.sh
```

## Configs

- `config/base.yaml`: shared defaults
- `config/baseline_*.yaml`: baseline experiments
- `config/exp3_timesteps_*.yaml`: timestep study configs

Training outputs and evaluation results are written under `results/` by default.
