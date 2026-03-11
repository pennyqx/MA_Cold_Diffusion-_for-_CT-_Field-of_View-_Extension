#!/usr/bin/env python3
"""
Training Script for CT Truncation Recovery Models

Supports:
- Cold Diffusion (main method)
- DDPM (baseline)
- UNet Inpainting (baseline)

Usage:
    python train.py --config config/baseline_cold_diffusion_k320.yaml
    python train.py --config config/base.yaml --data_dir ./data/sinograms
"""

import os
import sys
import signal
import subprocess
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import UNet, ColdDiffusion2D, DDPM2D, UNetInpaint
from models.cold_diffusion import build_cold_diffusion
from models.ddpm import build_ddpm
from models.unet_inpaint import build_unet_inpaint
from data import build_dataloaders
from training import Trainer
from utils.config import load_config, save_config, Config, resolve_keep_center

_requeue_requested = False
_trainer_ref = None


def _handle_slurm_signal(signum, frame):
    """Handle SLURM time limit signal (USR1) as early as possible."""
    global _requeue_requested
    _requeue_requested = True
    print("\n" + "=" * 60, flush=True)
    print("[SLURM] Received SIGUSR1 signal - time limit approaching.", flush=True)
    print("[SLURM] Will save checkpoint and exit gracefully.", flush=True)
    print("=" * 60, flush=True)

    if _trainer_ref is not None:
        _trainer_ref.request_requeue()
        return

    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        print(f"[SLURM] Trainer not ready, requeueing job {job_id} now.", flush=True)
        try:
            subprocess.run(["scontrol", "requeue", job_id], check=False)
        except Exception as e:
            print(f"[SLURM] Failed to requeue job: {e}", flush=True)
    else:
        print("[SLURM] SLURM_JOB_ID not set; cannot requeue.", flush=True)

    sys.exit(0)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_model(config: Config) -> torch.nn.Module:
    """Build model based on configuration"""
    model_type = config.model.type.lower()
    image_size = tuple(config.data.image_size)
    keep_center = resolve_keep_center(config.diffusion, image_size[1])
    use_wce_input = bool(getattr(config.diffusion, 'use_wce_input', False))

    if model_type == 'cold_diffusion':
        model = build_cold_diffusion(
            image_size=image_size,
            dim=config.model.dim,
            dim_mults=tuple(config.model.dim_mults),
            timesteps=config.diffusion.timesteps,
            keep_center=keep_center,
            sampling_routine=config.diffusion.sampling_routine,
            loss_type=config.diffusion.loss_type,
            use_wce_input=use_wce_input
        )
        print(f"[Model] Cold Diffusion: timesteps={config.diffusion.timesteps}, "
              f"keep_center={keep_center}, use_wce_input={use_wce_input}")

    elif model_type == 'ddpm':
        model = build_ddpm(
            image_size=image_size,
            dim=config.model.dim,
            dim_mults=tuple(config.model.dim_mults),
            timesteps=config.diffusion.timesteps,
            beta_schedule=config.diffusion.beta_schedule,
            loss_type=config.diffusion.loss_type,
            keep_center=keep_center,
            cond_channels=2,
            use_wce_input=use_wce_input
        )
        print(f"[Model] DDPM: timesteps={config.diffusion.timesteps}, "
              f"beta_schedule={config.diffusion.beta_schedule}, "
              f"keep_center={keep_center}, use_wce_input={use_wce_input}")

    elif model_type == 'unet_inpaint':
        model = build_unet_inpaint(
            image_size=image_size,
            dim=config.model.dim,
            dim_mults=tuple(config.model.dim_mults),
            keep_center=keep_center,
            loss_type='full',
            use_wce_input=use_wce_input
        )
        print(f"[Model] UNet Inpainting: keep_center={keep_center}, use_wce_input={use_wce_input}")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable parameters: {n_params:,}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train CT Truncation Recovery Model')

    # Configuration
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration YAML file')
    parser.add_argument('--base_config', type=str, default=None,
                       help='Path to base configuration (for inheritance)')

    # Override options
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Override data directory')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Override results directory')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--total_steps', type=int, default=None,
                       help='Override total training steps')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')

    # Other options
    parser.add_argument('--seed', type=int, default=None,
                       help='Override random seed')
    parser.add_argument('--no_amp', action='store_true',
                       help='Disable mixed precision training')

    args = parser.parse_args()

    # Register signal handler early to catch time-limit signals during setup
    signal.signal(signal.SIGUSR1, _handle_slurm_signal)

    # Load configuration
    print(f"[Config] Loading from {args.config}")
    config = load_config(args.config, args.base_config)

    # Apply overrides
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.results_dir:
        config.results_dir = args.results_dir
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.total_steps:
        config.training.total_steps = args.total_steps
    if args.lr:
        config.training.lr = args.lr
    if args.seed:
        config.seed = args.seed
    if args.no_amp:
        config.training.enable_amp = False

    # Set random seed
    set_seed(config.seed)
    print(f"[Config] Random seed: {config.seed}")

    # Create results directory
    results_dir = Path(config.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    save_config(config, results_dir / 'config.yaml')
    print(f"[Config] Saved to {results_dir / 'config.yaml'}")

    # Print configuration summary
    print("\n" + "=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"Experiment: {config.experiment_name}")
    print(f"Model: {config.model.type}")
    print(f"Image size: {config.data.image_size}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Total steps: {config.training.total_steps}")
    print(f"Learning rate: {config.training.lr}")
    print(f"Results: {config.results_dir}")
    print("=" * 60 + "\n")

    # Build data loaders
    print("[Data] Building dataloaders...")
    split_dir = config.data.split_dir
    if split_dir is None:
        candidate = Path(config.data.data_dir).parent / 'splits'
        if candidate.exists():
            split_dir = str(candidate)
    train_loader, val_loader, data_mean, data_std = build_dataloaders(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        split_ratio=config.data.split_ratio,
        val_n_center=config.data.val_n_center,
        val_n_uniform=config.data.val_n_uniform,
        num_workers=config.data.num_workers,
        val_num_workers=getattr(config.data, 'val_num_workers', None),
        seed=config.seed,
        split_dir=split_dir
    )

    # Build model
    print("[Model] Building model...")
    model = build_model(config)

    # Create trainer
    print("[Train] Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,

        # Training params
        lr=config.training.lr,
        total_steps=config.training.total_steps,

        # Validation
        val_interval=config.training.val_interval,
        early_stop_patience=config.training.early_stop_patience,
        early_stop_delta=config.training.early_stop_delta,
        val_max_batches=getattr(config.training, 'val_max_batches', None),
        val_truncation_ratio=getattr(config.training, 'val_truncation_ratio', 0.25),

        # EMA
        ema_beta=config.training.ema_beta,
        ema_start_step=config.training.ema_start_step,
        ema_update_every=config.training.ema_update_every,

        # Optimization
        enable_amp=config.training.enable_amp,
        grad_clip_norm=config.training.grad_clip_norm,
        lr_schedule=config.training.lr_schedule,
        lr_min=config.training.lr_min,
        warmup_steps=config.training.warmup_steps,

        # Checkpointing
        results_dir=str(config.results_dir),
        save_interval=config.training.save_interval,
        keep_last_ckpt=config.training.keep_last_ckpt,

        # Data stats
        data_mean=data_mean,
        data_std=data_std,

        # Model type
        model_type=config.model.type
    )
    global _trainer_ref
    _trainer_ref = trainer
    if _requeue_requested:
        trainer.request_requeue()

    # Resume if specified
    if args.resume:
        print(f"[Train] Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\n[Train] Starting training...")
    trainer.train()

    if getattr(trainer, "_requeue_requested", False):
        print("\n[Train] Requeue requested; skip final visualization in this run.")
        print(f"[Train] Current step: {trainer.global_step}")
        return

    # Final visualization
    print("\n[Train] Generating final visualization...")
    trainer.visualize()

    print("\n[Train] Training completed!")
    print(f"[Train] Best PSNR: {trainer.best_metric:.2f} dB at step {trainer.best_step}")
    print(f"[Train] Results saved to: {config.results_dir}")


if __name__ == '__main__':
    main()
