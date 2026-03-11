"""
Unified Trainer for Diffusion Models

Supports:
- Cold Diffusion (ColdDiffusion2D)
- Standard DDPM (DDPM2D)
- UNet Inpainting (UNetInpaint)

Features:
- Exponential Moving Average (EMA)
- Mixed precision training (AMP)
- Learning rate scheduling
- Early stopping
- Checkpoint management
- Validation and metrics logging
"""

import os
import copy
import json
import time
import math
import re
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import utils as vutils

import numpy as np


class EarlyStopping:
    """
    Early stopping based on validation metric.

    Stops training if the metric doesn't improve for `patience` validations.
    """
    def __init__(self, min_delta: float = 0.05, patience: int = 5):
        """
        Args:
            min_delta: Minimum improvement to count as progress
            patience: Number of validations without improvement before stopping
        """
        self.min_delta = min_delta
        self.patience = patience
        self.best = -float('inf')
        self.bad_epochs = 0

    def step(self, metric: float) -> bool:
        """
        Check if training should stop.

        Args:
            metric: Current validation metric (higher is better)

        Returns:
            True if training should stop
        """
        if metric > self.best + self.min_delta:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience


class EMA:
    """Exponential Moving Average of model parameters"""
    def __init__(self, model: nn.Module, beta: float = 0.9995):
        """
        Args:
            model: Model to track
            beta: EMA decay rate
        """
        self.model = copy.deepcopy(model)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.beta = beta

    def update(self, model: nn.Module):
        """Update EMA parameters"""
        with torch.no_grad():
            for ema_p, p in zip(self.model.parameters(), model.parameters()):
                ema_p.lerp_(p, 1 - self.beta)

            # Also sync buffers
            for ema_b, b in zip(self.model.buffers(), model.buffers()):
                if torch.is_tensor(ema_b):
                    ema_b.copy_(b)

    def copy_from(self, model: nn.Module):
        """Hard copy parameters from model"""
        self.model.load_state_dict(model.state_dict())


class Trainer:
    """
    Unified trainer for diffusion and inpainting models.

    Handles training loop, validation, checkpointing, and logging.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        # Training parameters
        lr: float = 1e-4,
        total_steps: int = 150000,
        batch_size: int = 8,

        # Validation
        val_interval: int = 2000,
        early_stop_patience: int = 10,
        early_stop_delta: float = 0.05,
        val_max_batches: Optional[int] = None,
        val_truncation_ratio: float = 0.25,

        # EMA
        ema_beta: float = 0.9995,
        ema_start_step: int = 2000,
        ema_update_every: int = 10,

        # AMP and optimization
        enable_amp: bool = True,
        grad_clip_norm: Optional[float] = 1.0,
        lr_schedule: str = 'cosine',  # 'none', 'cosine', 'cosine_warmup'
        lr_min: float = 5e-6,
        warmup_steps: int = 2000,

        # Checkpointing
        results_dir: str = './results',
        save_interval: int = 5000,
        keep_last_ckpt: int = 3,

        # Data stats (for denormalization)
        data_mean: float = 0.0,
        data_std: float = 1.0,

        # Model type for correct evaluation
        model_type: str = 'cold_diffusion',  # 'cold_diffusion', 'ddpm', 'unet_inpaint'
    ):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model_type = model_type

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Data stats
        self.data_mean = data_mean
        self.data_std = data_std

        # Training parameters
        self.lr = lr
        self.total_steps = total_steps
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.val_max_batches = val_max_batches
        self.val_truncation_ratio = val_truncation_ratio

        # Setup results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'checkpoints').mkdir(exist_ok=True)
        (self.results_dir / 'vis').mkdir(exist_ok=True)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-6,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler
        self.scheduler = None
        if lr_schedule == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=lr_min
            )
        elif lr_schedule == 'cosine_warmup':
            def lr_lambda(step):
                if step < warmup_steps:
                    return step / warmup_steps
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return lr_min / lr + (1 - lr_min / lr) * 0.5 * (1 + math.cos(math.pi * progress))
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # AMP
        self.use_amp = enable_amp and torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        self.grad_clip_norm = grad_clip_norm

        # EMA
        self.ema = EMA(self.model, beta=ema_beta)
        self.ema_start_step = ema_start_step
        self.ema_update_every = ema_update_every

        # Early stopping
        self.early_stopping = EarlyStopping(
            min_delta=early_stop_delta,
            patience=early_stop_patience
        )

        # Checkpoint management
        self.keep_last_ckpt = keep_last_ckpt
        self.ckpt_slots = []

        # Training state
        self.global_step = 0
        self.val_count = 0
        self.best_metric = -float('inf')
        self.best_step = -1
        self.current_metric = None

        # Logging
        self.train_csv = self.results_dir / 'train_loss.csv'
        self.val_csv = self.results_dir / 'val_metrics.csv'
        self._init_logs()

        # Track requeue request state
        self._requeue_requested = False

    def request_requeue(self) -> None:
        """Request a graceful stop and requeue at the next safe point."""
        self._requeue_requested = True
    def _init_logs(self):
        """Initialize log files"""
        if not self.train_csv.exists():
            with open(self.train_csv, 'w') as f:
                f.write('step,loss,lr\n')

        if not self.val_csv.exists():
            with open(self.val_csv, 'w') as f:
                f.write('step,MAE,RMSE,PSNR\n')

    def _denorm(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor"""
        return x * (self.data_std + 1e-6) + self.data_mean

    def _fixed_truncation(self, img: torch.Tensor) -> int:
        """Compute fixed truncation level (per side) from detector width."""
        width = img.size(-1)
        return int(width * self.val_truncation_ratio)

    def _update_ema(self):
        """Update EMA model"""
        if self.global_step < self.ema_start_step:
            self.ema.copy_from(self.model)
        elif self.global_step % self.ema_update_every == 0:
            self.ema.update(self.model)

    def train(self):
        """Main training loop"""
        print(f"[Train] Starting training for {self.total_steps} steps")
        print(f"[Train] Device: {self.device}")
        print(f"[Train] Model type: {self.model_type}")

        train_iter = iter(self.train_loader)
        start_time = time.time()
        first_batch_logged = False

        while self.global_step < self.total_steps:
            if self._requeue_requested:
                print("[Train] Requeue requested, saving checkpoint...")
                try:
                    self._save_checkpoint()
                    print("[Train] Checkpoint saved for requeue.")
                except Exception as e:
                    print(f"[Train] Failed to save checkpoint: {e}")

                job_id = os.environ.get('SLURM_JOB_ID')
                if job_id:
                    print(f"[Train] Requesting requeue for job {job_id}")
                    os.system(f"scontrol requeue {job_id}")

                print("[Train] Exiting training loop for requeue.")
                break

            self.model.train()

            # Get batch
            try:
                if not first_batch_logged:
                    print("[Train] Fetching first training batch...")
                    fetch_t0 = time.time()
                batch = next(train_iter)
                if not first_batch_logged:
                    fetch_elapsed = time.time() - fetch_t0
                    print(f"[Train] First batch fetched in {fetch_elapsed:.2f}s")
                    first_batch_logged = True
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)

            img = batch['sinogram'].to(self.device)

            # Forward and backward
            with autocast(enabled=self.use_amp):
                loss = self.model(img)

            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.grad_clip_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()

            # EMA update
            self._update_ema()

            # Logging
            if self.global_step % 100 == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"[{self.global_step}] loss={loss.item():.6f} lr={lr:.2e}")
                with open(self.train_csv, 'a') as f:
                    f.write(f"{self.global_step},{loss.item():.6f},{lr:.2e}\n")

            # Validation
            if (self.global_step + 1) % self.val_interval == 0:
                self.validate()

                if self._requeue_requested:
                    continue

                # Check early stopping
                if self.current_metric is None:
                    print("[EarlyStop] Validation metric unavailable, skipping early-stop check.")
                elif self.early_stopping.step(self.current_metric):
                    print(f"[EarlyStop] No improvement for {self.early_stopping.patience} validations")
                    break

            # Checkpoint
            if (self.global_step + 1) % self.save_interval == 0:
                self._save_checkpoint()

            self.global_step += 1

        elapsed = time.time() - start_time
        print(f"[Train] Training completed in {elapsed/3600:.2f} hours")
        print(f"[Train] Best PSNR: {self.best_metric:.2f} dB at step {self.best_step}")

    @torch.no_grad()
    def validate(self):
        """Validation loop"""
        if self._requeue_requested:
            print("[Val] Requeue requested, skipping validation.")
            return

        self.current_metric = None
        self.model.eval()
        self.val_count += 1

        metrics = {'MAE': 0.0, 'RMSE': 0.0, 'PSNR': 0.0}
        count = 0
        total_batches = len(self.val_loader)
        max_batches = self.val_max_batches if self.val_max_batches is not None else total_batches
        log_every = int(os.environ.get("VAL_LOG_EVERY", "50"))
        start_time = time.time()
        print(f"[Val] Starting validation: total_batches={total_batches}, max_batches={max_batches}")

        for batch_idx, batch in enumerate(self.val_loader):
            if self._requeue_requested:
                print("[Val] Requeue requested, aborting validation early.")
                return
            img = batch['sinogram'].to(self.device)

            # Get prediction based on model type
            if self.model_type == 'cold_diffusion':
                _, _, pred = self.ema.model.sample(img)
            elif self.model_type == 'ddpm':
                # Conditional DDPM: sample using truncated sinogram + mask
                truncation = self._fixed_truncation(img)
                mask = self.ema.model.get_mask_at_level(truncation, img.size(0)).to(self.device)
                truncated = img * mask
                pred = self.ema.model.sample_conditional(truncated, mask)
            elif self.model_type == 'unet_inpaint':
                # For inpainting, we need to truncate first
                truncation = self._fixed_truncation(img)
                mask = self.ema.model.get_mask_at_level(truncation, img.size(0)).to(self.device)
                truncated = img * mask
                pred = self.ema.model.inpaint(truncated, mask)
            else:
                pred = img  # Fallback

            # Compute metrics
            diff = img - pred
            mae = diff.abs().mean().item()
            rmse = torch.sqrt((diff ** 2).mean()).item()

            # PSNR with dynamic range
            dyn_range = (img.max() - img.min()).clamp(min=1e-8).item()
            mse = (diff ** 2).mean().item()
            psnr = 20 * math.log10(dyn_range / math.sqrt(mse + 1e-8))

            metrics['MAE'] += mae
            metrics['RMSE'] += rmse
            metrics['PSNR'] += psnr
            count += 1

            if self.val_max_batches is not None and count >= self.val_max_batches:
                print(f"[Val] Reached val_max_batches={self.val_max_batches}, stopping early.")
                break
            if log_every > 0 and (count % log_every) == 0:
                elapsed = time.time() - start_time
                print(f"[Val] Progress {count}/{max_batches} batches, elapsed={elapsed:.1f}s")

        if count == 0:
            print("[Val] No validation batches processed.")
            return

        # Average
        for k in metrics:
            metrics[k] /= count

        self.current_metric = metrics['PSNR']

        elapsed = time.time() - start_time
        print(f"[Val] Completed {count} batches in {elapsed:.1f}s")
        print(f"\n{'='*60}")
        print(f"[Val #{self.val_count}] Step {self.global_step}")
        print(f"  MAE:  {metrics['MAE']:.6f}")
        print(f"  RMSE: {metrics['RMSE']:.6f}")
        print(f"  PSNR: {metrics['PSNR']:.2f} dB")
        print(f"{'='*60}\n")

        # Log to CSV
        with open(self.val_csv, 'a') as f:
            f.write(f"{self.global_step},{metrics['MAE']:.6f},"
                    f"{metrics['RMSE']:.6f},{metrics['PSNR']:.3f}\n")

        # Save best model
        if metrics['PSNR'] > self.best_metric:
            self.best_metric = metrics['PSNR']
            self.best_step = self.global_step
            self._save_best()
            print(f"[Best] New best PSNR: {self.best_metric:.2f} dB")

    def _save_checkpoint(self):
        """Save checkpoint with rotation"""
        ckpt_dir = self.results_dir / 'checkpoints'

        ckpt = {
            'step': self.global_step,
            'model': self.model.state_dict(),
            'ema': self.ema.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'data_mean': self.data_mean,
            'data_std': self.data_std,
            'best_metric': self.best_metric,
            'early_stop_best': self.early_stopping.best,
            'early_stop_bad_epochs': self.early_stopping.bad_epochs,
        }

        if self.scheduler is not None:
            ckpt['scheduler'] = self.scheduler.state_dict()

        # Rotate checkpoints
        ckpt_path = ckpt_dir / f'ckpt_step{self.global_step}.pt'
        torch.save(ckpt, ckpt_path)
        self.ckpt_slots.append(ckpt_path)

        # Remove old checkpoints
        while len(self.ckpt_slots) > self.keep_last_ckpt:
            old = self.ckpt_slots.pop(0)
            if old.exists():
                old.unlink()

        print(f"[Checkpoint] Saved {ckpt_path}")

    def _save_best(self):
        """Save best model"""
        ckpt_dir = self.results_dir / 'checkpoints'

        ckpt = {
            'step': self.global_step,
            'model': self.model.state_dict(),
            'ema': self.ema.model.state_dict(),
            'data_mean': self.data_mean,
            'data_std': self.data_std,
            'best_metric': self.best_metric,
        }

        torch.save(ckpt, ckpt_dir / 'best.pt')

        # Save metadata
        with open(ckpt_dir / 'best.txt', 'w') as f:
            f.write(f"step,{self.global_step}\n")
            f.write(f"PSNR,{self.best_metric:.4f}\n")

    def load_checkpoint(self, ckpt_path: str, load_optimizer: bool = True):
        """Load checkpoint"""
        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt['model'])
        self.ema.model.load_state_dict(ckpt['ema'])

        if load_optimizer and 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scaler.load_state_dict(ckpt['scaler'])
            if self.scheduler is not None and 'scheduler' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler'])

        self.global_step = ckpt['step']
        self.data_mean = ckpt['data_mean']
        self.data_std = ckpt['data_std']
        self.best_metric = ckpt.get('best_metric', -float('inf'))
        self.early_stopping.best = ckpt.get('early_stop_best', -float('inf'))
        self.early_stopping.bad_epochs = ckpt.get('early_stop_bad_epochs', 0)

        # Rebuild checkpoint slots from existing files so rotation stays correct after resume.
        ckpt_dir = self.results_dir / 'checkpoints'
        parsed = []
        for p in ckpt_dir.glob('ckpt_step*.pt'):
            m = re.search(r'ckpt_step(\d+)\.pt$', p.name)
            if m:
                parsed.append((int(m.group(1)), p))
        parsed.sort(key=lambda x: x[0])
        self.ckpt_slots = [p for _, p in parsed]
        while len(self.ckpt_slots) > self.keep_last_ckpt:
            old = self.ckpt_slots.pop(0)
            if old.exists():
                old.unlink()

        print(f"[Load] Loaded checkpoint from step {self.global_step}")

    @torch.no_grad()
    def visualize(self, save_path: Optional[str] = None):
        """Generate visualization of model predictions"""
        self.model.eval()

        # Get a sample batch
        batch = next(iter(self.val_loader))
        img = batch['sinogram'].to(self.device)[:4]  # Take first 4 samples

        if self.model_type == 'cold_diffusion':
            x_T, _, pred = self.ema.model.sample(img)
            # Grid: original, truncated, reconstructed
            grid_imgs = torch.cat([
                self._denorm(img),
                self._denorm(x_T),
                self._denorm(pred)
            ], dim=0)
        elif self.model_type == 'ddpm':
            truncation = self._fixed_truncation(img)
            mask = self.ema.model.get_mask_at_level(truncation, img.size(0)).to(self.device)
            truncated = img * mask
            pred = self.ema.model.sample_conditional(truncated, mask)
            grid_imgs = torch.cat([
                self._denorm(img),
                self._denorm(truncated),
                self._denorm(pred)
            ], dim=0)
        elif self.model_type == 'unet_inpaint':
            truncation = self._fixed_truncation(img)
            mask = self.ema.model.get_mask_at_level(truncation, img.size(0)).to(self.device)
            truncated = img * mask
            pred = self.ema.model.inpaint(truncated, mask)
            grid_imgs = torch.cat([
                self._denorm(img),
                self._denorm(truncated),
                self._denorm(pred)
            ], dim=0)
        else:
            grid_imgs = self._denorm(img)

        # Normalize for visualization
        grid_imgs = (grid_imgs - grid_imgs.min()) / (grid_imgs.max() - grid_imgs.min() + 1e-8)

        if save_path is None:
            save_path = self.results_dir / 'vis' / f'vis_step{self.global_step}.png'

        vutils.save_image(grid_imgs, save_path, nrow=4)
        print(f"[Vis] Saved to {save_path}")


if __name__ == "__main__":
    print("Trainer module test")

    # Create dummy model and data
    import torch.nn as nn

    class DummyModel(nn.Module):
        def forward(self, x):
            return x.mean()

        def sample(self, x):
            return x, x, x

    model = DummyModel()

    # Create dummy dataloader
    from torch.utils.data import TensorDataset

    dummy_data = torch.randn(100, 1, 360, 640)
    dataset = TensorDataset(dummy_data)

    def collate_fn(batch):
        return {'sinogram': torch.stack([b[0] for b in batch])}

    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    # Test trainer initialization
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        total_steps=100,
        val_interval=50
    )

    print("Trainer initialized successfully")
