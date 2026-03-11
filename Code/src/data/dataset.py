"""
Dataset and DataLoader for CT Sinogram Data

Handles loading of pre-computed sinograms stored as .npy files.
Each sinogram file has shape (V, W, H) = (views, detector_width, detector_height)
or a single 2D slice of shape (V, W).
"""

import os
import glob
import math
import random
import json
import time
import hashlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional
from pathlib import Path


def _close_mmap(arr: np.ndarray) -> None:
    """Close mmap file handle if present to avoid too many open files."""
    mm = getattr(arr, "_mmap", None)
    if mm is not None:
        mm.close()


def compute_global_stats(file_list: List[str]) -> Tuple[float, float]:
    """
    Compute global mean and std from training files.

    Uses streaming computation to avoid loading all data into memory.

    Args:
        file_list: List of .npy file paths

    Returns:
        Tuple of (mean, std)
    """
    total_sum = 0.0
    total_sq = 0.0
    total_cnt = 0

    for fp in file_list:
        arr = np.load(fp, mmap_mode='r')
        if arr.ndim == 4:  # Remove batch dimension if present
            arr = arr[0]

        total_sum += float(arr.sum())
        total_sq += float((arr ** 2).sum())
        total_cnt += arr.size
        _close_mmap(arr)

    mean = total_sum / total_cnt
    std = math.sqrt(max(total_sq / total_cnt - mean ** 2, 1e-12))

    return mean, std


class SinogramDataset2D(Dataset):
    """
    Dataset for 2D sinogram slices.

    Loads 3D sinogram volumes and extracts 2D slices along the detector height axis.
    If the file is already 2D, it is treated as a single slice.

    Args:
        files: List of .npy file paths
        mean: Global mean for normalization
        std: Global std for normalization
        mode: 'train', 'val', or 'test'
        val_n_center: Number of center slices to use for validation
        val_n_uniform: Number of uniformly sampled slices for validation
    """
    def __init__(
        self,
        files: List[str],
        mean: float,
        std: float,
        mode: str = 'train',
        val_n_center: int = 11,
        val_n_uniform: int = 10
    ):
        super().__init__()

        self.files = sorted(files)
        self.mean = float(mean)
        self.std = float(std)
        self.mode = mode.lower()

        # Build index: (file_path, slice_idx)
        # Note: do not keep mmap handles open for all files to avoid FD exhaustion.
        self.index = []
        for fp in self.files:
            arr = np.load(fp, mmap_mode='r')
            if arr.ndim == 4:
                arr = arr[0]

            if arr.ndim == 2:
                # Already a 2D sinogram slice
                self.index.append((fp, 0))
                _close_mmap(arr)
                continue

            if arr.ndim != 3:
                _close_mmap(arr)
                raise ValueError(f"Unexpected sinogram shape: {arr.shape} in {fp}")

            H = arr.shape[2]  # detector height dimension

            if self.mode == 'val':
                # Validation: use center + uniformly sampled slices
                mid = H // 2
                half = val_n_center // 2

                # Center slices
                center_idx = np.arange(max(0, mid - half), min(H, mid + half + 1))

                # Uniform sampling from remaining slices
                remaining = np.setdiff1d(np.arange(H), center_idx)
                if len(remaining) <= val_n_uniform:
                    extra_idx = remaining
                else:
                    extra_idx = remaining[
                        np.linspace(0, len(remaining) - 1, val_n_uniform, dtype=int)
                    ]

                idx_use = np.sort(np.concatenate([center_idx, extra_idx]))
            else:
                # Train/Test: use all slices
                idx_use = np.arange(H)

            self.index.extend([(fp, int(z)) for z in idx_use])
            _close_mmap(arr)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single 2D sinogram slice.

        Returns:
            Dictionary with keys:
                - sinogram: (1, V, W) normalized tensor
                - volume_path: Source file path
                - slice_idx: Slice index in the volume
                - num_slices: Total slices in the volume
                - file_name: Base filename
        """
        fp, z = self.index[idx]

        # Load slice on demand (avoid holding many open file handles)
        arr = np.load(fp, mmap_mode='r')
        if arr.ndim == 4:
            arr = arr[0]

        if arr.ndim == 2:
            slice_vw = arr.astype(np.float32)
            num_slices = 1
            slice_idx = 0
        else:
            slice_vw = arr[:, :, z].astype(np.float32)
            num_slices = arr.shape[2]
            slice_idx = z
        _close_mmap(arr)

        # Normalize
        slice_vw = (slice_vw - self.mean) / (self.std + 1e-6)

        # Convert to tensor with channel dimension
        tensor = torch.from_numpy(slice_vw).unsqueeze(0)  # [1, V, W]

        return {
            'sinogram': tensor,
            'volume_path': fp,
            'slice_idx': slice_idx,
            'num_slices': num_slices,
            'file_name': os.path.basename(fp)
        }


class SinogramDataset2DWithMask(SinogramDataset2D):
    """
    Dataset that also returns truncation masks.
    Useful for baseline methods that need explicit masks.
    """
    def __init__(
        self,
        files: List[str],
        mean: float,
        std: float,
        mode: str = 'train',
        keep_center: int = 11,
        max_truncation: Optional[int] = None,
        val_n_center: int = 11,
        val_n_uniform: int = 10
    ):
        super().__init__(files, mean, std, mode, val_n_center, val_n_uniform)

        self.keep_center = keep_center

        # Get W from first file (open on demand)
        first_arr = np.load(self.files[0], mmap_mode='r')
        if first_arr.ndim == 4:
            first_arr = first_arr[0]
        self.W = first_arr.shape[1]
        self.V = first_arr.shape[0]
        _close_mmap(first_arr)

        if max_truncation is None:
            max_truncation = (self.W - keep_center) // 2
        self.max_truncation = max_truncation

    def get_random_mask(self) -> Tuple[torch.Tensor, int]:
        """Generate random truncation mask"""
        n = random.randint(0, self.max_truncation)
        mask = torch.ones(1, self.V, self.W)
        if n > 0:
            mask[:, :, :n] = 0
            mask[:, :, -n:] = 0
        return mask, n

    def __getitem__(self, idx: int) -> Dict:
        data = super().__getitem__(idx)

        if self.mode == 'train':
            mask, truncation_level = self.get_random_mask()
            data['mask'] = mask
            data['truncation_level'] = truncation_level

        return data


def build_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    split_ratio: float = 0.8,
    val_n_center: int = 11,
    val_n_uniform: int = 10,
    num_workers: int = 4,
    val_num_workers: Optional[int] = None,
    seed: int = 42,
    precomputed_stats: Optional[Tuple[float, float]] = None,
    split_dir: Optional[str] = None
) -> Tuple[DataLoader, DataLoader, float, float]:
    """
    Build training and validation dataloaders.

    Args:
        data_dir: Directory containing .npy sinogram files
        batch_size: Batch size
        split_ratio: Fraction of data for training
        val_n_center: Number of center slices for validation
        val_n_uniform: Number of uniformly sampled slices for validation
        num_workers: Number of data loading workers (train)
        val_num_workers: Number of data loading workers (val). None -> use num_workers.
        seed: Random seed for reproducibility
        precomputed_stats: Optional (mean, std) tuple to skip computation
        split_dir: Optional directory containing train/val split files

    Returns:
        Tuple of (train_loader, val_loader, mean, std)
    """
    def _load_split_files(split_name: str) -> Optional[List[str]]:
        if not split_dir:
            return None
        split_path = os.path.join(split_dir, f"{split_name}.txt")
        if not os.path.exists(split_path):
            return None
        files = []
        with open(split_path, 'r') as f:
            for line in f:
                name = line.strip()
                if not name:
                    continue
                if os.path.isabs(name) and os.path.exists(name):
                    files.append(name)
                else:
                    candidate = os.path.join(data_dir, name)
                    if os.path.exists(candidate):
                        files.append(candidate)
                    else:
                        files.append(name)
        return files

    train_files = _load_split_files('train')
    val_files = _load_split_files('val')

    if train_files is None or val_files is None:
        # Collect all .npy files
        all_files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
        if not all_files:
            raise ValueError(f"No .npy files found in {data_dir}")

        # Shuffle and split
        rng = random.Random(seed)
        rng.shuffle(all_files)

        split_idx = int(len(all_files) * split_ratio)
        train_files = all_files[:split_idx]
        val_files = all_files[split_idx:]

    print(f"[Data] Found {len(train_files) + len(val_files)} files: {len(train_files)} train, {len(val_files)} val")

    # Compute or use provided stats
    if precomputed_stats is not None:
        mean, std = precomputed_stats
    else:
        stats_cache_file = os.path.join(data_dir, 'normalization_stats.json')
        train_list_fingerprint = hashlib.sha1(
            "\n".join(train_files).encode("utf-8")
        ).hexdigest()

        if os.path.exists(stats_cache_file):
            try:
                with open(stats_cache_file, 'r') as f:
                    cached_stats = json.load(f)
                cached_n_files = cached_stats.get('n_files')
                cached_fingerprint = cached_stats.get('train_fingerprint')
                cached_split_dir = cached_stats.get('split_dir')

                if (
                    cached_n_files == len(train_files)
                    and cached_fingerprint == train_list_fingerprint
                    and cached_split_dir == split_dir
                ):
                    mean = float(cached_stats['mean'])
                    std = float(cached_stats['std'])
                    print(f"[Data] Loaded cached statistics from {stats_cache_file}")
                else:
                    print("[Data] Cached stats are stale, recomputing...")
                    mean, std = compute_global_stats(train_files)
            except Exception as e:
                print(f"[Data] Warning: Failed to load cached stats: {e}")
                print("[Data] Computing global statistics from training data...")
                mean, std = compute_global_stats(train_files)
        else:
            print("[Data] Computing global statistics from training data...")
            mean, std = compute_global_stats(train_files)

        if 'mean' in locals() and 'std' in locals():
            try:
                with open(stats_cache_file, 'w') as f:
                    json.dump(
                        {
                            'mean': float(mean),
                            'std': float(std),
                            'n_files': len(train_files),
                            'train_fingerprint': train_list_fingerprint,
                            'split_dir': split_dir,
                            'computed_at': time.strftime('%Y-%m-%d %H:%M:%S')
                        },
                        f,
                        indent=2
                    )
                print(f"[Data] Saved statistics cache to {stats_cache_file}")
            except Exception as e:
                print(f"[Data] Warning: Failed to save stats cache: {e}")

    print(f"[Data] Global stats: mean={mean:.4f}, std={std:.4f}")

    # Create datasets
    train_ds = SinogramDataset2D(
        train_files,
        mean=mean,
        std=std,
        mode='train'
    )
    val_ds = SinogramDataset2D(
        val_files,
        mean=mean,
        std=std,
        mode='val',
        val_n_center=val_n_center,
        val_n_uniform=val_n_uniform
    )

    print(f"[Data] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(num_workers > 0)
    )
    val_workers = num_workers if val_num_workers is None else val_num_workers
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=(val_workers > 0)
    )

    return train_loader, val_loader, mean, std


def build_test_dataloader(
    data_dir: str,
    mean: float,
    std: float,
    batch_size: int = 8,
    num_workers: int = 4
) -> DataLoader:
    """
    Build test dataloader.

    Args:
        data_dir: Directory containing test .npy files
        mean: Normalization mean (from training)
        std: Normalization std (from training)
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        Test DataLoader
    """
    test_files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
    if not test_files:
        raise ValueError(f"No .npy files found in {data_dir}")

    test_ds = SinogramDataset2D(
        test_files,
        mean=mean,
        std=std,
        mode='test'
    )

    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test dataset
    import tempfile

    # Create dummy data
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            dummy = np.random.randn(360, 640, 100).astype(np.float32)
            np.save(os.path.join(tmpdir, f"sino_{i:04d}.npy"), dummy)

        # Test dataloader building
        train_loader, val_loader, mean, std = build_dataloaders(
            tmpdir,
            batch_size=4,
            split_ratio=0.8,
            num_workers=0
        )

        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Test iteration
        batch = next(iter(train_loader))
        print(f"Batch sinogram shape: {batch['sinogram'].shape}")
        print(f"Batch slice indices: {batch['slice_idx']}")
