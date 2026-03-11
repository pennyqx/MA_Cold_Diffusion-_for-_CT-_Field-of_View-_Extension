"""
Configuration Management

Handles loading, saving, and merging YAML configuration files.
Supports hierarchical configs with base config inheritance.
"""

import os
import yaml
from yaml.constructor import ConstructorError
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    type: str = 'cold_diffusion'  # 'cold_diffusion', 'ddpm', 'unet_inpaint'
    dim: int = 64
    dim_mults: tuple = (1, 2, 4, 8)
    channels: int = 1
    with_time_emb: bool = True
    residual: bool = False


@dataclass
class DiffusionConfig:
    """Diffusion process configuration"""
    timesteps: int = 100
    keep_center: int = 11
    # Optional total truncation ratio (0.25 means keep 75% center, e.g., 480/640).
    truncation_ratio: Optional[float] = None
    sampling_routine: str = 'x0_step_down'  # 'default' or 'x0_step_down'
    loss_type: str = 'l1'  # 'l1' or 'l2'
    beta_schedule: str = 'cosine'  # For DDPM: 'linear' or 'cosine'
    # If true, apply WCE to truncated sinogram before feeding model.
    use_wce_input: bool = False


@dataclass
class DataConfig:
    """Data configuration"""
    image_size: tuple = (360, 640)  # (H, W)
    data_dir: str = './data/sinograms'
    split_dir: Optional[str] = None
    batch_size: int = 8
    num_workers: int = 4
    # Optional override for validation loader workers (None -> use num_workers)
    val_num_workers: Optional[int] = None
    split_ratio: float = 0.8
    val_n_center: int = 11
    val_n_uniform: int = 10


@dataclass
class TrainingConfig:
    """Training configuration"""
    lr: float = 1e-4
    lr_min: float = 5e-6
    lr_schedule: str = 'cosine'  # 'none', 'cosine', 'cosine_warmup'
    warmup_steps: int = 2000
    total_steps: int = 150000
    val_interval: int = 2000
    # Optional cap for validation batches (None = full validation)
    val_max_batches: Optional[int] = None
    # Validation truncation ratio per side (e.g., 0.25 = remove 1/4 from each side)
    val_truncation_ratio: float = 0.25
    save_interval: int = 5000

    # EMA
    ema_beta: float = 0.9995
    ema_start_step: int = 2000
    ema_update_every: int = 10

    # Regularization
    enable_amp: bool = True
    grad_clip_norm: float = 1.0

    # Early stopping
    early_stop_patience: int = 10
    early_stop_delta: float = 0.05

    # Checkpointing
    keep_last_ckpt: int = 3


@dataclass
class Config:
    """Main configuration"""
    experiment_name: str = 'default'
    seed: int = 42

    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    results_dir: str = './results'


def _dict_to_dataclass(d: Dict, cls):
    """Recursively convert dictionary to dataclass"""
    if not hasattr(cls, '__dataclass_fields__'):
        return d

    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}

    for key, value in d.items():
        if key in field_types:
            field_type = field_types[key]
            if hasattr(field_type, '__dataclass_fields__'):
                kwargs[key] = _dict_to_dataclass(value, field_type)
            elif isinstance(value, list) and field_type == tuple:
                kwargs[key] = tuple(value)
            else:
                kwargs[key] = value

    return cls(**kwargs)


def load_config(
    config_path: Union[str, Path],
    base_config_path: Optional[Union[str, Path]] = None
) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file
        base_config_path: Optional path to base configuration (for inheritance)

    Returns:
        Config dataclass
    """
    config_path = Path(config_path)

    # Auto-discover base config when not explicitly provided.
    # This keeps experiment configs small while still inheriting defaults.
    if base_config_path is None:
        candidate = config_path.parent / 'base.yaml'
        if config_path.name != 'base.yaml' and candidate.exists():
            base_config_path = candidate

    # Load base config first
    if base_config_path:
        with open(base_config_path, 'r') as f:
            base_dict = yaml.safe_load(f)
    else:
        base_dict = {}

    # Load main config
    with open(config_path, 'r') as f:
        try:
            config_dict = yaml.safe_load(f)
        except ConstructorError:
            # Fallback for legacy configs containing !!python/tuple
            f.seek(0)
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

    # Merge (main overrides base)
    merged = _deep_merge(base_dict, config_dict)

    # Convert to dataclass
    return _dict_to_dataclass(merged, Config)


def save_config(config: Config, save_path: Union[str, Path]):
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dataclass
        save_path: Path to save file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = _dataclass_to_dict(config)

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def _dataclass_to_dict(obj) -> Dict:
    """Recursively convert dataclass to dictionary"""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    elif isinstance(obj, tuple):
        return list(obj)
    return obj


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def get_experiment_dir(config: Config) -> Path:
    """Get experiment results directory"""
    return Path(config.results_dir) / config.experiment_name


def create_experiment_config(
    base_config: Config,
    **overrides
) -> Config:
    """
    Create new config with overrides.

    Args:
        base_config: Base configuration
        **overrides: Key-value pairs to override (use dot notation for nested keys)

    Returns:
        New Config with overrides applied
    """
    config_dict = _dataclass_to_dict(base_config)

    for key, value in overrides.items():
        parts = key.split('.')
        target = config_dict

        for part in parts[:-1]:
            target = target[part]

        target[parts[-1]] = value

    return _dict_to_dataclass(config_dict, Config)


def resolve_keep_center(diffusion_cfg: DiffusionConfig, detector_width: int) -> int:
    """
    Resolve keep_center from either explicit keep_center or truncation_ratio.

    truncation_ratio is interpreted as total removed detector fraction:
    keep_center = round((1 - truncation_ratio) * detector_width)
    """
    ratio = getattr(diffusion_cfg, 'truncation_ratio', None)
    if ratio is None:
        return int(diffusion_cfg.keep_center)

    ratio = float(ratio)
    ratio = max(0.0, min(0.99, ratio))
    keep_center = int(round((1.0 - ratio) * float(detector_width)))
    keep_center = max(1, min(detector_width, keep_center))
    return keep_center


if __name__ == "__main__":
    # Test configuration
    print("Testing configuration management...")

    # Create default config
    config = Config()
    print(f"Default config: {config.experiment_name}")
    print(f"Model dim: {config.model.dim}")
    print(f"Timesteps: {config.diffusion.timesteps}")

    # Test override
    new_config = create_experiment_config(
        config,
        experiment_name='exp1_k11',
        **{'diffusion.keep_center': 11, 'training.total_steps': 100000}
    )
    print(f"\nOverridden config: {new_config.experiment_name}")
    print(f"keep_center: {new_config.diffusion.keep_center}")
    print(f"total_steps: {new_config.training.total_steps}")

    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        save_config(config, f.name)
        loaded = load_config(f.name)
        print(f"\nLoaded config: {loaded.experiment_name}")
        os.unlink(f.name)

    print("\nAll tests passed!")
