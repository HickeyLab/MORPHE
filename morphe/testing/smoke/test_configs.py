"""Smoke tests: verify all config classes instantiate with defaults."""
from __future__ import annotations

from pathlib import Path

from core.autoencoder.config import AutoencoderTrainerConfig
from core.gcnn.config import GCNNTrainerConfig
from core.latent_diffusion.train.config import LatentTrainerConfig
from core.pixel_diffusion.config import PixelDiffusionTrainerConfig
from core.preprocessor.config import PreProcessConfig


def test_autoencoder_config_defaults() -> None:
    cfg = AutoencoderTrainerConfig()
    assert isinstance(cfg.save_dir, Path)
    assert cfg.bottleneck_dim > 0
    assert cfg.hidden_dim > 0
    assert cfg.num_epochs > 0
    assert cfg.lr > 0


def test_gcnn_config_defaults() -> None:
    cfg = GCNNTrainerConfig()
    assert cfg.hidden_channels > 0
    assert cfg.epochs > 0
    assert cfg.lr > 0
    assert cfg.K > 0


def test_latent_config_defaults() -> None:
    cfg = LatentTrainerConfig()
    assert cfg.batch_size > 0
    assert cfg.epochs > 0
    assert cfg.lr > 0
    assert cfg.mixed_precision in {"no", "fp16", "bf16"}


def test_pixel_diffusion_config_defaults() -> None:
    cfg = PixelDiffusionTrainerConfig()
    assert cfg.train_batch_size > 0
    assert cfg.epochs > 0
    assert cfg.lr > 0
    assert cfg.mixed_precision in {"no", "fp16", "bf16"}


def test_preprocess_config_defaults() -> None:
    cfg = PreProcessConfig()
    assert len(cfg.original_dimensions) == 2
    assert all(d > 0 for d in cfg.original_dimensions)
    assert len(cfg.pos_cols) == 2
    assert all(isinstance(c, str) and c for c in cfg.pos_cols)
