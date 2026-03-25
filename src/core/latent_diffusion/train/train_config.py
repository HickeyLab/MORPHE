from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class LatentTrainerConfig:
    """
    Configuration for latent diffusion training.

    This config centralizes optimization, checkpointing, architecture, and
    optional learning-rate decay behavior for the latent diffusion trainer.

    Notes:
        - ``supports_decay`` controls whether this training setup is allowed
          to use LR decay at all.
        - ``decay_enabled`` controls whether LR decay is actually turned on.
        - When decay is enabled, all required decay hyperparameters must be
          provided and valid.
    """
    # Learning-rate decay
    decay_enabled: bool = False
    patience: int | None = None
    lr_decay_every: int | None = None
    lr_decay_factor: float | None = None

    # Checkpointing / output
    save_dir: str | Path = "checkpoints"
    save_best_only: bool = True

    # Pretrained backbone
    unet_pretrained_path: str = "runwayml/stable-diffusion-v1-5"
    ae_pretrained_path: str = "runwayml/stable-diffusion-v1-5"
    scheduler_pretrained_path: str = "runwayml/stable-diffusion-v1-5"

    # Optimization / training loop
    lr: float = 2e-5
    mixed_precision: str = "fp16"
    grad_clip: float = 1.0
    batch_size: int = 8
    val_batch_size: int = 8
    epochs: int = 20

    # Optional encoder construction kwargs
    cond_encoder_kwargs: dict[str, Any] | None = None
    coord_encoder_kwargs: dict[str, Any] | None = None
    bbox_encoder_kwargs: dict[str, Any] | None = None
    
    def __post_init__(self) -> None:
        """Validate config immediately after construction."""
        self.validate()
        object.__setattr__(self, "save_dir", Path(self.save_dir))

    def validate(self) -> None:
        """Validate that all config values are internally consistent."""
        # ------------------------------------------------------------------
        # Decay configuration
        # ------------------------------------------------------------------
        if not isinstance(self.decay_enabled, bool):
            raise TypeError("decay_enabled must be a bool.")

        if self.decay_enabled:
            if self.patience is None:
                raise ValueError("patience must be provided when decay_enabled is True.")
            if self.lr_decay_every is None:
                raise ValueError("lr_decay_every must be provided when decay_enabled is True.")
            if self.lr_decay_factor is None:
                raise ValueError("lr_decay_factor must be provided when decay_enabled is True.")

        if self.patience is not None:
            if not isinstance(self.patience, int):
                raise TypeError("patience must be an int or None.")
            if self.patience <= 0:
                raise ValueError("patience must be > 0.")

        if self.lr_decay_every is not None:
            if not isinstance(self.lr_decay_every, int):
                raise TypeError("lr_decay_every must be an int or None.")
            if self.lr_decay_every <= 0:
                raise ValueError("lr_decay_every must be > 0.")

        if self.lr_decay_factor is not None:
            if not isinstance(self.lr_decay_factor, (int, float)):
                raise TypeError("lr_decay_factor must be a float or None.")
            if not (0.0 < float(self.lr_decay_factor) < 1.0):
                raise ValueError("lr_decay_factor must be between 0 and 1.")

        # ------------------------------------------------------------------
        # Checkpointing / output
        # ------------------------------------------------------------------
        if not isinstance(self.save_dir, str) or not self.save_dir:
            raise TypeError("save_dir must be a non-empty str.")
        if not isinstance(self.save_best_only, bool):
            raise TypeError("save_best_only must be a bool.")

        # ------------------------------------------------------------------
        # Pretrained backbone
        # ------------------------------------------------------------------
        if not isinstance(self.unet_pretrained_path, str) or not self.unet_pretrained_path:
            raise TypeError("unet_pretrained_path must be a non-empty str.")
        if not isinstance(self.ae_pretrained_path, str) or not self.ae_pretrained_path:
            raise TypeError("ae_pretrained_path must be a non-empty str.")
        if not isinstance(self.scheduler_pretrained_path, str) or not self.scheduler_pretrained_path:
            raise TypeError("scheduler_pretrained_path must be a non-empty str.")

        # ------------------------------------------------------------------
        # Optimization / training loop
        # ------------------------------------------------------------------
        if not isinstance(self.lr, (int, float)):
            raise TypeError("lr must be a float > 0.")
        if float(self.lr) <= 0.0:
            raise ValueError("lr must be > 0.")

        if not isinstance(self.mixed_precision, str) or not self.mixed_precision:
            raise TypeError("mixed_precision must be a non-empty str.")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of {'no', 'fp16', 'bf16'}.")

        if not isinstance(self.grad_clip, (int, float)):
            raise TypeError("grad_clip must be a float >= 0.")
        if float(self.grad_clip) < 0.0:
            raise ValueError("grad_clip must be >= 0.")

        for name, val in (
            ("batch_size", self.batch_size),
            ("val_batch_size", self.val_batch_size),
            ("epochs", self.epochs),
        ):
            if not isinstance(val, int):
                raise TypeError(f"{name} must be an int.")
            if val <= 0:
                raise ValueError(f"{name} must be > 0.")

        # ------------------------------------------------------------------
        # Encoder kwargs
        # ------------------------------------------------------------------
        for name, val in (
            ("cond_encoder_kwargs", self.cond_encoder_kwargs),
            ("coord_encoder_kwargs", self.coord_encoder_kwargs),
            ("bbox_encoder_kwargs", self.bbox_encoder_kwargs),
        ):
            if val is not None and not isinstance(val, dict):
                raise TypeError(f"{name} must be a dict or None.")